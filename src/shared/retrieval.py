"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, MongoDB, and Chroma.
"""

import os
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from threading import RLock
from typing import Generator

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from shared.configuration import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_PATH,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RETRIEVER_PROVIDER,
    DEFAULT_SEARCH_KWARGS,
    RETRIEVER_PROVIDERS,
    get_config,
)

## Encoder constructors

_embedding_init_lock = RLock()
_chroma_init_lock = RLock()


class LocalEmbeddings(Embeddings):
    """LangChain-compatible embeddings that run locally via sentence-transformers.

    Wraps a ``sentence_transformers.SentenceTransformer`` model so it can be used
    with any LangChain vector store. Mirrors the PDFReader_Agent behaviour:
    normalized embeddings, batched encoding.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        *,
        normalize_embeddings: bool = True,
        batch_size: int = 64,
    ) -> None:
        """Wrap a local sentence-transformers model as LangChain embeddings."""
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents into a list of vectors."""
        vectors = self._model.encode(
            texts,
            normalize_embeddings=self._normalize_embeddings,
            batch_size=self._batch_size,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query into a vector."""
        return self.embed_documents([text])[0]


@cache
def _get_embedding_model(model: str, device: str) -> LocalEmbeddings:
    """Return a cached local embedding model.

    Loading a model is expensive, so each (model, device) pair is loaded only
    once per process.
    """
    return LocalEmbeddings(model, device)


def make_text_encoder(model: str, device: str = "cpu") -> Embeddings:
    """Connect to the configured local text encoder.

    ``model`` is passed straight to ``sentence_transformers.SentenceTransformer``:
    it can be a HuggingFace model name or a local directory path.
    """
    # functools.cache alone permits concurrent misses to initialize repeatedly.
    with _embedding_init_lock:
        return _get_embedding_model(model, device)


## Retriever constructors


@contextmanager
def make_elastic_retriever(
    configurable: dict, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    connection_options = {}
    if (
        configurable.get("retriever_provider", DEFAULT_RETRIEVER_PROVIDER)
        == "elastic-local"
    ):
        connection_options = {
            "es_user": os.environ["ELASTICSEARCH_USER"],
            "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
        }

    else:
        connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}

    vstore = ElasticsearchStore(
        **connection_options,  # type: ignore
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )

    yield vstore.as_retriever(search_kwargs=configurable.get("search_kwargs") or DEFAULT_SEARCH_KWARGS)


@contextmanager
def make_pinecone_retriever(
    configurable: dict, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_pinecone import PineconeVectorStore

    vstore = PineconeVectorStore.from_existing_index(
        os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
    )
    yield vstore.as_retriever(search_kwargs=configurable.get("search_kwargs") or DEFAULT_SEARCH_KWARGS)


@contextmanager
def make_mongodb_retriever(
    configurable: dict, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="langgraph_retrieval_agent.default",
        embedding=embedding_model,
    )
    yield vstore.as_retriever(search_kwargs=configurable.get("search_kwargs") or DEFAULT_SEARCH_KWARGS)


@cache
def _get_chroma_client(path: str):
    """Create one persistent client per absolute path under the init lock."""
    import chromadb

    return chromadb.PersistentClient(path=path)


@contextmanager
def make_chroma_retriever(
    configurable: dict, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a local Chroma persistent store."""
    from langchain_chroma import Chroma

    path = str(Path(configurable.get("chroma_path", DEFAULT_CHROMA_PATH)).resolve())
    # Chroma's process-wide Rust system registry is not safe during concurrent
    # first initialization. Serialize client/collection setup, not retrieval.
    with _chroma_init_lock:
        vstore = Chroma(
            client=_get_chroma_client(path),
            collection_name=configurable.get(
                "chroma_collection", DEFAULT_CHROMA_COLLECTION
            ),
            embedding_function=embedding_model,
        )
    yield vstore.as_retriever(search_kwargs=configurable.get("search_kwargs") or DEFAULT_SEARCH_KWARGS)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configurable = get_config(config)
    embedding_model = make_text_encoder(
        configurable.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        configurable.get("embedding_device", DEFAULT_EMBEDDING_DEVICE),
    )
    provider = configurable.get("retriever_provider", DEFAULT_RETRIEVER_PROVIDER)
    match provider:
        case "elastic" | "elastic-local":
            with make_elastic_retriever(configurable, embedding_model) as retriever:
                yield retriever

        case "pinecone":
            with make_pinecone_retriever(configurable, embedding_model) as retriever:
                yield retriever

        case "mongodb":
            with make_mongodb_retriever(configurable, embedding_model) as retriever:
                yield retriever

        case "chroma":
            with make_chroma_retriever(configurable, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(RETRIEVER_PROVIDERS)}\n"
                f"Got: {provider}"
            )
