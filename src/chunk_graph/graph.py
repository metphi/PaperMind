"""Main entrypoint for the chunker graph.

Turns a PDF (local path or remote bytes) into index-ready chunks:

    START → pdf_to_raw_docs → raw_docs_to_chunks → END

- ``pdf_to_raw_docs`` validates the input boundary, then asks the configured
  ``DocumentLoader`` to emit one raw ``Document`` per page.
- ``raw_docs_to_chunks`` asks the configured ``TextSplitter`` to split those
  pages into chunks (plus an optional ``min_chunk_words`` post-filter).

The output ``chunks`` are ``list[Document]`` ready to feed
``indexer.ainvoke({"docs": chunks})`` directly.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

# Importing these modules seeds the provider registries with the built-in
# loaders/splitters (their classes auto-register via __init_subclass__).
from chunk_graph import loaders as _loaders  # noqa: F401
from chunk_graph import splitters as _splitters  # noqa: F401
from chunk_graph.base import DocumentLoader, TextSplitter
from chunk_graph.state import ChunkerInputState, ChunkerState
from shared.configuration import get_config


def _configurable(config: RunnableConfig | None) -> dict:
    """Return the raw ``configurable`` mapping, or ``{}`` when absent.

    Runtime-config style: nodes read their knobs straight from this dict.
    """
    return get_config(config)


def _strategy_config(configurable: dict, key: str) -> dict:
    """Return a strategy's private sub-config (``key``), or ``{}``.

    Custom loader/splitter parameters live under a dedicated namespace inside
    ``configurable`` (``loader_configuration`` / ``split_configuration``) so
    they can never collide with the graph's own top-level knobs, which are read
    straight from ``configurable``. Pass the sub-dict to ``from_provider`` so
    ``configure`` only ever sees the strategy's own parameters.
    """
    sub = configurable.get(key)
    return sub if isinstance(sub, dict) else {}


def pdf_to_raw_docs(
    state: ChunkerState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Load a PDF (local path or remote bytes) into page-level raw documents.

    Enforces the input boundary before delegating to the configured loader:
    ``source`` must be a non-empty string; when no ``data`` is given the
    ``source`` path must exist on disk; online uploads are capped by
    ``max_upload_bytes`` when configured.

    Args:
        state (ChunkerState): The current state (source/data/metadata).
        config (RunnableConfig): Configuration selecting the loader provider.

    Returns:
        dict[str, list[Any]]: A dictionary with the 'raw_docs' key.

    Raises:
        ValueError: If the source is invalid or the local path does not exist.
    """
    source = state.source
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"source must be a non-empty string, got {source!r}")

    configurable = _configurable(config)
    max_upload_bytes = configurable.get("max_upload_bytes")
    if state.data is None:
        if not Path(source).exists():
            raise ValueError(f"source path does not exist: {source}")
    else:
        if max_upload_bytes is not None and len(state.data) > max_upload_bytes:
            raise ValueError(
                f"upload too large: {len(state.data)} bytes exceeds "
                f"max_upload_bytes={max_upload_bytes}"
            )

    loader = DocumentLoader.from_provider(
        configurable.get("loader_provider", "pdf"),
        _strategy_config(configurable, "loader_configuration"),
    )
    raw_docs = loader.load(source, data=state.data, metadata=state.metadata)
    return {"raw_docs": raw_docs}


def raw_docs_to_chunks(
    state: ChunkerState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Split page-level raw documents into retrieval-ready chunks.

    Delegates to the configured ``TextSplitter``, then applies the optional
    ``min_chunk_words`` post-filter (word count per ``page_content``).

    Args:
        state (ChunkerState): The current state, including ``raw_docs``.
        config (RunnableConfig): Configuration selecting the split provider.

    Returns:
        dict[str, list[Any]]: A dictionary with the 'chunks' key.
    """
    configurable = _configurable(config)
    splitter = TextSplitter.from_provider(
        configurable.get("splitter_provider", "recursive"),
        _strategy_config(configurable, "splitter_configuration"),
    )
    chunks = splitter.split(state.raw_docs)
    min_chunk_words = configurable.get("min_chunk_words", 0)
    if min_chunk_words > 0:
        chunks = [
            c for c in chunks
            if len(c.page_content.split()) >= min_chunk_words
        ]
    return {"chunks": chunks}


# Define the graph
builder = StateGraph(ChunkerState, input_schema=ChunkerInputState)
builder.add_node(pdf_to_raw_docs)
builder.add_node(raw_docs_to_chunks)
builder.add_edge(START, "pdf_to_raw_docs")
builder.add_edge("pdf_to_raw_docs", "raw_docs_to_chunks")
builder.add_edge("raw_docs_to_chunks", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "ChunkerGraph"
