"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

import json

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from index_graph.configuration import DEFAULT_DOCS_FILE
from index_graph.state import IndexState
from shared import retrieval
from shared.configuration import get_config
from shared.state import reduce_docs


def index_docs(
    state: IndexState, *, config: RunnableConfig | None = None
) -> dict[str, str]:
    """Index documents in LangGraph's executor using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    If docs are not provided in the state, they will be loaded
    from the configuration.docs_file JSON file.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    if not config:
        raise ValueError("Configuration required to run index_docs.")

    configurable = get_config(config)
    docs = state.docs
    if not docs:
        with open(configurable.get("docs_file", DEFAULT_DOCS_FILE)) as f:
            serialized_docs = json.load(f)
            docs = reduce_docs([], serialized_docs)

    with retrieval.make_retriever(config) as retriever:
        if configurable.get("retriever_provider", "chroma") == "chroma":
            # Chroma upsert by content-derived id (the MD5 of the full page_content,
            # already stored as metadata.uuid by reduce_docs), so re-running the
            # indexer with the same docs never duplicates vectors.
            ids = [doc.metadata["uuid"] for doc in docs]
            retriever.add_documents(docs, ids=ids)
        else:
            retriever.add_documents(docs)

    return {"docs": "delete"}


# Define the graph
builder = StateGraph(IndexState)
builder.add_node(index_docs)
builder.add_edge(START, "index_docs")
builder.add_edge("index_docs", END)
# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"
