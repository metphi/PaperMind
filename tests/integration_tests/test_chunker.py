"""Integration tests for the chunker graph.

Covers the four contracts from ``ADD_CHUNKER.md`` step 6:
- local-path chunking (``source`` = a real file),
- online bytes chunking (``source`` = opaque doc_id, ``data`` = file bytes),
- invalid-input robustness (clean ``ValueError``),
- chunker→indexer end-to-end contract (chunks feed ``indexer.ainvoke`` and
  ``reduce_docs`` stamps the content ``uuid``).

The graphs are async (nodes use ``aadd_documents`` etc.), so each test drives
them through ``asyncio.run`` — no pytest-asyncio plugin required.
"""

import asyncio
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableConfig

from chunk_graph import graph as chunk_graph
from index_graph import graph as index_graph
from shared.chunking import normalize_source
from shared.state import reduce_docs

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.pdf"

_SCALAR_TYPES = (str, int, float, bool)
EMBEDDING_MODEL = "E:/python_work/PDFReader_Agent/bge-m3"


def _run(coro):
    """Run an async graph call to completion on a fresh event loop."""
    return asyncio.run(coro)


def test_chunker_local_path() -> None:
    result = _run(chunk_graph.ainvoke({"source": str(FIXTURE)}))
    chunks = result["chunks"]
    assert chunks, "local path produced no chunks"
    expected_source = normalize_source(str(FIXTURE))
    for chunk in chunks:
        assert chunk.metadata["source"] == expected_source
        assert "page" in chunk.metadata
        assert all(
            isinstance(v, _SCALAR_TYPES) for v in chunk.metadata.values()
        )


def test_chunker_online_bytes() -> None:
    doc_id = "sample-doc-001"
    result = _run(
        chunk_graph.ainvoke({"source": doc_id, "data": FIXTURE.read_bytes()})
    )
    chunks = result["chunks"]
    assert chunks, "online bytes produced no chunks"
    for chunk in chunks:
        assert chunk.metadata["source"] == doc_id
        assert "page" in chunk.metadata
        assert all(
            isinstance(v, _SCALAR_TYPES) for v in chunk.metadata.values()
        )


def test_chunker_invalid_input() -> None:
    with pytest.raises(ValueError):
        _run(chunk_graph.ainvoke({"source": ""}))
    with pytest.raises(ValueError):
        _run(chunk_graph.ainvoke({"source": "E:/definitely/not/here.pdf"}))
    config = RunnableConfig(configurable={"max_upload_bytes": 10})
    with pytest.raises(ValueError):
        _run(
            chunk_graph.ainvoke(
                {"source": "doc", "data": FIXTURE.read_bytes()}, config
            )
        )


def test_chunker_to_indexer_contract(tmp_path) -> None:
    chunks = _run(chunk_graph.ainvoke({"source": str(FIXTURE)}))["chunks"]
    assert chunks, "chunker produced no chunks to hand to the indexer"

    config = RunnableConfig(
        configurable={
            "retriever_provider": "chroma",
            "chroma_path": str(tmp_path / "chroma"),
            "embedding_model": EMBEDDING_MODEL,
        }
    )
    # A small slice keeps embedding fast while still exercising the whole path.
    sample = chunks[:5]
    indexed = _run(index_graph.ainvoke({"docs": sample}, config))
    assert indexed["docs"] == []  # indexer clears docs from state after ingesting

    # ``uuid`` (content identity, full-text MD5) is stamped by reduce_docs.
    reduced = reduce_docs([], sample)
    assert all("uuid" in doc.metadata for doc in reduced)
