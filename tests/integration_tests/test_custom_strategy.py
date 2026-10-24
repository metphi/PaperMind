"""End-to-end proof that the pluggable strategy contract is open to users.

A user only needs to know the two ABCs (``DocumentLoader`` / ``TextSplitter``):
subclass one and implement the single hook. Strategy-private parameters (e.g.
``size``, ``prefix``) live under a dedicated namespace inside ``configurable``
(``loader_configuration`` / ``split_configuration``) so they can never collide
with the graph's own top-level knobs, and are handed to ``configure`` as a
plain dict — the chunker graph no longer filters them through a config schema
(that was the "problem 1" contract lock).

Covered here:
- custom ``TextSplitter`` with a private ``size`` param (unit + graph level),
- custom ``DocumentLoader`` with a private ``prefix`` param (online branch),
- dotted-path providers (built-ins imported lazily from ``configurable``).
"""

import asyncio
from pathlib import Path

from langchain_core.documents import Document

from chunk_graph import graph as chunk_graph
from chunk_graph.base import DocumentLoader, TextSplitter

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.pdf"


def _run(coro):
    """Run an async graph call to completion on a fresh event loop."""
    return asyncio.run(coro)


class SizeSplitter(TextSplitter):
    """Chunk page content into fixed-width slices of ``size`` characters."""

    provider = "size"

    def __init__(self) -> None:
        self.size = 100  # default; overridden by ``configure`` when ``size`` is set

    def configure(self, config):
        if isinstance(config, dict) and "size" in config:
            self.size = config["size"]

    def _split(self, raw_docs):
        out = []
        for doc in raw_docs:
            text = doc.page_content
            out += [
                Document(page_content=text[i : i + self.size], metadata=dict(doc.metadata))
                for i in range(0, len(text), self.size)
            ]
        return out


class DummyLoader(DocumentLoader):
    """Load one Document without touching PyMuPDF; exercises the online branch."""

    provider = "dummy"

    def __init__(self) -> None:
        self.prefix = "x"

    def configure(self, config):
        if isinstance(config, dict) and "prefix" in config:
            self.prefix = config["prefix"]

    def _load(self, source, *, data=None, metadata=None):
        return [
            Document(
                page_content=f"{self.prefix}:{source}",
                metadata=dict(metadata or {}),
            )
        ]

    def _is_local(self, source, data):
        return False


def test_custom_splitter_reads_private_param() -> None:
    # Unit level: ``configure`` receives the strategy sub-config dict.
    splitter = TextSplitter.from_provider("size", {"size": 20})
    assert splitter.size == 20

    # Graph level: private params go under ``split_configuration``, not top-level.
    result = _run(
        chunk_graph.ainvoke(
            {"source": str(FIXTURE)},
            {
                "configurable": {
                    "split_provider": "size",
                    "split_configuration": {"size": 20},
                }
            },
        )
    )
    chunks = result["chunks"]
    assert chunks, "custom splitter produced no chunks"
    assert all(len(c.page_content) <= 20 for c in chunks)


def test_custom_loader_reads_private_param() -> None:
    result = _run(
        chunk_graph.ainvoke(
            {"source": "doc-id", "data": b"payload"},
            {
                "configurable": {
                    "loader_provider": "dummy",
                    "loader_configuration": {"prefix": "P"},
                }
            },
        )
    )
    raw = result["raw_docs"]
    assert raw and raw[0].page_content == "P:doc-id"
    assert raw[0].metadata["source"] == "doc-id"


def test_dotted_path_providers() -> None:
    result = _run(
        chunk_graph.ainvoke(
            {"source": str(FIXTURE)},
            {
                "configurable": {
                    "loader_provider": "chunk_graph.loaders.PdfDocumentLoader",
                    "split_provider": "chunk_graph.splitters.RecursiveSplitter",
                }
            },
        )
    )
    assert result["chunks"]
