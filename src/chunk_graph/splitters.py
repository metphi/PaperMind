"""Concrete text splitters for the chunker graph.

:class:`RecursiveSplitter` wraps LangChain's ``RecursiveCharacterTextSplitter``,
splitting each raw page document independently so chunks never cross page
boundaries and the ``page``/``source`` metadata survives into every chunk. It
only implements the :meth:`TextSplitter._split` hook and reads its parameters
from the graph configuration via :meth:`RecursiveSplitter.configure`.
"""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunk_graph.base import TextSplitter


class RecursiveSplitter(TextSplitter):
    """Split raw page documents into chunks with ``RecursiveCharacterTextSplitter``."""

    provider = "recursive"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        """Create a recursive splitter with the given size and overlap."""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def configure(self, config: Any) -> None:
        """Read ``chunk_size`` / ``chunk_overlap`` from ``splitter_configuration``.

        Only keys actually present override the ``__init__`` defaults, so a
        caller can tweak one knob (or none) and keep the rest.
        """
        if not isinstance(config, dict):
            return
        if "chunk_size" in config:
            self._chunk_size = config["chunk_size"]
        if "chunk_overlap" in config:
            self._chunk_overlap = config["chunk_overlap"]

    def _split(self, raw_docs: list[Document]) -> list[Document]:
        """Split page-level documents into chunks, preserving their metadata."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        return splitter.split_documents(raw_docs)
