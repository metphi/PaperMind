"""State management for the chunker graph."""

from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass(kw_only=True)
class ChunkerInputState:
    """Narrow external interface of the chunker graph.

    Only ``source`` / ``data`` / ``metadata`` are settable by callers. The
    intermediate and final products (``raw_docs`` / ``chunks``) live on
    :class:`ChunkerState` and are produced by the nodes, not the caller.
    """

    source: str
    """Document identity: a local filesystem path, or an opaque client
    doc_id for online uploads."""

    data: bytes | None = None
    """Remote document bytes to parse instead of reading from ``source``."""

    metadata: dict = field(default_factory=dict)
    """Optional paper metadata (title/year/author). Must be scalar-valued."""


@dataclass(kw_only=True)
class ChunkerState(ChunkerInputState):
    """Full internal state of the chunker graph.

    A single-producer sequential pipeline, so no reducers are needed: node1
    writes ``raw_docs``, node2 reads it and writes ``chunks``.
    """

    raw_docs: list[Document] = field(default_factory=list)
    """Page-level documents produced by the document loader (node1)."""

    chunks: list[Document] = field(default_factory=list)
    """Final chunks ready to feed ``indexer.ainvoke({"docs": chunks})``."""
