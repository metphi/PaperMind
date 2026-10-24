"""Index every PDF in ``papers/`` into the default chroma store.

Pipeline (all defaults, no configurable overrides):
    chunker.ainvoke({"source": <pdf>})          # pdf loader, recursive splitter 512/64
    indexer.ainvoke({"docs": chunks})           # chroma ./chroma_db, bge-m3 on cpu

Usage (from repo root):
    PYTHONPATH=src python scripts/index_papers.py [papers_dir]

Idempotent: chroma upserts by the content MD5 (``metadata.uuid``) stamped by
``reduce_docs``, so re-running never duplicates vectors.
"""

import asyncio
import sys
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from chunk_graph import graph as chunk_graph
from index_graph import graph as index_graph

DEFAULT_PAPERS_DIR = Path(__file__).resolve().parents[1] / "papers"


async def _index_one(path: Path) -> int:
    """Chunk a single PDF with default settings, then index into chroma."""
    chunked = await chunk_graph.ainvoke({"source": str(path)})
    chunks = chunked["chunks"]
    if not chunks:
        raise RuntimeError("chunker returned no chunks")
    # Empty config -> indexer falls back to every default constant
    # (provider chroma, ./chroma_db, bge-m3 on cpu).
    indexed = await index_graph.ainvoke({"docs": chunks}, RunnableConfig())
    if indexed["docs"]:
        raise RuntimeError("indexer did not clear docs from state")
    return len(chunks)


async def _main(papers_dir: Path) -> None:
    pdfs = sorted(papers_dir.glob("*.pdf"))
    print(f"found {len(pdfs)} pdfs under {papers_dir}")

    total_chunks = 0
    ok: list[tuple[str, int]] = []
    failed: list[tuple[str, str]] = []

    for i, path in enumerate(pdfs, 1):
        try:
            n = await _index_one(path)
            total_chunks += n
            ok.append((path.name, n))
            print(f"[{i}/{len(pdfs)}] OK  {path.name}: {n} chunks")
        except Exception as exc:  # keep going so one bad PDF doesn't block the rest
            failed.append((path.name, repr(exc)))
            print(f"[{i}/{len(pdfs)}] ERR {path.name}: {exc!r}")

    print("\n=== summary ===")
    print(f"indexed {len(ok)}/{len(pdfs)} papers, total chunks: {total_chunks}")
    for name, n in ok:
        print(f"  {n:5d}  {name}")
    if failed:
        print("failed:")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PAPERS_DIR
    asyncio.run(_main(target))
