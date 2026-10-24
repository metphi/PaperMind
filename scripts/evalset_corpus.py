"""Dump per-page text for every PDF in ``papers/`` into ``evalset/corpus/``.

The gold spans in the evaluation set (§7 of ``need2modify.md``) are anchored to
*source + page + text* taken straight from the PDF, completely decoupled from
chunking. This script produces that corpus: for each paper, one JSON file with
the normalized ``source`` identity and the per-page text.

Why reuse the loader instead of calling ``pymupdf`` directly: the page text here
must match what the indexer actually embedded. ``DocumentLoader.from_provider(
"pdf").load()`` runs the exact same pipeline the chunker runs (``_parse_pages``
block filtering + ``normalize_source`` identity + ``sanitize_metadata``), so a
span anchored to ``{source, page, text}`` here will line up with the chunks
retrieved from chroma.

Output::

    evalset/corpus/<slug>.json
    {
      "source": "E:/.../LIIF.pdf",          # normalized absolute path
      "title": "Learning Continuous Image Representation...",
      "pages": [{"page": 1, "text": "..."}, ...]
    }

Usage (from repo root):
    PYTHONPATH=src python scripts/evalset_corpus.py [papers_dir]
"""

import asyncio
import json
import re
import sys
from pathlib import Path

# Importing these modules seeds the provider registries (auto-register via
# __init_subclass__) so "pdf" resolves to the built-in PdfDocumentLoader.
import chunk_graph.loaders as _loaders  # noqa: F401
from chunk_graph.base import DocumentLoader

DEFAULT_PAPERS_DIR = Path(__file__).resolve().parents[1] / "papers"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "evalset" / "corpus"


def _slug(path: Path) -> str:
    """Filesystem-safe name for a corpus JSON, stable across runs."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)


def _one_paper(path: Path) -> dict:
    """Extract source identity + per-page text for one PDF (sync, no graph)."""
    loader = DocumentLoader.from_provider("pdf", {})
    raw_docs = loader.load(str(path))  # one Document per page
    pages = [
        {"page": int(doc.metadata["page"]), "text": doc.page_content}
        for doc in raw_docs
    ]
    if not pages:
        raise RuntimeError(f"loader returned no pages: {path.name}")
    title = raw_docs[0].metadata.get("title") or path.stem
    return {
        "source": raw_docs[0].metadata["source"],
        "title": title,
        "pages": pages,
    }


async def _main(papers_dir: Path, out_dir: Path) -> None:
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        print(f"no pdfs under {papers_dir}")
        sys.exit(1)
    print(f"found {len(pdfs)} pdfs under {papers_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    total_pages = 0
    ok: list[tuple[str, int]] = []
    failed: list[tuple[str, str]] = []

    for i, path in enumerate(pdfs, 1):
        try:
            data = await asyncio.to_thread(_one_paper, path)
            out_path = out_dir / f"{_slug(path)}.json"
            out_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            total_pages += len(data["pages"])
            ok.append((path.name, len(data["pages"])))
            print(f"[{i}/{len(pdfs)}] OK  {path.name}: {len(data['pages'])} pages")
        except Exception as exc:
            failed.append((path.name, repr(exc)))
            print(f"[{i}/{len(pdfs)}] ERR {path.name}: {exc!r}")

    print("\n=== summary ===")
    print(f"extracted {len(ok)}/{len(pdfs)} papers, total pages: {total_pages}")
    for name, n in ok:
        print(f"  {n:4d}  {name}")
    if failed:
        print("failed:")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    papers_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PAPERS_DIR
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT_DIR
    asyncio.run(_main(papers_dir, out_dir))
