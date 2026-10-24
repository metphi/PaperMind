"""Concrete document loaders for the chunker graph.

:class:`PdfDocumentLoader` ports the page-level parsing logic from
``PDFReader_Agent/ingest.py`` (``parse_pdf``): it reads a PDF either from a
local path or from remote bytes, keeps only visually meaningful text blocks,
drops the bibliography/references section, and emits one raw ``Document`` per
page. It only implements the :meth:`DocumentLoader._load` hook — identity
normalization and metadata sanitization are handled by the base class.
"""

import re
import unicodedata

from langchain_core.documents import Document

from chunk_graph.base import DocumentLoader

# ── Block filtering constants (ported from PDFReader_Agent/ingest.py) ──
MIN_WORDS, MIN_FRAGMENT, ALPHA_KEEP, ALPHA_FLOOR, END_PUNCT = 12, 5, 0.7, 0.5, ".?!"

_REF_HEAD = re.compile(r"^\s*\d{0,2}[.)]?\s*(references|bibliography)\b", re.IGNORECASE)
_REF_MIN_FRAC = 0.4


def is_meaningful_block(text: str) -> bool:
    """Return True if a text block carries enough real content to keep."""
    text = text.strip()
    if not text:
        return False
    n_words = len(text.split())
    non_space = sum(1 for c in text if not c.isspace())
    if non_space == 0:
        return False
    alpha_only = sum(c.isalpha() for c in text) / non_space
    if n_words < MIN_FRAGMENT:
        return False
    if n_words >= MIN_WORDS:
        return alpha_only >= ALPHA_FLOOR
    return alpha_only >= ALPHA_KEEP and text[-1] in END_PUNCT


def is_ref_heading(text: str, page_idx: int, num_pages: int) -> bool:
    """Return True if ``text`` looks like a reference-section heading.

    Two guards, matching ``ingest.py``: the heading must appear past the
    first ``_REF_MIN_FRAC`` of the document, and it must occupy its own line.
    """
    if page_idx < num_pages * _REF_MIN_FRAC:  # guard 1: position
        return False
    m = _REF_HEAD.match(text)
    if not m:
        return False
    rest = text[m.end() :].lstrip(" \t　:")  # strip inline whitespace only
    return rest == "" or rest[0] in "\r\n"  # guard 2: heading alone on its line


class PdfDocumentLoader(DocumentLoader):
    """Load a PDF (local path or remote bytes) into one raw doc per page."""

    provider = "pdf"

    def _load(
        self,
        source: str,
        *,
        data: bytes | None = None,
        metadata: dict | None = None,
    ) -> list[Document]:
        """Load a PDF from a local path or remote bytes into raw page docs.

        When ``data`` is given the bytes are parsed in-memory and ``source`` is
        treated as an opaque doc_id; otherwise the file at ``source`` is read
        from disk. The base class stamps the ``source`` identity and sanitizes
        the metadata afterwards.
        """
        import pymupdf  # lazy import: keeps the rest of the package usable without PyMuPDF

        document = (
            pymupdf.open(stream=data, filetype="pdf")
            if data is not None
            else pymupdf.open(filename=source)
        )
        try:
            return self._parse_pages(document, metadata)
        finally:
            document.close()

    def _parse_pages(
        self,
        document,  # pymupdf.Document, typed loosely to keep pymupdf import lazy
        metadata: dict | None,
    ) -> list[Document]:
        """Extract one raw document per page from a PyMuPDF document."""
        num_pages = document.page_count
        doc_meta = dict(document.metadata or {})
        if metadata:
            doc_meta.update(metadata)
        docmetadata = {
            "title": doc_meta.get("title") or "",
            "year": doc_meta.get("year") or 0,
            "author": doc_meta.get("author") or "",
        }

        raw_docs: list[Document] = []
        hit_ref = False
        for i in range(num_pages):
            if hit_ref:
                break
            page_string = ""
            page = document[i]
            for block in page.get_text("blocks"):
                if block[-1] != 0:  # skip non-text blocks (images etc.)
                    continue
                text = unicodedata.normalize("NFKC", block[4])
                text = text.replace("\r\n", "\n").replace("\r", "\n")
                if hit_ref or is_ref_heading(text, i + 1, num_pages):
                    hit_ref = True  # reference section: drop it and stop
                    break
                if is_meaningful_block(text):
                    page_string += text.strip() + "\n\n"
            if page_string.strip():
                raw_docs.append(
                    Document(
                        page_content=page_string,
                        metadata={**docmetadata, "page": i + 1},
                    )
                )
        return raw_docs
