"""Trim re-anchored spans to word boundaries.

Qwen sometimes returns spans cut mid-word (e.g. "rator that leverages a
Galerkin-type..."). The C9 verbatim check passes (they are page substrings) but
such fragments are ugly as gold evidence. This pass extends every span's raw
text at both ends to full words within its page, re-checks verbatim, and rewrites
gen_log + dev.jsonl.

Usage:
    PYTHONPATH=src python scripts/evalset_trim_spans.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import generate_evalset as ge  # noqa: E402


def _norm_aligned(raw: str) -> tuple[str, list[tuple[int, str]]]:
    """Simulate ``norm()`` while tracking each output char's source raw index.

    Returns ``(normed_string, [(raw_index, char)])`` — one pair per output char.
    Reproduces norm()'s transform order: \\r->space, drop "-" followed by
    whitespace, collapse whitespace runs to one space, drop remaining "-",
    strip leading/trailing whitespace, lowercase.
    """
    s = [(i, (" " if c == "\r" else c)) for i, c in enumerate(raw)]

    # drop "-" that is immediately followed by a whitespace run (join the words)
    out: list[tuple[int, str]] = []
    k = 0
    while k < len(s):
        i, c = s[k]
        if c == "-":
            j = k + 1
            while j < len(s) and s[j][1].isspace():
                j += 1
            if j > k + 1:  # hyphen + whitespace -> remove both
                k = j
                continue
        out.append((i, c))
        k += 1
    s = out

    # collapse whitespace runs to a single space
    out = []
    k = 0
    while k < len(s):
        i, c = s[k]
        if c.isspace():
            out.append((i, " "))
            while k + 1 < len(s) and s[k + 1][1].isspace():
                k += 1
        else:
            out.append((i, c))
        k += 1
    s = out

    # drop remaining hyphens
    s = [(i, c) for i, c in s if c != "-"]

    # strip leading/trailing whitespace, lowercase
    start, end = 0, len(s)
    while start < end and s[start][1].isspace():
        start += 1
    while end > start and s[end - 1][1].isspace():
        end -= 1
    s = [(i, c.lower()) for i, c in s[start:end]]
    return "".join(c for _, c in s), s


def _fix(text: str, page: str) -> str:
    """Extend a span to full-word boundaries within its page (norm-aligned).

    Works even when the span starts mid-word across a PDF line-break hyphen
    ("rator" from "opera-\\ntor") because matching happens in normed space.
    Line-break hyphens inside the returned span are joined for clean display.
    """
    norm_page, aligned = _norm_aligned(page)
    n_span = ge.norm(text)
    i = norm_page.find(n_span)
    if i < 0:
        return text
    j = i + len(n_span)
    # extend to word boundaries in normed space
    while i > 0 and norm_page[i - 1] != " ":
        i -= 1
    while j < len(norm_page) and norm_page[j] != " ":
        j += 1
    raw_span = page[aligned[i][0]: aligned[j - 1][0] + 1]
    cleaned = re.sub(r"-\s+", "", raw_span)  # join PDF line-break hyphens
    if len(ge.norm(cleaned)) >= 30 and ge.norm(cleaned) in ge.norm(page):
        return cleaned
    return text


def main() -> None:
    corpus = ge.load_corpus()
    pages = corpus["pages"]
    lines = [json.loads(l) for l in ge.GEN_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    fixed_total = 0
    for rec in lines:
        if not rec.get("ok"):
            continue
        for span in rec.get("gold_spans", []):
            page = pages.get(span["source"], {}).get(span["page"])
            if not page:
                continue
            new = _fix(span["text"], page)
            if new != span["text"]:
                span["text"] = new
                fixed_total += 1
    with ge.GEN_LOG.open("w", encoding="utf-8") as fh:
        for r in lines:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ge._write_dev()
    print(f"trimmed {fixed_total} spans; gen_log + dev.jsonl rewritten")


if __name__ == "__main__":
    main()
