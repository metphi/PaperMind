"""Evaluate the retrieval side against the eval set (§7.6 of need2modify.md).

For each pair in ``evalset/dev.jsonl``, retrieve top-k chunks from the vector
store and score, purely over text (no chroma internals):

    recall@k      per span: coverage = |topk_token_union & span_tokens| / |span_tokens|
                  hitrate    = mean coverage over ALL gold spans (uncovered
                  spans count as 0); no 0-1 threshold, so a question that needs
                  multiple spans / multiple sources must cover most of its
                  evidence points to score high.
                  recall@k = sum(hitrate) / total pairs
    有效命中率     fraction of the top-k chunks whose tokens contribute to covering
                  any gold span (a proxy for precision of the evidence window)

Cross-paper spans are pooled together (user decision): a cross question's
spans from all involved papers are averaged as one list, so a paper whose
evidence was never retrieved drags the mean down via its 0-coverage spans.

Retrieval uses the exact production path: ``make_retriever(config)`` with the
configurable ``search_kwargs.k``, so this doubles as the parameter-scan harness
for §2 (chunk_size) / §4 (k) without touching the pipeline.

Usage:
    PYTHONPATH=src python scripts/evalset_eval.py [--k 8] [--subset single|cross|all]

Output: summary lines + evalset/eval_report.csv (per-pair: pair_id, subset,
question, n_spans, hitrate, coverage_best, effective_hits, k).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from langchain_core.runnables import RunnableConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shared.retrieval import make_retriever  # noqa: E402

DEV_FILE = ROOT / "evalset" / "dev.jsonl"
REPORT_FILE = ROOT / "evalset" / "eval_report.csv"


def norm(s: str) -> str:
    """Canonical text (matches generate_evalset.norm): join hyphenated
    line-breaks, collapse whitespace, drop hyphens, lowercase."""
    s = s.replace("\r", " ")
    s = re.sub(r"-\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.replace("-", "").strip().lower()


def tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", norm(s)))


def _pair_id(pair: dict) -> str:
    """Unique report key per pair (same scheme as eval_generation._pair_id so the
    C-1 and C-2 reports join by pair_id). Cross pairs key on the paper tuple;
    single pairs key on the paper slug PLUS a stable hash of the question, so the
    4 questions of one paper don't collapse into a single row (2026-08-13)."""
    papers = pair.get("_meta", {}).get("papers")
    if papers:
        return str(papers)
    paper = pair.get("_meta", {}).get("paper")
    qhash = hashlib.md5(pair["question"].encode()).hexdigest()[:8]
    return f"{paper}#{qhash}"


async def _score_pair(
    pair: dict, k: int
) -> dict:
    question = pair["question"]
    spans = pair["gold_spans"]
    span_tokens = [tokenize(s["text"]) for s in spans]
    config = RunnableConfig(
        {"configurable": {"search_kwargs": {"k": k}}}
    )
    with make_retriever(config) as retriever:
        docs = await retriever.ainvoke(question)
    chunk_tokens = [tokenize(d.page_content) for d in docs]
    topk_union: set[str] = set()
    for ct in chunk_tokens:
        topk_union |= ct
    # per-span coverage; hitrate = mean over ALL gold spans (uncovered = 0),
    # pooled across sources for cross pairs (see module docstring)
    coverages = [
        len(topk_union & st) / len(st) if st else 0.0 for st in span_tokens
    ]
    best = max(coverages)
    hitrate = sum(coverages) / len(coverages) if coverages else 0.0
    # 有效命中率: chunks contributing tokens to any gold span
    any_span: set[str] = set()
    for st in span_tokens:
        any_span |= st
    contributing = sum(1 for ct in chunk_tokens if ct & any_span)
    return {
        "pair_id": _pair_id(pair),
        "subset": "cross" if pair.get("cross_paper") else "single",
        "question": question,
        "n_spans": len(spans),
        "hitrate": hitrate,
        "coverage_best": best,
        "effective_hits": contributing,
        "k": k,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--subset", choices=["single", "cross", "all"], default="all")
    ap.add_argument("--dev", type=Path, default=DEV_FILE)
    args = ap.parse_args()

    pairs = [
        json.loads(l)
        for l in args.dev.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    if args.subset != "all":
        want = args.subset == "cross"
        pairs = [p for p in pairs if p.get("cross_paper") == want]

    rows = [await _score_pair(p, args.k) for p in pairs]
    n = len(rows)
    recall = sum(r["hitrate"] for r in rows) / n if n else 0.0
    eff = sum(r["effective_hits"] for r in rows) / (n * args.k) if n else 0.0
    print(f"pairs={n}  recall@{args.k}={recall:.2%}  有效命中率={eff:.2f}")
    if args.subset == "all":
        for sub in ("single", "cross"):
            sub_rows = [r for r in rows if r["subset"] == sub]
            if not sub_rows:
                continue
            sr = sum(r["hitrate"] for r in sub_rows) / len(sub_rows)
            print(f"  {sub}: recall@{args.k}={sr:.2%}")

    with REPORT_FILE.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"per-pair -> {REPORT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
