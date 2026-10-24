"""Diagnose cross-paper retrieval recall by span and source.

Read-only diagnostic using the same retriever and token coverage definition as
``evalset_eval.py``. It prints the lowest-scoring cross-paper questions plus
per-source coverage and the source distribution of retrieved chunks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.runnables import RunnableConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shared.retrieval import make_retriever  # noqa: E402
from evalset_eval import norm, tokenize  # noqa: E402


def short_source(value: str) -> str:
    return Path(value).stem if value else "<unknown>"


async def score(pair: dict, k: int) -> dict:
    config = RunnableConfig({"configurable": {"search_kwargs": {"k": k}}})
    with make_retriever(config) as retriever:
        docs = await retriever.ainvoke(pair["question"])

    union = set().union(*(tokenize(doc.page_content) for doc in docs))
    span_rows = []
    by_source: dict[str, list[float]] = defaultdict(list)
    for span in pair["gold_spans"]:
        tokens = tokenize(span["text"])
        coverage = len(union & tokens) / len(tokens) if tokens else 0.0
        source = short_source(span["source"])
        by_source[source].append(coverage)
        span_rows.append({"source": source, "page": span["page"], "coverage": coverage})

    retrieved_sources = Counter(
        short_source(str(doc.metadata.get("source", ""))) for doc in docs
    )
    all_coverages = [row["coverage"] for row in span_rows]
    return {
        "question": pair["question"],
        "papers": pair.get("_meta", {}).get("papers", []),
        "hitrate": sum(all_coverages) / len(all_coverages),
        "source_coverage": {
            source: sum(values) / len(values) for source, values in by_source.items()
        },
        "spans": span_rows,
        "retrieved_sources": dict(retrieved_sources),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    pairs = [
        json.loads(line)
        for line in (ROOT / "evalset" / "dev.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = [pair for pair in pairs if pair.get("cross_paper")]
    rows = await asyncio.gather(*(score(pair, args.k) for pair in pairs))
    rows.sort(key=lambda row: row["hitrate"])
    print(json.dumps(rows[: args.limit], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
