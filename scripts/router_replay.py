"""Route-only replay over the eval set — routing distribution / correctness.

For each pair in ``evalset/dev.jsonl``, run ONLY the router step — no research
plan, no retrieval, no respond — and record the decision (type + logic). This
is the cheap "routing correctness" metric (CLAUDE.md §10 C 档) that quantifies
the 2026-08-13 router prompt fix over the whole set, and surfaces any remaining
implicit-reference questions the loosened ``more-info`` prompt now catches.

The routing call replicates ``analyze_and_route_query``'s body (same
ROUTER_SYSTEM_PROMPT constant, same default query model, same ``Router``
structured output) rather than calling the node directly — the node reads
``state.messages`` by attribute, which the graph runtime wraps but a bare
function call with a plain dict does not.

One structured-output call per question (~1-5s), no vector DB, no embeddings.
Checkpoints incrementally to OUT_FILE (resume skips done pair_ids; a killed
run loses only the in-flight question).

Usage:
    PYTHONPATH=src python scripts/router_replay.py [--limit 0]

Output: evalset/router_replay.jsonl (per-question: pair_id / subset /
difficulty / question / router / logic / error) + a distribution summary and
a listing of every non-langchain decision with its reason.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import cast

from langchain_core.runnables import RunnableConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval_graph.configuration import (  # noqa: E402
    DEFAULT_QUERY_MODEL,
    DEFAULT_ROUTER_SYSTEM_PROMPT,
)
from retrieval_graph.state import Router  # noqa: E402
from shared.configuration import get_config  # noqa: E402
from shared.utils import load_chat_model  # noqa: E402

DEV_FILE = ROOT / "evalset" / "dev.jsonl"
OUT_FILE = ROOT / "evalset" / "router_replay.jsonl"


def _pair_id(pair: dict) -> str:
    """Same scheme as eval_generation._pair_id / evalset_eval._pair_id."""
    papers = pair.get("_meta", {}).get("papers")
    if papers:
        return str(papers)
    paper = pair.get("_meta", {}).get("paper")
    qhash = hashlib.md5(pair["question"].encode()).hexdigest()[:8]
    return f"{paper}#{qhash}"


async def _route(question: str) -> Router:
    """Mirror of analyze_and_route_query's body: same prompt, model, structured
    output — but fed the question as a user message instead of via graph state."""
    configurable = get_config(RunnableConfig({"configurable": {}}))
    model = load_chat_model(configurable.get("query_model", DEFAULT_QUERY_MODEL))
    messages = [
        {
            "role": "system",
            "content": configurable.get(
                "router_system_prompt", DEFAULT_ROUTER_SYSTEM_PROMPT
            ),
        },
        {"role": "user", "content": question},
    ]
    return cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--dev", type=Path, default=DEV_FILE)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()

    pairs = [
        json.loads(l)
        for l in args.dev.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    if args.limit > 0:
        pairs = pairs[: args.limit]

    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["pair_id"])
    todo = [p for p in pairs if _pair_id(p) not in done]
    print(f"total={len(pairs)}  already done={len(done)}  to run={len(todo)}", flush=True)

    with args.out.open("a", encoding="utf-8") as fh:
        for i, p in enumerate(todo, 1):
            try:
                router = await _route(p["question"])
                row = {
                    "pair_id": _pair_id(p),
                    "subset": "cross" if p.get("cross_paper") else "single",
                    "difficulty": p.get("difficulty"),
                    "question": p["question"],
                    "router": router["type"],
                    "logic": router.get("logic", ""),
                    "error": "",
                }
            except Exception as exc:
                row = {
                    "pair_id": _pair_id(p),
                    "subset": "cross" if p.get("cross_paper") else "single",
                    "difficulty": p.get("difficulty"),
                    "question": p["question"],
                    "router": "ERR",
                    "logic": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(
                f"[{i:>3}/{len(todo)}] router={row['router']:10s} "
                f"{row['pair_id'][:45]}",
                flush=True,
            )
            if row["error"]:
                print(f"      ERR {row['error'][:80]}", flush=True)

    rows = [
        json.loads(l)
        for l in args.out.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(
        f"\nrouted={len(rows)}  "
        f"distribution={dict(Counter(r['router'] for r in rows))}"
    )
    for r in rows:
        if r["router"] != "langchain":
            logic = (r.get("logic") or "").replace("\n", " ")
            print(f"  {r['router']:10s} {r['pair_id'][:45]}")
            print(f"      logic: {logic[:160]}")
    print(f"detail -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
