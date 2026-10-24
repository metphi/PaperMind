"""Re-run the full graph for a few pairs and dump the retrieved evidence docs.

The C-2 judge scored answers against evidence, but eval_generation.py did not
persist that evidence. To manually verify judge verdicts, re-run the graph for
the target questions and dump what the judge saw (top 25 docs, 1500 chars each).

Usage:
    PYTHONPATH=src python evalset/c2_review/dump_evidence.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from retrieval_graph.graph import graph

OUT = Path(__file__).resolve().parent

TARGETS = [
    ("5-2-2", "A_Phase_Shift_Deep_Neural_Network_for_High-Frequency_Approximation"),
    ("1-1-1", "NeurIPS-2020-fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains-Paper_1_"),
    ("5-5-5", "LMF"),
]


async def main() -> None:
    from shared.utils import load_chat_model  # noqa: E402
    load_chat_model  # ensure imports resolve
    import json as _json
    pairs = [
        _json.loads(l)
        for l in (ROOT / "evalset" / "dev.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    for tag, key in TARGETS:
        pair = next(p for p in pairs if key in (p["_meta"].get("paper") or "") or
                    key in (p["_meta"].get("papers") or []) or
                    key in p["question"])
        q = pair["question"]
        print(f"== {tag}: {q[:60]}...", flush=True)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(q)]}, RunnableConfig({"configurable": {}})
        )
        docs = result.get("documents", [])
        lines = [f"question: {q}", f"router: {result.get('router', {}).get('type')}",
                 f"n_docs: {len(docs)}", "=" * 60]
        for d in docs[:25]:
            src = d.metadata.get("source")
            page = d.metadata.get("page")
            lines.append(f"[source={src} | page={page}]")
            lines.append(d.page_content[:1500])
            lines.append("-" * 60)
        (OUT / f"evidence_{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"  -> evidence_{tag}.txt ({len(docs)} docs)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
