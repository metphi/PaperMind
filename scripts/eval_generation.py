"""Generate-side (end-to-end) evaluation against the eval set (CLAUDE.md §10 C 档).

For each pair in ``evalset/dev.jsonl``, run the FULL retrieval graph
(routing -> research plan -> parallel retrieval -> respond) on the question,
then have a Qwen judge score the final answer on three reference-free
dimensions:

    relevance  1-5  does the answer actually address the question?
    grounded   1-5  is every factual claim supported by the retrieved
                    evidence (no fabrication / hallucination)?
    citation   1-5  do the in-answer citations (source, page) point at
                    evidence that exists in the retrieved docs and matches
                    the cited sentence?

The graph runs with defaults (router / research-plan prompts, query/response
models, k=8) — deliberately *not* bypassing routing, so this doubles as an
end-to-end smoke test of the routing + planning changes.

Robustness (2026-08-13): a first run hung for 5h with no output because no
network call had a timeout. Now: `load_chat_model` sets a per-request timeout
(180s), each pair is bounded by GRAPH_TIMEOUT / JUDGE_TIMEOUT
(``asyncio.wait_for``), each pair prints a progress line with per-phase
timings, and results are checkpointed to ``DETAIL_FILE`` incrementally so a
killed run resumes by skipping already-done pairs. A timed-out pair becomes a
row with ``error`` set instead of hanging the run.

Notes:
- GRAPH_TIMEOUT was raised 300->600 after the first 20-run: the graph
  legitimately takes 100-180s (respond node sends 30+ chunks to Qwen), and
  two slow-but-valid pairs were cut at 300s. The per-call 180s timeout in
  load_chat_model is the anti-hang; GRAPH_TIMEOUT is only an outer net.
- Resume keys on ``pair_id`` (unique per pair). Re-running skips pairs whose
  row already exists in ``DETAIL_FILE``.

Usage:
    PYTHONPATH=src python scripts/eval_generation.py [--limit 20] [--dev evalset/dev.jsonl]
                         [--force]   # re-run pairs already in DETAIL_FILE (e.g. after rewriting questions)

Per-pair record persisted to DETAIL_FILE (each JSONL line):
    question / answer / router (type) / router_logic (routing reason) /
    evidence (the top-EVIDENCE_CAP docs the judge actually saw, source+page+text) /
    relevance / grounded / citation / rationale (judge's one-line reason).

Output: evalset/eval_generation_report.csv (per-pair scores) +
        evalset/eval_generation_detail.jsonl (answers + evidence, for review).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval_graph.configuration import DEFAULT_RESPONSE_MODEL  # noqa: E402
from retrieval_graph.graph import graph  # noqa: E402
from shared.utils import chat_client_session, load_chat_model  # noqa: E402
from shared.configuration import get_config  # noqa: E402

DEV_FILE = ROOT / "evalset" / "dev.jsonl"
REPORT_FILE = ROOT / "evalset" / "eval_generation_report.csv"
DETAIL_FILE = ROOT / "evalset" / "eval_generation_detail.jsonl"

# Hard bounds: a stuck network call must surface as a per-pair error, never
# hang the run (a previous run hung 5h with zero output). The per-call
# timeout=180 in load_chat_model is the real anti-hang; these caps are an
# outer net for pathological accumulation (e.g. every call timing out).
GRAPH_TIMEOUT = 600  # whole graph (router+plan+retrieve+respond) for one pair
JUDGE_TIMEOUT = 120  # one judge call

# Evidence caps passed to the judge: bound prompt size on doc-heavy pairs.
EVIDENCE_CAP = 1500  # chars per chunk
MAX_EVIDENCE_DOCS = 25  # chunks per pair


JUDGE_SYSTEM_PROMPT = """你是一个严谨的生成式问答评估员。你会收到：用户问题、助手回答、以及助手实际检索到的文献证据（每个片段标注 source 与 page）。请从三个维度给助手回答打分（整数 1-5），并给出一句话理由。

评分维度：
1. **relevance（回答相关性）**：回答是否直接、完整地回应了用户问题？有没有答非所问或漏掉问题的主要方面？5=完全切题且覆盖充分，1=完全答非所问。
2. **grounded（证据支撑）**：回答中的事实性内容是否都能在证据片段中找到支撑？是否编造了证据中没有的内容（幻觉）？未标注出处的论断视为支撑不足。5=每句论断都有证据且无编造，1=大量内容无据可查或明显编造。
3. **citation（引文准确率）**：回答中引用的出处（如"（文献：X，p.N）"）是否真实对应证据片段中确实存在的内容？引注与所在句子的内容是否匹配？5=引注全部准确对应，1=引注大量虚标或错配。

只依据提供的证据片段判断，不要假设证据之外的信息。三个分数都要给整数，并给出一句话理由（中文）。"""


class Judge(TypedDict):
    """Judge score for one answer."""

    relevance: int
    grounded: int
    citation: int
    rationale: str


def _pair_id(pair: dict) -> str:
    """Unique checkpoint/report key per pair.

    Cross pairs key on the paper tuple (``str(papers)`` — unique per pair).
    Single pairs key on the paper slug PLUS a stable hash of the question, so
    the 4 questions of one paper don't collapse into a single row (2026-08-13).
    """
    papers = pair.get("_meta", {}).get("papers")
    if papers:
        return str(papers)
    paper = pair.get("_meta", {}).get("paper")
    qhash = hashlib.md5(pair["question"].encode()).hexdigest()[:8]
    return f"{paper}#{qhash}"


def load_checkpoint() -> dict[str, dict]:
    """Read already-scored pairs from DETAIL_FILE, keyed by pair_id."""
    done: dict[str, dict] = {}
    if DETAIL_FILE.exists():
        for line in DETAIL_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["pair_id"]] = r
    return done


async def _run(coro, timeout: float, what: str):
    """Await ``coro`` with a hard timeout; raise RuntimeError on timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise RuntimeError(f"{what} timed out after {timeout}s") from None


async def _score_pair(pair: dict, judge_model) -> dict:
    question = pair["question"]
    row = {
        "pair_id": _pair_id(pair),
        "subset": "cross" if pair.get("cross_paper") else "single",
        "difficulty": pair.get("difficulty"),
        "question": question,
        "router": "",
        "router_logic": "",
        "n_docs": 0,
        "graph_s": "",
        "judge_s": "",
        "relevance": "",
        "grounded": "",
        "citation": "",
        "rationale": "",
        "evidence": [],
        "error": "",
        "answer": "",
    }
    try:
        t0 = time.monotonic()
        result = await _run(
            graph.ainvoke(
                {"messages": [HumanMessage(question)]},
                RunnableConfig({"configurable": {}}),
            ),
            GRAPH_TIMEOUT,
            "graph",
        )
        row["graph_s"] = round(time.monotonic() - t0, 1)
    except Exception as exc:  # network / provider / timeout on this pair
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    row["router"] = result.get("router", {}).get("type", "?")
    row["router_logic"] = result.get("router", {}).get("logic", "")
    messages = result.get("messages", [])
    if not messages:
        row["error"] = "no messages in result"
        return row
    answer = messages[-1].content
    row["answer"] = answer if isinstance(answer, str) else str(answer)

    docs = result.get("documents", [])
    row["n_docs"] = len(docs)
    evidence = [
        {
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "text": d.page_content[:EVIDENCE_CAP],
        }
        for d in docs[:MAX_EVIDENCE_DOCS]
    ]
    row["evidence"] = evidence
    try:
        t1 = time.monotonic()
        verdict = await _run(
            judge_model.ainvoke(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"question": question, "answer": answer, "evidence": evidence},
                            ensure_ascii=False,
                        ),
                    },
                ]
            ),
            JUDGE_TIMEOUT,
            "judge",
        )
        row["judge_s"] = round(time.monotonic() - t1, 1)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row["relevance"] = verdict["relevance"]
    row["grounded"] = verdict["grounded"]
    row["citation"] = verdict["citation"]
    row["rationale"] = verdict["rationale"]
    return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20, help="0 = all pairs")
    ap.add_argument("--dev", type=Path, default=DEV_FILE)
    ap.add_argument("--judge-model", default=get_config().get("response_model", DEFAULT_RESPONSE_MODEL))
    ap.add_argument("--force", action="store_true",
                    help="re-run pairs already checkpointed in DETAIL_FILE")
    ap.add_argument("--out", type=str, default="",
                    help="output prefix (e.g. --out evalset/gen_10c10s -> "
                         "evalset/gen_10c10s_report.csv / ..._detail.jsonl); "
                         "defaults to evalset/eval_generation_*")
    args = ap.parse_args()
    if args.out:
        globals()["REPORT_FILE"] = ROOT / f"{args.out}_report.csv"
        globals()["DETAIL_FILE"] = ROOT / f"{args.out}_detail.jsonl"

    pairs = [
        json.loads(l)
        for l in args.dev.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    if args.limit > 0:
        pairs = pairs[: args.limit]

    # Resume: skip pairs already checkpointed in DETAIL_FILE, unless --force.
    done = load_checkpoint() if not args.force else {}
    todo = [p for p in pairs if _pair_id(p) not in done]
    if todo:
        judge_model = load_chat_model(args.judge_model).with_structured_output(Judge)
    else:
        judge_model = None
    print(f"total={len(pairs)}  already done={len(done)}  to run={len(todo)}", flush=True)

    with DETAIL_FILE.open("a", encoding="utf-8") as fh:
        for i, p in enumerate(todo, 1):
            t0 = time.monotonic()
            r = await _score_pair(p, judge_model)
            dt = time.monotonic() - t0
            # checkpoint immediately: a killed run loses only the in-flight pair
            fh.write(
                json.dumps(
                    {**r, "gold_answer": p.get("gold_answer"),
                     "gold_sources": p.get("gold_sources")},
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()
            tag = f"router={r['router']:9s} graph={r['graph_s']}s judge={r['judge_s']}s"
            if r["error"]:
                print(f"[{i:>2}/{len(todo)}] {tag}  ERR {r['error'][:70]}", flush=True)
            else:
                why = (r.get("rationale") or "").replace("\n", " ")
                print(
                    f"[{i:>2}/{len(todo)}] {tag}  "
                    f"rel={r['relevance']} gnd={r['grounded']} cit={r['citation']}  "
                    f"{r['pair_id'][:40]}  ({dt:.0f}s)\n"
                    f"      judge: {why[:110]}",
                    flush=True,
                )

    # Summary + report.csv derived from the full checkpoint (done + new).
    rows = list(load_checkpoint().values())
    ok = [r for r in rows if not r.get("error") and r.get("relevance") != ""]
    n = len(ok)
    print(f"\ncheckpointed pairs={len(rows)}  judged={n}  failed={len(rows) - n}")
    if n:
        for dim in ("relevance", "grounded", "citation"):
            vals = [int(r[dim]) for r in ok]
            print(f"  {dim:9s} mean={sum(vals) / n:.2f}  min={min(vals)}  max={max(vals)}")
        for key in ("single", "cross"):
            sub = [r for r in ok if r["subset"] == key]
            if sub:
                g = sum(int(r["grounded"]) for r in sub) / len(sub)
                print(f"  {key}: n={len(sub)} grounded={g:.2f}")
    from collections import Counter

    print("  router:", dict(Counter(r["router"] for r in rows)))

    with REPORT_FILE.open("w", encoding="utf-8", newline="") as fh:
        fields = ["pair_id", "subset", "difficulty", "question", "router",
                  "router_logic", "n_docs", "graph_s", "judge_s", "relevance",
                  "grounded", "citation", "rationale", "error"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)
    print(f"per-pair scores -> {REPORT_FILE}")
    print(f"detail (answers+evidence, for review) -> {DETAIL_FILE}")


if __name__ == "__main__":
    async def run():
        async with chat_client_session():
            await main()

    asyncio.run(run())
