"""Generate the evaluation set (dev) for the 20 papers with Qwen.

Two-stage generation per ``need2modify.md`` §7.3:

  stage 1 (question): feed ONLY the paper abstract(s) (no body text) so the
      question cannot word-copy the evidence -> question + gold_answer + difficulty
  stage 2 (anchor):   feed the question + gold_answer + the involved papers'
      per-page text (from ``evalset/corpus/``) -> 2..5 verbatim ``gold_spans``
      ``{source, page, text}`` that suffice to derive the answer. Every span is
      checked character-verbatim against the corpus page text; bad spans are
      dropped and the pair is regenerated if too few survive.

Layout (§7.4): 80 single-paper pairs (20 papers x 4 slots) + 20 cross-paper
pairs (2-3 papers sampled inside the same subtopic bucket, seeded, with a
per-paper coverage check). Cross-pair answers must be "缺一不可": they cite
*every* paper in the combo, so dropping any one paper leaves the answer
incomplete.

Incremental: every finished pair is appended to ``evalset/gen_log.jsonl``; a
re-run skips pairs already in the log (use ``--force`` to regenerate). At the
end the successful pairs are written to ``evalset/dev.jsonl`` in the §7.3
schema (plus ``_meta`` for attribution).

Usage (from repo root):
    PYTHONPATH=src python scripts/generate_evalset.py [--seed 42] [--model qwen/qwen3.7-plus]
    PYTHONPATH=src python scripts/generate_evalset.py --single-n 4 --cross-n 2   # pilot
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

from shared.utils import load_chat_model  # noqa: E402

CORPUS_DIR = ROOT / "evalset" / "corpus"
TOPICS_FILE = ROOT / "evalset" / "topics.json"
GEN_LOG = ROOT / "evalset" / "gen_log.jsonl"
DEV_FILE = ROOT / "evalset" / "dev.jsonl"

# 4 slots per paper (definition / method / cross-section synthesis / challenge),
# covering abstract·method·method+results·results·discussion (§7.4) and spreading
# difficulty ~ easy 1, medium 2, hard 1.
SLOTS = [
    {
        "key": "definition",
        "section": "abstract/intro",
        "difficulty": "easy",
        "paraphrase": False,
        "desc": (
            "a conceptual/definitional question: what the method is, the problem "
            "it solves, and its key idea. The answer is a definitional fact the "
            "paper states."
        ),
    },
    {
        "key": "method",
        "section": "method",
        "difficulty": "medium",
        "paraphrase": True,
        "desc": (
            "a question about the method's architecture / mechanism / design "
            "choice: how it works, its novel components, losses or training "
            "objective. Ask for one specific mechanism the paper introduces."
        ),
    },
    {
        "key": "synthesis",
        "section": "method+results",
        "difficulty": "medium",
        "paraphrase": True,
        "desc": (
            "a cross-section synthesis question that connects a mechanism from "
            "the method to evidence in the experiments (e.g. which design "
            "component drives which reported result / ablation)."
        ),
    },
    {
        "key": "challenge",
        "section": "results/discussion",
        "difficulty": "hard",
        "paraphrase": True,
        "desc": (
            "a question about the results / limitations / trade-offs / "
            "comparisons: what the experiments showed, which claim they support, "
            "or a limitation or cost the paper acknowledges."
        ),
    },
]

# How many cross-paper pairs per subtopic bucket (total = 20).
CROSS_QUOTAS = {
    "A_inr_image_sr": 7,
    "B_frequency_operator_sr": 5,
    "C_gaussian_flow_sr": 2,
    "D_scientific_computing": 6,
}
# A couple of 3-paper combos (rest are 2-paper) for genuinely "缺一不可" hard pairs.
TRIPLES = {"A_inr_image_sr", "D_scientific_computing"}

SYSTEM_STAGE1 = """You are a benchmark writer building a retrieval evaluation set over machine-learning research papers. You write ONE question per call. You are given the paper's abstract(s) ONLY — you must never see the body text, so the question cannot reuse the paper's exact wording. The question and gold answer must be fully answerable from the given paper(s)."""

SYSTEM_STAGE2 = """You are an evidence-annotation assistant. Given a question, its gold answer, and the verbatim page text of the paper(s) involved, you locate the smallest verbatim text excerpts (spans) that together are sufficient to derive the gold answer. You copy excerpts EXACTLY from the provided page text — no paraphrasing, no adding text."""

STAGE1_SINGLE = """Paper title: {title}
Abstract (the ONLY source of evidence; do not invent facts beyond it):
{abstract}

Write ONE evaluation question of the following type:
{desc}

The question should target the paper's {section}.

Requirements:
- The gold answer must be a specific factual claim the paper makes (1-3 sentences), verifiable from the paper.
- {paraphrase_rule}
- difficulty: {difficulty}

Return only JSON: {{"question": str, "gold_answer": str, "difficulty": "{difficulty}"}}"""

STAGE1_CROSS = """Papers (abstracts are the ONLY evidence; do not invent facts beyond them):
{abstracts_block}

Write ONE cross-paper evaluation question.

Type: compare/contrast two of the approaches, OR synthesize findings across the papers.
The question and gold answer must be "缺一不可": the correct answer must draw on a specific fact/claim from EACH paper, and if any one paper's evidence is removed the answer becomes incomplete. Do NOT ask to propose a new method.

Requirements:
- The gold answer must explicitly invoke each paper's relevant contribution.
- Paraphrase hard: do not reuse the abstracts' wording.
- difficulty: {difficulty}

Return only JSON: {{"question": str, "gold_answer": str, "difficulty": "{difficulty}"}}"""

STAGE2 = """Question: {question}
Gold answer: {gold_answer}

Paper page text (verbatim, page-tagged):
{pages_block}

Return only JSON with the smallest 2-5 verbatim excerpts that together suffice to derive the gold answer:
{{"gold_spans": [{{"source": str, "page": int, "text": str}}]}}
Rules:
- "text" must be copied EXACTLY from the page text above (same words, same order). The page text has PDF line-break hyphens (e.g. "ar-\\nbitrary"): when you copy such a word, write it WITHOUT the hyphen and line break ("arbitrary").
- Prefer the fewest, smallest spans that still fully support the answer.
- "page" must be the page number the text came from; "source" must be one of the given sources.
- For cross-paper questions, the spans MUST cover every source listed above."""


# ── text helpers ─────────────────────────────────────────────────────────────


def norm(s: str) -> str:
    """Canonical text for verbatim / overlap comparisons.

    Joins hyphenated line-breaks ("ar-\\nbitrary" -> "arbitrary"), collapses
    whitespace, removes remaining hyphens, lowercases. Both sides of a
    comparison go through the same norm, so PDF line-break artifacts on one
    side never cause a false mismatch.
    """
    s = s.replace("\r", " ")
    s = re.sub(r"-\s+", "", s)  # hyphen + line-break whitespace -> join words
    s = re.sub(r"\s+", " ", s)
    return s.replace("-", "").strip().lower()


def extract_json(text: str) -> dict:
    """Robustly pull the first balanced JSON object out of an LLM reply."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in reply: {t[:120]!r}")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("unbalanced JSON object in reply")


def tokenize(s: str) -> set[str]:
    """Lowercased alphanumeric tokens, stopwords removed."""
    STOP = {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with",
        "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
        "these", "those", "as", "at", "by", "from", "we", "our", "they", "their",
        "how", "what", "which", "why", "can", "does", "do", "not", "no", "but",
        "such", "than", "then", "via", "using", "use", "uses", "used", "between",
    }
    return {w for w in re.findall(r"[A-Za-z]+", s.lower()) if w not in STOP and len(w) > 2}


# ── data loading ─────────────────────────────────────────────────────────────


def load_corpus() -> dict[str, dict]:
    pages: dict[str, dict[int, str]] = {}
    titles: dict[str, str] = {}
    for f in CORPUS_DIR.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        pages[d["source"]] = {p["page"]: p["text"] for p in d["pages"]}
        titles[d["source"]] = d["title"]
    return {"pages": pages, "titles": titles}


def load_topics() -> list[dict]:
    return json.loads(TOPICS_FILE.read_text(encoding="utf-8"))["papers"]


# ── cross-paper pair sampling ────────────────────────────────────────────────


def sample_cross_pairs(papers: list[dict], seed: int = 42) -> list[list[str]]:
    """Deterministic per-bucket sampling of 2-3 paper combos with coverage.

    Enforces: each paper appears in at least one cross pair and at most 4.
    """
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for p in papers:
        by_bucket[p["bucket"]].append(p)

    chosen: list[list[str]] = []
    counts: dict[str, int] = defaultdict(int)
    for bucket, quota in CROSS_QUOTAS.items():
        members = by_bucket[bucket]
        rng = random.Random(f"{seed}:{bucket}")
        got = 0
        uncovered = {m["slug"] for m in members}

        def _add(slugs: list[str]) -> None:
            nonlocal got
            chosen.append(slugs)
            got += 1
            for s in slugs:
                counts[s] += 1
                uncovered.discard(s)

        if bucket in TRIPLES and len(members) >= 3:
            triples = list(itertools.combinations(members, 3))
            rng.shuffle(triples)
            for triple in triples:  # one seeded 3-paper combo ("缺一不可" hard)
                slugs = [m["slug"] for m in triple]
                if any(counts[s] >= 4 for s in slugs):
                    continue
                _add(slugs)
                break

        # phase 1: actively cover every member before random fill
        order = rng.sample(members, len(members))
        for i in range(len(order)):
            if got >= quota or not uncovered:
                break
            a, b = order[i], order[(i + 1) % len(order)]
            if counts[a["slug"]] >= 4 or counts[b["slug"]] >= 4:
                continue
            if a["slug"] in uncovered or b["slug"] in uncovered:
                _add([a["slug"], b["slug"]])

        # phase 2: fill remaining quota with random same-bucket combos
        combos = list(itertools.combinations(members, 2))
        rng.shuffle(combos)
        for combo in combos:
            if got >= quota:
                break
            slugs = [m["slug"] for m in combo]
            if any(counts[s] >= 4 for s in slugs):
                continue
            _add(slugs)
        if got < quota:
            raise RuntimeError(f"bucket {bucket}: only {got}/{quota} pairs sampled")

    # coverage sanity
    unused = [p["slug"] for p in papers if counts[p["slug"]] == 0]
    over = {s: c for s, c in counts.items() if c > 4}
    if unused:
        raise RuntimeError(f"papers never in a cross pair: {unused}")
    if over:
        raise RuntimeError(f"papers over cap: {over}")
    return chosen


# ── Qwen calls ───────────────────────────────────────────────────────────────


async def _invoke_json(
    llm, system: str, user: str, validate, attempts: int = 4
) -> dict:
    """Call the model, extract JSON, validate, retry until valid."""
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    last: Exception | None = None
    for mode in ("json", "plain"):
        for _ in range(attempts):
            try:
                if mode == "json":
                    resp = await llm.bind(
                        response_format={"type": "json_object"}
                    ).ainvoke(messages)
                else:
                    resp = await llm.ainvoke(messages)
                obj = extract_json(str(resp.content))
                if not isinstance(obj, dict):
                    # LLM returned a JSON array or scalar; force a retry so the
                    # caller can always rely on dict output (no .get crashes).
                    raise ValueError(
                        f"expected JSON object, got {type(obj).__name__}"
                    )
                validate(obj)  # raises ValueError -> retry
                return obj
            except Exception as exc:  # noqa: BLE001 — regenerate on any failure
                last = exc
                print(f"    retry ({mode}): {type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError(f"Qwen failed after retries: {last}")


def _validate_stage1(obj: dict) -> None:
    for k in ("question", "gold_answer"):
        if not isinstance(obj.get(k), str) or not obj[k].strip():
            raise ValueError(f"missing/invalid {k}")
    d = obj.get("difficulty")
    if d not in ("easy", "medium", "hard"):
        raise ValueError(f"bad difficulty: {d!r}")


async def stage1_single(llm, paper: dict, slot: dict) -> dict:
    rule = (
        "HARD PARAPHRASE: the question must NOT reuse the abstract's wording or "
        "key phrases. Only the method/paper name may appear verbatim; every other "
        "content word must be paraphrased so that a word-overlap matcher would "
        "score the question as lexically distant from the abstract."
        if slot["paraphrase"]
        else "Paraphrase where reasonable, but a definitional question may name "
        "the concept."
    )
    user = STAGE1_SINGLE.format(
        title=paper["title"], abstract=paper["abstract"],
        desc=slot["desc"], section=slot["section"],
        paraphrase_rule=rule, difficulty=slot["difficulty"],
    )
    obj = await _invoke_json(llm, SYSTEM_STAGE1, user, _validate_stage1)
    return obj


async def stage1_cross(llm, combo: list[dict], difficulty: str) -> dict:
    abstracts_block = "\n\n".join(
        f"[{i+1}] {p['title']}\nAbstract: {p['abstract']}" for i, p in enumerate(combo)
    )
    user = STAGE1_CROSS.format(
        abstracts_block=abstracts_block, difficulty=difficulty
    )
    obj = await _invoke_json(llm, SYSTEM_STAGE1, user, _validate_stage1)
    return obj


def _pages_block(pages_by_source: dict[str, dict[int, str]], source: str) -> str:
    """Page text of one paper as a page-tagged block, capped for token budget.

    Feeds the first ~14 pages (enough for abstract + method + early results);
    long papers' tails are skipped to keep the prompt bounded. Pairs that need
    late results pages may fail stage-2 and get flagged for regeneration.
    """
    page_texts = pages_by_source[source]
    chunks = []
    for pg in sorted(page_texts)[:14]:
        t = page_texts[pg]
        if len(t) > 4000:
            t = t[:4000] + " ..."
        chunks.append(f"[page {pg}]\n{t}")
    return "\n\n".join(chunks)


def _validate_span(span: dict, pages_by_source: dict, sources: list[str]) -> bool:
    src = span.get("source")
    pg = span.get("page")
    text = span.get("text")
    if not isinstance(src, str) or not isinstance(pg, int) or not isinstance(text, str):
        return False
    if src not in sources or pg not in pages_by_source[src]:
        return False
    n_text, n_page = norm(text), norm(pages_by_source[src][pg])
    return len(n_text) >= 30 and n_text in n_page


async def stage2(
    llm,
    question: str,
    gold_answer: str,
    sources: list[str],
    pages_by_source: dict,
) -> list[dict]:
    for _ in range(4):
        block = "\n\n".join(
            f"=== SOURCE {src} ===\n{_pages_block(pages_by_source, src)}"
            for src in sources
        )
        user = STAGE2.format(
            question=question, gold_answer=gold_answer, pages_block=block
        )
        obj = await _invoke_json(
            llm, SYSTEM_STAGE2, user, lambda o: None, attempts=3
        )
        spans = obj.get("gold_spans")
        if not isinstance(spans, list):
            continue
        valid = [s for s in spans if _validate_span(s, pages_by_source, sources)]
        seen = set()
        dedup = []
        for s in valid:
            key = (s["source"], s["page"], norm(s["text"]))
            if key not in seen:
                seen.add(key)
                dedup.append(s)
        covered = {s["source"] for s in dedup}
        if len(dedup) >= 2 and set(sources) <= covered:
            return dedup[:5]
    return []


# ── drivers ──────────────────────────────────────────────────────────────────


def _load_existing() -> set[str]:
    if not GEN_LOG.exists():
        return set()
    keys = []
    for line in GEN_LOG.read_text(encoding="utf-8").splitlines():
        try:
            keys.append(json.loads(line)["pair_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return set(keys)


def _append_log(record: dict) -> None:
    with GEN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _run_single(
    llm, paper: dict, pages_by_source: dict, done: set[str], force: bool
) -> list[dict]:
    out = []
    for slot in SLOTS:
        pair_id = f"single:{paper['slug']}:{slot['key']}"
        if pair_id in done and not force:
            continue
        print(f"  {pair_id}", flush=True)
        try:
            s1 = await stage1_single(llm, paper, slot)
            spans = await stage2(
                llm, s1["question"], s1["gold_answer"],
                [paper["source"]], pages_by_source,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    FAIL {type(exc).__name__}: {exc}", flush=True)
            continue
        rec = {
            "pair_id": pair_id,
            "paper": paper["slug"],
            "slot": slot["key"],
            "cross_paper": False,
            "question": s1["question"],
            "gold_answer": s1["gold_answer"],
            "difficulty": s1.get("difficulty", slot["difficulty"]),
            "gold_sources": [paper["source"]],
            "gold_spans": spans,
            "ok": bool(spans),
        }
        _append_log(rec)
        out.append(rec)
    return out


async def _run_cross(
    llm, papers_by_slug: dict, combos: list[list[str]], pages_by_source: dict,
    done: set[str], force: bool,
) -> list[dict]:
    out = []
    for idx, combo_slugs in enumerate(combos):
        pair_id = f"cross:{idx}"
        if pair_id in done and not force:
            continue
        combo = [papers_by_slug[s] for s in combo_slugs]
        difficulty = "hard" if len(combo) >= 3 else "medium"
        print(f"  {pair_id} [{', '.join(combo_slugs)}]", flush=True)
        try:
            s1 = await stage1_cross(llm, combo, difficulty)
            sources = [p["source"] for p in combo]
            spans = await stage2(
                llm, s1["question"], s1["gold_answer"], sources, pages_by_source
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    FAIL {type(exc).__name__}: {exc}", flush=True)
            continue
        rec = {
            "pair_id": pair_id,
            "papers": combo_slugs,
            "cross_paper": True,
            "question": s1["question"],
            "gold_answer": s1["gold_answer"],
            "difficulty": s1.get("difficulty", difficulty),
            "gold_sources": sources,
            "gold_spans": spans,
            "ok": bool(spans),
        }
        _append_log(rec)
        out.append(rec)
    return out


def _write_dev(records: list[dict] | None = None) -> None:
    """Rebuild ``dev.jsonl`` from the ok records in ``gen_log.jsonl``.

    ``gen_log.jsonl`` is the source of truth (incremental, resume-safe), so the
    output always reflects every successfully anchored pair regardless of which
    ones were produced by this particular run. ``records`` is accepted for
    backward compatibility but ignored.
    """
    good = [
        json.loads(line)
        for line in GEN_LOG.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    good = [r for r in good if r.get("ok")]
    with DEV_FILE.open("w", encoding="utf-8") as fh:
        for r in sorted(good, key=lambda r: r["pair_id"]):
            out = {
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "difficulty": r["difficulty"],
                "cross_paper": r["cross_paper"],
                "gold_sources": r["gold_sources"],
                "gold_spans": r["gold_spans"],
                "_meta": {"paper": r.get("paper"), "slot": r.get("slot"),
                          "papers": r.get("papers")},
            }
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="qwen/qwen3.7-plus")
    ap.add_argument("--single-n", type=int, default=80)
    ap.add_argument("--cross-n", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    corpus = load_corpus()
    papers = load_topics()
    papers_by_slug = {p["slug"]: p for p in papers}
    pages_by_source = corpus["pages"]

    combos = sample_cross_pairs(papers, seed=args.seed)
    print("cross-paper combos:")
    for c in combos:
        print("  ", c, flush=True)

    llm = load_chat_model(args.model)
    print(f"model: {args.model}", flush=True)

    done = set() if args.force else _load_existing()
    if args.force:
        # rewrite log fresh
        GEN_LOG.write_text("", encoding="utf-8")

    records: list[dict] = []
    single_n = min(args.single_n, len(papers) * len(SLOTS))
    cross_n = min(args.cross_n, len(combos))

    for paper in papers:
        if sum(1 for r in records if not r["cross_paper"]) >= single_n:
            break
        records.extend(
            await _run_single(llm, paper, pages_by_source, done, args.force)
        )

    records.extend(
        await _run_cross(
            llm, papers_by_slug, combos[:cross_n], pages_by_source,
            done, args.force,
        )
    )

    _write_dev(records)
    good = sum(1 for r in records if r.get("ok"))
    print(f"\n=== summary === total done {len(records)}, ok {good}, "
          f"dev.jsonl written with {good} pairs")
    for r in records:
        mark = "ok" if r.get("ok") else "NO-SPANS"
        srcs = r.get("papers") or [r.get("paper")]
        print(f"  [{mark}] {r['pair_id']:<34} diff={r['difficulty']} "
              f"n_span={len(r.get('gold_spans') or [])} srcs={srcs}")


if __name__ == "__main__":
    asyncio.run(main())
