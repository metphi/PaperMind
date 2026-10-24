"""Re-anchor the failed stage-2 pairs: candidate-region prompting.

Why: the original stage-2 fed the ENTIRE paper (14 pages x 4000 chars). Qwen,
faced with a wall of text, paraphrases or embellishes instead of copying
verbatim, so the strict substring validator rejects every span (14 pairs all
ended with gold_spans=[]). ``evalset_diagnose.py`` confirms the answer content
IS in the body (57-100% token overlap) and that PDF line-break hyphens
("Galerkin-\\ntype") are already normalized by ``norm()`` -- so the fix is
prompt design, not text cleaning.

New approach per failed pair:
  1. find candidate regions: score each page by answer-token overlap, then
     extract the densest word-windows (the "long paragraph") from the top pages.
  2. send Qwen ONLY: question + gold_answer + those candidate paragraphs, with a
     hardened prompt: quote EXACTLY from the given text, never fabricate, and
     return {"gold_spans": []} when the claim is absent.
  3. validate verbatim as before; >=2 spans covering all sources -> ok.

Only single-paper pairs are re-anchored here (all 14 failures are single-paper).

Usage:
    PYTHONPATH=src python scripts/evalset_reanchor.py [--model qwen/qwen3.7-plus]
    PYTHONPATH=src python scripts/evalset_reanchor.py --only single:SRNO:method   # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

import generate_evalset as ge  # noqa: E402  (reuses norm / _validate_span / _invoke_json / loaders)
from shared.utils import load_chat_model  # noqa: E402

SYSTEM = """You are an evidence-annotation assistant. Given a question, its gold answer, and candidate verbatim paragraphs from the paper, you locate verbatim text excerpts (spans) that together are sufficient to derive the gold answer.

HARD RULES:
- Return between 2 and 5 spans. Do NOT return fewer than 2: even if one paragraph would suffice, still give at least two separate excerpts -- e.g. the sentence stating the mechanism and a second sentence/page stating its property or evidence. Spans may come from different paragraphs or different pages.
- Spans must be DISJOINT (no nested or overlapping excerpts).
- Copy spans EXACTLY from the candidate paragraphs below: same words, same order, character-for-character. NO paraphrasing, NO rewording, NO adding words.
- The candidate paragraphs are the ONLY text you may quote. Never quote or invent text not present in them.
- If the gold answer's claim is NOT supported by the paragraphs, return {"gold_spans": []}. Never fabricate or embellish facts to force a match.
- PDF line-break hyphens: the page text may split a word across a line with a hyphen, e.g. "Galerkin-\\ntype". When you copy such a word, write the joined word without the hyphen and break ("Galerkin-type").
- Return ONLY JSON: {"gold_spans": [{"source": str, "page": int, "text": str}]}"""

PROMPT = """Question: {question}
Gold answer: {gold_answer}

Candidate paragraphs (verbatim, from the paper -- quote from these only):
{blocks}"""

STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "these", "those", "as", "at", "by", "from", "we", "our", "they", "their",
    "how", "what", "which", "why", "can", "does", "do", "not", "no", "but",
    "such", "than", "then", "via", "using", "use", "uses", "used", "between",
}


def _answer_words(answer: str) -> set[str]:
    """Plain content words of the answer (hyphen-splitting like tokenize)."""
    return {w for w in re.findall(r"[A-Za-z]+", answer.lower())
            if w not in STOP and len(w) > 2}


def _find_regions(raw: str, ans_words: set[str], max_regions: int = 2,
                  pad: int = 130, cap: int = 1800) -> list[tuple[int, str]]:
    """Return the densest (matched_word_count, verbatim_window) regions of a page.

    Windows are cut on the RAW page text (preserving exact chars for Qwen to copy).
    A seed is a run of >=2 consecutive matched content words (a phrase-like signal);
    clustered seeds form a region, expanded by ``pad`` chars and trimmed to word/newline
    boundaries and ``cap`` length.
    """
    words = [
        (m.group(0).lower().strip("'-"), m.start(), m.end())
        for m in re.finditer(r"[A-Za-z][A-Za-z'\-]*", raw)
    ]
    if not words:
        return []
    matched = []
    for w, _s, _e in words:
        parts = [p for p in w.split("-") if p]
        matched.append(bool(ans_words & set(parts)) if parts else False)
    seeds = [i for i in range(len(words) - 1) if matched[i] and matched[i + 1]]
    if not seeds:
        seeds = [i for i in range(len(words)) if matched[i]]
    if not seeds:
        return []
    clusters: list[list[int]] = []
    cur = [seeds[0], seeds[0]]
    for s in seeds[1:]:
        if s - cur[1] <= 8:
            cur[1] = s
        else:
            clusters.append(cur)
            cur = [s, s]
    clusters.append(cur)

    scored: list[tuple[int, int, int]] = []
    for a, b in clusters:
        n_m = sum(matched[a:b + 1])
        start = max(0, words[a][1] - pad)
        end = min(len(raw), words[b][2] + pad)
        # trim to word boundaries (start of first word / end of last word inside window)
        while start < len(raw) and not raw[start].isalpha():
            start += 1
        while end > start and not raw[end - 1].isalpha():
            end -= 1
        if end - start > cap:
            cut = raw.rfind("\n", start, start + cap)
            end = start + cap if cut < start else cut
        scored.append((n_m, start, end))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(n_m, raw[s:e]) for n_m, s, e in scored[:max_regions] if e > s + 30]


def _candidate_blocks(source: str, pages_of: dict[int, str], ans_words: set[str],
                      max_pages: int = 2, top_regions: int = 3) -> list[dict]:
    """Pick top pages by answer-token overlap, extract their densest regions."""
    page_scores = []
    for pg, text in pages_of.items():
        ov = len(ans_words & ge.tokenize(text))
        if ov:
            page_scores.append((ov, pg, text))
    page_scores.sort(key=lambda x: x[0], reverse=True)

    regions: list[tuple[int, int, str]] = []  # (matched, page, text)
    for ov, pg, text in page_scores[:max_pages]:
        for n_m, win in _find_regions(text, ans_words):
            regions.append((n_m, pg, win))
    regions.sort(key=lambda x: x[0], reverse=True)
    return [
        {"source": source, "page": pg, "text": win}
        for n_m, pg, win in regions[:top_regions]
    ]


def _blocks_block(blocks: list[dict]) -> str:
    return "\n\n".join(
        f"=== SOURCE {b['source']} page {b['page']} ===\n{b['text']}" for b in blocks
    )


async def reanchor(llm, rec: dict) -> dict:
    src = rec["gold_sources"][0]
    corpus = ge.load_corpus()
    pages_of = corpus["pages"][src]
    ans_words = _answer_words(rec["gold_answer"])
    blocks = _candidate_blocks(src, pages_of, ans_words)

    print(f"    candidate pages: {sorted({b['page'] for b in blocks})}  "
          f"blocks={len(blocks)}  chars={sum(len(b['text']) for b in blocks)}",
          flush=True)

    for _ in range(4):
        user = PROMPT.format(
            question=rec["question"], gold_answer=rec["gold_answer"],
            blocks=_blocks_block(blocks),
        )
        obj = await ge._invoke_json(llm, SYSTEM, user, lambda o: None, attempts=3)
        spans = obj.get("gold_spans")
        if not isinstance(spans, list):
            continue
        valid = [s for s in spans
                 if ge._validate_span(s, corpus["pages"], rec["gold_sources"])]
        seen = set()
        dedup = []
        for s in valid:
            key = (s["source"], s["page"], ge.norm(s["text"]))
            if key not in seen:
                seen.add(key)
                dedup.append(s)
        covered = {s["source"] for s in dedup}
        if len(dedup) >= 2 and set(rec["gold_sources"]) <= covered:
            return dict(rec, gold_spans=dedup[:5], ok=True)
    return dict(rec, gold_spans=[], ok=False)


def _rewrite_log(replaced: list[dict]) -> None:
    """Remove the old failed records for these pair_ids, append new ones, sort."""
    ids = {r["pair_id"] for r in replaced}
    keep = [
        json.loads(line)
        for line in ge.GEN_LOG.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("pair_id") not in ids
    ]
    keep.extend(replaced)
    keep.sort(key=lambda r: r["pair_id"])
    with ge.GEN_LOG.open("w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen3.7-plus")
    ap.add_argument("--only", action="append", help="re-anchor only these pair_ids (repeatable)")
    args = ap.parse_args()

    failed = [
        json.loads(l)
        for l in ge.GEN_LOG.read_text(encoding="utf-8").splitlines()
        if l.strip() and not json.loads(l).get("ok")
    ]
    if args.only:
        want = set(args.only)
        failed = [r for r in failed if r["pair_id"] in want]
    if not failed:
        print("no failed pairs to re-anchor")
        return

    llm = load_chat_model(args.model)
    print(f"model: {args.model}   re-anchoring {len(failed)} pairs", flush=True)

    replaced = []
    for rec in failed:
        print(f"\n### {rec['pair_id']}", flush=True)
        new = await reanchor(llm, rec)
        mark = "OK" if new["ok"] else "STILL-FAIL"
        print(f"    {mark}  n_span={len(new['gold_spans'])}", flush=True)
        if new["ok"]:
            for s in new["gold_spans"]:
                print(f"      - p{s['page']}: {s['text'][:80]!r}", flush=True)
        replaced.append(new)

    _rewrite_log(replaced)
    ge._write_dev()
    good = sum(1 for r in replaced if r["ok"])
    print(f"\n=== done === re-anchored {len(replaced)}, ok {good}; "
          f"dev.jsonl rebuilt")


if __name__ == "__main__":
    asyncio.run(main())
