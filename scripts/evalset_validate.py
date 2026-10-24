"""Validate ``evalset/dev.jsonl``: schema + structure (C9) and paraphrase
distance (C8), per ``need2modify.md`` §7.

C9 — per-pair schema (§7.3) and cross-set structure (§7.4):
    * required fields, difficulty enum, gold_sources/gold_spans shape
    * every span is a verbatim excerpt of its corpus page
    * cross_paper=true  -> spans cover >= 2 distinct sources
    * cross_paper=false -> exactly one source, all spans from it
    * 80 single (20 papers x 4 slots: definition/method/synthesis/challenge)
      + 20 cross; every paper appears in >=1 cross pair and <=4; cross pairs
      stay inside one subtopic bucket

C8 — paraphrase distance (the §7.3 命门): the question must not word-copy the
evidence. For each pair we report the fraction of the question's content tokens
(paper-title tokens excluded) that also appear in the gold spans. A high value
means retrieval would hit trivially and the pair loses all discriminating power.
§7.3 wants at least half of the single-paper pairs to be lexically distant.

Usage:
    PYTHONPATH=src python scripts/evalset_validate.py [dev.jsonl]
Exits non-zero on hard structural failures; prints the paraphrase-distance
distribution and the worst offenders for review.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "evalset" / "corpus"
TOPICS_FILE = ROOT / "evalset" / "topics.json"
DEV_FILE = ROOT / "evalset" / "dev.jsonl"

SLOT_KEYS = {"definition", "method", "synthesis", "challenge"}


def norm(s: str) -> str:
    """Canonical text (matches generate_evalset.norm): join hyphenated
    line-breaks, collapse whitespace, drop hyphens, lowercase."""
    s = s.replace("\r", " ")
    s = re.sub(r"-\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.replace("-", "").strip().lower()


def tokenize(s: str) -> set[str]:
    STOP = {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with",
        "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
        "these", "those", "as", "at", "by", "from", "we", "our", "they", "their",
        "how", "what", "which", "why", "can", "does", "do", "not", "no", "but",
        "such", "than", "then", "via", "using", "use", "uses", "used", "between",
        "does", "would", "could", "should",
    }
    return {w for w in re.findall(r"[A-Za-z]+", s.lower()) if w not in STOP and len(w) > 2}


def load_corpus() -> dict:
    pages: dict[str, dict[int, str]] = {}
    for f in CORPUS_DIR.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        pages[d["source"]] = {p["page"]: p["text"] for p in d["pages"]}
    return pages


def load_topics() -> list[dict]:
    return json.loads(TOPICS_FILE.read_text(encoding="utf-8"))["papers"]


def main() -> int:
    dev_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEV_FILE
    pairs = [json.loads(l) for l in dev_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    pages = load_corpus()
    topics = load_topics()
    bucket_of = {p["slug"]: p["bucket"] for p in topics}
    title_toks = {p["slug"]: tokenize(p["title"]) for p in topics}
    slug_by_source = {p["source"]: p["slug"] for p in topics}

    errors: list[str] = []
    warnings: list[str] = []
    overlap_rows: list[tuple[float, str, str]] = []  # (overlap, pair_id, question)

    single_by_paper: dict[str, list[str]] = defaultdict(list)
    cross_sets: list[list[str]] = []

    for i, p in enumerate(pairs):
        pid = p.get("_meta", {}).get("paper") or p.get("_meta", {}).get("papers")
        label = f"#{i} {pid}"
        # ── C9 schema ──────────────────────────────────────────────────────
        for k in ("question", "gold_answer", "difficulty", "cross_paper",
                  "gold_sources", "gold_spans"):
            if k not in p:
                errors.append(f"{label}: missing {k}")
        if p.get("difficulty") not in ("easy", "medium", "hard"):
            errors.append(f"{label}: difficulty={p.get('difficulty')!r}")
        spans = p.get("gold_spans")
        if not isinstance(spans, list) or not (2 <= len(spans) <= 5):
            errors.append(f"{label}: gold_spans must be 2..5, got {len(spans) if isinstance(spans, list) else spans!r}")
        sources = p.get("gold_sources")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{label}: gold_sources empty")

        if not isinstance(spans, list):
            continue
        for s in spans:
            src, pg, txt = s.get("source"), s.get("page"), s.get("text")
            if not (isinstance(src, str) and isinstance(pg, int) and isinstance(txt, str) and txt.strip()):
                errors.append(f"{label}: malformed span {s}")
                continue
            if src not in pages:
                errors.append(f"{label}: span source not in corpus: {src}")
                continue
            if pg not in pages[src]:
                errors.append(f"{label}: span page {pg} not in corpus page range")
                continue
            if len(norm(txt)) < 30 or norm(txt) not in norm(pages[src][pg]):
                errors.append(f"{label}: span not verbatim (page {pg}): {txt[:60]!r}")

        span_srcs = {s["source"] for s in spans} if isinstance(spans, list) else set()
        if p.get("cross_paper"):
            cross_sets.append(list(pid or []))
            if len(span_srcs) < 2:
                errors.append(f"{label}: cross_paper but spans cover only {len(span_srcs)} source(s)")
            if sources and not (set(span_srcs) <= set(sources)):
                errors.append(f"{label}: spans reference sources outside gold_sources")
        else:
            meta = p.get("_meta", {})
            paper = meta.get("paper")
            slot = meta.get("slot")
            if paper and slot:
                single_by_paper[paper].append(slot)
            if sources and len(sources) != 1:
                errors.append(f"{label}: single-paper pair has {len(sources)} gold_sources")
            if span_srcs and sources and len(span_srcs & set(sources)) == 0:
                errors.append(f"{label}: single spans not from gold_sources")

        # ── C8 paraphrase distance ─────────────────────────────────────────
        involved_slugs = p.get("_meta", {}).get("papers") or ([p.get("_meta", {}).get("paper")] if p.get("_meta", {}).get("paper") else [])
        allow = set()
        for slug in involved_slugs:
            allow |= title_toks.get(slug, set())
        q = tokenize(p.get("question", "")) - allow
        ev = set()
        for s in (spans or []):
            ev |= tokenize(s.get("text", ""))
        if q:
            ov = len(q & ev) / len(q)
            overlap_rows.append((ov, f"#{i} {involved_slugs}", p.get("question", "")))
            if p.get("_meta", {}).get("slot") == "definition":
                pass  # definition slots may name the concept; not judged
            elif ov >= 0.4:
                warnings.append(
                    f"{label}: question word-copies evidence (overlap={ov:.2f})"
                )

    # ── C9 cross-set structure (§7.4) ──────────────────────────────────────
    n_single = sum(1 for p in pairs if not p.get("cross_paper"))
    n_cross = sum(1 for p in pairs if p.get("cross_paper"))
    if n_single != 80:
        errors.append(f"expected 80 single-paper pairs, got {n_single}")
    if n_cross != 20:
        errors.append(f"expected 20 cross-paper pairs, got {n_cross}")

    for paper, slots in single_by_paper.items():
        missing = SLOT_KEYS - set(slots)
        dupes = [s for s, c in Counter(slots).items() if c > 1]
        if missing:
            errors.append(f"single {paper}: missing slots {sorted(missing)}")
        if dupes:
            errors.append(f"single {paper}: duplicate slots {dupes}")

    if cross_sets:
        cross_cnt = Counter(s for combo in cross_sets for s in combo)
        unused = [s for s, c in cross_cnt.items() if c == 0]
        over = {s: c for s, c in cross_cnt.items() if c > 4}
        for combo in cross_sets:
            buckets = {bucket_of[s] for s in combo if s in bucket_of}
            if len(buckets) != 1:
                errors.append(f"cross combo {combo} spans buckets {buckets}")
        if unused:
            errors.append(f"papers never in a cross pair: {unused}")
        if over:
            errors.append(f"papers over cap in cross pairs: {over}")

    # ── report ─────────────────────────────────────────────────────────────
    print(f"pairs: {len(pairs)}  (single {n_single}, cross {n_cross})")
    print(f"errors: {len(errors)}  warnings: {len(warnings)}")
    for e in errors[:40]:
        print("  [ERR]", e)
    for w in warnings[:20]:
        print("  [WARN]", w)

    if overlap_rows:
        single_ov = [ov for ov, _, _ in overlap_rows]
        frac_lexical = sum(1 for ov in single_ov if ov < 0.2) / len(single_ov)
        print(f"\nparaphrase distance: mean overlap={sum(single_ov)/len(single_ov):.2f} "
              f"| pairs with overlap<0.2 (lexically distant): {frac_lexical:.0%}")
        print("worst offenders (question word-copies evidence):")
        for ov, label, q in sorted(overlap_rows, reverse=True)[:8]:
            print(f"  {ov:.2f}  {label}: {q[:110]}")

    report = ROOT / "evalset" / "validate_report.txt"
    report.write_text(
        "\n".join([f"pairs: {len(pairs)} (single {n_single}, cross {n_cross})",
                   f"errors: {len(errors)}  warnings: {len(warnings)}",
                   *[f"ERR {e}" for e in errors],
                   *[f"WARN {w}" for w in warnings]]),
        encoding="utf-8",
    )
    print(f"\nreport -> {report}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
