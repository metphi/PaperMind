"""Diagnose the 14 failed stage-2 pairs BEFORE re-anchoring.

Read-only, no Qwen calls. For each failed pair we answer:

  1. Is the gold_answer's content present in the paper's body at all?
     (per-page content-token overlap, norm-insensitive to hyphens)
  2. Which pages carry the strongest evidence (candidate pages for re-anchor)?
  3. Can the answer's key multi-word phrases be found verbatim (normed) in any page?
  4. Hyphenation scan: are answer tokens split across PDF line-breaks as
     hyphenated words in the matching region (e.g. "Galerkin-\ntype")?

This tells us whether each failure is a hyphenation artifact, an abstract-vs-body
wording drift, or a genuine fabrication (answer claims something the text never
says) -- which is what the strengthened re-anchor prompt must not mask.

Usage:
    PYTHONPATH=src python scripts/evalset_diagnose.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import generate_evalset as ge  # noqa: E402  (reuses norm / tokenize / loaders)


def main() -> None:
    corpus = ge.load_corpus()
    pages = corpus["pages"]
    topics = ge.load_topics()
    slug_by_source = {p["source"]: p["slug"] for p in topics}

    failed = [
        json.loads(l)
        for l in ge.GEN_LOG.read_text(encoding="utf-8").splitlines()
        if l.strip() and not json.loads(l).get("ok")
    ]
    print(f"failed pairs to diagnose: {len(failed)}\n")

    for rec in failed:
        pid = rec["pair_id"]
        src = rec["gold_sources"][0]
        slug = slug_by_source.get(src, "?")
        answer = rec["gold_answer"]
        ans_toks = ge.tokenize(answer)
        pages_of = pages.get(src, {})

        # per-page content-token overlap (norm-insensitive)
        scored = []
        for pg, text in pages_of.items():
            inter = ans_toks & ge.tokenize(text)
            scored.append((len(inter) / len(ans_toks), pg, inter))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        top3 = scored[:3]

        # whole-answer / long-phrase verbatim check (normed substring)
        n_ans = ge.norm(answer)
        verbatim_pages = [
            pg for pg, text in pages_of.items() if n_ans in ge.norm(text)
        ]
        # 6-gram content-word windows of the answer, verbatim?
        STOP = {
            "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with",
            "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
            "these", "those", "as", "at", "by", "from", "we", "our", "they", "their",
            "how", "what", "which", "why", "can", "does", "do", "not", "no", "but",
            "such", "than", "then", "via", "using", "use", "uses", "used", "between",
        }
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]*", answer.lower())
                 if w not in STOP and len(w) > 2]
        windows = [" ".join(words[i:i + 6]) for i in range(0, max(len(words) - 5, 1))]
        found = [w for w in windows if any(ge.norm(w) in ge.norm(t) for t in pages_of.values())]

        # hyphenation: is the top page's best region riddled with word-\n breaks?
        hyphens = 0
        if top3 and top3[0][0] > 0.2:
            best_pg_text = pages_of[top3[0][1]]
            hyphens = len(re.findall(r"[A-Za-z]-\s*\n", best_pg_text))

        print(f"### {pid}  (slug={slug})  diff={rec['difficulty']}")
        print(f"  answer: {answer[:110]}")
        print(f"  ans tokens: {len(ans_toks)}  "
              f"best page overlap={best[0]:.0%}  top pages={[p for _, p, _ in top3]}")
        print(f"  whole-answer verbatim pages: {verbatim_pages or 'NONE'}")
        print(f"  verbatim 6-grams: {len(found)}/{len(windows)}")
        if top3 and top3[0][0] > 0.2:
            pg = top3[0][1]
            print(f"  top page {pg} hyphen-break count: {hyphens}  (best shared tokens: "
                  f"{sorted(top3[0][2])[:8]})")
        else:
            print(f"  ** LOW OVERLAP: answer content weakly present in body **")
        print()

    # specific: does SRNO mention Galerkin anywhere?
    srno_src = next(s for s, p in slug_by_source.items() if p == "SRNO")
    gal = []
    for pg, text in pages.get(srno_src, {}).items():
        hits = [m.start() for m in re.finditer(r"[Gg]alerkin", text)]
        if hits:
            gal.append((pg, len(hits)))
    print(f"=== SRNO Galerkin mentions: {gal if gal else 'NONE in any page'} ===")
    for pg, text in pages.get(srno_src, {}).items():
        if re.search(r"[Gg]alerkin", text):
            i = re.search(r"[Gg]alerkin", text).start()
            print(f"  SRNO p{pg} context: ...{text[max(0, i-120):i+120]!r}")


if __name__ == "__main__":
    main()
