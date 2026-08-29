"""Count the tics that make prose read as machine-written.

WHY THIS IS A CHECK AND NOT A MATTER OF TASTE. A reviewer's judgement that
writing "sounds like an LLM" is not usually about vocabulary; it is about
DENSITY. Any one antithesis ("this is not X, it is Y") is good writing. Forty of
them in twenty-nine pages is a mannerism, and a reader feels it as monotony long
before naming it. Density is countable, so it is audited rather than argued
about.

The patterns below are the ones that actually recur in machine-drafted academic
prose. Each is legitimate in isolation; each becomes a tell above a rate.
Thresholds are per 1,000 words and are deliberately generous -- they flag
habits, not sentences.

  antithesis      "not X, it is Y" / "X is not Y; it is Z". The single most
                  over-used device in LLM expository writing.
  em_dash         Parenthetical dashes. Human academic prose averages far
                  below what an unedited draft produces.
  which_is        "which is the point", "which is what X exists to Y" --
                  self-commentary appended to a finished clause.
  importantly     Sentence-initial "Crucially/Importantly/Notably/Critically".
  it_is_worth     "It is worth noting", "It should be noted", "It bears
                  emphasis".
  triads          Three parallel items in a row, repeatedly.
  showcase_verbs  "underscores", "highlights", "showcases", "delves",
                  "leverages", "sheds light on".
  hedge_stack     Two or more hedges in one clause ("may potentially",
                  "could arguably suggest").

WHAT THIS DOES NOT DO. It does not rewrite anything, and it does not know
whether a given instance is earned. It reports where to look.

    sh paper-a/src/_py.sh paper-a/src/audit_prose_style.py
"""
from __future__ import annotations

import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"

# (name, pattern, per-1000-word ceiling, note)
PATTERNS = [
    ("antithesis",
     r"\b(?:is|was|are|were)\s+not\s+[^.;,]{1,60}[,;]\s*(?:it|they|that)\s+(?:is|was|are|were)\b",
     1.0, "'X is not Y, it is Z' -- vary the construction"),
    ("not_but_rather",
     r"\bnot\s+[^.;]{1,50}\s+but\s+rather\b", 0.5, "'not X but rather Y'"),
    # THE CEILING IS MEASURED, NOT CHOSEN. Across the 23 full-text papers in
    # lit/text/ over 3,000 words, em-dash density runs 0.00 to 3.26 per 1,000
    # words with a median of 0.35: Bertrand and Mullainathan 0.42, Sclar et al.
    # 0.00, Gaebler et al. 3.26 at the top. A draft of this paper sat at 10.07,
    # which is the single most visible way its prose differed from the
    # literature it is joining. The ceiling is the corpus maximum, so the paper
    # is asked only to stay inside the range the field actually uses.
    ("em_dash", r"—", 3.26, "parenthetical dashes; field max is 3.26/1k, "
                            "median 0.35"),
    ("which_is_self",
     r",\s*which\s+is\s+(?:the\s+point|what|why|exactly|precisely)\b",
     0.6, "self-commentary tacked onto a finished clause"),
    ("importantly",
     r"(?:^|\.\s+)(?:Crucially|Importantly|Notably|Critically|Significantly)\b",
     0.4, "sentence-initial emphasis adverb"),
    ("it_is_worth",
     r"\bIt (?:is|would be) worth (?:noting|emphasi[sz]ing)\b|\bIt should be noted\b",
     0.2, "empty throat-clearing"),
    ("showcase_verbs",
     r"\b(?:underscore[sd]?|highlight[sd]?|showcase[sd]?|delve[sd]?|leverag(?:e|es|ed|ing)|sheds? light on)\b",
     0.6, "reviewer-bait verbs"),
    ("hedge_stack",
     r"\b(?:may|might|could)\s+(?:potentially|arguably|possibly|conceivably)\b",
     0.2, "stacked hedges"),
    ("in_order_to", r"\bin order to\b", 0.6, "'to' is usually enough"),
    ("the_fact_that", r"\bthe fact that\b", 0.6, "usually deletable"),
]


def main() -> int:
    if not PDF.exists():
        sys.exit("paper not built")
    import fitz  # noqa: PLC0415
    with fitz.open(PDF) as doc:
        text = "\n".join(p.get_text() for p in doc)
    # A TABLE CELL IS NOT PROSE. Suppressed cells render as a lone em dash --
    # 25 of them, where a ratio has a denominator covering zero -- and counting
    # those as a stylistic tic would penalise the paper for the very
    # suppression an earlier critique round required. Drop lines that are
    # nothing but a dash before measuring.
    text = "\n".join(ln for ln in text.split("\n") if ln.strip() != "—")
    flat = " ".join(text.split())
    words = len(flat.split())
    k = words / 1000.0

    print(f"{PDF.name}: {words:,} words\n")
    print(f"{'pattern':<16}{'count':>7}{'per 1k':>9}{'ceiling':>9}  note")
    print("-" * 78)
    over = []
    for name, pat, ceiling, note in PATTERNS:
        n = len(re.findall(pat, flat, re.I if name != "importantly" else 0))
        rate = n / k
        flag = " OVER" if rate > ceiling else ""
        if rate > ceiling:
            over.append((name, n, rate, ceiling, note))
        print(f"{name:<16}{n:>7}{rate:>9.2f}{ceiling:>9.2f}  {note}{flag}")

    # Sentence length: variety matters more than the mean. A draft with a
    # narrow spread reads as machine-paced even when every sentence is fine.
    sents = [s for s in re.split(r"(?<=[.!?])\s+", flat) if len(s.split()) > 2]
    lens = sorted(len(s.split()) for s in sents)
    if lens:
        import statistics
        mean = statistics.mean(lens)
        sd = statistics.pstdev(lens)
        print(f"\n  sentences {len(lens):,}   mean {mean:.1f} words   "
              f"sd {sd:.1f}   short(<12) {sum(1 for x in lens if x < 12)}   "
              f"long(>40) {sum(1 for x in lens if x > 40)}")
        print(f"  length variety (sd/mean) {sd / mean:.2f}  "
              f"-- below about 0.55 reads as uniform")

    print()
    if over:
        print(f"  {len(over)} pattern(s) above ceiling: "
              + ", ".join(o[0] for o in over))
    else:
        print("  no pattern above its ceiling")
    return 0


if __name__ == "__main__":
    sys.exit(main())
