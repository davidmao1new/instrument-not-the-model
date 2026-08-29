"""How much of the paper is the paper talking about its own earlier drafts?

WHERE THE WORDS WENT. This paper is roughly 28,000 words against a field median
near 10,500. Some of that excess is real: it reports eleven studies where a
typical paper reports one. But a large share is a habit the critique loop
created. Each round found a defect; each fix was written into the paper as a
narrated correction -- "an earlier draft said X, which was wrong twice over",
"we reported that every interval excluded zero; so it does, and that is
arithmetic rather than evidence", "it was found by an adversarial audit of this
paper, not by us".

That habit was right at the time and it is the reason the paper is trustworthy.
It is also, at this volume, a tax on every reader who was not present for the
argument. A reader wants to know what is true; the record of what we briefly
believed belongs in the changelog, which exists, is released, and is where a
sceptic will look.

WHAT THIS COUNTS. Sentences carrying the vocabulary of self-correction, and the
share of the document they occupy. It does not decide what to cut -- some of
these earn their place, particularly where the correction IS the finding (the
estimator that carried its own effect, §10.1) or where a reader would otherwise
repeat our mistake. It says how large the category is, so the decision is made
against a number instead of an impression.

    sh paper-a/src/_py.sh paper-a/src/audit_correction_narration.py
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

MARKERS = [
    ("earlier draft", r"\ban earlier (?:draft|version)\b"),
    ("previous version", r"\b(?:a |the )?previous version\b|\bwe previously\b"),
    ("we first wrote", r"\bwe (?:first|originally) (?:wrote|said|reported|"
                       r"claimed|made)\b|\bthan the one we first\b"),
    ("we said/reported", r"\bwe (?:said|reported|wrote|claimed) that\b"),
    ("withdrawn", r"\bis withdrawn\b|\bwe withdraw\b|\bhas been withdrawn\b"),
    ("an audit of this paper", r"\ban? (?:adversarial )?(?:audit|critique) of "
                               r"this paper\b|\ba (?:critique|reviewer) "
                               r"(?:of this paper )?(?:caught|pointed|found)\b"),
    ("was wrong", r"\bwas wrong\b|\bis wrong\b|\bgot (?:it|that) wrong\b"),
    ("corrected", r"\bwe (?:corrected|report the correction)\b|"
                  r"\bthe correction\b"),
    ("not by us", r"\bnot by us\b|\bfound by an\b"),
]


def main() -> int:
    if not PDF.exists():
        sys.exit("paper not built")
    import fitz  # noqa: PLC0415
    with fitz.open(PDF) as doc:
        text = "\n".join(p.get_text() for p in doc)
    flat = " ".join(text.split())
    sents = [s for s in re.split(r"(?<=[.!?])\s+", flat) if s.split()]
    total_w = sum(len(s.split()) for s in sents)

    tagged, per_marker = [], {}
    for s in sents:
        hit = None
        for name, pat in MARKERS:
            if re.search(pat, s, re.I):
                hit = name
                break
        if hit:
            tagged.append((hit, s))
            per_marker[hit] = per_marker.get(hit, 0) + len(s.split())

    words = sum(len(s.split()) for _h, s in tagged)
    print(f"{PDF.name}: {total_w:,} words in {len(sents):,} sentences\n")
    print(f"{'marker':<26}{'sentences':>11}{'words':>8}{'% of paper':>12}")
    print("-" * 58)
    counts = {}
    for h, _s in tagged:
        counts[h] = counts.get(h, 0) + 1
    for name, _pat in MARKERS:
        if name not in counts:
            continue
        print(f"{name:<26}{counts[name]:>11}{per_marker[name]:>8}"
              f"{100 * per_marker[name] / total_w:>11.1f}%")
    print("-" * 58)
    print(f"{'TOTAL':<26}{len(tagged):>11}{words:>8}"
          f"{100 * words / total_w:>11.1f}%")
    print()
    print("  A field-median paper in lit/text/ is about 10,500 words; this one "
          f"is {total_w:,}.")
    print(f"  Removing the self-correction narration alone would take it to "
          f"about {total_w - words:,}.")
    print()
    print("  Longest instances, which are where the reading is:")
    for _h, s in sorted(tagged, key=lambda t: -len(t[1].split()))[:8]:
        print(f"    [{len(s.split()):>3}w] {s[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
