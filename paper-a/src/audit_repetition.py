"""How many times does the paper say each thing?

WHY THIS IS THE PLACE TO CUT. The paper is roughly 28,000 words against a field
median near 10,500, and the excess is not stylistic: the self-correction
narration everyone notices is only 5.7 % of it. It is structural. The document
has five places that each state the whole set of findings -- §1.2 contributions,
the §4 results themselves, §9 recommendations, §10 threats, §11 conclusion --
so a single result can be written out five times, each time with its own
hedges, before a reader has learned anything new.

Cutting hedges would undo seven rounds of critique. Cutting the fourth and
fifth statement of a finding costs nothing but length.

WHAT THIS MEASURES. Every distinct numeric value the paper prints, where it
appears, and how many separate sections repeat it. A value in one section is a
result. The same value in five sections is a result plus four reminders.

The output is a worklist ordered by how many words the repetitions occupy, so
the cut is made against evidence rather than by feel.

    sh paper-a/src/_py.sh paper-a/src/audit_repetition.py
"""
from __future__ import annotations

import collections
import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"

# A printed measurement: a percentage, a decimal, or an "n of m" tally. Bare
# integers are excluded because section numbers and counts of things swamp them.
VALUE = re.compile(r"\b\d+(?:\.\d+)?\s?%|\b\d+\.\d{2,4}\b|\b\d+ of \d+\b")

# Values that recur legitimately because they ARE the subject: the conventional
# interval and alpha, and the count of models.
BORING = {"95 %", "80 %", "0.05", "5 %"}


def sections(text: str):
    """(section label, span) for each numbered heading in reading order."""
    heads = [(m.start(), m.group(1))
             for m in re.finditer(r"(?m)^\s*(\d{1,2}(?:\.\d)?)\s+[A-Z][^\n]{2,60}$",
                                  text)]
    heads.append((len(text), "END"))
    for i in range(len(heads) - 1):
        yield heads[i][1], text[heads[i][0]:heads[i + 1][0]]


def main() -> int:
    if not PDF.exists():
        sys.exit("paper not built")
    import fitz  # noqa: PLC0415
    with fitz.open(PDF) as doc:
        text = "\n".join(p.get_text() for p in doc)

    where = collections.defaultdict(set)
    sent_of = collections.defaultdict(list)
    for label, body in sections(text):
        flat = " ".join(body.split())
        for s in re.split(r"(?<=[.!?])\s+", flat):
            for m in VALUE.finditer(s):
                v = " ".join(m.group(0).split())
                if v in BORING:
                    continue
                where[v].add(label.split(".")[0])
                sent_of[v].append((label, s))

    repeated = {v: secs for v, secs in where.items() if len(secs) >= 3}
    # words spent on the second and later statements
    waste = 0
    rows = []
    for v, secs in repeated.items():
        ss = sent_of[v]
        extra = sum(len(s.split()) for _l, s in ss[1:])
        waste += extra
        rows.append((extra, v, sorted(secs), len(ss)))
    rows.sort(reverse=True)

    total_w = len(" ".join(text.split()).split())
    print(f"{PDF.name}: {total_w:,} words\n")
    print(f"{len(repeated)} measurements appear in three or more sections.\n")
    print(f"{'words in repeats':>17}  {'value':<12}{'times':>6}  sections")
    print("-" * 74)
    for extra, v, secs, n in rows[:24]:
        print(f"{extra:>17}  {v:<12}{n:>6}  {', '.join(secs)}")
    print("-" * 74)
    print(f"{waste:>17}  total words in second-and-later statements "
          f"({100 * waste / total_w:.1f}% of the paper)")
    print()
    print("  These are not all cuttable: a result belongs in its own section "
          "and, once, in the abstract.")
    print("  What is cuttable is the third, fourth and fifth statement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
