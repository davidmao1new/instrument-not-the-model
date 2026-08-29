r"""No figure may carry a cross-reference into a paper.

WHY THIS EXISTS. The variance-components figure's x-axis ended with
"see §6.2". That is correct in the preprint, where §6.2 is the section on
the log-odds to percentage-point conversion. The same image file ships in
the ICLR fork, which has no subsections at all, so a submitted paper
contained a pointer that resolved to nothing.

A caption can be rewritten for each venue, and this project already has a
test that checks the ported captions. Text drawn inside the image cannot be
rewritten without regenerating the figure, and nobody looks at an axis label
twice. So the rule is that a figure must be self-contained: it may name a
quantity, a unit, a transform or a caveat, but it may not point at a
numbered part of a document it might not be bound into.

WHAT IT SCANS. Every figure PDF the papers actually include, in both the
preprint's figure directory and the fork's copy of it.

    sh paper-a/src/_py.sh paper-a/src/audit_figure_refs.py
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

# ONLY THE PORTED COPIES. paper-a/figures/ holds the preprint's own figure
# files, which draw their captions into the image, and a caption in the
# preprint is entitled to reference the preprint. The directories below hold
# the caption-free copies the LaTeX venues include and give their own
# captions to, so anything drawn inside those images has to stand alone.
DIRS = [
    ROOT / "paper-a" / "iclr" / "figures",
    ROOT / "paper-a" / "facct" / "figures",
]

# A pointer into a numbered document. "§6.2", "Section 6.2", "see Section 4",
# "Table 3", "Figure 2" -- all of them assume a surrounding structure the
# image may not be bound into.
#
# CASE, PLURAL AND ABBREVIATION. This rule was case-sensitive, singular and
# spelled out, so "Fig. 2", "figures 2 and 3" and "see table 3" all passed --
# the three forms a figure is most likely to draw, because an axis label has
# less room than a sentence. The plural failure is the one that defeated the
# section-pointer audit too: \bFigure\s+ cannot match "Figures", because the
# "s" is exactly where the space has to be.
#
# The abbreviations require their period. "sec" is also the abbreviation for
# seconds and "tab" for a user-interface tab, and a tick label reading "5 sec"
# is not a cross-reference. None of the fourteen ported figures contains any
# of these words in any case today, so this widening flags nothing new right
# now; the period is what keeps it quiet if a timing axis is ever added.
REF = re.compile(
    r"(?:§\s*\d+(?:\.\d+)*"
    r"|\bSections?\s+\d+(?:\.\d+)*"
    r"|\bSecs?\.\s*\d+(?:\.\d+)*"
    r"|\bTables?\s+\d+"
    r"|\bTabs?\.\s*\d+"
    r"|\bFigures?\s+\d+"
    r"|\bFigs?\.\s*\d+"
    r"|\bAppendix\s+[A-Z]\b"
    r"|\bAppendices\s+[A-Z]\b"
    r"|\bApps?\.\s*[A-Z]\b)", re.I)

# The built papers live in the same directory as the figures and are not
# figures: a paper is entitled to reference its own sections. Matched by
# prefix so the condensed variant and any future build are covered without
# anyone remembering to add them.
SKIP_PREFIXES = ("paper_instrument_validity",)


def main() -> int:
    import fitz
    problems: list[str] = []
    scanned = 0
    for d in DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.pdf")):
            if f.name.startswith(SKIP_PREFIXES):
                continue
            scanned += 1
            try:
                with fitz.open(f) as doc:
                    text = " ".join(" ".join(p.get_text().split())
                                    for p in doc)
            except Exception as e:  # noqa: BLE001
                problems.append(f"{f.name}: unreadable ({e})")
                continue
            for m in REF.finditer(text):
                i = m.start()
                problems.append(
                    f"{f.relative_to(ROOT).as_posix()}: {m.group()!r} in "
                    f"...{text[max(0, i - 55):i + 35].strip()}...")

    print("=" * 74)
    print("CROSS-REFERENCES DRAWN INSIDE FIGURES")
    print("=" * 74)
    print(f"  {scanned} figure(s) scanned in {len(DIRS)} directories")
    if problems:
        print("-" * 74)
        for p in problems:
            print("  " + p)
        print("=" * 74)
        print(f"{len(problems)} reference(s) baked into a figure. A caption "
              "can be rewritten per venue; an axis label cannot.")
        return 1
    print("=" * 74)
    print("clean: every figure is self-contained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
