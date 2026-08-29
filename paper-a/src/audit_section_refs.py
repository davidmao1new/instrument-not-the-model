r"""Every section a document points at must be a section the document has.

WHY THIS EXISTS. A restructure moved five analyses out of the numbered body
and into lettered appendices. The numbering closed over the vacated slots,
so the preprint ran 4.1, 4.2, 4.4, 4.5 and 5, 5.2, and nineteen references
in the prose plus six rows of the design table went on pointing at 3.1,
4.3, 4.6, 4.7, 4.7.1, 5.1 and 6.4. None of those existed. A reader who
followed the paper's own pointer to the evidence for its headline
reproducibility claim arrived nowhere.

Nothing caught it for three rounds. The number audits verify figures
against artifacts and a missing heading has no figure. The claim audits
read prose against data and a dangling pointer states nothing false. It
was found by listing the headings and noticing a gap, which is why this
check exists: it compares the set of pointers to the set of headings, and
that comparison is cheap and total.

WHAT IT CHECKS. Both papers. For each, every "SS n.n" reference in the
built PDF must correspond to a heading the same document defines. Statute
citations are excluded by pattern, not by exception: "SS 20-870" is a
section of the N.Y.C. Administrative Code, not of this paper.

    sh paper-a/src/_py.sh paper-a/src/audit_section_refs.py
"""
from __future__ import annotations

import pathlib
import re
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]

SECTION_SIGN = "§"

# The statute the paper is about is cited by section number and is not a
# pointer into the paper. Local Law 144 lives at N.Y.C. Admin. Code
# 20-870 to 20-874, and the rule at 6 RCNY 5-300.
STATUTE = re.compile(r"^(20-|5-|20$|5$)")

DOCS = [
    ("preprint",
     ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf",
     ROOT / "paper-a" / "src" / "build_paper_v3.py",
     re.compile(r'H\("([0-9A-G]+(?:\.[0-9]+)?)\s')),
    ("iclr fork",
     ROOT / "paper-a" / "iclr" / "build" / "main.pdf",
     None, None),
]


def headings_from_source(src: pathlib.Path, pat: re.Pattern) -> set[str]:
    return set(pat.findall(src.read_text(encoding="utf-8")))


def refs_from_pdf(pdf: pathlib.Path) -> Counter:
    import fitz
    with fitz.open(pdf) as d:
        text = " ".join(" ".join(p.get_text().split()) for p in d)
    # BOTH FORMS. The paper writes pointers as "SS 4.5" and as "Section 4.5",
    # and an earlier version of this audit matched only the sign, so half the
    # pointers were never checked. They were sound, but that was luck.
    return Counter(
        re.findall(SECTION_SIGN + r"\s?([0-9]+(?:\.[0-9]+){0,2})", text)
        + re.findall(r"\bSection\s+([0-9]+(?:\.[0-9]+){0,2})", text))


def main() -> int:
    problems: list[str] = []
    checked = 0
    preprint_heads: set[str] = set()
    print("=" * 74)
    print("SECTION REFERENCES vs SECTIONS THAT EXIST")
    print("=" * 74)

    for name, pdf, src, pat in DOCS:
        if not pdf.exists():
            print(f"  {name}: not built")
            continue
        refs = refs_from_pdf(pdf)
        if src is None:
            # The fork resolves its OWN sections with \ref, which LaTeX
            # checks. Its bare section-sign pointers come from the embedded
            # design table, and its appendix tells reviewers those numbers
            # refer to the technical report. So they are validated against
            # the PREPRINT's headings, which is the promise the fork makes.
            live = {r: n for r, n in refs.items() if not STATUTE.match(r)}
            checked += sum(live.values())
            if preprint_heads:
                for r, n in sorted(live.items()):
                    if r not in preprint_heads:
                        problems.append(
                            f"{name}: {SECTION_SIGN}{r} referenced {n} "
                            "time(s); the fork says these point into the "
                            "technical report, which has no such section")
            print(f"  {name}: {sum(live.values())} pointer(s) into the "
                  "technical report")
            continue
        heads = headings_from_source(src, pat)
        if name == "preprint":
            preprint_heads = heads
        assert heads, f"{name}: no headings found; the pattern has gone stale"
        for r, n in sorted(refs.items()):
            if STATUTE.match(r):
                continue
            checked += n
            if r not in heads:
                problems.append(
                    f"{name}: {SECTION_SIGN}{r} referenced {n} time(s) but no "
                    "such heading exists")
        print(f"  {name}: {len(heads)} headings, "
              f"{sum(n for r, n in refs.items() if not STATUTE.match(r))} "
              "pointers")

    print(f"  {checked} pointer(s) checked")
    if problems:
        print("-" * 74)
        for p in problems:
            print("  " + p)
        print("=" * 74)
        print(f"{len(problems)} dangling reference(s). A pointer to a section "
              "that does not exist sends the reader nowhere.")
        return 1
    print("=" * 74)
    print("clean: every pointer resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
