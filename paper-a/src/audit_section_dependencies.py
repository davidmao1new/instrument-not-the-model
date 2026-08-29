"""What does a section carry that nothing else does?

WHY THIS EXISTS. Five sections have to lose 6,900 words between them, and the
dangerous cut is not the long one -- it is the cut that takes out the only
statement of a number, or the target of a cross-reference somewhere else, or
the caveat that licenses a claim three sections away. Those are invisible while
reading the section itself, because the thing that depends on it is elsewhere.

So, per section, three questions:

  DUPLICATED    which sentences restate something the paper already says
                somewhere else. These are the safe cuts and they are where the
                6,900 words are.
  UNIQUE        which numbers appear in this section and nowhere else. Cutting
                a sentence carrying one of these loses the number from the
                paper, which may or may not be intended -- but it should never
                be an accident.
  INBOUND       which other sections point AT this one. A cross-reference whose
                target is deleted is a dangling pointer, and the render-order
                bug showed how long those survive unnoticed.

    sh paper-a/src/_py.sh paper-a/src/audit_section_dependencies.py
"""
from __future__ import annotations

import pathlib
import re
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"

# The five sections being compressed, and their word budgets.
TARGETS = {"1": 700, "2": 550, "9": 600, "10": 400, "11": 250}

SHINGLE = 5          # a duplicate is this many consecutive words in common
NUMBER = re.compile(r"\d+(?:[.,]\d+)*\s*%?|\d+\.\d+")
XREF = re.compile(r"§\s?(\d+(?:\.\d+)*)")


def sections(text: str) -> dict[str, str]:
    """Split the body on its headings, taken from the builder.

    Guessing headings out of the extracted text does not work: PyMuPDF
    collapses the double space that separates a heading's number from its
    title, so "1  Introduction" arrives as "1 Introduction" and is
    indistinguishable from any sentence containing a numeral before a capital.
    The builder knows the exact strings, so ask it.
    """
    src = (ROOT / "paper-a" / "src" / "build_paper_v3.py").read_text(
        encoding="utf-8")
    heads = []
    for m in re.finditer(r'H\(\s*"(\d+(?:\.\d+)*)\s+([^"]{3,80})"', src):
        num, title = m.group(1), " ".join(m.group(2).split())
        heads.append((num, f"{num} {title}"))

    # STOP AT THE BACK MATTER. §11 is the last numbered heading, so without
    # this it swallows the data-availability note, the endmatter and the whole
    # reference list -- which showed up as §11 "uniquely" carrying arXiv
    # identifiers and journal page numbers.
    body = text
    for tail in ("Appendices", "Data and code availability", "Endmatter",
                 "References"):
        i = body.find(tail)
        if i > 0:
            body = body[:i]
    marks = []
    at = 0
    for num, needle in heads:
        i = body.find(needle, at)
        if i < 0:                      # a heading that moved to the appendix
            continue
        marks.append((i, num))
        at = i + len(needle)

    out: dict[str, str] = {}
    for i, (pos, num) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(body)
        top = num.split(".")[0]
        out[top] = out.get(top, "") + " " + body[pos:end]
    return out


def sentences(s: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z“])", s)
            if len(x.split()) >= 8]


def shingles(s: str) -> set[str]:
    w = re.findall(r"[a-z]+", s.lower())
    return {" ".join(w[i:i + SHINGLE]) for i in range(len(w) - SHINGLE + 1)}


def main() -> int:
    if not PDF.exists():
        sys.exit("paper not built")
    try:
        import fitz
    except ImportError:
        sys.exit("PyMuPDF required")
    with fitz.open(PDF) as doc:
        text = " ".join(" ".join(p.get_text().split()) for p in doc)

    sec = sections(text)
    missing = [k for k in TARGETS if k not in sec]
    if missing:
        print(f"  [warn] sections not located: {', '.join(missing)}")

    # shingles per section, for duplicate detection
    sh = {k: shingles(v) for k, v in sec.items()}
    # numbers per section
    nums = {k: set(n.strip() for n in NUMBER.findall(v)) for k, v in sec.items()}
    # inbound cross-references
    inbound: dict[str, set[str]] = defaultdict(set)
    for k, v in sec.items():
        for tgt in XREF.findall(v):
            top = tgt.split(".")[0]
            if top != k:
                inbound[top].add(k)

    print("=" * 78)
    print("SECTION DEPENDENCIES  --  what each of the five carries alone")
    print("=" * 78)

    for k in sorted(TARGETS, key=lambda x: int(x)):
        if k not in sec:
            continue
        body = sec[k]
        words = len(body.split())
        budget = TARGETS[k]
        print(f"\n{'-' * 78}\n§{k}   {words:,} words -> {budget}  "
              f"(cut {words - budget:,}, {100 * (1 - budget / words):.0f} %)")

        # ---- inbound references
        ref = sorted(inbound.get(k, ()), key=lambda x: int(x))
        print(f"  POINTED AT BY: {', '.join('§' + r for r in ref) or 'nothing'}")

        # ---- numbers only here
        others = set().union(*(v for j, v in nums.items() if j != k)) \
            if len(nums) > 1 else set()
        only = sorted(n for n in nums[k] - others if len(n) > 2)
        print(f"  NUMBERS FOUND ONLY HERE ({len(only)}): "
              f"{', '.join(only[:14])}{' …' if len(only) > 14 else ''}")

        # ---- duplicated sentences
        elsewhere = set().union(*(v for j, v in sh.items() if j != k)) \
            if len(sh) > 1 else set()
        dup = []
        for s in sentences(body):
            g = shingles(s)
            if not g:
                continue
            share = len(g & elsewhere) / len(g)
            if share >= 0.30:
                dup.append((share, len(s.split()), s))
        dup.sort(reverse=True)
        dw = sum(n for _, n, _ in dup)
        print(f"  RESTATED ELSEWHERE: {len(dup)} sentences, {dw:,} words "
              f"({100 * dw / max(words, 1):.0f} % of the section)")
        for share, n, s in dup[:4]:
            print(f"      {share:>4.0%} {n:>3}w  {s[:96]}")

        # ---- repetition WITHIN the section, which is where the length is.
        # Cross-section duplication turned out to be near zero; what makes
        # these sections long is making the same point two or three times
        # inside them, in different words. Pairs, not shingle overlap against
        # a pool, because a pool hides which sentence is the twin.
        sents = sentences(body)
        gs = [(s, shingles(s)) for s in sents]
        pairs = []
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                a, b = gs[i][1], gs[j][1]
                if not a or not b:
                    continue
                share = len(a & b) / min(len(a), len(b))
                if share >= 0.22:
                    pairs.append((share, gs[i][0], gs[j][0]))
        pairs.sort(reverse=True)
        print(f"  NEAR-TWIN PAIRS INSIDE THE SECTION: {len(pairs)}")
        for share, a, b in pairs[:3]:
            print(f"      {share:>4.0%}  {a[:88]}")
            print(f"            {b[:88]}")

    print("\n" + "=" * 78)
    tot = sum(len(sec[k].split()) for k in TARGETS if k in sec)
    print(f"the five total {tot:,} words against {sum(TARGETS.values()):,}")
    print("A sentence restated elsewhere can go. A sentence carrying a number")
    print("found only here cannot, unless the number goes too -- deliberately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
