"""How is this paper organised, against the papers it is joining?

WHY MEASURE RATHER THAN IMITATE BY FEEL. "Write like a good paper" is not
actionable. What IS actionable is the shape of the papers in lit/text/: how many
top-level sections they use, how long a section runs before a reader gets a
heading, where the results sit relative to the methods, how much of the document
is front matter before the first result, and how long the average paragraph is.
Those are countable, and the differences between this paper and that corpus are
the places a reader will feel it is unusual.

The corpus is the same one Table 18 surveys, plus the methodological anchors:
Bertrand and Mullainathan, Sclar et al., Simonsohn et al., Steegen et al.,
Kleinberg et al. Those last few matter because they are the papers this one is
formally closest to -- an argument about measurement rather than a report of a
single experiment.

WHAT THE NUMBERS MEAN FOR A REVISION, stated plainly because the temptation is
to treat any deviation as a fault:

  * Sections. A paper with many more top-level sections than the field is
    asking the reader to hold more structure in their head. A paper with far
    fewer is hiding its structure.
  * Words before the first result. This is the patience the paper demands
    before it pays anything back.
  * Paragraph length. Long paragraphs are where arguments go to hide.
  * Heading interval. The distance between signposts.

A deviation is a question, not a verdict. Some of this paper's shape is forced
by what it is: an audit that reports eleven studies has more sections than a
paper that reports one, and that is not a flaw.

    sh paper-a/src/_py.sh paper-a/src/audit_structure_vs_field.py
"""
from __future__ import annotations

import pathlib
import re
import statistics
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIT = ROOT / "lit" / "text"
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"

# A top-level heading in an extracted paper: a line starting with a small
# integer, then a title in title case. Deliberately conservative -- it will
# miss unnumbered headings, and that is better than counting equations.
HEAD = re.compile(r"^\s*(\d{1,2})\.?\s+([A-Z][A-Za-z][^\n]{2,60})$", re.M)
RESULT_WORDS = re.compile(
    r"^\s*\d{0,2}\.?\s*(Results?|Findings?|Experiments? and Results?)\b",
    re.M | re.I)


def profile(text: str, name: str) -> dict | None:
    flat_words = len(" ".join(text.split()).split())
    if flat_words < 3000:
        return None
    heads = [(m.start(), m.group(2).strip()) for m in HEAD.finditer(text)]
    # keep only ascending top-level numbers, so figure captions and
    # numbered lists do not masquerade as sections
    kept, last = [], 0
    for m in HEAD.finditer(text):
        n = int(m.group(1))
        if n == last + 1:
            kept.append((m.start(), m.group(2).strip()))
            last = n
    heads = kept

    paras = [p for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 25]
    plens = [len(p.split()) for p in paras]

    r = RESULT_WORDS.search(text)
    before = len(text[:r.start()].split()) if r else None

    return dict(
        name=name, words=flat_words,
        n_sections=len(heads),
        words_per_section=round(flat_words / len(heads), 0) if heads else None,
        median_paragraph=round(statistics.median(plens)) if plens else None,
        p90_paragraph=(round(statistics.quantiles(plens, n=10)[-1])
                       if len(plens) > 10 else None),
        words_before_results=before,
        frac_before_results=(round(before / flat_words, 2)
                             if before else None),
        section_titles=[h[1][:34] for h in heads][:14],
    )


def main() -> int:
    rows = []
    for p in sorted(LIT.glob("*.txt")):
        pr = profile(p.read_text(encoding="utf-8", errors="replace"), p.stem[:38])
        if pr:
            rows.append(pr)

    ours = None
    if PDF.exists():
        import fitz  # noqa: PLC0415
        with fitz.open(PDF) as doc:
            t = "\n".join(pg.get_text() for pg in doc)
        ours = profile(t, "THIS PAPER")

    print(f"{'paper':<40}{'words':>7}{'secs':>6}{'w/sec':>7}"
          f"{'med para':>10}{'p90':>6}{'pre-results':>12}")
    print("-" * 88)
    for r in sorted(rows, key=lambda r: r["n_sections"]):
        print(f"{r['name']:<40}{r['words']:>7}{r['n_sections']:>6}"
              f"{str(r['words_per_section'] or '-'):>7}"
              f"{str(r['median_paragraph'] or '-'):>10}"
              f"{str(r['p90_paragraph'] or '-'):>6}"
              f"{str(r['frac_before_results'] or '-'):>12}")
    if ours:
        print("-" * 88)
        print(f"{ours['name']:<40}{ours['words']:>7}{ours['n_sections']:>6}"
              f"{str(ours['words_per_section'] or '-'):>7}"
              f"{str(ours['median_paragraph'] or '-'):>10}"
              f"{str(ours['p90_paragraph'] or '-'):>6}"
              f"{str(ours['frac_before_results'] or '-'):>12}")

    def med(key):
        v = [r[key] for r in rows if r.get(key)]
        return statistics.median(v) if v else None

    print()
    for key, label in (("n_sections", "top-level sections"),
                       ("words_per_section", "words per section"),
                       ("median_paragraph", "median paragraph (words)"),
                       ("p90_paragraph", "90th-pct paragraph (words)")):
        m = med(key)
        o = ours.get(key) if ours else None
        if m is None or o is None:
            continue
        rel = o / m if m else float("nan")
        print(f"  {label:<28} field median {m:>6.0f}   ours {o:>6.0f}   "
              f"{rel:>4.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
