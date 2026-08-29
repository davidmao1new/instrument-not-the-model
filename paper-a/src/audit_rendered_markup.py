r"""No literal markup tag may appear in any built PDF.

WHY THIS EXISTS. Body paragraphs parse the builder's <b>/<i> run markup;
table captions and the title-block abstract did not, so a caps-to-italics
pass left 18 literal tags printed on the page -- "Choices made <i>after</i>
the data exists" in the abstract among them -- through every audit this
project runs, because text extraction yields the same characters whether a
tag rendered or printed. A reader found it on the page, in the file that had
just been sent to them.

The sinks now strip markup, and this audit closes the class: it extracts the
text of EVERY PDF the project builds and fails on any literal tag, so a new
sink that forgets to parse or strip is caught at build time rather than by a
correspondent.

    sh paper-a/src/_py.sh paper-a/src/audit_rendered_markup.py

Exit code 1 if any tag is found.
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

# Every PDF the project builds. Globbed, not listed: a new deliverable is
# covered the day it exists.
PDF_DIRS = [
    ROOT / "paper-a" / "figures",
    ROOT / "paper-a" / "releases",
    ROOT / "outreach",
]

TAG = re.compile(r"</?[bi]>")


def scan(pdf: pathlib.Path):
    import fitz
    with fitz.open(pdf) as d:
        text = "\n".join(p.get_text() for p in d)
    hits = []
    for m in TAG.finditer(text):
        ctx = re.sub(r"\s+", " ", text[max(0, m.start() - 50):m.start() + 50])
        hits.append(ctx)
    return hits


def main() -> int:
    print("=" * 74)
    print("RENDERED-MARKUP AUDIT  --  literal <b>/<i> tags on any built page")
    print("=" * 74)
    bad = 0
    n_pdfs = 0
    for d in PDF_DIRS:
        for pdf in sorted(d.glob("*.pdf")):
            n_pdfs += 1
            hits = scan(pdf)
            if hits:
                bad += 1
                print(f"\n  {pdf.relative_to(ROOT)}: {len(hits)} literal "
                      "tag(s)")
                for h in hits[:4]:
                    print(f"    ...{h}...")
            else:
                print(f"  ok    {pdf.relative_to(ROOT)}")
    print(f"\n  {n_pdfs} PDFs scanned, {bad} carrying literal tags")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
