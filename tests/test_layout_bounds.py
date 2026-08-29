"""Nothing may be drawn outside the column it belongs to.

WHY. paperkit draws headings with drawString, which paints past the measure
without complaining. It never showed while every heading was short. Moving six
subsections into an appendix promoted their titles from level 2 to level 1, and
two of them bled: "A  Instrument validation, and one limitation it exposed" ran
265 pt into a 242 pt column and over the gutter into the neighbouring text, and
"F  Multiplicity, and a verdict that moved without its data" ran 29 pt past the
right margin and into the page edge.

Both were plainly visible on the page and invisible to 497 tests, the block
overlap check and four audits, because every check read the TEXT and none read
the GEOMETRY. This one reads the geometry.
"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
PDFS = [ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf",
        ROOT / "paper-c" / "figures" / "paper_token_matching.pdf"]


def _pk():
    sys.path.insert(0, str(SRC))
    return pytest.importorskip("paperkit")


def _blocks(pdf):
    fitz = pytest.importorskip("fitz")
    out = []
    with fitz.open(pdf) as doc:
        for pno, page in enumerate(doc):
            for b in page.get_text("dict")["blocks"]:
                if "lines" not in b:
                    continue
                spans = [s for ln in b["lines"] for s in ln["spans"]]
                if not spans:
                    continue
                t = " ".join(" ".join(
                    "".join(s["text"] for s in ln["spans"])
                    for ln in b["lines"]).split())
                if t:
                    out.append((pno + 1, b["bbox"], t, spans))
    return out


@pytest.mark.parametrize("pdf", PDFS, ids=lambda p: p.parent.parent.name)
def test_no_block_is_drawn_past_the_right_margin(pdf):
    """The unambiguous one: nothing may run into the page edge."""
    if not pdf.exists():
        pytest.skip(f"{pdf.name} not built")
    pk = _pk()
    right = pk.PAGE_W - pk.MARGIN_X
    bad = [(p, round(bb[2]), t[:70]) for p, bb, t, _ in _blocks(pdf)
           if bb[2] > right + 2]
    assert not bad, f"{len(bad)} block(s) past the right margin: {bad[:4]}"


@pytest.mark.parametrize("pdf", PDFS, ids=lambda p: p.parent.parent.name)
def test_no_heading_crosses_the_gutter(pdf):
    """A heading is never a full-width float here, so it must fit one column."""
    if not pdf.exists():
        pytest.skip(f"{pdf.name} not built")
    pk = _pk()
    col2 = pk.MARGIN_X + pk.COL_W + pk.GUTTER
    bad = []
    for page, bb, t, spans in _blocks(pdf):
        # a numbered or lettered heading: bold, set larger than body text
        if not re.match(r"^(\d+(\.\d+)*|[A-Z](\.\d+)?)\s+[A-Z]", t):
            continue
        if len(t) > 120 or max(s["size"] for s in spans) < 9.4:
            continue
        if "Bold" not in spans[0]["font"]:
            continue
        x0, x1 = bb[0], bb[2]
        if x0 < col2 - 2 and x1 > col2 + 2:
            bad.append((page, round(x1 - x0), t[:70]))
    assert not bad, f"heading(s) wider than a column: {bad}"


def test_the_heading_renderer_wraps():
    """The fix, pinned at the source: a heading that cannot wrap will bleed
    again the next time a long one is written."""
    src = (SRC / "paperkit.py").read_text(encoding="utf-8")
    head = src.split("def heading(", 1)[1].split("\n    def ", 1)[0]
    assert "self.wrap(" in head, "heading() no longer wraps"
    assert "for ln in _l" in head, "heading() draws a single line again"
    # and it must measure against the right width for span2 vs column
    assert "PAGE_W - 2 * MARGIN_X) if span2 else COL_W" in head
