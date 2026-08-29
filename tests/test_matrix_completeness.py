"""Table 18 must print every choice the survey scored.

THE DEFECT. The table printed 11 of the artifact's 22 choices, chosen by a
literal list in the build script, and both the caption and §8.1 called it "the
matrix". The 11 omitted were the rows on which the surveyed field looks
competent -- exact prompt published 11/13, occupations 12/13, name list source
9/10, headline effect 10/12. Presenting the unflattering half of your evidence
as the whole is the practice this paper exists to object to, and it was doing
it in the table its own novelty claim rests on.

Ordering the rows is still an editorial choice, and the caption states it. What
may not happen again is a row existing in the artifact and not on the page.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
DATA = ROOT / "paper-a" / "data"
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"


def _matrix():
    p = DATA / "reference" / "reporting_practice_matrix.json"
    if not p.exists():
        pytest.skip("matrix not built")
    return json.loads(p.read_text(encoding="utf-8"))


def _paper():
    if not PDF.exists():
        pytest.skip("paper not built")
    fitz = pytest.importorskip("fitz")
    with fitz.open(PDF) as doc:
        return " ".join(" ".join(p.get_text().split()) for p in doc)


def test_every_scored_choice_appears_in_the_table():
    m = _matrix()
    t = _paper()
    missing = [c["pretty"] for c in m["counts"].values()
               if c["pretty"].lower() not in t.lower()]
    assert not missing, f"choices scored but not printed: {missing}"


def test_the_build_does_not_hand_pick_a_subset():
    """Structural: the row list must be derived, not enumerated."""
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    assert "HEAD = [(k, cts[k][\"pretty\"]) for k in _order]" in src, (
        "the table's rows are enumerated again rather than derived from cts")


def test_the_lead_order_drops_nothing():
    """LEAD reorders; it must not filter."""
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    m = re.search(r"_order = \[k for k in LEAD if k in cts\] \+ \[", src)
    assert m, "the order is no longer LEAD followed by the remainder"


def test_the_caption_does_not_claim_completeness_it_lacks():
    """If a future edit re-selects rows, the caption must stop saying this."""
    m = _matrix()
    t = _paper()
    printed = sum(1 for c in m["counts"].values()
                  if c["pretty"].lower() in t.lower())
    if printed == len(m["counts"]):
        assert "not a selection of them" in t
    else:
        assert "not a selection of them" not in t, (
            "the caption claims completeness the table does not have")
