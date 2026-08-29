r"""Literal markup tags must never print on a built page.

A caps-to-italics pass wrote <i>...</i> into strings bound for the two render
paths that parse no runs (table captions, the title-block abstract), and 18
literal tags printed in every build after -- through every audit, because
text extraction yields the same characters either way. A correspondent found
them on the page.
"""

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"

sys.path.insert(0, str(SRC))


@pytest.mark.skipif(not PDF.exists(), reason="paper not built")
def test_no_built_pdf_carries_a_literal_tag():
    r = subprocess.run(
        [sys.executable, str(SRC / "audit_rendered_markup.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT))
    assert r.returncode == 0, r.stdout


def test_the_sinks_strip_what_they_cannot_parse():
    """The founding cases: a caption and an abstract carrying markup."""
    pk_src = (SRC / "paperkit.py").read_text(encoding="utf-8")
    # Both non-parsing sinks strip tags before drawing.
    assert pk_src.count(r'''re.sub(r"</?[bi]>"''') >= 2, (
        "paperkit's caption or abstract sink lost its markup strip")
    import build_paper_v3 as B
    out = B.arxiv_abstract("Choices made <i>after</i> the data exist.")
    assert "<i>" not in out and "after" in out
