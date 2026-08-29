"""The paper must be typeset in Libertinus, and must say so if it is not.

WHY THIS FILE EXISTS. The public release shipped `tools/fonts/Libertinus-7.051/
static/OTF` (which matplotlib reads for the figures) but not `tools/fonts/ttf`
(which reportlab reads for the body). `paperkit.register_fonts()` caught the
resulting exception and returned Times-Roman, so the build produced a 31-page
PDF with the correct page count, the correct tables, no warning, and no error --
in the wrong face throughout.

It was only detectable downstream: Times-Roman is a PDF base-14 font with no
U+2009 THIN SPACE and no U+0394, so every "0.0 %" set with a thin space
extracted as "0.0I%" and every Delta became U+2206. Sixty-two tokens differed
between the two builds of what was supposed to be the same document.

The two assertions below are the cheap check that would have caught it on the
first build rather than the eighth.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper-a" / "src"))


@pytest.mark.parametrize("module", ["paperkit", "onepager"])
def test_the_typesetter_is_not_silently_on_the_fallback_face(module):
    m = __import__(module)
    assert m.FONT_FALLBACK is False, (
        f"{module} fell back to Times-Roman. The PDF will build and look "
        f"roughly right, but thin spaces and Greek letters render wrong. "
        f"Expected fonts at {m.TTF}"
    )
    assert m.R == "Lib", f"{module} body font is {m.R!r}, expected 'Lib'"


def test_the_faces_reportlab_needs_are_on_disk():
    """A separate directory from the OTFs matplotlib uses. That was the bug."""
    ttf = ROOT / "tools" / "fonts" / "ttf"
    need = ["LibertinusSerif-Regular.ttf", "LibertinusSerif-Bold.ttf",
            "LibertinusSerif-Italic.ttf"]
    missing = [n for n in need if not (ttf / n).exists()]
    assert not missing, f"missing from {ttf}: {missing}"


def test_the_font_licence_travels_with_the_fonts():
    """SIL OFL 1.1 permits redistribution only if the licence is included."""
    ofl = ROOT / "tools" / "fonts" / "Libertinus-7.051" / "OFL.txt"
    assert ofl.exists(), f"redistributing Libertinus without {ofl}"
    assert "SIL Open Font License" in ofl.read_text(encoding="utf-8")
