"""The released abstract must be pasteable into arXiv's form without editing.

arXiv's abstract field is ASCII plus a small TeX subset. Paste one em dash and
the form returns "bad character(s) in field Abstract" naming neither the
character nor its position. That happened on 2026-08-19: the artifact was clean,
a hand-edit introduced a single U+2014, and the only way to find it was to
enumerate every codepoint.

The generator cannot stop someone editing in a browser. What it can do is
guarantee that the text it hands over is clean to start with, so a failure is
known to come from the edit.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper-a" / "src"))
ABS = ROOT / "paper-a" / "releases" / "abstract_arxiv.txt"
ABS_C = ROOT / "paper-c" / "releases" / "abstract_arxiv.txt"


def checker():
    return pytest.importorskip("check_arxiv_fields")


@pytest.mark.parametrize("path", [ABS, ABS_C], ids=["paper-a", "paper-c"])
def test_the_released_abstract_would_be_accepted(path):
    if not path.exists():
        pytest.skip(f"{path.name} not generated")
    problems = checker().check(path.read_text(encoding="utf-8"))
    hard = [p for p in problems if "under the limit" not in p]
    assert not hard, "\n".join(hard)


@pytest.mark.parametrize("path", [ABS, ABS_C], ids=["paper-a", "paper-c"])
def test_the_released_abstract_is_pure_ascii(path):
    """The single rule that matters. Stated separately from the checker so the
    failure message names the codepoint rather than a paragraph."""
    if not path.exists():
        pytest.skip(f"{path.name} not generated")
    t = path.read_text(encoding="utf-8")
    bad = sorted({f"U+{ord(c):04X} {c!r}" for c in t if ord(c) > 126})
    assert not bad, f"non-ASCII in {path.name}: {bad}"


def test_the_checker_catches_the_character_that_actually_broke_it():
    """A checker that passes everything is worse than none."""
    c = checker()
    problems = c.check("An abstract with an em dash\u2014like this one.")
    assert any("U+2014" in p for p in problems)
    assert any("bad character" in p for p in problems)


def test_the_checker_catches_the_other_common_paste_artifacts():
    c = checker()
    for ch, name in [("\u2019", "curly apostrophe"), ("\u201c", "curly quote"),
                     ("\ufb01", "fi ligature"), ("\u00a0", "non-breaking space")]:
        assert c.check(f"text {ch} text"), f"{name} not caught"


def test_the_checker_accepts_clean_ascii():
    c = checker()
    assert not c.check("A perfectly ordinary abstract, 25 % of it plain ASCII "
                       "-- and nothing else.")


def test_the_length_limit_is_enforced():
    c = checker()
    assert any("rejects an abstract over" in p
               for p in c.check("x" * (c.MAX_CHARS + 1)))
