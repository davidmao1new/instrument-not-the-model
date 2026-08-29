r"""Pointers must resolve, and the check must keep being able to see them.

A restructure vacated five section numbers into appendices and nineteen
references went on pointing at them. Three earlier audit rounds missed it,
because a missing heading has no figure to verify and a dangling pointer
states nothing false.
"""

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
sys.path.insert(0, str(SRC))

import audit_section_refs as asr  # noqa: E402


def test_the_audit_is_clean():
    r = subprocess.run([sys.executable, str(SRC / "audit_section_refs.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_heading_pattern_still_finds_headings():
    """If the builder stops using H("n  Title"), this audit goes blind and
    would pass forever while protecting nothing."""
    src = ROOT / "paper-a" / "src" / "build_paper_v3.py"
    pat = re.compile(r'H\("([0-9A-G]+(?:\.[0-9]+)?)\s')
    heads = asr.headings_from_source(src, pat)
    assert len(heads) >= 25, f"only {len(heads)} headings found"
    for expect in ("1", "4.1", "11", "A", "G"):
        assert expect in heads, f"{expect} missing; the pattern has drifted"


def test_the_numbering_gaps_are_real_and_deliberate():
    """4.3, 4.6, 4.7, 5.1 and 6.4 were vacated when those analyses became
    appendices. This pins the fact so a future reader does not 'restore'
    them and silently re-point the references."""
    src = ROOT / "paper-a" / "src" / "build_paper_v3.py"
    pat = re.compile(r'H\("([0-9A-G]+(?:\.[0-9]+)?)\s')
    heads = asr.headings_from_source(src, pat)
    for vacated in ("4.3", "4.6", "4.7", "5.1", "6.4", "3.1"):
        assert vacated not in heads, (
            f"section {vacated} exists again; if that is deliberate, the "
            "references re-aimed to an appendix need revisiting")
    for appendix in ("A", "B", "C", "D", "E", "F", "G"):
        assert appendix in heads, f"appendix {appendix} is missing"


def test_statute_citations_are_not_treated_as_pointers():
    """The paper cites N.Y.C. Admin. Code 20-870 and 6 RCNY 5-300. Those are
    sections of law, not of the paper, and flagging them would drown the
    real signal."""
    for stat in ("20-870", "20-874", "5-300", "20", "5"):
        assert asr.STATUTE.match(stat), f"{stat} should be exempt"
    for real in ("4.3", "6.4", "3.1", "4.7.1"):
        assert not asr.STATUTE.match(real), f"{real} must not be exempt"
