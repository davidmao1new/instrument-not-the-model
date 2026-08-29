r"""The documented-figures checker, and proof that it still bites.

A checker that has stopped matching anything passes silently forever. These
tests pin the three ways this one could rot: the claim list going empty, a
pattern ceasing to fire because a document was reworded, and the comparison
being loose enough to accept a wrong number.
"""

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
sys.path.insert(0, str(SRC))

import audit_doc_figures as adf  # noqa: E402


def test_the_checker_is_clean_on_the_repository():
    r = subprocess.run([sys.executable, str(SRC / "audit_doc_figures.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    assert r.returncode == 0, r.stdout + r.stderr


def test_every_claim_actually_fires():
    """The founding failure this checker exists to prevent is a hand-quoted
    figure drifting. A pattern that matches nothing cannot catch that, so
    each one must find at least one occurrence in the file it names."""
    T = adf.truth()
    if not T:
        pytest.skip("nothing built")
    dead = []
    for rel, pattern, key, why in adf.CLAIMS:
        if key not in T:
            continue
        p = ROOT / rel
        assert p.exists(), f"{rel} is claimed but absent"
        text = p.read_text(encoding="utf-8", errors="replace")
        if not re.search(pattern, text):
            dead.append(f"{rel}: /{pattern}/ ({why})")
    assert not dead, "patterns that no longer match anything: " + "; ".join(dead)


def test_the_claim_list_is_not_empty():
    assert len(adf.CLAIMS) >= 5
    assert adf.HISTORICAL, (
        "the historical exemptions must stay listed with reasons; an empty "
        "dict means someone deleted the record of why files are skipped")


def test_a_planted_drift_is_caught(tmp_path, monkeypatch):
    """Change one documented figure and the checker must fail. Without
    this, a refactor could make the comparison vacuous and every run would
    still say clean."""
    T = adf.truth()
    if "full_pages" not in T:
        pytest.skip("full paper not built")
    real = ROOT / "CLAUDE.md"
    original = real.read_text(encoding="utf-8")
    wrong = original.replace(f"paper is {T['full_pages']} pages",
                             f"paper is {T['full_pages'] + 7} pages", 1)
    assert wrong != original, "the CLAUDE.md claim moved; re-aim this test"
    try:
        real.write_text(wrong, encoding="utf-8")
        r = subprocess.run([sys.executable, str(SRC / "audit_doc_figures.py")],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=str(ROOT))
        assert r.returncode == 1, "the checker passed a figure it should catch"
        assert "CLAUDE.md" in r.stdout
    finally:
        real.write_text(original, encoding="utf-8")


def test_historical_records_are_left_alone():
    """CHANGELOG.md and sent emails state what was true when written. If a
    future edit 'corrects' them, this fails and says why not to."""
    ch = ROOT / "CHANGELOG.md"
    if not ch.exists():
        pytest.skip("no changelog")
    assert "CHANGELOG.md" in adf.HISTORICAL
    assert re.search(r"\d+-page preprint", ch.read_text(encoding="utf-8")), (
        "the changelog's historical page counts were edited away; they are "
        "a record of what each revision was, not a description of now")
