"""A round that did not run has not converged.

WHAT HAPPENED. Round 6 launched both halves and every one of the sixteen critic
agents died on a session limit. Each half returned `n_examined: 0,
n_confirmed: 0, n_serious: 0, converged: true` -- because the script computed
`converged: serious.length === 0`, and zero findings from zero examinations
looks exactly like zero findings from a clean paper.

Merging those two halves would have written `critique_round6.json` with
`n_serious: 0`, which is the condition the loop's stopping rule reads before
deleting its own fallback job. The loop would have terminated itself, reporting
success, on the one round where nothing looked at the paper.

A stopping rule that fires on an empty round is worse than none: it ends the
work precisely when the work failed. These tests pin the guard.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
MERGE = SRC / "merge_critique_rounds.py"
PY = sys.executable


def _half(tmp, name, **kw):
    d = dict(version="test", n_examined=0, n_confirmed=0, n_serious=0,
             converged=True, confirmed=[], refuted_count=0)
    d.update(kw)
    p = tmp / name
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def _run(round_name, *halves):
    out = subprocess.run(
        [PY, str(MERGE), round_name, *[str(h) for h in halves]],
        capture_output=True, text=True, cwd=str(ROOT))
    written = ROOT / "paper-a" / "releases" / f"critique_{round_name}.json"
    data = json.loads(written.read_text(encoding="utf-8"))
    written.unlink(missing_ok=True)
    return out.stdout, data


def test_an_empty_round_does_not_report_convergence():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        a = _half(tmp, "critique_test_half1.json")
        b = _half(tmp, "critique_test_half2.json")
        stdout, data = _run("guardtest_empty", a, b)
    assert data["converged"] is False, (
        "an all-empty round reported convergence; the loop would stop here")
    assert data["complete"] is False
    assert "INCOMPLETE" in stdout


def test_a_round_with_a_failed_lens_does_not_report_convergence():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        a = _half(tmp, "critique_test_half1.json",
                  n_examined=12, complete=False, n_lenses_failed=1)
        b = _half(tmp, "critique_test_half2.json",
                  n_examined=9, complete=True, n_lenses_failed=0)
        stdout, data = _run("guardtest_failed", a, b)
    assert data["converged"] is False
    assert data["n_halves_incomplete"] == 1


def test_a_missing_half_does_not_report_convergence():
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        a = _half(tmp, "critique_test_half1.json",
                  n_examined=12, complete=True, n_lenses_failed=0)
        missing = tmp / "critique_test_half2.json"
        stdout, data = _run("guardtest_missing", a, missing)
    assert data["converged"] is False


def test_a_complete_clean_round_DOES_report_convergence():
    """Guard the guard: the rule must still be able to fire."""
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        a = _half(tmp, "critique_test_half1.json",
                  n_examined=20, complete=True, n_lenses_failed=0)
        b = _half(tmp, "critique_test_half2.json",
                  n_examined=17, complete=True, n_lenses_failed=0)
        stdout, data = _run("guardtest_clean", a, b)
    assert data["complete"] is True
    assert data["converged"] is True, (
        "a complete round with no serious findings must be able to converge, "
        "or the loop can never stop")
    assert "CONVERGED" in stdout


def test_the_round_script_requires_lenses_to_report():
    """The same hole existed one level up, in critique_round.js."""
    js = (SRC / "critique_round.js").read_text(encoding="utf-8")
    assert "converged: serious.length === 0," not in js, (
        "the round script can again report convergence without running")
    assert "n_lenses_failed" in js
    assert "critic_failed" in js
