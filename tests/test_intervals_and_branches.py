"""Two defects in how uncertainty was published.

  1. dispersion_budget.json deposited the token-matching interval as
     [abs(lo), abs(hi)] re-sorted, which turns an interval that SPANS zero into
     one that EXCLUDES it. It happened on three of four models, so a reader of
     the released artifact would have concluded the component is separable from
     zero everywhere -- the opposite of what the surrounding comment says the
     interval is carried to show, and the opposite of what `p` records.

  2. The arm contrast is fitted twice, with and without N1 (byte-identical to
     S1, so not an independent draw from the null arm). fit_arm_contrast.py
     says "The fit is reported both ways"; the paper reported one -- the
     branch whose own docstring calls it biased.

Neither is an arithmetic error. Both are about which number reaches the page.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
DATA = ROOT / "paper-a" / "data"


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------- the magnitude interval ----
def test_a_magnitude_interval_starts_at_zero_when_the_signed_one_spans_zero():
    b = _load(DATA / "delta_stability" / "dispersion_budget.json")
    seen = 0
    for m, rec in b["models"].items():
        c = rec["components"].get("token matching")
        if not c or "ci_signed" not in c:
            continue
        lo, hi = c["ci_signed"]
        if lo * hi <= 0:
            seen += 1
            assert c["ci"][0] == 0.0, (
                f"{m}: signed interval {c['ci_signed']} covers zero but the "
                f"magnitude interval {c['ci']} excludes it")
            assert c["ci_covers_zero"] is True, m
        else:
            assert c["ci"][0] > 0.0, m
            assert c["ci_covers_zero"] is False, m
    assert seen >= 1, "no interval spans zero; this test pins nothing"


def test_the_signed_interval_is_deposited_beside_the_magnitude_one():
    b = _load(DATA / "delta_stability" / "dispersion_budget.json")
    nl = _load(DATA / "instrument" / "name_length_effect.json")
    for m, rec in b["models"].items():
        c = rec["components"].get("token matching")
        if not c:
            continue
        assert "ci_signed" in c, m
        src = nl[m]["token_matched_first_name_clustered"]["matched_minus_all"]
        assert c["ci_signed"] == src["ci"], (
            f"{m}: the deposited signed interval is not the fitted one")


def test_the_magnitude_interval_contains_the_magnitude_estimate():
    b = _load(DATA / "delta_stability" / "dispersion_budget.json")
    for m, rec in b["models"].items():
        for name, c in rec["components"].items():
            if not c.get("ci"):
                continue
            assert c["ci"][0] <= c["value"] <= c["ci"][1] + 1e-12, (m, name)


def test_the_absolute_value_is_not_taken_endpointwise():
    """Structural: the construction that caused it must not return."""
    src = (SRC / "figures_budget.py").read_text(encoding="utf-8")
    assert 'ci=[abs(x) for x in sorted(_mm["ci"], key=abs)]' not in src


# ------------------------------------------------------ both fitted branches ----
def test_the_arm_contrast_artifact_has_both_branches():
    a = _load(DATA / "delta_stability" / "arm_contrast.json")
    n = 0
    for m, v in a.items():
        if m.startswith("_") or not isinstance(v, dict):
            continue
        assert "all" in v and "dropN1" in v, m
        n += 1
    assert n >= 2


def test_the_two_branches_differ_so_reporting_one_is_a_choice():
    a = _load(DATA / "delta_stability" / "arm_contrast.json")
    diffs = 0
    for m, v in a.items():
        if m.startswith("_") or not isinstance(v, dict):
            continue
        if v["all"]["prob_null_exceeds_semantic"] != \
                v["dropN1"]["prob_null_exceeds_semantic"]:
            diffs += 1
    assert diffs >= 1, "the branches coincide; the omission would be harmless"


def test_the_paper_reports_both_branches():
    pdf = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
    if not pdf.exists():
        pytest.skip("paper not built")
    fitz = pytest.importorskip("fitz")
    with fitz.open(pdf) as doc:
        t = " ".join(" ".join(p.get_text().split()) for p in doc)
    assert "Dropping N1" in t
    assert "Both fits are in the artifact" in t
