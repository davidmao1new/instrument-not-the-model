"""One rule about ratios, applied everywhere the paper states a ratio.

THE RULE, in the paper's own words (§4.5): "A ratio against such a denominator
has no finite upper bound, so the spread this paper previously quoted as a bare
184 percentage points in fact has an interval of [38, 7196] and is not a number
we may print."

The paper enforced it in Table 4 and Table 13 and broke it in Table 3, in §4.1,
in §9 item 1 and in the abstract -- and Table 3 is the one the other two quote.
The failure mode is not arithmetic, so no numeric check found it: every printed
value was correctly computed from a fresh artifact. What was wrong was printing
it at all.

These tests are therefore about WHERE a ratio may appear, not about its value.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
DATA = ROOT / "paper-a" / "data"


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


def _unidentified():
    """Models whose log-odds effect interval covers zero."""
    s2 = _load(DATA / "delta_stability" / "study2_v2.json")
    out = []
    for m, v in s2.items():
        if m.startswith("_"):
            continue
        ci = v["overall"]["logodds"]["ci"]
        if ci[0] * ci[1] <= 0:
            out.append(m)
    return out


def test_there_is_a_model_the_rule_applies_to():
    """Guard the guard: if every effect were identified these tests are vacuous."""
    assert _unidentified(), "no unidentified model; the rule pins nothing"


def test_table_3_suppresses_the_ratio_where_the_effect_covers_chance():
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    assert 'if m in survives else "—"' in src, (
        "Table 3's SD / effect cell no longer suppresses on unidentified "
        "models")


def test_the_rendered_table_3_row_shows_a_dash_not_a_number():
    pdf = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
    if not pdf.exists():
        pytest.skip("paper not built")
    fitz = pytest.importorskip("fitz")
    with fitz.open(pdf) as doc:
        t = " ".join(" ".join(p.get_text().split()) for p in doc)
    m = re.search(r"model P\(sup\.\)(.{0,400})", t)
    assert m, "Table 3 not found in the rendered paper"
    block = m.group(1)
    # Two models are unidentified, so two dashes must appear in the block.
    assert block.count("—") >= len(_unidentified()), (
        f"Table 3 prints a ratio where the denominator covers chance: {block}")


def test_no_prose_quotes_the_unbounded_ratio_range():
    """The abstract, §4.1 and §9 item 1 each printed the upper endpoint.

    The endpoint came from a model whose effect interval covers 0.5. Asserting
    on the SOURCE rather than the rendered value, because the rendered value
    changes with the data and the defect is the interpolation itself.
    """
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    live = [ln for ln in src.split("\n")
            if "_r_un" in ln and not ln.strip().startswith("#")]
    for ln in live:
        assert "fmt(min(_r_un)" not in ln and "fmt(max(_r_un)" not in ln, (
            f"prose interpolates the unbounded ratio: {ln.strip()}")


def test_the_ratio_interval_artifact_shows_why():
    """The upper endpoint really is unstable, so the rule is not pedantry."""
    ri = _load(DATA / "reference" / "ratio_intervals.json")
    wide = [v for v in ri["local"].values()
            if v.get("ci") and v["ci"][1] > 5.0]
    assert wide, ("no local ratio has a wide upper endpoint; the suppression "
                  "rule would be pinning nothing")


def test_every_ratio_table_says_it_suppresses():
    """All four ratio tables state the rule in their own caption.

    Tables 4, 11, 12 and 13 each already suppressed; Table 3 now does too.
    Asserting on the captions rather than on the cell expressions, because the
    caption is what tells a reader that a dash means "unbounded" rather than
    "missing" -- and a suppressed cell with no explanation is its own defect.
    """
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    n = src.count("suppressed")
    assert n >= 5, (
        f"only {n} suppression notices in the builder; a ratio table has "
        f"stopped explaining its dashes")


def test_the_frontier_table_suppresses_on_the_artifact_flag():
    """Table 13's dash is driven by the analyser, not by a hand-kept list."""
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    assert 'if v.get("ratio_sd_to_effect") else "—"' in src
    fm = _load(DATA / "frontier" / "frontier_margin_analysis.json")
    for m, v in fm["models"].items():
        if v.get("unmeasurable"):
            continue
        if v.get("ratio_suppressed_because_effect_covers_zero"):
            assert v.get("ratio_sd_to_effect") is None, m
