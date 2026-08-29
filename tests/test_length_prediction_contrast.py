"""The prediction must be built on the same contrast it predicts.

THE DEFECT. `observed` is matched_minus_all: the mean over the token-matched
subset minus the mean over ALL pairs. Under a linear length effect with slope
b, the matched arm contributes delta = 0, so the predicted shift is

    -b * mean(delta over ALL pairs)

The analysis averaged delta over the DROPPED pairs instead, which inflates the
prediction by n_all / n_dropped -- a factor of 1.3 to 1.5 here -- because it
throws away exactly the zeros the "all" arm contains.

It flattered the result: the observed-over-predicted ratios ran 0.47 to 2.20
with Llama-3.1 at 1.07, and that 1.07 was quoted in the prose as the agreement
the cross-check was built to find. Corrected, they run 0.71 to 3.30 and
Llama-3.1 sits at 1.42. The correction goes against the paper, which is the
direction that makes it worth pinning.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
DATA = ROOT / "paper-a" / "data"
sys.path.insert(0, str(SRC))


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


def test_the_prediction_uses_the_mean_over_all_pairs():
    lp = _load(DATA / "instrument" / "length_prediction.json")
    for m, v in lp["models"].items():
        assert "mean_delta_over_all" in v, m
        assert np.isclose(v["predicted_shift"],
                          -v["slope"] * v["mean_delta_over_all"],
                          rtol=1e-9), m


def test_both_means_are_recorded_and_they_differ():
    """If they were equal the correction would be pinning nothing."""
    lp = _load(DATA / "instrument" / "length_prediction.json")
    seen = 0
    for m, v in lp["models"].items():
        assert "mean_delta_over_dropped" in v, m
        if v["mean_delta_over_all"] != v["mean_delta_over_dropped"]:
            seen += 1
        # the all-pairs mean is the smaller: it includes the matched zeros
        assert abs(v["mean_delta_over_all"]) <= \
            abs(v["mean_delta_over_dropped"]) + 1e-12, m
    assert seen == len(lp["models"]), (
        "some model has no matched pairs; the two means coincide there")


def test_the_inflation_factor_is_the_count_ratio():
    """mean over dropped = mean over all * n_all / n_dropped, exactly."""
    lp = _load(DATA / "instrument" / "length_prediction.json")
    for m, v in lp["models"].items():
        assert np.isclose(
            v["mean_delta_over_dropped"] * v["n_dropped"] / v["n_pairs"],
            v["mean_delta_over_all"], rtol=1e-9), m




# ---------------------------------------------------------------------------
# THE CLAIM MUST BE IN ONE OF THE PAPERS. The token-matching study moved to
# paper-c, so a test that reads only paper-a would fail on a claim that was
# moved rather than lost. Reading both also catches the real hazard of a split:
# a claim that falls out of BOTH papers.
def _corpus():
    fitz = pytest.importorskip("fitz")
    out = []
    for p in (ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf",
              ROOT / "paper-c" / "figures" / "paper_token_matching.pdf"):
        if p.exists():
            with fitz.open(p) as doc:
                out.append(" ".join(" ".join(pg.get_text().split())
                                    for pg in doc))
    if not out:
        pytest.skip("no paper built")
    return "\n".join(out)


def test_the_paper_reports_the_corrected_column():
    """The column moved to the companion paper; it must still be right."""
    t = _corpus()
    assert "over ALL pairs" in t, "the corrected contrast is not described"
    assert "mean Δ, dropped" not in t


def test_the_prose_no_longer_leans_on_the_flattered_ratio():
    """The 1.07 sentence was quoting a number the estimator inflated."""
    pdf = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
    if not pdf.exists():
        pytest.skip("paper not built")
    t = _corpus()
    assert "sitting at 1.07 on the model" not in t


def test_the_sign_agreement_survives_the_correction():
    """The claim that does survive: the two designs agree in direction."""
    lp = _load(DATA / "instrument" / "length_prediction.json")
    s = lp["summary"]
    assert s["n_same_sign"] == s["n_models"]
