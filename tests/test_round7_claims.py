"""Three claims round 7 showed were scoped wrong, powered wrong, or backwards.

Each survived six earlier rounds for the same reason: the arithmetic was right.
A number can be correctly computed from the correct artifact and still be the
wrong number for the sentence it sits in, and that is what these pin.

  1. SCOPE. The abstract's headline wording ratio was min/max over Study 2's
     two identified checkpoints -- one posting, open-weight only -- in a
     sentence naming eleven checkpoints. The same statistic is 46-51 % on the
     nurse posting and 54-64 % on the frontier panel. Round 6's arithmetic lens
     verified "25-33 %" as reproducing exactly, which it did, against the
     artifact it was drawn from.

  2. POWER. The reported within-race correlation is the POOLED statistic, with
     all 24 names in it; the minimum detectable correlation quoted beside it
     was computed at n = 12. Every use ran in the direction of excusing a null.

  3. DIRECTION. §4.4 claimed the position conditions D8/D9 produce a shift of
     the same sign as the Table 8 slope, which is what promoted the token
     length channel from admitted confound to "live mechanism". Over 16 cells
     the sign agrees on 4 and opposes on 12.
"""
from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "paper-a" / "data"
SRC = ROOT / "paper-a" / "src"
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


def _paper():
    if not PDF.exists():
        pytest.skip("paper not built")
    fitz = pytest.importorskip("fitz")
    with fitz.open(PDF) as doc:
        return " ".join(" ".join(p.get_text().split()) for p in doc)

def _companion():
    """The token-matching study moved to paper-c; its claims moved with it."""
    p = ROOT / "paper-c" / "figures" / "paper_token_matching.pdf"
    if not p.exists():
        return ""
    fitz = pytest.importorskip("fitz")
    with fitz.open(p) as doc:
        return " ".join(" ".join(pg.get_text().split()) for pg in doc)



# ------------------------------------------------------------------ scope ----
def test_the_wording_ratio_spans_every_identified_cell():
    """If a cell outside Study 2 carries the ratio, the range must include it."""
    occ = _load(DATA / "occupation" / "occupation_analysis.json")
    fr = _load(DATA / "frontier" / "frontier_margin_analysis.json")
    vals = []
    for m, v in occ.items():
        if m.startswith("_") or not isinstance(v, dict):
            continue
        for k in ("BA", "SWE", "RN"):
            c = v.get(k)
            if not isinstance(c, dict) or c.get("ratio_sd_to_effect") is None:
                continue
            ci = c.get("ps_ci")
            if ci and (ci[0] - 0.5) * (ci[1] - 0.5) > 0:
                vals.append(c["ratio_sd_to_effect"])
    for m in (fr.get("summary", {}).get("identified_models") or []):
        r = fr["models"][m].get("ratio_sd_to_effect")
        if r is not None:
            vals.append(r)
    assert vals, "no identified cells; this pins nothing"
    hi = max(vals)
    t = _paper()
    assert f"{round(hi * 100)} %" in t or f"{round(hi * 100)} %" in t, (
        f"the paper's range should reach {hi:.0%}")
    assert "25 % to 33 % of itself" not in t.replace(" ", " ")


# ------------------------------------------------------------------ power ----
def test_the_quoted_power_matches_the_reported_statistic():
    cv = _load(DATA / "names" / "construct_validity.json")
    mde = cv["mde"]
    assert "mde_rho_pooled_within_race" in mde
    # the pooled statistic has all names in it, so its MDE sits below the
    # per-race figure and above the plain all-names one
    assert mde["mde_rho_all_names"] <= mde["mde_rho_pooled_within_race"] \
        <= mde["mde_rho_within_race"]
    t = _paper().replace("≥", ">=")
    assert f"{mde['mde_rho_pooled_within_race']:.2f}" in t


# -------------------------------------------------------------- direction ----
def test_the_position_conditions_have_no_predicted_sign_to_corroborate_with():
    """Twice wrong, in opposite directions, and the second version is the one
    this test used to defend.

    The first draft said the position conditions corroborate the length
    channel. Round 7 counted the cells and found the sign agreed on a minority,
    so the claim was withdrawn and replaced by the count. The count is also
    wrong, and for a reason no tally could reach: D8 and D9 insert their extra
    tokens INSIDE the job posting, identically in both arms of every pair, so
    the within-pair token-length difference the slope regresses on is unchanged
    and the linear model predicts a shift of exactly zero. A manipulation with
    no predicted sign cannot agree or disagree with a slope, and "agrees on 4
    of 16" was arithmetic on an undefined comparison.

    So what is pinned is the design fact rather than any tally: the regressor
    does not move, and neither paper reports a sign agreement.
    """
    nl = _load(DATA / "instrument" / "name_length_effect.json")
    live = {k: v for k, v in nl.items()
            if not k.startswith("_") and isinstance(v, dict)}
    assert live
    t = _paper() + "\n" + _companion()
    assert "a live mechanism rather than a speculation" not in t
    assert "opposes on" not in t, "the sign tally is back on the page"
    assert "agree with it on a minority of cells" not in t
    assert "predicts a shift of exactly zero" in t
    assert "identically in both arms" in t
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    assert "_d89_agree" not in src, "the tally is back in the builder"
