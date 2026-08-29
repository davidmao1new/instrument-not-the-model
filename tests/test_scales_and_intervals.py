"""Tests for scale commensurability and for the two new frontier analyses.

THE BUG THESE EXIST FOR. Two of the paper's ratios divided a LOG-ODDS standard
deviation by a PROBABILITY-OF-SUPERIORITY effect. Nothing caught it: the
arithmetic is valid, the artifacts were fresh, the numbers were interpolated
rather than typed, and both results were plausible small percentages (3 % and
5.4 %). The like-for-like values are 5 % and 8.7 %. §4.1 states the rule --
"Both quantities are on the log-odds scale ... so the ratio compares like with
like" -- three sections before the ratio that broke it.

A ratio is only checkable against its definition, so the definitions are what
these tests assert: each quantity's scale, traced to the field it is built
from, and the requirement that a ratio's halves agree.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
DATA = ROOT / "paper-a" / "data"
sys.path.insert(0, str(SRC))


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- scales ----
def test_the_noise_floor_ratio_is_log_odds_over_log_odds():
    """sigma_noise is built from margins, so its denominator must be beta.

    Dividing by |P(superiority) - 0.5| returned about 3 %; the comparable
    figure is about 5 %. This pins the identity the budget artifact encodes.
    """
    nf = _load(DATA / "delta_stability" / "noise_floor.json")
    bud = _load(DATA / "delta_stability" / "dispersion_budget.json")
    for m, rec in bud["models"].items():
        comp = rec.get("components", {}).get("noise floor")
        if not comp or m not in nf:
            continue
        beta = rec["beta_logodds"]
        assert math.isclose(
            comp["over_beta"],
            nf[m]["sigma_noise_on_variant_mean"] / abs(beta),
            rel_tol=1e-9), (
            f"{m}: the budget's noise-floor ratio is not sigma_noise / |beta|")


def test_the_paper_quotes_the_budget_artifact_for_the_noise_floor():
    """One computation, not two.

    The prose and Figure 1 disagreed because each divided independently. The
    builder now reads over_beta straight from the artifact the figure draws.
    """
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    assert '_bb[m]["components"]["noise floor"]["over_beta"]' in src
    assert 'noise[m]["sigma_noise_on_variant_mean"] / abs(ps[m] - 0.5)' not in src


def test_the_serving_bound_is_log_odds_over_log_odds():
    """effect_difference contrasts margin differences, so it is log-odds."""
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    assert 'max(abs(x) for x in _ci) / abs(lo[_m]["est"])' in src
    assert 'max(abs(x) for x in _ci) / abs(ps[_m] - 0.5)' not in src


def test_the_dispersion_ratio_is_superiority_over_superiority():
    """The one ratio that was always consistent stays consistent."""
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    assert "ratio_ps = {m: sd_word[m] / abs(ps[m] - 0.5) for m in models}" in src


def test_a_log_odds_sd_over_a_superiority_effect_is_a_different_number():
    """Guard the guard: confirm the two scales actually disagree here.

    If the mismatch happened to be numerically harmless the tests above would
    be pinning nothing. On the identified models it changes 5 % into 3 %.
    """
    s2 = _load(DATA / "delta_stability" / "study2_v2.json")
    nf = _load(DATA / "delta_stability" / "noise_floor.json")
    seen = 0
    for m, v in s2.items():
        if m.startswith("_") or m not in nf:
            continue
        lo = v["overall"]["logodds"]
        if lo["ci"][0] * lo["ci"][1] <= 0:
            continue
        seen += 1
        sig = nf[m]["sigma_noise_on_variant_mean"]
        right = sig / abs(lo["est"])
        wrong = sig / abs(v["overall"]["superiority"]["est"] - 0.5)
        assert not math.isclose(right, wrong, rel_tol=0.05), (
            f"{m}: the two scalings agree, so these tests pin nothing")
    assert seen >= 2


# ------------------------------------------------- the frontier replicate ----
def test_s1_and_n1_are_the_same_prompt():
    """The premise the frontier noise floor rests on, checked not trusted."""
    import experiment_delta_stability as ds  # noqa: PLC0415

    s1, n1 = ds.VARIANTS["S1"], ds.VARIANTS["N1"]
    assert s1["system"] == n1["system"]
    n = 0
    for _t, body in ds.TEMPLATES.items():
        for wname, bname in ds.PAIRS:
            for nm in (wname, bname):
                assert ds.user_message(s1, nm, body) == \
                    ds.user_message(n1, nm, body)
                n += 1
    assert n > 0


def test_the_frontier_noise_floor_verified_that_premise_itself():
    fn = _load(DATA / "frontier" / "frontier_noise_floor.json")
    pv = fn["_premise_verified"]
    assert pv["s1_and_n1_are_the_same_prompt"]
    assert pv["n_prompt_slots_differing"] == 0


def test_the_frontier_dispersion_exceeds_its_noise_floor():
    """The claim §4.7 now rests on. If this flips, §4.7 must be withdrawn."""
    fn = _load(DATA / "frontier" / "frontier_noise_floor.json")
    s = fn["summary"]
    assert s["dispersion_exceeds_the_floor_on_every_model"]
    assert s["n_models_where_noise_exceeds_dispersion"] == 0


def test_the_noise_correction_only_shrinks_the_dispersion():
    """Subtracting a variance cannot raise an SD; a sign slip would."""
    fn = _load(DATA / "frontier" / "frontier_noise_floor.json")
    for m, v in fn["models"].items():
        if not v.get("usable") or v.get("observed_sd_across_wordings") is None:
            continue
        assert v["wording_sd_corrected_for_noise"] <= \
            v["observed_sd_across_wordings"] + 1e-12, m
        assert v["wording_sd_corrected_for_noise"] >= 0.0, m
        assert 0.0 <= v["noise_share_of_observed_variance"] <= 1.0, m


def test_the_corrected_ratio_scales_with_the_corrected_sd():
    """The denominator is the effect and the replicate says nothing about it."""
    fn = _load(DATA / "frontier" / "frontier_noise_floor.json")
    for m, v in fn["models"].items():
        if not v.get("usable") or v.get("ratio_sd_to_effect_corrected") is None:
            continue
        assert math.isclose(
            v["ratio_sd_to_effect_corrected"],
            v["published_ratio_sd_to_effect"]
            * v["wording_sd_corrected_for_noise"]
            / v["observed_sd_across_wordings"], rel_tol=1e-9), m


def test_the_api_is_not_bitwise_reproducible():
    """The contrast with §5.2's local floor of exactly zero."""
    fn = _load(DATA / "frontier" / "frontier_noise_floor.json")
    assert not fn["summary"]["api_is_bitwise_reproducible"]


# ------------------------------------------------------ the ratio interval ----
def test_the_ratio_intervals_bracket_their_point_estimates():
    ri = _load(DATA / "reference" / "ratio_intervals.json")
    for panel in ("frontier", "local"):
        for m, v in ri[panel].items():
            if not v.get("ci") or v.get("est") is None:
                continue
            assert v["ci"][0] <= v["ci"][1], m
            assert v["ci"][0] > 0, m


def test_the_paper_may_not_claim_the_frontier_ratio_is_larger():
    """The finding that forced three sentences to change.

    Every frontier point estimate sits above every local one, but no interval
    separates, so the ordering is a direction. If a later run makes the
    intervals disjoint this test fails and the prose may be strengthened --
    which is the point of asserting the verdict rather than the wording.
    """
    ri = _load(DATA / "reference" / "ratio_intervals.json")
    c = ri["comparison"]
    assert c["every_frontier_point_above_every_local_point"]
    assert not c["intervals_disjoint"]
    assert not c["ranking_is_supported"]


def test_the_ranking_claims_are_gone_from_the_source():
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    for banned in ("larger than the open-weight panel",
                   "The frontier ratio is larger on the like-for-like"):
        assert banned not in src, f"unsupported ranking claim restored: {banned}"


def test_the_comparison_is_restricted_to_identified_models():
    """A ratio against a denominator covering zero is what §4.5 rules out."""
    ri = _load(DATA / "reference" / "ratio_intervals.json")
    ident = ri["_identified"]
    for m in ri["comparison"]["frontier_models"]:
        assert m in ident["frontier"]
    for m in ri["comparison"]["local_models"]:
        assert m in ident["local"]
