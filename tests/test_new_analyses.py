"""Tests for the analyses added in the v4 round.

Each of these four modules produced a number that changed something the paper
says, so each needs a test that would catch a regression:

  * analyze_occupation_null       withdrew §4.5's dispersion claim
  * analyze_asymmetry_uncertainty qualified Table 6's crossover count
  * analyze_frontier_margin       added §4.7 and its saturation finding
  * analyze_noise_vs_probability  tested Fu et al.'s prediction in §5.2

The emphasis is on the properties that make a result WRONG if they break --
the resampling unit, the direction of a bound, the classification of missing
data -- not on reproducing the numbers, which the artifacts already hold.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
DATA = ROOT / "paper-a" / "data"
sys.path.insert(0, str(SRC))

import analyze_asymmetry_uncertainty as au  # noqa: E402
import analyze_frontier_margin as fm  # noqa: E402
import analyze_noise_vs_probability as nv  # noqa: E402
import analyze_occupation_null as on  # noqa: E402


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# analyze_occupation_null -- the statistic that could never be negative
# --------------------------------------------------------------------------
def test_spread_is_non_negative_by_construction():
    """The defect this module exists to fix, asserted so it stays fixed.

    max(SD) - min(SD) over non-negative SDs cannot be negative, which is why a
    percentile interval on it excluding zero was never evidence. Any future
    'interval excludes zero' claim about this statistic is wrong for the same
    reason, and this test records why.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        M = rng.normal(size=(3, 12))
        s, sd = on.spread(M)
        assert s >= 0.0
        assert math.isclose(s, sd.max() - sd.min())


def test_permutation_preserves_each_posting_mean():
    """Centring before permuting is what makes this a test of DISPERSION.

    If the permutation moved the posting means it would also be testing the
    effect, and the null it calibrates against would not be the null §4.5
    needs to beat.
    """
    rng = np.random.default_rng(1)
    M = rng.normal(loc=[[0.2], [0.5], [0.9]], scale=0.1, size=(3, 12))
    centred = M - M.mean(axis=1, keepdims=True)
    assert np.allclose(centred.mean(axis=1), 0.0, atol=1e-12)
    sp, vr = on.permute(M, 50, rng)
    assert len(sp) == 50 and np.all(sp >= 0)
    assert np.all(vr >= 1.0)  # max variance over min variance


def test_equal_dispersion_is_not_rejected():
    """Calibration check: three groups drawn from one distribution should not
    look like three different dispersions."""
    rng = np.random.default_rng(7)
    M = rng.normal(scale=0.1, size=(3, 12))
    obs, _ = on.spread(M)
    sp, _ = on.permute(M, 2000, rng)
    p = (1 + (sp >= obs).sum()) / (1 + 2000)
    assert p > 0.01


def test_occupation_null_artifact_agrees_with_its_verdict():
    d = _load(DATA / "occupation" / "occupation_dispersion_null.json")
    s = d["summary"]
    n_sig = sum(1 for v in d["models"].values() if v["significant_at_05"])
    assert n_sig == s["n_significant_spread"]
    # every p is a valid permutation p-value
    for v in d["models"].values():
        assert 0 < v["spread"]["p"] <= 1
        assert v["spread"]["est"] >= 0
    # the paper's §4.5 text is written for the not-rejected case; if this ever
    # flips, the section has to be rewritten rather than silently reinterpreted
    assert s["n_significant_spread"] == 0, (
        "the dispersion claim §4.5 withdraws would now be supported; §4.5 "
        "must be rewritten, not left as a withdrawal")


# --------------------------------------------------------------------------
# analyze_asymmetry_uncertainty -- the bound must bracket the point estimate
# --------------------------------------------------------------------------
def test_asymmetry_bound_brackets_the_point_estimate():
    d = _load(DATA / "names" / "asymmetry_uncertainty.json")
    for m, v in d["models"].items():
        for k in d["ks"]:
            c = v[f"k_{k}"]
            lo, hi = c["bound"]
            assert lo <= c["est"] <= hi, f"{m} k={k}: {lo} <= {c['est']} <= {hi}"


def test_asymmetry_ratio_falls_with_list_size():
    """The monotone claim §4.2 rests on, which holds for any sigmas because
    only the numerator carries the 1/sqrt(k)."""
    d = _load(DATA / "names" / "asymmetry_uncertainty.json")
    for m, v in d["models"].items():
        ests = [v[f"k_{k}"]["est"] for k in d["ks"]]
        assert ests == sorted(ests, reverse=True), f"{m}: {ests}"


def test_asymmetry_side_labels_match_the_bound():
    d = _load(DATA / "names" / "asymmetry_uncertainty.json")
    for v in d["models"].values():
        for k in d["ks"]:
            c = v[f"k_{k}"]
            lo, hi = c["bound"]
            expect = ("name larger" if lo > 1 else
                      "wording larger" if hi < 1 else "not determined")
            assert c["side"] == expect
            assert c["bound_determines_side"] == (expect != "not determined")


# --------------------------------------------------------------------------
# analyze_frontier_margin -- saturation is not the same as censoring
# --------------------------------------------------------------------------
def test_superiority_scoring_matches_the_local_panel():
    assert fm.superiority(1.0, 0.0) == 1.0
    assert fm.superiority(0.0, 1.0) == 0.0
    assert fm.superiority(0.5, 0.5) == 0.5


def test_cluster_bootstrap_resamples_pairs_not_rows():
    """Twelve pairs of wildly unequal size must contribute equally, because the
    pair is the unit. A row-level bootstrap would let the big pair dominate.
    """
    rng = np.random.default_rng(3)
    by_pair = {0: [1.0] * 200}
    for i in range(1, 12):
        by_pair[i] = [0.0]
    est, ci = fm.boot_ps(by_pair, rng, n_boot=2000)
    assert math.isclose(est, 1 / 12, rel_tol=1e-9), est


def test_frontier_artifact_separates_saturation_from_measurement():
    d = _load(DATA / "frontier" / "frontier_margin_analysis.json")
    for m, v in d["models"].items():
        assert v["n_usable"] + v["n_censored"] == v["n_design_cells"]
        if v.get("unmeasurable"):
            # an unmeasurable model must not carry an effect the paper could quote
            assert "superiority" not in v
            assert v["n_saturated"] > 0
        else:
            assert 0.0 <= v["superiority"]["est"] <= 1.0
            lo, hi = v["superiority"]["ci"]
            assert lo <= v["superiority"]["est"] <= hi


def test_frontier_ratio_is_suppressed_where_the_effect_covers_half():
    """The rule Table 16 announced and did not implement, asserted here so the
    frontier table cannot repeat it."""
    d = _load(DATA / "frontier" / "frontier_margin_analysis.json")
    for m, v in d["models"].items():
        if v.get("unmeasurable"):
            continue
        lo, hi = v["superiority"]["ci"]
        identified = (lo - 0.5) * (hi - 0.5) > 0
        assert v["effect_identified"] == identified
        if not identified:
            assert v["ratio_sd_to_effect"] is None, m


def test_frontier_summary_ratios_come_only_from_identified_models():
    d = _load(DATA / "frontier" / "frontier_margin_analysis.json")
    s = d["summary"]
    if not s.get("ratio_min"):
        pytest.skip("no identified model")
    ids = s["identified_models"]
    vals = [d["models"][m]["ratio_sd_to_effect"] for m in ids]
    assert math.isclose(s["ratio_min"], min(vals))
    assert math.isclose(s["ratio_max"], max(vals))


def test_frontier_arm_uses_study_2s_exact_design():
    """§4.7's whole claim is that the comparison is exact rather than analogical.

    That is only true if the frontier run swept the SAME wordings, templates and
    name pairs as Study 2. The experiment imports them rather than restating
    them, but an import can be edited; this checks the collected data, which
    cannot be.
    """
    def sweep(p):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        return (sorted({r["variant"] for r in rows}),
                sorted({r["template"] for r in rows}),
                sorted({r["pair"] for r in rows}),
                sorted({(r["white_name"], r["black_name"]) for r in rows}))

    s2 = DATA / "delta_stability" / "delta_llama-2-7b-chat.jsonl"
    if not s2.exists():
        pytest.skip("study 2 raw data not present")
    ref = sweep(s2)
    n = 0
    for f in sorted((DATA / "frontier").glob("margin_*.jsonl")):
        assert sweep(f) == ref, f"{f.name} does not sweep Study 2's design"
        n += 1
    assert n, "no frontier margin files found"


def test_frontier_censoring_by_template_sums_to_the_total():
    d = _load(DATA / "frontier" / "frontier_margin_analysis.json")
    for m, v in d["models"].items():
        bt = v.get("censoring_by_template")
        if not bt:
            continue
        assert sum(x["lost"] for x in bt.values()) == v["n_censored"], m
        assert sum(x["n"] for x in bt.values()) == v["n_design_cells"], m


def test_frontier_corpus_censoring_matches_the_corpus_walker():
    """Two independent counts of the same thing, from different code paths.

    analyze_frontier_margin.py counts censored cells while analysing them;
    analyze_corpus_size.py counts null-margin rows by walking every raw file
    under paper-a/data with no knowledge of the frontier study. They must agree.
    """
    fr = _load(DATA / "frontier" / "frontier_margin_analysis.json")
    cs = _load(DATA / "reference" / "corpus_size.json")
    walked = cs["by_study"]["frontier"]["n_pair_rows_with_a_null_margin"]
    assert fr["summary"]["total_censored"] == walked


# --------------------------------------------------------------------------
# analyze_noise_vs_probability -- the rank correlation
# --------------------------------------------------------------------------
def test_spearman_against_known_values():
    assert math.isclose(nv.spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
    assert math.isclose(nv.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
    sp = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(11)
    for _ in range(20):
        x = rng.normal(size=40)
        y = 0.6 * x + rng.normal(size=40)
        assert math.isclose(nv.spearman(x, y), sp.spearmanr(x, y).statistic,
                            rel_tol=1e-9)


def test_spearman_handles_ties():
    """Many replicate cells are bitwise identical, so their SD is exactly zero
    and the ranks tie. Untied ranks would bias the correlation."""
    sp = pytest.importorskip("scipy.stats")
    x = [0.0, 0.0, 0.0, 1.0, 2.0, 2.0]
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert math.isclose(nv.spearman(x, y), sp.spearmanr(x, y).statistic,
                        rel_tol=1e-9)


def test_noise_artifact_records_every_rho_and_its_sign():
    d = _load(DATA / "replicate" / "noise_vs_probability.json")
    rhos = [v["spearman_rho_sd_vs_centrality"] for v in d["models"].values()]
    assert all(r is not None for r in rhos)
    assert math.isclose(d["summary"]["rho_min"], min(rhos))
    assert math.isclose(d["summary"]["rho_max"], max(rhos))
    assert d["summary"]["all_rho_positive"] == (min(rhos) > 0)


# --------------------------------------------------------------------------
# the second task's replicate arm, which is also a reproducibility result
# --------------------------------------------------------------------------
def test_s1_and_n1_assemble_to_byte_identical_prompts():
    """The premise of §4.6's zero noise floor, checked against the generator.

    S1 is the semantic arm's baseline and N1 the null arm's, and the design
    requires them to be the same prompt so that any disagreement between them
    is measurement error and nothing else. If an edit to the wording table ever
    breaks that, the noise floor silently becomes a wording effect.
    """
    import experiment_second_task as est  # noqa: PLC0415

    n = 0
    for name, dom in est.DOMAINS.items():
        v = est.build_variants(dom)
        assert v["S1"]["system"] == v["N1"]["system"], name
        for body in dom["levels"].values():
            for w, b in est.PAIRS:
                for nm in (w, b):
                    a = est.user_message(dom, v["S1"], nm, body)
                    c = est.user_message(dom, v["N1"], nm, body)
                    assert a == c, f"{name}: S1 and N1 differ for {nm}"
                    n += 1
    assert n > 0


def test_second_task_replicate_is_exact():
    d = _load(DATA / "second_task" / "second_task_analysis.json")
    s = d["summary"]
    if not s.get("n_replicate_cells"):
        pytest.skip("replicate arm not present")
    # the paper states this as an exact result, so a regression must fail here
    assert s["n_replicate_cells_identical"] == s["n_replicate_cells"], (
        "§4.6 says every byte-identical replicate cell returned a "
        "bitwise-identical margin; it no longer does, and the section's "
        "zero-noise-floor claim must be rewritten")
    assert s["n_cells_with_zero_noise_floor"] == s["n_domain_model_cells"]


def test_second_task_ratio_range_excludes_unidentified_cells():
    d = _load(DATA / "second_task" / "second_task_analysis.json")
    s = d["summary"]
    if not s.get("ratio_min"):
        pytest.skip("no identified cell")
    vals = [r["ratio_ps"] for r in s["per_cell"]
            if r.get("ratio_ps") is not None and r["identified"]]
    assert math.isclose(s["ratio_min"], min(vals))
    assert math.isclose(s["ratio_max"], max(vals))


def test_second_task_summary_ratio_is_on_the_superiority_scale():
    """The scale every other panel uses, and the reason this test exists.

    Tables 3 and 17 report the probability of superiority. Table 16 reported
    log-odds, and the abstract put all three dispersion-to-effect ranges side
    by side -- a ratio is not invariant under the logit transform, so that is
    the operating-point error §6.2 is a section about, committed by the paper.
    If the summary ever reverts to log-odds this fails.
    """
    d = _load(DATA / "second_task" / "second_task_analysis.json")
    s = d["summary"]
    if not s.get("ratio_min"):
        pytest.skip("no identified cell")
    assert "P(sup)" in s.get("ratio_basis", "")
    ps = [r["ratio_ps"] for r in s["per_cell"]
          if r.get("ratio_ps") is not None and r["identified"]]
    lo = [r["ratio"] for r in s["per_cell"]
          if r.get("ratio") is not None and r["identified"]]
    assert math.isclose(s["ratio_min"], min(ps))
    # and the two scales genuinely differ, so the choice is not cosmetic
    if lo:
        assert not math.isclose(min(ps), min(lo), rel_tol=1e-3)


def test_frontier_yes_no_mass_excludes_the_token_class_explanation():
    """§4.7 concludes gpt-4.1 is saturated rather than emitting a non-verdict.

    The two explanations make opposite predictions about p(yes) + p(no): near 1
    under saturation, near 0 under a token-class mismatch. If a future run ever
    shows low mass, the paragraph is wrong and must be rewritten.
    """
    d = _load(DATA / "frontier" / "frontier_yes_no_mass.json")
    for m, v in d["models"].items():
        assert v["frac_below_0_01"] < 0.01, (
            f"{m}: mass is off the verdict tokens on "
            f"{v['frac_below_0_01']:.1%} of calls; §4.7's saturation reading "
            "no longer holds")
        assert v["mean_mass"] > 0.95, m
    assert d["summary"]["n_token_class_mismatch"] == 0


def test_noise_band_membership_is_consistent():
    d = _load(DATA / "replicate" / "noise_vs_probability.json")
    for m, v in d["models"].items():
        assert v["n_in_band"] <= v["n_cells"]
        if v["n_in_band"] == 0:
            assert v["mean_sd_in_band"] is None
            assert "p_permutation" not in v
        else:
            assert v["mean_sd_in_band"] is not None
