"""Tests for the effect-size layer, which every number in the paper passes through.

This module replaced a constant (`LOGIT_TO_PP = 25.0`) that was wrong by factors
of up to 108, and added a cluster bootstrap that widened the paper's main
intervals by 3 to 5 times and turned two significant effects null. Both changes
moved published conclusions, so both need tests that would catch a regression.
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "paper-a" / "src"
sys.path.insert(0, str(SRC))

import effectsize as es  # noqa: E402


# --------------------------------------------------------------------------
# Benjamini-Hochberg
# --------------------------------------------------------------------------
def test_bh_matches_statsmodels():
    ps = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205,
          0.212, 0.216, 0.222, 0.251, 0.269, 0.275, 0.34]
    mine = es.benjamini_hochberg(ps)
    sm = pytest.importorskip("statsmodels.stats.multitest")
    ref = sm.multipletests(ps, method="fdr_bh")[1]
    assert np.allclose(mine, ref)


def test_bh_is_order_invariant():
    ps = [0.9, 0.01, 0.5, 0.03, 0.2]
    straight = es.benjamini_hochberg(ps)
    order = [3, 0, 4, 1, 2]
    shuffled = es.benjamini_hochberg([ps[i] for i in order])
    back = [0.0] * len(ps)
    for pos, i in enumerate(order):
        back[i] = shuffled[pos]
    assert np.allclose(back, straight)


def test_bh_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    ps = sorted(rng.uniform(size=40).tolist())
    adj = es.benjamini_hochberg(ps)
    assert all(0.0 <= a <= 1.0 for a in adj)
    assert all(b >= a - 1e-12 for a, b in zip(adj, adj[1:]))


def test_bh_never_shrinks_a_pvalue():
    ps = [0.001, 0.5, 0.9]
    assert all(a >= p - 1e-12 for a, p in zip(es.benjamini_hochberg(ps), ps))


# --------------------------------------------------------------------------
# Scale conversions
# --------------------------------------------------------------------------
def test_prob_is_the_logistic():
    assert es.prob(0.0) == pytest.approx(0.5)
    assert es.prob(100.0) == pytest.approx(1.0)
    assert es.prob(-100.0) == pytest.approx(0.0, abs=1e-12)
    for m in (-3.0, -0.5, 0.0, 0.7, 4.2):
        assert es.prob(m) == pytest.approx(1 / (1 + math.exp(-m)))


def rows_from(margins):
    """margins: list of (white, black)."""
    return [{"white_margin": w, "black_margin": b, "pair": i,
             "template": "T1", }
            for i, (w, b) in enumerate(margins)]


def test_local_slope_is_small_when_saturated():
    """The bug this module exists to fix.

    A model pinned near probability 1 has almost no probability-scale
    sensitivity, so a shared Jacobian of 0.25 overstates its effect enormously.
    """
    saturated = rows_from([(8.0, 7.5)] * 20)
    balanced = rows_from([(0.1, -0.1)] * 20)
    assert es.local_slope(saturated) < 0.01
    assert es.local_slope(balanced) == pytest.approx(0.25, abs=0.01)


def test_pp_interval_brackets_the_legacy_number():
    rows = rows_from([(8.0, 7.5)] * 30)
    d = es.describe(rows, n_boot=200)
    lo_, hi_ = sorted(d["pp_interval"])
    assert lo_ <= d["legacy_pp"] <= hi_ or hi_ <= d["legacy_pp"] <= lo_ \
        or abs(d["legacy_pp"] - hi_) < 1e-9
    # and the legacy conversion is the p = 0.5 end, i.e. the extreme one
    assert abs(d["legacy_pp"]) == pytest.approx(max(abs(lo_), abs(hi_)), rel=1e-6)


# --------------------------------------------------------------------------
# Probability of superiority
# --------------------------------------------------------------------------
def test_superiority_endpoints():
    ps, cliff = es.superiority(np.array([1.0, 2.0, 3.0]))
    assert ps == 1.0 and cliff == 1.0
    ps, cliff = es.superiority(np.array([-1.0, -2.0]))
    assert ps == 0.0 and cliff == -1.0
    ps, cliff = es.superiority(np.array([1.0, -1.0]))
    assert ps == pytest.approx(0.5) and cliff == pytest.approx(0.0)


def test_superiority_ties_split_evenly():
    ps, cliff = es.superiority(np.array([0.0, 0.0, 0.0, 0.0]))
    assert ps == pytest.approx(0.5)
    assert cliff == pytest.approx(0.0)


def test_cliff_delta_is_twice_ps_minus_one():
    rng = np.random.default_rng(3)
    d = rng.normal(0.3, 1.0, size=200)
    ps, cliff = es.superiority(d)
    assert cliff == pytest.approx(2 * ps - 1)


# --------------------------------------------------------------------------
# The cluster bootstrap. This is the correction that changed conclusions.
# --------------------------------------------------------------------------
def _clustered_rows(n_clusters=12, reps=3, spread=1.0, seed=0):
    """Each cluster has a large offset, replicated `reps` times with small noise.

    This is the structure of the real design: a name pair appears under every
    wording and template, so its idiosyncratic effect recurs. An i.i.d.
    bootstrap treats the repeats as independent evidence and understates the
    uncertainty.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        offset = rng.normal(0, spread)
        for r in range(reps):
            w = offset + rng.normal(0, 0.01)
            rows.append({"white_margin": w, "black_margin": 0.0,
                         "pair": c, "template": f"T{r}"})
    return rows


def test_clustering_widens_intervals_on_clustered_data():
    rows = _clustered_rows()
    naive = es.describe(rows, n_boot=3000, rng=np.random.default_rng(1),
                        cluster_key=None)
    clust = es.describe(rows, n_boot=3000, rng=np.random.default_rng(1))
    w_naive = naive["logodds"]["ci"][1] - naive["logodds"]["ci"][0]
    w_clust = clust["logodds"]["ci"][1] - clust["logodds"]["ci"][0]
    assert clust["clustered"] is True
    assert naive["clustered"] is False
    assert w_clust > 1.4 * w_naive, (w_naive, w_clust)


def test_clustering_is_a_noop_when_one_row_per_cluster():
    """A single specification has exactly one row per name pair, so the row
    bootstrap already IS the cluster bootstrap and nothing should change."""
    rows = _clustered_rows(n_clusters=20, reps=1, seed=5)
    naive = es.describe(rows, n_boot=4000, rng=np.random.default_rng(7),
                        cluster_key=None)
    clust = es.describe(rows, n_boot=4000, rng=np.random.default_rng(7))
    assert clust["clustered"] is False          # auto-detect declines to cluster
    assert clust["logodds"]["ci"] == pytest.approx(naive["logodds"]["ci"])


def test_cluster_count_is_recorded():
    rows = _clustered_rows(n_clusters=7, reps=4)
    d = es.describe(rows, n_boot=500)
    assert d["n_clusters"] == 7
    assert d["n"] == 28


def test_boot_ci_clusters_argument_resamples_whole_clusters():
    """With two clusters of wildly different means, a cluster bootstrap must
    sometimes produce a mean equal to one cluster's mean alone."""
    x = np.array([0.0] * 10 + [100.0] * 10)
    cl = np.array([0] * 10 + [1] * 10)
    b = es.boot_ci(x, lambda a: float(a.mean()), n_boot=2000,
                   rng=np.random.default_rng(0), clusters=cl)
    vals = set(np.round(b["boots"], 6))
    assert {0.0, 50.0, 100.0} <= vals


# --------------------------------------------------------------------------
# p-values
# --------------------------------------------------------------------------
def test_pvalue_floor_is_one_over_nboot():
    b = np.full(500, 10.0)
    p = es.pvalue_from_boots(b, est=10.0, null=0.0, n_boot=500)
    assert p == pytest.approx(1 / 500)


def test_pvalue_near_one_when_estimate_is_at_the_null():
    rng = np.random.default_rng(11)
    b = rng.normal(0.0, 1.0, size=4000)
    p = es.pvalue_from_boots(b, est=0.0, null=0.0, n_boot=4000)
    assert p > 0.9


def test_pvalue_uses_the_stated_null_not_zero():
    """Ps is tested against 0.5, not against 0. Passing the wrong null would
    make every superiority result look overwhelmingly significant."""
    rng = np.random.default_rng(2)
    b = rng.normal(0.5, 0.02, size=4000)
    at_null = es.pvalue_from_boots(b, est=0.5, null=0.5, n_boot=4000)
    at_zero = es.pvalue_from_boots(b, est=0.5, null=0.0, n_boot=4000)
    assert at_null > 0.9
    assert at_zero < 0.01


# --------------------------------------------------------------------------
# Paired contrast
# --------------------------------------------------------------------------
def test_paired_contrast_pairs_on_the_key_not_on_row_order():
    a = [{"white_margin": 1.0, "black_margin": 0.0, "template": "T1", "pair": 0},
         {"white_margin": 3.0, "black_margin": 0.0, "template": "T1", "pair": 1}]
    b = [{"white_margin": 3.0, "black_margin": 0.0, "template": "T1", "pair": 1},
         {"white_margin": 0.5, "black_margin": 0.0, "template": "T1", "pair": 0}]
    c = es.paired_contrast(a, b, ("template", "pair"), n_boot=200)
    # pair 0: 1.0 - 0.5 = 0.5 ; pair 1: 3.0 - 3.0 = 0.0 ; mean 0.25
    assert c["logodds"] == pytest.approx(0.25)
    assert c["n"] == 2


def test_paired_contrast_uses_only_shared_cells():
    a = [{"white_margin": 1.0, "black_margin": 0.0, "template": "T1", "pair": i}
         for i in range(5)]
    b = [{"white_margin": 0.0, "black_margin": 0.0, "template": "T1", "pair": i}
         for i in range(3)]
    c = es.paired_contrast(a, b, ("template", "pair"), n_boot=200)
    assert c["n"] == 3


def test_paired_contrast_returns_none_with_no_overlap():
    a = [{"white_margin": 1.0, "black_margin": 0.0, "template": "T1", "pair": 0}]
    b = [{"white_margin": 1.0, "black_margin": 0.0, "template": "T2", "pair": 9}]
    assert es.paired_contrast(a, b, ("template", "pair"), n_boot=100) is None


# --------------------------------------------------------------------------
# describe: incomplete rows must be excluded, not coerced
# --------------------------------------------------------------------------
def test_describe_drops_incomplete_rows():
    rows = rows_from([(1.0, 0.0), (2.0, 0.0)])
    rows.append({"white_margin": None, "black_margin": 0.0, "pair": 99,
                 "template": "T1"})
    d = es.describe(rows, n_boot=200)
    assert d["n"] == 2


def test_describe_returns_none_on_empty():
    assert es.describe([], n_boot=10) is None
    assert es.describe([{"white_margin": None, "black_margin": None}],
                       n_boot=10) is None


# ---------------------------------------------------------------------------
# paired_contrast resamples NAME PAIRS, not rows.
#
# The mechanism panel keys a cell on (template, pair), so one name pair
# contributes three rows. Bootstrapping those 72 rows independently treats
# three correlated measurements of the same name as three independent draws.
# Measured on the complete panel this widens 34% of contrast intervals by more
# than 10% and 9% by more than 25%, so it is not a cosmetic distinction.
# ---------------------------------------------------------------------------
def _panel_rows(n_pairs=24, templates=("T1_strong", "T2_mid", "T3_marginal"),
                pair_effect=1.0, noise=0.02, seed=0):
    """Rows with a large per-PAIR effect and almost no within-pair variation.

    Under this structure the effective sample size is n_pairs, not
    n_pairs * n_templates, so a row bootstrap must produce a visibly narrower
    interval than a cluster bootstrap.
    """
    rng = np.random.default_rng(seed)
    per_pair = rng.normal(0.0, pair_effect, size=n_pairs)
    a, b = [], []
    for p in range(n_pairs):
        for t in templates:
            a.append(dict(template=t, pair=p, white_margin=per_pair[p],
                          black_margin=rng.normal(0, noise)))
            b.append(dict(template=t, pair=p, white_margin=0.0,
                          black_margin=rng.normal(0, noise)))
    return a, b


def test_paired_contrast_clusters_on_pair_by_default():
    a, b = _panel_rows()
    clustered = es.paired_contrast(a, b, ("template", "pair"), 4000,
                                   np.random.default_rng(3))
    iid = es.paired_contrast(a, b, ("template", "pair"), 4000,
                             np.random.default_rng(3), cluster_on=None)
    assert clustered["n"] == iid["n"] == 72
    assert clustered["n_clusters"] == 24
    assert iid["n_clusters"] == 72
    wc = clustered["ci"][1] - clustered["ci"][0]
    wi = iid["ci"][1] - iid["ci"][0]
    # sqrt(3) is the ideal ratio when within-pair variance is negligible
    assert wc / wi > 1.4, f"cluster bootstrap did not widen the interval: {wc/wi:.2f}"


def test_paired_contrast_clustering_is_a_noop_with_one_row_per_pair():
    """Guards the default: turning clustering on must not change a design that
    already has exactly one row per cluster."""
    a, b = _panel_rows(templates=("T1_strong",))
    clustered = es.paired_contrast(a, b, ("template", "pair"), 4000,
                                   np.random.default_rng(5))
    iid = es.paired_contrast(a, b, ("template", "pair"), 4000,
                             np.random.default_rng(5), cluster_on=None)
    assert clustered["n_clusters"] == iid["n_clusters"] == 24
    assert clustered["ci"] == pytest.approx(iid["ci"], abs=1e-12)


def test_paired_contrast_ignores_cluster_field_absent_from_key():
    a, b = _panel_rows()
    r = es.paired_contrast(a, b, ("template", "pair"), 2000,
                           np.random.default_rng(7), cluster_on="model")
    assert r["n_clusters"] == r["n"] == 72


# ---------------------------------------------------------------------------
# THE REDUCTION THE MECHANISM-PANEL CLASS CONTRAST RESTS ON.
#
# analyze_mech_panel.class_panel() replaces each contrast by one number per NAME
# PAIR -- that pair's mean over templates -- and bootstraps those. Everything
# built on top of it (the interval on the difference between the two condition
# classes, and the injected-mechanism power calibration that gives the section
# its floor) is only a bootstrap of the right thing because the panel is
# BALANCED: every pair contributes the same number of cells to every contrast,
# so "resample clusters and concatenate their rows" and "resample pair means"
# are the same arithmetic. Unbalance the panel and the equality fails silently,
# taking every interval computed from the reduction with it. Pinned here rather
# than trusted.
# ---------------------------------------------------------------------------
def _cell_deltas_by_pair(a, b):
    ka = {(r["template"], r["pair"]): r["white_margin"] - r["black_margin"]
          for r in a}
    kb = {(r["template"], r["pair"]): r["white_margin"] - r["black_margin"]
          for r in b}
    by = {}
    for k in sorted(set(ka) & set(kb)):
        by.setdefault(k[1], []).append(ka[k] - kb[k])
    return by


def test_pair_mean_reduction_is_the_cluster_bootstrap():
    a, b = _panel_rows()
    by = _cell_deltas_by_pair(a, b)
    pairs = sorted(by)
    means = np.array([np.mean(by[p]) for p in pairs])
    rng = np.random.default_rng(11)
    for _ in range(200):
        pick = rng.integers(0, len(pairs), size=len(pairs))
        concat = np.concatenate([by[pairs[i]] for i in pick])
        assert concat.mean() == pytest.approx(means[pick].mean(), abs=1e-12)


def test_pair_mean_reduction_matches_paired_contrast_point_estimate():
    a, b = _panel_rows()
    by = _cell_deltas_by_pair(a, b)
    means = np.array([np.mean(by[p]) for p in sorted(by)])
    r = es.paired_contrast(a, b, ("template", "pair"), 500,
                           np.random.default_rng(2))
    assert r["logodds"] == pytest.approx(float(means.mean()), abs=1e-12)


def test_pair_mean_reduction_breaks_when_the_panel_is_unbalanced():
    """The guard has to be able to fail, or it is not a guard."""
    a, b = _panel_rows()
    a = [r for r in a if not (r["pair"] == 0 and r["template"] == "T2_mid")]
    by = _cell_deltas_by_pair(a, b)
    assert len(by[0]) == 2 and len(by[1]) == 3, "fixture did not unbalance pair 0"
    pairs = sorted(by)
    means = np.array([np.mean(by[p]) for p in pairs])
    # the short pair now carries less weight in the concatenation than it does
    # in the mean of the pair means, so the two disagree
    concat = np.concatenate([by[p] for p in pairs])
    assert concat.mean() != pytest.approx(means.mean(), abs=1e-12)
