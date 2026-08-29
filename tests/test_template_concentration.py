"""Tests for the template-concentration statistic, which moved a conclusion.

The paper reported that condition effects concentrate on the middling résumé.
The statistic that established it compared |one estimate| against |the average
of two|, and averaging shrinks a magnitude even when nothing differs, so the
statistic returned a concentration under its own null. Five sixths of the
published value was that artefact. The corrected form and the biased one are
both retained in the artifact, so both need tests: one that the old statistic
really is biased in the direction claimed, and one that the new statistic is
not. Without the first, a future reader has only our word that the correction
was necessary.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

SRC = pathlib.Path(__file__).resolve().parents[1] / "paper-a" / "src"
sys.path.insert(0, str(SRC))

import analyze_template_concentration as tc  # noqa: E402

# E|N(0, s^2)| = s*sqrt(2/pi); averaging two independent draws first divides
# the standard deviation by sqrt(2), so the legacy statistic returns
# s*sqrt(2/pi)*(1 - 1/sqrt(2)) rather than zero.
BIAS_PER_SD = float(np.sqrt(2 / np.pi) * (1 - 1 / np.sqrt(2)))


def _exchangeable(mu, sd, n_pairs=24, reps=4000, seed=7):
    """Three templates carrying the SAME true shift mu, i.i.d. noise sd."""
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sd, size=(reps, n_pairs, 3))


def test_bias_constant_is_what_the_docstring_says():
    """0.2337 is quoted in the analysis docstring and in the artifact."""
    assert abs(BIAS_PER_SD - 0.2337) < 5e-5


def test_legacy_statistic_is_biased_positive_under_the_null():
    for sd in (0.05, 0.10, 0.20):
        s = _exchangeable(0.0, sd)
        got = float(tc.stat_legacy(s).mean())
        assert got > 0.0
        assert abs(got - BIAS_PER_SD * sd) < 0.05 * BIAS_PER_SD * sd


def test_corrected_statistic_is_unbiased_under_the_null():
    for mu in (0.0, 0.05, 0.40):
        for sd in (0.05, 0.10, 0.20):
            s = _exchangeable(mu, sd)
            assert abs(float(tc.stat_unbiased(s).mean())) < 0.002


def test_correction_matters_most_where_the_effect_is_small():
    """The artefact shrinks as the true shift grows relative to the noise.

    This is why it was invisible: on a model with large shifts the two
    statistics agree, and on a model with small ones the legacy statistic
    manufactures the concentration.
    """
    small = float(tc.stat_legacy(_exchangeable(0.0, 0.10)).mean())
    large = float(tc.stat_legacy(_exchangeable(0.60, 0.10)).mean())
    assert small > 10 * abs(large)


def test_corrected_statistic_still_finds_a_real_concentration():
    """Unbiased is not the same as blind. Give T2 a genuinely larger shift."""
    rng = np.random.default_rng(11)
    s = rng.normal(0.10, 0.10, size=(4000, 24, 3))
    s[:, :, 0] += 0.30                      # column 0 is T2 by construction
    assert float(tc.stat_unbiased(s).mean()) > 0.25


def test_t2_is_column_zero():
    assert tc.TEMPLATES[0] == tc.T2
    assert set(tc.TEMPLATES[1:]) == set(tc.OTHER)


def test_permutation_null_zeroes_the_corrected_statistic_not_the_legacy_one():
    """The null the test claims to test is exchangeability, not zero shift."""
    rng = np.random.default_rng(3)
    s = rng.normal(0.05, 0.15, size=(24, 3))
    legacy, unbiased = tc.permutation_null(s, n_perm=4000,
                                           rng=np.random.default_rng(4))
    assert legacy > 0.01
    assert abs(unbiased) < 0.005


def test_perm_pvalue_centres_on_the_null_mean():
    """A p-value against zero would be the wrong test for the legacy statistic."""
    null = np.random.default_rng(5).normal(0.024, 0.004, size=5000)
    assert tc.perm_pvalue(null, 0.026) > 0.30          # sits inside its own null
    assert tc.perm_pvalue(null, 0.000) < 0.01          # zero does not
