"""Effect sizes for a paired margin, on every scale the paper needs, and an
honest account of why the scale matters more here than it usually does.

--------------------------------------------------------------------------
WHY THIS MODULE EXISTS: A CORRECTION.
--------------------------------------------------------------------------
Studies 2 and 3 reported the demographic effect in percentage points via

    LOGIT_TO_PP = 25.0        # dlogit x p(1-p) x 100, evaluated at p = 0.5

with a comment calling it an upper bound. It is an upper bound. It is also, on
this panel, a very loose one, and the looseness is not uniform across models,
which is worse than being loose. Measured on the Study 2 data, the ratio of the
p = 0.5 conversion to the exact probability-scale effect is

    llama-2-7b-chat            1.8x
    llama-3.1-8b-instruct      7.8x
    mistral-7b-instruct-v0.1  66.9x
    mistral-7b-instruct-v0.3 108.5x

The reason is visible in the distribution of the renormalised preference
p = P(yes) / (P(yes) + P(no)). The four models do not live in the same place:

    llama-2-7b-chat            6.5% of cells saturated (p > .99 or p < .01)
    llama-3.1-8b-instruct     44.0%   bimodal, median 0.012 with an upper mode
    mistral-7b-instruct-v0.1   0.0%   never saturated, but always below p = .51
    mistral-7b-instruct-v0.3  99.7%   saturated almost everywhere

A shared Jacobian of 0.25 assigns every model the sensitivity of a model sitting
at p = 0.5. Only one of these is near p = 0.5, and one of them is essentially
never anywhere except 0 or 1. Cross-model comparison in those percentage points
is therefore not comparing like with like, and the paper did it.

--------------------------------------------------------------------------
WHAT REPLACES IT.
--------------------------------------------------------------------------
Three scales, each reported for what it is good for, none doing a job it cannot.

1. LOG-ODDS MARGIN, d = m_white - m_black. The raw measurement. Unbounded, so
   it is the right scale for the hierarchical variance decomposition, and it is
   the only scale on which a saturated model still carries information. It is
   not interpretable to a reader, and it is not comparable to the correspondence
   literature.

2. PROBABILITY OF SUPERIORITY, Ps = P(d > 0), with Cliff's delta = 2 Ps - 1.
   THE PRIMARY EFFECT SIZE. The fraction of matched pairs in which the
   White-named resume is ranked above the Black-named one. Bounded, free of the
   model's operating point, and decision-relevant, because shortlisting is a
   ranking operation and not a probability readout. This is the common-language
   effect size of McGraw and Wong (1992) and Cliff (1993), so it is a standard
   statistic rather than one invented here.

3. THRESHOLD RATE GAP. Rank every candidate by margin, take the top k, and
   report the difference in selection rate between the two name groups. This is
   the only quantity in the paper that is genuinely commensurable with a
   callback-rate gap of the Bertrand and Mullainathan kind, and it is reported
   at several stated thresholds because the choice of k is itself a researcher
   degree of freedom.

Percentage points are still reported, as an INTERVAL from the exact local-slope
value to the p = 0.5 bound, never as a bare number. That the interval spans two
orders of magnitude for two of the four models is not an embarrassment to be
hidden; it is the same phenomenon the paper is about. The reporting scale is
part of the instrument, and two papers describing one model can differ by 100x
without either being wrong.
"""
from __future__ import annotations

import math

import numpy as np

# Retained for provenance: this is what Studies 2 and 3 used. Present so old
# artifacts can be recomputed and the correction can be shown, not so new
# numbers can be produced with it.
LEGACY_LOGIT_TO_PP = 25.0

# THE SAME NUMBER, USED HONESTLY. 25.0 is 100 x dp/dlogit at p = 0.5, and p(1-p)
# is maximised there, so 25 x |log-odds| is the LARGEST percentage-point movement
# any log-odds quantity could correspond to. As a point estimate of a pp effect
# that is wrong by up to 108x on this panel (see above). As an upper bound it is
# exactly correct, and it is the only pp statement that can be made about a
# quantity -- a variance component, say -- that has no single p attached to it.
#
# Anything converted with this constant must be reported as a bound and named as
# one. `describe()` reports the exact probability-scale effect instead, per cell,
# and that is what the paper's effect sizes use.
PP_PER_LOGIT_MAX = 25.0


def prob(margin: float) -> float:
    """Renormalised preference P(yes) / (P(yes) + P(no)) from the log-odds."""
    return 1.0 / (1.0 + math.exp(-margin))


def paired_deltas(rows) -> np.ndarray:
    return np.array([r["white_margin"] - r["black_margin"] for r in rows], dtype=float)


def paired_prob_deltas(rows) -> np.ndarray:
    return np.array([prob(r["white_margin"]) - prob(r["black_margin"])
                     for r in rows], dtype=float)


def local_slope(rows) -> float:
    """Mean p(1-p) at the pair's own operating point.

    The delta-method Jacobian evaluated where the model actually sits, rather
    than where it would be most sensitive.
    """
    s = []
    for r in rows:
        pw, pb = prob(r["white_margin"]), prob(r["black_margin"])
        pbar = 0.5 * (pw + pb)
        s.append(pbar * (1.0 - pbar))
    return float(np.mean(s)) if s else float("nan")


def superiority(d: np.ndarray) -> tuple[float, float]:
    """(Ps, Cliff's delta). Ties, which do not occur here, split evenly."""
    if len(d) == 0:
        return float("nan"), float("nan")
    gt = float((d > 0).mean())
    lt = float((d < 0).mean())
    return gt + 0.5 * (1.0 - gt - lt), gt - lt


def boot_ci(x: np.ndarray, stat, n_boot: int = 10000, rng=None, alpha: float = 0.05,
            clusters=None):
    """Percentile bootstrap, optionally resampling CLUSTERS rather than rows.

    WHY CLUSTERING MATTERS HERE, AND WHERE IT DOES NOT.

    The design is crossed: every name pair appears under every wording and every
    résumé template. Pooling those rows and resampling them independently treats
    36 observations of the same twelve names as 36 independent draws, and the
    interval comes out far too narrow. Measured against a bootstrap that
    resamples name pairs instead, the i.i.d. interval on the per-model effect is
    too narrow by a factor of **3.1 to 4.8** across the four models.

    IT MATTERS LESS FOR THE PAIRED CONTRASTS, BUT IT IS NOT FREE. A contrast is
    a difference of two conditions measured on the same cell, so the name and
    the résumé cancel inside each observation before any resampling happens.
    On the two-template panel that cancellation was complete enough that the
    clustered and i.i.d. intervals were interchangeable: the width ratio ran
    from 0.83 to 1.14 across 168 contrasts, and this docstring previously
    concluded that the contrasts were sound computed either way.

    THAT CONCLUSION DID NOT SURVIVE THE THIRD TEMPLATE. Re-measured on the
    complete panel -- three templates, so three rows per name pair rather than
    two -- the ratio runs from 0.639 to 1.462. The median is still 1.010, so on
    a typical contrast the choice genuinely does not matter, but 33.9% of
    contrasts widen by more than 10% and 8.9% by more than 25% once name pairs
    are resampled as units. Cancellation within a pair is partial, not total,
    and adding a third correlated row per pair exposed the difference.

    Every contrast in Studies 3 and 5 is therefore computed with `clusters`
    set to the name pair. "Usually indistinguishable" is not a reason to use
    the estimator that is wrong when they differ, and this paper's own subject
    is what happens when that reasoning is accepted.

    It also does not matter where one row per cluster is being resampled
    already: a single specification (one wording, one template) has exactly one
    row per name pair, so the row bootstrap IS the cluster bootstrap there.

    `clusters` is an array of cluster labels parallel to `x`. Clusters are drawn
    with replacement and their member rows concatenated, which is the standard
    nonparametric cluster bootstrap.
    """
    rng = rng or np.random.default_rng(20260728)
    n = len(x)
    if n == 0:
        return None
    if clusters is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        b = np.array([stat(x[i]) for i in idx])
    else:
        clusters = np.asarray(clusters)
        groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
        k = len(groups)
        b = np.empty(n_boot)
        for j in range(n_boot):
            pick = rng.integers(0, k, size=k)
            b[j] = stat(x[np.concatenate([groups[i] for i in pick])])
    return dict(est=float(stat(x)),
                ci=[float(np.percentile(b, 100 * alpha / 2)),
                    float(np.percentile(b, 100 * (1 - alpha / 2)))],
                boots=b)


def pvalue_from_boots(b: np.ndarray, est: float, null: float = 0.0,
                      n_boot: int | None = None) -> float:
    """Two-sided p by shifting the bootstrap distribution to the null.

    Floored at 1/n_boot: a bootstrap cannot resolve a p-value finer than its own
    resolution, and reporting p = 0 would claim it can.
    """
    n_boot = n_boot or len(b)
    centred = b - est + null
    p = float((np.abs(centred - null) >= abs(est - null)).mean())
    return max(p, 1.0 / n_boot)


def describe(rows, n_boot: int = 10000, rng=None, cluster_key="pair") -> dict | None:
    """Every scale for one set of matched pairs.

    Intervals resample NAME PAIRS, not rows, whenever the rows carry a `pair`
    field and more than one row shares a pair. See boot_ci for why: pooling a
    crossed design and resampling rows independently understated these
    intervals by a factor of 3.1 to 4.8. When there is exactly one row per pair
    the two are identical and nothing changes.

    `pp_interval` is the honest percentage-point statement: [exact local, p=0.5
    bound], sorted, with the legacy number alongside so the correction is
    visible rather than silent.
    """
    rows = [r for r in rows
            if r.get("white_margin") is not None and r.get("black_margin") is not None]
    if not rows:
        return None
    rng = rng or np.random.default_rng(20260728)
    d = paired_deltas(rows)
    dp = paired_prob_deltas(rows)
    n = len(d)

    cl = None
    if cluster_key and all(cluster_key in r for r in rows):
        labels = np.array([r[cluster_key] for r in rows])
        if len(np.unique(labels)) < n:      # rows genuinely share clusters
            cl = labels

    bl = boot_ci(d, lambda a: float(a.mean()), n_boot, rng, clusters=cl)
    bp = boot_ci(dp, lambda a: float(a.mean()) * 100.0, n_boot, rng, clusters=cl)
    bs = boot_ci(d, lambda a: superiority(a)[0], n_boot, rng, clusters=cl)

    slope = local_slope(rows)
    lo = bl["est"] * slope * 100.0
    hi = bl["est"] * 0.25 * 100.0
    ps, cliff = superiority(d)

    return dict(
        n=n,
        logodds=dict(est=bl["est"], ci=bl["ci"],
                     p=pvalue_from_boots(bl["boots"], bl["est"], 0.0, n_boot)),
        prob_pp=dict(est=bp["est"], ci=bp["ci"],
                     p=pvalue_from_boots(bp["boots"], bp["est"], 0.0, n_boot)),
        superiority=dict(est=ps, ci=bs["ci"], cliff=cliff,
                         p=pvalue_from_boots(bs["boots"], ps, 0.5, n_boot)),
        local_slope=slope,
        clustered=cl is not None,
        n_clusters=int(len(np.unique(cl))) if cl is not None else n,
        pp_interval=sorted([lo, hi], key=abs),
        legacy_pp=bl["est"] * LEGACY_LOGIT_TO_PP,
        saturated_frac=float(np.mean([
            (prob(r[s + "_margin"]) > 0.99 or prob(r[s + "_margin"]) < 0.01)
            for r in rows for s in ("white", "black")])),
    )


def paired_contrast(a, b, key=("template", "pair"), n_boot: int = 10000, rng=None,
                    cluster_on: str | None = "pair"):
    """Difference between two conditions, paired on the shared cell key.

    Pairing on the cell rather than resampling the two conditions independently
    is what makes the contrast a within-pair comparison. Every condition sees
    the same names and the same resumes, so almost all the between-pair variance
    cancels, and treating them as independent samples would discard that.

    THE RESAMPLING UNIT. `key` identifies a cell, and on the mechanism panel a
    cell is (template, name pair) -- so one name pair contributes three rows,
    one per template, and those three are not independent draws. `cluster_on`
    names the field within `key` that is the true unit of randomisation; its
    values become cluster labels and the bootstrap resamples pairs rather than
    rows. Set it to None to recover the row bootstrap.

    When `cluster_on` is absent from `key`, or every key has a distinct value
    for it, the two are identical and nothing changes -- so this is safe to
    leave on by default.
    """
    rng = rng or np.random.default_rng(20260728)
    ka = {tuple(r[k] for k in key): r["white_margin"] - r["black_margin"]
          for r in a if r.get("white_margin") is not None and r.get("black_margin") is not None}
    kb = {tuple(r[k] for k in key): r["white_margin"] - r["black_margin"]
          for r in b if r.get("white_margin") is not None and r.get("black_margin") is not None}
    keys = sorted(set(ka) & set(kb))
    if not keys:
        return None
    d = np.array([ka[k] - kb[k] for k in keys], dtype=float)
    clusters = None
    if cluster_on is not None and cluster_on in key:
        pos = list(key).index(cluster_on)
        clusters = np.array([k[pos] for k in keys])
    bt = boot_ci(d, lambda x: float(x.mean()), n_boot, rng, clusters=clusters)
    return dict(n=len(keys), logodds=bt["est"], ci=bt["ci"],
                n_clusters=int(len(np.unique(clusters))) if clusters is not None else len(keys),
                p=pvalue_from_boots(bt["boots"], bt["est"], 0.0, n_boot))


def benjamini_hochberg(ps: list[float]) -> list[float]:
    """BH-adjusted p-values, order preserved, monotone by construction."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        prev = min(prev, ps[i] * m / k)
        adj[i] = min(1.0, prev)
    return adj
