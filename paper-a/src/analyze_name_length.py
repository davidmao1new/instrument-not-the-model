"""Is part of the measured "race effect" a token-length effect?

THE THREAT. A matched-pair audit's warrant is that the two prompts are identical
apart from the name. That holds for the characters. It does not hold for the
TOKENS, and tokens are what the model receives. Measured across the 48-pair
factorial grid, only about a third of pairs put the same number of tokens on
both sides; the Black-associated name is longer in half of them, by up to two
tokens.

WHY THIS PAPER CAN ACTUALLY TEST IT. Study 5's conditions D8 and D9 displace the
name by exactly one and two tokens while destroying no delimiter and changing no
word, and on several checkpoints those displacements move the measured effect by
an amount comparable to the effect itself. So a two-token displacement is
established as non-free in a controlled design -- and the matched-pair design
embeds one, correlated with race by construction.

THE TEST. For each model, regress the per-pair demographic effect on the
per-pair token-length difference, over the full grid with wording and template
averaged within pair. Bootstrap over pairs, since the pair is the unit.

    delta_pair = a + b * (tokens(black name) - tokens(white name)) + e

  b indistinguishable from zero  -> the design's warrant survives a threat that
                                    had not previously been checked.
  b distinguishable from zero    -> part of the measured effect is attributable
                                    to tokenisation, and correspondence audits of
                                    language models need a length control.

A SECOND, SHARPER CUT. Restrict to the subset of pairs that DO tokenise to the
same length and re-estimate the demographic effect there. If the effect is
unchanged, length is not carrying it. That subset analysis is the one a sceptical
reader will want, because it needs no model at all.

A CORRECTION TO THAT SECOND CUT, ADDED LATER. The subset analysis above
resampled its rows independently, and its rows are not independent. The name
grid is a complete 12 x 4 rectangle: twelve FIRST-NAME pairs, each crossed with
the same four surname pairs. Token length in context is driven mostly by the
first name, so the token-matched subset does not draw a scattered sample of the
48 cells -- on Llama-3.1-8B-Instruct it is exactly three first-name pairs
(Anne/Ebony, Emily/Kenya, Brett/Jamal) crossed with all four surnames. The "12
pairs" in Table 9 are 3 clusters of 4, not 12 independent draws, and the
interval and p-value were computed as if they were 12.

This is the error Section 6.1 of this paper is about. It was committed in the
one place where nobody looked for it, because the subset analysis was sold as
the version that "needs no model at all" -- which is true of its point estimate
and false of its interval. A paper that argues the resampling unit is a
researcher degree of freedom cannot get its own resampling unit wrong in a
table, so the subset arm is recomputed here with the FIRST-NAME PAIR as the
unit, and the old i.i.d. numbers are kept alongside under _iid_superseded keys
so the correction is visible rather than silent.

WITH THREE CLUSTERS THE BOOTSTRAP ITSELF IS NOT TRUSTWORTHY, which is part of
the finding rather than a caveat to it. Drawing 3 clusters from 3 with
replacement puts an 11.1% atom on "all three draws are the same cluster", so
the 2.5th and 97.5th percentiles are pinned to the smallest and largest of the
three cluster means and the interval cannot be a 95% interval at all. Three
answers are therefore reported for the difference, and they are reported
together because they disagree: the cluster bootstrap the paper uses elsewhere,
an exact cluster-relabelling permutation test (enumerable at 220 outcomes when
the matched subset is whole clusters), and a cluster-level t interval on k - 1
degrees of freedom. Quoting whichever of the three is smallest would be the
same degree of freedom this paper spends sixty pages measuring.

    .venv/Scripts/python.exe paper-a/src/analyze_name_length.py
"""
from __future__ import annotations

import itertools
import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import effectsize as es  # noqa: E402
import stimuli as st  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
INSTR = ROOT / "paper-a" / "data" / "instrument"
NAMES = ROOT / "paper-a" / "data" / "names"
OUT = INSTR / "name_length_effect.json"
N_BOOT = 20_000
RNG = np.random.default_rng(20260730)

# The clustered recomputation draws from its OWN generator, and runs after
# everything above has finished drawing from RNG. Both are deliberate: the
# i.i.d. numbers this section supersedes have to keep reproducing bit for bit,
# or the artifact could no longer show what was corrected.
CLUSTER_SEED = 20260801
N_PERM = 200_000
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}  # two-sided 95%

ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SHORT = {"llama-2-7b-chat": "Llama-2-7B-chat",
         "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
         "mistral-7b-instruct-v0.1": "Mistral-7B-Instruct v0.1",
         "mistral-7b-instruct-v0.3": "Mistral-7B-Instruct v0.3"}


def load_effects(model):
    """{pair_idx: mean paired margin difference} over the Study 4 grid."""
    by = defaultdict(list)
    for f in sorted(NAMES.glob("names_*.jsonl")):
        for r in st.read_jsonl(f):
            if r.get("model") != model:
                continue
            if r.get("white_margin") is None or r.get("black_margin") is None:
                continue
            by[r["pair"]].append(r["white_margin"] - r["black_margin"])
    return {k: float(np.mean(v)) for k, v in by.items()}, \
           {k: len(v) for k, v in by.items()}


def load_grid(model, eff):
    """The Study 4 name grid as parallel arrays, keyed by what varies in it.

    Returns (values, first-name-pair label, surname-pair label, token-matched,
    token-length difference). The two labels are what make the clustering
    visible: the grid is a crossing of the first, the second and nothing else,
    so a subset selected on a tokenizer property can be -- and on one model is
    -- a handful of first names wearing four surnames each.
    """
    L = json.loads((INSTR / f"name_length_{model}.json").read_text(encoding="utf-8"))
    meta = {r["idx"]: r for r in L["pairs"]}
    keys = sorted(set(meta) & set(eff))
    y = np.array([eff[k] for k in keys], dtype=float)
    first = np.array([f"{meta[k]['white_first']}/{meta[k]['black_first']}"
                      for k in keys])
    last = np.array([f"{meta[k]['white_last']}/{meta[k]['black_last']}"
                     for k in keys])
    matched = np.array([meta[k]["delta_in_context"] == 0 for k in keys])
    delta = np.array([meta[k]["delta_in_context"] for k in keys], dtype=float)
    return y, first, last, matched, delta


def slope_clustered(y, delta, first):
    """The Table 7 slope with the same unit, reported because it is the same
    defect and not because this finding was about it.

    The regression arm bootstraps the 48 grid cells as 48 draws, and they are
    the same 12 first-name pairs crossed with the same 4 surnames as the subset
    arm. Nobody asked for this number; leaving it uncomputed after establishing
    that the grid is clustered would be choosing not to look. It is written to
    its own key and wired to nothing, for the author to adjudicate.
    """
    cl = sorted(set(first.tolist()))
    groups = [np.flatnonzero(first == c) for c in cl]
    k = len(groups)

    def fit(i):
        xi, yi = delta[i], y[i]
        return 0.0 if xi.std() == 0 else float(np.polyfit(xi, yi, 1)[0])

    rng = np.random.default_rng(CLUSTER_SEED)
    b = np.array([fit(np.concatenate([groups[i] for i in rng.integers(0, k, k)]))
                  for _ in range(N_BOOT)])
    est = fit(np.arange(len(y)))
    ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return dict(est=est, ci=ci, p=es.pvalue_from_boots(b, est, 0.0, N_BOOT),
                n=int(len(y)), n_clusters=k, cluster_unit="first_name_pair")


def _boot_clusters(y, groups, rng, n_boot=N_BOOT):
    """Mean of y over a cluster bootstrap: draw k clusters from k, concatenate."""
    k = len(groups)
    return np.array([y[np.concatenate([groups[i]
                                       for i in rng.integers(0, k, k)])].mean()
                     for _ in range(n_boot)])


def _cluster_t(y, groups):
    """Cluster-level t interval: the textbook remedy when k is small.

    The statistic is the UNWEIGHTED mean of the per-cluster means, which equals
    the row mean only when the clusters are balanced. It is reported next to the
    row mean rather than instead of it, because the row mean is what Table 9
    quotes and the two answer slightly different questions when they differ.
    """
    means = np.array([y[g].mean() for g in groups], dtype=float)
    k = len(means)
    if k < 2:
        return dict(est=float(means[0]) if k else float("nan"),
                    ci=[float("nan"), float("nan")], df=0, se=float("nan"),
                    cluster_means=[float(v) for v in means])
    se = float(means.std(ddof=1) / math.sqrt(k))
    t = T_CRIT.get(k - 1, 1.96)
    m = float(means.mean())
    return dict(est=m, ci=[m - t * se, m + t * se], df=k - 1, se=se,
                t_crit=t, cluster_means=[float(v) for v in means])


def _relabel_test(y, first, last, matched, rng):
    """Exact-where-possible test: does the matched subset differ from the grid
    only because of WHICH first-name pairs happen to fall in it?

    The null relabels the first-name clusters and carries the matched pattern
    along, holding the number of matched cells and their surname composition
    fixed. It needs no distributional assumption and no bootstrap, so it is the
    one statement here that a three-cluster subset cannot break. When every
    matched cluster is matched in all four of its cells the relabelling reduces
    to choosing k of 12 clusters, and all C(12, k) outcomes are enumerated.
    """
    cl = sorted(set(first.tolist()))
    sl = sorted(set(last.tolist()))
    pos = {(c, s): np.flatnonzero((first == c) & (last == s)) for c in cl for s in sl}
    if any(len(v) != 1 for v in pos.values()) or len(cl) * len(sl) != len(y):
        return None                       # not a complete rectangle; no test
    Y = np.array([[y[pos[(c, s)][0]] for s in sl] for c in cl])
    M = np.array([[bool(matched[pos[(c, s)][0]]) for s in sl] for c in cl])
    nm = int(M.sum())
    if nm == 0:
        return None
    obs = float(Y[M].mean() - Y.mean())
    per_cluster = M.sum(1)
    whole = bool(np.all((per_cluster == 0) | (per_cluster == M.shape[1])))
    if whole:
        km = int((per_cluster > 0).sum())
        mu = Y.mean(1)
        grand = float(Y.mean())
        stats = np.array([mu[list(c)].mean() - grand
                          for c in itertools.combinations(range(len(cl)), km)])
        exact = True
    else:
        stats = np.empty(N_PERM)
        for i in range(N_PERM):
            stats[i] = (Y * M[rng.permutation(len(cl))]).sum() / nm - Y.mean()
        exact = False
    p = float((np.abs(stats) >= abs(obs) - 1e-12).mean())
    return dict(obs=obs, p=max(p, 1.0 / len(stats)), exact=exact,
                n_outcomes=int(len(stats)),
                whole_cluster_subset=whole)


def token_matched_clustered(y, first, last, matched):
    """Table 9's subset arm, recomputed with the first-name pair as the unit.

    Everything here is the same statistic on the same rows as the i.i.d.
    version. Only the resampling unit changes, so every point estimate is
    unchanged by construction and only the uncertainty moves. That is the point:
    the defect was never in the estimate.
    """
    cl = sorted(set(first.tolist()))
    groups_all = [np.flatnonzero(first == c) for c in cl]
    mcl = [c for c in cl if bool((matched & (first == c)).any())]
    groups_m = [np.flatnonzero(matched & (first == c)) for c in mcl]
    k_all, k_m = len(groups_all), len(groups_m)

    est_all = float(y.mean())
    est_m = float(y[matched].mean())

    ba = _boot_clusters(y, groups_all, np.random.default_rng(CLUSTER_SEED))
    bm = _boot_clusters(y, groups_m, np.random.default_rng(CLUSTER_SEED))

    # The difference, resampled JOINTLY: the matched rows are a subset of the
    # grid rows, not a second independent sample, and the old two-sample
    # bootstrap treated them as one. Draws that happen to contain no matched
    # cluster are undefined rather than zero, and are counted, not silently
    # dropped -- on Llama-3.1 they are 3% of draws precisely because only three
    # of the twelve clusters carry a matched cell.
    rng = np.random.default_rng(CLUSTER_SEED)
    dif, rat, degen = [], [], 0
    for _ in range(N_BOOT):
        idx = np.concatenate([groups_all[i] for i in rng.integers(0, k_all, k_all)])
        sub = idx[matched[idx]]
        if len(sub) == 0:
            degen += 1
            continue
        ma, mm = float(y[idx].mean()), float(y[sub].mean())
        dif.append(mm - ma)
        if ma != 0.0:
            rat.append(abs(mm) / abs(ma) - 1.0)
    dif = np.array(dif)
    rat = np.array(rat)

    # And the same difference with the two arms drawn independently, which is
    # what the superseded number did, so the clustering effect can be separated
    # from the nesting fix rather than confounded with it.
    rng2 = np.random.default_rng(CLUSTER_SEED)
    di = np.array([
        y[np.concatenate([groups_m[i] for i in rng2.integers(0, k_m, k_m)])].mean()
        - y[np.concatenate([groups_all[i]
                            for i in rng2.integers(0, k_all, k_all)])].mean()
        for _ in range(N_BOOT)])

    est_d = est_m - est_all
    ci_d = [float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))]
    perm = _relabel_test(y, first, last, matched,
                         np.random.default_rng(CLUSTER_SEED))
    tt = _cluster_t(y, groups_m)

    # With k clusters the bootstrap puts an atom of k^-(k-1) on "every draw is
    # the same cluster". At k = 3 that is 11.1%, above 2.5%, so the percentile
    # interval is pinned to the extreme cluster means and is not a 95% interval.
    atom = float(k_m ** (-(k_m - 1)))
    degenerate = bool(atom > 0.025)

    out = dict(
        cluster_unit="first_name_pair",
        n_grid_rows=int(len(y)),
        n_grid_clusters=k_all,
        n_matched_rows=int(matched.sum()),
        n_matched_clusters=k_m,
        matched_cluster_labels=mcl,
        matched_cluster_sizes=[int(len(g)) for g in groups_m],
        n_surname_pairs_in_matched=int(len(set(last[matched].tolist()))),
        effect_all_pairs=dict(
            est=est_all, ci=[float(np.percentile(ba, 2.5)),
                             float(np.percentile(ba, 97.5))],
            p=es.pvalue_from_boots(ba, est_all, 0.0, N_BOOT),
            n=int(len(y)), n_clusters=k_all),
        effect_token_matched=dict(
            est=est_m, ci=[float(np.percentile(bm, 2.5)),
                           float(np.percentile(bm, 97.5))],
            p=es.pvalue_from_boots(bm, est_m, 0.0, N_BOOT),
            n=int(matched.sum()), n_clusters=k_m),
        matched_minus_all=dict(
            est=est_d, ci=ci_d,
            p=es.pvalue_from_boots(dif, est_d, 0.0, len(dif)),
            n_draws=int(len(dif)), n_draws_degenerate=degen),
        matched_minus_all_independent_arms=dict(
            est=float(di.mean()),
            ci=[float(np.percentile(di, 2.5)), float(np.percentile(di, 97.5))],
            p=es.pvalue_from_boots(di, float(di.mean()), 0.0, N_BOOT)),
        growth_ratio=dict(
            est=(abs(est_m) / abs(est_all) - 1.0) if est_all else float("nan"),
            ci=[float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5))]
            if len(rat) else [float("nan"), float("nan")]),
        cluster_t_token_matched=tt,
        cluster_relabel_test=perm,
        bootstrap_degenerate=degenerate,
        same_cluster_draw_probability=atom,
        n_boot=N_BOOT, seed=CLUSTER_SEED)
    # A ratio against an effect that is itself indistinguishable from zero has
    # no interpretation, whatever its point value.
    out["growth_ratio"]["interpretable"] = bool(
        out["effect_all_pairs"]["ci"][0] * out["effect_all_pairs"]["ci"][1] > 0)
    out["difference_significant"] = bool(out["matched_minus_all"]["p"] < 0.05)
    out["difference_significant_by_ci"] = bool(ci_d[0] * ci_d[1] > 0)
    out["difference_significant_by_relabel"] = bool(
        perm is not None and perm["p"] < 0.05)
    return out


def main() -> int:
    out = {}
    print("=" * 100)
    print("IS PART OF THE MEASURED EFFECT A TOKEN-LENGTH EFFECT?")
    print("Per-pair demographic effect regressed on the pair's token-length")
    print("difference. The pair is the resampling unit.")
    print("=" * 100)
    print(f"{'model':<26}{'pairs':>7}{'same len':>10}{'slope b':>11}"
          f"{'95% CI':>22}{'p':>8}{'r':>8}")

    for m in ORDER:
        lf = INSTR / f"name_length_{m}.json"
        if not lf.exists():
            continue
        L = json.loads(lf.read_text(encoding="utf-8"))
        dl = {r["idx"]: r["delta_in_context"] for r in L["pairs"]}
        eff, nobs = load_effects(m)
        keys = sorted(set(dl) & set(eff))
        if len(keys) < 20:
            print(f"{m:<26}  only {len(keys)} shared pairs; skipped")
            continue
        x = np.array([dl[k] for k in keys], dtype=float)
        y = np.array([eff[k] for k in keys], dtype=float)

        def slope(idx):
            xi, yi = x[idx], y[idx]
            if xi.std() == 0:
                return 0.0
            return float(np.polyfit(xi, yi, 1)[0])

        n = len(keys)
        boots = np.array([slope(RNG.integers(0, n, n)) for _ in range(N_BOOT)])
        est = slope(np.arange(n))
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        p = es.pvalue_from_boots(boots, est, 0.0, N_BOOT)
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() else float("nan")
        same = int((x == 0).sum())
        out[m] = dict(n_pairs=n, n_same_length=same, slope=est, ci=ci, p=p,
                      correlation=r)
        print(f"{m:<26}{n:>7}{same:>10}{est:>+11.4f}"
              f"{f'[{ci[0]:+.4f},{ci[1]:+.4f}]':>22}{p:>8.3f}{r:>+8.3f}")

    # ---- the subset analysis a sceptic will want --------------------------
    print("\n" + "=" * 100)
    print("THE EFFECT ON EQUAL-LENGTH PAIRS ONLY. No model, no regression: just")
    print("the demographic effect recomputed on the pairs where both names")
    print("occupy the same number of tokens.")
    print("=" * 100)
    print(f"{'model':<26}{'all pairs':>24}{'equal-length only':>26}"
          f"{'difference':>24}{'p':>8}")
    for m in ORDER:
        if m not in out:
            continue
        L = json.loads((INSTR / f"name_length_{m}.json").read_text(encoding="utf-8"))
        dl = {r["idx"]: r["delta_in_context"] for r in L["pairs"]}
        eff, _ = load_effects(m)
        keys = sorted(set(dl) & set(eff))
        allv = np.array([eff[k] for k in keys])
        eqk = [k for k in keys if dl[k] == 0]
        if len(eqk) < 8:
            continue
        eqv = np.array([eff[k] for k in eqk])
        ba = es.boot_ci(allv, lambda z: float(z.mean()), N_BOOT, RNG)
        be = es.boot_ci(eqv, lambda z: float(z.mean()), N_BOOT, RNG)
        # two-sample bootstrap on the difference, pairs independent
        diff = np.array([
            eqv[RNG.integers(0, len(eqv), len(eqv))].mean()
            - allv[RNG.integers(0, len(allv), len(allv))].mean()
            for _ in range(N_BOOT)])
        dp = es.pvalue_from_boots(diff, float(diff.mean()), 0.0, N_BOOT)
        out[m]["effect_all_pairs"] = dict(est=ba["est"], ci=ba["ci"], n=len(allv))
        out[m]["effect_equal_length"] = dict(est=be["est"], ci=be["ci"], n=len(eqv))
        out[m]["equal_minus_all"] = dict(est=float(diff.mean()), p=dp)
        _sa = f"{ba['est']:+.4f} [{ba['ci'][0]:+.4f},{ba['ci'][1]:+.4f}]"
        _se = (f"{be['est']:+.4f} [{be['ci'][0]:+.4f},{be['ci'][1]:+.4f}] "
               f"n={len(eqv)}")
        print(f"{m:<26}{_sa:>24}{_se:>26}{diff.mean():>+24.4f}{dp:>8.3f}")

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    sl = [v for v in out.values() if "slope" in v]
    nsig = sum(1 for v in sl if v["p"] < 0.05)
    dsig = sum(1 for v in out.values()
               if v.get("equal_minus_all", {}).get("p", 1) < 0.05)
    if nsig == 0 and dsig == 0:
        verdict = "NO LENGTH CONFOUND DETECTED"
        print("  NO LENGTH CONFOUND DETECTED. The per-pair effect does not track")
        print("  the token-length difference on any model, and restricting to")
        print("  equal-length pairs does not move the effect. The matched-pair")
        print("  warrant survives a threat that had not been checked. This is")
        print("  worth reporting precisely BECAUSE the paper shows elsewhere that")
        print("  a two-token displacement is not free: the displacement here is")
        print("  evidently too small, or too uncorrelated with the demographic")
        print("  signal, to carry the effect.")
    else:
        verdict = "LENGTH CONFOUND PRESENT"
        print(f"  LENGTH CONFOUND PRESENT on {nsig} of {len(sl)} models by slope")
        print(f"  and {dsig} by the equal-length subset. Part of what the audit")
        print("  attributes to the demographic signal is attributable to how the")
        print("  names tokenise. Correspondence audits of language models need a")
        print("  token-length control, which the field does not currently apply.")
    out["_verdict"] = dict(verdict=verdict, n_models=len(sl),
                           n_slope_significant=nsig,
                           n_subset_significant=dsig)

    # ---- the same subset, with the first-name pair as the resampling unit --
    # Runs last and on its own generator so everything above still reproduces.
    print("\n" + "=" * 100)
    print("THE SUBSET ARM AGAIN, RESAMPLING FIRST-NAME PAIRS. The grid is 12")
    print("first-name pairs x 4 surname pairs. A subset chosen by tokenization")
    print("is not a subset chosen at random: it selects first names.")
    print("=" * 100)
    print(f"{'model':<26}{'matched n':>10}{'clusters':>10}{'surnames':>10}"
          f"{'iid p':>9}{'clustered p':>13}{'relabel p':>11}{'survives':>10}")
    clus = {}
    for m in ORDER:
        if m not in out or "effect_equal_length" not in out[m]:
            continue
        eff, _ = load_effects(m)
        y, first, last, matched, delta = load_grid(m, eff)
        c = token_matched_clustered(y, first, last, matched)
        c["slope_same_unit"] = slope_clustered(y, delta, first)
        clus[m] = c
        # Rule: nothing is deleted. The superseded i.i.d. numbers are copied to
        # keys that say what they are, so that rewiring the live keys to the
        # clustered values cannot lose them.
        out[m]["effect_all_pairs_iid_superseded"] = dict(out[m]["effect_all_pairs"])
        out[m]["effect_equal_length_iid_superseded"] = \
            dict(out[m]["effect_equal_length"])
        out[m]["equal_minus_all_iid_superseded"] = dict(out[m]["equal_minus_all"])
        out[m]["token_matched_first_name_clustered"] = c
        surv = (c["difference_significant"] and c["difference_significant_by_ci"]
                and c["difference_significant_by_relabel"])
        print(f"{m:<26}{c['n_matched_rows']:>10}{c['n_matched_clusters']:>10}"
              f"{c['n_surname_pairs_in_matched']:>10}"
              f"{out[m]['equal_minus_all']['p']:>9.3f}"
              f"{c['matched_minus_all']['p']:>13.3f}"
              f"{(c['cluster_relabel_test'] or {}).get('p', float('nan')):>11.3f}"
              f"{str(surv):>10}")

    for m, c in clus.items():
        print(f"\n  {SHORT.get(m, m)}")
        print(f"    matched subset = {c['n_matched_rows']} rows carrying "
              f"{c['n_matched_clusters']} first-name pair(s): "
              f"{', '.join(c['matched_cluster_labels'])}")
        print(f"    effect, all {c['n_grid_rows']} pairs "
              f"({c['n_grid_clusters']} clusters): "
              f"{c['effect_all_pairs']['est']:+.4f} "
              f"[{c['effect_all_pairs']['ci'][0]:+.4f},"
              f"{c['effect_all_pairs']['ci'][1]:+.4f}]")
        print(f"    effect, token-matched: {c['effect_token_matched']['est']:+.4f} "
              f"[{c['effect_token_matched']['ci'][0]:+.4f},"
              f"{c['effect_token_matched']['ci'][1]:+.4f}]  "
              f"p = {c['effect_token_matched']['p']:.4f}"
              + ("   <- percentile interval pinned to the extreme cluster means; "
                 "not a 95% interval" if c["bootstrap_degenerate"] else ""))
        t = c["cluster_t_token_matched"]
        print(f"    same, cluster-t on {t['df']} df: {t['est']:+.4f} "
              f"[{t['ci'][0]:+.4f},{t['ci'][1]:+.4f}]")
        d = c["matched_minus_all"]
        print(f"    difference: {d['est']:+.4f} [{d['ci'][0]:+.4f},"
              f"{d['ci'][1]:+.4f}]  p = {d['p']:.4f}   "
              f"(was p = {out[m]['equal_minus_all']['p']:.4f} i.i.d.)")
        g = c["growth_ratio"]
        print(f"    growth in magnitude: {g['est']:+.1%} "
              f"[{g['ci'][0]:+.1%},{g['ci'][1]:+.1%}]"
              + ("" if g["interpretable"] else
                 "   (ratio against an effect indistinguishable from zero)"))
        s = c["slope_same_unit"]
        print(f"    [same unit applied to the Table 7 slope, not part of this "
              f"finding: {s['est']:+.4f} [{s['ci'][0]:+.4f},{s['ci'][1]:+.4f}] "
              f"p = {s['p']:.4f}, was p = {out[m]['p']:.4f}]")

    if clus:
        headline = max(clus, key=lambda m: abs(clus[m]["matched_minus_all"]["est"]))
        h = clus[headline]
        survives = (h["difference_significant"]
                    and h["difference_significant_by_ci"]
                    and h["difference_significant_by_relabel"])
        out["_token_matched_clustering"] = dict(
            cluster_unit="first_name_pair",
            grid="12 first-name pairs x 4 surname pairs",
            n_clusters_in_matched_subset={m: c["n_matched_clusters"]
                                          for m, c in clus.items()},
            n_rows_in_matched_subset={m: c["n_matched_rows"]
                                      for m, c in clus.items()},
            headline_model=headline,
            headline_growth=h["growth_ratio"]["est"],
            headline_difference_p_iid=out[headline]["equal_minus_all"]["p"],
            headline_difference_p_clustered=h["matched_minus_all"]["p"],
            headline_difference_p_relabel=(
                h["cluster_relabel_test"] or {}).get("p"),
            headline_survives=bool(survives),
            n_models_difference_significant_iid=dsig,
            n_models_difference_significant_clustered=sum(
                1 for c in clus.values() if c["difference_significant"]),
            n_models_slope_significant_iid=nsig,
            n_models_slope_significant_same_unit=sum(
                1 for c in clus.values() if c["slope_same_unit"]["p"] < 0.05),
            note=("The point estimates are unchanged; only the resampling unit "
                  "is. On the headline model the token-matched subset is three "
                  "first-name pairs crossed with four surnames, so its interval "
                  "was computed on 12 rows that carry 3 independent draws."))
        print("\n" + "=" * 100)
        print("DOES THE HEADLINE SURVIVE THE CORRECT RESAMPLING UNIT?")
        print("=" * 100)
        print(f"  {SHORT.get(headline, headline)}: the effect is still "
              f"{h['growth_ratio']['est']:+.0%} larger in magnitude on the")
        print("  token-matched subset -- the point estimate never depended on the")
        print("  bootstrap. What does not survive is the claim that the DIFFERENCE")
        print("  is distinguishable from zero." if not survives else
              "  and the difference remains distinguishable from zero.")
        if not survives:
            print(f"    i.i.d. over 12 rows      p = "
                  f"{out[headline]['equal_minus_all']['p']:.4f}")
            print(f"    3 first-name clusters    p = "
                  f"{h['matched_minus_all']['p']:.4f}   "
                  f"[{h['matched_minus_all']['ci'][0]:+.4f},"
                  f"{h['matched_minus_all']['ci'][1]:+.4f}]")
            print(f"    independent arms         p = "
                  f"{h['matched_minus_all_independent_arms']['p']:.4f}")
            if h["cluster_relabel_test"]:
                print(f"    exact cluster relabel    p = "
                      f"{h['cluster_relabel_test']['p']:.4f}   over "
                      f"{h['cluster_relabel_test']['n_outcomes']} outcomes")
            print("  Three tests, three answers straddling 0.05, on a subset whose")
            print("  effective n is three names. The reportable finding is that the")
            print("  token-matched subset is too small to adjudicate the question")
            print("  it was introduced to settle, not that the effect grows by 61%.")

    # ---- the verdict, recomputed on the correct resampling unit -----------
    # The block above computes `_verdict` from the ROW-level bootstrap, because
    # it runs before the clustered arm exists. That count -- 2 of 4 by slope --
    # reached the abstract, §1.2 and §4.4 of the paper and was wrong: the 48
    # rows are 12 first-name pairs x 4 surnames, and §6.1 of the paper forbids
    # resampling them as 48. The verdict is therefore recomputed here, last,
    # from `slope_same_unit`, and the superseded count is kept beside it under
    # its own name so nothing downstream can pick up the wrong one by accident.
    _cl_models = [m for m, v in out.items()
                  if not m.startswith("_")
                  and "token_matched_first_name_clustered" in v]
    _nsig_cl = sum(
        1 for m in _cl_models
        if out[m]["token_matched_first_name_clustered"]
        ["slope_same_unit"]["p"] < 0.05)
    _dsig_cl = sum(
        1 for m in _cl_models
        if out[m]["token_matched_first_name_clustered"]
        ["matched_minus_all"]["p"] < 0.05)
    out["_verdict"] = dict(
        verdict=("LENGTH CONFOUND PRESENT" if (_nsig_cl or _dsig_cl)
                 else "NO LENGTH CONFOUND DETECTED AT THE 0.05 THRESHOLD"),
        resampling_unit="first_name_pair",
        n_models=len(_cl_models),
        n_slope_significant=_nsig_cl,
        n_subset_significant=_dsig_cl,
        superseded_row_level=dict(
            n_slope_significant=nsig, n_subset_significant=dsig,
            why_superseded="rows are 12 first-name pairs x 4 surnames; "
                           "resampling them as 48 independent draws is the "
                           "error the paper measures in §6.1"))
    print("\n" + "=" * 100)
    print("VERDICT, ON THE FIRST-NAME-PAIR RESAMPLING UNIT")
    print("=" * 100)
    print(f"  slope distinguishable from zero on {_nsig_cl} of "
          f"{len(_cl_models)} models (row-level bootstrap said {nsig})")
    print(f"  subset difference on {_dsig_cl} of {len(_cl_models)} "
          f"(row-level bootstrap said {dsig})")

    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
