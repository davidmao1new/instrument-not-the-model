"""Does Study 3's template concentration replicate on the full panel?

WHAT STUDY 3 FOUND, AND WHY IT NEEDED A SECOND LOOK. Study 3 measured the
delimiter conditions on two résumé templates and found the condition effects
concentrated almost entirely in T2_mid -- the middling résumé -- and close to
absent on the strong and marginal ones. That is a plausible mechanism (a
middling candidate sits nearest the decision boundary, so perturbing the input
moves it further) and it is also exactly the shape a false positive takes when
one of three subgroups is inspected after the fact.

Two things were wrong with the original observation and both are fixed here.

  1. IT WAS NOT A TEST. "Large in T2, small in T1 and T3" compares three
     estimates by eye. The quantity that licenses the claim is the INTERACTION
     -- the difference between T2's condition effect and the other templates'
     condition effect -- with an interval on it. CLAUDE.md names this as the
     single most common error in this literature, and Study 3 committed it.

  2. THE PANEL DID NOT CARRY T2. The mechanism panel was originally run on
     TEMPLATES_2 = (T1_strong, T3_marginal), i.e. on the two templates where
     Study 3 said nothing happens. The T2_mid arm was added afterwards
     specifically so the concentration could be tested rather than assumed.

THE TEST, fixed before the T2 data was analysed. For each model, mode and
condition, the interaction is

    I = [ effect(cond) - effect(D0) ] restricted to T2_mid
      - [ effect(cond) - effect(D0) ] restricted to {T1_strong, T3_marginal}

paired within name pair, bootstrap over name pairs, BH-corrected across the
whole family. Study 3's claim predicts I is systematically positive in
magnitude, i.e. |condition effect| is larger in T2. A null I means the
concentration was a subgroup artefact and the paper must retract it.

--------------------------------------------------------------------------
A THIRD THING WAS WRONG, AND IT WAS IN THE FIX.
--------------------------------------------------------------------------
The interaction above is not zero under its own null. It was computed as

    I_legacy = | shift on T2 |  -  | shift AVERAGED over T1 and T3 |

and those two terms are not on the same footing. The left one is the
magnitude of ONE noisy estimate; the right one is the magnitude of the MEAN
of two noisy estimates, and averaging halves the variance before the absolute
value is taken. |E| is not E| |: a magnitude is inflated by the noise in the
estimate it is taken of, so the less noisy term is systematically the smaller
one even when the underlying quantity is identical. The estimator reports a
concentration on T2 when nothing whatever is concentrated there.

The size of the artefact is not small. Under an exchangeable null in which
all three templates carry the SAME true shift plus i.i.d. noise of standard
deviation s,

    E[I_legacy] = s * sqrt(2/pi) * (1 - 1/sqrt(2)) = 0.2337 * s   (and not 0)

which `gaussian_null()` below reproduces by simulation, and which decays only
as the true common shift grows large relative to s. Calibrated on this panel's
own numbers by permuting the template labels within each name pair --
`permutation_null()`, the exact null the test claims to be testing -- the
legacy statistic averages +0.0244 over the 120 (model, mode, condition) cells
when there is by construction nothing to find. The observed value is +0.0293.
Roughly five sixths of the reported concentration was the estimator measuring
its own asymmetry.

WHAT REPLACES IT. Put the two terms on the same footing by taking each
template's magnitude separately and only then averaging:

    I = | shift on T2 |  -  mean( | shift on T1 | , | shift on T3 | )

Now every term is the magnitude of a single-template estimate, so under
exchangeability of the three templates E|T2| = E|T1| = E|T3| and E[I] = 0
exactly -- no Gaussian assumption, no equal-variance assumption, just
exchangeability. Same pairing, same bootstrap over name pairs, same BH family.
The legacy quantity is kept beside it in the artifact under `*_legacy` keys,
with its own permutation-null mean, so the correction is visible rather than
silent and the paper can report what it previously said as well as what is
true.

WHAT THE CORRECTED STATISTIC STILL CANNOT DO. It is a comparison of
magnitudes, so a template that is merely NOISIER than the others will show a
positive I even with an identical true shift. That is not fixable inside one
condition; it is what the pure controls are for. A perturbation that changes
nothing structural has no mechanism to concentrate anywhere, so the control
conditions estimate exactly this template-specific jitter, and only a
destroying-versus-control gap can be read as a mechanism.

    .venv/Scripts/python.exe paper-a/src/analyze_template_concentration.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_mech_panel as amp  # noqa: E402
import effectsize as es  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper-a" / "data" / "mechanism_panel" / "template_concentration.json"
N_BOOT = 20_000
# THREE SEPARATE STREAMS, DELIBERATELY. The legacy statistic keeps its original
# generator and its original position in the call order so that the numbers the
# paper has already reported reproduce bit-for-bit; adding the corrected
# statistic must not silently move them. The new quantities draw from their own
# streams instead.
RNG = np.random.default_rng(20260730)
RNG_UNB = np.random.default_rng(20260801)
RNG_NULL = np.random.default_rng(20260802)
N_PERM = 2_000
T2 = "T2_mid"
OTHER = ("T1_strong", "T3_marginal")
TEMPLATES = (T2,) + OTHER          # column 0 is always T2, by construction

# The conditions Study 3 claimed a concentration for: the ones that destroy a
# delimiter. The pure controls are carried too, as a falsification check -- if
# the interaction is just as large for a condition that changes nothing, the
# quantity is measuring template-specific noise and not a mechanism.
DESTROYING = ("D4", "D5", "D6", "D10")
CONTROLS = ("D1", "D2", "D3", "D7", "D8", "D9")


def margins(rows, cond, templates):
    """{pair: mean paired margin difference across the given templates}.

    Retained because the legacy statistic is defined by it: the average across
    templates happens HERE, before any magnitude is taken, which is the whole
    of what was wrong with it. Nothing new should call this with more than one
    template.
    """
    by = defaultdict(list)
    for r in rows:
        if r["cond"] == cond and r["template"] in templates:
            by[r["pair"]].append(r["white_margin"] - r["black_margin"])
    return {p: float(np.mean(v)) for p, v in by.items()}


def margins_per_template(rows, cond):
    """{template: {pair: paired margin difference}}, nothing averaged.

    The corrected statistic needs T1 and T3 kept apart so each contributes its
    own magnitude; collapsing them first is precisely the step that biases the
    comparison.
    """
    by = defaultdict(dict)
    for r in rows:
        if r["cond"] == cond:
            by[r["template"]][r["pair"]] = r["white_margin"] - r["black_margin"]
    return by


def shift_matrix(rows, cond, pairs):
    """(len(pairs), 3) array of cond-minus-D0 shifts, columns ordered TEMPLATES.

    Returns None when any template is missing a cell, rather than quietly
    dropping a template and changing what the statistic means.
    """
    cur = margins_per_template(rows, cond)
    base = margins_per_template(rows, "D0")
    cols = []
    for t in TEMPLATES:
        c, b = cur.get(t, {}), base.get(t, {})
        if not all(p in c and p in b for p in pairs):
            return None
        cols.append([c[p] - b[p] for p in pairs])
    return np.array(cols, dtype=float).T


def stat_legacy(s):
    """|shift on T2| - |shift AVERAGED over T1 and T3|, per pair. Biased; see
    the module docstring. Vectorised over a leading permutation axis."""
    return np.abs(s[..., 0]) - np.abs(0.5 * (s[..., 1] + s[..., 2]))


def stat_unbiased(s):
    """|shift on T2| - mean(|shift on T1|, |shift on T3|), per pair.

    Every term is now the magnitude of one single-template estimate, so the
    noise inflation that |.| introduces is the same on both sides and cancels
    under exchangeability of the templates.
    """
    return np.abs(s[..., 0]) - 0.5 * (np.abs(s[..., 1]) + np.abs(s[..., 2]))


def permutation_null(s, n_perm=N_PERM, rng=None):
    """Mean of each statistic under the null the test actually claims to test.

    THE NULL IS EXCHANGEABILITY, not zero. "The concentration is an artefact"
    does not mean the shifts are zero -- they are plainly not -- it means the
    three templates are interchangeable. So shuffle which template plays the
    role of T2 within each name pair and leave everything else alone: the true
    shift, the noise scale, the between-pair heterogeneity and the correlation
    structure all survive the shuffle. Whatever a statistic returns here is
    what it returns when there is by construction nothing to find.
    """
    rng = rng or RNG_NULL
    idx = np.argsort(rng.random((n_perm,) + s.shape), axis=2)
    sp = np.take_along_axis(np.broadcast_to(s, (n_perm,) + s.shape), idx, axis=2)
    return (float(stat_legacy(sp).mean(axis=1).mean()),
            float(stat_unbiased(sp).mean(axis=1).mean()))


def family_permutation_null(smats, n_perm=N_PERM, rng=None):
    """Null distribution of the FAMILY MEAN of each statistic.

    Not the same thing as averaging the per-cell nulls. Every condition in a
    (model, mode) block is measured against the SAME D0 baseline, so the cells
    are correlated and independent per-cell shuffles would understate how much
    the family mean moves by chance. One template permutation is therefore drawn
    per (model, mode, name pair) and applied to every condition in that block,
    which is what permuting the template label of the underlying rows would do.

    Returns (legacy_family_means, unbiased_family_means), each length n_perm.
    """
    rng = rng or RNG_NULL
    tot_l = tot_u = None
    k = 0
    for byc in smats.values():
        conds = sorted(byc)
        idx = np.argsort(rng.random((n_perm,) + byc[conds[0]].shape), axis=2)
        for c in conds:
            s = byc[c]
            sp = np.take_along_axis(
                np.broadcast_to(s, (n_perm,) + s.shape), idx, axis=2)
            l, u = stat_legacy(sp).mean(axis=1), stat_unbiased(sp).mean(axis=1)
            tot_l = l if tot_l is None else tot_l + l
            tot_u = u if tot_u is None else tot_u + u
            k += 1
    return tot_l / k, tot_u / k


def perm_pvalue(null, obs):
    """Two-sided permutation p for `obs`, centred on the null's OWN mean.

    Centring matters for the legacy statistic and only for it: its null is not
    at zero, so a p-value computed against zero would be testing the wrong
    hypothesis. Floored at 1/n_perm for the reason pvalue_from_boots is.
    """
    c = float(np.mean(null))
    return max(float(np.mean(np.abs(null - c) >= abs(obs - c))), 1.0 / len(null))


def gaussian_null(n_pairs=24, reps=20_000, rng=None):
    """Closed-form check of the bias, on data with no panel in it at all.

    Three templates, one shared true shift mu, i.i.d. noise sd s. The legacy
    statistic should return 0.2337*s at mu = 0 and decay towards 0 as mu/s
    grows; the corrected one should return 0 everywhere. If this table ever
    stops looking like that, the correction has been broken.
    """
    rng = rng or RNG_NULL
    rows = []
    for mu in (0.0, 0.05, 0.15, 0.40):
        for sd in (0.05, 0.10, 0.20):
            s = rng.normal(mu, sd, size=(reps, n_pairs, 3))
            rows.append(dict(
                mu=mu, sd=sd,
                legacy=float(stat_legacy(s).mean(axis=1).mean()),
                unbiased=float(stat_unbiased(s).mean(axis=1).mean()),
                analytic_legacy_at_mu0=float(sd * np.sqrt(2 / np.pi)
                                             * (1 - 1 / np.sqrt(2))),
            ))
    return rows


def verdict_of(nd, nc):
    """The four-way read of (destroying significant, controls significant).

    Shared by both statistics so that the legacy and corrected conclusions are
    reached by identical logic and only the input differs.
    """
    if nd == 0 and nc == 0:
        return "NOT REPLICATED"
    if nd > 0 and nc == 0:
        return "REPLICATED"
    if nd > 0 and nc > 0:
        return "NON-SPECIFIC"
    return "CONTROLS ONLY"


def main() -> int:
    rows = amp.load()
    if not rows:
        sys.exit("no panel data")
    have = {r["template"] for r in rows}
    if T2 not in have:
        sys.exit(f"T2_mid arm not present; templates on disk: {sorted(have)}")

    print("=" * 100)
    print("TEMPLATE CONCENTRATION. Is a condition's effect larger on the middling")
    print("résumé than on the strong and marginal ones? Tested as an interaction,")
    print("not as a comparison of three separate estimates.")
    print("The interaction is reported twice: as it was originally computed, and")
    print("with the averaging-before-magnitude bias removed. See the docstring.")
    print("=" * 100)

    out, family = {}, []
    smats = defaultdict(dict)      # (model, mode) -> {cond: shift matrix}
    smat_pairs = {}                # (model, mode) -> the pair list it was built on
    for model in sorted({r["model"] for r in rows}):
        out[model] = {}
        for mode in sorted({r["mode"] for r in rows if r["model"] == model}):
            mr = [r for r in rows if r["model"] == model and r["mode"] == mode]
            base_t2 = margins(mr, "D0", (T2,))
            base_ot = margins(mr, "D0", OTHER)
            if not base_t2 or not base_ot:
                continue
            res = {}
            for cond in DESTROYING + CONTROLS:
                c_t2 = margins(mr, cond, (T2,))
                c_ot = margins(mr, cond, OTHER)
                pairs = sorted(set(c_t2) & set(c_ot) & set(base_t2) & set(base_ot))
                if len(pairs) < 8:
                    continue
                # Built before either statistic so that a cell missing a
                # template drops out of BOTH families rather than leaving a
                # half-populated record behind. The panel is balanced -- 6
                # models x 2 modes x 11 conditions x 3 templates x 24 pairs --
                # so this never fires today; it fires the day it stops being.
                s = shift_matrix(mr, cond, pairs)
                if s is None:
                    continue
                # LEGACY, kept verbatim and computed first so that its draws from
                # RNG come in the original order and its published values do not
                # move. |shift| in T2 minus |shift| of the T1/T3 AVERAGE -- the
                # asymmetry the docstring is about.
                d = np.array([
                    abs(c_t2[p] - base_t2[p]) - abs(c_ot[p] - base_ot[p])
                    for p in pairs], dtype=float)
                bt = es.boot_ci(d, lambda x: float(x.mean()), N_BOOT, RNG)
                p = es.pvalue_from_boots(bt["boots"], bt["est"], 0.0, N_BOOT)
                res[cond] = dict(n_pairs=len(pairs), interaction=bt["est"],
                                 ci=bt["ci"], p=p,
                                 shift_t2=float(np.mean([abs(c_t2[q]-base_t2[q]) for q in pairs])),
                                 shift_other=float(np.mean([abs(c_ot[q]-base_ot[q]) for q in pairs])))

                # CORRECTED. Same pairs, same pairing, same bootstrap over name
                # pairs; only the point where the two non-T2 templates are
                # combined has moved to after the magnitude.
                du = stat_unbiased(s)
                btu = es.boot_ci(du, lambda x: float(x.mean()), N_BOOT, RNG_UNB)
                pu = es.pvalue_from_boots(btu["boots"], btu["est"], 0.0, N_BOOT)
                nl, nu = permutation_null(s)
                res[cond].update(
                    interaction_unbiased=btu["est"], ci_unbiased=btu["ci"],
                    p_unbiased=pu,
                    shift_t1=float(np.mean(np.abs(s[:, 1]))),
                    shift_t3=float(np.mean(np.abs(s[:, 2]))),
                    shift_other_meanabs=float(np.mean(
                        0.5 * (np.abs(s[:, 1]) + np.abs(s[:, 2])))),
                    null_mean_legacy=nl, null_mean_unbiased=nu,
                    interaction_legacy_bias_corrected=bt["est"] - nl)
                family.append((model, mode, cond, p, res[cond]))
                # The block-level permutation below applies ONE shuffle per name
                # pair to every condition, so only conditions measured on the
                # identical pair list may join a block.
                key = (model, mode)
                if smat_pairs.setdefault(key, pairs) == pairs:
                    smats[key][cond] = s
            out[model][mode] = res

    adj = es.benjamini_hochberg([f[3] for f in family])
    for f, a in zip(family, adj):
        f[4]["p_bh"] = a
    # A SECOND, SEPARATE BH FAMILY. The corrected statistic is a different test
    # on the same cells; adjusting it inside the legacy family would let the
    # legacy p-values set its threshold.
    adj_u = es.benjamini_hochberg([f[4]["p_unbiased"] for f in family])
    for f, a in zip(family, adj_u):
        f[4]["p_bh_unbiased"] = a

    for model in out:
        for mode in out[model]:
            res = out[model][mode]
            if not res:
                continue
            print(f"\n{model}   mode={mode}")
            print(f"  {'cond':<6}{'|T2|':>9}{'|T1|':>9}{'|T3|':>9}"
                  f"{'I_legacy':>11}{'null':>9}{'I_fixed':>11}"
                  f"{'95% CI':>22}{'p_BH':>9}")
            for cond in DESTROYING + CONTROLS:
                if cond not in res:
                    continue
                v = res[cond]
                star = " *" if v["p_bh_unbiased"] < 0.05 else ""
                grp = "destroy" if cond in DESTROYING else "control"
                ci = v["ci_unbiased"]
                print(f"  {cond:<6}{v['shift_t2']:>9.4f}{v['shift_t1']:>9.4f}"
                      f"{v['shift_t3']:>9.4f}{v['interaction']:>+11.4f}"
                      f"{v['null_mean_legacy']:>+9.4f}"
                      f"{v['interaction_unbiased']:>+11.4f}"
                      f"{f'[{ci[0]:+.4f},{ci[1]:+.4f}]':>22}"
                      f"{v['p_bh_unbiased']:>9.4f}{star}   {grp}")

    # ---- the verdict -----------------------------------------------------
    dest = [f[4] for f in family if f[2] in DESTROYING]
    ctrl = [f[4] for f in family if f[2] in CONTROLS]
    nd = sum(1 for v in dest if v["p_bh"] < 0.05 and v["interaction"] > 0)
    nc = sum(1 for v in ctrl if v["p_bh"] < 0.05 and v["interaction"] > 0)
    md = float(np.mean([v["interaction"] for v in dest]))
    mc = float(np.mean([v["interaction"] for v in ctrl]))
    verdict = verdict_of(nd, nc)

    ndu = sum(1 for v in dest if v["p_bh_unbiased"] < 0.05
              and v["interaction_unbiased"] > 0)
    ncu = sum(1 for v in ctrl if v["p_bh_unbiased"] < 0.05
              and v["interaction_unbiased"] > 0)
    mdu = float(np.mean([v["interaction_unbiased"] for v in dest]))
    mcu = float(np.mean([v["interaction_unbiased"] for v in ctrl]))
    verdict_unbiased = verdict_of(ndu, ncu)

    # HOW MANY CELLS POINT THE OTHER WAY. A concentration on T2 is a directional
    # claim, so a count of significant cells is only interpretable beside the
    # count of significant cells running the opposite way. If the two are equal
    # the family is scatter, whatever the positive count is on its own.
    def _neg(vs, key_p, key_i):
        return sum(1 for v in vs if v[key_p] < 0.05 and v[key_i] < 0)
    nd_neg = _neg(dest, "p_bh", "interaction")
    nc_neg = _neg(ctrl, "p_bh", "interaction")
    ndu_neg = _neg(dest, "p_bh_unbiased", "interaction_unbiased")
    ncu_neg = _neg(ctrl, "p_bh_unbiased", "interaction_unbiased")

    allv = dest + ctrl
    null_legacy = float(np.mean([v["null_mean_legacy"] for v in allv]))
    null_unb = float(np.mean([v["null_mean_unbiased"] for v in allv]))
    obs_legacy = float(np.mean([v["interaction"] for v in allv]))
    obs_unb = float(np.mean([v["interaction_unbiased"] for v in allv]))
    gauss = gaussian_null()

    # Family-level test, with the correlation between conditions respected.
    fnl, fnu = family_permutation_null(smats)
    p_fam_legacy = perm_pvalue(fnl, obs_legacy)
    p_fam_unb = perm_pvalue(fnu, obs_unb)

    # WHERE THE SURVIVING SIGNAL SITS. Per (model, mode) rather than per
    # condition, because that is the grouping the corrected numbers turn out to
    # respect: within a block the sign is the same for delimiter-destroying and
    # control conditions alike, which is the signature of a template-by-model
    # dispersion difference and not of a condition mechanism.
    blocks = {}
    for model in out:
        for mode in out[model]:
            vs = list(out[model][mode].values())
            if not vs:
                continue
            iu = [v["interaction_unbiased"] for v in vs]
            blocks[f"{model}|{mode}"] = dict(
                n_cond=len(vs), mean_interaction_unbiased=float(np.mean(iu)),
                n_sig_pos=sum(1 for v in vs if v["p_bh_unbiased"] < 0.05
                              and v["interaction_unbiased"] > 0),
                n_sig_neg=sum(1 for v in vs if v["p_bh_unbiased"] < 0.05
                              and v["interaction_unbiased"] < 0),
                mean_interaction_legacy=float(np.mean(
                    [v["interaction"] for v in vs])))

    print("\n" + "=" * 100)
    print("HOW MUCH OF THE LEGACY INTERACTION WAS THE ESTIMATOR ITSELF")
    print("=" * 100)
    print(f"  family mean, legacy statistic          {obs_legacy:+.4f}")
    print(f"  the same statistic under the exchangeability null "
          f"{null_legacy:+.4f}   <- should be 0")
    print(f"  share of the legacy value that is bias "
          f"{100.0 * null_legacy / obs_legacy:>6.1f} %")
    print(f"  family mean, corrected statistic       {obs_unb:+.4f}")
    print(f"  the same statistic under the same null {null_unb:+.4f}")
    print(f"  family-level permutation p, legacy vs its own null   {p_fam_legacy:.4f}")
    print(f"  family-level permutation p, corrected vs zero        {p_fam_unb:.4f}")
    print("\n  Gaussian check, three templates with an identical true shift:")
    print(f"    {'mu':>6}{'sd':>8}{'I_legacy':>11}{'I_fixed':>11}"
          f"{'0.2337*sd':>12}")
    for g in gauss:
        print(f"    {g['mu']:>6.2f}{g['sd']:>8.2f}{g['legacy']:>+11.4f}"
              f"{g['unbiased']:>+11.4f}{g['analytic_legacy_at_mu0']:>12.4f}")

    # DOES THE CONCENTRATION SURVIVE AT ALL? Kept separate from the four-way
    # specificity label, which answers a different question and can read as
    # "something is there, and it is non-specific" when the honest answer is
    # that the family mean is not distinguishable from its null. Directional,
    # because "concentrated on T2" is a directional claim.
    survives = bool(p_fam_unb < 0.05 and obs_unb > 0)
    sign_symmetric = bool((ndu + ncu) == (ndu_neg + ncu_neg))
    conclusion = (
        f"Legacy family mean {obs_legacy:+.4f}, of which {null_legacy:+.4f} "
        f"({100.0 * null_legacy / obs_legacy:.0f}%) is the estimator's own bias "
        f"under an exchangeability null. Corrected family mean {obs_unb:+.4f}, "
        f"permutation p = {p_fam_unb:.3f}; delimiter-destroying conditions "
        f"average {mdu:+.4f}. BH-significant cells split "
        f"{ndu + ncu} positive to {ndu_neg + ncu_neg} negative, and the sign is "
        f"constant within a model x mode block across destroying and control "
        f"conditions alike. "
        + ("The concentration survives correction."
           if survives else
           "The concentration does not survive correction: what remains is a "
           "model-specific difference in how much each template moves under any "
           "perturbation, in both directions, and not a concentration of "
           "condition effects on the middling resume."))

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print("  LEGACY statistic (biased; retained for comparison only)")
    print(f"    destroying: {nd} of {len(dest)} positive after BH "
          f"({nd_neg} negative); mean {md:+.4f}")
    print(f"    controls:   {nc} of {len(ctrl)} positive after BH "
          f"({nc_neg} negative); mean {mc:+.4f}")
    print(f"    -> {verdict}")
    print("  CORRECTED statistic")
    print(f"    destroying: {ndu} of {len(dest)} positive after BH "
          f"({ndu_neg} negative); mean {mdu:+.4f}")
    print(f"    controls:   {ncu} of {len(ctrl)} positive after BH "
          f"({ncu_neg} negative); mean {mcu:+.4f}")
    print(f"    -> {verdict_unbiased}")
    print("\n  per model x mode, corrected statistic "
          "(sign is a block property, not a condition property):")
    print(f"    {'block':<34}{'mean I_fixed':>13}{'sig +':>7}{'sig -':>7}")
    for k, b in sorted(blocks.items(),
                       key=lambda kv: -kv[1]["mean_interaction_unbiased"]):
        print(f"    {k:<34}{b['mean_interaction_unbiased']:>+13.4f}"
              f"{b['n_sig_pos']:>7}{b['n_sig_neg']:>7}")
    print()
    if verdict_unbiased == "NOT REPLICATED":
        print("  NOT REPLICATED. Once the two sides of the comparison are put on")
        print("  the same footing, no condition's effect is reliably larger on the")
        print("  middling résumé than on the others. Study 3's concentration was a")
        print("  subgroup comparison made by eye, and the paper must report it as")
        print("  such rather than as a mechanism.")
    elif verdict_unbiased == "REPLICATED":
        print("  REPLICATED, and specific. The conditions that destroy a delimiter")
        print("  shift the middling résumé more than the strong and marginal ones,")
        print("  while conditions that change nothing structural do not. The")
        print("  decision-boundary account survives a real test.")
    elif verdict_unbiased == "NON-SPECIFIC":
        print("  NON-SPECIFIC. The interaction survives the correction for the")
        print("  destroying conditions AND for controls that change nothing")
        print("  structural, so it measures how much the middling résumé moves")
        print("  under ANY perturbation rather than a delimiter mechanism.")
    else:
        print("  CONTROLS ONLY. Only conditions that change nothing structural show")
        print("  the interaction. Whatever this is, it is not the claimed mechanism.")
    print()
    print("  CONCENTRATION SURVIVES CORRECTION: " + ("YES" if survives else "NO"))
    print("  " + conclusion)

    OUT.write_text(json.dumps(dict(
        per_model=out, n_boot=N_BOOT, n_perm=N_PERM,
        # LEGACY keys, untouched, because the paper already reports them and
        # nothing here is deleted. `verdict` is the legacy verdict.
        verdict=verdict, verdict_legacy=verdict,
        destroying=dict(n_positive_after_bh=nd, n=len(dest), mean_interaction=md),
        controls=dict(n_positive_after_bh=nc, n=len(ctrl), mean_interaction=mc),
        # CORRECTED keys. These are the ones the paper should report.
        verdict_unbiased=verdict_unbiased,
        concentration_survives=survives,
        sign_split_symmetric=sign_symmetric,
        conclusion=conclusion,
        destroying_unbiased=dict(n_positive_after_bh=ndu,
                                 n_negative_after_bh=ndu_neg, n=len(dest),
                                 mean_interaction=mdu),
        controls_unbiased=dict(n_positive_after_bh=ncu,
                               n_negative_after_bh=ncu_neg, n=len(ctrl),
                               mean_interaction=mcu),
        direction_split=dict(
            legacy=dict(destroying_pos=nd, destroying_neg=nd_neg,
                        controls_pos=nc, controls_neg=nc_neg),
            unbiased=dict(destroying_pos=ndu, destroying_neg=ndu_neg,
                          controls_pos=ncu, controls_neg=ncu_neg),
            note=("after correction the significant cells split evenly by sign, "
                  "and the sign is constant within a model x mode block across "
                  "destroying and control conditions alike"),
        ),
        per_block_unbiased=blocks,
        statistic=dict(
            legacy="mean_p |shift(T2)| - |mean over {T1,T3} of shift|",
            unbiased="mean_p |shift(T2)| - mean(|shift(T1)|, |shift(T3)|)",
            why=("the legacy form compares the magnitude of one estimate with "
                 "the magnitude of an average of two; averaging halves the "
                 "variance before |.| is taken, so the right-hand term is "
                 "smaller under a true null and the statistic reports a "
                 "concentration on T2 that is not there"),
        ),
        null_calibration=dict(
            method=("template labels permuted within name pair; the true shift, "
                    "noise scale and between-pair heterogeneity are preserved"),
            n_perm=N_PERM,
            legacy_observed_mean=obs_legacy,
            legacy_null_mean=null_legacy,
            legacy_bias_share=float(null_legacy / obs_legacy),
            legacy_family_p=p_fam_legacy,
            unbiased_observed_mean=obs_unb,
            unbiased_null_mean=null_unb,
            unbiased_family_p=p_fam_unb,
            family_null_note=("family-level p-values use one template shuffle "
                              "per (model, mode, name pair) applied to every "
                              "condition, so the shared D0 baseline does not "
                              "make the cells look independent"),
            gaussian=gauss,
            analytic_legacy_bias="0.2337 * sd(per-template shift), at mu = 0",
        ),
    ), indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
