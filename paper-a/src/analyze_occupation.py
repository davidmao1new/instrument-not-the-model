"""Study 7. Is the wording instability a property of the model or of the job?

THE CONFIRMATORY QUESTION, fixed before the data existed: is the ratio of
between-wording dispersion to effect size stable across occupations? If it is,
the instability is a property of the model and the single-occupation limitation
dissolves. If it is not, this paper's headline numbers are specific to a
Business Analyst posting and must be reported that way.

WHAT IS COMPARED. Three occupations, structurally matched by construction (see
occupations.py and tests/test_occupations.py): same requirement slots, same
three-section résumé skeleton, same employer and date style, postings within
four words and résumés within six words of one another. The Business Analyst arm
is Study 2's data, not a re-run, and a test asserts its strings have not drifted.

TWO SECONDARY QUESTIONS, both labelled exploratory because neither is why this
was run.

  GENDER TYPING. The occupations span the labour market's gender typing --
  Software Engineer male-typed, Registered Nurse female-typed, Business Analyst
  balanced -- and the name grid is gender-balanced, so an interaction between
  occupational typing and the demographic effect is estimable. Three occupations
  cannot support a claim about occupations in general, and the result is
  reported as a description of these three.

  DOES THE SIGN TRAVEL? The Llama family reverses sign across the 2024 boundary
  on the Business Analyst posting. Whether that reversal is a property of the
  models or of that posting is a question this study can answer and Study 2
  could not.

--------------------------------------------------------------------------
TWO CORRECTIONS, ADDED AFTER AN AUDIT OF TABLE 10.
--------------------------------------------------------------------------
1. THE RATIO HAD NO INTERVAL, AND ITS DENOMINATOR IS OFTEN ZERO.

   ratio_sd_to_effect is SD_wording(Ps) / |Ps - 0.5|. The denominator is the
   effect itself, and on this panel the effect is frequently not distinguishable
   from nothing: in EIGHT of the twelve cells the 95% interval on Ps contains
   0.5. Dividing by a quantity whose interval covers zero does not produce a
   large ratio, it produces an undefined one, and printing it to the nearest
   percent hid that. The confirmatory claim -- "the ratio moves by up to 184
   percentage points within a single model" -- is a max-minus-min over three
   such cells and the maximum is taken from the worst of them.

   The fix is not to drop the ratio. It is to report what it is worth. Every
   ratio, and the spread, now carries a percentile interval from a bootstrap
   that resamples NAME PAIRS, which is what was randomised; the point estimates
   are unchanged and kept, and each cell carries a flag saying whether its
   denominator is separable from zero. Where it is not, the interval runs to the
   top of the reportable range and the cell should be read as "no ratio is
   estimable here", not as "the ratio is enormous".

   The three occupation arms use the SAME twelve name pairs, so one shared
   resample of pairs drives all three arms in every replicate. That is what
   makes the spread interval a statement about the same names seeing three jobs
   rather than about three independent samples.

2. "MATCHED ON STRUCTURE, SO NOT A CONFOUND OF RESUME QUALITY" WAS AN ASSERTION.

   occupations.py matches the three arms on surface structure -- slots, skeleton,
   section headers, word counts -- and tests/test_occupations.py enforces it. It
   cannot match them on DIFFICULTY, because difficulty is a property of the model
   reading the posting, not of the strings. It is measurable, so it is now
   measured: the acceptance rate at each of the three résumé strength levels, per
   model and per arm. The mismatch is not subtle. On Mistral v0.3 the middle
   strength level is accepted 0.7% of the time in the business-analyst arm and
   94.4% in the software-engineer arm -- the same nominal résumé strength putting
   the two arms at opposite ends of the decision scale. A difference between arms
   can therefore be a difference in operating point, and the sentence claiming
   otherwise cannot stand on structural matching alone.

    .venv/Scripts/python.exe paper-a/src/analyze_occupation.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import effectsize as es  # noqa: E402
import stimuli as st  # noqa: E402
from occupations import OCCUPATIONS  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
OCCDIR = ROOT / "paper-a" / "data" / "occupation"
BADIR = ROOT / "paper-a" / "data" / "delta_stability"
OUT = OCCDIR / "occupation_analysis.json"
RNG = np.random.default_rng(20260729)

# A SECOND, INDEPENDENT STREAM for the interval work added by the audit fix.
# Drawing the new replicates from RNG would shift every subsequent draw and
# silently move the effect sizes and Ps intervals this file has already
# published. A separate generator keeps the pre-existing numbers reproducible
# byte for byte, so the diff on the artifact is additions only.
RNG_INTERVAL = np.random.default_rng(20260801)

BOOT = 6000                  # replicates; matches es.describe's budget here
DEN_FLOOR = 1e-9             # the floor the ratio point estimate already uses
UNBOUNDED_AT = 10.0          # a ratio of 1000%: past this the CI reports 1/0
STRENGTHS = ["T1_strong", "T2_mid", "T3_marginal"]

SHORT = {
    "mistral-7b-instruct-v0.1": "Mistral-7B-Instruct v0.1",
    "llama-2-7b-chat": "Llama-2-7B-chat",
    "mistral-7b-instruct-v0.3": "Mistral-7B-Instruct v0.3",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
}
ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
OCC_ORDER = ["BA", "SWE", "RN"]


def load_rows():
    """All three occupations in one table. BA comes from Study 2."""
    rows = []
    for f in sorted(BADIR.glob("delta_*.jsonl")):
        for r in st.read_jsonl(f):
            if (r.get("white_margin") is not None
                    and r.get("black_margin") is not None):
                rows.append({**r, "occupation": "BA"})
    if OCCDIR.exists():
        for f in sorted(OCCDIR.glob("occ_*.jsonl")):
            for r in st.read_jsonl(f):
                if (r.get("white_margin") is not None
                        and r.get("black_margin") is not None):
                    rows.append(r)
    best = {}
    for r in rows:
        best[(r["model"], r["occupation"], r["variant"], r["template"],
              r["pair"])] = r
    return list(best.values())


def per_cell(rows):
    """Effect and wording dispersion for one (model, occupation)."""
    d = es.describe(rows, 6000, RNG)
    byv = defaultdict(list)
    for r in rows:
        byv[r["variant"]].append(r["white_margin"] - r["black_margin"])
    per = np.array([np.mean(v) for v in byv.values()])
    ps_by_v = np.array([float(np.mean(np.array(v) > 0)) for v in byv.values()])
    return dict(
        n=d["n"], n_variants=len(byv),
        logodds=d["logodds"]["est"], ci=d["logodds"]["ci"],
        p=d["logodds"]["p"],
        ps=d["superiority"]["est"], ps_ci=d["superiority"]["ci"],
        sigma_variant_raw=float(per.std(ddof=1)),
        ps_sd_across_wordings=float(ps_by_v.std(ddof=1)),
        ratio_sd_to_effect=float(ps_by_v.std(ddof=1)
                                 / max(abs(d["superiority"]["est"] - 0.5), 1e-9)),
        wording_range=[float(per.min()), float(per.max())])


# ==========================================================================
# The audit fix. Everything below this line was added after Table 10 was
# challenged; nothing above it changed, so the published point estimates are
# reproduced unaltered and the new quantities sit beside them.
# ==========================================================================
def cube(rows):
    """One (model, occupation) cell as a (pair x wording x strength) array.

    WHY A CUBE AND NOT A LIST OF ROWS. Every interval added here resamples name
    pairs, and a name pair is not a row: it is a whole slab of this design,
    twelve wordings by three strength levels. Laying the cell out rectangularly
    makes the cluster bootstrap one fancy-index on axis 0, and makes an
    incomplete cell impossible to bootstrap by accident -- a missing observation
    survives as a nan and stops the run, instead of quietly producing a narrower
    interval than the data support.

    Returns (superiority indicator, acceptance indicator, pairs, wordings,
    strengths). The superiority cube carries 1 / 0.5 / 0 exactly as
    effectsize.superiority scores a pair, so its mean IS Ps. The acceptance cube
    carries the fraction of the pair's two résumés the model said yes to, so its
    mean is the acceptance rate over both sides.
    """
    pairs = sorted({r["pair"] for r in rows})
    words = sorted({r["variant"] for r in rows})
    seen = {r["template"] for r in rows}
    strengths = [t for t in STRENGTHS if t in seen] + sorted(seen - set(STRENGTHS))
    ip = {p: i for i, p in enumerate(pairs)}
    iw = {w: i for i, w in enumerate(words)}
    it = {t: i for i, t in enumerate(strengths)}

    shape = (len(pairs), len(words), len(strengths))
    sup = np.full(shape, np.nan)
    acc = np.full(shape, np.nan)
    for r in rows:
        i, j, k = ip[r["pair"]], iw[r["variant"]], it[r["template"]]
        d = r["white_margin"] - r["black_margin"]
        sup[i, j, k] = 1.0 if d > 0 else (0.0 if d < 0 else 0.5)
        acc[i, j, k] = 0.5 * sum(
            str(r.get(side, "")).strip().lower() == "yes" for side in ("white", "black"))
    if np.isnan(sup).any() or np.isnan(acc).any():
        raise SystemExit(
            f"incomplete cell: {int(np.isnan(sup).sum())} of {sup.size} "
            "(pair, wording, strength) combinations are missing; the cluster "
            "bootstrap needs a complete grid")
    return sup, acc, pairs, words, strengths


def ratio_stats(sup):
    """(Ps, wording SD of Ps, |Ps - 0.5|, ratio) for a superiority cube.

    Identical arithmetic to per_cell above, restated on the cube so the
    bootstrap recomputes the WHOLE statistic -- numerator and denominator
    together -- on each resample. Holding the denominator fixed at its observed
    value and resampling only the numerator would be the error the audit is
    about: it is the denominator that is unstable.
    """
    ps = float(sup.mean())
    num = float(sup.mean(axis=(0, 2)).std(ddof=1))
    den = abs(ps - 0.5)
    return ps, num, den, num / max(den, DEN_FLOOR)


def pctl(b, alpha=0.05):
    return [float(np.percentile(b, 100 * alpha / 2)),
            float(np.percentile(b, 100 * (1 - alpha / 2)))]


def model_intervals(cubes, accs, strengths, rng, n_boot=BOOT):
    """Pair-cluster percentile bootstrap over all of one model's arms at once.

    ONE RESAMPLE OF NAME PAIRS PER REPLICATE, SHARED BY THE THREE ARMS. The same
    twelve pairs were run against all three postings, so a replicate that used
    different names in different arms would inflate the spread with a difference
    the design does not contain. Sharing the draw is what makes the spread
    interval a statement about the same twelve people applying for three jobs.
    """
    occs = list(cubes)
    npair = cubes[occs[0]].shape[0]
    ratio_b = {o: np.empty(n_boot) for o in occs}
    ps_b = {o: np.empty(n_boot) for o in occs}
    num_b = {o: np.empty(n_boot) for o in occs}
    acc_b = {o: {t: np.empty(n_boot) for t in strengths + ["all_strengths"]}
             for o in occs}

    for j in range(n_boot):
        idx = rng.integers(0, npair, size=npair)
        for o in occs:
            ps, num, _den, ratio = ratio_stats(cubes[o][idx])
            ps_b[o][j], num_b[o][j], ratio_b[o][j] = ps, num, ratio
            a = accs[o][idx]
            for k, t in enumerate(strengths):
                acc_b[o][t][j] = float(a[..., k].mean())
            acc_b[o]["all_strengths"][j] = float(a.mean())

    stacked = np.vstack([ratio_b[o] for o in occs])
    spread_b = stacked.max(axis=0) - stacked.min(axis=0)
    numspread_b = np.vstack([num_b[o] for o in occs])
    numspread_b = numspread_b.max(axis=0) - numspread_b.min(axis=0)
    return dict(ratio=ratio_b, ps=ps_b, num=num_b, acc=acc_b,
                spread=spread_b, num_spread=numspread_b)


def add_intervals(out, rows, rng=RNG_INTERVAL):
    """Attach the pair-cluster intervals, the zero-denominator flags and the
    measured per-strength acceptance rates to every cell already in `out`.

    The existing keys are left exactly as they were. `ratio_sd_to_effect` is not
    superseded by `ratio_sd_to_effect_ci`; it is the same number with its
    uncertainty finally attached, and the paper can keep quoting it as long as it
    quotes the interval and the flag with it.
    """
    for m, arms in out.items():
        cubes, accs, strengths, axis0 = {}, {}, None, None
        for o in arms:
            sub = [r for r in rows if r["model"] == m and r["occupation"] == o]
            s, a, pairs, words, st = cube(sub)
            cubes[o], accs[o] = s, a
            # Row i of every arm's cube must be the SAME name pair, or the shared
            # resample in model_intervals would be silently comparing different
            # people across arms and the spread interval would be meaningless.
            if strengths is None:
                strengths, axis0 = st, pairs
            elif st != strengths or pairs != axis0:
                raise SystemExit(
                    f"{m}: arms are not aligned -- strengths {st} vs {strengths}, "
                    f"pairs {pairs} vs {axis0}")
            arms[o]["n_pair_clusters"] = len(pairs)
            arms[o]["n_wordings"] = len(words)

        b = model_intervals(cubes, accs, strengths, rng)

        for o in arms:
            c = arms[o]
            ps, num, den, ratio = ratio_stats(cubes[o])
            ps_ci = pctl(b["ps"][o])
            den_ci = pctl(np.abs(b["ps"][o] - 0.5))
            r_ci = pctl(b["ratio"][o])
            # The denominator is |Ps - 0.5|; it is separable from zero exactly
            # when the interval on Ps stays on one side of 0.5.
            undefined = bool(ps_ci[0] <= 0.5 <= ps_ci[1])
            side = np.sign(ps - 0.5)
            c["ratio_sd_to_effect_ci"] = r_ci
            c["ratio_sd_to_effect_ci_unbounded"] = bool(r_ci[1] >= UNBOUNDED_AT)
            c["ratio_denominator"] = float(den)
            c["ratio_denominator_ci"] = den_ci
            c["ratio_denominator_indistinguishable_from_zero"] = undefined
            c["ps_ci_pair_cluster"] = ps_ci
            c["ps_sd_across_wordings_ci"] = pctl(b["num"][o])
            c["boot_frac_effect_sign_flip"] = float(
                np.mean(np.sign(b["ps"][o] - 0.5) != side)) if side != 0 else 1.0
            c["boot_frac_ratio_over_1000pct"] = float(
                np.mean(b["ratio"][o] >= UNBOUNDED_AT))
            c["acceptance_by_strength"] = {
                t: dict(rate=float(accs[o][..., k].mean()),
                        n_yes=int(round(accs[o][..., k].sum() * 2)),
                        n_decisions=int(accs[o][..., k].size * 2),
                        ci=pctl(b["acc"][o][t]))
                for k, t in enumerate(strengths)}
            c["acceptance_by_strength"]["all_strengths"] = dict(
                rate=float(accs[o].mean()),
                n_yes=int(round(accs[o].sum() * 2)),
                n_decisions=int(accs[o].size * 2),
                ci=pctl(b["acc"][o]["all_strengths"]))

        # ---- model-level: the spread, and the difficulty mismatch ----------
        occs = list(arms)
        vals = {o: arms[o]["ratio_sd_to_effect"] for o in occs}
        hi_o = max(vals, key=vals.get)
        lo_o = min(vals, key=vals.get)
        sp_ci = pctl(b["spread"])
        arms["ratio_spread_across_occupations"] = dict(
            est=float(vals[hi_o] - vals[lo_o]), ci=sp_ci,
            ci_unbounded=bool(sp_ci[1] >= UNBOUNDED_AT),
            argmax=hi_o, argmin=lo_o,
            argmax_denominator_indistinguishable_from_zero=bool(
                arms[hi_o]["ratio_denominator_indistinguishable_from_zero"]),
            n_cells_with_undefined_denominator=int(sum(
                arms[o]["ratio_denominator_indistinguishable_from_zero"]
                for o in occs)),
            n_cells=len(occs),
            note=("max minus min of ratio_sd_to_effect across the three arms; "
                  "this is the quantity the paper calls the spread. The flag "
                  "says whether the arm supplying the maximum has an effect "
                  "separable from zero. If it does not, the spread is a "
                  "statement about a ratio that is not estimable."))

        # A denominator-free companion. SD_wording(Ps) needs no effect size to
        # divide by, so its spread across occupations is bounded and estimable
        # in every cell, including the ones where the ratio is not.
        nvals = {o: arms[o]["ps_sd_across_wordings"] for o in occs}
        nhi, nlo = max(nvals, key=nvals.get), min(nvals, key=nvals.get)
        arms["wording_sd_spread_across_occupations"] = dict(
            est=float(nvals[nhi] - nvals[nlo]), ci=pctl(b["num_spread"]),
            argmax=nhi, argmin=nlo,
            note=("spread across occupations of SD_wording(Ps) alone -- the "
                  "numerator of the ratio, with no division. Reportable in "
                  "every cell because it has no denominator to be near zero."))

        gaps = {}
        for t in strengths + ["all_strengths"]:
            rates = {o: arms[o]["acceptance_by_strength"][t]["rate"] for o in occs}
            g_hi, g_lo = max(rates, key=rates.get), min(rates, key=rates.get)
            gaps[t] = dict(max_rate=float(rates[g_hi]), max_occ=g_hi,
                           min_rate=float(rates[g_lo]), min_occ=g_lo,
                           gap_pp=float((rates[g_hi] - rates[g_lo]) * 100.0))
        worst = max((t for t in strengths), key=lambda t: gaps[t]["gap_pp"])
        arms["acceptance_by_strength_across_occupations"] = dict(
            by_strength=gaps, worst_strength=worst,
            worst_gap_pp=gaps[worst]["gap_pp"],
            note=("acceptance rate = fraction of the model's yes/no decisions "
                  "that were yes, over both résumés of every pair. The three "
                  "arms are matched on surface structure by construction; this "
                  "measures whether they are matched on difficulty, and the gap "
                  "at a fixed strength level is the answer."))
    return out


def main() -> int:
    rows = load_rows()
    occs = sorted({r["occupation"] for r in rows})
    if len(occs) < 2:
        sys.exit(f"only occupation(s) {occs} present; nothing to compare yet")

    out = {}
    print("=" * 104)
    print("STUDY 7. The same design across three structurally matched occupations.")
    print("Confirmatory: is the wording dispersion, relative to the effect, a")
    print("property of the model or of the job?")
    print("=" * 104)
    print(f"{'model':<26}{'occupation':<7}{'n':>5}{'effect':>10}{'95% CI':>20}"
          f"{'Ps':>7}{'wording SD':>12}{'SD/effect':>11}")

    for m in [x for x in ORDER if x in {r["model"] for r in rows}]:
        out[m] = {}
        for o in [x for x in OCC_ORDER if x in occs]:
            sub = [r for r in rows if r["model"] == m and r["occupation"] == o]
            if len(sub) < 100:
                continue
            c = per_cell(sub)
            c["label"] = OCCUPATIONS[o]["label"]
            c["gender_typing"] = OCCUPATIONS[o]["gender_typing"]
            out[m][o] = c
            print(f"{SHORT.get(m, m)[:25]:<26}{o:<7}{c['n']:>5}"
                  f"{c['logodds']:>+10.4f}"
                  f"{f'[{c[chr(99)+chr(105)][0]:+.3f}, {c[chr(99)+chr(105)][1]:+.3f}]':>20}"
                  f"{c['ps']:>7.3f}{c['ps_sd_across_wordings']:>12.4f}"
                  f"{c['ratio_sd_to_effect']*100:>10.0f}%")

    # ---- the confirmatory comparison -------------------------------------
    print("\nIS THE RATIO STABLE ACROSS OCCUPATIONS?")
    print(f"{'model':<26}" + "".join(f"{o:>12}" for o in OCC_ORDER if o in occs)
          + f"{'spread':>10}")
    ratios_all = []
    for m in out:
        vals = [out[m][o]["ratio_sd_to_effect"] for o in OCC_ORDER if o in out[m]]
        if len(vals) < 2:
            continue
        ratios_all.append(max(vals) - min(vals))
        print(f"{SHORT.get(m, m)[:25]:<26}"
              + "".join(f"{out[m][o]['ratio_sd_to_effect']*100:>11.0f}%"
                        for o in OCC_ORDER if o in out[m])
              + f"{(max(vals)-min(vals))*100:>9.0f}pp")

    # ---- does the sign travel? -------------------------------------------
    print("\nDOES THE DIRECTION OF THE EFFECT TRAVEL ACROSS OCCUPATIONS?")
    for m in out:
        signs = {o: ("+" if out[m][o]["logodds"] > 0 else "-")
                 for o in OCC_ORDER if o in out[m]}
        excl = {o: (out[m][o]["ci"][0] * out[m][o]["ci"][1] > 0)
                for o in OCC_ORDER if o in out[m]}
        agree = len(set(signs.values())) == 1
        print(f"  {SHORT.get(m, m)[:25]:<26}"
              + "  ".join(f"{o}:{signs[o]}{'*' if excl[o] else ''}" for o in signs)
              + ("   consistent" if agree else "   SIGN CHANGES ACROSS OCCUPATIONS"))
    print("  (* = interval excludes zero)")

    # ---- exploratory: gender typing --------------------------------------
    print("\nEXPLORATORY: effect against the occupation's gender typing.")
    print("Three occupations cannot support a claim about occupations in general.")
    for m in out:
        parts = []
        for o in OCC_ORDER:
            if o in out[m]:
                parts.append(f"{out[m][o]['gender_typing'][:6]} "
                             f"{out[m][o]['logodds']:+.3f}")
        print(f"  {SHORT.get(m, m)[:25]:<26}" + "   ".join(parts))

    # ---- the audit fix ----------------------------------------------------
    # Run last so the RNG usage above is untouched and the published point
    # estimates reproduce exactly.
    add_intervals(out, rows)
    models = [m for m in ORDER if m in out]

    def band(lo, hi, scale=100.0, unit="%"):
        top = ">1000" if hi >= UNBOUNDED_AT else f"{hi * scale:.0f}"
        return f"[{lo * scale:.0f}, {top}]{unit}"

    print("\n" + "=" * 104)
    print("TABLE 10 WITH INTERVALS. Pair-cluster percentile bootstrap, "
          f"{BOOT} draws, 12 name-pair clusters.")
    print("A ratio whose denominator's interval covers zero is not a large "
          "ratio; it is an undefined one.")
    print("=" * 104)
    print(f"{'model':<26}{'occ':<5}{'Ps':>7}{'Ps 95% CI':>18}"
          f"{'|Ps-.5|':>9}{'ratio':>8}{'ratio 95% CI':>18}  flag")
    for m in models:
        for o in [x for x in OCC_ORDER if x in out[m]]:
            c = out[m][o]
            flag = ("DENOMINATOR NOT SEPARABLE FROM ZERO"
                    if c["ratio_denominator_indistinguishable_from_zero"] else "")
            psci = c["ps_ci_pair_cluster"]
            psband = "[%.3f, %.3f]" % (psci[0], psci[1])
            print(f"{SHORT.get(m, m)[:25]:<26}{o:<5}{c['ps']:>7.3f}{psband:>18}"
                  f"{c['ratio_denominator']:>9.3f}"
                  f"{c['ratio_sd_to_effect'] * 100:>7.0f}%"
                  f"{band(*c['ratio_sd_to_effect_ci']):>18}  {flag}")
    n_bad = sum(out[m][o]["ratio_denominator_indistinguishable_from_zero"]
                for m in models for o in OCC_ORDER if o in out[m])
    n_tot = sum(1 for m in models for o in OCC_ORDER if o in out[m])
    print(f"\n  {n_bad} of {n_tot} cells have a denominator whose 95% interval "
          "covers zero.")

    print("\nTHE SPREAD, WITH ITS INTERVAL, AND WHAT IT RESTS ON.")
    for m in models:
        s = out[m]["ratio_spread_across_occupations"]
        w = out[m]["wording_sd_spread_across_occupations"]
        print(f"  {SHORT.get(m, m)[:25]:<26}spread {s['est'] * 100:>6.0f}pp  "
              f"{band(*s['ci'], unit='pp'):>18}  max from {s['argmax']}"
              + ("  <- that arm's effect is not separable from zero"
                 if s["argmax_denominator_indistinguishable_from_zero"] else ""))
        print(f"  {'':<26}denominator-free companion: "
              f"SD_wording(Ps) spread {w['est']:.4f} "
              f"[{w['ci'][0]:.4f}, {w['ci'][1]:.4f}]")

    print("\nARE THE THREE ARMS MATCHED ON DIFFICULTY? "
          "Acceptance rate at each résumé strength.")
    print("Structural matching cannot make a posting equally hard; only "
          "measurement can say whether it is.")
    print(f"{'model':<26}{'occ':<5}"
          + "".join(f"{t.split('_')[1]:>12}" for t in STRENGTHS) + f"{'all':>10}")
    for m in models:
        for o in [x for x in OCC_ORDER if x in out[m]]:
            a = out[m][o]["acceptance_by_strength"]
            print(f"{SHORT.get(m, m)[:25]:<26}{o:<5}"
                  + "".join(f"{a[t]['rate'] * 100:>11.1f}%" for t in STRENGTHS)
                  + f"{a['all_strengths']['rate'] * 100:>9.1f}%")
        g = out[m]["acceptance_by_strength_across_occupations"]
        b = g["by_strength"][g["worst_strength"]]
        print(f"{'':<26}worst mismatch: {g['worst_strength']} accepted "
              f"{b['min_rate'] * 100:.1f}% in {b['min_occ']} vs "
              f"{b['max_rate'] * 100:.1f}% in {b['max_occ']} "
              f"({g['worst_gap_pp']:.1f} pp)")

    OCCDIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
