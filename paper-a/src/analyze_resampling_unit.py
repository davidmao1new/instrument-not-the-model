"""How much does the bootstrap's resampling unit change the answer?

WHY THIS IS A SCRIPT AND NOT A REMEMBERED NUMBER. The paper claims that
resampling rows rather than name pairs understates the interval on the per-model
effect "by a factor of 3.1 to 4.8", and that on the mechanism contrasts the two
estimators diverge less but not negligibly. Both figures were measured in an
earlier session and then carried in prose. Every other number in the paper is
interpolated from an artifact by the script that typesets it; these two were the
exception, which makes them exactly the kind of claim this paper argues against.

WHAT IS MEASURED.

  A. THE POOLED EFFECT, per model, on the Study 2 design. Every name pair appears
     under twelve wordings and three templates, so 432 rows carry only 12
     independent name pairs. Resampling rows treats 36 measurements of the same
     name as 36 draws.

  B. THE MECHANISM CONTRASTS, all 168 of them. A contrast differences two
     conditions on the same cell, so the name and the resume cancel inside each
     observation before any resampling happens. The question is whether that
     cancellation is complete enough to make the resampling unit irrelevant.
     It is not: the ratio runs 0.640 to 1.446 and a third of the contrasts
     widen by more than 10%. Arm C shows the third template made this worse
     but did not create it.

  C. THE SAME CONTRASTS ON A TWO-TEMPLATE PANEL. Arm B is the "after" of a
     before/after argument the paper makes in section 6.1, and until this arm
     existed the "before" -- a width ratio of 0.83 to 1.14 across 168 contrasts
     -- was hand-typed into the typesetting script and into effectsize.boot_ci's
     docstring. It appeared under no artifact and could not be recomputed, which
     makes it the same kind of remembered number this file exists to abolish,
     and it is load-bearing: it is the whole reason the earlier draft concluded
     the contrasts were safe either way. Arm C recomputes it by dropping one
     template from the panel and changing nothing else.

     WHICH TWO TEMPLATES, AND WHY THOSE. Recoverable exactly, from three
     independent places rather than by inference. stimuli.TEMPLATES_2 is
     (T1_strong, T3_marginal) and is still what experiment_mechanism_panel.py
     uses when --templates is not given; run_suite.py adds the T2_mid arm as
     separate `mech-chat-t2` / `mech-raw-t2` jobs with an explicit
     `--templates T2_mid`, commented as having been "added after the first panel
     pass"; and analyze_template_concentration.py states outright that the panel
     was originally run on TEMPLATES_2. So the historical two-template panel is
     T1_strong + T3_marginal, and that subset is the headline. The other two
     subsets are computed as well, because the claim the paper needs is not
     "the range differs on one subset" but "the reported range is not the range
     of any two-template panel", and that requires all three.

All arms use the same seed and the same number of replicates, so a comparison
between them is of estimators and of panels, not of Monte Carlo error.

    .venv/Scripts/python.exe paper-a/src/analyze_resampling_unit.py
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_mech_panel as amp  # noqa: E402
import effectsize as es  # noqa: E402
import stimuli as st  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DS = ROOT / "paper-a" / "data" / "delta_stability"
OUT = DS / "resampling_unit.json"
N_BOOT = 20_000
# The contrast arms run at 8,000 rather than 20,000 because 168 contrasts x two
# estimators x four panels is the cost driver here, and unlike analyze_mech_panel
# nothing downstream ranks these by p-value, so bootstrap quantisation cannot
# reorder anything. Named rather than inlined so arms B and C provably share it.
N_BOOT_CONTRAST = 8_000
SEED = 20260730

# The two résumé templates the mechanism panel carried before the T2_mid arm was
# added. Not a guess: stimuli.TEMPLATES_2 is the default that
# experiment_mechanism_panel.py still falls back to, run_suite.py adds T2_mid as
# a separate job with an explicit --templates flag, and
# analyze_template_concentration.py records the same history in prose.
HISTORICAL_PAIR = tuple(st.TEMPLATES_2)

# The range that was carried in prose instead of in an artifact, kept so the
# paper can report what was claimed beside what recomputes.
REPORTED_TWO_TEMPLATE_RANGE = (0.83, 1.14)
REPORTED_TWO_TEMPLATE_N = 168

# WHY ARM C RE-RUNS THE HISTORICAL PANEL UNDER SPARE SEEDS. min and max over 168
# contrasts are order statistics, and order statistics are exactly where a
# bootstrap's own Monte Carlo error shows up first. Without this the obvious
# objection to arm C is "your endpoints are noisy, the earlier ones were not",
# and that objection has to be answerable with a number rather than a claim.
ALT_SEEDS = (20260728, 20260729, 7)

ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SHORT = {"llama-2-7b-chat": "Llama-2-7B-chat",
         "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
         "mistral-7b-instruct-v0.1": "Mistral-7B-Instruct v0.1",
         "mistral-7b-instruct-v0.3": "Mistral-7B-Instruct v0.3"}


def load_study2(model):
    rows = []
    for f in sorted(DS.glob("delta_*.jsonl")):
        for r in st.read_jsonl(f):
            if (r.get("model") == model
                    and r.get("white_margin") is not None
                    and r.get("black_margin") is not None):
                rows.append(r)
    return rows


def contrast_ratios(rows, templates=None, seed=SEED):
    """clustered / i.i.d. interval width, one value per panel contrast.

    `templates` restricts the panel BEFORE the contrast is formed, which is the
    only honest way to ask what a smaller panel would have shown. The cell key
    is (template, pair), so dropping a template removes one correlated row from
    every name-pair cluster and leaves the contrast family, the pair count and
    the i.i.d. estimator's own logic untouched. Every call uses the same
    replicate count, and by default the same seed, so a difference between two
    calls is a difference between panels. `seed` is overridable only so arm C
    can measure how much of its own answer is Monte Carlo error.
    """
    out = []
    if templates is not None:
        keep = set(templates)
        rows = [r for r in rows if r["template"] in keep]
    for model in sorted({r["model"] for r in rows}):
        for mode in sorted({r["mode"] for r in rows if r["model"] == model}):
            mr = [r for r in rows if r["model"] == model and r["mode"] == mode]
            by = {c: [r for r in mr if r["cond"] == c] for c in amp.ORDER}
            by = {c: v for c, v in by.items() if v}
            for _label, a, b in amp.CONTRASTS:
                if a not in by or b not in by:
                    continue
                ka = {(r["template"], r["pair"]):
                      r["white_margin"] - r["black_margin"] for r in by[a]}
                kb = {(r["template"], r["pair"]):
                      r["white_margin"] - r["black_margin"] for r in by[b]}
                keys = sorted(set(ka) & set(kb))
                if not keys:
                    continue
                d = np.array([ka[k] - kb[k] for k in keys])
                cl = np.array([k[1] for k in keys])
                i = es.boot_ci(d, lambda x: float(x.mean()), N_BOOT_CONTRAST,
                               np.random.default_rng(seed))
                c = es.boot_ci(d, lambda x: float(x.mean()), N_BOOT_CONTRAST,
                               np.random.default_rng(seed), clusters=cl)
                wi = i["ci"][1] - i["ci"][0]
                if wi > 0:
                    out.append((c["ci"][1] - c["ci"][0]) / wi)
    return np.array(out)


def summarise(r, templates, rows_per_cluster):
    """The five numbers the paper quotes, plus what panel produced them.

    `templates` and `rows_per_cluster` travel with the summary because the whole
    point of arm C is that these numbers are meaningless without knowing how
    many correlated rows a name pair contributed.
    """
    lo, hi = REPORTED_TWO_TEMPLATE_RANGE
    return dict(
        n=int(len(r)), min=float(r.min()), median=float(np.median(r)),
        mean=float(r.mean()), max=float(r.max()),
        frac_widen_over_10pct=float((r > 1.10).mean()),
        frac_widen_over_25pct=float((r > 1.25).mean()),
        frac_inside_reported_range=float(((r >= lo) & (r <= hi)).mean()),
        templates_used=list(templates), n_templates=len(templates),
        rows_per_cluster=rows_per_cluster)


def main() -> int:
    out = {"n_boot": N_BOOT, "n_boot_contrast": N_BOOT_CONTRAST, "seed": SEED}

    # ---- A. the pooled per-model effect ----------------------------------
    print("=" * 100)
    print("A. THE POOLED EFFECT. Rows are crossed: every name pair appears under")
    print("   every wording and every template, so the rows are not independent.")
    print("=" * 100)
    print(f"{'model':<26}{'rows':>6}{'pairs':>7}{'i.i.d. width':>14}"
          f"{'clustered width':>17}{'ratio':>8}{'iid sig':>9}{'clu sig':>9}")
    per = {}
    for m in ORDER:
        rows = load_study2(m)
        if not rows:
            continue
        d = np.array([r["white_margin"] - r["black_margin"] for r in rows])
        cl = np.array([r["pair"] for r in rows])
        i = es.boot_ci(d, lambda x: float(x.mean()), N_BOOT,
                       np.random.default_rng(SEED))
        c = es.boot_ci(d, lambda x: float(x.mean()), N_BOOT,
                       np.random.default_rng(SEED), clusters=cl)
        wi = i["ci"][1] - i["ci"][0]
        wc = c["ci"][1] - c["ci"][0]
        sig_i = i["ci"][0] * i["ci"][1] > 0
        sig_c = c["ci"][0] * c["ci"][1] > 0
        per[m] = dict(n_rows=len(d), n_pairs=int(len(set(cl))),
                      est=i["est"], iid_ci=i["ci"], clustered_ci=c["ci"],
                      iid_width=wi, clustered_width=wc, ratio=wc / wi,
                      iid_significant=bool(sig_i),
                      clustered_significant=bool(sig_c))
        print(f"{m:<26}{len(d):>6}{len(set(cl)):>7}{wi:>14.4f}{wc:>17.4f}"
              f"{wc / wi:>8.2f}{str(sig_i):>9}{str(sig_c):>9}")
    out["pooled_effect"] = per
    if per:
        rr = [v["ratio"] for v in per.values()]
        lost = [m for m, v in per.items()
                if v["iid_significant"] and not v["clustered_significant"]]
        out["pooled_summary"] = dict(
            min_ratio=float(min(rr)), max_ratio=float(max(rr)),
            n_models=len(rr),
            n_significant_iid=sum(1 for v in per.values() if v["iid_significant"]),
            n_significant_clustered=sum(1 for v in per.values()
                                        if v["clustered_significant"]),
            models_losing_significance=lost)
        print(f"\n  the i.i.d. interval is too narrow by a factor of "
              f"{min(rr):.1f} to {max(rr):.1f}")
        print(f"  effects distinguishable from zero: "
              f"{out['pooled_summary']['n_significant_iid']} under the row "
              f"bootstrap, {out['pooled_summary']['n_significant_clustered']} "
              f"under the cluster bootstrap")
        if lost:
            print(f"  lose significance under the correction: "
                  f"{', '.join(SHORT.get(m, m) for m in lost)}")

    # ---- B. the mechanism contrasts --------------------------------------
    print("\n" + "=" * 100)
    print("B. THE MECHANISM CONTRASTS. Pairing cancels the name inside each")
    print("   observation, so the resampling unit should matter less. Does it?")
    print("=" * 100)
    rows = amp.load()
    all_templates = tuple(sorted({r["template"] for r in rows})) if rows else ()
    r = contrast_ratios(rows) if rows else np.array([])
    if len(r):
        out["contrasts"] = summarise(r, all_templates, len(all_templates))
        print(f"  {len(r)} contrasts on {len(all_templates)} templates "
              f"({', '.join(all_templates)})")
        print(f"  clustered / i.i.d. interval width:  min {r.min():.3f}   "
              f"median {np.median(r):.3f}   max {r.max():.3f}")
        print(f"  widen by more than 10%: {(r > 1.10).mean():.1%}")
        print(f"  widen by more than 25%: {(r > 1.25).mean():.1%}")
        print("\n  READING. The median is close to 1, so on a typical contrast the")
        print("  choice is immaterial. It is not immaterial on all of them, which")
        print("  is the only fact that matters when choosing an estimator.")

    # ---- C. the same contrasts on a two-template panel --------------------
    lo_rep, hi_rep = REPORTED_TWO_TEMPLATE_RANGE
    if len(r) and len(all_templates) >= 3:
        print("\n" + "=" * 100)
        print("C. THE TWO-TEMPLATE PANEL. Arm B is the 'after'. This is the 'before',")
        print(f"   recomputed rather than remembered. Reported in prose as "
              f"{lo_rep} to {hi_rep} across {REPORTED_TWO_TEMPLATE_N} contrasts.")
        print("=" * 100)
        print(f"{'templates':<28}{'n':>5}{'min':>9}{'median':>9}{'max':>9}"
              f"{'>10%':>8}{'>25%':>8}{'inside':>9}   {'reproduces prose range?'}")
        subsets = {}
        for sub in itertools.combinations(all_templates, 2):
            rr = contrast_ratios(rows, sub)
            if not len(rr):
                continue
            sm = summarise(rr, sub, 2)
            # "Reproduces" is deliberately generous: it asks only whether the
            # subset's own min and max sit inside the reported interval, not
            # whether they equal its endpoints. A claim that cannot pass even
            # the generous test has not been rounded, it is wrong.
            sm["reproduces_reported_range"] = bool(
                sm["min"] >= lo_rep and sm["max"] <= hi_rep)
            sm["is_historical_panel"] = (tuple(sub) == HISTORICAL_PAIR)
            subsets["+".join(sub)] = sm
            mark = "yes" if sm["reproduces_reported_range"] else "NO"
            star = "  <- historical panel" if sm["is_historical_panel"] else ""
            print(f"{'+'.join(sub):<28}{sm['n']:>5}{sm['min']:>9.3f}"
                  f"{sm['median']:>9.3f}{sm['max']:>9.3f}"
                  f"{sm['frac_widen_over_10pct']:>8.1%}"
                  f"{sm['frac_widen_over_25pct']:>8.1%}"
                  f"{sm['frac_inside_reported_range']:>9.1%}   {mark}{star}")
        out["contrasts_two_template_all_subsets"] = subsets
        hist_key = "+".join(HISTORICAL_PAIR)
        if hist_key in subsets:
            out["contrasts_two_template"] = subsets[hist_key]

        # Is the gap between 0.83-1.14 and what arm C measures large enough to
        # survive the bootstrap's own noise in the endpoints?
        if hist_key in subsets:
            mins = [subsets[hist_key]["min"]]
            maxs = [subsets[hist_key]["max"]]
            for alt in ALT_SEEDS:
                rr = contrast_ratios(rows, HISTORICAL_PAIR, seed=alt)
                mins.append(float(rr.min()))
                maxs.append(float(rr.max()))
            out["contrasts_two_template_seed_stability"] = dict(
                seeds=[SEED, *ALT_SEEDS],
                min_across_seeds=min(mins), max_of_min_across_seeds=max(mins),
                min_of_max_across_seeds=min(maxs), max_across_seeds=max(maxs),
                reported_min_reachable=bool(max(mins) >= lo_rep),
                reported_max_reachable=bool(min(maxs) <= hi_rep),
                note="Endpoints are order statistics over 168 contrasts, so "
                     "this is where Monte Carlo error would show first. If "
                     "neither reported endpoint is reachable under any seed, "
                     "the prose range differs from the data by more than the "
                     "bootstrap's own noise.")
            print(f"\n  seed stability on {hist_key}, seeds "
                  f"{[SEED, *ALT_SEEDS]}:")
            print(f"    min ranges {min(mins):.3f}-{max(mins):.3f}  "
                  f"(reported min {lo_rep} reachable: "
                  f"{'yes' if max(mins) >= lo_rep else 'NO'})")
            print(f"    max ranges {min(maxs):.3f}-{max(maxs):.3f}  "
                  f"(reported max {hi_rep} reachable: "
                  f"{'yes' if min(maxs) <= hi_rep else 'NO'})")
        out["contrasts_two_template_prior_claim"] = dict(
            min=lo_rep, max=hi_rep, n=REPORTED_TWO_TEMPLATE_N,
            source="hand-typed in paper-a/src/build_paper_v3.py 6.1 and in "
                   "effectsize.boot_ci's docstring; never held in an artifact",
            reproduced_by_any_subset=any(
                s["reproduces_reported_range"] for s in subsets.values()),
            reproduced_by_historical_panel=bool(
                subsets.get(hist_key, {}).get("reproduces_reported_range")),
            note="Superseded by contrasts_two_template, which recomputes the "
                 "same quantity on the same contrast family. Retained so the "
                 "paper can report what the earlier draft claimed beside what "
                 "the data gives.")
        if subsets:
            hist = subsets.get(hist_key)
            any_ok = out["contrasts_two_template_prior_claim"][
                "reproduced_by_any_subset"]
            print()
            if any_ok:
                print("  READING. The reported range is reproducible on a two-template")
                print("  panel, and section 6.1's before/after stands as written.")
            else:
                print(f"  READING. No two-template subset reproduces {lo_rep} to "
                      f"{hi_rep}. On the")
                print(f"  historical panel ({'+'.join(HISTORICAL_PAIR)}) the ratio "
                      f"runs {hist['min']:.3f} to {hist['max']:.3f},")
                print(f"  with only {hist['frac_inside_reported_range']:.1%} of "
                      f"contrasts inside the reported interval.")
                print("  The DIRECTION of section 6.1's argument survives -- dropping a")
                print(f"  correlated row per cluster does pull the estimators together, "
                      f"from")
                print(f"  {out['contrasts']['min']:.3f}-{out['contrasts']['max']:.3f} "
                      f"to {hist['min']:.3f}-{hist['max']:.3f}, and the share of "
                      f"contrasts widening")
                print(f"  by more than 25% falls from "
                      f"{out['contrasts']['frac_widen_over_25pct']:.1%} to "
                      f"{hist['frac_widen_over_25pct']:.1%}. What does not survive is")
                worst = max(1.0 - hist["min"], hist["max"] - 1.0)
                print("  the 'before' being narrow enough to call the two estimators")
                print(f"  interchangeable: {hist['min']:.2f} is not {lo_rep}, and a "
                      f"contrast whose interval moves by")
                print(f"  {worst:.0%} with the resampling unit is not one the "
                      f"resampling unit left alone.")

    DS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
