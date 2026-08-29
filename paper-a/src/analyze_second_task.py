"""Does instrument variance survive leaving the hiring domain?

WHAT THIS COMPARES. Study 2 measured, per model, the between-wording standard
deviation of the demographic effect and the ratio of that dispersion to the
effect itself. This runs the identical analysis on two non-hiring domains
measured with the identical twelve wordings, the identical twelve name pairs and
the identical null perturbations, and puts the three side by side.

THE COMPARISON IS OF RATIOS AND OF MULTIPLES OF A NOISE FLOOR, NEVER OF RAW
MAGNITUDES. Three domains put three models at three different places on the
logistic curve, and §6.2 is a whole section on why a raw magnitude does not
travel across operating points. Each domain is therefore reported against its
own effect and against its own byte-identical replicate.

CENSORING, WHICH THIS DESIGN HAS AND STUDY 2 LARGELY DID NOT. The margin is
read from the top-100 next-token window. When a model is very confident the
losing option falls outside that window, no margin can be computed, and the cell
is dropped. That censoring is NOT random: it happens exactly where |margin| is
largest. Two things are therefore measured rather than assumed --

  1. the censoring rate, per domain, model and strength level;
  2. whether censoring is DIFFERENTIAL BY ARM. If White-named and Black-named
     prompts are censored at different rates, dropping censored cells biases the
     paired estimate, and the size of that threat has to be stated. A pair is
     dropped when EITHER side is censored, so a per-arm rate difference is the
     quantity that matters.

A domain whose effect is indistinguishable from zero still answers the question
this experiment asks, because a dispersion around zero is still a dispersion;
what it cannot support is a ratio, and the ratio is suppressed where the
denominator is not separable from zero, exactly as in §4.5.

    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/analyze_second_task.py
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "paper-a" / "data" / "second_task"
S2 = ROOT / "paper-a" / "data" / "delta_stability" / "study2_v2.json"
OUT = D / "second_task_analysis.json"

N_BOOT = 10_000
SEED = 20260801

ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SHORT = {"llama-2-7b-chat": "Llama-2-7B", "llama-3.1-8b-instruct": "Llama-3.1-8B",
         "mistral-7b-instruct-v0.1": "Mistral v0.1",
         "mistral-7b-instruct-v0.3": "Mistral v0.3"}
LABELS = {"housing": "rental tenancy screening",
          "moderation": "content moderation"}


def load(domain: str, model: str) -> list[dict]:
    f = D / f"{domain}_{model}.jsonl"
    if not f.exists():
        return []
    # Deduplicate on the design cell, keeping the last write, which is the rule
    # every other analysis in this project uses for a resumable experiment.
    best = {}
    for r in st.read_jsonl(f):
        best[(r["variant"], r["level"], r["pair"])] = r
    return list(best.values())


def censoring(rows: list[dict]) -> dict:
    """How much was lost to the top-100 window, and was it lost evenly?"""
    n = len(rows)
    w_cens = sum(1 for r in rows if r["white_margin"] is None)
    b_cens = sum(1 for r in rows if r["black_margin"] is None)
    pair_cens = sum(1 for r in rows
                    if r["white_margin"] is None or r["black_margin"] is None)
    by_level = defaultdict(lambda: [0, 0])
    for r in rows:
        e = by_level[r["level"]]
        e[1] += 1
        if r["white_margin"] is None or r["black_margin"] is None:
            e[0] += 1
    # McNemar-style exact test on discordant censoring: of the pairs censored on
    # exactly one side, is the side systematic?
    only_w = sum(1 for r in rows
                 if r["white_margin"] is None and r["black_margin"] is not None)
    only_b = sum(1 for r in rows
                 if r["black_margin"] is None and r["white_margin"] is not None)
    p_diff = None
    if only_w + only_b:
        from scipy.stats import binomtest  # noqa: PLC0415
        p_diff = float(binomtest(only_w, only_w + only_b, 0.5).pvalue)
    return dict(
        n_cells=n, n_white_censored=w_cens, n_black_censored=b_cens,
        n_pairs_dropped=pair_cens,
        frac_pairs_dropped=pair_cens / n if n else None,
        censored_white_only=only_w, censored_black_only=only_b,
        differential_p=p_diff,
        by_level={k: dict(dropped=v[0], n=v[1],
                          frac=v[0] / v[1] if v[1] else None)
                  for k, v in by_level.items()},
    )


def censoring_bounds(rows: list[dict]) -> dict | None:
    """Worst-case bounds on the effect when censored pairs are not dropped.

    WHY THIS IS NEEDED AND DROPPING IS NOT ENOUGH. A cell is censored when the
    losing option falls outside the top-100 window, which happens exactly when
    the model is most confident, so the missingness is informative. If it were
    symmetric across the two arms the paired estimate would lose precision and
    stay unbiased. It is not always symmetric: on at least one domain-model cell
    the White-named side is censored significantly more often than the
    Black-named side, and dropping the pair then removes high-magnitude cells
    from one arm preferentially.

    WHAT IS COMPUTED. A censored margin is not unknown, only unobserved from
    below: the model emitted a verdict, so the margin has that verdict's sign,
    and it is at least as extreme as the largest margin the window did resolve
    in the same direction. Substituting that bound for every censored value
    gives one extreme; substituting it with the opposite sign gives the other.
    The true effect lies between them. Wide bounds are an honest statement that
    the design cannot resolve the question; narrow ones say the drop is safe.
    """
    resolved = [r for r in rows
                if r["white_margin"] is not None and r["black_margin"] is not None]
    censored = [r for r in rows
                if r["white_margin"] is None or r["black_margin"] is None]
    if not resolved:
        return None
    mags = [abs(r[s + "_margin"]) for r in resolved for s in ("white", "black")]
    big = float(max(mags))

    def fill(r, side, sign):
        v = r[side + "_margin"]
        if v is not None:
            return v
        # verdict carries the sign the model actually chose; `sign` selects
        # which extreme we are testing.
        return sign * big if r[side] == "yes" else -sign * big

    lo_hi = []
    for sign in (+1, -1):
        d = [fill(r, "white", sign) - fill(r, "black", sign) for r in rows]
        lo_hi.append(float(np.mean(d)))
    obs = float(np.mean([r["white_margin"] - r["black_margin"] for r in resolved]))
    return dict(
        n_censored_pairs=len(censored), extreme_magnitude_used=big,
        effect_dropping_censored=obs,
        bound_low=float(min(lo_hi)), bound_high=float(max(lo_hi)),
        bounds_contain_zero=bool(min(lo_hi) <= 0 <= max(lo_hi)),
        sign_stable_under_bounds=bool(min(lo_hi) * max(lo_hi) > 0),
    )


def per_model(rows: list[dict], rng) -> dict | None:
    usable = [r for r in rows
              if r["white_margin"] is not None and r["black_margin"] is not None]
    if len(usable) < 24:
        return None
    overall = es.describe(usable, n_boot=N_BOOT, rng=rng, cluster_key="pair")
    if overall is None:
        return None

    per_variant = {}
    for v in sorted({r["variant"] for r in usable}):
        vr = [r for r in usable if r["variant"] == v]
        d = np.array([r["white_margin"] - r["black_margin"] for r in vr])
        ps, _ = es.superiority(d)
        per_variant[v] = dict(n=len(vr), logodds=float(d.mean()),
                              ps=float(ps),
                              kind=vr[0]["variant_kind"])
    lo_by_v = np.array([per_variant[v]["logodds"] for v in per_variant])
    ps_by_v = np.array([per_variant[v]["ps"] for v in per_variant])

    # Noise floor: S1 and N1 are byte-identical by construction, so any
    # disagreement between them is measurement error and nothing else.
    floor = None
    if "S1" in per_variant and "N1" in per_variant:
        a = {(r["level"], r["pair"]): r["white_margin"] - r["black_margin"]
             for r in usable if r["variant"] == "S1"}
        b = {(r["level"], r["pair"]): r["white_margin"] - r["black_margin"]
             for r in usable if r["variant"] == "N1"}
        keys = sorted(set(a) & set(b))
        if keys:
            diff = np.array([a[k] - b[k] for k in keys])
            n_id = int((diff == 0).sum())
            # SD of the per-cell replicate difference, halved to a per-
            # measurement SD, then scaled to the SD of a per-wording MEAN --
            # the same construction analyze_noise_floor.py uses, so the ratio
            # below is comparable with §4.1's.
            sd_cell = float(diff.std(ddof=1) / np.sqrt(2)) if len(diff) > 1 else 0.0
            floor = dict(n_cells=len(keys), n_identical=n_id,
                         frac_identical=n_id / len(keys),
                         sigma_per_cell=sd_cell,
                         sigma_on_variant_mean=sd_cell / np.sqrt(len(keys)))

    sd_lo = float(lo_by_v.std(ddof=1))
    sd_ps = float(ps_by_v.std(ddof=1))
    eff_lo = abs(overall["logodds"]["est"])
    eff_ps = abs(overall["superiority"]["est"] - 0.5)
    identified = overall["logodds"]["ci"][0] * overall["logodds"]["ci"][1] > 0

    # THE ARM SPLIT ON THE SAME SCALE AS THE REST OF §4.6. Table 16 reports
    # probability of superiority, so an arm SD in log-odds beside it invites
    # exactly the cross-scale comparison §6.2 forbids. Both are emitted; the
    # paper quotes the P(sup) pair.
    sem = np.array([per_variant[v]["logodds"] for v in per_variant
                    if per_variant[v]["kind"] == "semantic"])
    nul = np.array([per_variant[v]["logodds"] for v in per_variant
                    if per_variant[v]["kind"] == "null"])
    sem_ps = np.array([per_variant[v]["ps"] for v in per_variant
                       if per_variant[v]["kind"] == "semantic"])
    nul_ps = np.array([per_variant[v]["ps"] for v in per_variant
                       if per_variant[v]["kind"] == "null"])

    return dict(
        overall=overall, per_variant=per_variant,
        sd_across_wordings_logodds=sd_lo,
        sd_across_wordings_ps=sd_ps,
        effect_logodds=overall["logodds"]["est"],
        effect_ps=overall["superiority"]["est"],
        effect_identified=bool(identified),
        ratio_sd_to_effect_logodds=(sd_lo / eff_lo) if eff_lo > 1e-9 else None,
        ratio_sd_to_effect_ps=(sd_ps / eff_ps) if eff_ps > 1e-9 else None,
        noise_floor=floor,
        # A ZERO FLOOR IS A RESULT, NOT A MISSING MEASUREMENT. This used to
        # return None whenever the floor was zero, and the printed summary then
        # said the replicate was "not complete" -- which was false and threw
        # away the finding. The floor here is exactly zero on every cell: all
        # 284 byte-identical replicate cells returned bitwise-identical
        # margins. The ratio is therefore unbounded rather than unknown, and
        # what a reader needs is the agreement count, so both are recorded and
        # the None is reserved for a genuinely absent replicate.
        sd_over_noise=(sd_lo / floor["sigma_on_variant_mean"])
        if floor and floor["sigma_on_variant_mean"] > 0 else None,
        noise_floor_is_exactly_zero=bool(
            floor and floor["sigma_on_variant_mean"] == 0.0),
        replicate_agreement=(f"{floor['n_identical']}/{floor['n_cells']}"
                             if floor else None),
        arm_sd_ps=dict(
            semantic=float(sem_ps.std(ddof=1)) if len(sem_ps) > 1 else None,
            null=float(nul_ps.std(ddof=1)) if len(nul_ps) > 1 else None),
        arm_sd=dict(semantic=float(sem.std(ddof=1)) if len(sem) > 1 else None,
                    null=float(nul.std(ddof=1)) if len(nul) > 1 else None),
        wording_range_logodds=[float(lo_by_v.min()), float(lo_by_v.max())],
    )


def main() -> int:
    rng = np.random.default_rng(SEED)
    domains = sorted({f.name.split("_")[0] for f in D.glob("*.jsonl")})
    if not domains:
        print("no second-task data yet", file=sys.stderr)
        return 1

    out = {"_what": "Study 2's analysis, on non-hiring domains, same wordings "
                    "and names.",
           "n_boot": N_BOOT, "seed": SEED,
           "domains": domains, "domain_labels": LABELS,
           "by_domain": {}}

    s2 = json.loads(S2.read_text(encoding="utf-8")) if S2.exists() else {}

    for dom in domains:
        blk = {"models": {}, "censoring": {}}
        for m in ORDER:
            rows = load(dom, m)
            if not rows:
                continue
            blk["censoring"][m] = censoring(rows)
            _b = censoring_bounds(rows)
            if _b:
                blk["censoring"][m]["bounds"] = _b
            r = per_model(rows, rng)
            if r:
                blk["models"][m] = r
        out["by_domain"][dom] = blk

    # ---- the cross-domain statement the paper makes ----------------------
    rows_summary = []
    for dom in domains:
        for m, v in out["by_domain"][dom]["models"].items():
            rows_summary.append(dict(
                domain=dom, model=m,
                sd=v["sd_across_wordings_logodds"],
                ratio=v["ratio_sd_to_effect_logodds"],
                ratio_ps=v["ratio_sd_to_effect_ps"],
                sd_ps=v["sd_across_wordings_ps"],
                effect_ps=v["effect_ps"],
                identified=v["effect_identified"],
                sd_over_noise=v["sd_over_noise"],
                noise_floor=v.get("noise_floor"),
                noise_floor_is_exactly_zero=v.get("noise_floor_is_exactly_zero"),
                replicate_agreement=v.get("replicate_agreement")))
    hiring = []
    for m in ORDER:
        if m in s2:
            e = abs(s2[m]["overall"]["logodds"]["est"])
            sd = s2[m].get("sigma_variant_raw")
            if sd is None:
                sd = (s2[m].get("overall") or {}).get("sd_logodds_across_wordings")
            hiring.append(dict(model=m, effect=e))
    # A DISPERSION-TO-EFFECT RATIO IS ONLY DEFINED WHERE THE EFFECT IS. An
    # earlier version of this summary took the range over every cell, so the
    # paper's §4.6 printed "0.76x to 7.75x where the effect is separable from
    # zero" while the 7.75 came from the one cell where it is NOT -- the
    # quotient of a spread by something indistinguishable from nothing. The
    # range is now taken over identified cells only, and the count of
    # suppressed cells travels with it so their absence is visible.
    # THE SUMMARY RATIO IS ON THE PROBABILITY-OF-SUPERIORITY SCALE, because
    # every other panel in the paper is. Table 3 (§4.1) and Table 17 (§4.7)
    # both report P(sup), and the abstract puts all three ranges side by side.
    # A dispersion-to-effect ratio is NOT invariant under the logit transform,
    # so quoting this study in log-odds beside those two in P(sup) is the
    # operating-point error §6.2 is a whole section about, committed by this
    # paper. The log-odds values stay in the artifact and in `ratio_logodds`.
    ratios = [r["ratio_ps"] for r in rows_summary
              if r.get("ratio_ps") is not None and r["identified"]]
    ratios_logodds = [r["ratio"] for r in rows_summary
                      if r["ratio"] is not None and r["identified"]]
    son = [r["sd_over_noise"] for r in rows_summary if r["sd_over_noise"]]
    out["summary"] = dict(
        n_domain_model_cells=len(rows_summary),
        n_with_identified_effect=sum(1 for r in rows_summary if r["identified"]),
        n_ratio_suppressed_unidentified=sum(
            1 for r in rows_summary
            if r["ratio"] is not None and not r["identified"]),
        ratio_basis="cells whose effect interval excludes zero; ratio is "
                    "SD / |P(sup) - 0.5|, the same scale as Tables 3 and 17",
        ratio_logodds_min=float(min(ratios_logodds)) if ratios_logodds else None,
        ratio_logodds_max=float(max(ratios_logodds)) if ratios_logodds else None,
        ratio_min=float(min(ratios)) if ratios else None,
        ratio_max=float(max(ratios)) if ratios else None,
        sd_over_noise_min=float(min(son)) if son else None,
        sd_over_noise_max=float(max(son)) if son else None,
        # The replicate arm, summarised. Every cell of this study carries an
        # S1/N1 pair whose assembled prompts are byte-identical, so the arm is
        # a reproducibility measurement as well as a dispersion one.
        n_replicate_cells=sum((r.get("noise_floor") or {}).get("n_cells", 0)
                              for r in rows_summary),
        n_replicate_cells_identical=sum(
            (r.get("noise_floor") or {}).get("n_identical", 0)
            for r in rows_summary),
        n_cells_with_zero_noise_floor=sum(
            1 for r in rows_summary if r.get("noise_floor_is_exactly_zero")),
        per_cell=rows_summary,
    )

    # THE CLAUSE SHOULD NOT REQUIRE THE STRONGEST COMPARISON TO EXIST.
    # It used to be emitted only when the noise-floor ratio was available, and
    # that ratio needs an S1/N1 replicate this study does not yet have -- so a
    # completed eight-cell study across two domains contributed NOTHING to the
    # abstract because one derived quantity was missing. The clause now
    # degrades: the noise-floor comparison when it exists, the
    # dispersion-to-effect ratio otherwise, and both are interpolated.
    _doms = " and ".join(LABELS.get(d, d) for d in domains)
    _n_cells = len(rows_summary)
    _n_id = sum(1 for r in rows_summary if r["identified"])
    if ratios and son:
        out["abstract_clause"] = (
            "The same instrument variance appears outside hiring: running the "
            "identical twelve wordings, names and null edits on "
            + _doms + ", the between-wording standard deviation is "
            f"{min(son):.0f} to {max(son):.0f} times each domain\u2019s own "
            "measurement noise floor, so the finding is about prompted "
            "measurement rather than about hiring.")
    elif ratios:
        out["abstract_clause"] = (
            # One sentence. The abstract carries eight such clauses, one per
            # study; the domain names, cell count and identification count
            # are all in §4.6 and Table 16 for a reader who wants them.
            "Outside hiring — " + _doms + f", {_n_cells} domain-by-model "
            f"cells — the same wordings move the effect by {min(ratios):.2f} "
            f"to {max(ratios):.2f} of itself on the {_n_id} cells where it is "
            "separable from zero, so this is a fact about prompted "
            "measurement rather than about hiring.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    # ---- print -----------------------------------------------------------
    print("=" * 104)
    print("INSTRUMENT VARIANCE OUTSIDE HIRING")
    print("=" * 104)
    print(f"{'domain':<12}{'model':<15}{'n':>6}{'effect':>10}{'id?':>5}"
          f"{'SD(word)':>10}{'SD/eff':>9}{'SD/noise':>10}{'dropped':>9}")
    for dom in domains:
        for m in ORDER:
            v = out["by_domain"][dom]["models"].get(m)
            c = out["by_domain"][dom]["censoring"].get(m)
            if not v:
                continue
            rr = v["ratio_sd_to_effect_logodds"]
            son_v = v["sd_over_noise"]
            rr_s = f"{rr:.2f}" if rr else "-"
            son_s = f"{son_v:.1f}" if son_v else "-"
            drop = c["frac_pairs_dropped"] if c else 0.0
            print(f"{dom:<12}{SHORT.get(m, m):<15}{v['overall']['n']:>6}"
                  f"{v['effect_logodds']:>+10.4f}"
                  f"{('yes' if v['effect_identified'] else 'no'):>5}"
                  f"{v['sd_across_wordings_logodds']:>10.4f}"
                  f"{rr_s:>9}{son_s:>10}{drop:>8.1%}")
    print("\nCENSORING (top-100 window): differential-by-arm test, "
          "and what the drop could be hiding")
    for dom in domains:
        for m, c in out["by_domain"][dom]["censoring"].items():
            if c["n_pairs_dropped"]:
                print(f"  {dom:<12}{SHORT.get(m, m):<15}"
                      f"dropped {c['frac_pairs_dropped']:>6.1%}  "
                      f"white-only {c['censored_white_only']:>3}  "
                      f"black-only {c['censored_black_only']:>3}  "
                      f"p={c['differential_p'] if c['differential_p'] is not None else float('nan'):.3f}", end="")
                b = c.get("bounds")
                if b:
                    print(f"  | effect {b['effect_dropping_censored']:+.4f}"
                          f"  bounds [{b['bound_low']:+.4f}, "
                          f"{b['bound_high']:+.4f}]"
                          + ("  SIGN STABLE" if b["sign_stable_under_bounds"]
                             else "  sign not determined"))
                else:
                    print()
    s = out["summary"]
    if s["ratio_min"] is not None:
        print(f"\n  dispersion / effect        {s['ratio_min']:.2f} to "
              f"{s['ratio_max']:.2f}")
    if s["sd_over_noise_min"] is not None:
        print(f"  dispersion / noise floor   {s['sd_over_noise_min']:.1f} to "
              f"{s['sd_over_noise_max']:.1f}")
    else:
        _rep = s.get("n_replicate_cells") or 0
        _id = s.get("n_replicate_cells_identical") or 0
        if _rep and _id == _rep:
            print(f"  dispersion / noise floor   unbounded: the floor is "
                  f"EXACTLY ZERO on "
                  f"{s['n_cells_with_zero_noise_floor']} of "
                  f"{s['n_domain_model_cells']} cells")
            print(f"                             all {_id} byte-identical "
                  f"replicate cells returned bitwise-identical margins")
        else:
            print("  dispersion / noise floor   not estimable "
                  f"({_id} of {_rep} replicate cells agree)")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
