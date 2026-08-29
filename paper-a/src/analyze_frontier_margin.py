"""The wording study on frontier API models, analysed exactly as Study 2 is.

WHY THIS MATTERS FOR THE PAPER'S SCOPE. §10.2 concedes that the panel is
open-weight only and expects, without showing, that the stimulus-side results
transfer. The obstacle was never cost -- the whole run below is $0.67 at list
price -- it was that the paper's outcome needs a next-token distribution, and
`probe_frontier_api.py` found that none of fourteen Gemini models exposes one.
Four OpenAI checkpoints do. So the same twelve wordings, the same twelve name
pairs, the same three strength levels and the SAME OUTCOME can be run on
frontier models, and the comparison is exact rather than analogical.

WHAT IS COMPUTED, and it is the same list Study 2 computes:

  * the demographic effect as a probability of superiority over matched pairs,
    with a cluster bootstrap on NAME PAIRS (§6.1's unit, not the row);
  * the between-wording standard deviation of that effect;
  * their ratio, reported only where the effect's own interval excludes zero;
  * the same split by arm -- six semantic paraphrases against six edits that
    change no word -- because the null arm is what makes the dispersion a
    statement about the instrument rather than about meaning;
  * the censoring rate, because this API returns a 20-token window where the
    local stack returns 100, and a margin that falls outside it is missing for
    a reason correlated with its own size.

WHAT THIS CANNOT SETTLE. Serving-side choices -- quantization, batching, cache
residency -- are not manipulable behind an API, so §5's results have no
frontier counterpart and this arm is silent about them. The checkpoints are
also aliases, not pinned weights: "gpt-4o" names whatever the vendor currently
serves, which is the pinning failure Table 15 records against the field and
which we cannot escape either. Both are stated in §10.2 rather than implied.

    C:/research-toolchain/venv/Scripts/python.exe \
        paper-a/src/analyze_frontier_margin.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "paper-a" / "data" / "frontier"
OUT = D / "frontier_margin_analysis.json"

ORDER = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
N_BOOT = 20000
SEED = 20260801


def superiority(w, b):
    """1 / 0.5 / 0 exactly as effectsize.superiority scores a pair."""
    return 1.0 if w > b else (0.0 if w < b else 0.5)


def load(model: str):
    f = D / f"margin_{model}.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def boot_ps(by_pair, rng, n_boot=N_BOOT):
    """Cluster bootstrap on name pairs. The pair is the unit, per §6.1."""
    keys = list(by_pair)
    per = np.array([np.mean(by_pair[k]) for k in keys])
    est = float(per.mean())
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = per[rng.integers(0, len(keys), len(keys))].mean()
    return est, [float(np.percentile(draws, 2.5)),
                 float(np.percentile(draws, 97.5))]


def main() -> int:
    rng = np.random.default_rng(SEED)
    out = {
        "_what": "Study 2's wording design, outcome and estimand, on frontier "
                 "API models that return log probabilities.",
        "_outcome": "probability of superiority over matched pairs, from the "
                    "renormalised margin log P(yes) - log P(no)",
        "_resampling_unit": "name pair",
        "_window": "top-20 (the API cap); the local panel uses top-100",
        "n_boot": N_BOOT, "seed": SEED, "models": {},
    }
    print(f"{'model':<15}{'cells':>7}{'cens':>7}{'P(sup)':>9}{'95% CI':>20}"
          f"{'SD word':>9}{'ratio':>8}   arms (sem / null)")
    print("-" * 104)
    for m in ORDER:
        rows = load(m)
        if not rows:
            continue
        usable = [r for r in rows
                  if r.get("white_margin") is not None
                  and r.get("black_margin") is not None and not r.get("error")]
        n_cens = len(rows) - len(usable)

        # WHY A CELL IS MISSING MATTERS MORE THAN HOW MANY ARE. There are two
        # mechanisms and they mean opposite things:
        #
        #   SATURATION. The model puts essentially all mass on one answer, so
        #   the other answer is not in the returned window at all and its
        #   probability reads as exactly zero. The margin is not unknown -- it
        #   is enormous, and bounded below by the window's own resolution. On
        #   gpt-4.1 this is 431 of 432 cells: the model is so confident on this
        #   task that the outcome cannot be resolved at any window this API
        #   offers. That is a fact about auditing frontier models, not a defect
        #   of the run, and it is reported as one.
        #
        #   ORDINARY CENSORING. The losing option is present but below the
        #   window edge for a reason less extreme. Rarer, and the one §4.6
        #   treats for the second task.
        #
        # They are separated by whether the emitted verdict agrees on both arms
        # AND the missing side reads exactly zero.
        def saturated(r):
            for arm in ("white", "black"):
                if r.get(f"{arm}_margin") is None:
                    py, pn = r.get(f"{arm}_p_yes"), r.get(f"{arm}_p_no")
                    if py is not None and pn is not None and min(py, pn) == 0.0 \
                            and max(py, pn) > 0.99:
                        return True
            return False

        n_sat = sum(1 for r in rows
                    if (r.get("white_margin") is None
                        or r.get("black_margin") is None) and saturated(r))
        # The race breakdown of the censoring is recorded on BOTH branches.
        # The unmeasurable early return used to omit it, so the integrity
        # audit's claim that differential attrition was "measured and
        # reported" was false for exactly the models this branch covers.
        _wl_early = sum(1 for r in rows if r.get("white_margin") is None
                        and r.get("black_margin") is not None)
        _bl_early = sum(1 for r in rows if r.get("black_margin") is None
                        and r.get("white_margin") is not None)
        if len(usable) < 24:
            out["models"][m] = dict(
                n_design_cells=len(rows), n_usable=len(usable),
                n_censored=n_cens, frac_censored=n_cens / len(rows),
                n_saturated=n_sat, frac_saturated=n_sat / len(rows),
                censored_white_only=_wl_early, censored_black_only=_bl_early,
                unmeasurable=True,
                why="the model places essentially all probability mass on one "
                    "answer, so the other is outside the top-20 window the API "
                    "returns and the margin cannot be formed; this is model "
                    "confidence, not a sampling or budget limit")
            print(f"{m:<15}{len(usable):>7}{n_cens:>7}   OUTCOME UNMEASURABLE: "
                  f"{n_sat}/{len(rows)} cells saturated")
            continue

        by_pair = collections.defaultdict(list)
        by_variant = collections.defaultdict(lambda: collections.defaultdict(list))
        by_arm = collections.defaultdict(lambda: collections.defaultdict(
            lambda: collections.defaultdict(list)))
        for r in usable:
            s = superiority(r["white_margin"], r["black_margin"])
            by_pair[r["pair"]].append(s)
            by_variant[r["variant"]][r["pair"]].append(s)
            by_arm[r["variant_kind"]][r["variant"]][r["pair"]].append(s)

        est, ci = boot_ps(by_pair, rng)
        identified = (ci[0] - 0.5) * (ci[1] - 0.5) > 0
        per_variant = {v: float(np.mean([np.mean(x) for x in d.values()]))
                       for v, d in by_variant.items()}
        sd_word = float(np.std(list(per_variant.values()), ddof=1))
        ratio = sd_word / abs(est - 0.5) if abs(est - 0.5) > 1e-9 else None

        arms = {}
        for kind, d in by_arm.items():
            vals = [float(np.mean([np.mean(x) for x in pd.values()]))
                    for pd in d.values()]
            if len(vals) >= 2:
                arms[kind] = dict(n_wordings=len(vals),
                                  sd=float(np.std(vals, ddof=1)),
                                  mean=float(np.mean(vals)))

        # Censoring, tested for differential loss by arm. A margin is censored
        # when one of yes/no falls outside the returned window; that happens
        # where the distribution is most concentrated, so the loss is not at
        # random and a paired estimate can be biased by it.
        wl = sum(1 for r in rows if r.get("white_margin") is None
                 and r.get("black_margin") is not None)
        bl = sum(1 for r in rows if r.get("black_margin") is None
                 and r.get("white_margin") is not None)

        # CENSORING BY STRENGTH LEVEL, because on one model it is not scattered
        # -- gpt-4.1-mini loses every one of the 144 marginal-résumé cells and
        # none of the others. An estimate computed over what survives is then
        # an estimate on a DIFFERENT DESIGN: two strength levels, not three.
        # Reporting only an overall censoring fraction would hide that, and the
        # paper's own §4.6 makes the same point about the second task.
        tpl = collections.Counter(r["template"] for r in rows)
        tpl_lost = collections.Counter(
            r["template"] for r in rows
            if r.get("white_margin") is None or r.get("black_margin") is None)
        by_template = {t: dict(n=tpl[t], lost=tpl_lost.get(t, 0),
                               frac_lost=tpl_lost.get(t, 0) / tpl[t])
                       for t in sorted(tpl)}
        wiped = [t for t, v in by_template.items() if v["frac_lost"] == 1.0]
        out["models"][m] = dict(
            n_design_cells=len(rows), n_usable=len(usable),
            n_censored=n_cens, frac_censored=n_cens / len(rows),
            n_saturated=n_sat, frac_saturated=n_sat / len(rows),
            unmeasurable=False,
            censored_white_only=wl, censored_black_only=bl,
            censoring_by_template=by_template,
            strength_levels_wholly_lost=wiped,
            design_intact=not wiped,

            n_wordings=len(per_variant), n_pairs=len(by_pair),
            superiority=dict(est=est, ci=ci), effect_identified=bool(identified),
            per_variant=per_variant, sd_across_wordings=sd_word,
            ratio_sd_to_effect=ratio if identified else None,
            ratio_suppressed_because_effect_covers_zero=not identified,
            arms=arms)
        _a = "  ".join(f"{k}={v['sd']:.4f}" for k, v in sorted(arms.items()))
        _r = f"{ratio:.2f}x" if (ratio and identified) else "—"
        if wiped:
            _r += "!"

        print(f"{m:<15}{len(usable):>7}{n_cens:>7}{est:>9.4f}"
              f"{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>20}{sd_word:>9.4f}{_r:>8}   {_a}")

    mods = out["models"]
    live = {m: v for m, v in mods.items() if not v.get("unmeasurable")}
    dead = [m for m, v in mods.items() if v.get("unmeasurable")]
    ids = [m for m, v in live.items() if v["effect_identified"]]
    ratios = [v["ratio_sd_to_effect"] for v in live.values()
              if v["ratio_sd_to_effect"]]
    nulls = [v["arms"]["null"]["sd"] for v in live.values()
             if "null" in v.get("arms", {})]
    sems = [v["arms"]["semantic"]["sd"] for v in live.values()
            if "semantic" in v.get("arms", {})]
    out["summary"] = dict(
        n_models=len(mods), n_measurable=len(live),
        models_outcome_unmeasurable=dead, n_identified=len(ids), identified_models=ids,
        sd_min=min((v["sd_across_wordings"] for v in live.values()), default=None),
        sd_max=max((v["sd_across_wordings"] for v in live.values()), default=None),
        ratio_min=min(ratios) if ratios else None,
        ratio_max=max(ratios) if ratios else None,
        ratio_basis="models whose effect interval excludes 0.5",
        null_arm_sd_min=min(nulls) if nulls else None,
        null_arm_sd_max=max(nulls) if nulls else None,
        n_models_null_arm_at_least_semantic=sum(
            1 for n, s in zip(nulls, sems) if n >= s),
        total_censored=sum(v["n_censored"] for v in mods.values()),
        total_saturated=sum(v.get("n_saturated", 0) for v in mods.values()),
        models_with_a_strength_level_wholly_lost=[
            m for m, v in mods.items() if v.get("strength_levels_wholly_lost")],
    )
    s = out["summary"]
    print()
    print(f"  outcome measurable on {s['n_measurable']} of {s['n_models']}; "
          f"effect identified on {s['n_identified']}")
    print(f"  between-wording SD {s['sd_min']:.4f} to {s['sd_max']:.4f}")
    if s["ratio_min"]:
        print(f"  dispersion / effect {s['ratio_min']:.2f}x to "
              f"{s['ratio_max']:.2f}x (identified models only)")
    print(f"  null-arm SD {s['null_arm_sd_min']:.4f} to "
          f"{s['null_arm_sd_max']:.4f}; the null arm is at least as large as "
          f"the semantic arm on {s['n_models_null_arm_at_least_semantic']} of "
          f"{s['n_models']}")
    print(f"  censored cells {s['total_censored']} in total")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
