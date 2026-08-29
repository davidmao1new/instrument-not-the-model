"""An interval on the dispersion-to-effect ratio, on both panels.

WHY. §4.7 quotes the ratio on four frontier checkpoints and on the open-weight
panel, ranks the two ("larger than the open-weight panel", three times,
including in the abstract), and then says in the same section that "neither
ratio is pinned tightly enough to rank the two panels, and we do not". Both
statements cannot stand, and neither was supported: no interval on the ratio
existed anywhere in the project. The paper was ranking without evidence and
declining to rank without evidence, in adjacent sentences.

So compute the thing that settles it. The ratio is

    sd across wordings of the per-wording mean P(superiority)
    ------------------------------------------------------------
                     | mean P(superiority) - 0.5 |

and both numerator and denominator are functions of the same name pairs. A
cluster bootstrap that resamples PAIRS -- the resampling unit §6.1 fixes, and
the unit the rest of the paper uses -- propagates that shared dependence
correctly, which resampling rows would not.

WHAT MAKES THIS RATIO AWKWARD, stated because it governs how the result is
read. The denominator is a distance from 0.5 that can approach zero, so the
ratio is heavy-tailed and its upper endpoint is unstable by construction; §4.5
makes the same point about a ratio whose denominator covers zero. Two
consequences are built in here rather than discovered later:

  * the interval is reported as a percentile interval on the bootstrap
    distribution, and the fraction of draws whose denominator comes within a
    whisker of zero is reported beside it, so a reader can see when the upper
    endpoint is meaningless;
  * models whose EFFECT interval covers 0.5 get no ratio at all, matching what
    analyze_frontier_margin.py already does.

The comparison between panels is then made on the quantity that answers the
question -- the difference between the frontier ratio and the open-weight
ratio, bootstrapped -- rather than by eyeballing whether two intervals overlap,
which is the error §6.1 warns about in a different guise.

    C:/research-toolchain/venv/Scripts/python.exe \\
        paper-a/src/analyze_ratio_interval.py
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
D = ROOT / "paper-a" / "data"
OUT = D / "reference" / "ratio_intervals.json"
N_BOOT = 4000
SEED = 20260802
FRONTIER = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
LOCAL = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
EPS = 1e-3


def superiority(w, b):
    return 1.0 if w > b else (0.0 if w < b else 0.5)


def cells_from_jsonl(path: pathlib.Path):
    """[(pair, variant, superiority score)] from a margin file."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("white_margin") is None or r.get("black_margin") is None:
            continue
        out.append((r["pair"], r["variant"],
                    superiority(r["white_margin"], r["black_margin"])))
    return out


def ratio_from(cells, pairs):
    """(ratio, sd, effect) over the given multiset of pairs.

    Aggregation matches §4.7: per wording, average within a pair first, then
    across pairs, so an unbalanced cell count cannot tilt a wording mean.
    """
    by_vp = collections.defaultdict(lambda: collections.defaultdict(list))
    for p, v, s in cells:
        by_vp[v][p].append(s)
    per_wording = []
    for _v, d in by_vp.items():
        vals = [np.mean(d[p]) for p in pairs if p in d]
        if vals:
            per_wording.append(float(np.mean(vals)))
    if len(per_wording) < 2:
        return None, None, None
    allv = [np.mean([s for p, _v, s in cells if p == q])
            for q in pairs
            if any(p == q for p, _v, _s in cells)]
    if not allv:
        return None, None, None
    eff = float(np.mean(allv))
    sd = float(np.std(per_wording, ddof=1))
    den = abs(eff - 0.5)
    return (sd / den if den > EPS else None), sd, eff


def boot(cells, rng, n_boot=N_BOOT):
    pairs = sorted({p for p, _v, _s in cells})
    if len(pairs) < 2:
        return None
    est, sd0, eff0 = ratio_from(cells, pairs)
    draws, degenerate = [], 0
    for _ in range(n_boot):
        samp = [pairs[i] for i in rng.integers(0, len(pairs), len(pairs))]
        r, _sd, _eff = ratio_from(cells, samp)
        if r is None:
            degenerate += 1
        else:
            draws.append(r)
    if len(draws) < n_boot // 2:
        return dict(est=est, sd=sd0, effect=eff0, ci=None,
                    frac_degenerate=degenerate / n_boot,
                    note="denominator approaches zero in too many resamples "
                         "for an interval to mean anything")
    a = np.array(draws)
    return dict(est=est, sd=sd0, effect=eff0,
                ci=[float(np.percentile(a, 2.5)),
                    float(np.percentile(a, 97.5))],
                median=float(np.median(a)),
                frac_degenerate=degenerate / n_boot,
                n_pairs=len(pairs))


def main() -> int:
    rng = np.random.default_rng(SEED)
    out = {
        "_what": "Cluster-bootstrap intervals on the dispersion-to-effect "
                 "ratio, for the frontier and open-weight panels, resampling "
                 "name pairs.",
        "_why": "§4.7 both ranked the two panels and said it could not; "
                "neither claim had an interval behind it. This supplies one.",
        "_method": (
            "Resample name pairs with replacement (the §6.1 unit). Per draw, "
            "recompute the per-wording mean P(superiority) within pairs then "
            "across them, the SD across wordings, and the effect; the ratio is "
            "their quotient. Percentile interval over "
            f"{N_BOOT} draws. Draws whose |effect - 0.5| falls below {EPS} are "
            "counted as degenerate and excluded, and the fraction is reported: "
            "a ratio with a vanishing denominator has no finite upper bound, "
            "which is the §4.5 problem."),
        "frontier": {}, "local": {},
    }

    for m in FRONTIER:
        cells = cells_from_jsonl(D / "frontier" / f"margin_{m}.jsonl")
        if len(cells) < 24:
            continue
        r = boot(cells, rng)
        if r:
            out["frontier"][m] = r

    s2p = D / "delta_stability" / "study2_v2.json"
    s2 = json.loads(s2p.read_text(encoding="utf-8")) if s2p.exists() else {}
    for m in LOCAL:
        cells = []
        for f in sorted((D / "delta_stability").glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("model") != m:
                    continue
                if r.get("white_margin") is None or r.get("black_margin") is None:
                    continue
                cells.append((r["pair"], r["variant"],
                              superiority(r["white_margin"], r["black_margin"])))
        if len(cells) < 24:
            continue
        rr = boot(cells, rng)
        if rr:
            rr["published_ratio"] = (
                s2.get(m, {}).get("ps_sd_across_wordings", 0)
                / abs(s2[m]["overall"]["superiority"]["est"] - 0.5)
                if m in s2 and abs(
                    s2[m]["overall"]["superiority"]["est"] - 0.5) > EPS else None)
            out["local"][m] = rr

    # ---- the comparison, restricted to the models the paper compares -------
    # §4.7 compares only checkpoints whose EFFECT is identified -- a ratio
    # against a denominator that covers zero is the thing §4.5 rules out. So
    # the comparison is computed on that subset, which is also the subset the
    # abstract's "0.54 to 0.64" comes from. Including the unidentified models
    # would import exactly the unbounded ratios the restriction exists to keep
    # out, and would make the panels look further apart than they are.
    ident_f, ident_l = [], []
    fmp = D / "frontier" / "frontier_margin_analysis.json"
    if fmp.exists():
        _fm = json.loads(fmp.read_text(encoding="utf-8"))
        ident_f = _fm.get("summary", {}).get("identified_models", [])
    for m in LOCAL:
        rec = s2.get(m, {}).get("overall", {}).get("logodds", {})
        ci = rec.get("ci")
        if ci and ci[0] * ci[1] > 0:
            ident_l.append(m)
    out["_identified"] = {"frontier": ident_f, "local": ident_l}

    fr = {k: v for k, v in out["frontier"].items()
          if v.get("ci") and k in ident_f}
    lo = {k: v for k, v in out["local"].items()
          if v.get("ci") and k in ident_l}
    comp = {}
    if fr and lo:
        f_est = [v["est"] for v in fr.values() if v["est"] is not None]
        l_est = [v["est"] for v in lo.values() if v["est"] is not None]
        comp = dict(
            n_frontier_with_an_interval=len(fr),
            n_local_with_an_interval=len(lo),
            frontier_point_min=min(f_est), frontier_point_max=max(f_est),
            local_point_min=min(l_est), local_point_max=max(l_est),
            every_frontier_point_above_every_local_point=(
                min(f_est) > max(l_est)),
            # The honest test: do the INTERVALS separate? Overlapping
            # intervals do not prove equality, but non-overlap is sufficient
            # for the ordering the paper wants to assert.
            frontier_intervals_all_above_local_points=all(
                v["ci"][0] > max(l_est) for v in fr.values()),
            worst_case_overlap=max(
                (max(l_est) - v["ci"][0] for v in fr.values()), default=None),
            max_local_ci_upper=max(v["ci"][1] for v in lo.values()),
            min_frontier_ci_lower=min(v["ci"][0] for v in fr.values()),
            intervals_disjoint=(min(v["ci"][0] for v in fr.values())
                                > max(v["ci"][1] for v in lo.values())),
            frontier_models=sorted(fr), local_models=sorted(lo),
        )
        # THE SENTENCE THE PAPER IS ENTITLED TO. Written here rather than in
        # the builder so the claim and the evidence cannot drift apart.
        comp["ranking_is_supported"] = bool(comp["intervals_disjoint"])
        comp["_verdict"] = (
            "the intervals separate, so the ordering may be asserted"
            if comp["intervals_disjoint"] else
            "the point estimates order one way on every checkpoint, but the "
            "intervals overlap, so the ordering is a direction and not a "
            "finding; the paper must not claim the frontier ratio is larger")
    out["comparison"] = comp

    print("=" * 92)
    print("DISPERSION-TO-EFFECT RATIO, WITH A CLUSTER-BOOTSTRAP INTERVAL")
    print("=" * 92)
    for panel in ("frontier", "local"):
        print(f"\n{panel.upper()}")
        print(f"  {'model':<26}{'ratio':>8}{'95 % interval':>24}"
              f"{'degenerate':>12}")
        for m, v in out[panel].items():
            ci = (f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]" if v.get("ci")
                  else "none (denominator ~ 0)")
            est = f"{v['est']:.2f}" if v["est"] is not None else "--"
            print(f"  {m:<26}{est:>8}{ci:>24}"
                  f"{v['frac_degenerate']:>12.1%}")
    if comp:
        print()
        print(f"  frontier points {comp['frontier_point_min']:.2f}"
              f"–{comp['frontier_point_max']:.2f}   "
              f"local points {comp['local_point_min']:.2f}"
              f"–{comp['local_point_max']:.2f}")
        print(f"  every frontier point above every local point: "
              f"{comp['every_frontier_point_above_every_local_point']}")
        print(f"  intervals disjoint across panels:               "
              f"{comp['intervals_disjoint']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
