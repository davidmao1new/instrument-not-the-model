"""Do the two token-length designs agree on a MAGNITUDE, not just a direction?

THE PROBLEM THIS ANSWERS. §4.4 rests on two estimators of the same channel:

  * Table 8, a regression of the per-pair effect on the pair's token-length
    difference, which is distinguishable from zero on one model; and
  * Table 9, the effect recomputed on the token-matched subset, whose shift is
    distinguishable from zero on none.

They are significant on DIFFERENT models -- the slope on Mistral v0.3, the
subset shift (marginally, and only by one of three procedures) on Llama-3.1-8B
-- and a reviewer reading the two tables side by side will notice that the
section claims they agree. On point estimates they do; on verdicts they do not.
"Two weak results pointing the same way" is a much weaker object than the
section needs.

WHAT MAKES THEM ONE RESULT. The two are quantitatively linked, and the link is
a prediction rather than a restatement. If a length channel with slope b exists,
then dropping the pairs whose length difference is non-zero should move the
effect by approximately

    predicted shift  =  -b * (mean length difference over the DROPPED pairs)

because those pairs carry, on average, that much length signal and the matched
ones carry none. The slope is fitted on all 48 pairs; the shift is measured on a
12-pair subset; neither is used to compute the other. So comparing them is a
falsifiable check on the mechanism, of exactly the kind §6.2 runs when it tests
the realised Jacobian error against the predicted one.

WHAT WOULD FALSIFY IT. A predicted shift of the wrong sign, or off by an order
of magnitude, on a model where the slope is well determined. That is reported as
readily as agreement; the ratio observed/predicted is printed per model with no
threshold applied to it.

    C:/research-toolchain/venv/Scripts/python.exe \\
        paper-a/src/analyze_length_prediction.py
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
OUT = D / "instrument" / "length_prediction.json"

ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SHORT = {"llama-2-7b-chat": "Llama-2-7B", "llama-3.1-8b-instruct": "Llama-3.1-8B",
         "mistral-7b-instruct-v0.1": "Mistral v0.1",
         "mistral-7b-instruct-v0.3": "Mistral v0.3"}


def per_pair_effect(model: str):
    """Mean (white - black) margin for each of the 48 full-name pairs."""
    by = collections.defaultdict(list)
    for f in sorted((D / "names").glob("names_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("model") != model or r.get("error"):
                continue
            if r.get("white_margin") is None or r.get("black_margin") is None:
                continue
            by[(r["white_name"], r["black_name"])].append(
                r["white_margin"] - r["black_margin"])
    return {k: float(np.mean(v)) for k, v in by.items()}


def deltas(model: str):
    """Token-length difference per full-name pair, from the length artifact."""
    f = D / "instrument" / f"name_length_{model}.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text(encoding="utf-8"))
    rows = d.get("pairs") or d.get("rows") or []
    out = {}
    for r in rows:
        k = (r.get("white_name") or r.get("white"),
             r.get("black_name") or r.get("black"))
        v = r.get("delta_in_context", r.get("delta"))
        if k[0] and k[1] and v is not None:
            out[k] = float(v)
    return out


def main() -> int:
    nlen = json.loads((D / "instrument" / "name_length_effect.json")
                      .read_text(encoding="utf-8"))
    out = {
        "_what": "Does the token-matched subset shift agree with the shift the "
                 "Table 8 slope predicts?",
        "_prediction": "shift ~= -slope * mean(delta over ALL pairs), because "
                       "the observed contrast is matched-minus-ALL and the "
                       "matched arm contributes delta = 0; the slope is fitted "
                       "on all 48 pairs and the shift is measured on the "
                       "matched subset, so neither is used to compute the "
                       "other. An earlier version averaged delta over the "
                       "DROPPED pairs, which inflates the prediction by "
                       "n_all / n_dropped and flattered the agreement.",
        "models": {},
    }
    print(f"{'model':<15}{'slope':>9}{'mean d':>9}{'predicted':>11}"
          f"{'observed':>10}{'obs/pred':>10}  same sign")
    print("-" * 78)
    for m in ORDER:
        if m not in nlen:
            continue
        cl = nlen[m]["token_matched_first_name_clustered"]
        slope = cl["slope_same_unit"]["est"]
        eff = per_pair_effect(m)
        dl = deltas(m)
        common = [k for k in eff if k in dl]
        if not common:
            continue
        dropped = [k for k in common if dl[k] != 0]
        if not dropped:
            continue
        # THE PREDICTION MUST MATCH THE CONTRAST IT PREDICTS.
        # `observed` is matched_minus_all: mean over the token-matched subset
        # minus mean over ALL pairs. Under a linear length effect with slope b,
        #   mean(matched) = mu + b * 0            (matched pairs have delta 0)
        #   mean(all)     = mu + b * mean(delta over ALL pairs)
        # so the predicted shift is -b * mean(delta over ALL pairs). Averaging
        # delta over the DROPPED pairs instead inflates it by n_all/n_dropped
        # -- a factor of 1.3 to 1.5 here -- because it discards the zeros that
        # the "all" arm of the contrast actually contains. The correction moves
        # the observed-over-predicted ratios away from 1, i.e. against the
        # agreement this analysis was built to look for.
        mean_d = float(np.mean([dl[k] for k in common]))
        mean_d_dropped = float(np.mean([dl[k] for k in dropped]))
        predicted = -slope * mean_d
        observed = cl["matched_minus_all"]["est"]
        ratio = observed / predicted if abs(predicted) > 1e-12 else None
        same = bool(predicted * observed > 0)
        out["models"][m] = dict(
            slope=slope, n_pairs=len(common), n_dropped=len(dropped),
            mean_delta_over_all=mean_d,
            mean_delta_over_dropped=mean_d_dropped,
            predicted_shift=predicted, observed_shift=observed,
            ratio_observed_to_predicted=ratio, same_sign=same,
            slope_p=cl["slope_same_unit"]["p"],
            subset_p=cl["matched_minus_all"]["p"])
        print(f"{SHORT[m]:<15}{slope:>+9.4f}{mean_d:>9.2f}{predicted:>+11.4f}"
              f"{observed:>+10.4f}"
              + (f"{ratio:>10.2f}" if ratio is not None else f"{'-':>10}")
              + f"  {'yes' if same else 'NO'}")

    mm = out["models"]
    n_same = sum(1 for v in mm.values() if v["same_sign"])
    ratios = [v["ratio_observed_to_predicted"] for v in mm.values()
              if v["ratio_observed_to_predicted"] is not None]
    within = [r for r in ratios if 0.2 <= abs(r) <= 5.0]
    out["summary"] = dict(
        n_models=len(mm), n_same_sign=n_same,
        n_within_a_factor_of_five=len(within),
        ratio_min=min(ratios) if ratios else None,
        ratio_max=max(ratios) if ratios else None,
        verdict=("the two designs agree on magnitude, not only on sign"
                 if n_same == len(mm) and len(within) == len(mm)
                 else "the two designs do not agree on every model"))
    s = out["summary"]
    print()
    print(f"  same sign on {n_same} of {s['n_models']}; within a factor of "
          f"five on {s['n_within_a_factor_of_five']}")
    print(f"  observed/predicted from {s['ratio_min']:.2f} to "
          f"{s['ratio_max']:.2f}")
    print(f"  VERDICT: {s['verdict']}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
