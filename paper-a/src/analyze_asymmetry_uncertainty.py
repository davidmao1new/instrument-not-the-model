"""How firmly is the name-versus-wording crossover established?

WHY THIS EXISTS. §4.2 reports that the name-draw component averages down with
list size while the wording component does not, and that by k = 9 the wording
has become the larger of the two on some of the panel. The count of models on
which that has happened is read off Table 6, whose cells are POINT ESTIMATES:
each is a ratio of two posterior medians. A critique of this paper pointed out
that the paper argues against itself here -- §4.1 says twelve wordings pin the
dispersion only "to about a factor of two" and that this "is not enough to rank
two models by it, which the paper does not do" -- and then Table 6 ranks all
four against a threshold at 1.0 with no interval anywhere.

WHAT THIS COMPUTES. The ratio is

    ratio(k) = sigma_first * sqrt(2) / (sqrt(k) * sigma_variant)

-- the sqrt(2) because a matched PAIR draws two names, and the sqrt(k) because
a study averaging k pairs inherits the name component divided by sqrt(k) while
inheriting the whole wording component. Both sigmas are posterior quantities
with wide credible intervals, so the ratio has one too, and the question "is
the crossing at 1.0 established?" is a question about that interval.

HOW THE INTERVAL IS FORMED, AND ITS LIMIT. Only the 2.5 / 50 / 97.5 quantiles
of each sigma were retained, not the joint posterior draws, so this takes the
corners: the ratio is smallest when sigma_first is at its low end and
sigma_variant at its high end, and largest the other way round. That is a
CONSERVATIVE bound. It ignores any posterior correlation between the two, and
because they are estimated from different fits on the same data any real
correlation would narrow it. The bound is reported as a bound and the paper
says so; the alternative, quoting a point estimate with no interval, is what
this replaces.

    C:/research-toolchain/venv/Scripts/python.exe \
        paper-a/src/analyze_asymmetry_uncertainty.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "paper-a" / "data"
OUT = D / "names" / "asymmetry_uncertainty.json"

ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SHORT = {"llama-2-7b-chat": "Llama-2-7B", "llama-3.1-8b-instruct": "Llama-3.1-8B",
         "mistral-7b-instruct-v0.1": "Mistral v0.1",
         "mistral-7b-instruct-v0.3": "Mistral v0.3"}
KS = (3, 6, 9)


def main() -> int:
    nv = json.loads((D / "names" / "name_variance.json").read_text(encoding="utf-8"))
    vc = json.loads((D / "delta_stability" / "variance_components.json")
                    .read_text(encoding="utf-8"))

    out = {
        "_what": "Credible bounds on the name-to-wording ratio of Table 6, "
                 "propagating the posterior intervals of both sigmas.",
        "_formula": "ratio(k) = sigma_first * sqrt(2) / (sqrt(k) * sigma_variant)",
        "_bound_type": "corner bound from marginal 2.5/50/97.5 quantiles; "
                       "ignores posterior correlation between the two sigmas "
                       "and is therefore conservative (wider than the truth)",
        "ks": list(KS), "models": {},
    }
    print(f"{'model':<15}{'k':>4}{'ratio':>9}{'95% bound':>22}   crossing at 1.0")
    print("-" * 82)
    for m in ORDER:
        if m not in nv or m not in vc:
            continue
        sf = nv[m]["sigma_first"]          # [lo, med, hi]
        sv = vc[m]["all"]["sigma_variant"]  # [lo, med, hi]
        rec = {}
        for k in KS:
            f = math.sqrt(2) / math.sqrt(k)
            est = sf[1] * f / sv[1]
            lo = sf[0] * f / sv[2]
            hi = sf[2] * f / sv[0]
            # Does the bound settle which side of 1.0 the ratio is on?
            side = ("name larger" if lo > 1 else
                    "wording larger" if hi < 1 else "not determined")
            rec[f"k_{k}"] = dict(est=est, bound=[lo, hi],
                                 point_says_wording_larger=bool(est < 1.0),
                                 bound_determines_side=side != "not determined",
                                 side=side)
            if k == 9:
                print(f"{SHORT[m]:<15}{k:>4}{est:>9.2f}"
                      f"{f'[{lo:.2f}, {hi:.2f}]':>22}   {side}")
        out["models"][m] = dict(sigma_first=sf, sigma_variant=sv, **rec)

    mods = out["models"]
    per_k = {}
    for k in KS:
        key = f"k_{k}"
        pt = sum(1 for v in mods.values() if v[key]["point_says_wording_larger"])
        det = sum(1 for v in mods.values() if v[key]["bound_determines_side"])
        det_word = sum(1 for v in mods.values()
                       if v[key]["side"] == "wording larger")
        per_k[key] = dict(
            n_models=len(mods),
            n_wording_larger_at_point_estimate=pt,
            n_models_where_the_bound_determines_the_side=det,
            n_wording_larger_and_determined=det_word)
    out["summary"] = dict(
        per_k=per_k,
        crossover_established=bool(
            per_k["k_9"]["n_models_where_the_bound_determines_the_side"]),
        note="At the point estimate the wording is the larger source on "
             f"{per_k['k_9']['n_wording_larger_at_point_estimate']} of "
             f"{len(mods)} models at k = 9. Propagating both posteriors, the "
             "side of 1.0 is determined on "
             f"{per_k['k_9']['n_models_where_the_bound_determines_the_side']} "
             "of them. The MONOTONE claim -- that the ratio falls with k, "
             "because only the numerator carries the 1/sqrt(k) -- holds "
             "regardless, since it does not depend on either sigma's value.")
    print()
    for k in KS:
        v = per_k[f"k_{k}"]
        print(f"  k = {k}: wording larger at the point estimate on "
              f"{v['n_wording_larger_at_point_estimate']} of {v['n_models']}; "
              f"side determined by the bound on "
              f"{v['n_models_where_the_bound_determines_the_side']}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
