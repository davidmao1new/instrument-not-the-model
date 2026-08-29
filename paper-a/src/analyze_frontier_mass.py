"""How much of each frontier model's first token is a verdict at all?

TWO QUESTIONS THIS ANSWERS, both raised against §4.7.

  1. IS gpt-4.1 SATURATED, OR IS IT SAYING SOMETHING ELSE? §4.7 concludes that
     gpt-4.1 places essentially all its mass on one answer, so the other falls
     outside the twenty-token window. A reviewer offered a competing
     explanation that the section had not excluded: with no grammar constraint
     the first token might not be a verdict at all -- a preamble, a JSON
     delimiter, a capitalisation or leading-space variant our matcher missed --
     in which case the window contains neither answer and the diagnosis is a
     normalisation bug, not model confidence.

     The two make opposite predictions about a quantity already on disk. Under
     saturation, p(yes) + p(no) is approximately 1: the mass is on a verdict,
     just all of it on one. Under a token-class mismatch, p(yes) + p(no) is
     near ZERO: the mass is somewhere else entirely.

  2. IS THE FRONTIER COMPARISON CONFOUNDED BY THE GRAMMAR? The open-weight
     panel constrains emission to yes|no; the API cannot. §3.1 reports the
     yes/no mass per open-weight checkpoint precisely because the grammar "is
     doing real work" where that mass is low. The same quantity is computable
     for the API models from the top-20 window, and until it is computed the
     §4.7 comparison varies the model AND the forcing mechanism. This produces
     the missing rows.

WHAT IS MEASURED, per model: the renormalisable mass p(yes) + p(no) summed over
the window, its distribution, and the share of calls where that mass is above
0.99, above 0.5, and below 0.01 -- the last being the signature of the
mismatch explanation.

    C:/research-toolchain/venv/Scripts/python.exe \\
        paper-a/src/analyze_frontier_mass.py
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
OUT = D / "frontier_yes_no_mass.json"
ORDER = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]


def main() -> int:
    out = {
        "_what": "Renormalisable yes/no mass in the first token of each "
                 "frontier call, the same quantity Table 1 reports for the "
                 "open-weight panel.",
        "_why": "It distinguishes saturation (mass near 1, all on one side) "
                "from a token-class mismatch (mass near 0), and it is the "
                "quantity that makes the grammar-free API arm comparable with "
                "the grammar-constrained local one.",
        "models": {},
    }
    print(f"{'model':<15}{'calls':>7}{'mean mass':>11}{'median':>9}"
          f"{'>0.99':>8}{'>0.5':>8}{'<0.01':>8}   verdict")
    print("-" * 98)
    for m in ORDER:
        f = D / f"margin_{m}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        mass, raws, oneside = [], collections.Counter(), 0
        for r in rows:
            for arm in ("white", "black"):
                py, pn = r.get(f"{arm}_p_yes"), r.get(f"{arm}_p_no")
                if py is None or pn is None:
                    continue
                # the API returns renormalised-to-1 probabilities per token, so
                # the sum can exceed 1 by float error; clip for reporting only
                tot = min(py + pn, 1.0)
                mass.append(tot)
                if min(py, pn) == 0.0 and max(py, pn) > 0.5:
                    oneside += 1
                raws[(r.get(f"{arm}_raw") or "")[:12]] += 1
        if not mass:
            continue
        a = np.array(mass)
        rec = dict(
            n_calls=len(a), mean_mass=float(a.mean()),
            median_mass=float(np.median(a)),
            frac_above_0_99=float((a > 0.99).mean()),
            frac_above_0_5=float((a > 0.5).mean()),
            frac_below_0_01=float((a < 0.01).mean()),
            min_mass=float(a.min()),
            n_all_mass_on_one_verdict=oneside,
            frac_all_mass_on_one_verdict=oneside / len(a),
            emitted_first_tokens=dict(raws.most_common(6)))
        # The diagnosis, stated by the data rather than by us.
        if rec["frac_above_0_99"] > 0.95 and rec["frac_all_mass_on_one_verdict"] > 0.5:
            v = "SATURATED (mass on a verdict, all of it on one side)"
        elif rec["frac_below_0_01"] > 0.5:
            v = "TOKEN-CLASS MISMATCH (mass is not on a verdict at all)"
        elif rec["frac_above_0_99"] > 0.95:
            v = "verdict-dominated, both sides present"
        else:
            v = "mixed"
        rec["diagnosis"] = v
        out["models"][m] = rec
        print(f"{m:<15}{len(a):>7}{a.mean():>11.4f}{np.median(a):>9.4f}"
              f"{rec['frac_above_0_99']:>8.3f}{rec['frac_above_0_5']:>8.3f}"
              f"{rec['frac_below_0_01']:>8.3f}   {v}")

    mm = out["models"]
    out["summary"] = dict(
        n_models=len(mm),
        min_mean_mass=min((v["mean_mass"] for v in mm.values()), default=None),
        max_mean_mass=max((v["mean_mass"] for v in mm.values()), default=None),
        n_saturated=sum(1 for v in mm.values() if v["diagnosis"].startswith("SAT")),
        n_token_class_mismatch=sum(
            1 for v in mm.values() if v["diagnosis"].startswith("TOKEN")),
        all_verdict_dominated=all(v["frac_above_0_99"] > 0.95
                                  for v in mm.values()),
    )
    s = out["summary"]
    print()
    print(f"  mean yes/no mass {s['min_mean_mass']:.4f} to "
          f"{s['max_mean_mass']:.4f} across {s['n_models']} models")
    print(f"  token-class mismatch on {s['n_token_class_mismatch']} models; "
          f"saturation on {s['n_saturated']}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
