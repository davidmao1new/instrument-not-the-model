"""How well is the between-wording dispersion itself estimated?

THE OBJECTION. Table 3 reports a standard deviation across twelve wordings and a
ratio of that SD to the effect, both as point estimates. A standard deviation
computed on twelve observations has eleven degrees of freedom and a wide
sampling distribution; a paper whose thesis is that others should report the
dispersion of their estimate is in a poor position to report its own dispersion
without one. A reviewer made exactly this point and it is correct.

WHAT IS COMPUTED. Per model, a bootstrap interval on

  * the between-wording SD of the per-wording effect, resampling WORDINGS,
    which is the unit that generates that dispersion;
  * the ratio of that SD to the model's own effect;
  * for reference, the chi-square interval a normal-theory calculation gives,
    so the reader can see the bootstrap is not doing anything exotic.

TWO SOURCES OF UNCERTAINTY, AND THIS ADDRESSES ONE. Resampling wordings
captures the fact that twelve wordings are a sample from the space of
defensible wordings. It does NOT capture the sampling noise inside each
per-wording estimate; §6.1's partial-pooling model handles that separately and
its pooled figure is the one net of it. Reporting both is the point: the raw SD
is inflated by within-wording noise and is itself imprecisely estimated, and a
reader deserves to know both rather than neither.

    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/analyze_dispersion_uncertainty.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DS = ROOT / "paper-a" / "data" / "delta_stability"
OUT = DS / "dispersion_uncertainty.json"

N_BOOT = 20_000
SEED = 20260801
ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SHORT = {"llama-2-7b-chat": "Llama-2-7B", "llama-3.1-8b-instruct": "Llama-3.1-8B",
         "mistral-7b-instruct-v0.1": "Mistral v0.1",
         "mistral-7b-instruct-v0.3": "Mistral v0.3"}


def main() -> int:
    s2 = json.loads((DS / "study2_v2.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    out = {"n_boot": N_BOOT, "seed": SEED,
           "_unit": "wordings are resampled; within-wording sampling noise is "
                    "NOT included here and is handled by the partial-pooling "
                    "model of section 6.1",
           "models": {}}

    print("=" * 96)
    print("HOW PRECISELY IS THE DISPERSION ITSELF ESTIMATED?")
    print("=" * 96)
    print(f"{'model':<15}{'k':>4}{'SD(ps)':>9}{'95% CI on SD':>22}"
          f"{'rel. width':>12}{'SD/effect':>11}{'95% CI on ratio':>22}")

    for m in ORDER:
        if m not in s2:
            continue
        pv = s2[m]["per_variant"]
        ps = np.array([pv[v]["superiority"]["est"] for v in sorted(pv)])
        k = len(ps)
        sd = float(ps.std(ddof=1))
        eff = abs(s2[m]["overall"]["superiority"]["est"] - 0.5)

        boots = np.empty(N_BOOT)
        ratios = np.empty(N_BOOT)
        for b in range(N_BOOT):
            s = ps[rng.integers(0, k, k)]
            boots[b] = s.std(ddof=1)
            ratios[b] = boots[b] / eff if eff > 1e-9 else np.nan
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rl, rh = (np.nanpercentile(ratios, [2.5, 97.5])
                  if eff > 1e-9 else (np.nan, np.nan))

        # normal-theory reference: (k-1)s^2/sigma^2 ~ chi^2_{k-1}
        chi_lo = sd * np.sqrt((k - 1) / stats.chi2.ppf(0.975, k - 1))
        chi_hi = sd * np.sqrt((k - 1) / stats.chi2.ppf(0.025, k - 1))

        out["models"][m] = dict(
            n_wordings=k, sd_ps=sd, sd_ci=[float(lo), float(hi)],
            sd_ci_relative_width=float((hi - lo) / sd),
            chi2_ci=[float(chi_lo), float(chi_hi)],
            effect_ps=eff,
            ratio=float(sd / eff) if eff > 1e-9 else None,
            ratio_ci=[float(rl), float(rh)] if eff > 1e-9 else None,
            effect_identified=bool(
                s2[m]["overall"]["logodds"]["ci"][0]
                * s2[m]["overall"]["logodds"]["ci"][1] > 0),
        )
        print(f"{SHORT.get(m, m):<15}{k:>4}{sd:>9.4f}"
              f"{f'[{lo:.4f}, {hi:.4f}]':>22}"
              f"{(hi - lo) / sd:>12.2f}"
              f"{sd / eff:>11.2f}"
              f"{f'[{rl:.2f}, {rh:.2f}]':>22}")

    ws = [v["sd_ci_relative_width"] for v in out["models"].values()]
    out["summary"] = dict(
        min_relative_width=float(min(ws)), max_relative_width=float(max(ws)),
        note="relative width = (upper - lower) / point estimate")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  the interval on the SD is {min(ws):.0%} to {max(ws):.0%} of the "
          "point estimate itself")
    print("  READING. Twelve wordings pin the dispersion to roughly a factor of")
    print("  two. That is enough to say the dispersion is large relative to the")
    print("  effect and not enough to rank two models by it.")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
