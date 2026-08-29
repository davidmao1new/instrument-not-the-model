r"""Wild cluster bootstrap-t intervals for the four per-model effects.

WHY. Nearly every interval in the paper is a percentile bootstrap over twelve
name-pair clusters, and with a dozen clusters percentile intervals are known
to under-cover, sometimes badly (Cameron–Miller); the paper approvingly
cites a surveyed audit's cluster-robust wild bootstrap and then never runs
one on its own estimates. A reviewer asked, fairly, for the comparison.

WHAT. For each model, the headline effect (mean over cells of the paired
white-minus-black margin, collapsed to per-pair means so the unit is the name
pair) with three intervals side by side:

  percentile   pair-cluster percentile bootstrap, the paper's estimator
  wild-t       Rademacher wild cluster bootstrap-t on pair-level residuals,
               the estimator the small-cluster literature recommends
  ratio        wild-t width / percentile width

If the ratio sits near one, the paper's intervals are not hiding
under-coverage large enough to matter here; if it is well above one, the
percentile intervals are optimistic and the paper must say so.

    sh paper-a/src/_py.sh paper-a/src/analyze_wild_bootstrap.py

Writes paper-a/data/instrument/wild_bootstrap.json.
"""
import json
import pathlib
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "paper-a" / "data"
OUT = DATA / "instrument" / "wild_bootstrap.json"

MODELS = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
          "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SEED = 20260825
N_BOOT = 20000


def pair_means(model):
    """Per-pair mean of (white - black) margin over the 36 cells it owns."""
    by = {}
    p = DATA / "delta_stability" / f"delta_{model}.jsonl"
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        assert not r["error"]
        by.setdefault(r["pair"], []).append(
            r["white_margin"] - r["black_margin"])
    pairs = sorted(by)
    assert all(len(by[k]) == 36 for k in pairs), "expected 12x3 rows per pair"
    return np.array([np.mean(by[k]) for k in pairs])


def main() -> int:
    rng = np.random.default_rng(SEED)
    results = {}
    print(f"{'model':28s} {'est':>8s} {'percentile CI':>20s} "
          f"{'wild-t CI':>20s} {'width ratio':>11s}")
    for m in MODELS:
        g = pair_means(m)                       # 12 pair-level effects
        G = len(g)
        est = g.mean()
        se = g.std(ddof=1) / np.sqrt(G)

        # Percentile pair bootstrap (the paper's estimator).
        idx = rng.integers(0, G, size=(N_BOOT, G))
        boot_means = g[idx].mean(axis=1)
        p_lo, p_hi = np.percentile(boot_means, [2.5, 97.5])

        # Wild cluster bootstrap-t: impose the null-centred residuals, flip
        # each PAIR's residual with a Rademacher weight, and studentise.
        resid = g - est
        signs = rng.choice([-1.0, 1.0], size=(N_BOOT, G))
        gb = est + signs * resid                # each draw: 12 pair values
        tb = (gb.mean(axis=1) - est) / (gb.std(axis=1, ddof=1) / np.sqrt(G))
        t_lo, t_hi = np.percentile(tb, [2.5, 97.5])
        w_lo, w_hi = est - t_hi * se, est - t_lo * se

        ratio = (w_hi - w_lo) / (p_hi - p_lo)
        results[m] = dict(
            n_pairs=G, est=float(est), se_cluster=float(se),
            ci_percentile=[float(p_lo), float(p_hi)],
            ci_wild_t=[float(w_lo), float(w_hi)],
            width_ratio_wild_over_percentile=float(ratio),
            excl_zero_percentile=bool(p_lo * p_hi > 0),
            excl_zero_wild_t=bool(w_lo * w_hi > 0),
        )
        print(f"{m:28s} {est:+8.4f} [{p_lo:+8.4f},{p_hi:+8.4f}] "
              f"[{w_lo:+8.4f},{w_hi:+8.4f}] {ratio:11.2f}")

    ratios = [r["width_ratio_wild_over_percentile"] for r in results.values()]
    flips = [m for m, r in results.items()
             if r["excl_zero_percentile"] != r["excl_zero_wild_t"]]
    summary = dict(ratio_min=float(min(ratios)), ratio_max=float(max(ratios)),
                   n_verdict_changes=len(flips), verdict_changes=flips)
    print(f"\nwidth ratios {min(ratios):.2f} to {max(ratios):.2f}; "
          f"significance verdicts changed on {len(flips)} of {len(MODELS)}")

    OUT.write_text(json.dumps(dict(
        _what="Wild cluster bootstrap-t vs percentile pair-bootstrap "
              "intervals for the per-model headline effects; twelve "
              "name-pair clusters.",
        _why="Percentile intervals with a dozen clusters can under-cover "
             "(Cameron-Miller); this is the check a reviewer asked for.",
        seed=SEED, n_boot=N_BOOT, models=results, summary=summary,
    ), indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
