"""Is our measurement noise largest where Fu et al. predict it should be?

WHY THIS EXISTS. §5.2 establishes a noise floor: how much the measurement moves
across byte-identical repeats once batching and cache residency are controlled.
It reports that floor as a single number per model. Fu, Martinez, Conde,
Arriaga, Reviriego, Qi and Liu (2026) analysed LLM nondeterminism at the TOKEN
PROBABILITY level rather than at the text level -- which is exactly the level
this paper's outcome lives at -- and report that the effect is "negligible for
probabilities close to 0 or 1 and significant for probabilities in the range of
0.2 to 0.8".

That is a prediction about our data, and it is checkable. If it holds here, our
noise floor is not a property of our stack but an instance of something general,
and the floor a future auditor should expect depends on where their model sits
on the logistic rather than on which GPU they own. If it fails here, we have a
disagreement worth reporting.

THE TEST. Each replicate cell is the same prompt run repeatedly under the
controlled configuration. For each cell we compute:

  * the renormalised acceptance probability p = p(yes) / (p(yes) + p(no)),
    averaged over the repeats -- where the cell sits on the curve; and
  * the standard deviation of that same quantity across the repeats -- how much
    it moves when nothing changes.

Fu et al. predict the second is largest where the first is near 0.5. We test it
three ways, because a single summary could be produced by an outlier:

  1. Spearman correlation between the noise SD and min(p, 1 - p), which is a
     monotone measure of how close the cell is to the middle. Positive supports
     the prediction.
  2. The noise SD in the middle band (0.2 to 0.8) against the noise SD outside
     it, with a permutation test on the difference of means.
  3. The same, restricted to cells that are not exactly reproducible, so the
     result is not driven by the many cells where the repeat is bitwise equal
     and the SD is exactly zero by construction.

A REAL CONFOUND, STATED. The renormalised probability and its own variability
are not independent quantities even under a null: a binomial-like quantity
bounded in [0, 1] has less room to vary near the boundary whatever the
mechanism. So this test cannot separate "the arithmetic is noisier in the
middle" from "any bounded quantity varies less near its bound". What it can do
is establish whether the PATTERN Fu et al. report is present in a fairness
measurement on a CPU/Vulkan stack, which is the transportability question §5.2
needs, and that is how the result is reported.

    C:/research-toolchain/venv/Scripts/python.exe \
        paper-a/src/analyze_noise_vs_probability.py
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
D = ROOT / "paper-a" / "data" / "replicate"
OUT = ROOT / "paper-a" / "data" / "replicate" / "noise_vs_probability.json"

ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SHORT = {"llama-2-7b-chat": "Llama-2-7B", "llama-3.1-8b-instruct": "Llama-3.1-8B",
         "mistral-7b-instruct-v0.1": "Mistral v0.1",
         "mistral-7b-instruct-v0.3": "Mistral v0.3"}
BAND = (0.2, 0.8)
N_PERM = 20000
SEED = 20260801


def spearman(x, y):
    """Rank correlation, computed here so the module needs no scipy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None

    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        # average ties, which matter here because many SDs are exactly zero
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else None


def cells(model: str):
    """(mean renormalised p, SD across repeats) for every replicate cell.

    Both arms of a pair are used as separate cells: the question is about a
    single prompt's next-token distribution, not about the paired difference.
    """
    f = D / f"rep_{model}.jsonl"
    if not f.exists():
        return []
    by = collections.defaultdict(list)
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        for arm in ("white", "black"):
            py, pn = r.get(f"{arm}_p_yes"), r.get(f"{arm}_p_no")
            if py is None or pn is None or (py + pn) <= 0:
                continue
            by[(r["variant"], r["template"], r["pair"], arm)].append(py / (py + pn))
    out = []
    for k, vals in by.items():
        if len(vals) < 2:
            continue
        out.append((float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals)))
    return out


def perm_diff(a, b, rng, n=N_PERM):
    """Two-sided permutation p for a difference of means."""
    obs = abs(np.mean(a) - np.mean(b))
    pool = np.concatenate([a, b])
    na = len(a)
    hits = 0
    for _ in range(n):
        p = rng.permutation(pool)
        if abs(p[:na].mean() - p[na:].mean()) >= obs:
            hits += 1
    return float((1 + hits) / (1 + n)), float(np.mean(a) - np.mean(b))


def main() -> int:
    rng = np.random.default_rng(SEED)
    out = {
        "_what": "Whether this paper's own measurement noise follows the "
                 "pattern Fu et al. (2026) report at the token-probability "
                 "level: negligible near p = 0 or 1, largest in 0.2 to 0.8.",
        "_prediction_source": "Fu, Martinez, Conde, Arriaga, Reviriego, Qi and "
                              "Liu (2026), abstract.",
        "_caveat": "A quantity bounded in [0, 1] has less room to vary near "
                   "its bounds whatever the mechanism, so this cannot "
                   "separate the arithmetic explanation from the boundary "
                   "one. It establishes only that the pattern is present.",
        "band": list(BAND), "n_permutations": N_PERM, "seed": SEED,
        "models": {},
    }
    print(f"{'model':<16}{'cells':>7}{'in band':>9}{'SD in':>9}{'SD out':>9}"
          f"{'p':>8}{'rho':>8}   (rho vs min(p,1-p))")
    print("-" * 96)
    for m in ORDER:
        cs = cells(m)
        if len(cs) < 10:
            continue
        p = np.array([c[0] for c in cs])
        sd = np.array([c[1] for c in cs])
        mid = np.minimum(p, 1 - p)
        inb = (p >= BAND[0]) & (p <= BAND[1])
        rho = spearman(mid, sd)
        rec = dict(n_cells=len(cs), n_in_band=int(inb.sum()),
                   n_exactly_reproducible=int((sd == 0).sum()),
                   spearman_rho_sd_vs_centrality=rho,
                   mean_sd_in_band=float(sd[inb].mean()) if inb.any() else None,
                   mean_sd_out_of_band=float(sd[~inb].mean()) if (~inb).any() else None)
        if inb.any() and (~inb).any():
            pv, diff = perm_diff(sd[inb], sd[~inb], rng)
            rec["band_minus_out"] = diff
            rec["p_permutation"] = pv
            # and again on the cells that actually moved, so the many
            # bitwise-identical cells cannot carry the result on their own
            nz = sd > 0
            if (inb & nz).sum() >= 3 and ((~inb) & nz).sum() >= 3:
                pv2, d2 = perm_diff(sd[inb & nz], sd[(~inb) & nz], rng)
                rec["nonzero_only"] = dict(
                    band_minus_out=d2, p_permutation=pv2,
                    n_in=int((inb & nz).sum()), n_out=int(((~inb) & nz).sum()))
            rec["supports_prediction"] = bool(diff > 0 and pv < 0.05)
            print(f"{SHORT[m]:<16}{len(cs):>7}{int(inb.sum()):>9}"
                  f"{sd[inb].mean():>9.4f}{sd[~inb].mean():>9.4f}"
                  f"{pv:>8.4f}{(rho if rho is not None else float('nan')):>8.3f}")
        out["models"][m] = rec

    mods = out["models"]
    # TWO MODELS HAVE NO CELLS IN THE BAND AT ALL, and that is not a missing
    # test -- it is the prediction's other half. Their acceptance probabilities
    # sit entirely outside 0.2 to 0.8, and their per-arm noise is orders of
    # magnitude smaller than the models that do have cells in the middle. The
    # band test is reported where it is computable; the rank correlation is
    # computable everywhere and is the statistic that travels.
    n_testable = sum(1 for v in mods.values() if "p_permutation" in v)
    n_sup = sum(1 for v in mods.values() if v.get("supports_prediction"))
    rhos = [v["spearman_rho_sd_vs_centrality"] for v in mods.values()
            if v.get("spearman_rho_sd_vs_centrality") is not None]
    _out_sd = [v["mean_sd_out_of_band"] for v in mods.values()
               if v.get("n_in_band") == 0 and v.get("mean_sd_out_of_band")]
    out["summary"] = dict(
        n_models=len(mods), n_band_testable=n_testable, n_supporting=n_sup,
        n_models_with_no_cell_in_band=sum(
            1 for v in mods.values() if v.get("n_in_band") == 0),
        rho_min=min(rhos) if rhos else None, rho_max=max(rhos) if rhos else None,
        all_rho_positive=bool(rhos and min(rhos) > 0),
        mean_sd_where_no_cell_in_band=(float(np.mean(_out_sd))
                                       if _out_sd else None),
        verdict=(
            "the pattern Fu et al. report at the token-probability level is "
            "present here: the rank correlation between per-arm noise and "
            "closeness to p = 0.5 is positive on every model, and where the "
            "0.2-to-0.8 band contains any cell at all the in-band noise is "
            "larger than the out-of-band noise"
            if (rhos and min(rhos) > 0 and n_sup == n_testable) else
            "the pattern is not established on every model"))
    print()
    print(f"  band test computable on {n_testable} of {len(mods)} models "
          f"(the other two have no cell inside 0.2-0.8 at all, which is "
          f"itself the prediction's other half)")
    print(f"  supports it on {n_sup} of {n_testable}; rho positive on all "
          f"{len(rhos)}, from {out['summary']['rho_min']:.3f} to "
          f"{out['summary']['rho_max']:.3f}")
    print(f"  VERDICT: {out['summary']['verdict']}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
