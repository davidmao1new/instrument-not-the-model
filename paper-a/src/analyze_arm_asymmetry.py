"""Per-arm volatility: is the Black-name arm the less stable one?

PROVENANCE. The hypothesis tested here was raised externally rather than
originated by the authors: that audit volatility is epistemic, the model
having less information about minority names and so more ways to resolve
the uncertainty, which would make the minority arm the volatile one. The
test, the computation and the verdict are ours. It was drafted inside an
adversarial audit workflow and independently recomputed from the method
description alone before being adopted; every number matched.

Run from the repo root:
    cd "research" && sh paper-a/src/_py.sh <this_file>

Hypothesis under test: audit volatility is driven by the arm the model
knows less about, i.e. the Black-name arm should show LARGER between-wording
variance than the White-name arm.

Data (read-only):
  paper-a/data/delta_stability/delta_<model>.jsonl
      432 rows/model = 12 wording variants (S1-S6 semantic, N1-N6 null)
      x 3 templates x 12 name pairs; per-arm continuous margins
      white_margin / black_margin per row.
  paper-a/data/names/names_<model>.jsonl
      1728 rows/model = 12 variants x 3 templates x 48 name pairs
      (2 genders x 6 paired first names x 4 paired last names, crossed).

Statistic, per model:
  For each cell c = (name pair, template): the 12 wording variants form a
  paired set (identical wordings hit both arms).  v_B(c), v_W(c) = sample
  variance (ddof=1) of black_margin / white_margin across the 12 wordings.
  D = mean_c v_B(c) - mean_c v_W(c)   (paired across wordings by design)
  R = mean_c v_B(c) / mean_c v_W(c)
  Robustness: within-kind variance (variance computed inside the 6 semantic
  and 6 null wordings separately, then averaged) to strip the semantic-vs-
  null mean shift out of the "between-wording" variance.

Uncertainty:
  Cluster bootstrap, 10000 draws, percentile 95% CI.
  Resampling unit = the NAME cluster, never the row:
    - delta_stability: the 12 name pairs (matches the paper's own resampling
      unit, resampling_unit.json: n_pairs=12, clustered=true).
    - names: the 12 first-name pairs (gender x first_i).  Last names are
      crossed with first names; name_variance.json shows sigma_last is far
      below sigma_first, so first-name clusters are the binding dependence.
  Each resampled cluster carries ALL its templates and ALL 12 wordings for
  BOTH arms, preserving the pairing across wordings and arms.
"""
import json
import numpy as np
from collections import defaultdict

RNG_SEED = 20260825
N_BOOT = 10000
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = str(ROOT / "paper-a" / "data")
OUT = ROOT / "paper-a" / "data" / "instrument" / "arm_asymmetry.json"
MODELS = [
    "llama-2-7b-chat",
    "llama-3.1-8b-instruct",
    "mistral-7b-instruct-v0.1",
    "mistral-7b-instruct-v0.3",
]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build_cells(rows, cluster_of_pair):
    """cells[(pair, template)] -> dict variant -> (black_margin, white_margin);
    returns list of (cluster_id, v_black, v_white, v_black_wk, v_white_wk)."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["pair"], r["template"])].append(r)
    out = []
    for (pair, _tmpl), rs in sorted(cells.items()):
        assert len(rs) == 12, "expected 12 wording variants per cell"
        b = np.array([r["black_margin"] for r in rs])
        w = np.array([r["white_margin"] for r in rs])
        kinds = np.array([r["variant_kind"] for r in rs])
        vb, vw = b.var(ddof=1), w.var(ddof=1)
        # within-kind robustness: variance inside semantic-6 and null-6, averaged
        vb_wk = np.mean([b[kinds == k].var(ddof=1) for k in ("semantic", "null")])
        vw_wk = np.mean([w[kinds == k].var(ddof=1) for k in ("semantic", "null")])
        out.append((cluster_of_pair[pair], vb, vw, vb_wk, vw_wk))
    return out


def stat(cells_arr, idx_by_cluster, chosen):
    """Mean-difference and ratio over the cells of the chosen clusters
    (with multiplicity)."""
    rows = np.concatenate([idx_by_cluster[c] for c in chosen])
    sub = cells_arr[rows]
    mb, mw = sub[:, 0].mean(), sub[:, 1].mean()
    mb_wk, mw_wk = sub[:, 2].mean(), sub[:, 3].mean()
    return mb - mw, mb / mw, mb_wk - mw_wk


def analyse(rows, cluster_of_pair, label, model, rng):
    cells = build_cells(rows, cluster_of_pair)
    clusters = sorted({c for c, *_ in cells})
    arr = np.array([[vb, vw, vbk, vwk] for _c, vb, vw, vbk, vwk in cells])
    idx_by_cluster = {
        c: np.array([i for i, (cc, *_r) in enumerate(cells) if cc == c])
        for c in clusters
    }
    d_pt, r_pt, dwk_pt = stat(arr, idx_by_cluster, clusters)
    frac_cells = float(np.mean(arr[:, 0] > arr[:, 1]))
    boots = np.array([
        stat(arr, idx_by_cluster,
             [clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        for _ in range(N_BOOT)
    ])
    d_ci = np.percentile(boots[:, 0], [2.5, 97.5])
    r_ci = np.percentile(boots[:, 1], [2.5, 97.5])
    dwk_ci = np.percentile(boots[:, 2], [2.5, 97.5])
    print(f"{model:28s} {label:6s} "
          f"varB={arr[:,0].mean():7.4f} varW={arr[:,1].mean():7.4f} "
          f"D={d_pt:+7.4f} CI[{d_ci[0]:+7.4f},{d_ci[1]:+7.4f}] "
          f"R={r_pt:5.2f} CI[{r_ci[0]:5.2f},{r_ci[1]:5.2f}] "
          f"Dwk={dwk_pt:+7.4f} CI[{dwk_ci[0]:+7.4f},{dwk_ci[1]:+7.4f}] "
          f"frac(vB>vW)={frac_cells:.2f} "
          f"n_cells={len(cells)} n_clusters={len(clusters)}")
    return dict(model=model, dataset=label, mean_var_black=arr[:, 0].mean(),
                mean_var_white=arr[:, 1].mean(), D=d_pt, D_ci=list(d_ci),
                R=r_pt, R_ci=list(r_ci), D_withinkind=dwk_pt,
                D_withinkind_ci=list(dwk_ci), frac_cells_black_higher=frac_cells,
                n_cells=len(cells), n_clusters=len(clusters))


def main():
    rng = np.random.default_rng(RNG_SEED)
    print(f"cluster bootstrap, {N_BOOT} draws, seed {RNG_SEED}; "
          f"D = mean_cell Var_12wordings(black_margin) - same for white; "
          f"Dwk = within-kind (6+6) version")
    results = []
    for m in MODELS:
        # Study-2 deltas: 12 name pairs, cluster = the name pair itself.
        rows = load(f"{DATA}/delta_stability/delta_{m}.jsonl")
        assert all(not r["error"] for r in rows)
        c_of_p = {r["pair"]: r["pair"] for r in rows}
        results.append(analyse(rows, c_of_p, "delta", m, rng))
        # Names grid: 48 pairs crossed from 12 first-name pairs x 4 last-name
        # pairs; cluster = first-name pair (gender, first_i).
        rows = load(f"{DATA}/names/names_{m}.jsonl")
        assert all(not r["error"] for r in rows)
        c_of_p = {r["pair"]: (r["gender"], r["first_i"]) for r in rows}
        results.append(analyse(rows, c_of_p, "names", m, rng))
    print()
    n_pos = sum(1 for r in results if r["D"] > 0)
    n_excl = sum(1 for r in results if r["D_ci"][0] > 0)
    n_rev = sum(1 for r in results if r["D_ci"][1] < 0)
    reversed_models = sorted({r["model"] for r in results
                              if r["D_ci"][1] < 0})
    print(f"datasets x models with D>0: {n_pos}/{len(results)}; "
          f"with 95% CI excluding 0: {n_excl}/{len(results)}; "
          f"reversed (CI < 0): {n_rev} ({', '.join(reversed_models)})")
    OUT.write_text(json.dumps(dict(
        _what="Between-wording variance of the Black-name arm minus the "
              "White-name arm, per (model, dataset); cluster bootstrap on "
              "name clusters.",
        _provenance="Hypothesis raised externally, not originated by the "
                    "authors; computation independently reproduced before "
                    "adoption.",
        seed=RNG_SEED, n_boot=N_BOOT,
        results=results,
        summary=dict(
            n_combinations=len(results),
            n_positive=n_pos,
            n_ci_excludes_zero_positive=n_excl,
            n_ci_excludes_zero_negative=n_rev,
            reversed_models=reversed_models),
    ), indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
