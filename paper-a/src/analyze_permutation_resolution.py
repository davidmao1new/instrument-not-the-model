r"""How small a p-value the token-matched design can even return.

WHY THIS EXISTS. The paper reports that restricting to token-matched name
pairs "leaves too few independent clusters to carry inference". That is a
statement about power, and a reader is entitled to ask how few is too few.
There is a sharper answer, and it does not depend on the data at all.

The matched-pair randomization test conditions on the observed pairs and
re-randomizes the label within each pair. With n pairs there are exactly
2^n equally likely sign assignments, so the smallest attainable one-tailed
p-value is 2^-n and the smallest two-tailed one is 2^{1-n}. Stratifying the
permutation within token-length equivalence classes (the correct test once
token length is controlled) does not enlarge that set: the classes are
complete blocks, so the randomization distribution is a product over
classes and the arrangement count is unchanged.

The consequence is an impossibility rather than a power shortfall. Where
the matched subset is small enough that 2^{1-n} exceeds the conventional
threshold, the exact test cannot report significance whatever the effect
turns out to be. A null from such an analysis carries no evidence, and a
"significant" result from a non-exact approximation of it is an artifact of
the approximation.

WHAT IT READS.
  instrument/name_length_effect.json  matched first-name-pair clusters per
                                      checkpoint, on the design as run.
  instrument/name_list_power.json     maximum matching over the full source
                                      list, and the per-name yield used to
                                      price a remedy.

    sh paper-a/src/_py.sh paper-a/src/analyze_permutation_resolution.py
"""
from __future__ import annotations

import json
import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "paper-a" / "data"
OUT = DATA / "instrument" / "permutation_resolution.json"

ALPHA = 0.05


def _j(rel: str) -> dict:
    return json.loads((DATA / rel).read_text(encoding="utf-8"))


def floor_for(n: int) -> dict:
    """The exact randomization test's resolution at n matched pairs."""
    if n <= 0:
        return {"n_pairs": n, "n_assignments": 0,
                "p_min_one_tailed": None, "p_min_two_tailed": None,
                "can_reach_alpha": False}
    n_assign = 2 ** n
    one = 1.0 / n_assign
    two = min(1.0, 2.0 * one)
    return {"n_pairs": n, "n_assignments": n_assign,
            "p_min_one_tailed": one, "p_min_two_tailed": two,
            "can_reach_alpha": two <= ALPHA}


def smallest_workable_n() -> int:
    """Fewest pairs whose two-tailed exact floor clears alpha."""
    n = 1
    while not floor_for(n)["can_reach_alpha"]:
        n += 1
        if n > 64:                       # unreachable; guards a bad ALPHA
            raise AssertionError("no n clears alpha")
    return n


def main() -> int:
    nle = _j("instrument/name_length_effect.json")
    power = _j("instrument/name_list_power.json")

    # ---- the design as run, per checkpoint ------------------------------
    per_model = {}
    for model, v in nle.items():
        if not isinstance(v, dict):
            continue
        tm = v.get("token_matched_first_name_clustered")
        if not tm:
            continue
        n = tm["n_matched_clusters"]
        rec = floor_for(n)
        rec["n_clusters_before_matching"] = tm["n_grid_clusters"]
        rec["matched_pairs"] = tm["matched_cluster_labels"]
        per_model[model] = rec

    assert per_model, "no token-matched cluster counts found"

    # WHICH CHECKPOINT IS WHICH, AND WHY THIS BLOCK EXISTS. An earlier
    # version read `headline_model` out of the artifact and described it as
    # "the checkpoint carrying the significant length slope". It is not
    # that: it names the model whose effect-difference was the headline of
    # a different comparison, one that does not survive clustering. The two
    # are different checkpoints, so the prose attributed a three-pair floor
    # to a checkpoint that has eight. Both are computed here, separately
    # and from their own evidence, so a sentence cannot borrow one and mean
    # the other.
    slope_sig = []
    for model, v in nle.items():
        if not isinstance(v, dict):
            continue
        rec = v.get("token_matched_first_name_clustered") or {}
        ci = (rec.get("slope_same_unit") or {}).get("ci")
        if ci and ci[0] * ci[1] > 0:
            slope_sig.append(model)

    # The binding case for the impossibility claim is the checkpoint where
    # fewest pairs survive, not whichever one carries a slope.
    fewest = min(per_model, key=lambda m: per_model[m]["n_pairs"])

    # ---- the source list at its maximum, across the panel ---------------
    panel_max = power["max_matching_panel_total"]
    panel_rec = floor_for(panel_max)
    panel_rec["possible_pairs_before_matching"] = power["possible_pairs"]

    # ---- what it would take to clear alpha ------------------------------
    n_need = smallest_workable_n()
    yield_per_name = power["extrapolation"]["matched_pairs_per_name_per_cell"]
    names_now = power["extrapolation"]["names_per_cell_now"]
    names_needed = n_need / yield_per_name
    remedy = {
        "n_pairs_needed": n_need,
        "why": f"smallest n with 2^(1-n) <= {ALPHA}",
        "p_min_two_tailed_at_that_n": floor_for(n_need)["p_min_two_tailed"],
        "names_per_cell_now": names_now,
        "matched_pairs_per_name_per_cell": yield_per_name,
        "names_per_cell_needed": names_needed,
        "list_size_multiple": names_needed / names_now,
        "_caveat": power["extrapolation"]["_caveat"],
    }

    n_blocked = sum(1 for r in per_model.values() if not r["can_reach_alpha"])

    out = {
        "_what": "The exact randomization test's resolution floor on the "
                 "token-matched subset: the smallest p-value attainable at "
                 "n matched pairs, independent of the data.",
        "_why": "A matched-pair randomization test has 2^n equally likely "
                "sign assignments, so the one-tailed floor is 2^-n and the "
                "two-tailed floor 2^(1-n). Stratifying within token-length "
                "classes leaves the count unchanged, because the classes "
                "are complete blocks and the randomization distribution "
                "factorizes over them.",
        "_alpha": ALPHA,
        "per_model_design_as_run": per_model,
        "fewest_pairs_model": fewest,
        "fewest_pairs_floor_two_tailed":
            per_model[fewest]["p_min_two_tailed"],
        "fewest_pairs_can_reach_alpha":
            per_model[fewest]["can_reach_alpha"],
        "slope_significant_models": slope_sig,
        "slope_significant_pairs":
            {m: per_model[m]["n_pairs"] for m in slope_sig},
        "_note_on_naming": (
            "fewest_pairs_model is the binding case for the resolution "
            "floor. slope_significant_models is a different set, and the "
            "two must not be conflated in prose: the checkpoint carrying "
            "the significant token-length slope is not the one with the "
            "fewest surviving pairs."),
        "n_models_that_cannot_reach_alpha": n_blocked,
        "n_models": len(per_model),
        "source_list_panel_maximum": panel_rec,
        "remedy": remedy,
        "_verdict": (
            "On the checkpoint the length confound was identified on, the "
            f"token-matched subset holds {per_model[fewest]['n_pairs']} "
            "pairs, so the exact test's smallest two-tailed p-value is "
            f"{per_model[fewest]['p_min_two_tailed']:.3f}. It cannot "
            "return significance at any conventional level whatever the "
            "data show. Across the full source list the panel-wide maximum "
            f"matching is {panel_max} pairs, with the same floor. Reaching "
            f"even {n_need} pairs, the fewest that clear {ALPHA}, would "
            f"take about {names_needed:.0f} names per cell against the "
            f"{names_now} the standard list carries."),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("PERMUTATION RESOLUTION FLOOR")
    print("=" * 66)
    for m, r in sorted(per_model.items()):
        flag = "" if r["can_reach_alpha"] else "   <- cannot reach alpha"
        print(f"  {m:28s} n={r['n_pairs']:2d}  "
              f"p_min(2t)={r['p_min_two_tailed']:.4f}{flag}")
    print(f"\n  source list, panel maximum: n={panel_max}  "
          f"p_min(2t)={panel_rec['p_min_two_tailed']:.4f}")
    print(f"  to clear alpha={ALPHA}: n>={n_need} pairs, about "
          f"{names_needed:.0f} names per cell "
          f"({remedy['list_size_multiple']:.1f}x the standard list)")
    print(f"\n  -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
