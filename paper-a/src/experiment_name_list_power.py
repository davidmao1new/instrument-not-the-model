r"""How large a name list would have to be before the token-matched design has power.

WHY THIS EXISTS. A statistician reviewing the design on 24 August 2026 made
the point the paper had not: at three independent clusters, standard asymptotic
inference is not reliable, three procedures disagreeing is what that looks like
from the inside, and an exact cluster-level permutation test on three clusters
has only 2^3 = 8 arrangements, so it cannot produce a two-sided p below 0.05 at
all. Her conclusion was that the most defensible claim is not that the
restricted analysis establishes a disparity, but that the existing name list
cannot answer the question once tokenization is controlled.

That is a claim about the LIST, and a claim about a list is measurable.

THE KEY STRUCTURAL FACT, which makes this cheap. Token-balance is an
equivalence relation, not an arbitrary graph. Two names can be paired iff they
occupy the same number of tokens, so the bipartite graph is a disjoint union of
complete bipartite blocks, one per token-length vector across the panel. The
maximum matching is therefore exact and needs no search:

    M  =  sum over vectors v of  min( #white with v , #black with v )

That identity is what this script measures, and it says immediately why the
number is small and how it would grow: it grows only where the two races'
length distributions OVERLAP. A larger list helps in proportion to the overlap,
not in proportion to its size.

WHAT IS MEASURED
  1. The panel-wide matching, which should reproduce the 3 already reported.
  2. The per-model matching, which is what an analysis that gives up
     cross-model comparability would get instead.
  3. The matching as a function of how many tokenizers must agree, 1 through 4.
  4. The per-vector histogram, and from it the names-per-cell a list would need
     to reach a target number of independent pairs, at the overlap this list
     shows.

Item 4 is an extrapolation and is labelled as one. It assumes a larger list
draws from the same length distribution, which is exactly what a list built
without regard to tokenization does.

    sh paper-a/src/_py.sh paper-a/src/experiment_name_list_power.py

Writes paper-a/data/instrument/name_list_power.json. Tokenization only: no
generation, no sampling, so nothing here depends on decoding settings.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import localserve as ls  # noqa: E402
import names as nm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper-a" / "data" / "instrument" / "name_list_power.json"

PANEL = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]

# The exact context the name sits in inside the prompt, identical to
# build_token_balanced_grid.py so the counts are comparable with Table 7.
CTX_PRE, CTX_POST = "Candidate: ", "\n"


def measure_model(label: str, port: int) -> dict[str, int]:
    """In-context token count for every first name in the BM2004 list."""
    # start() returns (process, weights_path): the weights are returned so the
    # caller can record which file was actually served, since config.yaml pins
    # it by SHA-256 rather than by path.
    proc, weights = ls.start(label, port=port)
    try:
        base = len(ls.tokenize(port, f"{CTX_PRE}{CTX_POST}"))
        out = {}
        for (_race, _gender), xs in nm.BM2004.items():
            for x in xs:
                out[x] = len(ls.tokenize(port, f"{CTX_PRE}{x}{CTX_POST}")) - base
        out["_served"] = pathlib.Path(weights).name
        return out
    finally:
        ls.stop(proc)


def matching(vec: dict[str, tuple], white: list[str], black: list[str]) -> int:
    """Maximum matching = sum over length-vectors of min(#white, #black).

    Exact because token-balance is an equivalence relation: every eligible edge
    joins two names with the same vector, so the graph is a union of complete
    bipartite blocks and the matching decomposes block by block.
    """
    w = collections.Counter(vec[n] for n in white)
    b = collections.Counter(vec[n] for n in black)
    return sum(min(w[v], b[v]) for v in set(w) | set(b))


def race_gap(counts: dict) -> dict:
    """Mean in-context token count by race, per tokenizer.

    This is the mechanism. If the two arms differed only by noise, matching on
    token length would keep most of the list. They do not: the Black arm is
    about one token longer on every tokenizer, so the two length distributions
    barely overlap and the matching is thin by construction rather than by
    accident.
    """
    import statistics
    out = {}
    for m in PANEL:
        w = [counts[m][n] for (r, _g), xs in nm.BM2004.items()
             if r == "white" for n in xs]
        b = [counts[m][n] for (r, _g), xs in nm.BM2004.items()
             if r == "black" for n in xs]
        out[m] = {"white_mean": round(statistics.mean(w), 3),
                  "black_mean": round(statistics.mean(b), 3),
                  "gap": round(statistics.mean(b) - statistics.mean(w), 3)}
    return out


def main() -> int:
    counts: dict[str, dict[str, int]] = {}
    if "--from-cache" in sys.argv and OUT.exists():
        counts = json.loads(OUT.read_text(encoding="utf-8"))["token_counts"]
        print("  recomputing from stored token counts, no models served")
    else:
        for i, label in enumerate(PANEL):
            print(f"  tokenizing on {label} ...", flush=True)
            counts[label] = measure_model(label, port=8080 + i)

    names_by_cell = {f"{r}_{g}": list(xs) for (r, g), xs in nm.BM2004.items()}
    all_names = sorted({n for xs in names_by_cell.values() for n in xs})

    def vectors(models: list[str]) -> dict[str, tuple]:
        return {n: tuple(counts[m][n] for m in models) for n in all_names}

    res: dict = {
        "_why": "Answers whether the token-matched design can be given power "
                "by a larger name list, a question raised in external review "
                "of the design.",
        "_identity": "max matching = sum_v min(#white_v, #black_v), exact "
                     "because token-balance is an equivalence relation",
        "panel": PANEL,
        "context": {"pre": CTX_PRE, "post": CTX_POST},
        "token_counts": counts,
        "per_gender": {},
        "by_n_tokenizers": {},
        "per_model": {},
        "vector_histogram": {},
        "race_gap_in_tokens": race_gap(counts),
        "possible_pairs": 2 * len(nm.BM2004[("white", "female")]),
    }

    # ---- panel-wide, per gender
    vec4 = vectors(PANEL)
    for gender in ("female", "male"):
        w = names_by_cell[f"white_{gender}"]
        b = names_by_cell[f"black_{gender}"]
        res["per_gender"][gender] = {
            "n_white": len(w), "n_black": len(b),
            "max_matching_panel": matching(vec4, w, b),
        }
        hw = collections.Counter(vec4[n] for n in w)
        hb = collections.Counter(vec4[n] for n in b)
        res["vector_histogram"][gender] = {
            "white": {str(k): v for k, v in sorted(hw.items())},
            "black": {str(k): v for k, v in sorted(hb.items())},
            "overlap": {str(k): min(hw[k], hb[k])
                        for k in set(hw) & set(hb) if min(hw[k], hb[k])},
        }
    res["max_matching_panel_total"] = sum(
        res["per_gender"][g]["max_matching_panel"] for g in ("female", "male"))

    # ---- how much of the collapse is the cross-model requirement
    for k in range(1, len(PANEL) + 1):
        sub = PANEL[:k]
        v = vectors(sub)
        res["by_n_tokenizers"][k] = {
            "models": sub,
            "total": sum(matching(v, names_by_cell[f"white_{g}"],
                                  names_by_cell[f"black_{g}"])
                         for g in ("female", "male")),
        }
    for m in PANEL:
        v = vectors([m])
        res["per_model"][m] = sum(
            matching(v, names_by_cell[f"white_{g}"],
                     names_by_cell[f"black_{g}"])
            for g in ("female", "male"))

    # ---- what a larger list would buy, at this list's overlap
    per_cell = len(names_by_cell["white_female"])
    got = res["max_matching_panel_total"]
    rate = got / (2 * per_cell)          # matched pairs per name-per-cell
    res["extrapolation"] = {
        "_caveat": "Linear in list size only if a larger list draws from the "
                   "same length distribution. A list BUILT to be token-matched "
                   "does far better; that is the point of the recommendation.",
        "names_per_cell_now": per_cell,
        "matched_pairs_now": got,
        "matched_pairs_per_name_per_cell": round(rate, 4),
        "names_per_cell_for": {
            str(t): (None if rate == 0 else int(-(-t // rate) // 2 * 2 or 2))
            for t in (10, 20, 30)
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")

    print(f"\n  panel-wide matched first-name pairs: "
          f"{res['max_matching_panel_total']}")
    for g in ("female", "male"):
        print(f"    {g:<8}{res['per_gender'][g]['max_matching_panel']}")
    print("\n  matching as tokenizers are added:")
    for k, v in res["by_n_tokenizers"].items():
        print(f"    {k} tokenizer(s): {v['total']}")
    print("\n  single-model matching:")
    for m, v in res["per_model"].items():
        print(f"    {m:<28}{v}")
    print("\n  mean in-context tokens by race:")
    for m, v in res["race_gap_in_tokens"].items():
        print(f"    {m:<28}white {v['white_mean']:.2f}  "
              f"black {v['black_mean']:.2f}  gap {v['gap']:+.2f}")
    print(f"\n  of {res['possible_pairs']} possible pairs, "
          f"{max(res['per_model'].values())} survive the best single tokenizer "
          f"and {res['max_matching_panel_total']} survive all four")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
