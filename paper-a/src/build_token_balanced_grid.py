"""Construct a name grid whose matched pairs are token-balanced by construction.

WHY. Section 4.3 shows that between two thirds and three quarters of the matched
pairs in the standard validated hiring name list put a different number of tokens
on the two sides, that the difference predicts the measured effect on half the
panel, and that dropping the unmatched pairs moves one model's reported disparity
by 61 %. The limitation section then conceded that the design cannot separate
token length from the name properties correlated with it, and that "a name list
constructed to be token-balanced within pair would settle it".

That concession was too quick in one direction and too slow in the other.

  TOO QUICK: An and Rudinger (2023) already established, on social commonsense
  reasoning, that tokenization length affects a model's treatment of a first
  name independently of the name's demographic attributes, and they built
  name sets stratified by (race, gender, tokenization length) to show it. The
  mechanism is not ours to discover. What is ours is that the correspondence-
  audit design used by LLM hiring audits does not control for it, while their
  design did.

  TOO SLOW: nothing prevents building the balanced list. This script builds it,
  out of the SAME validated list -- Bertrand and Mullainathan's -- so the result
  can be set beside the unbalanced estimate rather than confounded with
  a change of name source.

THE CONSTRUCTION. The grid must stay FACTORIAL, because crossing first names
with surnames is what makes the within-race between-first-name variance
separately estimable (§4.2). For a factorial grid with positional pairing, every
cell (i, j) is balanced iff

    d_first(i) + d_last(j) = 0    for all i, j

where d is tokens(Black) - tokens(White). That forces d_first and d_last each to
be constant, and the only constant available on both sides at once is zero. So
the construction is: keep first-name pairs whose two names occupy the same
number of tokens, and surname pairs likewise -- ON EVERY TOKENIZER IN THE PANEL
SIMULTANEOUSLY, since a grid balanced for one model and not another would just
move the confound rather than remove it.

Token counts are measured IN CONTEXT, in the exact surrounding string the
prompt uses, because a name's segmentation depends on what precedes it. Both the
in-context count and the isolated count are recorded so the choice is auditable.

Selection among the survivors is alphabetical, the same behaviour-blind rule
§3 uses, so that no name enters or leaves this grid because of how it scored.

    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/build_token_balanced_grid.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import localserve as ls  # noqa: E402
import names as nm  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper-a" / "data" / "instrument" / "token_balanced_grid.json"

PANEL = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]

# The exact context the name sits in inside the prompt. Measured rather than
# assumed: `probe_name_length.py` uses this same string, so the counts here are
# comparable with Table 7's.
CTX_PRE, CTX_POST = "Candidate: ", "\n"


def count_in_context(port: int, text: str) -> int:
    return len(ls.tokenize(port, f"{CTX_PRE}{text}{CTX_POST}"))


def measure(port: int) -> dict:
    """Token counts for every first name and every surname in the BM2004 list."""
    out = {"first": {}, "last": {}, "full": {}}
    base = len(ls.tokenize(port, f"{CTX_PRE}{CTX_POST}"))
    for (race, gender), xs in nm.BM2004.items():
        for x in xs:
            out["first"][x] = dict(
                race=race, gender=gender,
                ctx=count_in_context(port, x) - base,
                iso=len(ls.tokenize(port, x)))
    for race, xs in nm.BM2004_SURNAMES.items():
        for x in xs:
            out["last"][x] = dict(
                race=race,
                ctx=count_in_context(port, x) - base,
                iso=len(ls.tokenize(port, x)))
    # full names, which is what actually has to balance
    for (race, gender), firsts in nm.BM2004.items():
        for f in firsts:
            for l in nm.BM2004_SURNAMES[race]:
                out["full"][f"{f} {l}"] = count_in_context(port, f"{f} {l}")
    out["_base_tokens"] = base
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--models", nargs="*", default=PANEL)
    args = ap.parse_args()

    per_model = {}
    for label in args.models:
        print(f"[serve] {label}", flush=True)
        proc = None
        try:
            proc, weights = ls.start(label, port=args.port, parallel=1)
            print(f"  weights {weights}", flush=True)
            per_model[label] = measure(args.port)
            print(f"  measured {len(per_model[label]['full'])} full names",
                  flush=True)
        finally:
            ls.stop(proc)

    if not per_model:
        print("no models measured", file=sys.stderr)
        return 1

    models = list(per_model)

    # ---- which cross-race FULL-NAME pairings balance on every tokenizer? ----
    # reported first as a descriptive fact about the list, independent of any
    # grid we then choose.
    def full_tokens(m, name):
        return per_model[m]["full"][name]

    all_full = {"white": [], "black": []}
    for (race, gender), firsts in nm.BM2004.items():
        for f in firsts:
            for l in nm.BM2004_SURNAMES[race]:
                all_full[race].append((f"{f} {l}", gender))

    pair_stats = {}
    for gender in ("female", "male"):
        w = [n for n, g in all_full["white"] if g == gender]
        b = [n for n, g in all_full["black"] if g == gender]
        tot = len(w) * len(b)
        bal_all = 0
        bal_any = 0
        for a, c in itertools.product(w, b):
            eq = [full_tokens(m, a) == full_tokens(m, c) for m in models]
            bal_all += all(eq)
            bal_any += any(eq)
        pair_stats[gender] = dict(
            n_white_full=len(w), n_black_full=len(b), n_candidate_pairings=tot,
            balanced_on_every_tokenizer=bal_all,
            balanced_on_at_least_one=bal_any,
            frac_balanced_all=bal_all / tot if tot else None)

    # ---- the factorial construction --------------------------------------
    # first-name pairs, same gender, equal in-context tokens on every model
    first_pairs = {}
    for gender in ("female", "male"):
        wf = sorted(nm.BM2004[("white", gender)])
        bf = sorted(nm.BM2004[("black", gender)])
        ok = []
        for a, c in itertools.product(wf, bf):
            if all(per_model[m]["first"][a]["ctx"] ==
                   per_model[m]["first"][c]["ctx"] for m in models):
                ok.append((a, c))
        first_pairs[gender] = ok

    wl = sorted(nm.BM2004_SURNAMES["white"])
    bl = sorted(nm.BM2004_SURNAMES["black"])
    last_pairs = [(a, c) for a, c in itertools.product(wl, bl)
                  if all(per_model[m]["last"][a]["ctx"] ==
                         per_model[m]["last"][c]["ctx"] for m in models)]

    # A grid needs each name used at most once, so take a maximum matching.
    # Greedy in alphabetical order is a valid matching and is behaviour-blind;
    # we also report the true maximum so the reader knows what greed cost.
    def greedy_matching(pairs):
        usedw, usedb, out = set(), set(), []
        for a, c in sorted(pairs):
            if a in usedw or c in usedb:
                continue
            usedw.add(a)
            usedb.add(c)
            out.append((a, c))
        return out

    def max_matching_size(pairs):
        # Hopcroft-Karp is overkill for 9x9; brute-force augmenting paths.
        adj = {}
        for a, c in pairs:
            adj.setdefault(a, []).append(c)
        matchb = {}

        def try_aug(a, seen):
            for c in adj.get(a, []):
                if c in seen:
                    continue
                seen.add(c)
                if c not in matchb or try_aug(matchb[c], seen):
                    matchb[c] = a
                    return True
            return False

        n = 0
        for a in sorted(adj):
            if try_aug(a, set()):
                n += 1
        return n

    grid = []
    sel_first = {g: greedy_matching(first_pairs[g]) for g in ("female", "male")}
    sel_last = greedy_matching(last_pairs)
    for gender in ("female", "male"):
        for i, (a, c) in enumerate(sel_first[gender]):
            for j, (x, y) in enumerate(sel_last):
                grid.append(dict(
                    idx=len(grid), gender=gender, first_i=i, last_j=j,
                    white_first=a, white_last=x, black_first=c, black_last=y,
                    white=f"{a} {x}", black=f"{c} {y}"))

    # verify the constructed grid really is balanced, on every model, in context
    residual = {m: [] for m in models}
    for p in grid:
        for m in models:
            d = full_tokens(m, p["black"]) - full_tokens(m, p["white"])
            if d != 0:
                residual[m].append(dict(pair=p["idx"], delta=d,
                                        white=p["white"], black=p["black"]))

    out = {
        "models": models,
        "context": {"pre": CTX_PRE, "post": CTX_POST},
        "source_list": "Bertrand & Mullainathan (2004), Tables 8 / A1 / fn.20",
        "prior_work": ("An & Rudinger (2023), ACL 2023 Short Papers 388-401, "
                       "stratify names by (race, gender, tokenization length) "
                       "for social commonsense reasoning. This grid applies the "
                       "same control to the matched-pair correspondence-audit "
                       "design, within pair rather than within stratum."),
        "descriptive": pair_stats,
        "first_name_pairs_balanced": {g: [list(p) for p in first_pairs[g]]
                                      for g in first_pairs},
        "surname_pairs_balanced": [list(p) for p in last_pairs],
        "max_matching": {
            "female_first": max_matching_size(first_pairs["female"]),
            "male_first": max_matching_size(first_pairs["male"]),
            "surnames": max_matching_size(last_pairs),
        },
        "selected": {
            "female_first": [list(p) for p in sel_first["female"]],
            "male_first": [list(p) for p in sel_first["male"]],
            "surnames": [list(p) for p in sel_last],
        },
        "grid": grid,
        "n_pairs": len(grid),
        "residual_imbalance": {m: v for m, v in residual.items()},
        "fully_balanced": all(not v for v in residual.values()),
        "per_model_tokens": per_model,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 96)
    print("HOW MANY CROSS-RACE FULL-NAME PAIRINGS ARE TOKEN-BALANCED AT ALL?")
    print("=" * 96)
    for g, s in pair_stats.items():
        print(f"  {g:<8} {s['n_candidate_pairings']:>6} candidate pairings   "
              f"{s['balanced_on_every_tokenizer']:>5} balanced on all "
              f"{len(models)} tokenizers ({s['frac_balanced_all']:.1%})")
    print("\nFACTORIAL CONSTRUCTION")
    for g in ("female", "male"):
        print(f"  {g:<8} first-name pairs available {len(first_pairs[g]):>3}, "
              f"maximum matching {out['max_matching'][g[0] == 'f' and 'female_first' or 'male_first']}, "
              f"selected {len(sel_first[g])}")
    print(f"  surnames  pairs available {len(last_pairs):>3}, "
          f"maximum matching {out['max_matching']['surnames']}, "
          f"selected {len(sel_last)}")
    print(f"\n  GRID: {len(grid)} matched pairs, fully balanced on every "
          f"tokenizer: {out['fully_balanced']}")
    for p in grid[:6]:
        print(f"    {p['white']:<22} | {p['black']:<22} {p['gender']}")
    if len(grid) > 6:
        print(f"    ... {len(grid) - 6} more")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
