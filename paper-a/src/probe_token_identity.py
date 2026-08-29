"""Are any of the 'different' prompts actually identical to the model?

A perturbation only matters if it survives tokenisation. Whitespace is exactly
the kind of edit a BPE tokeniser can absorb: a trailing newline at the very end
of a user turn may be stripped by the chat template, and repeated spaces may map
onto a single existing whitespace token.

So before asking why a perturbation changed the answer, the prior question is
whether the model saw a different input at all. This probe compares the FULL
token sequence of every variant against the baseline, for both names and all
three templates, and partitions the twelve variants into:

    IDENTICAL   the model receives exactly the same integers
    SHIFTED     same tokens, but the name sits at a different index
    ALTERED     the token multiset itself differs

The partition is the mechanistic backbone. A variant in IDENTICAL that shows a
different δ would mean the measurement is nondeterministic, which at temperature
0 with a fixed seed it should not be. A variant in SHIFTED isolates position. A
variant in ALTERED confounds position and content.
"""
from __future__ import annotations

import json
import pathlib
import sys

import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from experiment_delta_stability import (  # noqa: E402
    PAIRS, TEMPLATES, VARIANTS, user_message,
)

OUT = ROOT / "paper-a" / "data" / "mechanism"


def tok(api, text):
    r = requests.post(f"{api}/tokenize", json={"content": text}, timeout=120)
    r.raise_for_status()
    return r.json()["tokens"]


def classify(base: list[int], other: list[int]) -> str:
    if base == other:
        return "IDENTICAL"
    if sorted(base) == sorted(other):
        return "REORDERED"
    if len(base) == len(other):
        return "ALTERED_SAME_LEN"
    return "ALTERED"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model-label", required=True)
    args = ap.parse_args()
    api = f"http://127.0.0.1:{args.port}"
    OUT.mkdir(parents=True, exist_ok=True)

    base_v = "N1"
    results = {}
    print(f"token-identity probe vs {base_v}: {args.model_label}\n")

    for vname, var in VARIANTS.items():
        classes, dlen = set(), set()
        for tname, body in TEMPLATES.items():
            for wname, bname in PAIRS[:4]:
                for nm in (wname, bname):
                    # BOTH TURNS. This originally tokenised only the user
                    # message, and N4 -- the variant that swaps the two
                    # sentences of the SYSTEM message and changes nothing else
                    # -- was therefore classified IDENTICAL, because the part
                    # of the prompt it edits was never looked at. A probe whose
                    # answer is "the model receives exactly the same integers"
                    # has to have counted all of them.
                    bv = VARIANTS[base_v]
                    b = (tok(api, bv["system"])
                         + tok(api, user_message(bv, nm, body)))
                    o = (tok(api, var["system"])
                         + tok(api, user_message(var, nm, body)))
                    classes.add(classify(b, o))
                    dlen.add(len(o) - len(b))
        cls = ("IDENTICAL" if classes == {"IDENTICAL"}
               else "MIXED" if len(classes) > 1 else classes.pop())
        results[vname] = dict(kind=var["kind"], klass=cls,
                              delta_len=sorted(dlen),
                              note=var.get("note", ""))
        print(f"  {vname:<4}{var['kind']:<10}{cls:<18}"
              f"Δlen {sorted(dlen)}   {var.get('note','')}")

    ident = [v for v, r in results.items() if r["klass"] == "IDENTICAL"]
    print(f"\n{len(ident)} of {len(VARIANTS)} variants are TOKEN-IDENTICAL to the "
          f"baseline: {ident or 'none'}")
    if ident:
        print("  For these the model receives exactly the same integers, so any")
        print("  difference in δ would have to be measurement noise rather than")
        print("  a property of the perturbation.")

    path = OUT / f"token_identity_{args.model_label}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
