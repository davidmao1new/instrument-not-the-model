"""What do the null perturbations actually do to the token sequence?

The obvious mechanistic story for whitespace sensitivity, and the one every
reader reaches for first, is that whitespace changes how the CANDIDATE'S NAME is
tokenised: "Jamal" as ["Jam","al"] becomes " Jamal" as a single token, so the
model is fed a different integer and everything downstream follows.

That story cannot be right for this design, and it is worth establishing rather
than asserting. None of the six null perturbations touches the name or its
immediate context. N2 appends a trailing newline at the very end of the prompt,
after the name. N5 inserts a space into blank lines. N3 doubles the spaces after
colons. If the name's local token sequence is byte-identical across all twelve
variants, the name-retokenisation hypothesis is dead and the mechanism must lie
somewhere else.

This probe measures, per variant, using the model's own tokeniser via
llama.cpp's /tokenize endpoint:

  1. total prompt length in tokens
  2. the token index at which the candidate's name begins
  3. the exact token ids covering the name
  4. the number of tokens between the name and the final question mark
  5. whether the name's own tokenisation changed at all

Those five quantities separate the candidate mechanisms:

  - if (3) is constant but δ still moves, name retokenisation is ruled out
  - if (2) moves, absolute position of the name is a live candidate
  - if (4) moves, the name-to-question distance is a live candidate
  - if only (1) moves, the effect is about global sequence length rather than
    anything local to the name
"""
from __future__ import annotations

import json
import pathlib
import sys

import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from experiment_delta_stability import (  # noqa: E402
    PAIRS, POSTING, TEMPLATES, VARIANTS, user_message,
)

OUT = ROOT / "paper-a" / "data" / "mechanism"


def tokenize(api: str, text: str) -> list[int]:
    r = requests.post(f"{api}/tokenize", json={"content": text}, timeout=120)
    r.raise_for_status()
    return r.json()["tokens"]


def detokenize(api: str, toks: list[int]) -> str:
    r = requests.post(f"{api}/detokenize", json={"tokens": toks}, timeout=120)
    r.raise_for_status()
    return r.json()["content"]


def find_sub(hay: list[int], needle: list[int]) -> int:
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i
    return -1


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model-label", required=True)
    args = ap.parse_args()
    api = f"http://127.0.0.1:{args.port}"
    OUT.mkdir(parents=True, exist_ok=True)

    # One template, one name, all twelve variants. Holding everything else fixed
    # isolates what the perturbation itself does to the token stream.
    tname = "T2_mid"
    body = TEMPLATES[tname]
    wname, bname = PAIRS[8]          # "Greg Murphy" / "Jermaine Robinson"

    rows = []
    print(f"tokeniser probe: {args.model_label}, template {tname}\n")
    print(f"{'variant':<8}{'kind':<10}{'n_tok':>7}{'name@':>7}{'name_len':>9}"
          f"{'to_end':>8}   name tokens")
    for vname, var in VARIANTS.items():
        for label, nm in (("white", wname), ("black", bname)):
            text = user_message(var, nm, body)
            toks = tokenize(api, text)
            # Tokenise the bare name in the same left context the prompt gives
            # it, so the comparison is like-for-like rather than context-free.
            probe = tokenize(api, f"Name: {nm}\n")
            # strip the "Name:" prefix tokens by finding the longest suffix of
            # `probe` that occurs in `toks`
            name_toks, at = [], -1
            for start in range(len(probe)):
                cand = probe[start:]
                pos = find_sub(toks, cand)
                if pos >= 0 and len(cand) >= 2:
                    name_toks, at = cand, pos
                    break
            rows.append(dict(
                variant=vname, kind=var["kind"], side=label, name=nm,
                n_tokens=len(toks), name_at=at, name_len=len(name_toks),
                tokens_after_name=(len(toks) - at - len(name_toks)) if at >= 0 else None,
                name_tokens=name_toks,
                note=var.get("note", ""),
            ))
            if label == "white":
                r = rows[-1]
                print(f"{vname:<8}{var['kind']:<10}{r['n_tokens']:>7}"
                      f"{r['name_at']:>7}{r['name_len']:>9}"
                      f"{r['tokens_after_name']:>8}   {name_toks}")

    path = OUT / f"tokenization_{args.model_label}.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # ---- the decisive summary ----
    print()
    w = [r for r in rows if r["side"] == "white"]
    name_sets = {tuple(r["name_tokens"]) for r in w}
    print(f"distinct name tokenisations across all 12 variants: {len(name_sets)}")
    if len(name_sets) == 1:
        print("  -> the name's own token sequence is IDENTICAL in every variant.")
        print("     Name retokenisation is ruled out as the mechanism.")
    lens = {r["variant"]: r["n_tokens"] for r in w}
    ats = {r["variant"]: r["name_at"] for r in w}
    ends = {r["variant"]: r["tokens_after_name"] for r in w}
    print(f"prompt length in tokens : min {min(lens.values())}, max {max(lens.values())}, "
          f"distinct {len(set(lens.values()))}")
    print(f"name start index        : min {min(ats.values())}, max {max(ats.values())}, "
          f"distinct {len(set(ats.values()))}")
    print(f"tokens after the name   : min {min(ends.values())}, max {max(ends.values())}, "
          f"distinct {len(set(ends.values()))}")
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
