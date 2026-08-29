"""Is the outcome measure reading the model's answer, or reading the grammar?

WHY. The paper's primary measure is a margin, log P(yes) - log P(no), where each
probability is a SUM over surface forms of the word taken from the unconstrained
next-token distribution. Two things could be wrong with that and neither would
announce itself:

  1. COVERAGE. YES_TOKENS and NO_TOKENS are hand-written string sets. Any
     affirmative or negative surface form the tokenizer emits that is not in
     those sets is silently dropped from the sum. Different tokenizers spell
     the same word differently -- SentencePiece prefixes a word-initial marker,
     byte-level BPE does not -- so a set that is complete for one model can be
     lossy for another, which would make the margin non-comparable across
     exactly the comparison the paper is built on.

  2. ADDRESSED MASS. If the model overwhelmingly wants to emit something that is
     neither yes nor no, then both instruments are describing a corner of the
     distribution the model was never going to visit. The grammar forces an
     answer; that answer is a fact about the grammar as much as about the model.

This probe measures both, plus the agreement between the three ways the same
call can be turned into a verdict:

     GRAMMAR     the grammar-constrained emission, what a structured-output
                 deployment would see
     ARGMAX      whichever single yes-form or no-form token has the highest
                 probability
     MASS        the sign of the summed-mass margin, the paper's measure

Study 2 shows GRAMMAR and MASS disagreeing on 21% of Llama-2-7B-chat calls and
26% of Mistral-v0.1 calls. That is either a real property of the instrument or a
bug in the token sets, and the paper cannot report the number until it knows
which.

    .venv/Scripts/python.exe paper-a/src/probe_token_coverage.py --model-label <id>
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import unicodedata

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402
from experiment_delta_stability import VARIANTS, user_message  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper-a" / "data" / "instrument"

# A token counts as an affirmative or negative SURFACE FORM if, once stripped of
# whitespace and of the marker characters tokenizers use for a word boundary,
# it spells the word or a prefix that can only continue into it. Deliberately
# generous: the point is to find what the hand-written sets MISS, so a false
# positive here is informative and a false negative is not.
_MARKERS = "▁ĠĊ "          # SentencePiece, byte-level BPE, space


def canon(tok: str) -> str:
    s = unicodedata.normalize("NFKC", tok)
    return s.strip(_MARKERS).strip().lower()


YES_FORMS = {"yes", "y", "ye", "yeah", "yep", "affirmative", "true"}
NO_FORMS = {"no", "n", "nope", "negative", "false"}


def classify(tok: str) -> str | None:
    c = canon(tok)
    if c in YES_FORMS:
        return "yes"
    if c in NO_FORMS:
        return "no"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n-cells", type=int, default=48)
    # Without this the probe re-ran on every suite restart. Five probes at two
    # to four minutes each is twenty minutes of work redone, which is longer
    # than the watchdog's stall window, so the restart that triggered it
    # guaranteed the next one.
    ap.add_argument("--force", action="store_true",
                    help="re-run even if a complete output already exists")
    args = ap.parse_args()

    existing = OUT_DIR / f"token_coverage_{args.model_label}.json"
    if existing.exists() and not args.force:
        try:
            n = len(json.loads(existing.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            n = 0
        if n >= args.n_cells:
            print(f"  [skip] {existing.name} already has {n} calls; "
                  f"pass --force to re-run", flush=True)
            return 0

    api = f"http://127.0.0.1:{args.port}"
    served = st.assert_serving(args.port, args.model_label)
    print(f"  [guard] port {args.port} serving {served}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = []
    for v in ("S1", "S3", "N2", "N5"):
        for t in ("T1_strong", "T3_marginal"):
            for p in st.NAME_GRID[:args.n_cells // 8]:
                for side in ("white", "black"):
                    cells.append((v, t, p[side]))
    cells = cells[:args.n_cells]

    rows = []
    for v, t, name in cells:
        var = VARIANTS[v]
        msg = user_message(var, name, st.TEMPLATES[t])
        body = {"model": "local", "temperature": 0, "max_tokens": 1,
                "grammar": st.VERDICT_GRAMMAR, "logprobs": True,
                "top_logprobs": st.TOP_LOGPROBS,
                "messages": [{"role": "system", "content": var["system"]},
                             {"role": "user", "content": msg}]}
        try:
            j = requests.post(f"{api}/v1/chat/completions", json=body,
                              timeout=600).json()["choices"][0]
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {type(e).__name__}: {e}", flush=True)
            continue
        emitted = (j["message"]["content"] or "").strip().lower()
        top = j.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])

        set_yes = set_no = 0.0        # what the paper's hand-written sets capture
        gen_yes = gen_no = 0.0        # what a generous canonical classifier captures
        total = 0.0
        missed = []
        best_yes = best_no = (-1.0, "")
        for e in top:
            tok, lp = e["token"], e["logprob"]
            pr = math.exp(lp)
            total += pr
            if tok in st.YES_TOKENS:
                set_yes += pr
            elif tok in st.NO_TOKENS:
                set_no += pr
            cls = classify(tok)
            if cls == "yes":
                gen_yes += pr
                if pr > best_yes[0]:
                    best_yes = (pr, tok)
                if tok not in st.YES_TOKENS:
                    missed.append((tok, pr, "yes"))
            elif cls == "no":
                gen_no += pr
                if pr > best_no[0]:
                    best_no = (pr, tok)
                if tok not in st.NO_TOKENS:
                    missed.append((tok, pr, "no"))

        grammar_v = "yes" if emitted.startswith("y") else (
            "no" if emitted.startswith("n") else None)
        argmax_v = ("yes" if best_yes[0] > best_no[0] else "no") \
            if max(best_yes[0], best_no[0]) > 0 else None
        mass_v = ("yes" if set_yes > set_no else "no") \
            if (set_yes > 0 and set_no > 0) else None

        rows.append(dict(
            variant=v, template=t, name=name,
            grammar=grammar_v, argmax=argmax_v, mass=mass_v,
            set_yes=set_yes, set_no=set_no, gen_yes=gen_yes, gen_no=gen_no,
            addressed_mass=set_yes + set_no, top_total_mass=total,
            missed=[{"token": m[0], "p": m[1], "class": m[2]} for m in missed],
            best_yes_token=best_yes[1], best_no_token=best_no[1]))

    path = OUT_DIR / f"token_coverage_{args.model_label}.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    n = len(rows)
    if not n:
        sys.exit("no rows")
    miss_rows = [r for r in rows if r["missed"]]
    lost = [max(m["p"] for m in r["missed"]) for r in miss_rows]
    agree = lambda a, b: sum(1 for r in rows if r[a] and r[b] and r[a] == r[b])  # noqa: E731
    both = lambda a, b: sum(1 for r in rows if r[a] and r[b])  # noqa: E731

    print(f"\nTOKEN COVERAGE AUDIT: {args.model_label}   ({n} calls)")
    print(f"  top-{st.TOP_LOGPROBS} captured mass, mean            "
          f"{sum(r['top_total_mass'] for r in rows)/n:.4f}")
    print(f"  mass addressed by the yes/no sets, mean  "
          f"{sum(r['addressed_mass'] for r in rows)/n:.4f}")
    print(f"  calls with a yes/no surface form MISSED by the hand-written sets: "
          f"{len(miss_rows)}/{n}")
    if lost:
        print(f"    largest single missed probability: {max(lost):.5f}")
        seen = {}
        for r in miss_rows:
            for m in r["missed"]:
                seen[m["token"]] = max(seen.get(m["token"], 0.0), m["p"])
        for tok, pr in sorted(seen.items(), key=lambda x: -x[1])[:10]:
            print(f"      {tok!r:14} up to p={pr:.5f}")
    print("  verdict agreement between the three readings:")
    for a, b in (("grammar", "mass"), ("grammar", "argmax"), ("argmax", "mass")):
        d = both(a, b)
        print(f"    {a:8} vs {b:8} {agree(a,b)}/{d}"
              f"  ({(agree(a,b)/d if d else float('nan')):.1%} agree)")
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
