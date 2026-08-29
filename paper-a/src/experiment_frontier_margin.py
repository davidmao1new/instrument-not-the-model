"""Study 2, unchanged, on frontier API models that return log probabilities.

WHY THIS CLOSES THE PAPER'S LARGEST GAP. Section 10.2 concedes that the panel is
open-weight only and that the stimulus-side results are expected to transfer
without being demonstrated. The obstacle was never money -- the whole run costs
well under a dollar -- it was that the paper's outcome needs the next-token
distribution, and `probe_frontier_api.py` establishes that no Gemini model
available to us returns one. OpenAI's 4o and 4.1 families do. So the wording
study can be run with the SAME outcome, and the comparison is exact rather than
analogical.

WHAT IS HELD IDENTICAL TO STUDY 2: the twelve wordings in their two arms, the
six semantically-null perturbations character for character, the twelve
Bertrand and Mullainathan name pairs in order, the three resume templates, the
posting, the system message, and the outcome -- the renormalised margin
log P(yes) - log P(no) read from the top of the next-token distribution. All of
it is imported from experiment_delta_stability rather than restated, so the two
studies cannot drift apart.

WHAT NECESSARILY DIFFERS, and both are recorded rather than waved at:

  1. NO GRAMMAR. llama.cpp constrains the emission to yes|no; this API has no
     equivalent. The margin does not depend on the constraint -- it is read
     from the unconstrained distribution in both designs -- but the THRESHOLDED
     verdict does, so the verdict here is whichever of the two the model
     actually emitted, and rows where it emitted neither are flagged.
  2. A 20-TOKEN WINDOW, not 100. OpenAI caps `top_logprobs` at 20. That is a
     censoring risk of exactly the kind section 4.6 documents for the second
     task: when one option falls outside the window the margin is unobserved,
     and the loss is informative because it happens where the margin is
     largest. The per-arm censoring rate is recorded so the analysis can test
     whether it differs by arm, which is what would bias a paired estimate.

SPEND. 432 cells x 2 calls x ~122 input tokens is about 105k input tokens per
model, one output token per call. Every run prints its measured token usage and
the script refuses to start without an explicit --budget-usd, so a mistake costs
a stated maximum rather than an open-ended bill.

    OPENAI_API_KEY=... C:/research-toolchain/venv/Scripts/python.exe \
        paper-a/src/experiment_frontier_margin.py --model gpt-4o-mini --budget-usd 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import experiment_delta_stability as ds  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper-a" / "data" / "frontier"
BASE = "https://api.openai.com/v1/chat/completions"
TOP_LOGPROBS = 20

# List price per million input tokens, for the spend guard only. Output is one
# token per call and is ignored. These are recorded in the artifact so a reader
# can see what the estimate assumed rather than trusting a total.
PRICE_IN_PER_M = {
    "gpt-4o-mini": 0.15, "gpt-4o": 2.50,
    "gpt-4.1-mini": 0.40, "gpt-4.1": 2.00, "gpt-4.1-nano": 0.10,
}


def call(key: str, model: str, system: str, user: str, tries: int = 5):
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_completion_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": TOP_LOGPROBS}
    data = json.dumps(body).encode()
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(tries):
        req = urllib.request.Request(BASE, method="POST", data=data, headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return 200, json.load(r), ""
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if e.code in (429, 500, 502, 503) and attempt < tries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return e.code, None, msg
        except Exception as e:  # noqa: BLE001
            if attempt < tries - 1:
                time.sleep(5)
                continue
            return -1, None, f"{type(e).__name__}: {e}"
    return -1, None, "exhausted retries"


def read_margin(payload):
    """Renormalised margin, verdict, and the two masses, from the top window.

    Identical in construction to stimuli._margin_from_top: sum the probability
    on the yes-tokens and on the no-tokens, then take the difference of logs.
    Returns margin None when either side is unrepresented in the window, which
    is the censoring this design has to report rather than silently drop.
    """
    ch = payload["choices"][0]
    content = (ch.get("logprobs") or {}).get("content") or []
    txt = (ch["message"].get("content") or "").strip()
    low = txt.lower()
    verdict = "yes" if low.startswith("y") else ("no" if low.startswith("n") else None)
    p_yes = p_no = 0.0
    if content:
        for t in content[0].get("top_logprobs", []):
            tok = t["token"]
            if tok in ds.YES_TOKENS:
                p_yes += math.exp(t["logprob"])
            elif tok in ds.NO_TOKENS:
                p_no += math.exp(t["logprob"])
    margin = (math.log(p_yes) - math.log(p_no)) if (p_yes > 0 and p_no > 0) else None
    return verdict, margin, p_yes, p_no, txt[:40]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget-usd", type=float, required=True,
                    help="hard ceiling; the run stops before exceeding it")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        sys.exit("set OPENAI_API_KEY in the environment (never in a file)")
    price = PRICE_IN_PER_M.get(args.model)
    if price is None:
        sys.exit(f"no price recorded for {args.model}; add it to PRICE_IN_PER_M "
                 f"so the spend guard means something")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"margin_{args.model}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["variant"], r["template"], r["pair"]))
        print(f"  resuming: {len(done)} cells recorded", flush=True)

    total = len(ds.VARIANTS) * len(ds.TEMPLATES) * len(ds.PAIRS)
    n = 0
    in_tok = 0
    t0 = time.time()
    stopped = None
    with out.open("a", encoding="utf-8") as fh:
        for vname, variant in ds.VARIANTS.items():
            for tname, body in ds.TEMPLATES.items():
                for pi, (wname, bname) in enumerate(ds.PAIRS):
                    n += 1
                    if (vname, tname, pi) in done:
                        continue
                    spent = in_tok / 1e6 * price
                    if spent > args.budget_usd:
                        stopped = f"budget reached: ${spent:.4f}"
                        break
                    wu = ds.user_message(variant, wname, body)
                    bu = ds.user_message(variant, bname, body)
                    c1, r1, e1 = call(key, args.model, variant["system"], wu)
                    time.sleep(args.sleep)
                    c2, r2, e2 = call(key, args.model, variant["system"], bu)
                    time.sleep(args.sleep)
                    if c1 == 200:
                        in_tok += r1.get("usage", {}).get("prompt_tokens", 0)
                    if c2 == 200:
                        in_tok += r2.get("usage", {}).get("prompt_tokens", 0)
                    wv, wm, wpy, wpn, wtxt = read_margin(r1) if c1 == 200 \
                        else (None, None, None, None, "")
                    bv, bm, bpy, bpn, btxt = read_margin(r2) if c2 == 200 \
                        else (None, None, None, None, "")
                    fh.write(json.dumps({
                        "provider": "openai", "model": args.model,
                        "variant": vname, "variant_kind": variant["kind"],
                        "template": tname, "pair": pi,
                        "gender": "female" if pi < 6 else "male",
                        "white_name": wname, "black_name": bname,
                        "white": wv, "black": bv,
                        "white_margin": wm, "black_margin": bm,
                        "white_p_yes": wpy, "white_p_no": wpn,
                        "black_p_yes": bpy, "black_p_no": bpn,
                        "white_raw": wtxt, "black_raw": btxt,
                        "http_white": c1, "http_black": c2,
                        "top_logprobs": TOP_LOGPROBS,
                        "error": (e1 or e2)[:200],
                    }) + "\n")
                    fh.flush()
                    if n % 48 == 0:
                        print(f"  {n}/{total}  {in_tok/1e6*price:.4f} USD  "
                              f"{(time.time()-t0)/60:.1f} min", flush=True)
                if stopped:
                    break
            if stopped:
                break

    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"  input tokens {in_tok:,}   estimated spend "
          f"${in_tok/1e6*price:.4f} at ${price}/M")
    if stopped:
        print(f"  STOPPED EARLY: {stopped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
