"""The wording study on a frontier API model, with the outcome it will give us.

WHY THIS EXISTS AND WHAT IT CAN AND CANNOT SHOW. Section 10.2 concedes that the
panel is open-weight only and that the stimulus-side results are expected to
transfer without being demonstrated. `probe_frontier_api.py` establishes why the
demonstration is hard: of fourteen Gemini text models probed, NONE returns log
probabilities to this key, so the paper's primary outcome -- the renormalised
margin log P(yes) - log P(no) -- cannot be computed on any of them. That is a
fact about the product surface, not about budget: the whole wording study costs
under a dollar at list price.

So this measures what an API WILL give: the thresholded verdict.

WHAT THAT BUYS, AND WHAT IT DOES NOT. It does not buy a demographic effect
worth reporting. Section 3 explains why the binary is not the primary outcome
here -- when a template sits far from a model's threshold every pair is
concordant, discordance goes to zero, and the contrast is unmeasurable at any
feasible n. Twelve name pairs on a binary outcome is exactly that regime, and
the discordant-pair count is recorded so a reader can see whether there was any
information in it rather than take a number on trust.

What it does buy is the simpler and more basic quantity: does the ACCEPTANCE
RATE -- the thing a demographic effect is a difference of -- move when the
question is reworded without being changed? That needs no log probabilities, it
has full power on 432 binary observations, and if it moves on a frontier model
then the instrument variance this paper measures is not an artefact of small
open-weight checkpoints.

DESIGN. The twelve Study 2 wordings, the three Study 2 résumé templates and six
of its name pairs, giving 12 x 3 x 12 = 432 calls per model -- the same 36
observations per wording that Study 2 has. Same posting, same system message,
same question. Only the model and the outcome differ.

TWO API FACTS THIS DESIGN HAD TO ACCOMMODATE, both measured rather than assumed:
  * a one-token output budget returns nothing on thinking models, because the
    reasoning consumes it and the reply never arrives; and thinkingBudget = 0
    is rejected. The budget here is 24 tokens.
  * replies are markdown ("**Yes.**"), so the verdict parser strips it.

    GEMINI_API_KEY=... C:/research-toolchain/venv/Scripts/python.exe \
        paper-a/src/experiment_frontier_wording.py --model gemini-3.1-flash-lite
"""
from __future__ import annotations

import argparse
import json
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
BASE = "https://generativelanguage.googleapis.com/v1beta"
MAX_OUT = 24

# Six of Study 2's twelve pairs, gender-balanced, taken in order. Six rather
# than twelve because the free tier is rate- and quota-limited and the
# acceptance rate -- the quantity this arm can actually measure -- needs
# observations per WORDING, not per name.
PAIRS = ds.PAIRS[:3] + ds.PAIRS[6:9]


def verdict_of(payload) -> tuple[str | None, str]:
    cand = (payload.get("candidates") or [{}])[0]
    txt = "".join(p.get("text", "")
                  for p in (cand.get("content") or {}).get("parts", []) or [])
    t = txt.strip().lower().lstrip("*_# \t\n")
    v = "yes" if t.startswith("y") else ("no" if t.startswith("n") else None)
    return v, txt.strip()[:80]


def call(key: str, model: str, system: str, user: str, tries: int = 5):
    """One call, retrying on 429 with a widening wait.

    The free tier is a few requests per minute. A 429 here is a scheduling
    fact, not a failure, so it is waited out rather than recorded as missing
    data -- silently dropping rate-limited cells would bias the sample toward
    whatever the limiter happened to let through.
    """
    body = {"contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"temperature": 0, "maxOutputTokens": MAX_OUT}}
    req_data = json.dumps(body).encode()
    for attempt in range(tries):
        req = urllib.request.Request(
            f"{BASE}/models/{model}:generateContent", method="POST",
            data=req_data,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return 200, json.load(r), ""
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if e.code == 429 and attempt < tries - 1:
                time.sleep(20 * (attempt + 1))
                continue
            return e.code, None, msg
        except Exception as e:  # noqa: BLE001
            if attempt < tries - 1:
                time.sleep(10)
                continue
            return -1, None, f"{type(e).__name__}: {e}"
    return -1, None, "exhausted retries"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sleep", type=float, default=4.5,
                    help="seconds between calls; the free tier is a few RPM")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit("set GEMINI_API_KEY in the environment (never in a file)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"wording_{args.model}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["variant"], r["template"], r["pair"]))
        print(f"  resuming: {len(done)} cells recorded", flush=True)

    total = len(ds.VARIANTS) * len(ds.TEMPLATES) * len(PAIRS)
    n = 0
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for vname, variant in ds.VARIANTS.items():
            for tname, body in ds.TEMPLATES.items():
                for pi, (wname, bname) in enumerate(PAIRS):
                    n += 1
                    if (vname, tname, pi) in done:
                        continue
                    wu = ds.user_message(variant, wname, body)
                    bu = ds.user_message(variant, bname, body)
                    c1, r1, e1 = call(key, args.model, variant["system"], wu)
                    time.sleep(args.sleep)
                    c2, r2, e2 = call(key, args.model, variant["system"], bu)
                    time.sleep(args.sleep)
                    wv, wtxt = verdict_of(r1) if c1 == 200 else (None, "")
                    bv, btxt = verdict_of(r2) if c2 == 200 else (None, "")
                    fh.write(json.dumps({
                        "provider": "google", "model": args.model,
                        "variant": vname, "variant_kind": variant["kind"],
                        "template": tname, "pair": pi,
                        "gender": "female" if pi < 3 else "male",
                        "white_name": wname, "black_name": bname,
                        "white": wv, "black": bv,
                        "white_raw": wtxt, "black_raw": btxt,
                        "http_white": c1, "http_black": c2,
                        "error": (e1 or e2)[:200],
                    }) + "\n")
                    fh.flush()
                    if n % 24 == 0:
                        print(f"  {n}/{total}  {(time.time()-t0)/60:.1f} min",
                              flush=True)
    print(f"wrote {out.relative_to(ROOT)}  ({total} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
