"""Elicit each model's perceived prestige of each proxy organisation.

WHY THIS EXISTS. `PROTOCOL.md` §9 requires proxy variants to be matched on
prestige so the manipulation is demographic signal and not quality signal. That
requirement is right, and it is normally satisfied by assertion, which a reviewer
is entitled to disbelieve.

Iso et al. (2025) show what happens when it fails. They compare Ivy League
against HBCUs, women's colleges and lesser-known colleges, which varies prestige
and demographic association together, so a lower score for the HBCU resume is
equally consistent with a prestige effect and a demographic-proxy effect. Their
institution finding cannot distinguish them. Tan et al. hold education constant
precisely to dodge the problem.

We measure it instead. The quantity that actually moves a model's score is not an
organisation's objective selectivity but the model's internal representation of
it, and that is directly elicitable. The elicited rating enters the primary model
as a covariate, so the demographic effect is estimated NET OF perceived prestige.

Two elicitations per organisation:

  PRESTIGE       0-100 selectivity/prestige rating. The covariate.
  RECOGNITION    Free-text "what is the primary focus of X". If a model cannot
                 identify the organisation, its membership cannot be functioning
                 as a demographic proxy for that model, and the cell is reported
                 separately rather than pooled. This is the affiliation analogue
                 of Tan et al.'s demographic-recoverability check: a null is
                 uninterpretable if the signal never arrived.

PRE-REGISTRATION. The prompts below and the covariate specification are fixed
before any confirmatory run, and this script is run BEFORE the bias cells, so the
covariate cannot be chosen after seeing disparities. Neither elicitation mentions
hiring, resumes, candidates or demographics, so the rating cannot be contaminated
by the downstream task.

Run against a llama-server already serving the model under test:
    .venv/Scripts/python.exe paper-a/src/probe_affiliation_prestige.py --model-label <id>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

# Port is a PARAMETER, not a constant. It was hardcoded to 8080 on 2026-07-26
# while run_prestige.py was starting a second stream's servers on 8081, so that
# stream's probes silently queried the OTHER stream's model and two result files
# were written under the wrong model name. Nothing errored; the numbers were
# simply someone else's. See CHANGELOG, session 9.
API = ""  # set in main() from --port
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "prestige"

# (organisation, race_signal, match_key). Mirrors proxies.AFFILIATIONS plus the
# institution family, which needs the same treatment for the same reason.
ORGS: tuple[tuple[str, str, str], ...] = (
    ("the National Association of Black Accountants",        "black",    "affil_fin"),
    ("the Association of Latino Professionals For America",  "hispanic", "affil_fin"),
    ("Ascend, the Pan-Asian leaders organisation",           "asian",    "affil_fin"),
    ("the Institute of Management Accountants",              "neutral",  "affil_fin"),
    ("the National Society of Black Engineers",              "black",    "affil_eng"),
    ("the Society of Hispanic Professional Engineers",       "hispanic", "affil_eng"),
    ("the Society of Asian Scientists and Engineers",        "asian",    "affil_eng"),
    ("the American Society for Quality",                     "neutral",  "affil_eng"),
)

# Five paraphrases, matching the prompt-variant panel used everywhere else.
# Between-variant spread is reported: if a model's prestige rating for an
# organisation is itself unstable across paraphrases, that bounds how far the
# covariate can be trusted, and that bound belongs in the paper.
PRESTIGE_PROMPTS = (
    "On a scale from 0 to 100, how selective and prestigious is membership in {org}? Answer with the number only.",
    "Rate the prestige of membership in {org} from 0 to 100. Give only the number.",
    "How prestigious is it to be a member of {org}? Answer with a number between 0 and 100.",
    "Using a 0 to 100 scale where 100 is the most prestigious, score membership in {org}. Number only.",
    "Assign a prestige score from 0 to 100 to membership in {org}. Respond with just the integer.",
)
RECOGNITION_PROMPT = "In one sentence, what is the primary focus of {org}?"

INT_GRAMMAR = 'root ::= [0-9] | [1-9] [0-9] | "100"'


def chat(prompt: str, max_tokens: int, grammar: str | None = None, timeout: int = 900):
    body = {"model": "local", "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    if grammar:
        body["grammar"] = grammar
    try:
        r = requests.post(API, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"], ""
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {e}"



def served_model(port: int) -> str:
    """Name of the checkpoint a llama.cpp server is serving.

    Builds disagree on the response shape: the Vulkan build returns the OpenAI
    {"data":[...]} form while the CPU build returns {"models":[...]}. Accepting
    only one silently disabled the contamination guard on the other backend,
    which is precisely the failure the guard exists to prevent, so both are
    handled and anything unrecognised raises rather than passing.
    """
    j = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=15).json()
    for key in ("data", "models"):
        if isinstance(j.get(key), list) and j[key]:
            e = j[key][0]
            return e.get("id") or e.get("model") or e.get("name") or ""
    raise RuntimeError(f"unrecognised /v1/models shape: {list(j)[:5]}")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    global API
    API = f"http://127.0.0.1:{args.port}/v1/chat/completions"

    # Fail loudly if the server on that port is not the model we were told to
    # label. The bug this guards against produced perfectly plausible numbers
    # under the wrong name, which is the worst kind of failure.
    try:
        served = served_model(args.port)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"cannot reach a server on port {args.port}: {e}")
    stem = pathlib.Path(served).stem.lower()
    key = args.model_label.lower().replace("-", "").replace(".", "")
    if key[:14] not in stem.lower().replace("-", "").replace(".", ""):
        sys.exit(f"REFUSING TO WRITE: port {args.port} is serving '{pathlib.Path(served).name}' "
                 f"but --model-label says '{args.model_label}'. This is the "
                 f"cross-stream contamination guard.")
    print(f"  [guard] port {args.port} serving {pathlib.Path(served).name}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [(org, race, key, vi)
            for (org, race, key) in ORGS
            for vi in range(len(PRESTIGE_PROMPTS))]

    def prestige_one(j):
        org, race, key, vi = j
        txt, err = chat(PRESTIGE_PROMPTS[vi].format(org=org), 4, INT_GRAMMAR)
        m = re.search(r"\d{1,3}", txt)
        val = int(m.group()) if m else None
        if val is not None and not 0 <= val <= 100:
            val = None
        return {"kind": "prestige", "org": org, "race_signal": race,
                "match_key": key, "variant": vi, "score": val,
                "raw": txt, "error": err}

    def recognition_one(o):
        org, race, key = o
        txt, err = chat(RECOGNITION_PROMPT.format(org=org), 90)
        return {"kind": "recognition", "org": org, "race_signal": race,
                "match_key": key, "raw": txt, "error": err}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        rows = list(ex.map(prestige_one, jobs)) + list(ex.map(recognition_one, ORGS))

    path = OUT_DIR / f"prestige_{args.model_label}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8")

    print(f"Perceived-prestige elicitation: {args.model_label}")
    print(f"{len(ORGS)} organisations x {len(PRESTIGE_PROMPTS)} prompt variants\n")
    print(f"{'organisation':<52}{'signal':<10}{'mean':>6}{'sd':>7}{'range':>9}")
    by_key: dict[str, list] = {}
    for org, race, key in ORGS:
        vals = [r["score"] for r in rows
                if r["kind"] == "prestige" and r["org"] == org and r["score"] is not None]
        if not vals:
            print(f"{org[:50]:<52}{race:<10}{'n/a':>6}{'':>7}{'':>9}")
            continue
        mean = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        by_key.setdefault(key, []).append((race, mean))
        print(f"{org[:50]:<52}{race:<10}{mean:>6.1f}{sd:>7.1f}{max(vals)-min(vals):>9d}")

    print("\nWithin-match-key prestige gap (demographic mean minus neutral):")
    for key, entries in by_key.items():
        neutral = next((m for r, m in entries if r == "neutral"), None)
        if neutral is None:
            print(f"  {key}: no neutral comparator elicited"); continue
        for race, m in sorted(entries):
            if race != "neutral":
                print(f"  {key:<12}{race:<10}{m - neutral:+6.1f} vs neutral ({neutral:.1f})")

    print("\nThis gap is the covariate. It is NOT a bias measurement: it is the")
    print("prestige difference the design must control for so that the demographic")
    print("effect is not confounded with a perceived-quality effect. A large gap")
    print("here means the raw affiliation contrast would have been unfalsifiable,")
    print("which is exactly the problem in Iso et al.'s institution result.")
    print(f"\nRaw: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
