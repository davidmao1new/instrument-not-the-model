"""Does the model's own surprisal predict how far a perturbation moves the effect?

The token-healing literature makes a specific claim about why greedy tokenisation
matters: seeing a token conveys not only its embedding but also the information
that the tokeniser did NOT merge it with what follows. A prompt containing
'.\\n' followed by ' \\n' therefore tells the model that the standard merge into
'.\\n\\n' did not occur, a configuration that is close to absent from clean
training text.

If that is the mechanism, it makes a quantitative prediction. The perturbations
that move the demographic effect most should be the ones the model finds most
surprising, because surprisal is the model's own measure of how far a token
sequence sits from what it was trained on.

This probe measures, for each condition, the total surprisal the model assigns to
the perturbed region: the negative log probability of the actual next token at
each edited position, given everything before it. It is the model grading its own
input rather than us asserting what is unusual.

Prediction to be falsified: |Δδ| should increase with added surprisal. A null
correlation would mean off-distribution-ness is not what drives the effect, and
would send us back to position or length.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from experiment_mechanism import CONDITIONS, build  # noqa: E402
from experiment_delta_stability import PAIRS, TEMPLATES  # noqa: E402

OUT = ROOT / "paper-a" / "data" / "mechanism"


def token_surprisals(api: str, text: str) -> list[tuple[int, float]]:
    """Per-token negative log probability under the model, via /completion with
    a zero-token continuation. llama.cpp returns the prompt's own token
    probabilities when asked to echo them."""
    r = requests.post(f"{api}/completion",
                      json={"prompt": text, "n_predict": 0, "n_probs": 1,
                            "post_sampling_probs": False,
                            "return_tokens": True, "echo": True},
                      timeout=300)
    r.raise_for_status()
    j = r.json()
    out = []
    for item in (j.get("prompt_ms_probs") or j.get("completion_probabilities") or []):
        lp = item.get("logprob")
        if lp is not None:
            out.append((item.get("id", -1), -lp))
    return out


def total_logprob(api: str, text: str) -> float | None:
    """Fallback: total prompt log probability, if per-token echo is unavailable.

    Uses the native /completion endpoint. Returns None when the build does not
    expose prompt probabilities, in which case the probe reports NOT AVAILABLE
    rather than substituting something else.
    """
    try:
        r = requests.post(f"{api}/completion",
                          json={"prompt": text, "n_predict": 1, "n_probs": 0,
                                "temperature": 0},
                          timeout=300)
        r.raise_for_status()
        j = r.json()
        for k in ("prompt_logprob", "prompt_logprobs"):
            if k in j and isinstance(j[k], (int, float)):
                return float(j[k])
    except Exception:  # noqa: BLE001
        pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    api = f"http://127.0.0.1:{args.port}"
    OUT.mkdir(parents=True, exist_ok=True)

    body = TEMPLATES["T2_mid"]
    rows = []
    print(f"surprisal probe: {args.model_label}\n")
    print(f"{'cond':<5}{'n_tok':>7}{'total surprisal (nats)':>26}"
          f"{'mean/token':>13}   condition")

    ok = True
    for cond in CONDITIONS:
        per_name = []
        for wn, bn in PAIRS[:6]:
            for nm in (wn, bn):
                text = build(cond, nm, body)
                sur = token_surprisals(api, text)
                if not sur:
                    ok = False
                    break
                per_name.append((len(sur), sum(s for _, s in sur)))
            if not ok:
                break
        if not ok:
            break
        n = sum(a for a, _ in per_name) / len(per_name)
        tot = sum(b for _, b in per_name) / len(per_name)
        rows.append(dict(cond=cond, note=CONDITIONS[cond][0],
                         n_delims=CONDITIONS[cond][1],
                         n_tokens=n, total_surprisal=tot,
                         mean_surprisal=tot / n))
        print(f"{cond:<5}{n:>7.0f}{tot:>26.2f}{tot/n:>13.3f}   {CONDITIONS[cond][0]}")

    if not ok or not rows:
        print("\nPer-token prompt probabilities are NOT AVAILABLE from this "
              "llama.cpp build's /completion endpoint.")
        print("Reporting this rather than substituting a proxy. The surprisal "
              "prediction is therefore left untested here and is flagged as such "
              "in the paper.")
        (OUT / f"surprisal_{args.model_label}.json").write_text(
            json.dumps({"available": False}, indent=2), encoding="utf-8")
        return 0

    base = next(r for r in rows if r["cond"] == "D0")
    print(f"\n{'cond':<5}{'Δ surprisal vs baseline (nats)':>34}   condition")
    for r in rows:
        if r["cond"] == "D0":
            continue
        print(f"{r['cond']:<5}{r['total_surprisal']-base['total_surprisal']:>34.2f}"
              f"   {r['note']}")

    (OUT / f"surprisal_{args.model_label}.json").write_text(
        json.dumps({"available": True, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {(OUT / f'surprisal_{args.model_label}.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
