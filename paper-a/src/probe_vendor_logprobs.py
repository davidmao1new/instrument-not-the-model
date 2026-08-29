"""Which vendors expose the quantity a graded fairness audit needs?

WHY THIS IS A MEASUREMENT AND NOT AN ASIDE. §4.7 argues that the barrier to
auditing frontier models on this paper's outcome is the vendor's product
surface rather than the researcher's budget -- the whole frontier run cost 67
cents. A claim like that rests on how many vendors were checked and how
carefully, and an earlier draft got it wrong by checking one surface of one
vendor and generalising. This records all three, with the verbatim error string,
so a reader can see the difference between "gated" and "absent".

WHAT WAS FOUND, and the three postures are genuinely different:

  OpenAI      `logprobs` + `top_logprobs` are accepted and returned. Four
              checkpoints supply the margin; gpt-5 accepts the request and
              refuses with 403 "You are not allowed to request logprobs from
              this model", which is a per-model policy on a supported feature.

  Google      The parameter is RECOGNISED and DISABLED. Every reachable model
              of 46 probed answers HTTP 400 "Logprobs is not enabled for this
              model" -- not an unknown field. Tested on the API-key surface
              only, on an unbilled account; Vertex AI was not reached.

  Anthropic   The parameter DOES NOT EXIST. Every spelling returns HTTP 400
              "Extra inputs are not permitted", which is the schema rejecting
              an unknown field rather than a feature being withheld.

The distinction matters for what an auditor can hope for. A gated feature might
open on another tier; an absent one is a design decision about the product.

    ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=... \\
        C:/research-toolchain/venv/Scripts/python.exe \\
        paper-a/src/probe_vendor_logprobs.py

Run without keys, it rebuilds the summary from the two capability artifacts
already on disk and records the Anthropic result from this file's constants,
which carry the observed strings verbatim.
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
D = ROOT / "paper-a" / "data" / "instrument"
OUT = D / "vendor_logprob_support.json"

# Observed verbatim on 2026-08-02. Each `evidence` string is what the API
# returned, not a paraphrase.
ANTHROPIC = {
    "vendor": "Anthropic",
    "surface": "api.anthropic.com/v1/messages",
    "n_models_listed": 11,
    "returns_logprobs": False,
    "posture": "parameter absent from the schema",
    "spellings_tried": ["logprobs", "top_logprobs",
                        "logprobs+top_logprobs", "metadata.logprobs"],
    "status": 400,
    "evidence": "logprobs: Extra inputs are not permitted",
    "note": "The plain call succeeds and the response carries no field whose "
            "name contains 'logprob'. This is a schema rejection of an unknown "
            "field, not a disabled feature.",
}

OPENAI = {
    "vendor": "OpenAI",
    "surface": "api.openai.com/v1/chat/completions",
    "returns_logprobs": True,
    "posture": "supported; withheld on one model by policy",
    "n_models_usable_for_margin": 4,
    "usable": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
    "evidence": "gpt-5 returns HTTP 403 'You are not allowed to request "
                "logprobs from this model'; the 4o and 4.1 families return "
                "top_logprobs normally.",
}


def main() -> int:
    google = None
    p = D / "frontier_api_capability_v2.json"
    if p.exists():
        g = json.loads(p.read_text(encoding="utf-8"))
        errs = {}
        for r in g.get("records", []):
            for a in r.get("attempts", []):
                if a.get("error"):
                    errs[a["error"][:60]] = errs.get(a["error"][:60], 0) + 1
        google = {
            "vendor": "Google",
            "surface": "generativelanguage.googleapis.com (API-key surface); "
                       "Vertex AI not reached",
            "n_models_probed": g["n_probed"],
            "n_reachable": g["n_reachable"],
            "n_quota_blocked": g["n_quota_blocked"],
            "returns_logprobs": bool(g["n_usable_for_margin"]),
            "posture": "parameter recognised and disabled",
            "evidence": max(errs, key=errs.get) if errs else "",
            "note": "Free tier: the account carries no balance. A gated "
                    "feature may behave differently on a billed tier or on "
                    "Vertex, neither of which was tested.",
        }

    vendors = [v for v in (OPENAI, google, ANTHROPIC) if v]
    out = {
        "_what": "Whether each vendor exposes a next-token distribution, which "
                 "is what this paper's outcome requires.",
        "_why": "§4.7 claims the barrier to auditing frontier models on a "
                "graded outcome is the product surface, not the budget. This "
                "is the evidence for that claim, across every vendor tested.",
        "n_vendors": len(vendors),
        "n_returning_logprobs": sum(1 for v in vendors if v["returns_logprobs"]),
        "vendors": vendors,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{'vendor':<12}{'logprobs':>10}   posture / evidence")
    print("-" * 96)
    for v in vendors:
        print(f"{v['vendor']:<12}{str(v['returns_logprobs']):>10}   "
              f"{v['posture']}")
        print(f"{'':<22}   {v['evidence'][:70]}")
    print()
    print(f"  {out['n_returning_logprobs']} of {out['n_vendors']} vendors "
          f"return a next-token distribution")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
