"""Do the two names in a matched pair tokenise to the same length? And if not,
does the measured demographic effect track the difference?

WHY THIS EXISTS. The matched-pair design's whole warrant is that the two prompts
are identical apart from the name. That is true of the CHARACTERS. It is not
necessarily true of the TOKENS, and the token sequence is what the model
receives. On the one pair already probed, "Greg Murphy" is 5 tokens and
"Jermaine Robinson" is 7, so the Black-named prompt is two tokens longer and
every token after the name sits at a different absolute index.

WHY THAT IS NOT A PEDANTIC POINT IN THIS PAPER SPECIFICALLY. Study 5 includes
conditions D8 and D9, which displace the name by exactly one and two tokens
while destroying no delimiter and changing no word. On several checkpoints those
displacements move the measured effect by an amount comparable to the effect
itself. So the paper has already established, in a controlled design, that a
two-token displacement is not free -- and the matched-pair design embeds a
displacement of about that size, correlated with race by construction.

WHAT THIS PROBE MEASURES.

  1. Across all 48 pairs of the factorial grid, the distribution of
     len(tokens(Black name)) - len(tokens(White name)), per model.

  2. Whether that difference PREDICTS the per-pair demographic effect. If names
     that tokenise longer produce systematically different margins, then part of
     what the audit attributes to race is attributable to token length.

Both outcomes are informative. A null means the design's warrant survives a
threat that had not been checked. A positive means correspondence audits of
language models need a length control, which is not something the field
currently does.

    .venv/Scripts/python.exe paper-a/src/probe_name_length.py --model-label <id>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper-a" / "data" / "instrument"


def tok(api: str, text: str) -> list[int]:
    r = requests.post(f"{api}/tokenize", json={"content": text}, timeout=120)
    r.raise_for_status()
    return r.json()["tokens"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    api = f"http://127.0.0.1:{args.port}"
    served = st.assert_serving(args.port, args.model_label)
    print(f"  [guard] port {args.port} serving {served}", flush=True)

    rows = []
    for p in st.NAME_GRID:
        # Tokenised in isolation AND in context: a name's token count can differ
        # depending on what precedes it, and the in-context figure is the one
        # that matters for the prompt the model actually sees.
        w_iso, b_iso = tok(api, p["white"]), tok(api, p["black"])
        w_ctx = tok(api, f"Candidate: {p['white']}\n")
        b_ctx = tok(api, f"Candidate: {p['black']}\n")
        rows.append(dict(
            idx=p["idx"], gender=p["gender"],
            white=p["white"], black=p["black"],
            white_first=p["white_first"], black_first=p["black_first"],
            white_last=p["white_last"], black_last=p["black_last"],
            white_tokens_isolated=len(w_iso), black_tokens_isolated=len(b_iso),
            white_tokens_in_context=len(w_ctx), black_tokens_in_context=len(b_ctx),
            delta_isolated=len(b_iso) - len(w_iso),
            delta_in_context=len(b_ctx) - len(w_ctx),
            white_chars=len(p["white"]), black_chars=len(p["black"]),
            delta_chars=len(p["black"]) - len(p["white"])))

    d = np.array([r["delta_in_context"] for r in rows])
    di = np.array([r["delta_isolated"] for r in rows])
    dc = np.array([r["delta_chars"] for r in rows])
    print(f"\n{args.model_label}: {len(rows)} matched pairs")
    print(f"  token-length difference (Black - White), in context:")
    print(f"     mean {d.mean():+.3f}   median {np.median(d):+.1f}   "
          f"range [{d.min():+d}, {d.max():+d}]")
    print(f"     pairs where the two names tokenise to the SAME length: "
          f"{int((d == 0).sum())} of {len(d)} ({(d == 0).mean():.1%})")
    print(f"     pairs where the Black name is LONGER: "
          f"{int((d > 0).sum())} ({(d > 0).mean():.1%})")
    print(f"  isolated:  mean {di.mean():+.3f}  range [{di.min():+d}, {di.max():+d}]")
    print(f"  characters: mean {dc.mean():+.2f}  "
          f"(correlation with token delta r = {np.corrcoef(dc, d)[0, 1]:+.3f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"name_length_{args.model_label}.json"
    out.write_text(json.dumps(dict(
        model=args.model_label, served=served, n_pairs=len(rows),
        summary=dict(
            mean_delta_in_context=float(d.mean()),
            median_delta_in_context=float(np.median(d)),
            min_delta=int(d.min()), max_delta=int(d.max()),
            frac_equal_length=float((d == 0).mean()),
            frac_black_longer=float((d > 0).mean()),
            mean_delta_isolated=float(di.mean()),
            mean_delta_chars=float(dc.mean()),
            r_chars_vs_tokens=float(np.corrcoef(dc, d)[0, 1])),
        pairs=rows), indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
