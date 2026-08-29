"""What explains the nondeterminism that survives sequential execution?

Study 8 tested the paper's stated mechanism and returned a PARTIAL verdict:
running strictly sequentially raises bitwise agreement between repeats from
47.9% to 93.8% and cuts the per-cell standard deviation by roughly eight times.
Batching is therefore most of the story. It is not all of it -- about six per
cent of cells still disagree with themselves when nothing else is in flight.

THE REMAINING CANDIDATE. llama.cpp reuses the KV cache for whatever prefix a
new prompt shares with the previous one. Which prefix is cached depends on what
ran immediately before, so two calls with identical text can take different
arithmetic paths purely because of what preceded them. That is order dependence
without concurrency, and it predicts exactly the residual Study 8 found.

THE TEST. The same cells, five repeats, strictly sequential, with the prompt
cache DISABLED. If the residual disappears, prompt-cache reuse is the remainder
and the mechanism is fully accounted for. If it survives, something below the
server is responsible and the paper should say the residual is unexplained
rather than invent a cause for it.

This is deliberately narrow: two checkpoints, one wording, and the full template
and name grid of Study 8, so every cell here has a counterpart in the
concurrency-1 arm already measured and the two can be differenced cell by cell.

    .venv/Scripts/python.exe paper-a/src/probe_cache_residual.py --model-label <id>
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402
from experiment_delta_stability import PAIRS, VARIANTS, user_message  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper-a" / "data" / "replicate"
VARIANT = "S1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    api = f"http://127.0.0.1:{args.port}"
    served = st.assert_serving(args.port, args.model_label)
    print(f"  [guard] port {args.port} serving {served}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"nocache_{args.model_label}.jsonl"

    done = set()
    for r in st.read_jsonl(out):
        if r.get("white_margin") is not None and r.get("black_margin") is not None:
            done.add((r["repeat"], r["template"], r["pair"]))
    if done:
        print(f"[resume] {len(done)} cells complete", flush=True)

    var = VARIANTS[VARIANT]
    cells = [(rep, t, i) for rep, t, i in itertools.product(
        range(args.repeats), st.TEMPLATES, range(len(PAIRS)))
        if (rep, t, i) not in done]
    print(f"[{args.model_label}] {len(cells)} cells, sequential, cache OFF",
          flush=True)
    t0 = time.time()

    with out.open("a", encoding="utf-8") as fh:
        for k, (rep, tname, idx) in enumerate(cells, 1):
            body = st.TEMPLATES[tname]
            wname, bname = PAIRS[idx]
            wmsg = user_message(var, wname, body)
            bmsg = user_message(var, bname, body)
            wv, wm, wpy, wpn, we = st.call_chat(api, var["system"], wmsg,
                                                cache_prompt=False)
            bv, bm, bpy, bpn, be = st.call_chat(api, var["system"], bmsg,
                                                cache_prompt=False)
            fh.write(json.dumps({
                "model": args.model_label, "concurrency": 1, "cache": False,
                "repeat": rep, "variant": VARIANT, "template": tname,
                "pair": idx, "white_name": wname, "black_name": bname,
                "white": wv, "black": bv,
                "white_margin": wm, "black_margin": bm,
                "white_p_yes": wpy, "white_p_no": wpn,
                "black_p_yes": bpy, "black_p_no": bpn,
                "white_prompt_sha": st.prompt_sha(var["system"], wmsg),
                "black_prompt_sha": st.prompt_sha(var["system"], bmsg),
                "error": we or be}, ensure_ascii=False) + "\n")
            if k % 40 == 0:
                fh.flush()
                print(f"    {k}/{len(cells)}  {(time.time()-t0)/60:.1f} min",
                      flush=True)

    print(f"[{args.model_label}] done in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
