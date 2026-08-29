"""Does the cache-off measurement reproduce across SESSIONS, not just within one?

THE GAP THIS CLOSES. Studies 8 and 9 established that repeats of a cell agree
bitwise once requests are sequential and prompt-cache reuse is off. Every arm of
both studies is a repeat INSIDE ONE SERVER LAUNCH. The paper separately reports a
second and larger failure of reproducibility -- identical prompts re-measured in
a different process on a different day agree on 0 of 504 cells, with differences
up to 1.15 log-odds -- and nothing in Studies 8 or 9 speaks to it. Concluding
from them that the nondeterminism is "fully accounted for" therefore claims more
than was measured, since the cross-session arm was never re-run with the controls
applied.

THE TEST. Re-measure exactly the cells of `nocache_<model>.jsonl` in a FRESH
server process, still sequential, still with the cache off, and difference them
cell by cell against the stored run.

  identical            the controls fix cross-session reproducibility too, and
                       the paper's claim is exactly right
  still disagreeing    within-run determinism does not imply cross-session
                       determinism; something about process state -- memory
                       layout, thread scheduling, driver version, GPU work
                       partitioning -- survives both controls, and the paper
                       must say the cross-session component is unexplained

Either answer is publishable. The one thing not permitted is to keep asserting
the strong version without having run this.

    .venv/Scripts/python.exe paper-a/src/probe_cache_crosssession.py --model-label <id>
"""
from __future__ import annotations

import argparse
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
    args = ap.parse_args()

    api = f"http://127.0.0.1:{args.port}"
    served = st.assert_serving(args.port, args.model_label)
    print(f"  [guard] port {args.port} serving {served}", flush=True)

    src = OUT_DIR / f"nocache_{args.model_label}.jsonl"
    if not src.exists():
        sys.exit(f"no first-session run to compare against: {src}")
    # One row per cell from the FIRST session. Those five repeats agreed
    # bitwise, so any of them is the session's value; take the first.
    first = {}
    for r in st.read_jsonl(src):
        k = (r["template"], r["pair"])
        if k not in first and r.get("white_margin") is not None:
            first[k] = r
    print(f"[{args.model_label}] {len(first)} cells to re-measure in a fresh process",
          flush=True)

    out = OUT_DIR / f"nocache_session2_{args.model_label}.jsonl"
    done = {(r["template"], r["pair"]) for r in st.read_jsonl(out)
            if r.get("white_margin") is not None}
    todo = [k for k in sorted(first) if k not in done]
    if done:
        print(f"[resume] {len(done)} already done", flush=True)

    var = VARIANTS[VARIANT]
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for n, (tname, idx) in enumerate(todo, 1):
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
                "session": 2, "variant": VARIANT, "template": tname,
                "pair": idx, "white_name": wname, "black_name": bname,
                "white": wv, "black": bv,
                "white_margin": wm, "black_margin": bm,
                "white_prompt_sha": st.prompt_sha(var["system"], wmsg),
                "black_prompt_sha": st.prompt_sha(var["system"], bmsg),
                "error": we or be}, ensure_ascii=False) + "\n")
            if n % 12 == 0:
                fh.flush()
                print(f"    {n}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"[{args.model_label}] done in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
