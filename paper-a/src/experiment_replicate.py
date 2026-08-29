"""Study 8. Is the nondeterminism caused by batching? A designed replicate.

WHY THIS EXISTS. The noise floor -- the thing that lets the paper say the
wording effect is 5.7 to 7.0 times the arithmetic rather than merely "not
arithmetic" -- currently rests on an ACCIDENT. Two of the twelve Study 2
wordings happen to be byte-identical, so they happen to constitute a replicate
nobody designed. That is a fragile foundation for a headline, and it gives only
two measurements per cell.

It also leaves the causal claim unsupported. The paper asserts that batched
inference is responsible: four server slots and four concurrent clients mean a
prompt is matrix-multiplied alongside whatever else is in flight, and
floating-point reduction order is not invariant to batch composition. That is a
plausible mechanism and it has not been tested.

THE TEST. Run the identical cell R times at two client concurrency levels.

    CONCURRENCY 4   requests overlap, so a prompt shares a batch with whatever
                    else is in flight, and batch composition varies run to run.

    CONCURRENCY 1   requests are strictly sequential, so every prompt is
                    processed alone and batch composition is fixed.

The prediction is sharp and was written down before the run: at concurrency 1
the repeats should agree BITWISE, and at concurrency 4 they should not. If both
are nondeterministic the batching explanation is wrong and something else is
responsible, which we would then have to find. If both are deterministic, the
accidental replicate was measuring something else entirely and the noise-floor
section needs rewriting.

Either way the study also yields what the accident could not: a proper
per-cell variance estimate from R repeats rather than 2, at the concurrency the
rest of the paper actually used.

DESIGN
    1 wording (the baseline) x 3 résumés x 12 name pairs x R repeats
    x 2 concurrency levels = 360 cells = 720 calls per model at R = 5.

Resumable per (concurrency, repeat, template, pair).
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402
from experiment_delta_stability import PAIRS, VARIANTS, user_message  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper-a" / "data" / "replicate"

VARIANT = "S1"          # the baseline wording, identical to N1
CONCURRENCIES = (4, 1)  # 4 is what every other study used


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
    out = OUT_DIR / f"rep_{args.model_label}.jsonl"

    done: set[tuple[int, int, str, int]] = set()
    for r in st.read_jsonl(out):
        if r.get("white_margin") is not None and r.get("black_margin") is not None:
            done.add((r["concurrency"], r["repeat"], r["template"], r["pair"]))
    if done:
        print(f"[resume] {len(done)} cells complete", flush=True)

    var = VARIANTS[VARIANT]
    t0 = time.time()
    total_written = 0

    with out.open("a", encoding="utf-8") as fh:
        for conc in CONCURRENCIES:
            cells = [(rep, t, i) for rep, t, i in itertools.product(
                range(args.repeats), st.TEMPLATES, range(len(PAIRS)))
                if (conc, rep, t, i) not in done]
            if not cells:
                print(f"[concurrency {conc}] already complete", flush=True)
                continue
            print(f"[{args.model_label}] concurrency {conc}: {len(cells)} cells "
                  f"({len(cells)*2} calls)", flush=True)

            def run_cell(cell, _conc=conc):
                rep, tname, idx = cell
                body = st.TEMPLATES[tname]
                wname, bname = PAIRS[idx]
                wmsg = user_message(var, wname, body)
                bmsg = user_message(var, bname, body)
                # Every repeat of a cell must carry the same prompt hash. If two
                # repeats of what should be one cell differ here, the design
                # broke rather than the arithmetic, and the noise estimate this
                # study exists to produce would be measuring the wrong thing.
                wsha = st.prompt_sha(var["system"], wmsg)
                bsha = st.prompt_sha(var["system"], bmsg)
                wv, wm, wpy, wpn, we = st.call_chat(api, var["system"], wmsg)
                bv, bm, bpy, bpn, be = st.call_chat(api, var["system"], bmsg)
                return {"model": args.model_label, "concurrency": _conc,
                        "white_prompt_sha": wsha, "black_prompt_sha": bsha,
                        "repeat": rep, "variant": VARIANT,
                        "template": tname, "pair": idx,
                        "white_name": wname, "black_name": bname,
                        "white": wv, "black": bv,
                        "white_margin": wm, "black_margin": bm,
                        "white_p_yes": wpy, "white_p_no": wpn,
                        "black_p_yes": bpy, "black_p_no": bpn,
                        "error": we or be}

            with ThreadPoolExecutor(max_workers=conc) as ex:
                for k, row in enumerate(ex.map(run_cell, cells), 1):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_written += 1
                    if k % 60 == 0:
                        fh.flush()
                        el = time.time() - t0
                        print(f"    conc {conc}: {k}/{len(cells)}  {el/60:.1f} min",
                              flush=True)
            fh.flush()

    print(f"[{args.model_label}] done in {(time.time()-t0)/60:.1f} min, "
          f"{total_written} cells -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
