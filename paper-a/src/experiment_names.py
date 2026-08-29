"""Study 4. Is the name list part of the instrument too?

THE GAP (docs/GAPS.md G1). Studies 2 and 3 measure a demographic effect from
twelve hand-fixed name pairs. Every audit in this literature fixes its own list,
and the lists barely overlap: Bertrand and Mullainathan's 2004 newspaper-ad
names, Rosenman et al.'s voter-file names, hand-picked sets, SSA frequency
draws. If the measured effect depends materially on WHICH names were drawn, then
the name list is a researcher degree of freedom of the same kind as the wording,
and a more consequential one, because it varies across papers by construction
while the wording varies only by accident.

Gaddis (2017) gives the reason to expect this: BM2004's names carry perceived
socioeconomic status alongside race, unevenly, so different names within a race
carry different amounts and different kinds of signal. No LLM audit has
decomposed its effect into a race component and a which-names-you-picked
component.

DESIGN. A factorial name grid rather than a hand-fixed list.

    6 first names x 4 surnames x 2 genders, per race  =  48 names per race
    index-paired within gender                        =  48 matched pairs
    x 12 wording variants (6 semantic, 6 null)        <- identical to Study 2
    x 2 resume templates (strongest and weakest)
    = 1,152 cells = 2,304 calls per model

The factorial is what makes this worth running. Forty-eight hand-picked full
names would give the same n and would not let first-name and surname effects be
separated. Crossed, they can be: the analysis fits the per-name margin with race
as a fixed effect and first name and surname as crossed random effects nested
within race, putting sigma_first, sigma_surname, sigma_variant and beta on one
scale where they can be compared directly.

The wording variants are IMPORTED from experiment_delta_stability rather than
restated, so Study 4's wordings are byte-identical to Study 2's. That is what
makes the overlapping cells a valid reproducibility check.

WHAT WOULD FALSIFY THE PAPER'S OWN THESIS. If sigma_first turns out negligible
next to beta, then names are a solved problem, the wording result stands alone,
and this study is reported as a null that strengthens Studies 2 and 3 by
elimination. That outcome is written down here before the run.

Resumable per (variant, template, pair). Run against a live llama-server:
    .venv/Scripts/python.exe paper-a/src/experiment_names.py --model-label <id>
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
from experiment_delta_stability import VARIANTS, user_message  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper-a" / "data" / "names"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--templates", default=None,
                    help="comma-separated; default is the two extremes")
    args = ap.parse_args()
    templates = (tuple(args.templates.split(",")) if args.templates
                 else st.TEMPLATES_2)
    for t in templates:
        if t not in st.TEMPLATES:
            sys.exit(f"unknown template {t!r}; have {sorted(st.TEMPLATES)}")

    api = f"http://127.0.0.1:{args.port}"
    served = st.assert_serving(args.port, args.model_label)
    print(f"  [guard] port {args.port} serving {served}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"names_{args.model_label}.jsonl"

    done: set[tuple[str, str, int]] = set()
    for r in st.read_jsonl(out):
        if r.get("white_margin") is not None and r.get("black_margin") is not None:
            done.add((r["variant"], r["template"], r["pair"]))
    if done:
        print(f"[resume] {len(done)} cells complete", flush=True)

    want = set(args.variants.split(",")) if args.variants else set(VARIANTS)
    cells = [(v, t, p["idx"]) for v, t, p in itertools.product(
        [v for v in VARIANTS if v in want], templates, st.NAME_GRID)
        if (v, t, p["idx"]) not in done]

    print(f"[{args.model_label}] {len(cells)} cells to run ({len(cells)*2} calls), "
          f"{args.concurrency}-way concurrent", flush=True)
    t0 = time.time()
    by_idx = {p["idx"]: p for p in st.NAME_GRID}

    def run_cell(cell):
        vname, tname, idx = cell
        var = VARIANTS[vname]
        body = st.TEMPLATES[tname]
        p = by_idx[idx]
        wmsg = user_message(var, p["white"], body)
        bmsg = user_message(var, p["black"], body)
        # Provenance: a hash of exactly what was sent, so any row can be checked
        # against the current stimulus definitions rather than trusted.
        wsha = st.prompt_sha(var["system"], wmsg)
        bsha = st.prompt_sha(var["system"], bmsg)
        wv, wm, wpy, wpn, we = st.call_chat(api, var["system"], wmsg)
        bv, bm, bpy, bpn, be = st.call_chat(api, var["system"], bmsg)
        return {"model": args.model_label, "variant": vname,
                "variant_kind": var["kind"], "template": tname, "pair": idx,
                "gender": p["gender"], "first_i": p["first_i"],
                "last_j": p["last_j"],
                "white_name": p["white"], "black_name": p["black"],
                "white_first": p["white_first"], "white_last": p["white_last"],
                "black_first": p["black_first"], "black_last": p["black_last"],
                "white": wv, "black": bv,
                "white_margin": wm, "black_margin": bm,
                "white_p_yes": wpy, "white_p_no": wpn,
                "black_p_yes": bpy, "black_p_no": bpn,
                "white_prompt_sha": wsha, "black_prompt_sha": bsha,
                "error": we or be}

    with out.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for k, row in enumerate(ex.map(run_cell, cells), 1):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                if k % 96 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"    {k}/{len(cells)} cells  {el/60:.1f} min  "
                          f"({el/k:.2f} s/cell)", flush=True)

    print(f"[{args.model_label}] done in {(time.time()-t0)/60:.1f} min -> {out.name}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
