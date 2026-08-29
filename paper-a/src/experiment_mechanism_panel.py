"""Study 5. Does the delimiter mechanism generalise, and is it specific?

Study 3 found that destroying a structural delimiter token moves the measured
demographic effect on Llama-3.1-8B-Instruct, with every position and proximity
control null, and that none of it replicates on Llama-2-7B-chat. It offered an
explanation for the split -- heavy instruction tuning teaches a model to read a
paragraph break as a section boundary -- and said in as many words that the
explanation was fitted to two models rather than tested. This study tests it,
and closes two further gaps at the same time.

--------------------------------------------------------------------------
GAP G2. TWO MODELS CANNOT DISCRIMINATE THE HYPOTHESIS.
--------------------------------------------------------------------------
Llama-2-7B-chat and Llama-3.1-8B-Instruct differ in instruction tuning, and also
in tokenizer, pretraining corpus, parameter count, architecture details and
release date. At n = 2 every one of those is a candidate explanation. The panel
on disk breaks the confounds:

    mistral-7b-v0.1-BASE  vs  mistral-7b-INSTRUCT-v0.1
        Identical pretrained weights. Identical tokenizer, architecture,
        parameter count, corpus, release month. One differs from the other by
        instruction tuning and by nothing else. This is the decisive row, and
        H_tune makes a sharp prediction: the base model shows NO delimiter
        effect.

    mistral-instruct-v0.1  vs  mistral-instruct-v0.3
        Tuning recipe generation, architecture held fixed.

    llama-2-7b-chat  vs  llama-2-13b-chat
        Scale, tuning recipe held fixed. If the effect is about capacity rather
        than tuning, it should appear here and not in the Mistral pair.

    llama-2-7b-chat  vs  llama-3.1-8b-instruct
        The original contrast, retained.

--------------------------------------------------------------------------
CHAT MODE AND RAW MODE.
--------------------------------------------------------------------------
The base checkpoint has no chat template, so comparing it to a tuned model
through /v1/chat/completions confounds the weights with whatever template
llama.cpp wraps around them. Every model is therefore run twice: once
chat-templated, once as a raw completion with a byte-identical wrapper. Chat
mode is comparable to Study 3 and to the rest of the literature. Raw mode is the
only one in which base and instruct are asked the same question. Reporting both
means neither result can be dismissed as an artifact of its own path.

The wrapper for raw mode deliberately uses single newlines, so it contributes no
paragraph break of its own and the conditions still target exactly the two
delimiters they targeted in Study 3.

--------------------------------------------------------------------------
GAP G4. IS THE EFFECT SPECIFIC TO DEMOGRAPHIC SIGNAL?
--------------------------------------------------------------------------
Study 3 showed delimiter destruction moves the demographic contrast. It never
asked whether it moves ANY contrast. Running every condition against both the
strongest and the weakest resume yields, from the same calls and at no extra
cost, a second contrast on the same scale:

    demographic   m(White name) - m(Black name),  template held fixed
    qualification m(strong resume) - m(weak resume),  name held fixed

If destruction moves both, the finding is generic decision instability and the
demographic framing is decoration. If it moves the demographic contrast and
leaves the qualification contrast intact, the finding is specific: damaging the
format costs the model its grip on a weak social signal while leaving a strong
competence signal alone. That is a much stronger claim and the design can now
distinguish them.

DESIGN
    6 checkpoints x 2 modes x 8 conditions x 24 name pairs x 2 templates
    = 384 cells = 768 calls per model-mode.

Resumable per (mode, condition, template, pair).
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
from experiment_mechanism import CONDITIONS, build  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper-a" / "data" / "mechanism_panel"

SYSTEM = ("You are an experienced hiring screener. You evaluate candidates "
          "against a job posting and make a screening decision.")


def raw_prompt(conditioned: str) -> str:
    """Wrap a conditioned user message for the raw-completion path.

    Single newlines only. A wrapper containing its own paragraph break would add
    delimiters outside the conditions' reach and change what "both delimiters
    destroyed" means between the two modes.
    """
    return f"{SYSTEM}\n{conditioned}\nAnswer:"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--mode", required=True, choices=("chat", "raw"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--concurrency", type=int, default=4)
    # Templates are a parameter because the first pass carried only the two
    # extremes, and reconciling Study 3 against this panel then showed that
    # Study 3's headline delimiter effect was concentrated in the MIDDLE
    # template it happened to include: on Llama-3.1-8B, D6-D0 was +0.058 under
    # T1, +0.256 under T2, +0.078 under T3. Pooling the three gave +0.131 and
    # read as a general effect. Dropping T2, as the first panel pass did, gives
    # +0.046. Template is therefore not a nuisance dimension here, it is a
    # factor, and it has to be run rather than chosen.
    ap.add_argument("--templates", default=None,
                    help="comma-separated; default is the two extremes")
    # A RE-MEASUREMENT MODE, not a convenience. D9's definition was corrected
    # in the working tree minutes after it was written, and no combination of
    # file timestamps, git history and append order could establish which
    # definition produced the base-arm rows. Rather than delete data that may
    # be perfectly good, these two flags re-measure the cells in question into
    # a SEPARATE file so the old and new values can be compared against the
    # measured noise floor. Agreement vindicates the original rows; systematic
    # disagreement condemns them. Deleting first would have destroyed the only
    # evidence that could decide it.
    ap.add_argument("--conditions", default=None,
                    help="comma-separated subset, e.g. D9")
    ap.add_argument("--out-suffix", default="",
                    help="write to mech_<suffix>_<mode>_<model>.jsonl instead")
    args = ap.parse_args()
    conds = (tuple(x.strip() for x in args.conditions.split(","))
             if args.conditions else tuple(CONDITIONS))
    for c in conds:
        if c not in CONDITIONS:
            sys.exit(f"unknown condition {c!r}; have {sorted(CONDITIONS)}")
    templates = (tuple(args.templates.split(",")) if args.templates
                 else st.TEMPLATES_2)
    for t in templates:
        if t not in st.TEMPLATES:
            sys.exit(f"unknown template {t!r}; have {sorted(st.TEMPLATES)}")

    api = f"http://127.0.0.1:{args.port}"
    served = st.assert_serving(args.port, args.model_label)
    print(f"  [guard] port {args.port} serving {served} (mode={args.mode})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.out_suffix}_" if args.out_suffix else ""
    out = OUT_DIR / f"mech_{tag}{args.mode}_{args.model_label}.jsonl"

    done: set[tuple[str, str, int]] = set()
    for r in st.read_jsonl(out):
        if r.get("white_margin") is not None and r.get("black_margin") is not None:
            done.add((r["cond"], r["template"], r["pair"]))
    if done:
        print(f"[resume] {len(done)} cells complete", flush=True)

    cells = [(c, t, p["idx"]) for c, t, p in itertools.product(
        conds, templates, st.MECH_GRID)
        if (c, t, p["idx"]) not in done]
    print(f"[{args.model_label}/{args.mode}] {len(cells)} cells "
          f"({len(cells)*2} calls)", flush=True)
    t0 = time.time()
    by_idx = {p["idx"]: p for p in st.MECH_GRID}

    def one(cond, tname, name):
        """Returns the call result plus a hash of exactly what was sent."""
        body = st.TEMPLATES[tname]
        conditioned = build(cond, name, body)
        if args.mode == "chat":
            sha = st.prompt_sha(SYSTEM, conditioned)
            return st.call_chat(api, SYSTEM, conditioned) + (sha,)
        full = raw_prompt(conditioned)
        return st.call_raw(api, full) + (st.prompt_sha(full),)

    def run_cell(cell):
        cond, tname, idx = cell
        p = by_idx[idx]
        wv, wm, wpy, wpn, we, wsha = one(cond, tname, p["white"])
        bv, bm, bpy, bpn, be, bsha = one(cond, tname, p["black"])
        return {"model": args.model_label, "mode": args.mode, "cond": cond,
                "cond_note": CONDITIONS[cond][0],
                "n_delims_destroyed": CONDITIONS[cond][1],
                "template": tname, "pair": idx, "gender": p["gender"],
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
                if k % 64 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"    {k}/{len(cells)}  {el/60:.1f} min "
                          f"({el/k:.2f} s/cell)", flush=True)

    print(f"[{args.model_label}/{args.mode}] done in "
          f"{(time.time()-t0)/60:.1f} min -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
