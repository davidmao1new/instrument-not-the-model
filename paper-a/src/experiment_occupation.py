"""Study 7. Is the wording instability a property of the model or of the job?

THE GAP (docs/GAPS.md G7). Everything else in this paper is measured on one job
posting in one occupation. That is the paper's largest stated limitation and the
first thing a reviewer will raise, since Wilson and Caliskan vary occupation and
we did not. Closing it costs about four hours of local compute, which is not a
good reason to leave a limitation standing.

DESIGN. The Study 2 design, unchanged, run against two further occupations:

    12 wording variants (6 semantic, 6 null)   <- imported, byte-identical
    x 3 résumé templates
    x 12 matched name pairs                    <- imported, byte-identical
    x 2 new occupations
    = 864 cells = 1,728 calls per model

The Business Analyst arm is NOT re-run; it is the existing Study 2 data, and
`occupations.py` asserts that its strings have not drifted from the published
ones. Everything except the occupation is therefore held fixed by construction
rather than by intention.

WHAT WOULD FALSIFY WHAT. The confirmatory question is whether the ratio of
between-wording dispersion to effect size is stable across occupations. If it
is, the instability is a property of the model and the single-occupation
limitation dissolves. If it is not, the paper's headline numbers are specific to
a Business Analyst posting and must be reported as such -- which would be a
worse result for us and a more useful one for the field.

A SECOND, EXPLORATORY QUESTION comes free. The occupations were chosen to span
the gender typing of the labour market: Software Engineer is strongly
male-typed, Registered Nurse strongly female-typed, Business Analyst roughly
balanced. The name grid is gender-balanced, so an interaction between
occupational gender typing and the demographic effect is estimable. It is
labelled exploratory because it was not the reason for running this, and because
three occupations cannot support a claim about occupations in general.

Resumable per (occupation, variant, template, pair).
    .venv/Scripts/python.exe paper-a/src/experiment_occupation.py --model-label <id>
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
from experiment_delta_stability import PAIRS, VARIANTS  # noqa: E402
from occupations import OCCUPATIONS, check_matched  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper-a" / "data" / "occupation"

# BA is Study 2 and is not re-run.
NEW_OCCUPATIONS = ("SWE", "RN")


def user_message(variant, posting, name, body):
    """Byte-identical in structure to Study 2's, with the posting swapped."""
    msg = (f"Job posting:\n{posting}\n\n"
           f"Candidate:\nName: {name}\n{body}\n\n"
           f"{variant['ask']}")
    return variant["post"](msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--occupations", default=",".join(NEW_OCCUPATIONS))
    args = ap.parse_args()

    problems = check_matched()
    if problems:
        sys.exit("occupations are not structurally matched:\n  "
                 + "\n  ".join(problems))
    occs = tuple(x.strip() for x in args.occupations.split(","))
    for o in occs:
        if o not in OCCUPATIONS:
            sys.exit(f"unknown occupation {o!r}; have {sorted(OCCUPATIONS)}")

    api = f"http://127.0.0.1:{args.port}"
    served = st.assert_serving(args.port, args.model_label)
    print(f"  [guard] port {args.port} serving {served}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"occ_{args.model_label}.jsonl"

    done: set[tuple[str, str, str, int]] = set()
    for r in st.read_jsonl(out):
        if r.get("white_margin") is not None and r.get("black_margin") is not None:
            done.add((r["occupation"], r["variant"], r["template"], r["pair"]))
    if done:
        print(f"[resume] {len(done)} cells complete", flush=True)

    cells = [(o, v, t, i) for o, v, t, i in itertools.product(
        occs, VARIANTS, sorted(OCCUPATIONS["BA"]["templates"]), range(len(PAIRS)))
        if (o, v, t, i) not in done]
    print(f"[{args.model_label}] {len(cells)} cells to run ({len(cells)*2} calls), "
          f"occupations={','.join(occs)}", flush=True)
    t0 = time.time()

    def run_cell(cell):
        occ, vname, tname, idx = cell
        o = OCCUPATIONS[occ]
        var = VARIANTS[vname]
        body = o["templates"][tname]
        wname, bname = PAIRS[idx]
        wv, wm, wpy, wpn, we = st.call_chat(
            api, var["system"], user_message(var, o["posting"], wname, body))
        bv, bm, bpy, bpn, be = st.call_chat(
            api, var["system"], user_message(var, o["posting"], bname, body))
        return {"model": args.model_label, "occupation": occ,
                "occupation_label": o["label"],
                "gender_typing": o["gender_typing"],
                "variant": vname, "variant_kind": var["kind"],
                "template": tname, "pair": idx,
                "white_name": wname, "black_name": bname,
                "white": wv, "black": bv,
                "white_margin": wm, "black_margin": bm,
                "white_p_yes": wpy, "white_p_no": wpn,
                "black_p_yes": bpy, "black_p_no": bpn,
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
