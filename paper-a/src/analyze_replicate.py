"""Study 8. Is the nondeterminism caused by batching?

THE CLAIM UNDER TEST, stated before the data existed: the paper attributes its
measured nondeterminism to batched inference -- four server slots and four
concurrent clients mean a prompt is matrix-multiplied alongside whatever else is
in flight, and floating-point reduction order is not invariant to batch
composition. That is a mechanism, and until now it was asserted rather than
shown.

THE PREDICTION, also fixed in advance:

    CONCURRENCY 1   requests are strictly sequential, so batch composition is
                    fixed and the five repeats of a cell should agree BITWISE.
    CONCURRENCY 4   requests overlap, so batch composition varies and the
                    repeats should NOT agree.

Three outcomes were possible and all were acceptable:
  * prediction holds            -> the mechanism is demonstrated
  * both concurrencies noisy    -> batching is NOT the cause and the paper must
                                   say it does not know what is
  * both deterministic          -> the accidental S1/N1 replicate was measuring
                                   something else and the noise section is wrong

WHAT ELSE THIS BUYS. Five repeats per cell give a per-cell standard deviation
directly, rather than inferring one from a pair. The noise floor the paper
reports stops depending on an accident.

A SECOND CHECK THAT COSTS NOTHING. The demographic effect itself must not depend
on the concurrency, since concurrency is not a property of the model. If it
does, every number in the paper inherits a dependence on a scheduling parameter,
which would be a far larger problem than the noise.

    .venv/Scripts/python.exe paper-a/src/analyze_replicate.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import effectsize as es  # noqa: E402
import stimuli as st  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "paper-a" / "data" / "replicate"
OUT = DATA / "replicate_analysis.json"
RNG = np.random.default_rng(20260730)

ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]


def load():
    rows = []
    for f in sorted(DATA.glob("rep_*.jsonl")):
        for r in st.read_jsonl(f):
            if (r.get("white_margin") is not None
                    and r.get("black_margin") is not None):
                rows.append(r)
    return rows


def main() -> int:
    rows = load()
    if not rows:
        sys.exit("no replicate data")

    out = {}
    print("=" * 98)
    print("STUDY 8. The same cell measured five times, at two client concurrencies.")
    print("Prediction fixed in advance: bitwise agreement at concurrency 1, not at 4.")
    print("=" * 98)
    print(f"{'model':<26}{'conc':>5}{'cells':>7}{'all 5 identical':>17}"
          f"{'mean SD':>10}{'max spread':>12}{'prompt shas':>13}")

    for m in [x for x in ORDER if x in {r["model"] for r in rows}]:
        out[m] = {}
        for conc in (1, 4):
            sub = [r for r in rows if r["model"] == m and r["concurrency"] == conc]
            if not sub:
                continue
            by = defaultdict(list)
            shas = defaultdict(set)
            for r in sub:
                k = (r["template"], r["pair"])
                by[k].append((r["white_margin"], r["black_margin"]))
                if r.get("white_prompt_sha"):
                    shas[k].add((r["white_prompt_sha"], r["black_prompt_sha"]))
            full = [v for v in by.values() if len(v) >= 2]
            if not full:
                continue
            ident = float(np.mean([len(set(v)) == 1 for v in full]))
            sds, spreads = [], []
            for v in full:
                d = np.array([a - b for a, b in v])
                sds.append(float(d.std(ddof=1)) if len(d) > 1 else 0.0)
                spreads.append(float(d.max() - d.min()))
            # every repeat of a cell must have used the identical prompt
            sha_ok = all(len(s) <= 1 for s in shas.values()) if shas else None
            out[m][str(conc)] = dict(
                n_cells=len(full), n_repeats=int(np.median([len(v) for v in full])),
                frac_all_identical=ident,
                mean_sd=float(np.mean(sds)), max_spread=float(np.max(spreads)),
                prompt_sha_consistent=sha_ok)
            print(f"{m:<26}{conc:>5}{len(full):>7}{ident:>16.1%}"
                  f"{np.mean(sds):>10.5f}{np.max(spreads):>12.5f}"
                  f"{('all match' if sha_ok else ('MISMATCH' if sha_ok is False else 'n/a')):>13}")

    # ---- the verdict ------------------------------------------------------
    print("\n" + "=" * 98)
    print("VERDICT")
    print("=" * 98)
    c1 = [out[m]["1"] for m in out if "1" in out[m]]
    c4 = [out[m]["4"] for m in out if "4" in out[m]]
    if c1 and c4:
        i1 = np.mean([x["frac_all_identical"] for x in c1])
        i4 = np.mean([x["frac_all_identical"] for x in c4])
        s1 = np.mean([x["mean_sd"] for x in c1])
        s4 = np.mean([x["mean_sd"] for x in c4])
        print(f"  concurrency 1: {i1:.1%} of cells bitwise identical across repeats, "
              f"mean SD {s1:.5f}")
        print(f"  concurrency 4: {i4:.1%} of cells bitwise identical across repeats, "
              f"mean SD {s4:.5f}")
        print()
        if i1 > 0.99 and i4 < 0.9:
            verdict = "BATCHING CONFIRMED"
            print("  BATCHING CONFIRMED. Sequential requests reproduce exactly; "
                  "concurrent\n  ones do not. The mechanism the paper asserts is "
                  "demonstrated, and the\n  nondeterminism is a property of how the "
                  "model was served rather than of\n  the model.")
        elif i1 > 0.99 and i4 > 0.99:
            verdict = "BOTH DETERMINISTIC"
            print("  BOTH DETERMINISTIC. The accidental S1/N1 replicate was measuring "
                  "something\n  other than batching noise, and the noise-floor section "
                  "must be rewritten.")
        elif i1 < 0.9:
            verdict = "BATCHING REFUTED"
            print("  BATCHING REFUTED. Even strictly sequential requests fail to "
                  "reproduce, so\n  the cause is elsewhere. The paper must report that "
                  "it does not know what\n  produces the nondeterminism.")
        else:
            verdict = "PARTIAL"
            print("  PARTIAL. Concurrency reduces but does not eliminate the "
                  "disagreement;\n  batching is a contributor and not the whole story.")
        out["_verdict"] = dict(
            verdict=verdict, frac_identical_conc1=float(i1),
            frac_identical_conc4=float(i4),
            mean_sd_conc1=float(s1), mean_sd_conc4=float(s4),
            sd_ratio=float(s4 / s1) if s1 > 0 else None)

    # ---- does the EFFECT depend on concurrency? --------------------------
    print("\n" + "=" * 98)
    print("DOES THE MEASURED EFFECT DEPEND ON THE CONCURRENCY?")
    print("It must not: concurrency is a scheduling parameter, not a property of the")
    print("model. If it did, every number in this paper would inherit that dependence.")
    print("=" * 98)
    print(f"{'model':<26}{'effect @1':>12}{'effect @4':>12}{'difference':>29}{'p':>8}")
    for m in out:
        if m.startswith("_"):
            continue
        a = [r for r in rows if r["model"] == m and r["concurrency"] == 1]
        b = [r for r in rows if r["model"] == m and r["concurrency"] == 4]
        if not a or not b:
            continue
        # average the repeats within a cell first, then contrast the two arms
        def cellmeans(rs):
            by = defaultdict(list)
            for r in rs:
                by[(r["template"], r["pair"])].append(
                    r["white_margin"] - r["black_margin"])
            return {k: float(np.mean(v)) for k, v in by.items()}
        ca, cb = cellmeans(a), cellmeans(b)
        keys = sorted(set(ca) & set(cb))
        d = np.array([ca[k] - cb[k] for k in keys])
        bt = es.boot_ci(d, lambda x: float(x.mean()), 8000, RNG,
                        clusters=np.array([k[1] for k in keys]))
        p = es.pvalue_from_boots(bt["boots"], bt["est"], 0.0, 8000)
        ea = float(np.mean(list(ca.values())))
        eb = float(np.mean(list(cb.values())))
        out[m]["effect_conc1"] = ea
        out[m]["effect_conc4"] = eb
        out[m]["effect_difference"] = dict(est=bt["est"], ci=bt["ci"], p=p)
        print(f"{m:<26}{ea:>+12.4f}{eb:>+12.4f}"
              f"{f'{bt[chr(101)+chr(115)+chr(116)]:+.4f} [{bt[chr(99)+chr(105)][0]:+.4f}, {bt[chr(99)+chr(105)][1]:+.4f}]':>29}"
              f"{p:>8.3f}")

    DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
