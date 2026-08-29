"""Does prompt-cache reuse explain the nondeterminism that survives sequencing?

THE CHAIN OF CLAIMS, AND WHERE THIS SITS IN IT.

  1. The measurement is not reproducible even at temperature 0. Established by
     the S1/N1 byte-identical replicate and by cross-session re-measurement.
  2. Batched inference is the cause. Study 8 tested this and returned PARTIAL:
     forcing strictly sequential requests raises bitwise agreement from 47.9% to
     93.8% and cuts the per-cell SD roughly eightfold, so batching is most of the
     story -- but 6.2% of cells still disagree with themselves.
  3. Something else accounts for the remainder. THIS FILE.

THE CANDIDATE. llama.cpp reuses the KV cache for whatever prefix a new prompt
shares with the previous one. Which prefix is resident depends on what ran
immediately before, so two calls with byte-identical text can take different
arithmetic paths purely because of what preceded them. That is order dependence
WITHOUT concurrency, and it predicts precisely the residual Study 8 measured.

THE COMPARISON. Three arms on the same 36 cells, five repeats each:

    concurrency 4, cache on    the suite's normal setting
    concurrency 1, cache on    Study 8's sequential arm
    concurrency 1, cache OFF   this probe

If the third arm reaches full bitwise agreement, the mechanism is completely
accounted for: batching plus cache reuse, both artefacts of serving rather than
properties of the model. If it does not, the paper reports a residual it cannot
explain, which is the honest outcome and a better one than a confident guess.

A SECOND CHECK THAT COSTS NOTHING, as in Study 8: the measured demographic
effect must not depend on whether the cache is on. Caching is a serving
optimisation, not a property of the model.

    .venv/Scripts/python.exe paper-a/src/analyze_cache_residual.py
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
OUT = DATA / "cache_residual.json"
N_BOOT = 20_000
RNG = np.random.default_rng(20260730)


def cells(rows):
    """{(template, pair): [(white, black), ...]} plus the prompt SHAs seen."""
    by, shas = defaultdict(list), defaultdict(set)
    for r in rows:
        if r.get("white_margin") is None or r.get("black_margin") is None:
            continue
        k = (r["template"], r["pair"])
        by[k].append((r["white_margin"], r["black_margin"]))
        if r.get("white_prompt_sha"):
            shas[k].add((r["white_prompt_sha"], r["black_prompt_sha"]))
    return by, shas


def stats_for(by):
    full = [v for v in by.values() if len(v) >= 2]
    if not full:
        return None
    ident = float(np.mean([len(set(v)) == 1 for v in full]))
    sds, spreads = [], []
    for v in full:
        d = np.array([a - b for a, b in v])
        sds.append(float(d.std(ddof=1)))
        spreads.append(float(d.max() - d.min()))
    return dict(n_cells=len(full), frac_identical=ident,
                mean_sd=float(np.mean(sds)), max_spread=float(np.max(spreads)))


def cellmeans(by):
    return {k: float(np.mean([a - b for a, b in v])) for k, v in by.items()}


def main() -> int:
    arms = {}
    for f in sorted(DATA.glob("rep_*.jsonl")):
        for r in st.read_jsonl(f):
            arms.setdefault((r["model"], r["concurrency"], True), []).append(r)
    for f in sorted(DATA.glob("nocache_*.jsonl")):
        if f.name.startswith("nocache_session2_"):
            continue
        for r in st.read_jsonl(f):
            arms.setdefault((r["model"], 1, False), []).append(r)

    # The second session is a separate arm, not more repeats of the first.
    session2 = {}
    for f in sorted(DATA.glob("nocache_session2_*.jsonl")):
        for r in st.read_jsonl(f):
            if r.get("white_margin") is not None:
                session2.setdefault(r["model"], {})[(r["template"], r["pair"])] = r
    if not any(not k[2] for k in arms):
        sys.exit("no cache-off data yet")

    models = sorted({k[0] for k in arms if not k[2]})
    print("=" * 100)
    print("THE RESIDUAL. Same 36 cells, five repeats, three serving configurations.")
    print("Study 8 showed sequencing removes most of the disagreement. Does turning")
    print("off prompt-cache reuse remove the rest?")
    print("=" * 100)
    print(f"{'model':<26}{'arm':<26}{'cells':>6}{'all 5 identical':>17}"
          f"{'mean SD':>10}{'max spread':>12}{'prompt shas':>13}")

    out = {}
    for m in models:
        out[m] = {}
        for label, key in (("concurrency 4, cache on", (m, 4, True)),
                           ("concurrency 1, cache on", (m, 1, True)),
                           ("concurrency 1, cache OFF", (m, 1, False))):
            if key not in arms:
                continue
            by, shas = cells(arms[key])
            s = stats_for(by)
            if s is None:
                continue
            sha_ok = all(len(x) <= 1 for x in shas.values()) if shas else None
            s["prompt_sha_consistent"] = sha_ok
            out[m][label] = s
            print(f"{m:<26}{label:<26}{s['n_cells']:>6}{s['frac_identical']:>16.1%}"
                  f"{s['mean_sd']:>10.5f}{s['max_spread']:>12.5f}"
                  f"{('all match' if sha_ok else 'MISMATCH' if sha_ok is False else 'n/a'):>13}")

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    on, off = [], []
    for m in out:
        a = out[m].get("concurrency 1, cache on")
        b = out[m].get("concurrency 1, cache OFF")
        if a and b:
            on.append(a)
            off.append(b)
    if not off:
        sys.exit("no model has both sequential arms")
    i_on = float(np.mean([x["frac_identical"] for x in on]))
    i_off = float(np.mean([x["frac_identical"] for x in off]))
    s_on = float(np.mean([x["mean_sd"] for x in on]))
    s_off = float(np.mean([x["mean_sd"] for x in off]))
    print(f"  sequential, cache ON   {i_on:.1%} identical, mean SD {s_on:.5f}")
    print(f"  sequential, cache OFF  {i_off:.1%} identical, mean SD {s_off:.5f}")
    print()
    if i_off >= 0.995:
        verdict = "FULLY EXPLAINED"
        print("  FULLY EXPLAINED. With requests sequential and prompt-cache reuse")
        print("  disabled, repeats of a cell agree bitwise. The nondeterminism is")
        print("  entirely an artefact of how the model was served -- batch")
        print("  composition and cache residency -- and not of the model. Both are")
        print("  configuration a paper never reports.")
    elif i_off > i_on + 0.02:
        verdict = "PARTLY EXPLAINED"
        print("  PARTLY EXPLAINED. Disabling the cache raises agreement further, so")
        print("  cache residency is a real contributor, but some disagreement")
        print("  survives both controls and the paper must say it is unaccounted for.")
    else:
        verdict = "NOT THE CACHE"
        print("  NOT THE CACHE. Disabling prompt-cache reuse does not improve")
        print("  agreement. The residual has some other source, and the paper")
        print("  reports it as unexplained rather than naming a mechanism it has")
        print("  not demonstrated.")
    out["_verdict"] = dict(verdict=verdict, frac_identical_cache_on=i_on,
                           frac_identical_cache_off=i_off,
                           mean_sd_cache_on=s_on, mean_sd_cache_off=s_off,
                           n_models=len(off))

    # ---- does the EFFECT depend on the cache setting? --------------------
    print("\n" + "=" * 100)
    print("DOES THE MEASURED EFFECT DEPEND ON THE CACHE SETTING? It must not.")
    print("=" * 100)
    print(f"{'model':<26}{'cache on':>11}{'cache off':>11}{'difference':>29}{'p':>8}")
    for m in out:
        if m.startswith("_"):
            continue
        if (m, 1, True) not in arms or (m, 1, False) not in arms:
            continue
        a, _ = cells(arms[(m, 1, True)])
        b, _ = cells(arms[(m, 1, False)])
        ca, cb = cellmeans(a), cellmeans(b)
        keys = sorted(set(ca) & set(cb))
        d = np.array([ca[k] - cb[k] for k in keys])
        bt = es.boot_ci(d, lambda x: float(x.mean()), N_BOOT, RNG,
                        clusters=np.array([k[1] for k in keys]))
        p = es.pvalue_from_boots(bt["boots"], bt["est"], 0.0, N_BOOT)
        ea = float(np.mean([ca[k] for k in keys]))
        eb = float(np.mean([cb[k] for k in keys]))
        out[m]["effect_cache_on"] = ea
        out[m]["effect_cache_off"] = eb
        out[m]["effect_difference"] = dict(est=bt["est"], ci=bt["ci"], p=p)
        print(f"{m:<26}{ea:>+11.4f}{eb:>+11.4f}"
              f"{f'{bt[chr(101)+chr(115)+chr(116)]:+.4f} [{bt[chr(99)+chr(105)][0]:+.4f}, {bt[chr(99)+chr(105)][1]:+.4f}]':>29}"
              f"{p:>8.3f}")

    # ---- and does it reproduce ACROSS SESSIONS? --------------------------
    #
    # Everything above is repeats inside one server launch. The paper reports a
    # second and larger reproducibility failure -- identical prompts measured in
    # a different process agree on 0 of 504 cells -- and nothing above speaks to
    # it. Without this arm, "the nondeterminism is fully accounted for" would be
    # an extrapolation from within-run repeats to a between-run claim.
    if session2:
        print()
        print("=" * 100)
        print("AND ACROSS SESSIONS? Same cells, same controls, a FRESH server process.")
        print("=" * 100)
        print(f"{'model':<26}{'cells':>7}{'identical':>12}{'sigma':>12}"
              f"{'max |diff|':>12}{'effect s1':>11}{'effect s2':>11}")
        xs = {}
        for m, b in sorted(session2.items()):
            if (m, 1, False) not in arms:
                continue
            a = {}
            for r in arms[(m, 1, False)]:
                k = (r["template"], r["pair"])
                a.setdefault(k, r)
            keys = sorted(set(a) & set(b))
            if not keys:
                continue
            sha_ok = all(a[k]["white_prompt_sha"] == b[k]["white_prompt_sha"]
                         and a[k]["black_prompt_sha"] == b[k]["black_prompt_sha"]
                         for k in keys)
            ident = float(np.mean([
                (a[k]["white_margin"], a[k]["black_margin"])
                == (b[k]["white_margin"], b[k]["black_margin"]) for k in keys]))
            da = np.array([a[k]["white_margin"] - a[k]["black_margin"] for k in keys])
            db = np.array([b[k]["white_margin"] - b[k]["black_margin"] for k in keys])
            d = da - db
            xs[m] = dict(n_cells=len(keys), frac_identical=ident,
                         sigma=float(d.std(ddof=1)),
                         max_abs_diff=float(np.abs(d).max()),
                         effect_session1=float(da.mean()),
                         effect_session2=float(db.mean()),
                         prompt_sha_consistent=bool(sha_ok))
            print(f"{m:<26}{len(keys):>7}{ident:>11.1%}{d.std(ddof=1):>12.6f}"
                  f"{np.abs(d).max():>12.6f}{da.mean():>+11.4f}{db.mean():>+11.4f}")
        if xs:
            out["_cross_session"] = xs
            allid = all(v["frac_identical"] >= 0.995 for v in xs.values())
            print()
            if allid:
                print("  The controls hold ACROSS PROCESSES. Repeats in a fresh server")
                print("  reproduce the stored run bitwise, so the cross-session")
                print("  disagreement the paper reports elsewhere is attributable to the")
                print("  same two causes and not to anything about the process itself.")
            else:
                print("  The controls do NOT survive a process restart. Within-run")
                print("  determinism does not imply cross-session determinism, and the")
                print("  paper must report the residual as unexplained.")
            out["_verdict"]["cross_session_identical"] = bool(allid)

    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
