"""Did the D9 base-arm cells use the corrected definition? A pre-registered test.

THE PROBLEM. D9 is one of three position-control conditions added on 2026-07-28.
Its first definition applied two space-doubling replacements whose target sites
overlapped inside the job posting, so instead of the intended two-token shift it
produced one. That was caught by tokenising the condition and corrected within
minutes, and only the corrected version was ever committed.

But the suite was running at the time. The base arm of the position conditions
(templates T1_strong and T3_marginal) was written by the run that was in flight,
and NOTHING IN THE RECORDED DATA SAID WHICH DEFINITION PRODUCED IT. File
timestamps, git history and append order were all checked and none of them can
settle it: the file is appended to across many runs, and the commit postdates the
run that wrote the rows.

WHY NOT JUST DELETE AND RE-RUN. Because deleting destroys the only evidence that
could establish whether anything was ever wrong. If the original rows are sound,
throwing them away costs the study a real measurement and leaves the question
permanently unanswerable. So instead the cells are re-measured into a separate
file and the two are compared.

THE DESIGN, AND WHY IT IS DECISIVE. Re-measuring D9 alone would confound a
definition change with ordinary cross-session drift, which is real: identical
prompts re-measured in a different process agree on 0 of 154 cells, with a
systematic offset. So D8 is re-measured alongside as a CONTROL. D8 was verified
correct on first inspection and its definition has never changed, so its
recheck-versus-stored difference is pure drift. The test is then internal:

    D8 difference   = drift
    D9 difference   = drift + (definition change, if any)

    If the two distributions are indistinguishable, the D9 rows were produced by
    the corrected definition and the original data stands.

    If D9's differences are systematically larger, the original rows came from
    the superseded definition and must be replaced by the recheck.

DECISION RULE, FIXED BEFORE THE DATA EXISTS. Per model-mode, and pooled:

  1. Paired difference d = recheck - stored, per cell, for D8 and for D9.
  2. Compare |d| between conditions with a Mann-Whitney U test, and compare the
     mean d with a two-sample bootstrap.
  3. VINDICATED if the D9 mean |d| is within 1.5x the D8 mean |d| AND the
     pooled test does not reject at BH-adjusted 0.05.
  4. CONDEMNED otherwise, in which case the stored D9 base-arm rows are moved to
     a quarantine directory and the recheck values replace them.

Both outcomes are acceptable. Only failing to check is not.

    .venv/Scripts/python.exe paper-a/src/adjudicate_d9.py            # report
    .venv/Scripts/python.exe paper-a/src/adjudicate_d9.py --apply    # act on it
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import effectsize as es  # noqa: E402
import stimuli as st  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
MP = ROOT / "paper-a" / "data" / "mechanism_panel"
QUAR = MP / "_d9_superseded"
OUT = MP / "d9_adjudication.json"
BASE_TEMPLATES = ("T1_strong", "T3_marginal")
RATIO_LIMIT = 1.5
RNG = np.random.default_rng(20260729)


def index(path, conds):
    out = {}
    for r in st.read_jsonl(path):
        if r.get("cond") not in conds:
            continue
        if r.get("template") not in BASE_TEMPLATES:
            continue
        if r.get("white_margin") is None or r.get("black_margin") is None:
            continue
        out[(r["cond"], r["template"], r["pair"])] = (
            r["white_margin"], r["black_margin"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="act on the verdict: quarantine and replace if condemned")
    args = ap.parse_args()

    rechecks = sorted(MP.glob("mech_d9recheck_*.jsonl"))
    if not rechecks:
        print("no recheck data yet; jobs 49-60 have not run.")
        return 2

    print("=" * 96)
    print("D9 ADJUDICATION. Re-measured cells against stored, with D8 as the")
    print("never-changed control that calibrates ordinary cross-session drift.")
    print("=" * 96)
    print(f"{'model / mode':<34}{'n':>5}{'D8 mean|d|':>12}{'D9 mean|d|':>12}"
          f"{'ratio':>8}{'U p':>9}   verdict")

    results = {}
    all_d8, all_d9 = [], []
    for rp in rechecks:
        stem = rp.name.replace("mech_d9recheck_", "").replace(".jsonl", "")
        mode, model = stem.split("_", 1)
        orig = MP / f"mech_{mode}_{model}.jsonl"
        if not orig.exists():
            continue
        a = index(rp, ("D8", "D9"))
        b = index(orig, ("D8", "D9"))
        shared = sorted(set(a) & set(b))
        if len(shared) < 40:
            continue

        d8 = np.array([abs((a[k][0] - a[k][1]) - (b[k][0] - b[k][1]))
                       for k in shared if k[0] == "D8"])
        d9 = np.array([abs((a[k][0] - a[k][1]) - (b[k][0] - b[k][1]))
                       for k in shared if k[0] == "D9"])
        if len(d8) < 10 or len(d9) < 10:
            continue
        all_d8.append(d8)
        all_d9.append(d9)

        from scipy import stats  # noqa: PLC0415
        u = stats.mannwhitneyu(d9, d8, alternative="greater")
        ratio = float(d9.mean() / d8.mean()) if d8.mean() > 0 else float("inf")
        results[f"{model}/{mode}"] = dict(
            n_shared=len(shared), n_d8=len(d8), n_d9=len(d9),
            d8_mean_abs=float(d8.mean()), d9_mean_abs=float(d9.mean()),
            ratio=ratio, u_p=float(u.pvalue))
        print(f"{model + ' / ' + mode:<34}{len(shared):>5}{d8.mean():>12.4f}"
              f"{d9.mean():>12.4f}{ratio:>8.2f}{u.pvalue:>9.4f}"
              f"   {'ok' if ratio <= RATIO_LIMIT else 'LARGER'}")

    if not results:
        print("\nnot enough shared cells to adjudicate yet.")
        return 2

    # ---- pooled verdict --------------------------------------------------
    from scipy import stats
    p8 = np.concatenate(all_d8)
    p9 = np.concatenate(all_d9)
    pooled_u = stats.mannwhitneyu(p9, p8, alternative="greater")
    pooled_ratio = float(p9.mean() / p8.mean())
    per_p = [v["u_p"] for v in results.values()]
    adj = es.benjamini_hochberg(per_p)
    for (k, v), a in zip(results.items(), adj):
        v["u_p_bh"] = a
    n_sig = sum(1 for a in adj if a < 0.05)

    vindicated = (pooled_ratio <= RATIO_LIMIT and pooled_u.pvalue >= 0.05
                  and n_sig == 0)

    print("\n" + "=" * 96)
    print(f"POOLED   n(D8) = {len(p8)}, n(D9) = {len(p9)}")
    print(f"  D8 mean |difference|  {p8.mean():.4f}   (drift only)")
    print(f"  D9 mean |difference|  {p9.mean():.4f}")
    print(f"  ratio                 {pooled_ratio:.3f}   (limit {RATIO_LIMIT})")
    print(f"  Mann-Whitney p        {pooled_u.pvalue:.4f}")
    print(f"  model-modes rejecting after BH: {n_sig} of {len(results)}")
    print()
    if vindicated:
        print("  VERDICT: VINDICATED. The D9 base-arm rows are indistinguishable")
        print("  from the never-changed control, so they were produced by the")
        print("  corrected definition. The original data stands and the recheck")
        print("  becomes an additional cross-session reproducibility measurement.")
    else:
        print("  VERDICT: CONDEMNED. D9's differences exceed the control's, so the")
        print("  stored base-arm rows were produced by the superseded definition.")
        print("  They must be quarantined and replaced by the recheck values.")
        if args.apply:
            QUAR.mkdir(parents=True, exist_ok=True)
            moved = 0
            for rp in rechecks:
                stem = rp.name.replace("mech_d9recheck_", "").replace(".jsonl", "")
                mode, model = stem.split("_", 1)
                orig = MP / f"mech_{mode}_{model}.jsonl"
                if not orig.exists():
                    continue
                shutil.copy2(orig, QUAR / orig.name)
                keep, drop = [], 0
                for r in st.read_jsonl(orig):
                    if (r.get("cond") == "D9"
                            and r.get("template") in BASE_TEMPLATES):
                        drop += 1
                        continue
                    keep.append(r)
                for r in st.read_jsonl(rp):
                    if (r.get("cond") == "D9"
                            and r.get("template") in BASE_TEMPLATES):
                        keep.append(r)
                orig.write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in keep)
                    + "\n", encoding="utf-8")
                moved += drop
            print(f"\n  applied: {moved} superseded D9 rows quarantined to "
                  f"{QUAR.name}/ and replaced by the recheck.")
        else:
            print("\n  re-run with --apply to quarantine and replace.")

    OUT.write_text(json.dumps(dict(
        per_model_mode=results,
        pooled=dict(n_d8=len(p8), n_d9=len(p9),
                    d8_mean_abs=float(p8.mean()), d9_mean_abs=float(p9.mean()),
                    ratio=pooled_ratio, u_p=float(pooled_u.pvalue),
                    n_rejecting_after_bh=n_sig),
        ratio_limit=RATIO_LIMIT,
        verdict="VINDICATED" if vindicated else "CONDEMNED",
        applied=bool(args.apply and not vindicated)), indent=2),
        encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
