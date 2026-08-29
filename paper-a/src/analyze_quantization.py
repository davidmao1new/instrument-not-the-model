"""Study 6. Does the measured effect survive a change of numerical precision?

THE GAP (docs/GAPS.md G3). Every number elsewhere in this paper is measured at
Q4_K_M. Not one of the six audits in our literature table states its
quantization, and most run through APIs where it is undisclosed and can change
without notice. If the demographic effect moves between Q4_K_M and Q8_0 by an
amount comparable to the wording dispersion, then a second silent researcher
choice is as consequential as the wording, and the paper's thesis generalises
from "the prompt is part of the instrument" to "so is the arithmetic".

WHY THIS IS A STRONGER TEST THAN IT LOOKS. The comparison is CELL-BY-CELL: the
same wording, the same résumé, the same name pair, measured under two
quantizations of the same weights. Everything a paired design can cancel is
cancelled, and what remains is attributable to numerical precision alone. The
design is Study 2's, unchanged, so the Q4 arm is not re-run and cannot drift.

THREE QUANTITIES.

  SHIFT       the paired difference in the demographic effect, Q8 minus Q4,
              with a cluster bootstrap over name pairs. This is the headline:
              does changing precision change the answer?

  AGREEMENT   the correlation between per-cell margins across quantizations,
              and the fraction of cells where the two disagree about the SIGN
              of the paired difference. A high correlation with frequent sign
              disagreement means the two quantizations agree about the model
              and disagree about individual candidates.

  IN CONTEXT  the shift expressed as a multiple of that model's between-wording
              standard deviation, so the reader can see whether precision is a
              smaller or larger nuisance than the wording.

    .venv/Scripts/python.exe paper-a/src/analyze_quantization.py
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
Q8DIR = ROOT / "paper-a" / "data" / "quantization"
Q4DIR = ROOT / "paper-a" / "data" / "delta_stability"
OUT = Q8DIR / "quantization_analysis.json"
RNG = np.random.default_rng(20260729)

SHORT = {
    "mistral-7b-instruct-v0.1": "Mistral-7B-Instruct v0.1",
    "llama-2-7b-chat": "Llama-2-7B-chat",
}


def load(folder, pattern):
    rows = {}
    for f in sorted(pathlib.Path(folder).glob(pattern)):
        for r in st.read_jsonl(f):
            if (r.get("white_margin") is not None
                    and r.get("black_margin") is not None):
                rows[(r["model"], r["variant"], r["template"], r["pair"])] = r
    return rows


def main() -> int:
    if not Q8DIR.exists():
        sys.exit("no quantization data yet")
    q8 = load(Q8DIR, "delta_*.jsonl")
    q4 = load(Q4DIR, "delta_*.jsonl")
    if not q8:
        sys.exit("no Q8 rows")

    vc = {}
    p = Q4DIR / "variance_components.json"
    if p.exists():
        vc = json.loads(p.read_text(encoding="utf-8"))

    out = {}
    print("=" * 98)
    print("STUDY 6. Q4_K_M against Q8_0, cell by cell, same design and same weights.")
    print("=" * 98)

    for q8_label in sorted({k[0] for k in q8}):
        base = q8_label[:-3] if q8_label.endswith("-q8") else q8_label
        pairs = []
        for (m, v, t, i), r8 in q8.items():
            if m != q8_label:
                continue
            r4 = q4.get((base, v, t, i))
            if r4 is None:
                continue
            d8 = r8["white_margin"] - r8["black_margin"]
            d4 = r4["white_margin"] - r4["black_margin"]
            pairs.append(dict(variant=v, template=t, pair=i, d4=d4, d8=d8,
                              w4=r4["white_margin"], w8=r8["white_margin"],
                              b4=r4["black_margin"], b8=r8["black_margin"]))
        if len(pairs) < 20:
            print(f"\n{q8_label}: only {len(pairs)} shared cells, skipped")
            continue

        d4 = np.array([p["d4"] for p in pairs])
        d8 = np.array([p["d8"] for p in pairs])
        shift = d8 - d4
        clusters = np.array([p["pair"] for p in pairs])

        bt = es.boot_ci(shift, lambda a: float(a.mean()), 8000, RNG,
                        clusters=clusters)
        pval = es.pvalue_from_boots(bt["boots"], bt["est"], 0.0, 8000)

        # per-name margins, both arms pooled, for the agreement statistics
        m4 = np.array([p["w4"] for p in pairs] + [p["b4"] for p in pairs])
        m8 = np.array([p["w8"] for p in pairs] + [p["b8"] for p in pairs])
        r_margin = float(np.corrcoef(m4, m8)[0, 1])
        r_delta = float(np.corrcoef(d4, d8)[0, 1])
        sign_flip = float(np.mean(np.sign(d4) != np.sign(d8)))
        identical = float(np.mean(m4 == m8))

        # the same effect, estimated independently under each quantization
        e4 = es.describe([{"white_margin": p["w4"], "black_margin": p["b4"],
                           "pair": p["pair"]} for p in pairs], 4000, RNG)
        e8 = es.describe([{"white_margin": p["w8"], "black_margin": p["b8"],
                           "pair": p["pair"]} for p in pairs], 4000, RNG)

        swq = vc.get(base, {}).get("all", {}).get("sigma_variant", [None] * 3)
        sw = swq[1]

        # AN INTERVAL ON THE RATIO, BECAUSE BOTH HALVES ARE ESTIMATED.
        # shift_over_sigma_variant was published as a bare point value and is
        # the sole support for the abstract's "quantization shifts it by up to
        # the full between-wording standard deviation" and for §9 item 6. Its
        # numerator has a bootstrap interval and its denominator a posterior
        # one spanning a factor of several, so the point value is far better
        # pinned than the quantity it estimates. §4.2 already propagates a
        # structurally identical ratio by a corner bound over the marginal
        # quantiles; the same method is used here so the two are comparable.
        #
        # The corners: the ratio is smallest with the smallest |shift| over the
        # largest sigma, largest with the largest |shift| over the smallest.
        # If the shift interval straddles zero the lower corner IS zero -- the
        # data are consistent with no shift at all, and a bound that hid that
        # would be the same error this fixes.
        ratio = (abs(bt["est"]) / sw) if sw else None
        ratio_ci = None
        if sw and swq[0] and swq[2]:
            lo_s, hi_s = bt["ci"]
            abs_lo = 0.0 if lo_s * hi_s <= 0 else min(abs(lo_s), abs(hi_s))
            abs_hi = max(abs(lo_s), abs(hi_s))
            ratio_ci = [abs_lo / swq[2], abs_hi / swq[0]]

        out[q8_label] = dict(
            base=base, n_cells=len(pairs),
            shift=bt["est"], shift_ci=bt["ci"], shift_p=pval,
            r_per_name_margin=r_margin, r_paired_delta=r_delta,
            sign_disagreement=sign_flip, bitwise_identical=identical,
            q4=dict(logodds=e4["logodds"]["est"], ci=e4["logodds"]["ci"],
                    ps=e4["superiority"]["est"]),
            q8=dict(logodds=e8["logodds"]["est"], ci=e8["logodds"]["ci"],
                    ps=e8["superiority"]["est"]),
            sigma_variant=sw, sigma_variant_ci=[swq[0], swq[2]],
            shift_over_sigma_variant=ratio,
            shift_over_sigma_variant_ci=ratio_ci,
            _ratio_bound_type=(
                "corner bound over the shift's bootstrap interval and the "
                "pooled between-wording SD's posterior interval, both on the "
                "log-odds scale; the same construction §4.2 uses"),
            ratio_exceeds_one_determined=(
                bool(ratio_ci) and (ratio_ci[0] > 1.0 or ratio_ci[1] < 1.0)))

        r = out[q8_label]
        print(f"\n{SHORT.get(base, base)}   ({len(pairs)} shared cells)")
        print(f"  effect at Q4_K_M           {r['q4']['logodds']:+.4f} "
              f"[{r['q4']['ci'][0]:+.4f}, {r['q4']['ci'][1]:+.4f}]   Ps {r['q4']['ps']:.3f}")
        print(f"  effect at Q8_0             {r['q8']['logodds']:+.4f} "
              f"[{r['q8']['ci'][0]:+.4f}, {r['q8']['ci'][1]:+.4f}]   Ps {r['q8']['ps']:.3f}")
        print(f"  paired shift Q8 - Q4       {r['shift']:+.4f} "
              f"[{r['shift_ci'][0]:+.4f}, {r['shift_ci'][1]:+.4f}]  p = {r['shift_p']:.4f}")
        if sw:
            print(f"  as a multiple of the between-wording SD ({sw:.4f})   "
                  f"{r['shift_over_sigma_variant']:.2f}x")
        print(f"  correlation of per-name margins across quantizations   "
              f"r = {r_margin:.4f}")
        print(f"  correlation of paired differences                      "
              f"r = {r_delta:.4f}")
        print(f"  cells where the two disagree about the SIGN            "
              f"{sign_flip:.1%}")
        print(f"  cells bitwise identical                                "
              f"{identical:.1%}")

    if out:
        print("\nREADING. A shift indistinguishable from zero would mean precision is")
        print("free. A shift comparable to the between-wording SD would mean the")
        print("quantization a study happened to download is as consequential as the")
        print("sentence it happened to write, and no audit in our table reports either.")
        OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
