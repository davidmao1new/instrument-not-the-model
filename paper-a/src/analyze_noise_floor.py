"""How much of the measured wording variance is arithmetic?

THE CLAIM BEING RETRACTED. Section 5.1 of the paper states: "Two of the twelve
wordings are token-identical by construction, an accidental control we did not
design. Their estimated effects differ by 0.03 pp. The measurement is
deterministic, so variation elsewhere is not sampling noise and no account
resting on numerical instability can be correct."

The first two sentences are true. The third does not follow from them and is
false. S1 and N1 are byte-identical -- same system string, same user message --
and their AGGREGATE effects agree to 0.03 pp because per-cell disagreements
average out. Per cell they do not agree:

    llama-2-7b-chat            24 of 36 cells identical, max |diff| 0.323
    llama-3.1-8b-instruct       5 of 36 cells identical, max |diff| 0.219
    mistral-7b-instruct-v0.1   21 of 36 cells identical, max |diff| 0.099
    mistral-7b-instruct-v0.3   16 of 36 cells identical, max |diff| 0.364

Identical bytes, temperature 0, greedy decoding, and the answers still move.
The cause is batched inference: llama-server runs with four parallel slots and
the client issues four concurrent requests, so a given prompt is matrix-
multiplied alongside whatever else is in flight, and floating-point reduction
order is not invariant to batch composition. Nothing is wrong with the server;
this is what batched GPU inference does.

WHY THIS IS WORTH MEASURING RATHER THAN JUST CONFESSING. S1 and N1 are a free
replicate experiment. Two measurements of the same quantity give a direct
estimate of the measurement error, and that turns an embarrassment into the one
thing a measurement-validity paper most needs: a NOISE FLOOR. Every variance the
paper attributes to wording can then be stated as a multiple of the noise it
would have shown even if wording did nothing at all.

    sigma_noise   per-cell measurement SD, from the paired replicate
    sigma_variant between-wording SD of the estimated effect

The comparison must be made at the level the paper reports, which is the
per-wording MEAN over cells, not the individual cell. Noise on a mean of n cells
falls as sigma_noise / sqrt(n), so the fair question is whether sigma_variant
exceeds that. If it does not, the wording result is arithmetic and the paper
should be withdrawn. If it does, the paper can say by how much.

    .venv/Scripts/python.exe paper-a/src/analyze_noise_floor.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DELTA = ROOT / "paper-a" / "data" / "delta_stability"
NAMES = ROOT / "paper-a" / "data" / "names"
OUT = DELTA / "noise_floor.json"


def load(d, pattern, key):
    rows = {}
    for f in sorted(pathlib.Path(d).glob(pattern)):
        for r in st.read_jsonl(f):
            if (r.get("white_margin") is not None
                    and r.get("black_margin") is not None):
                rows[tuple(r[k] for k in key)] = r
    return rows


def main() -> int:
    d2 = load(DELTA, "delta_*.jsonl", ("model", "variant", "template", "pair"))
    if not d2:
        sys.exit("no Study 2 data")

    out = {}
    print("=" * 96)
    print("NOISE FLOOR from the S1/N1 byte-identical replicate.")
    print("Both are the same prompt. Any disagreement is measurement error.")
    print("=" * 96)
    print(f"{'model':<26}{'n cells':>8}{'identical':>11}{'sigma_noise':>13}"
          f"{'max|diff|':>11}{'noise on a':>12}")
    print(f"{'':<26}{'':>8}{'':>11}{'(per cell)':>13}{'':>11}{'12-cell mean':>12}")

    models = sorted({k[0] for k in d2})
    for m in models:
        pairs = []
        for (mm, v, t, p), r in d2.items():
            if mm != m or v != "S1":
                continue
            q = d2.get((m, "N1", t, p))
            if q is None:
                continue
            a = r["white_margin"] - r["black_margin"]
            b = q["white_margin"] - q["black_margin"]
            pairs.append((a, b))
        if not pairs:
            continue
        arr = np.array(pairs)
        diff = arr[:, 0] - arr[:, 1]
        # Two independent measurements of one quantity: Var(diff) = 2 sigma^2.
        sigma = float(np.std(diff, ddof=1) / np.sqrt(2))
        n_cells_per_variant = len({(t, p) for (mm, v, t, p) in d2
                                   if mm == m and v == "S1"})
        out[m] = dict(
            n_replicate_cells=len(diff),
            n_identical=int((diff == 0).sum()),
            frac_identical=float((diff == 0).mean()),
            sigma_noise_per_cell=sigma,
            max_abs_diff=float(np.abs(diff).max()),
            n_cells_per_variant=n_cells_per_variant,
            sigma_noise_on_variant_mean=sigma / np.sqrt(n_cells_per_variant))
        o = out[m]
        print(f"{m:<26}{len(diff):>8}{o['n_identical']:>7}/{len(diff):<3}"
              f"{sigma:>13.5f}{o['max_abs_diff']:>11.4f}"
              f"{o['sigma_noise_on_variant_mean']:>12.5f}")

    # ---- the comparison that decides whether the paper stands -------------
    print()
    print("=" * 96)
    print("DOES WORDING MOVE THE EFFECT BY MORE THAN THE ARITHMETIC DOES?")
    print("sigma_variant is the SD of the per-wording mean effect across the 12 wordings.")
    print("It is compared against the noise expected on a per-wording mean of the same size.")
    print("=" * 96)
    print(f"{'model':<26}{'sigma_variant':>15}{'noise floor':>13}{'ratio':>9}"
          f"{'verdict':>34}")
    for m in models:
        if m not in out:
            continue
        byv = defaultdict(list)
        for (mm, v, t, p), r in d2.items():
            if mm == m:
                byv[v].append(r["white_margin"] - r["black_margin"])
        per = np.array([np.mean(x) for x in byv.values()])
        sd = float(per.std(ddof=1))
        floor = out[m]["sigma_noise_on_variant_mean"]
        ratio = sd / floor if floor > 0 else float("inf")
        out[m]["sigma_variant_raw"] = sd
        out[m]["ratio_variant_to_noise"] = ratio
        verdict = ("wording dominates" if ratio > 3 else
                   "wording exceeds noise" if ratio > 1.5 else
                   "NOT SEPARABLE FROM NOISE")
        print(f"{m:<26}{sd:>15.5f}{floor:>13.5f}{ratio:>9.1f}x{verdict:>33}")

    # ---- cross-session replication, if Study 4 has landed ------------------
    d4 = load(NAMES, "names_*.jsonl", ("model", "variant", "template", "pair"))
    if d4:
        print()
        print("=" * 96)
        print("CROSS-SESSION REPLICATION. Study 4 re-measures names Study 2 already")
        print("measured, in a different process on a different day. Stronger than the")
        print("within-run replicate above, because the server was restarted in between.")
        print("=" * 96)
        idx2 = defaultdict(dict)
        for (m, v, t, p), r in d2.items():
            idx2[m][(v, t, r["white_name"])] = r["white_margin"]
            idx2[m][(v, t, r["black_name"])] = r["black_margin"]
        print(f"{'model':<26}{'shared cells':>14}{'identical':>11}"
              f"{'sigma_cross':>13}{'max|diff|':>11}")
        for m in sorted({k[0] for k in d4}):
            diffs = []
            for (mm, v, t, p), r in d4.items():
                if mm != m:
                    continue
                for side in ("white", "black"):
                    key = (v, t, r[f"{side}_name"])
                    if key in idx2.get(m, {}):
                        diffs.append(r[f"{side}_margin"] - idx2[m][key])
            if not diffs:
                continue
            a = np.array(diffs)
            s = float(np.std(a, ddof=1) / np.sqrt(2))
            out.setdefault(m, {})["cross_session"] = dict(
                n=len(a), n_identical=int((a == 0).sum()),
                sigma=s, max_abs=float(np.abs(a).max()))
            print(f"{m:<26}{len(a):>14}{int((a==0).sum()):>7}/{len(a):<4}"
                  f"{s:>13.5f}{np.abs(a).max():>11.4f}")

    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
