"""Does the between-wording dispersion really differ across postings?

WHY THIS EXISTS. §4.5 claims that the occupation changes the dispersion itself,
and rests that claim on the SPREAD of the between-wording standard deviation
across three postings -- max minus min of three SDs -- reported with a
percentile bootstrap interval and the observation that the interval excludes
zero.

That observation is arithmetic, not evidence. Max minus min of three
non-negative quantities is non-negative by construction, so a percentile
interval on it CANNOT contain zero no matter what the data says. An audit of
this paper caught it. A statistic whose interval excludes zero by construction
carries no information about the null, and the null here is the interesting
one: that the three postings share a single true between-wording dispersion and
the observed spread is what three noisy estimates of one number look like.

WHAT THIS DOES INSTEAD. A permutation test of equal dispersion, calibrated
against exactly that null.

  1. Build, per model, the 3 x 12 matrix of per-wording probability of
     superiority: three postings by twelve wordings. Each cell is a mean over
     the pairs and strength levels, computed the same way §4.1 computes it.
  2. Center each posting's twelve values on that posting's own mean. This
     removes the EFFECT differences between postings, which are not what is
     being tested, and leaves the deviations that carry the dispersion.
  3. Under the null the 36 centered deviations are exchangeable across
     postings. So permute them, reassign twelve to each posting, recompute the
     three SDs and take max minus min.
  4. The p-value is the fraction of permutations whose spread is at least the
     observed one.

This is a Fligner-Killeen-style permutation test of homogeneity of dispersion,
done by permutation rather than by an asymptotic reference distribution because
there are three groups of twelve.

WHAT IT ASSUMES, STATED. The twelve per-wording estimates within a posting are
treated as exchangeable draws under the null. They carry sampling error of
their own, and if that error differed systematically by posting the test would
see it as a dispersion difference. The design holds it comparable -- the same
twelve wordings, the same twelve name pairs, the same three strength levels in
every posting -- but a posting sitting near the top or bottom of the logistic
has less room to vary, which is a real asymmetry and is why the SECOND
statistic below is reported beside the first.

THE SECOND STATISTIC. The spread is a two-point summary that throws away the
middle group. The variance ratio max(SD^2)/min(SD^2) is reported under the same
permutation, so a reader can see whether the conclusion depends on the summary.

    C:/research-toolchain/venv/Scripts/python.exe \
        paper-a/src/analyze_occupation_null.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "paper-a" / "data"
OUT = D / "occupation" / "occupation_dispersion_null.json"

ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SHORT = {"llama-2-7b-chat": "Llama-2-7B",
         "llama-3.1-8b-instruct": "Llama-3.1-8B",
         "mistral-7b-instruct-v0.1": "Mistral v0.1",
         "mistral-7b-instruct-v0.3": "Mistral v0.3"}
N_PERM = 20000
SEED = 20260801


def rows_for(model: str):
    """Every usable pair row for this model, tagged with its posting.

    BA comes from the main wording study and SWE/RN from the occupation study,
    because BA IS the main study's posting -- rerunning it would have produced
    a second, needlessly different estimate of the same cell.
    """
    out = []
    ba = D / "delta_stability" / f"delta_{model}.jsonl"
    if ba.exists():
        for line in ba.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["occupation"] = "BA"
            out.append(r)
    occ = D / "occupation" / f"occ_{model}.jsonl"
    if occ.exists():
        for line in occ.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return [r for r in out
            if not r.get("error")
            and r.get("white_margin") is not None
            and r.get("black_margin") is not None]


def ps_matrix(rows):
    """(postings, wordings, 3 x 12 matrix of probability of superiority).

    A cell is scored 1 when the white-named margin is larger, 0 when smaller
    and 0.5 on an exact tie -- identical to effectsize.superiority, so the mean
    of the cube IS the probability of superiority.
    """
    occs = sorted({r["occupation"] for r in rows})
    wordings = sorted({r["variant"] for r in rows})
    M = np.full((len(occs), len(wordings)), np.nan)
    counts = np.zeros((len(occs), len(wordings)), dtype=int)
    for i, o in enumerate(occs):
        for j, w in enumerate(wordings):
            sel = [r for r in rows
                   if r["occupation"] == o and r["variant"] == w]
            if not sel:
                continue
            s = np.array([1.0 if r["white_margin"] > r["black_margin"]
                          else (0.0 if r["white_margin"] < r["black_margin"]
                                else 0.5) for r in sel])
            M[i, j] = s.mean()
            counts[i, j] = len(sel)
    return occs, wordings, M, counts


def spread(M):
    sd = M.std(axis=1, ddof=1)
    return float(sd.max() - sd.min()), sd


def var_ratio(M):
    v = M.var(axis=1, ddof=1)
    return float(v.max() / v.min()) if v.min() > 0 else float("inf")


def permute(M, n_perm: int, rng):
    """Null distribution of both statistics under exchangeable deviations."""
    centred = M - M.mean(axis=1, keepdims=True)
    pool = centred.ravel()
    n_occ, n_w = M.shape
    sp = np.empty(n_perm)
    vr = np.empty(n_perm)
    for b in range(n_perm):
        P = rng.permutation(pool).reshape(n_occ, n_w)
        sp[b], _ = spread(P)
        vr[b] = var_ratio(P)
    return sp, vr


def main() -> int:
    rng = np.random.default_rng(SEED)
    out = {
        "_what": "Permutation test of the null that the three postings share "
                 "one between-wording dispersion.",
        "_why": "The spread statistic §4.5 quoted is max minus min of three "
                "non-negative SDs. Its percentile bootstrap interval cannot "
                "contain zero by construction, so 'the interval excludes "
                "zero' was arithmetic rather than evidence.",
        "_test": "Deviations from each posting's own mean are pooled and "
                 "permuted across postings; the p-value is the fraction of "
                 "permutations whose statistic is at least the observed one.",
        "n_permutations": N_PERM, "seed": SEED, "models": {},
    }
    print(f"{'model':<16}{'postings':>10}{'spread':>10}{'p':>9}"
          f"{'var ratio':>11}{'p':>9}   SD by posting")
    print("-" * 96)
    for m in ORDER:
        rows = rows_for(m)
        if not rows:
            continue
        occs, wordings, M, counts = ps_matrix(rows)
        if np.isnan(M).any() or M.shape[0] < 3:
            print(f"{SHORT[m]:<16}  incomplete grid, skipped")
            continue
        obs_sp, sd = spread(M)
        obs_vr = var_ratio(M)
        sp, vr = permute(M, N_PERM, rng)
        # +1 in numerator and denominator: the observed arrangement is itself
        # one of the equally likely ones under the null, and omitting it can
        # return p = 0, which no permutation test can license.
        p_sp = float((1 + (sp >= obs_sp).sum()) / (1 + N_PERM))
        p_vr = float((1 + (vr >= obs_vr).sum()) / (1 + N_PERM))
        out["models"][m] = dict(
            postings=occs, n_wordings=len(wordings),
            n_pairs_per_cell=counts.tolist(),
            sd_by_posting={o: float(s) for o, s in zip(occs, sd)},
            spread=dict(est=obs_sp, p=p_sp,
                        null_median=float(np.median(sp)),
                        null_p95=float(np.percentile(sp, 95))),
            variance_ratio=dict(est=obs_vr, p=p_vr,
                                null_median=float(np.median(vr)),
                                null_p95=float(np.percentile(vr, 95))),
            significant_at_05=bool(p_sp < 0.05))
        sds = "  ".join(f"{o}={s:.4f}" for o, s in zip(occs, sd))
        print(f"{SHORT[m]:<16}{len(occs):>10}{obs_sp:>10.4f}{p_sp:>9.4f}"
              f"{obs_vr:>11.2f}{p_vr:>9.4f}   {sds}")

    # ---- the EFFECT across postings, which is a different claim ----------
    # Withdrawing the dispersion claim does not touch this one, and the two
    # were entangled in §4.5. The occupation component of the dispersion
    # budget is the SD of the EFFECT across the three postings, not of the
    # dispersion, so it needs its own test: is the largest pairwise difference
    # in effect between two postings distinguishable from zero?
    #
    # The resampling unit is the NAME PAIR and the same pairs are resampled in
    # all three postings at once, because the three postings score the same
    # twelve pairs and a bootstrap that drew them independently would discard
    # that pairing and inflate the interval.
    print()
    print(f"{'model':<16}{'largest pairwise effect gap':>30}{'95% CI':>22}"
          f"{'excl. 0':>9}")
    print("-" * 96)
    for m in ORDER:
        if m not in out["models"]:
            continue
        rows = rows_for(m)
        occs = sorted({r["occupation"] for r in rows})
        pairs = sorted({r["pair"] for r in rows})
        by = {(o, p): [] for o in occs for p in pairs}
        for r in rows:
            s = (1.0 if r["white_margin"] > r["black_margin"]
                 else (0.0 if r["white_margin"] < r["black_margin"] else 0.5))
            by[(r["occupation"], r["pair"])].append(s)
        cube = np.array([[np.mean(by[(o, p)]) if by[(o, p)] else np.nan
                          for p in pairs] for o in occs])
        if np.isnan(cube).any():
            continue

        def gap(c):
            e = c.mean(axis=1)
            return float(e.max() - e.min())

        obs = gap(cube)
        boot = np.empty(N_PERM // 4)
        for b in range(len(boot)):
            idx = rng.integers(0, len(pairs), len(pairs))
            boot[b] = gap(cube[:, idx])
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        # A max-minus-min is non-negative by construction here too, so the
        # interval is reported for magnitude and the TEST is a permutation of
        # posting labels within each name pair, which is valid because every
        # posting scores every pair.
        nullg = np.empty(N_PERM // 4)
        for b in range(len(nullg)):
            perm = np.array([rng.permutation(cube[:, j])
                             for j in range(cube.shape[1])]).T
            nullg[b] = gap(perm)
        p_gap = float((1 + (nullg >= obs).sum()) / (1 + len(nullg)))
        out["models"][m]["effect_gap_across_postings"] = dict(
            est=obs, ci=ci, p_permutation=p_gap,
            effect_by_posting={o: float(v)
                               for o, v in zip(occs, cube.mean(axis=1))},
            note="probability-of-superiority scale; max minus min over the "
                 "three postings; p from permuting posting labels within "
                 "name pair, so the pairing is preserved under the null")
        print(f"{SHORT[m]:<16}{obs:>30.4f}"
              f"{f'[{ci[0]:.4f}, {ci[1]:.4f}]':>22}"
              f"{'yes' if p_gap < 0.05 else 'no':>9}   p={p_gap:.4f}")

    mods = out["models"]
    n_sig = sum(1 for v in mods.values() if v["significant_at_05"])

    _sp = sorted(v["spread"]["p"] for v in mods.values())
    _bh_any = any(pv <= 0.05 * (k + 1) / len(_sp)
                  for k, pv in enumerate(_sp))
    out["n_effect_gap_significant"] = sum(
        1 for v in mods.values()
        if v.get("effect_gap_across_postings", {}).get("p_permutation", 1)
        < 0.05)
    out["summary"] = dict(
        n_models=len(mods), n_significant_spread=n_sig,
        n_significant_variance_ratio=sum(
            1 for v in mods.values() if v["variance_ratio"]["p"] < 0.05),
        min_p=min((v["spread"]["p"] for v in mods.values()), default=None),
        n_effect_gap_significant=out["n_effect_gap_significant"],
        # n_sig counts RAW per-model rejections; a verdict on any-of-four
        # uncorrected tests is the multiple-comparison practice the paper
        # criticises. The verdict now requires survival under BH across the
        # four models' spread p-values (the pre-named primary statistic).
        verdict=("the postings do not share one between-wording dispersion"
                 if _bh_any else
                 "the observed spread is within what one shared dispersion "
                 "produces; §4.5 cannot claim the postings differ in "
                 "dispersion on this evidence"))
    print()
    print(f"  spread significant on {n_sig} of {len(mods)} models")
    print(f"  VERDICT: {out['summary']['verdict']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
