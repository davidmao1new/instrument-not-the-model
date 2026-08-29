"""What is the error rate of the screening rule §9.1 recommends?

THE OBJECTION THIS ANSWERS, WHICH IS THE PAPER'S OWN ARGUMENT TURNED AROUND.
§9.1 tells an auditor to treat a disparity as established only if the sign is
stable across wordings AND the smallest per-wording estimate still excludes the
null. It then concedes that this is "a screening criterion, not a statistical
test with a calibrated error rate", because "taking the minimum over twelve
estimates is a selection, and its behaviour under the null depends on how
correlated the wordings are, which is not something we have characterised."

A paper whose central complaint is that the field reports numbers with
uncharacterised properties cannot leave its own recommendation in that state.
This characterises it.

THE RULE IS NOT ACTUALLY A MINIMUM-SELECTION PROBLEM. Requiring that the sign
be stable and that the SMALLEST estimate exclude the null is the same event as
requiring that EVERY wording's interval exclude the null on the same side. Once
stated that way the null distribution is a standard quantity and no simulation
is needed to get it right -- though we run one anyway as a check.

THE MODEL. Under the global null every wording has a true effect of zero. The
k per-wording estimates are jointly normal, mean zero, equicorrelated at rho:

    theta_w = sqrt(rho) * U + sqrt(1 - rho) * E_w,     U, E_w ~ N(0, 1) iid

which is exactly the structure the design induces -- the wordings share the
same name pairs, so U is what the pairs contribute in common and E_w is what
the wording adds. Each wording reports theta_w +/- z * se, so "excludes the
null" is |theta_w| > z. Conditioning on U makes the k estimates independent:

    P(all k exceed +z) = INTEGRAL phi(u) Phi((sqrt(rho) u - z)/sqrt(1-rho))^k du

and the false-positive rate is twice that, the two tails being disjoint.

WHAT COMES OUT, and it is a stronger result than we expected. The rate is
bounded above by the per-wording alpha for every rho, k and z: it rises
monotonically in rho from alpha^k at rho = 0 to alpha at rho = 1, and never
exceeds it. So the rule cannot be more liberal than the single-wording test an
auditor would otherwise have run, whatever the correlation is -- which is
precisely the thing §9.1 said it could not characterise. The selection worry
had the sign backwards: taking the minimum makes the rule harder to pass, not
easier.

We also estimate rho from this paper's own data, so the panel can be placed on
the curve rather than left somewhere on it.

    sh paper-a/src/_py.sh paper-a/src/analyze_screening_rule_null.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np
from scipy import integrate, stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "paper-a" / "data"
OUT = D / "instrument" / "screening_rule_null.json"

RHOS = [0.0, 0.3, 0.6, 0.9]
ALPHA = 0.05
Z = float(stats.norm.ppf(1 - ALPHA / 2))     # 1.9600, the usual two-sided cut
K_MAIN = 12                                   # wordings in this paper's design
K_GRID = [3, 6, 12, 24]
N_SIM = 400_000
SEED = 20260806


def fpr_exact(rho: float, k: int, z: float = Z) -> float:
    """P(all k estimates exclude zero on the SAME side), under the global null.

    Conditioning on the shared component U makes the k draws independent, so
    the k-fold probability is a one-dimensional integral. Exact to quadrature
    tolerance, which matters because at rho = 0 the answer is around 1e-19 and
    no simulation could ever see it.
    """
    if rho <= 0:
        return 2.0 * stats.norm.sf(z) ** k
    if rho >= 1:
        return 2.0 * stats.norm.sf(z)
    a, b = np.sqrt(rho), np.sqrt(1.0 - rho)

    def integrand(u):
        return stats.norm.pdf(u) * stats.norm.cdf((a * u - z) / b) ** k

    val, _err = integrate.quad(integrand, -12, 12, limit=400)
    return float(2.0 * val)


def fpr_sim(rho: float, k: int, n: int = N_SIM, z: float = Z, rng=None) -> dict:
    """The same quantity by Monte Carlo, plus the two halves of the rule.

    The rule is stated as a conjunction -- sign stable AND the smallest
    estimate excludes the null -- so we report each half separately. Anyone
    who reads the rule as two independent hurdles can see that they are not
    independent, and that the conjunction is what the second half already
    implies.
    """
    rng = rng or np.random.default_rng(SEED)
    u = rng.standard_normal((n, 1))
    e = rng.standard_normal((n, k))
    th = np.sqrt(rho) * u + np.sqrt(1.0 - rho) * e
    sign_stable = (np.all(th > 0, axis=1) | np.all(th < 0, axis=1))
    min_excludes = np.min(np.abs(th), axis=1) > z
    return dict(
        rule=float(np.mean(sign_stable & min_excludes)),
        sign_stable_only=float(np.mean(sign_stable)),
        min_excludes_only=float(np.mean(min_excludes)),
        any_one_excludes=float(np.mean(np.max(np.abs(th), axis=1) > z)),
    )


def empirical_rho() -> dict:
    """How correlated ARE the wordings, on this paper's own data?

    Builds, per model, the matrix of paired differences indexed by name pair
    and wording -- averaging over templates, because a wording mean in this
    paper is an average over templates -- and takes the mean off-diagonal
    correlation between wording columns. That is the rho the equicorrelated
    model above is parameterised by.
    """
    out = {}
    for f in sorted((D / "delta_stability").glob("delta_*.jsonl")):
        cells = collections.defaultdict(list)
        model = None
        for ln in f.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("white_margin") is None or r.get("black_margin") is None:
                continue
            model = r.get("model", model)
            cells[(r["pair"], r["variant"])].append(
                r["white_margin"] - r["black_margin"])
        if not cells or model is None:
            continue
        pairs = sorted({p for p, _v in cells})
        variants = sorted({v for _p, v in cells})
        if len(pairs) < 3 or len(variants) < 2:
            continue
        X = np.full((len(pairs), len(variants)), np.nan)
        for i, p in enumerate(pairs):
            for j, v in enumerate(variants):
                vals = cells.get((p, v))
                if vals:
                    X[i, j] = float(np.mean(vals))
        if np.isnan(X).any():
            keep = ~np.isnan(X).any(axis=1)
            X = X[keep]
        if X.shape[0] < 3:
            continue
        # a column with no variation carries no correlation information
        live = X[:, X.std(axis=0) > 1e-12]
        if live.shape[1] < 2:
            out[model] = dict(n_pairs=int(X.shape[0]),
                              n_wordings=int(X.shape[1]),
                              mean_offdiagonal_r=None,
                              note="no wording column varies across pairs")
            continue
        C = np.corrcoef(live, rowvar=False)
        iu = np.triu_indices_from(C, k=1)
        r = C[iu]
        out[model] = dict(
            n_pairs=int(live.shape[0]), n_wordings=int(live.shape[1]),
            mean_offdiagonal_r=float(np.mean(r)),
            median_offdiagonal_r=float(np.median(r)),
            min_offdiagonal_r=float(np.min(r)),
            max_offdiagonal_r=float(np.max(r)),
        )
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    rows = []
    for rho in RHOS:
        ex = fpr_exact(rho, K_MAIN)
        sim = fpr_sim(rho, K_MAIN, rng=rng)
        rows.append(dict(rho=rho, k=K_MAIN, exact=ex, **sim))

    by_k = {}
    for k in K_GRID:
        by_k[str(k)] = {str(r): fpr_exact(r, k) for r in RHOS + [0.99]}

    emp = empirical_rho()
    emp_vals = [v["mean_offdiagonal_r"] for v in emp.values()
                if v.get("mean_offdiagonal_r") is not None]
    emp_min = min(emp_vals) if emp_vals else None
    emp_max = max(emp_vals) if emp_vals else None
    at_emp = ({m: fpr_exact(max(0.0, min(1.0, v["mean_offdiagonal_r"])), K_MAIN)
               for m, v in emp.items()
               if v.get("mean_offdiagonal_r") is not None})

    # the headline: does the rate ever exceed the per-wording alpha?
    dense = [fpr_exact(r, K_MAIN) for r in np.linspace(0, 0.999, 200)]
    bounded = bool(max(dense) <= ALPHA + 1e-12)
    monotone = bool(np.all(np.diff(dense) >= -1e-12))

    out = {
        "_what": "The false-positive rate of the screening rule of §9.1 -- "
                 "sign stable across wordings and the smallest per-wording "
                 "estimate excluding the null -- under the global null, as a "
                 "function of how correlated the wordings are.",
        "_why": "§9.1 recommended the rule and said its behaviour under the "
                "null 'depends on how correlated the wordings are, which is "
                "not something we have characterised'. A paper about "
                "uncharacterised measurement choices should not leave its own "
                "recommendation uncharacterised.",
        "_model": "theta_w = sqrt(rho) U + sqrt(1-rho) E_w, U and E_w iid "
                  "N(0,1); reject when every wording's two-sided interval "
                  "excludes zero on the same side.",
        "_equivalence": "sign stable AND min|estimate| excludes the null is "
                        "the same event as ALL intervals excluding the null on "
                        "one side, so the rule is not a minimum-selection "
                        "problem and has a closed form.",
        "alpha_per_wording": ALPHA,
        "z": Z,
        "k_wordings": K_MAIN,
        "n_sim": N_SIM,
        "seed": SEED,
        "by_rho": rows,
        "by_k_and_rho": by_k,
        "rate_never_exceeds_alpha": bounded,
        "rate_monotone_increasing_in_rho": monotone,
        "max_rate_over_rho": float(max(dense)),
        "empirical_rho_per_model": emp,
        "empirical_rho_min": emp_min,
        "empirical_rho_max": emp_max,
        "rate_at_empirical_rho": at_emp,
        "rate_at_empirical_rho_max": (max(at_emp.values()) if at_emp else None),
        "_verdict": (
            "The rule is conservative for every correlation: its false-positive "
            "rate rises monotonically in rho from alpha^k to alpha and never "
            "exceeds alpha, so it cannot be more liberal than the "
            "single-wording test it replaces. The selection worry ran the "
            "wrong way -- taking the minimum makes the rule harder to pass."),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"screening rule: sign stable over k={K_MAIN} wordings AND the "
          f"smallest estimate excludes the null")
    print(f"per-wording alpha = {ALPHA}, z = {Z:.4f}\n")
    print(f"{'rho':>6}{'exact FPR':>14}{'simulated':>12}"
          f"{'sign stable':>14}{'any one excl.':>15}")
    print("-" * 62)
    for r in rows:
        print(f"{r['rho']:>6.2f}{r['exact']:>14.3e}{r['rule']:>12.5f}"
              f"{r['sign_stable_only']:>14.4f}{r['any_one_excludes']:>15.4f}")
    print()
    print(f"  never exceeds the per-wording alpha: {bounded} "
          f"(max {max(dense):.4f} against alpha {ALPHA})")
    print(f"  monotone increasing in rho:          {monotone}")
    print()
    if emp_vals:
        print(f"  wording correlation on our own panel: "
              f"{emp_min:.3f} to {emp_max:.3f}")
        for m, v in sorted(at_emp.items()):
            print(f"    {m:<28} rho={emp[m]['mean_offdiagonal_r']:.3f}  "
                  f"FPR={v:.3e}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
