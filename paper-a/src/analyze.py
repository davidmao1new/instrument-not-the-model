"""
Analysis. Mixed-effects logistic regression + the pre-specified contrasts.

The three things reviewers will check, in order:

  1. Did you report EFFECT SIZES in percentage points, not just p-values?
     With 10,000 pairs everything is significant. Only magnitude means anything.

  2. Did you test H2b as an INTERACTION, rather than comparing two separate
     significance tests? "Significant in A, not in B" is not evidence that A
     and B differ. This is the most common error in this literature.

  3. Did you correct for multiple comparisons across model x group cells?

Everything below is built around those three.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("data/results.jsonl")
OUT = Path("data/analysis")

PRE_2024_MODELS: set[str] = set()   # fill from config.yaml
POST_2024_MODELS: set[str] = set()


def load() -> pd.DataFrame:
    if not RESULTS.exists():
        raise FileNotFoundError(f"No results at {RESULTS}. Run the audit first.")
    df = pd.DataFrame([json.loads(l) for l in RESULTS.open(encoding="utf-8")])

    df["generation"] = np.where(
        df["model"].isin(PRE_2024_MODELS), "pre_2024",
        np.where(df["model"].isin(POST_2024_MODELS), "post_2024", "unclassified"),
    )
    if (df["generation"] == "unclassified").any():
        bad = sorted(df.loc[df["generation"] == "unclassified", "model"].unique())
        raise ValueError(
            f"Models not assigned to a generation: {bad}. "
            "Every model must be classified in config.yaml before analysis, "
            "and the classification must be pre-specified, not chosen after "
            "seeing the results."
        )
    return df


def refusal_report(df: pd.DataFrame) -> pd.DataFrame:
    """Refusal and parse-failure rates by model and demographic condition.

    REPORT THIS IN THE PAPER EVEN IF IT IS BORING. If refusal is differential
    by race signal, the advance-rate analysis is conditioned on a
    non-random subset and every downstream estimate is suspect.
    """
    g = (df.groupby(["model", "race_signal", "channel"])
           .agg(n=("advance", "size"),
                refused=("refused", "mean"),
                parse_fail=("parse_ok", lambda s: 1 - s.mean()),
                errored=("error", lambda s: s.notna().mean()))
           .reset_index())
    g["usable"] = 1 - g["refused"] - g["parse_fail"]
    return g


def check_differential_refusal(df: pd.DataFrame, tol: float = 0.02) -> list[str]:
    """Flag models where refusal rate varies by race signal beyond `tol`."""
    warnings: list[str] = []
    for model, sub in df.groupby("model"):
        rates = sub.groupby("race_signal")["refused"].mean()
        if len(rates) > 1 and (rates.max() - rates.min()) > tol:
            warnings.append(
                f"{model}: refusal varies by race signal "
                f"({rates.min():.3f} to {rates.max():.3f}). "
                "Advance-rate estimates are conditioned on a non-random subset."
            )
    return warnings


def paired_gaps(df: pd.DataFrame, reference: str = "white") -> pd.DataFrame:
    """Within-pair advance-rate gap in percentage points, with bootstrap CIs.

    A pair is one resume template + job posting + schema + channel, evaluated
    under two race signals. The gap is (rate_group - rate_reference) * 100.
    Negative means the group is favored over the reference.
    """
    usable = df[(df["parse_ok"]) & (~df["refused"]) & (df["advance"].notna())].copy()
    rows = []

    keys = ["model", "generation", "channel", "schema", "gender_signal"]
    for key, sub in usable.groupby(keys):
        ref = sub[sub["race_signal"] == reference]
        if ref.empty:
            continue
        ref_rate = ref["advance"].mean()

        for group, gsub in sub.groupby("race_signal"):
            if group == reference:
                continue
            gap = (gsub["advance"].mean() - ref_rate) * 100
            lo, hi = _bootstrap_gap_ci(gsub["advance"].values, ref["advance"].values)
            rows.append({
                **dict(zip(keys, key)),
                "group": group, "reference": reference,
                "n_group": len(gsub), "n_ref": len(ref),
                "gap_pp": gap, "ci_lo_pp": lo, "ci_hi_pp": hi,
            })
    return pd.DataFrame(rows)


def _bootstrap_gap_ci(a: np.ndarray, b: np.ndarray,
                      n_boot: int = 2000, seed: int = 20260726) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = a.astype(float); b = b.astype(float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (rng.choice(a, a.size, replace=True).mean()
                    - rng.choice(b, b.size, replace=True).mean()) * 100
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def benjamini_hochberg(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH-adjusted p-values. Report raw and adjusted side by side."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def fit_primary_model(df: pd.DataFrame):
    """Mixed-effects logistic regression, the primary specification.

    advance ~ race_signal * channel * schema * generation
              + (1 | template_id) + (1 | job_id)

    statsmodels' MixedLM is linear-only, so for a binomial outcome with
    crossed random effects the practical options are:
      - Bayesian: pymc or numpyro (recommended; gives you posterior intervals
        on the interaction, which is exactly what H2b needs)
      - R via rpy2: lme4::glmer, the field standard
      - Approximation: GEE with cluster-robust SEs (weaker but fast)

    Do not silently fall back to plain logistic regression while calling it
    mixed-effects in the paper.
    """
    raise NotImplementedError(
        "Choose a backend and implement explicitly. Recommended: numpyro or\n"
        "pymc. NOTE, updated 2026-07-30: what actually got fitted is GAUSSIAN\n"
        "crossed random effects on a continuous log-odds decision margin, not a\n"
        "hierarchical LOGISTIC model. The thresholded binary verdict is\n"
        "degenerate on this panel -- one model accepts 99.8% of candidates -- so\n"
        "a logistic outcome has almost no dynamic range to model. Crossed\n"
        "random intercepts\n"
        "on template_id and job_id. H2b is a channel x generation interaction,\n"
        "so you need a posterior on that interaction term, not two separate\n"
        "significance tests. See PROTOCOL.md section 7."
    )


def h2b_ratio_table(gaps: pd.DataFrame) -> pd.DataFrame:
    """Descriptive companion to the H2b interaction test.

    Ratio of |proxy-channel gap| to |name-channel gap|, by generation.
    H2b predicts this ratio is HIGHER post-2024: models moved their
    sensitivity from explicit names to distributed proxies rather than
    eliminating it.

    THIS TABLE IS DESCRIPTIVE ONLY. The inferential claim must come from the
    interaction term in fit_primary_model(). Ratios of noisy estimates are
    themselves very noisy and cannot carry the conclusion.
    """
    piv = (gaps.pivot_table(index=["model", "generation", "group", "schema"],
                            columns="channel", values="gap_pp")
                .reset_index())
    if "name_only" not in piv or "proxy_only" not in piv:
        raise ValueError("Need both 'name_only' and 'proxy_only' channels.")
    piv["ratio_proxy_to_name"] = (
        piv["proxy_only"].abs() / piv["name_only"].abs().replace(0, np.nan)
    )
    return piv


def schema_sensitivity_table(gaps: pd.DataFrame) -> pd.DataFrame:
    """H3: does identical content produce different disparity by schema?

    H3b predicts ATS_SUBSET reduces name-channel disparity while increasing
    proxy-channel disparity. If that holds, the industry-standard move of
    parsing resumes into fields makes the proxy problem worse while making
    the name problem look better on a compliance audit. That is the headline.
    """
    return (gaps.pivot_table(index=["model", "generation", "group", "channel"],
                             columns="schema", values="gap_pp")
                .reset_index())


def run_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()

    ref = refusal_report(df)
    ref.to_csv(OUT / "refusal_report.csv", index=False)
    for w in check_differential_refusal(df):
        print("WARNING:", w)

    gaps = paired_gaps(df)
    gaps.to_csv(OUT / "paired_gaps.csv", index=False)

    try:
        h2b_ratio_table(gaps).to_csv(OUT / "h2b_ratios.csv", index=False)
        schema_sensitivity_table(gaps).to_csv(OUT / "h3_schema.csv", index=False)
    except ValueError as e:
        print("Skipped derived tables:", e)

    print(f"Wrote descriptive tables to {OUT}")
    print("Next: implement fit_primary_model() for the inferential claims.")


if __name__ == "__main__":
    run_all()
