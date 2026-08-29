"""The dispersion budget, as one figure.

WHY THIS FIGURE DID NOT EXIST AND SHOULD HAVE. §1.2 names "a dispersion budget
for an audited effect" as the paper's first contribution, and until now the
budget was distributed across six tables on three scales. A reader could
assemble it. Nobody will.

WHAT IS PLOTTED. For each model, every unreported choice this paper measures,
expressed as a multiple of that model's own demographic effect, all on the
log-odds scale so the components are commensurable:

    component / |beta|,   beta = the pooled paired effect on the primary posting

A bar reaching 1.0 means the choice moves the estimate by as much as the whole
effect the study exists to report. The measurement noise floor is plotted first,
in outline, because it is the reference that makes the rest interpretable: it is
what a byte-identical repeat gives, and every bar taller than it is dispersion
rather than error.

WHY LOG-ODDS AND NOT THE PRIMARY SCALE. The probability of superiority is the
paper's primary effect size and is scale-free, but two components -- the
quantization shift and the name-draw counterfactual -- are estimated on
log-odds, and converting them requires exactly the fixed-Jacobian move §6.2
shows to be wrong by up to two orders of magnitude. Plotting a budget on a scale
the paper argues against would be self-refuting. The one component that has no
log-odds representation is the pairing (§4.5): re-pairing leaves the mean paired
difference algebraically invariant and moves only the pairwise statistic, so it
is reported in the caption rather than drawn as a bar it cannot honestly own.

    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/figures_budget.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "paper-a" / "data"
FIGS = ROOT / "paper-a" / "figures"
OUT_JSON = D / "delta_stability" / "dispersion_budget.json"

ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SHORT = {"llama-2-7b-chat": "Llama-2-7B", "llama-3.1-8b-instruct": "Llama-3.1-8B",
         "mistral-7b-instruct-v0.1": "Mistral v0.1",
         "mistral-7b-instruct-v0.3": "Mistral v0.3"}


def jload(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build() -> dict:
    s2 = jload(D / "delta_stability" / "study2_v2.json")
    occ = jload(D / "occupation" / "occupation_analysis.json")
    noise = jload(D / "delta_stability" / "noise_floor.json")
    names = jload(D / "names" / "name_variance.json")
    quant = jload(D / "quantization" / "quantization_analysis.json")
    nlen = jload(D / "instrument" / "name_length_effect.json")
    pair = jload(D / "names" / "pairing_freedom.json")

    out = {"_scale": "log-odds; every component divided by |beta| for that model",
           "models": {}}
    for m in ORDER:
        if not s2 or m not in s2:
            continue
        beta = abs(s2[m]["overall"]["logodds"]["est"])
        row = {"beta_logodds": s2[m]["overall"]["logodds"]["est"], "components": {}}

        if noise and m in noise:
            row["components"]["noise floor"] = dict(
                value=noise[m]["sigma_noise_on_variant_mean"],
                kind="sd", note="byte-identical replicate, on a per-wording mean")
        if occ and m in occ and "BA" in occ[m]:
            row["components"]["wording"] = dict(
                value=occ[m]["BA"]["sigma_variant_raw"], kind="sd",
                note="raw SD across twelve wordings")
        # THE POPULATION ARM, not the conditional one. `counterfactual` draws
        # names WITHOUT replacement from the 12-pair list, which applies a
        # finite-population correction and shrinks the dispersion toward zero
        # as k approaches the list size -- at k = 9 of 12 it halves it, for a
        # reason that has nothing to do with names. §4.2 says the paper reports
        # the population estimand and its tables do; this figure did not, and
        # the difference reached the abstract (0.084 against 0.162 on the
        # headline model, which set the paper's "8 %" lower bound).
        if names and m in names and names[m].get(
                "counterfactual_with_replacement"):
            cf = names[m]["counterfactual_with_replacement"]
            for k in (3, 9):
                key = f"draw_{k}_per_race"
                if key in cf:
                    row["components"][f"name draw, k={k}"] = dict(
                        value=cf[key]["beta_sd"], kind="sd",
                        estimand="population (draws with replacement)",
                        note=f"SD over draws of {k} matched pairs")
        if occ and m in occ:
            vals = [occ[m][j]["logodds"] for j in ("BA", "SWE", "RN")
                    if j in occ[m]]
            if len(vals) >= 3:
                row["components"]["occupation"] = dict(
                    value=float(np.std(vals, ddof=1)), kind="sd",
                    note="SD across three structurally matched postings")
        if quant:
            for key, v in quant.items():
                if isinstance(v, dict) and v.get("base") == m:
                    row["components"]["quantization"] = dict(
                        value=abs(v["shift"]), kind="shift",
                        note="Q4_K_M to Q8_0, absolute shift")
        # Read from the clustered arm, whose point estimate is identical but
        # whose uncertainty is the one §6.1 permits. The interval is carried
        # into the artifact so the figure's consumers can see that this
        # component, unlike the SD components, is not separable from zero.
        if nlen and m in nlen and "token_matched_first_name_clustered" in nlen[m]:
            _mm = nlen[m]["token_matched_first_name_clustered"]["matched_minus_all"]
            # THE INTERVAL ON A MAGNITUDE IS NOT THE MAGNITUDES OF THE
            # INTERVAL. This line took abs() of each endpoint and re-sorted,
            # which turns an interval that SPANS zero into one that EXCLUDES
            # it: [-0.044, +0.006] was deposited as [0.006, 0.044]. That
            # happened on three of the four models, so a reader of the
            # released artifact would have concluded the token-matching
            # component is separable from zero on every model -- the exact
            # opposite of what the comment above says the interval is carried
            # to show, and the opposite of what `p` records.
            #
            # When the signed interval covers zero the shift's magnitude can be
            # as small as zero, so the interval on |shift| is [0, max|endpoint|].
            # The signed interval is deposited beside it, because that is the
            # quantity anyone re-deriving the component needs.
            _lo, _hi = _mm["ci"]
            _abs_ci = ([0.0, max(abs(_lo), abs(_hi))] if _lo * _hi <= 0
                       else [min(abs(_lo), abs(_hi)), max(abs(_lo), abs(_hi))])
            row["components"]["token matching"] = dict(
                value=abs(_mm["est"]), kind="shift",
                ci=_abs_ci, ci_signed=[_lo, _hi],
                ci_covers_zero=bool(_lo * _hi <= 0),
                p=_mm["p"], separable_from_zero=bool(_mm["p"] < 0.05),
                note="effect on token-matched pairs minus effect on all "
                     "pairs, resampling first-name pairs")

        for name, c in row["components"].items():
            c["over_beta"] = c["value"] / beta if beta else None
        out["models"][m] = row

    if pair:
        out["pairing_note"] = {
            m: dict(perm_sd_psup=v["perm_sd"],
                    best_worst_range_psup=v["range_best_worst"])
            for m, v in pair.get("models", {}).items()}
    return out


ROWS = ["noise floor", "wording", "name draw, k=3", "name draw, k=9",
        "occupation", "quantization", "token matching"]


def main() -> int:
    data = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Write only when the bytes actually change. An unconditional rewrite moves
    # this artifact's mtime on every figure build, and the fork's own
    # supplementary gate compares the archive against the newest .py/.json
    # under src, data and tests -- so a no-op rewrite failed that gate against
    # an artifact nothing had touched.
    payload = json.dumps(data, indent=2)
    if not (OUT_JSON.exists()
            and OUT_JSON.read_text(encoding="utf-8") == payload):
        OUT_JSON.write_text(payload, encoding="utf-8")

    models = [m for m in ORDER if m in data["models"]]
    fs.use_paper_style()
    fig, axes = plt.subplots(1, len(models), figsize=(fs.FULL_W, 2.5 * fs.HEIGHT_SCALE),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    ymax = 0
    for m in models:
        for r in ROWS:
            c = data["models"][m]["components"].get(r)
            if c and c["over_beta"] is not None:
                ymax = max(ymax, c["over_beta"])
    ymax = min(ymax * 1.18, 8.0)

    for ax, m in zip(axes, models):
        comps = data["models"][m]["components"]
        # Two things are being distinguished: which choice a bar belongs to
        # (carried by position and by the x tick label) and what KIND of
        # quantity it is -- a standard deviation over a set of defensible
        # choices, or a single shift between two specific ones. The kind was
        # carried by colour alone, which does not survive greyscale printing,
        # so it is hatched as well. Colour is now redundant on every axis of
        # this figure.
        xs, hs, cols, edge, hatch = [], [], [], [], []
        for i, r in enumerate(ROWS):
            c = comps.get(r)
            if not c or c["over_beta"] is None:
                # AN ABSENT BAR IS NOT A ZERO. Quantization was measured on two
                # of the four checkpoints, so two panels carried a labelled
                # x-tick with nothing above it, which reads as "this choice
                # moves nothing" -- the opposite of what is known, which is
                # that it was not measured. Say so on the axis.
                ax.annotate("not\nmeasured", (i, ymax * 0.045), ha="center",
                            va="bottom", fontsize=fs.pt(4.4), color=fs.RULE,
                            linespacing=0.95)
                continue
            xs.append(i)
            hs.append(min(c["over_beta"], ymax))
            if r == "noise floor":
                cols.append("white")
                edge.append(fs.INK)
                hatch.append("")
            elif c["kind"] == "shift":
                cols.append(fs.ACCENT2)
                edge.append(fs.INK)
                hatch.append("///")
            else:
                cols.append(fs.ACCENT)
                edge.append(fs.ACCENT)
                hatch.append("")
        bars = ax.bar(xs, hs, width=0.72, color=cols, edgecolor=edge,
                      linewidth=0.6)
        for b, h in zip(bars, hatch):
            if h:
                b.set_hatch(h)
        # clipped bars get a caret so a truncated axis cannot mislead
        for x, r in zip(xs, [ROWS[i] for i in xs]):
            v = comps[r]["over_beta"]
            if v > ymax:
                ax.plot([x], [ymax * 0.985], marker="^", ms=3.2,
                        color=fs.INK, clip_on=False)
                ax.annotate(f"{v:.1f}\u00d7", (x, ymax * 0.90), ha="center",
                            fontsize=fs.pt(5.4), color=fs.INK)
        ax.axhline(1.0, color=fs.INK, lw=0.7, ls=(0, (3, 2)))
        ax.set_xticks(range(len(ROWS)))
        ax.set_xticklabels(ROWS, rotation=55, ha="right", fontsize=fs.pt(5.4))
        ax.set_ylim(0, ymax)
        ax.set_title(SHORT.get(m, m), fontsize=fs.pt(7))
        ax.tick_params(axis="y", labelsize=fs.pt(6))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("multiple of the\nmodel's own effect", fontsize=fs.pt(6.2))

    fig.tight_layout()
    _null = [SHORT[m] for m in models
             if abs(data["models"][m]["beta_logodds"]) < 0.10]
    fs.caption(
        fig, "fig11_dispersion_budget",
        "The unreported choices measured here, each as a multiple of the "
        "effect it perturbs.",
        "Each bar is the dispersion a single unreported choice contributes to "
        "the demographic effect, divided by that model's own effect, on the "
        "log-odds scale so the components are commensurable. These are the "
        "choices that act on the MEASUREMENT; three further choices act after "
        "the data exists -- the resampling unit, the reporting scale and the "
        "pairing -- and are not commensurable with these, so they are "
        "reported in Sections 6.1 to 6.3 instead. The dashed line "
        "at 1.0 is where a choice moves the estimate by as much as the whole "
        "effect the study exists to report. The open bar is the measurement "
        "noise floor -- what a byte-identical repeat gives -- so every bar "
        "taller than it is dispersion rather than error. Solid bars are "
        "standard deviations over a set of defensible choices; hatched bars "
        "are single shifts between two specific choices, not spreads. "
        + (f"On {' and '.join(_null)} the denominator is an effect not "
           "distinguishable from zero, so those panels' heights say more "
           "about a small divisor than about a large numerator. " if _null else "")
        + "The pairing of Section 6.3 has no bar because re-pairing leaves the "
          "mean paired difference algebraically unchanged; it moves the "
          "pairwise statistic instead.",
        y=-0.62)
    for ext in ("png", "pdf"):
        fig.savefig(fs.resolve(FIGS / f"fig11_dispersion_budget.{ext}"), dpi=400,
                    bbox_inches="tight")
    plt.close(fig)

    print("=" * 96)
    print("DISPERSION BUDGET, as a multiple of each model's own effect")
    print("=" * 96)
    hdr = f"{'component':<20}" + "".join(f"{SHORT[m]:>15}" for m in models)
    print(hdr)
    for r in ROWS:
        line = f"{r:<20}"
        for m in models:
            c = data["models"][m]["components"].get(r)
            cell = "-" if not c else f"{c['over_beta']:.2f}x"
            line += f"{cell:>15}"
        print(line)
    print(f"\nwrote {(FIGS / 'fig11_dispersion_budget.png').relative_to(ROOT)}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
