"""Figures for Study 2: does the measured demographic effect move with wording?

Two figures, both regenerated from artifacts:

  Figure 4  A forest plot of the demographic effect estimated separately under
            each of twelve wordings, per model, with semantic paraphrases and
            semantically null perturbations distinguished by marker. The point
            of the figure is the vertical scatter within a model: every marker
            in a row is the same question asked of the same candidates.

  Figure 5  The hierarchical decomposition. For each model, the demographic
            effect beta against the between-wording standard deviation
            sigma_variant, fitted separately for the semantic and null arms.
            If the null bar is as tall as the semantic bar, surface form moves
            the estimate as much as meaning does.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "paper-a" / "data" / "delta_stability"
FIGDIR = ROOT / "paper-a" / "figures"
# See effectsize.PP_PER_LOGIT_MAX: this is the logistic's maximum slope, so
# every pp value on these axes is an UPPER BOUND on the movement, not an
# estimate of it. Axis labels say so.
PP_PER_LOGIT_MAX = 25.0

fs.use_paper_style()

SHORT = {
    "llama-2-7b-chat": "Llama-2-7B-chat",
    "mistral-7b-instruct-v0.1": "Mistral-7B-Inst v0.1",
    "mistral-7b-instruct-v0.3": "Mistral-7B-Inst v0.3",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Inst",
}
ORDER = ["llama-2-7b-chat", "mistral-7b-instruct-v0.1",
         "mistral-7b-instruct-v0.3", "llama-3.1-8b-instruct"]


def load(p):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


# --------------------------------------------------------------------------
def fig_forest(s2) -> pathlib.Path:
    """One row per model, one marker per wording, on the primary effect size.

    Reads study2_v2.json, whose per-wording intervals resample NAME PAIRS. An
    earlier version of this figure read the quarantined margin_analysis.json,
    whose intervals resample rows, and converted to percentage points at the
    p = 0.5 slope. Both are errors this paper devotes a section to, so both are
    gone; the scale here is the probability of superiority, which is bounded,
    scale-free and needs no Jacobian.
    """
    models = [m for m in ORDER if m in s2]
    fig, ax = plt.subplots(figsize=(fs.FULL_W, 0.62 * len(models) + 1.25))

    ax.axvline(0.5, color=fs.INK, lw=0.7, zorder=1)

    for i, m in enumerate(models):
        d = s2[m]
        pooled = d["overall"]["superiority"]["est"]
        pci = d["overall"]["superiority"]["ci"]
        ax.plot(pci, [i, i], color=fs.INK, lw=2.2, solid_capstyle="butt",
                alpha=0.9, zorder=2)
        ax.scatter([pooled], [i], s=64, marker="D", facecolor=fs.INK,
                   edgecolor=fs.INK, zorder=5)

        for v, s in d["per_variant"].items():
            null = v.startswith("N")
            dy = (0.20 if null else -0.20)
            x = s["superiority"]["est"]
            ci = s["superiority"]["ci"]
            ax.plot(ci, [i + dy, i + dy], color=fs.RULE, lw=0.55, zorder=3)
            ax.scatter([x], [i + dy], s=20, zorder=4, linewidth=0.6,
                       marker=("s" if null else "o"),
                       facecolor=("white" if null else fs.ACCENT),
                       edgecolor=(fs.ACCENT2 if null else fs.INK))

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([SHORT[m] for m in models])
    ax.set_ylim(len(models) - 0.45, -0.55)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability of superiority (0.5 = no effect; above 0.5 "
                  "favours White-associated names)")
    ax.tick_params(axis="y", length=0)

    ax.legend(handles=[
        Line2D([], [], ls="", marker="D", ms=5, mfc=fs.INK, mec=fs.INK,
               label="pooled over all wordings"),
        Line2D([], [], ls="", marker="o", ms=4.2, mfc=fs.ACCENT, mec=fs.INK,
               mew=0.6, label="semantic paraphrase"),
        Line2D([], [], ls="", marker="s", ms=4.2, mfc="white", mec=fs.ACCENT2,
               mew=0.6, label="semantically null"),
    ], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
        handletextpad=0.4, columnspacing=1.4)

    fs.caption(fig, "fig4_forest_by_wording",
               "The same question, asked twelve ways, of the same candidates.",
               "Each small marker is the demographic effect estimated under one "
               "wording; circles are paraphrases, open squares are perturbations "
               "that change no meaning at all. Diamonds and heavy bars are the "
               "estimate pooled over all twelve, with a 95 % interval. Every "
               "interval here resamples NAME PAIRS rather than rows: an earlier "
               "version of this figure resampled rows and reported percentage "
               "points converted at p = 0.5, which are the two errors \u00a76.1 "
               "and \u00a76.2 are about. The scale is the probability of "
               "superiority, which needs no conversion.",
               y=-0.06)
    return fs.save(fig, FIGDIR / "fig4_forest_by_wording.png")


def fig_components(vc) -> pathlib.Path:
    models = [m for m in ORDER if m in vc]
    fig, ax = plt.subplots(figsize=(fs.FULL_W, 0.52 * len(models) + 1.3))

    h = 0.17
    for i, m in enumerate(models):
        for arm, off, face, edge in (("semantic", -h, fs.ACCENT, fs.INK),
                                     ("null", h, "white", fs.ACCENT2)):
            r = vc[m].get(arm)
            if not r:
                continue
            lo, mid, hi = r["sigma_variant_pp_max"]
            ax.plot([lo, hi], [i + off, i + off], color=fs.INK, lw=0.9, zorder=2)
            for e in (lo, hi):
                ax.plot([e, e], [i + off - 0.06, i + off + 0.06],
                        color=fs.INK, lw=0.7, zorder=2)
            ax.scatter([mid], [i + off], s=34, zorder=3, linewidth=0.7,
                       marker=("o" if arm == "semantic" else "s"),
                       facecolor=face, edgecolor=edge)
        allr = vc[m].get("all")
        if allr:
            b = abs(allr["beta_pp_max"][1])
            ax.plot([b, b], [i - 0.32, i + 0.32], color=fs.ACCENT2, lw=1.1,
                    ls=(0, (2, 2)), zorder=4)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([SHORT[m] for m in models])
    ax.set_ylim(len(models) - 0.45, -0.55)
    ax.set_xlim(left=0)
    # THE AXIS HAS TO NAME ITS OWN CONVERSION. The plotted quantity is
    # `sigma_variant_pp_max`: a log-odds variance component multiplied by the
    # logistic's MAXIMUM slope, 0.25, so it must be reported as an upper bound
    # and named as one. The conversion error runs 2.0x to 484x across this
    # panel, so magnitudes are not comparable between rows either.
    #
    # AND THE LABEL CARRIES NO CROSS-REFERENCE. It used to end "see §6.2",
    # which is right in the preprint and meaningless anywhere else. The same
    # image ships in the ICLR fork, which has no subsections at all, so the
    # pointer resolved to nothing in a submitted paper. A caption can be
    # rewritten per venue; text baked into the image cannot, so a figure has
    # to be self-contained. audit_figure_refs.py now enforces that.
    ax.set_xlabel("Between-wording standard deviation of the demographic "
                  "effect\n(percentage points, upper bound at the "
                  "p = 0.5 logistic slope)")
    ax.tick_params(axis="y", length=0)

    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=4.2, mfc=fs.ACCENT, mec=fs.INK,
               mew=0.6, label="semantic arm"),
        Line2D([], [], ls="", marker="s", ms=4.2, mfc="white", mec=fs.ACCENT2,
               mew=0.6, label="null arm"),
        Line2D([], [], ls=(0, (2, 2)), color=fs.ACCENT2, lw=1.1,
               label="|effect| for that model"),
    ], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
        handletextpad=0.5, columnspacing=1.3)

    # THE HEADLINE RANGE IS INTERPOLATED, NOT TYPED. It read "a fifth to a
    # half", which was a literal in this file, was wrong at both ends, and
    # broke the paper's own rule that every number is read from an artifact.
    # The ratio's own credible interval is printed with it, because on three of
    # the four models that interval spans an order of magnitude and a reader
    # who takes the medians as a ranking would be reading noise.
    _rt = [(m, arm, vc[m][arm]["ratio_sigma_v_to_beta"])
           for m in models for arm in ("semantic", "null")
           if vc[m].get(arm) and vc[m][arm].get("ratio_sigma_v_to_beta")]
    _meds = [r[2][1] for r in _rt]
    _widest = max(_rt, key=lambda r: r[2][2] - r[2][0]) if _rt else None
    fs.caption(fig, "fig5_variance_components",
               (f"Wording moves the demographic effect by {min(_meds):.2f} to "
                f"{max(_meds):.2f} of the effect itself in point estimate, and "
                "surface form does as much work as meaning."
                if _meds else
                "Wording moves the demographic effect by a large fraction of "
                "the effect itself, and surface form does as much work as "
                "meaning."),
               "Posterior median and 95% credible interval for the between-wording "
               "standard deviation from a crossed random-effects model, fitted "
               "separately to the six semantic paraphrases and the six semantically "
               "null perturbations. Sampling noise is partialled out by partial "
               "pooling, so this is dispersion rather than dispersion-plus-noise. The "
               "dashed rule marks the magnitude of that model's own demographic "
               "effect. Percentage points are computed at the logistic's maximum "
               "slope and are therefore upper bounds (§6.2). "
               + (f"The dispersion-to-effect ratio itself is poorly pinned: on "
                  f"{SHORT.get(_widest[0], _widest[0])}'s {_widest[1]} arm its "
                  f"95% interval runs [{_widest[2][0]:.2f}, {_widest[2][2]:.2f}]. "
                  if _widest else "")
               + "This figure orders nothing; it shows that the component is "
                 "large and that the two arms overlap.",
               y=-0.06)
    return fs.save(fig, FIGDIR / "fig5_variance_components.png")


def main() -> int:
    s2 = load(DATA / "study2_v2.json")
    vc = load(DATA / "variance_components.json")
    if not s2 or not vc:
        sys.exit("run analyze_study2_v2.py and fit_variance_components.py first")
    print(f"font: {fs.FONT}")
    print(f"  wrote {fig_forest(s2).relative_to(ROOT)}")
    print(f"  wrote {fig_components(vc).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
