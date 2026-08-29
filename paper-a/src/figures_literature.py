"""Figure 9. Published effect sizes, and how many prompts produced each one.

The paper's central practical claim is that effects of the size this literature
reports are not large compared to what wording alone moves. That claim was
previously asserted from memory. This figure computes it.

LEFT. Every percentage-point demographic gap in Gao, Jiang and Yan's panel --
the largest set of LLM callback-rate gaps measured on ONE protocol by one team,
which is what makes them comparable to each other, though not to gaps from other
papers measured differently -- plus the Bertrand and Mullainathan field-experiment
anchor every LLM audit measures itself against. The median published gap is 0.64
of a point. Eleven of fourteen fall below 1.1 points.

That was "half a point, ten of twelve" until 2026-08-01, when two rows of Table I
that had been recorded as untranscribable turned out to transcribe cleanly. The
plotted values come from the artifact, so the panel and the annotation both
followed automatically; only this docstring had to be corrected by hand. See
`_transcription_completeness` in published_effects.json.

RIGHT. The number of distinct prompt wordings under which each study's effect
was separately estimated. This is the figure's argument, and it is a count
rather than an estimate, so there is nothing to dispute: it is zero everywhere.
Two studies use one prompt. Two average over many and publish only the mean,
which discards the dispersion rather than reporting it. None publishes a range.

A note on scales, which the figure states rather than hides. The left panel is
in percentage points of callback rate. Our own effect is a log-odds decision
margin and converting it to percentage points requires a base rate the models do
not share, so OUR NUMBER IS NOT PLOTTED HERE. What transfers between the two is
the ratio of wording-induced dispersion to effect size, and that is annotated
instead.

    .venv/Scripts/python.exe paper-a/src/figures_literature.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIT = ROOT / "paper-a" / "data" / "reference" / "published_effects.json"
OURS = ROOT / "paper-a" / "data" / "delta_stability" / "study2_v2.json"
FIGDIR = ROOT / "paper-a" / "figures"

fs.use_paper_style()

# study -> (label, prompts USED, prompts the effect was SEPARATELY estimated
# under, annotation).
#
# The two counts differ for exactly the studies that use many prompts, and the
# difference is the point: averaging over wordings before the bias test is run
# collapses the design to a single effective wording, so the dispersion this
# paper measures is not recoverable from those studies either. Plotting only the
# first column would flatter them.
# Display labels only. The COUNTS come from the artifact -- they were
# hard-coded here until the v3 audit, and one of them was wrong by a factor of
# 164: An et al. were plotted at 820 wordings, which is their count of
# INSTANTIATED prompts (4 qualification levels x 5 base templates x 41 roles),
# not of distinct wordings. A transcribed number in a figure is exactly what
# this paper argues against, so it is derived now.
PROMPT_LABEL = {
    "gao_jiang_yan_2026": "Gao, Jiang & Yan 2026",
    "iso_etal_2025": "Iso et al. 2025",
    "fu_shi_2025": "Fu & Shi 2025",
    "tan_etal_2026": "Tan et al. 2026",
    "wilson_caliskan_2024": "Wilson & Caliskan 2024",
    "an_etal_2024": "An et al. 2024",
}
NUMWORD = {1: "one", 2: "two", 3: "three", 5: "five", 10: "ten"}


def build_prompts(lit):
    """(label, wordings used, wordings separately estimated, annotation).

    `neff` is None where the study reports no demographic effect at all, so
    there is nothing for a wording to have been estimated under; that bar is
    omitted rather than drawn as a one.
    """
    hp = lit.get("how_many_prompts_the_field_uses") or {}
    out = []
    for key, label in PROMPT_LABEL.items():
        v = hp.get(key)
        if not isinstance(v, dict):
            continue
        n = v["n_prompts"]
        averaged = bool(v.get("averaged_before_testing"))
        if "n_prompts_effective" in v:
            neff = v["n_prompts_effective"]
        else:
            neff = 1 if averaged else n
        word = NUMWORD.get(n, str(n))
        if neff is None:
            note = f"{word} used; reports no demographic effect"
        elif averaged:
            note = f"{word} used, averaged before testing"
        elif n == 1:
            note = "one fixed prompt"
        else:
            note = f"{word}, differing in what they ask"
        out.append((label, n, neff, note))
    return out


def main() -> int:
    lit = json.load(open(LIT, encoding="utf-8"))
    ours = json.load(open(OURS, encoding="utf-8"))
    eff = lit["gao_jiang_yan_2026"]["effects"]
    bm = lit["bertrand_mullainathan_2004"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.FULL_W, 3.05),
                                   gridspec_kw=dict(wspace=0.62, width_ratios=[1.25, 1.0]))

    # ---------------- left: published pp gaps -----------------------------
    rows = sorted(eff, key=lambda e: e["delta_pp"])
    y = np.arange(len(rows))
    val = np.array([e["delta_pp"] for e in rows])
    lo = np.array([e["ci"][0] for e in rows])
    hi = np.array([e["ci"][1] for e in rows])
    sig = np.array([e["sig"] != "n.s." for e in rows])

    ax1.axvline(0, color=fs.RULE, lw=0.7, ls=(0, (3, 3)), zorder=1)
    ax1.hlines(y, lo, hi, color=fs.INK, lw=0.9, zorder=2)
    for mask, fc, ec in ((sig, fs.ACCENT2, fs.INK), (~sig, "white", fs.ACCENT)):
        if mask.any():
            ax1.scatter(val[mask], y[mask], s=22, zorder=3, linewidth=0.7,
                        facecolor=fc, edgecolor=ec)
    ax1.set_yticks(y)
    ax1.set_yticklabels([e["model"] for e in rows], fontsize=7.2)
    ax1.invert_yaxis()
    ax1.tick_params(axis="y", length=0)
    ax1.set_xlabel("published gap in callback rate (pp)\nWhite-associated minus Black-associated",
                   fontsize=8)

    # the field-experiment anchor, on the same axis
    ax1.axvline(bm["delta_pp"], color=fs.ACCENT, lw=0.9, ls=(0, (1, 2)), zorder=1)
    # Axes-fraction y, so the label cannot land outside an inverted data axis.
    # Placing it at a data coordinate past the last row clipped it away entirely.
    ax1.annotate(f"Bertrand & Mullainathan\n{bm['delta_pp']:+.2f} pp, field experiment",
                 xy=(bm["delta_pp"] - 0.15, 0.98),
                 xycoords=("data", "axes fraction"), fontsize=6.6,
                 color=fs.ACCENT, ha="right", va="top", linespacing=1.35)

    med = lit["summary_of_published_pp_gaps"]["median_abs_pp"]
    ax1.annotate(f"median |gap| {med:.2f} pp\n"
                 f"{lit['summary_of_published_pp_gaps']['n_below_1_1_pp']} of "
                 f"{lit['summary_of_published_pp_gaps']['n']} below 1.1 pp",
                 xy=(0.02, 0.20), xycoords="axes fraction", fontsize=6.9,
                 color=fs.INK)

    # ---------------- right: how many prompts -----------------------------
    PROMPTS = build_prompts(lit)
    labs = [p[0] for p in PROMPTS]
    n = np.array([p[1] for p in PROMPTS], dtype=float)
    neff = np.array([np.nan if p[2] is None else p[2]
                     for p in PROMPTS], dtype=float)
    y2 = np.arange(len(labs))
    ax2.barh(y2, n, height=0.55, color=fs.ACCENT, edgecolor=fs.INK, lw=0.6,
             label="wordings used")
    ax2.barh(y2, neff, height=0.55, color="none", edgecolor=fs.INK, lw=0.9,
             hatch="///", label="effect separately estimated")
    ax2.legend(fontsize=6.4, loc="upper right",
               bbox_to_anchor=(1.0, -0.20), frameon=False,
               ncol=2, handletextpad=0.4, columnspacing=1.2)
    ax2.set_xscale("log")
    ax2.set_xlim(0.7, 40)
    for i, p in enumerate(PROMPTS):
        ax2.text(p[1] * 1.35, i, p[3], va="center", fontsize=6.8, color=fs.INK)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(labs, fontsize=7.2)
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlabel("distinct prompt wordings used\n(log scale)", fontsize=8)
    ax2.annotate("dispersion across wordings reported: none",
                 xy=(0.5, 1.05), xycoords="axes fraction", fontsize=7.4,
                 color=fs.ACCENT2, ha="center", va="bottom")

    # A RATIO NEEDS A DENOMINATOR THAT EXISTS. Two of the four models have a
    # log-odds effect whose interval covers zero; on those the ratio is a
    # spread divided by something not separable from nothing, and §4.1 says so
    # about the very number this caption used to print.
    def _identified(r):
        ci = r["overall"]["logodds"]["ci"]
        return ci[0] * ci[1] > 0

    def _ratio(r):
        return r["ps_sd_across_wordings"] / abs(r["overall"]["superiority"]["est"] - 0.5)

    ratios = sorted(_ratio(r) for r in ours.values() if _identified(r))
    ratios_un = sorted(_ratio(r) for r in ours.values() if not _identified(r))

    _hp = {k: v for k, v in (lit.get("how_many_prompts_the_field_uses") or {}).items()
           if isinstance(v, dict)}
    # A study with no demographic effect has no effective wording count, and §8
    # excludes it from every wording-count statement. It stays in the panel
    # because its reporting practice is informative in Table 15. The three
    # categories must partition the studies the counts APPLY to, not the panel.
    _excl = {k for k, v in _hp.items()
             if "n_prompts_effective" in v and v["n_prompts_effective"] is None}
    _cnt = {k: v for k, v in _hp.items() if k not in _excl}
    _n_one = sum(1 for v in _cnt.values()
                 if v.get("n_prompts") == 1 and not v.get("averaged_before_testing"))
    _n_two = sum(1 for v in _cnt.values()
                 if v.get("n_prompts") == 2 and not v.get("averaged_before_testing"))
    _n_avg = sum(1 for v in _cnt.values() if v.get("averaged_before_testing"))
    assert _n_one + _n_two + _n_avg == len(_cnt), (
        "the caption's three categories must partition the studies the count "
        f"applies to; they do not: {_n_one}+{_n_two}+{_n_avg} != {len(_cnt)}")

    fs.caption(fig, "fig9_literature",
               "The field's effects are small, and none of them is published with a range.",
               "Left: every demographic gap in the largest published model panel "
               "(Gao, Jiang and Yan, Table I), with 95% intervals; filled points are "
               "significant. The dotted line is the Bertrand and Mullainathan field "
               "experiment, the anchor this literature compares itself to. The median "
               f"published gap is {med:.2f} points. Right: how many distinct wordings each "
               "study estimated its effect under. "
               f"{NUMWORD.get(_n_one, _n_one)} use one prompt, "
               f"{NUMWORD.get(_n_two, _n_two)} uses two, and "
               f"{NUMWORD.get(_n_avg, _n_avg)} average over "
               "many and publish only the mean, which discards the "
               "dispersion rather than measuring it. "
               + (f"{NUMWORD.get(len(_excl), len(_excl))} further "
                  + ("study reports" if len(_excl) == 1 else "studies report")
                  + " no demographic effect, so no wording count applies to "
                  + ("it" if len(_excl) == 1 else "them")
                  + " and " + ("its" if len(_excl) == 1 else "their")
                  + " hatched overlay is omitted. " if _excl else "")
               + "Our own effect is on a log-odds "
               "scale and is deliberately not plotted on the left axis, because "
               "converting it to points requires a base rate these models do not share. "
               "The quantity that does transfer is the ratio of wording-induced "
               "dispersion to effect size, which on the "
               f"{NUMWORD.get(len(ratios), len(ratios))} of our models whose "
               "effect is distinguishable from zero runs from "
               f"{ratios[0]*100:.0f} to {ratios[-1]*100:.0f} per cent"
               + (f", and on the other {NUMWORD.get(len(ratios_un), len(ratios_un))} "
                  f"reaches {ratios_un[-1]*100:.0f} per cent against a "
                  "denominator that is not separable from zero" if ratios_un
                  else "")
               + ".",
               y=-0.36)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = fs.resolve(FIGDIR / "fig9_literature.png")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
