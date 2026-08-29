"""Specification curve: every defensible way to run this study, sorted.

Twelve wordings x three resume templates is thirty-six analyses a competent
researcher could have run and published, each defensible on its own. A paper
reports one of them. Simonsohn, Simmons and Nelson (2020) named the display for
exactly this situation: estimate all of them, sort by point estimate, and show
the reader the whole distribution alongside which specification produced which
result.

The lower panel is the part that does the work. It marks, for each
specification, whether the perturbation was a paraphrase or a semantically null
edit. If the sorted curve separated cleanly into a paraphrase half and a null
half, the instability would be about meaning. It does not separate.

    .venv/Scripts/python.exe paper-a/src/figures_speccurve.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "paper-a" / "data" / "delta_stability" / "study2_v2.json"
FIGDIR = ROOT / "paper-a" / "figures"

fs.use_paper_style()

SHORT = {
    "mistral-7b-instruct-v0.1": "Mistral-7B-Instruct v0.1",
    "llama-2-7b-chat": "Llama-2-7B-chat",
    "mistral-7b-instruct-v0.3": "Mistral-7B-Instruct v0.3",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
}
ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]


def main() -> int:
    d = json.load(open(DATA, encoding="utf-8"))
    models = [m for m in ORDER if m in d]

    fig, axes = plt.subplots(2, len(models), figsize=(fs.FULL_W, 3.55),
                             sharex="col",
                             gridspec_kw=dict(height_ratios=[3.0, 1.0], hspace=0.10,
                                              wspace=0.22))
    if len(models) == 1:
        axes = axes.reshape(2, 1)

    for j, m in enumerate(models):
        specs = d[m]["spec_curve"]["specs"]
        top, bot = axes[0, j], axes[1, j]
        x = np.arange(len(specs))
        est = np.array([s["logodds"] for s in specs])
        lo = np.array([s["ci"][0] for s in specs])
        hi = np.array([s["ci"][1] for s in specs])
        sig = np.array([s["p_bh"] < 0.05 for s in specs])

        top.axhline(0, color=fs.RULE, lw=0.7, ls=(0, (3, 3)), zorder=1)
        top.vlines(x, lo, hi, color=fs.RULE, lw=0.7, zorder=2)
        for mask, col, ec in ((sig, fs.ACCENT2, fs.INK), (~sig, "white", fs.ACCENT)):
            if mask.any():
                top.scatter(x[mask], est[mask], s=9, zorder=3, linewidth=0.6,
                            facecolor=col, edgecolor=ec)
        top.set_title(SHORT.get(m, m), fontsize=8.6, pad=5)
        top.tick_params(labelbottom=False)
        if j == 0:
            top.set_ylabel("demographic effect\n(log-odds)", fontsize=8)

        # lower panel: which arm each specification came from
        isnull = np.array([s["kind"] == "null" for s in specs])
        bot.scatter(x[isnull], np.zeros(isnull.sum()), s=7, marker="s",
                    color=fs.ACCENT2, linewidth=0)
        bot.scatter(x[~isnull], np.ones((~isnull).sum()), s=7, marker="o",
                    color=fs.ACCENT, linewidth=0)
        bot.set_ylim(-0.7, 1.7)
        bot.set_yticks([0, 1])
        bot.set_yticklabels(["null edit", "paraphrase"] if j == 0 else ["", ""],
                            fontsize=7.4)
        bot.tick_params(axis="y", length=0)
        bot.set_xlabel("specification, sorted", fontsize=8)
        for s in ("top", "right", "left"):
            bot.spines[s].set_visible(False)

        sc = d[m]["spec_curve"]
        note = f"{sc['n_sig_positive']}+ / {sc['n_sig_negative']}- / {sc['n_null']} n.s."
        top.annotate(note, xy=(0.03, 0.09), xycoords="axes fraction",
                     fontsize=7.2, color=fs.INK)

    axes[0, -1].legend(handles=[
        Line2D([], [], ls="", marker="o", ms=3.6, mfc=fs.ACCENT2, mec=fs.INK,
               mew=0.6, label="significant after BH"),
        Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec=fs.ACCENT,
               mew=0.6, label="not significant"),
    ], loc="upper left", frameon=False, fontsize=6.9, handletextpad=0.3,
        borderpad=0.1, labelspacing=0.3)

    fs.caption(fig, "fig7_spec_curve",
               "Every defensible analysis of the same data, sorted by the answer it gives.",
               "Thirty-six specifications per model: twelve wordings that ask the same "
               "question, crossed with three résumé templates. Each is an analysis a "
               "researcher could have run and published. Bars are 95% bootstrap "
               "intervals; filled points survive Benjamini-Hochberg across that model's "
               "thirty-six specifications. "
               "On Mistral-7B-Instruct v0.3 twelve specifications are significantly "
               "positive and twelve are significantly negative, so the sign of the "
               "reported effect is determined by a choice the method section does not "
               "usually record. The lower strip marks whether each specification came "
               "from a paraphrase or from an edit that changes no meaning at all; the "
               "two do not separate.",
               y=-0.30)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = fs.resolve(FIGDIR / "fig7_spec_curve.png")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
