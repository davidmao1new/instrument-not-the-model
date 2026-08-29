"""Generate Paper A figures programmatically from analysis output.

PROTOCOL.md phase 4: "Generate figures programmatically from the analysis output
so they regenerate on rerun. No screenshots." Every figure reads
`panel_gate_results.json` and regenerates from it, so a rerun of the experiment
regenerates the paper's figures with no manual step.

Style is set in `figstyle.py` and matches how figures in this literature are
actually typeset: body serif, no title inside the axes, captions below in the
"Figure N." convention, restrained colour with shape as the redundant channel,
and hairline rules.

Every proportion is drawn with its Wilson 95% interval. A point estimate alone
was how an n=4 result came to look precise; see CHANGELOG, session 7.
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
GATE = ROOT / "paper-a" / "data" / "panel_gate" / "panel_gate_results.json"
FIGDIR = ROOT / "paper-a" / "figures"

fs.use_paper_style()

SHORT = {
    "mistral-7b-instruct-v0.1": "Mistral-7B-Instruct v0.1",
    "mistral-7b-v0.1-base": "Mistral-7B v0.1 (base)",
    "llama-2-7b-chat": "Llama-2-7B-chat",
    "llama-2-13b-chat": "Llama-2-13B-chat",
    "mistral-7b-instruct-v0.3": "Mistral-7B-Instruct v0.3",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
}


def load_gate() -> list[dict]:
    if not GATE.exists():
        sys.exit(f"missing {GATE}; run paper-a/src/panel_gate.py first")
    rows = json.load(open(GATE, encoding="utf-8"))
    return [r for r in rows if "load_error" not in r]


def _order(rows):
    """Pre-2024 first, then post, each alphabetical. Plotted top-down."""
    return sorted(rows, key=lambda r: (r["generation"] != "pre_2024", r["model"]))


def _mark(r):
    """Filled circle for pre-2024, open square for post. Shape, not colour, is
    the primary channel, so the figure survives greyscale."""
    if r["generation"] == "pre_2024":
        return dict(marker="o", facecolor=fs.ACCENT, edgecolor=fs.INK)
    return dict(marker="s", facecolor="white", edgecolor=fs.ACCENT2)


# --------------------------------------------------------------------------
# Figure 1 — position bias
# --------------------------------------------------------------------------
def fig_position(rows) -> pathlib.Path:
    rows = _order(rows)
    fig, ax = plt.subplots(figsize=(fs.FULL_W, 0.265 * len(rows) + 0.85))

    ax.axvline(0.5, color=fs.RULE, lw=0.6, ls=(0, (3, 3)), zorder=1)

    for i, r in enumerate(rows):
        p = r["position"]["position_a_rate"]
        ci = r["position"].get("position_a_ci")
        n = r["position"].get("position_a_n")
        if ci:
            ax.plot(ci, [i, i], color=fs.INK, lw=1.0, solid_capstyle="butt", zorder=2)
            for e in ci:
                ax.plot([e, e], [i - 0.11, i + 0.11], color=fs.INK, lw=0.8, zorder=2)
        m = _mark(r)
        ax.scatter([p], [i], s=26, zorder=3, linewidth=0.7,
                   marker=m["marker"], facecolor=m["facecolor"], edgecolor=m["edgecolor"])
        if n:
            ax.annotate(f"{p:.2f}  (n = {n})", (1.015, i),
                        xycoords=("axes fraction", "data"), fontsize=7.4,
                        color=fs.INK, va="center", ha="left", annotation_clip=False)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SHORT.get(r["model"], r["model"]) for r in rows])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlabel("Proportion of forced choices selecting the first-listed candidate")
    ax.tick_params(axis="y", length=0)

    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=4.5, mfc=fs.ACCENT, mec=fs.INK, mew=0.7,
               label="pre-2024"),
        Line2D([], [], ls="", marker="s", ms=4.5, mfc="white", mec=fs.ACCENT2, mew=0.7,
               label="post-2024"),
    ], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
        handletextpad=0.4, columnspacing=1.6)

    fs.caption(fig, 1,
               "Forced-choice screening tracks prompt position rather than the candidate.",
               "Two résumés identical except the candidate's name, presented in both "
               "orders; an unbiased screener sits at 0.50. Bars are Wilson 95% intervals. "
               "The direction reverses within a model family across one generation: "
               "Mistral-7B-Instruct moves from 0.92 to 0.00 between v0.1 and v0.3, and the "
               "Llama family moves from 0.00 to 0.81.",
               y=-0.05)
    return fs.save(fig, FIGDIR / "fig1_position_bias.png")


# --------------------------------------------------------------------------
# Figure 2 — prompt-wording instability
# --------------------------------------------------------------------------
def fig_template(rows) -> pathlib.Path:
    rows = [r for r in _order(rows) if (r.get("template") or {}).get("per_template")]
    fig, ax = plt.subplots(figsize=(fs.FULL_W, 0.30 * len(rows) + 0.95))

    # Reference band: the largest demographic effect in this literature, drawn
    # at the same scale so the comparison is visual rather than asserted.
    ax.axvspan(0.5 - 0.0301, 0.5 + 0.0301, color=fs.HILITE, zorder=0, lw=0)

    for i, r in enumerate(rows):
        t = r["template"]
        pts = [(j, v) for j, v in enumerate(t["per_template"]) if v is not None]
        if not pts:
            continue
        vs = [v for _, v in pts]
        ax.plot([min(vs), max(vs)], [i, i], color=fs.RULE, lw=0.8,
                solid_capstyle="butt", zorder=1)

        cis = t.get("per_template_ci") or []
        m = _mark(r)
        for j, v in pts:
            ci = cis[j] if j < len(cis) and cis[j] else None
            if ci:
                ax.plot(ci, [i, i], color=fs.INK, lw=0.55, alpha=0.5, zorder=2)
            ax.scatter([v], [i], s=22, zorder=3, linewidth=0.7,
                       marker=m["marker"], facecolor=m["facecolor"],
                       edgecolor=m["edgecolor"])

        sp = t.get("spread"); sci = t.get("spread_ci")
        lbl = f"{sp*100:.0f}" + (f"  [{sci[0]*100:.0f}, {sci[1]*100:.0f}]" if sci else "")
        ax.annotate(lbl, (1.015, i), xycoords=("axes fraction", "data"),
                    fontsize=7.4, color=fs.INK, va="center", ha="left",
                    annotation_clip=False)

    ax.annotate("spread (pp)", (1.015, 1.0), xycoords="axes fraction",
                fontsize=7.4, color=fs.INK, va="bottom", ha="left",
                annotation_clip=False)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SHORT.get(r["model"], r["model"]) for r in rows])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlabel("Acceptance rate under each of five paraphrases of one instruction")
    ax.tick_params(axis="y", length=0)

    fs.caption(fig, 2,
               "Rewording one instruction moves the screening decision by far more "
               "than any demographic effect reported in this literature.",
               "Each marker is one of five paraphrases, at temperature 0 with identical "
               "candidates; thin bars are Wilson 95% intervals and the grey rule spans the "
               "range. The shaded band is 3.01 pp wide, the largest demographic gap in Gao, "
               "Jiang and Yan (2026). Among models passing the quality gate the spread runs "
               "from 50 to 97 pp.",
               y=-0.045)
    return fs.save(fig, FIGDIR / "fig2_template_spread.png")


# --------------------------------------------------------------------------
# Figure 3 — instrument-validity gates
# --------------------------------------------------------------------------
def fig_gates(rows) -> pathlib.Path:
    rows = _order(rows)
    cols = ["distinguishes strong\nfrom unqualified",
            "stable under\nrewording",
            "reads candidate,\nnot position"]
    fig, ax = plt.subplots(figsize=(fs.COL_W + 1.35, 0.27 * len(rows) + 1.15))

    for i, r in enumerate(rows):
        t = r.get("template") or {}
        sp, sci = t.get("spread"), t.get("spread_ci")
        pci = r["position"].get("position_a_ci")
        checks = [
            bool(r["admit"]),
            sp is not None and (sci or [sp, sp])[1] <= 0.25,
            bool(pci) and pci[0] <= 0.75 and pci[1] >= 0.25,
        ]
        for j, ok in enumerate(checks):
            if ok:
                ax.scatter([j], [i], s=42, marker="o", facecolor=fs.INK,
                           edgecolor=fs.INK, linewidth=0.7, zorder=3)
            else:
                ax.scatter([j], [i], s=42, marker="o", facecolor="white",
                           edgecolor=fs.RULE, linewidth=0.7, zorder=3)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SHORT.get(r["model"], r["model"]) for r in rows])
    ax.set_xlim(-0.55, len(cols) - 0.45)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", color=fs.FAINT, lw=0.5)
    ax.set_axisbelow(True)

    fs.caption(fig, 3,
               "No model in the panel passes all three instrument-validity gates.",
               "Filled marks indicate a pass. A model that cannot distinguish a "
               "statistician from a retail associate cannot meaningfully be described as "
               "biased or unbiased with respect to names, so these checks precede any "
               "reported disparity.",
               y=-0.06)
    return fs.save(fig, FIGDIR / "fig3_admission_gates.png")


def main() -> int:
    rows = load_gate()
    print(f"{len(rows)} models · font: {fs.FONT}")
    for fn in (fig_position, fig_template, fig_gates):
        print(f"  wrote {fn(rows).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
