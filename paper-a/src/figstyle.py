"""Shared figure style, matched to how papers in this literature actually look.

The first version of these figures used matplotlib's DejaVu Sans with a large
bold headline inside the axes, colour-washed zone bands and marketing-voice
annotations. That reads as a slide deck. Papers at FAccT, AIES and in the
Scientific Data / QJE lineage this work cites look nothing like it.

What real figures in this literature do:

  - Set in the SAME serif as the body text. ACM's `acmart` class uses Linux
    Libertine; Libertinus is its maintained successor and is what is loaded here.
  - Carry no title inside the axes. The title lives in the caption below the
    figure, set in the body serif at a smaller size, beginning "Figure N."
  - Use restrained colour. Often black plus one accent. Shape carries the
    categorical distinction so the figure survives greyscale printing, and
    colour is redundant rather than load-bearing.
  - Are physically small and dense. A single-column figure is about 3.3 inches
    wide, a double-column one about 6.9.
  - Use hairline rules. Default matplotlib line weights are noticeably heavy for
    print.

`use_paper_style()` applies all of that. `caption()` renders a figure caption in
the convention the body text uses, so the standalone PNG and the in-paper
version look the same.
"""
from __future__ import annotations

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / "tools" / "fonts" / "Libertinus-7.051" / "static" / "OTF"

# Column widths follow ACM two-column proportions.
COL_W = 3.33
FULL_W = 6.90

# DRAW AT THE WIDTH THE DOCUMENT WILL SHOW IT AT. A figure drawn wider than
# its text block gets shrunk by \includegraphics, and the shrink applies to
# the type inside it: the conference fork's \linewidth is 397.485 pt (5.52 in), so a
# 6.90 in figure arrived at x0.82 and 5.4 pt tick labels rendered at 4.45 pt --
# below that style's own \tiny. Overriding FULL_W makes the scale 1.0 so
# nominal sizes are the sizes a reader sees. FONT_SCALE and HEIGHT_SCALE then
# buy back what a narrower canvas costs. All three default to the preprint's
# geometry, so nothing changes unless a builder opts in.
FONT_SCALE = 1.0
HEIGHT_SCALE = 1.0


def pt(size: float) -> float:
    """A point size in the figure's own units, after any scaling."""
    return size * FONT_SCALE

# Restrained palette. Ink carries everything; the two accents distinguish model
# generation and are backed up by marker shape so colour is never the sole cue.
INK = "#111111"
RULE = "#999999"
FAINT = "#d9d9d9"
ACCENT = "#1f4e79"    # deep blue, prints near-black in greyscale
ACCENT2 = "#8c3b12"   # burnt sienna
HILITE = "#e8e2d0"    # parchment, for reference bands

# ---------------------------------------------------------------------------
# CAPTION SINK AND OUTPUT REDIRECT -- how the same figures reach LaTeX.
#
# The captions here are COMPUTED -- they interpolate artifact numbers, so
# Figure 8's caption states the median published gap the artifact holds -- and
# then BAKED INTO THE IMAGE. That is right for the reportlab build, which has
# no caption machinery, and it means a figure pulled out of the PDF and
# attached to an email still explains itself.
#
# It is wrong for LaTeX. `acmart` wants \caption{} under the float: set in the
# document's own type, selectable, numbered by LaTeX, \label-able, and visible
# to the accessibility tooling ACM requires a \Description for. A baked caption
# is a picture of words. It would also be numbered twice -- once by matplotlib
# inside the image and once by LaTeX underneath it.
#
# Rather than maintain a second copy of every caption, the port sets
# CAPTION_SINK and caption() records the text instead of drawing it. One
# computation, two renderings; the LaTeX caption cannot drift from the
# preprint's, because there is only one place the sentence exists.
CAPTION_SINK: "dict | None" = None

# Where figures are written. None means the path the caller asked for, which is
# every existing caller. The port sets it so caption-free figures land beside
# main.tex instead of overwriting the preprint's, which still need their
# captions baked in.
OUTDIR: "pathlib.Path | None" = None


def resolve(out: pathlib.Path) -> pathlib.Path:
    """The path a figure should actually be written to.

    Identity unless OUTDIR is set, so nothing changes for the preprint build.
    """
    if OUTDIR is None:
        return out
    OUTDIR.mkdir(parents=True, exist_ok=True)
    return OUTDIR / out.name


def _register_fonts() -> str:
    """Register Libertinus if present; fall back to a Windows serif."""
    if FONT_DIR.exists():
        for f in FONT_DIR.glob("Libertinus*.otf"):
            try:
                fm.fontManager.addfont(str(f))
            except Exception:  # noqa: BLE001
                pass
        names = {f.name for f in fm.fontManager.ttflist}
        if "Libertinus Serif" in names:
            return "Libertinus Serif"
    for fallback in ("Times New Roman", "Cambria", "Georgia", "DejaVu Serif"):
        if fallback in {f.name for f in fm.fontManager.ttflist}:
            return fallback
    return "serif"


FONT = _register_fonts()


def use_paper_style() -> None:
    # A MISSING GLYPH IS A DEFECT IN THE PDF, SO STOP RATHER THAN WARN.
    # matplotlib substitutes a hollow box for any character the font lacks and
    # says so in a UserWarning. The figure builds, the warning scrolls past
    # with everything else, and the released preprint ends up with a tofu
    # square in an axis label -- which is what happened to the operating-point
    # figure's legend key, where "❘" was picked to match a marker and
    # Libertinus has no such glyph. Nothing caught it for eight builds.
    warnings.filterwarnings(
        "error", message=r"Glyph \d+ .* missing from font", category=UserWarning)
    plt.rcParams.update({
        "font.family": FONT,
        # FONT_SCALE reaches the rcParams sizes too, not only the explicit
        # fontsize= arguments. Axis tick labels come from here, and a log
        # axis renders its exponents at 0.7x the tick size -- so an 8 pt
        # tick label puts a 5.6 pt digit on the page, under the 6 pt floor,
        # without any call site naming a size below 6.4.
        "font.size": pt(8.5),
        "axes.labelsize": pt(8.5),
        "axes.titlesize": pt(9),
        "xtick.labelsize": pt(8),
        "ytick.labelsize": pt(8),
        "legend.fontsize": pt(8),
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "xtick.direction": "out",
        "lines.linewidth": 0.9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Type 42 (TrueType) rather than matplotlib's default Type 3. Type 3
        # glyphs are uncompressed drawing programs: they render, but they
        # search and copy badly in many viewers, and some publishers reject
        # them outright. Nothing else in the figure changes.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
    })


# Figure numbers as they appear IN THE PAPER, keyed by output stem.
#
# WHY A CENTRAL MAP. Each figure script used to hard-code its own number, and
# those numbers were the order the figures were BUILT in, not the order the
# paper presents them. After the paper was restructured the two diverged: the
# specification curve is the paper's Figure 1 and its baked caption still said
# "Figure 7". A caption is part of the image, so a reader pulling one figure out
# of the PDF carries the wrong number with it. One map here, and renumbering the
# paper is a one-line edit rather than a hunt through six scripts.
# Figure numbers, in the order the figures appear in the paper. Two were added
# after the v3 audit -- the dispersion budget, which the paper's first stated
# contribution promises and had no figure of, and the reporting scale, which
# replaces a prose explanation whose stated mechanism did not produce its own
# numbers -- and inserting them renumbered everything after. The paper reads
# these numbers from here rather than writing "Figure 3" in prose, so a future
# insertion cannot leave a reference pointing at the wrong panel.
PAPER_FIGURE_NUMBER = {
    "fig11_dispersion_budget": 1,   # 1.2, contributions
    "fig7_spec_curve": 2,           # 4.1
    "fig4_forest_by_wording": 3,    # 4.1
    "fig8_noise_floor": 4,          # 5.2
    "fig5_variance_components": 5,  # 6.1
    "fig12_reporting_scale": 6,     # 6.2
    "fig10_mech_panel": 7,          # 7
    "fig9_literature": 8,           # 8
}


def caption(fig, number, bold_lead: str, body: str, width_frac: float = 1.0,
            y: float = -0.02) -> None:
    """Draw a caption below the axes, in the convention a paper body uses.

    "Figure 1. <lead sentence>. <supporting detail>." The lead sentence states
    the finding, so the figure stands alone when it is pulled out of the paper
    and attached to an email.

    `number` may be an int or an output stem present in PAPER_FIGURE_NUMBER; the
    stem form is preferred, because it keeps the number where the paper's order
    is decided. An unknown stem raises rather than guessing: a figure with a
    silently wrong number is worse than a build that stops.
    """
    stem = number if isinstance(number, str) else None
    if isinstance(number, str):
        if number not in PAPER_FIGURE_NUMBER:
            raise KeyError(
                f"{number!r} has no entry in figstyle.PAPER_FIGURE_NUMBER; add "
                f"one so the figure's caption matches the paper's order")
        number = PAPER_FIGURE_NUMBER[number]

    # THE LATEX PORT TAKES THE TEXT AND DRAWS NOTHING. See CAPTION_SINK above.
    # Keyed by stem, because that is what the paper's figure ORDER is keyed by;
    # a caller passing a bare int has no stable identity to file the caption
    # under, and the three that do so are the superseded figures 1-3.
    if CAPTION_SINK is not None:
        if stem is None:
            raise ValueError(
                f"caption(number={number}) passed an int while the LaTeX port "
                f"is capturing captions. Pass the output stem instead, so the "
                f"caption can be filed against the figure it belongs to.")
        CAPTION_SINK[stem] = {"number": number, "lead": " ".join(bold_lead.split()),
                              "body": " ".join(body.split())}
        return
    # THE GAP IS MEASURED IN POINTS, NOT IN FIGURE FRACTIONS.
    #
    # `y` used to be passed straight to fig.transFigure, so a caller writing
    # y = -0.34 got 0.34 x the figure HEIGHT of white space -- 76 pt under a
    # 3.1-inch figure and 27 pt under a 1.1-inch one. The same argument meant
    # different things on different figures, and the tall ones opened a visible
    # hole between the axes and their caption. Callers still pass `y`; it is
    # now only a floor, and the actual offset is a constant number of points
    # converted through this figure's own height, so every caption sits the
    # same distance below its axes.
    gap_pt = 10.0
    h_in = fig.get_size_inches()[1]
    # WHERE THE CONTENT ACTUALLY ENDS, not where the canvas does. y = 0 in
    # figure coordinates is the bottom EDGE of the figure, and the lowest
    # artist -- an x-axis label, a legend, a multi-line unit note -- usually
    # sits well above it. Placing the caption a fixed distance below y = 0
    # therefore leaves a band of white whose size depends on how tall the axis
    # furniture happens to be, which is what opened the hole under the
    # operating-point figure. Measuring the drawn extent first makes the gap
    # the same 10 pt under every figure.
    try:
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        y0_frac = bb.y0 / max(h_in, 1e-6)
    except Exception:  # noqa: BLE001
        y0_frac = 0.0
    y_pts = y0_frac - (gap_pt / 72.0) / max(h_in, 1e-6)
    fig.text(0.0, y_pts, f"Figure {number}. {bold_lead} {body}",
             ha="left", va="top", fontsize=7.6, color=INK,
             wrap=True, transform=fig.transFigure,
             fontfamily=FONT)


def save(fig, out: pathlib.Path) -> pathlib.Path:
    out = resolve(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out
