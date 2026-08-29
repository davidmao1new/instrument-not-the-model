"""Why a percentage-point conversion is not a common scale — and which
version of that claim the data actually supports.

THE CLAIM §6.2 MADE, AND THE DEFECT IN IT. Converting a log-odds effect to
percentage points multiplies it by the logistic slope at the model's operating
point. Audits convert at p = 0.5, where the slope takes its maximum of 0.25. The
paper reported the resulting overstatement as "1.8x, 7.8x, 66.9x, 108.5x", a
ratio of the fixed-Jacobian figure to the measured mean per-cell probability
difference. Building this figure showed that the stated MECHANISM does not
produce two of those four numbers:

    model            0% cells      0.25 / mean p(1-p)      reported ratio
    Mistral v0.1     saturated              2.0x                  66.9x

Mistral v0.1 sits near the steep part of the curve, so the operating-point
story predicts a factor of two and the paper printed sixty-seven. The reason is
that the reported ratio has the model's effect in its denominator, and that
model's effect is not distinguishable from zero: it is a quotient of two nearly
zero quantities, unstable by construction, and it is not evidence about a scale.

WHAT THIS FIGURE PLOTS INSTEAD. The quantity that answers the question without
the effect size in it:

    conversion-factor error  =  0.25  /  mean_cells p(1-p)

This is how wrong the assumed Jacobian is at the place the model actually sits.
It is defined whether or not the model has an effect, it is a property of the
operating point alone, and it is what makes percentage points incomparable
across a panel. Across these four checkpoints it runs from 2.0x to 484x.

The realised overstatement of a MEASURED effect is plotted too, but only for the
models whose effect is distinguishable from zero, where the quotient is stable —
and there it agrees with the Jacobian prediction closely (2.2 predicted against
1.8 realised, 8.1 against 7.8), which is the check that the mechanism is right
where it is identified.

    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/figures_scale.py
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
OUT_JSON = D / "delta_stability" / "reporting_scale.json"

ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
SHORT = {"llama-2-7b-chat": "Llama-2-7B", "llama-3.1-8b-instruct": "Llama-3.1-8B",
         "mistral-7b-instruct-v0.1": "Mistral v0.1",
         "mistral-7b-instruct-v0.3": "Mistral v0.3"}
MAXSLOPE = 0.25


def main() -> int:
    s2 = json.loads((D / "delta_stability" / "study2_v2.json")
                    .read_text(encoding="utf-8"))
    models = [m for m in ORDER if m in s2]

    pts = []
    for m in models:
        o = s2[m]["overall"]
        slope = o["local_slope"]
        lo = o["logodds"]
        distinguishable = lo["ci"][0] * lo["ci"][1] > 0
        # Invert mean p(1-p) to an operating point, on the side of 0.5 the
        # model's own preference sits. Printed so the inversion is auditable.
        disc = max(0.0, 1 - 4 * slope)
        p = ((1 + np.sqrt(disc)) / 2 if o["superiority"]["est"] >= 0.5
             else (1 - np.sqrt(disc)) / 2)
        pts.append(dict(
            model=m, slope=slope, operating_p=float(p),
            saturated=o["saturated_frac"],
            jacobian_error=MAXSLOPE / slope,
            effective_slope=o["prob_pp"]["est"] / (lo["est"] * 100),
            realised_ratio=abs(o["legacy_pp"] / o["prob_pp"]["est"]),
            effect_distinguishable=bool(distinguishable)))

    art = {
        "_definition": {
            "jacobian_error": "0.25 / mean_cells p(1-p): how wrong the assumed "
                              "conversion factor is at the model's operating "
                              "point. Independent of the effect size.",
            "realised_ratio": "fixed-Jacobian pp figure / measured mean per-cell "
                              "probability difference. Has the effect in its "
                              "denominator, so it is only interpretable where "
                              "the effect is distinguishable from zero.",
        },
        "models": {p["model"]: p for p in pts},
    }
    ok = [p for p in pts if p["effect_distinguishable"]]
    art["summary"] = dict(
        jacobian_error_min=min(p["jacobian_error"] for p in pts),
        jacobian_error_max=max(p["jacobian_error"] for p in pts),
        n_models=len(pts),
        n_distinguishable=len(ok),
        realised_min=min((p["realised_ratio"] for p in ok), default=None),
        realised_max=max((p["realised_ratio"] for p in ok), default=None),
        predicted_for_distinguishable=[p["jacobian_error"] for p in ok],
    )
    OUT_JSON.write_text(json.dumps(art, indent=2), encoding="utf-8")

    fs.use_paper_style()
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(fs.FULL_W, 2.45),
        gridspec_kw={"width_ratios": [1.5, 1.0]})

    # ---- left: the curve, the assumed tangent, and where models sit ------
    x = np.linspace(-9, 9, 900)
    ax.plot(x, 1 / (1 + np.exp(-x)), color=fs.INK, lw=1.0, zorder=2)
    xs = np.linspace(-3.4, 3.4, 10)
    ax.plot(xs, 0.5 + MAXSLOPE * xs, color=fs.RULE, lw=0.9, ls=(0, (4, 2)),
            zorder=1)
    ax.annotate("the slope every audit assumes\n(tangent at p = 0.5, slope 0.25)",
                xy=(0, 0.5), xytext=(-8.6, 0.72), fontsize=5.6, color=fs.RULE)

    mk = ["o", "s", "^", "D"]
    for i, q in enumerate(pts):
        p = min(max(q["operating_p"], 1e-5), 1 - 1e-5)
        xp = float(np.log(p / (1 - p)))
        s = q["slope"]
        xt = np.linspace(xp - 2.6, xp + 2.6, 10)
        ax.plot(xt, p + s * (xt - xp), color=fs.ACCENT, lw=0.8, zorder=3)
        ax.plot([xp], [p], marker=mk[i % 4], ms=4.2, color=fs.ACCENT,
                mec=fs.INK, mew=0.5, zorder=4)
        lab = f"{SHORT[q['model']]}"
        dy = 0.085 if p < 0.55 else -0.10
        ax.annotate(lab, xy=(xp, p), xytext=(xp + 0.4, p + dy),
                    fontsize=5.8, color=fs.INK)
    ax.set_xlim(-9, 9)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("decision margin (log-odds)", fontsize=6.4)
    ax.set_ylabel("P(yes)", fontsize=6.4)
    ax.tick_params(labelsize=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- right: the conversion-factor error ------------------------------
    labels = [SHORT[q["model"]] for q in pts]
    vals = [q["jacobian_error"] for q in pts]
    ypos = np.arange(len(vals))
    ax2.barh(ypos, vals, color=fs.ACCENT, height=0.56)
    for i, q in enumerate(pts):
        ax2.annotate(f"{q['jacobian_error']:.0f}\u00d7"
                     if q["jacobian_error"] >= 10
                     else f"{q['jacobian_error']:.1f}\u00d7",
                     (q["jacobian_error"], i), xytext=(3, 0),
                     textcoords="offset points", va="center", fontsize=6)
        if q["effect_distinguishable"]:
            ax2.plot([q["realised_ratio"]], [i], marker="|", ms=7,
                     mew=1.1, color=fs.ACCENT2, zorder=5)
    ax2.axvline(1.0, color=fs.INK, lw=0.7, ls=(0, (3, 2)))
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(labels, fontsize=6)
    ax2.set_xscale("log")
    ax2.set_xlim(0.8, max(vals) * 3.0)
    # THE LEGEND KEY HAS TO BE A GLYPH THE FIGURE'S FONT ACTUALLY HAS.
    # This read "\u2758 = realised overstatement" -- LIGHT VERTICAL BAR, chosen
    # to match the marker="|" tick drawn above. Libertinus Serif has no such
    # glyph, so matplotlib substituted its missing-character box and the
    # released preprint carries a tofu square where the key should be. It warns
    # on every build ("Glyph 10072 missing from font(s) Libertinus Serif") and
    # the warning was there long enough to become scenery.
    #
    # U+007C is the same mark, is in every font, and is what the marker is.
    ax2.set_xlabel("conversion-factor error, 0.25 / mean p(1\u2212p)\n"
                   "(log axis; | = realised overstatement, where the\n"
                   "effect is distinguishable from zero)", fontsize=5.8)
    ax2.tick_params(axis="x", labelsize=6)
    ax2.invert_yaxis()
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)

    fig.tight_layout()
    fs.caption(
        fig, "fig12_reporting_scale",
        "The four models do not share an operating point, so they do not "
        "share a percentage-point scale.",
        "Left: the logistic curve, the tangent an audit assumes when it "
        "converts at p = 0.5 (grey, slope 0.25), and the tangent that is "
        "actually there at each model's mean operating point (blue). The gap "
        "between the two slopes is the conversion error. Right: that error, "
        f"0.25 divided by the model's own mean p(1-p), on a log axis. It "
        "ranges over more than two orders of magnitude across four "
        "checkpoints of the same size measured on the same design. The "
        "quantity has no effect size in it, so it is defined whether or not "
        "the model has an effect to convert; the sienna tick marks the "
        "overstatement realised on a measured effect, shown only for the two "
        "models whose effect is distinguishable from zero, where that "
        "quotient is stable.",
        y=-0.52)
    for ext in ("png", "pdf"):
        fig.savefig(fs.resolve(FIGS / f"fig12_reporting_scale.{ext}"), dpi=400,
                    bbox_inches="tight")
    plt.close(fig)

    print(f"{'model':<15}{'slope':>9}{'oper. p':>10}{'satur.':>9}"
          f"{'0.25/slope':>12}{'eff.slope':>11}{'realised':>10}{'b != 0':>8}")
    for q in pts:
        print(f"{SHORT[q['model']]:<15}{q['slope']:>9.5f}{q['operating_p']:>10.4f}"
              f"{q['saturated']:>8.1%}{q['jacobian_error']:>12.1f}"
              f"{q['effective_slope']:>11.5f}{q['realised_ratio']:>10.1f}"
              f"{str(q['effect_distinguishable']):>8}")
    s = art["summary"]
    print(f"\n  conversion-factor error across the panel: "
          f"{s['jacobian_error_min']:.1f}x to {s['jacobian_error_max']:.0f}x")
    print(f"  realised overstatement, {s['n_distinguishable']} models with a "
          f"distinguishable effect: {s['realised_min']:.1f}x to "
          f"{s['realised_max']:.1f}x")
    print(f"  predicted for those same models: "
          + ", ".join(f"{v:.1f}x" for v in s["predicted_for_distinguishable"]))
    print(f"\nwrote {(FIGS / 'fig12_reporting_scale.png').relative_to(ROOT)}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
