"""Figure 10. Breaking the confound, and what is left afterwards.

SUPERSEDES fig6_mechanism.png, which showed the original eight-condition
dissection. That figure is not wrong about its own data and is misleading about
what the data mean: the two conditions it marks as destroying a delimiter are
also the only two that push the candidate's name one token later, so the panel
cannot distinguish the two explanations and the earlier caption implied it
could.

LEFT. The design that separates them, drawn as the 2x2 it is. The horizontal
axis is how far the name was pushed; the two series are whether a structural
delimiter was fragmented. If position drove the effect the two series would rise
together. If fragmentation drove it the series would be separated at every x.

RIGHT. The four decisive contrasts across every model and both inference modes.
Position alone is the control that the original design lacked. One contrast out
of thirty-two survives correction, and the figure is drawn so that a reader can
count them.

    .venv/Scripts/python.exe paper-a/src/figures_mechpanel.py
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
DATA = ROOT / "paper-a" / "data" / "mechanism_panel" / "mech_panel_analysis.json"
FIGDIR = ROOT / "paper-a" / "figures"

fs.use_paper_style()

SHORT = {
    "mistral-7b-instruct-v0.1": "Mistral-7B-Instr v0.1",
    "mistral-7b-v0.1-base": "Mistral-7B v0.1 base",
    "llama-2-7b-chat": "Llama-2-7B-chat",
    "llama-2-13b-chat": "Llama-2-13B-chat",
    "mistral-7b-instruct-v0.3": "Mistral-7B-Instr v0.3",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instr",
}
ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3",
         "mistral-7b-v0.1-base", "llama-2-13b-chat"]

# The 2x2. (condition, name shift in tokens, delimiter fragmented?)
CELLS = [("D0", 0, False), ("D8", 1, False), ("D9", 2, False),
         ("D5", 0, True), ("D4", 1, True)]

CONTRASTS = [("P1", "position alone\n(name +1, nothing fragmented)"),
             ("P2", "position dose\n(name +2)"),
             ("P3", "fragmentation,\nposition held equal"),
             ("P4", "fragmentation added\non top of position")]


def con(d, m, mode, prefix):
    for k, v in d[m][mode]["contrasts"].items():
        if k.startswith(prefix):
            return v
    return None


def main() -> int:
    d = json.load(open(DATA, encoding="utf-8"))
    pairs = [(m, mode) for m in ORDER if m in d for mode in ("chat", "raw")
             if mode in d[m]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.FULL_W, 3.1),
                                   gridspec_kw=dict(wspace=0.52,
                                                    width_ratios=[1.0, 1.30]))

    # ---------------- left: the 2x2, on ONE model-mode where P3 survives ---
    # The caption used to call this "the one model-mode where fragmentation
    # survives correction". It is not the only one: P3 survives on two, and if
    # P4 counts as a fragmentation contrast then three model-modes have one.
    # The panel has to pick a single cell to draw, which is fine; describing
    # that pick as uniqueness is not. Counted here so the caption states what
    # is true and follows the data if a re-run changes it.
    _p3_survivors = [(m, mode) for m in d if not m.startswith("_")
                     for mode in d[m]
                     if (con(d, m, mode, "P3") or {}).get("p_bh", 1) < 0.05]
    _frag_survivors = {(m, mode) for m in d if not m.startswith("_")
                       for mode in d[m] for pre in ("P3", "P4")
                       if (con(d, m, mode, pre) or {}).get("p_bh", 1) < 0.05}
    target = ("mistral-7b-instruct-v0.1", "raw")
    if target[0] in d and target[1] in d[target[0]]:
        per = d[target[0]][target[1]]["per_condition"]
        base = per["D0"]["logodds"]["est"]
        for frag, col, mk, lab in ((False, fs.ACCENT, "o", "delimiter intact"),
                                   (True, fs.ACCENT2, "D", "delimiter fragmented")):
            off = 0.045 if frag else -0.045
            xs, ys, es = [], [], []
            for c, shift, f in CELLS:
                if f != frag or c not in per:
                    continue
                e = per[c]["logodds"]
                xs.append(shift + off)
                ys.append(e["est"] - base)
                es.append((e["ci"][1] - e["ci"][0]) / 2)
            if xs:
                ax1.errorbar(xs, ys, yerr=es, marker=mk, ms=5, lw=1.1,
                             capsize=2.4, color=col, mec=fs.INK, mew=0.6,
                             label=lab, zorder=3)
        ax1.axhline(0, color=fs.RULE, lw=0.7, ls=(0, (3, 3)), zorder=1)
        ax1.set_xticks([0, 1, 2])
        ax1.set_xlabel("tokens the candidate's name was pushed", fontsize=8)
        ax1.set_ylabel("shift in demographic effect\n(log-odds, vs baseline)",
                       fontsize=8)
        ax1.set_title(f"{SHORT.get(target[0], target[0])}, {target[1]} mode",
                      fontsize=8.4, pad=5)
        ax1.legend(frameon=False, fontsize=7.1, loc="upper left",
                   handletextpad=0.4, borderpad=0.2)

    # ---------------- right: the four contrasts, everywhere ---------------
    labels, ys, xs, los, his, sig, raw = [], [], [], [], [], [], []
    # Which contrast class each drawn point belongs to, so the caption can
    # COUNT the position-only survivals instead of asserting they are zero.
    # The caption used to say position was "null everywhere after adjustment"
    # in the same sentence that reported seven survivals, and §4.4 depends on
    # the opposite reading. P1 and P2 move the name and fragment nothing.
    kind = []
    row = 0
    ticks, ticklabs = [], []
    for m, mode in pairs:
        for pi, (pref, _) in enumerate(CONTRASTS):
            c = con(d, m, mode, pref)
            if c is None:
                continue
            ys.append(row)
            xs.append(c["logodds"])
            los.append(c["ci"][0])
            his.append(c["ci"][1])
            sig.append(c["p_bh"] < 0.05)
            raw.append(c["p"] < 0.05)
            kind.append("position" if pref in ("P1", "P2") else "fragmentation")
            row += 1
        ticks.append(row - len(CONTRASTS) / 2 - 0.5)
        ticklabs.append(SHORT.get(m, m) + "\n" + mode + " mode")
        row += 1.2

    ys = np.array(ys); xs = np.array(xs)
    los = np.array(los); his = np.array(his)
    sig = np.array(sig); raw = np.array(raw)
    ax2.axvline(0, color=fs.RULE, lw=0.7, ls=(0, (3, 3)), zorder=1)
    ax2.hlines(ys, los, his, color=fs.INK, lw=0.7, zorder=2)
    ax2.scatter(xs[~raw], ys[~raw], s=13, facecolor="white", edgecolor=fs.ACCENT,
                linewidth=0.6, zorder=3)
    ax2.scatter(xs[raw & ~sig], ys[raw & ~sig], s=13, facecolor=fs.ACCENT,
                edgecolor=fs.INK, linewidth=0.6, zorder=3)
    ax2.scatter(xs[sig], ys[sig], s=30, facecolor=fs.ACCENT2, edgecolor=fs.INK,
                linewidth=0.8, marker="D", zorder=4)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(ticklabs, fontsize=6.8)
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlabel("contrast in demographic effect (log-odds)", fontsize=8)
    ax2.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=3.4, mfc="white", mec=fs.ACCENT,
               mew=0.6, label="not significant"),
        Line2D([], [], ls="", marker="o", ms=3.4, mfc=fs.ACCENT, mec=fs.INK,
               mew=0.6, label="raw p < 0.05"),
        Line2D([], [], ls="", marker="D", ms=4.2, mfc=fs.ACCENT2, mec=fs.INK,
               mew=0.7, label="survives BH"),
    ], frameon=False, fontsize=6.8, loc="lower right", handletextpad=0.35,
        borderpad=0.2, labelspacing=0.3)

    _first = next(iter(next(iter(d.values())).values()))
    fam = _first.get("bh_family_size", "?")
    # THE DESIGN, READ RATHER THAN REMEMBERED. This caption said each contrast
    # was "a paired difference over 48 name pairs and two résumés". The panel
    # is 24 name pairs crossed with THREE templates, giving 72 -- and §7's own
    # cell count, 9,504 = 11 x 6 x 2 x 72, says so. Both numbers were wrong,
    # and one of them is the resampling unit the paper spends §6.1 fixing, so
    # a reader checking the cluster count against the caption would have been
    # told the wrong thing about the most load-bearing choice in the analysis.
    _n_cells = _first.get("n_pairs_per_condition")
    _n_clusters = (_first.get("per_condition", {}).get("D0", {})
                   .get("n_clusters"))
    _n_tpl = (_n_cells // _n_clusters) if (_n_cells and _n_clusters) else None
    _NUM = {2: "two", 3: "three", 4: "four"}
    _design = (f"a paired difference over {_n_cells} cells — "
               f"{_n_clusters} name pairs crossed with "
               f"{_NUM.get(_n_tpl, _n_tpl)} résumé templates, resampled by "
               "pair"
               if _n_cells and _n_clusters and _n_tpl
               else "a paired difference over the full name-by-template grid")
    n_sig = int(sig.sum())
    kind = np.array(kind)
    n_pos = int((kind == "position").sum())
    n_pos_sig = int((sig & (kind == "position")).sum())
    n_frag_sig = int((sig & (kind == "fragmentation")).sum())
    fs.caption(fig, "fig10_mech_panel",
               "The apparent mechanism was confounded with token position, and little "
               "survives once that is controlled.",
               "Left: the design that separates the two accounts, on one of "
               f"the {len(_p3_survivors)} model-modes where fragmentation "
               "with position held equal survives correction "
               f"({len(_frag_survivors)} survive on either fragmentation "
               "contrast). The horizontal "
               "axis is how far the candidate's name was pushed by the edit; the "
               "series are whether a structural delimiter was fragmented. Under a "
               "pure position account the two series would coincide. Right: four "
               "contrasts per model and inference mode, in each case "
               + _design + ". Position alone is the "
               "control the original eight-condition design lacked; it is not null. "
               f"Of {len(xs)} contrasts drawn here, within a Benjamini-Hochberg "
               f"family of {fam}, {n_sig} "
               + ("survives" if n_sig == 1 else "survive") + " — "
               f"{n_pos_sig} of the {n_pos} position-only contrasts and "
               f"{n_frag_sig} of the {len(xs) - n_pos} fragmentation contrasts. "
               "The two classes do not separate, which is the finding: moving the "
               "name does what fragmenting a delimiter does.",
               y=-0.34)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = fs.resolve(FIGDIR / "fig10_mech_panel.png")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}  ({len(xs)} contrasts, {n_sig} survive BH)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
