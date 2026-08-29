"""Figure 8. The measurement's own noise, and how much of the result it explains.

Left panel. Two prompts in the Study 2 panel are byte-identical by construction:
S1 and N1 have the same system string and the same user message. They are a
replicate experiment nobody designed, and at temperature 0 with greedy decoding
they ought to agree exactly. The panel shows how often they do. Llama-3.1-8B
agrees on 5 of 36 cells.

Right panel. What that noise implies for the paper's claim. The between-wording
standard deviation is compared against the noise expected on a per-wording mean
of the same size. If the bars were comparable the wording result would be
arithmetic; they are not, by a factor of between five and seven.

This figure exists because the paper previously asserted the measurement was
deterministic. It is not, and saying so with a number is worth more than the
assertion was.

    .venv/Scripts/python.exe paper-a/src/figures_noise.py
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
DATA = ROOT / "paper-a" / "data" / "delta_stability" / "noise_floor.json"
FIGDIR = ROOT / "paper-a" / "figures"

fs.use_paper_style()

SHORT = {
    "mistral-7b-instruct-v0.1": "Mistral-7B-Instr v0.1",
    "llama-2-7b-chat": "Llama-2-7B-chat",
    "mistral-7b-instruct-v0.3": "Mistral-7B-Instr v0.3",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instr",
}
ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]


def main() -> int:
    d = json.load(open(DATA, encoding="utf-8"))
    models = [m for m in ORDER if m in d and "sigma_noise_per_cell" in d[m]]
    y = np.arange(len(models))
    lab = [SHORT.get(m, m) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.35),
                                   gridspec_kw=dict(wspace=0.55))

    # ---- left: how often a byte-identical repeat agrees exactly ----------
    frac = np.array([d[m]["frac_identical"] for m in models])
    ax1.barh(y, frac, height=0.55, color=fs.ACCENT, edgecolor=fs.INK, lw=0.6)
    for i, m in enumerate(models):
        ax1.text(frac[i] + 0.02, i,
                 f"{d[m]['n_identical']}/{d[m]['n_replicate_cells']}",
                 va="center", fontsize=7.4, color=fs.INK)
    ax1.set_yticks(y)
    ax1.set_yticklabels(lab, fontsize=7.8)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("fraction of byte-identical repeats\nthat agree exactly", fontsize=8)
    ax1.invert_yaxis()
    ax1.tick_params(axis="y", length=0)

    # ---- right: wording SD against the noise floor -----------------------
    sv = np.array([d[m]["sigma_variant_raw"] for m in models])
    nf = np.array([d[m]["sigma_noise_on_variant_mean"] for m in models])
    h = 0.34
    ax2.barh(y - h / 2, sv, height=h, color=fs.ACCENT2, edgecolor=fs.INK, lw=0.6,
             label="between-wording SD")
    ax2.barh(y + h / 2, nf, height=h, color="white", edgecolor=fs.INK, lw=0.6,
             hatch="////", label="noise floor")
    for i, m in enumerate(models):
        ax2.text(sv[i] + 0.002, i - h / 2, f"{d[m]['ratio_variant_to_noise']:.1f}×",
                 va="center", fontsize=7.4, color=fs.INK)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("log-odds", fontsize=8)
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", length=0)
    ax2.legend(frameon=False, fontsize=7.2, loc="upper center",
               bbox_to_anchor=(0.5, 1.30), ncol=2, handlelength=1.3,
               handletextpad=0.4, columnspacing=1.2, borderpad=0.1)
    ax2.set_xlim(0, max(sv.max(), nf.max()) * 1.30)

    # Report the cross-session check from the model with the MOST shared cells.
    # Taking whichever came first picked a five-cell overlap and printed it as
    # though it were the whole comparison.
    # THE CAUSE IS NOT ONE THING, AND THIS CAPTION USED TO SAY IT WAS.
    # "The cause is batched inference" states a single mechanism that §5.2
    # explicitly refutes: forcing requests sequential raises agreement a long
    # way but not to unity, and the residue closes only when key-value cache
    # reuse is disabled as well. A caption asserting the simpler story
    # contradicts the section it sits in, on the paper's own headline
    # reproducibility result.
    _cause = " The cause is the serving stack rather than the model"
    try:
        _rv = json.loads((ROOT / "paper-a" / "data" / "replicate"
                          / "replicate_analysis.json")
                         .read_text(encoding="utf-8"))["_verdict"]
        _cv = json.loads((ROOT / "paper-a" / "data" / "replicate"
                          / "cache_residual.json")
                         .read_text(encoding="utf-8"))["_verdict"]
        _cause = (
            f" Batching is most of the cause and not all of it: agreement runs "
            f"{_rv['frac_identical_conc4']:.1%} under concurrent requests and "
            f"{_rv['frac_identical_conc1']:.1%} once they are forced "
            f"sequential, and reaches "
            f"{_cv['frac_identical_cache_off']:.0%} only when key-value cache "
            "reuse is disabled as well (§5.2)")
    except Exception:  # noqa: BLE001
        pass

    cross = [m for m in d if isinstance(d[m], dict) and "cross_session" in d[m]]
    extra = ""
    if cross:
        best = max(cross, key=lambda m: d[m]["cross_session"]["n"])
        c = d[best]["cross_session"]
        extra = (f" Re-measuring {c['n']} of these cells on {SHORT.get(best, best)} "
                 f"in a separate process on a different day reproduced "
                 f"{c['n_identical']} of them exactly.")

    fs.caption(fig, "fig8_noise_floor",
               "The measurement is not deterministic, and the wording result survives that.",
               "Left: two of the twelve wordings are byte-identical by construction, so "
               "at temperature 0 they should agree exactly on every cell. They do not."
               + _cause + "." + extra +
               " Right: the between-wording standard deviation against the noise "
               "expected on a per-wording mean of the same size, estimated from that "
               "same replicate. Wording moves the measured effect by five to seven "
               "times what the arithmetic does.",
               y=-0.44)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = fs.resolve(FIGDIR / "fig8_noise_floor.png")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
