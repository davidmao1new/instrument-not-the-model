r"""Everything the eight figure captions need except the sentences.

WHY THIS EXISTS. The captions arrive from `build_facct_figures.py` already
typeset and looking finished, which is exactly why they get forgotten: nothing
about them looks unwritten. They are preprint prose, so they carry the same
obligation the body does and have to be rewritten. Rewriting them by hand
against the rendered text means retyping every number they state, and the
paper's central claim about itself is that no measurement in it is typed.

So this script closes the gap the other way. It slices each number OUT of the
generated caption and emits it as a macro, so a hand-written caption can state
the number without typing it.

    \renewcommand{\FigCapNoiseFloor}{... agreement runs \CapNoiseConcurrent{}
      under concurrent requests ...}

THE VALUE IS NOT RE-DERIVED. `figstyle.caption()` computes the caption from the
released artifacts and the figure is drawn from that same call, so the string in
`captions.tex` IS the artifact's rendering. Lifting a number out of it keeps one
computation behind both the figure and the macro. A second derivation would not:
`build_facct_tex.py` carries a comment about the first attempt at that, which
got the headline ratio wrong (54 % against the paper's 25 %) by reconstructing
the wrong cell set.

MATCHED ON CONTEXT, NEVER ON VALUE. Matching a caption number against the
existing macros by value looked attractive and is a trap. Measured across the
eight captions it fires twelve times and is mostly coincidence: the mechanism
panel's "24 name pairs" matches \NMdeObs, which is 24 observations behind a
minimum detectable effect, and its "2 model-modes" matches \QuantN, which is 2
quantizations. Every number would be right and every reason wrong, and no
consistency check in this repository could see it. So each macro below names a
short pattern of the words AROUND its number, and the build fails if that
pattern stops matching exactly once.

WHAT THIS WILL NOT DO. It does not write a caption or a \Description. FAccT
prohibits the use of LLMs to generate text for publications, and the
submission's generative-AI statement asserts the author wrote the prose.
CAPTIONS.md states what each caption has to land and what it may not drop; the
sentences are the author's.

    sh paper-a/src/_py.sh paper-a/src/build_caption_kit.py
"""
from __future__ import annotations

import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACCT = ROOT / "paper-a" / "facct"
GEN = FACCT / "generated"
CAPTIONS = GEN / "captions.tex"

# Which figures cost body pages. A body caption is charged against the
# fourteen-page limit and an appendix caption is free, so they get different
# budgets. Source: facct/FIGURES.md.
BODY = {"DispersionBudget", "NoiseFloor"}
WORD_BUDGET = {True: 60, False: 110}      # keyed by "is in the body"
DESC_BUDGET = (40, 70)

# --------------------------------------------------------------------------
# The numbers each caption states, named.
#
# (macro, figure, pattern, what it is). The pattern must contain exactly one
# capture group and must match exactly once in that caption. It is deliberately
# a few words of context rather than the bare numeral: see the module docstring.
# --------------------------------------------------------------------------
NAMED = [
    ("CapNoiseConcurrent", "NoiseFloor",
     r"agreement runs ([\d.]+)\\% under concurrent",
     "exact-agreement rate between byte-identical wordings, concurrent"),
    ("CapNoiseSequential", "NoiseFloor",
     r"([\d.]+)\\% once they are forced sequential",
     "the same rate once requests are forced sequential"),
    ("CapNoiseNoCache", "NoiseFloor",
     r"reaches ([\d.]+)\\% only when key-value cache",
     "the same rate again with key-value cache reuse disabled"),
    ("CapNoiseReplicateCells", "NoiseFloor",
     r"Re-measuring ([\d,]+) of these cells",
     "cells re-measured in a separate process on a different day"),
    ("CapNoiseReplicateExact", "NoiseFloor",
     r"reproduced (\d+) of them exactly",
     "how many of those reproduced exactly"),

    ("CapVarRatioLo", "VarianceComponents",
     r"effect by ([\d.]+) to [\d.]+ of the effect",
     "low end of between-wording SD as a fraction of the effect"),
    ("CapVarRatioHi", "VarianceComponents",
     r"effect by [\d.]+ to ([\d.]+) of the effect",
     "high end of the same fraction"),
    ("CapVarIntervalLo", "VarianceComponents",
     r"interval runs \[([\d.]+), [\d.]+\]",
     "low end of the credible interval on that ratio, Mistral v0.1 semantic"),
    ("CapVarIntervalHi", "VarianceComponents",
     r"interval runs \[[\d.]+, ([\d.]+)\]",
     "high end of the same interval"),

    ("CapMechModes", "MechPanel",
     r"one of the (\d+) model-modes",
     "model-modes where fragmentation survives correction with position held"),
    ("CapMechFragEither", "MechPanel",
     r"\((\d+) survive on either fragmentation",
     "model-modes surviving on either fragmentation contrast"),
    ("CapMechCells", "MechPanel",
     r"paired difference over (\d+) cells",
     "cells behind each contrast"),
    ("CapMechNamePairs", "MechPanel",
     r"(\d+) name pairs crossed with",
     "name pairs in that cell count"),
    ("CapMechContrasts", "MechPanel",
     r"Of (\d+) contrasts drawn here",
     "contrasts drawn in the right-hand panel"),
    ("CapMechFamily", "MechPanel",
     r"Benjamini-Hochberg family of (\d+)",
     "size of the correction family"),
    ("CapMechSurviving", "MechPanel",
     r"family of \d+, (\d+) survive",
     "contrasts surviving correction"),
    ("CapMechPositionN", "MechPanel",
     r"of the (\d+) position-only contrasts",
     "position-only contrasts drawn"),
    ("CapMechPositionSurv", "MechPanel",
     r"survive---(\d+) of the \d+ position-only",
     "position-only contrasts surviving"),
    ("CapMechFragN", "MechPanel",
     r"of the (\d+) fragmentation contrasts",
     "fragmentation contrasts drawn"),
    ("CapMechFragSurv", "MechPanel",
     r"and (\d+) of the \d+ fragmentation contrasts",
     "fragmentation contrasts surviving"),

    ("CapLitMedianGap", "Literature",
     r"median published gap is ([\d.]+) points",
     "median absolute published demographic gap, percentage points"),
    ("CapLitRatioLo", "Literature",
     r"runs from (\d+) to \d+ per cent",
     "dispersion-to-effect ratio, low end, models separable from zero"),
    ("CapLitRatioHi", "Literature",
     r"runs from \d+ to (\d+) per cent",
     "the same ratio, high end"),
    ("CapLitRatioMax", "Literature",
     r"reaches (\d+) per cent against a denominator",
     "the same ratio on the two models whose effect is not separable"),
]

# Numbers that are properties of the METHOD rather than measurements of it. A
# caption may type these, because there is no artifact behind them to disagree
# with -- 0.25 is the logistic's maximum slope, not something this study found.
# check_draft.py reads this list so the two cannot drift apart.
CONSTANTS = {
    "1.0": "the parity line, where a choice moves the estimate as much as the "
           "effect",
    "0.5": "no effect on the probability-of-superiority scale; also the "
           "operating point an audit assumes when it converts",
    "0.25": "the logistic's maximum slope, at p = 0.5",
    "95": "the interval level",
}
# NOT a constant: 100. The noise-floor caption's "reaches 100 % only when
# key-value cache reuse is disabled as well" is a measured agreement rate that
# happens to land on the ceiling, and it is \CapNoiseNoCache. Listing it here
# as well would let the one serving configuration that reproduces exactly be
# typed into the paper, which is the case the reader is most likely to check.

# UNITS ARE THE AUTHOR'S. These macros hold a bare numeral, unlike numbers.tex
# where a percentage macro bakes in "\%". The same quantity is written "47.9 %"
# in one caption and "25 to 33 per cent" in another, and a baked sign forces
# the first. So write \CapNoiseConcurrent{}\% or \CapLitRatioLo{} per cent as
# the sentence needs.


def caption_bodies() -> dict[str, str]:
    """The eight caption strings, without their WHAT-IS-DRAWN comment blocks."""
    if not CAPTIONS.exists():
        sys.exit(f"{CAPTIONS.relative_to(ROOT)} not found. Run "
                 "build_facct_figures.py first.")
    text = CAPTIONS.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"\\newcommand\{\\FigCap(\w+)\}\{%\n", text):
        tail = text[m.end():]
        stop = re.search(r"^\s*%", tail, re.M)
        out[m.group(1)] = (tail[:stop.start()] if stop else tail).strip()
    return out


def drawn(name: str) -> list[str]:
    """The WHAT IS DRAWN inventory the figure port extracted for one figure."""
    text = CAPTIONS.read_text(encoding="utf-8")
    m = re.search(r"\\newcommand\{\\FigCap" + name + r"\}\{%\n.*?\n"
                  r"%\s+WHAT IS DRAWN[^\n]*\n(.*?)\\providecommand",
                  text, re.S)
    if not m:
        return []
    return [ln.strip(" %-") for ln in m.group(1).split("\n") if ln.strip(" %-")]


def extract(bodies: dict[str, str]) -> tuple[dict, list]:
    """Pull each named number out of its caption. Loud on any mismatch."""
    values, problems = {}, []
    for macro, fig, pattern, what in NAMED:
        body = bodies.get(fig)
        if body is None:
            problems.append(f"{macro}: no caption named {fig}")
            continue
        hits = re.findall(pattern, body)
        if len(hits) != 1:
            problems.append(
                f"{macro}: pattern matched {len(hits)} times in {fig}, "
                f"expected 1  ({pattern})")
            continue
        values[macro] = (hits[0], fig, what)
    return values, problems


def write_macros(values: dict) -> None:
    lines = [
        "% GENERATED by paper-a/src/build_caption_kit.py -- do not edit.",
        "%",
        "% One macro per number the eight figure captions state. The value is",
        "% sliced out of generated/captions.tex, which figstyle.caption()",
        "% computed from the released artifacts in the same call that drew the",
        "% figure -- so a macro here cannot disagree with the figure above it.",
        "%",
        "% Use these when you rewrite a caption, so the rewrite does not put a",
        "% typed measurement into a paper whose claim is that it has none.",
        "%",
        "% Each holds a BARE NUMERAL. numbers.tex bakes \\% into a percentage",
        "% macro; these do not, because the same quantity is written \"47.9 %\"",
        "% in one caption and \"25 to 33 per cent\" in another. Supply the unit:",
        "%     \\CapNoiseConcurrent{}\\%        \\CapLitRatioLo{} per cent",
        "",
    ]
    for macro in sorted(values):
        val, fig, what = values[macro]
        lines.append(f"% {fig}: {what}")
        lines.append(f"\\newcommand{{\\{macro}}}{{{val}}}")
    (GEN / "capnumbers.tex").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")


def write_briefs(bodies: dict, values: dict) -> None:
    """CAPTIONS.md -- what each caption has to land. Not a draft."""
    out = [
        "# The eight captions, and the eight `\\Description`s",
        "",
        "Generated by `paper-a/src/build_caption_kit.py`. Regenerate rather "
        "than edit.",
        "",
        "**These are briefs and not drafts.** FAccT prohibits the use of LLMs "
        "to generate text for publications, and the submission's "
        "generative-AI statement asserts you wrote the prose. The numbers, "
        "the inventory of what is drawn, and the scope limits below all come "
        "from the artifacts. The sentences do not exist yet and are yours.",
        "",
        "## How to use this file",
        "",
        "`main.tex` carries a prepared slot for each figure, after "
        "`\\input{generated/captions}`. Fill the slot; do not edit "
        "`generated/`.",
        "",
        "```latex",
        "\\renewcommand{\\FigCapNoiseFloor}{%",
        "  \\textbf{Your one-line claim.} Your sentences, with "
        "\\CapNoiseConcurrent{} and friends where a number belongs.}",
        "\\renewcommand{\\FigDescNoiseFloor}{%",
        "  What someone who cannot see the figure needs.}",
        "```",
        "",
        "Then `sh paper-a/src/_py.sh paper-a/src/check_draft.py`, which counts "
        "words against the budget, flags a typed measurement, flags a macro "
        "that does not exist, and flags a `\\Description` that is only the "
        "caption again.",
        "",
        "**The caption macros hold a bare numeral and you supply the unit** "
        "-- `\\CapNoiseConcurrent{}\\%` or `\\CapLitRatioLo{} per cent`. This "
        "differs from `numbers.tex`, where a percentage macro such as "
        "`\\RatioLo` already carries its sign, and it differs on purpose: the "
        "same quantity is written both ways in different captions.",
        "",
        f"**{len(CONSTANTS)} numbers may be typed**, because they are "
        "properties of the method rather than measurements of it and there is "
        "no artifact behind them to disagree with: "
        + ", ".join(f"`{k}` ({v})" for k, v in sorted(CONSTANTS.items()))
        + ". Everything else is a macro.",
        "",
        "## The rule for a caption, and the different rule for a "
        "`\\Description`",
        "",
        "A **caption** says what the reader should conclude. It is an "
        "argument with a number in it.",
        "",
        "A **`\\Description`** says what is on the page, for a reader who "
        "cannot see it. ACM requires one on every float. It is not the "
        "caption repeated, and it is not a second copy of the conclusion: "
        "it is panel structure, axes and their units, what the series are, "
        "and what the reference lines mark. The inventory under each figure "
        "below is extracted from the figure itself. Use it; do not paste it.",
        "",
    ]

    by_fig: dict[str, list] = {}
    for macro, (val, fig, what) in values.items():
        by_fig.setdefault(fig, []).append((macro, val, what))

    for name in bodies:
        body = bodies[name]
        in_body = name in BODY
        words = len(re.sub(r"\\[A-Za-z]+\*?", " ", body).split())
        out += [
            "---",
            "",
            f"## `\\FigCap{name}`  ({'body' if in_body else 'appendix'})",
            "",
            f"**Budget: {WORD_BUDGET[in_body]} words.** The preprint's is "
            f"{words}. "
            + ("A body caption is charged against the fourteen pages."
               if in_body else
               "Appendix captions are free, so this one only has to be "
               "readable."),
            "",
        ]
        macros = sorted(by_fig.get(name, []))
        if macros:
            out += ["**The numbers, as macros.** Every one of these is a "
                    "measurement; typing it instead of using the macro is "
                    "the one thing that breaks the paper's claim about "
                    "itself.", "",
                    "| macro | value | what it is |",
                    "|---|---:|---|"]
            for macro, val, what in macros:
                out.append(f"| `\\{macro}{{}}` | {val} | {what} |")
            out.append("")
        else:
            out += ["**No measurement macros needed.** This caption states no "
                    "measurement that is not already a macro in "
                    "`numbers.tex`, or spells its counts as words. Keep it "
                    "that way.", ""]

        inv = drawn(name)
        if inv:
            out += [f"**What is drawn** ({len(inv)} labels extracted from the "
                    "figure). This is the raw material for the "
                    f"`\\Description`, which wants "
                    f"{DESC_BUDGET[0]}--{DESC_BUDGET[1]} words:", ""]
            out += [f"- `{x}`" for x in inv]
            out.append("")

    (FACCT / "CAPTIONS.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    bodies = caption_bodies()
    values, problems = extract(bodies)

    print("=" * 78)
    print("CAPTION KIT  --  macros and briefs for the eight figures")
    print("=" * 78)
    print(f"\n  {len(bodies)} captions read from generated/captions.tex")

    if problems:
        print(f"\n  {len(problems)} PATTERN(S) NO LONGER MATCH:\n")
        for p in problems:
            print(f"    {p}")
        print("\n  A caption's wording changed under a pattern in NAMED. That "
              "is not\n  a failure to route around: re-read the caption and "
              "fix the pattern,\n  because the number it names may have "
              "changed too.\n")
        return 1

    write_macros(values)
    write_briefs(bodies, values)
    print(f"  {len(values)} caption numbers named and emitted")
    print(f"  {len(CONSTANTS)} method constants declared typeable")
    print(f"\n  wrote  {(GEN / 'capnumbers.tex').relative_to(ROOT)}")
    print(f"  wrote  {(FACCT / 'CAPTIONS.md').relative_to(ROOT)}")
    print("\n  Next: fill the prepared slots in main.tex, then check_draft.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
