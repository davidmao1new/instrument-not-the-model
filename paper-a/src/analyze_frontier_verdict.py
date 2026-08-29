"""What a thresholded verdict can see, measured on a frontier model.

WHY THIS ARM EXISTS AND WHY IT WAS NEARLY LOST. §4.7 reaches four OpenAI
checkpoints because they return a next-token distribution. One vendor that does
not was run anyway, on the same wording design, recording the thresholded
verdict per candidate: 220 matched pairs on gemini-3.1-flash-lite. The rows sat
in the corpus unreported, and the record-classifier keyed on the margin, so it
scored each of those pairs as a single prompt costing one call -- the corpus was
short 220 pairs and 220 calls, and the attribution of every single-prompt record
to the admission gate, the affiliation probe and the smoke tests was wrong by
exactly that. Fixing the count made the arm visible, and it turns out to say
something.

WHAT IT SAYS. The verdict is a deterministic function of the résumé template.
Every T1_strong cell advances; no T2_mid or T3_marginal cell does. That holds
across both arms of every pair, all twelve wordings, both genders, and the four
cells that were run twice. So on this instrument:

  * the name moves nothing -- 0 of 220 pairs disagree between arms;
  * the wording moves nothing once the template is fixed;
  * and the four repeated cells return the same verdict, so it is not noise
    swamping a signal.

THIS IS NOT A NULL RESULT ABOUT THE MODEL. It is a measurement of the OUTCOME.
The model plainly reads the name: its free-text rationales use the candidate's
first name and gendered pronouns. What has no resolution is the yes/no
threshold, which quantises everything the model might be doing into three
constant values. §4.2 argues that binarising a graded outcome discards the
comparison; here that argument is not an argument but an observation, on a
frontier model, where the discarding is total.

The honest limit, stated because it cuts the other way: an arm that finds
nothing cannot distinguish "no effect" from "no resolution". What licenses the
second reading is that the SAME design on models that do return a distribution
produces effects that are not zero, and that the constancy here is perfect
rather than merely small -- a genuine null would be noisy around zero, not
identical in all 220 cells.

    C:/research-toolchain/venv/Scripts/python.exe \\
        paper-a/src/analyze_frontier_verdict.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "paper-a" / "data" / "frontier"
OUT = D / "frontier_verdict_analysis.json"
SRC = D / "wording_gemini-3.1-flash-lite.jsonl"


def cell(r: dict) -> tuple:
    """The design cell a row occupies, so re-runs can be found."""
    return (r["variant"], r["template"], r["pair"], r["gender"])


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC.relative_to(ROOT)}")
        return 1
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = [r for r in rows if not r.get("error")]

    disagree = [r for r in rows if r["white"] != r["black"]]
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[cell(r)].append(r)
    repeats = {k: v for k, v in by_cell.items() if len(v) > 1}
    repeats_disagreeing = sum(
        1 for v in repeats.values()
        if len({(x["white"], x["black"]) for x in v}) > 1)

    # Is the verdict a function of the template alone? Collect the set of
    # verdicts each template ever produced, on either arm.
    by_tpl = collections.defaultdict(set)
    tpl_n = collections.Counter()
    for r in rows:
        by_tpl[r["template"]] |= {r["white"], r["black"]}
        tpl_n[r["template"]] += 1
    deterministic = all(len(v) == 1 for v in by_tpl.values())

    by_var = collections.defaultdict(list)
    for r in rows:
        by_var[r["variant"]].append(r)

    def advance(rs):
        return sum((x["white"] == "yes") + (x["black"] == "yes")
                   for x in rs) / (2 * len(rs))

    # WORDING EFFECT, HELD AT A FIXED TEMPLATE. Comparing raw advance rates
    # across wordings would be misleading here: four cells were re-run, so the
    # variants do not all carry the same template mix, and an unbalanced mix
    # alone shifts the rate. Conditioning on the template removes that.
    within = {}
    for t in sorted(by_tpl):
        rates = {v: advance([r for r in rs if r["template"] == t])
                 for v, rs in by_var.items()
                 if any(r["template"] == t for r in rs)}
        within[t] = dict(n_variants=len(rates),
                         min=min(rates.values()), max=max(rates.values()),
                         spread=max(rates.values()) - min(rates.values()))

    # The model reads the name even though the verdict never moves: count the
    # rationales that name the candidate. Evidence for "no resolution" over
    # "no effect", and checked rather than asserted.
    named = 0
    for r in rows:
        for arm in ("white", "black"):
            raw = (r.get(f"{arm}_raw") or "")
            first = (r.get(f"{arm}_name") or "").split()[:1]
            if first and first[0] in raw:
                named += 1

    n_pairs = len(rows)
    out = {
        "_what": "The Gemini arm of the frontier wording study: 220 matched "
                 "pairs scored on a thresholded yes/no verdict, because the "
                 "endpoint returns no next-token distribution to any caller.",
        "_why": "It measures what a binary outcome can resolve on a frontier "
                "model, which is the empirical form of the argument in §4.2 "
                "that binarising a graded outcome discards the comparison.",
        "model": rows[0]["model"],
        "provider": rows[0]["provider"],
        "n_pairs": n_pairs,
        "n_model_calls": 2 * n_pairs,
        "n_design_cells": len(by_cell),
        "n_cells_run_twice": len(repeats),
        "n_repeated_cells_disagreeing": repeats_disagreeing,
        "n_variants": len(by_var),
        "n_templates": len(by_tpl),

        "n_pairs_where_the_arms_disagree": len(disagree),
        "frac_pairs_where_the_arms_disagree": len(disagree) / n_pairs,

        "verdict_is_a_function_of_the_template_alone": deterministic,
        "by_template": {t: {"n_pairs": tpl_n[t],
                            "verdicts_observed": sorted(by_tpl[t]),
                            "advance_rate": advance(
                                [r for r in rows if r["template"] == t])}
                        for t in sorted(by_tpl)},
        "wording_spread_within_template": within,
        "max_wording_spread_within_any_template": max(
            v["spread"] for v in within.values()),

        "n_rationales_naming_the_candidate": named,
        "n_arm_responses": 2 * n_pairs,
        "_naming_note": (
            "A rationale that uses the candidate's first name shows the model "
            "read it. That the verdict is constant anyway is the point: the "
            "threshold, not the model, is what has no resolution."),
        "_limit": (
            "An arm that finds nothing cannot by itself separate 'no effect' "
            "from 'no resolution'. Two things license the second reading: the "
            "same design on models returning a distribution yields effects "
            "that are not zero, and the constancy here is exact rather than "
            "small -- a true null would scatter around zero, not repeat one "
            "value in every cell."),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"FRONTIER VERDICT ARM  {out['model']} ({out['provider']})")
    print("=" * 78)
    print(f"  matched pairs                {n_pairs}  "
          f"({out['n_model_calls']} model calls)")
    print(f"  design cells                 {out['n_design_cells']} "
          f"({out['n_cells_run_twice']} run twice, "
          f"{repeats_disagreeing} disagreeing)")
    print(f"  pairs where the arms differ  {len(disagree)} "
          f"({out['frac_pairs_where_the_arms_disagree']:.1%})")
    print()
    print(f"  {'template':<16}{'pairs':>7}{'advance':>10}   verdicts seen")
    for t, v in out["by_template"].items():
        print(f"  {t:<16}{v['n_pairs']:>7}{v['advance_rate']:>10.1%}   "
              f"{', '.join(v['verdicts_observed'])}")
    print()
    print(f"  verdict is a function of the template alone: {deterministic}")
    print(f"  largest wording spread within a fixed template: "
          f"{out['max_wording_spread_within_any_template']:.1%}")
    print(f"  rationales naming the candidate: {named} of "
          f"{out['n_arm_responses']}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
