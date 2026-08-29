"""The design table: what each analysis treats, measures, assigns and clusters.

WHY THIS EXISTS. An external reviewer of the two-page summary (2026-08-19)
asked for exactly this and named the columns:

    "A design table for each analysis: treatment, outcome, assignment unit,
     observation unit, block, target population, estimator, and uncertainty
     estimator."

He asked because three of his objections turned out to be about scope. The
paper's claims about the resampling unit and the pairing are true of a design
that scores each résumé alone and imposes the pairing afterwards, and false of
a correspondence design that preconstructs profiles inside a prespecified
block. A reader cannot tell which one a paper has from prose spread over two
sections, and that ambiguity is what he was objecting to. One table settles it
for every analysis at once.

It is also a referee's first question and the paper did not have an answer in
one place.

HOW IT IS BUILT. The design columns are a DECLARATION -- prose describing what
the experiment did, which no artifact contains -- and every count in the
`n` column is read from the artifact that analysis produced. So the table
cannot claim a sample size the analysis did not have, and `test_design_table.py`
asserts that every artifact named here exists and every declared count matches
what is on disk.

    sh paper-a/src/_py.sh paper-a/src/build_design_table.py
"""
from __future__ import annotations

import json
import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "paper-a" / "data"
OUT_MD = ROOT / "paper-a" / "docs" / "DESIGN_TABLE.md"
OUT_TEX = ROOT / "paper-a" / "facct" / "generated" / "tab-design.tex"
OUT_JSON = D / "reference" / "design_table.json"


def art(*parts) -> pathlib.Path:
    return D.joinpath(*parts)


def load(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------------------
# THE COUNTS. Each is a function of the artifact, so the table's n column is
# interpolated rather than typed -- the same guarantee the paper makes.
def n_study2(a):
    m = a["llama-3.1-8b-instruct"]
    return (f"{m['overall']['n']} matched pairs per cell "
            f"({m['binary']['n']} résumé scorings)")


def n_namevar(a):
    m = a["llama-3.1-8b-instruct"]
    return (f"{m['n_pairs']:,} pairs, {m['n_first']} first names, "
            f"{m['n_last']} surnames, {m['n_variant']} wordings")


def n_spec(a):
    m = a["llama-3.1-8b-instruct"]["spec_curve"]
    return f"{m['n_specs']} specifications per model"


def n_second(a):
    s = a["summary"]
    return (f"{s['n_domain_model_cells']} domain-by-model cells, "
            f"{s['n_with_identified_effect']} with an identified effect")


def n_resamp(a):
    return (f"{a['n_boot']:,} bootstrap replicates "
            f"({a['n_boot_contrast']:,} for contrasts)")


def n_pairfree(a):
    return f"{a['n_perm']:,} random re-pairings"


def n_scale(a):
    return f"{a['summary']['n_models']} checkpoints"


def n_quant(a):
    return f"{len(a)} checkpoints re-run at a second quantization"


def n_matrix(a):
    return f"{a['n_llm_hiring_audits']} audits, {len(a['field_order'])} fields"


def n_cval(a):
    return f"{a['n_boot']:,} bootstrap replicates"


def n_tbal(a):
    return (f"{a['max_matching']['female_first'] + a['max_matching']['male_first']} "
            f"independent first-name pairs survive balancing")


def n_srn(a):
    return "closed-form, equicorrelated normal"


def n_mech(a):
    return f"{len([k for k in a if not k.startswith('_')])} checkpoints"


def n_front(a):
    # FOUR HERE, FIVE IN THE PAPER, AND BOTH ARE RIGHT. This artifact holds the
    # APIs that return a next-token distribution, so a margin can be computed.
    # The paper's frontier count is checkpoints TOUCHED, which additionally
    # includes the Gemini verdict-only arm below, whose endpoint returns no
    # distribution to any caller. Saying which is which is the entire purpose
    # of this table.
    return (f"{len(a['models'])} frontier APIs returning log probabilities "
            f"(a fifth, verdict-only, is the row below)")


# ---------------------------------------------------------------------------
# THE DECLARATION. Eight columns, exactly the ones the reviewer named.
#
# "Target population" is the column that most needs care and is the one most
# papers in this literature leave implicit. Almost nothing here generalises to
# a population of job applicants; it generalises to the instrument's own design
# space. Saying so is the point.
COMMON_POP = ("the twelve Bertrand–Mullainathan name pairs under the twelve "
              "wordings; NOT a population of applicants")

DESIGN = [
    {
        "sec": "4.1",
        "name": "The demographic effect (the inner estimand)",
        "treatment": "applicant first name, White vs Black, held matched on gender",
        "outcome": "renormalised decision margin, log P(yes) − log P(no); "
                   "and its probability-of-superiority transform",
        "assignment": "first-name pair, assigned to byte-identical résumé text",
        "observation": "matched pair (name pair × wording × template)",
        "block": "model × job posting",
        "population": COMMON_POP,
        "estimator": "mean paired difference; probability of superiority",
        "uncertainty": "percentile bootstrap resampling NAME PAIRS; "
                       "exact within-race permutation null",
        "artifact": art("delta_stability", "study2_v2.json"),
        "n": n_study2,
    },
    {
        "sec": "4.1",
        "name": "Dispersion of that effect across instruction wordings",
        "treatment": "instruction wording: 6 paraphrases (S1–S6) and 6 "
                     "null edits (N1–N6) changing no word, only punctuation "
                     "or line breaks",
        "outcome": "the row-1 effect, re-estimated under each wording",
        "assignment": "none: the wording set is crossed, every wording is "
                      "applied to every cell",
        "observation": "(model, posting, wording) cell",
        "block": "model × job posting",
        "population": COMMON_POP,
        "estimator": "SD across the twelve wordings, and that SD divided by "
                     "the cell's own effect",
        "uncertainty": "bootstrap over name pairs with numerator and "
                       "denominator resampled together",
        "artifact": art("delta_stability", "study2_v2.json"),
        "n": n_spec,
    },
    {
        "sec": "4.2",
        "name": "Which names were drawn",
        "treatment": "the identity of the first name and surname drawn",
        "outcome": "renormalised decision margin",
        "assignment": "first name and surname, crossed",
        "observation": "matched pair",
        "block": "model × wording",
        "population": "the Bertrand–Mullainathan list as transcribed in "
                      "names.py; not names in general",
        "estimator": "crossed random-effects variance decomposition, "
                     "σ_first / σ_last / σ_wording",
        "uncertainty": "cluster bootstrap over first-name pairs; "
                       "posterior interval where the model is fitted",
        "artifact": art("names", "name_variance.json"),
        "n": n_namevar,
    },
    {
        "sec": "4.5",
        "name": "Which job was posted",
        "treatment": "occupation: Business Analyst, Software Engineer, "
                     "Registered Nurse",
        "outcome": "renormalised decision margin",
        "assignment": "posting text, fixed per occupation",
        "observation": "matched pair",
        "block": "model × occupation",
        "population": "three postings chosen for gender-typing contrast; "
                      "not occupations in general",
        "estimator": "per-occupation effect and its across-wording SD",
        "uncertainty": "bootstrap over name pairs; dispersion null by "
                       "permutation across occupations",
        "artifact": art("occupation", "occupation_analysis.json"),
        "n": lambda a: f"{len([k for k in a['llama-3.1-8b-instruct'] if len(k) <= 4])} occupations",
    },
    {
        "sec": "B",
        "name": "Manipulation check: do the names carry the construct?",
        "treatment": "none; observational over the name list",
        "outcome": "correlation between the model's within-race name ranking "
                   "and external perception, callback and SES measures",
        "assignment": "n/a",
        "observation": "first name",
        "block": "race (correlations computed within race, then pooled)",
        "population": "the drawn first names",
        "estimator": "Spearman correlation, ranked inside each race then pooled",
        "uncertainty": "stratified re-ranking bootstrap; exact within-race "
                       "permutation null; MDE by simulation",
        "artifact": art("names", "construct_validity.json"),
        "n": n_cval,
    },
    {
        "sec": "C",
        "name": "The same wordings outside hiring",
        "treatment": "applicant/subject first name, as in row 1",
        "outcome": "renormalised decision margin",
        "assignment": "first-name pair",
        "observation": "matched pair",
        "block": "model × domain",
        "population": "two further decision domains; a check that the result "
                      "is not a hiring artefact",
        "estimator": "as row 2, per domain",
        "uncertainty": "bootstrap over name pairs",
        "artifact": art("second_task", "second_task_analysis.json"),
        "n": n_second,
    },
    {
        "sec": "D",
        "name": "The same wordings on frontier APIs",
        "treatment": "as row 2",
        "outcome": "renormalised margin from returned log probabilities",
        "assignment": "first-name pair",
        "observation": "matched pair",
        "block": "model",
        "population": "frontier APIs that expose log probabilities",
        "estimator": "as row 2, with a noise floor subtracted",
        "uncertainty": "bootstrap over name pairs; repeat-based noise floor",
        "artifact": art("frontier", "frontier_margin_analysis.json"),
        "n": n_front,
    },
    {
        "sec": "D",
        "name": "The verdict-only frontier arm",
        "treatment": "as row 2",
        "outcome": "thresholded yes/no verdict: this endpoint returns no "
                   "next-token distribution to any caller, so no margin exists",
        "assignment": "first-name pair",
        "observation": "matched pair",
        "block": "model",
        "population": "one frontier API without log probabilities; it is "
                      "here to show what a binary outcome can resolve",
        "estimator": "difference in verdict rates",
        "uncertainty": "bootstrap over name pairs; repeated cells used to "
                       "measure disagreement directly",
        "artifact": art("frontier", "frontier_verdict_analysis.json"),
        "n": lambda a: (f"{a['model']}, {a['n_pairs']} matched pairs, "
                        f"{a['n_model_calls']:,} calls"),
    },
    {
        "sec": "E",
        "name": "Which quantization was downloaded",
        "treatment": "weight quantization of the same checkpoint",
        "outcome": "the row-1 effect",
        "assignment": "n/a: the same grid re-run on a second build",
        "observation": "matched pair",
        "block": "model",
        "population": "two checkpoints for which a second quantization was "
                      "obtainable",
        "estimator": "shift in the effect, divided by the wording SD",
        "uncertainty": "bootstrap over name pairs",
        "artifact": art("quantization", "quantization_analysis.json"),
        "n": n_quant,
    },
    {
        "sec": "5.2",
        "name": "How the requests were served",
        "treatment": "request batching and key-value cache residency",
        "outcome": "cell-level agreement between byte-identical repeats",
        "assignment": "n/a: repeated measurement of a fixed grid",
        "observation": "cell (one prompt, one scoring)",
        "block": "model, and separately server process / session",
        "population": "this serving stack; NOT a claim about deployed serving",
        "estimator": "fraction of cells agreeing exactly; bound on the induced "
                     "shift in the effect",
        "uncertainty": "reported as a bound, not an absence",
        "artifact": art("delta_stability", "noise_floor.json"),
        "n": lambda a: f"{len([k for k in a if not k.startswith('_')])} checkpoints",
    },
    {
        "sec": "6.1",
        "name": "The resampling unit",
        "treatment": "n/a: an analysis-time comparison of two estimators on "
                     "one dataset",
        "outcome": "width of the 95 % interval on the row-1 effect",
        "assignment": "n/a",
        "observation": "matched pair (row) vs first-name pair (cluster)",
        "block": "model",
        "population": "this crossed design; the unit follows the assignment "
                      "process and is not universal",
        "estimator": "ratio of interval widths, row bootstrap ÷ cluster "
                     "bootstrap",
        "uncertainty": "both intervals computed at the same replicate count "
                       "and seed",
        "artifact": art("delta_stability", "resampling_unit.json"),
        "n": n_resamp,
    },
    {
        "sec": "6.2",
        "name": "The reporting scale and the operating point",
        "treatment": "n/a: an analysis-time comparison",
        "outcome": "the log-odds-to-percentage-point conversion factor",
        "assignment": "n/a",
        "observation": "model",
        "block": "model",
        "population": "this panel; the quantity is defined per model and does "
                      "not transfer",
        "estimator": "0.25 ÷ the model's own mean p(1−p); and the "
                     "overstatement realised on a measured effect",
        "uncertainty": "realised value reported only for the checkpoints whose "
                       "effect is distinguishable from zero",
        "artifact": art("delta_stability", "reporting_scale.json"),
        "n": n_scale,
    },
    {
        "sec": "6.3",
        "name": "Which résumé was paired with which",
        "treatment": "n/a: an analysis-time degree of freedom that exists "
                     "only because each résumé is scored alone",
        "outcome": "probability of superiority under a re-pairing",
        "assignment": "n/a; in a blocked correspondence design this row does "
                      "not arise at all",
        "observation": "a whole pairing of the grid",
        "block": "gender (re-pairings drawn within gender)",
        "population": "this design; not designs that preconstruct and assign "
                      "profiles within a prespecified block",
        "estimator": "SD of the statistic over random re-pairings; and the "
                     "best-worst range under a maximum-weight matching",
        "uncertainty": "permutation over pairings",
        "artifact": art("names", "pairing_freedom.json"),
        "n": n_pairfree,
    },
    {
        "sec": "7",
        "name": "Position or fragmentation?",
        "treatment": "edits that move the name's token position, with "
                     "delimiter fragmentation held equal, and vice versa",
        "outcome": "renormalised decision margin",
        "assignment": "edit condition, applied to a fixed grid",
        "observation": "contrast (model × inference mode × condition pair)",
        "block": "model × inference mode",
        "population": "this edit family; a mechanism probe, not an estimate "
                      "of a population quantity",
        "estimator": "per-contrast difference",
        "uncertainty": "Benjamini–Hochberg across the declared contrast family",
        "artifact": art("mechanism_panel", "mech_panel_analysis.json"),
        "n": n_mech,
    },
    {
        "sec": "8",
        "name": "What the field reports",
        "treatment": "n/a: a literature survey",
        "outcome": "whether each audit reports each of the design fields above",
        "assignment": "n/a",
        "observation": "study × field cell",
        "block": "field",
        "population": "LLM hiring audits readable in full text; the frame is "
                      "stated and the exclusions are listed",
        "estimator": "count of studies reporting, per field, with an evidence "
                     "quote or a negative search behind every cell",
        "uncertainty": "'partial' counts as not reported, because the question "
                       "is whether a reader could reconstruct the choice",
        "artifact": art("reference", "reporting_practice_matrix.json"),
        "n": n_matrix,
    },
    {
        "sec": "9.1",
        "name": "The screening rule's false-positive rate",
        "treatment": "n/a: a calibration of the paper's own recommendation",
        "outcome": "rejection rate under the global null",
        "assignment": "n/a",
        "observation": "simulated wording set",
        "block": "n/a",
        "population": "equicorrelated normal wordings, as a function of ρ",
        "estimator": "closed-form rejection probability of the stated rule",
        "uncertainty": "exact, not simulated",
        "artifact": art("instrument", "screening_rule_null.json"),
        "n": n_srn,
    },
    {
        "sec": "C",
        "name": "Token matching of the pair (companion paper)",
        "treatment": "restriction to token-length-matched name pairs",
        "outcome": "the row-1 effect, before and after restriction",
        "assignment": "first-name pair",
        "observation": "matched pair",
        "block": "model (the tokenizer differs per model)",
        "population": "the Bertrand–Mullainathan list under four tokenizers",
        "estimator": "effect on the restricted subset vs the full grid",
        "uncertainty": "three procedures reported, which disagree at 0.05; "
                       "the disagreement is the result",
        "artifact": art("instrument", "token_balanced_grid.json"),
        "n": n_tbal,
    },
]

COLS = [("sec", "§"), ("name", "analysis"), ("treatment", "treatment"),
        ("outcome", "outcome"), ("assignment", "assignment unit"),
        ("observation", "observation unit"), ("block", "block"),
        ("population", "target population"), ("estimator", "estimator"),
        ("uncertainty", "uncertainty estimator"), ("_n", "n")]


def main() -> int:
    rows, missing = [], []
    for d in DESIGN:
        a = load(d["artifact"])
        if a is None:
            missing.append(str(d["artifact"].relative_to(ROOT)))
            n = ", artifact missing —"
        else:
            try:
                n = d["n"](a)
            except Exception as e:  # noqa: BLE001
                n = f", count failed: {e} —"
                missing.append(f"{d['artifact'].name}: {e}")
        r = {k: d[k] for k, _ in COLS if k in d}
        r["_n"] = n
        r["artifact"] = str(d["artifact"].relative_to(ROOT)).replace("\\", "/")
        rows.append(r)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"_what": "One row per analysis: treatment, outcome, units, block, "
                  "target population, estimator, uncertainty estimator.",
         "_why": "Requested by an external reviewer, 2026-08-19. Three of the "
                 "review's objections were about scope, which prose spread over two "
                 "sections cannot settle and one table can.",
         "n_analyses": len(rows), "rows": rows}, indent=1, ensure_ascii=False),
        encoding="utf-8")

    # ---- markdown, one block per analysis. A 10-column grid is unreadable
    # ---- at this text length, so each analysis gets a definition list.
    md = ["# The design table",
          "",
          "Generated by `paper-a/src/build_design_table.py`. Regenerate rather "
          "than edit. Every `n` is read from the artifact named beneath it; "
          "the design fields are a declaration of what the experiment did.",
          "",
          "> Requested by an external reviewer on 2026-08-19, who named these columns "
          "exactly. Three of that review's objections turned out to be about scope: the "
          "resampling-unit and pairing claims hold for a design that scores "
          "each résumé alone and pairs afterwards, and not for one that "
          "preconstructs profiles inside a prespecified block. A reader cannot "
          "tell which from prose spread across two sections.",
          "",
          "**The target-population column is the one to read first.** Almost "
          "nothing here generalises to a population of job applicants. It "
          "generalises to the instrument's own design space, and saying so is "
          "the point of the paper.",
          ""]
    for r in rows:
        md.append(f"### §{r['sec']}: {r['name']}")
        md.append("")
        for k, label in COLS:
            if k in ("sec", "name"):
                continue
            md.append(f"- **{label}.** {r[k]}")
        md.append(f"- *artifact:* `{r['artifact']}`")
        md.append("")
    if missing:
        md += ["## Artifacts that did not resolve", ""] + \
              [f"- `{m}`" for m in missing] + [""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    # ---- LaTeX: a landscape longtable, which is the only shape that fits ten
    # ---- columns of prose on a single-column page.
    def tex(s: str) -> str:
        for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                     ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                     ("−", "$-$"), ("×", "$\\times$"), ("÷", "$\\div$"),
                     ("—", "---"), ("–", "--"), ("’", "'"), ("‘", "`"),
                     ("“", "``"), ("”", "''"), ("σ", "$\\sigma$"),
                     ("ρ", "$\\rho$"), ("≤", "$\\leq$")):
            s = s.replace(a, b)
        return s

    tl = ["% GENERATED by paper-a/src/build_design_table.py -- do not edit.",
          "% Ten columns of prose do not fit a single-column page as a tabular.",
          "% Each analysis is a paragraph block instead, which is also easier to",
          "% read than a grid at this text length.",
          "\\begingroup\\small"]
    for r in rows:
        tl.append("\\par\\smallskip\\noindent\\textbf{"
                  + ("Appendix~" if r["sec"][:1].isalpha() else "\\S{}")
                  + tex(r["sec"])
                  + "\\quad " + tex(r["name"]) + "}\\par\\nopagebreak")
        items = [f"\\textit{{{tex(label)}.}} {tex(r[k])}"
                 for k, label in COLS if k not in ("sec", "name")]
        tl.append("\\begin{itemize}\\setlength\\itemsep{0pt}")
        tl += [f"  \\item {i}" for i in items]
        tl.append("\\end{itemize}")
    tl.append("\\endgroup")
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(tl) + "\n", encoding="utf-8")

    print("=" * 78)
    print("DESIGN TABLE")
    print("=" * 78)
    for r in rows:
        print(f"  §{r['sec']:<4} {r['name'][:52]:<54} {r['_n'][:36]}")
    print(f"\n  {len(rows)} analyses")
    if missing:
        print(f"  [!] {len(missing)} artifact(s) unresolved: {missing}")
    print(f"  wrote {OUT_MD.relative_to(ROOT)}")
    print(f"  wrote {OUT_TEX.relative_to(ROOT)}")
    print(f"  wrote {OUT_JSON.relative_to(ROOT)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
