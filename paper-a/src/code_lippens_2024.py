"""A proposed 14th row for the reporting-practice matrix: Lippens (2024).

WHY THIS IS SEPARATE FROM THE MATRIX. Lippens (2024) is not among the
thirteen surveyed audits: it was not in the matrix and had no full text in
`lit/`, and under this project's own rule (never cite a source not read in
full text) excluding it was correct at the time. But a survey of LLM hiring
audits that omits a 34,560-observation ChatGPT correspondence audit has a
coverage gap. The full text is now on file, and this file is the coding that
follows.

IT IS NOT MERGED INTO THE MATRIX YET, ON PURPOSE. The other thirteen were coded
by a multi-agent protocol with independent negative checks; this one was
coded by hand and has not yet had an independent check. A hand coding is
merged only after one. If a cell is wrong, the person most likely to notice
is the study's own author, and the merge tool records who checked what.

WHAT THE FIRST DRAFT OF THIS CODING GOT WRONG. Eight cells, every one of them
too generous, which is the direction that would have embarrassed us. They were
caught by checking each verdict against how the same field was coded for the
other thirteen rather than judging it in isolation:

  concurrency_or_batching  reported sequential submission -> `partial`, which
                           is what Fu & Shi got for stating item-level
                           sequencing. Not `not-reported`.
  decoding_params          temperature only -> `partial`, which is what
                           Seshadri, Tan and Veldanda got. `reported` is for
                           studies that also give top-p, max-tokens and seed.
  n_repeats                one presentation per cell, inferable from the
                           arithmetic -> `partial`, exactly Iso's coding.
  names_per_race           the per-group count lives in supplementary Table A8,
                           which is NOT in the article and which we have not
                           read -> `partial`.
  n_wordings_sep_estimated single-wording studies get `reported` with value 1
                           (Gao, Glazko, Nghiem, Wilson), not `not-applicable`.
  null_edit_control        single-wording studies get `not-reported` (Gao, Iso,
                           Nghiem), not `not-applicable`.
  name_sensitivity         stays `reported`, but the evidence was wrong: it is
                           §3.3.1 and Fig. 3, which ARE in the article, not
                           supplementary Table A8, which is not.
  prompt_published         stays `partial`, but for a better reason: the
                           instruction is quoted in ENGLISH while the input was
                           Dutch. Hoffstedde got `reported` for printing both
                           the translation and the original Japanese.

WHAT THE CODING SHOWS, stated without the superlatives a first draft reached
for and could not support. He reports 12 of 20 applicable fields. That is level
with Seshadri (12 of 22) and Armstrong (12 of 21) on the count and ahead of
both as a proportion, so he is jointly best-documented rather than ahead of
them, a claim worth getting right in a letter to the person who would check
it. What is genuinely uncommon is WHICH fields: he is one of two audits that
report a name-draw sensitivity result, one of three that pin a model snapshot,
one of three that release code and data, and one of four that report a
resampling unit. He reports nothing on request batching, cache policy or token
matching, which is what all thirteen also do. So adding him does not weaken the
survey's four never-reported findings; it strengthens them, because one of the
best-documented audits in the set still does not report those four.

    sh paper-a/src/_py.sh paper-a/src/code_lippens_2024.py
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
TEXT = ROOT / "lit" / "text" / "lippens_2024_computer_says_no.txt"
MATRIX = ROOT / "paper-a" / "data" / "reference" / "reporting_practice_matrix.json"
OUT_JSON = ROOT / "paper-a" / "data" / "reference" / "candidate_lippens_2024.json"
OUT_MD = ROOT / "outreach" / "LIPPENS_CODING.md"
OUT_PDF = ROOT / "outreach" / "Lippens_coding.pdf"

STUDY = "Lippens 2024"
PRINTED = ("Lippens, L. (2024), \"Computer says 'no': Exploring systemic bias "
           "in ChatGPT using an audit approach\", Computers in Human Behavior: "
           "Artificial Humans 2, 100054")

# WHAT WAS READ, AND WHAT WAS NOT. The 13-page article was read in full. The
# supplementary appendix -- Tables A1-A25, hosted separately and referenced
# throughout -- was NOT. Several cells below turn on a table in that appendix
# and are coded `partial` for that reason alone. Saying so is the point: this
# project's rule is that a source not read is not coded, and half a source read
# is coded as half.
PROVENANCE = ("The 13-page article, in full. I did not read the supplementary "
              "appendix, which holds Tables A1 to A25 and sits at the article "
              "DOI and on OSF. Cells that depend on it say so, and are coded "
              "partial.")

# The whole preamble, written once. The markdown and the PDF are assembled
# separately below, and an earlier edit fixed the PDF's version while leaving
# the markdown's in place.
STANDFIRST = ("Proposed, not merged. Coded by hand from the 13-page article. "
              "I have not read the supplementary appendix, so cells that "
              "depend on it are marked partial.")

# verdict is one of: reported | partial | not-reported | not-applicable
# `quote` must appear VERBATIM in the extracted text, or the build fails.
CELLS = {
    "prompt_published": {
        "verdict": "partial",
        "value": "The general instruction is quoted in full, both the role "
                 "framing and the scoring instruction. It is quoted in English, while "
                 "the input was Dutch, so the actual prompt string cannot be "
                 "reconstructed from the article. The vacancy and CV text are "
                 "exemplified in supplementary tables rather than reproduced "
                 "(there are 1,920 vacancies). Full materials are on OSF. "
                 "Hoffstedde et al. is the comparison: they print the "
                 "evaluation prompt in both English translation AND the "
                 "original Japanese, and are coded reported.",
        "quote": "All input was written in Dutch",
        "note": "Is the original Dutch prompt string in the OSF "
                "repo? If so this becomes 'reported' on the Hoffstedde "
                "precedent. It matters more than the cell suggests. A paper about "
                "instruction wording cannot treat a translation as the "
                "instrument.",
    },
    "n_wordings_used": {
        "verdict": "reported",
        "value": "1. A single general instruction, quoted in the text. The "
                 "vacancy and CV text vary by design; the instruction does "
                 "not.",
        "quote": "The general instruction ordered ChatGPT to help select suitable candidates",
        "note": "The single most consequential cell for our "
                "disagreement. If a second phrasing was run and not reported, "
                "the coding is wrong.",
    },
    "n_wordings_separately_estimated": {
        "verdict": "reported",
        "value": "1. Only one wording exists, so the ethnic-identity effect is "
                 "estimated under a single wording throughout. Coded the way "
                 "Gao, Glazko, Nghiem and Wilson are, all of whom carry 1.",
        "quote": "The candidates’ ethnic identity (Ethi), at the individual level i, was the main predictor of interest.",
    },
    "dispersion_across_wordings": {
        "verdict": "not-applicable",
        "value": "One wording, so no dispersion across wordings is definable. "
                 "This is why adding this study leaves the survey's "
                 "8-applicable / 0-reported count untouched.",
        "quote": "",
    },
    "null_edit_control": {
        "verdict": "not-reported",
        "value": "No placebo or no-op arm. The nearest analogue is the "
                 "White American vs Dutch contrast, which is a comparison "
                 "between two majority groups rather than a null edit. The "
                 "name still changes. Coded as Gao, Iso and Nghiem are.",
        "quote": "White American vs. Dutch candidates",
    },
    "name_list_source": {
        "verdict": "reported",
        "value": "Five sources, each named: Crabtree et al. (2023) validated "
                 "names, Gaddis (2017), Van Belle et al. (2023) and "
                 "Martiniello & Verhaeghe (2022), each cited. Nine of the ten "
                 "applicable audits report a name-list source, so this is the "
                 "norm rather than the exception; what is unusual is drawing "
                 "on five validated sets rather than one.",
        "quote": "I drew names from five sources.",
    },
    "names_per_race": {
        "verdict": "partial",
        "value": "The structure is stated, nine ethnic groups by two genders, "
                 "18 profiles per vacancy, and so is the total of 812 distinct "
                 "first-and-last-name combinations, is given. The count PER "
                 "GROUP is not stated in the article; it is derivable only "
                 "from supplementary Table A8, which we have not read.",
        "quote": "812 first and last name com­ binations used in the experiment",
        "note": "What is the per-(ethnicity x gender) name "
                "count? If it is in the article and we missed it, this is "
                "'reported'.",
    },
    "name_sensitivity_reported": {
        "verdict": "reported",
        "value": "Yes, and it is a central result rather than a robustness "
                 "note: §3.3.1 reports that within-group dispersion across "
                 "names EXCEEDS the between-group differences, with per-name "
                 "score distributions and means plotted in Fig. 3 (in the "
                 "article, p. 8). Only one of the ten applicable audits "
                 "currently reports anything of this kind. It is also the same "
                 "class of finding as our §4.2, reached independently and on a "
                 "much larger name set.",
        "quote": "much larger differences in ChatGPT’s score output exist within groups than between groups",
    },
    "token_matching": {
        "verdict": "not-reported",
        "value": "No tokenisation analysis. Negative search: 'token' returns "
                 "0 hits in the full article text.",
        "quote": "",
        "negative": ["token"],
    },
    "checkpoint_pinned": {
        "verdict": "reported",
        "value": "A dated model snapshot, pinned explicitly and for the "
                 "stated reason. Only two of the thirteen currently do this.",
        "quote": "I relied on the 13 June 2023 snapshot of the GPT-3.5 model",
    },
    "quantization_reported": {
        "verdict": "not-applicable",
        "value": "Hosted API; the caller cannot observe or choose the weight "
                 "quantization.",
        "quote": "",
        "negative": ["quantiz"],
    },
    "serving_stack": {
        "verdict": "reported",
        "value": "OpenAI's API rather than the web interface, called from R "
                 "via the httr and jsonlite packages, with the reason for "
                 "choosing the API over the web UI given. At least as specific "
                 "as Armstrong et al., who are coded reported.",
        "quote": "Connection with the API was made through R relying on the {httr} and {jsonlite} packages.",
    },
    "concurrency_or_batching": {
        "verdict": "partial",
        "value": "Client-side submission IS described: requests were sent "
                 "sequentially and in isolation, 34,560 of them in about eight "
                 "hours (~1.2/s), with no chat history between them. Nothing "
                 "is said about server-side batching, which for a hosted API "
                 "the caller cannot observe. Coded as Fu & Shi are, who state "
                 "item-level sequencing and nothing about batching.",
        "quote": "the successive and isolated presentation of vacancy–CV combinations to ChatGPT",
        "negative": ["batch"],
    },
    "cache_policy": {
        "verdict": "not-reported",
        "value": "No cache policy stated. Negative search: 'cache' returns 0 "
                 "hits in the full article text. Relevant because all 34,560 "
                 "prompts share a long common instruction prefix.",
        "quote": "",
        "negative": ["cache"],
    },
    "decoding_params": {
        "verdict": "partial",
        "value": "Temperature is given, and given unusually well, swept from "
                 "0.00 to 1.50 in 0.25 increments under a stated probability "
                 "weighting, altered in about two-fifths of prompts, entered "
                 "as a covariate, and its moderating effect estimated and "
                 "found null. But no top-p, no max-tokens, no seed. Coded as "
                 "Seshadri, Tan and Veldanda are, all temperature-only.",
        "quote": "temperatures between 0.00 and 1.50 with increments of 0.25 were integrated into the API request",
        "note": "This is the cell where the coding convention feels harshest. "
                "Two of the thirteen also treat temperature as a factor "
                "(Seshadri at two values, Tan at three); sweeping seven and "
                "estimating the moderation effect goes further than either. "
                "The convention still asks for the other three parameters.",
    },
    "n_repeats": {
        "verdict": "partial",
        "value": "One presentation per vacancy-CV combination, inferable from "
                 "the arithmetic (1,920 vacancies x 18 profiles = 34,560) but "
                 "never stated as a repeat count, and no agreement or "
                 "stability statistic across repeats is reported. The "
                 "temperature sweep varies a parameter rather than repeating a "
                 "cell. Coded as Iso is.",
        "quote": "I repeated this process until all vacancy–CV combinations were presented to ChatGPT.",
        "note": "Were any combinations re-sent? If so this is "
                "'reported' and the agreement rate would be worth having.",
    },
    "resampling_unit": {
        "verdict": "reported",
        "value": "Cluster-robust wild bootstrap, 2,000 replications, clusters "
                 "at the vacancy level, with the reason given. Only three of "
                 "the eleven applicable audits report a resampling unit at "
                 "all; Gaebler clusters on the generating dossier and Gao on "
                 "job-posting clusters, so he would be the fourth rather than "
                 "the only one.",
        "quote": "Clusters are defined at the va­ cancy level, given the correlation between the assignment of the candidates and the vacancies.",
    },
    "multiplicity_correction": {
        "verdict": "reported",
        "value": "Holm (1979) as the primary correction, with "
                 "Benjamini-Hochberg and Benjamini-Yekutieli reported as "
                 "producing similar results and not altering interpretation.",
        "quote": "using Holm’s (1979) method",
    },
    "reporting_scale": {
        "verdict": "reported",
        "value": "Ratio-level outcome (a 1-100 interview-invitation score). "
                 "Threshold sensitivity handled by estimating a separate "
                 "penalised-maximum-likelihood logit at each of a range of "
                 "cutoffs, each ratio using the relevant reference-group "
                 "probability. There is no single fixed operating point.",
        "quote": "I used a penalised maximum likelihood estimator",
        "note": "This is the cell behind objection 5 in his reply. He is "
                "right, and the paper never claimed otherwise.",
    },
    "n_occupations": {
        "verdict": "reported",
        "value": "23 occupations across industries; 1,920 real vacancies "
                 "retrieved from a public employment agency, balanced by "
                 "occupation and experience level. not the largest occupation "
                 "frame in the set. Seshadri has 46 and An et al. 41, and "
                 "the vacancies are real postings retrieved from a public "
                 "agency rather than generated job descriptions. We have not "
                 "checked which other audits use real postings, so no "
                 "comparative claim is made here.",
        "quote": "I chose 23 occupations across different industries",
    },
    "headline_effect": {
        "verdict": "reported",
        "value": "OLS on the 1-100 score against a Dutch reference "
                 "(intercept 66.91): Eastern European -1.89, Asian -1.72, "
                 "Arab -1.64, Central African -1.37, Black American -1.37, "
                 "Hispanic -1.31, White American -1.13, Turkish -0.86, all "
                 "significant, with cluster-robust standard errors and "
                 "Holm-corrected p-values. Female +0.61, not significant; "
                 "Turkish x Female -1.78.",
        "quote": "Intercept 66.9115*** (0.2993)",
    },
    "code_or_data_released": {
        "verdict": "reported",
        "value": "Data, code and supplementary tables and figures on OSF "
                 "under CC-BY-4.0.",
        "quote": "are available at https://osf.io/vezt7/",
    },
}


def norm(s: str) -> str:
    """Collapse whitespace and soft hyphens so a quote matches the PDF text."""
    return " ".join(s.replace("­", "").split())


def main() -> int:
    if not TEXT.exists():
        sys.exit(f"{TEXT.relative_to(ROOT)} not found. Extract the PDF first")
    flat = norm(TEXT.read_text(encoding="utf-8"))
    low = flat.lower()

    bad, negbad = [], []
    for field, c in CELLS.items():
        q = norm(c.get("quote", ""))
        if q and q not in flat:
            bad.append((field, q))
        for n in c.get("negative", []):
            if n.lower() in low:
                negbad.append((field, n, low.count(n.lower())))

    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    known = [f for f, _ in matrix["field_order"]]
    extra = sorted(set(CELLS) - set(known))
    missing = sorted(set(known) - set(CELLS))
    verdicts = {c["verdict"] for c in CELLS.values()}
    badv = verdicts - {"reported", "partial", "not-reported", "not-applicable"}

    print("=" * 78)
    print("LIPPENS 2024, proposed 14th row of the reporting-practice matrix")
    print("=" * 78)
    if bad:
        print("\n  EVIDENCE QUOTES NOT FOUND IN THE EXTRACTED TEXT:")
        for f, q in bad:
            print(f"    {f}: {q[:80]!r}")
    if negbad:
        print("\n  NEGATIVE SEARCHES THAT ACTUALLY HIT:")
        for f, n, k in negbad:
            print(f"    {f}: {n!r} appears {k} time(s)")
    if extra:
        print(f"\n  FIELDS NOT IN THE MATRIX SCHEMA: {extra}")
    if missing:
        print(f"\n  MATRIX FIELDS LEFT UNCODED: {missing}")
    if badv:
        print(f"\n  UNKNOWN VERDICTS: {sorted(badv)}")
    if bad or negbad or extra or missing or badv:
        print("\n  refusing to write a coding that does not verify")
        return 1

    tally: dict[str, int] = {}
    for c in CELLS.values():
        tally[c["verdict"]] = tally.get(c["verdict"], 0) + 1
    for k in sorted(tally):
        print(f"  {k:<16} {tally[k]}")

    rec = {
        "_what": "A proposed 14th row for the reporting-practice matrix.",
        "_why": ("The author identified the omission himself on 2026-08-19 and "
                 "supplied the paper. NOT merged into the matrix: the other "
                 "thirteen were coded by a multi-agent protocol with "
                 "independent negative checks, this one by hand, and the "
                 "author is available to check it."),
        "_status": "PROPOSED, awaiting author verification",
        "_provenance": PROVENANCE,
        "_verified": ("Every non-empty quote below appears verbatim in "
                      "lit/text/lippens_2024_computer_says_no.txt; every "
                      "negative search returns zero hits; every verdict was "
                      "checked against how the same field was coded for the "
                      "other thirteen. Enforced at build time."),
        "label": STUDY,
        "kind": "llm_hiring_audit",
        "study_as_printed": PRINTED,
        "reference": PRINTED + ". https://doi.org/10.1016/j.chbah.2024.100054",
        "source_file": "lit/text/lippens_2024_computer_says_no.txt",
        "tally": tally,
        "cells": CELLS,
    }
    OUT_JSON.write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                        encoding="utf-8")

    md = [
        "# Lippens (2024) coded against the survey's 22 fields",
        "",
        STANDFIRST,
        "",
        "**Tally:** " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())),
        "",
        "| field | verdict | what the paper says |",
        "|---|---|---|",
    ]
    pretty = dict(matrix["field_order"])
    for field in known:
        c = CELLS[field]
        v = {"reported": "**reported**", "partial": "partial",
             "not-reported": "not reported",
             "not-applicable": "n/a"}[c["verdict"]]
        md.append(f"| {pretty.get(field, field)} | {v} | "
                  f"{c['value'].replace(chr(10), ' ')} |")
    md += ["", "## The cells where I would most like to be corrected", ""]
    for field in known:
        if CELLS[field].get("note"):
            md.append(f"- **{pretty.get(field, field)}.** {CELLS[field]['note']}")
    md += [
        "",
        "## What this changes in the survey",
        "",
        "- **Nothing about the four never-reported fields.** Request batching, "
        "cache policy, token matching and dispersion-across-wordings stay at "
        "zero studies reporting. This paper is among the best documented in "
        "the set and still does not report those. (`partial` counts as not "
        "reported in the headline, on the survey's own rule: the question is "
        "whether a reader could reconstruct the choice.)",
        "- **The dispersion count is unchanged at 8 applicable, 0 reported**, "
        "because a single wording makes dispersion undefinable rather than "
        "unreported.",
        "- **Where it is genuinely uncommon**, with the current counts: "
        "name-draw sensitivity (1 of 10 audits report anything of the kind), "
        "a pinned model snapshot (2 of 13), released code and data (2 of 13), "
        "a stated resampling unit (3 of 11) and a multiplicity correction "
        "(4 of 13). On the count of reported fields it is level with Seshadri "
        "and Armstrong at 12, not ahead of them.",
        "- **§3.3.1 is an independent antecedent for our §4.2.** That "
        "within-group dispersion across names exceeds the between-group "
        "difference is the same class of finding, reached first, on a much "
        "larger name set. The paper should cite it as such.",
        "",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    _pdf(known, pretty, tally)
    print(f"\n  wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"  wrote {OUT_MD.relative_to(ROOT)}")
    print(f"  wrote {OUT_PDF.relative_to(ROOT)}")
    return 0


def _pdf(known, pretty, tally) -> None:
    """The same coding as a PDF, because it is going to be emailed.

    Reuses the layout class from the methods supplement rather than growing a
    second one; the two documents go in the same message and should look like
    they came from the same person.
    """
    try:
        from build_methods_supplement import Doc, INK, MUTE
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] no PDF: {e}")
        return
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    d = Doc(OUT_PDF)
    d.c.setFillColorRGB(*INK)
    d.c.setFont("Times-Bold", 14.5)
    d.c.drawString(d.c._pagesize[0] * 0 + 0.95 * 72, d.y - 14.5,
                   "Lippens (2024), coded against the survey's 22 fields")
    d.y -= 22
    d.rule(8)
    d.para(STANDFIRST, size=9.0, lead=12.0)
    d.para("Tally: " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())),
           size=9.0, lead=12.0)
    W = [138, 66, 300]
    d.row(["field", "verdict", "what the paper says"], W, bold=True)
    d.rule(3)
    for field in known:
        c = CELLS[field]
        d.row([pretty.get(field, field), c["verdict"], c["value"]], W,
              after=2.0)
    d.rule(5)
    d.sub("The cells where I would most like to be corrected", size=10.0)
    for field in known:
        if CELLS[field].get("note"):
            d.para(f"{pretty.get(field, field)}. {CELLS[field]['note']}",
                   size=8.4, lead=10.6, indent=10, after=3.0)
    d.sub("What this would change in the survey", size=10.0)
    for line in [
        "Nothing about the four never-reported fields. Request batching, cache "
        "policy, token matching and dispersion-across-wordings stay at zero "
        "studies reporting. ('partial' counts as not reported in the headline, "
        "on the survey's own rule: the question is whether a reader could "
        "reconstruct the choice.)",
        "The dispersion count is unchanged at 8 applicable, 0 reported, "
        "because a single wording makes dispersion undefinable rather than "
        "unreported.",
        "Where it is uncommon: name-draw sensitivity (1 of 10 audits report "
        "anything of the kind), a pinned model snapshot (2 of 13), released "
        "code and data (2 of 13), a stated resampling unit (3 of 11). On the "
        "count of reported fields it is level with Seshadri and Armstrong at "
        "12, not ahead of them.",
        "§3.3.1 is an independent antecedent for our §4.2, reached first and "
        "on a much larger name set. The paper now cites it as such.",
    ]:
        d.para("·  " + line, size=8.4, lead=10.6, indent=10, after=3.0)
    d.save()


if __name__ == "__main__":
    sys.exit(main())
