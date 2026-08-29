"""Typeset the token-matching paper.

WHY THIS IS A SEPARATE PAPER. The parent audit (paper-a) measures how far an
LLM hiring disparity moves under choices no published audit reports: the
wording, the names, the job, the quantization, the resampling unit, the
reporting scale. One of its findings does not belong in that list, because it is
not a choice at all -- it is an assumption the design makes silently and that
turns out to be false. A correspondence audit sends two applications identical
in every respect but the name. To a language model they are not identical in
every respect but the name: they are different lengths in tokens, and only a
quarter to a third of the pairs drawn from the field's standard validated list
are token-matched.

In the parent paper that finding sits inside the longest subsection of the
results, where it is crowded out by the six choices around it. It is a
self-contained argument with its own literature and its own remedy, and it
reads better alone.

WHAT IS AND IS NOT REPEATED. The design, panel and outcome are shared with the
parent and are described here in full, because a reader should not need both
papers. Every number is interpolated from the same artifacts the parent uses --
there is one set of measurements, not two -- so the two papers cannot disagree.
The parent's other studies are cited, not restated.

WHAT THIS PAPER CLAIMS, in order of how much it rests on:

  1. Only a minority of pairs from the standard list are token-matched, on
     every tokenizer tested. This is arithmetic on the list and is not in
     doubt.
  2. The token-length difference predicts the measured effect on one of four
     checkpoints, on the resampling unit the design requires.
  3. Restricting to matched pairs moves the disparity, and the subset is too
     small to say by how much: three procedures on the name-pair unit give
     answers straddling 0.05.
  4. A grid balanced by construction can be built from the standard list and is
     too small to carry an audit.

    sh paper-a/src/_py.sh paper-c/src/build_paper_c.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper-a" / "src"))

import paperkit as pk  # noqa: E402

D = ROOT / "paper-a" / "data"
OUT = ROOT / "paper-c" / "figures" / "paper_token_matching.pdf"
FIGS = ROOT / "paper-a" / "figures"

MISSING: list[str] = []

SRC = {
    "nlen": D / "instrument" / "name_length_effect.json",
    "lpred": D / "instrument" / "length_prediction.json",
    "tbal": D / "instrument" / "token_balanced_grid.json",
    "mech": D / "mechanism_panel" / "mech_panel_analysis.json",
    "matrix": D / "reference" / "reporting_practice_matrix.json",
    "s2": D / "delta_stability" / "study2_v2.json",
}

SHORT = {
    "llama-2-7b-chat": "Llama-2-7B-chat",
    "llama-2-13b-chat": "Llama-2-13B-chat",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
    "mistral-7b-instruct-v0.1": "Mistral-7B-Instruct v0.1",
    "mistral-7b-instruct-v0.3": "Mistral-7B-Instruct v0.3",
    "mistral-7b-v0.1-base": "Mistral-7B v0.1 base",
}
TINY = {k: v.replace("-Instruct", "").replace("-chat", "")
        for k, v in SHORT.items()}
ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
NUM = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
       6: "six", 8: "eight", 12: "twelve", 48: "forty-eight"}


def load(key):
    p = SRC[key]
    if not p.exists():
        MISSING.append(key)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(x, n=3, sign=False):
    if x is None:
        return "--"
    s = f"{x:+.{n}f}" if sign else f"{x:.{n}f}"
    return s


def pct(x, n=0):
    return "--" if x is None else f"{100 * x:.{n}f} %"


def main() -> int:
    nlen = load("nlen")
    lpred = load("lpred")
    tbal = load("tbal")
    mech = load("mech")
    matrix = load("matrix")

    if not nlen:
        sys.exit("name_length_effect.json is required")

    models = [m for m in ORDER if m in nlen]
    CL = {m: nlen[m]["token_matched_first_name_clustered"] for m in models}

    # ---- quantities used in more than one place ------------------------
    _frac = {m: nlen[m]["n_same_length"] / nlen[m]["n_pairs"] for m in models}
    _fmin, _fmax = min(_frac.values()), max(_frac.values())
    # §10 RECOMMENDATION 1 ASKS FOR TWO NUMBERS and this paper printed one.
    # The only delta it showed was Table 4's mean SIGNED difference, which the
    # shift prediction needs and which is two to four times smaller than the
    # absolute mean an auditor would be reporting.
    _absd = {}
    for _m in models:
        _p = D / "instrument" / f"name_length_{_m}.json"
        if _p.exists():
            _dl = [abs(r["delta_in_context"])
                   for r in json.loads(_p.read_text(encoding="utf-8"))["pairs"]]
            if _dl:
                _absd[_m] = sum(_dl) / len(_dl)
    # HOW MANY AUDITS ACTUALLY USE THIS LIST. Every version of this paper said
    # the literature "standardised on" the Bertrand and Mullainathan list, in
    # the abstract, §1, §2 and the conclusion. The survey artifact this paper
    # cites for every other claim about the literature says otherwise, and
    # nobody had asked it. It is the canonical list and it is not the common
    # one: the rest draw on voter files, administrative records, other
    # name-selection studies and a generator library.
    # WHAT ONE ROW OF TABLES 2-4 IS AN AVERAGE OF. §3 named the templates and
    # never the wordings, so a reader could not tell that a per-pair effect is
    # a mean over the template-by-wording grid rather than a single reading.
    # Derived from the same rows the analysis reads, so it cannot drift.
    _rows0 = [json.loads(x) for x in
              (D / "names" / f"names_{models[0]}.jsonl")
              .read_text(encoding="utf-8").splitlines() if x.strip()]
    _tpl = sorted({r["template"] for r in _rows0})
    _wrd = sorted({r["variant"] for r in _rows0})
    _cells = len(_tpl) * len(_wrd)
    _nls = ((matrix or {}).get("counts", {}).get("name_list_source") or {})
    _nbm = sum(1 for s in (matrix or {}).get("studies", [])
               if s["kind"] == "llm_hiring_audit"
               and s["cells"]["name_list_source"]["verdict"] != "not-applicable"
               and "Bertrand" in str(s["cells"]["name_list_source"]["value"]))
    # The token-matching row's denominator is its own n_applicable, not the
    # panel size: one audit manipulates no names, so the choice does not apply
    # to it, and the matrix excludes such cells from both halves of the count.
    _tmc = ((matrix or {}).get("counts", {}).get("token_matching") or {})
    _slope_sig = [m for m in models if CL[m]["slope_same_unit"]["p"] < 0.05]
    _away = [m for m in models
             if abs(CL[m]["effect_token_matched"]["est"])
             > abs(CL[m]["effect_all_pairs"]["est"])]
    _interp = [m for m in models if CL[m]["growth_ratio"].get("interpretable")]
    _interp_away = [m for m in _interp if m in _away]
    # TALLIES THE PROSE USED TO TYPE. Each of these was a count over an
    # artifact field, written out as an English word and never read back:
    # the row-level slope count in Table 2's caption, and the number of
    # procedures that reject in §6.1. Both were right on the current
    # artifacts, which is exactly why nothing caught them.
    _slope_sig_row = sum(1 for m in models if nlen[m]["p"] < 0.05)
    _best = max(_interp, key=lambda m: CL[m]["growth_ratio"]["est"]) \
        if _interp else None

    # Does the position manipulation corroborate the slope? Counted, because
    # the parent paper asserted that it does and it does not.
    # ONLY THE CELL COUNT. This used to also tally how often the displacement
    # shift agreed in sign with the model's slope. Both conditions add their
    # tokens inside the posting, identically in both arms, so the regressor of
    # §5 is unchanged and §7's model predicts a shift of exactly zero: there
    # is no sign for the data to agree or disagree with, and the tally was
    # arithmetic on an undefined comparison.
    _d89_total = 0
    if mech:
        for m, v in mech.items():
            if m.startswith("_") or not isinstance(v, dict):
                continue
            if m not in nlen:
                continue
            for mode in ("chat", "raw"):
                pc = (v.get(mode) or {}).get("per_condition") or {}
                if (pc.get("D0") or {}).get("logodds", {}).get("est") is None:
                    continue
                for c in ("D8", "D9"):
                    if (pc.get(c) or {}).get("logodds", {}).get("est") is None:
                        continue
                    _d89_total += 1

    # ---- abstract -------------------------------------------------------
    abstract = (
        "A correspondence audit sends two applications that differ in one "
        "attribute and attributes the difference in outcome to that attribute. "
        "Applied to a language model, the design assumes the two prompts are "
        "identical apart from the name. They are not: they are different "
        "lengths in tokens. Drawing names from the canonical Bertrand and "
        "Mullainathan list, only "
        + pct(_fmin, 0) + " to " + pct(_fmax, 0) + " of matched pairs are "
        "token-matched on the tokenizers of four open-weight checkpoints, so "
        "most of the design carries a length difference alongside the "
        "demographic one."
        + f" The difference predicts the measured effect on "
        + NUM.get(len(_slope_sig), str(len(_slope_sig)))
        + " of " + NUM.get(len(models), str(len(models)))
        + " checkpoints when name pairs rather than rows are resampled, and "
        "restricting the audit to token-matched pairs moves the reported "
        "disparity further from zero on "
        + NUM.get(len(_away), str(len(_away))) + " of "
        + NUM.get(len(models), str(len(models))) + ". How far it moves cannot "
        "be settled here: the matched subset is small enough that three "
        "standard procedures on the same data give answers straddling 0.05, "
        "and we report that rather than the one that agrees with us."
        + " A grid balanced by construction can be built from the same list "
        + ((f"and reaches {tbal['n_pairs']} rows built from "
            f"{tbal['max_matching']['female_first'] + tbal['max_matching']['male_first']} "
            "independent name pairs") if tbal else "")
        + ", which is too small to carry an audit. The remedy is not to "
        "abandon the list but to report the token statistics of the pairs "
        "used, which no surveyed audit does"
        + ((f": of {_tmc.get('n_applicable', '--')} LLM hiring audits read in "
            "full text whose design manipulates a name, none reports them")
           if matrix else "")
        + "."
    )

    paper = pk.Paper(OUT, "The Matched Pair Is Not Token-Matched", "David Mao")
    paper.title_block(
        ["The Matched Pair Is Not Token-Matched:",
         "A Silent Assumption in Correspondence Audits",
         "of Language Models"],
        "David Mao", "Independent", abstract,
        keywords=("algorithmic audit · tokenization · correspondence "
                  "study · résumé screening · measurement "
                  "validity"),
        email="davidmao.xyz@gmail.com")

    P = paper.para
    H = paper.heading

    # ---- 1 --------------------------------------------------------------
    H("1  Introduction")
    P("Bertrand and Mullainathan sent employers résumés drawn from a bank "
      "and assigned each one a White-sounding or a Black-sounding name at "
      "random, and attributed the gap in callbacks to the name. The "
      "design’s strength is that the name is assigned independently of "
      "everything else on the page, so whatever else is true of the "
      "résumés cannot explain the result.", indent=False)
    P("Language models are now audited the same way. The two prompts are "
      "built from one template with one substitution, so they are identical "
      "in characters apart from the name, and the inference proceeds as "
      "before. But a model does not read characters. It reads tokens, and two "
      "strings identical apart from a name are not in general the same length "
      # NAMES AND COUNTS FROM THE ARTIFACT. This read "Allison may be one
      # token where Lakisha is three": Allison is two on all four tokenizers
      # standing alone and one in context, so the printed pair mixed one
      # name's in-context count with the other's isolated count and matched
      # neither measure. Anne is one on all four, and the measure is now named.
      "in tokens: standing alone, “Anne” is "
      + (f"{sorted({v['first']['Anne']['iso'] for v in tbal['per_model_tokens'].values()})[0]} "
         "token on every tokenizer here where “Lakisha” is "
         f"{sorted({v['first']['Lakisha']['iso'] for v in tbal['per_model_tokens'].values()})[0]}"
         if tbal else "shorter than “Lakisha”")
      + ". The pair is matched in the units the auditor wrote and "
      "unmatched in the units the model consumes.")
    P("This paper measures how often that happens on the canonical Bertrand "
      "and Mullainathan list, whether the difference predicts the "
      "measured effect, and what happens to a reported disparity when the "
      "audit is restricted to pairs that are matched in tokens. The answer to "
      "the first question is not close: on every tokenizer we tested, most "
      "pairs are unmatched. The answers to the second and third are weaker "
      "than we would like, and we report the weakness rather than the "
      "strongest reading available. The measurements come from a larger audit "
      "of this literature’s design choices by the same author, called the "
      "parent study below and cited as Mao (2026); §3 gives the design in "
      "full, so this paper can be read on its own.")

    P("<b>What this is not.</b> This is not a claim that measured disparities "
      "are artifacts of tokenization. The length difference is correlated "
      "with the demographic contrast by construction — a rarer name "
      "segments into more pieces — and no design that varies the name can "
      "separate them. It is a claim that a design describing itself as "
      "matched is unmatched on a dimension the model is sensitive to, that "
      "the imbalance is large, and that no audit we surveyed reports it. "
      "Simonsohn, Simmons and Nelson (2020) ran a specification curve over "
      "Bertrand and Mullainathan’s own data. Every specification in it "
      "varies the analysis, which is what the method is for; none varies how "
      "the pair was built, which is the layer this paper opens.")

    # ---- 2 --------------------------------------------------------------
    H("2  Background")
    P("<b>Correspondence audits.</b> The canonical instance is Bertrand and "
      "Mullainathan (2004), who report a callback gap between "
      "White-sounding and Black-sounding names and validate the list with a "
      "human perception survey. A name enters that list by having, in their "
      "words, the highest ratio of frequency in one racial group to frequency "
      "in the other, so distinctiveness there IS rarity in the contrasting "
      "group. That is the selection rule this paper’s opening caveat "
      "refers to: the correlation between a name’s demographic signal "
      "and its rarity is not a coincidence of the list but the criterion that "
      "built it, and a rarer string segments into more tokens. Veldanda et "
      "al. (2023) reuse those first names to audit language models, which is "
      "the list and the construction measured here. The LLM audits as a whole "
      "have not settled on it: of the "
      + str(_nls.get("n_applicable", "--"))
      + " audits in our survey that use a name list at all, "
      + str(_nbm)
      + " draw on it, and the rest use voter files, administrative records, "
      "other published name-selection studies, or a generator library. The "
      "measurement below is therefore about the canonical list rather than "
      "about the field's modal one.", indent=False)
    P("<b>Tokenization and names.</b> An and Rudinger (2023) establish on a "
      "social-commonsense task that a first name’s tokenization length "
      "affects how a model treats it, independently of the name’s "
      "demographic association, and stratify names by race, gender and "
      "tokenization length to separate the two. That result is the direct "
      "antecedent of this paper: it establishes the channel on a different "
      "task, and we ask what the channel does to a hiring audit that does not "
      "control for it.")
    P("<b>What names are selected on.</b> The audits that build their own "
      "list still select on distinctiveness, and some already balance on a "
      "second property that nobody reads as demographic. Gaebler et al. "
      "(2024) rank names by predicted demographic probability from a voter "
      "file and take the top pairs. Seshadri et al. (2025) bin names by their "
      "frequency in a pre-training corpus and match within bins, so the "
      "principle this paper argues for is already accepted practice; "
      "tokenization is simply not the property anyone has picked. Wilson and "
      "Caliskan (2024) show the pick is consequential: changing the "
      "frequency-matching strategy alters whether Black names or White names "
      "are favoured in a majority of their cases.")
    P("<b>What is left.</b> An and Rudinger control for length in a design "
      "built to allow it. A correspondence audit cannot: its unit is the "
      "matched PAIR, and balancing the pair on tokens restricts which names "
      "may be paired with which. Whether that restriction leaves enough of "
      "the validated list to run an audit is an empirical question, and "
      "§9 answers it.")

    # ---- 3 --------------------------------------------------------------
    H("3  Design")
    P("<b>Models.</b> Four open-weight instruction-tuned checkpoints spanning "
      "a generational boundary, served locally with grammar-constrained "
      "decoding at temperature zero: "
      + ", ".join(SHORT[m] for m in models)
      + ". Each checkpoint’s SHA-256 digest is recorded and re-verified "
      "at analysis time.", indent=False)
    P("<b>Stimuli.</b> One job posting, "
      + NUM.get(len(_tpl), str(len(_tpl)))
      + " résumé templates "
      "spanning strong, middling and marginal qualification, and a name grid "
      "built factorially from the Bertrand and Mullainathan list: "
      f"{nlen[models[0]]['n_pairs']} matched pairs, crossing first names with "
      "surnames so that the first-name and surname contributions are "
      "separately estimable. Names are selected by a fixed mechanical rule, "
      "alphabetical order within each cell, because any rule depending on the "
      "names’ measured behaviour would introduce the degree of freedom "
      "this paper is about. Every pair is run under each template crossed "
      "with "
      + NUM.get(len(_wrd), str(len(_wrd)))
      + " instruction wordings, paraphrases and semantically null edits "
      "alike, so a per-pair effect here is the mean over "
      f"{_cells} prompt realisations: the wording and the template are "
      "averaged over rather than fixed at one setting.")
    P("<b>Outcome.</b> The model is constrained by grammar to emit exactly "
      "“yes” or “no” and we read the full top-token "
      "distribution rather than the sampled token. The outcome is the "
      "renormalised decision margin in log-odds, log P(yes) − log P(no).")
    P("<b>Token length.</b> For each pair we take the difference in token "
      "count between the two full names under that model’s own "
      "tokenizer. A pair is TOKEN-MATCHED when the difference is zero. The "
      "measure is per model, because tokenizers differ.")
    P("<b>Resampling unit.</b> The grid is "
      f"{nlen[models[0]]['n_pairs']} rows built from "
      f"{CL[models[0]]['n_grid_clusters']} first-name pairs crossed with "
      "surnames, so the rows are not independent: a first name appears in "
      "several of them. Every interval and every test in this paper resamples "
      "FIRST-NAME PAIRS, not rows. Resampling rows would treat "
      f"{CL[models[0]]['n_grid_clusters']} independent draws as "
      f"{nlen[models[0]]['n_pairs']}, and the difference is not cosmetic: it "
      "is the difference between the significance verdicts reported here and "
      "a more favourable set we do not claim.")

    # ---- 4 --------------------------------------------------------------
    H("4  How much of the standard list is token-matched")
    P("The first question is arithmetic on the list and the tokenizers, and "
      "needs no model calls. Across the "
      f"{nlen[models[0]]['n_pairs']}-pair grid, the fraction of pairs whose "
      "two names occupy the same number of tokens is "
      + ", ".join(f"{pct(_frac[m], 0)} on {TINY[m]}" for m in models)
      + ".", indent=False)
    paper.table(
        ["model", ">pairs", ">token-matched", ">share", ">mean |Δ|",
         ">first-name pairs"],
        [[TINY[m], str(nlen[m]["n_pairs"]), str(nlen[m]["n_same_length"]),
          pct(_frac[m], 0), fmt(_absd.get(m), 2),
          str(CL[m]["n_matched_clusters"])]
         for m in models],
        [120, 46, 74, 54, 52, 80], span2=True, size=7.8,
        caption=("Table 1. How much of a standard correspondence grid is "
                 "matched in tokens. The grid is identical across models; only "
                 "the tokenizer changes. The last column is the number of "
                 "distinct first-name pairs surviving the restriction, which "
                 "is the resampling unit and the reason the matched subset is "
                 "weaker than its row count suggests. |Δ| is the absolute "
                 "within-pair token-length difference, averaged over pairs; "
                 "Table 4 reports its signed mean, which the shift prediction "
                 "requires and which is the smaller number."))
    P("Most of the design is unmatched on every tokenizer. That is the "
      "paper’s least contestable claim and it does not depend on any "
      "model’s behaviour: an auditor who never runs a model can compute "
      "it from the list and the tokenizer in a few lines.")
    P("In Table 1 the first-name-pair column matters more than the "
      "token-matched count. Restricting to "
      "token-matched pairs does not merely shrink the sample; it collapses "
      "the number of independent first-name pairs, on one model to "
      f"{min(CL[m]['n_matched_clusters'] for m in models)}. Any statistic "
      "computed on that subset inherits that, and §6 is about what it "
      "inherits.")

    # ---- 5 --------------------------------------------------------------
    H("5  Does the token-length difference predict the effect?")
    P("If tokenization is a live channel rather than an accounting curiosity, "
      "pairs with a larger token-length difference should show a different "
      "effect. We regress the per-pair effect on the within-pair token-length "
      "difference, resampling first-name pairs.", indent=False)
    paper.table(
        ["model", ">slope", ">95 % interval", ">p"],
        [[TINY[m], fmt(CL[m]["slope_same_unit"]["est"], 4, sign=True),
          f"[{fmt(CL[m]['slope_same_unit']['ci'][0], 4, sign=True)}, "
          f"{fmt(CL[m]['slope_same_unit']['ci'][1], 4, sign=True)}]",
          f"{CL[m]['slope_same_unit']['p']:.3f}"] for m in models],
        [150, 82, 154, 60], span2=True, size=7.8,
        caption=("Table 2. The per-pair regression of the effect on the "
                 "within-pair token-length difference, resampling first-name "
                 "pairs. On the row-level resampling this paper rejects, "
                 + NUM.get(_slope_sig_row, str(_slope_sig_row))
                 + " of these slopes would be distinguishable from zero "
                 "rather than "
                 + NUM.get(len(_slope_sig), str(len(_slope_sig)))
                 + "; the unit is the difference."))
    P("The slope is distinguishable from zero on "
      + NUM.get(len(_slope_sig), str(len(_slope_sig)))
      + " of " + NUM.get(len(models), str(len(models))) + " checkpoints"
      + ((": " + ", ".join(TINY[m] for m in _slope_sig) + ".")
         if _slope_sig else ".")
      + " That is weak evidence for the channel and we do not present it as "
      "more. What it is not is evidence against: a slope estimated on "
      f"{CL[models[0]]['n_grid_clusters']} independent first-name pairs has "
      "wide intervals on every model, and the design was not built to "
      "estimate it.")

    # ---- 6 --------------------------------------------------------------
    H("6  What restricting to token-matched pairs does")
    P("The question an auditor cares about is not whether the slope is "
      "significant but whether the reported disparity would have been "
      "different had the design been balanced. We estimate the effect twice "
      "on each model: over all pairs, and over the token-matched subset "
      "alone.", indent=False)
    paper.table(
        ["model", ">all pairs", ">token-matched", ">difference", ">p"],
        [[TINY[m], fmt(CL[m]["effect_all_pairs"]["est"], 4, sign=True),
          fmt(CL[m]["effect_token_matched"]["est"], 4, sign=True),
          fmt(CL[m]["matched_minus_all"]["est"], 4, sign=True),
          f"{CL[m]['matched_minus_all']['p']:.3f}"] for m in models],
        [136, 80, 92, 80, 58], span2=True, size=7.8,
        caption=("Table 3. The same audit, restricted to pairs matched in "
                 "tokens. Effects are in log-odds; intervals and p-values "
                 "resample first-name pairs. The p column is the cluster "
                 "bootstrap alone and clears 0.05 on every model; §6.1 gives "
                 "two further procedures that disagree with it on the model "
                 "with the largest movement."))
    P("The restriction moves the disparity further from zero on "
      + NUM.get(len(_away), str(len(_away))) + " of "
      + NUM.get(len(models), str(len(models))) + " models"
      + ((f", by {pct(CL[_best]['growth_ratio']['est'], 0)} "
          f"[{pct(CL[_best]['growth_ratio']['ci'][0], 0)}, "
          f"{pct(CL[_best]['growth_ratio']['ci'][1], 0)}] on {TINY[_best]}")
         if _best else "")
      + ". The direction favours the confound MASKING a disparity rather than "
      "manufacturing one, which is the opposite of the usual worry about "
      "measurement artifacts. It is also the direction of "
      + NUM.get(len(_away), str(len(_away)))
      + " point estimates and no more than that: of the "
      + NUM.get(len(_interp), str(len(_interp)))
      + " models whose baseline all-pairs effect is distinguishable from zero, "
      "and which can therefore carry a ratio at all, the split is "
      + NUM.get(len(_interp_away), str(len(_interp_away)))
      + " away and "
      + NUM.get(len(_interp) - len(_interp_away),
                str(len(_interp) - len(_interp_away)))
      + " toward.")

    H("6.1  Three procedures, three answers", 2)
    _c = CL[_best] if _best else CL[models[0]]
    P("On the model with the largest movement the three standard ways of "
      "asking whether the restriction changed anything disagree, and the "
      "disagreement is the result rather than a nuisance to be resolved by "
      "choosing one.", indent=False)
    _nrej = sum(bool(_c.get(k)) for k in ("difference_significant_by_ci",
                                          "difference_significant",
                                          "difference_significant_by_relabel"))
    P("The percentile interval of the cluster bootstrap "
      + ("excludes" if _c.get("difference_significant_by_ci") else "contains")
      + " zero. The "
      f"same bootstrap shifted to the null gives p = {_c['matched_minus_all']['p']:.3f}. "
      "The exact test that relabels whole clusters, which is enumerable at "
      f"this size, gives p = {_c['cluster_relabel_test']['p']:.3f} over "
      f"{_c['cluster_relabel_test']['n_outcomes']} outcomes. "
      + NUM.get(_nrej, str(_nrej)) + " of the three reject at 0.05 and "
      + NUM.get(3 - _nrej, str(3 - _nrej))
      + (" does not." if 3 - _nrej == 1 else " do not."))
    P("The reason is visible in the design rather than in the arithmetic. The "
      f"matched subset on this model has {_c['n_matched_clusters']} "
      "first-name pairs in it. A bootstrap that resamples "
      f"{_c['n_matched_clusters']} clusters with replacement puts probability "
      f"{_c['same_cluster_draw_probability']:.2f} on drawing a single cluster "
      "every time, so its percentile interval is pinned to the extreme "
      "cluster means and is not a 95 % interval in the usual sense. "
      "Separately, "
      f"{_c['matched_minus_all']['n_draws_degenerate']:,} of "
      f"{_c['n_boot']:,} draws of the difference bootstrap contained no "
      "matched cluster at all and are undefined rather than zero.")
    P("<b>The reportable finding is that the subset cannot adjudicate the "
      "question it was introduced to settle.</b> Not that the effect grows by "
      + (pct(_c["growth_ratio"]["est"], 0) if _best else "some amount")
      + ", and not that it does not. An auditor who restricts to token-matched "
      "pairs on this list will be estimating from a handful of first names, "
      "and should say so.")

    # ---- 7 --------------------------------------------------------------
    H("7  Two designs, checked against each other")
    if lpred and lpred.get("models"):
        _lp = lpred["models"]
        _ls = lpred["summary"]
        P("The slope of §5 and the subset shift of §6 are fitted on the same "
          f"{nlen[models[0]]['n_pairs']} rows, and neither is used to compute "
          "the other, so the second can be checked against what the first "
          "predicts. If the effect "
          "depends on token length with slope b, then dropping the unmatched "
          "pairs should move it by about −b times the mean length "
          "difference over ALL pairs — all, not just the dropped ones, "
          "because the matched arm of that contrast contributes a difference "
          "of zero.", indent=False)
        paper.table(
            ["model", ">slope", ">mean Δ", ">predicted", ">observed",
             ">obs / pred"],
            [[TINY[m], fmt(_lp[m]["slope"], 4, sign=True),
              f"{_lp[m]['mean_delta_over_all']:.2f}",
              fmt(_lp[m]["predicted_shift"], 4, sign=True),
              fmt(_lp[m]["observed_shift"], 4, sign=True),
              f"{_lp[m]['ratio_observed_to_predicted']:.2f}"]
             for m in ORDER if m in _lp],
            [96, 56, 48, 62, 60, 56], span2=True, size=7.6,
            caption=("Table 4. The two token-length designs checked against "
                     "each other. Neither quantity enters the other’s "
                     "computation, so the comparison can fail. Δ here is "
                     "the SIGNED within-pair token-length difference, which "
                     "the prediction requires; Table 1 reports its absolute "
                     "mean."))
        P("The sign agrees on "
          f"{_ls['n_same_sign']} of {_ls['n_models']} models and the "
          "magnitude within a factor of five on "
          f"{_ls['n_within_a_factor_of_five']}, with the ratio running "
          f"{_ls['ratio_min']:.2f} to {_ls['ratio_max']:.2f}. The observed "
          "shift is the larger of the two on "
          + NUM.get(sum(1 for v in _lp.values()
                        if (v.get("ratio_observed_to_predicted") or 0) > 1),
                    "most")
          + f" of {_ls['n_models']}, so the slope under-predicts what dropping "
          "the unmatched pairs does. What this check is, precisely: because "
          "the matched arm contributes a length difference of zero and the "
          "regression residuals average to zero over all rows, the observed "
          "shift equals the predicted shift plus the mean residual over the "
          "matched rows. It is a decomposition of one fit rather than two "
          "independent estimates, and what it shows is that the matched-arm "
          "residual is small beside the length term. That is worth less than "
          "a designed test.")

    # ---- 8 --------------------------------------------------------------
    H("8  A manipulation that does not corroborate it")
    if _d89_total:
        P("A stronger form of corroboration would come from a design that "
          "moves the name’s POSITION without changing the name, since "
          "that varies the token context while holding the demographic "
          "content fixed. The parent study (Mao 2026) runs such conditions, "
          "displacing the name by one and by two tokens and changing nothing "
          "else. Its cells cross the four checkpoints with two inference "
          "modes, chat-templated and raw completion behind a byte-identical "
          "wrapper, and the two displacements.", indent=False)
        P("They do not corroborate the length channel, and no arrangement of "
          "the counts could. Both conditions add their extra tokens inside "
          "the job posting, identically in both arms of every pair, so the "
          "within-pair token-length difference §5 regresses on is unchanged, "
          "and the model of §7 predicts a shift of exactly zero in all "
          f"{_d89_total} model-by-mode-by-condition cells. The measured "
          "shifts are not zero, which is itself a sign that the instrument "
          "reads token position, but a manipulation with no predicted sign "
          "cannot agree or disagree with a slope. We report this because an "
          "earlier version of the parent paper counted these cells as "
          "corroboration, and because a reader entitled to the corroborating "
          "evidence is entitled to its absence.")
        P("The honest reading is the one §1 gave: this design cannot "
          "separate token length from the properties correlated with it, and "
          "a position manipulation on this panel does not separate them "
          "either.")

    # ---- 9 --------------------------------------------------------------
    H("9  A balanced grid, and why it is too small")
    if tbal:
        # THE BALANCED GRID IN CLUSTERS, NOT ROWS. §9 quoted its row count
        # inside a precision claim, which is the error this paper exists to
        # name and which §10's own recommendation 2 tells others to avoid.
        _bal_fn = (tbal["max_matching"]["female_first"]
                   + tbal["max_matching"]["male_first"])
        P("The obvious remedy is to build the grid balanced: pair names so "
          "that every pair is token-matched by construction. It can be done, "
          "from the same validated list, and the result is the reason this "
          "paper recommends reporting rather than rebuilding.", indent=False)
        P("Selecting by maximum-weight bipartite matching over the list, "
          "subject to matching within race and gender and to token equality "
          f"on all {NUM.get(len(models), len(models))} tokenizers "
          "simultaneously, yields a grid of "
          f"{tbal['n_pairs']} matched pairs"
          + (" that is fully balanced" if tbal.get("fully_balanced") else "")
          + ". That is a quarter of the "
          f"{nlen[models[0]]['n_pairs']}-pair grid the same list supports "
          "unbalanced. But the row count is the wrong unit, which is this "
          "paper's own point: those rows are "
          f"{_bal_fn} distinct first-name pairs crossed with "
          f"{tbal['max_matching']['surnames']} surname pairs, so on the "
          f"resampling unit of §3 the balanced grid is {_bal_fn} independent "
          f"draws and not {tbal['n_pairs']}. That is the size §6.1 dissects, "
          "and it is why the remedy here is reporting rather than rebuilding: "
          "crossing in more surnames adds rows and no clusters, because the "
          "binding constraint is how few first-name pairs of the validated "
          "list are token-balanced at all.")
        P("<b>An earlier statement of this was wrong in the useful "
          "direction.</b> A draft of the parent paper said such a list "
          "“does not exist”. It does; it is simply small. The "
          "distinction matters because “impossible” closes the "
          "question and “too small at this list size” names what "
          "would fix it, which is a larger validated pool than this "
          "literature currently draws from.")

    # ---- 10 -------------------------------------------------------------
    H("10  What an audit should report")
    P("Nothing here licenses discarding the standard list, and we are not "
      "proposing that an auditor rebuild their design. Three lines of "
      "reporting would let a reader judge the threat, and none of them costs "
      "a model call.", indent=False)
    P("<b>1. The token statistics of the pairs used.</b> The fraction of "
      "pairs that are token-matched under the tokenizer of each model "
      "audited, and the mean absolute token-length difference. This is "
      "computable from the name list and the tokenizer before any data is "
      "collected. One audit comes close: Tan et al. (2026) rule out a length "
      "preference by relating each résumé’s character count to its "
      "outcome. That check is on the whole document and in the units the "
      "auditor wrote in, which leaves the manipulated span uncounted.")
    P("<b>2. The effect on the token-matched subset, if it is large enough "
      "to estimate.</b> With the count of independent name pairs in that "
      "subset, not only the row count — §6.1 is what happens when "
      "the two are confused.")
    P("<b>3. The resampling unit.</b> Whether intervals resample rows or name "
      "pairs, because on this grid the two give different significance "
      "verdicts for the same data.")
    if matrix:
        _tm = matrix["counts"].get("token_matching", {})
        P("Of the "
          f"{_tm.get('n_applicable', matrix['n_llm_hiring_audits'])} LLM "
          "hiring audits we read in full text whose design manipulates a "
          f"name, {_tm.get('n_reported', 0)} report the first"
          + ((f" and {len(_tm.get('partial_by') or [])} ("
              + ", ".join(_tm["partial_by"])
              + ") mention it without specifying it")
             if _tm.get("partial_by") else "")
          + "; all of them are in the references below. The reporting cost is "
          "three numbers, and the current reporting rate is what it is "
          "because nobody has asked.")

    # ---- 11 -------------------------------------------------------------
    H("11  Threats to validity")
    P("<b>The channel is not identified.</b> Token length and name "
      "distinctiveness are correlated by construction, so no design that "
      "varies the name can attribute the effect to one of them. This paper "
      "measures an IMBALANCE and its consequences for a reported number; it "
      "does not claim that disparities are a tokenization artifact, and "
      "§8 reports the manipulation that would have supported a stronger "
      "reading and does not.", indent=False)
    P("<b>The matched subset is small.</b> On one model it rests on "
      f"{min(CL[m]['n_matched_clusters'] for m in models)} first-name pairs. "
      "Every statistic computed on it is correspondingly weak, which is the "
      "finding of §6.1 rather than a caveat attached to it.")
    P("<b>Four checkpoints, one list, one posting.</b> The token-matched "
      "fractions of §4 are a property of the list and the tokenizers and "
      "would transfer to any audit using both. They are also a property of a "
      "grid that varies first name and surname together: an audit drawing "
      "from a probability-thresholded voter-file list, or holding the surname "
      "fixed as Wilson and Caliskan (2024) do, would compute a different "
      "fraction. The behavioural results of "
      "§§5 and 6 are measured on four open-weight instruction-tuned "
      "models on a single job posting, and we do not claim they transfer.")
    P("<b>The remedy is untested.</b> We recommend reporting token statistics "
      "and have not shown that doing so changes any conclusion in the "
      "literature, because that would require re-running the surveyed audits "
      "rather than reading them.")

    # ---- 12 -------------------------------------------------------------
    H("12  Conclusion")
    P("A correspondence audit rests on the two conditions being identical "
      "apart from the manipulated attribute. On a language model they are "
      "identical in characters and not in tokens, and on the canonical "
      "Bertrand and Mullainathan list the mismatch affects most of the "
      "design. The consequences we can measure are real but modest and "
      "imprecisely estimated: the length difference predicts the effect on "
      + NUM.get(len(_slope_sig), str(len(_slope_sig)))
      + " checkpoint" + ("" if len(_slope_sig) == 1 else "s")
      + " of " + NUM.get(len(models), str(len(models)))
      + ", restricting to matched pairs moves the disparity further from zero "
      "on " + NUM.get(len(_away), str(len(_away)))
      + ", and the subset is too small to say by how much.", indent=False)
    P("The reason to report it anyway is not the size of the effect. It is "
      "that a design describing itself as matched is unmatched on a dimension "
      "the instrument is demonstrably sensitive to, that the imbalance is "
      "computable before any data is collected, and that no audit we "
      "surveyed computes it.")

    H("Data and code")
    P("The name grid, the per-pair token counts under every tokenizer, the "
      "per-model effects and every interval reported here are interpolated at "
      "typesetting time from the artifact set deposited with the parent "
      "study, Mao (2026), and are deposited with this preprint as well. The "
      "two papers read the same files and cannot disagree.", indent=False)

    H("References")
    # THE SURVEYED PANEL BELONGS HERE. §10's claim is a census of the audits
    # that manipulate a name, and not one of them was named or citable: a
    # reader could not check the count against anything. The entries come from
    # the survey artifact, so they cannot drift from the panel they describe.
    for r in sorted([
        "An, H. and Rudinger, R. (2023). Nichelle and Nancy: The Influence of "
        "Demographic Attributes and Tokenization Length on First Name Biases. "
        "Proceedings of ACL 2023 (Short Papers), 388–401.",
        "Bertrand, M. and Mullainathan, S. (2004). Are Emily and Greg More "
        "Employable than Lakisha and Jamal? A Field Experiment on Labor "
        "Market Discrimination. American Economic Review 94(4), 991–1013.",
        # Volume and pages are deliberately absent: the copy read is the
        # online-first version, which prints the journal and the DOI and no
        # pagination. Identifiers here are as printed on the document read.
        "Simonsohn, U., Simmons, J. P. and Nelson, L. D. (2020). "
        "Specification Curve Analysis. Nature Human Behaviour, "
        "doi:10.1038/s41562-020-0912-z.",
        "Mao, D. (2026). The Instrument Is Not the Model: How Much of an LLM "
        "Hiring Disparity Comes from Unreported Design Choices. Companion "
        "preprint; the model panel, the name grid and the artifact set from "
        "which this paper’s numbers are interpolated are shared with it.",
    ] + [s["reference"] for s in ((matrix or {}).get("studies") or [])
         if s["kind"] == "llm_hiring_audit" and s.get("reference")]):
        P(r, indent=False, size=7.8, lead=9.6, space_after=1.6)

    # THE ARXIV SUBMISSION FIELD, from the same string the PDF sets, using the
    # parent builder's transliteration table so the two papers cannot diverge
    # on what arXiv accepts. paper-c/releases/abstract_arxiv.txt was a hand
    # copy until now, which is exactly the drift this repository forbids
    # everywhere else.
    sys.path.insert(0, str(ROOT / "paper-a" / "src"))
    from build_paper_v3 import arxiv_abstract  # noqa: PLC0415
    _ax = arxiv_abstract(abstract)
    _axp = ROOT / "paper-c" / "releases" / "abstract_arxiv.txt"
    _axp.parent.mkdir(parents=True, exist_ok=True)
    _axp.write_text(_ax + "\n", encoding="utf-8", newline="\n")
    print(f"  arXiv abstract: {len(_ax)} chars -> "
          f"{_axp.relative_to(ROOT)}")

    paper.render()
    if MISSING:
        print(f"  [WARNING] missing artifacts: {', '.join(MISSING)}")
    print(f"wrote {OUT.relative_to(ROOT)}  ({paper.page} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
