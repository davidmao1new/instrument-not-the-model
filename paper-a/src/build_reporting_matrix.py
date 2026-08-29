"""Normalise the reporting-practice readings into one artifact the paper reads.

WHAT THIS IS. Section 8 argues that the field does not report the choices this
paper shows are consequential. Until now that argument rested on a count of
prompt wordings and on prose. A count of one variable is an anecdote; the claim
is about a PRACTICE, so it needs a matrix: every surveyed study by every
reporting dimension, each cell carrying the evidence that put it there.

HOW THE CELLS WERE PRODUCED, stated plainly because it bears on how much weight
they can carry. Each study was read end to end by an independent reader working
only from the full text in `lit/text/` and the PDF in `lit/`, and every cell
returned a verdict, a value and a verbatim quote or the exact search that
returned nothing. In the first sweep, every NEGATIVE verdict was then
re-checked by a second reader who re-extracted the PDF independently --
necessary, because several of these papers put load-bearing detail in figures
whose text is not in the PDF text layer, and a negative from a grep over a
lossy extraction would be a false claim about someone else's work. Two limits
on that, both visible in the raw files rather than asserted away: the second
reader worked from the first reader's evidence rather than blind, and the
studies added in the later sweep carry a single coding, so `negative_check` is
present for the first set and absent for the second.
The raw readings, including the disagreements, are retained under
`paper-a/data/reference/raw/`.

THE HEADLINE COUNTS THIS FILE COMPUTES are the ones the paper quotes, and they
are computed here rather than typed. Two rules make them defensible:

  * `not-applicable` is excluded from both numerator and denominator. A field
    experiment has no quantization; counting it as "did not report" would
    inflate the indictment.
  * `partial` is counted as NOT reported for the headline, and reported
    separately, because the claim being tested is whether a reader could
    reconstruct the choice -- and a partially specified choice cannot be.

    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/build_reporting_matrix.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = ROOT / "paper-a" / "data" / "reference"
RAW = REF / "raw"
OUT = REF / "reporting_practice_matrix.json"

# Short labels, and whether the study is an LLM audit of a hiring task. The
# headline "of the N studies" counts run over LLM HIRING AUDITS only: Bertrand
# and Mullainathan is a field experiment and Sclar et al. is a format-sensitivity
# study on task accuracy, so neither belongs in a count about hiring-audit
# reporting practice. Both are kept in the matrix because they are the two most
# informative comparisons in it.
# Matched on the leading author string only. An earlier version tested
# substring membership anywhere in the title, which classified
# "Wilson & Caliskan" as "An" because "Caliskan" contains "an" -- and every
# label in the printed table came out wrong while every count stayed plausible.
CLASS = [
    ("wilson", "Wilson & Caliskan 2024", "llm_hiring_audit"),
    ("an, ", "An et al. 2024", "llm_hiring_audit"),
    ("an ", "An et al. 2024", "llm_hiring_audit"),
    ("iso", "Iso et al. 2025", "llm_hiring_audit"),
    ("fu,", "Fu & Shi 2025", "llm_hiring_audit"),
    ("fu ", "Fu & Shi 2025", "llm_hiring_audit"),
    ("gao", "Gao et al. 2026", "llm_hiring_audit"),
    ("tan", "Tan et al. 2026", "llm_hiring_audit"),
    ("nghiem", "Nghiem et al. 2024", "llm_hiring_audit"),
    ("gaebler", "Gaebler et al. 2024", "llm_hiring_audit"),
    ("armstrong", "Armstrong et al. 2024", "llm_hiring_audit"),
    ("veldanda", "Veldanda et al. 2023", "llm_hiring_audit"),
    ("seshadri", "Seshadri et al. 2025", "llm_hiring_audit"),
    ("hoffstedde", "Hoffstedde et al. 2026", "llm_hiring_audit"),
    ("glazko", "Glazko et al. 2024", "llm_hiring_audit"),
    ("bertrand", "Bertrand & Mullainathan 2004", "field_experiment"),
    ("sclar", "Sclar et al. 2024", "format_sensitivity"),
]

# The dimensions the paper's own sections show to matter, in the order the paper
# introduces them, so the table reads as an index of the argument.
FIELD_ORDER = [
    ("prompt_published", "exact prompt published"),
    ("n_wordings_used", "wordings used"),
    ("n_wordings_separately_estimated", "wordings separately estimated"),
    ("dispersion_across_wordings", "dispersion across wordings"),
    ("null_edit_control", "null-edit control"),
    ("name_list_source", "name list source"),
    ("names_per_race", "names per race"),
    ("name_sensitivity_reported", "name-draw sensitivity"),
    ("token_matching", "token matching of the pair"),
    ("checkpoint_pinned", "checkpoint pinned"),
    ("quantization_reported", "quantization"),
    ("serving_stack", "serving stack"),
    ("concurrency_or_batching", "request batching"),
    ("cache_policy", "cache policy"),
    ("decoding_params", "decoding parameters"),
    ("n_repeats", "repeats / reproducibility"),
    ("resampling_unit", "resampling unit"),
    ("multiplicity_correction", "multiplicity correction"),
    ("reporting_scale", "reporting scale + operating point"),
    ("n_occupations", "occupations"),
    ("headline_effect", "headline effect"),
    ("code_or_data_released", "code or data released"),
]


# Canonical reference strings for the studies added in the second sweep, each
# transcribed from the title page of the PDF in lit/ and cross-checked against
# the arXiv stamp. Where a venue is not printed in the document it is not
# asserted here -- several of these are arXiv preprints whose conference
# attribution is only inferable from an acknowledgement, and inventing a venue
# in a bibliography is exactly the kind of unverified claim the project rule
# exists to stop.
REFERENCE = {
    # The six studies of the first sweep are re-stated here in the same author
    # convention as the rest. Their `citation` fields in published_effects.json
    # were transcribed verbatim from each source's own title page, which put
    # given names first, so the reference list was internally inconsistent and
    # therefore not alphabetised by anything a reader could see. The
    # bibliographic facts are unchanged; only the convention is.
    "An et al. 2024":
        "An, H., Acquaye, C., Wang, C. K., Li, Z. and Rudinger, R. (2024). Do "
        "Large Language Models Discriminate in Hiring Decisions on the Basis of "
        "Race, Ethnicity, and Gender? arXiv:2406.10486v1.",
    "Fu & Shi 2025":
        "Fu, D. and Shi, D. (2025). “You Are Rejected!”: An Empirical Study "
        "of Large Language Models Taking Hiring Evaluations. "
        "arXiv:2510.19167v2, 23 Oct 2025.",
    "Gao et al. 2026":
        "Gao, Z., Jiang, W. and Yan, Y. (2026). Can LLMs Hire Fairly? Racial "
        "Bias in Resume Screening. arXiv:2606.28978v1, 27 Jun 2026.",
    "Iso et al. 2025":
        "Iso, H., Pezeshkpour, P., Bhutani, N. and Hruschka, E. (2025). "
        "Evaluating Bias in LLMs for Job-Resume Matching: Gender, Race, and "
        "Education. arXiv:2503.19182v1, 24 Mar 2025.",
    "Tan et al. 2026":
        "Tan, B. C. Z., Khoo, S., Doan, B. N., Liu, Z., Chen, N. F. and Lee, "
        "R. K.-W. (2026). Small Changes, Big Impact: Demographic Bias in "
        "LLM-Based Hiring Through Subtle Sociocultural Markers in Anonymised "
        "Resumes. arXiv:2603.05189v2, 5 May 2026.",
    "Wilson & Caliskan 2024":
        "Wilson, K. and Caliskan, A. (2024). Gender, Race, and Intersectional "
        "Bias in Resume Screening via Language Model Retrieval. "
        "arXiv:2407.20371v2, 20 Aug 2024. Carries an AAAI copyright line; the "
        "proceedings title is not printed in the document.",
    # THE PROCEEDINGS VERSION, WHICH IS THE ONE ON DISK. The sweep ran twice
    # and found this study both as an arXiv preprint and as the EMNLP paper;
    # the dedup kept the better-resolved reading for counting, and this dict
    # was never updated from the preprint. Venue and pages are the running
    # head of lit/nghiem_etal_2024_emnlp_name_based_bias.pdf p.1.
    "Nghiem et al. 2024":
        "Nghiem, H., Prindle, J., Zhao, J. and Daumé III, H. (2024). “You "
        "Gotta be a Doctor, Lin”: An Investigation of Name-Based Bias of Large "
        "Language Models in Employment Recommendations. Proceedings of the "
        "2024 Conference on Empirical Methods in Natural Language Processing, "
        "7268–7287.",
    "Gaebler et al. 2024":
        "Gaebler, J. D., Goel, S., Huq, A. and Tambe, P. (2024). Auditing the "
        "Use of Language Models to Guide Hiring Decisions. arXiv:2404.03086v1.",
    "Armstrong et al. 2024":
        "Armstrong, L., Liu, A., MacNeil, S. and Metaxa, D. (2024). The Silicon "
        "Ceiling: Auditing GPT’s Race and Gender Biases in Hiring. "
        # v3 is what the PDF's own margin stamp prints; this entry said v2
        # while the same script's transcribed study_as_printed said v3.
        "arXiv:2405.04412v3.",
    "Veldanda et al. 2023":
        "Veldanda, A. K., Grob, F., Thakur, S., Pearce, H., Tan, B., Karri, R. "
        "and Garg, S. (2023). Are Emily and Greg Still More Employable than "
        "Lakisha and Jamal? Investigating Algorithmic Hiring Bias in the Era of "
        "ChatGPT. arXiv:2310.05135v1.",
    "Seshadri et al. 2025":
        "Seshadri, P., Chen, H., Singh, S. and Goldfarb-Tarrant, S. (2025). "
        "Small Changes, Large Consequences: Analyzing the Allocational Fairness "
        "of LLMs in Hiring Contexts. arXiv:2501.04316v2.",
    # THE ONE ENTRY A READER COULD NOT LOCATE. This carried no identifier and
    # no venue, which made it the single unlocatable row in a table whose whole
    # point is that other people's reporting should be checkable. The arXiv
    # stamp is on the document's own first page --
    # "arXiv:2606.18649v1 [cs.MA] 17 Jun 2026" -- so it was there to be read.
    "Hoffstedde et al. 2026":
        "Hoffstedde, S. A., Hirota, M., Nadayanur Sathis Kanna, A., Kotani, R., "
        "Kumar, U., Trovato, G. and Phan, X. T. (2026). Gender Bias in LLM "
        "Hiring Decisions: Evidence from a Japanese Context and Evaluation of "
        "Mitigation Strategies. arXiv:2606.18649v1, 17 June 2026.",
    "Glazko et al. 2024":
        "Glazko, K., Mohammed, Y., Kosa, B., Potluri, V. and Mankoff, J. "
        "(2024). Identifying and Improving Disability Bias in GPT-Based Resume "
        "Screening. Proceedings of the 2024 ACM Conference on Fairness, "
        "Accountability, and Transparency (FAccT ’24).",
}


def classify(study: str):
    s = study.strip().lower()
    for key, label, kind in CLASS:
        if s.startswith(key):
            return label, kind
    return study.split(",")[0].split("(")[0].strip()[:40], "llm_hiring_audit"


def load_rounds() -> list[dict]:
    rows = []
    for f in sorted(RAW.glob("reporting_matrix_agents_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for entry in d:
            row = entry.get("row") if isinstance(entry, dict) else None
            if not row:
                continue
            if row.get("included") is False:
                continue
            rows.append(dict(row=row,
                             negative_check=entry.get("negative_check"),
                             source_file=f.name))
    return rows


def main() -> int:
    raw = load_rounds()
    if not raw:
        print("no raw readings found under "
              f"{RAW.relative_to(ROOT)}", file=sys.stderr)
        return 1

    # The search sweep ran twice from different angles and both passes found
    # Nghiem et al., so the same paper arrives under two slightly different
    # study strings. Keep the reading with more resolved cells; a silent
    # duplicate would inflate every denominator by one.
    studies = []
    seen_labels = {}
    for r in raw:
        label, kind = classify(r["row"]["study"])
        cells = {}
        for f in r["row"].get("fields", []):
            cells[f["field"]] = dict(verdict=f["verdict"], value=f.get("value", ""),
                                     evidence=f.get("evidence", ""))
        entry = dict(
            label=label, kind=kind, study_as_printed=r["row"]["study"],
            citation_check=r["row"].get("citation_check", ""),
            reference=REFERENCE.get(label, ""),
            negative_check=r.get("negative_check", ""),
            source_file=r["source_file"], cells=cells)
        resolved = sum(1 for c in cells.values()
                       if c["verdict"] != "not-applicable")
        prev = seen_labels.get(label)
        if prev is None:
            seen_labels[label] = (resolved, entry)
            studies.append(entry)
        elif resolved > prev[0]:
            studies[studies.index(prev[1])] = entry
            seen_labels[label] = (resolved, entry)

    # THE IDENTIFIER IN THE HAND-TYPED REFERENCE MUST MATCH THE ONE THE READER
    # TRANSCRIBED OFF THE PDF. These are two independent records of the same
    # fact and they were free to disagree: Armstrong's said v2 in one and v3 in
    # the other for two rounds, and nothing in the build noticed. A reference
    # with no arXiv identifier is fine (it cites a published venue instead);
    # two that disagree is not.
    _ax = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", re.I)
    for s in studies:
        a = _ax.search(s.get("reference") or "")
        b = _ax.search(s.get("study_as_printed") or "")
        if a and b and a.group(0).replace(" ", "") != b.group(0).replace(" ", ""):
            raise SystemExit(
                f"{s['label']}: REFERENCE says {a.group(0)} but the document "
                f"read prints {b.group(0)}")

    studies.sort(key=lambda s: (s["kind"] != "llm_hiring_audit", s["label"]))
    audits = [s for s in studies if s["kind"] == "llm_hiring_audit"]

    # ---- headline counts, over LLM hiring audits only ---------------------
    counts = {}
    for field, pretty in FIELD_ORDER:
        applicable = [s for s in audits
                      if s["cells"].get(field, {}).get("verdict")
                      not in (None, "not-applicable")]
        rep = [s for s in applicable
               if s["cells"][field]["verdict"] == "reported"]
        par = [s for s in applicable
               if s["cells"][field]["verdict"] == "partial"]
        counts[field] = dict(
            pretty=pretty,
            n_applicable=len(applicable), n_reported=len(rep),
            n_partial=len(par),
            n_not_reported=len(applicable) - len(rep) - len(par),
            reported_by=[s["label"] for s in rep],
            partial_by=[s["label"] for s in par],
            not_reported_by=[s["label"] for s in applicable
                             if s["cells"][field]["verdict"] == "not-reported"],
        )

    # ---- HOW MANY ACTUALLY USED MORE THAN ONE WORDING --------------------
    # `n_reported` on this field counts studies whose wording count we could
    # DETERMINE, not studies whose count exceeds one. Eight studies have a
    # determinable count and four of those determined it to be exactly one, so
    # a sentence built on n_reported ("8 of 12 estimate the effect separately
    # under more than one wording") is false about eight of them. The number
    # the paper wants is derived from the VALUE, which begins with the count.
    # The parse is recorded per study so a reader can check it rather than
    # trust it, and anything unparseable is listed rather than dropped.
    import re as _re  # noqa: PLC0415
    multi, single, unparsed = [], [], []
    for s_ in audits:
        cell = s_["cells"].get("n_wordings_separately_estimated", {})
        if cell.get("verdict") in (None, "not-applicable"):
            continue
        m = _re.match(r"\s*(\d+)", str(cell.get("value", "")))
        if not m:
            unparsed.append(s_["label"])
            continue
        (multi if int(m.group(1)) > 1 else single).append(
            (s_["label"], int(m.group(1))))
    counts["n_wordings_separately_estimated"]["derived_more_than_one"] = dict(
        note="studies whose SEPARATELY-ESTIMATED wording count exceeds one, "
             "parsed from the leading integer of the evidence cell; "
             "n_reported above counts determinable counts, not counts > 1",
        n_more_than_one=len(multi), n_exactly_one=len(single),
        n_unparseable=len(unparsed),
        more_than_one=[f"{a} ({b})" for a, b in sorted(multi)],
        exactly_one=[f"{a} ({b})" for a, b in sorted(single)],
        unparseable=unparsed)

    # the fields this paper shows to move the number, and nobody fully reports
    never = [f for f, c in counts.items()
             if c["n_applicable"] > 0 and c["n_reported"] == 0]
    never_even_partial = [f for f, c in counts.items()
                          if c["n_applicable"] > 0 and c["n_reported"] == 0
                          and c["n_partial"] == 0]

    out = dict(
        _how_produced=(
            "Each study read end to end from lit/text/ and lit/; every negative "
            "verdict independently re-checked against a fresh PDF extraction, "
            "including rendering figure pages whose text is not in the text "
            "layer. Raw readings retained in paper-a/data/reference/raw/."),
        _counting_rules=(
            "'not-applicable' is excluded from numerator and denominator. "
            "'partial' counts as NOT reported in the headline, because the "
            "question is whether a reader could reconstruct the choice."),
        field_order=[list(x) for x in FIELD_ORDER],
        studies=studies,
        n_studies=len(studies),
        n_llm_hiring_audits=len(audits),
        llm_hiring_audits=[s["label"] for s in audits],
        counts=counts,
        never_reported_by_any=never,
        never_reported_even_partially=never_even_partial,
    )
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- print ------------------------------------------------------------
    print("=" * 108)
    print(f"REPORTING PRACTICE, {len(audits)} LLM HIRING AUDITS "
          f"(+{len(studies) - len(audits)} reference studies)")
    print("=" * 108)
    w = max(len(p) for _, p in FIELD_ORDER) + 2
    hdr = " " * w + "".join(f"{s['label'].split()[0][:9]:>10}" for s in studies)
    print(hdr)
    sym = {"reported": "  yes", "partial": "  part", "not-reported": "   NO",
           "not-applicable": "    -"}
    for field, pretty in FIELD_ORDER:
        line = f"{pretty:<{w}}"
        for s in studies:
            v = s["cells"].get(field, {}).get("verdict")
            line += f"{sym.get(v, '    ?'):>10}"
        print(line)

    print("\nNOT REPORTED BY A SINGLE LLM HIRING AUDIT (of those to which it applies):")
    for f in never:
        c = counts[f]
        extra = f", {c['n_partial']} partial" if c["n_partial"] else ""
        print(f"  - {c['pretty']:<40} 0 of {c['n_applicable']}{extra}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
