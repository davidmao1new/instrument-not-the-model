"""Count the corpus by RECORD KIND, so the paper stops calling every row a pair.

WHY THIS EXISTS. `corpus_size()` in build_paper_v3.py counts every non-quarantined
JSONL row and the paper interpolates that one number into the sentence "N
matched-pair measurements ... across all studies". Most rows are matched pairs,
but three studies do not produce pairs at all: panel_gate, prestige and the smoke
tests each score a SINGLE prompt and carry one outcome. Calling those matched
pairs overstates the paired corpus by 1,461 records, and the fix is to publish
the two subtotals separately.

AND ONCE, IN THE OTHER DIRECTION. The first version of the predicate asked for
a margin on each arm, which was the right question until a study arrived whose
API answers it for nobody. The 220 Gemini rows in `frontier` carry a thresholded
verdict per arm and two HTTP responses -- two calls, one pair -- and the margin
test scored each as one single prompt costing one call. So the corpus was short
220 pairs AND 220 calls, and the sentence attributing every single-prompt record
to the admission gate, the affiliation probe and the smoke tests was false by
the same 220. Fixed by asking what the row actually records.

HOW A ROW IS CLASSIFIED. By its keys, not by its directory. A row is a MATCHED
PAIR if it records an outcome for both arms -- `white_margin`/`black_margin`, or
`white_raw`/`black_raw` where there is no margin to be had. That is the same
predicate `corpus_size()` uses to decide a row cost two model calls, so the two
numbers cannot drift apart. Everything else is SINGLE PROMPT. Directory names
are recorded in the breakdown but never used to decide, because the next study
added to a folder need not have the same shape as the last one.

The stricter form -- both margins present AND non-null -- is evaluated too, and
its disagreements are split into the two kinds that look alike and are not: a
margin CENSORED by the log-probability window, which is a property of this
instrument, and a margin ABSENT because the endpoint returns none, which is a
property of somebody's product. Reporting them as one number would claim a
censoring rate the instrument does not have.

That the corrected single-prompt total is now exactly panel_gate + prestige +
smoke -- the three studies named in EXPECTED_SINGLE_PROMPT, which is written
down independently of the count -- is the check that the widening went far
enough and no further.

TWO INCLUSION RULES, INHERITED RATHER THAN INVENTED.

  QUARANTINE. The same four directory names build_paper_v3.py and
  audit_consistency.py exclude: `_contaminated`, `_superseded`,
  `_binary_only_superseded`, `_d9_superseded`. Rows under them are counted and
  reported under `quarantined_excluded` so the exclusion is visible, and are in
  none of the headline totals.

  THE D9 RECHECK IS INCLUDED. `mechanism_panel/mech_d9recheck_*.jsonl` re-measures
  the D8 and D9 base-arm cells. analyze_mech_panel.py drops those files, because
  for the panel CONTRASTS they would silently overwrite the stored cells --
  adjudicate_d9.py returned VINDICATED, so the stored rows stand and the recheck
  is an independent cross-session reproduction, not a replacement. A corpus count
  is a different question: those 1,152 pairs are measurements that were really
  taken, and `corpus_size()` has always counted them. They stay in the headline,
  and the totals net of them are published beside it so a sentence that wants the
  count of DISTINCT design cells has a key to point at.

    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/analyze_corpus_size.py
"""
from __future__ import annotations

import json
import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "paper-a" / "data"
OUT = D / "reference" / "corpus_size.json"

# Verbatim from build_paper_v3.corpus_size() and audit_consistency.py. Duplicated
# rather than imported: build_paper_v3.py is the paper builder and importing it
# would run its module-level setup, and this script must be runnable when the
# builder is mid-edit.
QUAR = ("_contaminated", "_superseded", "_binary_only_superseded",
        "_d9_superseded")

# Studies whose rows score one prompt rather than a matched pair. Not used to
# classify anything -- classification is by keys -- but stated here so that the
# printed report can flag a directory whose composition changed.
EXPECTED_SINGLE_PROMPT = ("panel_gate", "prestige", "smoke")

# Studies whose PAIR rows may legitimately carry a null margin. The top-100
# logprob window censors a cell when the model is so confident that the losing
# option falls outside it; the row is still a matched pair and still cost two
# calls, and the analysis drops it explicitly and reports the rate. Before the
# second-task study no study had any, so "no pair row has a null margin" was a
# true invariant and worth asserting. It is no longer, and widening it to
# "no UNEXPECTED study has one" keeps the check sharp instead of deleting it.
#
# `frontier` joined the list for a DIFFERENT reason and the difference matters.
# In the second task a margin is censored because the top-100 window occasionally
# fails to resolve a confident cell. On the OpenAI arm the window is twenty
# tokens, not a hundred, and on one checkpoint the model concentrates so much
# mass on one answer that the other is absent from the window on 431 of 432
# cells -- saturation, not sampling. analyze_frontier_margin.py separates the
# two mechanisms and reports both; the count it arrives at independently is the
# same number this file arrives at by walking the raw rows, which is the point
# of keeping the check rather than deleting it.
EXPECTED_CENSORING = ("second_task", "frontier")

# Studies with PAIRED rows that carry no margin at all, because the API they
# came from does not return one to anybody. The Gemini arm of `frontier`
# records the thresholded verdict per candidate -- see probe_vendor_logprobs.py
# for the measurement establishing that no Gemini model reachable to us returns
# log probabilities. The OpenAI arm of the same study DOES carry margins.
#
# This is now a POSITIVE expectation rather than an exemption: these rows are
# counted as the matched pairs they are, and the check below asserts that no
# OTHER study has any, so an endpoint quietly dropping its margins would show
# up here instead of shrinking the corpus in silence.
VERDICT_ONLY = ("frontier",)

RECHECK_PREFIX = "mech_d9recheck_"


def is_matched_pair(row: dict) -> bool:
    """The predicate corpus_size() uses to charge a row two model calls.

    A ROW IS PAIRED WHEN BOTH ARMS WERE MEASURED, NOT WHEN BOTH ARMS WERE
    MEASURED ON A PARTICULAR SCALE. Keying on the margin alone was right until
    a study arrived whose API returns no margin: the 220 Gemini rows in
    `frontier` carry a verdict for each arm and two HTTP responses, so they are
    matched pairs that cost two calls, and the margin test scored them as one
    single prompt costing one call. Falling back to the raw per-arm response
    catches them. It stops there deliberately -- a bare `white`/`black` pair is
    not enough, because the outcome probe in `smoke` uses those two keys to
    hold the two NAMES in a forced choice, and widening the predicate that far
    would score three single prompts as pairs.
    """
    return (("white_margin" in row and "black_margin" in row)
            or ("white_raw" in row and "black_raw" in row))


def is_matched_pair_strict(row: dict) -> bool:
    """As above, but a present-and-null margin does not count as measured."""
    return (row.get("white_margin") is not None
            and row.get("black_margin") is not None)


def has_no_margin_field(row: dict) -> bool:
    """A paired row from an API that returns no distribution at all.

    Distinct from a censored margin, and the report keeps them apart. A
    CENSORED margin is a key that is present and null: the model answered, the
    log-probability window was asked for, and the losing option fell outside
    it. An ABSENT margin is a key that was never there: the Gemini endpoint
    does not return log probabilities to any caller, so the row records the
    thresholded verdict and nothing else. Rolling both into one count would
    read as a much higher censoring rate than the instrument actually has.
    """
    return "white_margin" not in row and "black_margin" not in row


def blank_tally() -> dict:
    return dict(n_records=0, n_matched_pair_records=0,
                n_single_prompt_records=0, n_model_calls=0)


def tally(dst: dict, row: dict) -> None:
    pair = is_matched_pair(row)
    dst["n_records"] += 1
    dst["n_matched_pair_records" if pair else "n_single_prompt_records"] += 1
    dst["n_model_calls"] += 2 if pair else 1


def main() -> int:
    total = blank_tally()
    recheck = blank_tally()
    quarantined = blank_tally()
    by_study: dict[str, dict] = {}
    by_file: list[dict] = []
    n_strict_disagreements = 0
    n_no_margin_field = 0
    n_rows_with_error = 0

    for f in sorted(D.rglob("*.jsonl")):
        rel = f.relative_to(D).as_posix()
        is_quar = any(q in f.parts for q in QUAR)
        study = f.relative_to(D).parts[0]
        this = blank_tally()
        this_null = 0
        this_absent = 0
        for row in st.read_jsonl(f):
            tally(this, row)
            if is_quar:
                continue
            if is_matched_pair(row) != is_matched_pair_strict(row):
                if has_no_margin_field(row):
                    n_no_margin_field += 1
                    this_absent += 1
                else:
                    n_strict_disagreements += 1
                    this_null += 1
            if row.get("error"):
                n_rows_with_error += 1

        if is_quar:
            for k, v in this.items():
                quarantined[k] += v
            by_file.append(dict(path=rel, study=study, quarantined=True, **this))
            continue

        for k, v in this.items():
            total[k] += v
        if f.name.startswith(RECHECK_PREFIX):
            for k, v in this.items():
                recheck[k] += v
        s = by_study.setdefault(
            study,
            blank_tally() | dict(n_files=0, n_pair_rows_with_a_null_margin=0,
                                 n_pair_rows_with_no_margin_field=0))
        for k, v in this.items():
            s[k] += v
        s["n_files"] += 1
        # FOLD THE PER-FILE COUNT IN. This line was missing, and its absence
        # made the reconciliation check below VACUOUS: every study record got
        # the key from a setdefault(0) afterwards, so "no study outside
        # EXPECTED_CENSORING has a null margin" compared 0 against 0 for all of
        # them and could never fail. Caught by re-reading the check rather than
        # by the check firing, which is the only way a vacuous test is ever
        # caught.
        s["n_pair_rows_with_a_null_margin"] += this_null
        s["n_pair_rows_with_no_margin_field"] += this_absent
        by_file.append(dict(path=rel, study=study, quarantined=False,
                            d9_recheck=f.name.startswith(RECHECK_PREFIX), **this))

    for study, s in by_study.items():
        s["kind"] = ("single_prompt" if s["n_matched_pair_records"] == 0
                     else "matched_pair" if s["n_single_prompt_records"] == 0
                     else "mixed")

    net = {k: total[k] - recheck[k] for k in total}

    out = {
        "_what": (
            "Every non-quarantined JSONL row under paper-a/data, classified by "
            "its keys as a matched-pair measurement or a single-prompt "
            "measurement. Produced by paper-a/src/analyze_corpus_size.py."),
        "_classification_rule": (
            "matched pair iff the row records an outcome for BOTH arms: either "
            "white_margin and black_margin, or -- where the API returns no "
            "distribution -- white_raw and black_raw. This is the same "
            "predicate build_paper_v3.corpus_size() uses to charge the row two "
            "model calls. All other rows are single-prompt and cost one call. "
            "The raw fallback exists for the 220 Gemini rows in `frontier`, "
            "which are matched pairs carrying a thresholded verdict per arm; "
            "keying on the margin alone scored them as single prompts and "
            "undercounted both the paired corpus and the call total."),
        "_quarantine_rule": (
            "rows under " + ", ".join(QUAR) + " are excluded from every "
            "headline total, and counted under quarantined_excluded."),
        "_d9_recheck_rule": (
            "mechanism_panel/mech_d9recheck_*.jsonl IS included in the headline "
            "totals, as it always was in corpus_size(). analyze_mech_panel.py "
            "excludes it from the panel CONTRASTS because adjudicate_d9.py "
            "returned VINDICATED, making the recheck an independent "
            "cross-session reproduction rather than a replacement for the "
            "stored cells. Those rows are real measurements, so they count "
            "toward corpus size; the totals net of them are published under "
            "the *_excluding_d9_recheck keys for any sentence that means "
            "distinct design cells."),
        "_superseded_note": (
            "n_matched_pair_records_superseded is the number the abstract "
            "previously reported as 'matched-pair measurements across all "
            "studies'. It is the count of ALL non-quarantined rows, which "
            "includes the 1,461 single-prompt rows in panel_gate, prestige and "
            "smoke. Retained under its own key because nothing is deleted and "
            "the paper may want to report the corpus both ways; the sentence "
            "about matched pairs should use n_matched_pair_records."),
        "_n_studies_note": (
            "There are 10 data directories but the paper says nine studies: "
            "`smoke` holds the pilot and instrument smoke tests, not a study. "
            "n_studies excludes it; n_data_directories does not. Wire prose "
            "that says 'nine studies' to n_studies."),

        # ---- the numbers the paper should interpolate ---------------------
        "n_matched_pair_records": total["n_matched_pair_records"],
        "n_single_prompt_records": total["n_single_prompt_records"],
        "n_model_calls": total["n_model_calls"],
        "n_all_records": total["n_records"],
        "n_matched_pair_records_superseded": total["n_records"],

        "n_matched_pair_records_excluding_d9_recheck":
            net["n_matched_pair_records"],
        "n_model_calls_excluding_d9_recheck": net["n_model_calls"],
        "n_d9_recheck_records": recheck["n_records"],
        "n_d9_recheck_model_calls": recheck["n_model_calls"],

        "n_data_directories": len(by_study),
        "n_studies": len(by_study) - (1 if "smoke" in by_study else 0),
        "n_studies_matched_pair": sum(1 for s in by_study.values()
                                      if s["kind"] == "matched_pair"),
        "n_studies_single_prompt": sum(1 for s in by_study.values()
                                       if s["kind"] == "single_prompt"),
        "n_files": sum(1 for r in by_file if not r["quarantined"]),
        "n_rows_with_error": n_rows_with_error,
        "n_pair_rows_with_a_null_margin": n_strict_disagreements,
        "n_pair_rows_with_no_margin_field": n_no_margin_field,
        "_null_vs_absent_note": (
            "n_pair_rows_with_a_null_margin counts pairs whose margin was "
            "REQUESTED and CENSORED -- the key is there and null because the "
            "losing option fell outside the log-probability window. "
            "n_pair_rows_with_no_margin_field counts pairs from an endpoint "
            "that returns no distribution to anyone, so the key was never "
            "written. Only the first is a property of this instrument; "
            "combining them would report a censoring rate the instrument does "
            "not have."),

        "by_study": by_study,
        "by_file": by_file,
        "quarantined_excluded": quarantined,
    }

    # ---- reconciliation, written to the artifact so the paper can cite it --
    checks = {
        "pair_plus_single_equals_old_total": (
            total["n_matched_pair_records"] + total["n_single_prompt_records"]
            == total["n_records"]),
        "calls_equal_two_per_pair_plus_one_per_single": (
            2 * total["n_matched_pair_records"] + total["n_single_prompt_records"]
            == total["n_model_calls"]),
        "study_breakdown_sums_to_total": (
            sum(s["n_records"] for s in by_study.values()) == total["n_records"]
            and sum(s["n_model_calls"] for s in by_study.values())
            == total["n_model_calls"]),
        "null_margins_confined_to_studies_that_expect_them": all(
            s.get("n_pair_rows_with_a_null_margin", 0) == 0
            for k, s in by_study.items() if k not in EXPECTED_CENSORING),
        # THE EXEMPTION IS GONE, WHICH IS THE POINT. `frontier` used to be
        # listed here so the check would pass while 220 of its paired rows were
        # miscounted as single prompts -- an exemption that hid the defect it
        # was added to accommodate. With the predicate fixed the study has no
        # single-prompt rows at all, so the check can go back to naming only
        # the three studies that genuinely produce them.
        "single_prompt_rows_are_confined_to_the_expected_studies": all(
            s["n_single_prompt_records"] == 0
            for k, s in by_study.items() if k not in EXPECTED_SINGLE_PROMPT),
        "single_prompt_total_equals_the_expected_studies": (
            total["n_single_prompt_records"]
            == sum(by_study[k]["n_single_prompt_records"]
                   for k in EXPECTED_SINGLE_PROMPT if k in by_study)),
        "absent_margins_confined_to_studies_that_expect_them": all(
            s.get("n_pair_rows_with_no_margin_field", 0) == 0
            for k, s in by_study.items() if k not in VERDICT_ONLY),
    }
    out["reconciliation"] = checks

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- print ------------------------------------------------------------
    print("=" * 84)
    print("CORPUS SIZE BY RECORD KIND")
    print("=" * 84)
    print(f"{'study':<22}{'kind':>14}{'rows':>8}{'pairs':>8}{'single':>8}"
          f"{'calls':>9}{'files':>7}")
    for study in sorted(by_study):
        s = by_study[study]
        print(f"{study:<22}{s['kind']:>14}{s['n_records']:>8}"
              f"{s['n_matched_pair_records']:>8}"
              f"{s['n_single_prompt_records']:>8}{s['n_model_calls']:>9}"
              f"{s['n_files']:>7}")
    print("-" * 84)
    print(f"{'TOTAL':<22}{'':>14}{total['n_records']:>8}"
          f"{total['n_matched_pair_records']:>8}"
          f"{total['n_single_prompt_records']:>8}{total['n_model_calls']:>9}"
          f"{out['n_files']:>7}")
    print(f"\n  matched-pair records          {total['n_matched_pair_records']:,}"
          f"   <- what the abstract means")
    print(f"  single-prompt records         {total['n_single_prompt_records']:,}"
          f"    (panel_gate, prestige, smoke)")
    print(f"  all rows (superseded figure)  {total['n_records']:,}")
    print(f"  model calls                   {total['n_model_calls']:,}"
          f"   ({n_no_margin_field} verdict-only pairs cost two apiece)")
    print(f"\n  net of the D9 recheck: {net['n_matched_pair_records']:,} pairs "
          f"from {net['n_model_calls']:,} calls "
          f"({recheck['n_records']:,} recheck rows excluded)")
    if quarantined["n_records"]:
        print(f"  excluded by quarantine: {quarantined['n_records']:,} rows, "
              f"{quarantined['n_model_calls']:,} calls")

    print("\nRECONCILIATION")
    for name, passed in checks.items():
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
