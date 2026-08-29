"""Tests for the companion paper's first audit.

Thirty-three findings survived refutation on four pages, which is roughly what
seven rounds on the parent paper would predict for prose that had never been
audited. Three of them are worth stating plainly, because they are the kinds a
second reader catches and an author does not.

  The framing claim was never checked against the artifact.  Every version of
  this paper said the LLM-audit literature "standardised on" the Bertrand and
  Mullainathan name list -- in the abstract, the introduction, the background
  and the conclusion. The survey artifact it cites for every other claim about
  that literature says two of ten. Load-bearing framing gets written first,
  before the artifact exists, and is the last thing anyone re-reads.

  The paper committed its own central error in its remedy section.  §9 quoted
  the balanced grid's ROW count inside a claim about interval width, while §10
  recommendation 2 tells other people to report the cluster count and not only
  the row count. Twelve rows, three first-name pairs.

  A negative result was reported with a statistic that could not exist.  The
  two displacement conditions add their tokens inside the posting, identically
  in both arms, so the regressor is unchanged and the model predicts a shift of
  exactly zero. Counting how often the observed shift "agreed in sign" with the
  slope was arithmetic on a comparison with no sign to agree with.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-c" / "src" / "build_paper_c.py"
DATA = ROOT / "paper-a" / "data"
PDF = ROOT / "paper-c" / "figures" / "paper_token_matching.pdf"
PDF_A = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


def _text(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    fitz = pytest.importorskip("fitz")
    with fitz.open(p) as doc:
        return " ".join(" ".join(x.get_text().split()) for x in doc)


def _paper():
    return _text(PDF)


def _matrix():
    return _load(DATA / "reference" / "reporting_practice_matrix.json")


def _tbal():
    return _load(DATA / "instrument" / "token_balanced_grid.json")


# ------------------------------------------------------- the framing claim ----
def test_the_paper_does_not_claim_the_field_standardised_on_this_list():
    """Two of ten, against a claim of "most"."""
    mx = _matrix()
    applicable = [s for s in mx["studies"]
                  if s["kind"] == "llm_hiring_audit"
                  and s["cells"]["name_list_source"]["verdict"] != "not-applicable"]
    bm = [s for s in applicable
          if "Bertrand" in str(s["cells"]["name_list_source"]["value"])]
    assert len(bm) * 2 < len(applicable), (
        "a majority of surveyed audits now use the B&M list; the original "
        "framing would be defensible again and this test should be revisited")
    t = _paper()
    assert "standardised on" not in t
    assert "the one most subsequent work draws from" not in t
    assert f"{len(applicable)} audits in our survey that use a name list" in t
    assert f"{len(bm)} draw on it" in t


def test_the_canonical_framing_is_used_consistently():
    """The abstract, §1, §2 and §12 all carried the same wrong claim."""
    t = _paper()
    assert t.count("canonical Bertrand and Mullainathan list") >= 2
    assert "canonical instance is Bertrand and Mullainathan" in t


# ------------------------------------------------------- the balanced grid ----
def test_the_balanced_grid_is_reported_in_clusters_not_only_rows():
    """§10 tells others to do exactly this; §9 did not."""
    tb = _tbal()
    fn = tb["max_matching"]["female_first"] + tb["max_matching"]["male_first"]
    sn = tb["max_matching"]["surnames"]
    assert fn * sn == tb["n_pairs"], (fn, sn, tb["n_pairs"])
    t = _paper()
    assert f"{fn} distinct first-name pairs crossed with {sn} surname pairs" in t
    assert f"{fn} independent draws and not {tb['n_pairs']}" in t
    # the cross-scale comparison that had no conversion behind it
    assert "are wider than the effects this literature reports" not in t


def test_the_abstract_gives_the_balanced_grid_its_cluster_count():
    tb = _tbal()
    fn = tb["max_matching"]["female_first"] + tb["max_matching"]["male_first"]
    t = _paper()
    assert f"reaches {tb['n_pairs']} rows built from {fn} independent name pairs" in t


def test_the_balanced_grid_is_as_small_as_the_subset_section_6_1_dissects():
    """If it ever stops being, §9's argument changes and so should the prose."""
    tb = _tbal()
    fn = tb["max_matching"]["female_first"] + tb["max_matching"]["male_first"]
    nl = _load(DATA / "instrument" / "name_length_effect.json")
    worst = min(v["token_matched_first_name_clustered"]["n_matched_clusters"]
                for k, v in nl.items()
                if not k.startswith("_") and isinstance(v, dict))
    assert fn <= worst + 1, (
        f"the balanced grid now has {fn} clusters against a worst matched "
        f"subset of {worst}; §9's 'that is the size §6.1 dissects' needs "
        f"rechecking")


# ---------------------------------------------------------- the token counts ----
def test_the_token_counts_in_the_introduction_come_from_the_artifact():
    """"Allison may be one token" was neither measure on any tokenizer."""
    tb = _tbal()
    pm = tb["per_model_tokens"]
    anne = {v["first"]["Anne"]["iso"] for v in pm.values()}
    lak = {v["first"]["Lakisha"]["iso"] for v in pm.values()}
    assert len(anne) == 1 and len(lak) == 1, (
        "the tokenizers no longer agree; the sentence says they do")
    t = _paper()
    assert f"“Anne” is {anne.pop()} token on every tokenizer here" in t
    assert f"“Lakisha” is {lak.pop()}" in t
    assert "“Allison” may be one token" not in t


def test_allison_is_not_one_token_on_any_tokenizer():
    """The claim the paper used to make, checked directly."""
    pm = _tbal()["per_model_tokens"]
    for m, v in pm.items():
        assert v["first"]["Allison"]["iso"] != 1, m


# ------------------------------------------------- the displacement conditions ----
def test_the_displacement_conditions_are_not_scored_for_agreement():
    """Both add tokens inside the posting, identically in both arms, so the
    within-pair difference is unchanged and the predicted shift is zero."""
    src = SRC.read_text(encoding="utf-8")
    assert "_d89_agree" not in src, (
        "the sign tally is back; it counts agreement with a prediction that "
        "has no sign")
    t = _paper()
    assert "no arrangement of the counts could" in t
    assert "predicts a shift of exactly zero" in t
    assert "opposes on" not in t


# ------------------------------------------------------ the parent, named ----
def test_the_parent_study_is_named_before_it_is_leant_on():
    t = _paper()
    assert "parent study below" in t
    assert "Mao (2026)" in t or "Mao, D. (2026)" in t
    # the reference entry exists
    assert "The Instrument Is Not the Model" in t
    # and it is introduced in §1, before §8 uses it
    i_intro = t.find("parent study below")
    i_use = t.find("The parent study (Mao 2026) runs such conditions")
    assert i_intro > 0
    assert i_use == -1 or i_intro < i_use


def test_every_surveyed_audit_is_citable():
    """§10's claim is a census; not one of its members was named."""
    mx = _matrix()
    t = _paper()
    audits = [s for s in mx["studies"] if s["kind"] == "llm_hiring_audit"]
    assert len(audits) >= 10
    for s in audits:
        first = s["reference"].split(",")[0]
        assert first in t, f"{s['label']} is counted and not cited"


# -------------------------------------------------------------- denominators ----
def test_the_token_matching_denominator_excludes_the_inapplicable_study():
    mx = _matrix()
    tm = mx["counts"]["token_matching"]
    assert tm["n_applicable"] < mx["n_llm_hiring_audits"], (
        "every audit now manipulates a name; the two denominators coincide "
        "and this test no longer distinguishes them")
    t = _paper()
    assert f"of {tm['n_applicable']} LLM hiring audits read in full text" in t
    assert f"of {mx['n_llm_hiring_audits']} LLM hiring audits read in" not in t
    # and it matches what the parent paper prints for the same cell
    a = _text(PDF_A)
    assert f"0 of {tm['n_applicable']}" in a or f"{tm['n_applicable']}" in a


# ------------------------------------------------------- the movement count ----
def test_the_three_of_four_count_carries_its_qualifier():
    """The restriction moves the estimate on all four; three is the count of
    models where it moves FURTHER FROM ZERO."""
    nl = _load(DATA / "instrument" / "name_length_effect.json")
    live = {k: v["token_matched_first_name_clustered"] for k, v in nl.items()
            if not k.startswith("_") and isinstance(v, dict)}
    moved = sum(1 for v in live.values()
                if abs(v["matched_minus_all"]["est"]) > 1e-9)
    away = sum(1 for v in live.values()
               if abs(v["effect_token_matched"]["est"])
               > abs(v["effect_all_pairs"]["est"]))
    assert moved > away, (
        "the restriction now moves the estimate on exactly the models it "
        "moves further from zero; the qualifier is no longer load-bearing")
    t = _paper()
    assert t.count("further from zero") >= 3, (
        "the abstract, §6 and the conclusion should all carry it")


# ------------------------------------------------------- the two-design check ----
def test_the_cross_check_is_described_as_a_decomposition():
    """Both statistics are fitted on all 48 rows, and the observed shift is
    the predicted shift plus the mean residual over the matched rows."""
    nl = _load(DATA / "instrument" / "name_length_effect.json")
    for k, v in nl.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        cl = v["token_matched_first_name_clustered"]
        assert cl["slope_same_unit"]["n"] == cl["effect_all_pairs"]["n"], k
    t = _paper()
    assert "estimated on different parts of the grid" not in t
    assert "Two estimators built on different subsets" not in t
    assert "decomposition of one fit rather than two independent estimates" in t


# ------------------------------------------------------------- the delta column ----
def test_the_paper_reports_the_statistic_it_demands():
    """Recommendation 1 asks for the mean absolute token-length difference."""
    absd = {}
    for p in sorted((DATA / "instrument").glob("name_length_*.json")):
        if p.name == "name_length_effect.json":
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "pairs" not in d:
            continue
        v = [abs(r["delta_in_context"]) for r in d["pairs"]]
        absd[d["model"]] = sum(v) / len(v)
    assert absd, "no per-pair token artifacts"
    t = _paper()
    assert "mean |Δ|" in t
    for m, v in absd.items():
        assert f"{v:.2f}" in t, f"{m}: {v:.2f} is not on the page"


def test_the_signed_and_absolute_deltas_are_distinguished():
    lp = _load(DATA / "instrument" / "length_prediction.json")
    t = _paper()
    assert "SIGNED within-pair token-length difference" in t
    assert "absolute within-pair token-length difference" in t
    # they really are different numbers, or the distinction is pedantry
    signed = [v["mean_delta_over_all"] for v in lp["models"].values()]
    assert max(signed) < 1.5


# ------------------------------------------------------- cross-reference sanity ----
def test_no_cross_reference_points_at_a_section_that_does_not_answer_it():
    src = SRC.read_text(encoding="utf-8")
    assert "§7 answers it" not in src
    t = _paper()
    assert "§9 answers it" in t
    # every §N referenced in the body has a heading with that number
    heads = set(re.findall(r"H\(\"(\d+(?:\.\d+)?)\s", src))
    refs = set(re.findall(r"§(\d+(?:\.\d+)?)", src))
    assert refs <= heads, f"dangling cross-references: {sorted(refs - heads)}"


# ------------------------------------------------------------- typed tallies ----
def test_the_tallies_in_captions_are_derived():
    src = SRC.read_text(encoding="utf-8")
    assert "_slope_sig_row" in src
    assert "_nrej" in src
    nl = _load(DATA / "instrument" / "name_length_effect.json")
    live = {k: v for k, v in nl.items()
            if not k.startswith("_") and isinstance(v, dict)}
    row_sig = sum(1 for v in live.values() if v["p"] < 0.05)
    cl_sig = sum(1 for v in live.values()
                 if v["token_matched_first_name_clustered"]
                 ["slope_same_unit"]["p"] < 0.05)
    assert row_sig != cl_sig, (
        "the two resampling units now agree; Table 2's caption makes no point")
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four"}
    t = _paper()
    assert (f"this paper rejects, {words[row_sig]} of these slopes would be "
            f"distinguishable from zero rather than {words[cl_sig]}") in t


# ----------------------------------------------------------- the arXiv field ----
def test_the_arxiv_abstract_is_generated_and_clean():
    p = ROOT / "paper-c" / "releases" / "abstract_arxiv.txt"
    if not p.exists():
        pytest.skip("not built")
    s = p.read_text(encoding="utf-8").strip()
    assert s
    assert len(s) <= 1920
    assert all(ord(c) < 128 for c in s), (
        [c for c in s if ord(c) > 127])
    assert "\n" not in s
    assert not s.lower().startswith("abstract")
    # generated from the same string the PDF sets, not hand-copied
    assert "arxiv_abstract" in SRC.read_text(encoding="utf-8")


# ------------------------------------------------------ the two papers agree ----
def test_the_shared_quantities_agree_across_the_two_papers():
    """Both read the same artifacts, so a disagreement is a builder bug."""
    a, c = _text(PDF_A), _paper()
    nl = _load(DATA / "instrument" / "name_length_effect.json")
    live = {k: v for k, v in nl.items()
            if not k.startswith("_") and isinstance(v, dict)}
    fracs = sorted(v["n_same_length"] / v["n_pairs"] for v in live.values())
    for f in (fracs[0], fracs[-1]):
        s = f"{100 * f:.0f} %"
        assert s in a and s in c, f"{s}: a={s in a} c={s in c}"
    n_pairs = str(next(iter(live.values()))["n_pairs"])
    assert n_pairs in a and n_pairs in c
