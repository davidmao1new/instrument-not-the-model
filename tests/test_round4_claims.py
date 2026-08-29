"""Tests for the claims round 4 forced the paper to weaken or source.

Each of these was a sentence that outran its evidence. None of them was an
arithmetic error -- every number was correctly computed from a fresh artifact --
so no numeric check could have caught any of them. What was wrong was the
epistemic status the prose assigned to a number: a bound stated as an absence,
a direction stated as settled, a ratio stated without the interval that makes
it undetermined, a premise stated without a source.

So these assert on what the prose is ALLOWED to say, given the artifacts.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
DATA = ROOT / "paper-a" / "data"
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


def _paper():
    if not PDF.exists():
        pytest.skip("paper not built")
    fitz = pytest.importorskip("fitz")
    with fitz.open(PDF) as doc:
        return " ".join(" ".join(p.get_text().split()) for p in doc)

def _companion():
    """The token-matching study moved to paper-c; its claims moved with it."""
    p = ROOT / "paper-c" / "figures" / "paper_token_matching.pdf"
    if not p.exists():
        return ""
    fitz = pytest.importorskip("fitz")
    with fitz.open(p) as doc:
        return " ".join(" ".join(pg.get_text().split()) for pg in doc)



# ------------------------------------------------- the quantization ratio ----
def test_the_quantization_ratio_has_an_interval():
    q = _load(DATA / "quantization" / "quantization_analysis.json")
    seen = 0
    for m, v in q.items():
        if m.startswith("_") or not isinstance(v, dict):
            continue
        assert v.get("shift_over_sigma_variant_ci"), m
        lo, hi = v["shift_over_sigma_variant_ci"]
        assert lo <= v["shift_over_sigma_variant"] <= hi, m
        seen += 1
    assert seen >= 1


def test_the_side_of_parity_is_undetermined_so_the_paper_may_not_claim_it():
    """If a re-run ever determines it, this fails and the prose may be firmed."""
    q = _load(DATA / "quantization" / "quantization_analysis.json")
    live = [v for m, v in q.items()
            if not m.startswith("_") and isinstance(v, dict)]
    assert not any(v["ratio_exceeds_one_determined"] for v in live)
    t = _paper()
    assert "up to the full between-wording standard deviation" not in t


# --------------------------------------------------- bound, not absence ----
def test_the_serving_result_is_stated_as_a_bound_everywhere():
    """§5.2 and §9 item 7 asserted an absence the Conclusion disowned."""
    t = _paper()
    assert "No number in this paper inherits a dependence" not in t
    assert "These do not move the effect" not in t
    # the interval is what we have, so the interval is what is claimed
    assert "bound rather than an absence" in t or "a bound and not an absence" in t


# ------------------------------------------------ the token-match direction ----
def test_the_masking_direction_is_not_asserted_flatly():
    """Two of four estimates are uninterpretable and one points the other way."""
    t = _paper()
    assert "The confound was MASKING the disparity, not manufacturing it" not in t
    assert "The confound was masking the finding, not manufacturing it" not in t
    nl = _load(DATA / "instrument" / "name_length_effect.json")
    interp = [m for m, v in nl.items()
              if not m.startswith("_") and isinstance(v, dict)
              and v.get("token_matched_first_name_clustered", {})
              .get("growth_ratio", {}).get("interpretable")]
    assert len(interp) >= 2, "the caveat needs at least two interpretable models"
    assert nl["_verdict"]["n_subset_significant"] == 0, (
        "a subset difference became significant; the softened wording can be "
        "revisited")


# ------------------------------------------------------- the §2 novelty claim ----
def test_section_2_does_not_contradict_table_18():
    """Table 18 credits Seshadri with a null-edit control; §2 denied it."""
    mx = _load(DATA / "reference" / "reporting_practice_matrix.json")
    credited = mx["counts"]["null_edit_control"]["reported_by"]
    assert credited, "nobody is credited; this test pins nothing"
    t = _paper()
    assert "neither runs a semantically-null arm" not in t
    assert "do run a null-edit control" in t


def test_the_novelty_claim_lists_only_choices_nobody_reports():
    mx = _load(DATA / "reference" / "reporting_practice_matrix.json")
    for f in mx["never_reported_by_any"]:
        assert mx["counts"][f]["n_reported"] == 0, f


# --------------------------------------------------------- the legal premise ----
def test_the_opening_premise_is_sourced():
    """It carried no citation, no reference entry and no ledger row."""
    t = _paper()
    assert "Local Law 144" in t
    assert "20-871" in t
    # the two limbs with nothing behind them are gone
    assert "regulation, procurement and litigation" not in t
    assert "vendor procurement and litigation" not in t


def test_the_legal_sources_are_in_the_reference_list_and_on_disk():
    t = _paper()
    assert "N.Y.C. Admin. Code" in t
    assert "Department of Consumer and Worker Protection" in t
    for p in ("lit/law/text/ll144_2021_enacted.txt",
              "lit/law/text/dcwp_final_rule_2023.txt"):
        assert (ROOT / p).exists(), f"{p} missing; the citation is unsourced"


# ------------------------------------------------------------ Sclar's Table 1 ----
def test_the_sclar_counts_match_their_table():
    """Three of six is half, not most; and the complement is three, not two."""
    t = _paper()
    assert "half the individual format features do not" in t
    assert "leaves three with a positive individual signal" in t


def test_an_external_table_reference_is_not_generated_by_our_numbering():
    """"their Table 1" must not come from this paper's TABLE_ORDER."""
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    # Case-insensitive on the leading letter. This pinned 'in their Table
    # 1' and failed when the sentence before it gained a full stop, making
    # it 'In their Table 1'. The claim is what matters, not the capital.
    assert 'in their table 1 the second' in src.lower()
    assert 'in their " + TAB(' not in src


# ------------------------------------------------------- the bootstrap atom ----
def test_the_atom_probability_is_described_as_what_it_is():
    """0.11 is P(all k draws are one cluster), not P(any two coincide).

    Twice corrected. The original said "any two draws coincide" (0.78, wrong
    quantity); the repair said the 602 discarded draws "did exactly that",
    fusing the matched-arm atom to the difference bootstrap's degenerate draws
    -- two different procedures whose rates differ by a factor of 3.7. What is
    pinned here is that the two are no longer asserted to be one event.
    """
    t = _paper() + "\n" + _companion()
    assert "any two draws coincide" not in t
    assert "resamples did exactly that" not in t, (
        "the atom and the discarded draws are fused again")
    assert "on drawing a single cluster every time" in t


def test_the_two_bootstraps_are_kept_apart_numerically():
    """The atom and the discard rate differ enough that fusing them is checkable."""
    nl = _load(DATA / "instrument" / "name_length_effect.json")
    v = nl["llama-3.1-8b-instruct"]["token_matched_first_name_clustered"]
    atom = v["same_cluster_draw_probability"]
    degen = v["matched_minus_all"]["n_draws_degenerate"] / v["n_boot"]
    assert abs(atom - degen) > 0.05, (
        "the two rates have converged; the prose distinction needs rechecking")
    # the discard rate should track (1 - m/g)^g for m matched of g grid clusters
    g, m = v["n_grid_clusters"], v["n_matched_clusters"]
    assert abs(degen - (1 - m / g) ** g) < 0.01


def test_the_atom_probability_equals_k_to_the_minus_k_minus_one():
    nl = _load(DATA / "instrument" / "name_length_effect.json")
    seen = 0
    for m, v in nl.items():
        if m.startswith("_") or not isinstance(v, dict):
            continue
        cl = v.get("token_matched_first_name_clustered", {})
        k = cl.get("n_matched_clusters")
        p = cl.get("same_cluster_draw_probability")
        if k is None or p is None:
            continue
        seen += 1
        assert abs(p - k ** -(k - 1)) < 1e-9, (
            f"{m}: the atom probability is not k^-(k-1) for k={k}")
    assert seen >= 1, "no clustering diagnostics found; this pins nothing"
