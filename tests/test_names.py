"""Tests for name-list construction.

CLAUDE.md requires "name-list matching on length and syllables" as a minimum
test. The reason is substantive rather than hygienic: PROTOCOL.md section 8
argues that unmatched name lists confound race signal with familiarity and
rarity, and Wilson & Caliskan show empirically (ledger A-24) that changing the
frequency-matching strategy can REVERSE the direction of the measured effect.
So the matching invariants are load-bearing, and a silent regression in
`_match_on_shape` would change the paper's conclusion rather than merely
breaking a build.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "paper-a" / "src"))

import names  # noqa: E402


# --------------------------------------------------------------------------
# Syllable heuristic
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("Greg", 1),
        ("Emily", 3),
        ("Jamal", 2),
        ("Aisha", 2),
        ("Todd", 1),
        ("Latonya", 3),
        ("Anne", 1),      # trailing silent e is stripped
        ("Kyle", 1),      # trailing silent e is stripped
    ],
)
def test_syllable_counts_on_known_names(name: str, expected: int) -> None:
    assert names.count_syllables(name) == expected


@given(st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu")), min_size=1, max_size=20))
@settings(max_examples=200)
def test_syllable_count_is_always_at_least_one(s: str) -> None:
    """A name contributing zero syllables would silently collapse a matching
    stratum and let unmatched names through."""
    assert names.count_syllables(s) >= 1


@given(st.text(alphabet="bcdfghjklmnpqrstvwxz", min_size=1, max_size=12))
@settings(max_examples=100)
def test_consonant_only_strings_get_one_syllable(s: str) -> None:
    assert names.count_syllables(s) == 1


def test_syllable_count_is_case_insensitive() -> None:
    for n in ("Emily", "EMILY", "emily", "eMiLy"):
        assert names.count_syllables(n) == names.count_syllables("Emily")


# --------------------------------------------------------------------------
# Bertrand & Mullainathan anchor set
# --------------------------------------------------------------------------
def test_bm2004_cells_are_balanced() -> None:
    """Unequal cell sizes would make the race contrast partly a count contrast."""
    sizes = {k: len(v) for k, v in names.BM2004.items()}
    assert len(set(sizes.values())) == 1, f"BM2004 cells are unbalanced: {sizes}"


def test_bm2004_covers_all_four_race_gender_cells() -> None:
    assert set(names.BM2004) == {
        ("white", "female"), ("white", "male"),
        ("black", "female"), ("black", "male"),
    }


def test_bm2004_has_no_duplicate_first_names_within_a_cell() -> None:
    for cell, firsts in names.BM2004.items():
        assert len(firsts) == len(set(firsts)), f"duplicate first name in {cell}"


def test_bm2004_records_are_wellformed() -> None:
    recs = names.bm2004_records()
    assert len(recs) == sum(len(v) for v in names.BM2004.values())
    for r in recs:
        assert r.first and r.last
        assert r.race in names.RACES
        assert r.gender in names.GENDERS
        assert r.n_chars == len(r.first)
        assert r.n_syllables >= 1
        assert r.source == "bertrand_mullainathan_2004"
        assert r.full == f"{r.first} {r.last}"


def test_bm2004_probabilities_are_nan_not_invented() -> None:
    """BM did not publish P(race|name); they curated by construction. Recording a
    number here would be fabricating provenance, so NaN is correct and the paper
    must say so."""
    for r in names.bm2004_records():
        assert pd.isna(r.p_race_first)
        assert pd.isna(r.p_race_last)
        assert pd.isna(r.freq_pctile)


# --------------------------------------------------------------------------
# The deliberately-unimplemented gender path
# --------------------------------------------------------------------------
def test_gender_assignment_still_refuses_to_guess() -> None:
    """CLAUDE.md: this raises on purpose so gender is never assigned by
    intuition. If someone satisfies it with plausible values, this test fails and
    forces the conversation."""
    with pytest.raises(NotImplementedError, match="gender"):
        names.build_matched_first_names()


# --------------------------------------------------------------------------
# Matching invariants — the ones that actually protect the paper
# --------------------------------------------------------------------------
def _fake_candidates() -> dict[str, pd.DataFrame]:
    """Synthetic pools that DO share (n_chars, n_syllables) strata across all
    four races, so the matcher has something to work with.

    Built deliberately: every race contributes at least two names in each of the
    (5 chars, 2 syllables) and (5 chars, 3 syllables) strata. Verified by
    `test_fixture_actually_shares_strata` below, so a change to the syllable
    heuristic surfaces as a fixture failure rather than as a confusing matcher
    failure.
    """
    return {
        "white": pd.DataFrame({"name": ["Sarah", "Karen", "Emily", "Megan", "Brian", "Nolan"]}),
        "black": pd.DataFrame({"name": ["Jamal", "Aisha", "Ebony", "Kenya", "Malik", "Deion"]}),
        "hispanic": pd.DataFrame({"name": ["Jorge", "Pablo", "Maria", "Elena", "Mateo", "Rosal"]}),
        "asian": pd.DataFrame({"name": ["Cheng", "Zhang", "Mihoo", "Naoki", "Yusuf", "Reiko"]}),
    }


def test_fixture_actually_shares_strata() -> None:
    """Guards the fixture itself. Without this, a change to count_syllables makes
    the matching tests fail for a reason that has nothing to do with matching."""
    strata = []
    for df in _fake_candidates().values():
        d = df.copy()
        d["n_chars"] = d["name"].str.len()
        d["n_syllables"] = d["name"].map(names.count_syllables)
        strata.append(set(zip(d["n_chars"], d["n_syllables"])))
    shared = set.intersection(*strata)
    assert len(shared) >= 2, f"fixture shares only {shared}; matching tests need at least 2 strata"


def test_matched_lists_share_shape_distribution() -> None:
    """The whole point of section 8's matching rule: the marginal distribution of
    name shape must be identical across races, or the race signal is confounded
    with familiarity and rarity."""
    out = names._match_on_shape(_fake_candidates(), per_cell=3)
    shapes = {
        race: sorted(zip(df["n_chars"], df["n_syllables"]))
        for race, df in out.items()
    }
    reference = next(iter(shapes.values()))
    for race, s in shapes.items():
        assert s == reference, f"{race} shape distribution differs from reference: {s} vs {reference}"


def test_matching_returns_requested_count_per_race() -> None:
    out = names._match_on_shape(_fake_candidates(), per_cell=3)
    for race, df in out.items():
        assert len(df) == 3, f"{race} got {len(df)} names, expected 3"


def test_matching_does_not_repeat_a_name_within_a_race() -> None:
    out = names._match_on_shape(_fake_candidates(), per_cell=3)
    for race, df in out.items():
        assert df["name"].is_unique, f"{race} list contains a duplicate"


def test_matching_holds_under_asymmetric_stratum_capacity() -> None:
    """Regression test for a real bug found by this suite on 2026-07-26.

    When races have DIFFERENT numbers of names available within a shared
    stratum, the earlier implementation drew unequal counts per stratum and the
    marginal shape distributions diverged, e.g. [(5,1),(5,1),(5,2)] against
    [(5,1),(5,2),(5,2)]. That silently reintroduces the exact length and
    familiarity confound PROTOCOL.md section 8 exists to eliminate.

    Here 'white' has one name at (5,1) and three at (5,2); 'black' has three at
    (5,1) and one at (5,2). Any correct matcher must return the same shape
    multiset for both.
    """
    candidates = {
        "white": pd.DataFrame({"name": ["Brian", "Sarah", "Karen", "Megan"]}),
        "black": pd.DataFrame({"name": ["Malik", "Deion", "Kayla", "Aisha"]}),
    }
    out = names._match_on_shape(candidates, per_cell=2)
    shapes = {r: sorted(zip(d["n_chars"], d["n_syllables"])) for r, d in out.items()}
    assert shapes["white"] == shapes["black"], (
        f"asymmetric capacity produced unmatched shapes: {shapes}"
    )


def test_matching_refuses_rather_than_returning_short_lists() -> None:
    """If the shared strata cannot supply per_cell names for every race, the
    only safe outcome is to raise. Returning short or unequal lists would let an
    unmatched manipulation reach the model, and every downstream number would be
    invalid without anything visibly failing."""
    thin = {
        "white": pd.DataFrame({"name": ["Sarah", "Karen"]}),
        "black": pd.DataFrame({"name": ["Aisha"]}),
    }
    with pytest.raises(ValueError, match="matched on shape|shared"):
        names._match_on_shape(thin, per_cell=5)


def test_matching_is_deterministic() -> None:
    """No RNG in name-set construction. Two runs must give byte-identical lists,
    or the 'pre-specified name set' claim in PROTOCOL.md section 8 is false."""
    a = names._match_on_shape(_fake_candidates(), per_cell=3)
    b = names._match_on_shape(_fake_candidates(), per_cell=3)
    for race in a:
        assert list(a[race]["name"]) == list(b[race]["name"]), f"{race} draw is not deterministic"


def test_matching_fails_loudly_when_no_shared_strata_exist() -> None:
    """Silently returning short or unmatched lists would invalidate every
    downstream number, so this must raise."""
    impossible = {
        "white": pd.DataFrame({"name": ["Bo"]}),                 # 2 chars, 1 syllable
        "black": pd.DataFrame({"name": ["Bartholomew"]}),        # 11 chars, 4 syllables
    }
    with pytest.raises(ValueError, match="shared"):
        names._match_on_shape(impossible, per_cell=1)


def test_prob_threshold_is_the_preregistered_value() -> None:
    """PROTOCOL section 8 pre-specifies 0.80. Tuning it after seeing results
    would be a pre-registration violation, so pin it."""
    assert names.PROB_THRESHOLD == 0.80
