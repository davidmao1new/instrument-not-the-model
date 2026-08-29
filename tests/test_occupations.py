"""The occupation arm is only a control if the three arms are matched.

Study 7 varies the occupation to test whether the wording instability is a
property of the model or of the job. That inference holds only if occupation is
the ONLY thing that varies. If the Software Engineer posting were longer, or its
résumés more detailed, a difference between arms could be a difference in prompt
length and the study would answer nothing.
"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "paper-a" / "src"
sys.path.insert(0, str(SRC))

import occupations as occ  # noqa: E402
import experiment_delta_stability as eds  # noqa: E402


def test_structural_matching_passes():
    problems = occ.check_matched()
    assert problems == [], problems


def test_the_business_analyst_arm_is_the_published_one():
    """Study 7 reuses Study 2's data for BA rather than re-running it, so the
    strings must not have drifted from the published ones."""
    assert occ.OCCUPATIONS["BA"]["posting"] == eds.POSTING
    assert occ.OCCUPATIONS["BA"]["templates"] == dict(eds.TEMPLATES)


def test_three_occupations_spanning_gender_typing():
    typings = {o["gender_typing"] for o in occ.OCCUPATIONS.values()}
    assert typings == {"balanced", "male-typed", "female-typed"}


@pytest.mark.parametrize("key", sorted(occ.OCCUPATIONS))
def test_every_occupation_has_the_same_three_strength_levels(key):
    assert sorted(occ.OCCUPATIONS[key]["templates"]) == sorted(eds.TEMPLATES)


@pytest.mark.parametrize("key", sorted(occ.OCCUPATIONS))
def test_postings_are_length_matched(key):
    lens = [len(o["posting"].split()) for o in occ.OCCUPATIONS.values()]
    assert max(lens) - min(lens) <= 4


@pytest.mark.parametrize("key", sorted(occ.OCCUPATIONS))
def test_no_occupation_leaks_a_demographic_cue(key):
    """The posting and résumés must carry no gender, race or age marker; the
    only demographic signal in the design is the candidate's name."""
    # Word boundaries, not substrings. A naive `"he " in text` matches inside
    # "the ", which fails on every occupation including the published one.
    banned = ("he", "she", "his", "her", "him", "hers", "male", "female",
              "man", "woman", "men", "women", "mr", "mrs", "ms", "aged")
    blob = (occ.OCCUPATIONS[key]["posting"] + " "
            + " ".join(occ.OCCUPATIONS[key]["templates"].values()))
    words = set(re.findall(r"[a-z]+", blob.lower()))
    leaked = sorted(words & set(banned))
    assert not leaked, f"{key} contains demographic cue(s): {leaked}"


@pytest.mark.parametrize("key", sorted(occ.OCCUPATIONS))
def test_strength_ordering_is_encoded_the_same_way(key):
    """T1 must be the strongest and T3 the weakest in every occupation, and the
    signal must be carried by the same fields: degree relevance, GPA, seniority."""
    t = occ.OCCUPATIONS[key]["templates"]
    assert "GPA 3.8" in t["T1_strong"]
    assert "GPA 3.3" in t["T2_mid"]
    assert "GPA 3.0" in t["T3_marginal"]
    assert "advanced" in t["T1_strong"].lower()
    assert "B.A. " in t["T3_marginal"] or "A.A. " in t["T3_marginal"]


def test_the_marginal_resume_is_off_field_in_every_occupation():
    """T3 is marginal because the degree does not match the posting. If one
    occupation's T3 were on-field it would be a different manipulation."""
    for key, o in occ.OCCUPATIONS.items():
        t3 = o["templates"]["T3_marginal"]
        assert ("Communications" in t3) or ("Liberal Studies" in t3), key


def test_employers_and_dates_are_held_constant():
    for o in occ.OCCUPATIONS.values():
        for t, body in o["templates"].items():
            assert "State University" in body
            assert "Keystone" in body
            assert ("2024-2026" in body) or ("2025-2026" in body)
