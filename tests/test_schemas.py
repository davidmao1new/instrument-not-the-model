"""Schema content-equivalence — the guard on Paper A's novel contribution.

RQ3 asks whether the SAME content, presented differently, changes measured
disparity. That question is only meaningful if the content really is the same.
If condition A silently omits a field that conditions B and C carry, then a
"schema effect" is partly an information effect and H3a/H3b are unfalsifiable.

This suite therefore tests the equivalence guarantee itself, not just the code.
It documents two real gaps found on 2026-07-26; see CHANGELOG.md.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "paper-a" / "src"))

import schemas  # noqa: E402
from schemas import Resume, SchemaCondition  # noqa: E402


def make_resume(**over) -> Resume:
    """A resume exercising every proxy family in proxies.py at once."""
    base = dict(
        name="Jordan Ellis",
        email="jordan.ellis@example.com",
        phone="215-555-0142",
        institution="State University",
        degree="B.S. Statistics",
        grad_year=2024,
        gpa=3.6,
        years_experience=2.0,
        job_titles=["Operations Analyst", "Data Intern"],
        experience=[
            {"title": "Operations Analyst", "org": "Keystone Logistics",
             "start": "2024", "end": "2026",
             "bullets": ["Owned weekly reporting for a 40-person team."]},
        ],
        skills=["Excel", "SQL", "Tableau"],
        certifications=["Tableau Desktop Specialist"],
        languages=["Spanish (fluent)"],
        affiliations=["Member, National Society of Black Engineers"],
        activities=["Volunteer tutor, Urban League youth program"],
        activity_descriptions=["Weekly algebra tutoring for high school students."],
        volunteering_detail=["Food bank, 3 hours monthly"],
        hobbies=["Chess", "Distance running"],
        personal_statement="Analytical and detail-oriented.",
        references=["Available on request"],
        template_id="T01",
        race_signal="black",
        gender_signal="none",
        channel="proxy_only",
    )
    base.update(over)
    return Resume(**base)


# --------------------------------------------------------------------------
# Renderers produce output at all
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cond", list(SchemaCondition))
def test_every_condition_renders_non_empty(cond: SchemaCondition) -> None:
    assert schemas.render(make_resume(), cond).strip()


@pytest.mark.parametrize("cond", [SchemaCondition.STRUCTURED, SchemaCondition.ATS_SUBSET])
def test_structured_conditions_emit_valid_json(cond: SchemaCondition) -> None:
    json.loads(schemas.render(make_resume(), cond))


def test_conditions_are_actually_different() -> None:
    """If two conditions render identically the factor has no levels."""
    rendered = {c: schemas.render(make_resume(), c) for c in SchemaCondition}
    assert len(set(rendered.values())) == 3


# --------------------------------------------------------------------------
# The equivalence guarantee
# --------------------------------------------------------------------------
def test_existing_guard_passes_on_a_wellformed_resume() -> None:
    schemas.assert_content_equivalence(make_resume())


def test_ats_subset_drops_only_the_declared_fields() -> None:
    r = make_resume()
    structured = json.loads(schemas.render_structured(r))
    ats = json.loads(schemas.render_ats_subset(r))
    assert set(structured) - set(ats) == set(schemas.ATS_DROPPED)


def test_ats_subset_never_alters_a_retained_value() -> None:
    """Condition C may drop fields. It may not change them."""
    r = make_resume()
    structured = json.loads(schemas.render_structured(r))
    ats = json.loads(schemas.render_ats_subset(r))
    for k, v in ats.items():
        assert structured[k] == v, f"field '{k}' changed value between B and C"


def test_ats_priority_fields_come_first() -> None:
    """Condition C's premise is that high-weight fields are surfaced first.
    JSON key order is the manipulation, so it has to actually hold."""
    ats = json.loads(schemas.render_ats_subset(make_resume()))
    present = [k for k in schemas.ATS_PRIORITY if k in ats]
    assert list(ats)[: len(present)] == present


# --------------------------------------------------------------------------
# GAP 1: the guard checks only two of the four proxy families
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "family_field,value",
    [
        ("skills", "Excel"),
        ("languages", "Spanish (fluent)"),
        ("affiliations", "Member, National Society of Black Engineers"),
        ("activities", "Volunteer tutor, Urban League youth program"),
    ],
)
def test_every_proxy_family_survives_the_free_text_render(family_field: str, value: str) -> None:
    """proxies.py manipulates FOUR families: language, affiliation, institution
    and activity. `assert_content_equivalence` only checks skills and languages.
    A proxy that vanished from condition A would make RQ2 and RQ3 confounded
    while every existing check still passed."""
    text = schemas.render_free_text(make_resume())
    assert value in text, f"{family_field} value missing from the free-text render"


def test_institution_survives_the_free_text_render() -> None:
    """The institution family is the one Iso et al. (arXiv:2503.19182) found
    persistent bias in, and the one PROTOCOL section 9's IPEDS matching is
    designed around. It must reach the model in every condition."""
    assert "State University" in schemas.render_free_text(make_resume())


def test_guard_catches_a_dropped_language() -> None:
    r = make_resume()
    r.languages = ["Klingon (fluent)"]
    original = schemas.render_free_text
    try:
        schemas.render_free_text = lambda res: original(res).replace("Klingon (fluent)", "")
        with pytest.raises(AssertionError, match="Language"):
            schemas.assert_content_equivalence(r)
    finally:
        schemas.render_free_text = original


# --------------------------------------------------------------------------
# GAP 2: fields present in B and C but absent from A
# --------------------------------------------------------------------------
def test_conditions_are_informationally_equivalent() -> None:
    """RQ3's validity condition, and now an enforced invariant rather than a
    limitation.

    Every field carried by the structured render must also reach the model in the
    free-text render. If condition A omits something B and C carry, part of any
    measured "schema effect" is an information effect and H3a/H3b are
    unfalsifiable.

    On 2026-07-26 this failed: `years_experience` and `references` were absent
    from condition A, and `job_titles` appeared only implicitly. Fixed in
    `render_free_text` rather than written up as a limitation. See
    docs/decisions_2026-07-26.md, decision 4.
    """
    # Use a sentinel value that cannot collide with a year, a GPA or a bullet,
    # so absence is real absence rather than a substring accident. An earlier
    # version of this test used years_experience=2.0 and matched "2" inside
    # "2024", reporting the field as present when it is not.
    r = make_resume(years_experience=7777.0, references=["REF-SENTINEL-9999"])
    text = schemas.render_free_text(r)
    structured = json.loads(schemas.render_structured(r))

    absent = {
        key
        for key, sentinel in (("years_experience", "7777"),
                              ("references", "REF-SENTINEL-9999"),
                              ("job_titles", "Operations Analyst"))
        if sentinel not in text
    }
    assert structured["years_experience"] == 7777.0
    assert structured["references"] == ["REF-SENTINEL-9999"]

    assert absent == set(), (
        f"Fields present in the structured render but missing from condition A: "
        f"{absent}. RQ3 requires informational equivalence across conditions. "
        "Add the field to render_free_text; do not accept the asymmetry as a "
        "limitation, because it makes H3a/H3b unfalsifiable."
    )


# --------------------------------------------------------------------------
# GAP 3: length, which Wilson & Caliskan show is not neutral
# --------------------------------------------------------------------------
def test_condition_lengths_are_recorded_not_assumed() -> None:
    """Wilson & Caliskan (ledger A-25) show shorter documents produce measurably
    MORE biased outcomes, up to +22.2%. The three conditions differ substantially
    in length, so length is a live confound in RQ3 and must be measured and
    reported per condition, not assumed away.

    This test does not enforce equal length -- equalising it would destroy the
    manipulation. It asserts the differences are large enough to matter, which is
    the fact the paper has to confront.
    """
    r = make_resume()
    lengths = {c.value: len(schemas.render(r, c)) for c in SchemaCondition}
    spread = max(lengths.values()) / min(lengths.values())
    assert spread > 1.2, (
        f"Condition lengths are unexpectedly similar: {lengths}. If this becomes "
        "true the length confound has gone away and the paper's limitation "
        "section should be revisited."
    )


# --------------------------------------------------------------------------
# Provenance must never reach the model
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cond", list(SchemaCondition))
def test_provenance_fields_never_reach_the_model(cond: SchemaCondition) -> None:
    """template_id, race_signal, gender_signal and channel are analysis joins.
    Leaking any of them into a rendered prompt would tell the model the answer."""
    r = make_resume()
    out = schemas.render(r, cond)
    for leaked in ("template_id", "race_signal", "gender_signal", "channel",
                   "T01", "proxy_only"):
        assert leaked not in out, f"provenance '{leaked}' leaked into {cond.value}"


def test_empty_optional_sections_do_not_appear_in_free_text() -> None:
    """A resume with no hobbies must not render an empty INTERESTS header:
    an empty header is itself a signal and would differ across pair members if
    the manipulation touches that field."""
    r = make_resume(hobbies=[], languages=[], affiliations=[], activities=[],
                    activity_descriptions=[], volunteering_detail=[])
    text = schemas.render_free_text(r)
    for header in ("INTERESTS", "LANGUAGES", "AFFILIATIONS", "ACTIVITIES", "VOLUNTEERING"):
        assert header not in text, f"empty section '{header}' still rendered"


def test_paired_resumes_differ_only_in_the_manipulated_field() -> None:
    """The core design claim in PROTOCOL section 3: pair members are identical
    except for the manipulation. Verify at the rendered-string level, because
    that is what the model actually sees."""
    a = make_resume(name="Allison Baker")
    b = make_resume(name="Aisha Jackson")
    for cond in SchemaCondition:
        ra, rb = schemas.render(a, cond), schemas.render(b, cond)
        assert ra.replace("Allison Baker", "X") == rb.replace("Aisha Jackson", "X"), (
            f"pair members differ beyond the name in condition {cond.value}"
        )
