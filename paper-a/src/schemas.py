"""
Input schema conditions. THIS IS THE NOVEL CONTRIBUTION OF PAPER A.

Prior audits (Wilson & Caliskan 2024, Gao et al. 2026, Tan et al. 2026) all
present resumes as free text. Production ATS pipelines do not: they parse a
document into a structured field set, silently drop what does not map, and
score against a schema. A model reading a resume and a model scoring a field
set are not performing the same task, and the schema determines which
demographic proxies survive to the model at all.

Three conditions, byte-identical underlying content:

  A. FREE_TEXT   - resume as one string. Replicates prior work.
  B. STRUCTURED  - full parse into explicit typed fields.
  C. ATS_SUBSET  - condition B minus fields commonly dropped by commercial
                   parsers, with high-weight fields ordered first.

IMPORTANT — CONFIDENTIALITY BOUNDARY. Condition C must be grounded in PUBLIC
sources only: HR Open Standards schemas, published ATS vendor API docs, and
open-source resume parsers. Do NOT encode any specific employer's proprietary
weighting, thresholds, or field handling. Your contribution is knowing which
public sources describe real practice and which do not; that is legitimate,
citable, and safe. See PROTOCOL.md section 10.

Every field-inclusion decision in ATS_DROPPED / ATS_PRIORITY must carry a
citation in docs/schema_sources.md. A reviewer will ask where the schema
came from, and "industry knowledge" is not an acceptable answer in a paper.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SchemaCondition(str, Enum):
    FREE_TEXT = "free_text"
    STRUCTURED = "structured"
    ATS_SUBSET = "ats_subset"


# Fields commonly discarded or not extracted by commercial resume parsers.
# EVERY ENTRY NEEDS A PUBLIC CITATION in docs/schema_sources.md before submission.
ATS_DROPPED: tuple[str, ...] = (
    "hobbies",
    "personal_statement",
    "references",
    "volunteering_detail",   # often collapsed to a boolean or dropped
    "activity_descriptions",  # titles survive, prose usually does not
)

# Fields typically surfaced first / weighted heavily in structured scoring.
# Same citation requirement.
ATS_PRIORITY: tuple[str, ...] = (
    "years_experience",
    "skills",
    "education",
    "certifications",
    "job_titles",
)


@dataclass
class Resume:
    """Canonical resume representation. Rendered into each schema condition.

    Demographic manipulation happens upstream (see resume_gen.py and
    proxies.py) and writes into `name`, `languages`, `affiliations`,
    `institution`, and `activities`. Everything else is held constant
    within a pair.
    """
    name: str
    email: str
    phone: str
    institution: str
    degree: str
    grad_year: int
    gpa: float | None
    years_experience: float
    job_titles: list[str] = field(default_factory=list)
    experience: list[dict[str, Any]] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    activity_descriptions: list[str] = field(default_factory=list)
    volunteering_detail: list[str] = field(default_factory=list)
    hobbies: list[str] = field(default_factory=list)
    personal_statement: str = ""
    references: list[str] = field(default_factory=list)

    # Provenance, never shown to the model. Used for analysis joins.
    template_id: str = ""
    race_signal: str = "none"
    gender_signal: str = "none"
    channel: str = "neither"   # name_only | proxy_only | both | neither


def render_free_text(r: Resume) -> str:
    """Condition A. Conventional resume prose.

    INFORMATIONAL EQUIVALENCE. Every field carried by render_structured must also
    appear here. RQ3 asks whether the same content presented differently changes
    measured disparity; if condition A silently omits a field that B and C carry,
    part of any measured "schema effect" is an information effect and H3a/H3b
    become unfalsifiable.

    An earlier version omitted `years_experience` and `references`, and carried
    `job_titles` only implicitly inside the experience prose. Found by
    tests/test_schemas.py::test_documents_fields_that_condition_a_does_not_carry
    on 2026-07-26 and fixed rather than written up as a limitation, because a
    limitation here would have permanently weakened the paper's novel claim.

    Length still differs across conditions and cannot be equalised without
    destroying the manipulation. It is measured per condition and entered as a
    covariate instead. See docs/decisions_2026-07-26.md, decision 4.
    """
    lines: list[str] = [r.name, f"{r.email} | {r.phone}", ""]

    if r.personal_statement:
        lines += [r.personal_statement, ""]

    lines += ["SUMMARY",
              f"{r.years_experience:g} years of professional experience"
              + (f" across {', '.join(r.job_titles)}." if r.job_titles else "."), ""]

    lines += ["EDUCATION",
              f"{r.degree}, {r.institution}, {r.grad_year}"
              + (f" (GPA {r.gpa:.2f})" if r.gpa is not None else ""), ""]

    lines.append("EXPERIENCE")
    for e in r.experience:
        lines.append(f"{e.get('title','')}, {e.get('org','')} "
                     f"({e.get('start','')}-{e.get('end','')})")
        for b in e.get("bullets", []):
            lines.append(f"  - {b}")
    lines.append("")

    if r.skills:
        lines += ["SKILLS", ", ".join(r.skills), ""]
    if r.certifications:
        lines += ["CERTIFICATIONS", ", ".join(r.certifications), ""]
    if r.languages:
        lines += ["LANGUAGES", ", ".join(r.languages), ""]
    if r.affiliations:
        lines += ["AFFILIATIONS", ", ".join(r.affiliations), ""]
    if r.activities:
        lines.append("ACTIVITIES")
        if r.activity_descriptions:
            for a, d in zip(r.activities, r.activity_descriptions):
                lines.append(f"  - {a}: {d}")
        else:
            lines += [", ".join(r.activities)]
        lines.append("")
    if r.volunteering_detail:
        lines += ["VOLUNTEERING"] + [f"  - {v}" for v in r.volunteering_detail] + [""]
    if r.hobbies:
        lines += ["INTERESTS", ", ".join(r.hobbies), ""]
    if r.references:
        lines += ["REFERENCES", ", ".join(r.references), ""]

    return "\n".join(lines).strip()


def render_structured(r: Resume) -> str:
    """Condition B. Full typed parse, nothing dropped."""
    payload = {
        "name": r.name,
        "contact": {"email": r.email, "phone": r.phone},
        "education": [{
            "institution": r.institution, "degree": r.degree,
            "graduation_year": r.grad_year, "gpa": r.gpa,
        }],
        "years_experience": r.years_experience,
        "job_titles": r.job_titles,
        "experience": r.experience,
        "skills": r.skills,
        "certifications": r.certifications,
        "languages": r.languages,
        "affiliations": r.affiliations,
        "activities": r.activities,
        "activity_descriptions": r.activity_descriptions,
        "volunteering_detail": r.volunteering_detail,
        "hobbies": r.hobbies,
        "personal_statement": r.personal_statement,
        "references": r.references,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def render_ats_subset(r: Resume) -> str:
    """Condition C. Structured minus commonly-dropped fields, priority first.

    The interesting prediction (H3b): this condition should REDUCE name-based
    disparity, because incidental prose cues disappear, while INCREASING
    proxy-based disparity, because languages/affiliations/institution become
    explicit weighted inputs rather than buried text.
    """
    full = json.loads(render_structured(r))
    for k in ATS_DROPPED:
        full.pop(k, None)

    ordered: dict[str, Any] = {}
    for k in ATS_PRIORITY:
        if k in full:
            ordered[k] = full.pop(k)
    ordered.update(full)
    return json.dumps(ordered, indent=2, ensure_ascii=False)


RENDERERS = {
    SchemaCondition.FREE_TEXT: render_free_text,
    SchemaCondition.STRUCTURED: render_structured,
    SchemaCondition.ATS_SUBSET: render_ats_subset,
}


def render(r: Resume, condition: SchemaCondition) -> str:
    return RENDERERS[condition](r)


def assert_content_equivalence(r: Resume) -> None:
    """Guard: the three conditions must differ in PRESENTATION only.

    Condition C legitimately drops fields, so equivalence is checked on the
    fields C retains. If a substantive value differs across conditions, the
    schema comparison is confounded and the result is meaningless.
    """
    structured = json.loads(render_structured(r))
    ats = json.loads(render_ats_subset(r))
    for k, v in ats.items():
        if structured.get(k) != v:
            raise AssertionError(
                f"Field '{k}' differs between STRUCTURED and ATS_SUBSET. "
                "Presentation may vary; values may not."
            )
    text = render_free_text(r)
    for skill in r.skills:
        if skill not in text:
            raise AssertionError(f"Skill '{skill}' missing from free-text render.")
    for lang in r.languages:
        if lang not in text:
            raise AssertionError(f"Language '{lang}' missing from free-text render.")
