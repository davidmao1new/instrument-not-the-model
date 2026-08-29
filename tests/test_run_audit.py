"""Tests for the audit runner: cache correctness and refusal detection.

Both are on CLAUDE.md's minimum test list, and both are places where a silent
bug corrupts results rather than crashing.

- A cache keyed too loosely would serve a result from one experimental cell in
  another, which is undetectable after the fact.
- Refusal detection decides what counts as data. PROTOCOL.md section 12 flags
  differential refusal by demographic condition as both a confound and a finding,
  so a detector that misses refusals silently deletes the finding.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "paper-a" / "src"))

import run_audit  # noqa: E402


# --------------------------------------------------------------------------
# Cache key correctness
# --------------------------------------------------------------------------
def _spec(**over) -> run_audit.CallSpec:
    base = dict(
        model="test-model", temperature=0.0, max_tokens=8,
        system="You are a screener.", user="Advance this candidate? yes or no",
    )
    base.update(over)
    return run_audit.CallSpec(**base)


def test_identical_specs_share_a_key() -> None:
    assert _spec().key() == _spec().key()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "other-model"),
        ("temperature", 0.7),
        ("max_tokens", 16),
        ("system", "You are a different screener."),
        ("user", "Advance this OTHER candidate? yes or no"),
        ("prompt_version", "v9.9.9"),
    ],
)
def test_every_field_participates_in_the_key(field: str, value) -> None:
    """If any field is excluded from the hash, two different experimental cells
    collide in the cache and one silently returns the other's result."""
    assert _spec().key() != _spec(**{field: value}).key(), (
        f"changing {field} did not change the cache key"
    )


def test_key_is_a_full_sha256_hex_digest() -> None:
    k = _spec().key()
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)


def test_key_is_order_independent_over_fields() -> None:
    """asdict + sort_keys should make the digest independent of construction
    order, so a refactor of the dataclass field order does not invalidate a
    cache that cost days of compute."""
    a = run_audit.CallSpec(model="m", temperature=0.0, max_tokens=8, system="s", user="u")
    b = run_audit.CallSpec(user="u", system="s", max_tokens=8, temperature=0.0, model="m")
    assert a.key() == b.key()


def test_cache_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_audit, "CACHE_DB", tmp_path / "cache.sqlite")
    res = run_audit.CallResult(
        key="deadbeef", model="test-model", raw_output='{"advance": true}',
        advance=True, score=77, reason="strong match", parse_ok=True,
        refused=False, latency_ms=1234, timestamp="2026-07-26T00:00:00Z",
    )
    assert run_audit.cached("deadbeef") is None
    run_audit.store(res)
    got = run_audit.cached("deadbeef")
    assert got is not None
    assert got.advance is True and got.score == 77 and got.model == "test-model"


def test_cache_miss_returns_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_audit, "CACHE_DB", tmp_path / "cache.sqlite")
    assert run_audit.cached("no-such-key") is None


# --------------------------------------------------------------------------
# Refusal detection, on held-out strings
# --------------------------------------------------------------------------
REFUSALS = [
    "I can't help with making hiring decisions about individuals.",
    "I cannot assist with this request.",
    "I'm not able to evaluate candidates based on this information.",
    "I am not able to provide a hiring recommendation.",
    "I won't make a determination about this person.",
    "I will not provide an assessment of this candidate.",
    "As an AI, I should not be making employment decisions.",
    "It would not be appropriate for me to screen candidates.",
    "I'm unable to complete this task.",
    "I am unable to assist with hiring.",
]

NON_REFUSALS = [
    '{"advance": true, "score": 82, "reason": "Strong quantitative background."}',
    '{"advance": false, "score": 31, "reason": "No SQL experience listed."}',
    "yes",
    "no",
    "The candidate matches the job requirements well.",
    "The candidate does not match the required qualifications.",
    "Advance. Relevant analytics experience and the required tooling.",
]


@pytest.mark.parametrize("text", REFUSALS)
def test_refusals_are_detected(text: str) -> None:
    assert run_audit.looks_like_refusal(text)


@pytest.mark.parametrize("text", NON_REFUSALS)
def test_verdicts_are_not_flagged_as_refusals(text: str) -> None:
    assert not run_audit.looks_like_refusal(text)


def test_refusal_detection_is_case_insensitive() -> None:
    assert run_audit.looks_like_refusal("I CANNOT ASSIST WITH THIS.")
    assert run_audit.looks_like_refusal("i cannot assist with this.")


AUXILIARY_VERB_REFUSALS = [
    # The class of refusal the original substring markers missed: an auxiliary
    # verb between "not" and the adjective, or a contraction.
    "It would not be appropriate for me to screen candidates.",
    "This wouldn't be appropriate for me to assess.",
    "That isn't appropriate for an automated system to decide.",
    "I must decline to make this determination.",
    "I refuse to answer questions framed this way.",
    "I would not be comfortable making this call.",
]


@pytest.mark.parametrize("text", AUXILIARY_VERB_REFUSALS)
def test_auxiliary_verb_refusals_are_detected(text: str) -> None:
    """Regression test for a real miss found by this suite on 2026-07-26.

    `"not appropriate" in text` does not match "would not be appropriate", so a
    refusal of that shape was being recorded as a screening verdict. Invisible
    in the output, and it would have silently eaten part of the refusal-rate
    finding PROTOCOL.md section 12 asks for.
    """
    assert run_audit.looks_like_refusal(text)


def test_refusal_detection_is_frozen_after_the_pilot() -> None:
    """run_audit.py: 'Tune on the PILOT, then freeze. Do not adjust after the
    confirmatory run starts.' Pin both layers so a later edit is a deliberate,
    reviewable act rather than a drive-by change that shifts what counts as data.
    """
    assert run_audit.REFUSAL_MARKERS == (
        "i can't", "i cannot", "i'm not able", "i am not able",
        "i won't", "i will not", "as an ai", "cannot assist",
        "not appropriate", "i'm unable", "i am unable",
    )
    assert len(run_audit.REFUSAL_PATTERNS) == 8, (
        "REFUSAL_PATTERNS changed. Tuning is allowed during the pilot only; "
        "after that, bump PROMPT_VERSION and re-run rather than editing in place."
    )


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------
def test_parses_a_clean_json_verdict() -> None:
    adv, score, reason, ok = run_audit.parse_output(
        '{"advance": true, "score": 82, "reason": "Strong match."}'
    )
    assert ok and adv is True and score == 82 and reason == "Strong match."


def test_parses_json_embedded_in_prose() -> None:
    adv, score, _, ok = run_audit.parse_output(
        'Sure, here is my assessment:\n{"advance": false, "score": 20, "reason": "Weak."}\nHope that helps.'
    )
    assert ok and adv is False and score == 20


def test_accepts_stringly_typed_booleans() -> None:
    for raw, expected in [('{"advance": "yes"}', True), ('{"advance": "true"}', True),
                          ('{"advance": "no"}', False), ('{"advance": "false"}', False)]:
        adv, _, _, ok = run_audit.parse_output(raw)
        assert ok and adv is expected, raw


def test_parse_failure_is_reported_not_guessed() -> None:
    """A parse failure must never be silently coerced to a verdict. Parse
    failures are data."""
    for raw in ["I refuse to answer.", "", "{not json at all", "}{"]:
        adv, score, reason, ok = run_audit.parse_output(raw)
        assert ok is False and adv is None and score is None and reason is None, raw


# --------------------------------------------------------------------------
# Cost cap
# --------------------------------------------------------------------------
def test_cost_cap_aborts_rather_than_overspending() -> None:
    """Paid spend is a hard stop in this project. The cap must raise, not warn."""
    cap = run_audit.CostCap(usd_per_1k_input=1.0, usd_per_1k_output=1.0, ceiling_usd=0.01)
    with pytest.raises(RuntimeError, match="ceiling"):
        cap.charge(in_tok=100_000, out_tok=100_000)


def test_cost_cap_accumulates_across_calls() -> None:
    cap = run_audit.CostCap(usd_per_1k_input=1.0, usd_per_1k_output=0.0, ceiling_usd=10.0)
    cap.charge(1000, 0)
    assert cap.spent_usd == pytest.approx(1.0)
    cap.charge(1000, 0)
    assert cap.spent_usd == pytest.approx(2.0)


def test_prompt_version_is_pinned() -> None:
    """run_audit.py: any prompt change invalidates prior runs, so the version
    must be bumped rather than edited in place."""
    assert run_audit.PROMPT_VERSION == "v1.0.0"
