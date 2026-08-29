"""Tests for the endmatter and the anonymisation switch.

FAccT desk-rejects a submission that carries identifying information and, separately,
desk-rejects one that omits the generative-AI statement. Those two rules land on the same
page and pull opposite ways, which is how a non-anonymised PDF gets uploaded at 11pm. The
switch is tested rather than remembered.

The third test is the one that matters most. FAccT's guide says verbatim that it
"prohibits the use of LLMs to generate text for publications". The anonymous build emits
a statement asserting the author wrote every sentence. That statement is FALSE until the
rewrite is done, and printing a false disclosure is a worse outcome than missing a
deadline, so the build refuses to run until an explicit flag says otherwise.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src" / "build_paper_v3.py"
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
PY = ROOT / "paper-a" / "src" / "_py.sh"


def _paper():
    if not PDF.exists():
        pytest.skip("paper not built")
    fitz = pytest.importorskip("fitz")
    with fitz.open(PDF) as doc:
        return " ".join(" ".join(p.get_text().split()) for p in doc)


def test_the_preprint_declares_the_competing_interest():
    """The author builds hiring software and audits hiring software. A reader of
    a fairness paper is owed that before the first result, not after."""
    t = _paper()
    assert "Competing interests" in t
    assert "a company that builds AI recruiting software" in t
    # The employer is described, not named: naming a small private company in a
    # paper auditing its category is commercially sensitive, and the function of
    # the disclosure -- telling a reader the author has a direct interest in the
    # class of system under audit -- is served without it. The clauses below are
    # what make the statement worth anything, so they are the ones pinned.
    assert "none of its data, models, prompts, thresholds or internal documents" in t
    assert "did not fund, commission, direct or review this study" in t
    assert "no system built by that company was audited" in t.lower()


def test_the_preprint_discloses_the_generative_ai_use_without_hedging():
    """What is pinned is the CONSEQUENCE, not the wording.

    How the drafting assistance gets characterised is the author's call and has
    already been reworded once. What cannot be softened is the sentence that
    lets a reader judge the characterisation for themselves: that this preprint
    would not satisfy a rule prohibiting LLM-generated text. Without it a reader
    has only an adjective to go on, and FAccT's rule turns on the substance.
    """
    t = _paper()
    assert "Generative AI usage" in t
    assert "drafting the prose of this document" in t, (
        "the disclosure no longer mentions prose drafting at all")
    assert "would not satisfy a venue rule prohibiting LLM-generated text" in t, (
        "the preprint must let a reader see that it is not FAccT-compliant")
    # and it must name the other three uses, which are the interesting ones
    for use in ("typesetting", "screened the literature", "audited the methodology"):
        assert use in t, f"the disclosure omits {use!r}"


def test_the_paper_carries_an_ethics_statement():
    t = _paper()
    assert "Ethical considerations" in t
    # the adverse impact a reviewer will ask about first
    assert "licence to dismiss any audit finding" in t


def test_the_anonymous_build_refuses_to_print_a_false_disclosure():
    """The guard, exercised for real rather than asserted from the source."""
    if not PY.exists():
        pytest.skip("no toolchain shim")
    env = dict(os.environ, PAPER_VENUE="facct")
    env.pop("PAPER_PROSE_REWRITTEN", None)
    r = subprocess.run(["sh", str(PY), str(SRC)], capture_output=True,
                       text=True, cwd=str(ROOT), env=env, timeout=900)
    assert r.returncode != 0, "the facct build ran without the rewrite flag"
    out = (r.stdout + r.stderr)
    assert "PAPER_PROSE_REWRITTEN=1" in out
    assert "prohibits LLM-generated text" in out
    # and it must not have overwritten the preprint PDF on its way out
    assert "wrote" not in out.split("PAPER_PROSE_REWRITTEN")[0][-200:]


def test_the_anonymous_mode_strips_every_identifying_field():
    """Source-level, because building twice per test run is expensive. Each of
    these is a place FAccT would desk-reject on."""
    src = SRC.read_text(encoding="utf-8")
    assert 'ANON = VENUE in ("facct", "anon", "anonymous")' in src
    assert '"Anonymous Author(s)" if ANON else "David Mao"' in src
    assert 'email=None if ANON else' in src
    # competing interests is inside `if not ANON`, which is what the guide requires
    i_guard = src.find("    if not ANON:")
    i_coi = src.find("<b>Competing interests.</b>")
    assert 0 < i_guard < i_coi, "the competing-interests block is not gated on ANON"
    # the gen-AI statement is NOT gated: FAccT requires it in the submission
    i_ai = src.find("<b>Generative AI usage.</b>")
    assert 0 < i_ai < i_guard, "the AI statement must be emitted in anonymous mode too"


def test_the_two_gen_ai_statements_say_opposite_things():
    """A rewrite flag that does not change the disclosure is a trap."""
    # the literals are split across source lines, so join the seams first
    src = SRC.read_text(encoding="utf-8")
    flat = " ".join(src.replace('"\n', '" ').split()).replace('" "', "")
    assert "No text in this paper was generated by a language model." in flat
    assert "assisted in drafting the prose of this document" in flat
    assert "The author wrote every sentence of this paper." in flat
    assert "if PROSE_REWRITTEN else" in src
    # the post-rewrite statement must not carry the not-compliant warning, and
    # the preprint one must -- that is the whole difference between them
    assert flat.count("would not satisfy a venue rule") == 1
