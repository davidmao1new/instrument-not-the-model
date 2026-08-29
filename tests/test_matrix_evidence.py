"""The survey's claims about other people's papers stay true.

Section 8 makes 22 claims about each of 13 other researchers' papers. About 140
of those cells rest on a NEGATIVE search -- "grep -i for 'cache' (0 hits)" --
and a wrong negative claim about somebody else's work is the most damaging
thing in this project to get wrong. It is also invisible: the cell renders, the
count sums, the table prints.

Those searches were run once, by readers, and nothing re-ran them until
`audit_matrix_evidence.py` existed. When it did, it found no coding error in
414 quotations and 658 negative searches -- but it took five iterations to get
there, because the first four runs drowned the real question in false alarms
from lossy PDF extraction. What these tests pin is the machinery that made the
answer trustworthy, since a regression in any of it silently turns the audit
back into noise.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
sys.path.insert(0, str(SRC))


def A():
    if not (SRC / "audit_matrix_evidence.py").exists():
        pytest.skip("audit not present")
    return pytest.importorskip("audit_matrix_evidence")


# ---------------------------------------------------------------------------
# Normalisation. Each of these is a real extraction defect found in lit/text/.

def test_line_wrap_hyphenation_is_rejoined():
    """"hyperparame- ters" is one word the PDF broke across a line. Comparing
    without rejoining reported 151 correctly-transcribed quotes as missing."""
    a = A()
    assert "hyperparameters" in a.norm("hyperparame- ters")
    # A real compound has no space after its hyphen and must survive as two
    # tokens, or "state-of-the-art" would collapse to one.
    assert a.norm("state-of-the-art") == "state of the art"


def test_digit_full_stop_ligatures_are_folded():
    """bm2004.txt carries 5,617 of these: the scan's font maps "8." to the
    single glyph U+248F, so every number in the paper that anchors this
    literature is unrecoverable without folding."""
    a = A()
    # "825-838." renders as "825-83⒏". Folded, the trailing digit comes back;
    # unfolded it is dropped as a non-ASCII character and the page range reads
    # 825-83, which is a different and wrong number.
    assert a.norm("825-83⒏") == "825 838", a.norm("825-83⒏")
    assert a.norm("pages 12-1⒉") == "pages 12 12."[:0] + "pages 12 12", \
        a.norm("pages 12-1⒉")
    # norm() strips punctuation on purpose, so a decimal becomes two tokens on
    # BOTH sides of every comparison and nothing is lost by it.
    assert a.norm("9.65 percent") == "9 65 percent"


def test_a_search_term_is_a_word_not_whatever_sits_between_two_quotes():
    """On "grep -i for 'snapshot' (0 hits), 'checkpoint' (0)" an unrestricted
    pattern matched from one term's closing quote to the next term's opening
    quote and captured "(0), " as a search term. It then "appeared" 875 times
    in every paper and buried the real findings."""
    a = A()
    terms = a.negatives("Searched: grep -i for 'snapshot' (0 hits), "
                        "'checkpoint' (0), 'revision' (0)")
    assert "snapshot" in terms and "checkpoint" in terms
    for junk in terms:
        assert "(" not in junk and ")" not in junk and ":" not in junk, \
            f"{junk!r} is punctuation, not a search term"


def test_commentary_is_not_held_to_a_verbatim_standard():
    """The readers used quotation marks for their own phrasing too. Treating
    "missing: the definition of the unit" as a claim about what a paper says
    is a false accusation against the reader."""
    a = A()
    for s in ("missing: the definition of the unit inside a replicate",
              "grep -i for 'top-p' (0), 'seed' (0)",
              "column header of Table 5, not numerical precision"):
        assert a.COMMENTARY.search(s), f"not classified as commentary: {s!r}"
    for s in ("We carry out our experiments using five instruction-tuned LLMs",
              "the mean paired difference is the difference of the two means"):
        assert not a.COMMENTARY.search(s), f"wrongly classified: {s!r}"


def test_the_commentary_pattern_compiled_as_a_regex_not_control_characters():
    """A patch once wrote literal backspace bytes where \\b word boundaries
    were meant, and the pattern silently matched nothing at all -- the audit
    reported no commentary and every finding came back. A regex that matches
    nothing looks exactly like a clean run."""
    a = A()
    assert "\x08" not in a.COMMENTARY.pattern, \
        "word boundaries were written as backspace characters"
    assert a.COMMENTARY.search("grep"), "the pattern matches nothing"


# ---------------------------------------------------------------------------
# The matching itself

def test_subsequence_matching_tolerates_interleaved_columns():
    """Two-column PDFs splice the right column into the middle of the left.
    An et al.'s model list extracts with a footnote URL and a table fragment
    inside the sentence."""
    a = A()
    text = ("we carry out our experiments using five state of the art models "
            "1 https www ssa gov babynames 7 occupational roles "
            "and gpt 3 5 turbo ouyang et al 2022").split()
    idx: dict[str, list[int]] = {}
    for i, w in enumerate(text):
        idx.setdefault(w, []).append(i)
    q = ("we carry out our experiments using five state of the art models "
         "and gpt 3 5 turbo ouyang et al 2022")
    assert a.subsequence_cover(q, text, idx) >= a.MIN_COVER


def test_subsequence_matching_still_rejects_an_absent_sentence():
    """The tolerance must not make everything match. A sentence that is not
    there has to fail, or the audit is decorative."""
    a = A()
    text = "the model was evaluated on a held out set of resumes".split()
    idx: dict[str, list[int]] = {}
    for i, w in enumerate(text):
        idx.setdefault(w, []).append(i)
    q = "we clustered standard errors at the vacancy level using a wild bootstrap"
    assert a.subsequence_cover(q, text, idx) < a.MIN_COVER


def test_despacing_fixes_small_caps_and_hyphenation_but_needs_length():
    a = A()
    assert a.despace("non reasoning") in a.despace("the rmse of nonreasoning models")
    assert a.despace("f ormat s pread") == a.despace("formatspread")


# ---------------------------------------------------------------------------
# Scope

def test_a_reference_list_is_not_the_paper():
    """B&M's extraction carries a thousand-entry citing-articles list, which is
    how a claim that the paper never says "ethics" collides with the Journal of
    Business Ethics."""
    a = A()
    t = ("the methods section says nothing about caching " + "x " * 400
         + "references smith 2020 journal of cache studies")
    # The old form of the first assertion ended in `or True`, which
    # operator precedence turns into a tautology -- the one property this
    # test exists to pin (negative searches must not run over the reference
    # list) was not tested at all.
    assert "journal of cache studies" not in a.body_of(t), (
        "body_of() failed to cut the reference list; a negative claim about "
        "another researcher's paper could collide with a journal title")
    assert len(a.body_of(t)) < len(t), "the reference list was not cut"


def test_extraction_damage_is_measured_not_asserted():
    a = A()
    clean = a.degradation("A perfectly ordinary sentence about resumes. "
                          "Another one follows it here.")
    assert not clean["degraded"]
    assert a.degradation("page 825-83⒏ and 12-1⒉ " * 60)["degraded"]


def test_every_surveyed_study_has_a_text_to_check_against():
    """A study with no source silently skips every check on it.

    SKIPS IN THE PUBLIC RELEASE. `lit/text/` holds extracted full text of
    thirteen copyrighted papers and is not redistributable, so a clone has none
    of it. That is a property of what may lawfully be shipped, not a defect in
    the codings -- and a test that fails for a reason nobody can fix teaches
    readers to ignore the suite.
    """
    a = A()
    texts = a.load_texts()
    if not texts:
        pytest.skip("no lit/text/ -- expected in the public release, which "
                    "cannot redistribute copyrighted full texts")
    missing = sorted(set(a.SOURCES) - set(texts))
    assert not missing, f"no extracted text for {missing}"


def test_the_audit_runs_clean_enough_to_be_read():
    """The point of the five iterations. If the residual grows back past a few
    dozen, the signal is buried again and the audit stops being used."""
    a = A()
    texts = a.load_texts()
    tok = {}
    for lab, tri in texts.items():
        full = tri[2]
        ft = full.split()
        idx: dict[str, list[int]] = {}
        for i, w in enumerate(ft):
            idx.setdefault(w, []).append(i)
        tok[lab] = (ft, idx)
    fails = 0
    for s in a.studies():
        lab = s["label"]
        if lab not in texts or a.DEGRADED.get(lab, {}).get("degraded"):
            continue
        full = texts[lab][2]
        ft, idx = tok[lab]
        for _f, c in s["cells"].items():
            for q in a.QUOTED.findall(str(c.get("evidence") or "")):
                qn = a.norm(q)
                if len(qn.split()) < 6 or "…" in q or "..." in q or "[" in q:
                    continue
                if a.COMMENTARY.search(q):
                    continue
                if qn in full or a.defootnote(qn) in a.defootnote(full):
                    continue
                if a.despace(qn) in a.despace(full):
                    continue
                if a.subsequence_cover(qn, ft, idx) >= a.MIN_COVER:
                    continue
                fails += 1
    assert fails <= 30, (
        f"{fails} quotations no longer verify. Every one of the 26 that "
        f"remained on 2026-08-19 was checked by hand and was an extraction "
        f"artifact, a mangled formula, or reader commentary. A jump above that "
        f"means either a coding changed or the normalisation regressed.")
