"""Tests for the three defects found in the round-4 critique.

Each of these shipped in a released PDF and passed the whole suite on the way
out, so each gets a test that fails if it comes back.

  * an f-string prefix on the wrong fragment of a concatenation, printing
    `{pct(min(sat.values()), 1)}` to the page in v6, v7 and v8;
  * a matched-pair predicate that keyed on the margin, so 220 paired rows from
    a marginless endpoint were scored as single prompts costing one call;
  * the widening that fixes it, which must not sweep in the outcome probe's
    white/black NAME fields.

The emphasis is on the property that makes each defect a defect, not on the
current values -- a test that asserts 31,468 breaks every time a study is
added, and would not have caught any of these.
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
DATA = ROOT / "paper-a" / "data"
sys.path.insert(0, str(SRC))

import analyze_corpus_size as cs  # noqa: E402

BRACE = re.compile(r"\{[^{}]{0,200}\}")


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- leaks ----
def _plain_string_constants(path):
    """Every string literal in `path` that is not an f-string or a docstring.

    Braces inside an f-string's literal half are already doubled by the parser
    and cannot leak; braces in a docstring are prose about the bug.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            if ast.get_docstring(node, clean=False) is not None and node.body:
                docs.add(id(node.body[0].value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            continue
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docs):
            out.append((node.lineno, node.value))
    return out


def test_no_plain_literal_in_the_builder_carries_a_placeholder():
    """The defect itself: a brace in a non-f string reaches the page verbatim.

    Scoped to the builder, where a literal brace has no legitimate use. The
    same rule over the whole source tree would be noise -- prompt templates and
    regexes are full of braces on purpose.
    """
    bad = [(n, m.group(0)) for n, v in _plain_string_constants(
        SRC / "build_paper_v3.py") for m in BRACE.finditer(v)]
    assert not bad, (
        "a plain string literal in the builder carries a format placeholder; "
        f"the `f` prefix is on a different fragment: {bad}")


def test_the_detector_fires_on_the_shape_that_shipped():
    """Guard the guard. A check that cannot fail is worse than no check.

    This is the exact construction from v6 -- `f""` glued to a plain string
    holding the braces -- and the detector must see it.
    """
    import tempfile
    src = 'x = f"" + g("k") + " ranges from {pct(v)} to "\n'
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(src)
        p = pathlib.Path(fh.name)
    try:
        hits = [m.group(0) for _, v in _plain_string_constants(p)
                for m in BRACE.finditer(v)]
        assert "{pct(v)}" in hits
    finally:
        p.unlink()


def test_the_rendered_pdf_has_no_placeholder_on_any_page():
    """The artifact check, which catches the defect whatever produced it."""
    pdf = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
    if not pdf.exists():
        pytest.skip("paper not built")
    fitz = pytest.importorskip("fitz")
    leaks = []
    with fitz.open(pdf) as doc:
        for pno, page in enumerate(doc, 1):
            leaks += [(pno, m.group(0)) for m in BRACE.finditer(page.get_text())]
    assert not leaks, f"unevaluated placeholder printed to the page: {leaks}"


# --------------------------------------------------------------- corpus ----
def test_a_verdict_only_pair_counts_as_a_pair():
    """Both arms measured is the criterion, not measured on a given scale."""
    row = {"white_raw": "Yes.", "black_raw": "No.", "white": "yes",
           "black": "no", "http_white": 200, "http_black": 200}
    assert cs.is_matched_pair(row)


def test_a_margin_pair_still_counts_as_a_pair():
    assert cs.is_matched_pair({"white_margin": 1.0, "black_margin": 0.5})


def test_a_censored_margin_pair_is_a_pair_but_not_strictly_measured():
    """Present-and-null is the censoring signature; it stays a pair."""
    row = {"white_margin": None, "black_margin": 0.5}
    assert cs.is_matched_pair(row)
    assert not cs.is_matched_pair_strict(row)
    assert not cs.has_no_margin_field(row)


def test_an_absent_margin_is_distinguished_from_a_censored_one():
    """Rolling the two together would report a censoring rate we do not have."""
    assert cs.has_no_margin_field({"white_raw": "Yes.", "black_raw": "No."})
    assert not cs.has_no_margin_field({"white_margin": None,
                                       "black_margin": None})


def test_the_outcome_probe_rows_are_not_swept_up_as_pairs():
    """The near-miss that makes the widening delicate.

    smoke/outcome_probe.jsonl uses `white` and `black` to hold the two NAMES in
    a forced choice. A predicate that accepted a bare white/black pair would
    score three single prompts as pairs -- so the fallback stops at the raw
    per-arm response, and this pins that.
    """
    assert not cs.is_matched_pair({"probe": "choice", "white": "Allison Baker",
                                   "black": "Jamal Williams", "fwd": "?",
                                   "rev": "?"})


def test_a_single_prompt_row_is_not_a_pair():
    assert not cs.is_matched_pair({"margin": 0.2, "name": "Greg Murphy"})


def test_single_prompt_records_are_exactly_the_three_expected_studies():
    """The invariant that shows the widening went far enough and no further.

    EXPECTED_SINGLE_PROMPT is written down independently of any count. When
    `frontier` was miscounted, the total exceeded the sum over those three
    studies by exactly the 220 misclassified pairs -- and the reconciliation
    check passed only because `frontier` had been added to an exemption list.
    """
    c = _load(DATA / "reference" / "corpus_size.json")
    by = c["by_study"]
    named = sum(by[k]["n_single_prompt_records"]
                for k in cs.EXPECTED_SINGLE_PROMPT if k in by)
    assert c["n_single_prompt_records"] == named
    for k, s in by.items():
        if k not in cs.EXPECTED_SINGLE_PROMPT:
            assert s["n_single_prompt_records"] == 0, (
                f"{k} has single-prompt rows and is not one of the studies "
                f"that produce them")


def test_calls_reconcile_with_two_per_pair():
    c = _load(DATA / "reference" / "corpus_size.json")
    assert (2 * c["n_matched_pair_records"] + c["n_single_prompt_records"]
            == c["n_model_calls"])
    assert all(c["reconciliation"].values()), c["reconciliation"]


def test_the_builder_and_the_analyser_use_the_same_predicate():
    """They are deliberately duplicated, so they are checked for drift.

    The builder cannot import the analyser -- it must run while the analyser is
    mid-edit -- so the predicate is written twice. Two copies that disagree is
    how the corpus and the call count would silently part company.
    """
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    assert '"white_raw" in r and "black_raw" in r' in src
    assert '"white_margin" in r and "black_margin" in r' in src


# ------------------------------------------------------- the verdict arm ----
def test_the_verdict_arm_is_constant_within_a_template():
    """The finding §4.7 rests on, stated as a property rather than a number.

    If a later re-run makes any template produce both verdicts, the claim that
    the threshold has no resolution is no longer true and the paragraph has to
    change.
    """
    v = _load(DATA / "frontier" / "frontier_verdict_analysis.json")
    assert v["verdict_is_a_function_of_the_template_alone"]
    for t, rec in v["by_template"].items():
        assert len(rec["verdicts_observed"]) == 1, t
        assert rec["advance_rate"] in (0.0, 1.0), t
    assert v["n_pairs_where_the_arms_disagree"] == 0
    assert v["max_wording_spread_within_any_template"] == 0.0


def test_the_repeated_cells_agree_so_constancy_is_not_noise():
    v = _load(DATA / "frontier" / "frontier_verdict_analysis.json")
    assert v["n_cells_run_twice"] > 0
    assert v["n_repeated_cells_disagreeing"] == 0


def test_the_verdict_arm_pairs_are_in_the_corpus_total():
    """The arm is reported AND counted, which is what went wrong before."""
    v = _load(DATA / "frontier" / "frontier_verdict_analysis.json")
    c = _load(DATA / "reference" / "corpus_size.json")
    assert c["by_study"]["frontier"]["n_single_prompt_records"] == 0
    assert c["n_pair_rows_with_no_margin_field"] == v["n_pairs"]
    assert v["n_model_calls"] == 2 * v["n_pairs"]


def test_the_model_reads_the_name_it_does_not_act_on():
    """Evidence for 'no resolution' over 'no effect', checked not asserted."""
    v = _load(DATA / "frontier" / "frontier_verdict_analysis.json")
    assert v["n_rationales_naming_the_candidate"] > 0
