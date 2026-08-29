"""The delimiter column in Table 2 must be a measurement, not a declaration.

WHY THIS TEST EXISTS. `experiment_mechanism.CONDITIONS` carries, beside each
condition's prose note, an integer saying how many structural delimiters that
condition destroys. That integer is the experimenter's intent. For a while it
was also the only delimiter number the tokenizer probe exported per condition,
so the paper's Table 2 column headed "delimiters destroyed" certified the design
against the intent that built it. The probe was measuring the real quantity the
whole time and no consumer read it.

On the reference checkpoint the two disagree on exactly one condition, and it is
the one the design can least afford: D7 is declared to destroy nothing, and
measured it removes BOTH merged period-plus-break tokens, because the string it
substitutes in is its own single vocabulary item rather than the one H_delim
names. D7 is nevertheless not the same edit as D6 -- it SUBSTITUTES a delimiter
where D6 FRAGMENTS one -- which is why the artifact records a disposition and
not just a count.

Two things are locked here. The first is a property of the code alone and needs
no server: what D7's edit does to the prompt at the character level. The second
is a property of the stored artifact: that both columns are present for every
condition, that they are internally consistent, and that the set of conditions
where they disagree is exactly {D7}. If someone re-declares D7, or drops the
measured column, or re-runs the probe on a checkpoint that merges differently,
this fails loudly instead of the paper quietly printing an intent as a
measurement again.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
sys.path.insert(0, str(SRC))

import stimuli as st  # noqa: E402
from experiment_mechanism import CONDITIONS, build  # noqa: E402

ARTIFACT = (ROOT / "paper-a" / "data" / "instrument"
            / "condition_tokens_llama-3.1-8b-instruct.json")

NAME = "Allison Baker"
BODY_KEY = "T1_strong"


# --------------------------------------------------------------------------
# What D7 does, from the code alone
# --------------------------------------------------------------------------
def test_d7_substitutes_every_paragraph_break_and_fragments_none():
    """D7 replaces both breaks with a longer one; it splits neither."""
    body = st.TEMPLATES[BODY_KEY]
    base, d7 = build("D0", NAME, body), build("D7", NAME, body)
    assert base.count("\n\n") == 2
    # every break became a triple, and none became the fragmented form D4-D6 use
    assert d7.count("\n\n\n") == 2
    assert "\n \n" not in d7
    # and no plain double break survives that is not part of a triple
    assert d7.replace("\n\n\n", "") .count("\n\n") == 0


def test_d7_is_not_the_same_edit_as_d6():
    """Both remove the merged token; only one fragments the boundary."""
    body = st.TEMPLATES[BODY_KEY]
    d6, d7 = build("D6", NAME, body), build("D7", NAME, body)
    assert d6 != d7
    assert d6.count("\n \n") == 2 and "\n \n" not in d7


# --------------------------------------------------------------------------
# What the stored artifact carries
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT.exists():
        pytest.skip(f"{ARTIFACT.name} not present; run probe_condition_tokens.py")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_covers_every_condition(artifact):
    assert set(artifact["conditions"]) == set(CONDITIONS)


@pytest.mark.parametrize("cond", sorted(CONDITIONS))
def test_both_columns_present_and_declared_matches_the_source(artifact, cond):
    r = artifact["conditions"][cond]
    for key in ("n_delims_destroyed_declared", "n_delims_destroyed_measured",
                "n_delims_fragmented_measured", "n_delims_substituted_measured",
                "delimiter_disposition", "declared_matches_measured"):
        assert key in r, f"{cond} is missing {key}"
    # the declared column must still be the literal it always was
    assert r["n_delims_destroyed_declared"] == CONDITIONS[cond][1]
    assert r["n_delims_destroyed"] == CONDITIONS[cond][1]


@pytest.mark.parametrize("cond", sorted(CONDITIONS))
def test_measured_column_is_internally_consistent(artifact, cond):
    """measured destroyed = baseline count - surviving count, and the
    disposition accounts for every destroyed delimiter."""
    r = artifact["conditions"][cond]
    assert (r["n_merged_delims_baseline"] - r["n_merged_delims"]
            == r["n_delims_destroyed_measured"])
    assert (r["n_delims_fragmented_measured"] + r["n_delims_substituted_measured"]
            == r["n_delims_destroyed_measured"])
    assert (r["declared_matches_measured"]
            is (r["n_delims_destroyed_measured"] == r["n_delims_destroyed_declared"]))


def test_the_only_disagreement_is_d7(artifact):
    a = artifact["delimiter_audit"]
    assert set(a["disagreements"]) == {"D7"}
    assert a["disagreements"]["D7"]["declared"] == 0
    assert a["disagreements"]["D7"]["measured"] == 2
    assert a["disagreements"]["D7"]["disposition"] == "substituted"


def test_d7_is_not_a_null_control_once_measured(artifact):
    a = artifact["delimiter_audit"]
    assert "D7" in a["null_controls_declared"]
    assert "D7" not in a["null_controls_measured"]
    assert a["reclassified"] == ["D7"]


def test_the_fragmenting_conditions_are_unaffected_by_the_correction(artifact):
    """D4, D5, D6 and D10 were right all along; the fix must not move them."""
    for cond in ("D4", "D5", "D6", "D10"):
        r = artifact["conditions"][cond]
        assert r["declared_matches_measured"] is True
        assert r["delimiter_disposition"] == "fragmented"
