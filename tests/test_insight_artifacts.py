r"""The three insight artifacts, pinned to what the paper says from them.

Each analysis was drafted inside an adversarial audit workflow and
independently recomputed from its method description before adoption. These
tests pin the invariants the paper's new passages interpolate, so a re-run
that silently changes a conclusion fails here rather than shipping.
"""

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "paper-a" / "data"
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
sys.path.insert(0, str(ROOT / "paper-a" / "src"))


def _load(rel):
    p = DATA / rel
    if not p.exists():
        pytest.skip(f"{rel} not built")
    return json.loads(p.read_text(encoding="utf-8"))


def test_arm_asymmetry_summary_is_internally_consistent():
    d = _load("instrument/arm_asymmetry.json")
    s = d["summary"]
    assert s["n_combinations"] == len(d["results"])
    n_pos = sum(1 for r in d["results"] if r["D_ci"][0] > 0)
    n_neg = sum(1 for r in d["results"] if r["D_ci"][1] < 0)
    assert s["n_ci_excludes_zero_positive"] == n_pos
    assert s["n_ci_excludes_zero_negative"] == n_neg
    # The reversal is the honest part of the claim: whatever models reverse
    # must be named, and a model cannot be both supporting and reversed.
    for m in s["reversed_models"]:
        rs = [r for r in d["results"] if r["model"] == m]
        assert all(r["D_ci"][1] < 0 for r in rs), (
            f"{m} listed as reversed but not reversed in every dataset")


def test_crossmodel_family_pair_ranks_first_in_every_specification():
    d = _load("instrument/crossmodel_wording.json")
    for label, spec in d["specifications"].items():
        assert spec["family_rank_of_6"] == 1, (
            f"{label}: the same-family pair no longer ranks first; the "
            "appendix passage claims it does in every specification")
        assert spec["family_r"] > spec["cross_family_mean"]
    assert d["reliability_floor"]["value"] > 0.9, (
        "profile reliability fell; the 'not measurement error' sentence "
        "no longer holds")


def test_matrix_structure_counts_recompute_from_the_matrix():
    d = _load("reference/matrix_structure.json")
    m = json.loads((DATA / "reference" / "reporting_practice_matrix.json")
                   .read_text(encoding="utf-8"))
    audits = [s for s in m["studies"] if s["kind"] == "llm_hiring_audit"]
    rerun = d["minimal_rerun"]["criterion"]
    gaps = {s["label"]: [f for f in rerun
                         if s["cells"][f]["verdict"] in ("partial",
                                                         "not-reported")]
            for s in audits}
    assert d["minimal_rerun"]["n_meeting"] == \
        sum(1 for g in gaps.values() if not g)
    for f, flips in d["minimal_rerun"]["flips"].items():
        assert sorted(flips) == sorted(
            lab for lab, g in gaps.items() if g == [f])


@pytest.mark.skipif(not PDF.exists(), reason="paper not built")
def test_the_paper_states_what_the_artifacts_hold():
    import fitz
    with fitz.open(PDF) as doc:
        t = re.sub(r"\s+", " ", "\n".join(p.get_text() for p in doc))
    arm = _load("instrument/arm_asymmetry.json")["summary"]
    ms = _load("reference/matrix_structure.json")["minimal_rerun"]
    NUM = {0: "none", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
           7: "seven", 8: "eight"}
    n_flip = len(ms["flips"]["checkpoint_pinned"])
    assert f"pinning the checkpoint alone would bring " \
           f"{NUM.get(n_flip, n_flip)}" in t
    assert "Which arm is the unstable one?" in t
    assert "Is the wording effect shared across models?" in t
    # The reversal must be named in the passage, not hidden.
    if arm["reversed_models"]:
        assert "shows the reverse" in t
