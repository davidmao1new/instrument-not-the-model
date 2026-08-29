"""The methods supplement is going to the person who asked for it.

He listed four things and will check whether they are there and whether the
numbers in them are the paper's. A supplement that quietly disagrees with the
paper it supplements is worse than not sending one -- it converts a reviewer
into an auditor, and he has already shown he audits.

So: every headline figure in the PDF is re-derived here from the artifact that
produced it, and the four sections he asked for must all be present.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "paper-a" / "data"
PDF = ROOT / "paper-a" / "releases" / "Mao_methods_supplement.pdf"


def text() -> str:
    if not PDF.exists():
        pytest.skip("supplement not built")
    fitz = pytest.importorskip("fitz")
    with fitz.open(PDF) as doc:
        return " ".join(" ".join(p.get_text().split()) for p in doc)


def j(rel: str):
    p = D / rel
    if not p.exists():
        pytest.skip(f"{rel} missing")
    return json.loads(p.read_text(encoding="utf-8"))


def test_all_four_of_his_requests_are_answered():
    """He numbered them. Missing one is the difference between answering and
    appearing to answer."""
    t = text()
    for n, needle in [
        (1, "Design table"),
        (2, "Models, prompts, names, matching rule"),
        (3, "Sample breakdown"),
        (4, "pairing and resampling claims"),
    ]:
        assert needle.lower() in t.lower(), f"request {n} unanswered: {needle!r}"


def test_the_corpus_totals_match_the_paper():
    t = text()
    import sys
    sys.path.insert(0, str(ROOT / "paper-a" / "src"))
    B = pytest.importorskip("build_paper_v3")
    n_rec, _n_single, n_calls = B.corpus_size()
    assert f"{n_rec:,} matched pairs" in t
    assert f"{n_calls:,} model calls" in t


def test_the_randomisation_test_is_reported_as_run():
    """The pairing claim rests on it and he asked for it by name."""
    pf, t = j("names/pairing_freedom.json"), text()
    s = pf["summary"]
    assert f"{pf['n_perm']:,} random re-pairings" in t
    assert f"{s['min_perm_sd']:.4f} to {s['max_perm_sd']:.4f}" in t
    assert f"{s['min_best_worst_range']:.3f} to {s['max_best_worst_range']:.3f}" in t
    assert str(pf["seed"]) in t, "the seed is not stated"


def test_the_bootstrap_comparison_matches_the_artifact():
    r, t = j("delta_stability/resampling_unit.json"), text()
    ps = r["pooled_summary"]
    assert f"{r['n_boot']:,} replicates" in t
    assert f"{ps['min_ratio']:.2f} to {ps['max_ratio']:.2f}" in t
    assert str(r["seed"]) in t


def test_every_one_of_the_twelve_prompts_is_printed():
    """The paper's claim is about exact strings. A described prompt is not the
    prompt -- which is precisely the defect the supplement points out in one of
    the surveyed papers, so getting it wrong here would be embarrassing."""
    import sys
    sys.path.insert(0, str(ROOT / "paper-a" / "src"))
    E = pytest.importorskip("experiment_delta_stability")
    t = text()
    for k in [f"S{i}" for i in range(1, 7)]:
        ask = " ".join(E.VARIANTS[k]["ask"].split())
        assert ask in t, f"{k}'s request text is not printed"
    for k in [f"N{i}" for i in range(1, 7)]:
        note = " ".join(E.VARIANTS[k].get("note", "").split())
        assert note and note in t, f"{k}'s perturbation is not described"


def test_the_per_model_breakdown_is_complete():
    s2, t = j("delta_stability/study2_v2.json"), text()
    import sys
    sys.path.insert(0, str(ROOT / "paper-a" / "src"))
    B = pytest.importorskip("build_paper_v3")
    for m in [x for x in B.SHORT if x in s2]:
        assert B.SHORT[m] in t, f"{m} missing from the sample breakdown"
        assert f"{s2[m]['ps_sd_across_wordings']:.4f}" in t, (
            f"{m}'s across-wording SD is not reported")


def test_the_supplement_concedes_what_the_paper_concedes():
    """Three of his objections changed the paper. If the supplement restates
    the old, wider claims it reads as though nothing was taken on board."""
    t = text().lower()
    import re as _re
    # "the difference of the two means" in the paper, "...two group means" in
    # the supplement. Both are right; the test should not pick a favourite.
    assert _re.search(r"difference of the two (group )?means", t), (
        "the mean-paired-difference concession is missing")
    assert "follows the assignment process" in t, (
        "the resampling-unit principle is missing")
    assert "prespecified block" in t, (
        "the blocked-design scope condition is missing")


def test_nothing_is_silently_excluded():
    """A supplement that lists retained pairs without stating the exclusion
    rule invites the obvious question."""
    assert "NOTHING IS EXCLUDED FROM THE MAIN ANALYSES" in text()
