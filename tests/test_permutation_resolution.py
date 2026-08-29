r"""The randomization test's resolution floor, and the arithmetic under it.

The paper makes an impossibility claim: on the token-matched subset the
exact test cannot return a conventionally significant p-value whatever the
data show. That claim rests on counting sign assignments, so the counting
is pinned here against hand-checked founding cases, and the artifact is
pinned against the inputs it was derived from.
"""

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "paper-a" / "data"
sys.path.insert(0, str(ROOT / "paper-a" / "src"))

import analyze_permutation_resolution as apr  # noqa: E402

ART = DATA / "instrument" / "permutation_resolution.json"

pytestmark = pytest.mark.skipif(
    not ART.exists(), reason="permutation_resolution.json not built")


def _art():
    return json.loads(ART.read_text(encoding="utf-8"))


@pytest.mark.parametrize("n, assignments, one, two", [
    (1, 2, 0.5, 1.0),          # one pair: the test is vacuous
    (3, 8, 0.125, 0.25),       # the headline case, by hand
    (5, 32, 0.03125, 0.0625),  # last n that cannot clear 0.05 two-tailed
    (6, 64, 0.015625, 0.03125),  # first n that can
    (8, 256, 1 / 256, 2 / 256),
    (12, 4096, 1 / 4096, 2 / 4096),
])
def test_the_floor_is_two_to_the_minus_n(n, assignments, one, two):
    r = apr.floor_for(n)
    assert r["n_assignments"] == assignments
    assert r["p_min_one_tailed"] == pytest.approx(one)
    assert r["p_min_two_tailed"] == pytest.approx(two)
    assert r["can_reach_alpha"] is (two <= apr.ALPHA)


def test_six_pairs_is_the_smallest_workable_design():
    """If this moves, the paper's "how many names would it take" number
    moves with it and the prose must be re-read."""
    assert apr.smallest_workable_n() == 6
    assert not apr.floor_for(5)["can_reach_alpha"]
    assert apr.floor_for(6)["can_reach_alpha"]


def test_the_binding_checkpoint_cannot_reach_significance():
    a = _art()
    head = a["per_model_design_as_run"][a["fewest_pairs_model"]]
    assert head["can_reach_alpha"] is False, (
        "the paper claims the exact test forecloses an answer here")
    assert a["fewest_pairs_floor_two_tailed"] > apr.ALPHA
    # the floor must agree with the pair count it is derived from
    assert head["p_min_two_tailed"] == pytest.approx(
        apr.floor_for(head["n_pairs"])["p_min_two_tailed"])


def test_the_slope_checkpoint_is_not_the_binding_one():
    """The bug this pins. The prose once said the floor applied to "the
    checkpoint carrying the significant length slope". It does not: that
    checkpoint has more surviving pairs and its exact test CAN reach 0.05.
    Keying a sentence on the wrong one understated the design by a factor
    of thirty in the p-value it can attain."""
    a = _art()
    slope = a["slope_significant_models"]
    if not slope:
        pytest.skip("no checkpoint carries a significant clustered slope")
    fewest = a["fewest_pairs_model"]
    assert fewest not in slope, (
        "the fewest-pairs checkpoint now also carries the slope; the prose "
        "distinction between them can be simplified, but check it first")
    for m in slope:
        rec = a["per_model_design_as_run"][m]
        assert rec["can_reach_alpha"], (
            f"{m} carries the slope and cannot reach alpha; the paper's "
            "framing of which checkpoint is blocked needs revisiting")


def test_the_artifact_agrees_with_its_sources():
    """No number in the artifact is independent of the artifacts it was
    built from; drift in either source must fail here, not in the PDF."""
    a = _art()
    nle = json.loads((DATA / "instrument" / "name_length_effect.json")
                     .read_text(encoding="utf-8"))
    for model, rec in a["per_model_design_as_run"].items():
        tm = nle[model]["token_matched_first_name_clustered"]
        assert rec["n_pairs"] == tm["n_matched_clusters"]
        assert rec["n_clusters_before_matching"] == tm["n_grid_clusters"]
    power = json.loads((DATA / "instrument" / "name_list_power.json")
                       .read_text(encoding="utf-8"))
    assert a["source_list_panel_maximum"]["n_pairs"] == \
        power["max_matching_panel_total"]
    ex = power["extrapolation"]
    assert a["remedy"]["names_per_cell_needed"] == pytest.approx(
        a["remedy"]["n_pairs_needed"]
        / ex["matched_pairs_per_name_per_cell"])


def test_matching_never_invents_pairs():
    """Sanity on the direction of the constraint: the matched subset is a
    subset, so it can never hold more clusters than the grid it came
    from."""
    a = _art()
    for rec in a["per_model_design_as_run"].values():
        assert rec["n_pairs"] <= rec["n_clusters_before_matching"]
