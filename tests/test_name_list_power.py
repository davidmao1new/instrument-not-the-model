r"""The name-list power result, pinned.

A statistician's review of 24 August 2026 said the restricted analysis cannot
establish a disparity at three clusters, and that the defensible claim is about
the list rather than about the models. `experiment_name_list_power.py` measured
whether a larger list could rescue it. It cannot, and the reason is a mechanism
rather than bad luck: the Black arm of the Bertrand and Mullainathan list costs
about one more token than the white arm on every tokenizer in the panel, so the
two length distributions barely overlap.

These numbers go into a letter to a statistician who will check them, and into
the paper. They are pinned here so a rebuild cannot move them silently.
"""

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
POWER = ROOT / "paper-a" / "data" / "instrument" / "name_list_power.json"
GRID = ROOT / "paper-a" / "data" / "instrument" / "token_balanced_grid.json"


@pytest.fixture(scope="module")
def power():
    if not POWER.exists():
        pytest.skip("name_list_power.json not built")
    return json.loads(POWER.read_text(encoding="utf-8"))


def test_the_panel_matching_reproduces_the_published_three(power):
    """The experiment must recover the number already in the paper.

    token_balanced_grid.json reports 2 female and 1 male first-name pairs. A
    separate measurement that disagreed would mean one of the two is wrong.
    """
    assert power["max_matching_panel_total"] == 3
    assert power["per_gender"]["female"]["max_matching_panel"] == 2
    assert power["per_gender"]["male"]["max_matching_panel"] == 1

    grid = json.loads(GRID.read_text(encoding="utf-8"))["max_matching"]
    assert grid["female_first"] == power["per_gender"]["female"]["max_matching_panel"]
    assert grid["male_first"] == power["per_gender"]["male"]["max_matching_panel"]


def test_the_black_arm_costs_about_one_more_token_on_every_tokenizer(power):
    """The mechanism. Without it the collapse looks like small-sample bad luck,
    and a bigger list would fix it. With it, a bigger list drawn the same way
    reproduces the same thin overlap."""
    gaps = power["race_gap_in_tokens"]
    assert len(gaps) == 4
    for model, v in gaps.items():
        assert v["gap"] > 0.5, (
            f"{model}: the Black arm is only {v['gap']} tokens longer, which "
            f"would make the overlap argument unsupported")
        assert v["black_mean"] > v["white_mean"]


def test_the_cross_model_requirement_is_what_costs_the_most(power):
    """Requiring balance on all four tokenizers, rather than one, is a bigger
    cut than tokenization itself. That is the trade a comparative design pays."""
    best_single = max(power["per_model"].values())
    assert best_single > power["max_matching_panel_total"], (
        "if the panel-wide matching equalled the best single-model matching, "
        "cross-model comparability would be free and there would be nothing "
        "to report")
    assert power["possible_pairs"] == 18
    assert best_single == 8


def test_the_extrapolation_is_labelled_as_one(power):
    """It assumes a larger list keeps the same length distribution. Said
    plainly, that is a real assumption; unsaid, it is a fabricated number."""
    ex = power["extrapolation"]
    assert "_caveat" in ex and "same length distribution" in ex["_caveat"]
    assert ex["names_per_cell_now"] == 9
    assert ex["matched_pairs_now"] == 3
