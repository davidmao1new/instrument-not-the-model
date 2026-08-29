"""Tests for the claims round 7's reconcile pass changed.

Forty-four confirmed findings, and the ones worth pinning are not the typos.
They fall into three kinds, and each kind gets a different sort of assertion.

  A statistic that was computed wrongly.  The pooled within-race bootstrap
  resampled frozen ranks without stratifying by race, which understated its own
  spread and produced the only "excludes zero" result in the construct-validity
  check. What is pinned is not the corrected number but the PROPERTY that makes
  it correct: a resampling interval and an exact permutation null estimate the
  same spread, so they have to agree. They disagreed by a third.

  A claim that outran the artifact under it.  Three of these were directional
  assertions inherited from a docstring or an earlier calibration and never
  re-derived: the sign of the N1 bias, the ordering of dispersion against
  effect, the size of the delimiter floor. Each is pinned against the artifact
  that decides it, so if a re-run moves the artifact the test fails rather than
  the prose quietly becoming false.

  A number that was scoped to the wrong set.  The three-rule disagreement rate
  was a minimum over all six models inside a sentence about the bottom of the
  panel. These are recomputed here from the raw coverage files, independently of
  the builder, because a builder bug cannot be caught by reading the builder.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
DATA = ROOT / "paper-a" / "data"
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


def _paper():
    if not PDF.exists():
        pytest.skip("paper not built")
    fitz = pytest.importorskip("fitz")
    with fitz.open(PDF) as doc:
        return " ".join(" ".join(p.get_text().split()) for p in doc)


def _cval():
    return _load(DATA / "names" / "construct_validity.json")


def _live(d):
    return {k: v for k, v in d.items()
            if not k.startswith("_") and isinstance(v, dict)}


# ------------------------------------------- the bootstrap and its own null ----
def test_the_within_race_bootstrap_agrees_with_the_exact_null():
    """The property the old bootstrap violated, stated as a property.

    A percentile interval and a permutation null are two routes to the same
    spread. If they disagree by a third, one of them is wrong -- and the one
    that was wrong is the one that produced a result. Printing both is what
    makes the error visible; asserting they agree is what keeps it visible.
    """
    seen = 0
    for m, v in _cval()["models"].items():
        for key in ("q1_perception", "q2_callback", "q3_ses"):
            w = (v.get(key) or {}).get("pooled_within_race")
            if not w:
                continue
            seen += 1
            assert w["null_sd"] > 0, (m, key)
            ratio = w["boot_sd"] / w["null_sd"]
            assert 0.75 < ratio < 1.25, (
                f"{m}/{key}: bootstrap SD {w['boot_sd']:.3f} against exact null "
                f"SD {w['null_sd']:.3f}; the resample is not describing its own "
                f"statistic")
    assert seen >= 9, "too few within-race statistics found; this pins nothing"


def test_the_within_race_statistic_is_stratified_and_re_ranked():
    src = (SRC / "analyze_construct_validity.py").read_text(encoding="utf-8")
    assert "def pooled_within_race_ci" in src
    # the two defects, both absent: an unstratified draw and a frozen rank
    body = src.split("def pooled_within_race_ci", 1)[1].split("\ndef ", 1)[0]
    assert "rng.integers(0, len(bx), (n_boot, len(bx)))" in body, (
        "the bootstrap no longer draws within a race block")
    assert "stats.rankdata(bx[idx], axis=1)" in body, (
        "the bootstrap no longer re-ranks inside the draw")
    assert "p_perm" in body


def test_every_within_race_check_is_null_on_every_checkpoint():
    """If a re-run ever produces one that is not, the prose has to change."""
    s = _cval()["summary"]
    for k in ("q1_perception_within_race", "q2_callback_within_race",
              "q3_ses_within_race"):
        assert s[k]["n_excluding_zero"] == 0, k
        assert s[k]["n_p_perm_below_05"] == 0, k
    t = _paper()
    assert "the interval excludes zero" not in t
    assert "it points AGAINST us" not in t


def test_the_manipulation_check_uses_perception_and_not_callback():
    """§4.3 filled the manipulation-check slot with the criterion statistic.

    The distinction is the whole point of the section: perception is what the
    name signals, callback is what employers did. Only the first can validate
    the manipulation.
    """
    cv = _cval()
    for m, v in cv["models"].items():
        assert v["q1_perception"].get("pooled_within_race"), m
    t = _paper()
    # Case-insensitive for the same reason as test_round4_claims: the
    # sentence now starts here, so the "h" is a capital.
    assert "how distinctly a name signals the race it was chosen for" \
        in t.lower()
    rhos = [v["q1_perception"]["pooled_within_race"]["rho"]
            for v in cv["models"].values()]
    for r in rhos:
        assert f"{r:+.2f}" in t, f"{r:+.2f} is not on the page"


def test_the_within_race_perception_predictor_is_signed():
    """Unsigned, the two race blocks predict opposite movements and cancel."""
    src = (SRC / "analyze_construct_validity.py").read_text(encoding="utf-8")
    assert 'q1["pooled_within_race"] = pooled_within_race_ci(signed, m, race, rng)' \
        in src


# ------------------------------------------------------ §4.1's ordering claim ----
def test_section_4_1_makes_no_between_model_claim():
    """Two within-model magnitudes cannot license a between-model statement,
    and §1 explicitly declares the reverse false."""
    t = _paper()
    assert "which wording the researcher wrote matters more than which model" \
        not in t
    assert "Two magnitudes compared directly say that" not in t


def test_the_dispersion_over_effect_ordering_carries_its_interval():
    du = _load(DATA / "delta_stability" / "dispersion_uncertainty.json")
    ci = du["models"]["mistral-7b-instruct-v0.1"]["ratio_ci"]
    assert ci[0] < 1.0 < ci[1], (
        "the interval no longer covers 1; the prose may state the ordering")
    t = _paper()
    assert f"[{ci[0]:.2f}, {ci[1]:.2f}]" in t
    assert "the ordering is where the point estimates put it" in t


# ------------------------------------------------------------ the N1 direction ----
def test_the_n1_bias_has_no_single_direction_and_the_paper_says_so():
    """Inherited verbatim from a docstring, wrong on three of four models."""
    arm = _load(DATA / "delta_stability" / "arm_contrast.json")
    live = {m: v for m, v in _live(arm) if isinstance(v, dict)} if False else \
        {m: v for m, v in arm.items() if not m.startswith("_")}
    down = sum(1 for v in live.values()
               if v["dropN1"]["sigma_null"][1] < v["all"]["sigma_null"][1])
    assert 0 < down < len(live), (
        "dropping N1 now moves every model the same way; the prose may state a "
        "direction again")
    t = _paper()
    assert "inclusion makes that arm look less dispersed than it is" not in t
    assert "has no single direction" in t


def test_the_fit_docstring_no_longer_asserts_the_direction():
    """The claim reached print by being copied out of here."""
    src = (SRC / "fit_arm_contrast.py").read_text(encoding="utf-8")
    head = src.split('"""', 2)[1]
    assert "Including it makes the null arm look" not in head
    assert "Do not\nput one back." in head or "Do not put one back." in \
        " ".join(head.split())


# ------------------------------------------------------------- the grid's floor ----
def test_the_name_grid_uses_its_own_noise_floor_not_5_2s():
    """§5.2's zero floor was measured at concurrency one with the cache off.
    Study 4 ran at the default concurrency with cache reuse on."""
    cv = _cval()
    ratios = []
    for m, v in cv["models"].items():
        n = v.get("replicate_noise")
        assert n, f"{m} has no replicate floor"
        assert n["n_replicate_pairs"] > 100, m
        ratios.append(n["noise_over_within_race_sd"])
    assert max(ratios) < 0.10, (
        "the grid's floor is no longer small against the spread it is used to "
        "license; §4.2's argument needs rechecking")
    t = _paper()
    assert "equal to zero once serving is controlled" not in t
    assert "of the within-race spread of per-name margins" in t


# ------------------------------------------------- scoping the disagreement rate ----
def test_the_three_rule_disagreement_is_scoped_to_the_models_it_describes():
    """Recomputed from the raw coverage files, not from the builder.

    The sentence is about the models whose unconstrained next token is usually
    neither yes nor no. The minimum used to run over all six rows, which took
    its number from a model at 62 % yes/no mass -- printed on its own row of
    Table 1, so the paper contradicted itself on one page.
    """
    import statistics
    files = sorted((DATA / "instrument").glob("token_coverage_*.json"))
    if not files:
        pytest.skip("no coverage artifacts")
    mass, agree = {}, {}
    for f in files:
        rows = json.loads(f.read_text(encoding="utf-8"))
        k = f.stem.replace("token_coverage_", "")
        mass[k] = statistics.fmean(r["addressed_mass"] for r in rows)
        agree[k] = statistics.fmean(
            float(r["grammar"] == r["argmax"] == r["mass"]) for r in rows)
    low = [k for k in agree if mass[k] < 0.5]
    assert low, "no low-mass models; the sentence has no referent"
    scoped = 1 - min(agree[k] for k in low)
    panel = 1 - min(agree.values())
    assert scoped < panel, (
        "the panel minimum now belongs to a low-mass model; the scoping is "
        "still correct but this test no longer distinguishes the two")
    t = _paper()
    assert f"disagree on up to {round(scoped * 100)} % of probes" in t
    assert f"disagree on up to {round(panel * 100)} % of probes" not in t


# ------------------------------------------------------------ the probe partition ----
def test_the_frontier_probe_partition_closes():
    f = DATA / "instrument" / "frontier_api_capability_v2.json"
    d = _load(f)
    parts = d["n_reachable"] + d["n_quota_blocked"] + d["n_unavailable"]
    assert parts == d["n_probed"], (parts, d["n_probed"])
    t = _paper()
    assert f"{d['n_unavailable']} could not be called at all" in t


# ------------------------------------------------------- the surname contrast ----
def test_the_first_name_surname_contrast_is_a_posterior_not_two_medians():
    nv = _load(DATA / "names" / "name_variance.json")
    live = {k: v for k, v in nv.items()
            if not k.startswith("_") and isinstance(v, dict)
            and "sigma_first_minus_last" in v}
    assert len(live) >= 4
    for m, v in live.items():
        d = v["sigma_first_minus_last"]
        assert d[0] <= d[1] <= d[2], m
        # the difference of the medians is not the median of the difference,
        # so this is a real second quantity and not a restatement
        assert abs(d[1] - (v["sigma_first"][1] - v["sigma_last"][1])) < 0.02, m
    n_excl = sum(1 for v in live.values()
                 if v["sigma_first_minus_last"][0] > 0)
    words = {1: "one", 2: "two", 3: "three", 4: "four"}
    t = _paper()
    assert f"the larger share on {words[n_excl]} of the" in t


# --------------------------------------------------------------- the references ----
def test_the_nghiem_entry_cites_the_version_that_is_on_disk():
    mx = _load(DATA / "reference" / "reporting_practice_matrix.json")
    ref = next(s["reference"] for s in mx["studies"]
               if s["label"] == "Nghiem et al. 2024")
    assert "Empirical Methods in Natural Language Processing" in ref
    assert "arXiv" not in ref
    # The reference must name the version actually read. The PDF itself is
    # copyrighted and is not redistributed, so its absence in a clone is
    # expected; what is checked above -- that the reference cites the EMNLP
    # proceedings rather than the arXiv preprint -- is the claim that matters
    # and it holds without the file.
    pdf = ROOT / "lit" / "nghiem_etal_2024_emnlp_name_based_bias.pdf"
    if not (ROOT / "lit").exists():
        pytest.skip("lit/ not present; copyrighted PDFs are not redistributed")
    assert pdf.exists()


def test_no_reference_disagrees_with_the_document_it_was_read_from():
    """Two independent records of one identifier were free to diverge, and did.

    This mirrors the guard now in build_reporting_matrix.py; keeping it here as
    well means a hand-edited artifact fails too.
    """
    mx = _load(DATA / "reference" / "reporting_practice_matrix.json")
    ax = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", re.I)
    for s in mx["studies"]:
        a = ax.search(s.get("reference") or "")
        b = ax.search(s.get("study_as_printed") or "")
        if a and b:
            assert a.group(0).replace(" ", "") == b.group(0).replace(" ", ""), \
                s["label"]


def test_the_reference_preamble_claims_only_what_was_done():
    t = _paper()
    assert "no peer-reviewed version was located" not in t
    assert "publisher records were not searched separately" in t


# ---------------------------------------------------------------- the abstract ----
def test_the_abstract_counts_a_dispersion_statistic_not_any_report():
    """Four of eight show per-wording estimates; none prints a dispersion."""
    mx = _load(DATA / "reference" / "reporting_practice_matrix.json")
    c = mx["counts"]["dispersion_across_wordings"]
    assert c["n_reported"] == 0
    assert c["n_partial"] > 0, (
        "no study shows per-wording estimates any more; the qualifier is "
        "unnecessary")
    t = _paper()
    assert "none reports a dispersion statistic for how far its effect moves" in t


def test_the_abstract_does_not_call_the_quantization_comparable():
    q = _load(DATA / "quantization" / "quantization_analysis.json")
    live = _live(q)
    assert live, "no quantization cells"
    t = _paper()
    assert "the quantization and the name drawn move it comparably" not in t
    lo = min(v["shift_over_sigma_variant"] for v in live.values())
    hi = max(v["shift_over_sigma_variant"] for v in live.values())
    assert f"{lo:.2f}× to {hi:.2f}×" in t


# ------------------------------------------------------------ the panel roster ----
def test_every_reference_in_the_list_is_cited_somewhere_in_the_body():
    """Three panel members had an entry and no citation: _who() collapses any
    set larger than three to "most of the panel", so a study never in a small
    set was never named."""
    mx = _load(DATA / "reference" / "reporting_practice_matrix.json")
    t = _paper()
    for label in mx["llm_hiring_audits"]:
        assert label in t, f"{label} has a reference entry and no citation"


def test_the_panel_is_described_as_a_sample():
    t = _paper()
    assert "every LLM audit we could obtain in full text" not in t
    assert "sample rather than a census" in t or "a sample and not a census" in t


# --------------------------------------------------------------- table ordering ----
def test_tables_render_in_the_order_they_are_numbered():
    """paperkit defers span2 tables to a later page's float slots, so emission
    order is not render order. The build asserted emission order and passed
    while Table 5 printed on page 7 and Table 4 on page 9."""
    fitz = pytest.importorskip("fitz")
    if not PDF.exists():
        pytest.skip("paper not built")
    src = (SRC / "build_paper_v3.py").read_text(encoding="utf-8")
    # THE SOURCE LITERAL IS A SEED, NOT THE TRUTH. The builder measures the
    # order the tables actually reach the page and caches it, because paperkit
    # defers span2 floats and the hand-maintained list was wrong three times in
    # one week. Reading the literal here made this test fail against a PDF whose
    # numbering was correct -- the test was checking the wrong thing.
    sys.path.insert(0, str(SRC))
    keys = list(pytest.importorskip("build_paper_v3").TABLE_ORDER)
    caps = re.findall(
        r"caption=\(f\"\{TAB\('([a-z0-9_]+)'\)\}\.\s*([^\"{]{8,40})", src)
    assert len(caps) == len(keys)
    with fitz.open(PDF) as doc:
        pages = [" ".join(p.get_text().split()) for p in doc]
    pos = []
    for key, head in caps:
        n = keys.index(key) + 1
        needle = f"Table {n}. " + " ".join(head.split())
        hit = next(((i, pages[i].find(needle)) for i in range(len(pages))
                    if pages[i].find(needle) >= 0), None)
        assert hit is not None, f"{key}: caption not in the PDF"
        pos.append((hit, n))
    pos.sort()
    nums = [n for _, n in pos]
    assert nums == sorted(nums), f"tables render out of order: {nums}"


# ------------------------------------------------------- assorted single claims ----
def test_the_screening_rule_alpha_is_a_limit_not_a_maximum():
    srn = _load(DATA / "instrument" / "screening_rule_null.json")
    t = _paper()
    assert "across the whole range it never exceeds" not in t
    assert "at the largest ρ we evaluate" in t
    assert srn["max_rate_over_rho"] < srn["alpha_per_wording"]


def test_the_no_band_noise_ratios_are_printed_rather_than_ordered():
    d = _load(DATA / "replicate" / "noise_vs_probability.json")
    ib = [v["mean_sd_in_band"] for v in _live(d["models"]).values()
          if v.get("mean_sd_in_band")]
    assert ib
    mean_ib = sum(ib) / len(ib)
    ratios = sorted(mean_ib / v["mean_sd_out_of_band"]
                    for v in _live(d["models"]).values()
                    if v.get("n_in_band") == 0 and v.get("mean_sd_out_of_band"))
    assert min(ratios) < 10 < max(ratios), (
        "the two no-band checkpoints are no longer far apart; 'orders of "
        "magnitude' may be defensible again")
    t = _paper()
    assert " and ".join(f"{r:.0f}×" for r in ratios) in t


def test_the_delimiter_floor_fraction_is_derived_from_the_calibration():
    d = _load(DATA / "mechanism_panel" / "mech_panel_analysis.json")
    # the builder reads it off the first model-mode block, which the analysis
    # script writes identically for every block
    first = next(iter(next(iter(d.values())).values()))
    pct = first["delimiter_class_contrast"]["mdd_80_calibrated_pct_of_baseline"]
    words = {17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
             20: "twentieth", 16: "sixteenth", 15: "fifteenth"}
    n = int(100 / pct)
    t = _paper()
    assert f"smaller than a {words[n]} of" in t
    assert "smaller than a twentieth of" not in t or n == 20


def test_the_relative_width_is_a_range_and_not_a_symmetric_half_width():
    du = _load(DATA / "delta_stability" / "dispersion_uncertainty.json")
    s = du["summary"]
    t = _paper()
    assert "±35 %" not in t
    assert f"{round(s['min_relative_width'] * 100)} % to " \
           f"{round(s['max_relative_width'] * 100)} % of the point estimate" in t


def test_the_second_task_per_wording_n_is_not_assumed_equal():
    d = _load(DATA / "second_task" / "second_task_analysis.json")
    rows = []
    for dv in d["by_domain"].values():
        for mv in (dv.get("models") or {}).values():
            ns = [x["n"] for x in (mv.get("per_variant") or {}).values()
                  if isinstance(x, dict) and "n" in x]
            if ns:
                rows.append(ns)
    same = sum(1 for ns in rows if len(set(ns)) == 1)
    assert rows
    t = _paper()
    assert f"the same cells under every wording on {same} of {len(rows)}" in t
    if same < len(rows):
        assert "carries a composition term as well as a wording term" in t
