"""Tests for the stimuli and the perturbation conditions.

The paper's central claim is that edits carrying NO MEANING move a measured
demographic effect. That claim is only as good as the guarantee that the edits
really carry no meaning, and that guarantee has so far been a docstring. These
tests make it mechanical: strip all whitespace from any conditioned prompt and it
must be byte-identical to the baseline. If a future edit accidentally changes a
word, the suite fails rather than the paper quietly overclaiming.

They also cover the contamination guard, which caught a real failure in which
one stream's probes silently recorded another model's answers, and the resume
reader, which must tolerate the truncated final line an interrupted run leaves
behind without inventing data.
"""
from __future__ import annotations

import inspect
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
sys.path.insert(0, str(SRC))

import stimuli as st  # noqa: E402
from experiment_mechanism import (  # noqa: E402
    CONDITIONS, build, base_prompt, _nth_break)
import experiment_delta_stability as eds  # noqa: E402


NOWS = re.compile(r"\s+")


def strip_ws(s: str) -> str:
    return NOWS.sub("", s)


# --------------------------------------------------------------------------
# The name grid
# --------------------------------------------------------------------------
def test_grid_shape():
    g = st.NAME_GRID
    assert len(g) == st.N_FIRST * st.N_LAST * 2 == 48
    assert len({p["white"] for p in g}) == 48
    assert len({p["black"] for p in g}) == 48


def test_grid_is_gender_balanced_and_pairs_share_gender():
    g = st.NAME_GRID
    assert sum(p["gender"] == "female" for p in g) == 24
    assert sum(p["gender"] == "male" for p in g) == 24
    for p in g:
        wf, bf = p["white_first"], p["black_first"]
        assert wf in st._BM_FIRST[("white", p["gender"])]
        assert bf in st._BM_FIRST[("black", p["gender"])]


def test_grid_selection_rule_is_alphabetical_and_behaviour_blind():
    """The selection rule must not be tunable after seeing results."""
    for gender in ("female", "male"):
        for race in ("white", "black"):
            chosen = {p[f"{race}_first"] for p in st.NAME_GRID
                      if p["gender"] == gender}
            expected = set(sorted(st._BM_FIRST[(race, gender)])[:st.N_FIRST])
            assert chosen == expected
    for race in ("white", "black"):
        chosen = {p[f"{race}_last"] for p in st.NAME_GRID}
        assert chosen == set(sorted(st._BM_LAST[race])[:st.N_LAST])


def test_grid_is_a_full_factorial():
    for race in ("white", "black"):
        for gender in ("female", "male"):
            sub = [p for p in st.NAME_GRID if p["gender"] == gender]
            firsts = {p[f"{race}_first"] for p in sub}
            lasts = {p[f"{race}_last"] for p in sub}
            combos = {(p[f"{race}_first"], p[f"{race}_last"]) for p in sub}
            assert len(combos) == len(firsts) * len(lasts)


def test_mech_grid_is_a_gender_balanced_half():
    assert len(st.MECH_GRID) == 24
    assert sum(p["gender"] == "female" for p in st.MECH_GRID) == 12
    assert {p["idx"] for p in st.MECH_GRID} <= {p["idx"] for p in st.NAME_GRID}


def test_grid_names_are_all_from_the_verified_bm2004_list():
    ref = json.loads((pathlib.Path(__file__).resolve().parents[1] / "paper-a" /
                      "data" / "reference" / "bm2004_names.json")
                     .read_text(encoding="utf-8"))
    allowed_first = {x["name"] for cell in ref["first_names"].values()
                     for x in cell}
    allowed_last = set(ref["surnames"]["white"]) | set(ref["surnames"]["black"])
    for p in st.NAME_GRID:
        for race in ("white", "black"):
            assert p[f"{race}_first"] in allowed_first
            assert p[f"{race}_last"] in allowed_last


# --------------------------------------------------------------------------
# THE CENTRAL INVARIANT: every mechanism condition is whitespace-only
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cond", sorted(CONDITIONS))
def test_condition_changes_only_whitespace(cond):
    name, body = st.NAME_GRID[0]["white"], st.TEMPLATES["T1_strong"]
    assert strip_ws(build(cond, name, body)) == strip_ws(build("D0", name, body))


@pytest.mark.parametrize("cond", sorted(CONDITIONS))
@pytest.mark.parametrize("tname", ["T1_strong", "T2_mid", "T3_marginal"])
def test_condition_never_touches_the_name(cond, tname):
    for p in st.NAME_GRID[:6]:
        for race in ("white", "black"):
            nm = p[race]
            out = build(cond, nm, st.TEMPLATES[tname])
            assert nm in out, f"{cond} mangled {nm!r}"
            assert out.count(nm) == 1


@pytest.mark.parametrize("cond", sorted(CONDITIONS))
def test_every_condition_actually_differs_from_baseline(cond):
    name, body = st.NAME_GRID[0]["white"], st.TEMPLATES["T1_strong"]
    out, base = build(cond, name, body), build("D0", name, body)
    if cond == "D0":
        assert out == base
    else:
        assert out != base, f"{cond} is a no-op; the replacement did not match"


def test_baseline_has_exactly_two_paragraph_breaks():
    """Every delimiter condition is defined relative to these two."""
    s = base_prompt("Allison Baker", st.TEMPLATES["T1_strong"])
    assert s.count("\n\n") == 2


def test_delimiter_conditions_target_the_intended_breaks():
    name, body = "Allison Baker", st.TEMPLATES["T1_strong"]
    base = base_prompt(name, body)
    i0, i1 = _nth_break(base, 0), _nth_break(base, 1)
    assert i0 < i1
    # D4 fragments the FIRST break only, D5 the second only, D6 both
    assert build("D4", name, body).count("\n \n") == 1
    assert build("D5", name, body).count("\n \n") == 1
    assert build("D6", name, body).count("\n \n") == 2
    assert build("D4", name, body)[:i0] == base[:i0]      # nothing before it moved
    # the first break precedes the name, the second follows it
    ni = base.index(name)
    assert i0 < ni < i1


def test_position_conditions_do_not_touch_any_delimiter():
    name, body = "Allison Baker", st.TEMPLATES["T1_strong"]
    for cond in ("D8", "D9"):
        out = build(cond, name, body)
        assert out.count("\n\n") == 2
        assert "\n \n" not in out


def test_d9_is_a_strictly_larger_edit_than_d8():
    name, body = "Allison Baker", st.TEMPLATES["T1_strong"]
    d8, d9 = build("D8", name, body), build("D9", name, body)
    assert len(d9) > len(d8) > len(build("D0", name, body))


# --------------------------------------------------------------------------
# Study 2's null arm
# --------------------------------------------------------------------------
NULL_WS_ONLY = ["N1", "N2", "N3", "N5", "N6"]


@pytest.mark.parametrize("v", NULL_WS_ONLY)
def test_null_variant_user_message_is_whitespace_only(v):
    name, body = "Allison Baker", eds.TEMPLATES["T1_strong"]
    a = eds.user_message(eds.VARIANTS[v], name, body)
    b = eds.user_message(eds.VARIANTS["N1"], name, body)
    assert strip_ws(a) == strip_ws(b)


def test_n4_swaps_two_independent_sentences_and_changes_nothing_else():
    """N4 is the one null variant that is not whitespace-only; it reorders two
    independent system sentences. The words must be preserved exactly."""
    a, b = eds.VARIANTS["N4"]["system"], eds.VARIANTS["N1"]["system"]
    assert a != b
    assert sorted(a.split()) == sorted(b.split())
    assert eds.VARIANTS["N4"]["ask"] == eds.VARIANTS["N1"]["ask"]


def test_s1_and_n1_are_byte_identical():
    """The accidental replicate the noise floor is estimated from. If a future
    edit breaks this, noise_floor.py is silently measuring something else."""
    name, body = "Allison Baker", eds.TEMPLATES["T1_strong"]
    assert eds.VARIANTS["S1"]["system"] == eds.VARIANTS["N1"]["system"]
    assert (eds.user_message(eds.VARIANTS["S1"], name, body)
            == eds.user_message(eds.VARIANTS["N1"], name, body))


def test_variant_arms_are_six_and_six():
    kinds = [v["kind"] for v in eds.VARIANTS.values()]
    assert kinds.count("semantic") == 6
    assert kinds.count("null") == 6


# --------------------------------------------------------------------------
# Shared stimuli must not drift between modules
# --------------------------------------------------------------------------
def test_stimuli_identical_across_modules():
    """stimuli.py claims in its docstring that a test asserts this. It does now."""
    assert st.POSTING == eds.POSTING
    assert st.TEMPLATES == eds.TEMPLATES
    assert st.YES_TOKENS == eds.YES_TOKENS
    assert st.NO_TOKENS == eds.NO_TOKENS
    assert st.VERDICT_GRAMMAR == eds.VERDICT_GRAMMAR
    assert st.TOP_LOGPROBS == eds.TOP_LOGPROBS


def test_templates_2_are_the_extremes():
    assert st.TEMPLATES_2 == ("T1_strong", "T3_marginal")
    assert set(st.TEMPLATES_2) <= set(st.TEMPLATES)


# --------------------------------------------------------------------------
# The resume reader
# --------------------------------------------------------------------------
def test_read_jsonl_tolerates_a_truncated_final_line(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b"', encoding="utf-8")
    rows = list(st.read_jsonl(p))
    assert [r["a"] for r in rows] == [1, 2]


def test_read_jsonl_raises_on_corruption_that_is_not_the_last_line(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text('{"a": 1}\nNOT JSON\n{"a": 3}\n', encoding="utf-8")
    with pytest.raises(RuntimeError):
        list(st.read_jsonl(p))


def test_read_jsonl_on_missing_file_is_empty(tmp_path):
    assert list(st.read_jsonl(tmp_path / "nope.jsonl")) == []


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"a": 1}\n\n\n{"a": 2}\n', encoding="utf-8")
    assert len(list(st.read_jsonl(p))) == 2


# --------------------------------------------------------------------------
# The contamination guard
# --------------------------------------------------------------------------
def test_expected_filename_resolves_labels_through_config():
    assert st.expected_filename("llama-2-7b-chat") == "llama-2-7b-chat.Q4_K_M.gguf"
    assert st.expected_filename("mistral-7b-v0.1-base") == "mistral-7b-v0.1.Q4_K_M.gguf"


def test_expected_filename_distinguishes_quantization():
    """The hole the old prefix guard had: a Q8 label matched Q4 weights, in the
    one study where quantization is the independent variable."""
    q4 = st.expected_filename("llama-2-7b-chat")
    q8 = st.expected_filename("llama-2-7b-chat-q8")
    assert q4 != q8
    assert "Q8" in q8 and "Q4" in q4


def test_expected_filename_returns_none_for_an_unknown_label():
    assert st.expected_filename("not-a-real-model") is None


def test_assert_serving_refuses_an_unknown_label(monkeypatch):
    monkeypatch.setattr(st, "served_model", lambda port: "whatever.gguf")
    with pytest.raises(SystemExit):
        st.assert_serving(8080, "not-a-real-model")


def test_assert_serving_refuses_a_filename_mismatch(monkeypatch):
    monkeypatch.setattr(
        st, "served_model",
        lambda port: r"C:\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    with pytest.raises(SystemExit):
        st.assert_serving(8080, "llama-2-7b-chat")


def test_assert_serving_accepts_the_right_checkpoint(monkeypatch):
    monkeypatch.setattr(
        st, "served_model", lambda port: r"C:\models\llama-2-7b-chat.Q4_K_M.gguf")
    assert st.assert_serving(8080, "llama-2-7b-chat") == "llama-2-7b-chat.Q4_K_M.gguf"


def test_assert_serving_refuses_q4_weights_under_a_q8_label(monkeypatch):
    monkeypatch.setattr(
        st, "served_model", lambda port: r"C:\models\llama-2-7b-chat.Q4_K_M.gguf")
    with pytest.raises(SystemExit):
        st.assert_serving(8080, "llama-2-7b-chat-q8")


# --------------------------------------------------------------------------
# Margin extraction
# --------------------------------------------------------------------------
def test_margin_sums_over_surface_forms():
    import math
    top = [{"token": "yes", "logprob": math.log(0.2)},
           {"token": " Yes", "logprob": math.log(0.1)},
           {"token": "no", "logprob": math.log(0.1)},
           {"token": "The", "logprob": math.log(0.6)}]
    m, py, pn = st._margin_from_top(top)
    assert py == pytest.approx(0.3)
    assert pn == pytest.approx(0.1)
    assert m == pytest.approx(math.log(0.3) - math.log(0.1))


def test_margin_is_none_when_an_option_is_absent():
    import math
    top = [{"token": "yes", "logprob": math.log(0.5)},
           {"token": "The", "logprob": math.log(0.5)}]
    m, py, pn = st._margin_from_top(top)
    assert m is None and pn == 0.0


# ---------------------------------------------------------------------------
# The mechanism-panel loader must not swallow the D9 adjudication files.
#
# `mech_d9recheck_*.jsonl` is a deliberate RE-MEASUREMENT of the D8 and D9
# base-arm cells, kept as an independent cross-session reproducibility check
# after adjudicate_d9.py returned VINDICATED. A bare `mech_*.jsonl` glob pulled
# it into the primary analysis, and because sorted() places
# mech_d9recheck_chat_* after mech_chat_* but before mech_raw_*, the recheck
# overwrote the originals in chat mode and lost to them in raw mode. 576 cells,
# one mode only, in exactly the conditions the position contrasts are built
# from. Five BH verdicts moved.
# ---------------------------------------------------------------------------
def test_mech_panel_loader_excludes_the_d9_recheck():
    import analyze_mech_panel as amp
    src = inspect.getsource(amp.load)
    assert "mech_d9recheck_" in src, (
        "analyze_mech_panel.load() no longer mentions the recheck prefix; if the "
        "glob was widened again the recheck will silently replace stored cells")
    rows = amp.load()
    if not rows:
        pytest.skip("no panel data on disk")
    # every stored cell must be unique, and none may carry the recheck marker
    seen = {}
    for r in rows:
        k = (r["model"], r["mode"], r["cond"], r["template"], r["pair"])
        assert k not in seen, f"duplicate design cell survived the loader: {k}"
        seen[k] = r


def test_d9_recheck_files_exist_and_are_a_separate_measurement():
    """If the recheck files vanish, the exclusion above becomes untestable and
    the reproducibility measurement built on them is unsupported."""
    d = ROOT / "paper-a" / "data" / "mechanism_panel"
    if not d.exists():
        pytest.skip("no panel data on disk")
    recheck = sorted(d.glob("mech_d9recheck_*.jsonl"))
    if not recheck:
        pytest.skip("recheck not run")
    assert len(recheck) == 12, f"expected 12 recheck files, found {len(recheck)}"
    verdict = d / "d9_adjudication.json"
    assert verdict.exists(), "recheck data present but never adjudicated"
    assert json.loads(verdict.read_text(encoding="utf-8"))["verdict"] in (
        "VINDICATED", "CONDEMNED")


# ---------------------------------------------------------------------------
# The consistency audit must flag live stale claims and NOT flag its own
# retractions of them. Correcting a claim means naming it, so a phrase regex
# alone reports every correction as an error and the report stops being usable.
# ---------------------------------------------------------------------------
def test_audit_retraction_guard_flags_live_claims():
    import audit_consistency as ac
    live = "some prose\nthe measurement is deterministic at temperature 0\nmore\n"
    assert not ac._is_retraction(live, live.index("deterministic"))


def test_audit_retraction_guard_skips_retractions():
    import audit_consistency as ac
    ret = "prose\nthis previously said the measurement is deterministic; retracted\np\n"
    assert ac._is_retraction(ret, ret.index("deterministic"))


def test_audit_retraction_guard_window_is_one_line():
    """A retraction three lines up must not exempt an unrelated live claim."""
    import audit_consistency as ac
    far = "retracted elsewhere\nline\nline\nthe measurement is deterministic\n"
    assert not ac._is_retraction(far, far.index("deterministic"))
