"""The §9.1 screening rule, its calibration, and the arXiv abstract.

THREE THINGS PINNED HERE.

1. The rule's null calibration. §9.1 recommended a decision procedure and said
   its behaviour under the null "depends on how correlated the wordings are,
   which is not something we have characterised" -- the paper's own complaint
   about the field, pointed at itself. The rate is now computed exactly, and
   the load-bearing property is that it never exceeds the per-wording alpha at
   any correlation, so the recommendation cannot be more liberal than the test
   it replaces.

2. The rule's verdict on our own panel. The paper asserted, in hand-typed
   prose, that the rule is passed by the sign-stable models. It is failed by
   all four: sign stability is only half of it, and on both sign-stable models
   the smallest per-wording interval covers the null.

3. The arXiv abstract. It must be under 1920 characters, pure ASCII, one line,
   free of the TeX arXiv rejects, and identical in content to the PDF's.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
DATA = ROOT / "paper-a" / "data"
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
sys.path.insert(0, str(SRC))


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return json.loads(p.read_text(encoding="utf-8"))


def _srn():
    return _load(DATA / "instrument" / "screening_rule_null.json")


def _paper():
    if not PDF.exists():
        pytest.skip("paper not built")
    fitz = pytest.importorskip("fitz")
    with fitz.open(PDF) as doc:
        return " ".join(" ".join(p.get_text().split()) for p in doc)


# ------------------------------------------------------- the calibration ----
def test_the_rate_never_exceeds_the_per_wording_alpha():
    """The property the recommendation rests on."""
    d = _srn()
    assert d["rate_never_exceeds_alpha"] is True
    assert d["max_rate_over_rho"] <= d["alpha_per_wording"] + 1e-12


def test_the_rate_increases_with_correlation():
    d = _srn()
    assert d["rate_monotone_increasing_in_rho"] is True
    rows = sorted(d["by_rho"], key=lambda r: r["rho"])
    assert all(a["exact"] <= b["exact"] + 1e-15
               for a, b in zip(rows, rows[1:]))


def test_the_simulation_agrees_with_the_closed_form():
    """Two independent routes to the same number, where the rate is visible."""
    for r in _srn()["by_rho"]:
        if r["exact"] < 1e-4:
            continue          # below what 400k draws can resolve
        assert abs(r["rule"] - r["exact"]) < 0.15 * r["exact"] + 5e-4, r["rho"]


def test_the_closed_form_matches_the_independent_case_analytically():
    """At rho = 0 the answer is 2*alpha_half^k and nothing else."""
    from scipy import stats  # noqa: PLC0415
    d = _srn()
    z, k = d["z"], d["k_wordings"]
    r0 = next(r for r in d["by_rho"] if r["rho"] == 0.0)
    assert abs(r0["exact"] - 2 * stats.norm.sf(z) ** k) < 1e-30


def test_the_empirical_correlation_is_measured_not_assumed():
    d = _srn()
    emp = d["empirical_rho_per_model"]
    vals = [v["mean_offdiagonal_r"] for v in emp.values()
            if v.get("mean_offdiagonal_r") is not None]
    assert len(vals) >= 2
    assert all(-1.0 <= v <= 1.0 for v in vals)
    assert d["rate_at_empirical_rho_max"] <= d["alpha_per_wording"]


# --------------------------------------------- the verdict on our own panel ----
def test_the_rule_fails_on_every_model_and_the_paper_says_so():
    s2 = _load(DATA / "delta_stability" / "study2_v2.json")
    passed = []
    for m, v in s2.items():
        if m.startswith("_") or not isinstance(v, dict):
            continue
        pv = v.get("per_variant")
        if not pv:
            continue
        e = {k: pv[k]["logodds"] for k in pv}
        stable = len({x["est"] > 0 for x in e.values()}) == 1
        w = min(e, key=lambda k: abs(e[k]["est"]))
        ci = e[w]["ci"]
        if stable and ci[0] * ci[1] > 0:
            passed.append(m)
    assert not passed, f"the rule now passes on {passed}; §9.1's text must change"
    t = _paper()
    assert "failed by all four" in t
    assert "passed by the models whose sign is stable" not in t


# ------------------------------------------------------ the arXiv abstract ----
def test_the_arxiv_abstract_meets_arxivs_rules():
    p = ROOT / "paper-a" / "releases" / "abstract_arxiv.txt"
    if not p.exists():
        pytest.skip("abstract not emitted")
    raw = p.read_bytes()
    t = raw.decode("ascii")            # raises if not pure ASCII
    body = t.rstrip("\n")
    assert len(body) <= 1920, f"{len(body)} characters, arXiv's limit is 1920"
    assert "\n" not in body, "must be a single line"
    assert not body.lower().startswith("abstract")
    for bad in ("~", "\\,", "\\ ", "\\em", "\\it", "\\bf"):
        assert bad not in body, f"arXiv rejects {bad!r}"


def test_the_arxiv_abstract_matches_the_printed_one():
    """Same text, transliterated -- not a second abstract that can drift."""
    p = ROOT / "paper-a" / "releases" / "abstract_arxiv.txt"
    if not p.exists():
        pytest.skip("abstract not emitted")
    body = p.read_text(encoding="ascii").rstrip("\n")
    t = _paper()
    i = t.find("ABSTRACT")
    j = t.find("Keywords:")
    printed = t[i + len("ABSTRACT"):j].strip()
    # compare on a shape both survive: the digits, in order
    digits_txt = [c for c in body if c.isdigit()]
    digits_pdf = [c for c in printed if c.isdigit()]
    assert digits_txt == digits_pdf, "the two abstracts carry different numbers"
    assert len(printed) <= 1920


def test_the_transliteration_is_total():
    """An unmapped non-ASCII character must raise, not slip through."""
    import build_paper_v3 as B  # noqa: PLC0415
    with pytest.raises(ValueError):
        B.arxiv_abstract("a Chinese character 中 slips through")
    with pytest.raises(ValueError):
        B.arxiv_abstract("x" * 2000)
    assert B.arxiv_abstract("résumé — 3×") == "resume -- 3x"
