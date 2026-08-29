r"""Prose written for this project must read like the literature it cites.

WHY THIS EXISTS. The author read the coding of a surveyed audit and the methods supplement and
said they sounded machine-written. He was right, and the useful part is that it
turned out to be measurable rather than a matter of taste. Against the 24
published papers in `lit/text/` -- Bertrand and Mullainathan in the AER,
Benjamini and Hochberg in JRSS-B, and 22 others already on disk because every
citation here was checked against full text -- the drafts were using em dashes
four to nine times as often as any paper in the corpus, capitals for emphasis
that appear in none of them, and sentences of far more uniform length.

`style_corpus.py` derives the baselines. `audit_ai_tells.py` scores a document
against them. This runs the audit so a regression fails the suite instead of
being noticed by the reader.

The thresholds are the corpus, not my judgement. Where a threshold is not from
the corpus -- `shout_caps` and `triad`, whose baselines are inflated by running
headers and author lists surviving PDF extraction -- the audit labels it BUDGET
rather than corpus, so nothing here claims a provenance it does not have.
"""

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT = ROOT / "paper-a" / "src" / "audit_ai_tells.py"
PROFILE = ROOT / "paper-a" / "data" / "reference" / "style_profile.json"


def test_the_style_profile_is_present_and_says_what_it_measured():
    """A profile with no corpus behind it is a set of invented numbers."""
    assert PROFILE.exists(), (
        "no style profile. Run:\n"
        "  sh paper-a/src/_py.sh paper-a/src/style_corpus.py --write")
    prof = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert prof["n_papers"] >= 20, (
        f"the profile is built from {prof['n_papers']} papers; the thresholds "
        f"are percentiles and stop meaning anything on a small sample")
    assert prof["n_words"] >= 100_000, f"only {prof['n_words']:,} words"
    # The features that actually separate machine prose from published prose.
    for k in ("em_dash", "reveal_colon", "not_x_but_y", "sent_len_sd",
              "discourse_opener", "hedge_stack"):
        assert k in prof["per_1000_words"], f"{k} missing from the profile"


def test_the_corpus_confirms_these_are_machine_tics():
    """The premise is checkable: published prose barely uses these at all.

    If the corpus turned out to use the antithesis frame constantly, the
    feature would be measuring register rather than authorship, and flagging it
    would be wrong. It does not.
    """
    prof = json.loads(PROFILE.read_text(encoding="utf-8"))["per_1000_words"]
    assert prof["not_x_but_y"]["median"] == 0.0, (
        "the corpus median for the antithesis frame is not zero; the "
        "assumption behind flagging it needs rechecking")
    assert prof["hedge_stack"]["median"] == 0.0
    assert prof["em_dash"]["median"] < 2.0, (
        f"corpus median em-dash rate is {prof['em_dash']['median']}, which is "
        f"high enough that flagging the dash needs justifying again")


@pytest.mark.skipif(not AUDIT.exists(), reason="no audit script")
def test_the_drafts_read_like_the_corpus():
    """Runs the audit over every tracked document. Fails on a hard breach.

    A hard breach means the document is outside the range every paper in the
    corpus occupies -- not merely above average, but past the worst case in
    twenty-four published papers.
    """
    r = subprocess.run([sys.executable, str(AUDIT), "--all"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    assert r.returncode == 0, (
        "prose has drifted outside the range published academic writing "
        f"occupies:\n\n{r.stdout}\n{r.stderr}")


def test_reveal_colon_spares_lists_and_catches_punchlines():
    """The detector's own docs permit a colon that introduces a list; for a
    while the regex flagged those too, which trains a reader to dismiss the
    real hits. Both founding cases, pinned."""
    import sys
    sys.path.insert(0, str(ROOT / "paper-a" / "src"))
    from style_corpus import measure
    lst = ("The survey codes the audits for four things: the wording set, "
           "the name list, the serving configuration, and the resampling "
           "unit. ") * 3
    pun = ("What I should have said is narrower: the freedom only exists "
           "when pairing happens after scoring. ") * 3
    assert measure(lst)["reveal_colon"] == 0, "a list colon was flagged"
    assert measure(pun)["reveal_colon"] > 0, "the punchline colon escaped"
