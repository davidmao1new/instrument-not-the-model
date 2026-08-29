r"""The paper's sentences must be sentences.

A bulk pass converted em dashes to full stops wherever the next word looked
like it could start a clause. The word list contained verbs, which cannot, and
on a paired dash the damage tripled: subject, aside and verb phrase each became
a separate "sentence".

    Whatever the two applications share. The job, the wording, the
    reviewer's mood. Cancels in the difference.

Six of those reached the built PDF across two passes. Nothing caught them. The
numbers were right, the claims were scoped, the style audit measures punctuation
and word choice rather than grammar, and the suite pins claims rather than
prose. A reader found them on the page, which is the worst place to find them.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT = ROOT / "paper-a" / "src" / "audit_fragments.py"
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"


@pytest.mark.skipif(not PDF.exists(), reason="paper not built")
def test_the_paper_contains_no_sentence_fragments():
    r = subprocess.run([sys.executable, str(AUDIT)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, (
        "the built paper contains sentences that cannot stand alone:\n\n"
        f"{r.stdout}\n{r.stderr}")


def test_the_detector_still_catches_the_bug_it_was_written_for():
    """A checker that no longer fires on its own founding case is decoration."""
    sys.path.insert(0, str(ROOT / "paper-a" / "src"))
    import audit_fragments as af

    broken = ("Its strength is that everything except the manipulated "
              "attribute is held identical, so whatever the two applications "
              "share. The job, the wording, the reviewer's mood. Cancels in "
              "the difference.")
    found = af.check(broken)
    assert found, "the detector no longer catches the fragment it exists for"
    assert any(why == "opens on a finite verb" for why, _ in found)

    fixed = ("Its strength is that everything except the manipulated "
             "attribute is held identical, so whatever the two applications "
             "share (the job, the wording, the reviewer's mood) cancels in "
             "the difference.")
    assert not af.check(fixed), "the repaired sentence is still flagged"

def test_the_comma_list_rule_catches_its_founding_case():
    """A short comma list standing alone is an orphaned dash aside.

    'A vacancy, a firm, an employer.' sat in §6.3 through two fragment
    sweeps: five words is below the no-verb floor, and the floor exists
    because section lead-ins are short noun phrases. Lead-ins carry no
    commas, which is the discriminator.
    """
    sys.path.insert(0, str(ROOT / "paper-a" / "src"))
    import audit_fragments as af
    hits = af.check("The design blocks on something. A vacancy, a firm, "
                    "an employer. The match is fixed before any outcome.")
    assert any(why == "comma list with no verb" for why, _ in hits), hits
    # A lead-in without commas stays legal.
    assert not af.check("Text before it ends here. A null-edit control "
                        "and a byte-identical replicate.")

