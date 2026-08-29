r"""Ported figures must be self-contained, and the check must still bite.

The variance-components figure shipped in the ICLR fork with "see §6.2"
drawn into its x-axis. That is correct in the preprint and meaningless in
the fork, which has no subsections, so a submitted paper pointed at nothing.
A caption can be rewritten per venue; an axis label cannot.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
sys.path.insert(0, str(SRC))

import audit_figure_refs as afr  # noqa: E402


def test_the_ported_figures_are_clean():
    r = subprocess.run([sys.executable, str(SRC / "audit_figure_refs.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    assert r.returncode == 0, r.stdout + r.stderr


def test_it_scans_something():
    """A scanner pointed at empty directories passes forever. The fork
    carries figures; if this drops to zero the audit has gone blind."""
    present = [d for d in afr.DIRS if d.exists()]
    if not present:
        pytest.skip("no ported figure directories in this checkout")
    n = sum(len(list(d.glob("*.pdf"))) for d in present)
    assert n >= 4, f"only {n} ported figures found; the audit may be misaimed"


@pytest.mark.parametrize("bad", [
    "see §6.2", "Section 4.1", "in Table 3", "Figure 2", "Appendix B",
])
def test_the_pattern_catches_real_reference_forms(bad):
    assert afr.REF.search(bad), f"{bad!r} should be caught"


@pytest.mark.parametrize("ok", [
    "percentage points, upper bound at the p = 0.5 logistic slope",
    "between-wording SD", "demographic effect (log-odds)",
    "specification, sorted", "0.25 0.50 0.75",
])
def test_the_pattern_leaves_ordinary_labels_alone(ok):
    assert not afr.REF.search(ok), f"{ok!r} is a false positive"


def test_the_preprints_own_figures_are_not_scanned():
    """paper-a/figures/ draws captions into the image, and a caption in the
    preprint may reference the preprint. Scanning it would flood."""
    assert all("iclr" in d.as_posix() or "facct" in d.as_posix()
               for d in afr.DIRS), (
        "the audit scope grew to include the preprint's own figures, which "
        "legitimately carry their captions and their cross-references")
