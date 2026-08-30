r"""Documented figures against the artifacts they describe.

WHY THIS EXISTS. Three times now a hand-quoted number in a document has
drifted from the thing it describes, and each time it was found by accident
or by a test that happened to cover one file. The paper grew from 31 to 33
pages and six live documents still said 31 or 32; the ICLR fork went from
eight main-text pages to nine and two said eight. Nothing was wrong with
the paper. What was wrong is that a reader of CLAUDE.md would have been
told the wrong thing, and the next person to reason from it, including a
future model, would have carried the error forward.

THE DISTINCTION THIS CHECKER RESTS ON. A page count in CHANGELOG.md or in
a sent email is a *record*: it says what was true when it was written and
must never be updated. A page count in CLAUDE.md is a *description of the
current state* and is simply wrong when stale. No regex can tell those
apart, so the live claims are listed here by hand, each with the reason it
is live. The list is short on purpose.

HOW IT AVOIDS PROTECTING NOTHING. Every pattern below must match at least
once. If a document is reworded so a pattern stops firing, that is a
failure, not a pass: a check that matches nothing silently protects
nothing, which is the failure mode this project has hit before.

    sh paper-a/src/_py.sh paper-a/src/audit_doc_figures.py
"""
from __future__ import annotations

import pathlib
import re
import sys
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
FULL_PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
ICLR_PDF = ROOT / "paper-a" / "iclr" / "build" / "main.pdf"
ZIP = ROOT / "paper-a" / "iclr" / "supplementary" / "supplementary_code.zip"

WORDS = {30: "thirty", 31: "thirty-one", 32: "thirty-two",
         33: "thirty-three", 34: "thirty-four", 35: "thirty-five"}


def truth() -> dict[str, int | float]:
    """Every figure read from the artifact, never typed."""
    import fitz
    t: dict[str, int | float] = {}
    if FULL_PDF.exists():
        with fitz.open(FULL_PDF) as d:
            t["full_pages"] = len(d)
    if ICLR_PDF.exists():
        # ONE MEASUREMENT. This used to count pages before the REFERENCES
        # page, which is a proxy for main text and stopped tracking it when
        # the required statements grew to fill a page of their own. It then
        # pinned PLAN.md and SUBMISSION.md to the wrong number -- the audit
        # against stale documents holding two of them stale.
        from build_iclr import main_text_end_page
        with fitz.open(ICLR_PDF) as d:
            t["iclr_total"] = d.page_count
            t["iclr_main"] = main_text_end_page(
                [p.get_text("blocks") for p in d])[0]
    if ZIP.exists():
        with zipfile.ZipFile(ZIP) as z:
            t["zip_members"] = len(z.namelist())
        t["zip_mb"] = round(ZIP.stat().st_size / 1e6, 1)
    return t


# (path, pattern with ONE capturing group, truth key, why this one is live)
CLAIMS: list[tuple[str, str, str, str]] = [
    ("CLAUDE.md", r"paper is (\d+) pages", "full_pages",
     "CLAUDE.md describes the project as it is now"),
    ("paper-a/src/build_iclr.py", r"(\d+)-page preprint", "full_pages",
     "the builder's docstring tells a reader what the fork is a fork of"),
    ("paper-a/iclr/main.tex", r"(\d+)-page preprint", "full_pages",
     "the submission source's header comment"),
    ("paper-a/iclr/PLAN.md", r"(\d+)-page preprint", "full_pages",
     "the runway document describes current state"),
    ("paper-a/iclr/PLAN.md", r"MAIN TEXT (\d+) PAGES", "iclr_main",
     "the runway document quotes the page budget"),
    ("paper-a/iclr/PLAN.md", r"(\d+) total\.", "iclr_total",
     "the runway document quotes the built page count"),
    ("paper-a/iclr/PLAN.md", r"\((\d+) files,", "zip_members",
     "the runway document quotes the supplementary size"),
    ("paper-a/iclr/SUBMISSION.md", r"main text (\d+) of", "iclr_main",
     "the runbook is read immediately before uploading"),
    # The runbook quoted a total page count that no entry pinned, so it sat
    # at 16 while the PDF grew to 17. A figure quoted in a document is a
    # figure that can go stale, whether or not this list happens to name it.
    ("paper-a/iclr/SUBMISSION.md", r"the submission PDF \((\d+) pp", "iclr_total",
     "the runbook quotes the built page count beside the page limit"),
    ("paper-a/iclr/SUBMISSION.md", r"\((\d+) files,", "zip_members",
     "the runbook quotes what the reviewer will receive"),
    # The size was quoted and unpinned, and sat at 9.4 against an actual 7.7.
    # \s+ rather than a literal space: the runbook wraps this figure onto the
    # next line, and a pattern that assumes a space would pin nothing while
    # looking like it pinned something.
    ("paper-a/iclr/SUBMISSION.md", r"\(\d+ files,\s+([\d.]+) MB\)", "zip_mb",
     "the runbook quotes the archive size beside its file count"),
    ("paper-a/iclr/PLAN.md", r"\(\d+ files,\s+([\d.]+) MB\)", "zip_mb",
     "the runway document quotes the archive size"),
    ("paper-a/iclr/SUBMISSION.md", r"100 MB \(ours is ([\d.]+) MB\)", "zip_mb",
     "the runbook weighs the archive against OpenReview's upload cap"),
]

# Correspondence drafts also quote these figures, but naming their paths in
# a file that ships would point a public repository at unpublished letters,
# and the release gate refuses exactly that. They load from a local file the
# release excludes, the same way audit_ai_tells.py loads its target list.
# Absent (a clone), the inline claims above still run.
_LOCAL = (ROOT / "paper-a" / "data" / "reference"
          / "doc_figure_claims.local.txt")
if _LOCAL.exists():
    for _ln in _LOCAL.read_text(encoding="utf-8").splitlines():
        _ln = _ln.strip()
        if not _ln or _ln.startswith("#"):
            continue
        # " :: " and not "|", because a regex may contain pipes and
        # stripping around them silently ate a space the first time.
        _parts = _ln.split(" :: ")
        if len(_parts) != 4:
            raise SystemExit(f"malformed claim line: {_ln!r}")
        CLAIMS.append((_parts[0].strip(), _parts[1],
                       _parts[2].strip(), _parts[3].strip()))

# Documents that quote a figure and must NOT be corrected, with the reason.
# Listed so the exemption is a decision on the record rather than an
# absence anyone has to notice.
HISTORICAL = {
    "CHANGELOG.md": "a log of what was true at each revision",
    "correspondence archive": "sent messages and posting records; the "
                              "directory is named only in the local claims "
                              "file, because a published source file may "
                              "not point at unpublished letters",
    "tests/test_figure_port.py": "narrates a past restructure",
    "paper-a/src/onepager.py": "narrates a past font incident",
    "paper-a/src/paperkit.py": "narrates a past font incident",
    "paper-a/src/build_facct_figures.py": "narrates a past restructure",
}


def main() -> int:
    T = truth()
    if not T:
        print("  nothing built yet; nothing to check")
        return 0

    # --fix rewrites the live claims to whatever the artifacts now say. It
    # is safe in a way an autofix usually is not, because the artifact IS
    # the truth here: there is no judgement to get wrong, only a number to
    # copy. It exists because the preprint's page count moves on almost
    # every content edit, and hand-patching six documents each time is how
    # the drift got in to begin with. It never touches the historical
    # records, which are not claims about now.
    fix = "--fix" in sys.argv
    fixed = 0

    problems: list[str] = []
    checked = 0
    for rel, pattern, key, _why in CLAIMS:
        if key not in T:
            continue
        p = ROOT / rel
        if not p.exists():
            problems.append(f"{rel}: listed as a live claim but missing")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        hits = re.findall(pattern, text)
        if not hits:
            problems.append(
                f"{rel}: /{pattern}/ matches nothing. The document was "
                "reworded, so this check now protects nothing. Re-aim it "
                "or drop it deliberately.")
            continue
        want = T[key]
        want_forms = {str(want), f"{want:,}", WORDS.get(int(want), "")} - {""}
        checked += len(hits)
        stale = [h for h in hits if h.strip().lower() not in want_forms]
        if not stale:
            continue
        if fix:
            # ONE PASS OVER EVERY MATCH, NOT ONE PASS PER STALE HIT. This
            # loop used to call re.subn(..., count=1) once per stale hit,
            # and count=1 always targets the FIRST match in the document
            # rather than the hit being handled. A document holding one
            # correct mention and one stale one therefore had its CORRECT
            # mention rewritten -- which changes nothing, it already said
            # the right thing -- while subn still returned n=1, so the
            # code printed "fixed", counted it and skipped the problem
            # report. The stale number survived and the audit said it had
            # repaired it. A fix that reports success without fixing is
            # worse than no fix, because it also suppresses the warning.
            #
            # Deciding per match instead: each match is rewritten only if
            # its own captured text is stale, so correct mentions are left
            # exactly as they are and every stale one is reached.
            def _sub(m: re.Match) -> str:
                cur = m.group(1)
                if cur.strip().lower() in want_forms:
                    return m.group(0)
                # A numeral stays a numeral and a spelled-out count stays
                # spelled out; only the group is touched, never the
                # sentence around it.
                spelled = not cur.strip().isdigit()
                repl = WORDS.get(int(want), str(want)) if spelled \
                    else str(want)
                return m.group(0).replace(cur, repl, 1)

            new_text = re.sub(pattern, _sub, text)
            if new_text != text:
                p.write_text(new_text, encoding="utf-8")
                fixed += len(stale)
                for hit in stale:
                    print(f"  fixed {rel}: {hit!r} -> {want}")
                continue
        for hit in stale:
            problems.append(
                f"{rel}: says {hit!r} where the artifact says {want} "
                f"({key})")

    print("=" * 74)
    print("DOCUMENTED FIGURES vs THE ARTIFACTS")
    print("=" * 74)
    for k in sorted(T):
        print(f"  {k:14s} {T[k]}")
    print(f"  {checked} quoted figure(s) checked across "
          f"{len({c[0] for c in CLAIMS})} live documents")
    print(f"  {len(HISTORICAL)} document(s) exempt as historical record")
    if problems:
        print("-" * 74)
        for pr in problems:
            print("  " + pr)
        print("=" * 74)
        print(f"{len(problems)} stale figure(s)")
        return 1
    print("=" * 74)
    print("clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
