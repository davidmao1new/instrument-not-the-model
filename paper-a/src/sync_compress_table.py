r"""Resync COMPRESS.md's budget table from audit_section_dependencies.

The table quotes the audit's output and a test pins the two together, so any
paper edit that moves a section's word count fails the suite until this runs.
That is by design (the pin caught three real drifts in its first day); this
script is the one-command repair.

    sh paper-a/src/_py.sh paper-a/src/sync_compress_table.py
"""
import pathlib
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    out = subprocess.run(
        [sys.executable,
         str(ROOT / "paper-a/src/audit_section_dependencies.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT)).stdout
    words = dict(re.findall(r"^§(\d+)\s+([\d,]+) words ->", out, re.M))
    cuts = dict(re.findall(
        r"^§(\d+)\s+[\d,]+ words -> \d+\s+\(cut [\d,]+, (\d+) %\)",
        out, re.M))
    only = {}
    for b in re.split(r"^-{10,}$", out, flags=re.M):
        m = re.search(r"^§(\d+)\s", b, re.M)
        c = re.search(r"ONLY HERE \((\d+)\)", b)
        if m and c:
            only[m.group(1)] = c.group(1)

    p = ROOT / "paper-a/facct/COMPRESS.md"
    t = p.read_text(encoding="utf-8")
    total = 0
    for sec, budget in (("1", "700"), ("2", "550"), ("9", "600"),
                        ("10", "400"), ("11", "250")):
        w = words[sec]
        total += int(w.replace(",", ""))
        bold = "**" if sec == "10" else ""
        new = (f"| {sec} | {w} | {budget} | {bold}{cuts[sec]} %{bold} | "
               f"{only.get(sec, '0')} |")
        t, k = re.subn(
            rf"^\| {sec} \| [\d,]+ \| {budget} \| \*{{0,2}}\d+ %\*{{0,2}} "
            rf"\| \d+ \|$", new, t, flags=re.M)
        assert k == 1, (sec, k)
        t, k = re.subn(rf"(## §{sec} [^\n]*?— )[\d,]+( → {budget})",
                       rf"\g<1>{w}\g<2>", t)
        assert k == 1, (sec, "heading")
    t, k = re.subn(r"^\| \| \*\*[\d,]+\*\* \| 2,500 \| \| \|$",
                   f"| | **{total:,}** | 2,500 | | |", t, flags=re.M)
    assert k == 1

    # The two figures COMPRESS.md quotes in prose from the S10 worksheet.
    # They were hand-copied and drifted by a word the first time the paper
    # grew, which test_facct_s10 caught. Sync them from the worksheet here
    # so the table and the prose cannot disagree with it separately.
    # ORDER MATTERS AND NOTHING ELSE ENFORCES IT. S10.md is regenerated from
    # the built paper by build_s10_disposition.py, and this script copies
    # figures out of S10.md. Run in the wrong order it faithfully copies the
    # previous build's numbers into COMPRESS.md and reports success, which
    # is exactly how a one-word drift survived a green sync once.
    _sheet_p = ROOT / "paper-a/facct/S10.md"
    _paper_p = ROOT / "paper-a/figures/paper_instrument_validity_v3.pdf"
    if (_paper_p.exists()
            and _sheet_p.stat().st_mtime < _paper_p.stat().st_mtime):
        raise SystemExit(
            "  REFUSING: S10.md is older than the built paper, so its "
            "figures are\n  stale and syncing them would spread the "
            "staleness. Run this first:\n"
            "    sh paper-a/src/_py.sh paper-a/src/build_s10_disposition.py")
    sheet = _sheet_p.read_text(encoding="utf-8")
    WORD = {9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve",
            13: "Thirteen", 14: "Fourteen"}

    m = re.search(r"cost \*\*(\d+) words\*\*", sheet)
    assert m, "the worksheet no longer states its must-survive word count"
    t, k = re.subn(r"(must survive are )\d+( words between)",
                   rf"\g<1>{m.group(1)}\g<2>", t)
    assert k == 1, "must-survive figure not found in COMPRESS.md"
    survive = m.group(1)

    m = re.search(r"\*\*(\d+) numbers appear only in §10", sheet)
    assert m, "the worksheet no longer states its only-here count"
    n_only = int(m.group(1))
    t, k = re.subn(r"\*\*[A-Z][a-z]+ numbers appear only",
                   f"**{WORD.get(n_only, n_only)} numbers appear only", t)
    assert k == 1, "only-here figure not found in COMPRESS.md"

    p.write_text(t, encoding="utf-8")
    print(f"  COMPRESS.md table resynced ({total:,} words over five sections)")
    print(f"  prose figures resynced (must-survive {survive} words, "
          f"{n_only} numbers only in the section)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
