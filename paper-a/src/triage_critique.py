"""Which critique findings still apply to the CURRENT build?

WHY THIS IS NEEDED. A critique round reads a frozen text. Fixes land while it
runs. By the time a round returns, some of its findings describe a paper that no
longer exists -- and acting on those wastes a fix cycle, or worse, reverts a
correction. Round 2 was launched against v4 and returned after eleven further
rebuilds, so a large share of its 128 findings are already answered.

WHAT THIS DOES, and what it deliberately does NOT do. It is a cheap first pass,
not a verdict. For each finding it asks one mechanical question: does the exact
sentence or table cell the critic quoted still appear in the current PDF?

  * quote absent   -> the text was rewritten. The finding MAY be answered, but
                      the underlying defect may survive in new wording, so it is
                      marked LIKELY-STALE and still goes to a human or an agent.
  * quote present  -> the sentence is still on the page verbatim. The finding
                      stands until refuted on its merits.

Matching is done on a normalised form -- whitespace collapsed, quotation marks
and dashes unified, digits preserved -- because a two-column PDF breaks lines
mid-sentence and the critic quotes a reflowed string.

WHY THE OBVIOUS MATCHER IS WRONG, AND WAS. The first version fell back to the
quote's first sixty characters whenever the full quote was absent, on the theory
that reflow had broken it. That inverts the test. A fix usually changes a few
words inside a long sentence and leaves the opening clause alone -- so the
prefix survives precisely when the defect is FIXED, and the fallback flipped
those findings back to "stands". After round 4 it reported sixteen findings as
outstanding that had each been verified fixed, including the f-string leak whose
entire content was the six characters removed. A triage that cannot tell a
repaired sentence from an untouched one is worse than none, because the loop's
convergence test reads its output.

So the match is GRADED, on word 8-grams of the normalised quote:

  * verbatim  -- the whole quote is on the page. The finding stands.
  * reflowed  -- >= 90 % of its 8-grams survive. Line-breaking moved it; the
                 sentence is the same one. The finding stands.
  * partly    -- 40 % to 90 %. Something was edited inside it. Needs a human;
                 this is the bucket a prefix match used to hide in.
  * gone      -- below 40 %. The passage was rewritten.

8-grams because they are long enough to be distinctive and short enough to
survive a column break once whitespace is normalised. The missing n-grams are
recorded for the "partly" bucket, since the edited region is the thing to look
at and it is usually where the defect was.

NEVER use "gone" alone to close a finding. The defect classes this project keeps
rediscovering -- a ratio with a null denominator, a claim outrunning its
evidence -- move between sentences when prose is edited.

    C:/research-toolchain/venv/Scripts/python.exe \\
        paper-a/src/triage_critique.py paper-a/releases/critique_round2.json
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"


def norm(s: str) -> str:
    s = (s or "")
    s = (s.replace("\u2019", "'").replace("\u2018", "'")
          .replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2014", "-").replace("\u2013", "-")
          .replace("\u00d7", "x").replace("\u2009", " ")
          .replace("\u00a0", " "))
    return re.sub(r"\s+", " ", s).strip().lower()


N = 8            # words per shingle
STANDS = 0.90    # >= this fraction present and the sentence is the same one
EDITED = 0.40    # below this and the passage was rewritten


def shingles(words: list[str], n: int = N) -> list[str]:
    if len(words) <= n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def match(quote: str, paper: str) -> dict:
    """How much of `quote` is still on the page, and which part is not."""
    q = norm(quote)
    if len(q) < 12:
        return dict(verdict="unmatchable", coverage=None, missing=[])
    if q in paper:
        return dict(verdict="verbatim", coverage=1.0, missing=[])
    sh = shingles(q.split())
    if not sh:
        return dict(verdict="unmatchable", coverage=None, missing=[])
    missing = [s for s in sh if s not in paper]
    cov = 1.0 - len(missing) / len(sh)
    verdict = ("reflowed" if cov >= STANDS
               else "partly" if cov >= EDITED else "gone")
    # Report the edited REGION rather than every overlapping shingle that
    # touches it: consecutive misses are one edit, and listing 40 of them
    # buries the one place worth reading.
    regions, run = [], []
    for s in sh:
        if s in missing:
            run.append(s)
        elif run:
            regions.append(run[0] if len(run) == 1
                           else run[0] + " ... " + run[-1])
            run = []
    if run:
        regions.append(run[0] if len(run) == 1 else run[0] + " ... " + run[-1])
    return dict(verdict=verdict, coverage=round(cov, 3), missing=regions[:4])


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit("usage: triage_critique.py <critique.json>")
    src = pathlib.Path(sys.argv[1]).resolve()
    d = json.loads(src.read_text(encoding="utf-8"))

    import fitz  # noqa: PLC0415
    doc = fitz.open(PDF)
    paper = norm(" ".join(p.get_text() for p in doc))

    rows = []
    for f in d["confirmed"]:
        m = match(f.get("quote"), paper)
        rows.append(dict(
            id=f.get("id"), severity=f.get("severity"),
            lenses=f.get("lenses") or ([f["lens"]] if f.get("lens") else []),
            n=f.get("n_independent_reports", 1),
            verdict=m["verdict"], coverage=m["coverage"],
            edited_regions=m["missing"],
            quote=(f.get("quote") or "")[:150],
            defect=(f.get("defect") or "")[:200],
            fix=(f.get("fix") or "")[:200]))

    def bucket(name):
        return [r for r in rows if r["verdict"] == name]

    stands = bucket("verbatim") + bucket("reflowed")
    partly = bucket("partly")
    gone = bucket("gone")
    unk = bucket("unmatchable")

    def sev_count(rs, *levels):
        return sum(1 for r in rs if r["severity"] in levels)

    out = dict(
        source=src.name, pdf=str(PDF.relative_to(ROOT)),
        n_findings=len(rows),
        _method=(f"word-{N} shingle coverage of the normalised quote against "
                 f"the normalised PDF text; >= {STANDS:.0%} stands, "
                 f"< {EDITED:.0%} is rewritten"),
        _warning="'gone' does NOT close a finding; the defect can survive a "
                 "rewrite. This is a triage aid only.",
        n_stands=len(stands), n_partly_edited=len(partly),
        n_rewritten=len(gone), n_unmatchable=len(unk),
        # The convergence test reads these two.
        n_standing_critical_or_major=sev_count(stands, "critical", "major"),
        n_partly_critical_or_major=sev_count(partly, "critical", "major"),
        stands=stands, partly_edited=partly, rewritten=gone, unmatchable=unk)
    p = src.with_name(src.stem + "_triaged.json")
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{len(rows)} findings against {PDF.name}")
    print(f"  quote intact on the page  {len(stands):>4}   <- these stand")
    print(f"  partly edited             {len(partly):>4}   <- read these")
    print(f"  rewritten                 {len(gone):>4}   <- likely answered")
    print(f"  unmatchable               {len(unk):>4}")
    print()
    for label, rs in (("STAND", stands), ("PARTLY EDITED", partly)):
        if not rs:
            continue
        sev = {}
        for r in rs:
            sev[r["severity"]] = sev.get(r["severity"], 0) + 1
        print(f"  {label}: "
              + ", ".join(f"{v} {k}" for k, v in sorted(sev.items())))
        for r in sorted(rs, key=lambda x: {"critical": 0, "major": 1}.get(
                x["severity"], 2)):
            cov = "" if r["coverage"] is None else f" {r['coverage']:.0%}"
            print(f"    [{r['severity']:<8}]{cov:>5} {str(r['id']):<34} "
                  f"{r['quote'][:52]}")
            for reg in r["edited_regions"][:1]:
                print(f"{'':>18}edited: {reg[:88]}")
        print()
    print(f"  critical/major still standing: "
          f"{out['n_standing_critical_or_major']}")
    print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
