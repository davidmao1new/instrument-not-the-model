"""What "adversarial review" actually was, as a table of counts.

WHY THIS EXISTS. A project summary said the manuscript had been through
"seven rounds of adversarial review against its own artifacts". An external
reviewer read that on 2026-08-19 and objected that the phrase is unverifiable
as stated: reviewed how, against what, and where is the record?

The objection is right. The phrase asserts rigour without exposing anything that could
falsify it, in a paper whose entire thesis is that a number reported without
its procedure cannot be checked. It was also WRONG: six rounds have released
artifacts, not seven, and this script is how that was discovered.

So the claim is replaced by its ledger. Every figure below is read from the
released critique files in `paper-a/releases/`, and a round with no released
artifact is reported as absent rather than quietly folded into a total.

WHAT THE COLUMNS MEAN.

  examined    findings raised by the adversarial readers in that round.
  confirmed   findings that survived a refutation pass. The gap between the
              two columns is the round's own false-positive rate, and it is
              large -- which is the point of running the refutation pass.
  serious     confirmed findings graded critical or major.
  stands      of the confirmed findings, how many left the criticised passage
              UNCHANGED after triage against the rebuilt document. A high
              number here is not a good sign; it means the round found things
              that had not yet been acted on.
  changed     passages partly edited or rewritten in response.

    sh paper-a/src/_py.sh paper-a/src/build_review_ledger.py
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
REL = ROOT / "paper-a" / "releases"
CHANGELOG = ROOT / "CHANGELOG.md"
OUT_MD = ROOT / "paper-a" / "docs" / "REVIEW_LEDGER.md"
OUT_JSON = ROOT / "paper-a" / "data" / "reference" / "review_ledger.json"

# The rounds this project ran, and the file each one's merged output landed in.
# ROUND 3 IS DELIBERATELY LISTED WITH NO FILE. There is no released artifact for
# it and no changelog section naming it; the numbering runs 1, 2, 4, 5, 6, 7
# because the two earliest audits were folded into Amendments 14 and 15 before
# the round vocabulary settled. Listing it as absent is the whole reason this
# script exists -- a silent renumber is how "six" became "seven".
ROUNDS = [
    (1, "critique_round1.json", None),
    (2, "critique_round2.json", "critique_round2_triaged.json"),
    (3, None, None),
    (4, "critique_round4_half1.json", "critique_round4_half1_triaged.json"),
    (5, "critique_round5.json", "critique_round5_triaged.json"),
    (6, "critique_round6.json", "critique_round6_triaged.json"),
    (7, "critique_round7.json", "critique_round7_triaged.json"),
]


def load(name: str | None) -> dict | None:
    if not name:
        return None
    p = REL / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def withdrawals() -> list[str]:
    """Claims the changelog records as withdrawn.

    A withdrawal is the most falsifiable thing this project can point at: a
    result that was in a draft, was tested, and lost. Counted by matching the
    changelog's own convention rather than by memory, and the matched lines are
    printed so the count can be audited by eye.
    """
    if not CHANGELOG.exists():
        return []
    out = []
    for ln in CHANGELOG.read_text(encoding="utf-8").split("\n"):
        s = ln.strip()
        if re.search(r"\bwithdraw(n|s|al)?\b", s, re.I) and len(s.split()) >= 4:
            out.append(" ".join(s.split()))
    return out


def main() -> int:
    rows, absent = [], []
    for num, merged, triaged in ROUNDS:
        d, t = load(merged), load(triaged)
        if d is None:
            absent.append(num)
            continue
        changed = None
        if t and "n_rewritten" in t:
            changed = t.get("n_rewritten", 0) + t.get("n_partly_edited", 0)
        rows.append({
            "round": num,
            "examined": d.get("n_examined"),
            "confirmed": d.get("n_confirmed"),
            "refuted": d.get("refuted_count"),
            "serious": d.get("n_serious"),
            "stands": (t or {}).get("n_stands"),
            "changed": changed,
        })

    tot = {k: sum(r[k] or 0 for r in rows)
           for k in ("examined", "confirmed", "refuted", "serious")}
    wd = withdrawals()

    led = {
        "_what": "What the adversarial critique loop actually did, per round.",
        "_why": ("The phrase 'seven rounds of adversarial review' is not "
                 "falsifiable and was also wrong. This is the ledger behind it."),
        "_source": "paper-a/releases/critique_round*.json",
        "n_rounds_with_artifacts": len(rows),
        "rounds_without_artifacts": absent,
        "rounds": rows,
        "totals": tot,
        "n_withdrawal_lines_in_changelog": len(wd),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(led, indent=1), encoding="utf-8")

    w = max(1, tot["examined"])
    md = [
        "# What the adversarial review actually was",
        "",
        "Generated by `paper-a/src/build_review_ledger.py` from the released "
        "critique artifacts. Regenerate rather than edit.",
        "",
        "> Written because an external reviewer read the phrase *“seven rounds "
        "of adversarial review”* in a project summary and pointed out that it "
        "means nothing unless it is reproducible. The objection was right, and "
        "the phrase was also wrong: **there are "
        f"{len(rows)} rounds with released artifacts, not seven.**",
        "",
        "| round | findings examined | confirmed | refuted | serious | passage unchanged after triage | passage edited |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        f = lambda v: "—" if v is None else str(v)  # noqa: E731
        md.append(f"| {r['round']} | {f(r['examined'])} | {f(r['confirmed'])} | "
                  f"{f(r['refuted'])} | {f(r['serious'])} | {f(r['stands'])} | "
                  f"{f(r['changed'])} |")
    md += [
        f"| **total** | **{tot['examined']}** | **{tot['confirmed']}** | "
        f"**{tot['refuted']}** | **{tot['serious']}** | | |",
        "",
        f"**{tot['examined']} findings were raised across {len(rows)} rounds. "
        f"{tot['refuted']} did not survive a refutation pass "
        f"({tot['refuted'] / w:.0%}), which is the loop's own false-positive "
        f"rate and the reason the refutation pass exists. {tot['confirmed']} "
        f"were confirmed and {tot['serious']} of those graded critical or "
        f"major.**",
        "",
        f"Round{'s' if len(absent) != 1 else ''} "
        f"{', '.join(str(a) for a in absent)} "
        f"{'have' if len(absent) != 1 else 'has'} no released artifact. The "
        "numbering runs 1, 2, 4, 5, 6, 7 because the two earliest audits were "
        "folded into changelog Amendments 14 and 15 before the round "
        "vocabulary settled. That is exactly how a six became a seven, and it "
        "is why this file reports absence rather than renumbering.",
        "",
        "## What a round consisted of",
        "",
        "1. **Raise.** Independent adversarial readers, given the built PDF and "
        "the artifact set, each working a distinct lens (statistics, "
        "literature, internal consistency, geometry). Rounds 6 and 7 record "
        "the lens count per half.",
        "2. **Refute.** Every finding re-derived against the artifacts before "
        "anything was changed. The `refuted` column is what did not survive.",
        "3. **Triage.** Each surviving finding matched back against the "
        "rebuilt document to see whether the criticised passage had actually "
        "changed — the `stands` column is how many had not.",
        "4. **Apply, or record why not.** Changes land in the builder; the "
        "claim-level consequences land in `CHANGELOG.md`.",
        "",
        "## The falsifiable part",
        "",
        f"`CHANGELOG.md` carries **{len(wd)} lines recording a withdrawal** — "
        "a claim that was in a draft, was tested, and lost. That is the number "
        "worth quoting, because a review process that never costs the author "
        "anything is not a review process.",
        "",
    ]
    for line in wd[:12]:
        md.append(f"- {line[:150]}")
    if len(wd) > 12:
        md.append(f"- …and {len(wd) - 12} more.")
    md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print("=" * 78)
    print("REVIEW LEDGER")
    print("=" * 78)
    print(f"  rounds with artifacts : {len(rows)}   (absent: {absent})")
    print(f"  findings examined     : {tot['examined']}")
    print(f"  confirmed             : {tot['confirmed']}")
    print(f"  refuted               : {tot['refuted']}  "
          f"({tot['refuted'] / w:.0%} of examined)")
    print(f"  serious               : {tot['serious']}")
    print(f"  withdrawal lines      : {len(wd)}")
    print(f"\n  wrote {OUT_MD.relative_to(ROOT)}")
    print(f"  wrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
