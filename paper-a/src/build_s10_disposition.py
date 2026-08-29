r"""Section 10, sentence by sentence, with what each sentence costs and carries.

WHY A WORKSHEET AND NOT A DRAFT. FAccT prohibits the use of LLMs to generate
text for publications and the submission's generative-AI statement asserts the
author wrote the prose, so the sentences of the FAccT §10 have to be his. What
can be done for him is the part that is arithmetic: §10 is the largest single
overage in the paper against its 400-word budget, and the reason it is hard to
cut is not padding. `audit_section_dependencies.py` measures zero near-twin
sentence pairs inside it. Every sentence says something, so the cut is most of
a section with no slack in it, and that is a decision list rather than a
writing task.

No figure is quoted in this docstring. Three were, and all three went stale the
moment §10.2 gained a paragraph, in the one script whose job is to stop exactly
that. They are printed at run time instead, where they are computed.

So this prints the decision list. For each sentence: what it costs in words,
which numbers it carries, which of those numbers the preprint states NOWHERE
ELSE, and whether it is narrating how the paper came to know a limitation
rather than stating the limitation.

THE COLUMN THAT ACTUALLY BINDS is "only here". The cut drops every number
that appears nowhere else in the paper unless each is deliberately kept. A number leaving the paper is a decision; a number leaving
because nobody counted is an accident, and the difference is invisible once the
section is rewritten.

    sh paper-a/src/_py.sh paper-a/src/build_s10_disposition.py
"""
from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import audit_section_dependencies as dep  # noqa: E402
import audit_correction_narration as narr  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
OUT = ROOT / "paper-a" / "facct" / "S10.md"

SECTION = "10"
BUDGET = 400

# The three subsection headings, from build_paper_v3.py. Named rather than
# discovered: a heading that stops appearing should fail this build loudly, not
# quietly fold its sentences into the subsection above.
SUBS = [
    ("10.1", "Internal validity. Are these numbers right?"),
    ("10.2", "External validity. How far does this go?"),
    ("10.3", "Construct validity. Is this measuring what it is called?"),
]

# COMPRESS.md: "Do not cut the three bound-not-absence framings. Those are the
# sentences that make the limitations section read as competence rather than
# hedging." They are matched here on the shape of the claim, not on a
# remembered wording, so a rewrite upstream does not silently lose the flag.
#
# Written against the sentences that are actually on the page, not against
# COMPRESS.md's paraphrase of them. The first version of this list was written
# from the paraphrase and two of its three patterns matched nothing, so §10's
# construct-validity and no-claim framings were carrying a "must survive" flag
# that never fired. test_facct_s10.py now fails if any of the three stops
# matching, because a protection that protects nothing is worse than none: it
# reads on the worksheet as though the job were done.
MUST_SURVIVE = [
    ("a bound, not an absence",
     r"\ba bound,? not an absence\b|\bbounded within\b"),
    ("the design's limit, not a result",
     r"design[’']s limit(?:s)? rather than results\b|"
     r"\bclaim only that none larger exists\b|"
     r"\bonly partly reassuring\b|\bthose are the design[’']s limit\b"),
    ("the claim is declined, not hedged",
     r"\bWe make none\b|\bnone is a place we have looked\b|"
     r"\bcannot support a claim about\b"),
]


def numbers_in(s: str) -> set[str]:
    """The measurements in a sentence, which is not everything that looks like
    one.

    "Appendix F states this" contains no measurement, and neither does "Appendix D.1".
    Counting them made cross-references show up in the list of numbers §10 is
    the only place in the paper to state, which is the one column of this
    worksheet that is supposed to be trustworthy: it is what tells the author
    that cutting a sentence removes a fact from the paper entirely.
    """
    out = set()
    for m in dep.NUMBER.finditer(s):
        before = s[max(0, m.start() - 3):m.start()]
        if "§" in before or re.search(r"\b[Ss]ection\s*$", before):
            continue
        out.add(m.group().strip())
    return out


def sentences(s: str) -> list[str]:
    """Every sentence, including the short ones.

    NOT `audit_section_dependencies.sentences`, which drops anything under
    eight words. That is right for its job -- it hunts near-duplicate pairs and
    a short sentence has too few shingles to match on -- and wrong for this
    one. A worksheet that silently omits sentences is worse than no worksheet:
    the first run of this script reported 2,329 words against the 2,472
    COMPRESS.md measured at the time, and the missing 143 were short sentences
    that would have been cut or kept without ever appearing on the list. The totals are
    asserted against the section below so this cannot recur quietly.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z“(])", s)
    return [p.strip() for p in parts if p.strip()]


def subsections(body: str) -> list[tuple[str, str, str]]:
    """(number, title, text) for each 10.x, in order."""
    marks = []
    for num, title in SUBS:
        needle = f"{num} {title}"
        i = body.find(needle)
        if i < 0:
            # Try the heading with its punctuation collapsed the way PyMuPDF
            # sometimes delivers it.
            i = body.find(f"{num} {title.split('.')[0]}")
        if i < 0:
            sys.exit(f"subsection heading not found in the extracted text: "
                     f"{needle!r}")
        marks.append((i, num, title))
    marks.sort()
    out = []
    # THE LEAD-IN COUNTS. Whatever sits between the §10 heading and §10.1 is
    # part of the section and part of the budget. Starting at the first
    # subsection dropped it, which is how the first two runs of this script
    # disagreed with the 2,472 words COMPRESS.md measured at the time.
    lead = body[:marks[0][0]].strip()
    # dep.sections() slices from the start of the heading, so the section's own
    # title is the first thing in the lead-in and would be counted and printed
    # as though it were a sentence of prose.
    lead = re.sub(r"^" + SECTION + r"\s+Threats to validity\s*", "", lead)
    if lead:
        out.append(("10", "(lead-in, before 10.1)", lead))
    for j, (i, num, title) in enumerate(marks):
        end = marks[j + 1][0] if j + 1 < len(marks) else len(body)
        start = i + len(f"{num} {title}")
        out.append((num, title, body[start:end].strip()))
    return out


def main() -> int:
    if not PDF.exists():
        sys.exit("paper not built; run build_paper_v3.py first")
    import fitz
    with fitz.open(PDF) as doc:
        text = " ".join(" ".join(p.get_text().split()) for p in doc)

    secs = dep.sections(text)
    if SECTION not in secs:
        sys.exit(f"§{SECTION} not found in the extracted text")
    body = secs[SECTION]

    # Numbers this section is the only place in the paper to state.
    nums = {k: numbers_in(v) for k, v in secs.items()}
    others = set().union(*(v for k, v in nums.items() if k != SECTION))
    only_here = {n for n in nums[SECTION] - others if len(n) > 2}
    # A HEADING NUMBER IS NOT A MEASUREMENT. "10.1" is unique to §10 in the
    # trivial sense that it is §10's own subsection, and it survived the
    # cross-reference filter because a heading carries no "§". Left in, the
    # count in this worksheet's header disagreed with the number of tags in
    # its body -- 13 against 10 -- which sends the author looking for three
    # facts that do not exist.
    only_here -= {n for n, _ in SUBS}

    rows, kept_words, total_words = [], 0, 0
    subs = subsections(body)
    for num, title, sub in subs:
        for s in sentences(sub):
            words = len(s.split())
            total_words += words
            carried = sorted(numbers_in(s) & only_here)
            tells = sorted({why for why, pat in narr.MARKERS
                            if re.search(pat, s, re.I)})
            must = sorted({why for why, pat in MUST_SURVIVE
                           if re.search(pat, s, re.I)})
            if must or carried:
                mark, kept_words = "KEEP", kept_words + words
            elif tells:
                mark = "CUT"
            else:
                mark = "--"
            rows.append((num, title, s, words, carried, tells, must, mark))

    # NOTHING IS SILENTLY DROPPED. Every word of the three subsections has to
    # appear on the worksheet, or the cut is being decided against a section
    # the author cannot see all of.
    head_words = sum(len(f"{num} {title}".split()) for num, title in SUBS)
    head_words += len(f"{SECTION} Threats to validity".split())
    in_subs = sum(len(sub.split()) for _, _, sub in subs)
    if total_words != in_subs:
        sys.exit(f"worksheet lists {total_words} words but the subsections "
                 f"hold {in_subs}; {in_subs - total_words} went missing in "
                 "the sentence split")

    # THE HEADER COUNT AND THE TAGS MUST AGREE. The header promises N numbers
    # that leave the paper if their sentence goes; if fewer than N are tagged
    # on a sentence, the author is being sent to look for a fact that is not
    # on the list.
    tagged = set().union(*[set(r[4]) for r in rows]) if rows else set()
    if tagged != only_here:
        sys.exit(f"{len(only_here)} numbers are unique to §{SECTION} but "
                 f"{len(tagged)} are tagged on a sentence; "
                 f"unattributed: {sorted(only_here - tagged)}")

    # ---- the worksheet -------------------------------------------------
    out = [
        "# §10 Threats to validity, sentence by sentence",
        "",
        "Generated by `paper-a/src/build_s10_disposition.py`. Regenerate "
        "rather than edit.",
        "",
        "**This is a worksheet and not a draft.** FAccT prohibits the use of "
        "LLMs to generate text for publications and the submission's "
        "generative-AI statement asserts you wrote the prose. What is below "
        "is the arithmetic of the cut: what each sentence costs, what it "
        "carries that nothing else carries, and whether it is narrating how "
        "the paper came to know a limitation rather than stating one. The "
        "sentences of the FAccT §10 do not exist yet.",
        "",
        f"**{total_words} words of prose against a {BUDGET}-word budget, "
        f"an {100 * (total_words - BUDGET) // max(total_words, 1)} % cut** "
        "-- the largest in the paper. `audit_section_dependencies.py` finds "
        "**zero** near-twin sentence pairs inside §10, so there is no "
        "redundancy to delete: every cut drops something the section "
        "currently says.",
        "",
        f"(A section word count that includes the {len(subs) - 1} "
        "subsection headings and the section title will read "
        f"{total_words + head_words} rather than {total_words}. This "
        "worksheet strips them, because a heading is not a sentence you can "
        "cut.)",
        "",
        "## The three marks",
        "",
        "| mark | meaning |",
        "|---|---|",
        "| `KEEP` | carries a number stated nowhere else in the paper, or one "
        "of the three bound-not-absence framings `COMPRESS.md` says must "
        "survive |",
        "| `CUT` | narrates how the paper came to know the limitation. The "
        "changelog carries that record; the paper needs the limitation |",
        "| `--` | neither. Yours to decide |",
        "",
        f"Keeping every `KEEP` sentence verbatim would cost **{kept_words} "
        f"words** against the {BUDGET}-word budget"
        + (f", which is {kept_words - BUDGET} words more than the whole "
           "section is allowed. So this is not a section you can assemble by "
           "keeping sentences. Every `KEEP` marks a FACT that has to survive, "
           "and most of them are one clause of substance inside a sentence of "
           "setup; the clause is what you are keeping."
           if kept_words > BUDGET else
           ". That is the floor if you keep them as written, and the mark "
           "means the fact has to survive rather than the sentence."),
        "",
        f"**{len(only_here)} numbers appear only in §10.** Each one below "
        "tagged `only here` leaves the paper entirely if its sentence goes "
        "and you do not move the number somewhere else.",
        "",
    ]

    for num, title in [(n, t) for n, t, _ in subs]:
        mine = [r for r in rows if r[0] == num and r[1] == title]
        if not mine:
            continue
        w = sum(r[3] for r in mine)
        out += ["---", "", f"## §{num} {title}", "",
                f"{len(mine)} sentences, {w} words.", ""]
        for _, _, s, words, carried, tells, must, mark in mine:
            flags = []
            if carried:
                flags.append("`only here`: " + ", ".join(carried))
            if must:
                flags.append("**must survive** -- " + "; ".join(must))
            if tells:
                flags.append("narration: " + ", ".join(tells))
            out.append(f"- **`{mark}`** ({words}w) {s}")
            if flags:
                out.append(f"  - {' | '.join(flags)}")
        out.append("")

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")

    n_keep = sum(1 for r in rows if r[7] == "KEEP")
    n_cut = sum(1 for r in rows if r[7] == "CUT")
    print("=" * 78)
    print("§10 DISPOSITION")
    print("=" * 78)
    print(f"\n  {len(rows)} sentences, {total_words} words, "
          f"budget {BUDGET}")
    print(f"  {n_keep} KEEP ({kept_words} words), {n_cut} CUT, "
          f"{len(rows) - n_keep - n_cut} to decide")
    print(f"  {len(only_here)} numbers stated only in §10")
    print(f"\n  wrote  {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
