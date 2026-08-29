r"""Find sentences that cannot stand on their own.

WHY THIS EXISTS. A bulk pass over the prose converted em dashes to full stops
wherever the following word looked like it could begin a clause. Its word list
contained verbs, which cannot begin a clause at all, and on a PAIRED dash the
damage compounded: "X -- the aside -- cancels in the difference" became three
fragments, none of them a sentence.

    Whatever the two applications share. The job, the wording, the
    reviewer's mood. Cancels in the difference.

Five of those reached the built PDF. Every other check passed them: the numbers
were right, the claims were scoped, the style audit measures punctuation and
vocabulary rather than grammar, and the test suite pins claims rather than
sentences. A reader caught it on the page.

WHAT THIS CHECKS, and what it does not. There is no parser here and no tagger
installed, so this is a lexicon heuristic with two rules:

  1. A sentence beginning with a finite verb form. "Cancels in the difference."
     Imperatives are the legitimate case and are allowed by an explicit list,
     because the reporting set in section 9 is written as instructions.

  2. A sentence of six words or more containing no verb at all. "The job, the
     wording, the reviewer's mood."

It will miss fragments those two rules do not describe. It catches the one that
actually happened, which is the class a bulk edit produces.

    sh paper-a/src/_py.sh paper-a/src/audit_fragments.py

Exit code 1 if anything is flagged.
"""
from __future__ import annotations

import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"

# Finite verb forms and auxiliaries. A sentence with none of these, and with
# enough words to need one, is doing something other than predicating.
AUX = {
    "is", "are", "was", "were", "be", "been", "being", "am", "has", "have",
    "had", "do", "does", "did", "can", "could", "will", "would", "shall",
    "should", "may", "might", "must", "cannot",
}
VERBISH = re.compile(
    r"\b\w{3,}(?:s|es|ed|ing)\b")   # a crude stand-in for "some verb is here"

# Irregular pasts have no -ed, so VERBISH misses them and a perfectly good
# sentence such as "Not one cell came back the same" is flagged as verbless.
IRREGULAR = {
    "came", "gave", "took", "made", "went", "saw", "found", "held", "put",
    "ran", "got", "kept", "left", "meant", "sent", "set", "told", "thought",
    "began", "broke", "chose", "drew", "fell", "knew", "led", "lost", "met",
    "paid", "read", "rose", "said", "sat", "spoke", "stood", "won", "wrote",
}
# Bare present-tense plurals end in no suffix at all, so VERBISH misses them
# and "Fu and Shi report no demographic effect of any kind" reads as verbless.
BASE_FORM = {
    "report", "move", "give", "show", "carry", "hold", "vary", "pin",
    "state", "run", "use", "need", "cost", "cover", "apply", "differ",
    "depend", "remain", "appear", "occur", "exist", "fail", "survive",
}

# Verbs this paper actually uses, in third person. A sentence opening with one
# of these is the fragment the dash pass produced.
OPENING_VERBS = {
    "cancels", "cancel", "contribute", "contributes", "gives", "give",
    "produce", "produces", "does", "moves", "move", "narrows", "narrow",
    "reports", "report", "shows", "show", "carries", "carry", "holds",
    "predicts", "predict", "collapses", "collapse", "survives", "survive",
}
# Section 9 is written as instructions to an auditor, so imperatives are
# expected there and are not fragments.
IMPERATIVES = {
    "report", "run", "include", "show", "state", "give", "use", "publish",
    "record", "pin", "fix", "check", "cut", "keep", "write", "read", "send",
    "treat", "avoid", "do", "make", "add", "set",
}


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(“])", text)
    return [p.strip() for p in parts if p.strip()]


def looks_like_a_table_row(s: str) -> bool:
    """Table cells and figure captions arrive as pseudo-sentences."""
    digits = sum(c.isdigit() for c in s)
    return (digits > len(s) * 0.18
            or s.count("%") >= 2
            or re.match(r"^(Table|Figure|Fig\.)\s+\d", s) is not None)


def looks_like_a_reference(s: str) -> bool:
    """A bibliography entry is not a sentence and is not trying to be.

    "Sclar, M., Choi, Y., Tsvetkov, Y. and Suhr, A." has no verb because an
    author list has no verb. Flagging the reference section produces fourteen
    findings that can never be fixed, which is how a checker gets ignored.
    """
    if re.match(r"^[A-Z][a-z]+(-[A-Z][a-z]+)?,\s+[A-Z]\.", s):
        return True                       # Surname, Initial.
    if re.search(r"\b(arXiv|doi|pp\.|Proceedings|Journal|Conference|"
                 r"Transactions|Press|Vol\.)\b", s):
        return True
    if re.search(r"\b(19|20)\d{2}[a-z]?\)", s):
        return True                       # a year in parentheses
    return False


def check(text: str) -> list[tuple[str, str]]:
    out = []
    for s in sentences(text):
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", s)
        if (len(words) < 4 or looks_like_a_table_row(s)
                or looks_like_a_reference(s)):
            continue
        # A sentence opening with a quotation begins with whatever the quote
        # begins with, so the first WORD is not the sentence's subject.
        if s.lstrip()[:1] in ('“', '‘', chr(34), "'"):
            continue
        first = words[0].lower()

        if first in OPENING_VERBS and first not in IMPERATIVES:
            out.append(("opens on a finite verb", s))
            continue

        low = {w.lower() for w in words}
        # A SHORT COMMA LIST STANDING ALONE. "A vacancy, a firm, an employer."
        # is five words, below the no-verb floor, and it is not a section
        # lead-in: lead-ins are short noun phrases WITHOUT commas. A
        # comma-separated list with no verb is the orphaned aside of a
        # converted dash pair, which is the bug this file exists for. One of
        # these sat in §6.3 through two fragment sweeps.
        if (3 <= len(words) <= 9 and s.count(",") >= 2
                and not (low & AUX) and not (low & IRREGULAR)
                and not (low & BASE_FORM) and not VERBISH.search(s)):
            out.append(("comma list with no verb", s))
            continue
        # TEN WORDS, NOT SIX. The paper's section lead-ins are short noun
        # phrases by design ("A null-edit control and a byte-identical
        # replicate."), and a heading is not a fragment. Below ten words the
        # rule produced only headings, which is a checker training its reader
        # to skim past it. The bug this file exists for is caught by the
        # opens-on-a-verb rule above, on both instances.
        if (len(words) >= 10 and not (low & AUX) and not (low & IRREGULAR)
                and not (low & BASE_FORM) and not VERBISH.search(s)):
            out.append(("no verb", s))
    return out


def main() -> int:
    if not PDF.exists():
        sys.exit(f"{PDF} not found. Build the paper first.")
    import fitz
    with fitz.open(PDF) as d:
        text = "\n".join(p.get_text() for p in d)

    # THE BIBLIOGRAPHY IS NOT PROSE. Author lists have no verbs, and the
    # annotations after a reference are deliberate fragments. Scanning it
    # produced fourteen findings that could never be fixed, which teaches a
    # reader to skim the report. Everything from the references heading on is
    # cut before checking.
    cut = re.search(r"(?m)^\s*References\s*$", text)
    body = text[:cut.start()] if cut else text

    bad = check(body)
    print("=" * 74)
    print("FRAGMENT AUDIT  --  sentences that cannot stand alone")
    print("=" * 74)
    if not bad:
        print("\n  none found\n")
        return 0
    print(f"\n  {len(bad)} flagged:\n")
    for why, s in bad:
        print(f"    [{why}] {s[:150]}")
    print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
