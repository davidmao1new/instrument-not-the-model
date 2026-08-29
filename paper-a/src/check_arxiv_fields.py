"""Validate text against arXiv's submission-form rules before you paste it.

WHY. arXiv's abstract field is ASCII plus a small TeX subset; Unicode is not
allowed. Paste one em dash and the form returns "bad character(s) in field
Abstract" without telling you which character or where. arXiv's own help names
the usual culprits -- curly quotes, em dashes, ligatures -- and notes they
"typically result from UTF characters being pasted from PDF viewers".

That is exactly what happened on 2026-08-19: the released abstract artifact is
pure ASCII, a hand-edit introduced a single U+2014, and the form rejected it.
The generator cannot prevent that, because the editing happens in a browser. So
this exists to be run on whatever is about to be pasted.

    sh paper-a/src/_py.sh paper-a/src/check_arxiv_fields.py                 # the released abstract
    sh paper-a/src/_py.sh paper-a/src/check_arxiv_fields.py some_file.txt
    ... | sh paper-a/src/_py.sh paper-a/src/check_arxiv_fields.py -

Exit code is non-zero if the text would be rejected.
"""
from __future__ import annotations

import pathlib
import re
import sys
import unicodedata

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "paper-a" / "releases" / "abstract_arxiv.txt"

# arXiv rejects an abstract longer than this.
MAX_CHARS = 1920

# What to use instead, when the character is one of the usual paste artifacts.
# The replacement is what a TeX-aware plain-text field renders sensibly.
SUGGEST = {
    "\u2014": " -- ",     # em dash
    "\u2013": "-",        # en dash
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2026": "...",
    "\u00a0": " ",        # non-breaking space
    "\u2212": "-",        # minus sign
    "\u00d7": "x",
    "\u2264": "<=", "\u2265": ">=", "\u2260": "!=",
    "\u00b1": "+/-",
    "\ufb01": "fi", "\ufb02": "fl",
    "\u00e9": "e",        # or \\'e, but plain is safer in a plain-text field
    "\u00fc": "u", "\u00f6": "o", "\u00e4": "a", "\u00e8": "e",
}

# TeX that arXiv tells you to leave out of an abstract.
UNWANTED_TEX = [
    (r"\\,", "a thin space; delete it"),
    (r"\\(em|it|bf|rm|sf|tt)\b", "a font command; abstracts carry no formatting"),
    (r"\\begin\{|\\end\{", "an environment; abstracts are one paragraph of text"),
    (r"\\footnote", "a footnote; abstracts cannot carry them"),
    (r"\\cite", "a citation command; write the reference out or drop it"),
    (r"\\ref|\\label", "a cross-reference; there is nothing to point at"),
]


def check(text: str, field: str = "Abstract") -> list[str]:
    problems: list[str] = []
    t = text.strip()

    # ---- the one that actually causes "bad character(s)"
    bad = [(i, c) for i, c in enumerate(t) if ord(c) > 126]
    if bad:
        seen: dict[str, int] = {}
        for i, c in bad:
            seen[c] = seen.get(c, 0) + 1
        problems.append(
            f"{len(bad)} non-ASCII character(s) -- this is what produces "
            f"\"bad character(s) in field {field}\":")
        for c, n in sorted(seen.items(), key=lambda kv: -kv[1]):
            i = t.find(c)
            fix = SUGGEST.get(c)
            problems.append(
                f"    U+{ord(c):04X}  {unicodedata.name(c, '?')}  (x{n})"
                + (f"   ->  use {fix!r}" if fix else "   -> no suggestion; retype it"))
            problems.append(f"      ...{t[max(0, i - 40):i + 32]}...")

    # ---- control characters, which paste in invisibly
    ctrl = {c for c in t if ord(c) < 32 and c not in "\n\t"}
    if ctrl:
        problems.append(
            f"control character(s) {[hex(ord(c)) for c in sorted(ctrl)]} -- "
            f"invisible, and they survive a visual check")

    # ---- length
    n = len(t)
    if field == "Abstract":
        if n > MAX_CHARS:
            problems.append(f"{n} characters; arXiv rejects an abstract over "
                            f"{MAX_CHARS}. Cut {n - MAX_CHARS} more.")
        elif n > MAX_CHARS - 30:
            problems.append(
                f"{n} characters, {MAX_CHARS - n} under the limit. That is "
                f"valid but tight -- any later edit can push it over, and the "
                f"form will not tell you that is why.")

    # ---- formatting arXiv asks you to omit
    for pat, why in UNWANTED_TEX:
        for m in re.finditer(pat, t):
            problems.append(f"{m.group()!r} is {why}")

    if t.lower().startswith("abstract"):
        problems.append("starts with the word \"Abstract\"; arXiv adds the "
                        "label itself")
    if "\n\n" in t:
        problems.append("contains a blank line; abstracts are one paragraph")
    # A newline NOT followed by whitespace is silently stripped, which can weld
    # two words together.
    for m in re.finditer(r"\n(?=\S)", t):
        problems.append(
            f"a line break at offset {m.start()} is not followed by "
            f"whitespace. arXiv strips it and the words either side join: "
            f"...{t[max(0, m.start() - 24):m.start() + 24]!r}...")
    return problems


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "-":
        text, src = sys.stdin.read(), "stdin"
    else:
        p = pathlib.Path(arg) if arg else DEFAULT
        if not p.exists():
            sys.exit(f"{p} not found")
        text, src = p.read_text(encoding="utf-8"), str(p)

    print("=" * 74)
    print(f"arXiv field check  --  {src}")
    print("=" * 74)
    problems = check(text)
    print(f"\n  {len(text.strip())} characters, "
          f"{sum(1 for c in text if ord(c) > 126)} non-ASCII\n")
    if not problems:
        print("  clean. Safe to paste.")
        return 0
    for p_ in problems:
        print("  " + p_)
    print("\n  Fix these, or re-run with the corrected text.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
