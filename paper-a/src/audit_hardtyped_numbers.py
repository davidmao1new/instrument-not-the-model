"""Find numbers the paper prints without reading them from an artifact.

THE CLAIM THIS CHECKS. §1.2 says "Every number in this paper is interpolated
from an artifact on disk by the script that typesets it", and §11 makes the
stronger form: "a number cannot be printed unsourced". Two independent lenses
of round 7 showed both are false -- several data-derived figures are typed
constants in ordinary string literals and would print unchanged if the artifact
they describe were deleted or its value changed.

That is the one claim in this paper whose whole point is that it must hold, so
it gets a check rather than a promise.

WHAT COUNTS AS A VIOLATION. A numeral inside a prose string literal that is NOT
an f-string, in the file that typesets the paper. That over-reports, because
plenty of numerals are not data:

  * section and table cross-references (§4.7, Table 13)
  * design constants fixed by the protocol, not measured (twelve wordings,
    three postings, 2004 for Bertrand and Mullainathan)
  * years, statute numbers, model names (Llama-2-7B, GPT-4o, Local Law 144)
  * ordinary English (one, two, first)

so those are filtered. What survives is a numeral that looks like a measurement:
a percentage, a decimal, a count of cells or records, a p-value. Each one is
either a number that should be interpolated, or a design constant that should
be on the allow-list with a reason.

    sh paper-a/src/_py.sh paper-a/src/audit_hardtyped_numbers.py
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "paper-a" / "src" / "build_paper_v3.py"

# Numerals that are not measurements. Each entry is a reason, not a hole.
EXEMPT = [
    # THE RHO GRID IS A DESIGN CONSTANT, NOT A MEASUREMENT. "0.3", "0.6" and
    # "0.9" are the correlation levels the simulation was run at, used as
    # dictionary keys to select rows out of the artifact and printed back as
    # the axis they label. They became visible when the audit stopped
    # discarding short strings; they are typed because the design says so,
    # and the values they select are interpolated.
    (r"^0\.[369]$", "a rho level in the correlation grid, a design constant"),
    # A FIGURE BELONGING TO A CITED WORK, not to this paper. It cannot be
    # interpolated because it is not our measurement, and citing a precedent
    # without its magnitude is weaker than citing it with one. Bound to the
    # sentence that names the source, so the exemption cannot slide onto one
    # of our own numbers: it matches only inside this attribution.
    (r"up to 12\.6 points of movement in a fairness metric",
     "Qian et al. (2021), NeurIPS: fairness spread across fixed-seed runs"),
    # THE STATUTE PATTERN MUST COME FIRST. The section-reference pattern below
    # matches the "§ 20" of "§ 20-871(a)" and leaves "871" behind to be flagged
    # as a measurement, which is how a statute number sat in this report for
    # several rounds looking like an unfixed defect.
    (r"§\s*20-?87\d(\([a-z]\))?", "statute section number"),
    # EACH OF THESE MATCHES A LIST, NOT A SINGLE NUMBER. "Sections 6.1 and
    # 6.3" used to strip "Sections 6" and leave "6.3" behind, which was then
    # reported as an unfixed typed measurement -- the same half-match failure
    # the statute entry above documents. A cross-reference is as likely to name
    # two targets as one, so the pattern has to consume the whole list.
    (r"§\s*\d+(\.\d+)?((\s*(and|to|,|&|–|-)\s*)(§\s*)?\d+(\.\d+)?)*",
     "section cross-reference, possibly a list"),
    (r"\bTables?\s+\d+((\s*(and|to|,|&|–|-)\s*)\d+)*",
     "table cross-reference, possibly a list"),
    (r"\bFigures?\s+\d+((\s*(and|to|,|&|–|-)\s*)\d+)*",
     "figure cross-reference, possibly a list"),
    (r"\bSections?\s+\d+(\.\d+)?((\s*(and|to|,|&|–|-)\s*)\d+(\.\d+)?)*",
     "section cross-reference, possibly a list"),
    (r"\b(19|20)\d{2}\b", "year"),
    (r"\bLocal Law 144\b|20-?87\d|\bEEO-1\b|\b1607\.4\b|\b29 C\.F\.R\b",
     "statute or rule number"),
    (r"\bLlama-?\s?[23](\.\d)?(-\d+B)?\b|\bMistral-?7B\b|\bv0\.[13]\b"
     r"|\bGPT-?4(\.\d)?o?\b|\bgpt-\d", "model name"),
    (r"\bQ[48]_[A-Z0-9_]+\b|\b[48]-bit\b", "quantization label"),
    (r"\bD\d{1,2}\b|\b[SN]\d\b|\bT\d\b", "condition label"),
    (r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
     "number word"),
    (r"\b(first|second|third|fourth)\b", "ordinal word"),
    (r"\bp = 0\.05\b|\b0\.05\b", "conventional alpha"),
    # ANOTHER STUDY'S NUMBERS, NOT THIS ONE'S. §2 quotes the three p-values
    # from Lahey and Beasley (2009) Table 1, where one half of an audit returns
    # 0.0225 and the other half 0.5000 on the same experiment. There is no
    # artifact here to interpolate them from; the artifact is their 2009 data.
    # Named as three exact literals rather than as a general "cited numbers"
    # rule, because a general rule would be a hole in the one check that
    # enforces the paper's central claim about itself.
    (r"\b0\.0225\b|\b0\.5000\b|\b0\.0513\b",
     "Lahey and Beasley (2009) Table 1, quoted"),
    (r"\b(0\.2|0\.8|0\.5)\b", "stated decision band"),
    (r"\bninety\b|\bhundred\b", "number word"),
    # Conventions, not measurements: the interval level and the power level are
    # choices stated once and used throughout, and printing them from a
    # variable would obscure rather than protect them.
    (r"\b95\s?%", "interval level, a stated convention"),
    (r"\b80\s?%\s+power|\bat\s+80\s?%", "power level, a stated convention"),
    # Design constants fixed by the protocol before any data existed. Each is
    # asserted against its artifact by audit_consistency.py, which is the right
    # guard for a constant: it should not move, and the check says so if it did.
    (r"\btop-100\b|\btop\s+100\b|\btop-20\b", "log-probability window size"),
    (r"\b(10|20|40|4),?000\b", "bootstrap or permutation draw count"),
    (r"\bSHA-?256\b", "hash algorithm name"),
    (r"\b(432|504|864|1,?728|3,?456)\b", "design cell count"),
    (r"\b12 first names\b|\b8 surnames\b|\b24 names\b", "grid dimension"),
    (r"\bHTTP \d{3}\b", "HTTP status code"),
    (r"\b4,?000 replicates\b|\bmultiples of 0\.00025\b",
     "replicate count and its reciprocal, both design constants"),
    # Numbers belonging to OTHER papers, quoted. Interpolating these would be
    # worse, not better: they are facts about a source document, and the right
    # guard for them is that a human read the source, which the citation ledger
    # records.
    (r"\b820\b", "a count from An and Rudinger, quoted"),
    (r"\b5,?748 (first )?names\b",
     "the size of An and Rudinger's name pool, quoted"),
    (r"\b(about|roughly) 200 observations\b|\b70 per male name\b",
     "counts from Bertrand and Mullainathan, quoted"),
    (r"\b24 of the\b", "historical count from a corrected defect, not a "
                       "current measurement"),
    (r"\b812 name combinations\b|\b1,920 vacancies\b",
     "the size of Lippens's name set and vacancy frame, quoted. §4.2 credits "
     "his within-group dispersion result as an antecedent and the comparison "
     "only means something with his numbers in it; they are facts about his "
     "paper and are read by a human from it, not computed from our artifacts."),
    (r"\btemperature 0\.6\b",
     "An et al.'s decoding temperature, quoted. The sentence exists because "
     "they claim seed-based reproducibility at a NON-zero temperature, so the "
     "value is the point and cannot be interpolated from our artifacts."),
]

# What a measurement looks like once the above are stripped.
MEASURE = re.compile(
    r"\d+\s?%"                      # a percentage
    r"|\b\d+\.\d+\b"                # a decimal
    r"|\b\d{3,}\b"                  # a count of any size
    r"|\b\d+ of \d+\b"              # a tally
    r"|\bp = \d"                    # a p-value
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # ONLY WHAT REACHES THE PAGE. Docstrings, headings, column labels and
    # reference entries are strings too, and none of them is prose a reader
    # takes a measurement from. Collect the arguments of P() and the caption=
    # keyword, and nothing else.
    printed = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name == "P":
            printed.extend(node.args)
        for kw in node.keywords:
            if kw.arg == "caption":
                printed.append(kw.value)

    def constants(n):
        """Plain string constants inside an expression, skipping f-strings."""
        for sub in ast.walk(n):
            if isinstance(sub, ast.JoinedStr):
                continue
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                inside_f = any(
                    sub is v for anc in ast.walk(n)
                    if isinstance(anc, ast.JoinedStr) for v in ast.walk(anc))
                if not inside_f:
                    yield sub

    # The reference registry is bibliographic: page ranges, DOIs, arXiv ids.
    ref_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") in ("REFS", "SURVEYED") for t in node.targets):
            ref_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    hits = []
    for expr in printed:
        for node in constants(expr):
            if node.lineno in ref_lines:
                continue
            text = node.value
            # NO LENGTH CUT. This read `if len(text) < 12: continue`, applied
            # before MEASURE ran, so a measurement in its own short
            # concatenation fragment was discarded for being short rather
            # than judged for being a measurement. Exactly one string in the
            # builder was hidden by it, and it was a hardtyped "83 %"
            # artifact fallback -- the single defect this audit exists to
            # find. A cheap filter that runs before the real test decides
            # what the real test is allowed to see.
            if not any(c.isdigit() for c in text):
                continue
            stripped = text
            for pat, _why in EXEMPT:
                stripped = re.sub(pat, " ", stripped, flags=re.I)
            for m in MEASURE.finditer(stripped):
                hits.append((node.lineno, m.group(0),
                             " ".join(text.split())[:88]))

    print(f"{SRC.name}: {len(hits)} numeral(s) in non-f-string prose that look "
          f"like measurements\n")
    for lineno, num, ctx in sorted(hits):
        print(f"  line {lineno:>5}  {num:<12} {ctx}")
    print()
    if hits:
        print("  Each is either a number to interpolate, or a design constant "
              "to exempt with a reason.")
    else:
        print("  every measurement on the page is interpolated")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
