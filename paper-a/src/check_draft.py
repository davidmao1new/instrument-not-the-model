r"""Check a hand-written FAccT section against everything the paper knows.

The author writes the prose; this reads it back. Six checks, each of which has
already caught something real in this project at least once:

  BUDGET      words against the section's target. Cheap and the one that
              actually governs whether the paper fits.
  TYPED       a measurement written as digits instead of a macro. The paper's
              central claim about itself is that no number is typed, and
              audit_hardtyped_numbers.py cannot see inside a .tex file.
  UNDEFINED   a macro used that numbers.tex does not define. In LaTeX an
              undefined macro is a compile error; better to hear it here.
  DROPPED     numbers the preprint's version of this section was the only
              place to state. Losing one is a decision, not an accident.
  DANGLING    a cross-reference to a section that no longer exists, or to one
              that moved into the appendix.
  REGISTER    the machine-writing tells audit_prose_style.py measures against
              the field: em-dash pile-ups, throat-clearing, showcase verbs.
  CAPTIONS    which figure captions are still the preprint's. A caption is
              prose and FAccT prohibits LLM-generated text, so the eight of
              them carry the same obligation the body does -- and they are
              easy to forget, because they arrive already typeset and looking
              finished. Also flags any figure built but never \input, and any
              \input twice.

    sh paper-a/src/_py.sh paper-a/src/check_draft.py
    sh paper-a/src/_py.sh paper-a/src/check_draft.py --section 10
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

ROOT = pathlib.Path(__file__).resolve().parents[2]
FACCT = ROOT / "paper-a" / "facct"
MAIN = FACCT / "main.tex"
NUMBERS = FACCT / "generated" / "numbers.tex"

BUDGET = {
    "Introduction": 700,
    "Background and related work": 550,
    "Design": 700,
    "What the auditor writes": 1300,
    "What the auditor runs": 500,
    "What the auditor computes": 800,
    "What the sensitivity is not": 350,
    "What the field reports": 600,
    "A minimum reporting set": 600,
    "Threats to validity": 400,
    "Conclusion": 250,
}

# LaTeX the author legitimately writes that is not a number macro.
KNOWN = {
    "section", "subsection", "paragraph", "label", "ref", "cite", "citep",
    "citet", "input", "emph", "textbf", "textit", "S", "times", "%", "&",
    "documentclass", "usepackage", "begin", "end", "title", "author",
    "maketitle", "keywords", "bibliographystyle", "bibliography", "appendix",
    "item", "footnote", "url", "href", "newcommand", "makecell", "toprule",
    "midrule", "bottomrule", "addlinespace", "centering", "caption",
    "textwidth", "alpha", "rho", "sigma", "geq", "leq", "pm", "approx",
    "quad", ",", "\\",
}

# The register tells, taken from audit_prose_style.py.
TELLS = [
    (r"\bit is worth noting\b|\bit is important to\b|\bnotably,", "throat-clearing"),
    (r"\bwhich is self-\w+|, which is itself\b", "self-commentary"),
    (r"\bdelve\b|\bunderscore[sd]?\b|\bshowcase[sd]?\b|\bleverage[sd]?\b", "showcase verb"),
    (r"\bmay potentially\b|\bcould possibly\b|\bmight perhaps\b", "stacked hedge"),
    (r"\bin order to\b", "'to' is usually enough"),
    (r"\bthe fact that\b", "usually deletable"),
    (r"\bnot only\b[^.]{0,60}\bbut also\b", "not-only-but-also"),
]


CAPNUMBERS = FACCT / "generated" / "capnumbers.tex"


def defined_macros() -> set[str]:
    """Every number macro the author may use.

    BOTH FILES. numbers.tex holds the body's quantities and capnumbers.tex the
    ones only a figure caption states, sliced out of the generated caption text
    by build_caption_kit.py. Reading only the first made every caption macro
    look undefined, which trains the author to ignore the UNDEFINED line.
    """
    out: set[str] = set()
    for f in (NUMBERS, CAPNUMBERS):
        if f.exists():
            out |= set(re.findall(r"\\newcommand\{\\(\w+)\}",
                                  f.read_text(encoding="utf-8")))
    return out


def typed_measurements(body: str, allow: set[str] = frozenset()) -> list[str]:
    """Numbers written as digits where a macro was available.

    Factored out of the section loop so the figure captions get the same check.
    They did not have it, and they are the densest numbers in the submission:
    the noise-floor caption alone states five measurements. A caption arrives
    already typeset and reads as finished, so a hand-rewrite that retypes
    "47.9 %" looks exactly like one that uses the macro.

    `allow` carries the numbers that are properties of the method rather than
    measurements of it -- the logistic's maximum slope, the interval level --
    which have no artifact behind them to disagree with. The list lives in
    build_caption_kit.CONSTANTS so the two cannot drift apart.
    """
    out = []
    # A COMMENT IS NOT PROSE. This gate read the whole file, so a template
    # note like "https://dl.acm.org/ccs (pick 2-3)" was scanned as if it
    # were a sentence in the paper. check_iclr.py has stripped comments
    # since it was written.
    body = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln)
                     for ln in body.split("\n"))
    for m in re.finditer(r"(?<![\\\w])\d[\d,.]*\s*\\?%?", body):
        s = m.group().strip()
        # TRAILING PUNCTUATION IS NOT PART OF THE NUMBER. "[\d,.]*" swallows
        # the comma in "slope 0.25, and the one that is there", which made the
        # match "0.25," -- equal to nothing in `allow`, so a declared method
        # constant was reported as a typed measurement whenever it happened to
        # end a clause. It also made every report of such a number ugly.
        if not s.endswith("%"):
            s = s.rstrip(",.")
        # NO SINGLE-DIGIT CUT. It ran before the real test and decided what
        # the real test was allowed to see -- the same defect removed from
        # check_iclr.py's TYPED gate, left standing in its sibling. 15 of
        # the 55 generated FAccT macros hold a single-digit value, five of
        # them Cap* macros, so about a quarter of this submission's
        # measurements could be typed as digits with this gate printing
        # "none". What the cut was really hiding is handled below, by
        # context, in the manner check_iclr.py uses.
        if s.rstrip("\\%").strip().rstrip(",.") in allow:
            continue
        # A NUMERAL INSIDE AN IDENTIFIER IS PART OF THE NAME. "Llama-2-7B",
        # "Mistral v0.1", "Llama-3.1-8B-Instruct": the whitespace-delimited
        # token carries letters, so the digits in it name a checkpoint
        # rather than measure one. This is the category the single-digit cut
        # was silently standing in for -- 66 of the matches it hid.
        # The match pattern ends in `\s*`, so m.end() can sit past a space
        # and the window would swallow the NEXT word -- which made "6 points"
        # read as a token containing letters, and the guard skipped the very
        # measurements removing the single-digit cut was meant to expose.
        _end = m.start() + len(m.group().rstrip())
        _lo = body.rfind(" ", 0, m.start()) + 1
        _hi = body.find(" ", _end)
        token = body[_lo:_hi if _hi > 0 else len(body)]
        if any(c.isalpha() for c in token.strip("\\%(),.;:")):
            continue
        # A section number is not a measurement. "\S{}4.7" and "§4.7" both
        # end in digits and both tripped this on the first run.
        before = body[max(0, m.start() - 30):m.start()]
        if re.search(r"(\\S\{\}|§)\s*$", before):
            continue
        # A cross-reference names a float, it does not measure one. The
        # captions point at the preprint's numbering throughout.
        if re.search(r"\b(Figures?|Tables?|Sections?|Appendix|Appendices)"
                     r"\s*$", before):
            continue
        if "\\label" in before or "\\ref" in before or "\\cite" in before:
            continue
        out.append(s)
    return out


def runs(text: str, n: int) -> set[str]:
    """Every run of n consecutive words, lowercased, markup stripped."""
    w = re.findall(r"[a-z]+", re.sub(r"\\[A-Za-z]+\*?", " ", text.lower()))
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def strip_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.split("\n"))


def sections(tex: str) -> list[tuple[str, str]]:
    out, marks = [], list(re.finditer(r"\\section\*?\{([^}]*)\}", tex))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(tex)
        out.append((m.group(1), tex[m.end():end]))
    return out


def preprint_uniques() -> dict[str, set[str]]:
    """Numbers the preprint states only in a given section."""
    try:
        import audit_section_dependencies as dep
    except Exception:  # noqa: BLE001
        return {}
    pdf = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
    if not pdf.exists():
        return {}
    try:
        import fitz
    except ImportError:
        return {}
    with fitz.open(pdf) as doc:
        text = " ".join(" ".join(p.get_text().split()) for p in doc)
    sec = dep.sections(text)
    nums = {k: set(n.strip() for n in dep.NUMBER.findall(v))
            for k, v in sec.items()}
    out = {}
    for k, v in nums.items():
        others = set().union(*(w for j, w in nums.items() if j != k))
        out[k] = {n for n in v - others if len(n) > 2}
    return out


# section title -> the preprint's number for it
NUMBERED = {"Introduction": "1", "Background and related work": "2",
            "Design": "3", "What the auditor writes": "4",
            "What the auditor runs": "5", "What the auditor computes": "6",
            "What the sensitivity is not": "7", "What the field reports": "8",
            "A minimum reporting set": "9", "Threats to validity": "10",
            "Conclusion": "11"}


def main() -> int:
    if not MAIN.exists():
        sys.exit(f"{MAIN.relative_to(ROOT)} not found")
    only = None
    if "--section" in sys.argv:
        only = sys.argv[sys.argv.index("--section") + 1]

    tex = strip_comments(MAIN.read_text(encoding="utf-8"))
    tex = tex.split(r"\appendix", 1)[0]
    macros = defined_macros()
    uniq = preprint_uniques()
    problems = 0
    written = 0

    print("=" * 78)
    print("DRAFT CHECK  --  hand-written sections against the artifacts")
    print("=" * 78)

    for title, body in sections(tex):
        if title not in BUDGET:
            continue
        num = NUMBERED.get(title)
        if only and num != only:
            continue
        prose = re.sub(r"\\input\{[^}]*\}", "", body)
        prose = re.sub(r"\\[A-Za-z]+\*?(\{[^}]*\})?", " ", prose)
        words = len(prose.split())
        if words < 25:
            print(f"\n§{num}  {title}   -- not written yet")
            continue
        written += 1
        budget = BUDGET[title]
        flag = "OVER" if words > budget else "ok"
        print(f"\n{'-' * 78}\n§{num}  {title}   {words} / {budget} words  [{flag}]")
        if words > budget:
            problems += 1

        # ---- typed measurements
        typed = typed_measurements(body)
        if typed:
            problems += 1
            print(f"  TYPED MEASUREMENTS ({len(typed)}): "
                  f"{', '.join(sorted(set(typed))[:10])}")
            print("     -> add to EXPORT in build_paper_v3.py and use a macro")
        else:
            print("  TYPED MEASUREMENTS: none")

        # ---- undefined macros
        used = set(re.findall(r"\\([A-Za-z]+)", body))
        undef = sorted(u for u in used - macros - KNOWN if u[0].isupper())
        if undef:
            problems += 1
            print(f"  UNDEFINED MACROS: {', '.join('\\' + u for u in undef)}")

        # ---- numbers the preprint stated only here
        if num in uniq and uniq[num]:
            kept = {n for n in uniq[num] if n in body}
            lost = sorted(uniq[num] - kept)
            print(f"  UNIQUE NUMBERS FROM THE PREPRINT: kept {len(kept)}, "
                  f"dropped {len(lost)}")
            if lost:
                print(f"     dropped: {', '.join(lost[:12])}")
                print("     -> each is now stated nowhere in the paper. "
                      "Intended?")

        # ---- dangling cross-references
        refs = set(re.findall(r"\\S\{\}\s*(\d+(?:\.\d+)*)", body)) | \
            set(re.findall(r"§\s?(\d+(?:\.\d+)*)", body))
        live = {v for v in NUMBERED.values()}
        dead = sorted(r for r in refs if r.split(".")[0] not in live)
        if dead:
            problems += 1
            print(f"  DANGLING CROSS-REFERENCES: "
                  f"{', '.join('§' + d for d in dead)}")

        # ---- register
        hits = []
        for pat, why in TELLS:
            for m in re.finditer(pat, prose, re.I):
                hits.append((m.group()[:34], why))
        em = prose.count("---") + prose.count("—")
        per_k = 1000 * em / max(words, 1)
        if hits or per_k > 3.0:
            print(f"  REGISTER: em-dashes {per_k:.1f}/1k "
                  f"(field median 0.35, paper 2.55)")
            for h, why in hits[:5]:
                print(f"     {why}: {h!r}")

    problems += check_captions()

    print("\n" + "=" * 78)
    if not written:
        print("nothing written yet. Start with §10 -- see facct/COMPRESS.md.")
    else:
        print(f"{written} section(s) written, {problems} with something to fix")
    return 0


CAPTIONS = FACCT / "generated" / "captions.tex"


def check_captions() -> int:
    """Which of the eight captions are still the preprint's, and are they in.

    A caption arrives from the generator already typeset and looking finished,
    which is exactly why it gets forgotten: nothing about it looks unwritten.
    It is still prose, and the submission's disclosure asserts the author wrote
    the prose.
    """
    if not CAPTIONS.exists():
        return 0
    raw = MAIN.read_text(encoding="utf-8")
    body = strip_comments(raw)
    defined = re.findall(r"\\newcommand\{\\FigCap(\w+)\}",
                         CAPTIONS.read_text(encoding="utf-8"))
    if not defined:
        return 0

    print(f"\n{'-' * 78}\nFIGURE CAPTIONS  ({len(defined)} figures)")
    caps = dict(brace_bodies(body, "FigCap"))
    descs = dict(brace_bodies(body, "FigDesc"))
    over, desc = set(caps), set(descs)
    todo = [d for d in defined if d not in over]
    print(f"  IN YOUR WORDS: {len(over)} of {len(defined)}")
    if todo:
        print(f"     still the preprint's: {', '.join(sorted(todo))}")
        print("     -> \\renewcommand{\\FigCap<Name>}{...} in main.tex, after "
              "\\input{generated/captions}")
    print(f"  \\Description WRITTEN: {len(desc)} of {len(defined)}"
          + ("" if len(desc) == len(defined)
             else "   -> ACM requires a real one on every float"))

    bad = read_written_captions(caps, descs)

    # A figure that is built and never included is dead weight in the repo; one
    # included twice is a duplicate float that LaTeX will happily number twice.
    inputs = re.findall(r"\\input\{generated/fig-([a-z0-9-]+)\}", body)
    want = {"-".join(re.findall(r"[A-Z][a-z0-9]*", d)).lower() for d in defined}
    orphan = sorted(want - set(inputs))
    twice = sorted({i for i in inputs if inputs.count(i) > 1})
    if orphan:
        print(f"  BUILT BUT NEVER \\input: {', '.join(orphan)}")
    if twice:
        print(f"  \\input MORE THAN ONCE: {', '.join(twice)}")
    return (1 if todo else 0) + (1 if orphan or twice else 0) + bad


REUSE = 10       # consecutive words shared with the preprint caption


def brace_bodies(tex: str, prefix: str) -> list[tuple[str, str]]:
    """(name, body) for each \\renewcommand{\\<prefix><Name>}{...}.

    Brace-counted rather than regexed. A caption contains \\textbf{...} and
    \\S\\ref{...}, so "\\{([^}]*)\\}" stops at the first inner closing brace and
    silently reads a fraction of the caption -- which would let a typed
    measurement past simply for sitting after the first macro.
    """
    out = []
    for m in re.finditer(r"\\renewcommand\{\\" + prefix + r"(\w+)\}\{", tex):
        i, depth = m.end(), 1
        while i < len(tex) and depth:
            if tex[i] == "{" and tex[i - 1] != "\\":
                depth += 1
            elif tex[i] == "}" and tex[i - 1] != "\\":
                depth -= 1
            i += 1
        out.append((m.group(1), tex[m.end():i - 1]))
    return out


def read_written_captions(caps: dict, descs: dict) -> int:
    """The checks that apply to a caption the author has actually written.

    The four of them exist because a rewritten caption is the easiest place in
    this submission to break a promise the paper makes elsewhere. It states
    more measurements per word than any section; it is written last, against a
    version that already reads as finished; and the temptation is to keep the
    preprint's sentence and change a word, which is not what the venue's policy
    asks for.
    """
    if not caps and not descs:
        return 0
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import build_caption_kit as kit
        allow = set(kit.CONSTANTS)
        budget, dlo, dhi = kit.WORD_BUDGET, *kit.DESC_BUDGET
        body_figs = kit.BODY
        preprint = kit.caption_bodies()
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (caption budgets unavailable: {exc})")
        return 0

    macros = defined_macros()
    problems = 0
    for name in sorted(set(caps) | set(descs)):
        notes = []
        cap = caps.get(name)
        if cap is not None:
            words = len(re.sub(r"\\[A-Za-z]+\*?", " ", cap).split())
            want = budget[name in body_figs]
            if words > want:
                notes.append(f"{words} words against a budget of {want}")
            typed = typed_measurements(cap, allow)
            if typed:
                notes.append("typed measurement: "
                             + ", ".join(sorted(set(typed))[:6])
                             + "  -> use the \\Cap... macro")
            undef = sorted(u for u in set(re.findall(r"\\([A-Za-z]+)", cap))
                           - macros - KNOWN if u[0].isupper())
            if undef:
                notes.append("undefined macro: "
                             + ", ".join("\\" + u for u in undef))
            # The policy check. A rewrite that keeps ten consecutive words of
            # the preprint is not a rewrite, and this is the submission where
            # that matters.
            shared = runs(cap, REUSE) & runs(preprint.get(name, ""), REUSE)
            if shared:
                notes.append(f"{len(shared)} run(s) of {REUSE} words still "
                             f"the preprint's: {sorted(shared)[0]!r}")
        d = descs.get(name)
        if d is not None:
            dw = len(re.sub(r"\\[A-Za-z]+\*?", " ", d).split())
            if dw < dlo:
                notes.append(f"\\Description is {dw} words; ACM wants what a "
                             f"reader who cannot see the figure needs "
                             f"({dlo}-{dhi})")
            elif dw > dhi:
                notes.append(f"\\Description is {dw} words against {dhi}")
            if cap and runs(d, 8) & runs(cap, 8):
                notes.append("\\Description repeats the caption; it should "
                             "say what is ON the page, not what to conclude")
        if notes:
            problems += 1
            print(f"  {name}:")
            for n in notes:
                print(f"     {n}")
    return problems


if __name__ == "__main__":
    sys.exit(main())
