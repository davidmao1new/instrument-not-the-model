"""Cross-document consistency audit: does everything still say the same thing?

A project this size accumulates statements in places that do not get rebuilt
when the data changes: the protocol, the gap register, the handoff notes, an
older paper builder, figure captions baked into PNGs. Numbers inside the paper
are safe by construction, because `build_paper_v3.py` interpolates every one of
them from an artifact and drops any sentence whose artifact is missing. Nothing
protects the prose ELSEWHERE, and nothing protects a count that a human typed
into a caption.

This script hunts for the drift that is left. It is deliberately noisy: it
reports candidates and the reader adjudicates, because a false positive costs a
glance and a false negative costs a wrong number in a preprint.

  A. STALE COUNTS      Statements naming a number of conditions, models, cells,
                       wordings or contrasts, checked against the data.
  B. SUPERSEDED PROSE  Phrases known to encode a claim that has since been
                       retracted: determinism, the p = 0.5 conversion, "one
                       delimiter destroyed", "hierarchical logistic", the arm
                       ordering.
  C. FIGURE NUMBERING  Figures referenced by the builder against figures that
                       exist, and captions whose baked-in number disagrees with
                       the file name.
  D. ARTIFACT FRESHNESS Analyses older than the raw data they summarise.
  E. ORPHANS           Analysis artifacts nothing reads, and figures nothing
                       includes.
  F. MARKUP            Tags and entities the typesetter cannot render and will
                       therefore print as literal source text.

    .venv/Scripts/python.exe paper-a/src/audit_consistency.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "paper-a" / "src"
DATA = ROOT / "paper-a" / "data"
FIGS = ROOT / "paper-a" / "figures"
DOCS = [ROOT / "paper-a" / "PROTOCOL.md", ROOT / "paper-a" / "docs" / "GAPS.md",
        ROOT / "CLAUDE.md", ROOT / "PROGRESS.md"]

ISSUES: list[tuple[str, str]] = []


def issue(kind, msg):
    ISSUES.append((kind, msg))
    print(f"  [{kind}] {msg}")


# --------------------------------------------------------------------------
# Phrases that encode a claim this project has since retracted. Each is paired
# with what superseded it, so the report is actionable rather than accusatory.
# --------------------------------------------------------------------------
SUPERSEDED = [
    (r"measurement is deterministic",
     "retracted: 5/36 byte-identical cells agree; see noise_floor.json"),
    (r"hierarchical logistic",
     "the model is Gaussian crossed random effects on a log-odds contrast"),
    (r"LOGIT_TO_PP\s*=\s*25",
     "superseded by effectsize.py; overstates by 1.8x-108.5x"),
    (r"one delimiter destroyed",
     "D4 and D5 differ 14-fold; the pooled label concealed it"),
    (r"directly comparable",
     "percentage points are not comparable across models at different "
     "operating points"),
    (r"nulls? (?:move|moves) the (?:estimate|effect) more than",
     "arms cannot be distinguished; P(null>semantic) 0.12-0.84, all intervals "
     "include zero"),
    (r"288 matched pairs per condition",
     "36 per condition, 288 in total"),
]

# Files where a superseded phrase is EXPECTED, because they are the record of
# the correction rather than an instance of the error.
EXEMPT = ("CHANGELOG.md", "VERIFICATION.md", "GAPS.md", "audit_consistency.py",
          "effectsize.py", "analyze_mech_panel.py", "analyze_noise_floor.py",
          "fit_arm_contrast.py", "figures_mechpanel.py", "figures_noise.py",
          "build_paper_v3.py")

# Which of the above actually skipped something, filled in as the scan runs.
# AN EXEMPTION THAT SKIPS NOTHING IS A SWITCH LEFT OFF. It removes no finding
# today, so it looks identical to one that is working, and it stays armed for
# whatever text lands in that file next -- exempted without anyone deciding it
# should be. build_paper_v2.py sat here after the file itself was deleted.
EXEMPT_USED: set[str] = set()


# Words that mark a nearby occurrence of a superseded phrase as a RETRACTION of
# it rather than an instance of it.
#
# WHY THIS IS NEEDED. Correcting a stale claim usually means naming the thing
# being corrected -- "the fitted model is NOT a hierarchical logistic model",
# "this clause originally required percentage points so the result would be
# directly comparable, and that is withdrawn". A regex over phrases cannot tell
# the claim from its withdrawal, so without this the audit flags its own
# corrections forever, and a reader can no longer tell a live error from a
# documented one. That failure mode is worse than the false negative it risks:
# the window is one line either side, and a genuine live claim that happens to
# sit beside a retraction word is a narrow case.
RETRACTION_MARKERS = re.compile(
    r"\b(?:retract\w*|withdraw\w*|supersede\w*|superseded|amend\w*|"
    r"no longer|not a|NOT a|originally|previously|stale|corrected|"
    r"does not exist|did not survive|is wrong|overstates)\b", re.I)


def _is_retraction(text: str, pos: int, window: int = 1) -> bool:
    """True if the match sits on, or within `window` lines of, a retraction."""
    lines = text.splitlines()
    ln = text[:pos].count("\n")
    lo, hi = max(0, ln - window), min(len(lines), ln + window + 1)
    return bool(RETRACTION_MARKERS.search("\n".join(lines[lo:hi])))


def check_markup():
    """Markup the typesetter will print literally instead of rendering.

    paperkit understands <b> and <i> and nothing else. An HTML entity or any
    other tag reaches the PDF as its own source text, which is invisible in a
    diff and obvious only to someone looking at the rendered page. One escaped
    entity -- a hair space written as a numeric reference -- shipped into a
    build this way.
    """
    print("\nF. UNRENDERABLE MARKUP")
    hits = 0
    for f in SRC.glob("build_paper*.py"):
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"&#?\w+;|</?(?!/?[bi]>)[a-zA-Z][a-zA-Z0-9]*>", text):
            ln = text[:m.start()].count("\n") + 1
            issue("markup", f"{f.relative_to(ROOT)}:{ln} {m.group(0)!r} "
                            f"will be typeset literally; paperkit renders only "
                            f"<b> and <i>")
            hits += 1
    if not hits:
        print("  none")

def check_superseded():
    print("\nB. SUPERSEDED PROSE")
    targets = [f for f in SRC.glob("*.py")] + [d for d in DOCS if d.exists()]
    hits = 0
    for f in targets:
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        for pat, why in SUPERSEDED:
            for m in re.finditer(pat, text, re.I):
                if _is_retraction(text, m.start()):
                    continue
                # The exemption is recorded at the moment it does something,
                # so an entry that never reaches this line is reported below
                # as one that no longer applies to any text.
                if f.name in EXEMPT:
                    EXEMPT_USED.add(f.name)
                    continue
                ln = text[:m.start()].count("\n") + 1
                issue("stale", f"{f.relative_to(ROOT)}:{ln} {m.group(0)!r} — {why}")
                hits += 1
    if not hits:
        print("  none")

    # RESOLVED AGAINST THE SET ACTUALLY SCANNED, not against guessed
    # directories. Checking SRC/ and ROOT/ reported GAPS.md as absent; it
    # lives at paper-a/docs/GAPS.md and is in DOCS, and was being scanned
    # all along.
    present = {f.name for f in targets}
    missing = [n for n in EXEMPT if n not in present]
    unused = [n for n in EXEMPT if n not in EXEMPT_USED and n not in missing]
    if missing:
        print(f"  exempt but absent: {', '.join(missing)} — the rule points "
              "at a file that is not here")
    if unused:
        print(f"  exempt but skipped nothing: {', '.join(unused)} — the "
              "reason no longer applies to any text in those files, and the "
              "rule stays armed for whatever lands there next")


# --------------------------------------------------------------------------
def check_counts():
    print("\nA. STALE COUNTS")
    from experiment_mechanism import CONDITIONS
    facts = {
        "mechanism conditions": len(CONDITIONS),
        "name-grid pairs": len(st.NAME_GRID),
        "mech-grid pairs": len(st.MECH_GRID),
    }
    for k, v in facts.items():
        print(f"  fact: {k} = {v}")

    # any file claiming "eight ... conditions" when there are now eleven
    n = len(CONDITIONS)
    # THE COUNTS THIS FACT HAS PREVIOUSLY HELD. Append when it changes.
    #
    # This was written `words.get(8) if n != 8 else None`, which hardcodes
    # the superseded value inside a lookup, where it reads like a parameter
    # and behaves like a constant: the only word ever searched for is
    # "eight", and the map's entries for eleven and twelve can never be
    # reached. Move the count from eleven to twelve and prose saying
    # "eleven conditions" goes unlooked-for, silently.
    #
    # Searching for EVERY word that is not the current count was tried and
    # reverted: "conditions" is too generic a noun and it produced four
    # false positives on the first run -- a schema's "Three conditions", a
    # control pair's "two conditions", a contrast definition's "two
    # conditions". Four standing false alarms in a gate that runs on every
    # build is worse than the narrow scan, because it is how a gate stops
    # being read.
    PRIOR_COUNTS = (8,)
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
             11: "eleven", 12: "twelve"}
    stale_words = [words[k] for k in PRIOR_COUNTS if k != n and k in words]
    hits = 0
    if not stale_words:
        print(f"  (no superseded count to look for: {n} is the only value "
              "this fact has held)")
    if stale_words:
        pat = re.compile(
            r"\b(?:" + "|".join(stale_words) + r")"
            r"\s+(?:semantically\s+null\s+)?conditions\b", re.I)
        for f in [f for f in SRC.glob("*.py")] + [d for d in DOCS if d.exists()]:
            if f.name in EXEMPT:
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                continue
            for m in pat.finditer(text):
                ln = text[:m.start()].count("\n") + 1
                issue("count", f"{f.relative_to(ROOT)}:{ln} says {m.group(0)!r}, "
                               f"there are now {n}")
                hits += 1
    if not hits:
        print("  none")


# --------------------------------------------------------------------------
def check_figures():
    print("\nC. FIGURE NUMBERING AND REFERENCES")
    builders = [SRC / "build_paper_v2.py", SRC / "build_paper_v3.py"]
    referenced = set()
    for b in builders:
        if not b.exists():
            continue
        for m in re.finditer(r'FIGS\s*/\s*"([^"]+)"', b.read_text(encoding="utf-8")):
            referenced.add(m.group(1))
    for r in sorted(referenced):
        if not (FIGS / r).exists():
            issue("figure", f"{r} is referenced by a builder and does not exist")
    produced = {p.name for p in FIGS.glob("fig*.png")}
    unused = sorted(produced - referenced)
    for u in unused:
        issue("orphan", f"{u} exists and no builder includes it")
    # caption number vs filename number
    for f in SRC.glob("figures*.py"):
        text = f.read_text(encoding="utf-8")
        cap = re.search(r"fs\.caption\(\s*fig\s*,\s*(\d+)", text)
        out = re.search(r'FIGDIR\s*/\s*"fig(\d+)_', text)
        if cap and out and cap.group(1) != out.group(1):
            issue("figure", f"{f.name}: caption says Figure {cap.group(1)} but "
                            f"writes fig{out.group(1)}_*.png")
    if not any(k in ("figure", "orphan") for k, _ in ISSUES):
        print("  none")


# --------------------------------------------------------------------------
def check_freshness():
    print("\nD. ARTIFACT FRESHNESS")
    pairs = [
        (DATA / "delta_stability" / "study2_v2.json",
         list((DATA / "delta_stability").glob("delta_*.jsonl"))),
        (DATA / "delta_stability" / "noise_floor.json",
         list((DATA / "delta_stability").glob("delta_*.jsonl"))
         + list((DATA / "names").glob("names_*.jsonl"))),
        (DATA / "mechanism_panel" / "mech_panel_analysis.json",
         list((DATA / "mechanism_panel").glob("mech_*.jsonl"))),
        (DATA / "names" / "name_variance.json",
         list((DATA / "names").glob("names_*.jsonl"))),
        (DATA / "quantization" / "quantization_analysis.json",
         list((DATA / "quantization").glob("delta_*.jsonl"))),
        (DATA / "occupation" / "occupation_analysis.json",
         list((DATA / "occupation").glob("occ_*.jsonl"))),
    ]
    stale = 0
    for art, raws in pairs:
        raws = [r for r in raws if r.exists()]
        if not art.exists() or not raws:
            continue
        newest = max(r.stat().st_mtime for r in raws)
        if art.stat().st_mtime < newest:
            issue("stale-artifact",
                  f"{art.relative_to(DATA)} is older than its raw data; re-run "
                  f"its analysis before quoting it")
            stale += 1
    if not stale:
        print("  every analysis artifact is newer than its raw data")


# --------------------------------------------------------------------------
def check_paper_reads_everything():
    print("\nE. ARTIFACTS NOTHING READS")
    b = SRC / "build_paper_v3.py"
    if not b.exists():
        b = SRC / "build_paper_v2.py"
    if not b.exists():
        print("  no builder found")
        return
    text = b.read_text(encoding="utf-8")

    # Quarantine directories are deliberately unreachable. A file in one of
    # them being unread is the intended state, not a finding.
    QUARANTINE = ("_contaminated", "_superseded", "_binary_only_superseded",
                  "_d9_superseded")
    # `reference/raw/` holds PROVENANCE, not analysis artifacts: the unedited
    # readings that a derived artifact was built from -- the per-study
    # literature readings behind reporting_practice_matrix.json, and the audit
    # findings behind this round of corrections. The builder is not supposed to
    # read them; they exist so a reviewer can check what the derived artifact
    # was derived from. Treating them as unread analysis artifacts would train
    # the reader to ignore this section, which is the only way it fails.
    arts = [p for p in DATA.rglob("*.json")
            if not any(q in p.parts for q in QUARANTINE)
            and "SUPERSEDED" not in p.name
            and p.parent.name != "reference"
            and "raw" not in p.parts]

    # The builder reads some artifact families by GLOB rather than by name --
    # one file per checkpoint, where hard-coding six filenames would break the
    # moment a model is added. Resolve those globs so a genuinely-read file is
    # not reported as an orphan; matching on the literal name alone made the
    # audit demand that the builder be written badly.
    globbed = set()
    for m in re.finditer(r'^[A-Z_]+_GLOB\s*=\s*["\']([^"\']+)["\']',
                         text, re.M):
        globbed.update(p.name for p in DATA.rglob(m.group(1)))

    unread = [p for p in arts if p.name not in text and p.name not in globbed]
    for p in sorted(unread):
        issue("unread", f"{p.relative_to(DATA)} is an analysis artifact the "
                        f"paper builder never opens")
    if not unread:
        print("  the builder opens every analysis artifact")


def check_page_overflow():
    """Does any text run off the page edge in the built PDF?

    WHY THIS IS A CHECK AND NOT A CONVENTION. `paperkit` takes explicit column
    widths and does not validate them. A table given single-column widths that
    sum past the column, or a span2 table whose widths sum past the text frame,
    renders with its right-hand columns off the paper and NOTHING WARNS. It
    happened to Table 14 in this build: six columns summing to 334 pt against a
    241.9 pt column, so the last column and half the caption were outside the
    trim. A reader of the PDF sees a table with a column missing; the builder
    exits 0.

    Reading the rendered PDF rather than the source is what makes this reliable:
    it catches wrapped captions and long unbroken tokens too, not just tables.
    """
    print("\nG. PAGE OVERFLOW")
    pdf = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
    if not pdf.exists():
        print("  no built PDF")
        return
    try:
        import fitz  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        print("  PyMuPDF not available; skipped")
        return
    doc = fitz.open(pdf)
    margin = 18.0
    bad = []
    for i, page in enumerate(doc):
        w = page.rect.width
        for b in page.get_text("blocks"):
            if b[2] > w - margin:
                bad.append((i + 1, round(b[2], 1),
                            " ".join(b[4].split())[:56]))
    for pg, x, txt in bad:
        issue("overflow", f"p.{pg} text reaches x={x} (page is "
                          f"{doc[0].rect.width:.0f} wide): {txt}")

    # COLUMN OVERPRINT, WHICH THE TRIM CHECK ABOVE CANNOT SEE. A table given
    # single-column widths that sum past COL_W renders its right-hand columns
    # and its caption ON TOP OF the other column's text. That block is still
    # comfortably inside the page, so the trim check passes and the page is
    # unreadable. It happened to Table 6, and a reader found it, not this file.
    #
    # This is checked from the SOURCE rather than from the rendered geometry.
    # Geometry cannot distinguish a span2 table narrower than the text frame
    # from a one-column table that overran: both start at the left margin and
    # end somewhere in the middle. The declared widths can, exactly.
    #
    # paperkit.table() now raises on the same condition, so a violation cannot
    # reach a PDF at all. This check exists so the condition is also reported
    # rather than only thrown, and so it is covered when the build is not run.
    import ast as _ast  # noqa: PLC0415
    import paperkit as _pk  # noqa: PLC0415
    src = ROOT / "paper-a" / "src" / "build_paper_v3.py"
    over = []
    if src.exists():
        tree = _ast.parse(src.read_text(encoding="utf-8"))
        for node in _ast.walk(tree):
            if not (isinstance(node, _ast.Call)
                    and isinstance(node.func, _ast.Attribute)
                    and node.func.attr == "table"):
                continue
            widths, span2 = None, False
            for a in node.args:
                if (isinstance(a, _ast.List) and a.elts
                        and all(isinstance(e, _ast.Constant)
                                and isinstance(e.value, (int, float))
                                for e in a.elts)):
                    widths = [e.value for e in a.elts]
            for kw in node.keywords:
                if kw.arg == "span2" and isinstance(kw.value, _ast.Constant):
                    span2 = bool(kw.value.value)
            if widths is None:
                note_line = f"line {node.lineno}: table widths not statically readable"
                over.append((node.lineno, None, None, note_line))
                continue
            avail = (_pk.PAGE_W - 2 * _pk.MARGIN_X) if span2 else _pk.COL_W
            if sum(widths) > avail + 0.5:
                over.append((node.lineno, sum(widths), avail,
                             f"span2={span2} widths={widths}"))
    for ln, tot, avail, txt in over:
        if tot is None:
            issue("overprint", txt)
        else:
            issue("overprint", f"build_paper_v3.py:{ln} table widths sum to "
                               f"{tot:.0f} pt but {avail:.0f} pt is available "
                               f"— it would overprint the next column ({txt})")
    if not bad and not over:
        print("  every text block sits inside the trim, and every table "
              "inside its column")


def check_suppressed_prose():
    """Did the build script decide not to print a result, and did anyone notice?

    WHY THIS IS A CHECK. Twice in one round the paper ran an experiment, wrote
    the analysis, built the table, and then printed nothing, because the
    paragraph was wrapped in a guard on a DIFFERENT quantity that happened to be
    absent. §4.6's headline was gated on a dispersion-to-noise-floor ratio that
    is None precisely BECAUSE the floor is zero -- the good case -- and the
    abstract clause for the same study had the same guard.

    A missing paragraph is invisible in the PDF: it looks exactly like a
    paragraph nobody wrote. It is visible from the source, so it is checked
    there. A suppressed block is not automatically a defect; it is a decision
    that has to be made by a person rather than by a falsy dictionary key.
    """
    print(chr(10) + "H. SUPPRESSED PROSE")
    try:
        import audit_suppressed_prose as asp  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        print(f"  could not import audit_suppressed_prose: {e}")
        return
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        asp.main()
    out = buf.getvalue()
    n = 0
    for line in out.splitlines():
        if "suppressed," in line and "guarded blocks printed" in line:
            n = int(line.split("printed,")[1].split("suppressed")[0].strip())
    if n:
        for line in out.splitlines():
            t = line.strip()
            if t and t[0].isdigit() and "is falsy" in line or "is missing" in line:
                issue("suppressed", t[:110])
    else:
        print("  every artifact-guarded prose block fired")


def check_heading_order():
    """Do the section headings appear in the order their numbers claim?

    WHY THIS IS A CHECK. §4.7 was added by inserting its block at the anchor for
    §4.5, so the built document read 4.4, 4.7, 4.5, 4.6. Every number in it was
    correct, every artifact was fresh, every reference resolved, and the paper
    was simply in the wrong order -- a defect that survives every other check in
    this file because none of them looks at sequence. Two further attempts to
    move it landed it inside §5, for the same reason: the separator comments are
    shared between an if-guarded section and the one after it.

    Read from the SOURCE rather than the PDF: headings are emitted by H() in
    source order, and the text layer of a two-column render interleaves columns
    in a way that makes ordering ambiguous.
    """
    print(chr(10) + "I. HEADING ORDER")
    src = ROOT / "paper-a" / "src" / "build_paper_v3.py"
    if not src.exists():
        print("  no build script")
        return
    import re as _re  # noqa: PLC0415
    pat = _re.compile(r'^\s*H\("(\d+(?:\.\d+)?)\s')
    nums = []
    for line in src.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            nums.append(m.group(1))

    def key(n):
        parts = n.split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else -1)

    bad = []
    for a, b in zip(nums, nums[1:]):
        if key(b) < key(a):
            bad.append((a, b))
    for a, b in bad:
        issue("order", f"section {b} is emitted after section {a}")
    if not bad:
        print(f"  {len(nums)} headings, all in ascending order")


def check_escape_leaks():
    """Did a unicode escape reach the page as six literal characters?

    WHY THIS HAPPENS AND WHY NOTHING ELSE CATCHES IT. Prose in this project is
    edited by patch scripts, and a script that writes a backslash-u escape into
    a source file can leave the backslash DOUBLED. The typesetter then receives
    the escape rather than the character and prints it verbatim. It is
    invisible to every numeric check, the build exits 0, and the only symptom
    is a reader seeing an escape sequence in the middle of a sentence -- which
    is where a reviewer found two, in 9.1.

    A stray backslash before a non-ASCII character is the same failure after a
    partial repair, and is checked too: inside a Python string literal a
    backslash before a letter with no escape meaning is preserved verbatim.

    Checked in the source, where it is fixed, and in the rendered PDF, where
    the damage shows.
    """
    print(chr(10) + "J. ESCAPE LEAKS")
    BS = chr(92)
    HEX = set("0123456789abcdefABCDEF")
    n = 0

    def doubled(text):
        """Count backslash runs of length >= 2 followed by u + 4 hex."""
        out, i = 0, 0
        while i < len(text) - 5:
            if text[i] == BS:
                k = 0
                while i + k < len(text) and text[i + k] == BS:
                    k += 1
                j = i + k
                if (j + 4 < len(text) and text[j] == "u"
                        and all(c in HEX for c in text[j + 1:j + 5])
                        and k >= 2):
                    out += 1
                i = j + 1
            else:
                i += 1
        return out

    for f in sorted((ROOT / "paper-a" / "src").glob("*.py")):
        txt = f.read_text(encoding="utf-8")
        d = doubled(txt)
        if d:
            issue("escape", f"{f.name}: {d} doubled unicode escape(s) -- the "
                            f"typesetter would print them literally")
            n += d
        stray = sum(1 for i in range(len(txt) - 1)
                    if txt[i] == BS and ord(txt[i + 1]) > 127)
        if stray:
            issue("escape", f"{f.name}: {stray} stray backslash(es) before a "
                            f"non-ASCII character")
            n += stray

    pdf = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
    if pdf.exists():
        try:
            import fitz  # noqa: PLC0415
            t = " ".join(pg.get_text() for pg in fitz.open(pdf))
            seen = set()
            i = 0
            while i < len(t) - 5:
                if (t[i] == BS and t[i + 1] == "u"
                        and all(c in HEX for c in t[i + 2:i + 6])):
                    seen.add(t[i:i + 6])
                i += 1
            for e in sorted(seen):
                issue("escape", f"the rendered PDF prints the literal {e}")
                n += 1
        except Exception:  # noqa: BLE001
            pass

    if not n:
        print("  no unicode escape reaches the page as literal text")


def check_placeholder_leaks():
    """Did an unevaluated format placeholder reach the page?

    THE MECHANISM, WHICH IS WORTH STATING BECAUSE IT DEFEATS EVERYTHING ELSE.
    Prose in the build script is assembled by concatenating fragments, some of
    which interpolate numbers. Put the `f` prefix on one fragment and the
    braces in the next, and Python is perfectly happy: the fragment holding the
    braces is an ordinary string literal, so it is emitted verbatim. There is
    no exception, no warning, and no failing test -- the build exits 0, the
    consistency audit passes, and 346 tests pass. The only symptom is a reader
    seeing `{pct(min(sat.values()), 1)}` where a percentage should be. Two of
    them shipped that way in v6, v7 and v8, one of them destroying the lower
    endpoint of a range in 6.2.

    Checked in two places, because they fail differently. The SOURCE check is
    scoped to the build script, where a brace in a plain literal has no
    legitimate use -- other modules carry prompt templates and regexes full of
    braces, so the same rule there would be noise. The PDF check is on the
    artifact and so catches the defect whatever produced it, including a route
    nobody has thought of yet. Docstrings are exempt: this one describes the
    bug.
    """
    print(chr(10) + "N. PLACEHOLDER LEAKS")
    import ast  # noqa: PLC0415

    pat = re.compile(r"\{[^{}]{0,200}\}")
    n = 0

    src = ROOT / "paper-a" / "src" / "build_paper_v3.py"
    if src.exists():
        tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
        docs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
                d = ast.get_docstring(node, clean=False)
                if d is not None and node.body:
                    docs.add(id(node.body[0].value))
        for node in ast.walk(tree):
            # A Constant inside a JoinedStr is the literal half of an
            # f-string; its braces are already doubled and cannot leak.
            if isinstance(node, ast.JoinedStr):
                continue
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            if id(node) in docs:
                continue
            for m in pat.finditer(node.value):
                issue("placeholder",
                      f"build_paper_v3.py:{node.lineno}: a plain string "
                      f"literal carries {m.group(0)[:60]} -- the `f` prefix is "
                      f"on a different fragment, so this prints verbatim")
                n += 1

    pdf = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
    if pdf.exists():
        try:
            import fitz  # noqa: PLC0415
            with fitz.open(pdf) as doc:
                for pno, page in enumerate(doc, 1):
                    for m in pat.finditer(page.get_text()):
                        issue("placeholder",
                              f"the rendered PDF prints {m.group(0)[:60]} on "
                              f"p.{pno}")
                        n += 1
        except ImportError:
            pass

    if not n:
        print("  no unevaluated placeholder in the build script or on the page")


def check_figure_captions():
    """The one part of the paper no text check has ever seen.

    WHY IT IS A BLIND SPOT. Figure captions are drawn INTO the matplotlib
    figure and rasterised into the PNG, so they are pixels by the time the PDF
    exists. Every check in this file that reads the rendered text -- escape
    leaks, placeholder leaks, hard-typed table numbers, overlap -- silently
    skips them. That is how a caption came to describe the mechanism panel as
    "48 name pairs and two résumés" when it is 24 pairs and three templates,
    and survived every audit and 377 tests: the sentence was never read by
    anything.

    So the captions are checked at the SOURCE, where they are still strings.
    The same three classes are applied, plus the one that matters most here --
    a number written by hand where an artifact value was available. That last
    one cannot be decided automatically, so what is enforced is narrower and
    still useful: a caption may not hard-type a table or figure number, and any
    caption carrying a bare count is listed for a human to confirm against the
    artifact rather than passed silently.
    """
    print(chr(10) + "P. FIGURE CAPTIONS")
    import ast as _ast  # noqa: PLC0415
    src = ROOT / "paper-a" / "src"
    files = sorted(src.glob("fig*.py"))
    if not files:
        print("  no figure scripts")
        return

    BS = chr(92)
    n = 0
    n_caps = 0
    for f in files:
        try:
            tree = _ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as e:
            issue("caption", f"{f.name}: will not parse ({e})")
            n += 1
            continue
        for node in _ast.walk(tree):
            if not (isinstance(node, _ast.Call)
                    and isinstance(node.func, _ast.Attribute)
                    and node.func.attr == "caption"):
                continue
            n_caps += 1
            for arg in node.args:
                for sub in _ast.walk(arg):
                    if isinstance(sub, _ast.JoinedStr):
                        continue
                    if not (isinstance(sub, _ast.Constant)
                            and isinstance(sub.value, str)):
                        continue
                    v = sub.value
                    for m in re.finditer(r"\{[^{}]{0,120}\}", v):
                        issue("caption", f"{f.name}:{node.lineno}: caption "
                                         f"carries the unevaluated "
                                         f"{m.group(0)[:50]}")
                        n += 1
                    for m in re.finditer(r"\b(Table|Figure)s?\s+\d+", v):
                        issue("caption", f"{f.name}:{node.lineno}: caption "
                                         f"hard-types {m.group(0)!r}; the "
                                         f"number cannot follow the object")
                        n += 1
                    if any(v[i] == BS and ord(v[i + 1]) > 127
                           for i in range(len(v) - 1)):
                        issue("caption", f"{f.name}:{node.lineno}: stray "
                                         f"backslash before a non-ASCII "
                                         f"character")
                        n += 1
    if not n:
        print(f"  {n_caps} captions across {len(files)} scripts, none "
              f"hard-typing a number or leaking a placeholder")


def check_duplicate_references():
    """Does the reference list print one work twice, or one identifier twice?

    THE FAILURE THIS CATCHES. Citing a new work means adding a paragraph AND a
    reference entry, and the second step has no natural place to notice that
    the work is already listed. Seshadri et al. went in twice, adjacently, once
    with arXiv:2501.04316 and once with arXiv:2503.19182 -- the latter being
    Iso et al.'s identifier, already correctly printed against Iso a few lines
    above. So the list asserted that two different papers share an arXiv
    number, and a reader following the wrong one lands on unrelated work.

    Three things are checked on the RENDERED list rather than the source,
    because that is where a reader meets it: an author-and-year that appears
    twice, an arXiv identifier that appears twice, and an identifier attached
    to two different titles. The last is the one that misdirects.
    """
    print(chr(10) + "O. DUPLICATE REFERENCES")
    pdf = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
    if not pdf.exists():
        print("  paper not built")
        return
    try:
        import fitz  # noqa: PLC0415
    except ImportError:
        print("  PyMuPDF unavailable")
        return
    with fitz.open(pdf) as doc:
        txt = " ".join(" ".join(pg.get_text().split()) for pg in doc)
    i = txt.rfind("References")
    body = txt[i:] if i >= 0 else txt

    n = 0
    # An entry starts at "Surname, X." and runs to the next such start.
    starts = [m.start() for m in
              re.finditer(r"[A-Z][A-Za-zÀ-ɏ'’-]+, [A-Z]\.", body)]
    entries = [body[a:b] for a, b in zip(starts, starts[1:] + [len(body)])]

    seen_head = {}
    for e in entries:
        m = re.match(r"([A-Z][A-Za-zÀ-ɏ'’-]+).{0,180}?\((\d{4})\)", e)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        # A surname-year pair can legitimately repeat (two papers, same author
        # and year), so compare the TITLE that follows before complaining.
        title = e[m.end():m.end() + 70].strip(" .")
        if key in seen_head and seen_head[key][:40] == title[:40]:
            issue("reference", f"{key[0]} ({key[1]}) is listed twice with the "
                               f"same title")
            n += 1
        seen_head.setdefault(key, title)

    ids = collections.defaultdict(list)
    for e in entries:
        for aid in re.findall(r"arXiv:(\d{4}\.\d{4,5})", e):
            ids[aid].append(" ".join(e.split())[:90])
    for aid, where in ids.items():
        if len(where) > 1:
            titles = {re.sub(r"^[^)]*\)\s*", "", w)[:45] for w in where}
            if len(titles) > 1:
                issue("reference", f"arXiv:{aid} is attached to "
                                   f"{len(titles)} different works: "
                                   + " | ".join(sorted(titles)))
            else:
                issue("reference", f"arXiv:{aid} appears {len(where)} times")
            n += 1

    if not n:
        print(f"  {len(entries)} entries, no duplicate work or identifier")


def check_table_numbering():
    """Are table numbers unique, in emission order, and all resolvable?

    WHY. Numbers were typed into captions, so adding a table produced two
    Table 10s and moving a section printed Tables 16 and 17 before 11 to 14 --
    both in a shipped build, neither caught by any check here. They are now
    assigned from TABLE_ORDER in build_paper_v3, the way figure numbers are
    assigned from figstyle. This verifies the three properties that makes true:

      * TABLE_ORDER has no duplicate key;
      * every key is emitted by exactly one caption, in the declared order;
      * every TAB(key) in prose names a key that exists.
    """
    print(chr(10) + "K. TABLE NUMBERING")
    import ast as _ast  # noqa: PLC0415
    import re as _re  # noqa: PLC0415
    src = ROOT / "paper-a" / "src" / "build_paper_v3.py"
    if not src.exists():
        print("  no build script")
        return
    text = src.read_text(encoding="utf-8")

    order = []
    for node in _ast.walk(_ast.parse(text)):
        if (isinstance(node, _ast.Assign) and node.targets
                and isinstance(node.targets[0], _ast.Name)
                and node.targets[0].id == "TABLE_ORDER"
                and isinstance(node.value, _ast.List)):
            order = [e.value for e in node.value.elts
                     if isinstance(e, _ast.Constant)]
    if not order:
        issue("tables", "TABLE_ORDER not found; numbers may be typed again")
        return

    # THE SOURCE LITERAL IS A SEED, NOT THE ORDER. paperkit defers span2 floats,
    # so emission order is not render order, and the builder now MEASURES the
    # order the captions reach the page and caches it. Two consumers read the
    # literal instead -- this audit and one test -- and both reported a
    # correctly-numbered PDF as broken, naming the same two tables. Parsing the
    # source is the right way to catch a duplicate or an orphaned key, and the
    # wrong way to learn what number a table carries.
    try:
        import build_paper_v3 as _bp
        if set(_bp.TABLE_ORDER) == set(order):
            order = list(_bp.TABLE_ORDER)
    except Exception:  # noqa: BLE001
        pass

    dups = {k for k in order if order.count(k) > 1}
    for k in sorted(dups):
        issue("tables", f"TABLE_ORDER lists {k!r} more than once")

    # captions, in source order
    emitted = _re.findall(r"caption=\(f\"\{TAB\('([a-z0-9_]+)'\)\}", text)
    missing = [k for k in order if k not in emitted]
    extra = [k for k in emitted if k not in order]
    for k in missing:
        issue("tables", f"{k!r} is declared in TABLE_ORDER but no caption "
                        f"emits it")
    for k in extra:
        issue("tables", f"a caption emits {k!r}, which TABLE_ORDER does not "
                        f"declare")
    # EMISSION ORDER IS NOT THE PROPERTY THAT MATTERS, and asserting it is why
    # two inversions shipped. paperkit defers span2 tables to a later page's
    # float slots while single-column tables flow inline, so a full-width table
    # emitted first can render a page AFTER a narrow one emitted later: Table 5
    # printed on page 7 and Table 4 on page 9, and this check passed. TABLE_ORDER
    # is now the RENDERED sequence, so emission order legitimately differs from
    # it and is reported rather than asserted.
    if emitted and not missing and not extra:
        pos = [order.index(k) for k in emitted]
        if pos != sorted(pos):
            out_of = [emitted[i] for i in range(1, len(pos))
                      if pos[i] < pos[i - 1]]
            print("  note: emitted out of declared order at "
                  + ", ".join(out_of)
                  + " (span2 floats defer; the rendered order is asserted below)")

    caps = _re.findall(
        r"caption=\(f\"\{TAB\('([a-z0-9_]+)'\)\}\.\s*([^\"{]{8,40})", text)
    pdf = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
    try:
        import fitz  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        fitz = None
    if fitz is None or not pdf.exists():
        print("  (render order not checked: no PDF or no PyMuPDF)")
    elif len(caps) < len(order):
        issue("tables", f"only {len(caps)} of {len(order)} captions expose a "
                        f"literal opening; render order cannot be checked")
    else:
        with fitz.open(pdf) as doc:
            pages = [" ".join(p.get_text().split()) for p in doc]
        found, seq = [], True
        for key, head in caps:
            n = order.index(key) + 1
            needle = f"Table {n}. " + " ".join(head.split())
            hit = next(((pno, pages[pno].find(needle))
                        for pno in range(len(pages))
                        if pages[pno].find(needle) >= 0), None)
            if hit is None:
                issue("tables", f"{key!r}: caption not found in the built PDF")
                seq = False
            else:
                found.append((hit, n))
        found.sort()
        nums = [n for _, n in found]
        if seq and nums != sorted(nums):
            issue("tables", "tables RENDER out of numeric order: "
                  + "; ".join(f"{nums[i - 1]} before {nums[i]}"
                              for i in range(1, len(nums))
                              if nums[i] < nums[i - 1]))
        elif seq:
            print(f"  {len(nums)} captions render in numeric order")

    refs = set(_re.findall(r'TAB\("([a-z0-9_]+)"\)', text))
    for k in sorted(refs - set(order)):
        issue("tables", f'prose calls TAB("{k}") but TABLE_ORDER has no such key')

    # A HARD-TYPED NUMBER IS A REFERENCE THAT CANNOT FOLLOW ITS TABLE. One
    # caption said "the scale Tables 3 and 17 use" to argue that three panels
    # share a scale; Table 17 is the mechanism-class table, in log-odds, so the
    # sentence named a panel on the other scale -- in a caption whose entire
    # purpose was scale commensurability. It read correctly when written and
    # stopped being true when §4.7 moved. Automatic numbering exists precisely
    # so this cannot happen, and it only helps where it is used.
    typed = 0
    _tree = _ast.parse(text)
    # Docstrings are exempt: they explain the rule, and TABS()'s own docstring
    # necessarily shows what its output looks like.
    _docs = set()
    for node in _ast.walk(_tree):
        if isinstance(node, (_ast.Module, _ast.FunctionDef,
                             _ast.AsyncFunctionDef, _ast.ClassDef)):
            if _ast.get_docstring(node, clean=False) is not None and node.body:
                _docs.add(id(node.body[0].value))
    for node in _ast.walk(_tree):
        if isinstance(node, _ast.JoinedStr):
            continue
        if (isinstance(node, _ast.Constant) and isinstance(node.value, str)
                and id(node) not in _docs):
            for m in _re.finditer(r"\bTables?\s+\d+", node.value):
                # A REFERENCE TO SOMEBODY ELSE'S TABLE IS NOT A DRIFT RISK --
                # it is the opposite. §8 cites Sclar et al.'s Table 1, and it
                # used to do so through TAB("instrument_validation"), this
                # paper's Table 1, which agreed only by coincidence and would
                # have silently renumbered somebody else's table on the next
                # reordering. External references are therefore hard-typed on
                # purpose, and marked by a possessive so the intent is legible
                # here and to a reader.
                before = node.value[max(0, m.start() - 12):m.start()].lower()
                if before.endswith(("their ", "'s ", "’s ")):
                    continue
                issue("tables", f"line {node.lineno}: prose hard-types "
                                f"{m.group(0)!r}; use TAB(key) so the number "
                                f"follows the table, or mark it as another "
                                f"paper's with a possessive")
                typed += 1

    if not dups and not missing and not extra and not typed:
        print(f"  {len(order)} tables, unique keys, every key emitted once")


def check_block_overlap():
    """Do any two text blocks physically overlap on the page?

    WHY THIS IS THE RIGHT CHECK. Every other geometric test here is a proxy:
    widths that sum inside a column, blocks that stay inside the trim, tables
    whose cells fit. All three passed while a full-width table painted its
    right-hand columns on top of the facing column's text on six pages, because
    a header wider than its column spills and nothing clips it. Overlap is what
    the reader actually sees, and two rectangles intersecting is exact -- no
    heuristic about which block is a float and which is a column.

    The tolerance is 2 pt so that adjacent lines, which share an edge, do not
    register.
    """
    print(chr(10) + "L. BLOCK OVERLAP")
    pdf = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
    if not pdf.exists():
        print("  no built PDF")
        return
    try:
        import fitz  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        print("  PyMuPDF not available; skipped")
        return
    TOL = 2.0
    n = 0
    for pno, page in enumerate(fitz.open(pdf), 1):
        blocks = [b for b in page.get_text("blocks") if " ".join(b[4].split())]
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                a, b = blocks[i], blocks[j]
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox > TOL and oy > TOL:
                    n += 1
                    if n <= 6:
                        issue("overlap", f"p.{pno} two blocks overlap "
                                         f"{ox:.0f}x{oy:.0f} pt: "
                                         f"{' '.join(a[4].split())[:40]!r} / "
                                         f"{' '.join(b[4].split())[:40]!r}")
    if n > 6:
        issue("overlap", f"...and {n - 6} further overlapping pairs")
    if not n:
        print("  no two text blocks overlap on any page")


def check_orphan_headings():
    """Is any section heading the last thing in its column?

    A heading with nothing under it is the most visible typesetting fault a
    paper can have, and three of them shipped: 10.1 and 11 at the foot of a
    column, and a page 1 whose right column was empty entirely. The cause was
    that paragraphs were atomic blocks -- keep_with_next reserved the heading
    plus two lines, those fitted, and then the whole paragraph did not. They
    split now, so a heading can always be followed by the start of its text.

    This checks the rendered page rather than the intent: a heading block with
    no other block below it in the same column is an orphan, whatever the flow
    believed.
    """
    print(chr(10) + "M. ORPHAN HEADINGS")
    pdf = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
    if not pdf.exists():
        print("  no built PDF")
        return
    try:
        import fitz  # noqa: PLC0415
        import paperkit as _pk  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        print("  dependencies unavailable; skipped")
        return
    import re as _re  # noqa: PLC0415
    HEAD = _re.compile(r"^\d+(\.\d+)?\s+[A-Z]")
    mid = _pk.MARGIN_X + _pk.COL_W + _pk.GUTTER / 2
    n = 0
    for pno, page in enumerate(fitz.open(pdf), 1):
        blocks = [b for b in page.get_text("blocks")
                  if " ".join(b[4].split())]
        for side, sel in (("left", lambda b: b[0] < mid),
                          ("right", lambda b: b[0] >= mid)):
            colb = sorted([b for b in blocks if sel(b)], key=lambda b: b[1])
            for k, b in enumerate(colb):
                txt = " ".join(b[4].split())
                if not HEAD.match(txt) or len(txt) > 90:
                    continue
                below = [x for x in colb[k + 1:] if x[1] > b[3] - 1]
                if not below:
                    n += 1
                    issue("orphan", f"p.{pno} {side} column ends with the "
                                    f"heading {txt[:60]!r}")
    if not n:
        print("  every heading is followed by text in its own column")


def main() -> int:
    print("=" * 78)
    print("CROSS-DOCUMENT CONSISTENCY AUDIT")
    print("=" * 78)
    check_counts()
    check_superseded()
    check_markup()
    check_figures()
    check_freshness()
    check_paper_reads_everything()
    check_page_overflow()
    check_suppressed_prose()
    check_heading_order()
    check_escape_leaks()
    check_table_numbering()
    check_block_overlap()
    check_orphan_headings()
    check_placeholder_leaks()
    check_duplicate_references()
    check_figure_captions()
    print("\n" + "=" * 78)
    kinds = {}
    for k, _ in ISSUES:
        kinds[k] = kinds.get(k, 0) + 1
    if ISSUES:
        print(f"{len(ISSUES)} candidate inconsistencies: "
              + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    else:
        print("no inconsistencies found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
