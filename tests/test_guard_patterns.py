r"""Founding cases for the four guards that were silently not guarding.

On 28 August 2026 an audit of the pipeline for the surname-boundary bug class
found the same defect in four more places. Every one of them reported clean
for months, because a dead pattern and a satisfied pattern produce identical
output: zero hits.

(The original defect is not spelled out here. Writing it as a regex literal
put the author's surname into the anonymised archive, where identity in the
supplementary material is a stated desk-rejection ground -- and the
anonymiser's own name rule did not catch it, because the character before the
name was the "b" of the escape sequence. See DENY_SCAN_NORMALISE in
build_iclr_supplementary.py, which now removes escapes before scanning.)

  1. build_iclr_supplementary.py loaded a file of regexes through re.escape,
     so all 14 private-correspondent patterns were literals that could only
     match text containing a backslash. 0 of 14 could ever fire.
  2. build_release_repo.py compiled the same file case-sensitively, so the
     sentence-initial "We thank <name>" -- the form an acknowledgement
     actually takes -- slipped a rule written for mid-sentence prose.
  3. build_release_repo.py wrapped the employer rule in \b boundaries: the
     founding bug with the name changed, blind to every underscore- and
     digit-bearing spelling, on a push to a public repository. (The term
     itself is not written here -- see the note below, and note that the
     release gate refused this very file until this line was reworded, which
     is the repaired rule working.)
  4. check_iclr.py's ALLOW list exempted five values that were also the
     values of live measurement macros, turning the typed-measurement gate
     off for those measurements.

NOTHING SENSITIVE IS SPELLED IN THIS FILE. It ships in the public release,
and the release gate scans tests/**/*.py for exactly the terms these rules
protect -- a test that named them would be refused by the gate it tests. So
every probe string here is CONSTRUCTED FROM THE PATTERN'S OWN PARSE TREE.
That also makes the tests self-maintaining: add a correspondent and the new
pattern is exercised without anyone editing this file.
"""

import pathlib
import re
import re._parser as sre
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "paper-a" / "src"
sys.path.insert(0, str(SRC))

GATE = SRC / "build_release_repo.py"
CHECK = SRC / "check_iclr.py"
PRIVATE = ROOT / "paper-a" / "data" / "reference" / "private_names.local.txt"

needs_gate = pytest.mark.skipif(not GATE.exists(),
                                reason="gate not shipped (public clone)")
needs_private = pytest.mark.skipif(
    not PRIVATE.exists(), reason="the private list never leaves this machine")

# THIS FILE SHIPS; ITS SUBJECTS DO NOT. build_release_repo.py, check_iclr.py
# and build_iclr_supplementary.py are all excluded from the public repo and
# from the reviewer archive by SKIP_FILES, and both READMEs tell the reader to
# run `python -m pytest tests/ -q`. Without these guards four tests import or
# read a module that is not there and ERROR rather than skip -- in the
# artifact of a paper about reproducibility, following its own instructions.
ANON = SRC / "build_iclr_supplementary.py"
needs_anon = pytest.mark.skipif(
    not ANON.exists(),
    reason="the submission anonymiser is not part of the release")
needs_check = pytest.mark.skipif(
    not CHECK.exists(),
    reason="the submission checker is not part of the release")


# ---------------------------------------------------------------------------
# One string that a pattern MUST match, derived from the pattern itself.
# ---------------------------------------------------------------------------
def _from_class(items: object) -> str:
    for op, av in items:
        name = str(op)
        if name == "LITERAL":
            return chr(av)
        if name == "RANGE":
            return chr(av[0])
        if name == "CATEGORY":
            return " " if "SPACE" in str(av) else "x"
    return "x"


def matching_string(pattern: str) -> str:
    """Build a string the pattern matches, by walking its parse tree.

    Guessing at the literal a pattern needs is how a test ends up asserting
    something weaker than it looks: the first attempt at this used the
    longest word in the pattern and reported 4 of 14 passing when the true
    answer was 14 of 14. The parse tree does not guess.
    """
    def walk(seq) -> str:
        parts = []
        for op, av in seq:
            name = str(op)
            if name == "LITERAL":
                parts.append(chr(av))
            elif name in ("NOT_LITERAL", "ANY"):
                parts.append("x")
            elif name == "IN":
                parts.append(_from_class(av))
            elif name == "RANGE":
                parts.append(chr(av[0]))
            elif name == "BRANCH":
                parts.append(walk(av[1][0]))          # first alternative
            elif name == "SUBPATTERN":
                parts.append(walk(av[3]))
            elif name in ("MAX_REPEAT", "MIN_REPEAT"):
                lo, _hi, item = av
                parts.append(walk(item) * max(lo, 1))
            elif name == "ATOMIC_GROUP":
                parts.append(walk(av))
            # anchors and lookarounds contribute no characters
        return "".join(parts)

    return walk(sre.parse(pattern))


def _code(path: pathlib.Path) -> str:
    """A file's executable lines only.

    These assertions look for a defect's code shape, and the fix for each
    defect is commented with the code it replaced -- so a naive substring
    search over the whole file finds the explanation and reports the bug as
    still present. Comments are where the reasoning lives; they are not what
    runs.
    """
    return "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines()
                     if not ln.lstrip().startswith("#"))


def _private_patterns() -> list[str]:
    return [ln.strip()
            for ln in PRIVATE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


# ---------------------------------------------------------------------------
# 1 + 2. the private-correspondent list
# ---------------------------------------------------------------------------
@needs_private
@needs_anon
def test_the_anonymiser_reads_the_private_file_as_regexes_not_literals():
    r"""re.escape on a file of regexes makes every rule unable to fire.

    The file's own header says "One regex per line". Passing each line
    through re.escape turned r"Name\s+Surname" into r"Name\\s\+Surname",
    which matches only text containing a literal backslash.

    This calls the ANONYMISER'S OWN LOADER. The first version of this test
    compiled the patterns itself and passed against the bug, because a test
    that reimplements the code under test is testing its own copy. That is
    why load_private_patterns() is module-level: so the test can reach the
    thing that actually ships.
    """
    import importlib

    import build_iclr_supplementary as anon
    importlib.reload(anon)

    rules = anon.load_private_patterns()
    written = _private_patterns()
    assert len(rules) == len(written), (
        "the anonymiser did not load every private-name rule")

    # The probe comes from the line AS WRITTEN IN THE FILE, never from
    # rx.pattern. Probing rx.pattern is circular: under re.escape the
    # compiled pattern is already escaped, so it matches its own escaped
    # probe and the test passes against the very bug it names. That is
    # exactly how the first version of this test passed on the defect.
    dead = [i for i, (rx, raw) in enumerate(zip(rules, written), 1)
            if not rx.search(matching_string(raw))]
    assert not dead, (
        f"{len(dead)} of {len(rules)} private-name rules, as loaded by the "
        f"anonymiser, cannot match the text their line was written to match "
        f"(lines {dead}). The patterns are being escaped into literals, and "
        "the rule is not running at all.")


@needs_private
def test_private_name_rules_are_case_insensitive():
    """"We thank <name> for the reply." is the form that matters.

    The release gate compiled these case-sensitively. An acknowledgement is
    the single most likely place a correspondent's name appears, and it puts
    the name in a position a rule written for mid-sentence prose can miss.
    """
    patterns = _private_patterns()
    missed = [p for p in patterns
              if not re.compile(p, re.I).search(matching_string(p).upper())]
    assert not missed, (
        f"{len(missed)} of {len(patterns)} rules miss an uppercased name")

    in_sentence = [p for p in patterns
                   if re.compile(p, re.I).search(
                       f"We thank {matching_string(p)} for the reply.")]
    assert len(in_sentence) == len(patterns), (
        "a name inside an acknowledgement sentence is not caught")


@needs_private
@needs_gate
def test_the_release_gate_loads_the_private_rules_case_insensitively():
    """The rules as the shipping gate actually compiles them."""
    import importlib

    import build_release_repo as g
    importlib.reload(g)
    rules = [rx for rx, why in g.DENY_TEXT if "private correspondent" in why]
    assert len(rules) == len(_private_patterns()), (
        "the gate did not load every private-name rule")
    for rx in rules:
        probe = matching_string(rx.pattern)
        assert rx.search(probe), "a loaded rule cannot match its own text"
        assert rx.search(probe.upper()), "a loaded rule is case-sensitive"


@needs_private
def test_published_work_stays_citable():
    """Citing a correspondent's PUBLISHED paper must survive these rules.

    The rules match full names and correspondence forms on purpose, not bare
    surname-plus-year citations. A rule that swallowed citations would push
    the author toward not citing relevant work, which is a worse failure than
    the one being prevented.
    """
    for pat in _private_patterns():
        rx = re.compile(pat, re.I)
        assert not rx.search("as Smith (2024) reports"), (
            "a private-name rule matches an unrelated citation")


@needs_anon
def test_a_name_written_inside_a_regex_literal_is_still_denied():
    r"""An escape sequence ends in a letter, and letter anchors trust letters.

    The anonymiser anchors names on letters rather than word characters,
    which fixed the original defect. It does not fix a name written inside a
    pattern: there the preceding character is the trailing letter of an
    escape, so the lookbehind reads it as mid-word and the rule stays quiet.
    This shipped the surname into the anonymised archive, and an unbounded
    substring sweep found it -- the deny rules did not.

    The name is reconstructed from the module's own byte list. Spelling it
    here would put it back in the archive and recreate the leak.
    """
    import importlib

    import build_iclr_supplementary as anon
    importlib.reload(anon)

    surname = anon.DENY_BYTES[0].decode("ascii").split()[-1]
    assert surname.isalpha(), "could not recover the name from DENY_BYTES"

    bare = f"the {surname} case"
    in_pattern = "the r\"\\b" + surname + "\\b\" case"

    assert anon.DENY.search(bare), "the plain form is not denied at all"
    assert not anon.DENY.search(in_pattern), (
        "the raw text now matches directly, so this test no longer "
        "exercises the escape-sequence blind spot")
    assert anon.DENY.search(anon.scannable(in_pattern)), (
        "a name written inside a regex literal is not denied. Escapes are "
        "not being normalised before the scan, and identity in the "
        "supplementary material is a stated desk-rejection ground.")

    # A correct helper that nothing calls is the same as no helper. The
    # assertions above would all pass with scannable() defined and unwired,
    # so the scan site itself is pinned.
    source = _code(ANON)
    assert "views_of(" in source, (
        "the member loop no longer builds its views from views_of()")
    assert "scannable(text)" in source, (
        "views_of() no longer includes the escape-normalised copy")
    assert "seen = views_of(text, rel.as_posix())" in source, (
        "the text branch no longer scans every view of the member")
    assert "DENY.search(v)" in source and "rx.search(v)" in source, (
        "the deny rules no longer read every view")


# ---------------------------------------------------------------------------
# 3. the employer rule
# ---------------------------------------------------------------------------
@needs_gate
def test_the_employer_rule_is_not_bounded_by_word_characters():
    r"""The founding bug with the name changed.

    r"\bTerm\b" does not match "Term_internal.json" or "TERM_SEAT_COUNT":
    "_" and digits are word characters, so the boundary never fires. The
    convention that confidential material lives in *.local.txt files makes
    underscore-bearing spellings the likely ones, and this rule is the only
    one in the public gate protecting that boundary.

    The term is reconstructed from the gate's own pattern, never typed here.
    """
    import importlib

    import build_release_repo as g
    importlib.reload(g)
    rx = next(rx for rx, why in g.DENY_TEXT if "employer" in why)
    term = matching_string(rx.pattern)
    assert term.isalpha() and len(term) > 3, "could not recover the term"

    for probe in (term,
                  term + "_internal.json",
                  term.upper() + "_SEAT_COUNT = 12",
                  term.lower() + "_notes.local.txt",
                  "the " + term + "2 pilot",
                  "x_" + term,
                  '{"source": "' + term.lower() + '_ats_schema"}'):
        assert rx.search(probe), (
            f"the employer rule misses a {len(probe)}-char spelling; it has "
            "been re-bounded with \\b")


@needs_gate
def test_the_email_rule_reaches_every_tld_not_an_enumerated_sample():
    """An enumerated allowlist standing in for a general shape.

    The rule listed edu|ac.XX|com|org|net|be|il|ca, so every European and
    modern TLD was unreachable -- 44 addresses across 15 domains sitting in
    the private outreach corpus, which is exactly the set the rule exists to
    keep out of a public repository. It also carried no re.I, so an uppercase
    domain escaped a second time.

    Every probe is ASSEMBLED AT RUNTIME rather than written out, so this file
    contains no literal address. Written out, the synthetic probes tripped the
    very rule under test and the gate refused to publish the file -- correctly.
    The alternative, adding this file to ALLOW_EMAIL_IN, would have exempted
    it wholesale and let a real address ship from here later.

    The probes also avoid the reserved example.* domain, which the rule
    excludes on purpose: probing with it would test the exclusion, not the TLD.
    """
    import importlib

    import build_release_repo as g
    importlib.reload(g)
    rx = next(rx for rx, why in g.DENY_TEXT if "email address" in why)

    at = chr(64)
    for local, domain in (("someone", "a-university.de"),
                          ("someone", "a-university.nl"),
                          ("someone", "a-university.se"),
                          ("someone", "a-lab.quebec"),
                          ("someone", "ai.a-school.ch"),
                          ("someone", "a-university.dk"),
                          ("someone", "a-university.no"),
                          ("someone", "a-university.it"),
                          ("someone", "a-university.cz"),
                          ("reader", "lab.a-startup.ai"),
                          ("reader", "start.a-startup.io"),
                          ("Jane.Doe", "A-UNIVERSITY.EDU"),
                          ("someone", "a-lab.gov")):
        probe = local + at + domain
        assert rx.search(probe), (
            f"the email rule cannot see .{domain.split('.')[-1]}; the TLD "
            "list has been re-enumerated")

    assert rx.flags & re.I, "the email rule lost re.I; uppercase domains escape"


@needs_gate
def test_binaries_are_scanned_as_bytes_not_as_mangled_text():
    r"""errors="ignore" invents matches and hides real ones.

    Reading a PNG as UTF-8 with errors ignored turns compressed pixel data
    into pseudo-prose. Broadening the email rule made that visible at once:
    the gate refused the release over 'r-sJ@O4I...oo.hZJ' recovered from a
    figure. The narrow TLD list had been concealing it, so the old rule
    looked well-behaved for the wrong reason.
    """
    import importlib

    import build_release_repo as g
    importlib.reload(g)

    source = GATE.read_text(encoding="utf-8")
    assert 'read_text(encoding="utf-8", errors="ignore")' not in source, (
        "the gate is reading files as text with errors ignored again")
    assert "except UnicodeDecodeError:" in source, (
        "the gate no longer distinguishes binary files from text")

    assert g.DENY_BYTES, "no literal byte rules for binary files"
    for lit, _why in g.DENY_BYTES:
        assert lit == lit.lower(), (
            f"{lit!r} is not lowercase, so the lowercased comparison in the "
            "scan can never match it")
        blob = b"\x89PNG\r\n\x1a\n\x00\xff" + lit.upper() + b"\x00\xfe"
        try:
            blob.decode("utf-8")
            raise AssertionError("probe is not binary")
        except UnicodeDecodeError:
            pass
        assert lit in blob.lower(), "the byte scan would miss this literal"


# ---------------------------------------------------------------------------
# 4. the typed-measurement exemption list
# ---------------------------------------------------------------------------
@needs_check
def test_the_allow_list_never_exempts_one_of_our_own_measurements():
    r"""An exemption that equals a generated value turns the gate off.

    CITED_FIGURES beside it was deliberately given a proximity guard so "the
    exemption cannot drift onto one of our own numbers". ALLOW is a bare
    set-membership test with no such condition, and five of its ten entries
    were the values of live macros: 144 (\NoisePairCells), 22 (\NFields),
    0.25 (\PermFloorHead), and 4 and 8 between seven more. Writing "144
    paired cells" instead of the macro passed the gate that exists to catch
    exactly that.
    """
    import check_iclr

    gen_dir = ROOT / "paper-a" / "iclr" / "generated"
    if not gen_dir.exists():
        pytest.skip("the fork's generated macros are not present")

    values: dict[str, list[str]] = {}
    for f in sorted(gen_dir.glob("*.tex")):
        for m in re.finditer(r"\\newcommand\{?\\([A-Za-z]+)\}?\{([^{}]*)\}",
                             f.read_text(encoding="utf-8")):
            values.setdefault(m.group(2).strip(), []).append(m.group(1))
    assert values, "no macros parsed; the pattern has drifted from the format"

    drift = {v: values[v] for v in check_iclr.ALLOW if v in values}
    assert not drift, (
        "ALLOW exempts values that are also our own measurements: "
        + "; ".join(f"{v} = " + ", ".join(names)
                    for v, names in sorted(drift.items())))


@needs_check
def test_typed_gate_sees_single_digits_and_still_excuses_non_measurements():
    r"""A cheap filter ahead of the real test decides what the test may see.

    TYPED discarded every single-digit numeral, and 33 of the paper's 104
    macros hold single-digit values, so a third of its measurements could be
    typed as digits with the gate reporting "none". Removing the filter
    surfaced eleven numerals, all of them legitimately not measurements:
    enumerators, quantization labels, and a digit inside math. Each now has a
    context guard of its own, in the manner of CITED_FIGURES, rather than a
    bare value in ALLOW that would exempt every macro sharing it.
    """
    src = _code(CHECK)
    assert 'if len(s.replace(",", "").replace(".", "")) < 2:' not in src, (
        "the single-digit length filter is back; a third of the paper's "
        "measurements are invisible to the typed-measurement gate again")
    for guard in ('re.match(r"-bit\\b", after)',
                  'before.endswith("(") and after.startswith(")")'):
        assert guard in src, (
            f"the context guard {guard!r} is gone; the numerals it excuses "
            "will be reported as typed measurements")


def test_hardtyped_audit_has_no_blanket_length_cut():
    """The one string it needed to see was four characters long.

    audit_hardtyped_numbers discarded every string constant under twelve
    characters BEFORE running MEASURE over it. The paper builder held exactly
    one hardtyped measurement, an artifact fallback printing a stale value,
    and it sat in its own four-character concatenation fragment -- so the
    single defect the audit exists to find was the single defect its cheap
    pre-filter hid. MEASURE matches the string perfectly well.
    """
    audit = SRC / "audit_hardtyped_numbers.py"
    if not audit.exists():
        pytest.skip("audit not shipped")
    assert "if len(text) < 12:" not in _code(audit), (
        "the blanket length cut is back, and it hides short measurements")

    import audit_hardtyped_numbers as ah
    measure = getattr(ah, "MEASURE", None)
    if measure is not None:
        assert measure.search("83 %"), (
            "MEASURE no longer matches a bare percentage")

    builder = SRC / "build_paper_v3.py"
    if builder.exists():
        # THE SHAPES, NOT ONE SPELLING OF THEM. This matched `else "<digits>"`
        # and nothing else, so it never saw `_n_panel = len(...) or 6` -- a
        # bare int after `or`, feeding \NPanel into the abstract of both
        # papers, and printing exactly the value the artifact holds today.
        # The AST names both shapes and can tell an artifact fallback from
        # `1 if fverd else 0`, which counts an optional arm rather than
        # standing in for a missing one: there, both branches are constants.
        import ast as _ast
        src_text = builder.read_text(encoding="utf-8")
        tree = _ast.parse(src_text)

        def _numeric(node):
            return (isinstance(node, _ast.Constant)
                    and not isinstance(node.value, bool)
                    and (isinstance(node.value, (int, float))
                         or (isinstance(node.value, str)
                             and node.value[:1].isdigit())))

        bad = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.BoolOp) and isinstance(node.op, _ast.Or) \
                    and _numeric(node.values[-1]):
                bad.append((node.lineno,
                            _ast.get_source_segment(src_text, node)))
            if isinstance(node, _ast.IfExp) and _numeric(node.orelse) \
                    and not _numeric(node.body):
                bad.append((node.lineno,
                            _ast.get_source_segment(src_text, node)))
        assert not bad, (
            "the builder has numeric artifact fallbacks again:\n  "
            + "\n  ".join(f"line {n}: {s}" for n, s in bad)
            + "\nA missing artifact must stop the build, not print a stale "
              "number into a paper claiming every number is interpolated.")


def test_the_facct_typed_gate_sees_single_digits_too():
    r"""The repair landed in one of the two gates that had the defect.

    check_iclr.py's TYPED gate dropped its single-digit cut and gained
    context guards. check_draft.py is the FAccT counterpart -- its docstring
    says it exists because audit_hardtyped_numbers.py cannot see inside a
    .tex file -- and kept the cut. 15 of the 55 generated FAccT macros hold a
    single-digit value, five of them Cap* macros, and this routine is also
    what checks the captions, "the densest numbers in the submission".

    Removing a cheap pre-filter is only half the repair; what it was standing
    in for has to be handled by context. Here that is one category above all
    -- a numeral inside a checkpoint name -- and it accounted for 66 of the
    matches the cut was hiding.
    """
    draft = SRC / "check_draft.py"
    if not draft.exists():
        pytest.skip("check_draft not shipped")
    assert "< 2:" not in _code(draft), (
        "the single-digit cut is back in the FAccT typed-measurement gate")

    import check_draft as CD
    allow = {"0.25", "0.5", "1.0", "95"}

    for prose in ("the effect moves by 6 points",
                  "agreement falls to 4 of 12 cells",
                  "the panel covers 6 checkpoints",
                  "47.9 % of contrasts widen"):
        assert CD.typed_measurements(prose, allow), (
            f"a typed measurement is invisible: {prose!r}")

    for quiet in ("measured on Llama-2-7B-chat",
                  "on Mistral-7B-Instruct v0.1 and v0.3",
                  "Llama-3.1-8B-Instruct saturates",
                  "see preprint Figure 4 for the panel",
                  "Table 7 lists the wordings",
                  "as Section 6 shows",
                  "the interval level is 95 %",
                  "% a comment saying pick 2-3 topics"):
        assert not CD.typed_measurements(quiet, allow), (
            f"a false positive would train the author to ignore this gate: "
            f"{quiet!r} -> {CD.typed_measurements(quiet, allow)}")


def test_negative_search_harvester_reads_every_documented_form():
    """Two of the forms its own docstring promises were never matched.

    negatives() required the zero to be immediately preceded by
    return/returns/returned/= and immediately followed by "for" or ":".
    "Negative greps returning 0 on the extracted text: grep -ic 'a', 'b'"
    fails both ways, so whole studies' negative codings were dropped without
    appearing in the unparsed disclosure.
    """
    audit = SRC / "audit_matrix_evidence.py"
    if not audit.exists():
        pytest.skip("audit not shipped")
    import audit_matrix_evidence as ame

    for ev, want in [
            ("returned 0 for 'cache', 'batching', 'prefill'", 3),
            ("0 hits for 'cache' and 'batching'", 2),
            ("Negative greps returning 0 on the extracted text: grep -ic "
             "'placebo', 'null edit', 'seed'", 3),
            ("Searches run over the full text, all returning 0 hits: grep -i "
             "'whitespace', 'padding'", 2)]:
        got = ame.negatives(ev)
        assert len(got) == want, (
            f"harvested {sorted(got)} from {ev[:52]!r}, expected {want} terms")

    # The over-harvest this rule was tightened against, still controlled: a
    # loose term class once captured "(0), " as a search term and buried the
    # findings under 875 false hits.
    control = ("grep -i for 'snapshot' (0 hits), 'checkpoint' (0), "
               "'revision' (0)")
    assert ame.negatives(control) == {"snapshot", "checkpoint", "revision"}, (
        "the term class has loosened; the harvester is capturing punctuation")


def test_quotation_check_has_no_upper_length_bound():
    """A ceiling on a verification rule exempts its biggest instance.

    QUOTED stopped at 400 characters, so the longest quotations of other
    researchers' papers -- the passages a reader leans on hardest -- were the
    ones never checked against the source they are attributed to.
    """
    audit = SRC / "audit_matrix_evidence.py"
    if not audit.exists():
        pytest.skip("audit not shipped")
    import audit_matrix_evidence as ame

    assert not re.search(r"\{\d+,\d+\}", ame.QUOTED.pattern), (
        f"QUOTED has an upper bound again: {ame.QUOTED.pattern}")
    long_quote = "x" * 900
    assert ame.QUOTED.findall(f'says "{long_quote}" in section 3'), (
        "a 900-character quotation is not picked up for checking")


# ---------------------------------------------------------------------------
# 5. the supplementary gate's haystack
# ---------------------------------------------------------------------------
@needs_check
def test_the_supplementary_gate_reads_the_paper_not_the_runbook():
    """A shared buffer name let one gate repoint another eighty lines away.

    check_iclr bound `raw = MAIN.read_text(...)`, and a later block bound the
    same name to SUBMISSION.md. The SUPPLEMENTARY gate below then tested the
    runbook for a promise the PAPER makes. Both passed only because the two
    documents matched different alternations of the same pattern, so one
    reword of a prose runbook would have retired the archive existence and
    freshness check in silence.

    This reads the source rather than the behaviour, because the behaviour
    was indistinguishable from correct on the day the bug was found.
    """
    import ast

    tree = ast.parse(CHECK.read_text(encoding="utf-8"))

    haystack = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "promises"
                        for t in node.targets)):
            call = node.value
            while isinstance(call, ast.Compare):
                call = call.left
            assert isinstance(call, ast.Call), "promises is not a re.search"
            haystack = call.args[-1].id
    assert haystack, "no assignment to `promises` found in check_iclr.py"

    sources = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == haystack
                        for t in node.targets)):
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Name):
                    sources.add(sub.id)

    assert "MAIN" in sources, (
        f"the supplementary gate tests `{haystack}`, which is never read "
        "from MAIN. The promise being checked is the paper's promise to "
        "reviewers, not a sentence in the runbook.")
    assert "sub" not in sources, (
        f"`{haystack}` is also bound from SUBMISSION.md, so which file the "
        "gate tests depends on statement order")


# ---------------------------------------------------------------------------
# 6. what `git push` actually publishes
# ---------------------------------------------------------------------------
@needs_gate
def test_the_gate_scans_git_history_not_only_the_working_tree(tmp_path,
                                                              monkeypatch):
    """A push sends history. The gate used to scan only the working tree.

    staged_files() strips any path containing ".git" -- a decision taken so
    git internals would not inflate the reported file count, which silently
    exempted them from the deny scan too. On 28 Aug 2026 that gap was found
    holding a private correspondent's name, in two commits already pushed to
    a public remote, while the working tree was clean and the gate said so.

    This builds the exact state that shipped: something committed, then
    deleted, so the tree is clean and the history is not.
    """
    import importlib
    import subprocess

    import build_release_repo as g
    importlib.reload(g)

    if not shutil.which("git"):
        pytest.skip("git is not on PATH")

    repo = tmp_path / "clone"
    repo.mkdir()

    def run(*args):
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, check=True)

    run("init", "-b", "main")
    run("config", "user.name", "Test")
    run("config", "user.email", "test" + chr(64) + "invalid.test")

    # A denied string, assembled so it is not a literal in this file.
    secret = "correspondence with someone" + chr(64) + "a-university.de"
    leaked = repo / "notes.md"
    leaked.write_text(secret + "\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-m", "add notes")

    leaked.unlink()
    run("add", "-A")
    run("commit", "-m", "remove notes")

    # The state that shipped: nothing on disk, everything in history.
    on_disk = [p for p in repo.rglob("*")
               if p.is_file() and ".git" not in p.parts]
    assert not any(secret in p.read_text(encoding="utf-8", errors="ignore")
                   for p in on_disk), "the probe is not testing history"

    monkeypatch.setattr(g, "STAGE", repo)
    found = g.history_problems()
    assert found, (
        "the gate reports a repository clean whose history carries a denied "
        "string. A push publishes history, not the working tree.")
    assert any("history" in f or "publish" in f for f in found), (
        f"the refusal does not say where the hit is: {found}")


@needs_gate
def test_the_history_scan_is_actually_wired_into_the_gate():
    """A correct check that nothing calls is the same as no check."""
    source = _code(GATE)
    assert "def history_problems" in source, "the history scan is gone"
    assert "hist = history_problems()" in source, (
        "history_problems() is defined but main() never calls it")
    assert "problems.extend(hist)" in source, (
        "the history findings are computed and then discarded")


@needs_gate
def test_the_history_scan_honours_the_cited_author_allowlist():
    """Strict is not the same as usable.

    The survey artifacts quote cited papers' own author blocks, which
    ALLOW_EMAIL_IN permits because republishing what a paper prints about
    itself is not disclosure. The first cut of the history scan had no path
    context and refused over exactly those files -- a gate that refuses over
    permitted content is a gate that gets switched off.
    """
    source = _code(GATE)
    assert "at_path" in source and "ALLOW_EMAIL_IN" in source, (
        "the history scan no longer maps blobs to their committed path, so "
        "the cited-author allowlist cannot apply in history")


# ---------------------------------------------------------------------------
# 7. what the anonymiser's rules are actually shown
# ---------------------------------------------------------------------------
@needs_anon
def test_the_member_path_is_scanned_not_only_its_contents():
    """The founding case was a FILENAME, and no rule ever saw the filename.

    Every repair after Mao_methods_supplement.pdf was about matching content.
    The builder still wrote `code/{rel}` into the archive without running one
    identity rule over `rel`. This tree actively generates surname-bearing
    artifact names, kept out today only by SKIP_DIRS entries added for
    unrelated curation reasons.

    The name is rebuilt from the module's own byte list rather than typed.
    """
    import importlib

    import build_iclr_supplementary as anon
    importlib.reload(anon)

    surname = anon.DENY_BYTES[0].decode("ascii").split()[-1]
    for path in (f"paper-a/data/reference/{surname}_methods_supplement.pdf",
                 f"paper-a/src/{surname}_notes.py",
                 f"figures/{surname.lower()}-summary.png"):
        assert any(v and anon.DENY.search(v)
                   for v in anon.views_of("", path)), (
            f"the member path {path} ships unscanned")

    # and an ordinary path is still fine
    assert not any(v and anon.DENY.search(v)
                   for v in anon.views_of("", "paper-a/src/build_paper_v3.py"))


@needs_anon
@needs_private
def test_a_name_joined_by_an_underscore_is_still_a_name():
    r"""The rules join parts with \s+; a filename joins with _.

    A name joined by an underscore or a hyphen missed every rule, in a
    whose own release naming convention is underscore-separated and whose
    founding case was exactly such a filename.

    Probes are generated from each pattern's parse tree, so no correspondent
    is named in this file.
    """
    import importlib

    import build_iclr_supplementary as anon
    importlib.reload(anon)

    pats = anon.load_private_patterns()
    assert pats, "no private patterns loaded"

    missed = []
    for rx in pats:
        plain = matching_string(rx.pattern)
        if " " not in plain:
            continue
        # only spaces BETWEEN LETTERS, which is what SEPARATOR_ALIAS
        # normalises. Substituting every space invents spellings nobody
        # writes, like "Hypothesis_:_Name", and tests the probe not the code.
        for sep in ("_", "-"):
            probe = re.sub(r"(?<=[A-Za-z]) (?=[A-Za-z])", sep, plain)
            if not any(rx.search(v) for v in anon.views_of(probe, "")):
                missed.append(probe.replace(plain.split()[0], "<name>"))
    assert not missed, (
        f"{len(missed)} underscore/hyphen spellings escape every view; the "
        "separator normalisation has been removed")


@needs_anon
def test_binaries_get_literal_rules_and_never_a_regex():
    """Two ways to get a binary wrong, and this file has had both.

    Too weak: DENY_BYTES was case-sensitive, omitted the school, and the
    private-correspondent patterns never touched a binary at all.

    Too clever: running those regexes over raw.decode(errors="ignore")
    invents matches out of compressed pixel data -- it refused three figures
    the moment it was tried. Content gets literal byte comparisons; the
    path, which is always real text, gets the full rule set.
    """
    import importlib

    import build_iclr_supplementary as anon
    importlib.reload(anon)

    names = [p.decode("ascii", "ignore").lower() for p in anon.DENY_BYTES]
    assert any("wissahickon" in n for n in names), (
        "the school is denied for text but not for binaries again")

    lits = anon.literal_bytes(anon.load_private_patterns())
    assert lits, "no correspondent literals derived for binary scanning"

    png = b"\x89PNG\r\n\x1a\n\x00\xff"
    for probe in (anon.DENY_BYTES[0].upper(),
                  anon.DENY_BYTES[0].replace(b" ", b"_"),
                  lits[0].upper()):
        blob = png + probe + b"\x00\xfe"
        assert any(p.lower() in blob.lower()
                   for p in anon.DENY_BYTES + lits), (
            "a binary member carrying a denied term is not caught")

    source = _code(ANON)
    assert 'decode("utf-8", errors="ignore")' not in source, (
        "the anonymiser is regex-scanning salvage-decoded binaries again; "
        "that invents matches out of compressed data")


@needs_gate
@needs_private
def test_the_public_gate_sees_the_same_views_the_anonymiser_does():
    r"""Two gates, one boundary, and only one of them could see it.

    A staged test file carrying an underscored correspondent name passed
    build_release_repo.py -- which printed "clean: ... no personal contact
    details" -- and was refused by build_iclr_supplementary.py one command
    later. The release gate joined name parts with \s+ only, so
    "Name_Surname" was invisible to the gate that publishes to the internet
    while being caught by the one that packages for reviewers.

    Probes come from each rule's own parse tree; no name is typed here.
    """
    import importlib

    import build_release_repo as g
    importlib.reload(g)

    rules = [rx for rx, why in g.DENY_TEXT if "private correspondent" in why]
    assert rules, "the private-correspondent rules did not load"

    missed = []
    for rx in rules:
        plain = matching_string(rx.pattern)
        if " " not in plain:
            continue
        for sep in ("_", "-"):
            probe = re.sub(r"(?<=[A-Za-z]) (?=[A-Za-z])", sep, plain)
            if not any(rx.search(v) for v in g.views_of(probe)):
                missed.append(sep)
    assert not missed, (
        f"{len(missed)} separator spellings escape the public gate; it has "
        "lost the views the anonymiser has")

    source = _code(GATE)
    assert "seen = views_of(text, rel)" in source, (
        "the gate scan no longer reads every view of the file")


# ---------------------------------------------------------------------------
# 8. what the gates are shown: line breaks, file lists, and folded literals
# ---------------------------------------------------------------------------
@needs_check
def test_a_name_broken_across_a_line_is_still_a_name():
    """A PDF line break is lossy in three ways and each needs its inverse.

    TeX INSERTS a hyphen when it breaks inside a word, KEEPS the hyphen when
    it breaks at one the word already had, and leaves only the break when it
    splits at a space. Dropping the hyphen repairs the first case and
    destroys the second, so no single normalisation works -- the scan has to
    try every reconstruction.

    Measured on the built PDF: of 39 hyphen breaks, four of the six longest
    are compounds broken at their own hyphen (between-group, within-group,
    maximum-weight, single-wording). The one LEAKS pattern built from
    hyphenated parts is the repository URL, which has three places to break.
    """
    import importlib

    import check_iclr as C
    importlib.reload(C)

    # THE MODULE'S OWN SCAN, not a copy of it. Written with the views
    # inlined here, this test passed whatever check_iclr.py did -- the same
    # way two of the first founding-case tests passed on the bugs they were
    # written for.
    def caught(t):
        return bool(C.pdf_leak_hits(t))

    # The school, hyphenated by the typesetter at three different places.
    school = next(rx.pattern for rx, why in C.LEAKS if "school" in why)
    for cut in (4, 8, 10):
        assert caught(f"{school[:cut]}-\n{school[cut:]} High School"), (
            f"the school broken after {cut} letters is not caught")

    # The repository URL, broken at each hyphen it already contains.
    repo = next(rx.pattern for rx, why in C.LEAKS if "repo" in why)
    for i, ch in enumerate(repo):
        if ch == "-":
            probe = f"{repo[:i + 1]}\n{repo[i + 1:]}"
            assert caught(probe), (
                f"the repository URL broken at hyphen {i} is not caught; "
                "a break at an EXISTING hyphen keeps it, so de-hyphenating "
                "cannot rebuild the word")

    # Ordinary hyphenation must stay quiet, including the paper's own
    # compounds, which is what stops the fourth view being a blanket join.
    for quiet in ("randomiza-\ntion", "between-\ngroup dispersion",
                  "maximum-\nweight matching", "a dis-\npersion budget"):
        assert not caught(quiet), f"false positive on {quiet!r}"


@needs_check
def test_the_freshness_gate_asks_what_the_archive_holds():
    """The file list is derived from the selection, never guessed at.

    Three copies of "three directories, two suffixes" once stood in for the
    archive's contents. Measured against the archive they guarded, that guess
    watched 55 files it never ships -- 43 under _superseded/, excluded on
    purpose -- and missed 50 it does, including all 19 figures, every bundled
    font, config.yaml and _py.sh. A gate wrong in both directions blocks on
    edits reviewers never see and waves through edits to the figures.
    """
    import importlib

    import build_iclr_supplementary as anon
    import build_release_repo as rel
    importlib.reload(rel)
    importlib.reload(anon)

    packaged = {p.relative_to(anon.REPO).as_posix()
                for p in anon.packaged_sources()}
    assert packaged, "the archive selection is empty"

    # Every figure and every font is watched. These are exactly what the
    # directory-and-suffix guess could not see.
    for suffix in (".png", ".pdf", ".otf", ".ttf", ".sh", ".yaml"):
        assert any(p.endswith(suffix) for p in packaged), (
            f"no {suffix} file is watched; the freshness scan has gone back "
            "to filtering by directory and suffix")

    # And nothing deliberately excluded is watched.
    assert not [p for p in packaged if "_superseded" in p], (
        "superseded files are watched again, so touching one demands a "
        "rebuild that changes nothing")

    # The archive on disk holds exactly what the selection names.
    zp = ROOT / "paper-a" / "iclr" / "supplementary" / "supplementary_code.zip"
    if zp.exists():
        import zipfile
        with zipfile.ZipFile(zp) as z:
            shipped = {n[len("code/"):] for n in z.namelist()
                       if n.startswith("code/") and not n.endswith("/")}
        shipped.discard("ANONYMIZED_FOR_REVIEW.md")
        assert shipped == packaged, (
            f"the archive and the selection disagree on "
            f"{len(shipped ^ packaged)} file(s)")


@needs_check
def test_freshness_is_decided_by_content_not_by_a_timestamp():
    """mtime is the pre-filter; the bytes are the answer.

    The FaccT tests rewrite capnumbers.tex with byte-identical content on
    every suite run, so a timestamp-only gate demanded a rebuild after every
    `pytest`. A gate that cries wolf on every test run is one people learn to
    skip, which is the same erosion the directory guess caused by the
    opposite route.

    mtime survives as a filter because it is sound in the direction that
    matters: a file whose mtime has not moved cannot have changed.
    """
    import importlib

    import build_release_repo as rel
    importlib.reload(rel)

    src = _code(pathlib.Path(rel.__file__))
    assert "def staged_copy_differs" in src, (
        "the content comparison is gone; freshness is back to trusting "
        "timestamps")

    # MOVING AN mtime WITHOUT CHANGING A BYTE MUST NOT READ AS STALE. The
    # first version of this took the first selected file with a staged
    # counterpart and asserted the two did not differ -- without checking
    # that their bytes were identical, which is what its failure message
    # claimed. It therefore passed whenever the stage happened to be current
    # and failed the moment a source was legitimately edited, blaming the
    # comparison for a real difference.
    import os

    sel = rel.selected_sources()
    assert sel, "nothing selected"
    probe = next((p for p in sel
                  if (rel.STAGE / p.relative_to(rel.ROOT)).is_file()
                  and (rel.STAGE / p.relative_to(rel.ROOT)).read_bytes()
                  == p.read_bytes()), None)
    if probe is None:
        pytest.skip("no staged file is currently byte-identical to its "
                    "source, so there is nothing to compare")
    staged = rel.STAGE / probe.relative_to(rel.ROOT)
    was = (probe.stat().st_atime, probe.stat().st_mtime)
    try:
        os.utime(probe, (was[0], was[1] + 120))
        assert not rel.staged_copy_differs(probe, staged), (
            f"{probe.name} reads as changed after its mtime moved, though "
            "not one byte of it did; the suite rewrites generated files "
            "with identical content on every run, so this demands a rebuild "
            "after every pytest and the gate becomes noise")
    finally:
        os.utime(probe, was)


@needs_check
def test_the_gate_and_the_test_share_one_freshness_definition():
    """Three copies of a list is how a list stays wrong.

    The same guess lived in check_iclr.py, in build_iclr_supplementary.py's
    pre-flight and in tests/test_iclr_fork.py. Fixing one left the other two
    to re-report the old answer, which is what happened: the gate was
    repaired and the suite went on failing from its own copy.
    """
    import importlib

    import check_iclr as C
    importlib.reload(C)
    assert hasattr(C, "archive_staleness"), (
        "the single freshness definition is gone")

    for name in ("build_iclr_supplementary.py", "check_iclr.py"):
        body = _code(pathlib.Path(C.__file__).parent / name)
        assert '("paper-a/src", "paper-a/data", "tests")' not in body, (
            f"{name} has grown its own copy of the directory guess again")
    fork = ROOT / "tests" / "test_iclr_fork.py"
    if fork.exists():
        assert '("paper-a/src", "paper-a/data", "tests")' not in _code(fork), (
            "the fork test has grown its own copy of the directory guess")


def test_a_literal_glued_to_an_fstring_is_still_typed():
    """Python folds implicit concatenation before the AST exists.

        P(f"the gap is {x:.1f} points"
          " and the legacy share is 83 %")

    is ONE JoinedStr. The second fragment interpolates nothing -- it is a
    plain literal -- but it lives in JoinedStr.values, so a rule that skipped
    everything inside a JoinedStr skipped 815 typed fragments in
    build_paper_v3.py: 63,695 of 165,207 characters, 38.6% of the printed
    prose. Twelve numerals were hiding there, eight of them artifact values.

    The format spec is the one thing inside an f-string that is not prose:
    `{x:.3f}` stores ".3f", where the 3 is a precision.
    """
    import ast as _ast
    import importlib

    import audit_hardtyped_numbers as H
    importlib.reload(H)

    # THE MODULE'S OWN WALK, not a copy of it. Spelling the traversal out
    # here made the test agree with itself no matter what the audit did.
    tree = _ast.parse(
        'P(f"the gap is {x:.1f} points"\n'
        '  " and the legacy share is 83 %")\n')
    call = tree.body[0].value
    found = [n.value for n in H.typed_literals(call)]

    assert any("83 %" in f for f in found), (
        "a plain literal implicitly concatenated onto an f-string is still "
        "invisible to the audit")
    assert not any(".1f" == f for f in found), (
        "the format spec is being read as prose; its digits are precisions")


def test_the_paper_types_no_measurement_anywhere_in_printed_prose():
    """The claim SS1.2 makes about itself, enforced end to end.

    With the f-string repair in place this found twelve numerals. Eight were
    artifact values -- the cross-day counts (212 of 504, none of 504 on three
    of four), the superseded literature median (0.51, 12 of the 14, the other
    two), the widening buckets and the saturation cutoff -- and are now read
    from the artifacts that hold them. The four that remain are not
    measurements and each carries a bound exemption.
    """
    import subprocess
    p = subprocess.run(
        [sys.executable, str(ROOT / "paper-a" / "src" /
                             "audit_hardtyped_numbers.py")],
        capture_output=True, text=True, cwd=str(ROOT))
    assert "0 numeral(s)" in p.stdout, (
        "a measurement is typed into the paper's prose again:\n" + p.stdout)


# ---------------------------------------------------------------------------
# 9. what leaves the machine: venues, and directories nobody scanned
# ---------------------------------------------------------------------------
@needs_gate
def test_venue_tooling_is_excluded_by_category_not_by_roll_call():
    """The comment named a category; the code named one member of it.

    SKIP_NAME_PATTERNS read ("iclr",) under a comment saying "anything whose
    NAME carries the venue is conference-submission tooling and never ships,
    whether or not anyone remembered to list it below". This repository
    builds submissions for two anonymous venues, and 33 files of the second
    reached the author's own named public repo -- five builders, four tests,
    and the generated tree holding that paper's own captions and tables.

    No venue is spelled here: the tokens come from the gate's own tuple, and
    the probe names are built from them.
    """
    import importlib

    import build_release_repo as rel
    importlib.reload(rel)

    venues = tuple(rel.ANONYMOUS_VENUES)
    assert len(venues) >= 2, (
        "the venue list is back to naming a single venue while the "
        "repository builds submissions for more than one")
    assert tuple(rel.SKIP_NAME_PATTERNS) == venues, (
        "the name rule and the venue list have come apart")

    for v in venues:
        for probe in (f"build_{v}_tex.py", f"check_{v}.py",
                      f"test_{v}_structure.py", f"BUILD_{v.upper()}.PY"):
            assert any(p in probe.lower() for p in rel.SKIP_NAME_PATTERNS), (
                f"{probe} would ship")

    # And the selection agrees: nothing carrying a venue token is staged.
    staged = [p.name for p in rel.selected_sources()
              if any(v in p.name.lower() for v in venues)]
    assert not staged, f"{len(staged)} venue-named file(s) still staged: " \
                       f"{sorted(staged)[:4]}"


@needs_anon
@needs_gate
def test_the_two_gates_agree_on_which_directories_are_transient():
    """One gate refused to scan .hypothesis; the other packaged it.

    build_release_repo.staged_files() has always ignored .hypothesis, so
    nothing in it is ever read by a deny rule. build_iclr_supplementary's
    SKIP_DIRS did not list it, so it would have packaged it. Running the
    suite inside the clone -- which the README invites -- leaves the
    directory behind, and 27 of its files were headed for the reviewer
    archive: bytes no rule had read, in the artifact where identity is a
    stated desk-rejection ground.
    """
    import importlib

    import build_iclr_supplementary as anon
    import build_release_repo as rel
    importlib.reload(rel)
    importlib.reload(anon)

    unscanned = {".git", "__pycache__", ".hypothesis", ".pytest_cache"}
    missing = sorted(unscanned - set(anon.SKIP_DIRS))
    assert not missing, (
        f"the archive would package {missing}, which the release gate "
        "never scans; unscanned bytes must not reach the reviewer zip")

    # Demonstrated rather than asserted from the constant: a file planted in
    # one of those directories is not among the packaged members.
    probe = anon.REPO / ".hypothesis" / "examples" / "probe.bin"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_bytes(b"transient")
    try:
        packaged = {p.as_posix() for p in anon.packaged_sources()}
        assert probe.as_posix() not in packaged, (
            "a file under .hypothesis is packaged into the reviewer archive")
    finally:
        probe.unlink()
        for d in (probe.parent, probe.parent.parent):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()


# ---------------------------------------------------------------------------
# 10. rules written to a category rather than to a sample of it
# ---------------------------------------------------------------------------
@needs_gate
def test_the_credential_rule_covers_the_shapes_vendors_issue_today():
    """sk-[A-Za-z0-9]{20,} stops at a hyphen, and the current keys have one.

    sk-proj- and sk-svcacct- have been OpenAI's issued formats since 2024 and
    both put a hyphen inside the first twenty characters, so neither matched;
    the bare sk- shape the rule did catch is the one being retired. ghp_ is
    not github_pat_, and the four sibling gh?_ kinds had no rule at all.

    The routes are real: experiment_frontier_margin.py posts a Bearer token
    and prints an OPENAI_API_KEY=... line in its own docstring, and
    paper-a/releases/critique_*.json ships verbatim command transcripts.
    Every probe below is a syntactically valid shape with an invented body.
    """
    import importlib

    import build_release_repo as rel
    importlib.reload(rel)

    def caught(s):
        return any(rx.search(s) for rx, _ in rel.DENY_TEXT)

    body = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ"
    for shape in (f"sk-proj-{body}", f"sk-svcacct-{body}", f"sk-ant-api03-{body}",
                  f"sk-{body}", f"github_pat_11ABCDEFG0{body}{body}",
                  f"ghp_{body}", f"ghs_{body}", f"gho_{body}", f"hf_{body}",
                  # Split: whole, this IS a credential shape, and a gate
                  # that let a file containing one ship would be the bug.
                  "AKIA" + "IOSFODNN7EXAMPLE", f"AIzaSyA-{body}",
                  # Split for the same reason as the AWS shape above.
                  "xoxb-" + "123456789012-" + body[:12]):
        assert caught(shape), f"{shape[:18]}... is not recognised as a credential"

    # Prose about a prefix is not a credential.
    for quiet in ("the sk- prefix is discussed in the appendix",
                  "import sklearn as sk", "see ghp for the format"):
        assert not caught(quiet), f"false refusal on {quiet!r}"


@needs_gate
def test_the_identity_tripwires_are_case_and_separator_agnostic():
    """Every identity rule in the table carries re.I except the one that did not.

    The account-name rule was the last line of defence for the home-directory
    form SANITISE misses -- that table replaces literals, so an uppercased
    path out of cmd is neither rewritten nor flagged. The outreach tripwire
    assumed a forward slash on a Windows tree whose recorded shell commands
    carry backslashes, and the LinkedIn rule missed the vendor's own
    capitalisation, which is the single most likely spelling.
    """
    import importlib

    import build_release_repo as rel
    importlib.reload(rel)

    def caught(s):
        return any(rx.search(s) for rx, _ in rel.DENY_TEXT)

    # NOTHING DENIED IS SPELLED HERE. Written out, these probes ARE the
    # strings the rules hunt, in a file that ships publicly -- and the gate
    # refused this file until they were built at runtime instead, which is
    # the widened rules working on their first real input. The account name
    # comes from its own rule's parse tree; the paths are split so the denied
    # sequence exists only after concatenation.
    acct = matching_string(
        next(rx for rx, why in rel.DENY_TEXT
             if "account name" in why).pattern)
    assert acct, "the account-name rule yielded no literal to probe with"

    for probe in (f"C:/Users/{acct.upper()}/AppData",
                  f"C:/Users/{acct.title()}/scratch",
                  f"{acct}_scratch", f"{acct}2", f"/{acct}/",
                  "see outreach" + "\\" + "NOTES.md",
                  "see OUTREACH" + "/notes.md",
                  "https://www.LinkedIn.com" + "/in/someone"):
        assert caught(probe), f"{probe!r} escapes every tripwire"

    # A surname that merely CONTAINS the account name is not the account name.
    # Bare, this rule refused a release over an ordinary citation; \b would
    # reintroduce the founding bug, since the live path forms are acct_ and
    # acct2 and "_" and digits are word characters.
    for quiet in (f"Chai{acct}ana et al. 2019",
                  f"the {acct.title()}enko method"):
        assert not caught(quiet), (
            f"false refusal on {quiet!r}; a citation must not stop a release")


@needs_gate
def test_the_byte_rules_know_more_than_one_spelling_of_a_name():
    """The binary branch is the only path where sanitise() never runs.

    b"david mao" assumed exactly one space, so the underscore, hyphen, dotted
    and inverted spellings a PDF /Author or XMP dc:creator field routinely
    carries all missed -- the founding separator case in byte form. The list
    also had no counterpart to the three text rules guarding the absolute
    local path, so a figure regenerated by a tool that stamps its input path
    would have shipped untouched and unflagged.

    The author's name is not spelled here: the probes are built from the
    module's own generated spellings.
    """
    import importlib

    import build_release_repo as rel
    importlib.reload(rel)

    lits = [p for p, _ in rel.DENY_BYTES]

    # BUILD THE PROBES, DO NOT READ THEM BACK. The first version of this test
    # asked each literal whether the list containing it matched -- true by
    # construction -- and it passed against a list carrying a single spelling.
    # The two name parts come from the inverted literal, which is the one
    # entry that spells both, and each separator form is rebuilt here.
    inverted = next(p for p, why in rel.DENY_BYTES
                    if why == "names the author, inverted")
    last, first = (x.strip() for x in inverted.split(b","))

    png = b"\x89PNG\r\n\x1a\n\x00\xff"
    for sep in (b" ", b"_", b"-", b".", b""):
        probe = first + sep + last
        blob = (png + probe.upper() + b"\x00\xfe").lower()
        assert any(p.lower() in blob for p in lits), (
            f"a binary carrying the {sep!r}-joined author name is not "
            "caught; the byte list is back to a single spelling")

    # And the private working tree is represented in bytes, not only in text.
    assert any(b"my drive" in p.lower() for p in lits), (
        "no byte rule guards the absolute private path, on the one branch "
        "where sanitise() does not run")
    # Recovered from the rule, not spelled: this file ships.
    acct = matching_string(
        next(rx for rx, why in rel.DENY_TEXT
             if "account name" in why).pattern)
    assert any(acct.encode("ascii") in p.lower() for p in lits), (
        "no byte rule guards the home directory")


# ---------------------------------------------------------------------------
# 11. the ICLR gates: what they read before they decide
# ---------------------------------------------------------------------------
@needs_check
def test_the_allow_drift_check_reads_a_decorated_macro_value():
    r"""The check exists because ALLOW drifted onto five generated values.

    It read them with `\{([^{}]*)\}` and keyed on the raw text, so a value
    with braces was invisible -- \RuleFprHi's "1.2\times 10^{-2}" and its
    sibling, both generated measurements -- and a percentage stored as
    "8.7\%" never matched a bare "8.7" in ALLOW. Measured on this build: 15
    of 106 values carry markup and 14 distinct numbers were unreachable.
    Adding a percentage to ALLOW is the edit that produced the founding
    failure, and it was the edit the check could not see.
    """
    import importlib

    import check_iclr as C
    importlib.reload(C)

    assert C.macro_numbers("8.7\\%") == {"8.7"}, "a percentage is not read"
    assert C.macro_numbers("1.2\\times 10^{-2}") == {"1.2", "10"}, (
        "a braced value is not read, or the exponent is read as a number")
    assert C.macro_numbers("0.62") == {"0.62"}

    gen = ROOT / "paper-a" / "iclr" / "generated"
    if not gen.is_dir():
        pytest.skip("no generated macros on this checkout")
    seen = {}
    for g in sorted(gen.glob("*.tex")):
        for gm in C.MACRO_DEF.finditer(g.read_text(encoding="utf-8")):
            for num in C.macro_numbers(gm.group(2)):
                seen.setdefault(num, []).append(gm.group(1))
    assert seen, "no macro values parsed at all"
    # EVERY DEFINED MACRO IS REACHABLE. The first version of this assertion
    # derived the expected set with C.MACRO_DEF -- the regex under test --
    # so reverting that regex emptied the set and the check became vacuous.
    # A macro's NAME can be read without parsing its value at all, which
    # makes it the independent quantity.
    names = {n for ns in seen.values() for n in ns}
    # Only a macro whose value CARRIES a number needs to be reachable: one
    # that renders a word (\NSlopeChecked prints "four") has nothing an
    # ALLOW entry could collide with. The definition line is scanned for a
    # digit rather than parsed, and not parsing the value is what keeps this
    # independent of the regex under test.
    declared = set()
    for g in gen.glob("*.tex"):
        for line in g.read_text(encoding="utf-8").split("\n"):
            dm = re.match(r"\s*\\newcommand\{?\\([A-Za-z]+)", line)
            if dm and any(c.isdigit() for c in line[dm.end():]):
                declared.add(dm.group(1))
    assert declared <= names, (
        "these macros are invisible to the drift check, so an ALLOW entry "
        f"equal to one of their values would never be reported: "
        f"{sorted(declared - names)}")
    # And the live ALLOW set is genuinely clean under the wider view.
    assert not [a for a in C.ALLOW if a in seen], (
        f"ALLOW has drifted onto a generated value: "
        f"{[a for a in C.ALLOW if a in seen]}")


@needs_check
def test_the_bibliography_regressions_survive_a_line_break():
    """Every literal here is multi-word and the PDF is line-broken.

    The same defect the ANONYMITY scan had, in the same function, fixed there
    and not here. "not a peer-reviewed paper" can break at a space or at its
    own hyphen, and one regeneration of the bibliography carrying the fuller
    registry note prints that sentence into the references.
    """
    import importlib

    import check_iclr as C
    importlib.reload(C)

    # REGRESSIONS is local to main(); what is reachable, and what the
    # gate actually depends on, is the reconstruction.
    for probe in ("not a peer-reviewed paper",
                  "not a peer-reviewed\npaper",
                  "not a peer-\nreviewed paper",
                  "not a\npeer-reviewed paper"):
        assert any("not a peer-reviewed paper" in v
                   for v in C.pdf_views(probe)), (
            f"the bibliography literal is invisible when broken as {probe!r}")


@needs_check
def test_the_runbook_gate_fails_on_its_own_founding_case():
    """A shortened quotation is still a substring.

    This gate was written after the runbook's quotation dropped the clause
    disclosing that a model screened the surveyed literature, and then argued
    from the shortened version that an OpenReview box could stay unticked.
    Compared with `in`, re-running that exact deletion passes. Compared as an
    equality, it cannot.
    """
    import importlib

    import check_iclr as C
    importlib.reload(C)

    full = ("draft and revise this version's text, implement the analysis "
            "and build-pipeline code, screen the surveyed literature for the "
            "reporting matrix of \\S\\ref{sec:field}, and run adversarial "
            "audits of the paper's claims against its artifacts")
    stated = C._disclosure(full)

    # The rendered section number and the LaTeX ref must compare equal: a
    # renumbering is not a disclosure change.
    rendered = full.replace("\\S\\ref{sec:field}", "§8")
    assert C._disclosure(rendered) == stated, (
        "a section renumbering would fail the gate")

    # The founding deletion: drop the literature-screening clause.
    shortened = ("draft and revise this version's text, implement the "
                 "analysis and build-pipeline code")
    assert C._disclosure(shortened) != stated, "the shortened quote compares equal"
    assert C._disclosure(shortened) in stated, (
        "the shortened quote is no longer a substring, so this test is not "
        "exercising the defect it names")

    # And the other direction: the paper drops a use the quote never had.
    trimmed = full.split(", and run adversarial")[0]
    assert C._disclosure(trimmed) != stated


@needs_check
def test_the_macro_gate_reads_the_appendix_and_what_it_inputs():
    r"""A compile error in the appendix is still a compile error.

    The gate exists to hear one "with a name instead of on Overleaf at the
    deadline", and it split main.tex at \appendix and read only what came
    before -- while the appendix \inputs generated/tab-design. Nothing slips
    through today, which is why it was worth closing now.

    Its uppercase-only rule is what keeps it from drowning in \sqrt and
    \rho, and that rule is sound exactly as long as every generated macro is
    capitalised. That was true and unstated.
    """
    import importlib

    import check_iclr as C
    importlib.reload(C)

    src = _code(pathlib.Path(C.__file__))
    assert "scanned = tex" in src, (
        "the macro gate is reading the pre-appendix body again")

    defined = C.defined_macros()
    assert defined, "no macros defined"
    assert not [m for m in defined if m[:1].islower()], (
        "a generated macro is lowercase, which the gate cannot see; the "
        "build should be saying so")


@needs_check
def test_a_lowercase_generated_macro_makes_the_gate_say_so():
    r"""The precondition is tested by breaking it, not by reading the source.

    The first version of this asserted that the name "lower_defined" appears
    in check_iclr.py and that no macro is lowercase today. A revert that
    keeps the name and empties the branch passes both. So this creates the
    condition -- one lowercase macro in generated/ -- and asks the gate what
    it says.
    """
    import subprocess

    gen = ROOT / "paper-a" / "iclr" / "generated"
    if not gen.is_dir():
        pytest.skip("no generated macros on this checkout")
    probe = gen / "_probe_lowercase.tex"
    probe.write_text("\\newcommand{\\probelowercase}{1}\n", encoding="utf-8")
    try:
        p = subprocess.run(
            [sys.executable, str(ROOT / "paper-a" / "src" / "check_iclr.py")],
            capture_output=True, text=True, cwd=str(ROOT))
        assert "probelowercase" in p.stdout, (
            "a lowercase generated macro is invisible to the gate and the "
            "gate does not say so, so its uppercase-only rule is silently "
            "unsound:\n" + p.stdout)
    finally:
        probe.unlink()


# ---------------------------------------------------------------------------
# The section-pointer audit: what counts as a pointer, and what it does with
# the ones it finds.
# ---------------------------------------------------------------------------
def _secrefs():
    import audit_section_refs
    return audit_section_refs


def test_a_pointer_that_names_more_than_one_section_is_read_in_full():
    r"""A pointer can carry two numbers, and both of them are pointers.

    "Sections 6.1 and 6.3" is in the built preprint. The audit read one
    number per pointer and matched the singular word form, so the plural
    defeated it outright and the tail of every list went unchecked. Both
    sections happen to exist, which is why nothing showed: the audit's own
    docstring already records the previous version of this same mistake --
    "they were sound, but that was luck."
    """
    A = _secrefs()
    sign = A.SECTION_SIGN
    got = A.refs_from_text(
        "Sections 6.1 and 6.3 concern the case where the conversion is "
        f"exact, and {sign}4.6 and 4.7 give the two panels.")
    missing = [n for n in ("6.1", "6.3", "4.6", "4.7") if not got[n]]
    assert not missing, (
        f"{missing} appear in a pointer but never reach the check, so a "
        "dangling one would be reported as clean")


def test_a_statute_citation_is_not_a_pointer_into_this_paper():
    r"""The statute is excluded by context: the hyphen that follows it.

    The rule this replaces tested the harvested VALUE, by which point the
    hyphen was gone -- so it exempted the bare "5" and "20", which is also
    what a pointer to this paper's own section 5 looks like.
    """
    A = _secrefs()
    sign = A.SECTION_SIGN
    got = A.refs_from_text(
        f"6 RCNY {sign} 5-300 et seq., adopted 6 April 2023; N.Y.C. Admin. "
        f"Code {sign}{sign} 20-870 to 20-874; a summary of the results is "
        f"published ({sign} 20-871(a)).")
    assert not got, (
        f"a statute citation was harvested as a pointer into this paper: "
        f"{dict(got)}; the audit must then exempt it by value, and the only "
        "values it can exempt are the ones real pointers also use")


def test_a_pointer_to_this_papers_own_section_five_is_harvested():
    r"""The other direction of the same rule.

    Excluding the statute must not exclude the sections whose numbers it
    happens to share. Section 5 is pointed at three times across the two
    documents -- "it says nothing about 5: quantization" and two in the fork.
    """
    A = _secrefs()
    sign = A.SECTION_SIGN
    for n in ("5", "20", "7"):
        got = A.refs_from_text(
            f"It says nothing about {sign}{n}: quantization, batching.")
        assert got[n] == 1, (
            f"a pointer to section {n} is not harvested, so a dangling one "
            f"would never be reported; harvested {dict(got)}")


def test_a_comma_after_a_pointer_is_punctuation_and_not_a_list():
    r"""Why the list separators are "and" and "&" and nothing else.

    They were read off the two built documents rather than guessed at. Every
    comma that follows a pointer there is ordinary punctuation, and admitting
    one invents a pointer to a section 0 out of the sentence below -- a
    dangling reference the paper does not contain, in the gate that exists to
    report dangling references. A gate has two failure modes and this is the
    one that teaches people to ignore it.
    """
    A = _secrefs()
    sign = A.SECTION_SIGN
    got = A.refs_from_text(
        f"Of the audits surveyed in {sign}8, 0 of 8 report it.")
    assert dict(got) == {"8": 1}, (
        f"harvested {dict(got)} where the only pointer is section 8")


def test_every_pointer_the_audit_finds_is_a_pointer_it_checks():
    r"""No pointer is harvested and then dropped on the way to the check.

    The audit used to filter statute-shaped values out AFTER harvesting, so
    its "N pointer(s) checked" was smaller than the number of pointers it had
    found, and the difference was silent. Whatever the exclusion rule is, it
    belongs at the harvest -- once a string is called a pointer, it gets
    checked. This compares the two counts against the real documents.
    """
    import subprocess

    A = _secrefs()
    pytest.importorskip("fitz", reason="pymupdf reads the built PDFs")
    harvested = 0
    built = 0
    for _name, pdf, _src, _pat in A.DOCS:
        if not pdf.exists():
            continue
        built += 1
        harvested += sum(A.refs_from_pdf(pdf).values())
    if not built:
        pytest.skip("neither document is built on this checkout")
    p = subprocess.run(
        [sys.executable, str(SRC / "audit_section_refs.py")],
        capture_output=True, text=True, cwd=str(ROOT))
    m = re.search(r"(\d+) pointer\(s\) checked", p.stdout)
    assert m, "the audit did not report how many pointers it checked:\n" + \
        p.stdout
    assert int(m.group(1)) == harvested, (
        f"the audit harvested {harvested} pointer(s) but checked "
        f"{m.group(1)}; the difference is exempted somewhere downstream and "
        "nothing says so")


# ---------------------------------------------------------------------------
# Exemptions in the typed-measurement gate: a reason, or a switch left off.
# ---------------------------------------------------------------------------
def _hardtyped():
    import audit_hardtyped_numbers
    if not audit_hardtyped_numbers.SRC.exists():
        pytest.skip("the preprint builder is not on this checkout")
    return audit_hardtyped_numbers


def test_an_exemption_that_matches_nothing_is_reported(capsys):
    r"""The report is tested by creating the condition, not by reading code.

    Every entry in EXEMPT turns the typed-measurement gate off over the span
    it matches. When the prose an entry was bound to is reworded away, the
    reason stops applying but the rule stays armed for whatever matches it
    next -- and nothing shows, because a dead exemption and a satisfied one
    both remove zero hits. Three entries were in exactly that state when this
    was written.
    """
    H = _hardtyped()
    why = "a probe bound to prose that is not in the builder"
    probe = (r"\bzzq-no-such-phrase-in-any-paper\b", why)
    H.EXEMPT.append(probe)
    try:
        H.main()
    finally:
        H.EXEMPT.remove(probe)
    out = capsys.readouterr().out
    assert why in out, (
        "an exemption that matched nothing went unmentioned, so a reason "
        "that no longer applies to any text stays armed silently:\n" + out)


def test_every_exemption_still_applies_to_something(capsys):
    r"""Drift check, run against the builder as it stands.

    Not a list of known-dead entries to maintain -- the audit computes it.
    If prose moves and an exemption stops matching, this says so and someone
    decides whether the sentence comes back or the rule goes.
    """
    H = _hardtyped()
    H.main()
    out = capsys.readouterr().out
    dead = [ln.strip() for ln in out.splitlines() if "unused [" in ln]
    assert not dead, (
        "an exemption no longer applies to any printed text:\n  "
        + "\n  ".join(dead))


# ---------------------------------------------------------------------------
# The documented-figures autofix: which mention does it actually repair?
# ---------------------------------------------------------------------------
def test_the_autofix_repairs_every_stale_mention_not_just_the_first(capsys):
    r"""A correct mention standing before a stale one used to absorb the fix.

    --fix looped once per stale hit and called re.subn(..., count=1), which
    always rewrites the FIRST match in the document rather than the hit being
    handled. So a file saying the right number once and the wrong number once
    had its RIGHT mention rewritten -- a no-op -- while subn still reported
    one substitution, which the code took as proof of repair: it printed
    "fixed", counted it, and suppressed the problem report. The stale number
    survived and the audit called it repaired.

    The document is built here rather than found, because no claim in the
    tree matches twice today. The audit's own docstring explains why one
    soon will: the preprint's page count moves on almost every content edit.
    """
    import audit_doc_figures as D

    if D.main() != 0:
        capsys.readouterr()
        pytest.skip("a real document is already stale; not running --fix")
    capsys.readouterr()

    T = D.truth()
    key = next((k for k, v in T.items() if isinstance(v, int)), None)
    if key is None:
        pytest.skip("no integer figure to build a probe from")
    want = T[key]

    probe = ROOT / "_probe_doc_figures.md"
    pattern = r"the probe count is (\d+) exactly"
    probe.write_text(
        f"First, correctly: the probe count is {want} exactly.\n"
        f"Later, staler: the probe count is {want + 1} exactly.\n",
        encoding="utf-8")
    claim = ("_probe_doc_figures.md", pattern, key, "a constructed probe")
    D.CLAIMS.append(claim)
    argv = sys.argv[:]
    sys.argv = [argv[0], "--fix"]
    try:
        D.main()
        found = re.findall(pattern, probe.read_text(encoding="utf-8"))
    finally:
        sys.argv = argv
        D.CLAIMS.remove(claim)
        probe.unlink(missing_ok=True)
        capsys.readouterr()

    assert found == [str(want), str(want)], (
        f"--fix left {found} where both mentions should read {want}; a "
        "correct mention standing first absorbed the substitution and the "
        "stale one was reported as repaired")


# ---------------------------------------------------------------------------
# Cross-references baked into a figure: the spellings a figure actually uses.
# ---------------------------------------------------------------------------
def test_a_reference_baked_into_a_figure_is_caught_in_every_spelling():
    r"""An axis label abbreviates; the rule only knew the long form.

    The rule was case-sensitive, singular and spelled out, so "Fig. 2",
    "figures 2 and 3" and "see table 3" all passed -- the three forms a
    figure is most likely to draw, because a label has less room than a
    sentence. A figure that points at a section cannot be re-captioned per
    venue, which is the whole reason this check exists.
    """
    import audit_figure_refs as A
    missed = [s for s in (
        "see Figure 2", "see Fig. 2", "Figs. 2", "figures 2 and 3",
        "see Table 3", "see table 3", "Tab. 3", "tables 2 and 3",
        "Section 6.2", "section 6.2", "Sec. 6.2", "Sections 6.1 and 6.3",
        "Appendix B", "appendix B", "App. B", "Appendices B",
    ) if not A.REF.search(s)]
    assert not missed, (
        f"a figure could draw {missed} and the check would call it "
        "self-contained")


def test_the_abbreviations_still_require_their_period():
    r"""Why "sec" and "tab" are not enough on their own.

    "sec" is also the abbreviation for seconds and "tab" for a user-interface
    tab. A tick label reading "5 sec" is not a cross-reference, and a gate
    that says it is would be a standing false alarm on any timing axis --
    which is the failure mode that gets a gate ignored.
    """
    import audit_figure_refs as A
    cried_wolf = [s for s in (
        "0 sec 5 sec 10", "elapsed secs", "tab 3 of the interface",
        "app B store", "the second run", "figurative",
    ) if A.REF.search(s)]
    assert not cried_wolf, (
        f"{cried_wolf} would be reported as baked cross-references")


# ---------------------------------------------------------------------------
# A number that is missing must not typeset as a dash.
# ---------------------------------------------------------------------------
def _other_venue_builder():
    """The second venue's macro builder, which the release does not carry."""
    p = SRC / ("build_" + "fac" + "ct_tex.py")
    if not p.exists():
        pytest.skip("the second venue's tooling is not part of the release")
    import importlib
    return importlib.import_module(p.stem)


def test_a_missing_percentage_leaves_its_macro_undefined():
    r"""The two helpers have to agree about what absence means.

    num() returns None when the artifact key is missing, M() then declines to
    define the macro, and the LaTeX run stops on an undefined control
    sequence -- loud, and correct. Its sibling returned "--" for the same
    case; M() stores anything that is not None, so the macro came out
    DEFINED, as an em dash, and typeset that way inside a sentence asserting
    a measurement. The quiet helper was the one carrying the headline
    dispersion ratio.
    """
    F = _other_venue_builder()
    assert F._pctnum(None) is None, (
        f"a missing percentage renders as {F._pctnum(None)!r}, which the "
        "macro writer stores rather than skips, so the paper typesets it "
        "where a measurement was promised")


def test_no_generated_macro_of_the_second_venue_is_a_dash(capsys):
    r"""The same property, read off the artifacts rather than the code.

    A drift check: whatever the helpers do, no macro that reaches the page
    may be defined as a dash, because a dash in a macro slot is a
    measurement that silently went missing.
    """
    _other_venue_builder()
    gen = ROOT / "paper-a" / ("fac" + "ct") / "generated"
    if not gen.is_dir():
        pytest.skip("no generated macros for that venue on this checkout")
    dashes = []
    for f in sorted(gen.glob("*.tex")):
        for m in re.finditer(r"\\newcommand\{?\\(\w+)\}?\{(-{1,3})\}",
                             f.read_text(encoding="utf-8")):
            dashes.append(f"{f.name}: \\{m.group(1)} = {m.group(2)!r}")
    assert not dashes, (
        "a macro is defined as a dash, so a number the paper promises is "
        "absent and nothing says so:\n  " + "\n  ".join(dashes))


# ---------------------------------------------------------------------------
# The other venue's desk-rejection checks: scope, spelling, and self-disabling.
# ---------------------------------------------------------------------------
def _other_venue_gate():
    """The structural gate for the second venue, which the release omits."""
    p = SRC / ("check_" + "fac" + "ct_tex.py")
    if not p.exists():
        pytest.skip("the second venue's gate is not part of the release")
    import importlib
    return importlib.import_module(p.stem)


def test_a_two_word_term_is_found_however_latex_splits_it():
    r"""The anonymity check matched one literal space and nothing else.

    LaTeX source wraps lines wherever it likes and ties words with ~, so a
    two-word name written across a line break, or tied, went unseen by the
    check that decides whether a double-blind submission stays blind. That is
    this project's founding bug with the separator changed.

    A neutral term is used: the property is general, and this file ships.
    """
    C = _other_venue_gate()
    rx = C.loose("Given Family")
    missed = [v for v in (
        "Given Family", "Given\nFamily", "Given~Family", "Given  Family",
        "{Given} {Family}", "Given\\ Family", "Given\n  Family",
        "GIVEN FAMILY", "given family",
    ) if not rx.search(v)]
    assert not missed, (
        f"a two-word identity term spelled {missed} would pass the "
        "anonymity check")


def test_the_loose_match_does_not_reach_across_words():
    r"""The gap may span spacing, never letters -- else it cries wolf."""
    C = _other_venue_gate()
    rx = C.loose("Given Family")
    wrong = [v for v in ("GivenXFamily", "Given and Family",
                         "Givenfamily", "Given, Family")
             if rx.search(v)]
    assert not wrong, f"{wrong} matched, so the gap spans more than spacing"


def test_a_banned_section_in_an_included_file_is_caught():
    r"""Two holes at once: the wrong construct, in a file never read.

    The banned list looked for a hypothetical \acks{...}; acmart's real
    construct is the environment \begin{acks}...\end{acks}, which is what an
    author writes. And the check read main.tex alone, while the document
    inputs its generated parts -- so the form that occurs, in the place it
    would occur, was invisible twice over.
    """
    import subprocess

    C = _other_venue_gate()
    gen = C.GEN
    if not gen.is_dir() or not C.MAIN.exists():
        pytest.skip("that venue's document is not on this checkout")

    main_before = C.MAIN.read_text(encoding="utf-8")
    probe = gen / "_probe_banned.tex"
    probe.write_text("\\begin{acks}\nWe thank nobody in particular.\n"
                     "\\end{acks}\n", encoding="utf-8")
    C.MAIN.write_text(
        main_before.replace("\\end{document}",
                            "\\input{generated/_probe_banned}\n"
                            "\\end{document}"),
        encoding="utf-8")
    try:
        p = subprocess.run(
            [sys.executable, str(SRC / ("check_" + "fac" + "ct_tex.py"))],
            capture_output=True, text=True, cwd=str(ROOT))
    finally:
        C.MAIN.write_text(main_before, encoding="utf-8")
        probe.unlink(missing_ok=True)
    assert C.MAIN.read_text(encoding="utf-8") == main_before, \
        "the document was not restored"

    assert p.returncode != 0 and "acks environment" in p.stdout, (
        "an acks environment inside an included file was not reported as a "
        "desk rejection:\n" + p.stdout)


def test_the_gate_will_not_certify_anonymity_it_could_not_check():
    r"""The confidential terms live in a file the repository never publishes.

    When that file is absent the terms are not checked. That used to be a
    note, so the gate still printed "anonymised" at the end -- a check that
    silently turned itself off and a satisfied one produced the same verdict.
    """
    import subprocess

    C = _other_venue_gate()
    local = C.FACCT / "identity_terms.local.txt"
    if not local.exists():
        pytest.skip("the confidential terms are not on this machine")
    aside = local.with_suffix(".txt.founding-case")
    body = local.read_bytes()
    local.rename(aside)
    try:
        p = subprocess.run(
            [sys.executable, str(SRC / ("check_" + "fac" + "ct_tex.py"))],
            capture_output=True, text=True, cwd=str(ROOT))
    finally:
        aside.rename(local)
    assert local.read_bytes() == body, "the terms file was not restored"

    assert p.returncode != 0, (
        "with the confidential terms unavailable the gate still passed, so "
        "it certified an anonymity it did not check:\n" + p.stdout)


def test_the_gate_itself_catches_an_identity_term_broken_across_lines():
    r"""Not the helper -- the gate.

    The companion case above proves loose() survives a line break, and that
    is not the same claim: reverting the call site to re.escape leaves it
    green, because it never runs the gate. So this puts a synthetic term into
    the confidential list, writes it into the document ACROSS A LINE BREAK,
    and asks the gate what it says. The term is invented; no real identity
    appears in this file, which ships.
    """
    import subprocess

    C = _other_venue_gate()
    local = C.FACCT / "identity_terms.local.txt"
    if not local.exists() or not C.MAIN.exists():
        pytest.skip("the confidential terms are not on this machine")

    term_a, term_b = "Zzqprobe", "Wwvterm"
    local_before = local.read_bytes()
    main_before = C.MAIN.read_text(encoding="utf-8")
    local.write_text(
        local_before.decode("utf-8").rstrip("\n") + f"\n{term_a} {term_b}\n",
        encoding="utf-8")
    C.MAIN.write_text(
        main_before.replace(
            "\\end{document}",
            f"A sentence mentioning {term_a}\n{term_b} here.\n"
            "\\end{document}"),
        encoding="utf-8")
    try:
        p = subprocess.run(
            [sys.executable, str(SRC / ("check_" + "fac" + "ct_tex.py"))],
            capture_output=True, text=True, cwd=str(ROOT))
    finally:
        local.write_bytes(local_before)
        C.MAIN.write_text(main_before, encoding="utf-8")
    assert local.read_bytes() == local_before, "the terms file was not restored"
    assert C.MAIN.read_text(encoding="utf-8") == main_before, \
        "the document was not restored"

    assert p.returncode != 0 and "confidential affiliation" in p.stdout, (
        "a confidential term written across a line break passed the "
        "anonymity check, so the gate is not using the loose match:\n"
        + p.stdout)


# ---------------------------------------------------------------------------
# The refusal detector is frozen. This watches the assumption that lets it be.
# ---------------------------------------------------------------------------
def test_no_recorded_output_hides_a_refusal_behind_a_curly_apostrophe():
    r"""The detector knows only the ASCII apostrophe, and that is left alone.

    Every contraction in REFUSAL_PATTERNS is written with '. Models routinely
    emit U+2019, and a refusal spelled with one matches nothing -- so it is
    recorded as a screening verdict, which is the invisible failure the
    module's own comment describes.

    The patterns are frozen after the Stage 0 pilot, with an explicit
    instruction to bump PROMPT_VERSION and re-run rather than adjust them
    mid-run, so the gap stays. What makes that safe is an empirical claim
    about the collected data, and an empirical claim should be checked rather
    than remembered: no recorded output contains a typographic apostrophe at
    all. The moment one does, this fails, and the decision has to be made
    again by someone rather than assumed.
    """
    import json

    data = ROOT / "paper-a" / "data"
    if not data.is_dir():
        pytest.skip("no collected data on this checkout")
    sys.path.insert(0, str(SRC))
    try:
        import run_audit
    except Exception:  # noqa: BLE001
        pytest.skip("the audit runner is not importable here")

    curly = "\u2019"
    scanned = 0
    hidden = []
    for f in sorted(data.rglob("*.jsonl")):
        for line in f.read_text(encoding="utf-8",
                                errors="replace").split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            for key in ("raw", "white_raw", "black_raw"):
                text = rec.get(key)
                if not isinstance(text, str) or not text:
                    continue
                scanned += 1
                if curly not in text:
                    continue
                if (not run_audit.looks_like_refusal(text)
                        and run_audit.looks_like_refusal(
                            text.replace(curly, "'"))):
                    hidden.append(f"{f.name}: {text[:100]!r}")

    if not scanned:
        pytest.skip("no model-generated text in the collected data")
    assert not hidden, (
        f"{len(hidden)} recorded output(s) read as a refusal only once the "
        "typographic apostrophe is normalised, so they were counted as "
        "screening verdicts. The detector is frozen: bump PROMPT_VERSION and "
        "re-run that arm rather than widening the pattern in place.\n  "
        + "\n  ".join(hidden[:5]))


# ---------------------------------------------------------------------------
# The reporting matrix: comparing identifiers, and keeping what was written.
# ---------------------------------------------------------------------------
def _matrix_builder():
    import build_reporting_matrix
    return build_reporting_matrix


def test_an_identifier_is_compared_by_value_not_by_the_text_around_it():
    r"""The pattern was case-insensitive; the comparison was not.

    The cross-check compared the raw matched text of the FIRST match in each
    field, so "ArXiv:" against "arXiv:", "V1" against "v1", and a reference
    that names another paper's identifier before its own were all reported as
    disagreements -- each raising SystemExit and stopping the build. A gate
    that halts a build over a capital letter is a gate someone deletes, which
    is the more insidious of the two ways a gate fails.
    """
    B = _matrix_builder()
    want = B.arxiv_ids("arXiv:2406.10486v1")
    assert want, "the identifier pattern matched nothing at all"
    for variant in ("ArXiv:2406.10486v1", "ARXIV:2406.10486V1",
                    "arXiv: 2406.10486v1", "arXiv:2406.10486v1 [cs.AI]"):
        assert B.arxiv_ids(variant) == want, (
            f"{variant!r} reads as a different identifier, so the build "
            "stops on a difference in spelling")
    both = B.arxiv_ids("see arXiv:2101.00001 and arXiv:2406.10486v1")
    assert want <= both, (
        "only the first identifier in a field is read, so a reference that "
        "mentions another paper first is compared against the wrong one")


def test_a_real_disagreement_between_versions_still_shows():
    r"""The widening must not cost the check its founding case.

    Armstrong's entry said v2 in the reference and v3 in the transcription
    for two rounds and nothing noticed. That must still be a disagreement.
    """
    B = _matrix_builder()
    assert B.arxiv_ids("arXiv:2407.20371v2") != B.arxiv_ids(
        "arXiv:2407.20371v3"), "two versions of a paper read as the same one"
    assert not (B.arxiv_ids("arXiv:2407.20371v3")
                <= B.arxiv_ids("arXiv:2407.20371v2"))


def test_every_note_in_the_built_matrix_comes_from_a_raw_reading():
    r"""The matrix is BUILT, so anything hand-added to it is deleted next run.

    A note recording why one venue line came from the ACL Anthology listing
    rather than from the PDF the row was coded against had been typed
    straight into the built file -- it sorted last among that entry's keys,
    which is the signature of an edit after the build -- and the next rebuild
    duly destroyed it. It lives in the raw reading now. This fails if another
    one is ever written into the output, and says where it belongs.
    """
    import json

    ref = ROOT / "paper-a" / "data" / "reference"
    built_p = ref / "reporting_practice_matrix.json"
    if not built_p.exists() or not (ref / "raw").is_dir():
        pytest.skip("the reporting matrix is not on this checkout")
    built = json.loads(built_p.read_text(encoding="utf-8"))
    raw_text = "\n".join(
        f.read_text(encoding="utf-8") for f in sorted((ref / "raw").glob("*.json")))

    orphaned = [
        f"{s['label']}: {s['citation_check_note'][:60]}..."
        for s in built["studies"]
        if s.get("citation_check_note")
        and json.dumps(s["citation_check_note"])[1:-1] not in raw_text]
    assert not orphaned, (
        "a note in the built matrix is in no raw reading, so the next "
        "rebuild deletes it. Put it on the study's row under "
        "paper-a/data/reference/raw/ instead:\n  " + "\n  ".join(orphaned))


# ---------------------------------------------------------------------------
# The consistency audit: a scan frozen at one transition, and dead exemptions.
# ---------------------------------------------------------------------------
def _consistency():
    import audit_consistency
    return audit_consistency


def test_the_stale_count_scan_still_finds_the_superseded_count(capsys):
    r"""Frozen at one transition, and it had to stay live through the repair.

    The scan read `words.get(8) if n != 8 else None`, hardcoding the
    superseded value inside a lookup so that only "eight" was ever searched
    for -- the map's entries for eleven and twelve could not be reached, and
    a later move of the count would go unlooked-for in silence. It is a named
    list of the counts this fact has held now.

    Widening it to EVERY word that is not the current count was tried and
    reverted: "conditions" is too generic a noun and it raised four false
    alarms on unrelated prose. So this checks the narrow property that
    matters -- the superseded count is still found where it is written.
    """
    A = _consistency()
    doc = next((d for d in A.DOCS if d.exists()), None)
    if doc is None:
        pytest.skip("none of the scanned documents is on this checkout")
    before = doc.read_text(encoding="utf-8")
    doc.write_text(
        before + "\n\nProbe: the panel used eight semantically null "
        "conditions.\n", encoding="utf-8")
    try:
        A.ISSUES.clear()
        A.check_counts()
        out = capsys.readouterr().out
    finally:
        doc.write_text(before, encoding="utf-8")
        A.ISSUES.clear()
    assert doc.read_text(encoding="utf-8") == before, "document not restored"
    assert "eight semantically null conditions" in out, (
        "a superseded condition count written into a scanned document was "
        "not found:\n" + out)


def test_an_exemption_that_skips_nothing_is_reported(capsys):
    r"""An exemption removes no finding when it is dead AND when it is working.

    Both states print the same thing -- nothing -- so a rule whose reason has
    expired stays armed for whatever text lands in that file next, and no one
    is told. One entry here named a builder that had already been deleted.
    """
    A = _consistency()
    probe = "zzq_no_such_module.py"
    original = A.EXEMPT
    A.EXEMPT = tuple(original) + (probe,)
    try:
        A.ISSUES.clear()
        A.check_superseded()
        out = capsys.readouterr().out
    finally:
        A.EXEMPT = original
        A.ISSUES.clear()
    assert probe in out, (
        "an exemption naming a file the audit never scans went unmentioned, "
        "so a rule that protects nothing looks exactly like one that "
        "does:\n" + out)


# ---------------------------------------------------------------------------
# Quarantined data: two definitions of "quarantined", one of them load-bearing.
# ---------------------------------------------------------------------------
def test_a_quarantine_the_analyses_do_not_know_about_is_reported():
    r"""Directory layout is not what keeps superseded records out.

    Two modules walk the data tree RECURSIVELY -- build_paper_v3.py and
    analyze_corpus_size.py -- and then drop anything whose path contains one
    of a hand-typed set of directory names. That set is the protection. The
    integrity audit, meanwhile, recognises a quarantine directory by a
    different rule: a leading underscore, or "SUPERSEDED" in the name. The
    two agreed only by coincidence of naming.

    So a directory quarantined under any other name is reported safe by the
    audit while the paper builder walks straight into it, and superseded
    records reach a corpus count with an audit saying that cannot happen.
    This creates exactly that directory and asks.
    """
    import shutil

    import analyze_corpus_size as acs
    import audit_data_integrity as adi

    if not adi.DATA.is_dir():
        pytest.skip("no data tree on this checkout")
    live = next((d for d in sorted(adi.DATA.iterdir())
                 if d.is_dir() and not d.name.startswith("_")), None)
    if live is None:
        pytest.skip("no live data folder to plant a probe in")

    name = "_probe_withdrawn"
    assert name not in acs.QUAR, "pick a name the analyses do not know"
    probe = live / name
    probe.mkdir(parents=True, exist_ok=False)
    (probe / "probe.jsonl").write_text("{}\n", encoding="utf-8")
    before_f, before_n = list(adi.FAILURES), list(adi.NOTES)
    try:
        adi.FAILURES.clear()
        adi.check_superseded_excluded()
        failures = list(adi.FAILURES)
    finally:
        shutil.rmtree(probe)
        adi.FAILURES[:] = before_f
        adi.NOTES[:] = before_n
    assert not probe.exists(), "the probe directory was not removed"

    assert any(name in f for f in failures), (
        "a directory this audit calls quarantined, but which the analyses' "
        "filter does not name, was not reported -- so it would be walked "
        "into by the recursive globs in the paper builder:\n  "
        + "\n  ".join(failures))
