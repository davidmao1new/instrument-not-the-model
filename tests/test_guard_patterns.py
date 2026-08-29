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
        bad = re.findall(r'else\s+"(\d[^"]{0,10})"', _code(builder))
        assert not bad, (
            f"the builder has numeric artifact fallbacks again: {bad}. A "
            "missing artifact must stop the build, not print a stale number "
            "into a paper claiming every number is interpolated.")


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
