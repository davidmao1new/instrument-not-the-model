r"""Build the condensed variation of the paper: claims without the biography.

WHAT THIS IS. The full paper narrates its own corrections -- "an earlier draft
said X", "found by an adversarial audit of this paper, not by us", "we report
it here rather than only in the changelog". That record is why the paper is
trustworthy and it is also, at ~6% of the prose, a tax on every reader who was
not present for the argument. This build drops the narration and keeps the
claims, and -- stage two -- replaces sections 1, 2, 9, 10 and 11 wholesale
with the compressed versions in condensed_sections.py (the COMPRESS.md briefs
executed; ~1,500 words where the full paper spends ~8,800), producing a
separate PDF beside the full one:

    paper-a/figures/paper_instrument_validity_v3_condensed.pdf

The FULL version remains the canonical artifact; nothing here modifies it.

HOW IT CUTS. Every paragraph passes through a sentence filter before
typesetting:

  DROPPED   a sentence matching the correction-narration markers
            (audit_correction_narration.MARKERS -- the measured category) or
            the meta-commentary patterns below, PROVIDED it carries no digit.
  KEPT+LOG  a narration sentence that carries a digit: dropping it would drop
            a measurement, so it stays and is logged for a manual rewrite.
  KEPT      everything else, untouched.

Every cut and every kept-but-flagged sentence is written to
paper-a/docs/CONDENSED_CUTS.md, so the condensation is reviewable line by
line rather than trusted.

WHAT THIS IS FOR, AND NOT FOR. It is the working draft for the ICLR/arXiv
length cut, where LLM assistance is disclosed and permitted. It is NOT FAccT
prose and never will be: FAccT prohibits LLM-generated text, and a machine-cut
paragraph is machine text. It is also a first pass, not a finished paper --
dropping a sentence can orphan a connective in the next one, and the cut log
exists precisely so the author can walk the seams.

    sh paper-a/src/_py.sh paper-a/src/build_paper_condensed.py
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

import audit_correction_narration as narr  # noqa: E402
import build_paper_v3 as B  # noqa: E402
import condensed_sections as cs  # noqa: E402
import paperkit as pk  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Meta-commentary beyond the measured narration category: sentences about the
# paper's own process rather than its findings. Same rule -- no digit, no stay.
META = [
    r"\bworth saying plainly\b",
    r"\ba reader found it\b",
    r"\bwe report it here rather than\b",
    r"\bwhich is how (?:it|this|that) was discovered\b",
    r"\bit is stated because\b",
    r"\bthe changelog (?:carries|records)\b",
    r"\bthis (?:paper|file|section) (?:previously|first|originally)\b",
    r"\bwhich (?:an|the) (?:audit|critique) (?:of this paper )?(?:caught|found)\b",
]
META_RX = [re.compile(p, re.I) for p in META]
NARR_RX = [(why, re.compile(p, re.I)) for why, p in narr.MARKERS]

# Split on sentence ends followed by a capital, an opening markup tag, or a
# quote -- the builder's strings carry <b>/<i> markup that a naive splitter
# would sever mid-tag.
SENT = re.compile(r"(?<=[.!?])\s+(?=(?:<[bi]>)?[A-Z“(])")

CUTS: list[tuple[str, str]] = []       # (reason, sentence)
FLAGGED: list[tuple[str, str]] = []    # narration that carries a measurement
REWRITTEN: list[tuple[str, str]] = []  # (before, after)

# FINDINGS THAT LOOK LIKE NARRATION. The markers are written for prose about
# the paper's own history, and two of the paper's actual results collide with
# them: "the conversion factor ... is wrong by 2.0x to 484x" is a measurement,
# and "2 survive the correction" is the resampling result. Sentences matching
# these stay untouched and unflagged.
KEEP_VERBATIM = [
    re.compile(r"\bwrong by [\d.]+", re.I),
    re.compile(r"\bsurvives? the correction\b", re.I),
]

# CLAUSE-LEVEL REWRITES: narration removed from inside a sentence whose claim
# (and numbers) stay. Numbers are never written here -- each pattern deletes
# or reshapes the story around whatever values the build interpolated, so a
# changed artifact changes the sentence and the rewrite still applies. Applied
# before classification; a sentence they clean no longer matches the markers.
CLAUSE_SUBS = [
    # "That exception is what makes D5 ... , and an earlier version of this
    # sentence generalised past it."
    (re.compile(r",? and an earlier version of this sentence generalised "
                r"past it"), ""),
    # "...the spread this paper previously quoted as a bare N percentage
    # points in fact has an interval of [a, b] and is not a number we may
    # print."
    (re.compile(r"the spread this paper previously quoted as a bare "
                r"[\d.]+ percentage points in fact has"), "the spread has"),
    (re.compile(r" and is not a number we may print"),
     ", so no point value is printed"),
    # "The claim does not survive ... , and this is the second form of it we
    # have had to withdraw."
    (re.compile(r",? and this is the second form of it we have had to "
                r"withdraw"), ""),
    # "Having lost the ratio to its denominator, we fell back on the
    # between-wording standard deviation alone, which moves ..."
    (re.compile(r"Having lost the ratio to its denominator, we fell back on "
                r"the between-wording standard deviation alone, which moves"),
     "The between-wording standard deviation alone moves"),
    (re.compile(r",? and we reported that every interval excluded zero"), ""),
    # "... the claim that X is withdrawn." -> state the non-claim.
    (re.compile(r",? and the claim that wording dispersion is a property of "
                r"the model-and-job <i>pair</i> rather than of the model is "
                r"withdrawn"),
     ", so no claim is made that the dispersion varies by job"),
    # "both panels get the same SUBTRACTION, which an earlier draft of this
    # section did not do." (the caps arrive as <i> markup on each word)
    (re.compile(r",? which an earlier draft of this section did not do"), ""),
    # "The two-template range in that sentence is interpolated now and was
    # not before. carried in prose ... an audit of this paper found that no
    # artifact reproduced it: recomputed on ..."
    (re.compile(r"The two-template range in that sentence is interpolated "
                r"now and was not before\.\s*carried in prose from a "
                r"measurement made before the third template existed, and an "
                r"audit of this paper found that no artifact reproduced it: "
                r"recomputed on the panel it described,"),
     "Recomputed on the two-template panel,"),
    # "What §4.5 supports is narrower than the claim an earlier draft made
    # from it, and is still sufficient for the argument it is used in: ..."
    (re.compile(r"What (§[\d.]+) supports is narrower than the claim an "
                r"earlier draft made from it, and is still sufficient for "
                r"the argument it is used in:"),
     r"\1 supports this much:"),
]


def _classify(s: str):
    plain = re.sub(r"<[^>]+>", "", s)
    for rx in KEEP_VERBATIM:
        if rx.search(plain):
            return None
    for why, rx in NARR_RX:
        if rx.search(plain):
            return why
    for rx in META_RX:
        if rx.search(plain):
            return "meta-commentary"
    return None


_orig_para = pk.Paper.para
_orig_heading = pk.Paper.heading

# Section replacement state. E and ART are filled by the first (plain) build;
# SUPPRESS drops the full paper's prose while a compressed section is active.
E: dict = {}
ART: dict = {}
STATE = {"suppress": False, "emitted": set()}
SECTION_WORDS: dict = {}


def condensed_heading(self, text, level=1, span2=False):
    if level == 1:
        sec = text.split()[0]
        if sec in cs.SECTIONS and sec not in STATE["emitted"]:
            STATE["emitted"].add(sec)
            STATE["suppress"] = True
            _orig_heading(self, text, level, span2)
            words = 0
            for i, para in enumerate(cs.SECTIONS[sec](E, ART)):
                words += len(para.split())
                _orig_para(self, para, indent=(i > 0))
            SECTION_WORDS[sec] = words
            budget = cs.BUDGET[sec]
            assert words <= budget * 1.15, (
                f"compressed section {sec} is {words} words against a "
                f"budget of {budget}; tighten condensed_sections.py")
            return
        STATE["suppress"] = False
        return _orig_heading(self, text, level, span2)
    if STATE["suppress"]:
        return                      # level-2 headings of a replaced section
    return _orig_heading(self, text, level, span2)


# PURE HISTORY CARRYING DIGITS. The digit rule keeps any narration sentence
# with a number in it, because dropping a number is dropping a measurement.
# These three carry numbers that describe the EARLIER DRAFT (how many tables
# it used, how many wordings it claimed), not anything the paper measures, so
# the digit rule spares exactly the wrong sentences. Full drops.
HISTORY_DROPS = [
    re.compile(r"An earlier draft did; §[\d.]+ reports the withdrawal"),
    re.compile(r"An earlier version of this paper used \d+ of the \d+ and "
               r"said the other two had not extracted cleanly"),
    re.compile(r"An earlier version of this paper said the second used "
               r"\d+ wordings"),
]


def condensed_para(self, text, *args, **kwargs):
    if STATE["suppress"]:
        CUTS.append(("replaced section", text[:80] + "\u2026"))
        return
    parts = SENT.split(text)
    kept = []
    for s in parts:
        if any(rx.search(re.sub(r"<[^>]+>", "", s)) for rx in HISTORY_DROPS):
            CUTS.append(("history with historical digits", s))
            continue
        cleaned = s
        for rx, rep in CLAUSE_SUBS:
            cleaned = rx.sub(rep, cleaned)
        if cleaned != s:
            REWRITTEN.append((s, cleaned))
            s = cleaned
        why = _classify(s)
        if why is None:
            kept.append(s)
        elif re.search(r"\d", s):
            FLAGGED.append((why, s))
            kept.append(s)
        else:
            CUTS.append((why, s))
    if not kept:
        return                      # the whole paragraph was narration
    _orig_para(self, " ".join(kept), *args, **kwargs)


def main() -> int:
    # PASS 1: a plain build IN A SUBPROCESS, to fill EXPORT for the
    # compressed sections. In-process, the builder's module-global table and
    # appendix registries carried state into pass 2: the appendices shipped
    # twice, and the full PDF's table numbering was left mid-renumber. A
    # subprocess gives the full paper a clean build every time and hands
    # EXPORT over through a file.
    import json as _json
    import subprocess as _sp
    import tempfile as _tf
    _ef = pathlib.Path(_tf.mkdtemp()) / "export.json"
    _code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(ROOT / 'paper-a/src')!r})\n"
        "import build_paper_v3 as B\n"
        "B.main()\n"
        "json.dump({k: v for k, v in B.EXPORT.items()\n"
        "           if isinstance(v, (int, float, str))},\n"
        f"          open({str(_ef)!r}, 'w'))\n")
    _r = _sp.run([sys.executable, "-c", _code], cwd=str(ROOT),
                 capture_output=True, text=True, encoding="utf-8",
                 errors="replace")
    if _r.returncode != 0:
        sys.exit("pass-1 full build failed:\n" + _r.stdout + _r.stderr)
    E.update(_json.loads(_ef.read_text(encoding="utf-8")))
    ART.update(srn=B.load("srn"), mstruct=B.load("mstruct"))

    # PASS 2: the instrumented build. Two shared-state hazards are fenced
    # off first. The renumber mechanism persists the measured table order to
    # a cache the FULL build also loads; letting the condensed layout write
    # it poisons the next full build (the suite caught this as a table-order
    # failure that a standalone rebuild fixed and the next condensed build
    # re-broke). The cache is redirected to a scratch file, and the
    # renumber retry (which re-executes the FULL build script) is disabled
    # for this pass -- a working draft may number its tables in page order
    # on the second regeneration rather than re-exec.
    import os as _os
    B.TABLE_ORDER_CACHE = _ef.parent / "table_order_condensed.json"
    _old_flag = _os.environ.get("PAPER_RENUMBER_PASS")
    _os.environ["PAPER_RENUMBER_PASS"] = "1"
    # B.APP is a module-global appendix
    # accumulator built for one main() per process; without a reset, pass
    # 1's stashed appendix blocks are emitted AGAIN in pass 2 and the
    # condensed PDF ships every appendix twice (7,807 words became 15,357
    # before this line existed).
    B.APP.__init__()
    B.OUT = B.FIGS / "paper_instrument_validity_v3_condensed.pdf"
    pk.Paper.para = condensed_para
    pk.Paper.heading = condensed_heading
    try:
        B.main()
    finally:
        pk.Paper.para = _orig_para
        pk.Paper.heading = _orig_heading
        if _old_flag is None:
            _os.environ.pop("PAPER_RENUMBER_PASS", None)
        else:
            _os.environ["PAPER_RENUMBER_PASS"] = _old_flag

    dropped_words = sum(len(s.split()) for _, s in CUTS)
    rep = [
        "# What the condensed variation cut",
        "",
        "Generated by `paper-a/src/build_paper_condensed.py`. Regenerate "
        "rather than edit.",
        "",
        "**This is the review ledger for the machine cut.** The condensed PDF "
        "is a working draft for the ICLR/arXiv length reduction (disclosed "
        "LLM assistance; never FAccT prose). Walk the seams: a dropped "
        "sentence can orphan a connective in its neighbour.",
        "",
        f"**{len(CUTS)} sentences dropped ({dropped_words:,} words); "
        f"{len(FLAGGED)} narration sentences kept because they carry a "
        "measurement** and need a manual rewrite that keeps the number and "
        "drops the story.",
        "",
        "## Dropped",
        "",
    ]
    for why, s in CUTS:
        rep.append(f"- *[{why}]* {s}")
    rep += ["", "## Rewritten (narration removed from inside the sentence, "
            "numbers untouched)", ""]
    for before, after in REWRITTEN:
        rep.append(f"- BEFORE: {before}")
        rep.append(f"  AFTER:  {after}")
    rep += ["", "## Kept but flagged (measurement inside narration)", ""]
    for why, s in FLAGGED:
        rep.append(f"- *[{why}]* {s}")
    out = ROOT / "paper-a" / "docs" / "CONDENSED_CUTS.md"
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")

    for sec in sorted(SECTION_WORDS, key=int):
        print(f"  \u00a7{sec}: {SECTION_WORDS[sec]} words "
              f"(budget {cs.BUDGET[sec]})")
    print(f"\n  condensed: {B.OUT.relative_to(ROOT)}")
    print(f"  cut {len(CUTS)} sentences ({dropped_words:,} words); "
          f"{len(FLAGGED)} flagged for manual rewrite")
    print(f"  ledger: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
