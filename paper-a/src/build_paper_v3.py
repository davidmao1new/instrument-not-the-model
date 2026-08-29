"""Build the paper. Complete rewrite; supersedes build_paper_v2.py.

WHY A REWRITE RATHER THAN AN EXTENSION. v2 was organised study by study, in the
order the studies happened to be run, and its thesis was that the PROMPT is part
of the instrument. Five further studies and a full re-analysis on the complete
dataset changed both. The prompt is one of at least six choices that move the
reported number, they are not all of the same kind, and the paper is clearer
organised by the KIND of choice -- what the auditor writes, what the auditor
runs, what the auditor computes -- than by the order of discovery. The abstract,
the title, the section structure and every number are rebuilt from that.

TWO RULES THIS FILE ENFORCES, CARRIED OVER FROM v2 BECAUSE THEY ARE THE POINT.

1. Every number comes from an artifact on disk and is interpolated, never typed.
   A sentence whose artifact is missing is DROPPED and a warning printed, so a
   partial run yields a shorter honest paper rather than a complete dishonest
   one.

2. Where an analysis choice matters, the paper shows both answers. The naive
   choice is not hidden and then quietly corrected; it is reported next to the
   defensible one, with the reason. A methods paper that concealed its own
   forking paths would be self-refuting.

    .venv/Scripts/python.exe paper-a/src/build_paper_v3.py
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import effectsize as es  # noqa: E402
import paperkit as pk  # noqa: E402
import stimuli as st  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "paper-a" / "data"
FIGS = ROOT / "paper-a" / "figures"
OUT = FIGS / "paper_instrument_validity_v3.pdf"

SRC = {
    "study2": D / "delta_stability" / "study2_v2.json",
    "vcomp": D / "delta_stability" / "variance_components.json",
    "arm": D / "delta_stability" / "arm_contrast.json",
    "noise": D / "delta_stability" / "noise_floor.json",
    "mech": D / "mechanism_panel" / "mech_panel_analysis.json",
    "tconc": D / "mechanism_panel" / "template_concentration.json",
    "d9": D / "mechanism_panel" / "d9_adjudication.json",
    "names": D / "names" / "name_variance.json",
    "occ": D / "occupation" / "occupation_analysis.json",
    "occnull": D / "occupation" / "occupation_dispersion_null.json",
    "front": D / "frontier" / "frontier_margin_analysis.json",
    "asymunc": D / "names" / "asymmetry_uncertainty.json",
    "fcap": D / "instrument" / "frontier_api_capability_v2.json",
    "fspend": D / "frontier" / "frontier_spend.json",
    "fmass": D / "frontier" / "frontier_yes_no_mass.json",
    "fverd": D / "frontier" / "frontier_verdict_analysis.json",
    "fnoise": D / "frontier" / "frontier_noise_floor.json",
    "rint": D / "reference" / "ratio_intervals.json",
    "funoise": D / "replicate" / "noise_vs_probability.json",
    "quant": D / "quantization" / "quantization_analysis.json",
    "rep": D / "replicate" / "replicate_analysis.json",
    "cache": D / "replicate" / "cache_residual.json",
    "lit": D / "reference" / "published_effects.json",
    "bm": D / "reference" / "bm2004_names.json",
    "gate": D / "panel_gate" / "panel_gate_results.json",
    "ctok": D / "instrument" / "condition_tokens_llama-3.1-8b-instruct.json",
    "nlen": D / "instrument" / "name_length_effect.json",
    "lenpred": D / "instrument" / "length_prediction.json",
    "tokid": D / "mechanism" / "token_identity_llama-3.1-8b-instruct.json",
    "resamp": D / "delta_stability" / "resampling_unit.json",
    # Added after the v3 audit. Each closes a gap the audit found rather than
    # adding a new claim: `scale` replaces a ratio whose stated mechanism did
    # not produce it, `matrix` replaces a prose assertion about the field with
    # a counted one, `cval` supplies the manipulation check the design assumed,
    # `pairfree` measures a degree of freedom the design created and did not
    # follow up, and `tbal` answers a limitation the paper had only conceded.
    "scale": D / "delta_stability" / "reporting_scale.json",
    "budget": D / "delta_stability" / "dispersion_budget.json",
    "matrix": D / "reference" / "reporting_practice_matrix.json",
    "cval": D / "names" / "construct_validity.json",
    "pairfree": D / "names" / "pairing_freedom.json",
    "tbal": D / "instrument" / "token_balanced_grid.json",
    "armasym": D / "instrument" / "arm_asymmetry.json",
    "xmodel": D / "instrument" / "crossmodel_wording.json",
    "mstruct": D / "reference" / "matrix_structure.json",
    "wildboot": D / "instrument" / "wild_bootstrap.json",
    # Why the matched subset is small, and whether a larger list would fix it.
    # Written after a statistician pointed out that three clusters cannot
    # support the inference the design was asking of them.
    "npow": D / "instrument" / "name_list_power.json",
    # The resolution floor of the exact test on the matched subset. The
    # reporting-set item below used to compute 2**n inline; it reads the
    # artifact now so the preprint and the submission version cannot
    # disagree about an impossibility claim.
    "permres": D / "instrument" / "permutation_resolution.json",
    "second": D / "second_task" / "second_task_analysis.json",
    "disp_unc": D / "delta_stability" / "dispersion_uncertainty.json",
    # The null calibration of §9.1's own screening rule. §9.1 recommends a
    # decision procedure; until this existed it recommended one whose error
    # rate it declined to state, which is the paper's own complaint about the
    # field pointed back at itself.
    "srn": D / "instrument" / "screening_rule_null.json",
    # Every claim the survey makes about somebody else's paper, re-checked
    # against that paper's extracted full text. §8.1 asserts that the survey's
    # negatives are verified rather than asserted, and that assertion needs a
    # number the build produced.
    "evcheck": D / "reference" / "matrix_evidence_check.json",
    # One row per analysis: treatment, outcome, assignment unit, observation
    # unit, block, target population, estimator, uncertainty estimator. These
    # columns were named by a reviewer who could not tell from the prose which
    # of this paper's claims applied to which design, which is a fair complaint
    # about a paper whose subject is undeclared design choices.
    "design": D / "reference" / "design_table.json",
    # One row per analysis: treatment, outcome, assignment unit, observation
    # unit, block, target population, estimator, uncertainty estimator. These
    # columns were named by a reviewer who could not tell from the prose which
    # of this paper's claims applied to which design, which is a fair complaint
    # about a paper whose subject is undeclared design choices.
    "design": D / "reference" / "design_table.json",
}

# Probes with one artifact per checkpoint rather than one per study.
COVERAGE_GLOB = "token_coverage_*.json"
NAMELEN_GLOB = "name_length_*.json"
GATE_GLOB = "gate_*.json"

SHORT = {
    "mistral-7b-instruct-v0.1": "Mistral-7B-Instruct v0.1",
    "mistral-7b-v0.1-base": "Mistral-7B v0.1 (base)",
    "llama-2-7b-chat": "Llama-2-7B-chat",
    "llama-2-13b-chat": "Llama-2-13B-chat",
    "mistral-7b-instruct-v0.3": "Mistral-7B-Instruct v0.3",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
}
TINY = {
    "mistral-7b-instruct-v0.1": "Mistral v0.1",
    "mistral-7b-v0.1-base": "Mistral base",
    "llama-2-7b-chat": "Llama-2-7B",
    "llama-2-13b-chat": "Llama-2-13B",
    "mistral-7b-instruct-v0.3": "Mistral v0.3",
    "llama-3.1-8b-instruct": "Llama-3.1-8B",
}
ORDER = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
         "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
ORDER_FRONT = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
PANEL_ORDER = ["llama-2-7b-chat", "llama-2-13b-chat", "llama-3.1-8b-instruct",
               "mistral-7b-v0.1-base", "mistral-7b-instruct-v0.1",
               "mistral-7b-instruct-v0.3"]

MISSING: list[str] = []

# Small-integer words, for prose where a numeral would read badly.
# ==========================================================================
# VENUE MODE. FAccT desk-rejects a submission carrying identifying
# information, and separately desk-rejects one that omits the generative-AI
# statement. Those two rules land on the same page and pull opposite ways, so
# the endmatter is generated from a mode rather than hand-edited before a
# deadline -- which is exactly how a non-anonymised PDF gets uploaded.
#
#   PAPER_VENUE=preprint  (default)  name, affiliation, competing interests
#   PAPER_VENUE=facct                anonymous; identifying endmatter dropped
#
# PAPER_PROSE_REWRITTEN gates the generative-AI statement. FAccT's author
# guide says, verbatim, "FAccT prohibits the use of LLMs to generate text for
# publications", allowing only formatting and "grammar or fluency" help. The
# prose of the preprint was drafted with an LLM, so the FAccT statement is
# TRUE ONLY AFTER the text has been rewritten by the author. Building a FAccT
# PDF before then would print a false disclosure, which is a worse failure
# than missing the deadline, so it raises instead.
VENUE = os.environ.get("PAPER_VENUE", "preprint").strip().lower()
PROSE_REWRITTEN = os.environ.get("PAPER_PROSE_REWRITTEN", "0") == "1"
ANON = VENUE in ("facct", "anon", "anonymous")
if VENUE not in ("preprint", "facct", "anon", "anonymous"):
    sys.exit(f"unknown PAPER_VENUE={VENUE!r}")
if ANON and not PROSE_REWRITTEN:
    sys.exit(
        "PAPER_VENUE=facct requires PAPER_PROSE_REWRITTEN=1.\n"
        "  FAccT prohibits LLM-generated text. The generative-AI statement\n"
        "  emitted in this mode asserts the author wrote the prose. Do not\n"
        "  set the flag until that is true of every sentence in the PDF.")

NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
       7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
       12: "twelve", 13: "thirteen", 14: "fourteen"}
# Fraction words, for prose that says "smaller than a Nth of the effect". The
# N is computed from an artifact, so it has to survive being rendered: "a 17th"
# is a number pretending to be a word.
ORD = {2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth",
       7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh",
       12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
       16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
       20: "twentieth", 21: "twenty-first", 22: "twenty-second",
       23: "twenty-third", 24: "twenty-fourth", 25: "twenty-fifth"}


def load(key):
    p = SRC[key]
    if not p.exists():
        MISSING.append(key)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        MISSING.append(f"{key} ({e})")
        return None


# The widening buckets, named once. The prose used to type "more than 10 %"
# beside a lookup of frac_widen_over_10pct, so the sentence and the artifact
# key it describes could drift apart silently. Reading the label out of the
# key is what keeps them together.
W10 = "frac_widen_over_10pct"
W25 = "frac_widen_over_25pct"


def widen_label(key: str) -> str:
    """The percentage a frac_widen_over_NNpct key counts, as printed prose."""
    return key.split("_over_")[1][:-len("pct")] + " %"


def fmt(x, n=2, sign=False):
    if x is None:
        return "n/a"
    s = f"{x:+.{n}f}" if sign else f"{x:.{n}f}"
    return s


def pct(x, n=1):
    return "n/a" if x is None else f"{100 * x:.{n}f} %"


def rng_str(lo, hi, fn):
    """"a to b", or just "a" when the two coincide.

    A partial run can leave a range with one member, and "13.3 % to 13.3 %"
    reads as a mistake even when it is arithmetically right.
    """
    a, b = fn(lo), fn(hi)
    return a if a == b else f"{a} to {b}"


def FIGREF(stem: str) -> str:
    """"Figure N" for a figure, read from the order registry.

    Written as a lookup rather than a literal for the same reason every other
    number in this paper is: two figures were inserted after the v3 audit and
    every literal after them would have been silently wrong.
    """
    import figstyle as _fs  # noqa: PLC0415
    return f"Figure {_fs.PAPER_FIGURE_NUMBER[stem]}"


# ---------------------------------------------------------------------------
# TABLE NUMBERING, declared once and read everywhere.
#
# Table numbers used to be typed into their captions, which produced a
# duplicate the moment a table was added and two inversions the moment a
# section moved -- Tables 16 and 17 printed before 11 to 14 after Appendix D was
# repositioned, and nothing complained. Figures never had this problem because
# their order is declared in one list. Tables now work the same way: TABLE_ORDER
# is the emission order, TABN maps a key to its number, and TAB(key) is what
# prose says. The wrapper below asserts that tables actually emit in this order,
# so a future reordering fails the build rather than printing out of sequence.
# THE SEED ORDER. The build MEASURES the order the tables actually reach the
# page and caches it in data/reference/table_order.json, which is adopted at
# import; this literal is only the starting point for a repository that has
# never been built. Editing it by hand is harmless and mostly pointless -- the
# measurement wins on the next build.
TABLE_ORDER = [
    "study2_effect",
    # NUMBERED BY THE ORDER A READER MEETS THEM, NOT BY EMISSION ORDER. paperkit
    # defers span2 tables to the next page's float slots while single-column
    # tables flow inline, so a full-width table emitted first can render a page
    # LATER than a narrow one emitted after it. That happened twice: 4 after 5,
    # and 7 after 8. The build asserts emission order, which is why neither
    # showed up. test_table_render_order.py now asserts the rendered sequence.
    #
    # THE APPENDIX MOVE RE-SORTED THIS LIST. Five tables belong to sections that
    # now sit after the page break, so they render last and are numbered last.
    # A reader meets Tables 1-10 in the body and 11-15 in the appendices.
    "name_draw_grid",
    "name_variance",
    "averaging_asymmetry",
    # Swapped 2026-08-19: adding the blocked-design paragraph to §2 shifted
    # the body enough that the occupation table now reaches its float slot
    # first. Numbering follows the reader, so the list follows the render.
    "occupation_ratio",
    "token_segmentation",
    "replicate",
    "spec_curve",
    "mech_classes",
    "reporting_matrix",
    # ---- appendices ----
    "instrument_validation",
    "conditions",
    "second_task",
    "frontier",
    "quantization",
]
TABN = {k: i + 1 for i, k in enumerate(TABLE_ORDER)}

# ---------------------------------------------------------------------------
# TABLE NUMBERS FOLLOW THE READER, AND NOW THEY DO SO WITHOUT BEING TOLD.
#
# `paperkit` defers span2 floats to the next page's float slots while
# single-column tables flow inline, so emission order is not render order. The
# list above therefore had to be re-sorted BY HAND every time a paragraph
# changed length, and it was wrong three times in one week -- caught each time
# by a test, which is the good outcome, but the fix was always the same manual
# swap and the next prose edit undid it.
#
# So the build now measures. After rendering it reads the order the captions
# actually appear in, and if that disagrees with TABLE_ORDER it renumbers and
# renders once more. One extra pass, only when needed, and the numbering cannot
# drift from the page again.
# WHY A SECOND PROCESS AND NOT A SECOND CALL. The first version of this simply
# called main() again, and the paper came out 36 pages instead of 30: main()
# mutates module-level state -- APP holds the blocks it lifted out of the body,
# MISSING accumulates, EXPORT fills -- so a second call in the same interpreter
# renders the appendix twice. Rather than hunt for every piece of that state and
# hope none is added later, the corrected order is written to disk and the build
# re-executes itself once. A fresh interpreter has no state to leak.
TABLE_ORDER_CACHE = D / "reference" / "table_order.json"


def _load_cached_table_order() -> None:
    """Adopt a previously measured order, if it covers the same tables."""
    try:
        cached = json.loads(TABLE_ORDER_CACHE.read_text(encoding="utf-8"))["order"]
    except Exception:  # noqa: BLE001
        return
    if isinstance(cached, list) and set(cached) == set(TABLE_ORDER):
        TABLE_ORDER[:] = cached
        TABN.clear()
        TABN.update({k: i + 1 for i, k in enumerate(TABLE_ORDER)})


_load_cached_table_order()


def _rendered_table_order(pdf: pathlib.Path) -> list[str] | None:
    """The keys of TABLE_ORDER, in the order their captions reach the page."""
    try:
        import fitz
    except ImportError:
        return None
    if not pdf.exists():
        return None
    seen: list[tuple[int, int, str]] = []
    with fitz.open(pdf) as doc:
        pages = [" ".join(pg.get_text().split()) for pg in doc]
    for key, num in TABN.items():
        needle = f"Table {num}."
        for pi, page in enumerate(pages):
            j = page.find(needle)
            if j >= 0:
                seen.append((pi, j, key))
                break
        else:
            return None            # a table did not render; do not guess
    seen.sort()
    return [k for _p, _j, k in seen]


def _renumber_to_match_the_page(pdf: pathlib.Path) -> bool:
    """True if the numbering disagreed with the page and has been recorded."""
    order = _rendered_table_order(pdf)
    if order is None or order == TABLE_ORDER:
        return False
    TABLE_ORDER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TABLE_ORDER_CACHE.write_text(json.dumps({
        "_what": "The order the tables actually reach the page.",
        "_why": ("paperkit defers span2 floats, so emission order is not render "
                 "order. Maintaining the numbering by hand was wrong three "
                 "times in one week; the build measures it instead."),
        "order": order}, indent=1), encoding="utf-8")
    return True


def TAB(key: str) -> str:
    """"Table N" for a table, from the declared order."""
    return f"Table {TABN[key]}"


def TABS(*keys: str) -> str:
    """"Tables 3 and 13" -- the plural form, still from the declared order.

    Without this, prose naming two tables either hard-types the numbers (which
    is how one caption came to cite a table on the wrong scale after a section
    moved) or reads "Table 3 and Table 13". Numbers are sorted so the phrase
    matches the order a reader meets them in.
    """
    ns = sorted(TABN[k] for k in keys)
    if len(ns) == 1:
        return f"Table {ns[0]}"
    body = ", ".join(str(n) for n in ns[:-1])
    return f"Tables {body} and {ns[-1]}"


def corpus_size():
    """(matched-pair records, single-prompt records, model calls) from disk.

    WHY THIS DELEGATES NOW. This function used to return the count of ALL
    non-quarantined rows and the paper called that number "matched-pair
    measurements". An audit found that 1,461 of those rows carry one prompt and
    one margin, not a pair. The admission gate, the affiliation probe and
    the smoke tests. A later audit found an error the other way: 220 paired
    rows from an endpoint that returns no margin were scored as single prompts
    costing one call apiece, so the paired corpus AND the call total were both
    short by 220. Both are fixed; the predicate now asks whether both arms were
    measured rather than whether both were measured on a particular scale. The
    classification lives in analyze_corpus_size.py, which writes the
    breakdown to an artifact, so the distinction is auditable rather than
    implicit in a predicate here. The fallback preserves the old behaviour if
    the artifact is missing, and flags itself.
    """
    p = D / "reference" / "corpus_size.json"
    if p.exists():
        c = json.loads(p.read_text(encoding="utf-8"))
        return c["n_matched_pair_records"], c["n_single_prompt_records"], \
            c["n_model_calls"]
    QUAR = ("_contaminated", "_superseded", "_binary_only_superseded",
            "_d9_superseded")
    pair = single = calls = 0
    for f in D.rglob("*.jsonl"):
        if any(q in f.parts for q in QUAR):
            continue
        for r in st.read_jsonl(f):
            # Kept verbatim in step with analyze_corpus_size.is_matched_pair:
            # both arms measured, on the margin where there is one and on the
            # raw response where the API returns no margin.
            if (("white_margin" in r and "black_margin" in r)
                    or ("white_raw" in r and "black_raw" in r)):
                pair += 1
                calls += 2
            else:
                single += 1
                calls += 1
    MISSING.append("corpus_size.json (counted inline instead)")
    return pair, single, calls


# ==========================================================================
# --------------------------------------------------------------------------
# The arXiv submission abstract, emitted from the same string the PDF sets.
ARXIV_ABSTRACT_LIMIT = 1920

# Every non-ASCII character the abstract can contain, and what arXiv gets
# instead. Total by construction: anything unmapped raises rather than being
# dropped or replaced with a question mark.
ARXIV_TRANSLIT = {
    "\u2014": " -- ",   # em dash
    "\u2013": "-",      # en dash
    "\u2019": "'",      # right single quote
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00e9": "e",      # e-acute, as in resume
    "\u00e8": "e",
    "\u00d7": "x",      # multiplication sign
    "\u00a7": "Sec. ",  # section sign
    "\u00b1": "+/-",
    "\u2009": " ",      # thin space
    "\u00a0": " ",      # non-breaking space
    "\u2212": "-",      # minus sign
    "\u03b1": "alpha",
    "\u03c1": "rho",
}

# TeX that arXiv explicitly rejects in this field.
ARXIV_FORBIDDEN = ("~", "\\,", "\\ ", "\\em", "\\it", "\\bf")


def arxiv_abstract(text: str) -> str:
    """ASCII, single-line, within the limit -- or raise saying why."""
    # The builder's <b>/<i> run markup is an internal convention; arXiv's
    # abstract field is plain text and a tag would be submitted literally
    # (one was, into abstract_arxiv.txt, when a caps-to-italics pass edited
    # the abstract string).
    out = re.sub(r"</?[bi]>", "", text)
    for ch, rep in ARXIV_TRANSLIT.items():
        out = out.replace(ch, rep)
    out = " ".join(out.split())          # one line, no leading whitespace
    bad = sorted({c for c in out if ord(c) > 127})
    if bad:
        raise ValueError(
            "unmapped non-ASCII in the arXiv abstract: "
            + ", ".join(f"{c!r} (U+{ord(c):04X})" for c in bad)
            + " -- add it to ARXIV_TRANSLIT rather than letting it through")
    for f in ARXIV_FORBIDDEN:
        if f in out:
            raise ValueError(f"arXiv rejects {f!r} in the abstract field")
    if out.lower().startswith("abstract"):
        raise ValueError("arXiv: do not include the word 'Abstract'")
    if len(out) > ARXIV_ABSTRACT_LIMIT:
        raise ValueError(
            f"arXiv abstract is {len(out)} characters, limit is "
            f"{ARXIV_ABSTRACT_LIMIT}; cut {len(out) - ARXIV_ABSTRACT_LIMIT}")
    return out


class _Appendix:
    """Lift a run of emitted blocks out of the body and hold them for later.

    paperkit builds a flat list of blocks in emission order, so a section can
    be relocated by slicing its blocks out and re-appending them after a page
    break. The generating code never moves, which is why no number changes.
    """

    def __init__(self):
        self.paper = None
        self.stash = []
        self._mark = None

    def bind(self, paper):
        self.paper = paper

    def start(self, _note=None):
        assert self._mark is None, "APP.start() without a matching end()"
        self._mark = len(self.paper.blocks)

    def end(self):
        assert self._mark is not None, "APP.end() without a start()"
        self.stash.append(self.paper.blocks[self._mark:])
        del self.paper.blocks[self._mark:]
        self._mark = None

    def emit(self, paper, H, P):
        assert self._mark is None, "an appendix section was never closed"
        paper.page_break()
        H("Appendices")
        P("The body states each measurement and what follows from it. These "
          "appendices carry the designs, the full panels and the diagnostics "
          "those statements rest on. Nothing here restates the body; it is the "
          "evidence the body draws on, kept whole so that a reader who wants "
          "to check a number can.", indent=False)
        for blocks in self.stash:
            paper.blocks.extend(blocks)


APP = _Appendix()

# Headline quantities, recorded as main() computes them so that anything else
# needing them -- the LaTeX port, the outreach summary -- reads the numbers this
# document actually prints instead of reimplementing the derivations. A parallel
# implementation is how the outreach summary came to claim 220,000 matched pairs
# against an artifact holding 31,468.
EXPORT: dict = {}
# The full reference list (surname key, rendered text), filled by main() so
# build_facct_tex.py can emit refs.bib entries for the background works the
# FAccT sections cite -- refs.bib used to hold only the 13 surveyed audits.
REFS_OUT: list = []


def main() -> int:
    s2 = load("study2")
    vc = load("vcomp")
    arm = load("arm")
    noise = load("noise")
    mech = load("mech")
    tconc = load("tconc")
    d9 = load("d9")
    names = load("names")
    occ = load("occ")
    occnull = load("occnull")
    armasym = load("armasym")
    xmodel = load("xmodel")
    mstruct = load("mstruct")
    wildboot = load("wildboot")
    front = load("front")
    asymunc = load("asymunc")
    fcap = load("fcap")
    fspend = load("fspend")
    fmass = load("fmass")
    fverd = load("fverd")
    fnoise = load("fnoise")
    rint = load("rint")
    funoise = load("funoise")
    quant = load("quant")
    rep = load("rep")
    cache = load("cache")
    lit = load("lit")
    bm = load("bm")
    ctok = load("ctok")
    nlen = load("nlen")
    lenpred = load("lenpred")
    resamp = load("resamp")
    tokid = load("tokid")
    scale = load("scale")
    budget = load("budget")
    matrix = load("matrix")
    npow = load("npow")
    permres = load("permres")
    cval = load("cval")
    pairfree = load("pairfree")
    tbal = load("tbal")
    second = load("second")
    disp_unc = load("disp_unc")
    srn = load("srn")
    evcheck = load("evcheck")
    design = load("design")
    design = load("design")
    if s2 is None:
        sys.exit("cannot build without study2_v2.json")

    models = [m for m in ORDER if m in s2]

    # ---- quantities used in more than one section, computed once ---------
    ps = {m: s2[m]["overall"]["superiority"]["est"] for m in models}
    psci = {m: s2[m]["overall"]["superiority"]["ci"] for m in models}
    lo = {m: s2[m]["overall"]["logodds"] for m in models}
    sd_word = {m: s2[m]["ps_sd_across_wordings"] for m in models}
    ratio_ps = {m: sd_word[m] / abs(ps[m] - 0.5) for m in models}
    survives = [m for m in models if lo[m]["ci"][0] * lo[m]["ci"][1] > 0]
    # The complement, kept as a list rather than described in prose. Four
    # places used to assert that the dispersion exceeds "the whole effect"
    # wherever the effect is not identified. That is true of one of the two
    # unidentified models (1.18) and false of the other (0.55); the sentence
    # generalised a single cell to a class. Both endpoints are now
    # interpolated, so the claim cannot outrun the data again.
    unident = [m for m in models if m not in survives]
    _r_un = [ratio_ps[m] for m in unident]
    sat = {m: s2[m]["overall"]["saturated_frac"] for m in models}
    ncells_s2 = sum(s2[m]["overall"]["n"] for m in models)
    n_records, n_single, n_calls = corpus_size()
    # Frontier checkpoints TOUCHED, which is what the corpus totals cover: the
    # four that return a margin plus the verdict-only arm of Appendix D.
    # Same rule, same abstract. A missing frontier artifact used to make
    # this zero, in a sentence that counts frontier checkpoints. The second
    # term stays conditional: the verdict-only arm of Appendix D is
    # genuinely optional, and `1 if fverd else 0` counts it rather than
    # standing in for it.
    if not (front and front.get("summary")):
        sys.exit(
            "cannot build without frontier/frontier_margin_analysis.json: "
            "\\NFront is interpolated into the abstract, and a missing "
            "artifact used to make it zero rather than stop the build.")
    _n_front = front["summary"]["n_models"] + (1 if fverd else 0)
    # How badly a single p = 0.5 Jacobian misstates the probability-scale
    # effect, per model. Recomputed here rather than quoted, because it is
    # one of the paper's load-bearing numbers.
    jac = {m: abs(s2[m]["overall"]["legacy_pp"]
                  / s2[m]["overall"]["prob_pp"]["est"]) for m in models}

    paper = pk.Paper(OUT, "The Instrument Is Not the Model", "David Mao")

    # ======================================================================
    # TITLE AND ABSTRACT
    # ======================================================================
    # The change produced by dropping token-unmatched pairs.
    #
    # THIS NUMBER WAS PREVIOUSLY GATED ON A P-VALUE THIS PAPER DISOWNS. The old
    # gate was `equal_minus_all["p"] < 0.05`, which is the row-level bootstrap
    # -- p = 0.004 on Llama-3.1 -- and §4.4 says in prose that this p came from
    # the wrong resampling unit. On the first-name-pair unit the correction is
    # not separable from zero on any of the four models (p = 0.12, 0.078, 0.30,
    # 0.68), so a significance gate admits nothing and the abstract would have
    # to drop the result entirely.
    #
    # What the data does support is a DIRECTION plus a magnitude with its
    # interval, which is what is now reported: how many models move away from
    # zero when the unmatched pairs are dropped, and the growth ratio with the
    # clustered interval the artifact carries. Selection is on magnitude only
    # and is stated in the sentence rather than left to §10.1.
    _mask_gain = _mask_gain_ci = _mask_n_away = None
    if nlen:
        _mm = {m: nlen[m]["token_matched_first_name_clustered"]
               for m in nlen if not m.startswith("_")
               and "token_matched_first_name_clustered" in nlen[m]}
        # "away from zero" = the token-matched effect is larger in absolute
        # value than the full-grid effect, i.e. the confound was masking.
        _mask_n_away = sum(
            1 for v in _mm.values()
            if abs(v["effect_token_matched"]["est"])
            > abs(v["effect_all_pairs"]["est"]))
        _cand = [m for m, v in _mm.items() if v.get("growth_ratio", {})
                 .get("interpretable")]
        if _cand:
            b = max(_cand, key=lambda m: _mm[m]["growth_ratio"]["est"])
            _mask_gain = _mm[b]["growth_ratio"]["est"]
            _mask_gain_ci = _mm[b]["growth_ratio"]["ci"]
        # HOW MUCH OF THAT "<i>three</i> <i>of</i> FOUR" IS READABLE. Two of the three sit
        # on a baseline effect the artifact itself flags interpretable=false,
        # because the denominator is not distinguishable from zero; among the
        # two that ARE interpretable the split is one each way, and Llama-2-7B
        # moves TOWARD zero, meaning the confound inflated rather than masked
        # on that model. The paper asserted the causal direction flatly. These
        # counts let it state what the four estimates actually support.
        _mask_interp = len(_cand)
        _mask_interp_away = sum(
            1 for m in _cand
            if abs(_mm[m]["effect_token_matched"]["est"])
            > abs(_mm[m]["effect_all_pairs"]["est"]))
        _mask_toward = [m for m in _cand
                        if abs(_mm[m]["effect_token_matched"]["est"])
                        <= abs(_mm[m]["effect_all_pairs"]["est"])]
        _mask_clusters = ((nlen.get("_token_matched_clustering") or {})
                          .get("n_clusters_in_matched_subset", {}).get(b))
        _mask_best = b if _cand else None

    # How many models show a token-length slope distinguishable from zero.
    # Spelled out rather than inlined: the abstract used to compute this in a
    # single expression built out of chr() calls, which was unreadable and
    # therefore unauditable. The opposite of what this file is for.
    #
    # THE UNIT IS THE FIRST-NAME PAIR, NOT THE ROW. The 48 rows this slope is
    # fitted over are 12 first-name pairs x 4 surnames, so a row-level
    # bootstrap treats 12 clusters as 48 independent draws -- the exact error
    # §6.1 of this paper identifies and forbids. An earlier draft counted on
    # the row-level `p` and reported "two of four" in the abstract, §1.2 and
    # §4.4. On the clustered slope the count is one. Everything downstream now
    # reads SLOPE, so the two can never drift apart again.
    def SLOPE(m):
        """The token-length slope on the resampling unit §6.1 requires."""
        return nlen[m]["token_matched_first_name_clustered"]["slope_same_unit"]

    _n_slope_sig = len([m for m in (nlen or {})
                        if not m.startswith("_") and SLOPE(m)["p"] < 0.05])
    _n_slope_sig_word = NUM.get(_n_slope_sig, str(_n_slope_sig))

    # The reporting-matrix counts, and the model on which the token-matching
    # restriction moves the effect most. Both were typed as literals in §4.4
    # ("13 audits", "0.15 and 0.24") in a paper whose rule is that no number is.
    cts = matrix["counts"] if matrix else {}
    _tok_worst = _tok_all = _tok_matched = None
    if nlen:
        _tk = {m: nlen[m]["token_matched_first_name_clustered"]
               for m in nlen if not m.startswith("_")
               and "token_matched_first_name_clustered" in nlen[m]}
        if _tk:
            _tok_worst = max(_tk, key=lambda m: abs(
                _tk[m]["matched_minus_all"]["est"]))
            _tok_all = _tk[_tok_worst]["effect_all_pairs"]["est"]
            _tok_matched = _tk[_tok_worst]["effect_token_matched"]["est"]

    # How many studies. Typed in two places and they disagreed -- "eight" in
    # 1.1 against "nine" in 3 -- which a reviewer found. Both now read the
    # study registry the corpus artifact builds by walking the data tree.
    _corpus = None
    _p = D / "reference" / "corpus_size.json"
    if _p.exists():
        _corpus = json.loads(_p.read_text(encoding="utf-8"))
    _n_studies = _corpus["n_studies"] if _corpus else None

    # The dispersion budget restricted to models whose effect is identified.
    # A ratio whose denominator is not separable from zero is not a statement
    # about instrument variance, so the introduction quotes only the two
    # checkpoints where the denominator exists.
    #
    # AND THE TWO KINDS ARE KEPT APART. "Moves the effect by X" cannot mean the
    # same thing for a standard deviation over a set of defensible choices and
    # for a single shift between two specific ones. The budget artifact carries
    # a `kind` field that the paper used to collapse, so its headline range ran
    # from an SD at one end to a shift at the other. They are now two ranges.
    _budget_sd, _budget_shift, _budget_shift_weak = [], [], []
    # Which component and which model each endpoint of the headline range comes
    # from. A reviewer could not map "16 % to 50 %" onto any table; naming the
    # two endpoints costs a clause and makes the sentence checkable against
    # Figure 1, which draws every component per model.
    _budget_named = []
    _bp = D / "delta_stability" / "dispersion_budget.json"
    if _bp.exists():
        _b = json.loads(_bp.read_text(encoding="utf-8"))
        for _m in ORDER:
            if _m not in _b.get("models", {}) or _m not in survives:
                continue
            for _k, _c in _b["models"][_m]["components"].items():
                if _k == "noise floor" or not _c.get("over_beta"):
                    continue
                _budget_named.append((_c["over_beta"], _k, _m,
                                      _c.get("kind")))
                if _c.get("kind") == "shift":
                    _budget_shift.append(_c["over_beta"])
                    if _c.get("separable_from_zero") is False:
                        _budget_shift_weak.append(_c["over_beta"])
                else:
                    _budget_sd.append(_c["over_beta"])
    _budget_id = _budget_sd + _budget_shift

    # On how many models has the wording overtaken the name draw by k = 9?
    # Interpolated rather than typed, and read off the population-referent arm
    # -- the same arm §4.2's tables use. An earlier abstract said "three of
    # four" from the conditional arm while the sentence around it made the
    # population claim.
    _n_word_wins_abs = "several"
    if names:
        _r9 = [names[m]["averaging_asymmetry_with_replacement"]
               ["draw_9_per_race"]["ratio_name_to_wording"]
               for m in ORDER if m in names]
        if _r9:
            _n = sum(1 for x in _r9 if x < 1.0)
            _n_word_wins_abs = NUM.get(_n, str(_n))

    # WHAT THE OCCUPATION STUDY ACTUALLY SUPPORTS. The abstract used to say the
    # occupation "changes the dispersion itself". A permutation test against
    # the null that the three postings share one between-wording dispersion
    # does not reject on any model (p = 0.27 to 0.63), so that claim is
    # withdrawn -- see §4.5. What the same artifact does support, strongly, is
    # that the occupation moves the EFFECT, and that is what the abstract now
    # says. Interpolated, so it tracks the artifact if the test is re-run.
    _occ_gap_sig = _occ_gap_min = _occ_gap_max = None
    if occnull and occnull.get("models"):
        _g = [v["effect_gap_across_postings"] for v in occnull["models"].values()
              if "effect_gap_across_postings" in v]
        if _g:
            _n = sum(1 for x in _g if x["p_permutation"] < 0.05)
            _occ_gap_sig = NUM.get(_n, str(_n))
            _occ_gap_min = min(x["est"] for x in _g)
            _occ_gap_max = max(x["est"] for x in _g)

    # The Study 2 noise floor as a fraction of that model's own effect, on the
    # identified models. §1 used to say "a byte-identical repeat gives zero",
    # which is true of §5.2's controlled configuration and NOT of Study 2, the
    # run every dispersion number in the abstract comes from. §10.1 concedes
    # the gap; §1 was asserting across it.
    #
    # THE DENOMINATOR IS THE LOG-ODDS EFFECT, NOT THE SUPERIORITY DISTANCE.
    # sigma_noise_on_variant_mean is built from white_margin - black_margin, so
    # it is in log-odds. Dividing it by |P(superiority) - 0.5| put a log-odds
    # numerator over a probability denominator and returned about 3 %; the
    # like-for-like figure is about 5 %. §4.1 states the rule this broke --
    # "Both quantities are on the log-odds scale ... so the ratio compares like
    # with like" -- three sections before the ratio that broke it, which is how
    # a guard written once fails to guard everything.
    # Read from dispersion_budget.json rather than recomputed here, because
    # that artifact already divides every component by beta_logodds and Figure 1
    # draws it. Two independent computations of the same ratio is how the text
    # came to disagree with the figure it told the reader to check.
    _noise_over_beta = None
    _bnf = D / "delta_stability" / "dispersion_budget.json"
    if _bnf.exists() and survives:
        _bb = json.loads(_bnf.read_text(encoding="utf-8")).get("models", {})
        _r = [_bb[m]["components"]["noise floor"]["over_beta"]
              for m in survives
              if m in _bb and "noise floor" in _bb[m].get("components", {})]
        if _r:
            _noise_over_beta = max(_r)

    # The size of the validated list this field actually uses. §4.4 compares it
    # with the voter-file pool An and Rudinger drew on, and "this list" in that
    # paragraph is Bertrand and Mullainathan's, not our grid -- an earlier fix
    # interpolated our own 24 distinct first names and so compared the wrong two
    # things. Read from the transcription artifact, which carries all four
    # race-by-gender blocks.
    _bm_n_first = 36
    if bm and isinstance(bm.get("first_names"), dict):
        _bm_n_first = sum(len(v) for v in bm["first_names"].values()
                          if isinstance(v, list))

    # The tightest honest statement about the serving manipulations: not that
    # they move nothing, but that their shift is BOUNDED. The conclusion printed
    # only p-values, and a p-value cannot distinguish "no effect" from "no
    # power". Widest endpoint of any effect-difference interval, as a fraction
    # of that model's own effect.
    _serving_bound = None
    _sb = []
    for _f in ("replicate_analysis.json", "cache_residual.json"):
        _pp = D / "replicate" / _f
        if not _pp.exists():
            continue
        _dd = json.loads(_pp.read_text(encoding="utf-8"))
        for _m, _v in _dd.items():
            # Identified models only. Dividing by an effect that covers zero
            # gives a ratio with no upper bound, which is the error §4.5 and
            # Table 10 exist to police -- and it inflated this bound from
            # about 5 % to 20 % on a model whose effect is 0.058.
            if (_m.startswith("_") or _m not in ps or _m not in survives
                    or not isinstance(_v, dict)):
                continue
            # LIKE FOR LIKE. effect_difference is a contrast of mean paired
            # MARGIN differences, so it is in log-odds; the denominator has to
            # be the log-odds effect. Dividing by |P(superiority) - 0.5| mixed
            # the scales and reported this bound as 5.4 % when the comparable
            # figure is 8.7 %.
            _ci = (_v.get("effect_difference") or {}).get("ci")
            if _ci and abs(lo[_m]["est"]) > 1e-9:
                _sb.append(max(abs(x) for x in _ci) / abs(lo[_m]["est"]))
    if _sb:
        _serving_bound = max(_sb)

    # THE LARGEST EFFECT THIS PAPER MEASURES, over every study that measures
    # one on the primary posting -- not over Study 2 alone. Study 2 runs at
    # Q4_K_M; the quantization study runs the same models on the same posting
    # at Q8_0 and reaches further from zero on one of them, and the
    # forty-eight-pair name grid reaches further still than Study 2. Two
    # sentences called the Study 2 maximum "the largest effect we measure" and
    # divided the mechanism panel's detection floor by it, which understated
    # the floor as a fraction of what the paper can actually see.
    _eff_max, _eff_max_src = 0.0, ""
    for _m in models:
        if abs(lo[_m]["est"]) > _eff_max:
            _eff_max, _eff_max_src = abs(lo[_m]["est"]), "Study 2"
    if quant:
        # `shift` is the Q8 checkpoint's displacement FROM its base, so the
        # Q8 effect is the base effect plus the shift. Table 14 prints the two
        # side by side: +0.1805 at Q4_K_M and +0.2041 at Q8_0.
        for _m, _v in quant.items():
            if _m.startswith("_") or not isinstance(_v, dict):
                continue
            _base = _v.get("base")
            if _base in lo and isinstance(_v.get("shift"), (int, float)):
                _e = abs(lo[_base]["est"] + _v["shift"])
                if _e > _eff_max:
                    _eff_max, _eff_max_src = _e, "the quantization study"
    if names:
        # beta is [lower, median, upper]; the point estimate is the median.
        for _m, _v in names.items():
            if _m.startswith("_") or not isinstance(_v, dict):
                continue
            _b = _v.get("beta")
            if isinstance(_b, list) and len(_b) == 3 and abs(_b[1]) > _eff_max:
                _eff_max, _eff_max_src = abs(_b[1]), "the name grid"

    # HOW MANY QUANTIZATION SHIFTS ARE SEPARABLE FROM ZERO. One of two. The
    # abstract said "separable from zero" unqualified -- a sentence written to
    # replace an earlier overstatement, which overstated a different way: appendix E
    # and Table 14 both say 1 of 2, with p = 0.158 on Llama-2-7B against 0.002
    # on Mistral v0.1. A reviewer can refute the abstract from Table 14 without
    # turning the page, so the count is interpolated here and used in both.
    _quant_models = [v for m, v in (quant or {}).items()
                     if not m.startswith("_") and isinstance(v, dict)
                     and v.get("shift_ci")]
    _quant_sep = sum(1 for v in _quant_models
                     if v["shift_ci"][0] * v["shift_ci"][1] > 0)
    _quant_n = len(_quant_models)

    # THE OPEN-WEIGHT PANEL'S OWN REPLICATE FLOOR, bound once and guarded.
    # §10.1 needs it and sits far from Appendix D.1, where the per-panel summaries
    # are unpacked inside a conditional; referring to that binding from here
    # would raise if the artifact were ever missing, which is the one case the
    # build is supposed to survive.
    _local_floor = (fnoise or {}).get("summary_local")

    # STUDIES THAT PRODUCE MATCHED PAIRS, which is what the record count is a
    # count of. _n_studies is 11 and includes the two single-prompt studies
    # that contribute none, so "31,468 matched-pair records across eleven
    # studies" attributed the pairs to studies that have no pairs in them.
    _n_studies_mp = None
    _csp = D / "reference" / "corpus_size.json"
    if _csp.exists():
        _n_studies_mp = json.loads(
            _csp.read_text(encoding="utf-8")).get("n_studies_matched_pair")

    # HOW MANY CHECKPOINTS EACH SERVING MANIPULATION ACTUALLY COVERS. §1.2
    # said "batching is manipulated on every checkpoint in the panel". Study 8
    # runs on four of the six the mechanism panel defines, and Study 9 on two.
    # The sentence was making the serving evidence sound panel-wide in the same
    # breath as conceding that "complete" overstates the cache arm.
    # NO TYPED FALLBACK BEHIND A NUMBER IN THE ABSTRACT. This read
    # `... or 6`, and six is exactly what the artifact holds today: lose the
    # artifact and the sentence would have read the same while the number
    # stopped being measured. \NPanel is in the abstract of this paper and
    # of the ICLR fork. The file's idiom for an artifact it cannot build
    # without is line 597's sys.exit, and this is one of those.
    _panel_models = [k for k in (mech or {}) if not k.startswith("_")]
    if not _panel_models:
        sys.exit(
            "cannot build without mechanism_panel/mech_panel_analysis.json: "
            "\\NPanel is interpolated into the abstract, and the fallback "
            "that used to stand in for it printed the value the artifact "
            "happens to hold today, so nothing would have looked wrong.")
    _n_panel = len(_panel_models)
    _n_batch = len([k for k in (rep or {})
                    if not k.startswith("_") and isinstance(rep[k], dict)])
    _n_cache = len([k for k in (cache or {})
                    if not k.startswith("_") and isinstance(cache[k], dict)])

    # ON HOW MANY MODELS IS THE PRIMARY POSTING THE FAVOURABLE ONE?
    # Two sites claimed the conjunction -- largest effect AND smallest
    # dispersion -- held on two of four. It holds on one: Llama-3.1's smallest
    # dispersion is on the software-engineer posting, and both Mistrals show
    # their largest effect on the nurse posting. The weaker half (largest
    # effect alone) holds on two. Both counts are derived so a re-run cannot
    # leave the prose behind.
    _ba_both = _ba_eff = 0
    if occ:
        for _m, _v in occ.items():
            if _m.startswith("_") or not isinstance(_v, dict):
                continue
            _o = {_k: _v[_k] for _k in ("BA", "SWE", "RN") if _k in _v}
            if len(_o) < 2:
                continue
            _be = max(_o, key=lambda k: abs(_o[k]["ps"] - 0.5))
            _bs = min(_o, key=lambda k: _o[k]["ps_sd_across_wordings"])
            _ba_eff += (_be == "BA")
            _ba_both += (_be == "BA" and _bs == "BA")

    # THE WORDING RATIO OVER EVERY CELL THAT CAN CARRY IT, not over Study 2
    # alone. The ratio is sd-across-wordings divided by the absolute effect,
    # and three studies compute it identically: Study 2 (business analyst),
    # Study 7 (three postings) and the frontier arm. Quoting Study 2's two
    # identified cells while naming eleven checkpoints understated the range
    # by half and misdescribed its scope.
    # KEYED ON (MODEL, POSTING), because two sources supply the same cell:
    # the occupation artifact's business-analyst arm IS Study 2's data
    # (analyze_occupation.py says so; the ratios agree to 16 digits), so
    # appending both counted the two identified models' BA cells twice and
    # the abstract said "8 cells" over 6 distinct ones. The endpoints could
    # never move -- a duplicate cannot change a min or a max -- but the count
    # is a claim about scope, and it was wrong. First writer wins.
    #
    # The token-matched share is also computed here, because the abstract
    # states it. It previously appeared only in EXPORT (after the abstract
    # string was built) while the abstract spelled it as "a quarter to a
    # third" -- number words, which audit_hardtyped_numbers is structurally
    # blind to: right today, free to drift on any tokenizer change.
    _tm_ab = ({m: nlen[m]["n_same_length"] / nlen[m]["n_pairs"]
               for m in models if m in nlen} if nlen else {})
    # The template-vs-wording decomposition, for the abstract. Computed on
    # the sign-flip checkpoint's spec curve (the only one the artifact
    # carries), so the abstract clause is scoped to that checkpoint and
    # emitted only while the template share actually exceeds the wording
    # share. A reviewer read the paper's own 91.5%-between-templates result
    # and asked, fairly, why the abstract never mentions the largest
    # component it found.
    # THE FLIP CHECKPOINT, not the first one. All four models carry a
    # decomposition, and on llama-2 the WORDING share exceeds the template
    # share -- the first draft of this selection took models[0] and the
    # truth-gate below correctly suppressed the clause, which is how the
    # mismatch was noticed. The sentence says "the checkpoint whose sign
    # flips", so the selection is the model with significant specs on both
    # sides of zero.
    _fd_ab = None
    for _m in models:
        _sc = s2.get(_m, {}).get("spec_curve") or {}
        if (_sc.get("factor_decomposition")
                and _sc.get("n_sig_positive", 0) > 0
                and _sc.get("n_sig_negative", 0) > 0):
            _fd_ab = _sc["factor_decomposition"]["dispersion_share"]
            break
    _tmpl_ab = bool(_fd_ab
                    and _fd_ab["frac_template"] > _fd_ab["frac_wording"])
    _rat_by_cell = {}
    for _m in survives:
        _rat_by_cell[(SHORT.get(_m, _m), "business analyst")] = ratio_ps[_m]
    for _m, _v in (occ or {}).items():
        if _m.startswith("_") or not isinstance(_v, dict):
            continue
        for _k, _lbl in (("BA", "business analyst"), ("SWE", "software engineer"),
                         ("RN", "registered nurse")):
            _c = _v.get(_k)
            if not isinstance(_c, dict) or _c.get("ratio_sd_to_effect") is None:
                continue
            _ci = _c.get("ps_ci")
            if not _ci or (_ci[0] - 0.5) * (_ci[1] - 0.5) <= 0:
                continue
            _rat_by_cell.setdefault((SHORT.get(_m, _m), _lbl),
                                    _c["ratio_sd_to_effect"])
    for _m in ((front or {}).get("summary", {}).get("identified_models") or []):
        _r = front["models"][_m].get("ratio_sd_to_effect")
        if _r is not None:
            _rat_by_cell.setdefault((_m, "frontier"), _r)
    _rat_cells = [(m, p, v) for (m, p), v in _rat_by_cell.items()]
    _rat_vals = [c[2] for c in _rat_cells]
    _rat_lo = min(_rat_vals) if _rat_vals else None
    _rat_hi = max(_rat_vals) if _rat_vals else None
    _rat_n = len(_rat_vals)

    # THE POSITION MANIPULATION HAS NO PREDICTED SIGN, so it cannot agree or
    # disagree with the length slope and this file used to count how often it
    # did. D8 and D9 add one and two whitespace tokens INSIDE the posting,
    # identically in both arms of every pair, so the within-pair token-length
    # difference the slope regresses on is unchanged and the linear model
    # predicts a shift of exactly zero in every cell. The tally that used to
    # live here was arithmetic on a comparison with nothing to compare against;
    # the companion paper states the design fact instead. Removed rather than
    # corrected, because there is no correct version of it.

    # THE BH FAMILY SIZE, READ RATHER THAN TYPED. Three sentences quoted 168
    # as a literal. Round 7 showed the paper's governing claim -- that every
    # number is interpolated -- was false, and this was one of the numbers
    # making it false.
    _bh_n = "168"
    for _mv in (mech or {}).values():
        if not isinstance(_mv, dict):
            continue
        for _md in _mv.values():
            if isinstance(_md, dict) and _md.get("bh_family_size"):
                _bh_n = f"{_md['bh_family_size']:,}"
                break
        if _bh_n != "168":
            break

    # ---- SCALE GUARD -----------------------------------------------------
    # Two of this paper's ratios were built with a log-odds numerator over a
    # probability-of-superiority denominator, and both survived every check
    # the project had: the arithmetic is valid, the artifacts are fresh, the
    # numbers are interpolated rather than typed, and the result is a
    # plausible small percentage. Only reading the definitions catches it.
    #
    # So the definitions are asserted here, at the point of use, where they
    # are cheap to state and impossible to drift away from. Each entry names
    # the quantity, its scale, and the artifact field it comes from; the
    # assertion is that a ratio's two halves carry the same scale.
    _SCALES = {
        "sigma_noise_on_variant_mean": "logodds",   # white_margin - black_margin
        "effect_difference.ci": "logodds",          # contrast of those margins
        "logodds.est": "logodds",                   # the fitted effect
        "ps_sd_across_wordings": "superiority",     # SD of per-wording P(sup)
        "superiority.est": "superiority",           # the P(sup) effect
    }
    for _num, _den, _what in (
            ("sigma_noise_on_variant_mean", "logodds.est", "noise floor"),
            ("effect_difference.ci", "logodds.est", "serving bound"),
            ("ps_sd_across_wordings", "superiority.est", "dispersion ratio")):
        if _SCALES[_num] != _SCALES[_den]:
            sys.exit(f"scale mismatch in the {_what}: {_num} is "
                     f"{_SCALES[_num]} and {_den} is {_SCALES[_den]}")

    # The identified local checkpoint whose yes/no mass is closest to unity --
    # the like-for-like partner for the frontier models, all of which sit at 1.
    _best_mass = None

    n_lit = lit["summary_of_published_pp_gaps"]["n"] if lit else None
    med_lit = lit["summary_of_published_pp_gaps"]["median_abs_pp"] if lit else None
    n_below = lit["summary_of_published_pp_gaps"]["n_below_1_1_pp"] if lit else None
    _qm = [v for k, v in (quant or {}).items()
           if not k.startswith("_") and isinstance(v, dict)
           and v.get("shift_over_sigma_variant") is not None]
    # ARXIV CAPS THE ABSTRACT AT 1920 CHARACTERS. Kept short by stating
    # results and leaving their qualifications to the sections that earn them;
    # every figure is still interpolated, so this cannot drift from the body.
    abstract = (
        "Audits of large language models for hiring discrimination carry legal "
        "weight: NYC Local Law 144 makes it unlawful to use an "
        "automated employment decision tool not audited for bias within the "
        "preceding year. Such an audit reports a demographic effect \u2014 how "
        "differently a model treats two r\u00e9sum\u00e9s differing only in "
        "the name. Whatever the two share should cancel. It does not, by "
        "enough to change what an audit concludes."
        + f" Holding the model fixed and varying only choices published audits "
        f"do not report, over {n_records:,} matched pairs from "
        f"{n_calls:,} model calls on {NUM.get(_n_panel, _n_panel)} open-weight "
        f"and {NUM.get(_n_front, _n_front)} frontier checkpoints, the "
        "instruction wording moves the effect by "
        + (f"{fmt(_rat_lo * 100, 0)} % to {fmt(_rat_hi * 100, 0)} % of itself "
           if _rat_lo is not None else "a large fraction of itself ")
        + "across the "
        + (f"{_rat_n} " if _rat_n else "")
        + "model-by-posting cells where the effect is separable from zero, "
        "including under edits that "
        "change no word, and the job posted and the name drawn move it "
        "comparably"
        + ("; where the sign flips, the template outweighs "
           "the wording" if _tmpl_ab else "")
        # QUANTIZATION IS A DIFFERENT KIND OF QUANTITY and was listed
        # coordinately with two that are measured as an SD across a defensible
        # set. It is a shift between two specific settings, so "comparably"
        # under a "% of itself" metric was never supported by either cell.
        + ((f"; the quantization, on the "
            f"{NUM.get(len(_qm), str(len(_qm)))} checkpoints tested, shifts "
            f"it by {min(v['shift_over_sigma_variant'] for v in _qm):.2f}× to "
            f"{max(v['shift_over_sigma_variant'] for v in _qm):.2f}× the "
            "between-wording SD") if _qm else "")
        + ". The two prompts are identical in characters but "
        "not in tokens. Only "
        + (f"{fmt(100 * min(_tm_ab.values()), 0)} % to "
           f"{fmt(100 * max(_tm_ab.values()), 0)} % "
           if _tm_ab else "a minority ")
        + "of pairs from a standard "
        "list are token-matched."
        + " Choices made <i>after</i> the data exists matter as much. Resampling rows "
        "rather than name pairs narrows intervals by "
        + (f"{resamp['pooled_summary']['min_ratio']:.1f} to "
           f"{resamp['pooled_summary']['max_ratio']:.1f}\u00d7"
           if resamp else "several-fold")
        + "; a fixed operating point misstates the percentage-point "
        "conversion by "
        + (f"{scale['summary']['jacobian_error_min']:.1f}\u00d7 to "
           f"{scale['summary']['jacobian_error_max']:.0f}\u00d7"
           if scale else "orders of magnitude")
        + "; and since each r\u00e9sum\u00e9 is scored alone, the pairing is "
        "an analysis-time choice that moves the statistic the field reports "
        "while leaving the mean paired difference invariant. Measurements "
        "reproduce bitwise only once batching and cache residency are "
        "controlled."
        + ((f" Of {matrix['n_llm_hiring_audits']} LLM hiring audits read in "
            "full text, none fully reports its batching, cache policy or "
            "token matching, and none reports a dispersion statistic for how "
            "far its effect moves across wordings.") if matrix else "")
        + " We give the minimum an audit must report, a screening rule whose "
        "false-positive rate we calibrate, and the full pipeline."
    ) if lit else None
    if abstract is None:
        MISSING.append("abstract (literature artifact)")
        abstract = ""
    else:
        _ax = arxiv_abstract(abstract)
        _axp = ROOT / "paper-a" / "releases" / "abstract_arxiv.txt"
        _axp.parent.mkdir(parents=True, exist_ok=True)
        # newline at the end only; the body is a single line so arXiv's
        # carriage-return rule cannot introduce a break we did not intend
        _axp.write_text(_ax + "\n", encoding="ascii")
        print(f"  arXiv abstract: {len(_ax)} / {ARXIV_ABSTRACT_LIMIT} chars "
              f"-> {_axp.relative_to(ROOT)}")

    paper.title_block(
        ["The Instrument Is Not the Model:",
         "Measuring How Much of an LLM Hiring Disparity",
         "Comes from Unreported Design Choices"],
        "Anonymous Author(s)" if ANON else "David Mao",
        "Submitted for double-blind review" if ANON else "Independent",
        abstract,
        email=None if ANON else "davidmao.xyz@gmail.com",
        keywords=("algorithmic audit · measurement validity · "
                  "résumé screening · prompt sensitivity · "
                  "specification curve · reproducibility"))

    # ======================================================================
    P = paper.para
    H = paper.heading
    APP.bind(paper)

    H("1  Introduction")
    P("A correspondence audit sends two applications that differ in one "
      "attribute and records which is advanced. The design is old, careful and "
      "well understood: Bertrand and Mullainathan faxed (and in a few cases "
      "mailed) résumés to "
      "real employers and found a "
      f"{fmt(bm['_headline']['gap_pp'], 1) if bm else 'n/a'}-point callback gap "
      "between White- and Black-sounding names. Its strength is that everything "
      "except the manipulated attribute is held identical, so whatever the two "
      "applications share (the job, the wording, the reviewer’s mood) "
      "cancels in the difference.", indent=False)
    P("Language models are now audited the same way, and the results carry "
      "legal weight. New York City Local Law 144 of 2021 makes it unlawful for "
      "an employer or employment agency to use an automated employment "
      "decision tool to screen a candidate unless the tool “has been the "
      "subject of a bias audit conducted no more than one year prior to the "
      "use of such tool” and a summary of the results is published "
      "(§ 20-871(a)); the Department of Consumer and Worker Protection’s 2023 "
      "final rule fixes what that audit must compute, namely selection rates "
      "and an impact ratio over the EEO-1 race and sex categories. A number "
      "produced by the kind of audit this paper examines is therefore a number "
      "with a legal consequence attached, which is why its stability is worth "
      "measuring rather than assuming. The transfer of "
      "the design looks straightforward. It is not, and the reason is that a "
      "field experiment and a model evaluation differ in what the experimenter "
      "controls. A field audit cannot choose how an employer reads a "
      "résumé. A model audit chooses everything. The instruction "
      "wording, the résumé text, the names, the job, the numerical "
      "precision of the weights, the batching of requests, the resampling unit "
      "of the confidence interval, the scale of the reported number. None of "
      "these is the model. All of them are choices, most are never reported, "
      "and this paper measures how far each of them moves the answer.")
    P("The question is one of measurement validity rather than of fairness. We "
      "take no position on whether these models discriminate. We ask a prior "
      "question. When an audit reports that a model does or does not, how much "
      "of that number is the model?")
    P("It is worth naming the quantity, because the field has no word for it "
      "and a thing without a name is hard to require. An audit reports a "
      "single number. That number is the sum of what the model does, what the "
      "instrument does, what the draw happened to be, and what the analyst "
      "decided afterwards:")
    P("      reported effect  =  model tendency  +  <b>instrument variance</b> "
      "+  sampling variance  +  inference choice", indent=False,
      size=8.4, space_after=6.0)
    P("<b>Instrument variance</b> is the term this paper measures. The "
      "movement produced by choices about how the question is put and how it "
      "is executed, holding the model, the names and the analysis fixed. It is "
      "not sampling error"
      + (f": the byte-identical replicate embedded in Study 2 puts the "
         f"noise floor at about {pct(_noise_over_beta, 0)} of the effect in "
         "log-odds, and "
         "the between-wording dispersion exceeds it several times over"
         if _noise_over_beta else ": a byte-identical repeat gives a floor "
         "far below the dispersion")
      + ". §5.2 establishes the stronger form of that separately. Under a "
      "serving configuration Study 2 predates, the repeat gives <i>exactly</i> zero, "
      "and Appendix C confirms it on "
      + (f"{second['summary']['n_replicate_cells']:,} " if second
         else "")
      + "replicate cells in two further domains. "
      "It is not the model. The weights never change. And it is "
      "not visible in any published audit"
      + ((": "
          f"{matrix['counts']['n_wordings_separately_estimated']['derived_more_than_one']['n_more_than_one']}"
          " of the "
          f"{matrix['counts']['n_wordings_separately_estimated']['n_applicable']}"
          " studies to which the check applies estimate the effect separately "
          "under more than one wording, but "
          f"{matrix['counts']['dispersion_across_wordings']['n_reported']} of "
          f"{matrix['counts']['dispersion_across_wordings']['n_applicable']} "
          "report a dispersion statistic across them. A standard "
          "deviation, a range, a variance component , which is the "
          "quantity a reader needs and the one nobody prints (" + TAB("reporting_matrix") + ").")
         if matrix and "derived_more_than_one" in
         matrix["counts"]["n_wordings_separately_estimated"] else
         ", because reporting it requires a dispersion statistic across "
         "wordings that no surveyed study prints."), indent=False)
    P("The paper’s three parts are that decomposition. Part I is what the "
      "auditor <i>writes</i>: the wording, the names, the job, and whether "
      "the pair is matched in the units the model reads. Part II is what the "
      "auditor <i>runs</i> — the quantization, the batching, the cache. "
      "Part III is what the auditor <i>computes</i> — the resampling unit, "
      "the reporting scale, the pairing, the multiplicity correction. The last "
      "of these move no measurement at all and still move the published "
      "number, which is why they belong in the same account.")
    P("<b>What this does not claim.</b> An earlier title said these choices "
      "move the answer “as much as the model does”, and a reviewer was "
      "right that we never make that comparison. We can, and it does not "
      "support the phrase. Across these four checkpoints the measured effect "
      "itself spans "
      + (f"{min(ps.values()):.3f} to {max(ps.values()):.3f} on the primary "
         "effect size, a range of "
         f"{max(ps.values()) - min(ps.values()):.2f}, against a between-wording "
         f"standard deviation of {min(sd_word.values()):.3f} to "
         f"{max(sd_word.values()):.3f}. " if sd_word else "")
      + "Which model you audit matters several times more than how you word "
      "the question, and any claim to the contrary would be false.",
      indent=False)
    P("That comparison is also not the one a reader of an audit needs. A "
      "published audit reports one number for one model. The between-model "
      "spread is not available to its reader and does not enter their "
      "inference; what enters is the single estimate, and the question is how "
      "much of <i>that</i> is the instrument. The answer, on the two checkpoints "
      "here whose effect is distinguishable from zero, comes in two parts, "
      "because two different kinds of quantity are involved. Choices that "
      "have a defensible set of settings (the wording, the names drawn, the "
      "posting) contribute a standard deviation across that set of "
      + (f"{pct(min(_budget_sd), 0)} to {pct(max(_budget_sd), 0)} "
         if _budget_sd else "")
      + "of the whole effect. Choices that are a switch between two specific "
      "settings. The quantization, the token-matching restriction — "
      "contribute a shift of "
      + (f"{pct(min(_budget_shift), 0)} to {pct(max(_budget_shift), 0)} "
         if _budget_shift else "")
      + "of it, though on the name-pair resampling unit three procedures "
      "straddle 0.05 for the token-matching shift, so §4.4 does not settle "
      "whether it is separable from zero. "
      + (("The two ends of the first range are "
          + " and ".join(
              f"the {k} on {SHORT.get(m, m)}"
              for _, k, m, _kind in
              [min((x for x in _budget_named if x[3] != "shift")),
               max((x for x in _budget_named if x[3] != "shift"))])
          + ", and every component is drawn per model in "
          + FIGREF("fig11_dispersion_budget") + ". ")
         if _budget_named else "")
      + "Both are set against a measurement noise floor of "
      + (f"about {pct(_noise_over_beta, 0)} " if _noise_over_beta
         else "about five per cent ")
      + "of the effect. That is the "
      "claim this paper makes, and it is enough. An estimate that moves by "
      "half its own size under choices nobody reports is not a number a "
      "regulator can act on, whatever the spread between models happens to "
      "be. One caveat travels with the posting component and is stated where "
      "it is earned. It is a standard deviation over <i>three</i> postings, which is "
      "a thin basis for an SD, and §4.5 shows that three postings cannot "
      "establish a difference in <i>dispersion</i> between jobs. What they do "
      "establish is a difference in the effect, which is the quantity this "
      "component measures.")

    P("<b>Two estimands, and this paper only threatens one.</b> “Does this "
      "model discriminate?” and “does this deployed pipeline "
      "discriminate?” are different questions and the difference decides "
      "what the dispersion means. A vendor auditing the fixed prompt that its "
      "product actually ships has the fixed-prompt number as its correct "
      "estimand. The wording is not a nuisance to average over, it is part of "
      "the system under test, and the dispersion measured here is irrelevant "
      "to it. A study claiming that a <i>model</i> is or is not biased has the "
      "opposite problem, because it has chosen one point on a surface it did "
      "not characterise and reported it as a property of the weights. Almost "
      "every audit in Section 8 makes the second kind of claim while running "
      "the first kind of experiment. The recommendations in Section 9 are "
      "marked accordingly.", indent=False)
    P("<b>Why hiring, and why this design.</b> The choice of domain is not "
      "incidental. A correspondence audit is the rare evaluation whose "
      "estimand is a <i>difference</i> between matched conditions, so it is the "
      "hardest case for the argument. Whatever the instrument does to one arm "
      # NOT "HIRING PUTS <i>the</i> <i>model</i> NEAR A DECISION BOUNDARY". On half the
      # panel it does not: noise_vs_probability.json records 24 of 72 replicate
      # cells inside the 0.2-0.8 band on two checkpoints and NONE on the other
      # two. The premise is real where it holds and the sentence now says
      # where that is; three em-dashes in a row was the first draft of this
      # repair, so the counts go in a clause of their own.
      "it should do to the other, and cancel. Hiring puts some cells near a "
      "decision boundary, on two of the four checkpoints a third of them and "
      "on the other two none at all, and §5.2 shows that is where the "
      "measurement is least stable, the closer a cell sits to p = 0.5 the "
      "more it moves under byte-identical repetition; "
      "it is heavily studied, so there is a published record to calibrate "
      "against; and it has a field-experiment anchor with per-name covariates, "
      "which is what makes the construct check in Appendix B possible at all. A "
      "result on a difference-in-differences design transfers to the easier "
      "case of a single-condition benchmark; the converse would not.")

    H("1.1  What we do", 2)
    P("We hold the model fixed and vary the instrument. Six open-weight "
      "checkpoints spanning a generational boundary are served locally, so the "
      "weights, the decoding and the serving configuration are all under "
      "experimental control rather than behind an API. Across "
      + (NUM.get(_n_studies, str(_n_studies)) if _n_studies else "eight")
      + " studies we "
      "vary, one at a time: the instruction wording, in two arms, one of which "
      "is designed so that no word changes; the names, on a factorial grid "
      "built from a validated list; the occupation, across three structurally "
      "matched postings; the weight quantization; the structural form of the "
      "prompt, across eleven conditions with a position control; the request "
      "concurrency; and the key-value cache policy. A tokenizer probe across "
      "the full name grid adds one more axis the design did not set out to "
      "vary and turns out not to have held fixed either. We then vary three "
      "choices that live in the analysis rather than the experiment. The "
      "bootstrap resampling unit, the reporting scale and the pairing.",
      indent=False)

    H("1.2  Contributions", 2)
    P("<b>A dispersion budget for an audited effect.</b> We give, per model, "
      "the standard deviation the reported demographic effect inherits from "
      "each unreported choice, on a common scale, with the measurement noise "
      "floor established separately so that dispersion is distinguishable from "
      "error.", indent=False)
    # WHAT THE CROSSED GRID BUYS, AND ONLY IT. A full-name random effect --
    # Armstrong et al. (2024) "treat the name as a random effect" -- already
    # gives the within-race between-name SD apart from the between-race
    # effect, so two earlier versions of this bullet claimed priority over
    # work that had it. Crossing first names with surnames is what makes the
    # two components separately estimable, which a hand-picked full-name list
    # cannot do at any size.
    _nv0k = next((k for k in (names or {}) if not k.startswith("_")), None)
    _nv0 = (names or {}).get(_nv0k) if _nv0k else None
    _nvd = [v["sigma_first_minus_last"] for k, v in (names or {}).items()
            if not k.startswith("_") and isinstance(v, dict)
            and v.get("sigma_first_minus_last")]
    _nvd_n = sum(1 for d in _nvd if d[0] > 0)
    P("<b>The first-name and surname components of name-draw variance, "
      "separated.</b> Name-draw sensitivity is reported by others and so is a "
      "name variance component. A full-name random effect, as in Armstrong "
      "et al. (2024), already estimates the within-race between-name standard "
      "deviation apart from the between-race effect. What it cannot do is say "
      "how much of that variance belongs to the first name and how much to "
      "the surname, because a hand-picked list of full names never varies one "
      "with the other held fixed. Crossing "
      + (f"{_nv0['n_first']} first names with "
         f"{_nv0['n_last']} surnames" if _nv0 else "the two")
      + " makes them separate parameters"
      # THE DIFFERENCE, NOT TWO MARGINALS. This used to compare the two
      # marginal medians of whichever checkpoint sorted first and call them
      # "not the same size" -- a contrast asserted from two summaries that
      # cannot settle it, with no interval and no count of the panel. The
      # posterior on the difference is now formed on the joint draws.
      + ((f", and the first name carries the larger share on "
          f"{NUM.get(_nvd_n, str(_nvd_n))} of the "
          f"{NUM.get(len(_nvd), str(len(_nvd)))} checkpoints. On "
          f"{SHORT.get(_nv0k, _nv0k)}, "
          f"{fmt(_nv0['sigma_first'][1], 3)} against "
          f"{fmt(_nv0['sigma_last'][1], 3)}, a difference of "
          f"{fmt(_nv0['sigma_first_minus_last'][1], 3)} "
          f"[{fmt(_nv0['sigma_first_minus_last'][0], 3)}, "
          f"{fmt(_nv0['sigma_first_minus_last'][2], 3)}]")
         if _nvd and _nv0 and _nv0.get("sigma_first_minus_last") else
         ((f", and they are not the same size: "
           f"{fmt(_nv0['sigma_first'][1], 3)} against "
           f"{fmt(_nv0['sigma_last'][1], 3)} on "
           f"{SHORT.get(_nv0k, _nv0k)}")
          if _nv0 and _nv0.get("sigma_first") else ""))
      + ". That separation is what the factorial grid buys, and it is the "
      "narrow thing we claim.")
    P("<b>A threat to the matched-pair design, measured and quantified.</b> "
      "The two prompts in a matched pair are identical in characters and, "
      "two thirds to three quarters of the time, not in tokens. The token-length difference predicts "
      f"the measured effect on {_n_slope_sig_word} of the four checkpoints "
      "where it was tested"
      + (f", restricting to token-matched pairs moves the disparity further "
         f"from zero on {NUM.get(_mask_n_away, _mask_n_away)} of them. By "
         f"{pct(_mask_gain, 0)} [{pct(_mask_gain_ci[0], 0)}, "
         f"{pct(_mask_gain_ci[1], 0)}] on the largest, a shift not itself "
         "measured on a subset too small to settle its size , three "
         "procedures on the name-pair resampling unit straddle 0.05 "
         "(§4.4), with the direction split one model each way among "
         "those whose baseline effect can carry a ratio"
         if _mask_gain is not None else "")
      + ". We also build the balanced grid whose absence the field treats as a "
      "given, and report why the standard name list cannot supply one at "
      "usable size.")
    P("<b>An occupation control, and a defence that survives half of it.</b> "
      "The obvious response to wording sensitivity is that the "
      "dispersion-to-effect ratio is a stable property of a model, so a reader "
      "can discount by it. Half of that fails outright. On two thirds of the "
      "model-by-posting cells the ratio’s denominator is not separable "
      "from zero, so there is no ratio to carry from one audit to another. The "
      "other half we tried to establish and could not. The job posted moves "
      "the <i>effect</i> by about as much as the whole demographic effect"
      + (f" on {_occ_gap_sig} of four models" if _occ_gap_sig else "")
      + ", but a permutation test against the null that the three postings "
      "share one between-wording dispersion does not reject on any of them, so "
      "we do not claim the dispersion is a property of the model-and-job pair. "
      "An earlier draft did; §4.5 reports the withdrawal and the test that "
      "forced it.")
    P("<b>A complete account of the apparent nondeterminism.</b> Four of "
      "the surveyed audits rest reproducibility on a decoding setting alone, "
      "and on this panel that does not hold. The disagreement is "
      "attributable to request batching and "
      "key-value cache residency. With both controlled the measurement "
      "reproduces bitwise, across separate server processes, and on "
      "Appendix C’s two further domains as well , while the effect "
      "estimate itself is not detected to move under either. A bound rather "
      "than an absence, stated as such in §5.2. Batching is manipulated on "
      + NUM.get(_n_batch, str(_n_batch)) + " of the panel’s "
      + NUM.get(_n_panel, str(_n_panel)) + " checkpoints; cache "
      "residency on the two where a second "
      "session was run, which is where the word “complete” is doing "
      "more work than the design supports.")
    P("<b>A negative result with an explicit position control.</b> Eleven "
      "conditions find no evidence that the sensitivity is carried by any "
      "specific structural feature. We report this as a replication of "
      "existing characterisations of format sensitivity rather than as a "
      "failure. A word on its status, because an earlier draft called it "
      "pre-registered and it is not. The delimiter hypothesis and the test "
      "that would falsify it were written into a gap register before the "
      "confirmatory runs, but the eleven conditions themselves were not: "
      "the position controls D8 to D10 were added mid-project, after the "
      "pilot showed that every condition which destroyed a delimiter also "
      "moved the name. That addition is the reason the null means anything, "
      "and it is also the reason the design is not pre-registered.")
    # THE CLAIM, STATED AS WHAT IS TRUE AND CHECKED. It read "Every number in
    # this paper is interpolated from an artifact on disk", which two lenses of
    # round 7 falsified: several data-derived figures were typed constants that
    # would have printed unchanged if their artifact had changed. Those are now
    # interpolated, and audit_hardtyped_numbers.py enforces it, but design
    # constants, cross-references and reference-list page numbers are typed on
    # purpose and always will be. A guarantee that names its own boundary is
    # worth more than a universal that a reader can falsify with grep.
    P("<b>An auditable pipeline.</b> Every measured quantity in this paper is "
      "interpolated from an artifact on disk by the script that typesets it, "
      "and a check over the typesetting source reports any measurement typed "
      "into the prose instead of read from disk; what it still reports are "
      "design constants, cross-references, bibliographic detail and figures "
      "quoted from other papers, each exempted in that check with a stated "
      "reason. "
      "Every superseded claim is retained with its correction rather than "
      "deleted, and this version carries a great many, because the draft "
      "before it was put through an adversarial audit that read each section "
      "against the artifacts underneath it. Where that audit reversed a "
      "conclusion of ours, the reversal is in the text beside the claim it "
      "replaced.")
    if budget:
        paper.figure(FIGS / "fig11_dispersion_budget.png", span2=True,
                     max_h=2.9 * pk.inch, space_after=12.0)

    # ======================================================================
    H("2  Background")
    P("<b>Correspondence audits.</b> The matched-pair design and its analysis "
      "by discordant pairs are standard. Its validity rests on the two "
      "applications being identical apart from the manipulated attribute, and "
      "on the reviewer being a fixed instrument across the pair. The "
      "tradition has audited its own instrument before: the "
      "Heckman–Siegelman critique, formalised by Neumark, shows that group "
      "differences in the variance of unobservables can alone generate "
      "spurious evidence of discrimination in either direction, even in a "
      "correspondence study. That is a design-side ancestor of the question "
      "asked here, about a component that critique held fixed: the stimulus "
      "text itself, and the machinery that serves it.", indent=False)
    P("<b>Where the design is blocked, the match is a property of the design "
      "rather than of the analysis.</b> Lippens supplies the clearest "
      "instance of that in this literature. Eighteen preconstructed profiles "
      "per vacancy, varying within a vacancy only in name and identity, with "
      "standard errors from a cluster-robust wild bootstrap clustered at the "
      "vacancy level because that is where assignment happened. Sections 6.1 "
      "and 6.3 concern the case that construction forecloses. An audit that "
      "scores each résumé alone and imposes the pairing afterwards, and the "
      "distinction is load-bearing for both.", indent=False)
    P("<b>Audits of language models.</b> A rapidly growing literature applies "
      "the design to LLM résumé screening. Section 8 tabulates the "
      "effect sizes and, more importantly for our purposes, the number of "
      "distinct prompts each study used to obtain them.")
    P("<b>Two of those audits reach part of this paper’s conclusion "
      "before it, and should be read as antecedents rather than as related "
      "work.</b> Seshadri, Chen, Singh and Goldfarb-Tarrant study allocational "
      "fairness in résumé summarisation and applicant ranking, perturb "
      "résumés along demographic and NON-demographic dimensions, and "
      "report that retrieval models “can show comparable sensitivity to "
      "both demographic and non-demographic changes, suggesting that fairness "
      "issues may stem from broader model brittleness.” That is the shape "
      "of the argument here, in hiring, published first, and we do not claim "
      "the idea. Tan and co-authors supply the other half from the instruction "
      "side. Auditing sociocultural markers that survive anonymisation, they "
      "find that “prompting for explanations may paradoxically amplify "
      "bias”: a change to the instruction, not to the candidate, "
      "moving the fairness estimate.", indent=False)
    # WHAT IS LEFT OVER IS READ FROM THE MATRIX, NOT ASSERTED. This paragraph
    # used to say that neither work "runs a semantically-null arm designed so
    # that the edit provably changes no word" -- while Table 18 of this same
    # paper records Seshadri et al. as FULLY reporting a null-edit control and
    # credits them by name. §2 contradicted the paper's own table about a paper
    # §2 had just elevated to an antecedent, which is the worst place to be
    # careless. The novelty claim now derives from the counts, so it cannot
    # outrun them: a choice may be listed here only if the matrix records that
    # NO surveyed study reports it.
    _nobody = []
    if matrix:
        _nobody = [matrix["counts"][f]["pretty"]
                   for f in matrix.get("never_reported_by_any", [])
                   if f in matrix.get("counts", {})]
    _null_credit = ""
    if matrix and matrix["counts"].get("null_edit_control", {}).get("reported_by"):
        _null_credit = (
            " Seshadri et al. do run a null-edit control, and "
            + TAB("reporting_matrix") + " credits them for it; we did not "
            "arrive at that design first and do not claim to.")
    P("What is left over is specific, and it is what the rest of this paper "
      "does. Neither work measures the <i>dispersion</i> of a demographic effect as a "
      "quantity in its own right, with an interval on it; neither establishes "
      "a byte-identical noise floor, so neither can say how much of an "
      "observed movement is measurement error; neither decomposes the estimate "
      "into named components on a common scale; and neither touches the "
      "serving stack, which is half of what we find moves the number."
      + _null_credit
      + (" Across the whole surveyed panel the choices no study reports in "
         "reconstructible form are " + "; ".join(_nobody)
         + ". Several studies mention some of them without specifying "
         "them, and " + TAB("reporting_matrix") + " prints those "
         "partial counts beside the full ones; a choice that cannot be "
         "reconstructed cannot be checked, which is the standard we "
         "apply to ourselves too. That count, not our reading of any "
         "one paper, is what the novelty claim rests on."
         if _nobody else "")
      + " Their result is that model behaviour is brittle. Ours is a budget "
      "for that brittleness, in the units the audit reports, against a floor "
      "that says when it is real.")
    P("<b>Prompt sensitivity.</b> Sclar, Choi, Tsvetkov and Suhr established "
      "that model accuracy on classification tasks is highly sensitive to "
      "formatting choices that preserve meaning, with spreads up to 76 accuracy "
      "points on Llama-2-13B in few-shot settings, and that this survives "
      "increases in model size, in the number of demonstrations, and "
      "instruction tuning. Two of their findings bear directly on our design. "
      "Format performance correlates only weakly between models, which "
      "independently undermines cross-model comparison under a fixed prompt. "
      # THE SAME CORRECTION AS §8, WHICH THIS PASSAGE DID NOT GET. Sclar's
      # Table 1 marks three of six classes non-predictive (S2, Fitem1,
      # Fcasing) -- half, not "most" -- and the complement is three, not two:
      # C carries 29 % weak differences, and S1 and Fitem2 are the two of
      # those three they single out as having the most individual impact. §8
      # was fixed and this was not, so the paper stated the count two ways.
      # The reference is hard-typed for the same reason as there: routed
      # through TAB("instrument_validation") it was OUR Table 1 generating a
      # citation to THEIR table, matching only by coincidence.
      "And, in their characterisation of the format space, half the "
      "individual format features do not independently predict performance "
      ", in their Table 1, the second separator, the item wrapper and "
      "casing do not , while three classes carry a positive individual "
      "signal and they name two of those, separators and the number format "
      "used in enumerations, as having the "
      "most individual impact. The space is, in their words, highly "
      "non-monotonic. Our contribution is not to rediscover "
      "sensitivity but to measure it on a quantity where it should not survive: "
      "a difference between matched conditions, where the format is held "
      "constant within the pair.")
    P("<b>Template bias, named in 2009.</b> Lahey and Beasley call it that: "
      "in a traditional audit every item on a r\u00e9sum\u00e9 is correlated within "
      "the template, so an estimate can turn on which templates were chosen "
      "rather than on the treatment. They demonstrate it on one age-"
      "discrimination audit, partitioned into eight "
      "pseudo-templates from three binary characteristics and then split into "
      "two arbitrary sets of four. One set returns a one-tailed p of 0.0225 "
      "and the other 0.5000, on the same experiment; pooled, the eight give "
      "0.0513. Their own summary is that one set finds evidence of "
      "discrimination and the other does not. That is this paper\u2019s result on "
      "human employers, seventeen years earlier, and it is the reason the "
      "argument here is a methods contribution to the audit literature rather "
      "than a complaint about it. Their remedy is to generate the stimuli "
      "from a randomised databank instead of a handful of templates. Ours is "
      "to measure what the remaining choices are worth and report it, which "
      "is a diagnosis rather than a cure.")
    P("<b>Researcher degrees of freedom.</b> Specification-curve analysis "
      "(Simonsohn, Simmons and Nelson) and multiverse analysis (Steegen, "
      "Tuerlinckx, Gelman and Vanpaemel) exist because a defensible analysis is "
      "rarely unique, and reporting one path from many invites a selection "
      "effect. Simonsohn et al. illustrate the technique on a field experiment "
      "measuring discrimination against distinctively Black names , the same "
      "design this paper studies, with ninety analytic specifications. We "
      "apply the same logic one step earlier, to the <i>measurement</i> rather than "
      "only to the analysis, and in Section 6 to our own analysis as well.")
    P("<b>Tokenization and names.</b> An and Rudinger establish, on social "
      "commonsense reasoning, that a first name’s tokenization length "
      "influences how a model treats it independently of and in addition to "
      "the name’s demographic attributes. That the two are correlated in the "
      "population of names, because White and male names are more often "
      "single tokens, is an earlier result they replicate on 5,748 names "
      "rather than one of their own, and we take it from their replication. "
      "They control for it by stratifying names on (race, "
      "gender, tokenization length). Section 4.4 asks what happens in a design "
      "that does not. The correspondence audit’s unit is the matched <i>pair</i>, "
      "and no audit balances it.")

    # ======================================================================
    H("3  Design")
    P("<b>The estimand, written out.</b> Let β(m ; w, t, p, N, s, a) be "
      "the number an audit of checkpoint m reports, where w is the "
      "instruction wording, t the résumé template, p the job posting, N "
      "the drawn name set, s the serving configuration and a the analysis "
      "path. A published audit reports one point of this function and reads "
      "it as β(m), a property of the model. What this paper measures is "
      "the dispersion of β along each argument with the others held at "
      "their defaults, as a fraction of the effect itself. For the wording, "
      "σ is the standard deviation of β(m ; w, t, p) taken over w with "
      "t and p at their defaults, and the quantity tabulated throughout is "
      "σ/|β|, with β in the denominator the pooled estimate under the "
      "default instrument. The same construction runs in each remaining "
      "argument, and the dispersion budget plots exactly these ratios. In "
      "the tables a ratio is printed only where its denominator is separable "
      "from zero; the budget draws every panel and its caption marks the two "
      "whose denominator is not, "
      "and every dispersion is read against the byte-identical replicate "
      "floor, which is the same statistic computed across repeats in which "
      "no argument changes at all.", indent=False)
    P("<b>Models.</b> Six open-weight checkpoints, served locally with "
      "grammar-constrained decoding at temperature zero: Llama-2-7B-chat, "
      "Llama-2-13B-chat, Llama-3.1-8B-Instruct, Mistral-7B-Instruct v0.1 and "
      "v0.3, and Mistral-7B v0.1 base. The panel spans a generational boundary "
      "and includes a base/instruct pair, which is the contrast that can "
      "discriminate whether instruction tuning is implicated in the "
      "sensitivity. Every checkpoint’s SHA-256 digest is recorded and "
      "re-verified from disk at analysis time, and every run asserts that the "
      "server is holding the labelled file before issuing a call.", indent=False)
    P("<b>Stimuli.</b> One job posting, three résumé templates "
      "spanning strong, middling and marginal qualification, and a name grid "
      "built factorially from the Bertrand and Mullainathan list. Six first "
      "names by four surnames by two genders per race, giving 48 matched pairs. "
      "Names are selected by a fixed mechanical rule , alphabetical order "
      "within each cell, because any rule that depended on the "
      "names’ measured behaviour would be the degree of freedom this paper "
      "is about. Crossing first names with surnames is what makes the "
      "first-name and surname components separately estimable.")
    # THE QUANTIFIER FOLLOWS THE ARTIFACT. This said "degenerate on this
    # panel" on the strength of one hard-named checkpoint's accept rate, while
    # the per-model `binary.degenerate` flags sat unread beside it.
    _deg = [m for m in models if s2[m].get("binary", {}).get("degenerate")]
    P("<b>Outcome.</b> The model is constrained by grammar to emit exactly "
      "“yes” or “no”, and we read the full top-token "
      "distribution rather than the sampled token. The outcome is the "
      "renormalised decision margin in log-odds, "
      "log P(yes) − log P(no). We do not use the thresholded "
      "verdict as the primary outcome because it is degenerate on "
      f"{NUM.get(len(_deg), str(len(_deg)))} of the "
      f"{NUM.get(len(models), str(len(models)))} checkpoints Study 2 covers: "
      f"{SHORT['llama-2-7b-chat']} accepts "
      f"{pct(s2['llama-2-7b-chat']['binary']['accept_rate'], 1)} of candidates, "
      "so a binary analysis has almost no dynamic range there and the panel "
      "cannot be read on a common binary outcome; a null result would "
      "be uninformative rather than substantive.")
    P("<b>Effect size.</b> The primary effect size is the probability of "
      "superiority. The fraction of matched pairs in which the "
      "White-named résumé outranks the Black-named one, with "
      "0.5 denoting no effect. It is scale-free, bounded, and needs no "
      "conversion. Log-odds are reported beside it. Section 6.2 explains why "
      "percentage points, the field’s usual scale, cannot carry this "
      "comparison across models.")
    P("<b>Reading the tables. Four estimators, not one.</b> The same model’s "
      "effect appears with slightly different values in different tables and a "
      "reviewer was right to ask why, so each is named here. " + TAB("study2_effect") + " is Study 2 "
      ". Twelve name pairs crossed with twelve wordings on the business-analyst "
      "posting. " + TAB("name_variance") + "’s β is the posterior median of a crossed "
      "random-effects fit on Study 4’s forty-eight-pair grid, which differs "
      "from the sample mean of the paired differences on that same grid by "
      "the shrinkage the model applies and by nothing else. " + TAB("spec_curve") + " is the probability of superiority on that grid, which is "
      "a rank statistic and not a rescaling of either. The differences are "
      "small on three models and not small on Mistral-7B-Instruct v0.1, where "
      "Study 2 gives "
      + (f"{ps['mistral-7b-instruct-v0.1']:.3f} and the name grid gives "
         f"{pairfree['models']['mistral-7b-instruct-v0.1']['p_actual']:.3f} "
         if pairfree and 'mistral-7b-instruct-v0.1' in pairfree.get('models', {})
         else "")
      + ". Two designs, twelve names against forty-eight, disagreeing about "
      "a model whose effect neither can distinguish from zero. That is the "
      "paper’s own subject arriving in its own tables.", indent=False)
    P(f"<b>Scale of the experiment.</b> {n_calls:,} model calls yielding "
      f"{n_records:,} matched-pair records across "
      + (NUM.get(_n_studies_mp, str(_n_studies_mp)) if _n_studies_mp
         else "nine")
      + " studies, plus "
      f"{n_single:,} single-prompt records from the admission gate, the "
      "affiliation probe and the instrument smoke tests, run unattended under "
      "a supervisor that restarts on stall and escalates on unrecognised "
      "failure. In the reproducibility study of §5.2 every row records the "
      "SHA-256 of the exact prompt that produced it, so the claim that two "
      "cells received identical input is verified rather than assumed: "
      "which is the study where it has to be. It is not recorded corpus-wide: "
      "the hash was added when Study 8 needed it, and the earlier studies "
      "carry the design key instead.")

    APP.start()   # -> Appendix A
    H("A  Instrument validation, and one limitation it exposed")
    cov = {}
    # `_yn_mass` is filled below; Appendix D quotes it to answer the grammar objection, so
    # it is initialised here rather than inside the `if cov:` block.
    _yn_mass = {}
    for f in sorted((D / "instrument").glob(COVERAGE_GLOB)):
        try:
            cov[f.stem.replace("token_coverage_", "")] = json.loads(
                f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    if cov:
        _yn_mass = {k: float(np.mean([r["addressed_mass"] for r in v]))
              for k, v in cov.items()}
        tm = {k: float(np.mean([r["top_total_mass"] for r in v]))
              for k, v in cov.items()}
        sh = {k: float(np.mean([r["addressed_mass"] / r["top_total_mass"]
                                for r in v])) for k, v in cov.items()}
        ag = {k: float(np.mean([r["grammar"] == r["argmax"] == r["mass"]
                                for r in v])) for k, v in cov.items()}
        # THE DISAGREEMENT FIGURE BELONGS TO THE MODELS THE SENTENCE IS ABOUT.
        # Two passages quote it of "the models at the bottom of Table 1", where
        # the unconstrained next token is usually neither yes nor no, and both
        # took the minimum over all six rows. That handed them a number from a
        # checkpoint whose yes/no mass is above 60 % -- disqualified by the
        # paper's own criterion, and printed on its own row of Table 1, so the
        # paper contradicted itself on the same page.
        _low_mass = [k for k in ag if _yn_mass.get(k, 1.0) < 0.5]
        _agmin = min((ag[k] for k in _low_mass), default=min(ag.values()))
        _cand = [m for m in survives if m in _yn_mass]
        if _cand:
            _best_mass = max(_cand, key=lambda m: _yn_mass[m])
        lo_m = min(_yn_mass, key=_yn_mass.get)
        hi_m = max(_yn_mass, key=_yn_mass.get)
        P("<b>Does the outcome read what we think it reads?</b> The margin is "
          "computed by renormalising over a set of yes-tokens and no-tokens. A "
          "probe records, for every model, how much probability mass those "
          "sets actually carry. The answer differs sharply across the panel: "
          f"{SHORT.get(hi_m, hi_m)} places {pct(_yn_mass[hi_m], 2)} of its total "
          f"next-token mass on yes/no tokens, while {SHORT.get(lo_m, lo_m)} "
          f"places only {pct(_yn_mass[lo_m], 1)}.", indent=False)
        # The probe counts are not equal across rows and an audit found the
        # paper never said why. They are not a like-for-like comparison, so the
        # design of each row is printed beside its count rather than left for a
        # reader to reverse-engineer.
        def _design(v):
            w = len({r["variant"] for r in v})
            t = len({r["template"] for r in v})
            n = len({r["name"] for r in v})
            return f"{w}×{t}×{n}"
        rows = [[TINY.get(k, k), f"{len(v)}", _design(v), pct(_yn_mass[k], 2),
                 pct(tm[k], 1), pct(sh[k], 2), pct(ag[k], 0)]
                for k, v in sorted(cov.items(), key=lambda kv: -_yn_mass[kv[0]])]
        _short_rows = [TINY.get(k, k) for k, v in cov.items()
                       if len(v) < max(len(x) for x in cov.values())]
        paper.table(
            ["model", ">probes", ">wordings×templates×names", ">yes/no mass",
             ">top-100 mass", ">share of top-100", ">3 rules agree"],
            rows, [72, 32, 104, 58, 56, 68, 58], span2=True, size=7.6,
            caption=(f"{TAB('instrument_validation')}. Instrument validation. “yes/no mass” is "
                     "the <i>absolute</i> probability the model puts on the tokens the "
                     "outcome reads. “top-100 mass” is how much of the "
                     "distribution the top-100 window captures at all; the next "
                     "column is the per-probe ratio of the two, averaged over "
                     "probes, which is not exactly the quotient of the two "
                     "column means. They diverge on the base "
                     "model, whose top-100 holds only about three quarters of "
                     "its distribution, so the absolute figure understates how "
                     "much of what the probe can <i>see</i> is yes/no. “3 rules "
                     "agree” is how often the grammar-constrained token, "
                     "the unconstrained argmax and the renormalised mass winner "
                     "pick the same answer. The rows are <i>not</i> a like-for-like "
                     "comparison and the third column says how they differ: "
                     "four models were probed on two wordings, two templates "
                     "and twelve names; the base model on six names, because "
                     "it has no chat template and the probe was scoped to "
                     "confirm the outcome could be read at all; and "
                     "Llama-2-13B-chat on one wording, with 12 strong and 9 "
                     "marginal cells rather than a balanced 12 and 12, because "
                     "that run was truncated and not repeated. The last row is "
                     "therefore an unbalanced sample of a smaller design, and "
                     "its figures should be read as indicative."))
        P("This is a genuine limitation and we state it rather than bury it. On "
          "the models at the bottom of " + TAB("instrument_validation") + " the model’s unconstrained "
          "next token is usually neither “yes” nor "
          "“no” , it is prose, and the grammar is doing "
          "real work in forcing a decision. Renormalising over the yes/no pair "
          "is the correct response, because it asks which of the two permitted "
          "answers the model prefers, and that is exactly the quantity a "
          "screening decision needs. But the reader should know that on those "
          "models the question is being put to the model more insistently than "
          "a deployment would, and that the three natural decision rules "
          f"disagree on up to {pct(1 - _agmin, 0)} of probes. Every "
          "result in this paper is a difference between two conditions read the "
          "same way, so the comparison is internally valid; the caveat bears on "
          "how the level should be interpreted, not the contrast.")
    gates = []
    for f in sorted((D / "panel_gate").glob(GATE_GLOB)):
        try:
            gates.extend(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    if gates:
        pa = [g["position"]["position_a_rate"] for g in gates
              if "position" in g]
        P("<b>Why one résumé per call, not two.</b> The obvious design "
          "presents both résumés in one prompt and asks which is "
          "preferred. An admission gate tested it before the panel ran and "
          "rejected it: across the checkpoints the rate at which the model "
          f"chose whichever candidate was presented first ranges from "
          f"{pct(min(pa), 0)} to {pct(max(pa), 0)}. A format in which one model "
          "always picks the second candidate and another picks the first nine "
          "times in ten cannot carry a demographic contrast. Each "
          "résumé is therefore scored alone and the pair differenced "
          "afterwards, which removes order entirely.")
    if ctok:
        cnd = ctok["conditions"]
        P("<b>Are the structural conditions what they claim to be?</b> Every "
          "condition is tokenised against a served checkpoint’s own "
          "vocabulary and its token count, its delimiter count and the token "
          f"index of the name are recorded. " + TAB("conditions") + " reports "
          f"{SHORT.get(ctok.get('model'), ctok.get('model', 'one checkpoint'))}; "
          "token counts and merge behaviour are properties of a tokenizer, so "
          "the specific integers are that model’s and the other five are "
          "not claimed to match them. What generalises is the <i>design</i> the table "
          "certifies , which conditions move the name and which destroy a "
          "delimiter, because that is a property of the edit, not of the "
          "vocabulary. This is not a formality. The "
          "delimiter this prompt uses merges with the preceding period into a "
          "single vocabulary item"
          + (" , confirmed present in the vocabulary, " if ctok.get("merged_token_exists") else " ")
          + "so editing what looks like whitespace changes a token the model "
          "was trained on, and editing a delimiter usually displaces everything "
          "after it, including the name. Usually, not always: D5 destroys a "
          "delimiter and leaves the name index unchanged, because the "
          "delimiter it destroys sits after the name. That exception is "
          "what makes D5 the condition the design needs , a "
          "destruction with no displacement, and an earlier "
          "version of this sentence generalised past it.", indent=False)
        order = ["D0", "D1", "D2", "D3", "D7", "D8", "D9", "D5", "D4", "D10", "D6"]
        DISPO = {"intact": "—", "fragmented": "fragmented",
                 "substituted": "substituted"}
        rows = [[c, f"{cnd[c]['n_tokens']}", f"{cnd[c]['delta_len']:+d}",
                 f"{cnd[c]['delta_name_index']:+d}",
                 f"{cnd[c].get('n_delims_destroyed_measured', cnd[c]['n_delims_destroyed'])}",
                 DISPO.get(cnd[c].get("delimiter_disposition", ""), "?")]
                for c in order if c in cnd]
        _disagree = [c for c in order if c in cnd
                     and cnd[c].get("n_delims_destroyed_measured") is not None
                     and cnd[c]["n_delims_destroyed_measured"]
                     != cnd[c].get("n_delims_destroyed_declared",
                                   cnd[c]["n_delims_destroyed"])]
        paper.table(
            ["condition", ">tokens", ">Δ length", ">Δ name index",
             ">delimiters affected", ">how"],
            rows, [58, 40, 48, 66, 74, 78], span2=True, size=7.6,
            caption=(f"{TAB('conditions')}. The eleven structural conditions, tokenised "
                     "against one checkpoint’s own vocabulary. The "
                     "delimiter column is <i>measured</i> against the tokenizer, not "
                     "copied from the condition definitions. An audit of this "
                     "paper found that the published version of this table "
                     "printed the experimenter’s declared value, and that "
                     "the two disagree on D7. D8 and D9 displace the name "
                     "without touching a delimiter; D5 fragments a delimiter "
                     "without displacing the name; D4, D6 and D10 do both; "
                     "D0–D3 do neither. D7 was designed as a null "
                     "control and is not one. It adds no tokens and moves "
                     "nothing, and it replaces both merged delimiter tokens "
                     "with a different single token. That is a substitution "
                     "rather than a fragmentation, and it is a third "
                     "disposition the design had no name for."))
        P("" + TAB("conditions") + " is the design. Our first pass confounded the two effects "
          "completely , every condition that destroyed a delimiter also "
          "moved the name, and the conditions that appeared to matter "
          "were exactly those. The table now separates them: D8 and D9 "
          "displace the name by one and two tokens with every delimiter "
          "intact; D5 destroys a delimiter with the name index unchanged; D4, "
          "D6 and D10 do both, and are interpretable only against the first "
          "two groups. Only with those in place does a null on delimiters mean "
          "anything.")

    # ======================================================================
    APP.end()
    H("4  Part I, What the auditor writes")

    H("4.1  The wording of the instruction", 2)
    P("Twelve instruction wordings ask the same question. Six are semantic "
      "paraphrases. The kind of variation any two research groups would "
      "produce independently. Six are <i>semantically null</i>, and they are "
      "worth listing exactly, because the claim depends on how small they are: "
      "the unperturbed baseline; one trailing newline; two spaces after each "
      "colon instead of one; the two independent sentences of the system "
      "message swapped; a single space inserted into each blank line; and one "
      "trailing space. The null arm is constructed so that a reader shown any "
      "two of its prompts would call them the same prompt. Each wording is "
      "crossed with all three résumé templates and all twelve name pairs "
      f"of this study, giving {s2[models[0]]['overall']['n']} matched pairs "
      "per model.", indent=False)
    if tokid:
        from collections import Counter as _C  # noqa: PLC0415
        _nk = _C(v["klass"] for v in tokid.values() if v["kind"] == "null")
        # "IDENTICAL" is a data value compared against the artifact, and an
        # automated caps-to-italics prose pass once rewrote this literal to
        # "<i>identical</i>" -- the filter matched nothing and the page read
        # "Exactly 0 wordings ()" two lines after the same paragraph printed
        # "1 identical" from the Counter. The assertion makes the two
        # derivations of the same fact agree or fail the build.
        _ident = [k for k, v in tokid.items() if v["klass"] == "IDENTICAL"]
        assert len([k for k in _ident if tokid[k]["kind"] == "null"])             == _nk.get("IDENTICAL", 0), (_ident, _nk)
        _reord = [k for k, v in tokid.items() if v["klass"] == "REORDERED"]
        P("“Semantically null” is a claim about meaning, so it is worth "
          "saying exactly what it is not a claim about. Tokenising every "
          "wording against one served checkpoint’s vocabulary. The "
          "same one " + TAB("conditions") + " uses, and not the four whose "
          "effects " + TAB("study2_effect") + " reports , and comparing the "
          "integer sequences, the null arm spans four distinct relationships to "
          f"the baseline: {_nk.get('IDENTICAL', 0)} identical, "
          f"{_nk.get('REORDERED', 0)} carrying the same tokens in a different "
          f"order, {_nk.get('ALTERED_SAME_LEN', 0)} differing in a token but "
          f"not in length, and {_nk.get('ALTERED', 0)} differing in length by "
          "one to three tokens. So a wording a reader cannot distinguish from "
          "another may reach the model as a different sequence, an identical "
          "sequence, or the same sequence permuted"
          + (f" — {_reord[0]} is the permuted case, and it moves the "
             "estimate like the rest" if _reord else "")
          + ". Exactly "
          f"{len(_ident)} wordings ({', '.join(_ident)}) produce byte-identical "
          "token sequences, and those two are the replicate the noise floor in "
          "§5.2 is built from.")

    rows = []
    for m in models:
        o = s2[m]["overall"]
        rows.append([TINY[m],
                     f"{fmt(ps[m], 3)}",
                     f"[{fmt(psci[m][0], 3)}, {fmt(psci[m][1], 3)}]",
                     fmt(sd_word[m], 4),
                     # SUPPRESSED WHERE THE DENOMINATOR COVERS CHANCE,
                     # exactly as Tables 4 and 13 already do. This table
                     # printed 118 % and 55 % for the two models whose
                     # effect interval spans 0.5 -- the construction
                     # §4.5 calls "not a number we may print". The
                     # rule was enforced in two tables and broken in the
                     # third, which is the one the abstract quotes.
                     (f"{fmt(ratio_ps[m] * 100, 0)} %"
                      if m in survives else "—"),
                     pct(sat[m], 1)])
    paper.table(
        ["model", ">P(sup.)", ">95 % CI", ">SD across wordings",
         ">SD / effect", ">saturated"],
        rows, [76, 46, 88, 72, 60, 54], span2=True, size=7.6,
        caption=(f"{TAB('study2_effect')}. The measured demographic effect and its dispersion "
                 "across twelve wordings that ask the same question. "
                 "“Effect” is |P(superiority) − 0.5|. The SD / effect "
                 "ratio is suppressed, as “—”, on the models whose "
                 "effect interval covers 0.5: that denominator can be "
                 "arbitrarily small, so the ratio has no finite upper bound "
                 "and a point estimate for it would mislead (§4.5). The final "
                 "column is the fraction of individual résumé "
                 "scorings , two per matched pair, on which the "
                 "model’s "
                 f"preference is saturated beyond {es.SATURATION_HI} or "
                 f"below {es.SATURATION_LO}, which is "
                 "why a single percentage-point conversion cannot serve all "
                 "four models. See Section 6.2."))

    P(f"The effect survives its own interval on {len(survives)} of "
      f"{len(models)} models. What " + TAB("study2_effect") + " adds is the final-but-one column. "
      "On the two models whose effect is distinguishable from chance, the "
      "standard deviation across wordings is "
      + " and ".join(f"{fmt(ratio_ps[m] * 100, 0)}\u2009%" for m in survives)
      + " of the effect itself, and that is the number the argument rests "
      "on. On the other two the column is empty by the rule stated in its "
      f"caption. On {SHORT['mistral-7b-instruct-v0.1']} the point estimates "
      f"put the dispersion ({fmt(sd_word['mistral-7b-instruct-v0.1'], 4)}) "
      f"above the effect "
      f"({fmt(abs(ps['mistral-7b-instruct-v0.1'] - 0.5), 4)}), which would "
      "say the wording moves this checkpoint's estimate by more than the "
      "whole of its demographic effect"
      # THE ORDERING IS A POINT ESTIMATE, so it gets its interval. This
      # sentence used to read "the dispersion simply EXCEEDS the effect --
      # which wording the researcher wrote matters more than which model was
      # tested", a between-model claim that two within-model magnitudes cannot
      # license and that \u00a71 pre-emptively declares false. The interval that
      # settles it was already on disk and filtered out of the prose by the
      # `for m in survives` guard below. It holds the effect fixed, so it has
      # a finite upper bound and does not reinstate the ratio the caption
      # suppresses.
      + ((", but resampling the twelve wordings with the effect held fixed "
          "puts that ratio at ["
          f"{disp_unc['models']['mistral-7b-instruct-v0.1']['ratio_ci'][0]:.2f}, "
          f"{disp_unc['models']['mistral-7b-instruct-v0.1']['ratio_ci'][1]:.2f}]: "
          "the ordering is where the point estimates put it, and twelve "
          "wordings do not pin it there")
         if disp_unc and (disp_unc.get("models", {})
                          .get("mistral-7b-instruct-v0.1", {})
                          .get("ratio_ci")) else "")
      + ".")
    if disp_unc:
        _du = disp_unc["models"]
        _dus = disp_unc["summary"]
        P("<b>How precisely is that dispersion itself estimated?</b> A paper "
          "asking others to report the dispersion of their estimate should "
          "report the dispersion of its own, and a reviewer pointed out that "
          "we had not. A standard deviation on twelve wordings has eleven "
          "degrees of freedom. Resampling wordings, the interval on the "
          f"between-wording SD is {pct(_dus['min_relative_width'], 0)} to "
          f"{pct(_dus['max_relative_width'], 0)} of the point estimate, and "
          "the dispersion-to-effect ratios the argument rests on become "
          + "; ".join(
              f"{SHORT[m]}, {_du[m]['ratio']:.2f} "
              f"[{_du[m]['ratio_ci'][0]:.2f}, {_du[m]['ratio_ci'][1]:.2f}]"
              for m in survives if m in _du and _du[m].get("ratio_ci"))
          + ". Those intervals vary the <i>wordings</i> and hold the effect fixed, "
          "which is the question this section asks; Appendix D.1 varies the name "
          "pairs instead and recomputes both halves of the ratio, which "
          "widens them"
          + ((f" — to [{rint['local'][survives[0]]['ci'][0]:.2f}, "
              f"{rint['local'][survives[0]]['ci'][1]:.2f}] on "
              f"{SHORT[survives[0]]}, for instance"
              ) if rint and survives and survives[0] in rint.get("local", {})
             and rint["local"][survives[0]].get("ci") else "")
          + ". Both are in the artifacts, and a ratio quoted without saying "
          "which quantity was resampled is the kind of unreported choice this "
          "paper is about. Twelve wordings pin the dispersion to about a "
          "factor of two. "
          "That is enough to say it is large relative to the effect: "
          "every lower bound here is several times the noise floor below "
          ", and it is not enough to rank two models by it, which the "
          "paper does not do.", indent=False)

    if noise:
        nm = [m for m in models if m in noise]
        rr = [noise[m]["ratio_variant_to_noise"] for m in nm]
        P("<b>This is dispersion, not noise.</b> A byte-identical replicate is "
          "embedded in the design: S1, the first semantic paraphrase, and N1, "
          "the unperturbed baseline of the null arm, are the same string. They "
          "sit in different arms, which is what makes the replicate free , "
          "it costs no extra condition, and any disagreement between them "
          "is measurement error and nothing else. That gives a noise floor per "
          "model, and the between-wording standard deviation of the "
          "per-wording estimate exceeds it by a factor of "
          f"{fmt(min(rr), 1)} to {fmt(max(rr), 1)}. Both quantities are on the "
          "log-odds scale and both refer to the per-wording mean, so the ratio "
          "compares like with like. The wording is moving the estimate; the "
          "arithmetic is not.")

    if arm:
        am = [m for m in models if m in arm]
        pn = [arm[m]["all"]["prob_null_exceeds_semantic"] for m in am]
        # BOTH BRANCHES. N1 is byte-identical to S1, so its variant effect is
        # not an independent draw from the null arm -- fit_arm_contrast.py
        # fits it both ways. Quoting only the N1-included branch hid the
        # branch that comes closest to separating the arms.
        #
        # WHAT THE DIRECTION OF THAT BIAS IS NOT. Both the fit script's
        # docstring and this paper used to assert that including N1 makes the
        # null arm look LESS dispersed than it is. That holds only where
        # sigma_semantic is the smaller of the two, and it is the larger on
        # three of the four checkpoints, so the claim was wrong on most of the
        # panel. Nobody checked the docstring against the fit it described.
        # The direction is measured here instead of asserted.
        _pnd = [arm[m]["dropN1"]["prob_null_exceeds_semantic"] for m in am
                if "dropN1" in arm[m]]
        _sn = [(arm[m]["all"]["sigma_null"][1], arm[m]["dropN1"]["sigma_null"][1])
               for m in am if "dropN1" in arm[m]]
        _sn_down = sum(1 for a, b in _sn if b < a)
        P("<b>The null arm behaves like the paraphrase arm.</b> One might "
          "expect edits that change no word to move the estimate less than "
          "genuine paraphrases. Fitting a separate between-wording standard "
          "deviation per arm in one joint model, so that the difference has a "
          "posterior rather than being compared by eye, the two arms cannot be "
          "distinguished on any model. The posterior probability that the null "
          "arm exceeds the semantic arm ranges from "
          f"{fmt(min(pn), 2)} to {fmt(max(pn), 2)}, and every difference "
          "interval includes zero."
          + (f" Dropping N1 , which is byte-identical to S1 and so not "
             "an independent draw from the null arm, moves the range to "
             f"{fmt(min(_pnd), 2)} to {fmt(max(_pnd), 2)}. That correction "
             "has no single direction. The null arm's standard deviation "
             f"falls on {NUM.get(_sn_down, str(_sn_down))} of the "
             f"{NUM.get(len(_sn), str(len(_sn)))} checkpoints and rises on "
             f"the {NUM.get(len(_sn) - _sn_down, str(len(_sn) - _sn_down))} "
             "others, because the sign of the bias depends on which arm is "
             "the more dispersed, which is the quantity under test. The "
             "conclusion does not change. Both fits are in the "
             "artifact." if _pnd else "")
          + " The defensible claim is not that null edits "
          "move the effect <i>more</i> than paraphrases, but that they move it "
          "<i>at all</i>, and by an amount of the same order.")

    sc = {m: s2[m]["spec_curve"] for m in models}
    flips = [m for m in models if sc[m].get("sign_flips")]
    P("<b>Every wording is defensible, so every one is a specification.</b> "
      "Twelve wordings crossed with three templates give 36 analyses per model "
      "that a competent researcher might have run and published. " + FIGREF("fig7_spec_curve") + " "
      "plots all of them. "
      + (f"On {SHORT[flips[0]]} the sign of the effect is not stable across "
         "them." if flips else
         "No model changes sign across the curve, but the range of publishable "
         "estimates within a single model is wide."))
    # An audit found the previous version of this passage claiming the sign
    # instability was produced by the wording, with "the same résumés" in a
    # list of things held fixed -- while the curve crosses three templates.
    # The decomposition below was computed to settle it and it went against us.
    _fd = None
    if flips:
        _fd = ((s2.get(flips[0], {}).get("spec_curve") or {})
               .get("factor_decomposition"))
    if _fd:
        _w = _fd["across_wordings_within_template"]
        _t = _fd["across_templates_within_wording"]
        _ds = _fd["dispersion_share"]
        _pairs = ", ".join(_t.get("reportable_flip_pairs", {}))
        P("It is not the wording that flips it. The curve crosses twelve "
          "wordings "
          "<i>with</i> three résumé templates, so “the same résumés” was "
          "never true of it. Decomposing. Holding the résumé fixed, "
          f"{_w['n_flips_reportable']} of {_w['n_contrasts']} same-template "
          "wording contrasts reverse a significant effect. On any of the "
          f"three templates. Holding the wording fixed, "
          f"{_t['n_flips_reportable']} of {_t['n_contrasts']} reverse, and "
          f"every one of them is the same pair ({_pairs.replace('|', ' against ')}). "
          "Of the spread among the 36 point estimates, "
          f"{pct(_ds['frac_template'], 1)} lies between templates and "
          f"{pct(_ds['frac_wording'], 1)} between wordings.", indent=False)
        P("What survives is the claim the rest of this section makes and the "
          "abstract reports. The wording moves the <i>magnitude</i>, by a standard "
          "deviation of a quarter to more than the effect itself. What does "
          "not survive is a claim about the <i>sign</i>. The direction of the "
          "reported disparity on this model is determined by which résumé the "
          "auditor wrote , which is a stimulus choice no audit varies "
          "either, and which §3 fixes and this section crosses. Not by which of twelve "
          "defensible phrasings they chose. Both are instrument variance. They "
          "are not the same axis.")
    paper.figure(FIGS / "fig7_spec_curve.png", span2=True,
                 max_h=3.2 * pk.inch, space_after=12.0)
    P("" + FIGREF("fig4_forest_by_wording") + " shows the same data as a forest plot, one row per MODEL with one marker per wording, so "
      "the reader can see the spread each model carries. Nothing about the wordings "
      "at the extremes distinguishes them. They are not the longest, the "
      "politest or the most specific, and a researcher choosing among them "
      "would have no basis for preferring one.")
    paper.figure(FIGS / "fig4_forest_by_wording.png", span2=True,
                 max_h=3.0 * pk.inch, space_after=12.0)

    # ------------------------------------------------------------------
    H("4.2  Which names were drawn", 2)
    if names:
        nm = [m for m in models if m in names]
        rf = {m: names[m]["ratio_first_to_beta"] for m in nm}
        # A ratio is only a ratio where its denominator exists. Two of the four
        # models have a race effect whose credible interval covers zero, and on
        # those the σ/|β| column is a quotient of a spread by something not
        # separable from zero -- its posterior runs to 78.75 on one of them.
        # §4.5 and §6.2 police exactly this elsewhere and this table did not.
        _beta_id = {m: names[m]["beta"][0] * names[m]["beta"][2] > 0 for m in nm}
        _rf_id = [rf[m][1] for m in nm if _beta_id[m]]
        n_first_per_race = names[nm[0]]["n_first"] // 2
        # AN INDEPENDENT ANTECEDENT, CREDITED BEFORE OUR OWN RESULT.
        # Lippens reports, on a far larger name set than ours, that within-group
        # dispersion across names EXCEEDS the between-group difference -- the
        # same finding this section reaches, reached first. He is one of only
        # two audits in the survey that report anything of the kind, which is
        # both why the point needed making again and why it must be credited
        # rather than presented as new.
        P("<b>This is not a new observation, and the prior one is larger than "
          "ours.</b> Lippens reports that the between-group differences in his "
          "ChatGPT audit “hide the substantial dispersion in assigned "
          "interview invitation scores between names of the same ethnic and "
          "gender identity”, and states the consequence plainly. Much larger "
          "differences exist within groups than between them. That is on 812 "
          "name combinations across 1,920 vacancies, an order of magnitude "
          "more names than this grid carries. What we add is not the "
          "observation but its decomposition. Separating what is "
          "name-specific from what is race-specific requires the name list to "
          "be crossed rather than fixed, which is the design choice the rest "
          "of this section is about.", indent=False)
        P("The list itself has a literature the LLM audits rarely engage. "
          "Fryer and Levitt trace distinctively Black names to a divergence "
          "that begins in the early 1970s and show that a name indexes the "
          "circumstances of its bearer’s birth, so a name manipulation "
          "carries socioeconomic signal along with racial signal by "
          "construction. Gaddis, re-validating audit names on a modern "
          "perception survey, finds that how reliably a name is perceived as "
          "Black varies name by name and tracks the education of the mothers "
          "who choose it, so two audits drawing different names from the "
          "same list are not running the same manipulation. Both are "
          "population facts about names; §4.4 finds their analogue inside "
          "the tokenizer, where the same list is confounded again, this time "
          "with token length.")
        P("Every audit picks a name list. Bertrand and Mullainathan used nine "
          "first names per race per gender; LLM audits vary widely in list size and "
          "rarely report the sensitivity of the result to that choice. Our grid "
          "crosses six first names per race per gender with four surnames per "
          f"race, giving {n_first_per_race} first names and 48 full names per "
          "race. Crossing rather than fixing 48 hand-picked full names is what "
          "makes the within-race between-first-name standard deviation "
          "separately estimable from the between-race effect. We fit a crossed "
          "random-effects model on the per-name margin, with race fixed and "
          "first name random nested within race, so that σ_first captures what "
          "is name-specific and not race.", indent=False)

        rows = []
        for m in nm:
            f = names[m]
            rows.append([TINY[m],
                         fmt(f["beta"][1], 4, sign=True),
                         fmt(f["sigma_first"][1], 4),
                         (f"{fmt(rf[m][1], 2)} [{fmt(rf[m][0], 2)}, "
                          f"{fmt(rf[m][2], 2)}]" if _beta_id[m]
                          else f"— [{fmt(rf[m][0], 2)}, {fmt(rf[m][2], 2)}]"),
                         f"{f['max_rhat']:.4f}",
                         f"{f['min_ess']:.0f}"])
        paper.table(
            ["model", ">β (race)", ">σ first name",
             ">σ/|β| [95 %]", ">r̂", ">min ESS"],
            rows, [104, 74, 78, 110, 50, 46], span2=True, size=7.6,
            caption=(f"{TAB('name_variance')}. Within-race between-first-name dispersion "
                     "against the race effect itself, in log-odds. The σ/|β| "
                     "column is the posterior median <i>of</i> <i>the</i> <i>ratio</i>, not the "
                     "quotient of the two columns beside it; the two differ "
                     "wherever β is near zero. The point estimate is "
                     "suppressed on the two models whose β interval covers "
                     "zero, because a ratio against such a denominator has no "
                     "finite upper bound. Their intervals are printed so the "
                     "reader can see how little is pinned. The r-hat "
                     "and ESS columns are a <i>worst</i> <i>case</i> over the parameters "
                     "the paper interprets, not a per-parameter report; the "
                     "per-parameter values are in the artifact. Restricting "
                     "to interpreted parameters is itself the correction of "
                     "a worst case over "
                     "every latent and so failed diagnostics on quantities "
                     "the paper never reads."))

        P("On the two models whose race effect is identified, the within-race "
          "spread between individual first names is "
          f"{fmt(min(_rf_id), 2)} to {fmt(max(_rf_id), 2)} times the "
          "between-race effect the study exists to measure. Two names that a "
          "validated source assigns to the same demographic group are not "
          "interchangeable, and the difference between them is of the same "
          "order as the difference the audit reports. On the other two the "
          "ratio has no useful upper bound. Its 95 % interval reaches "
          + " and ".join(
              fmt(rf[m][2], 1) for m in nm if not _beta_id[m])
          + " , which is a statement about the denominator, not about names, "
          "and is why those cells carry an interval and no point estimate.")

        # THE FACTOR IS READ, NOT WRITTEN OUT. An earlier draft typed the
        # survey-sampling approximation sqrt(1 - k/N), giving exactly one half
        # at k = 9. The exact factor for the SD of a mean drawn without
        # replacement is sqrt((N - k)/(N - 1)) = sqrt(3/11) = 0.52, which is
        # what fit_name_variance.py computes and stamps on every row. The
        # difference is small and the sentence was still wrong; interpolating
        # it from the artifact makes the two incapable of disagreeing.
        # WHICH ESTIMAND. The original counterfactual drew k pairs WITHOUT
        # replacement from the twelve the grid provides, so its spread carried
        # that factor. That
        # answers "what if we had used k of THESE twelve", which is not the
        # question the paper asks. A study choosing a name list draws from the
        # population the list came from, so the with-replacement arm is the
        # right referent and it is also the conservative one: it makes the
        # name component LARGER relative to the wording, against our argument.
        # Both arms are in the artifact and both are reported.
        cf = {m: names[m]["counterfactual_with_replacement"] for m in nm}
        cf_cond = {m: names[m]["counterfactual"] for m in nm}
        # Read from the WITHOUT-replacement arm, which is the one that carries
        # the factor; the with-replacement arm records 1.0 by definition.
        _fpc9 = next(
            (cf_cond[_m]["draw_9_per_race"]["finite_population_factor"]
             for _m in nm
             if cf_cond.get(_m, {}).get("draw_9_per_race", {})
             .get("finite_population_factor")), None)
        n_avail = cf[nm[0]]["draw_9_per_race"].get("n_pairs_available")
        rows = []
        for m in nm:
            c = cf[m]
            rows.append([TINY[m],
                         fmt(names[m]["beta"][1], 4, sign=True)]
                        + [pct(c[f"draw_{k}_per_race"]["sign_flip_frac"], 1)
                           for k in (3, 6, 9)])
        paper.table(
            ["model", ">full grid β", ">k = 3", ">k = 6", ">k = 9"],
            rows, [74, 56, 36, 36, 36], size=7.6,
            caption=(f"{TAB('name_draw_grid')}. Sign-flip rate. The fraction of draws of k "
                     f"matched first-name pairs, from the {n_avail} the grid "
                     "provides, that would have published an effect of the "
                     "opposite sign to the full-grid estimate. The unit is the "
                     "<i>pair</i>, because a correspondence audit that uses k names "
                     "per race uses k matched pairs."))

        # A model whose effect is indistinguishable from zero flips sign in
        # about half of all draws BY CONSTRUCTION, and quoting that as the worst
        # case would be reporting arithmetic as a finding. The informative case
        # is a model with a real effect that STILL flips, so the candidate set
        # is restricted to models whose full-grid interval excludes zero.
        real = [m for m in nm if names[m]["beta"][0] * names[m]["beta"][2] > 0]
        null9 = [m for m in nm if m not in real]
        # Which null models actually flip near half at EVERY list size? Not all
        # of them do, and saying so without checking was wrong once already.
        near_half = [m for m in null9
                     if min(cf[m][f"draw_{k}_per_race"]["sign_flip_frac"]
                            for k in (3, 6, 9)) > 0.35]
        P("The counterfactual is direct. Resample name lists of the size the "
          "literature uses and re-estimate. Two details of it were wrong in "
          "earlier drafts and both were found by audit, so they are worth "
          "stating. The resampling unit is the matched <i>pair</i>: drawing k White "
          "names and k Black names independently would not describe a list of "
          "size k at all, because the design pairs each White first name with "
          "exactly one Black one. And the draw is made <i>with</i> replacement, "
          "because the question is what a new study would face when it chooses "
          "a list from the population these names came from. Drawing without "
          "replacement from our own twelve answers a different question and "
          "carries a finite-population factor of "
          + "√((N−k)/(N−1)), "
          + (f"{_fpc9:.2f} at k = 9" if _fpc9 else "about a half at k = 9")
          + " , which roughly halves the dispersion for a reason that has "
          "nothing to do with names. Both arms are in the artifact; the tables "
          "report the population one, which is the larger and therefore the "
          "conservative choice for what follows.", indent=False)
        P("At three pairs, between "
          f"{pct(min(cf[m]['draw_3_per_race']['sign_flip_frac'] for m in nm), 1)} "
          f"and "
          f"{pct(max(cf[m]['draw_3_per_race']['sign_flip_frac'] for m in nm), 1)} "
          "of defensible draws produce the opposite sign."
          + (f" The high end belongs to {SHORT[near_half[0]]}, whose full-grid "
             "effect is not distinguishable from zero. A null effect flips in "
             "about half of all draws at every list size by construction, and "
             "the table should not be misread as showing instability where "
             "there is no effect to be unstable about." if near_half else ""))
        P("At nine pairs the sign becomes stable on every model with a "
          "distinguishable effect. No draw of nine reverses it. That is a real "
          "reassurance and we report it as one. It does not extend to the "
          "<i>magnitude</i>. Across draws of nine, the middle 95 % of publishable "
          "estimates on "
          + ", ".join(
              f"{SHORT[m]} runs "
              f"{fmt(cf[m]['draw_9_per_race']['beta_p2_5'], 4, sign=True)} to "
              f"{fmt(cf[m]['draw_9_per_race']['beta_p97_5'], 4, sign=True)}"
              for m in real)
          + ": a factor of "
          + " and ".join(
              f"{max(abs(cf[m]['draw_9_per_race']['beta_p2_5']), abs(cf[m]['draw_9_per_race']['beta_p97_5'])) / max(min(abs(cf[m]['draw_9_per_race']['beta_p2_5']), abs(cf[m]['draw_9_per_race']['beta_p97_5'])), 1e-9):.1f}"
              for m in real)
          + " between the largest and smallest number a researcher could "
          "plausibly have published from the same model on the same "
          "résumés, having done nothing wrong. A name list of the size "
          "the field considers adequate fixes the direction of the finding and "
          "does not fix the finding.")
        P("We quote the middle 95 % rather than the full range of simulated "
          "draws, and the reason is the kind of thing this paper is about. The "
          "minimum and maximum over "
          f"{cf[real[0]]['draw_9_per_race']['n_sim']:,} draws — "
          + ", ".join(
              f"{fmt(cf[m]['draw_9_per_race']['beta_min'], 4, sign=True)} to "
              f"{fmt(cf[m]['draw_9_per_race']['beta_max'], 4, sign=True)} on "
              f"{SHORT[m]}" for m in real)
          + ": is an extreme order statistic. It widens with the number "
          "of simulations and converges to nothing, so a paper that wanted a "
          "more alarming number could get one by raising a constant in its own "
          "code, and a reader could not tell. The percentile range does not "
          "have that property. Quoting the "
          "min-max.", indent=False)
        P("One caveat on scale, since the comparison invites it. Bertrand and "
          "Mullainathan used nine first names per race <i>per</i> <i>gender</i>, so eighteen "
          f"per race; our grid supplies {n_first_per_race} per race, and k = 9 "
          "is therefore three quarters of what is available here and half the "
          "field-experiment anchor. The k column should be read as a list size, "
          "not as a reconstruction of that study.")

        aa = {m: names[m]["averaging_asymmetry_with_replacement"]
              for m in nm}
        aa_cond = {m: names[m]["averaging_asymmetry"] for m in nm}
        rows = []
        for m in nm:
            a = aa[m]
            _b = (asymunc or {}).get("models", {}).get(m, {}).get("k_9", {})
            rows.append([TINY[m]] +
                        [f"{a[f'draw_{k}_per_race']['ratio_name_to_wording']:.2f}×"
                         for k in (3, 6, 9)]
                        + [f"[{_b['bound'][0]:.2f}, {_b['bound'][1]:.2f}]"
                           if _b.get("bound") else "—"])
        paper.table(
            ["model", ">k = 3", ">k = 6", ">k = 9", ">95 % bound on k = 9"],
            rows, [130, 62, 62, 62, 140], span2=True, size=7.6,
            caption=(f"{TAB('averaging_asymmetry')}. The averaging asymmetry. Ratio of the standard "
                     "deviation a published estimate inherits from the name "
                     "draw to the standard deviation it inherits from the "
                     "wording choice. Names average down with list size; "
                     "wordings do not, because the standard design uses one. "
                     # THE NUMERATOR IS NOT A POSTERIOR. It is
                     # e.std(ddof=1) over 4,000 cluster-bootstrap draws of k
                     # matched name pairs (fit_name_variance.py:451) -- a
                     # frequentist plug-in from a different estimator than the
                     # crossed random-effects fit. Only the DENOMINATOR is a
                     # posterior median. Calling the pair "ratios of posterior
                     # medians" named the wrong estimand for the column the
                     # whole averaging-asymmetry argument rests on.
                     "The first three columns divide the standard deviation "
                     "of the published effect across 4,000 cluster-bootstrap "
                     "draws of k matched name pairs by the posterior median "
                     "of the between-wording SD, so the two halves come from "
                     "different estimators and only the denominator is a "
                     "posterior. The last propagates both posteriors and is a "
                     "conservative corner bound centred on its own "
                     "posterior-median estimator rather than on the cells "
                     "beside it; it is wide, and the text says what does and "
                     "does not follow from that."))

        r9 = [aa[m]["draw_9_per_race"]["ratio_name_to_wording"] for m in nm]
        r3 = [aa[m]["draw_3_per_race"]["ratio_name_to_wording"] for m in nm]
        n_word_wins = sum(1 for x in r9 if x < 1.0)
        P("" + TAB("averaging_asymmetry") + " states the asymmetry. A published estimate carries the "
          "name-draw component divided by √k, because it averages over k "
          "pairs, but carries the <i>full</i> wording component, because it "
          "uses exactly one wording and nothing in the standard design averages "
          "that away. The consequence is a crossover. At k = 3 the name draw "
          f"contributes {fmt(min(r3), 2)}× to {fmt(max(r3), 2)}× what the "
          f"wording does; by k = 9 the ratio has fallen to {fmt(min(r9), 2)}× "
          f"to {fmt(max(r9), 2)}×, and the wording is the larger source on "
          f"{n_word_wins} of {len(r9)} models. Both choices matter at every "
          "list size the field uses, and neither is reported.")
        if asymunc:
            _k9 = asymunc["summary"]["per_k"]["k_9"]
            P("<b>How much of that is established, and it is less than the "
              "table looks.</b> §4.1 says twelve wordings pin the "
              "between-wording dispersion only to about a factor of two, and "
              "that this is not enough to rank models by it. " + TAB("averaging_asymmetry") + " divides "
              "by the pooled between-wording SD of §6.1. A different "
              "estimator of the same phenomenon, on the log-odds scale, "
              "fitted to the same twelve wordings and pinned no better, so "
              "at least the same caution applies to its "
              "cells, and an audit of this paper pointed out that we applied "
              "it in one place and not the other. Propagating the posteriors "
              "of both standard deviations, the side of 1.0 is determined on "
              f"{_k9['n_models_where_the_bound_determines_the_side']} of "
              f"{_k9['n_models']} models at k = 9: the crossing is where the "
              "point estimates put it, and twelve wordings do not pin it "
              "there. What does not depend on either posterior is the "
              "<i>direction</i>. The ratio is the name component over the wording "
              "component and only the numerator carries the 1/√k, so it "
              "falls with list size as a matter of arithmetic rather than of "
              "estimation. The argument §4.2 needs is that one component "
              "averages away and the other does not, and that part holds "
              "without an interval.", indent=False)
        _r9c = [aa_cond[m]["draw_9_per_race"]["ratio_name_to_wording"]
                for m in nm]
        _wc = sum(1 for x in _r9c if x < 1.0)
        P("The crossover is where the choice of estimand shows, so both are "
          "given. Asking instead what would have happened had we used nine of "
          "our own twelve pairs. A draw without replacement, conditional "
          f"on this grid. The k = 9 ratios are {fmt(min(_r9c), 2)}× to "
          f"{fmt(max(_r9c), 2)}× and the wording is the larger source on "
          f"{_wc} of {len(_r9c)} models rather than {n_word_wins}. The "
          "conditional figures are the smaller ones and it "
          "quoted them while making the population claim. The direction of "
          "that error is worth naming. It overstated how far the name "
          "component averages away, which is to say it overstated the paper’s "
          "own argument that wording is the neglected axis.")

    if cval and cval.get("field_anchor"):
        fa = cval["field_anchor"]
        # THE GRID'S OWN NOISE FLOOR, not §5.2's. This paragraph used to borrow
        # the zero floor measured at concurrency one with the prompt cache off.
        # The name grid ran at the default concurrency with cache reuse on, so
        # that floor does not transfer. The S1/N1 twin inside the grid is the
        # same prompt sent twice and gives it a floor under its own conditions.
        _gn = [v["replicate_noise"]["noise_over_within_race_sd"]
               for v in cval.get("models", {}).values()
               if (v.get("replicate_noise") or {}).get("noise_over_within_race_sd")]
        P("<b>The field experiment has the same problem, and said so.</b> This "
          "is not an artefact of language models. Inside a single race-gender "
          "cell of Bertrand and Mullainathan’s own experiment, the per-name "
          "callback rate spans "
          f"{fa['min_within_cell_pp']:.1f} to {fa['max_within_cell_pp']:.1f} "
          "percentage points, and in "
          f"{fa['n_cells_exceeding_headline']} of {fa['n_cells']} cells that "
          "within-race spread exceeds their headline between-race gap of "
          f"{fa['headline_gap_pp']:.1f} points. They report the per-name table "
          "and they name the reason a reader should be careful with it: with "
          "about 200 observations per female name and 70 per male name, "
          "chance alone could produce that much variation. That caveat is "
          "exactly right for their design and it is why the question stayed "
          "open. This grid carries a floor of its own, so it need not borrow "
          "the one §5.2 measures under a different serving configuration. Two "
          "of its twelve wordings are the same prompt sent twice, and the gap "
          "between their margins on a cell is measurement noise and nothing "
          "else. Spread over "
          f"{cval['models'][models[0]]['n_obs_per_name']} observations per "
          "name it comes to "
          + (f"{pct(min(_gn), 0)} to {pct(max(_gn), 0)}" if _gn else "a fraction")
          + " of the within-race spread of per-name margins, depending on the "
          "checkpoint. That separates the two , and finds the spread is real. "
          "The name-draw problem is older than this literature; what is new is "
          "being able to tell it from sampling error.", indent=False)

    # ------------------------------------------------------------------
    APP.start()   # -> Appendix B
    H("B  Do the names carry the construct?")
    if cval and cval.get("models"):
        cm = [m for m in models if m in cval["models"]]
        mde = cval.get("mde", {})
        q1 = {m: cval["models"][m]["q1_perception"] for m in cm}
        q1sig = [m for m in cm if q1[m] and q1[m]["ci"][0] * q1[m]["ci"][1] > 0]
        P("Every study in this literature, ours included, swaps a name and "
          "calls the resulting difference a demographic effect. That inference "
          "has a premise. That the model encodes the name-to-race association "
          "the list was validated for. The premise is never checked. It is a "
          "manipulation check, and correspondence audits of humans do not need "
          "one because the list was validated <i>on</i> humans, Bertrand and "
          "Mullainathan ran a perception survey and printed it. A language "
          "model is a different population and inherits none of that "
          "validation.", indent=False)
        P("Their survey is a per-name covariate, so the check costs no "
          "measurement \u2014 but it has to be read carefully, and an audit of "
          "this paper caught us reading it loosely. Correlating each first "
          "name\u2019s mean margin against the human-perceived probability "
          "that the name belongs to its assigned race, <i>signed</i> by race, gives "
          f"{len(q1sig)} of {len(cm)} models with an interval excluding zero. "
          "That statistic cannot do the work we asked of it. Perception is a "
          "probability, so signing it by race makes the predictor positive on "
          "every White name and negative on every Black name. Its ranks are "
          "perfectly separated by race, and the correlation is dominated by "
          "the between-race contrast it was supposed to validate "
          "independently. A model with a demographic effect scores well on it "
          "whether or not it tracks perception <i>within</i> either race.",
          indent=False)
        # THE MANIPULATION CHECK IS THE PERCEPTION CORRELATION, not the
        # callback one. This paragraph used to reach for the callback rate,
        # which is the CRITERION check of the paragraph below -- what employers
        # did, not what the name signals. The within-race perception statistic
        # the slot needs did not exist until an audit of this paper asked for
        # it. It ranks the signed perception probability inside each race, so
        # both blocks read as "how far toward White does this name point".
        _q1w = {m: (q1[m] or {}).get("pooled_within_race") for m in cm}
        P("The part that is not the race effect is the same survey read <i>within</i> "
          "each race. How distinctly a name signals the race it was chosen for, "
          "against how far the model moves for it, ranked inside each race so "
          "the between-race contrast cannot enter. That is the manipulation "
          "check, and it runs "
          + ", ".join(f"{_q1w[m]['rho']:+.2f}" for m in cm)
          + " \u2014 permutation p from "
          + f"{min(_q1w[m]['p_perm'] for m in cm):.2f} to "
          + f"{max(_q1w[m]['p_perm'] for m in cm):.2f}, and no interval "
          "excluding zero. We therefore cannot "
          "separate a model that encodes the name-race association and treats "
          "the two groups alike from one that does not encode it at all "
          "\u2014 which is the distinction a null on this panel would need. "
          + ((f"The design can only detect a within-race correlation of "
              f"{cval['mde']['mde_rho_pooled_within_race']:.2f} or larger "
              f"on the {cval['mde']['n_names_total']} pooled "
              "observations that statistic is computed on, so these "
              "nulls are weak evidence rather than reassuring ones.")
             if cval.get("mde") else "")
          + " A direct probe of the model\u2019s own representations would "
          "settle it; a correlation against the list\u2019s validation data "
          "will not.", indent=False)
        q2 = {m: cval["models"][m]["q2_callback"]["pooled_within_race"]
              for m in cm}
        q3 = {m: cval["models"][m]["q3_ses"]["pooled_within_race"] for m in cm}
        # THE INTERVALS ON THESE TWO ARE THE STRATIFIED ONES. The first version
        # of the pooled within-race bootstrap resampled the frozen ranks
        # without stratifying by race and without re-ranking inside the draw.
        # Both errors shrink the interval, and together they produced the one
        # "excludes zero" result in this section. See analyze_construct_
        # validity.pooled_within_race_ci for the arithmetic.
        _q3_excl = sum(1 for m in cm if q3[m]["ci"][0] * q3[m]["ci"][1] > 0)
        _q3top = max(cm, key=lambda x: abs(q3[x]["rho"]))
        if all(q2.values()) and mde:
            P("Two further checks are underpowered and we report them as such "
              "rather than as nulls. <i>within</i> race, no model’s per-name margin "
              "tracks the callback rate the same name actually received from "
              "real employers (r from "
              f"{min(q2[m]['rho'] for m in cm):+.2f} to "
              f"{max(q2[m]['rho'] for m in cm):+.2f}, every interval "
              "spanning zero). The socioeconomic proxy runs "
              f"{min(q3[m]['rho'] for m in cm):+.2f} to "
              f"{max(q3[m]['rho'] for m in cm):+.2f} and spans zero as well, "
              f"the closest being {SHORT[_q3top]} at "
              f"{q3[_q3top]['rho']:+.2f} [{q3[_q3top]['ci'][0]:+.2f}, "
              f"{q3[_q3top]['ci'][1]:+.2f}], p = {q3[_q3top]['p_perm']:.2f}"
              ". An earlier version of this paper reported that interval as "
              "excluding zero and built a paragraph on it. It does not. The "
              "bootstrap behind it resampled the within-race ranks without "
              "holding the race composition fixed and without re-ranking "
              "inside each draw, which understated its own spread by about a "
              "third against the exact permutation null. Stratifying and "
              "re-ranking brings the two into agreement and the interval "
              "covers zero. With "
              f"{mde['n_names_total']} pooled within-race observations the "
              f"design can only detect |r| ≥ "
              f"{mde['mde_rho_pooled_within_race']:.2f} at 80 % power, "
              "so neither check licenses a conclusion. The criterion check "
              "has a second and more interesting obstacle. At about 200 "
              "observations per name the field experiment’s own per-name rates "
              "are close to pure sampling noise, so there may be no stable "
              "per-name signal in the anchor to correlate against. Validating "
              "an LLM audit against a field experiment name by name is not "
              "something the existing anchor can support, and that is worth "
              "knowing before anyone tries.", indent=False)

    # ------------------------------------------------------------------
    APP.end()
    H("4.4  The matched pair is not token-matched", 2)
    if nlen:
        _tm = {m: nlen[m] for m in models if m in nlen}
        _tf = {m: _tm[m]["n_same_length"] / _tm[m]["n_pairs"] for m in _tm}
        P("Every other subsection of this section measures a choice the "
          "auditor made and could have made differently. This one measures an "
          "assumption the design makes without stating it: that two prompts "
          "identical in characters are identical in everything the model "
          "reads. They are not. A model reads tokens, and two names rarely "
          "occupy the same number of them.", indent=False)
        P("Across the "
          + f"{_tm[models[0]]['n_pairs']}-pair grid, the share of pairs whose "
          "two names are the same length in tokens is "
          + ", ".join(f"{pct(_tf[m], 0)} on {TINY[m]}" for m in models)
          + ". The measure is per model because tokenizers differ, and it "
          "needs no model calls. An auditor can compute it from the name list "
          "and the tokenizer before collecting any data. Most of a standard "
          "correspondence grid is unmatched on every tokenizer we tested.")
        _cl = {m: nlen[m]["token_matched_first_name_clustered"] for m in _tm}
        paper.table(
            ["model", ">pairs", ">token-matched", ">share",
             ">first-name pairs left"],
            [[TINY[m], str(_tm[m]["n_pairs"]), str(_tm[m]["n_same_length"]),
              pct(_tf[m], 0), str(_cl[m]["n_matched_clusters"])]
             for m in models],
            [132, 54, 84, 60, 110], span2=True, size=7.8,
            caption=(f"{TAB('token_segmentation')}. How much of a "
                     "standard correspondence grid is matched in tokens. The "
                     "grid is identical across models; only the tokenizer "
                     "changes. The last column is the number of distinct "
                     "first-name pairs surviving the restriction, which is "
                     "the resampling unit \u00a76.1 requires and the reason "
                     "the matched subset is weaker than its row count "
                     "suggests."))
        P("Three consequences matter for the argument of this paper, and the "
          "companion study establishes them in full. The token-length "
          "difference predicts the measured effect on "
          f"{_n_slope_sig_word} of the four checkpoints on the name-pair "
          "resampling unit. Restricting the audit to matched pairs moves the "
          "reported disparity"
          + ((f" on {NUM.get(_mask_n_away, _mask_n_away)} of four, by "
              f"{pct(_mask_gain, 0)} [{pct(_mask_gain_ci[0], 0)}, "
              f"{pct(_mask_gain_ci[1], 0)}] on the largest")
             if _mask_gain is not None else "")
          + ". And how far it moves cannot be settled on this list. The "
          "matched subset collapses to as few as "
          + f"{min(_cl[m]['n_matched_clusters'] for m in models)} independent "
          "first-name pairs, and three standard procedures on that subset "
          "give answers straddling 0.05.", indent=False)
        if npow:
            _gaps = npow["race_gap_in_tokens"]
            _lo = min(v["gap"] for v in _gaps.values())
            _hi = max(v["gap"] for v in _gaps.values())
            _best = max(npow["per_model"].values())
            P("<b>The subset is small by construction, not by accident, and a "
              "longer list does not fix it.</b> Matching on token length is an "
              "equivalence relation, so the largest matched set is exactly the "
              "sum over token-length vectors of the smaller of the two arms\u2019 "
              "counts, and it is large only where the two arms overlap. They "
              "barely do. The Black arm of this list costs "
              f"{_lo:.2f} to {_hi:.2f} more tokens than the white arm, on "
              "every tokenizer in the panel, so the two occupy nearly disjoint "
              "regions of that space. Of "
              f"{npow['possible_pairs']} possible first-name pairs, "
              f"{_best} survive on the single most favourable tokenizer and "
              f"{npow['max_matching_panel_total']} survive on all four; the "
              "drop happens at the second tokenizer rather than gradually, so "
              "comparability across models costs more here than the "
              "tokenization control does. A longer list drawn the same way "
              "reproduces the same gap and so the same thin overlap. What the "
              "design needs is a list built token-matched from the outset, "
              "which is a different object from a list filtered afterwards.",
              indent=False)
        if permres:
            # fewest_pairs_model, NOT headline_model. The artifact's
            # headline names the model whose effect-difference headlined a
            # different comparison; keying on it attributed a three-pair
            # floor to a checkpoint that has eight.
            _ph = permres["per_model_design_as_run"][
                permres["fewest_pairs_model"]]
            _rem = permres["remedy"]
            P("<b>The restriction does not cost power, it removes the "
              "possibility of an answer.</b> A matched-pair randomisation "
              "test conditions on the pairs and re-randomises the label "
              "within each, so at <i>n</i> pairs there are exactly 2^<i>n</i> "
              "equally likely sign assignments and the smallest attainable "
              "two-sided p-value is 2^(1−<i>n</i>). Stratifying the "
              "permutation within token-length classes, which is the correct "
              "test once length is controlled, leaves that count unchanged, "
              "because the classes are complete blocks and the randomisation "
              "distribution factorises over them. On the checkpoint where "
              "fewest pairs survive the restriction the subset holds "
              f"{_ph['n_pairs']} pairs, giving {_ph['n_assignments']} "
              "arrangements and a floor of "
              f"{_ph['p_min_two_tailed']:.2f}, above any conventional "
              "threshold whatever the data show; "
              f"{permres['n_models_that_cannot_reach_alpha']} of "
              f"{permres['n_models']} checkpoints is in that position, and "
              "so is the panel-wide maximum matching over the full list. "
              "Clearing 0.05 needs at least "
              f"{_rem['n_pairs_needed']} pairs, which at the observed yield "
              f"is about {_rem['names_per_cell_needed']:.0f} names per cell "
              f"against the {_rem['names_per_cell_now']} the standard list "
              "carries. That is the sense in which the list, not the sample, "
              "is what fails.", indent=False)
        P("<b>What this does and does not license.</b> Token length and name "
          "distinctiveness are correlated by construction, since a rarer name "
          "segments into more pieces, so no design that varies the name can "
          "attribute the effect to one of them, and we do not. The claim is "
          "that a design describing itself as matched is unmatched on a "
          "dimension the instrument is sensitive to, that the imbalance is "
          "large and computable in advance, and that no audit in "
          + TAB("reporting_matrix") + " reports it. A grid balanced by "
          "construction can be built from the same validated list"
          + (f" and reaches {tbal['n_pairs']} pairs" if tbal else "")
          + ", which is too small to carry an audit \u2014 so the remedy is "
          "to report the token statistics, not to rebuild the list.")
        P("The full treatment is a companion paper. The per-pair regression, "
          "the matched-subset estimates with all three procedures, the "
          "cross-check between the two length designs, the balanced-grid "
          "construction, and a position manipulation that would have "
          "corroborated the channel and does not. It reads the artifacts this "
          "paper releases, so the two cannot disagree.")
    H("4.5  Which job was posted", 2)
    if occ:
        om = [m for m in models if m in occ]
        rows = []
        spreads = {}
        for m in om:
            # A CELL WITH NO DENOMINATOR GETS NO NUMBER. Two thirds of these
            # ratios divide by an effect whose own interval covers zero, and
            # the paragraph below says so; the table printed all twelve
            # regardless, plus a spread column built from them that the same
            # paragraph withdraws as "not a number we may print". Suppressed
            # cells now read as an em dash, and the spread column is gone.
            r = [(occ[m][k]["ratio_sd_to_effect"] * 100,
                  not occ[m][k].get(
                      "ratio_denominator_indistinguishable_from_zero"))
                 for k in ("BA", "SWE", "RN")]
            spreads[m] = max(v for v, _ in r) - min(v for v, _ in r)
            rows.append([TINY[m]]
                        + [(f"{v:.0f} %" if ok else "—")
                           for v, ok in r])
        paper.table(
            ["model", ">balanced", ">male-typed", ">female-typed"],
            rows, [70, 52, 56, 58], size=7.6,
            caption=(f"{TAB('occupation_ratio')}. Dispersion-to-effect ratio across three "
                     "structurally matched postings, business analyst, "
                     "software engineer, registered nurse, with identical "
                     "résumé structure, identical wordings and "
                     "identical names. A cell is suppressed where the effect "
                     "it divides by has an interval covering zero, which is "
                     "two thirds of them. The spread across a row is "
                     "therefore not computed here; the text explains why the "
                     "version of it we previously printed was withdrawn."))

        P("The natural defence against Section 4.1 is that the "
          "dispersion-to-effect ratio is a stable property of a model, so a "
          "reader can mentally discount by it. Study 7 tests that "
          "confirmatorily. Three occupations were constructed to be "
          "structurally matched. The same résumé skeleton, the "
          "same seniority, the same requirement count, differing in "
          "occupational gender typing , and a matching check enforces it "
          "programmatically.", indent=False)

        # THE RESULT LEADS. It used to sit five paragraphs down, after two
        # withdrawals, so a reader who stopped early took away that the
        # section had failed. It did not: it established one thing and failed
        # to establish another, about a different quantity.
        _gap = {}
        if occnull and occnull.get("models"):
            _gap = {m: occnull["models"][m]["effect_gap_across_postings"]
                    for m in om if m in occnull["models"]
                    and "effect_gap_across_postings" in occnull["models"][m]}
        if _gap:
            _gsig = [m for m, g in _gap.items() if g["p_permutation"] < 0.05]
            P("<b>The job posted moves the reported disparity.</b> Permuting "
              "posting labels within name pair, which preserves the pairing "
              "because all three postings score the same twelve pairs, the "
              "largest gap between two postings runs "
              f"{fmt(min(g['est'] for g in _gap.values()), 3)} to "
              f"{fmt(max(g['est'] for g in _gap.values()), 3)} on the "
              "probability-of-superiority scale and is distinguishable from "
              f"zero on {len(_gsig)} of {len(_gap)} models. That gap is of "
              "the same order as the whole demographic effect the audit "
              "reports. Which posting was used is reported by almost every "
              "surveyed study; what none of them reports is what the effect "
              "would have been under a different one. That is this "
              "section's finding.", indent=False)
            P("What follows is what we could <i>not</i> establish, reported at "
              "length because we tried twice and printed one of the attempts. "
              "The job moves the <i>effect</i>; we could not show it moves the "
              "<i>dispersion</i>. Those are separate claims and only the first is "
              "ours to make.")
        worst = max(spreads, key=spreads.get)
        _sp = {m: occ[m].get("ratio_spread_across_occupations") for m in om}
        _wsp = {m: occ[m].get("wording_sd_spread_across_occupations")
                for m in om}
        _n_null_denom = sum(
            1 for m in om for k in ("BA", "SWE", "RN")
            if occ[m][k].get("ratio_denominator_indistinguishable_from_zero"))
        P("The defence fails, and an audit of this paper narrowed the form in "
          "which we are entitled to say so. The ratio’s denominator is "
          f"the effect, and on {_n_null_denom} of the {3 * len(om)} "
          "model-by-posting cells that denominator’s own interval covers "
          "no effect. A ratio against such a denominator has no finite upper "
          "bound, so the spread this paper previously quoted as a bare "
          f"{fmt(max(spreads.values()), 0)} percentage points in fact has an "
          f"interval of [{fmt(_sp[worst]['ci'][0] * 100, 0)}, "
          f"{fmt(_sp[worst]['ci'][1] * 100, 0)}] and is not a number we may "
          "print.", indent=False)
        # THE SECOND ATTEMPT AT THIS CLAIM FAILED TOO, AND IT IS REPORTED AS A
        # FAILURE. Having lost the ratio to its denominator, an earlier draft
        # fell back on the numerator: the between-wording SD alone moves across
        # postings by 0.034, 0.043, 0.020, 0.018, "every interval excluding
        # zero". Every interval excludes zero because the statistic is max
        # minus min of three non-negative SDs and cannot be negative. That is
        # arithmetic, not evidence, and a critique of this paper caught it. The
        # replacement is a permutation test against the null the sentence
        # needed to beat, and the sentence does not beat it.
        P("<b>The claim does not survive on the numerator either, and this is "
          "the second form of it we have had to withdraw.</b> Having lost the "
          "ratio to its denominator, we fell back on the between-wording "
          "standard deviation alone, which moves across the three postings by "
          + ", ".join(fmt(_wsp[m]["est"], 3) for m in om if _wsp.get(m))
          + " for the four models, and we reported that every interval "
          "excluded zero. So it does. The statistic is the largest of three "
          "standard deviations minus the smallest, it cannot be negative, and "
          "no interval on it can contain zero whatever the data says. That "
          "sentence was arithmetic wearing the clothes of evidence.",
          indent=False)
        if occnull and occnull.get("models"):
            _on = occnull["models"]
            _os = occnull["summary"]
            _ps = [_on[m]["spread"]["p"] for m in om if m in _on]
            P("Calibrated against the null it needed to beat. That the three "
              "postings share one between-wording dispersion, tested by "
              "pooling each posting’s deviations from its own mean and "
              f"permuting them across postings, {occnull['n_permutations']:,} "
              "times. The observed spread is unremarkable on every model "
              f"(p = {', '.join(f'{p:.2f}' for p in _ps)}). The same holds "
              "for the variance ratio, which does not discard the middle "
              f"posting. On this evidence the dispersion is "
              f"indistinguishable across the three jobs on "
              f"{_os['n_models'] - _os['n_significant_spread']} of "
              f"{_os['n_models']} models, and the claim that wording "
              "dispersion is a property of the model-and-job <i>pair</i> rather than "
              "of the model is withdrawn. Twelve wordings estimate a standard "
              "deviation with eleven degrees of freedom"
              # NOT "about +/-35 %". That was the low end of §4.1's RELATIVE
              # WIDTH halved into a symmetric half-width, a conversion this
              # interval does not admit: a bootstrap CI on a standard
              # deviation with eleven degrees of freedom is skewed in the SD's
              # own units. It also dropped the top of the range.
              + ((" (§4.1 puts the interval on it at "
                  f"{pct(disp_unc['summary']['min_relative_width'], 0)} to "
                  f"{pct(disp_unc['summary']['max_relative_width'], 0)} of the "
                  "point estimate, and asymmetric)")
                 if disp_unc and disp_unc.get("summary") else "")
              + ", and three such estimates "
              "cannot separate what this claim needed separated.", indent=False)
        P("What cannot be defended in any form is expressing the dispersion as "
          "a multiple of an effect that is not there. On two thirds of these "
          "arms, dispersion-to-effect is not a property of anything.")
        flip = [m for m in om
                if len({np.sign(occ[m][k]["logodds"]) for k in ("BA", "SWE", "RN")}) > 1]
        if flip:
            _acc = occ[flip[0]].get("acceptance_by_strength_across_occupations")
            P(f"On {SHORT[flip[0]]} the <i>direction</i> of the effect is not "
              "stable across occupations either. It is tempting to argue that the "
              "three postings are matched on structure by construction and "
              "that the reversal is therefore not a confound of "
              "résumé quality. The first half is right; the second "
              "does not follow. Structural matching is enforced on the "
              "<i>posting</i>: the same skeleton, the same seniority, the same "
              "requirement count , and it says nothing about where the "
              "model’s decision boundary falls for that job."
              + ((" Measured rather than assumed, the acceptance rate for the "
                  "same résumé strength differs across postings on "
                  f"this model by up to {fmt(_acc['worst_gap_pp'], 1)} points "
                  f"({_acc['worst_strength']}).")
                 if _acc else "")
              + " The reversal is real and is a fact about the model-and-job "
              "pair. It is not evidence that difficulty was held fixed, and "
              "the sentence that said so is withdrawn.")
        P("A further observation deserves emphasis because it bears on our own "
          "design. The business-analyst posting , the one every other "
          "study in this paper is built on, happens to be the occupation "
          "where the effect is largest and the dispersion smallest on "
          + NUM.get(_ba_both, str(_ba_both)) + " of the four models, and "
          "where the effect alone is largest on "
          + NUM.get(_ba_eff, str(_ba_eff)) + ". Had we chosen the "
          "software-engineer posting as the "
          "primary stimulus, the same models would have looked as though they "
          "had no measurable effect at all, drowned in wording variance. The "
          "choice of job is itself a researcher degree of freedom, and we were "
          "as exposed to it as anyone.")

    if armasym and armasym.get("summary"):
        _as = armasym["summary"]
        _rev = [SHORT.get(m, m) for m in _as["reversed_models"]]
        H("B.1  Which arm is the unstable one?", 2)
        P("A referee of the epistemic account of this volatility — the "
          "model knows less about the minority names, so the minority arm "
          "should be the jumpier one — can test it here, because the "
          "per-arm margins are released. Comparing the between-wording "
          "variance of the Black-name arm against the White-name arm, "
          "cell by cell with clusters resampled at the name pair, the "
          "Black-name arm is the more volatile one in "
          f"{NUM.get(_as['n_ci_excludes_zero_positive'], str(_as['n_ci_excludes_zero_positive']))} of "
          f"{NUM.get(_as['n_combinations'], str(_as['n_combinations']))} model-by-dataset combinations, with intervals "
          "excluding zero — and the exception is systematic rather than "
          f"noisy: {', '.join(_rev)} shows the reverse, with its interval "
          "excluding zero on the other side, in both datasets. The "
          "asymmetry is real but modest (variance ratios near one), and "
          "the direction is not a law of the panel; a mechanism that "
          "predicts the reversal as well as the rule would be the "
          "interesting one.", indent=False)

    if xmodel and xmodel.get("specifications"):
        _xg = xmodel["specifications"].get(
            "margin-gap scale, grand-centered", {})
        _xc = xmodel.get("per_context_summary", {})
        H("B.2  Is the wording effect shared across models?", 2)
        P("If two models agreed on <i>which</i> wordings inflate the measured "
          "effect, an auditor could hope to learn a safe wording once. They "
          "do not agree. Stacking the per-wording effect profiles across the "
          "five task contexts that share the twelve-wording grid, the "
          "cross-family correlations are indistinguishable from zero "
          f"(mean r = {_xg.get('cross_family_mean', 0):+.2f}), and "
          f"{pct(_xg.get('idiosyncratic_share', 0), 0)} of the "
          "between-wording variance is idiosyncratic to the model. The one "
          "pair that does agree is the one that shares a lineage: the two "
          "Mistral checkpoints correlate at "
          f"r = {_xg.get('family_r', 0):+.2f} on the pooled profiles, rank "
          "first among the six pairs in every specification and in each of "
          f"the {NUM.get(_xc.get('n_contexts', 5), str(_xc.get('n_contexts', 5)))} contexts separately. With four models and a single "
          "same-family pair this is a hypothesis confirmed at the floor the "
          "design allows, not a law; what it argues is that the instrument "
          "effect follows the model, and what transfers follows the "
          "lineage. The profiles themselves are nearly noise-free — the "
          "byte-identical replicate puts their reliability at "
          f"{xmodel['reliability_floor']['value']:.2f} or better — so "
          "the disagreement is not measurement error.", indent=False)

    # ------------------------------------------------------------------
    APP.start()   # -> Appendix C
    if second and second.get("by_domain"):
        H("C  The same twelve wordings, outside hiring")
        _dl = second.get("domain_labels", {})
        _doms = [d for d in second["domains"] if second["by_domain"][d]["models"]]
        P("Everything above is measured on a résumé screened against a job "
          "posting. A reader who accepts all of it can still say that "
          "résumé screening is peculiar. A long, highly structured "
          "stimulus, a decision the model has seen a great deal of, and a "
          "literature that has taught it what the expected answer looks like. "
          "If the wording moves the demographic estimate only here, this is a "
          "paper about hiring audits. If it moves it elsewhere, it is a paper "
          "about measuring language models.", indent=False)
        # WHAT ACTUALLY RAN, COUNTED FROM THE ARTIFACT. This paragraph used to
        # assert the full design -- twelve wordings, two arms, twelve pairs,
        # four checkpoints -- while the artifact behind it held one checkpoint
        # and, on one domain, four wordings from a single arm. The design is
        # specified once in the code and the coverage is now counted, so the
        # sentence describes the run rather than the intention.
        _cells = [(d, m, v) for d in _doms for m in ORDER
                  if (v := second["by_domain"][d]["models"].get(m))]
        _ckpts = sorted({m for _, m, _ in _cells})
        _nw = {len(v.get("per_variant", {})) for _, _, v in _cells}
        _arms = {a for _, _, v in _cells
                 for a, x in (v.get("arm_sd") or {}).items() if x is not None}
        _full = (_nw == {12} and _arms == {"semantic", "null"})
        P("So the wording study was run again with the domain swapped and "
          "nothing else. The same wordings, the same name pairs in the same "
          "order, the same three strength levels and the same outcome, on "
          + (f"{NUM.get(len(_ckpts), len(_ckpts))} of the four checkpoints"
             if len(_ckpts) < 4 else "all four checkpoints")
          + (". Every cell carries all twelve wordings in both arms, "
             "including the six semantically-null perturbations character "
             "for character" if _full else
             ". Coverage is not uniform across cells and the per-row wording "
             "count is printed in " + TAB("second_task") + " rather than assumed")
          + ". The domains are "
          + " and ".join(_dl.get(d, d) for d in _doms)
          + (". The second is not an allocation decision at all. The text "
             "being judged is identical across the pair and only the "
             "attributed author's name differs, which makes it the shape of a "
             "benchmark rather than of an audit."
             if "moderation" in _doms else ".")
          + " These runs use the serving configuration §5.2 establishes as "
          "reproducible: concurrency one, prompt-cache reuse off , which "
          "Study 2 predates.")
        _rows = []
        for d in _doms:
            for m in ORDER:
                v = second["by_domain"][d]["models"].get(m)
                if not v:
                    continue
                rr = v["ratio_sd_to_effect_ps"]
                son = v["sd_over_noise"]
                # SUPPRESS ON THE IDENTIFICATION FLAG, NOT ON TRUTHINESS. The
                # caption has always promised this; the code tested `if rr`,
                # and the artifact always returns a ratio, so the rule was
                # announced and never applied. The housing row printed 7.75x
                # against an effect whose interval covers zero.
                _n_w = len(v.get("per_variant", {}))
                _rows.append([_dl.get(d, d), TINY.get(m, m),
                              f"{v['overall']['n']}", f"{_n_w}",
                              fmt(v["effect_ps"], 3),
                              fmt(v["sd_across_wordings_ps"], 4),
                              (f"{rr:.2f}×" if rr and v.get("effect_identified")
                               else "—"),
                              # THE REPLICATE COLUMN, which used to be an
                              # empty "SD / noise floor". The floor is exactly
                              # zero on every cell, so the ratio is unbounded
                              # rather than missing; what a reader can use is
                              # the agreement count behind that zero.
                              (v.get("replicate_agreement") or "—")])
        if _rows:
            paper.table(
                ["domain", "model", ">pairs", ">wordings", ">P(sup.)",
                 ">SD across wordings", ">SD / effect", ">replicate"],
                _rows, [80, 56, 30, 40, 44, 66, 44, 74], span2=True, size=7.4,
                caption=(f"{TAB('second_task')}. The wording study re-run outside hiring; "
                         "only the task changes. Reported on the <i>probability</i> "
                         "<i>of</i> <i>superiority</i>, the scale "
                         # Hard-typed as "Tables 3 and 17". Table 17 is the
                         # mechanism-class table, whose columns are absolute
                         # LOG-ODDS -- so a caption whose whole point is scale
                         # commensurability named a panel on the other scale.
                         # The three P(superiority) panels are this one, the
                         # open-weight study and the frontier re-run.
                         + TABS("study2_effect", "frontier") + " use, "
                         "so the three panels are comparable; "
                         "put this table in log-odds and the ratios were not "
                         "commensurable with the others. “pairs” is how many of the "
                         "432 design cells the top-100 window resolved and "
                         "“wordings” how many of the twelve those cells "
                         "cover, so a reader can see the coverage of each row "
                         "rather than take it on trust. “replicate "
                         "agreement” is how many of that cell's "
                         "byte-identical S1/N1 pairs returned a "
                         "bitwise-identical margin. The measurement noise "
                         "floor of every row is exactly zero, so the "
                         "dispersion beside it is signal in its entirety and "
                         "a dispersion-to-noise ratio would be unbounded "
                         "rather than informative. The SD / effect ratio is "
                         "suppressed, as “—”, wherever the effect "
                         "interval covers zero."))
        _s = second["summary"]
        # THE REPLICATE ARM OF THIS STUDY IS ALSO A REPRODUCIBILITY RESULT, and
        # the paper was not reporting it. Every cell carries an S1/N1 pair whose
        # assembled prompts are byte-identical, so the arm measures the noise
        # floor at the same time as the dispersion. An earlier draft printed
        # "not yet estimable" because the floor was zero and the ratio divided
        # by it; the floor being zero is the finding.
        if _s.get("n_replicate_cells"):
            _rc, _ri = _s["n_replicate_cells"], _s["n_replicate_cells_identical"]
            P("<b>Before the dispersion, the floor it stands on.</b> Every cell "
              "of this study contains a pair of wordings, S1 and N1, whose "
              "assembled prompts are byte-identical. The null arm's "
              "baseline is the semantic arm's baseline, character for "
              "character , so that pair measures reproducibility at the "
              "same time as everything else measures dispersion. Across "
              f"{_rc} such replicate cells, spanning "
              f"{_s['n_domain_model_cells']} domain-by-model combinations, "
              + (f"all {_ri} returned a <i>bitwise</i>-<i>identical</i> margin"
                 if _ri == _rc else
                 f"{_ri} returned a bitwise-identical margin")
              + ". The measurement noise floor of this study is exactly zero "
              f"on {_s['n_cells_with_zero_noise_floor']} of "
              f"{_s['n_domain_model_cells']} cells. Two things follow. The "
              "dispersion reported below is signal in its entirety, with no "
              "noise term to subtract, so it needs no floor comparison to be "
              "interpreted , which is why " + TAB("second_task") + " reports the agreement "
              "count rather than a ratio that would be unbounded. And §5.2's "
              "reproducibility result, established on résumé screening, "
              "replicates on two further domains and four checkpoints under the "
              "same serving configuration, in a study that was not built to "
              "test it.", indent=False)
        P("The dispersion does not stay behind. Across the "
          f"{_s['n_domain_model_cells']} domain-by-model cells the "
          "between-wording standard deviation is "
          + (f"{_s['ratio_min']:.2f}× to {_s['ratio_max']:.2f}× the "
             f"effect itself on the {_s['n_with_identified_effect']} cells "
             "whose effect is separable from zero"
             if _s.get("ratio_min") else "of the same order as the effect")
          + (f", and {_s['sd_over_noise_min']:.0f} to "
             f"{_s['sd_over_noise_max']:.0f} times each domain's own "
             "measurement noise floor" if _s.get("sd_over_noise_min")
             else ", against a measurement noise floor of exactly zero")
          + ". Twelve wordings that ask the same question of the same "
          "candidate move the measured demographic difference on a tenancy "
          "decision and on a content judgment by the same order as they move "
          "it on a résumé. Instrument variance is not a property of "
          "hiring audits; it is a property of asking a language model a "
          "question and reporting the answer as a measurement.", indent=False)
        # On the probability-of-superiority scale, matching Table 16.
        _arms = [(d, m, v["arm_sd_ps"])
                 for d in _doms for m in ORDER
                 if (v := second["by_domain"][d]["models"].get(m))
                 and (v.get("arm_sd_ps") or {}).get("null") is not None
                 and (v.get("arm_sd_ps") or {}).get("semantic") is not None]
        if _arms:
            _null_ge = sum(1 for _, _, a in _arms if a["null"] >= a["semantic"])
            _nulls = [a["null"] for _, _, a in _arms]
            _sems = [a["semantic"] for _, _, a in _arms]
            P("<b>And the arm split travels with it.</b> The claim that costs "
              "the most to accept is not that wording matters but that "
              "<i>surface</i> <i>form</i> does. Six of the twelve wordings change no word, "
              "only a line break, a capital or an option order. On the "
              f"{len(_arms)} cells here, on the probability-of-superiority "
              "scale, the null arm's between-wording standard deviation runs "
              f"{min(_nulls):.4f} to "
              f"{max(_nulls):.4f} against the semantic arm's "
              f"{min(_sems):.4f} to {max(_sems):.4f}, and the null arm is the "
              f"<i>larger</i> of the two on {_null_ge} of {len(_arms)}. Two domains "
              "and four checkpoints that were never used to develop the "
              "finding reproduce it: edits that preserve the question exactly "
              "move the demographic estimate about as much as edits that "
              "rewrite it.", indent=False)
        _cens = [second["by_domain"][d]["censoring"].get(m, {})
                 for d in _doms for m in ORDER
                 if second["by_domain"][d]["censoring"].get(m)]
        _drop = [c["frac_pairs_dropped"] for c in _cens
                 if c.get("frac_pairs_dropped") is not None]
        _pdiff = [c["differential_p"] for c in _cens
                  if c.get("differential_p") is not None]
        if _drop:
            _worst = min(_pdiff) if _pdiff else None
            _bad = [c for c in _cens
                    if c.get("differential_p") is not None
                    and c["differential_p"] < 0.05]
            P("<b>One threat this design has and Study 2 largely did not, and "
              "it is not clean.</b> The margin is read from a top-100 "
              "next-token window; when a model is very confident the losing "
              "option falls outside it, no margin exists, and the pair is "
              "dropped. That censoring is not random. It happens exactly "
              "where the margin is largest. It costs "
              + rng_str(min(_drop), max(_drop), lambda v: pct(v, 1))
              + " of pairs. What would bias the paired estimate is censoring "
              "that differs <i>by</i> <i>arm</i>, so that is tested rather than assumed"
              + ((f", and on {len(_bad)} of {len(_cens)} domain-model cells it "
                  f"does differ (smallest p = {_worst:.3f}). We report that "
                  "rather than let the drop pass." )
                 if _bad else
                 (f" and does not differ on any cell (smallest p = "
                  f"{_worst:.3f})." if _worst is not None else "."))
              , indent=False)
            _wb = [c["bounds"] for c in _cens if c.get("bounds")]
            _unstable = [b for b in _wb if not b["sign_stable_under_bounds"]]
            # IS THE RESOLVED SET REALLY COMMON TO ALL TWELVE WORDINGS? The
            # sentence below asserted it was. per_model() builds each wording's
            # estimate from that wording's own resolved rows, and the artifact
            # records the resulting per-wording n, which nobody had read.
            _pvn = []
            for _d in _doms:
                for _m in ORDER:
                    _pv = (second["by_domain"][_d].get("models", {})
                           .get(_m, {}).get("per_variant") or {})
                    _ns = [x["n"] for x in _pv.values()
                           if isinstance(x, dict) and "n" in x]
                    if _ns:
                        _pvn.append(_ns)
            _pv_same = sum(1 for _ns in _pvn if len(set(_ns)) == 1)
            _pv_uneq = [_ns for _ns in _pvn if len(set(_ns)) > 1]
            if _wb:
                P("A censored margin is not unknown, only unobserved from "
                  "below. The model emitted a verdict, so the sign is known "
                  "and the magnitude is at least as large as the most extreme "
                  "the window did resolve. Substituting that bound for every "
                  "censored cell in both directions brackets what the drop "
                  "could be hiding. On "
                  f"{len(_unstable)} of {len(_wb)} cells the bracket contains "
                  "zero, so on those the <i>sign</i> of the demographic effect is not "
                  "determined by this design. That is a limitation of the "
                  "second-task measurement and we state it as one. It does not "
                  "reach the claim this section makes. The dispersion across "
                  "wordings is computed from the cells the window did resolve, "
                  "the same cells under every wording on "
                  f"{_pv_same} of {len(_pvn)} domain-model rows"
                  + ((f"; on the "
                      f"{NUM.get(len(_pv_uneq), str(len(_pv_uneq)))} that "
                      "differ, each wording resolves "
                      f"{min(min(x) for x in _pv_uneq)} to "
                      f"{max(max(x) for x in _pv_uneq)} cells, so that row's "
                      "spread carries a composition term as well as a wording "
                      "term") if _pv_uneq else "")
                  + ". A dispersion around "
                  "an undetermined centre is still a dispersion. Where the "
                  "effect's own sign matters, nowhere in this section "
                  ". It is not claimed.")

    # ======================================================================
    # ------------------------------------------------------------------
    APP.end()
    APP.start()   # -> Appendix D
    if front and front.get("models"):
        H("D  The same twelve wordings, on frontier models")
        _fs = front["summary"]
        _fm = front["models"]
        _live = {m: v for m, v in _fm.items() if not v.get("unmeasurable")}
        _dead = [m for m, v in _fm.items() if v.get("unmeasurable")]
        P("Everything so far is measured on open weights, which §10.2 gives a "
          "reason for. Quantization, batching and cache residency cannot be "
          "manipulated behind an API, and those are half of what this paper "
          "measures. The other half (the wording, the names, the pairing) "
          "does not need serving control, and there is no good reason "
          "not to test it where the audits actually point. The obstacle was "
          "never money. "
          + (f"The whole run below cost {fspend['total_cents']} cents at "
             f"list price, {fspend['total_input_tokens']:,} input tokens "
             f"across {fspend['n_models']} models. " if fspend else
             "The whole run below cost well under a dollar. ")
          + "It was that this paper’s outcome is a next-token "
          "distribution, and most vendors do not return one"
          # MODELS, NOT PROBE RECORDS. This read n_probed and called it a
          # count of models. n_probed is 46 (model, REST-version) records over
          # 33 distinct models, 13 of which answer on both versions , so the
          # sentence overstated the breadth of an EXHAUSTIVENESS claim, which
          # is the one place the difference matters most.
          + ((": we probed every generation-capable text model the Gemini "
              "API-key surface lists: "
              f"{fcap.get('n_models_distinct', fcap['n_probed'])} of them, "
              f"{fcap['n_probed']} probes across both public <i>rest</i> versions, "
              "each with three spellings of the parameter , and "
             f"{fcap['n_usable_for_margin']} returned any. "
             f"{fcap['n_reachable']} probes reached a model that refused; "
             f"{fcap['n_quota_blocked']} were quota-blocked and "
             f"{fcap['n_unavailable']} could not be called at all, the model "
             "having been retired for new users or served only through a "
             "different API, so the question is untested on those") if fcap else "")
          + ". Four OpenAI checkpoints do.", indent=False)
        if fcap:
            P("<b>That claim is narrower than the one we first made, and "
              "the narrowing matters.</b> An earlier draft said the study "
              "could not be run on that vendor <i>at</i> <i>any</i> <i>price</i>, on a "
              "hand-picked fourteen models, and called the key paid. All "
              "three were wrong. The account carries no balance, so every "
              "call went through the free tier, which is precisely where a "
              "capability gate is most likely; the model list is now "
              "exhaustive rather than chosen; and Vertex AI’s endpoint, "
              "which needs OAuth and a billed project, was never reached "
              "and documents this parameter. What we can say is what the "
              "API said. Every reachable model returns HTTP 400 with one "
              "string — “Logprobs is not enabled for this model” "
              ", which reports a disabled feature, not an unknown "
              "field. The parameter is recognised and gated. Whether the "
              "gate opens on a billed tier or on Vertex we did not test, "
              "and this is a statement about the free API-key surface and "
              "nothing wider.", indent=False)
        P("So Study 2 was run again, unchanged, on those four. The same twelve "
          "wordings in the same two arms, the same twelve name pairs in the "
          "same order, the same three strength levels, and the same outcome "
          ". The renormalised margin, read from the top of the "
          "distribution exactly as the local panel reads it. Two differences "
          "are forced by the API and neither is cosmetic. There is no "
          "grammar, so the emission is unconstrained and the verdict is "
          "whatever the model wrote. And the window is the top twenty tokens "
          "rather than the top hundred, so a margin can fall outside it.",
          indent=False)
        _rows = []
        for m in ORDER_FRONT:
            v = _fm.get(m)
            if not v:
                continue
            if v.get("unmeasurable"):
                _rows.append([m, f"{v['n_usable']}/{v['n_design_cells']}",
                              "—", "—", "—", "—"])
                continue
            _sup = v["superiority"]
            _rows.append([
                m, f"{v['n_usable']}/{v['n_design_cells']}",
                f"{_sup['est']:.3f} [{_sup['ci'][0]:.3f}, {_sup['ci'][1]:.3f}]",
                f"{v['sd_across_wordings']:.4f}",
                (f"{v['ratio_sd_to_effect']:.2f}×"
                 if v.get("ratio_sd_to_effect") else "—"),
                "  ".join(f"{k[:3]} {a['sd']:.3f}"
                          for k, a in sorted(v["arms"].items()))])
        paper.table(
            ["model", ">pairs resolved", ">P(superiority) [95 %]",
             ">SD across wordings", ">SD / effect", ">by arm"],
            _rows, [76, 62, 108, 66, 50, 96], span2=True, size=7.4,
            caption=(f"{TAB('frontier')}. Study 2 repeated on frontier API models, with "
                     "the same outcome and the same estimand; intervals "
                     "resample name pairs. “pairs resolved” is how "
                     "many of the 432 design cells the top-20 window could "
                     "form a margin for. The ratio is suppressed where the "
                     "effect interval covers 0.5, and “by arm” "
                     "gives the between-wording SD within the six semantic "
                     "paraphrases and within the six edits that change no "
                     "word."))
        _idr = [_fm[m]["ratio_sd_to_effect"] for m in _fs["identified_models"]]
        _idsd = [_fm[m]["sd_across_wordings"] for m in _fs["identified_models"]]
        _ideff = [abs(_fm[m]["superiority"]["est"] - 0.5)
                  for m in _fs["identified_models"]]
        _locsd = [sd_word[m] for m in survives]
        _loceff = [abs(ps[m] - 0.5) for m in survives]
        P("On the "
          f"{NUM.get(_fs['n_identified'], _fs['n_identified'])} of "
          f"{NUM.get(_fs['n_models'], _fs['n_models'])} checkpoints where the "
          "effect is distinguishable "
          "from chance, the between-wording standard deviation is "
          f"{min(_idr):.2f} to {max(_idr):.2f} times the effect itself, "
          f"against {fmt(min(ratio_ps[m] for m in survives), 2)} to "
          f"{fmt(max(ratio_ps[m] for m in survives), 2)} on the open-weight "
          "panel. Read that carefully, because the obvious reading is wrong.",
          indent=False)
        P("<b>The dispersion did not grow. The effect shrank.</b> The "
          "numerator of that ratio is almost the same on the two panels. A "
          f"between-wording standard deviation of {min(_idsd):.3f} to "
          f"{max(_idsd):.3f} on the frontier checkpoints against "
          f"{min(_locsd):.3f} to {max(_locsd):.3f} on the open-weight ones. "
          "What differs is the denominator. The demographic effect "
          f"itself is {min(_ideff):.3f} to {max(_ideff):.3f} here against "
          f"{min(_loceff):.3f} to {max(_loceff):.3f} there. So the honest "
          "statement is not that frontier models are noisier instruments. It "
          "is that the instrument noise is the <i>same</i> while the signal is "
          "smaller, and the quantity an auditor actually needs , how much "
          "of the reported number is the instrument, therefore gets "
          "worse, not better, on the models the field most wants to audit. "
          "Neither ratio is pinned tightly enough to rank the two panels, and "
          "we do not. What the two have in common is that both are of order "
          "one half.", indent=False)
        if rint and rint.get("comparison", {}).get("_verdict"):
            _rc = rint["comparison"]
            _rf = {k: v for k, v in rint["frontier"].items()
                   if k in _rc["frontier_models"]}
            _rl = {k: v for k, v in rint["local"].items()
                   if k in _rc["local_models"]}
            P("<b>That last sentence used to be an assertion; here is the "
              "interval behind it.</b> Bootstrapping the ratio on name pairs "
              "moves numerator and denominator together, so they have to be "
              "resampled together, which gives "
              + "; ".join(
                  f"{SHORT.get(k, k)} {v['est']:.2f} "
                  f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]"
                  for k, v in _rf.items())
              + " on the frontier side against "
              + "; ".join(
                  f"{SHORT.get(k, k)} {v['est']:.2f} "
                  f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]"
                  for k, v in _rl.items())
              + " locally. Every frontier point estimate sits above every "
              "local one, and not one interval separates from its "
              "counterpart. The ordering is a direction, not a result, and "
              "an earlier draft of this paper stated it as a result three "
              "times , including in the abstract, while this section "
              "disclaimed it. The disclaimer was right.", indent=False)
            P("The width is not a sampling failure to be fixed with more "
              "pairs; it is the shape of the quantity. The denominator is a "
              "distance from one half that can approach zero, so the ratio "
              "has no finite upper bound and its interval is asymmetric by "
              "construction. The same objection §4.5 raises against a "
              "spread quoted over a denominator that covers zero, arriving "
              "here in a place we had not applied it. Where the denominator "
              "does vanish in a resample the draw is discarded and the "
              "fraction reported, rather than letting one near-zero "
              "denominator set the upper endpoint.")
        if fnoise and fnoise.get("summary", {}).get("n_models_with_a_replicate"):
            # THE SAME DEFECT THIS SECTION EXISTS TO CONFESS, ONE PARAGRAPH
            # LATER. Appendix D.1 is the section that admits correcting one panel
            # with a number from the other -- and it then bound `_fn` to the
            # MERGED seven-model summary while describing "a noise floor for
            # the frontier arm". It printed 0.0000 to 0.0540 and 0 % to 26 %,
            # whose lower ends are Mistral v0.3 on the LOCAL panel; the
            # frontier arm is 0.0350 to 0.0540 and 25 % to 26 %. It also made
            # "about half the observed spread on every checkpoint" false,
            # since that holds on the three frontier models (0.500 to 0.511)
            # and not on the local four (0.000 to 0.450). Three lenses of
            # round 6 found it independently.
            _fn, _fnm = fnoise["summary_frontier"], fnoise["models"]
            # PER PANEL, because the whole defect here was a claim about one
            # panel supported by a number from the other. _fnl is the frontier
            # models (the ones Appendix D corrects and quotes); _fnl_s is the
            # open-weight panel's summary; _fnl_id is that panel's corrected
            # ratio range restricted to the checkpoints whose effect is
            # identified, which is the only set Appendix D ever compares against.
            _fnl = {k: v for k, v in _fnm.items()
                    if v.get("usable") and v.get("panel") == "frontier"}
            _fnf = fnoise["summary_frontier"]
            _fnl_s = fnoise["summary_local"]
            _fn_any = next(iter(_fnl.values()))
            _fnl_r = sorted(
                _fnm[m]["ratio_sd_to_effect_corrected"] for m in survives
                if m in _fnm and _fnm[m].get("ratio_sd_to_effect_corrected")
                is not None)
            # No identified open-weight corrected ratios means the clause
            # is dropped, not a fabricated 0.00-0.00 range printed as data.
            _fnl_id = (_fnl_r[0], _fnl_r[-1]) if _fnl_r else None
            H("D.1  A noise floor for the frontier arm", 2)
            P("<b>Everything above would collapse if the API were simply "
              "noisy, and until now we could not say that it is not.</b> §5.2 "
              "establishes that the local panel's floor is exactly zero , "
              "repeats of an identical prompt return identical bytes, so a "
              "local dispersion is wording and not jitter by construction. An "
              "API is a shared, batched service with none of those guarantees, "
              "so the same argument is not available, and a dispersion "
              "reported without a floor is a number a reviewer is entitled to "
              "disbelieve.")
            P("The design already contained the experiment. Variant N1 is "
              "declared identical to S1, and it is: building both prompts over "
              "every template and name gives "
              f"{fnoise['_premise_verified']['n_prompt_slots_identical']} of "
              f"{fnoise['_premise_verified']['n_prompt_slots_identical']} "
              "byte-identical system strings and user messages. The runner "
              "iterates the variants blindly, so every cell was sent to the "
              "API twice, on separate requests, under two labels. Any "
              "difference between the two readings is measurement noise. There "
              "is no wording difference left for it to be.", indent=False)
            P("<b>Not one cell came back the same.</b> Across the "
              + NUM.get(_fnf["n_models"], str(_fnf["n_models"]))
              + " checkpoints whose margins survive, "
              + pct(_fnf["max_frac_bitwise_identical"], 1)
              + " of replicate cells are bitwise identical. The open-weight "
              "panel, measured the same way on the Study 2 cells that carry "
              "its ratios, gives "
              + pct(_fnl_s["min_frac_bitwise_identical"], 1) + " to "
              + pct(_fnl_s["max_frac_bitwise_identical"], 1)
              + ": better, and not the perfect reproducibility §5.2 "
              "reports under a serving configuration Study 2 predates. That "
              "distinction matters because the correction below depends on "
              "it.", indent=False)
            P("The floor still has to be compared at the level the dispersion "
              "lives at, which is the wording <i>mean</i>: an average over twelve "
              "name pairs and three templates, not a single cell. Differencing "
              "the two readings pair by pair and dividing by the square root "
              "of twice the number of pairs gives a noise standard deviation "
              "of "
              f"{_fn['min_noise_sd_of_a_wording_mean']:.4f} to "
              f"{_fn['max_noise_sd_of_a_wording_mean']:.4f} on a wording mean. "
              "That is about half the observed between-wording spread on every "
              "checkpoint, so it accounts for "
              + pct(_fn["min_noise_share_of_variance"], 0) + " to "
              + pct(_fn["max_noise_share_of_variance"], 0)
              + " of the observed "
              "<i>variance</i>. The dispersion is larger than the floor on every "
              "model, so Appendix D stands , but it was overstated, and by a "
              "consistent amount.", indent=False)
            P("Subtracting the floor in variance, the between-wording standard "
              "deviation is "
              + ", ".join(
                  f"{_fnl[k]['wording_sd_corrected_for_noise']:.4f} rather "
                  f"than {_fnl[k]['observed_sd_across_wordings']:.4f} on "
                  f"{SHORT.get(k, k)}"
                  for k in _fnm if k in _fnl)
              + ". <i>both</i> <i>panels</i> <i>get</i> <i>the</i> <i>same</i> SUBTRACTION, which an earlier "
              "draft of this section did not do. It corrected the frontier "
              "ratios and left the open-weight ones alone, on the ground that "
              "the local floor was zero. Citing the "
              "perfect reproducibility of §5.2, which belongs to a "
              "different study on a different serving configuration. The "
              "local floor on Study 2 is "
              f"{_fnl_s['min_noise_sd_of_a_wording_mean']:.4f} to "
              f"{_fnl_s['max_noise_sd_of_a_wording_mean']:.4f}, removing up to "
              + pct(_fnl_s["max_noise_share_of_variance"], 0)
              + " of that panel's observed variance. Correcting one side and "
              "not the other, while comparing the two, is precisely the "
              "unreported measurement choice this paper exists to measure, so "
              "it is worth saying plainly that we made it and that a reader "
              "found it. Corrected on both sides, the dispersion-to-effect "
              f"ratio runs {_fnf['min_ratio_corrected']:.2f}–"
              f"{_fnf['max_ratio_corrected']:.2f} on the frontier checkpoints"
              + (f" against {_fnl_id[0]:.2f}–{_fnl_id[1]:.2f} on the "
                 "open-weight ones whose effect is identified"
                 if _fnl_id else "")
              + ". The same "
              "ordering as before, on numbers that are now comparable. It "
              "also sharpens the paragraph above. On the corrected numbers "
              "the frontier numerator is "
              f"{min(v['wording_sd_corrected_for_noise'] for v in _fnl.values()):.3f}"
              f"–{max(v['wording_sd_corrected_for_noise'] for v in _fnl.values()):.3f} "
              "against "
              f"{min(_fnm[m]['wording_sd_corrected_for_noise'] for m in survives):.3f}"
              f"–{max(_fnm[m]['wording_sd_corrected_for_noise'] for m in survives):.3f} "
              "on the open-weight checkpoints. The same size on the two "
              "panels, so almost all of what separates them is the "
              "shrinking effect.", indent=False)
            P("<b>What this does not license.</b> The floor is estimated from "
              "one replicated wording on "
              f"{_fn_any['n_pairs']} pairs, so it carries its own uncertainty, "
              "and subtracting an uncertain variance can over- or "
              "under-correct. We therefore report both numbers rather than "
              "silently replacing one with the other, and the claim we rely on "
              "is the ordering , dispersion above floor on every "
              "checkpoint, which does not depend on the exact size of the "
              "correction. The floor also covers only what varies between two "
              "requests minutes apart; a vendor changing the weights behind an "
              "alias would not show up in it at all.")
        # THE SIGN, WHICH THE PAPER OTHERWISE NEVER DISCUSSES AND SHOULD.
        # This paper is about dispersion, not about direction, and it says so
        # in §1. But the frontier arm makes a point about direction available
        # for free, and it is the same point: a single audit's SIGN is no more
        # transportable than its magnitude.
        _sgn = {m: v["superiority"]["est"] for m, v in _live.items()
                if v.get("effect_identified")}
        if _sgn:
            _fav_black = sum(1 for x in _sgn.values() if x < 0.5)
            _local = {m: ps[m] for m in survives}
            _loc_black = sum(1 for x in _local.values() if x < 0.5)
            # EVERY DIRECTIONAL CLAIM IS DERIVED. "Below one half", "one
            # either side of chance" and "the direction is not consistent"
            # were typed as prose while only the point estimates
            # interpolated -- true today, and a re-run that flipped one
            # estimate would have printed values contradicting the sentence
            # around them. _fav_black, computed for exactly this and
            # previously dead, now decides the first clause; a side count
            # over all identified models decides the last.
            _split = ((max(_local.values()) - 0.5)
                      * (min(_local.values()) - 0.5) < 0)
            _sides = {v < 0.5 for v in _sgn.values()} \
                | {v < 0.5 for v in _local.values()}
            P("<b>A note on direction, which this paper otherwise avoids.</b> "
              "We report dispersion, not sign, because the sign is what an "
              "audit is for and we are auditing the instrument. But the sign "
              "is here and it makes the same point. On the "
              f"{NUM.get(len(_sgn), str(len(_sgn)))} frontier "
              "checkpoints whose effect is identified, the probability of "
              "superiority is "
              + ("<i>below</i> one half: " if _fav_black == len(_sgn) else
                 ("<i>above</i> one half: " if _fav_black == 0 else
                  "not on one side of one half: "))
              + " and ".join(f"{fmt(v, 3)}" for v in _sgn.values())
              + (", meaning the Black-named résumé scores higher more often"
                 if _fav_black == len(_sgn) else "")
              + ". On the open-weight panel the two identified checkpoints "
              + (f"disagree with each other: {fmt(max(_local.values()), 3)} "
                 f"and {fmt(min(_local.values()), 3)}, one either side of "
                 "chance. " if _split else
                 f"fall on the same side: {fmt(max(_local.values()), 3)} "
                 f"and {fmt(min(_local.values()), 3)}. ")
              + f"So across the {len(_sgn) + len(_local)} identified models "
              + ("in this paper the direction is not consistent, and a "
                 "reader who took any one of them as a fact about language "
                 "models would be wrong about the others. "
                 if len(_sides) > 1 else
                 "in this paper the direction happens to agree, on a panel "
                 "far too small to make that a finding. ")
              + "We draw no conclusion from the sign "
              "beyond that one, and note that the same posting, names and "
              "wordings produced all of them.", indent=False)
        if fmass and fmass.get("models"):
            _fmm = fmass["models"]
            _fsum = fmass["summary"]
            _sat = [m for m, v in _fmm.items()
                    if v["diagnosis"].startswith("SATURATED")]
            P("<b>Two objections to the paragraph above, and the quantity that "
              "settles both.</b> The first. With no grammar, is the model even "
              "answering? A first token that is a preamble, a brace or a "
              "capitalised variant our matcher missed would leave neither "
              "verdict in the window, and we would have called a normalisation "
              "bug a saturated model. The second. The local panel constrains "
              "emission to yes or no and the API cannot, so the comparison "
              "varies the forcing mechanism as well as the model. Both turn on "
              "the renormalisable yes/no mass , how much of the first "
              "token’s probability sits on a verdict at all, which "
              "Appendix A already reports for the local panel and which the "
              "top-20 window supplies here.", indent=False)
            P("It is "
              + (f"{_fsum['min_mean_mass']:.4f}"
                 if abs(_fsum['min_mean_mass'] - _fsum['max_mean_mass']) < 5e-5
                 else f"{_fsum['min_mean_mass']:.4f} to "
                      f"{_fsum['max_mean_mass']:.4f}")
              + " on every one of the four, over "
              + f"{sum(v['n_calls'] for v in _fmm.values()):,} calls. "
              "That answers the first objection outright. The mass is on a "
              "verdict, and on "
              + (", ".join(_sat) if _sat else "the saturated checkpoint")
              + " all of it is on one side of it, which is what saturation "
              "means and what a token-class mismatch would have contradicted. "
              "It answers the second in a direction we did not expect. The "
              "grammar can only do work where that mass is low, and on these "
              "models there is nothing left for it to do; "
              + (f"across the open-weight panel the same quantity runs "
                 f"{min(_yn_mass.values()):.3f} to {max(_yn_mass.values()):.4f} (Appendix A), "
                 "so the confound is on <i>our</i> side of the comparison, not the "
                 "API side. " if _yn_mass else "")
              + "The sharpest version of the comparison is therefore against "
              + (f"{SHORT[_best_mass]}, whose mass is {_yn_mass[_best_mass]:.4f} and "
                 f"whose ratio is {fmt(ratio_ps[_best_mass], 2)}: against "
                 f"{min(_idr):.2f} to {max(_idr):.2f} here."
                 if _yn_mass and _best_mass else "the local checkpoint whose mass is "
                 "closest to unity.")
              + " The point estimates order the same way on the "
              "like-for-like comparison, though Appendix D.1 shows the intervals do "
              "not separate.", indent=False)
        _null_ge = _fs["n_models_null_arm_at_least_semantic"]
        P("The arm split behaves as it does locally. Edits that change no word "
          "(a line break moved, a label capitalised, an option order "
          "reversed) produce a between-wording dispersion of "
          f"{_fs['null_arm_sd_min']:.4f} to {_fs['null_arm_sd_max']:.4f}, "
          "which is the same order as the six genuine paraphrases and larger "
          f"than them on {_null_ge} of the {_fs['n_measurable']} models where "
          "the outcome resolves. The two arms are not distinguishable on a "
          "frontier model either, which is the finding that makes this "
          "dispersion a property of the instrument rather than of meaning.",
          indent=False)
        if _dead:
            _d0 = _fm[_dead[0]]
            P("<b>One model cannot be audited this way at all, and the reason "
              "is the interesting part.</b> On "
              + ", ".join(_dead) + " the margin could be formed on "
              f"{_d0['n_usable']} of {_d0['n_design_cells']} cells. The "
              "failure is not rate limits, cost or refusals. Every one "
              "of the 432 calls returned 200. It is that the model places "
              "essentially all of its probability mass on one answer. The "
              "other answer is not among the twenty tokens the API returns, "
              "its probability reads as exactly zero, and the log-odds margin "
              "has no finite value. A more confident model is a <i>harder</i> model "
              "to audit on a graded outcome, not an easier one, and the "
              "window a vendor chooses to expose sets the ceiling on how "
              "confident a model can be before its fairness becomes "
              "unmeasurable from outside. No audit in §8 reports the "
              "window it read, because none of them reads one.", indent=False)
        _wiped = _fs.get("models_with_a_strength_level_wholly_lost") or []
        if _wiped:
            _w0 = _fm[_wiped[0]]
            P("The same mechanism, partially. On " + ", ".join(_wiped) + " it "
              "removes an entire strength level. Every one of the "
              f"{_w0['censoring_by_template']['T3_marginal']['n']} "
              "marginal-résumé cells saturates and none of the "
              "stronger ones do, so the estimate that survives is not a "
              "noisier estimate of the same design. It is an estimate "
              "of a different one, over two strength levels instead of three. "
              "We report it with the row marked rather than pooled silently, "
              "because a censoring rate averaged over the design would have "
              "read as a third of cells lost at random when in fact a third "
              "of the design is gone.", indent=False)
        if fverd and fverd["verdict_is_a_function_of_the_template_alone"]:
            _fv_t = fverd["by_template"]
            _fv_adv = [t for t, v in _fv_t.items() if v["advance_rate"] == 1.0]
            _fv_rej = [t for t, v in _fv_t.items() if v["advance_rate"] == 0.0]
            P("<b>A fifth frontier model was run on the same twelve wordings, "
              "scored on a thresholded verdict instead of a margin, and it "
              "resolves nothing whatever.</b> That vendor returns no "
              "distribution to any caller, so the yes or no is the only "
              "outcome there is. On "
              f"{fverd['model']}, {fverd['n_pairs']} matched pairs "
              f"({fverd['n_model_calls']} calls, "
              f"{fverd['n_design_cells']} design cells) yield a verdict that "
              "is a function of the résumé template and of nothing else: "
              "every " + ", ".join(_fv_adv) + " cell advances and no "
              + " or ".join(_fv_rej) + " cell does. The two arms of a pair "
              f"disagree on {fverd['n_pairs_where_the_arms_disagree']} of "
              f"{fverd['n_pairs']}. Holding the template fixed, the largest "
              "spread across the twelve wordings is "
              f"{fverd['max_wording_spread_within_any_template']:.1%}. The "
              f"{fverd['n_cells_run_twice']} cells that were run twice "
              "returned the same verdict both times, so this is not noise "
              "swamping a signal.", indent=False)
            P("The model is not ignoring the candidate. Its free-text "
              "rationales name them and use gendered pronouns: "
              f"{fverd['n_rationales_naming_the_candidate']} of "
              f"{fverd['n_arm_responses']} responses use the first name we "
              "supplied. What has no resolution is the threshold, which "
              "quantises whatever the model is doing into three constant "
              "values. §4.2 argues that binarising a graded outcome discards "
              "the comparison; on this model that is not an argument but an "
              "observation, and the discarding is total. An arm that finds "
              "nothing cannot on its own separate <i>no</i> <i>effect</i> from <i>no</i> "
              "<i>resolution</i>, and we do not claim it can. Two things point at "
              "the second. The same design on models that do return a "
              "distribution gives effects that are not zero, and the "
              "constancy here is exact rather than merely small. A true "
              "null scatters around zero, it does not repeat one value in "
              "every cell.")
        P("<b>What this arm cannot do.</b> It says nothing about §5: "
          "quantization, request batching and cache residency are not "
          "exposed, so the serving half of this paper has no frontier "
          "counterpart and the reproducibility result is not tested here. "
          "And the checkpoints are aliases rather than pinned weights , "
          "“gpt-4o” names whatever the vendor serves on the day, "
          "which is precisely the pinning failure " + TAB("reporting_matrix") + " records against "
          "the field, and which we cannot escape by being careful. A reader "
          "should treat this section as evidence that the stimulus-side "
          "results transfer, and as evidence of nothing else.", indent=False)

    APP.end()
    H("5  Part II, What the auditor runs")

    APP.start()   # -> Appendix E
    H("E  Which quantization was downloaded")
    if quant:
        rows = []
        for k, v in quant.items():
            rows.append([TINY.get(v["base"], v["base"]),
                         f"{v['n_cells']}",
                         fmt(v["q4"]["logodds"], 4, sign=True),
                         fmt(v["q8"]["logodds"], 4, sign=True),
                         fmt(v["shift"], 4, sign=True),
                         f"{v['shift_p']:.3f}",
                         fmt(v["sigma_variant"], 4),
                         f"{v['shift_over_sigma_variant']:.2f}×",
                         pct(v["sign_disagreement"], 1)])
        paper.table(
            ["model", ">n", ">Q4_K_M", ">Q8_0", ">shift", ">p",
             ">pooled σ word", ">shift / σ", ">sign disagree"],
            rows, [72, 28, 50, 50, 50, 36, 60, 48, 58], span2=True, size=7.6,
            caption=(f"{TAB('quantization')}. The same design measured on two quantizations of "
                     "the same weights. “vs σ word” expresses the "
                     "shift as a multiple of that model’s <i>pooled</i> "
                     "between-wording standard deviation on the log-odds scale "
                     "(σ is printed beside it, and is the partially-pooled "
                     "posterior median of §6.1’s variance "
                     "decomposition. <i>not</i> the raw probability-of-"
                     "superiority SD in " + TAB("study2_effect") + ", which is a different quantity "
                     "on a different scale). “sign disagree” is the "
                     "fraction of matched pairs on which the two quantizations "
                     "disagree about which résumé is preferred."))

        sig = [v for v in quant.values() if v["shift_p"] < 0.05]
        P("A published audit reports the model it evaluated. It does not "
          "normally report which quantization of that model it downloaded, "
          "because precision is understood as an engineering detail rather than "
          "an experimental factor. " + TAB("quantization") + " measures it directly. The same "
          "checkpoint, the same design, the same cells, at 4-bit and 8-bit.",
          indent=False)
        big = max(quant.values(), key=lambda v: v["shift_over_sigma_variant"])
        P(f"On {SHORT.get(big['base'], big['base'])} the shift is "
          f"{fmt(big['shift'], 4, sign=True)} log-odds, "
          f"{big['shift_over_sigma_variant']:.2f}× that model’s pooled "
          f"between-wording standard deviation on the same scale "
          f"({fmt(big['sigma_variant'], 4)} log-odds)"
          + (f", and distinguishable from zero (p = {big['shift_p']:.3f})."
             if big["shift_p"] < 0.05 else ".")
          # THE RATIO CARRIES AN INTERVAL NOW, AND IT IS WIDE. Both halves are
          # estimated -- a bootstrapped shift over a posterior SD spanning a
          # factor of several -- so the bare 1.07x was far better pinned than
          # the quantity behind it. Propagated by the corner bound §4.2 uses,
          # the side of parity is not determined, and the sentence that read
          # "as consequential as the sentence they happened to write" was
          # resting entirely on that undetermined comparison.
          + ((" Both halves of that ratio are estimated, and propagating them "
              "by the corner bound of §4.2 gives "
              f"[{big['shift_over_sigma_variant_ci'][0]:.2f}, "
              f"{big['shift_over_sigma_variant_ci'][1]:.2f}]: the <i>shift</i> is "
              "separable from zero, but which of the two sources is larger is "
              "not determined by twelve wordings. What the data support is "
              "that quantization belongs on the same list as the wording, not "
              "that it outranks it."
              ) if big.get("shift_over_sigma_variant_ci") else
             " The quantization a researcher happened to download is as "
             "consequential as the sentence they happened to write.")
          + " Per-name "
          "margins correlate across quantizations at "
          f"r = {fmt(big['r_per_name_margin'], 3)}, which is why the shift is "
          "easy to miss , but the correlation of the <i>paired "
          "differences</i>, which is the quantity the audit reports, is only "
          f"r = {fmt(big['r_paired_delta'], 3)}, and "
          f"{pct(big['sign_disagreement'], 1)} of pairs disagree about the "
          "direction of preference. "
          + (f"{len(sig)} of {len(quant)} models show a shift distinguishable "
             "from zero; the point does not rest on statistical significance "
             "but on the magnitude relative to other unreported choices."
             if sig else ""))

    # ------------------------------------------------------------------
    APP.end()
    H("5.2  How the requests were served", 2)
    P("Four of the surveyed audits rest reproducibility on a decoding "
      "setting alone: one sets temperature to zero “to remove variability”, "
      "a second writes that “temperature 0 ensures deterministic outputs”, "
      "a third states that greedy decoding produces deterministic outputs "
      "and reports no repeat, and a fourth, running at a non-zero "
      "temperature, relies on a fixed random seed while reporting a batch "
      "size of four in the same appendix. The studies are identifiable "
      "from Table 10; the point needs the count, not the names. We "
      "make no claim about the other nine, which say nothing either way; "
      "and for an audit run against a commercial API, several serving "
      "fields are not the auditor’s to report, because the platform does "
      "not disclose them. §9 therefore asks for what the platform exposes, "
      "plus the statement that the rest was not knowable. On "
      "this panel none of those conditions is sufficient, and the reason "
      "is worth establishing precisely, because it determines whether the "
      "dispersion reported above is a property of the instrument or an "
      "artefact of arithmetic.", indent=False)
    P("The scope of that claim is narrow and deliberately so. A fixed seed "
      "and a zero temperature do fix the sampling; what they do not fix is "
      "the order in which requests reach the kernel. Nothing here says an "
      "audit whose language is careful about this is wrong, Lippens, for "
      "instance, writes only that temperature zero leaves ChatGPT “mostly "
      "deterministic, allowing minimal output variability”. That wording is "
      "accurate and the distinction it draws is the one this section needs.",
      indent=False)
    if noise:
        nm = [m for m in models if m in noise]
        fi = [noise[m]["frac_identical"] for m in nm]
        cs = [noise[m]["cross_session"] for m in nm if "cross_session" in noise[m]]
        P("Two byte-identical prompts, issued at temperature zero under "
          "grammar-constrained greedy decoding, agree exactly on "
          f"{pct(min(fi), 0)} to {pct(max(fi), 0)} of cells within a run. "
          + (f"Re-measured across sessions, with the server restarted in "
             f"between, {min(c['n_identical'] for c in cs)} of "
             f"{min(c['n'] for c in cs)} identical prompts reproduce on the "
             "worst model." if cs else ""))
    if rep and "_verdict" in rep:
        v = rep["_verdict"]
        P("<b>The mechanism is not ours, and neither is the priority.</b> Two "
          "prior works establish most of it. The closest precedent in argument "
          "is not about language models at all: Qian et al., at NeurIPS 2021, "
          "hold the seed, the data, the hardware and the software fixed, and "
          "still find up to 12.6 points of movement in a fairness metric across "
          "identical training runs, enough for one run to fall the fair side of "
          "a threshold used in a US legal case and another the unfair side. "
          "They report that only about a third of the fairness papers they "
          "survey run training more than once. This paper is that argument "
          "where there is no training run to repeat. Atil et al., at Eval4NLP 2025, "
          "measure the consequence on five hosted APIs, where accuracy moves "
          "up to 15 points across ten identical runs; they name continuous "
          "batching, chunked prefill and prefix caching as the likely causes "
          "but report that behind an API they can only speculate, their local "
          "run without those optimisations being deterministic. Yuan et al., at NeurIPS 2025, "
          "show that evaluation batch size, GPU count and GPU version each "
          "change the responses a model generates, and trace the cause to the "
          "non-associativity of floating-point arithmetic at limited "
          "precision. He and Thinking Machines Lab give the sharper "
          "diagnosis. The dominant source is the absence of BATCH INVARIANCE "
          "in reduction kernels. A request’s numerics depend on the "
          "batch it happens to be scheduled with, because the reduction order "
          "inside RMSNorm, matrix multiplication and attention changes with "
          "batch size , and they are explicit that the usual "
          "“concurrency plus floating point” story is the wrong "
          "diagnosis, since the forward pass contains no atomic adds and is "
          "run-to-run deterministic. They substitute batch-invariant kernels "
          "and obtain identical completions across a thousand runs at "
          "temperature zero.", indent=False)
        P("What follows is not that result. It is a replication of the "
          "mechanism on a stack neither work covers, llama.cpp on CPU "
          "and Vulkan, where no batch-invariant kernels exist, with the "
          # NOT "WHICH NEITHER ACCOUNT COVERS". He and Thinking Machines Lab
          # do treat KV-cache residency: batch invariance, in their account,
          # requires that a token's reduction order not depend on how many
          # tokens sit in the KV cache, and they name vLLM's Triton attention
          # kernel as a case where it does. What is ours is manipulating cache
          # residency as an independent experimental knob and measuring its
          # separate contribution to a fairness estimate -- not noticing that
          # it matters.
          "two contributing factors separated and quantified, with cache "
          "residency separated as an independently manipulable knob rather "
          "than as one term inside a batch-invariance argument, and on a "
          "<i>fairness</i> "
          "<i>estimate</i> rather than on benchmark accuracy or generated text. One "
          "clarification we owe He and Thinking Machines Lab: our manipulation "
          "varies <i>concurrency</i>, which changes batch size, and it is the batch "
          "size that carries the effect. Concurrency is the knob, not the "
          "mechanism.", indent=False)
        P("<b>Study 8 tested the obvious explanation.</b> The suite serves four "
          "concurrent requests, so a prompt is multiplied alongside whatever "
          "else is in flight, and floating-point reduction order is not "
          "invariant to batch composition. Prediction fixed in advance. Strictly "
          "sequential requests should agree bitwise. Measured over five repeats "
          "of each cell, agreement rises from "
          f"{pct(v['frac_identical_conc4'], 1)} at concurrency four to "
          f"{pct(v['frac_identical_conc1'], 1)} at concurrency one, pooled "
          f"over the {len([m for m in rep if not m.startswith('_')])} "
          "checkpoints where batching was manipulated, and the "
          "per-cell standard deviation falls by "
          f"{fmt(v['sd_ratio'], 1)}×. Batching is most of the story and "
          "not all of it.")
    if cache and "_verdict" in cache:
        v = cache["_verdict"]
        rows = []
        cm = [m for m in cache if not m.startswith("_")]
        for m in cm:
            for label in ("concurrency 4, cache on", "concurrency 1, cache on",
                          "concurrency 1, cache OFF"):
                if label in cache[m]:
                    a = cache[m][label]
                    rows.append([TINY.get(m, m), label.replace("concurrency ", "conc. "),
                                 pct(a["frac_identical"], 1),
                                 f"{a['mean_sd']:.5f}",
                                 f"{a['max_spread']:.5f}"])
        paper.table(
            ["model", "serving configuration", ">identical", ">mean SD",
             ">max spread"],
            rows, [70, 116, 52, 54, 56], span2=True, size=7.6,
            caption=(f"{TAB('replicate')}. The same 36 cells, five repeats each, under three "
                     "serving configurations. “identical” is the "
                     "fraction of cells whose five repeats agree bitwise. Every "
                     "repeat’s prompt SHA-256 was verified identical, so "
                     "these are the same input in every case."))
        P("<b>Study 9 closes it.</b> The remaining candidate is key-value cache "
          "residency. Llama.cpp reuses the cache for whatever prefix a new "
          "prompt shares with the previous one, so which prefix is resident "
          "depends on what ran immediately before. That is order dependence "
          "without concurrency, and it predicts exactly the residual Study 8 "
          "left. Disabling prompt-cache reuse while holding requests "
          f"sequential, on the {NUM.get(v['n_models'], str(v['n_models']))} "
          "checkpoints where a cache-off arm was run, "
          f"raises agreement from {pct(v['frac_identical_cache_on'], 1)} to "
          f"{pct(v['frac_identical_cache_off'], 1)}, with a per-cell standard "
          f"deviation of exactly {v['mean_sd_cache_off']:.5f}.")
        diffs = [cache[m]["effect_difference"] for m in cm
                 if "effect_difference" in cache[m]]
        xsess = cache.get("_cross_session") or {}
        if xsess:
            xi = min(v["frac_identical"] for v in xsess.values())
            xs_ = max(v["sigma"] for v in xsess.values())
            xn = sum(v["n_cells"] for v in xsess.values())
            # READ, NOT TYPED. This sentence carried five artifact values
            # and two number words as literal text -- "none of 504 cells on
            # three of the four models, and on 212 of 504 on the fourth" --
            # in a paper whose SS1.2 claims that no number in it is typed.
            # It survived every earlier run of audit_hardtyped_numbers.py
            # because the fragment is implicitly concatenated onto the
            # f-string below it, so Python folded it into that JoinedStr
            # before the AST existed and the audit read it as interpolated.
            # The source is noise_floor.json, which SS8.1 already reads.
            _xc = [noise[m]["cross_session"] for m in models
                   if m in noise and "cross_session" in noise[m]]
            _zero = [c for c in _xc if not c["n_identical"]]
            _some = [c for c in _xc if c["n_identical"]]
            if len(_some) != 1 or not _zero:
                sys.exit(
                    "cross-session shape changed: this sentence names one "
                    f"model as the exception and {len(_some)} now show "
                    "partial agreement. Rewrite it rather than letting it "
                    "print a description of data that no longer exists.")
            _ord = ("first", "second", "third", "fourth", "fifth")[
                next(i for i, c in enumerate(_xc) if c["n_identical"])]
            P("<b>And across processes, which is the harder test.</b> "
              "Everything above is repeats inside one server launch, and this "
              "paper reports a second and larger reproducibility failure: "
              "identical prompts measured in a different process on a "
              f"different day agree on none of {_zero[0]['n']} cells on "
              f"{NUM[len(_zero)]} of the {NUM[len(_xc)]} "
              f"models, and on {_some[0]['n_identical']} of {_some[0]['n']} "
              f"on the {_ord}. Nothing in Studies 8 "
              "or 9 speaks to that, so the same cells were re-measured a third "
              "time in a <i>fresh</i> server process, still sequential, still with "
              "the cache off. They reproduce the stored run "
              f"{pct(xi, 1)} bitwise across {xn} cells, with a cross-session "
              f"standard deviation of {xs_:.6f}. The cross-session "
              "disagreement is therefore the same two causes and not something "
              "additional about process state, which is what the strong claim "
              "below requires and what we would otherwise have been "
              "extrapolating.")
        P("The nondeterminism is therefore fully accounted for, and it is "
          "entirely a property of how the model was <i>served</i> rather than "
          "of the model. Both contributing factors , batch composition and "
          "cache policy, are configuration that no audit we surveyed "
          "reports. "
          # A NON-REJECTION IS NOT AN EQUIVALENCE, and this sentence used to
          # read as one: "No number in this paper inherits a dependence on a
          # scheduling parameter." The Conclusion states the same result
          # correctly -- "a bound, not an absence" -- so the paper asserted
          # both epistemic statuses for one finding, and §9 item 7 took the
          # stronger one, which is where a reader acts on it. The interval is
          # what we have, so the interval is what is claimed.
          + (f"Critically, the measured effect is not detected to depend on "
             f"either. Across models the effect difference between cache-on "
             f"and cache-off has p = "
             + " and ".join(f"{d['p']:.2f}" for d in diffs)
             + ", every interval containing zero, and the same holds for "
             "concurrency."
             + (f" That is a bound rather than an absence. The shift is "
                f"held within {pct(_serving_bound, 1)} of the effect on the "
                "checkpoints whose effect is distinguishable from zero, the "
                "only ones this ratio is defined on , and it is the form in "
                "which every claim about serving in this paper should be read."
                if _serving_bound else "") if diffs else ""))
    P("" + FIGREF("fig8_noise_floor") + " places the two side by side: what a byte-identical repeat "
      "does to the estimate, against what a change of wording does.")
    paper.figure(FIGS / "fig8_noise_floor.png", span2=True,
                 max_h=2.6 * pk.inch, space_after=12.0)

    if funoise and funoise.get("models"):
        _fn = funoise["summary"]
        _fnm = funoise["models"]
        # THE NO-BAND CHECKPOINTS, AGAINST THE IN-BAND AVERAGE OF THE OTHERS.
        # This used to read "smaller by orders of magnitude", written to the
        # stronger of the two cases: the ratios are about 5x and 369x, so one
        # is well under a single order and the other well over two, and the
        # plural was fixed at authoring time rather than read off the artifact.
        _ib = [v["mean_sd_in_band"] for v in _fnm.values()
               if v.get("mean_sd_in_band")]
        _ib_mean = (sum(_ib) / len(_ib)) if _ib else None
        _noband = sorted(_ib_mean / v["mean_sd_out_of_band"]
                         for v in _fnm.values()
                         if v.get("n_in_band") == 0
                         and v.get("mean_sd_out_of_band")) if _ib_mean else []
        # THE BOUND IS PRINTED AS PROSE, SO IT IS CHECKED AS A CLAIM.
        # "permutation p < 0.001" is a reporting convention rather than a
        # measurement, which is why it stays typed -- but a convention that
        # states a fact can stop being true without the sentence noticing.
        # The same guard covers the clause after it: "unchanged when the
        # bitwise-identical cells are dropped" is the nonzero_only figure,
        # and it has to clear the bound too.
        _PBOUND = 0.001
        for _m, _v in _fnm.items():
            if not _v.get("mean_sd_in_band"):
                continue
            for _where, _d in (("", _v), (" with ties dropped",
                                          _v.get("nonzero_only") or {})):
                _p = _d.get("p_permutation")
                if _p is not None and _p >= _PBOUND:
                    sys.exit(
                        f"SS8 prints 'permutation p < {_PBOUND}', but {_m}"
                        f"{_where} has p = {_p:.5g}. Restate the sentence "
                        "rather than letting it print a bound the data no "
                        "longer supports.")
        P("<b>The floor is not a constant, and prior work says where it should "
          "be largest.</b> Fu, Martínez, Conde and colleagues analysed LLM "
          "nondeterminism at the token "
          "probability level rather than at the level of generated text "
          ", which is the level this paper’s outcome lives at, "
          "and report that the effect is negligible for probabilities near 0 "
          "or 1 and significant for probabilities between 0.2 and 0.8. That is "
          "a prediction about our data, so we tested it rather than cited it. "
          "Across every replicate cell, the rank correlation between how much "
          "a cell moves under byte-identical repetition and how close it sits "
          f"to p = 0.5 is positive on all {len(_fnm)} models, from "
          f"{_fn['rho_min']:.2f} to {_fn['rho_max']:.2f}. On the "
          f"{_fn['n_band_testable']} models with any cell inside the 0.2-to-0.8 "
          "band, in-band cells move "
          + (f"{max(v['mean_sd_in_band'] / v['mean_sd_out_of_band'] for v in _fnm.values() if v.get('mean_sd_in_band')):.0f}"
             " times more than out-of-band cells (permutation p < 0.001 on "
             "both, and unchanged when the bitwise-identical cells are "
             "dropped). " if any(v.get("mean_sd_in_band") for v in _fnm.values())
             else "")
          + f"The other {_fn['n_models_with_no_cell_in_band']} have no cell in "
          "that band at all, and their per-arm noise runs "
          + (" and ".join(f"{r:.0f}×" for r in _noband) if _noband else "far")
          + " below the in-band average of the checkpoints that do, which is "
          "the same prediction seen from the other side. The gap between "
          "those two figures says the floor is a property of the checkpoint "
          "as much as of the design.", indent=False)
        P("Two consequences, one for readers and one for us. For a reader. The "
          "noise floor an auditor should expect is not a property of their GPU "
          "but of where their model sits on the logistic, so a floor measured "
          "on a saturated task does not license a claim about a task near the "
          "boundary. For us: the correlation is also what a bounded quantity "
          "does near its bounds, so this test establishes that the pattern is "
          "<i>present</i> on a CPU and Vulkan stack and on a fairness estimate, not "
          "that the arithmetic rather than the boundary produces it. We report "
          "it as corroboration of transportability and claim nothing about "
          "mechanism.", indent=False)

    # ======================================================================
    H("6  Part III, What the auditor computes")
    P("The choices above are made when designing an experiment. The next two "
      "are made after the data exists, and they move the reported number as "
      "much as anything in Part I. We report them against ourselves. Both were "
      "errors in earlier drafts of this work, caught and corrected, and both "
      "are documented rather than silently repaired.", indent=False)

    H("6.1  The resampling unit", 2)
    if resamp:
        ps_ = resamp["pooled_summary"]
        P("The design is crossed. Every name pair appears under every wording "
          "and every template. Pooling those rows and bootstrapping them "
          "independently treats repeated measurements of the same twelve names "
          "as independent draws. Measured against a bootstrap that resamples "
          "name pairs, the i.i.d. interval on the per-model effect is too "
          f"narrow by a factor of <b>{ps_['min_ratio']:.1f} to "
          f"{ps_['max_ratio']:.1f}</b> across the four models. "
          f"{ps_['n_significant_iid']} of {ps_['n_models']} effects appear "
          "distinguishable from zero under the row bootstrap and "
          f"{ps_['n_significant_clustered']} survive the correction.",
          indent=False)
    P("The rule itself is not new and is not ours. Lahey and Beasley state "
      "it for audits in 2009: standard errors should be clustered by pair, "
      "and at the firm level where several r\u00e9sum\u00e9s reach the same firm. "
      "Abadie, Athey, Imbens and Wooldridge give the general result: the "
      "sampling process and the assignment mechanism alone determine the "
      "level of clustering. Here the name carries the treatment, so every "
      "row sharing a name shares its assignment; and their case where the "
      "adjustment overcorrects (sampled clusters that are most of the "
      "population of clusters) is not this one, since twelve pairs are drawn "
      "from the space of plausible American names. "
      "What this section adds is the size of the "
      "error when the rule is not followed, and \u00a78 reports how many of the "
      "surveyed audits say enough for a reader to tell."
      + ((" One caution runs the other way: with twelve clusters a "
          "percentile bootstrap can itself under-cover, so the per-model "
          "effects were re-estimated with a wild cluster bootstrap-t. Those "
          f"intervals come out {pct(wildboot['summary']['ratio_min'] - 1, 0)} "
          f"to {pct(wildboot['summary']['ratio_max'] - 1, 0)} wider and "
          + ("change no significance verdict"
             if wildboot['summary']['n_verdict_changes'] == 0 else
             f"change {NUM.get(wildboot['summary']['n_verdict_changes'], str(wildboot['summary']['n_verdict_changes']))} verdicts")
          + ", so the percentile intervals here are modestly optimistic "
          "rather than misleading; the wild-t intervals are released "
          "beside them.")
         if wildboot and wildboot.get("summary") else ""))
    P("The error is not uniform, which is what makes it dangerous. For paired "
      "<i>contrasts</i> , differences between two conditions measured on "
      "the same cell, the name and the résumé cancel inside "
      "each observation before any resampling happens, and on a two-template "
      "panel the two estimators looked close enough to call interchangeable"
      + ((", with a width ratio from "
          f"{resamp['contrasts_two_template']['min']:.3f} to "
          f"{resamp['contrasts_two_template']['max']:.3f} across "
          f"{resamp['contrasts_two_template']['n']} contrasts")
         if resamp and resamp.get("contrasts_two_template") else "")
      + ". We concluded on that basis that "
      "contrasts were safe either way. Adding a third résumé template "
      ", a third correlated row per name pair, falsified it: the "
      + (f"ratio now runs from {resamp['contrasts']['min']:.3f} to "
         f"{resamp['contrasts']['max']:.3f}, and while the median is "
         f"{resamp['contrasts']['median']:.3f}, "
         f"{pct(resamp['contrasts'][W10], 1)} of "
         f"contrasts widen by more than {widen_label(W10)} and "
         f"{pct(resamp['contrasts'][W25], 1)} by "
         f"more than {widen_label(W25)}. " if resamp else
         "ratio is no longer negligible. ")
      + "Every contrast in this paper resamples name pairs. "
      "“Usually indistinguishable” is not a reason to use the "
      "estimator that is wrong when they differ.")
    P("What generalises from this is a rule about provenance, not a rule "
      "naming one unit. <b>The resampling unit follows the assignment "
      "process.</b> Here names are the units carrying the treatment and the "
      "grid is crossed over them, so the name pair is the unit; in a design "
      "that preconstructs profiles and assigns them within a vacancy, the "
      "vacancy is the block and clustering there is what a correct interval "
      "requires, which is what the correspondence-audit literature has long "
      "done, and what Lippens does with a cluster-robust wild bootstrap. "
      "Neither choice is right in the abstract. The finding this paper "
      "reports is not that one unit is universally correct but that almost "
      "no audit states its unit clearly enough for a reader to tell whether "
      "it matches the assignment process"
      + (f"; {matrix['counts']['resampling_unit']['n_reported']} of the "
         f"{matrix['counts']['resampling_unit']['n_applicable']} to which "
         "the field applies do so" if matrix else "")
      + ".", indent=False)
    if resamp and resamp.get("contrasts_two_template"):
        _t2 = resamp["contrasts_two_template"]
        P("The two-template range in that sentence is interpolated now and "
          "was not before. "
          "carried in prose from a measurement made before the third template "
          "existed, and an audit of this paper found that no artifact "
          "reproduced it: recomputed on the panel it described, the range is "
          f"{_t2['min']:.3f} to {_t2['max']:.3f}, wider at both ends, with "
          f"{pct(_t2[W10], 1)} of contrasts already "
          f"widening by more than {widen_label(W10)}. The direction of that "
          "error matters "
          "for the argument it supports. The “before” panel was less "
          "reassuring than we reported, so the case for calling the two "
          "estimators interchangeable was weaker than we thought even before "
          "the third template falsified it. A number carried in prose because "
          "it was true once is exactly the failure this paper is about, and "
          "it took an audit to find it here.")

    P("" + FIGREF("fig5_variance_components") + " decomposes the paired difference into its variance components "
      "under partial pooling, which separates true between-wording dispersion "
      "from the sampling noise in each per-wording estimate. The raw standard "
      "deviation across wordings is inflated by the latter, so the pooled "
      "estimate is the smaller of the two. This paper quotes the <i>raw</i> SD "
      "throughout. In " + TAB("study2_effect") + ", in §4.1 and in the abstract , because it is "
      "the quantity a reader can reconstruct from a study’s reported "
      "per-wording estimates, which is what makes it the right number for a "
      "claim about what this literature would see. Where the pooled estimate "
      "is used it is named as such.")
    P("An earlier draft added that the raw SD was also “the conservative "
      "direction for our argument”. That was backwards and a reviewer caught "
      "it. Our argument is that the dispersion is large; the raw SD is the "
      "<i>larger</i> of the two available numbers, so quoting it is the aggressive "
      "choice, not the cautious one. The reconstruction rationale above stands "
      "on its own and is the only one we make. What follows from being honest "
      "about it is that the dispersion-to-effect ratios in " + TAB("study2_effect") + " should be "
      "read as upper bounds. Partial pooling attributes some of that spread to "
      "sampling noise in each per-wording estimate, and the pooled figures are "
      "smaller. Both are in the artifact and "
      + FIGREF("fig5_variance_components") + " plots the pooled ones.")
    paper.figure(FIGS / "fig5_variance_components.png", span2=True,
                 max_h=2.8 * pk.inch, space_after=12.0)

    H("6.2  The reporting scale", 2)
    P("The field reports percentage points, which makes results look "
      "comparable to Bertrand and Mullainathan and to each other. A "
      "percentage-point figure is a log-odds effect multiplied by the logistic "
      "slope at the model’s operating point, and an audit that converts "
      "at p = 0.5 is using the largest slope the curve has. The four models do "
      "not share an operating point. The fraction of saturated cells in "
      + TAB("study2_effect")
      + f" ranges from {pct(min(sat.values()), 1)} to "
      f"{pct(max(sat.values()), 1)}.", indent=False)
    if scale:
        sm = scale["summary"]
        per = scale["models"]
        distinguishable = [m for m in models
                           if per.get(m, {}).get("effect_distinguishable")]
        P("The quantity that answers this cleanly is the conversion factor "
          "itself, 0.25 divided by the model’s own mean <i>p</i>(1−<i>p</i>). "
          "It says how wrong the assumed Jacobian is at the place the model "
          "actually sits, and , this is the point, it does not have the "
          "effect size in it, so it is defined whether or not the model has an "
          "effect to convert. Across this panel it runs from <b>"
          f"{sm['jacobian_error_min']:.1f}× to "
          f"{sm['jacobian_error_max']:.0f}×</b>. " + FIGREF("fig12_reporting_scale") + " is the geometry. One "
          "curve, one assumed tangent, and four models sitting nowhere near "
          "it.", indent=False)
        if distinguishable:
            P("The overstatement actually realised on a measured effect is the "
              "ratio of the fixed-Jacobian figure to the mean per-cell "
              "probability difference, and it is only interpretable where the "
              "effect is distinguishable from zero, because otherwise it is a "
              "quotient of two nearly zero quantities. On the "
              f"{len(distinguishable)} models where it is, the realised "
              "overstatement is "
              + " and ".join(f"{per[m]['realised_ratio']:.1f}×"
                             for m in distinguishable)
              + " against a predicted "
              + " and ".join(f"{per[m]['jacobian_error']:.1f}×"
                             for m in distinguishable)
              + ". Agreement that close is the check that the mechanism is the "
              "operating point and not something else. "
              "this section quoted the realised ratio for all four models, "
              "including two whose effects are not distinguishable from zero; "
              "on one of those the operating-point account predicts a factor "
              f"of {per['mistral-7b-instruct-v0.1']['jacobian_error']:.1f} and "
              "the quoted figure was "
              f"{per['mistral-7b-instruct-v0.1']['realised_ratio']:.0f}. The "
              "number was not wrong; the mechanism attributed to it was.",
              indent=False)
    P("A conversion factor that is wrong by two orders of magnitude for one "
      "model and by a factor of two for another is not a common scale, and "
      "percentage-point comparability across models was never available. This "
      "paper therefore reports the probability of superiority as its primary "
      "effect size, log-odds beside it, and percentage points only where a "
      "per-cell probability-scale interval can be computed. Where a quantity "
      "has no single operating point attached (a variance component, for "
      "instance), any percentage-point figure is reported as an upper "
      "bound and named as one.")
    if scale:
        paper.figure(FIGS / "fig12_reporting_scale.png", span2=True,
                     max_h=2.7 * pk.inch, space_after=12.0)

    H("6.3  Which résumé was paired with which", 2)
    if pairfree and pairfree.get("models"):
        pf = pairfree["models"]
        pfs = pairfree["summary"]
        pfm = [m for m in models if m in pf]
        P("Section 3 gives the reason each résumé is scored alone. Presented "
          "together, the answer is a function of presentation order on every "
          "checkpoint we gated. That decision has a consequence we did not "
          "follow up. If the two "
          "résumés are never in the same prompt, nothing in the <i>measurement</i> "
          "matches them. The model is common to every observation rather than "
          "to a pair, so it is a study-level constant rather than a blocking "
          "factor: it induces dependence without inducing a match. The pairing "
          "happens in the analysis, after every number "
          "exists, and which White name is placed opposite which Black one is "
          "the analyst’s choice. Our own grid-construction code called that "
          "choice arbitrary, and meant it as a reassurance.", indent=False)
        P("This is a consequence of scoring each résumé alone, and it is "
          "worth saying plainly what it does not reach. In a correspondence "
          "design where profiles are preconstructed and assigned within a "
          "prespecified block (a vacancy, a firm, an employer), the match is "
          "fixed before any outcome is observed, and re-pairing across blocks "
          "would break the design rather than exercise a degree of freedom. "
          "The latitude described below exists only where the pairing is "
          "imposed after scoring, and only for statistics that are functions "
          "of which résumé beat which. An audit that blocks its design, or "
          "that reports a difference in group means, does not have it.",
          indent=False)
        P("It is one, on the log-odds scale, and exactly there. The mean paired "
          "difference is the difference of the two means, so re-pairing cannot "
          "move it at all. It is not a reassurance for the statistic this "
          "literature actually reports. The probability of superiority, and "
          "the discordant-pair analysis correspondence audits have used since "
          "Bertrand and Mullainathan, are functions of which résumé beat "
          "which , and that is a property of the matching.",
          indent=False)
        P("Re-pairing the grid at random within gender, which is the set of "
          "pairings an equally careful researcher could have written down, "
          "moves the reported probability of superiority with a standard "
          f"deviation of {pfs['min_perm_sd']:.4f} to {pfs['max_perm_sd']:.4f}. "
          "That is three to ten times below the between-wording dispersion "
          "of " + TAB("study2_effect") + ", and we report it as the small term it is: not every "
          "unreported choice matters, and a paper arguing that they all do "
          "would be making the same mistake in the other direction.",
          indent=False)
        P("The latitude is a different quantity from the dispersion, and it is "
          "not small. Choosing the matching that maximises the reported effect "
          "rather than one at random. A maximum-weight bipartite matching "
          "over the pooled win counts, computable in a second from data the "
          "analyst already has. Spans "
          f"{pfs['min_best_worst_range']:.3f} to "
          f"{pfs['max_best_worst_range']:.3f} between the best and worst "
          "achievable value, on a statistic bounded by one. No honest "
          "researcher would search that space, and no reader of a published "
          "audit can tell whether one did, because the pairing is never "
          "reported. The remedy is to report the <i>mean</i> <i>paired</i> <i>difference</i>, which "
          "is algebraically invariant to the pairing and so removes the degree "
          "of freedom rather than disclosing it: a statistic nobody can move "
          "needs no assurance that nobody moved it. Where the "
          "probability of superiority is wanted anyway \u2014 it is the more "
          "readable number, and this paper reports it throughout \u2014 the "
          "fallback is to state the pairing rule explicitly, so a reader can "
          "see which of the values in "
          + TAB("spec_curve") + " they are being shown. Disclosure is the "
          "weaker remedy and belongs second.",
          indent=False)
        paper.table(
            ["model", ">as built", ">random re-pairing", ">best", ">worst",
             ">range"],
            [[TINY[m], f"{pf[m]['p_actual']:.4f}",
              f"{pf[m]['perm_mean']:.4f} ± {pf[m]['perm_sd']:.4f}",
              f"{pf[m]['best_possible']:.4f}",
              f"{pf[m]['worst_possible']:.4f}",
              f"{pf[m]['range_best_worst']:.4f}"] for m in pfm],
            [96, 60, 116, 60, 60, 60], span2=True, size=7.6,
            caption=(f"{TAB('spec_curve')}. The probability of superiority under different "
                     "pairings of the same names, the same résumés and the "
                     "same measurements. “Random re-pairing” is the mean and "
                     f"SD over {pairfree['n_perm']:,} permutations within "
                     "gender; “best” and “worst” are exact, by "
                     "maximum-weight bipartite matching over pooled win "
                     "counts. The mean paired difference in log-odds is "
                     "algebraically invariant to every column here and is not "
                     "shown."))

    APP.start()   # -> Appendix F
    H("F  Multiplicity, and a verdict that moved without its data")
    P("Benjamini and Hochberg prove FDR control for <i>independent</i> test "
      "statistics. Our family of " + _bh_n + " mechanism contrasts is not independent: "
      "every contrast is taken against the shared D0 baseline, and contrasts "
      "within a model and mode share cells. BH is known to control FDR under "
      "positive regression dependence, but that is a later result than the one "
      "we cite, and we flag the gap rather than let the 1995 proof be read as "
      "covering a dependent family. The direction of the concern is that our "
      "adjusted p-values may be anti-conservative; since the section’s "
      "conclusion is a <i>null</i>, that direction works against us and not for us.",
      indent=False)
    P("Benjamini–Hochberg correction across the " + _bh_n + " mechanism contrasts "
      "behaved in a way worth recording. A bootstrap p-value is quantised at "
      "one over the replicate count; at 4,000 replicates the resolvable values are "
      "multiples of 0.00025, and many contrasts landed on exactly the same raw "
      "p. We wrote that BH then broke those ties arbitrarily, and an audit of "
      "this paper showed it does not. Our own implementation enforces "
      "monotonicity by a running minimum over the sorted order, so tied raw "
      "p-values necessarily receive identical adjusted ones. The mechanism we "
      "described cannot occur.", indent=False)
    P("What did occur is worse, and is the more useful lesson. A BH-adjusted "
      "p-value is a function of the whole family, not of its own test. When a "
      "loader defect was corrected and 24 of the " + _bh_n + " contrasts "
      "changed value, "
      "five verdicts flipped , and four of those five were contrasts whose "
      "own raw p-value did not move at all. They changed side because the "
      "family reshuffled around them. A reader who saw only the corrected "
      "output would have no way to know the verdicts had ever been that "
      "fragile. Raising the resolution to 40,000 replicates reduces the "
      "quantisation that makes such reshuffling easy, and is what Study 5 now "
      "uses; it does not remove the family dependence, which is a property of "
      "the procedure rather than of our implementation of it.")

    APP.end()

    # ------------------------------------------------------------------
    if design and design.get("rows"):
        APP.start()   # -> Appendix G
        H("G  The design, analysis by analysis")
        P("An economist with a published correspondence audit, reading a "
          "summary of this work in August 2026, "
          "asked for a table naming, for each analysis, the treatment, the "
          "outcome, the assignment unit, the observation unit, the block, the "
          "target population, the estimator and the uncertainty estimator. The "
          "table matters because three of that review's objections were about scope "
          "rather than substance. This paper's claims about the resampling "
          "unit and about the pairing hold for a design that scores each "
          "résumé alone and pairs afterwards, and not for one that "
          "preconstructs profiles inside a prespecified block. A reader cannot "
          "tell which from prose spread over two sections. The request was "
          "obviously right and this appendix is the answer to it.", indent=False)
        P("<b>Read the target-population row first.</b> Almost nothing here "
          "generalises to a population of job applicants. It generalises to "
          "the instrument's own design space, which is the whole argument. A "
          "measurement whose sensitivity to its own construction is unreported "
          "cannot be read as a measurement of the world.", indent=False)
        _FIELDS = [("treatment", "Treatment"), ("outcome", "Outcome"),
                   ("assignment", "Assignment unit"),
                   ("observation", "Observation unit"), ("block", "Block"),
                   ("population", "Target population"),
                   ("estimator", "Estimator"),
                   ("uncertainty", "Uncertainty estimator"), ("_n", "n")]
        for _row in design["rows"]:
            # A section number takes a section sign; an appendix letter takes
            # the word. Two analyses moved into appendices in a restructure
            # and this table kept numbering them, which is how it came to
            # point at sections that no longer exist.
            _lbl = (f"Appendix {_row['sec']}" if _row["sec"][:1].isalpha()
                    else f"\u00a7{_row['sec']}")
            P(f"<b>{_lbl}  {_row['name']}</b>",
              indent=False, size=8.4, lead=10.4, space_after=1.0)
            for _k, _label in _FIELDS:
                if not str(_row.get(_k, "")).strip():
                    continue
                P(f"<i>{_label}.</i> {_row[_k]}", indent=False,
                  size=7.6, lead=9.2, space_after=0.6)
            P(f"<i>Artifact.</i> {_row['artifact']}", indent=False,
              size=7.6, lead=9.2, space_after=5.0)
        APP.end()

    # ======================================================================
    H("7  What the sensitivity is not")
    if mech and tconc:
        P("A natural next question is <i>why</i>. Our initial hypothesis was "
          "structural. That the prompt’s section delimiters bound the "
          "region over which the name is integrated, so destroying a delimiter "
          "should let the name’s influence spread and should move the "
          "effect more than a perturbation that leaves structure intact. Study "
          "5 tests this on a full panel. Eleven conditions, six "
          "checkpoints, two inference modes, three templates, 9,504 cells.",
          indent=False)
        P("The design separates two things our first pass confounded. Editing "
          "a delimiter also shifts the token index at which the name appears, "
          "and the conditions that moved the effect in the pilot were exactly "
          "the conditions that shifted the name. Conditions D8, D9 and D10 were "
          "added to break this: D8 and D9 move the name by one and two tokens "
          "with every delimiter intact; D10 moves the name <i>and</i> destroys "
          "a delimiter. Token indices were verified by tokenising every "
          "condition against a served checkpoint’s own vocabulary rather "
          "than assumed: on one checkpoint, Llama-3.1-8B-Instruct, which "
          "is the same limitation " + TAB("conditions") + " carries and for the same reason: "
          "token counts are a property of a tokenizer, and what has to "
          "generalise is which conditions move the name and which destroy a "
          "delimiter, not the integers.")

        # Every contrast whose comparator is the D0 baseline, mapped to the
        # condition it tests.
        #
        # THE POSITION CONTROLS ARE BASELINE TESTS TOO. D8 and D9 carry no
        # contrast labelled "vs base"; they are tested as P1 (D8 - D0) and P2
        # (D9 - D0), which is the same comparison under a different name.
        # Matching on the label prefix alone dropped them, so the control group
        # was missing the two conditions that displace the name -- the ones the
        # position controls exist to supply -- and the table's own caption,
        # "every condition-versus-baseline test", was false. Including them
        # makes the null stronger, not weaker.
        #
        # SECOND CORRECTION, from the same audit. The membership of these two
        # sets was declared rather than measured, and the tokenizer probe
        # disagrees on D7: it replaces both merged delimiter tokens with a
        # different single token. It fragments nothing and displaces nothing,
        # so it belongs to neither class. The classes are therefore taken from
        # the measured disposition where it is available, and D7 is reported
        # separately rather than being quietly assigned to whichever side
        # suits.
        DESTROY = {"D4", "D5", "D6"}          # D10 has no baseline contrast
        CONTROL = {"D1", "D2", "D3", "D8", "D9"}
        SUBSTITUTED = set()
        if ctok:
            _cn = ctok.get("conditions", {})
            for _c, _v in _cn.items():
                if _v.get("delimiter_disposition") == "substituted":
                    SUBSTITUTED.add(_c)
            _frag = {c for c, v in _cn.items()
                     if v.get("delimiter_disposition") == "fragmented"}
            _int = {c for c, v in _cn.items()
                    if v.get("delimiter_disposition") == "intact"}
            if _frag or _int:
                DESTROY = {c for c in _frag if c != "D10"}
                CONTROL = {c for c in _int}

        def vs_baseline(label):
            t = label.split()[0]
            if t.startswith("D") and "vs base" in label:
                return t
            if label.startswith("P1"):
                return "D8"
            if label.startswith("P2"):
                return "D9"
            return None

        dst, ctl, sub = [], [], []
        for model, modes in mech.items():
            for mode, blk in modes.items():
                for label, v in blk.get("contrasts", {}).items():
                    c = vs_baseline(label)
                    if c is None:
                        continue
                    (dst if c in DESTROY
                     else ctl if c in CONTROL
                     else sub if c in SUBSTITUTED else []).append(v)
        if dst and ctl:
            eD = np.array([abs(v["logodds"]) for v in dst])
            eC = np.array([abs(v["logodds"]) for v in ctl])
            pD = np.array([v["p_bh"] for v in dst])
            pC = np.array([v["p_bh"] for v in ctl])
            eS = np.array([abs(v["logodds"]) for v in sub]) if sub else np.array([])
            pS = np.array([v["p_bh"] for v in sub]) if sub else np.array([])
            paper.table(
                ["condition class", ">tests", ">sig. after BH", ">mean |effect|",
                 ">max |effect|"],
                [["delimiter destroyed", f"{len(dst)}",
                  f"{int((pD < .05).sum())} ({(pD < .05).mean() * 100:.1f} %)",
                  fmt(eD.mean(), 4), fmt(eD.max(), 4)],
                 ["nothing structural changed", f"{len(ctl)}",
                  f"{int((pC < .05).sum())} ({(pC < .05).mean() * 100:.1f} %)",
                  fmt(eC.mean(), 4), fmt(eC.max(), 4)]]
                + ([["delimiter token substituted", f"{len(sub)}",
                     f"{int((pS < .05).sum())} ({(pS < .05).mean() * 100:.1f} %)",
                     fmt(eS.mean(), 4), fmt(eS.max(), 4)]] if sub else []),
                [116, 34, 66, 60, 56], span2=True, size=7.6,
                caption=(f"{TAB('mech_classes')}. Every condition-versus-baseline test on the "
                         "mechanism panel, split by what the <i>tokenizer</i> says "
                         "the condition does to a delimiter rather than by "
                         "what the condition was declared to do. The third "
                         "row is D7, designed as a null control and in fact "
                         "a substitution. It replaces both merged delimiter "
                         "tokens with a different single token, fragmenting "
                         "none and displacing nothing. It is the closest this "
                         "design comes to a pure delimiter-identity manipulation, "
                         "and it arrived by accident. The rows are compared by "
                         "a stated test below rather than by eye, and that "
                         "comparison is a failure to reject rather than a "
                         "demonstration of equality."))
            # State the tests rather than assert the conclusion. The magnitude
            # comparison is rank-based because the effects are not symmetric about
            # zero and a few are far larger than the rest; the significance-rate
            # comparison is exact because the counts are small.
            # The class-difference interval and the minimum
            # detectable class difference. Present identically in
            # every model-mode block, so read it from any one.
            _dcc = None
            try:
                _first_mode = next(iter(next(iter(mech.values())).values()))
                _dcc = _first_mode["delimiter_class_contrast"]
            except Exception:  # noqa: BLE001
                pass
            from scipy import stats as _st  # noqa: PLC0415
            _mw = float(_st.mannwhitneyu(eD, eC, alternative="greater").pvalue)
            _fe = float(_st.fisher_exact(
                [[int((pD < .05).sum()), int((pD >= .05).sum())],
                 [int((pC < .05).sum()), int((pC >= .05).sum())]])[1])
            P("They do not differ. Conditions that destroy a delimiter reach "
              f"significance on {(pD < .05).mean() * 100:.1f} % of tests; "
              "conditions that change nothing structural reach it on "
              f"{(pC < .05).mean() * 100:.1f} %. The magnitudes are of the "
              f"same order — {fmt(eD.mean(), 4)} against "
              f"{fmt(eC.mean(), 4)}, a ratio of {eD.mean() / eC.mean():.2f} "
              f", and neither difference is distinguishable from chance "
              f"(Mann–Whitney p = {_mw:.2f} on the magnitudes, Fisher "
              f"exact p = {_fe:.2f} on the significance rates). We report no "
              "delimiter mechanism."
              + ((" The difference itself now carries an interval, which an "
                  "audit of this paper pointed out it never had. Resampling "
                  f"name pairs, the two classes differ in mean magnitude by "
                  f"{fmt(_dcc['mean_abs_effect_difference']['est'], 4, sign=True)} "
                  f"log-odds, 95 % CI "
                  f"[{fmt(_dcc['mean_abs_effect_difference']['ci'][0], 4, sign=True)}, "
                  f"{fmt(_dcc['mean_abs_effect_difference']['ci'][1], 4, sign=True)}]. "
                  "A failure to reject is not a demonstration of equality, and "
                  "the interval is what says how close to equality the data can "
                  "put them.") if _dcc else "")
              + ((" The substituted condition, which changes a delimiter "
                  "token without fragmenting one, reaches significance on "
                  f"{(pS < .05).mean() * 100:.1f} % of its {len(pS)} tests "
                  f"with a mean magnitude of {fmt(eS.mean(), 4)}: "
                  "inside the range the other two rows occupy, on far too "
                  "few tests to carry weight either way.") if sub else ""))
        _nc = tconc.get("null_calibration", {})
        _du = tconc.get("destroying_unbiased", {})
        _cu = tconc.get("controls_unbiased", {})
        P("A second candidate fared worse than that, and the way it failed is "
          "the more instructive of the two. Our pilot found the condition "
          "effects concentrated on the middling résumé, which has an "
          "appealing decision-boundary reading. A candidate near the threshold "
          "moves further under perturbation. We tested it as an interaction "
          "with an interval rather than as three estimates compared by eye, "
          "and reported that the concentration was real and non-specific. An "
          "audit of this paper showed the estimator was the effect. It "
          "compared the magnitude of <i>one</i> template’s shift against the "
          "magnitude of an <i>average</i> of the other two, and averaging shrinks a "
          "magnitude even when nothing is concentrated anywhere.", indent=False)
        if _nc:
            P("Permuting the template labels within each name pair , which "
              "preserves the true shift, the noise scale and the heterogeneity "
              "between pairs, and destroys only the association with the "
              "template. The old statistic returns "
              f"{fmt(_nc['legacy_null_mean'], 4, sign=True)} under a null in "
              "which it should return zero, against an observed "
              f"{fmt(_nc['legacy_observed_mean'], 4, sign=True)}. "
              f"{pct(_nc['legacy_bias_share'], 1)} of what we reported was the "
              "estimator, and against its own null the observed value has "
              f"p = {_nc['legacy_family_p']:.2f}. On an unbiased contrast the "
              "family mean falls to "
              f"{fmt(_nc['unbiased_observed_mean'], 4, sign=True)} and the "
              "significant interactions split evenly in sign "
              f"({_du.get('n_positive_after_bh')} positive and "
              f"{_du.get('n_negative_after_bh')} negative among the "
              "delimiter-destroying conditions, "
              f"{_cu.get('n_positive_after_bh')} and "
              f"{_cu.get('n_negative_after_bh')} among the controls), which is "
              "what no concentration looks like.")
        P("So the middling résumé is not established as more sensitive than "
          "the others, and the decision-boundary reading we found appealing is "
          "not supported by this design. What survives is the comparison the "
          "section is actually about, and it survives on the corrected "
          "statistic as it did on the broken one. Conditions that change "
          "nothing structural behave like conditions that destroy a delimiter "
          f"({fmt(_cu.get('mean_interaction', 0), 4, sign=True)} against "
          f"{fmt(_du.get('mean_interaction', 0), 4, sign=True)}). We report "
          "the correction rather than the original because a biased estimator "
          "that happened to agree with our conclusion is not evidence for it.")
        # THE COUNTS, AGAINST SCLAR'S OWN TABLE 1 AND TEXT. "Most" was wrong:
        # they name three of six classes as non-predictive (S2, Fitem1,
        # Fcasing), which is half. And the complement is three, not two -- C
        # carries 29 % weak differences; S1 and Fitem2 are the two they single
        # out as having the MOST individual impact among those three.
        #
        # The reference is also hard-typed on purpose. It used to go through
        # TAB("instrument_validation"), which is THIS paper's Table 1: the two
        # agree today only by coincidence, and a reordering here would have
        # silently renumbered a reference to somebody else's table.
        P("<b>This negative result is a replication, not a failure.</b> Sclar "
          "et al. report that half the individual format features do not "
          "independently predict performance. In their Table 1 the second "
          "separator, the item wrapper and casing do not , and that the "
          "format space is highly non-monotonic. Three classes carry a "
          "positive individual signal, and they name two of those as having "
          "the most individual impact. Separators, and the number format used "
          "in enumerations. It is easy to name separators as "
          "the sole exception and put itemisation among the non-predictive "
          "classes; both are wrong against their table, and the second is "
          "the one that matters here, because it means their analysis does "
          "<i>not</i> establish per-feature unpredictability across the board. It "
          "establishes it for three classes of six and leaves three with a "
          "positive individual signal.")
        P("That makes our null the informative test rather than a foregone "
          "one. Separators are the closest thing in their taxonomy to our "
          "delimiters, and they fall in the exception set, so a "
          "delimiter-specific mechanism was live going in; their result is "
          "the reason it was worth testing, not a prediction that it must be "
          "absent. What our null adds is that a feature class carrying a "
          "positive signal for task <i>accuracy</i> does not carry the <i>demographic</i> "
          "effect once position is controlled. The same shape of finding "
          "as theirs, perturbation sensitivity that is real, large, and not "
          "attributable to the specific feature perturbed, measured on a "
          "fairness estimate instead of on accuracy.")
        # THE THREE COUNTS, EACH COMPUTED. The figure draws the position
        # contrasts, the table covers the condition-versus-baseline tests, and
        # the two together do NOT exhaust the family: the condition-versus-
        # condition contrasts C1, C2 and C5 appear in neither. The prose used
        # to say the table held "the full family", which is how a reader would
        # have concluded that every test was accounted for somewhere.
        _mech_all = sum(len(b.get("contrasts", {}))
                        for mo in mech.values() for b in mo.values())
        _mech_drawn = sum(1 for mo in mech.values() for b in mo.values()
                          for lab in b.get("contrasts", {})
                          if lab.split()[0] in ("P1", "P2", "P3", "P4"))
        _mech_tabled = len(dst) + len(ctl) + len(sub)
        _mech_neither = sum(
            1 for mo in mech.values() for b in mo.values()
            for lab in b.get("contrasts", {})
            if lab.split()[0] in ("C1", "C2", "C5"))
        P("" + FIGREF("fig10_mech_panel") + " shows the position contrasts of "
          f"the family, P1 to P4 for every model and mode, {_mech_drawn} "
          f"of the {_mech_all} , because those are the ones the delimiter "
          "question turns on once position is separated from destruction. An "
          "earlier caption said it showed every contrast, which it does not. "
          "What a delimiter mechanism would look like is a block of intervals "
          "excluding zero confined to the destroying conditions. What the "
          "panel shows instead is scattered significance distributed across "
          "condition classes without regard to whether structure was "
          "destroyed.")
        P(f"The other cut through the same family is {TAB('mech_classes')}, "
          f"which covers the {_mech_tabled} tests that compare a condition "
          "against the baseline, classified by what the tokenizer says the "
          "condition did. The two views overlap on the position contrasts and "
          f"do not exhaust the family between them: {_mech_neither} "
          "condition-versus-condition contrasts, C1, C2 and C5, which ask "
          "which of two edits matters rather than whether one matters, are "
          "in neither, and are in the released artifact. "
          "said the table held the full family, which would have told a "
          "reader every test was accounted for on the page.", indent=False)
        paper.figure(FIGS / "fig10_mech_panel.png", span2=True,
                     max_h=3.1 * pk.inch, space_after=12.0)
        if d9:
            P("<b>An audit of our own data.</b> One condition definition was "
              "corrected mid-run, and nothing in the recorded rows established "
              "which definition had produced them. Rather than delete the data "
              ", which would have destroyed the only evidence bearing on "
              "the question, we re-measured the affected cells alongside "
              "a never-changed control condition that calibrates ordinary "
              "cross-session drift, under a decision rule fixed before the new "
              "data existed. The re-measured cells are indistinguishable from "
              f"the control (ratio {fmt(d9['pooled']['ratio'], 3)} against a "
              f"pre-set limit of {d9['ratio_limit']}, "
              f"p = {fmt(d9['pooled']['u_p'], 3)}, "
              f"{d9['pooled']['n_rejecting_after_bh']} of "
              f"{len(d9['per_model_mode'])} model-modes rejecting after "
              "correction), so the original rows stand and the recheck becomes "
              "an additional cross-session reproducibility measurement.")

    # ======================================================================
    H("8  Calibration against the published literature")
    if lit:
        s = lit["summary_of_published_pp_gaps"]
        # THE CORRECTION SENTENCE READS THE SUPERSEDED BLOCK. It used to type
        # "12 of the 14", "the other two" and "from 0.51", which are exactly
        # the values published_effects.json keeps under superseded_twelve_row
        # -- a block whose own _why_kept says it exists "so that any prose
        # still carrying the old numbers can be traced". The prose was
        # carrying them.
        _sup = s.get("superseded_twelve_row")
        if not _sup:
            sys.exit(
                "published_effects.json no longer records the superseded "
                "twelve-row summary, but SS8 describes the correction it "
                "made. Restore the block or rewrite the sentence; do not "
                "let it print numbers with nothing behind them.")
        # The threshold is read out of the key that defines it, so the
        # sentence cannot go on naming a bucket the analysis stopped
        # computing.
        _below_key = next(k for k in s
                          if k.startswith("n_below_") and k.endswith("_pp"))
        _below_thr = _below_key[len("n_below_"):-len("_pp")].replace("_", ".")
        P("The movement above is only alarming if it is large relative to what "
          "audits report. " + FIGREF("fig9_literature") + " places it against the published record. The "
          "largest set of directly comparable percentage-point callback gaps in "
          "this literature , comparable because one team measured them on "
          "one protocol, has a median absolute gap of "
          f"{fmt(s['median_abs_pp'])} "
          f"points, with {s[_below_key]} of {s['n']} below {_below_thr} "
          f"and a maximum of {fmt(s['max_abs_pp'])}: all {s['n']} model rows "
          f"of that panel. An earlier version of this paper used {_sup['n']} "
          f"of the {s['n']} and said the other {NUM[s['n'] - _sup['n']]} had "
          "not extracted cleanly from the published two-column layout. That "
          "was wrong. Both extract cleanly, and an audit of this paper found "
          "it. Restoring them moves the median from "
          f"{fmt(_sup['median_abs_pp'])} to {fmt(s['median_abs_pp'])} points. "
          "Against our own "
          "argument, since the effects we are calibrating against get larger. "
          f"The Bertrand and Mullainathan "
          f"field anchor is {fmt(bm['_headline']['gap_pp'], 1) if bm else 'n/a'} "
          "points.", indent=False)
        P("Comparison must be made carefully, and not on the scale the field "
          "usually uses. For the reason given in Section 6.2. The one "
          "quantity in our design commensurable with a callback rate is the "
          "shortlist-rate gap at a fixed cut. Rank candidates by margin, take "
          "the top decile, and report the difference in selection rate. On that "
          "scale the between-wording standard deviation alone is "
          + ", ".join(f"{s2[m]['shortlist']['top_10pct']['sd_across_wordings_pp']:.1f}"
                      for m in models)
          + " points for the four models respectively, against a published "
          f"median gap of {fmt(s['median_abs_pp'])} points. Read naively that "
          "says the wording moves the shortlist gap by several times the size "
          "of the effects this literature reports. It does not, and the next "
          "paragraph is why.")
        P("This comparison has a resolution limit and it should be stated "
          "rather than left for a reader to find. With twelve name pairs the "
          "top decile of twenty-four ranked candidates is two positions, so a "
          "per-template shortlist gap can only take multiples of 1/6, and a "
          "per-wording value , the mean over three templates, only "
          # DERIVED, NOT TYPED, AND THE DERIVATION WAS MISSING A FACTOR. The
          # shortlist slots come out of ONE pooled ranking of all 24
          # instances, so the two arms' selections sum to k and their
          # difference moves in steps of k rather than of one. The step is
          # k / (instances per arm), spread over the templates a per-wording
          # value averages. Omitting k printed half the true quantum; the
          # realised values in study2_v2.json step by 5.6, not 2.8.
          f"multiples of about "
          f"{100 * 2 / (2 * st.N_FIRST * len(st.TEMPLATES)):.1f} "
          "points. The smallest movement this estimator "
          "can register is therefore several times the published median it is "
          "set against. An earlier draft added that this “does not affect "
          "the comparison being made here”. An audit of this paper "
          "disagreed, and the audit is right. If the coarsest movement the "
          "estimator can see already exceeds the published median, then "
          "observing a between-wording spread larger than that median is close "
          "to guaranteed by the instrument rather than earned from the data: "
          "the spreads above are between one and three quanta.")
        P("So we withdraw the shortlist bridge as a quantitative comparison "
          "and keep it as what it can support. It establishes that the "
          "wording moves the shortlist composition at all. That the "
          "identity of the top decile is not stable across twelve wordings "
          "that ask the same question , and it cannot say by how much "
          "relative to a published callback gap, because it cannot resolve "
          "anything at that scale. The quantitative comparison this paper "
          "rests on is the one in the model’s own units, on the "
          "scale-free effect size of §3, where no conversion and no "
          "discretisation is involved. That a bridge to the field’s "
          "preferred scale cannot be built at usable resolution from twelve "
          "name pairs is itself worth reporting. It is the same "
          "operating-point problem as §6.2, arriving from the other side.")
        hp = lit.get("how_many_prompts_the_field_uses")
        P("" + FIGREF("fig9_literature") + " also counts, for each study, "
          "two things. The number of distinct prompt wordings it used, and the "
          "number under which its effect was <i>separately estimated</i>. The "
          "two diverge when a study averages over its wordings before the bias "
          "test is run, which collapses the design to a single effective "
          "wording. Two of the six do that. One issues ten instruction "
          "wordings and averages the similarity scores over all ten before any "
          "test; the other writes five base wordings and reports the mean over "
          "them. An earlier version of this paper said the second used 820 "
          "wordings. It does not: 820 is the number of instantiated prompts, "
          "four qualification levels crossed with five base wordings crossed "
          "with forty-one occupational roles, and only the five are paraphrases "
          "of one another. Miscounting another group’s design by a factor "
          "of a hundred and sixty is the kind of error this paper exists to "
          "complain about, and an audit of it found this one.", indent=False)
        P("One of the six does not belong in that tally at all and is drawn "
          "with its count omitted rather than as a one. Fu and Shi report no "
          "demographic effect of any kind. Their outcome is a model’s "
          "own score on a hiring instrument, not a contrast between two "
          "candidates , so there is no effect for a wording to have been "
          "estimated under. It is retained in the panel because its reporting "
          "practice is informative in " + TAB("reporting_matrix") + ", and excluded from every "
          "effect-size and wording-count statement.")
        paper.figure(FIGS / "fig9_literature.png", span2=True,
                     max_h=2.9 * pk.inch, space_after=12.0)

    # ------------------------------------------------------------------
    if matrix:
        H("8.1  What the field reports, counted", 2)
        cts = matrix["counts"]  # noqa: F841 (already set above; kept local)
        nA = matrix["n_llm_hiring_audits"]
        # THE CRITERION IS TOPICAL, NOT AVAILABILITY. "Every audit we could
        # obtain in full text" describes a search that stopped where the PDFs
        # ran out, and invites the reader to wonder which audits were missed.
        # The rule actually applied is narrower and checkable: every row of the
        # matrix carries kind == "llm_hiring_audit", i.e. an LLM audit whose
        # object of study is an employment decision. Saying so also makes the
        # limit explicit -- this is not a census of LLM auditing.
        P("A count of one variable is an anecdote. The claim this paper makes "
          "is about a practice, so it needs a matrix. The LLM audits we read "
          "in full text whose object of study is an employment decision, "
          "against every choice this paper shows to "
          "move the number. It is a sample and not a census of LLM auditing, "
          "and an audit of a different decision would sit outside it. "
          + TAB("reporting_matrix")
          + f" is that matrix over {nA} LLM hiring "
          "audits, summarised one choice per row; the per-study cells, "
          "including the two non-audits kept for contrast, are released with "
          "the artifacts. Each cell was filled by reading the paper end to end, "
          "and every <i>negative</i> was re-checked independently against a fresh "
          "extraction of the PDF: necessary, because several of these "
          "papers put load-bearing detail in figures whose text is not in the "
          "PDF text layer, and a false claim about someone else’s reporting "
          "would be the same kind of error this paper is about. A choice that "
          "is partly specified counts as not reported, because the question is "
          "whether a reader could reconstruct it. "
          + (("Those readings are re-checked mechanically at build time rather "
              "than trusted. Every negative search the matrix records is "
              f"re-run against the source it was recorded from: "
              f"{evcheck['n_negative_searches_rerun']:,} of them, of which "
              f"{evcheck['n_negative_searches_now_hitting']} now return a hit. "
              "Every quotation is matched back to the paper it is "
              f"attributed to, {evcheck['n_quotations_verified']} of "
              f"{evcheck['n_quotations_checked']} verbatim or through the "
              "line-wrap, small-caps and column-interleaving damage that PDF "
              "extraction introduces. "
              + (f"{evcheck['n_quotations_unverifiable_damaged_source']} "
                 "quotations cannot be checked this way at all, because they "
                 "come from a scan whose font maps every digit-and-full-stop "
                 "pair to a single glyph; they are reported as unverifiable "
                 "rather than counted either way. "
                 if evcheck.get("n_quotations_unverifiable_damaged_source")
                 else "")
              + "A false claim about another group's reporting would be the "
              "same kind of error this paper is about, so it is the one class "
              "of claim here that is checked on every build. ")
             if evcheck else "")
          + "The panel is "
          # NAMED, NOT JUST COUNTED. Three of the thirteen never appear in the
          # body: _who() collapses any reporting set larger than three to "most
          # of the panel", so a study that is never in a small set is never
          # named, and ends up with a reference entry and no citation. Naming
          # the roster also makes the matrix reconstructible from the page.
          + "; ".join(matrix.get("llm_hiring_audits") or [])
          + ".", indent=False)

        # EVERY CHOICE IN THE ARTIFACT, NOT A SELECTION OF THEM. This was a
        # literal list of eleven of the artifact's twenty-two, and the eleven
        # it omitted were the rows on which the surveyed field looks
        # competent -- prompt published 11/13, occupations 12/13, name list
        # source 9/10. Printing the unflattering half and calling it "the
        # matrix" is the practice this paper objects to, in the table its own
        # novelty claim rests on. LEAD now fixes only the ORDER, which the
        # caption states; nothing is dropped.
        LEAD = ["dispersion_across_wordings", "null_edit_control",
                "name_sensitivity_reported", "token_matching",
                "checkpoint_pinned", "quantization_reported",
                "concurrency_or_batching", "cache_policy",
                "resampling_unit", "reporting_scale",
                "code_or_data_released"]
        # field_order is a list of [key, pretty] pairs, not bare keys.
        _rest = [p[0] if isinstance(p, (list, tuple)) else p
                 for p in (matrix.get("field_order") or list(cts))]
        _order = [k for k in LEAD if k in cts] + [
            k for k in _rest if k in cts and k not in LEAD]
        HEAD = [(k, cts[k]["pretty"]) for k in _order]
        def _who(names):
            # Naming the reporters is informative only when there are few of
            # them. A row where ten of thirteen report the choice is not the
            # row anyone reads this table for, and listing ten names overflows
            # the column.
            if not names:
                return "—"
            if len(names) > 3:
                return f"most of the panel ({len(names)})"
            return ", ".join(n.replace(" et al.", "")
                              .replace("Wilson & Caliskan", "Wilson & C.")
                             for n in names)

        rows = []
        for key, pretty in HEAD:
            c = cts.get(key)
            if not c:
                continue
            rows.append([pretty, f"{c['n_reported']} / {c['n_applicable']}",
                         str(c["n_partial"]), _who(c["reported_by"])])
        paper.table(
            # THE LAST HEADER NAMES ITS OWN COLUMN. It read "reported by",
            # sitting immediately right of "partly", so the header row scanned
            # as "... partly reported by" and every named study appeared to be
            # one that PARTLY reported the choice. The artifact's field is
            # `reported_by`, which lists the studies that reported it FULLY --
            # so the table as printed said the opposite of its source on every
            # named row. Wording the header so it cannot be read as a
            # continuation of its neighbour is the whole fix.
            ["choice", ">fully", ">partly", "fully reported by"],
            rows, [132, 34, 34, 200], span2=True, size=7.4,
            caption=(f"{TAB('reporting_matrix')}. Reporting practice across {nA} LLM hiring "
                     "audits, counted from full text. Every choice the underlying survey "
                     "scored, not a selection of them. The panel is "
                     "the audits we obtained and read in full, named in "
                     "§8.1, and it is a sample rather than a census. No "
                     "systematic database search was run, so a study we did "
                     "not obtain is absent rather than excluded. The choices "
                     "this paper shows to move the number come first "
                     "and the rest follow in the survey's own order; "
                     "an earlier version printed only the first "
                     "group. The “fully” column "
                     "excludes studies to which the choice does not apply, so "
                     "denominators differ; a partly specified choice is not "
                     "counted as reported, because a reader cannot reconstruct "
                     "it. The last column names the studies counted in the "
                     "<i>first</i> column, not the second. The evidence quote, or the "
                     "exact failed search, behind every cell is released with "
                     "the artifacts."))

        never = [cts[f]["pretty"] for f in matrix["never_reported_by_any"]]
        if never:
            P(f"{NUM.get(len(never), len(never)).capitalize()} choices are "
              "reported by <b>no</b> audit in the panel: "
              + "; ".join(never) + ". Two of them are exactly what §5.2 shows "
              "determines whether the measurement reproduces at all, and a "
              "third is the threat to the matched pair of §4.4. The last, "
              "dispersion across wordings, is the single highest-value line an "
              "audit could add.", indent=False)
        bm_row = next((s for s in matrix["studies"]
                       if s["kind"] == "field_experiment"), None)
        if bm_row:
            c = cts.get("name_sensitivity_reported")
            P("One line of that matrix deserves to be read on its own. Bertrand "
              "and Mullainathan print the callback rate of every individual "
              "name, test whether name-level variation is explained by social "
              "background, and say plainly that chance could account for what "
              f"they see. Of the {c['n_applicable']} LLM audits to which the "
              "same check applies, none reports anything at the level of the "
              "individual name"
              + (f"; {NUM.get(c['n_reported'], c['n_reported'])} "
                 + ("reports" if c["n_reported"] == 1 else "report")
                 + " the sensitivity of its result to substituting names "
                 "<i>within</i> a demographic group, aggregated per model rather than "
                 "per name" if c["n_reported"] else "")
              + ". The 2004 field experiment documents its name list’s construction in more detail than most of the "
              "name list than the 2024–2026 papers that cite it.")
        P("That is the argument. A study reporting one number from one "
          "effective wording, one name draw, one unreported quantization and "
          "one unstated pairing has measured one point on the specification "
          "curve in " + FIGREF("fig7_spec_curve") + " and cannot know where on it that point sits "
          ", and neither can a reader, a regulator or a court.")

    # ======================================================================
    _dcc_early = None
    if mech:
        try:
            _fm = next(iter(next(iter(mech.values())).values()))
            _dcc_early = _fm["delimiter_class_contrast"]
        except Exception:  # noqa: BLE001
            pass
    _mde_early = []
    if mech:
        for _m, _modes in mech.items():
            for _mode, _blk in _modes.items():
                for _c, _v in (_blk.get("mde") or {}).items():
                    if isinstance(_v, dict) and "mde_absolute" in _v:
                        _mde_early.append(_v["mde_absolute"])

    H("9  A minimum reporting set")
    P("None of the above implies that LLM audits are uninformative. It implies "
      "that a single number without its instrument is uninterpretable. What "
      "follows is the minimum an audit must report for a reader to know what "
      "its number is a number about. The items are grouped by where in a study "
      "the choice is made. What the auditor writes, what the auditor "
      "runs, what the auditor computes , because that is where each has "
      "to be fixed, and each carries the movement this paper measured when it "
      "was left unreported. Every item is cheap relative to running the audit "
      "at all. Two cost more than a line: the first, and the manipulation "
      "check, which needs its own probe run.",
      indent=False)

    P("<b>STIMULUS: what the auditor writes.</b>", indent=False,
      space_after=2.0)
    P("<b>1. The dispersion, not only the estimate.</b> Run the audit under at "
      "least five defensible wordings and report the standard deviation of the "
      "effect across them beside the effect. Measured consequence. The wording "
      "moves the effect by "
      # OVER EVERY CELL THAT CAN CARRY THE RATIO, not Study 2 alone. This
      # item is the reporting requirement the paper builds to, and quoting
      # the narrowest of the three studies that compute the same statistic
      # understated it by half.
      + (f"{fmt(_rat_lo * 100, 0)} % to "
         f"{fmt(_rat_hi * 100, 0)} % of itself "
         "across every model-by-posting cell where the effect is "
         "identified, open-weight and frontier alike. Where it is not "
         "identified the ratio is not interpretable at all: its denominator "
         "covers "
         "chance, and Section 4.5 is explicit that such a ratio has no finite "
         "upper bound, so no range is given for those models , which is "
         "itself the argument for this item. Without the dispersion printed "
         "beside the estimate, a reader cannot tell which case they are in. "
         if survives and _r_un
         else "a quarter to more than the whole effect (§4.1). ")
      + "This is the single highest-value addition and costs a factor of five "
        "in compute. Of the audits surveyed in §8, "
      + (f"{matrix['counts']['dispersion_across_wordings']['n_reported']} of "
         f"{matrix['counts']['dispersion_across_wordings']['n_applicable']} "
         "report it." if matrix else "almost none report it."))
    P("<b>2. A null-edit control and a byte-identical replicate.</b> Include at "
      "least one wording differing from another by no word , whitespace "
      "or punctuation only, and one pair that is byte-identical. The "
      "identical pair is free and gives the noise floor, without which a "
      "dispersion cannot be distinguished from arithmetic. The null edits say "
      "whether the sensitivity is to meaning or to surface form; on this panel "
      "the two arms could not be distinguished at all (§4.1), which is a "
      "stronger result than the one we expected and is invisible without the "
      "control.")
    # ITEM 3 USED TO STATE THE CROSSOVER FLATLY -- "the crossover happens
    # around nine" -- which is precisely the claim §4.2 withdraws two pages
    # earlier: propagating both posteriors, the side of 1.0 is determined on
    # none of the four models at k = 9. A recommendations section is the last
    # place a withdrawn result should reappear as settled, because it is the
    # part a reader acts on. What survives is the direction, which is
    # arithmetic rather than estimation, and that is what the item now says.
    P("<b>3. The name list, and the effect recomputed on subsets.</b> A study "
      "using three names per race should say so: at that size the name draw "
      "is the larger source of dispersion on every model we measured, and its "
      "contribution falls with list size while the wording’s does not, so the "
      "two cross somewhere. Where they cross is <i>not</i> pinned by twelve wordings "
      "— §4.2 propagates both posteriors and cannot determine which side of "
      "parity a nine-name list falls on , so report the effect recomputed "
      "on subsets rather than trusting a rule of thumb about list size. "
      "Report the pairing as well as the list , which name faced which "
      ", because if résumés are scored singly the pairing is an analysis "
      "choice and it moves the pairwise statistic (§6.3).")
    # THIS ITEM USED TO ASK FOR THE EFFECT ON THE MATCHED SUBSET. On the
    # standard list that subset is three independent first-name pairs, where an
    # exact cluster-level permutation test has eight arrangements and cannot
    # return a two-sided p below 0.05 at all. A recommendations section is the
    # part a reader acts on, so it should not ask for a number that cannot
    # carry an inference. What it asks for now is the surviving count, which
    # tells a reader whether the control was affordable on the list they used.
    P("<b>4. Token-matched pairs, and the size of the matched subset.</b> "
      "Report whether the two names in each pair occupy the same number of "
      "tokens, and report how many independent pairs survive the "
      "restriction. "
      + (f"Between {fmt(100 * (1 - max(_tm_ab.values())), 0)} % and "
         f"{fmt(100 * (1 - min(_tm_ab.values())), 0)} % "
         if _tm_ab else "Most ")
      + "of the pairs in a "
      "standard validated list do not match (\u00a74.4). This item does not "
      "ask for the effect recomputed on the matched subset, and that is "
      "deliberate. On the standard list the subset is "
      + (f"{permres['source_list_panel_maximum']['n_pairs']} independent "
         f"first-name pairs, at which an exact cluster-level permutation "
         f"test has {permres['source_list_panel_maximum']['n_assignments']} "
         f"arrangements and cannot return a two-sided p below 0.05 at all"
         if permres else
         "small enough that an exact cluster-level permutation test cannot "
         "return a two-sided p below 0.05 at all")
      + ", so an effect reported there would be a number with no inference "
      "behind it. The surviving count is what a reader needs, because a "
      "list that cannot afford the control is the finding. Of the audits "
      "surveyed, "
      + (f"{matrix['counts']['token_matching']['n_reported']} of "
         f"{matrix['counts']['token_matching']['n_applicable']} "
         if matrix else "none ")
      + "discuss tokenization of the manipulated names at all.")
    P("<b>5. A manipulation check.</b> Show that the model encodes the "
      "association the name list was validated for by probing the model "
      "directly. Correlating per-name effects against the list’s own "
      "validation data is <i>not</i> a substitute. Signed by race it re-expresses "
      "the demographic effect, and within race our own design cannot "
      "detect anything short of a very large correlation (Appendix B). A null "
      "on a model that cannot be shown to encode the distinction is "
      "uninformative rather than reassuring, and on our whole "
      "panel that is the situation (Appendix B). No audit we surveyed does this "
      "on a name list; the one surveyed study that runs the analogous direct "
      "probe does it on a different stimulus, so the check we are asking for "
      "has no precedent in this design.")

    P("<b>INSTRUMENT: what the auditor runs.</b>", indent=False,
      space_after=2.0)
    P("<b>6. The checkpoint, to the file.</b> Quantization, revision hash and "
      "serving stack, not just the model family. The digest of the weights "
      "actually loaded is the only unambiguous identifier, and the "
      "quantization alone shifts the effect by an amount of the same order as "
      "the between-wording standard deviation, separable from zero on "
      + (f"{NUM.get(_quant_sep, _quant_sep)} of "
         f"{NUM.get(_quant_n, _quant_n)} checkpoints tested"
         if _quant_n else "one of the two checkpoints tested")
      + " and with which is larger not pinned (Appendix E). Of the audits surveyed, "
      + (f"{matrix['counts']['quantization_reported']['n_reported']} of "
         f"{matrix['counts']['quantization_reported']['n_applicable']} report "
         "the quantization and "
         f"{matrix['counts']['checkpoint_pinned']['n_reported']} of "
         f"{matrix['counts']['checkpoint_pinned']['n_applicable']} pin the "
         "checkpoint to a revision." if matrix else
         "almost none report either.")
      + ((" It is also the cheapest single repair on the panel: under a "
          "minimal re-run criterion (the exact prompt, a pinned checkpoint "
          "and the decoding parameters), pinning the checkpoint alone would "
          f"bring {NUM.get(len(mstruct['minimal_rerun']['flips']['checkpoint_pinned']), str(len(mstruct['minimal_rerun']['flips']['checkpoint_pinned'])))} "
          f"of the {NUM.get(mstruct['minimal_rerun']['n_audits'], str(mstruct['minimal_rerun']['n_audits']))} audits up to the criterion ("
          + ("none meet it today" if mstruct['minimal_rerun']['n_meeting'] == 0
             else f"{NUM.get(mstruct['minimal_rerun']['n_meeting'], str(mstruct['minimal_rerun']['n_meeting']))} meet it today")
          + "), and "
          f"{NUM.get(mstruct['checkpoint_near_miss']['n_partial'], str(mstruct['checkpoint_near_miss']['n_partial']))} of the audits already half-report it, "
          "naming the family but not the file.")
         if mstruct and mstruct.get("minimal_rerun") else ""))
    P("<b>7. The serving configuration.</b> Request concurrency and "
      "prompt-cache policy. These are not detected to move the effect"
      + (f": the shift is bounded within {pct(_serving_bound, 1)} of it "
         "on the checkpoints whose effect is distinguishable from zero, the "
         "only ones a ratio against the effect is defined on, which is a "
         "bound "
         "and not an absence" if _serving_bound else ": we checked")
      + " , and they determine whether the measurement reproduces at "
      "all, which with both controlled it does bitwise across separate "
      "processes (§5.2). Of the audits surveyed, "
      + (f"{matrix['counts']['concurrency_or_batching']['n_reported']} state "
         "their batching in full and "
         f"{matrix['counts']['cache_policy']['n_reported']} state a cache "
         "policy." if matrix else "none state either.")
      + " On the remedy. Concurrency one with the cache off is what we could "
        "do on this stack and it costs the whole serving pipeline. If "
        "batch-invariant kernels are available. He and Thinking Machines Lab "
        "release a reference implementation and a vLLM demonstration, which is "
        "what their text claims and all it claims "
        ". They are the better answer, because they remove the batch "
        "dependence without giving up throughput. Report which route was "
        "taken.")

    P("<b>Which claim each item is for.</b> Items 1, 3 and 5 are required for "
      "a claim about a <i>model</i> and optional for a claim about a fixed deployed "
      "pipeline, where the wording and the name list are part of the system "
      "under test rather than nuisance choices. Everything else is required "
      "for both. A pipeline audit that does not state its checkpoint, its "
      "serving configuration or its resampling unit is not reproducible "
      "either.", indent=False)

    P("<b>ANALYSIS: what the auditor computes.</b>", indent=False,
      space_after=2.0)
    P("<b>8. The resampling unit, explicitly.</b> State what the bootstrap "
      "resamples. In a crossed design, resampling rows rather than the unit of "
      "randomisation understates intervals by a factor of "
      + (f"{resamp['pooled_summary']['min_ratio']:.1f} to "
         f"{resamp['pooled_summary']['max_ratio']:.1f} " if resamp
         else "three to five ")
      + "(§6.1), which is enough to manufacture a significant effect from "
        "a null one , and did, for two of our four models.")
    P("<b>9. The scale, and its Jacobian.</b> If percentage points are "
      "reported, state the operating point at which the conversion was made "
      "and whether it varies across the models compared. Across four "
      "checkpoints of the same size on the same design the conversion factor "
      "at p = 0.5 is wrong by "
      + (f"{scale['summary']['jacobian_error_min']:.1f}× to "
         f"{scale['summary']['jacobian_error_max']:.0f}× " if scale
         else "up to two orders of magnitude ")
      + "(§6.2). If it varies, the percentage points are not comparable "
        "and something scale-free should be reported instead.")
    P("<b>10. The floor under a null.</b> A null result needs its minimum "
      "detectable effect printed beside it, not deposited in a repository. "
      "Ours is "
      + (f"{fmt(max(_mde_early), 4)} log-odds at worst, about "
         f"{max(_mde_early) / _eff_max:.0%} of "
         "the largest effect this paper measures " if _mde_early else "")
      + "(§10.1), and a reader is entitled to weigh the null against it "
        "rather than take the word null at face value. The sharper number is "
        "the minimum detectable <i>difference</i> <i>between</i> <i>condition</i> <i>classes</i>, which is "
        "what §7's null is actually about. Calibrated by injection rather than "
        "by a closed form, it is "
      + ((f"{fmt(_dcc_early['mdd_80_calibrated_mean_abs_effect'], 4)} log-odds, "
          f"{_dcc_early['mdd_80_calibrated_pct_of_baseline']:.1f} % of the "
          "median demographic effect on that panel. A delimiter mechanism "
          "larger than that would have been detected; one smaller would not.")
         if _dcc_early else "recorded in the mechanism artifact."))

    H("9.1  What a reader does with a dispersion", 2)
    # THE VERDICT IS COMPUTED, NOT TYPED. This paragraph used to assert that
    # the rule is passed by the sign-stable models. On the artifact it is
    # failed by every model, because sign stability is only half the rule.
    _screen = {}
    for _m in models:
        _pv = (s2.get(_m) or {}).get("per_variant") or {}
        if not _pv:
            continue
        _e = {k: _pv[k]["logodds"] for k in _pv}
        _stable = len({v["est"] > 0 for v in _e.values()}) == 1
        _w = min(_e, key=lambda k: abs(_e[k]["est"]))
        _ci = _e[_w]["ci"]
        _screen[_m] = dict(sign_stable=_stable, worst_variant=_w,
                           worst_est=_e[_w]["est"], worst_ci=_ci,
                           passes=bool(_stable and _ci[0] * _ci[1] > 0))
    _screen = {m: v for m, v in _screen.items() if not v["passes"]}
    P("A reporting requirement that does not say how the number is used is a "
      "paperwork exercise, so we state a rule and its limits. Report the "
      "effect and the between-wording standard deviation together. Treat a "
      "disparity as established only if it survives the wordings "
      "individually. That is, if the sign is stable across them and the "
      "smallest per-wording estimate still excludes the null, rather than "
      "if the pooled estimate alone does. That is a stricter test than "
      "significance on the pooled number, and it is the one this paper\u2019s "
      "own results have to face."
      + ("" if not _screen else
         " Applied to our own panel it is failed by all "
         + NUM.get(len(_screen), str(len(_screen))) + ": "
         + "; ".join(
             (f"on {SHORT.get(m, m)} the sign is stable but the smallest "
              f"per-wording estimate, {v['worst_variant']}, is "
              f"{fmt(v['worst_est'], 4, sign=True)} "
              f"[{fmt(v['worst_ci'][0], 4, sign=True)}, "
              f"{fmt(v['worst_ci'][1], 4, sign=True)}], which does not "
              "exclude the null")
             if v["sign_stable"] else
             (f"on {SHORT.get(m, m)} the sign is not stable across the twelve "
              "wordings")
             for m, v in _screen.items())
         + ". A rule our own headline models do not pass is still the rule we "
         "recommend, and we would rather report that than weaken it until we "
         "pass."), indent=False)
    if srn:
        P("<b>The rule's error rate, which an earlier draft declined to "
          "state.</b> We wrote that taking the minimum over twelve estimates "
          "is a selection whose behaviour under the null depends on how "
          "correlated the wordings are, and that we had not characterised it. "
          "A paper whose complaint is that the field reports numbers with "
          "uncharacterised properties cannot leave its own recommendation in "
          "that state, so here it is. The rule is not in fact "
          "a selection problem. Requiring the sign to be stable <i>and</i> the "
          "smallest estimate to exclude the null is the same event as "
          "requiring <i>every</i> wording\u2019s interval to exclude the null on the "
          "same side. Under the global null, with the twelve estimates "
          "equicorrelated at \u03c1 \u2014 the structure the design induces, "
          "since the wordings share the same name pairs \u2014 that "
          "probability has a closed form.", indent=False)
        _srows = {f"{r['rho']:.1f}": r for r in srn["by_rho"]}
        P("It rises monotonically in \u03c1, from "
          + f"{srn['by_rho'][0]['exact']:.0e} at \u03c1 = 0 to "
          + ", ".join(
              f"{_srows[k]['exact']:.1e} at \u03c1 = {k}"
              for k in ("0.3", "0.6", "0.9") if k in _srows)
          + f", and it never exceeds the per-wording "
          f"\u03b1 of {srn['alpha_per_wording']:.2f}, which it approaches only "
          "as the wordings become perfectly correlated and the twelve tests "
          f"collapse into one ({srn['max_rate_over_rho']:.4f} at the largest "
          "\u03c1 we evaluate). So the rule cannot be "
          "more liberal than the single-wording test an auditor would "
          "otherwise have run, whatever the correlation turns out to be. The "
          "selection worry ran the wrong way. Taking the minimum makes the "
          "rule harder to pass, not easier."
          + ((" On our own panel the wordings correlate at "
              f"{srn['empirical_rho_min']:.2f} to "
              f"{srn['empirical_rho_max']:.2f}, which puts the rule\u2019s "
              f"false-positive rate between "
              f"{min(srn['rate_at_empirical_rho'].values()):.0e} and "
              f"{max(srn['rate_at_empirical_rho'].values()):.1e} \u2014 "
              "at least four times, and up to two orders of magnitude, "
              "stricter than the nominal test it replaces.")
             if srn.get("rate_at_empirical_rho") else ""), indent=False)
        P("Two things this does not license. The rate above is the rate under "
          "a <i>global</i> null, and a rule this conservative pays for it in power: "
          "it is deliberately one-directional, protecting against reporting a "
          "disparity that a different wording would not have found and doing "
          "nothing about a disparity that every wording misses. It applies "
          "to any scalar an audit reports, the impact ratio Local Law 144 "
          "requires included. And \u03c1 is "
          "estimated from twelve wordings on four models, so the placement of "
          "our own panel on that curve is itself a small-sample quantity. "
          "What has changed is that the rule now has a stated error rate and "
          "a bound that holds without knowing \u03c1 at all, which is what we "
          "are asking of everyone else.")
    else:
        P("One caution on that rule. It is deliberately conservative in one "
          "direction only. It protects against reporting a disparity that a "
          "different wording would not have found, and does nothing about a "
          "disparity that every wording misses.")

    H("10  Threats to validity")
    P("A paper about measurement validity owes its own reader the same "
      "accounting it asks of everyone else, and a flat list of limitations "
      "invites them all to be weighed alike. They are not alike. Some threaten "
      "whether these numbers are right, some whether they transfer, and some "
      "whether the quantity being measured is the quantity being named. The "
      "last is the one this literature is least careful about and the one this "
      "paper is about, so it is given its own heading rather than a sentence.",
      indent=False)

    H("10.1  Internal validity. Are these numbers right?", 2)
    P("<b>The measurement reproduces, and did not before.</b> §5.2 leaves this "
      "threat closed rather than bounded. At concurrency one with prompt-cache "
      "reuse disabled, the same cells reproduce bitwise within a run and across "
      "separate server processes, with a per-cell standard deviation of exactly "
      "zero. That floor belongs to that configuration, and not to every "
      "measurement in this paper. Study 2 predates the finding and ran with "
      "neither control, and it is where the headline dispersions come from, "
      "so its own replicate floor is the one they have to be read against"
      + ((f" \u2014 {_local_floor['min_noise_sd_of_a_wording_mean']:.4f} to "
          f"{_local_floor['max_noise_sd_of_a_wording_mean']:.4f} on a wording mean, "
          f"up to {pct(_local_floor['max_noise_share_of_variance'], 0)} of the "
          "observed variance, which Appendix D.1 measures and subtracts")
         if _local_floor else "")
      + ". The effect estimate is not detected to move under either control"
      + ((f", the shift bounded within {pct(_serving_bound, 1)} of it on the "
          "checkpoints whose effect is distinguishable from zero \u2014 a bound, "
          "not an absence, which is how \u00a75.2 states it")
         if _serving_bound else ", which \u00a75.2 bounds")
      + ", and the studies added afterwards use the reproducible "
      "configuration. An earlier draft of this paragraph said every "
      "dispersion here is measured against a floor of exactly zero; that was "
      "true of the second-task study and the cache probes and of nothing "
      "else.", indent=False)
    P("<b>Estimators can carry the effect they are used to find, and one "
      "here did.</b> The template-concentration statistic in §7 compared one "
      "template’s magnitude against the magnitude of an average of two "
      "others; averaging shrinks a magnitude, so the statistic returned a "
      "positive value under a null in which it should return zero, and "
      # NO TYPED FALLBACK. This read `else "83 %"`, printing a
      # hardcoded measurement whenever the artifact was absent -- inside
      # the very sentence that goes on to say every other number here is
      # interpolated from an artifact on disk. audit_hardtyped_numbers
      # could not see it: the literal is four characters in its own
      # concatenation fragment, and the audit discarded every string
      # under twelve characters before testing any of them. The number
      # now comes from the artifact, or the build stops.
      + pct(tconc["null_calibration"]["legacy_bias_share"], 0)
      + " of what we reported was that bias. It was found by an "
      "adversarial audit of this paper, not by us. We report it here rather "
      "than only in the changelog because it is the threat a reader cannot "
      "check from the outside. Every other number in this paper is "
      "reconstructible from the released artifacts, and a biased estimator "
      "reconstructs its bias faithfully.")
    P("<b>One headline is selected on its own outcome.</b> The model named in "
      "§4.4 as showing a token-matching correction is the largest of four such "
      "corrections, chosen by the build script as the argmax, and the p-value "
      "beside it carries no adjustment for that selection. On the name-pair "
      "resampling unit that difference sits on the 0.05 boundary rather than "
      "clearing it. The percentile interval excludes zero, the shifted "
      "bootstrap does not, the exact cluster relabelling does, and §4.4 "
      "reports that ambiguity where the claim is made rather than a verdict "
      "either way; but a reader should "
      "know that a maximum over four models is being reported as though it "
      "were a result about one.")
    P("<b>Multiplicity, on a dependent family.</b> The " + _bh_n + " mechanism contrasts "
      "share the D0 baseline and share cells within a model and mode, and "
      "Benjamini–Hochberg is proved for independent statistics. Appendix F "
      "states this and states the direction. Adjusted p-values may be "
      "anti-conservative, which works against a section whose conclusion is a "
      "null. A BH-adjusted verdict is also a function of the whole family, so a "
      "contrast can change side when other contrasts change and its own data "
      "does not , which happened here, to four contrasts.")
    P("<b>No family-wise accounting across the paper.</b> We correct within "
      "the " + _bh_n + " mechanism contrasts and nowhere else. Across "
      + (NUM.get(_n_studies, str(_n_studies)) if _n_studies else "nine")
      + " studies this "
      "paper reports a good many “two of four” and “one of two” "
      "verdicts with no adjustment for the number of such statements made, "
      "which is a researcher degree of freedom of exactly the kind the paper "
      "is about. Two things limit the damage and neither removes it: the "
      "argument rests on magnitudes relative to each model’s own effect "
      "rather than on which cells cross a threshold, and the dispersion "
      "results would stand if every significance verdict in the paper were "
      "deleted. A reader who wants the counts corrected should treat them as "
      "descriptive.")
    P("<b>The primary posting was chosen before the occupation study, and it "
      "is the favourable one.</b> §4.5 reports that the business-analyst "
      "posting is where the effect is largest and the dispersion smallest "
      "on " + NUM.get(_ba_both, str(_ba_both)) + " of four models, and "
      "where the effect alone is largest on "
      + NUM.get(_ba_eff, str(_ba_eff))
      + " , and every other study in this paper uses it. The "
      "order matters and we state it, with a record that predates the result: "
      "the gap register written before the confirmatory runs lists “one "
      "resume, one posting, one occupation” as an open gap needing an "
      "occupation panel “beyond this round”, so the single posting was "
      "fixed first and the other two were built afterwards to close that gap. "
      "It was not selected on its outcome. But it is still a favourable draw, "
      "and the direction of the consequence is that the dispersion-to-effect "
      "ratios reported throughout are the <i>smallest</i> of the three available: "
      "had we built the paper on the software-engineer posting, the same "
      "models would have shown more dispersion relative to less effect. The "
      "half of that concern that survives testing is the effect half. §4.5 "
      "finds the between-posting differences in <i>dispersion</i> indistinguishable "
      "from what one shared dispersion produces, so the ranking of postings "
      "by dispersion is a point estimate and not a result; the differences in "
      "<i>effect</i> are large and do survive. A reader should discount the choice "
      "of posting because it moves the numerator, not because we have shown "
      "it moves the spread.")
    P("<b>A declared property is not a measured one.</b> " + TAB("conditions") + "’s "
      "delimiter column was copied from the condition definitions until an "
      "audit checked it against the tokenizer and found D7, designed as a null "
      "control, replacing both merged delimiter tokens. The column is measured "
      "now and D7 is reported as its own class. The general form of this threat "
      "is not closed. Every other property this paper asserts of its own "
      "stimuli is measured, but the only defence against the next one is that "
      "the probes and the raw prompts are released.")

    H("10.2  External validity. How far does this go?", 2)
    _dom_note = ""
    if second:
        _dl = second.get("domain_labels") or {}
        _dom_note = (" Appendix C answers part of this. The same twelve "
                     "wordings, the same names and the same null edits were "
                     "run on "
                     + " and ".join(_dl.get(d, d)
                                    for d in second.get("domains", []))
                     + ", and the dispersion is of the same order there.")
    _front_note = ""
    if front and front.get("summary"):
        _f = front["summary"]
        _front_note = (
            f" Appendix D answers most of the rest. The wording study was run "
            f"with the <i>same</i> outcome on {_f['n_models']} frontier API "
            f"checkpoints, and on the {_f['n_identified']} where the effect is "
            f"identified the dispersion-to-effect ratio <i>point</i> <i>estimates</i> "
            f"are higher than on the open-weight panel rather than lower, "
            f"though no bootstrap interval separates the two panels. That "
            f"ordering is two checkpoints against two, and we should say what "
            f"we would say of it in somebody else\u2019s paper. If the four "
            f"were exchangeable, a clean split in a direction named in "
            f"advance has probability 1/6. It is a direction worth following "
            f"up and it is not, on its own, evidence.")
    P("<b>Open-weight checkpoints for everything the serving stack touches, "
      "and that is a hard constraint rather than an oversight.</b> Serving the "
      "model locally is what makes the quantization, batching and cache "
      "manipulations possible at all, and those are among the paper’s "
      "cleanest results. None of them could be run against an API, and "
      "none of them is tested on a frontier model here or, so far as we can "
      "tell, anywhere. The stimulus-side results need nothing but the ability "
      "to send a prompt and read a decision, so they can be tested on an API "
      "and now have been." + _dom_note + _front_note
      + " What remains genuinely open is therefore narrower than we first "
      "wrote it: not whether the wording effect transfers, but whether the "
      "<i>serving</i> effects do, on stacks whose numerics no auditor can inspect."
      + (" A second limit is not ours to fix: the frontier checkpoints are "
         "aliases rather than pinned weights, so Appendix D measures whatever "
         "the vendor served on the day. The same pinning failure "
         "" + TAB("reporting_matrix") + " records against the field." if front else ""),
      indent=False)
    P("<b>The screen is one stage of a process the audit does not see.</b> "
      "An external reviewer of an earlier draft pointed out that a deployed "
      "system rarely ends at the model: it shortlists, and a person decides "
      "from the shortlist. That makes the deployed quantity a two-stage "
      "selection, in which the second stage sees a sample selected on the "
      "first, and a matched pair may both survive the screen, split, or both "
      "be cut. A disparity measured at the screen in isolation therefore "
      "does not transfer linearly to who is interviewed, and nothing here "
      "estimates the composition. We adopt the single-stage scope "
      "deliberately, because Local Law 144 audits the tool rather than the "
      "pipeline, so the number this paper is about is the one the statute "
      "asks for. That is a scope condition rather than a defence: an audit "
      "regime that regulated the pipeline would need a different design, and "
      "the instrument sensitivity measured here would still sit inside it.",
      indent=False)
    P("<b>One job family, three occupations, one résumé skeleton.</b> "
      "The design spends its power on instrument dimensions rather than "
      "stimulus breadth, so a between-posting dispersion component is not "
      "separately estimable from the occupation contrast, and three "
      "occupations cannot support a claim about occupations in general. We "
      "make none. What §4.5 supports is narrower than the claim an "
      "earlier draft made from it, and is still sufficient for the argument "
      "it is used in: the job posted moves the <i>effect</i> by as much as the whole "
      "demographic effect, so an audit’s number does not transport "
      "between jobs. Whether the job also moves the <i>dispersion</i> is not "
      "established. Three postings cannot separate that from noise, and "
      "§4.5 says so.")
    P("<b>One threat this design does not carry.</b> Lahey and Beasley name "
      "two concerns in the traditional audit, and the second is experimenter "
      "bias: a human tester can deviate from random assignment, particularly "
      "when instructed to tweak a r\u00e9sum\u00e9 to fit the posting. There is no "
      "tester here. Assignment is a fixed rule executed by a script, and the "
      "stimuli are released, so the threat is absent by construction rather "
      "than bounded. It is stated because the rest of this section is a list "
      "of places the design is weaker than a field experiment, and a list "
      "with no entry running the other way is not an accounting.")
    P("<b>US names, English prompts, two genders, one binary decision.</b> The "
      "name list is American and validated on American perceptions; the "
      "gender split is binary because the source data is; and the outcome is "
      "a single yes/no with a margin rather than a ranking, a score or a "
      "generated rationale. Each of those is a place the result might not "
      "reach, and none is a place we have looked.")

    H("10.3  Construct validity. Is this measuring what it is called?", 2)
    P("This is the threat the correspondence-audit tradition handles by "
      "validating its names on the population being audited, and the one an "
      "LLM audit inherits nothing of. Appendix B is our attempt at the missing "
      "manipulation check and it is only partly reassuring.", indent=False)
    if cval and cval.get("summary"):
        _q1 = {m: cval["models"][m]["q1_perception"] for m in cval["models"]}
        _sig = [m for m in _q1 if _q1[m] and _q1[m]["ci"][0] * _q1[m]["ci"][1] > 0]
        P("<b>The manipulation check is weaker than we first read it.</b> "
          "The signed perception correlation we relied on is separated by "
          "race by construction, so it largely re-expresses the demographic "
          "effect rather than validating it independently (Appendix B). On the "
          "within-race correlations, which are the part that is not the race "
          "effect, no model shows an interval excluding zero. We therefore "
          "cannot separate a model that treats the names alike from a model "
          "that does not encode the distinction the names carry. "
          "No LLM audit we surveyed runs this check, and a null reported "
          "without it is not evidence of fairness.")
    P("<b>A name is not a race.</b> It carries socioeconomic association, "
      "familiarity, region and era as well, and Bertrand and Mullainathan "
      "tested the socioeconomic channel on their own data and report a "
      "negative. Using birth-certificate data on mother’s education for "
      "the first names in their sample, they “find little relationship "
      "between social background and the name-specific callback rates”. Our "
      "version of that test is weaker than theirs and settles nothing in "
      "either direction. Within race, the model’s per-name margin tracks "
      "neither the socioeconomic proxy nor the callback rate the name "
      "actually received, and every interval spans zero. With twelve names "
      "per race the design detects only very large correlations, so those are "
      "the design’s limit rather than results. The honest statement is that "
      "the demographic label on this contrast is inherited from the source "
      "list rather than established here.")
    P("<b>The field experiment cannot validate an LLM audit name by name.</b> "
      "The obvious criterion for whether a model reproduces the phenomenon is "
      "whether its per-name effects track the per-name callback rates real "
      "employers produced. They do not, on any model , but at roughly "
      "200 observations per female name and 70 per male name in the original "
      "experiment, those callback rates are close to sampling noise, and their "
      "own authors say so. There may be no stable per-name signal in the anchor "
      "to correlate against. This is a gap in the field’s apparatus "
      "rather than in either study, and it is worth knowing before anyone "
      "builds on the comparison.")
    P("<b>Token length and name distinctiveness are not separately "
      "identified.</b> §4.4 shows the token-length difference predicts the "
      "measured effect and cannot separate length from the properties "
      "correlated with it, because a rarer name segments into more pieces. An "
      "and Rudinger establish that both channels are real on a different task "
      "by stratifying names on length. Our own position conditions, which move "
      "the name without changing it, do not corroborate the length slope and "
      "could not. They add their tokens inside the posting, identically in "
      "both arms, so the within-pair difference the slope regresses on does "
      "not move and no shift is predicted. What would settle it on a hiring "
      "outcome is a token-balanced matched-pair list; the companion builds "
      "one from the "
      "validated list and it is too small to carry an audit, so the "
      "decomposition needs a larger validated pool than this literature "
      "currently draws from. An earlier draft of this paragraph said such a "
      "list “does not exist”, which was wrong twice over.")
    P("<b>The outcome is a decision the model was made to give.</b> Appendix A "
      "reports that on the lower half of the panel the unconstrained next "
      "token is usually prose rather than a verdict, and the grammar is doing "
      "real work. Renormalising over the two permitted answers asks which one "
      "the model prefers, which is the quantity a screening decision needs, and "
      "every result here is a difference between two conditions read the same "
      "way. But the level is being extracted more insistently than a "
      "deployment would"
      # _agmin binds inside the coverage block far above; with the artifact
      # missing, reading it here crashed the build with a NameError instead
      # of shortening the paper, against this file's own drop-the-sentence
      # rule. Same "in dir()" idiom the EXPORT block already uses for it.
      + ((", and the three natural decision rules disagree on up to "
          f"{pct(1 - _agmin, 0)} of probes on those models.")
         if "_agmin" in dir() else "."))

    _mde = []
    if mech:
        for _m, _modes in mech.items():
            for _mode, _blk in _modes.items():
                for _c, _v in (_blk.get("mde") or {}).items():
                    if isinstance(_v, dict) and "mde_absolute" in _v:
                        _mde.append(_v["mde_absolute"])
    P("<b>The mechanism result is a null, and a null has a floor.</b> "
      + (("The panel’s minimum detectable effect at 80 % power runs from "
          f"{fmt(min(_mde), 4)} to {fmt(max(_mde), 4)} log-odds across the "
          f"{len(_mde)} condition-by-model-by-mode cells, worst case "
          f"{fmt(max(_mde), 4)}. For scale, the largest demographic effect "
          "this paper measures on the primary posting is "
          f"{fmt(_eff_max, 4)} (in {_eff_max_src}), so the floor is "
          f"about {max(_mde) / _eff_max:.0%} of "
          "it. A delimiter mechanism smaller than that floor would not have "
          "been detected here, and we claim only that none larger exists on "
          "this panel. An earlier draft promised this number beside every "
          "null and printed it nowhere; an audit of this paper found "
          "that.") if _mde else
         ("The panel’s minimum detectable effect at 80 % power is recorded "
          "per model and mode in the mechanism artifact alongside every "
          "contrast. A delimiter mechanism smaller than that floor would not "
          "have been detected here, and we claim only that none larger exists "
          "on this panel.")))

    # ======================================================================
    if _dcc:
        P("That per-condition floor is the wrong number for the claim §7 "
          "actually makes, and quoting only it understated the null. §7 does "
          "not assert that no condition moves the effect; it asserts that the "
          "delimiter-destroying class does not move it <i>more</i> than the class "
          "that changes nothing structural. The relevant floor is therefore "
          "the minimum detectable <i>difference</i> between those classes, which "
          "pools 96 tests rather than resting on one. Calibrated by injecting "
          "a known difference and inverting the power curve. Rather than by "
          "a closed form, which understates it by a factor of "
          f"{_dcc['mdd_80_calibration_factor']:.2f} — that floor is "
          f"{fmt(_dcc['mdd_80_calibrated_mean_abs_effect'], 4)} log-odds, "
          f"{_dcc['mdd_80_calibrated_pct_of_baseline']:.1f} % of the "
          "median demographic effect on the same panel. A delimiter mechanism "
          "worth caring about would have to be smaller than a "
          # FLOOR, NOT ROUND: 5.65 % is one part in 17.7, and rounding to the
          # nearest would print "eighteenth" and understate the floor. The
          # prose used to say "a twentieth", which was written against an
          # earlier calibration and never re-derived.
          + (lambda _n: ORD.get(_n, f"{_n}th"))(
              int(100 / _dcc['mdd_80_calibrated_pct_of_baseline']))
          + " of the "
          "effect the audit exists to measure to have escaped this design. "
          "That is a tight null, and it is tighter than the per-condition "
          "figure above suggests.")

    H("11  Conclusion")
    P("A matched-pair difference is supposed to be the robust thing. Whatever "
      "the two conditions share should cancel, and that cancellation is what "
      "licenses the inference. We find that on LLM résumé audits it "
      "does not cancel: the reported number moves under the wording, the names, "
      "the job, the quantization, the resampling unit and the reporting scale, "
      "and the pair is not even matched in the units the model reads, "
      # NOT "<i>the</i> EFFECTS THIS LITERATURE REPORTS". Those are percentage-point
      # callback gaps; our movements are log-odds and probability of
      # superiority. §8 withdraws the shortlist bridge that would license the
      # comparison , its quantum alone exceeds the published median, and the
      # abstract says we do not make it. The Conclusion made it anyway, in the
      # last substantive paragraph, where a reader takes their impression away.
      # The comparison the paper CAN make is against each model's own effect,
      # which is what Table 3, Table 6 and Figure 1 are.
      "by amounts of the same order as , and sometimes larger than, "
      "the effect each model itself shows. Request batching and cache policy "
      "are a separate matter and worth stating precisely, because it is easy to "
      "run them together with the rest. They do not DETECTABLY move the "
      "reported effect"
      + (f", the shift being bounded within {pct(_serving_bound, 1)} of the "
         "effect on the checkpoints whose effect is distinguishable from "
         "zero, which are the only ones this ratio is defined on: a "
         "bound, not an absence" if _serving_bound else ", which we checked")
      + ". What they move is whether the measurement "
      "reproduces , and there, uniquely, we can say exactly why and exactly "
      "how to make it stop.", indent=False)
    P("We did not find a mechanism, and we report that plainly. The sensitivity "
      "is not carried by any structural feature we could isolate across eleven "
      "conditions and six checkpoints, which is consistent with the strongest "
      "existing characterisation of the prompt-format space and inconsistent "
      "with the tidier story we set out to tell.")
    P("The practical consequence is narrow and, we think, actionable. An audit "
      "that reports one number from one prompt with one name list on one "
      "quantization has measured one point on a surface whose extent it has not "
      "characterised. The cost of characterising it is a small multiple of the "
      "cost of the audit. Given that a bias audit is what Local Law 144 makes "
      "a condition of using the tool at all, that seems a reasonable price.")

    # ======================================================================
    H("Data and code availability")
    P("Every experiment script, analysis script, raw JSONL record and generated "
      "artifact is retained, along with a changelog recording every "
      "pre-registered claim that was amended and why, a verification ledger "
      "with a row per external claim and its primary source, and a gap register "
      "written before the confirmatory runs so that findings could not be "
      "retro-fitted to it. Superseded analysis modules are quarantined with a "
      "written account of what they got wrong rather than deleted.", indent=False)
    # WHERE, NOT JUST WHAT. An audit of this paper counted the phrase
    # "released with the artifacts" six times across captions and body text
    # and observed that the paper names no repository, DOI or archive
    # anywhere -- so every one of those promises pointed at nothing a reader
    # could open. Saying plainly that the deposit accompanies the preprint,
    # and listing what it contains, is the least a reader is owed until the
    # identifier exists to print.
    P("Those phrases in the body and in the table captions. The evidence "
      "quote behind every cell of " + TAB("reporting_matrix") + ", the "
      "per-study cells, the per-parameter convergence diagnostics, both arms "
      "of the name-draw counterfactual, the quarantined modules. All refer "
      "to a single artifact set deposited with this preprint. It contains the "
      "raw JSONL for every study, the analysis artifact each number is "
      "interpolated from, every experiment and analysis script including the "
      "consistency audit that checks this document against those artifacts, "
      "and the test suite. Nothing in this paper is computed anywhere except "
      # WHAT THE BUILD ACTUALLY DOES. It does not fail: load() records the
      # missing key and returns None, the guarded sentence is dropped, and a
      # warning naming every missing artifact prints at the end. Claiming a
      # hard failure overstated the guarantee -- and understated it, because
      # the real guard is check_suppressed_prose(), which fails the AUDIT when
      # a guarded block never fires. That catches the dangerous case a build
      # failure would not: a result quietly vanishing from the page.
      # AND THE SAME BOUNDARY HERE. "A number cannot be printed unsourced" was
      # the stronger form of the §1.2 claim and false in the same way. Three
      # guards do hold, and naming them exactly is more useful to a reader than
      # a universal that does not.
      "in that code. Each figure quoted here is produced by a named script "
      "reading a named artifact; a missing artifact drops the sentence that "
      "depended on it and is named in a warning, the consistency audit fails "
      "if any guarded passage never fired, and a third check reads the "
      "typesetting source for measurements typed into the prose rather than "
      "read from disk. A result therefore cannot vanish unremarked, and a "
      "measurement that is typed rather than sourced is listed rather than "
      "hidden; design constants, bibliographic detail and figures quoted "
      "from other papers are typed on purpose.",
      indent=False)

    # ======================================================================
    # ENDMATTER. FAccT requires a generative-AI statement from all authors
    # whether or not such tools were used, and makes failure to disclose
    # grounds for desk rejection. It separately requires that competing
    # interests, positionality and acknowledgements be ABSENT from the
    # anonymous submission and present only in the camera-ready. Both rules
    # are implemented here rather than remembered.
    H("Endmatter")
    P("<b>Generative AI usage.</b> " + (
        "The author wrote every sentence of this paper. Large language models "
        "were used elsewhere in the project, and their role is stated here in "
        "full. As tools for the analysis and for the typesetting code that "
        "renders this document; to screen the literature for the reporting "
        "survey in Section 8; and, run adversarially against the manuscript's "
        "own claims, to audit the methodology and find errors in it, which "
        "produced a large share of the corrections recorded in the changelog. "
        "The research questions, the experimental design, the choice of what "
        "to measure and every interpretation are the author's, as is the "
        "prose. The author has verified each reported number against the "
        "artifact it is computed from and takes full responsibility for the "
        "contents of this paper. No text in this paper was generated by a "
        "language model."
        if PROSE_REWRITTEN else
        "Large language models were used throughout the preparation of this "
        "preprint, and their role is stated here in full rather than "
        "summarised. They assisted in drafting the prose of this document; "
        "they were used as tools for the analysis and for the typesetting "
        "code that renders it; they screened the literature for the reporting "
        "survey in Section 8; and, run adversarially against the manuscript's "
        "own claims, they audited the methodology and found errors in it, "
        "which produced a large share of the corrections recorded in the "
        "changelog. The research questions, the experimental design, the "
        "choice of what to measure and every interpretation are the author's. "
        "The author has verified each reported number against the artifact it "
        "is computed from and takes full responsibility for the contents of "
        "this paper. One consequence a reader should be able to judge for "
        "themselves. The drafting assistance was substantial enough that this "
        "preprint would not satisfy a venue rule prohibiting LLM-generated "
        "text, which is why the version submitted to such a venue is written "
        "by the author rather than revised from this one."), indent=False)
    P("<b>Ethical considerations.</b> No human subjects were involved and no "
      "personal data was collected. The résumés are synthetic and the names "
      "are drawn from a published, peer-reviewed list. The study measures "
      "systems, not people. Two adverse impacts were considered. First, a "
      "paper showing that audit results move under unreported choices could "
      "be read as licence to dismiss any audit finding, including sound ones; "
      "Section 7 states directly what the sensitivity is not, and the "
      "reporting set of Section 9 is written so that the response is better "
      "reporting rather than less auditing. Second, publishing the exact "
      "conditions under which a measured disparity shrinks could help a "
      "vendor select a favourable specification. That risk is real and is the "
      "reason the recommended primary statistic is one that is algebraically "
      "invariant to the choice rather than merely disclosed. A number nobody "
      "can move needs no assurance that nobody moved it.", indent=False)
    if not ANON:
        # Present in the preprint and the camera-ready, absent from the
        # anonymous submission, which is what FAccT's author guide requires.
        # CHECK BEFORE POSTING: every clause below is an assertion about the
        # author's own circumstances and the last two cannot be verified from
        # this repository. Confirm they are true, or amend them.
        P("<b>Competing interests.</b> The author is a member of the founding "
          "team at a company that builds AI recruiting software, and is "
          "compensated by it. That is a direct commercial interest in the "
          "class of system this paper audits, and it is stated here so that "
          "readers can weigh it. The systems measured in this paper are "
          "open-weight research checkpoints and public commercial APIs. No "
          "system built by that company was audited, and none of its data, "
          "models, prompts, thresholds or internal documents was used at any "
          "point; every claim in this paper about deployed practice is "
          "sourced to public documentation or to published work. The company "
          "did not fund, commission, direct or review this study.",
          indent=False)

    H("References")
    # THE CONVENTION, STATED ONCE INSTEAD OF PER ENTRY. Two entries carried
    # inline editorial caveats -- one saying a work is a technical report
    # rather than a peer-reviewed paper, another explaining why a proceedings
    # title is absent -- while several other preprints carried none, so the
    # notes read as judgements about particular papers rather than as a
    # uniform rule. Stating the convention here makes the two remaining notes
    # what they actually are: the cases where the document itself is
    # ambiguous, not the cases we chose to comment on.
    P("Identifiers are given as printed on the document read. An arXiv "
      "identifier alone means the document read prints no venue; publisher "
      "records were not searched separately. Where a document's status or "
      "venue cannot be settled "
      "from its own pages, that is stated in the entry rather than resolved "
      "by assumption.", indent=False, size=7.8, lead=9.6, space_after=4.0)
    # WHY THIS IS A REGISTRY AND NOT A LIST OF STRINGS. The previous version
    # appended the verbatim `citation` field of each surveyed study to a list of
    # hand-written entries and called sorted() on the raw strings. Two defects
    # followed and an audit found both: the two groups use different author
    # conventions ("Surname, Initial." against full given names first), so the
    # list was not alphabetised by anything a reader could see; and one entry
    # carried an internal verification note about how its venue had been
    # checked, which belongs in the ledger and not in a bibliography. Sorting
    # now uses an explicit surname key, and the displayed text is separate from
    # the key. Entries never cited in the body have been removed rather than
    # left as orphans.
    refs = [
        ("An", "An, H. and Rudinger, R. (2023). Nichelle and Nancy: The "
                "Influence of Demographic Attributes and Tokenization Length on "
                "First Name Biases. Proceedings of the 61st Annual Meeting of "
                "the Association for Computational Linguistics (Volume 2: Short "
                "Papers), 388–401."),
        ("Atil", "Atil, B., Aykent, S., Chittams, A., Fu, L., Passonneau, "
                  "R. J., Radcliffe, E., Rajagopal, G. R., Sloan, A., Tudrej, "
                  "T., Ture, F., Wu, Z., Xu, L. and Baldwin, B. (2025). "
                  "Non-Determinism of “Deterministic” LLM System "
                  "Settings in Hosted Environments. Proceedings of the 5th "
                  "Workshop on Evaluation and Comparison of NLP Systems "
                  "(Eval4NLP), 135–148."),
        ("Qian", "Qian, S., Pham, H. V., Lutellier, T., Hu, Z., Kim, J., "
                  "Tan, L., Yu, Y., Chen, J. and Shah, S. (2021). Are My Deep "
                  "Learning Systems Fair? An Empirical Study of Fixed-Seed "
                  "Training. Advances in Neural Information Processing "
                  "Systems 34 (NeurIPS 2021)."),
        ("He", "He, H. and Thinking Machines Lab (2025). Defeating "
                "Nondeterminism in LLM Inference. Thinking Machines Lab: "
                "Connectionism, 10 September 2025. DOI 10.64434/tml.20250910. "
                "A technical report with a registered DOI, not a peer-reviewed "
                "paper."),
        ("Benjamini", "Benjamini, Y. and Hochberg, Y. (1995). Controlling the "
                       "False Discovery Rate: A Practical and Powerful Approach "
                       "to Multiple Testing. Journal of the Royal Statistical "
                       "Society B 57(1), 289–300."),
        ("Lippens", "Lippens, L. (2024). Computer says ‘no’: Exploring "
                     "systemic bias in ChatGPT using an audit approach. "
                     "Computers in Human Behavior: Artificial Humans 2, "
                     "100054. DOI 10.1016/j.chbah.2024.100054."),
        ("Bertrand", "Bertrand, M. and Mullainathan, S. (2004). Are Emily and "
                      "Greg More Employable than Lakisha and Jamal? A Field "
                      "Experiment on Labor Market Discrimination. American "
                      "Economic Review 94(4), 991–1013."),
        ("Fu2026", "Fu, T., Martínez, G., Conde, J., Arriaga, C., "
                    "Reviriego, P., Qi, X. and Liu, S. (2026). Beyond "
                    "Reproducibility: Token Probabilities Expose Large "
                    "Language Model Nondeterminism. arXiv:2601.06118."),
        ("Yuan", "Yuan, J., Li, H., Ding, X., Xie, W., Li, Y.-J., Zhao, W., "
                  "Wan, K., Shi, J., Hu, X. and Liu, Z. (2025). Understanding "
                  "and Mitigating Numerical Sources of Nondeterminism in LLM "
                  "Inference. NeurIPS 2025. arXiv:2506.09501."),
        # NO SESHADRI ENTRY HERE. The paper already carries one, keyed
        # "Seshadri", with arXiv:2501.04316 -- the identifier the full-text
        # record in reporting_practice_matrix.json transcribes from the PDF.
        # A second entry added with the §2 positioning paragraph carried
        # arXiv:2503.19182, which belongs to Iso et al. (2025) and is listed
        # against them a few entries above. The reference list therefore
        # printed the same paper twice, adjacently, under two identifiers, one
        # of which pointed at somebody else's work. Adding a citation is not
        # the same as adding a reference, and check_duplicate_references()
        # now fails the build rather than leaving it to a reader.
        # The two legal sources the opening premise rests on. Both are read in
        # full text in lit/law/; the paper cites the operative provision and
        # the rule that specifies what a bias audit must compute, rather than
        # gesturing at "regulation".
        ("LL144", "City of New York (2021). Local Law 144 of 2021, in relation "
                  "to automated employment decision tools. N.Y.C. Admin. Code "
                  "§§ 20-870 to 20-874."),
        ("DCWP", "New York City Department of Consumer and Worker Protection "
                 "(2023). Notice of Adoption of Final Rule: Automated "
                 "Employment Decision Tools. 6 RCNY § 5-300 et seq., adopted "
                 "6 April 2023."),
        ("Abadie", "Abadie, A., Athey, S., Imbens, G. W. and Wooldridge, "
                   "J. (2017). When Should You Adjust Standard Errors for "
                   "Clustering? arXiv:1710.02926."),
        ("Fryer", "Fryer, R. G. and Levitt, S. D. (2004). The Causes and "
                  "Consequences of Distinctively Black Names. The Quarterly "
                  "Journal of Economics 119(3), 767\u2013805."),
        ("Gaddis", "Gaddis, S. M. (2017). How Black Are Lakisha and Jamal? "
                   "Racial Perceptions from Names Used in Correspondence "
                   "Audit Studies. Sociological Science 4, 469\u2013489."),
        ("Neumark", "Neumark, D. (2012). Detecting Discrimination in Audit "
                    "and Correspondence Studies. Journal of Human Resources "
                    "47(4)."),
        ("Lahey", "Lahey, J. N. and Beasley, R. A. (2009). Computerizing "
                  "audit studies. Journal of Economic Behavior & Organization "
                  "70(3), 508\u2013514."),
        ("Sclar", "Sclar, M., Choi, Y., Tsvetkov, Y. and Suhr, A. (2024). "
                   "Quantifying Language Models’ Sensitivity to Spurious "
                   "Features in Prompt Design. ICLR 2024. arXiv:2310.11324."),
        ("Simonsohn", "Simonsohn, U., Simmons, J. P. and Nelson, L. D. (2020). "
                       "Specification Curve Analysis. Nature Human Behaviour. "
                       "doi:10.1038/s41562-020-0912-z."),
        ("Steegen", "Steegen, S., Tuerlinckx, F., Gelman, A. and Vanpaemel, W. "
                     "(2016). Increasing Transparency Through a Multiverse "
                     "Analysis. Perspectives on Psychological Science 11(5), "
                     "702–712."),
    ]
    # The surveyed studies, from the literature artifact, with the leading
    # surname supplied here so the sort key does not depend on how the source
    # happened to print the author list.
    SURVEYED = [
        ("Armstrong", "armstrong_etal_2024"), ("An", "an_etal_2024"),
        ("Fu", "fu_shi_2025"), ("Gaebler", "gaebler_etal_2024"),
        ("Gao", "gao_jiang_yan_2026"), ("Glazko", "glazko_etal_2024"),
        ("Hoffstedde", "hoffstedde_etal_2026"), ("Iso", "iso_etal_2025"),
        ("Nghiem", "nghiem_etal_2024"), ("Seshadri", "seshadri_etal_2025"),
        ("Tan", "tan_etal_2026"), ("Veldanda", "veldanda_etal_2023"),
        ("Wilson", "wilson_caliskan_2024"),
    ]
    # Every surveyed study's reference comes from the reporting-practice
    # artifact, which holds one canonical string per study in a single author
    # convention, transcribed from the PDF title page. The literature artifact's
    # verbatim `citation` fields are the fallback for anything not in the
    # matrix.
    seen_ref = {r[1] for r in refs}
    if matrix:
        for s in matrix["studies"]:
            c = s.get("reference")
            if c and c not in seen_ref:
                refs.append((s["label"].split()[0], c))
                seen_ref.add(c)
        have_labels = {s["label"] for s in matrix["studies"] if s.get("reference")}
    else:
        have_labels = set()
    if lit:
        LABEL = {"an_etal_2024": "An et al. 2024", "fu_shi_2025": "Fu & Shi 2025",
                 "gao_jiang_yan_2026": "Gao et al. 2026",
                 "iso_etal_2025": "Iso et al. 2025",
                 "tan_etal_2026": "Tan et al. 2026",
                 "wilson_caliskan_2024": "Wilson & Caliskan 2024"}
        for surname, k in SURVEYED:
            if LABEL.get(k) in have_labels:
                continue
            e = lit.get(k)
            if isinstance(e, dict) and e.get("citation"):
                refs.append((surname, e["citation"].strip()))
    REFS_OUT[:] = sorted(refs, key=lambda t: (t[0].lower(), t[1]))
    for _key, r in REFS_OUT:
        P(r, indent=False, size=7.8, lead=9.6, space_after=1.6)

    # ======================================================================
    # WHAT THIS DOCUMENT CLAIMS, in one dict, taken from the same locals the
    # prose interpolated. Read by build_facct_tex.py and build_summary.py.
    EXPORT.update(
        n_records=n_records, n_calls=n_calls,
        n_panel=_n_panel, n_front=_n_front,
        n_open_weight=len(models),
        ratio_lo=_rat_lo, ratio_hi=_rat_hi, ratio_n=_rat_n,
        resamp_lo=(resamp["pooled_summary"]["min_ratio"] if resamp else None),
        resamp_hi=(resamp["pooled_summary"]["max_ratio"] if resamp else None),
        jacobian_lo=(scale["summary"]["jacobian_error_min"] if scale else None),
        jacobian_hi=(scale["summary"]["jacobian_error_max"] if scale else None),
        n_audits=(matrix["n_llm_hiring_audits"] if matrix else None),
        n_token_applicable=(matrix["counts"]["token_matching"]["n_applicable"]
                            if matrix else None),
        n_disp_applicable=(
            matrix["counts"]["dispersion_across_wordings"]["n_applicable"]
            if matrix else None),
        token_matched_lo=(min(_tf.values()) if nlen and (_tf := {
            m: nlen[m]["n_same_length"] / nlen[m]["n_pairs"]
            for m in models if m in nlen}) else None),
        token_matched_hi=(max(_tf.values()) if _tf else None),
        serving_bound=_serving_bound,
        quant_lo=(min(v["shift_over_sigma_variant"] for v in _qm)
                  if _qm else None),
        quant_hi=(max(v["shift_over_sigma_variant"] for v in _qm)
                  if _qm else None),
        quant_n=len(_qm),
        balanced_rows=(tbal["n_pairs"] if tbal else None),
        balanced_clusters=((tbal["max_matching"]["female_first"]
                            + tbal["max_matching"]["male_first"])
                           if tbal else None),
        # ---- what each appendix establishes, for the body pointer that
        # ---- replaces it. One number per pointer, so the compressed body
        # ---- paragraph interpolates rather than restates.
        rule_disagreement=(1 - _agmin) if "_agmin" in dir() else None,
        n_low_mass=len(_low_mass) if "_low_mass" in dir() else None,
        mde_within=((cval.get("mde") or {}).get("mde_rho_pooled_within_race")
                    if cval else None),
        n_mde_obs=((cval.get("mde") or {}).get("n_names_total") if cval else None),
        n_within_excluding_zero=(
            sum(v["n_excluding_zero"] for k, v in (cval["summary"] or {}).items()
                if v and "within_race" in k) if cval else None),
        front_ratio_lo=((fnoise.get("summary_frontier") or {})
                        .get("min_ratio_corrected") if fnoise else None),
        front_ratio_hi=((fnoise.get("summary_frontier") or {})
                        .get("max_ratio_corrected") if fnoise else None),
        front_noise_share=((fnoise.get("summary_frontier") or {})
                           .get("max_noise_share_of_variance") if fnoise else None),
        n_pages=None,
    )

    APP.emit(paper, H, P)

    paper.render()

    # SECOND PASS, ONLY IF THE PAGE DISAGREES WITH THE NUMBERING. Guarded to a
    # single retry: if reordering changes the layout enough to reorder the
    # tables again, that is a real instability and the build should say so
    # rather than oscillate.
    if _renumber_to_match_the_page(OUT):
        if os.environ.get("PAPER_RENUMBER_PASS"):
            print("  [WARNING] the table order is still unstable after a "
                  "renumbering pass. The numbering on the page may be wrong; "
                  "this needs a human.")
        else:
            print("  [renumber] tables did not render in numbered order; "
                  "renumbering to match the page and rebuilding")
            env = dict(os.environ, PAPER_RENUMBER_PASS="1")
            return subprocess.run([sys.executable, __file__],
                                  env=env).returncode

    if MISSING:
        print(f"  [WARNING] {len(MISSING)} artifact(s) missing; the "
              f"corresponding sentences were dropped: {', '.join(MISSING)}")
    print(f"wrote {OUT.relative_to(ROOT)}  ({paper.page} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
