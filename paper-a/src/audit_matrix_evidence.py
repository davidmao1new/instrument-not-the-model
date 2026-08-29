"""Re-run every negative search the reporting matrix claims, against the papers.

WHY THIS EXISTS. Section 8's table makes 22 claims about each of 13 other
researchers' papers, and roughly 140 of those cells rest on a NEGATIVE search:
"grep -i for 'cache' (0 hits)". A negative claim about somebody else's work is
the most damaging thing in this project to get wrong. It is also the easiest,
because nothing about a wrong one looks wrong: the cell renders, the count
sums, the table prints.

Those searches were run once, by readers, in 2026. They have never been re-run.
Meanwhile the extracted text under `lit/text/` has been re-extracted more than
once, one study's published version was added alongside its preprint, and the
codings have been edited by hand. Any of those can silently falsify a negative.

WHAT PROMPTED IT. On 2026-08-19 a hand-written coding of a fourteenth study was
found to contain eight wrong verdicts, every one too generous, plus four
superlatives that were false or unverifiable. The method that caught them was
not re-reading the paper -- it was checking each claim against something
external to the judgement that produced it. This is that method applied to the
thirteen codings nobody has re-checked.

WHAT IT CHECKS

  NEGATIVE   every "0 hits" claim, re-run against the study's full text. A term
             that now appears is either a wrong claim or a stale extraction,
             and both need a human.
  QUOTE      every verbatim quotation of forty characters or more, with no
             upper bound, verified to appear in the text it is attributed to.
             This said "six or more words" while the rule measured forty
             characters and stopped at four hundred, so it promised a floor
             it did not implement and did not mention a ceiling that
             exempted the longest quotations in the matrix.
  EMPTY      a `reported` or `partial` verdict with neither a value nor
             evidence behind it.

WHAT IT DELIBERATELY DOES NOT DO is fail on a bare hit. These extractions are
lossy in known ways -- several of these papers put load-bearing text in figures
that are rasters -- and a term appearing only in a reference list is not a
counterexample to a claim about a methods section. So a hit is reported with
its count and its context, ranked, for a human to judge. `--strict` turns
confirmed-body hits into a non-zero exit once the known-benign ones are
recorded in BENIGN below.

    sh paper-a/src/_py.sh paper-a/src/audit_matrix_evidence.py
    sh paper-a/src/_py.sh paper-a/src/audit_matrix_evidence.py --strict
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = ROOT / "paper-a" / "data" / "reference"
MATRIX = REF / "reporting_practice_matrix.json"
OUT = REF / "matrix_evidence_check.json"
CANDIDATE = REF / "candidate_lippens_2024.json"
TXT = ROOT / "lit" / "text"

# Study label -> the extracted full text(s) it was coded from. A study with two
# entries must have the term absent from BOTH before a negative claim stands,
# which is the strict reading: An et al.'s arXiv v1 and its published ACL
# version differ, and a claim about "the paper" should hold of either.
SOURCES = {
    "An et al. 2024": ["arxiv_2406.10486.txt", "an_etal_2024_acl_short37.txt"],
    "Armstrong et al. 2024": ["armstrong_etal_2024_silicon_ceiling.txt"],
    "Fu & Shi 2025": ["arxiv_2510.19167.txt"],
    "Gaebler et al. 2024": ["gaebler_etal_2024_auditing_lm_hiring.txt"],
    "Gao et al. 2026": ["gao_jiang_yan_2026_cuhk.txt"],
    "Glazko et al. 2024": ["glazko_etal_2024_facct_disability_bias_resume.txt"],
    "Hoffstedde et al. 2026": [
        "hoffstedde_etal_2026_gender_bias_japanese_hiring.txt"],
    "Iso et al. 2025": ["arxiv_2503.19182.txt"],
    "Nghiem et al. 2024": ["nghiem_etal_2024_emnlp_name_based_bias.txt",
                           "nghiem_etal_2024_name_based_bias_employment.txt"],
    "Seshadri et al. 2025": [
        "seshadri_etal_2025_small_changes_large_consequences.txt"],
    "Tan et al. 2026": ["tan_etal_2026_small_changes.txt"],
    "Veldanda et al. 2023": ["veldanda_etal_2023_emily_greg_chatgpt.txt"],
    "Wilson & Caliskan 2024": ["wilson_caliskan_2024_aies.txt"],
    "Bertrand & Mullainathan 2004": ["bm2004.txt"],
    "Sclar et al. 2024": ["sclar2024.txt"],
    "Lippens 2024": ["lippens_2024_computer_says_no.txt"],
}

# Hits a human has looked at and judged not to falsify the claim. Each needs a
# reason; an entry with no reason is a hole, not an exemption.
#
# ALL SIX ARE THE SAME THING: a search stem that also occurs inside an ordinary
# English word in an unrelated sense. Narrowing the stems would be the wrong
# repair -- the readers chose them to catch morphological variants, and
# 'quantiz' has to match quantization -- so the homographs are recorded
# individually instead. Checked by hand on 2026-08-19; each context is quoted.
BENIGN: dict[tuple[str, str, str], str] = {
    ("Gao et al. 2026", "names_per_race", "distinct"):
        "matches 'distinctively-black and distinctively-white name lists', "
        "which is the list's provenance, not a per-group name count.",
    ("Bertrand & Mullainathan 2004", "serving_stack", "capital"):
        "matches 'log(median per capital income)', a regressor in their zip-code "
        "table. A 2004 field experiment has no serving stack to report.",
    ("Wilson & Caliskan 2024", "n_repeats", "replicat"):
        "matches 'replicating real-world patterns of bias', a claim about "
        "external validity, not a count of experimental repeats.",
    ("Veldanda et al. 2023", "n_repeats", "replicate"):
        "matches 'we replicate this experiment on state-of-art LLMs', which is "
        "replicating Bertrand and Mullainathan, not repeating their own cells.",
    ("Gao et al. 2026", "concurrency_or_batching", "worker"):
        "matches 'discriminated against black applicants, women and older "
        "workers' in a news citation about an EEOC case.",
    ("Fu & Shi 2025", "multiplicity_correction", "adjust"):
        "matches a questionnaire item, 'I am willing to adjust my work based on "
        "others' feedback' -- an instrument item, not a p-value adjustment.",

    # SURFACED ON 28 AUG 2026, when NEG_LIST was widened to the forms the
    # readers actually wrote. These searches had been harvested by nobody and
    # run by nobody; each was read in the source text before being recorded
    # here, and none overturns its cell's negative verdict.
    ("Glazko et al. 2024", "checkpoint_pinned", "version"):
        "matches 'we used two versions of GPT-4: GPT-4, unmodified ... and a "
        "customized, trained GPT'. The word distinguishes their two "
        "experimental conditions, not a dated model snapshot; no checkpoint "
        "identifier appears anywhere in the paper.",
    ("Wilson & Caliskan 2024", "checkpoint_pinned", "version"):
        "matches 'a long-form version of this paper with appendix is available "
        "at arxiv.org', which is a version of the PAPER, not of a model.",
    ("Bertrand & Mullainathan 2004", "quantization_reported", "precision"):
        "matches 'the precision of the information' in their discussion of "
        "prior work. A 2004 field experiment with paper resumes has no "
        "numerical precision to report, for the same reason the serving_stack "
        "entry above is benign.",
    ("Glazko et al. 2024", "cache_policy", "history"):
        "matches 'GPT-4, unmodified with an empty prompt history'. That is the "
        "conversational state of a web session at its first turn, not a "
        "statement about prompt caching or KV reuse in the serving stack.",
    ("Wilson & Caliskan 2024", "null_edit_control", "formatting"):
        "matches 'the text formatting used to encode instructions and query "
        "texts varies between MTEs ... we followed the recommended structure "
        "as described in that model's documentation'. Formatting differing "
        "BETWEEN models by design is the opposite of a null edit, which holds "
        "meaning fixed and varies form within one model.",
    ("Glazko et al. 2024", "multiplicity_correction", "adjust"):
        "matches 'we took the rankings at face value and did not adjust the "
        "scores to match the tie descriptions' -- a scoring decision about "
        "their own procedure, not a p-value adjustment.",
    ("Armstrong et al. 2024", "multiplicity_correction", "correct"):
        "matches 'to ensure correctness in our method's execution'. The stem "
        "catches the English word, not a multiplicity correction.",
}

# Filled by load_texts(); label -> measured extraction damage.
DEGRADED: dict[str, dict] = {}

# Terms too short or too common to test. A one- or two-character search is not
# evidence either way, and the readers used some as shorthand.
TOO_SHORT = 4

# "…'term' (0 hits)…", "…'term' = 0…", "…'term' 0 hits…"
#
# THE TERM CLASS IS RESTRICTED ON PURPOSE. An earlier version allowed any
# characters between the quotes, so on a list like
#     grep -i for 'snapshot' (0 hits), 'checkpoint' (0), 'revision' (0)
# it matched from one term's CLOSING quote to the next term's OPENING quote and
# captured "(0), " and "missing: grep -i for" as search terms. Those then
# "appeared" hundreds of times in every paper and buried the real findings under
# 875 false hits. A search term is a word, a stem or a short phrase: letters,
# digits, and the few punctuation marks that occur inside identifiers.
# A QUOTE MARK BETWEEN LETTERS IS AN APOSTROPHE. Without the lookarounds,
# the apostrophe in "Bertrand's age-related name study. grep -i 'Black'"
# opened a quote that closed on the one before Black, so the harvested term
# was the sentence fragment between them and 'Black' was never re-run --
# in a paper about name-based bias. The fragment was then counted in n_neg,
# which the paper prints as \MatrixNegSearches: a search nobody made,
# returning zero forever, reading as a confirmation.
NEG_NEAR = re.compile(
    r"(?<![A-Za-z])['‘“\"]([a-zA-Z][a-zA-Z0-9 ._\-/]{%d,39})"
    r"['’”\"](?![A-Za-z])"
    r"[^.;\n]{0,18}?\b0\b(?!\.\d)" % (TOO_SHORT - 1))
# "…returned 0 for 'a', 'b', 'c'…" / "…0 hits for 'a' and 'b'…"
# THE VERB IS NOT ALWAYS "returned" AND THE DELIMITER IS NOT ALWAYS NEXT.
# This required `(?:returns?|returned|=)` immediately before the zero and
# `for`/`:` immediately after it, which dropped every cell written as
#     "Negative greps returning 0 on the extracted text: grep -ic 'cache', ..."
# -- "returning" defeats `returns?` (it matches "return", then wants \s*0 and
# finds "ing"), and the colon sits 26 characters away. The second form the
# docstring above promises, "0 hits for 'a' and 'b'", has no verb at all and
# was never matched either. Whole studies' negative codings were dropped in
# silence, and the `unparsed` disclosure could not see it because the same
# cells carried checkable quotations.
#
# The TERM class is now NEG_NEAR's restricted one. Loosening the prefix
# without it would rebuild the over-harvest described above, where an
# unrestricted class captured "(0), " as a search term.
NEG_LIST = re.compile(
    r"(?:(?:return(?:s|ed|ing)?|yield(?:s|ed|ing)?|=)\s*\b0\b(?:\s*hits?)?"
    r"|\b0\s*hits?\b)"
    r"[^:'‘“\"\n]{0,48}?(?::|\bfor\b)"
    r"[^'‘“\"\n]{0,24}?"
    # Same apostrophe rule as NEG_NEAR, inside the repeated group: each
    # quoted term must open on a mark that is not preceded by a letter and
    # close on one not followed by a letter.
    r"((?:(?<![A-Za-z])['‘“\"][a-zA-Z][a-zA-Z0-9 ._\-/]{%d,39}"
    r"['’”\"](?![A-Za-z])[,;\s]*(?:and\s*)?)+)" % (TOO_SHORT - 1),
    re.I)
# NO UPPER BOUND. The ceiling was 400 characters, which exempted the longest
# quotations in the matrix -- precisely the passages a reader would lean on
# hardest -- from the only check that tests whether they appear in the paper
# they are attributed to. One of them, a 655-character passage from An et al.
# on their headline effect, sits below the audit's own matching floor and had
# never been tested. (It was read by hand on 28 Aug 2026 and is accurate; it
# scores low because the extracted text interleaves a footnote into the
# middle of the passage and hyphenates across line breaks, which is the
# "interleaved page furniture" category the audit already reports.)
#
# A ceiling on a verification rule is an exemption for the biggest instance
# of the thing being verified.
QUOTED = re.compile(r"[“\"]([^”\"]{40,})[”\"]")

# NOT EVERYTHING IN QUOTATION MARKS IS A QUOTATION. The readers used quotes for
# their own phrasing too -- "missing: the definition of the unit inside a
# bootstrap replicate", "grep -i for 'top-p' (0), 'seed' (0)". Those are
# descriptions of a search, not claims about what a paper says, and there is no
# syntactic way to tell them apart from a real quotation. These markers are the
# practical way: a passage describing a grep, an absence, or a table coordinate
# is the reader talking, and holding it to a verbatim-match standard produced
# most of what looked like findings on the first four runs of this audit.
COMMENTARY = re.compile(
    r"\b(grep|searched|search(es|ing)?\b|missing\b|absent\b|returns? (no|zero)"
    r"|0 hits|not reported|never (stated|reported)|column header|on line \d"
    r"|nothing\b|no hit)", re.I)


# THE FIRST VERSION OF THIS AUDIT REPORTED 151 QUOTES AS MISSING AND ALL OF
# THEM WERE PRESENT. The cause was not the codings; it was comparing against
# text that still carried the PDF's line-wrap hyphenation. "Llama2 We mainly
# follow the hyperparame- ters recommended in the original Llama2 repos- itory,7
# where..." does not contain the sentence a reader correctly transcribed from
# it. An audit that cries wolf 151 times is worse than no audit, because the
# twelve real findings are then indistinguishable from the noise.
#
# So normalisation happens on both sides, and it is deliberately aggressive:
#   * soft hyphens deleted;
#   * a hyphen followed by whitespace rejoined -- that only ever happens at a
#     line wrap, since a real compound ("state-of-the-art") has no space after
#     its hyphen;
#   * curly quotes, dashes and ligatures folded to ASCII;
#   * everything but letters, digits and spaces dropped.
LIG = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
       "’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-",
       " ": " ", "−": "-"}


# DIGIT-FULL-STOP LIGATURES. bm2004.txt carries 5,617 characters from the
# U+2488-U+249B block -- the PDF is a JSTOR scan whose font maps "8." to the
# single glyph U+248F. Every digit in Bertrand and Mullainathan is therefore
# mangled, which made every quotation from the field experiment this entire
# literature is anchored to look fabricated. The PDF itself carries the defect,
# so re-extracting does not help; folding does.
DIGIT_LIG = re.compile(r"[\u2488-\u249b]")


def norm(s: str) -> str:
    s = s.replace("\u00ad", "")
    s = DIGIT_LIG.sub(lambda m: str(ord(m.group()) - 0x2487) + ".", s)
    for a, b in LIG.items():
        s = s.replace(a, b)
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)          # rejoin line wraps
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s)
    return " ".join(s.split()).lower()


# Footnote markers survive normalisation as stray short digit tokens glued to
# the preceding word ("repository,7 where" -> "repository7 where"). A quote
# transcribed by a human will not contain them. Only used as a FALLBACK, after
# a strict comparison fails, so a quote whose numbers are actually wrong is not
# excused by it.
# A SEARCH TERM'S PUNCTUATION IS PART OF THE SEARCH.
#
# `norm()` strips punctuation, which is right for comparing sentences and wrong
# for comparing search terms. A reader looking for a dated snapshot id searched
# for "gpt-3.5-turbo-" WITH A TRAILING HYPHEN, meaning "gpt-3.5-turbo-0613".
# Stripped, that becomes "gpt 3 5 turbo" and matches the bare model name, so a
# correct claim -- that no dated snapshot is pinned -- was reported as false in
# four papers at once. "top_p" matching "top performers" is the same defect.
#
# So negative searches run against text normalised WITHOUT discarding the
# characters that carry the distinction.
def norm_kw(s: str) -> str:
    s = s.replace("\u00ad", "")
    s = DIGIT_LIG.sub(lambda m: str(ord(m.group()) - 0x2487) + ".", s)
    for a, b in LIG.items():
        s = s.replace(a, b)
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
    s = re.sub(r"[^A-Za-z0-9 ._\-/]+", " ", s)
    return " ".join(s.split()).lower()


def defootnote(s: str) -> str:
    return " ".join(re.sub(r"(?<=[a-z])\d{1,2}\b", "", w) for w in s.split())


# HOW MUCH OF A QUOTE HAS TO BE THERE, AND WHY NOT ALL OF IT.
#
# These PDFs are two-column. The extractor reads them in visual order, so page
# furniture lands in the middle of sentences: An et al.'s model list extracts as
# "...and GPT-3.5" + a footnote URL + a table fragment + "-Turbo (Ouyang et al.,
# 2022)". A reader who transcribed that sentence correctly is then reported as
# having invented it, which is exactly backwards -- and it accounted for most of
# the 82 quote failures the strict test produced.
#
# So the test is in-order SUBSEQUENCE containment with a gap budget: every token
# of the quote must appear in the text, in order, with a bounded amount of
# interpolated material. Forty-one consecutive words in order is conclusive
# evidence a sentence is present; it is not evidence about what sits between
# them. A quote that is genuinely absent still fails, because its tokens will
# not appear in order anywhere.
GAP_BUDGET = 0.6          # interpolated tokens allowed, as a share of the quote
MIN_COVER = 0.92          # share of the quote's tokens that must be found


# WHERE THE SPACES ARE IS NOT EVIDENCE OF ANYTHING.
#
# PDF text extraction inserts and deletes spaces for reasons that have nothing
# to do with the words. Three of these papers demonstrate three different
# mechanisms:
#   * Fu & Shi print "non-reasoning"; the extractor emits "nonreasoning", so a
#     correctly transcribed hyphenated word tokenises differently on each side.
#   * Sclar's small-caps FORMATSPREAD extracts as "f ormat s pread", because
#     the drop-cap glyph is a separate run.
#   * Nghiem's formula carries a Delta and a multiplication sign that survive in
#     a human transcription and vanish from the extraction.
# In all three the sentence is present and a reader quoted it correctly.
#
# So the last comparison drops spaces from both sides. For a passage of forty
# characters or more this is still decisive -- an accidental spaceless match of
# that length does not happen -- while being immune to every one of the above.
def despace(s: str) -> str:
    return s.replace(" ", "")


def subsequence_cover(quote: str, text_tokens: list[str],
                      index: dict[str, list[int]]) -> float:
    """Best share of `quote`'s tokens found in order in the text."""
    q = quote.split()
    if not q:
        return 0.0
    starts = index.get(q[0], [])[:400]
    budget = int(GAP_BUDGET * len(q)) + 8
    best = 0.0
    for s in starts:
        i, matched, gap = s, 0, 0
        for tok in q:
            found = False
            while i < len(text_tokens) and gap <= budget:
                if text_tokens[i] == tok:
                    matched += 1
                    i += 1
                    found = True
                    break
                i += 1
                gap += 1
            if not found:
                break
        best = max(best, matched / len(q))
        if best >= 1.0:
            break
    return best


# Where the paper's own text stops and other people's begins. B&M's extraction
# carries a thousand-entry citing-articles list, which is how a claim that the
# paper never says "ethics" collides with the Journal of Business Ethics, and
# "language model" with a 2023 paper on text mining. A negative search means
# "not in this paper", not "not in anything ever printed near it".
#
# TWO MARKERS, BECAUSE ONE IS NOT ENOUGH. A reference list is usually announced
# by a heading; a publisher's citing-articles appendix is not, and B&M's is
# recognisable only by the "crossref" token repeated once per entry.
REFS = re.compile(r"\b(references|bibliography|works cited|literature cited)\b")
CITED_BY = re.compile(r"crossref")


def body_of(text: str) -> str:
    """Everything before the reference list, when one is identifiable."""
    cuts = [m.start() for m in REFS.finditer(text) if m.start() > 0.35 * len(text)]
    xr = [m.start() for m in CITED_BY.finditer(text)]
    if len(xr) >= 20:
        cuts.append(xr[0])
    return text[:min(cuts)] if cuts else text


# A SOURCE WHOSE EXTRACTION IS TOO DAMAGED TO VERIFY AGAINST.
#
# Measured, not asserted. Two signals, both objective:
#   * digit-full-stop ligatures, which mean the PDF's font maps "8." to one
#     glyph and every number in the document is unrecoverable without folding;
#   * column interleaving, estimated by how often a sentence-terminal token is
#     followed by a lower-case continuation -- the signature of a two-column
#     reader splicing the right column into the middle of the left one.
#
# Quotes from a degraded source are reported as UNVERIFIABLE rather than as
# missing. Calling them missing would be a false accusation against the reader
# who transcribed them correctly, and it is what buried the real findings on
# the first three runs of this audit.
def degradation(raw: str) -> dict:
    ligs = len(DIGIT_LIG.findall(raw))
    words = raw.split()
    splices = sum(1 for a, b in zip(words, words[1:])
                  if a.endswith(".") and b[:1].islower() and len(a) > 3)
    return {"ligatures": ligs, "splices": splices,
            "degraded": ligs >= 100 or splices > 0.02 * max(len(words), 1)}


def load_texts() -> dict[str, tuple[str, str, str]]:
    """label -> (body_kw, full_kw, full_sentence).

    THREE, BECAUSE THE TWO JOBS NEED DIFFERENT NORMALISATIONS. A negative search
    is about a token with its punctuation intact ("gpt-3.5-turbo-"), and runs
    over the body so a publisher's citing-articles list cannot falsify it. A
    quotation is about a sentence, where punctuation is noise and the whole
    document is fair game.
    """
    out = {}
    for label, files in SOURCES.items():
        parts = []
        for f in files:
            p = TXT / f
            if p.exists():
                parts.append(p.read_text(encoding="utf-8", errors="replace"))
        if parts:
            raw = " ".join(parts)
            full = norm(raw)
            DEGRADED[label] = degradation(raw)
            kw = norm_kw(raw)
            out[label] = (body_of(kw), kw, full)
    return out


def studies() -> list[dict]:
    m = json.loads(MATRIX.read_text(encoding="utf-8"))
    out = list(m["studies"])
    if CANDIDATE.exists():
        c = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        out.append({"label": c["label"], "kind": c["kind"],
                    "cells": {k: {"verdict": v["verdict"], "value": v["value"],
                                  "evidence": (v.get("quote", "") + " " +
                                               " ".join(f"'{n}' 0 hits"
                                                        for n in v.get("negative", [])))}
                              for k, v in c["cells"].items()}})
    return out


def negatives(evidence: str) -> set[str]:
    """Every term the evidence claims returns zero hits."""
    terms = {m.group(1) for m in NEG_NEAR.finditer(evidence)}
    for m in NEG_LIST.finditer(evidence):
        terms |= set(re.findall(r"['‘“\"]([^'’”\"]+)"
                                r"['’”\"]", m.group(1)))
    return {t.strip().lower() for t in terms
            if len(t.strip()) >= TOO_SHORT and not t.strip().isdigit()}


def main() -> int:
    strict = "--strict" in sys.argv
    texts = load_texts()
    missing_src = sorted(set(SOURCES) - set(texts))

    print("=" * 78)
    print("MATRIX EVIDENCE  --  re-running the survey's claims about other papers")
    print("=" * 78)
    if missing_src:
        print(f"\n  [warn] no extracted text for: {', '.join(missing_src)}")

    n_neg = n_quote = n_footnote = n_interleaved = n_commentary = 0
    n_despaced = 0
    neg_fail, quote_fail, empty, refs_only, unverifiable = [], [], [], [], []
    # Two exclusions the headline count used to hide. `benign_hits`
    # are searches that DO return a body hit and are exempted by name
    # in BENIGN while staying in the denominator; `unparsed` are
    # not-reported cells whose evidence negatives() cannot parse, so
    # nothing from them is ever re-run. Both are now reported.
    benign_hits, unparsed = [], []

    # Token index per study, built once: the subsequence test is O(quote x
    # starts) and there are several hundred quotes.
    tokens = {}
    for label, (_b, _kw, full) in texts.items():
        ft = full.split()
        idx: dict[str, list[int]] = {}
        for i, w in enumerate(ft):
            idx.setdefault(w, []).append(i)
        tokens[label] = (ft, idx)

    for s in studies():
        label = s["label"]
        text = texts.get(label)
        for field, c in s["cells"].items():
            ev = str(c.get("evidence") or "")
            val = str(c.get("value") or "")
            verdict = c.get("verdict")

            if verdict in ("reported", "partial") and not ev.strip() \
                    and not val.strip():
                empty.append((label, field, verdict))

            if not text:
                continue

            body, full_kw, full = text
            ftok, findex = tokens[label]
            terms = sorted(negatives(ev))
            n_quote_before = n_quote
            for term in terms:
                n_neg += 1
                k = (label, field, term)
                # A LEADING WORD BOUNDARY, BUT NO TRAILING ONE. Several of these
                # searches are deliberately stems -- 'quantiz' is meant to catch
                # quantize/quantized/quantization -- so a trailing boundary would
                # break them. A leading one stops 'holm' matching "Kiviholma"
                # and 'ethic' matching a surname, which is where the noise was.
                rx = re.compile(r"\b" + re.escape(norm_kw(term)))
                nb = len(rx.findall(body))
                nf = len(rx.findall(full_kw))
                if k in BENIGN:
                    # Still exempt, but no longer invisible: a hit that is
                    # adjudicated away is a hit that happened.
                    if nb:
                        benign_hits.append((nb, label, field, term))
                    continue
                if nb:
                    i = rx.search(body).start()
                    neg_fail.append((nb, label, field, term,
                                     body[max(0, i - 70):i + 70]))
                elif nf:
                    refs_only.append((nf, label, field, term))

            for q in QUOTED.findall(ev):
                qn = norm(q)
                # Elisions and editorial insertions make a quote unmatchable by
                # construction; those are the reader's shorthand, not a claim.
                if len(qn.split()) < 6 or "…" in q or "..." in q or "[" in q:
                    continue
                if COMMENTARY.search(q):
                    n_commentary += 1
                    continue
                n_quote += 1
                if qn in full:
                    continue
                if defootnote(qn) in defootnote(full):
                    n_footnote += 1
                    continue
                if despace(qn) in despace(full):
                    n_despaced += 1
                    continue
                cover = subsequence_cover(qn, ftok, findex)
                if cover >= MIN_COVER:
                    n_interleaved += 1
                    continue
                if DEGRADED.get(label, {}).get("degraded"):
                    unverifiable.append((label, field, cover, qn[:110]))
                    continue
                quote_fail.append((label, field, cover, qn[:110]))

            # UNCHECKED CELLS. A not-reported cell leaves this script's reach
            # only if it offers neither a search negatives() can parse nor a
            # quotation that survives the quote filters. Counting cells with
            # no parsable search alone overstates it: a long verbatim
            # quotation is evidence, and the loop above does re-check it.
            # These cells lower n_neg rather than raising a flag, which is
            # why the coverage claim has to name them rather than round them
            # into a clean total.
            if (verdict == "not-reported" and not terms
                    and n_quote == n_quote_before):
                unparsed.append((label, field))

    print(f"\n  negative searches re-run : {n_neg}")
    print(f"  verbatim quotes checked  : {n_quote}"
          + (f"  ({n_footnote} after footnote markers were stripped, "
             f"{n_despaced} after extraction spacing was normalised, "
             f"{n_interleaved} as in-order subsequences through interleaved "
             f"page furniture)"
             if n_footnote or n_interleaved or n_despaced else ""))
    deg = sorted(k for k, v in DEGRADED.items() if v["degraded"])
    if deg:
        print("\n  SOURCES TOO DAMAGED TO VERIFY AGAINST")
        for k in deg:
            d = DEGRADED[k]
            print(f"    {k:<30} {d['ligatures']:>6} digit ligatures, "
                  f"{d['splices']:>6} column splices")
        print(f"    {len(unverifiable)} quote(s) from these are reported as "
              f"unverifiable, not missing.")
    if n_commentary:
        print(f"  quoted passages that are the reader's own commentary rather "
              f"than a quotation: {n_commentary} — not checked")
    if refs_only:
        print(f"  terms found ONLY in a reference list, not the body: "
              f"{len(refs_only)} — not counted against the claim")
    print(f"  cells with a positive verdict and no evidence : {len(empty)}")

    if empty:
        print("\n  UNSUPPORTED VERDICTS")
        for label, field, v in empty[:20]:
            print(f"    {label:<26} {field:<30} {v}")

    if quote_fail:
        print(f"\n  QUOTES NOT FOUND IN THE PAPER THEY ARE ATTRIBUTED TO "
              f"({len(quote_fail)})")
        quote_fail.sort(key=lambda r: r[2])
        for label, field, cover, q in quote_fail[:25]:
            print(f"    {cover:>5.0%} of tokens found  {label:<24} {field:<26}")
            print(f"           {q}")

    if neg_fail:
        neg_fail.sort(reverse=True)
        print(f"\n  NEGATIVE SEARCHES THAT NOW RETURN HITS ({len(neg_fail)})")
        print("  Ranked by hit count. A hit is not automatically a wrong claim "
              "-- it may sit\n  in a reference list, or the reader may have "
              "scoped the search to a section --\n  but each needs a human "
              "before the cell can be trusted.\n")
        for hits, label, field, term, ctx in neg_fail[:40]:
            print(f"    {hits:>4}x  {label:<24} {field:<28} {term!r}")
            print(f"           …{ctx.strip()}…")
        if len(neg_fail) > 40:
            print(f"    …and {len(neg_fail) - 40} more")

    # THE PAPER INTERPOLATES THESE. §8.1 claims the survey's negative findings
    # are re-verified rather than asserted, and a claim like that has to be
    # backed by a number the build produced, not by a sentence about diligence.
    # WRITE ONLY ON CHANGE. This artifact is deterministic, but rewriting it
    # unconditionally moved its mtime on every run -- and the supplementary
    # archive refuses to build against sources newer than itself. So merely
    # RUNNING the test suite made the zip stale and failed two fork tests,
    # which reads as a real regression and is not one. An unchanged result
    # should leave the filesystem alone.
    payload = json.dumps({
        "_what": "Every claim the reporting matrix makes about another paper, "
                 "re-checked against that paper's extracted full text.",
        "_why": "A wrong negative claim about someone else's reporting is the "
                "same class of error this paper is about. Asserting the "
                "codings were careful is not evidence; re-running them is.",
        "_method": "Negative searches are re-run with a leading word boundary "
                   "over the body, excluding reference lists. Quotations are "
                   "matched verbatim, then after footnote markers are removed, "
                   "then with extraction spacing normalised, then as in-order "
                   "token subsequences through interleaved page furniture.",
        "n_studies": len(SOURCES),
        "n_negative_searches_rerun": n_neg,
        "n_negative_searches_now_hitting": len(neg_fail),
        "n_terms_only_in_a_reference_list": len(refs_only),
        "n_negative_searches_hitting_but_adjudicated_benign":
            len(benign_hits),
        "n_not_reported_cells_with_no_parsable_search":
            len(unparsed),
        "n_quotations_checked": n_quote,
        "n_quotations_verified": n_quote - len(quote_fail) - len(unverifiable),
        "n_quotations_unverifiable_damaged_source": len(unverifiable),
        "n_cells_with_a_positive_verdict_and_no_evidence": len(empty),
        "degraded_sources": {k: v for k, v in DEGRADED.items()
                             if v.get("degraded")},
    }, indent=1)
    unchanged = (OUT.exists()
                 and OUT.read_text(encoding="utf-8") == payload)
    if not unchanged:
        OUT.write_text(payload, encoding="utf-8")

    bad = len(neg_fail) + len(quote_fail) + len(empty)
    print(f"\n  {'unchanged' if unchanged else 'wrote'} "
          f"{OUT.relative_to(ROOT)}")
    print("\n" + "=" * 78)
    if not bad:
        print("  every negative search still returns nothing; every quote is "
              "in its paper")
        return 0
    print(f"  {bad} item(s) need a human. Record each in BENIGN with a reason, "
          f"or fix the cell.")
    return 1 if strict else 0


if __name__ == "__main__":
    sys.exit(main())
