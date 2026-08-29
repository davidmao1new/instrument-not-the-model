r"""Score a document against how published academic prose actually reads.

`style_corpus.py` measures 24 published papers -- Bertrand and Mullainathan in
the AER, Benjamini and Hochberg in JRSS-B, and 22 others already on disk because
every citation in this project was checked against full text. This scores a
target document against those baselines and names what to change.

    sh paper-a/src/_py.sh paper-a/src/audit_ai_tells.py <file.md>
    sh paper-a/src/_py.sh paper-a/src/audit_ai_tells.py --all

A feature over the corpus p90 is flagged; over the corpus maximum it is a hard
failure, because no published paper in the corpus writes that way.

TWO THRESHOLDS ARE NOT FROM THE CORPUS, and the reason is worth stating. The
corpus is text extracted from PDFs, so running headers, journal furniture and
table fragments survive as capitalised strings; its `shout_caps` baseline is
inflated by extraction rather than by authors shouting. A markdown file written
by hand has no such contamination, so it is held to a flat, stricter budget.
`triad` is likewise inflated by author lists and reference strings. Both are
marked BUDGET rather than CORPUS in the output, so nothing here pretends to a
provenance it does not have.

Exit code 1 if any feature exceeds its hard limit.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style_corpus import measure  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROFILE = ROOT / "paper-a" / "data" / "reference" / "style_profile.json"

# Hand-set budgets where the corpus baseline is contaminated by PDF extraction.
BUDGET = {"shout_caps": (0.5, 2.0), "triad": (6.0, 9.0)}

WHY = {
    "em_dash": "commas, parentheses and full stops; keep the dash for one real "
               "aside per page",
    "shout_caps": "capitals for emphasis are not used in published prose -- "
                  "recast the sentence so the emphasis falls on the last word",
    "not_x_but_y": "state the thing; the antithesis frame is a machine tic and "
                   "the corpus median is zero",
    "reveal_colon": "a colon should introduce a list or a quotation, not a "
                    "punchline",
    "hedge_stack": "one hedge or none",
    "triad": "two items, or a list set as a list",
    "discourse_opener": "cut it; if the sentence needs Crucially to matter, it "
                        "does not matter",
    "sent_len_sd": "vary the sentence lengths; uniform middling sentences "
                   "are the clearest signature of machine prose",
    # Wikipedia's "Signs of AI writing", via the humanizer skill.
    "ai_words": "swap the stock word for the specific one: delve, crucial, "
                "pivotal, tapestry, testament, underscore, showcase, leverage",
    "filler": "cut it; 'it is important to note that X' says no more than 'X'",
    "announce": "make the point instead of announcing it. The corpus median "
                "for this is zero: published papers do not say what they are "
                "about to say",
    "saying": "state the claim. No paper in the corpus uses this construction "
              "even once",
    "qualifier": "one qualifier or none",
    "fragment_run": "two or more very short sentences in a row reads as a "
                    "forced punchline. One is emphasis; a row of them is a tic",
}
LOWER_IS_WORSE = {"sent_len_sd"}   # burstiness: too little of it is the tell

# BURSTINESS IS A GENRE PROPERTY, so it is not checked on correspondence.
#
# The baseline comes from research papers, where a long methodological sentence
# sits next to a short assertion and the spread is naturally wide. A 170-word
# email has neither the room nor the reason. Measured on the drafts: mean
# sentence length 12.8 words against the corpus 22.9, with 33 sentences under
# eight words. They are not uniform-and-middling, which is the thing the
# feature exists to catch; they are short, because they are emails.
#
# Holding one genre to another genre's baseline is a measurement error, and the
# wrong repair would have been to lower the threshold until the prose passed.
# There is no corpus of published human correspondence here to calibrate
# against, so the check is skipped and labelled rather than fudged.
# The per-thread reply files are named after the people who sent private
# replies, so listing them here would publish who corresponded. The names
# live in a file this repository does not ship (same mechanism as
# TARGET_LIST below); this source keeps only the generic campaign files.
# test_outreach_numbers.py asserts every reply file on disk is registered.
CORRESPONDENCE = {"FOLLOWUPS.md", "EMAILS_TO_SEND.md", "EMAILS.md"}
_CORR_LIST = (pathlib.Path(__file__).resolve().parents[2] / "paper-a" / "data"
              / "reference" / "correspondence.local.txt")
if _CORR_LIST.exists():
    CORRESPONDENCE |= {ln.strip() for ln in
                       _CORR_LIST.read_text(encoding="utf-8").splitlines()
                       if ln.strip() and not ln.startswith("#")}

# The documents scored by --all live in a file this repository does not
# publish. Naming them inline would put those paths into a public source
# file, and the release gate refused the build when they were written here.
TARGET_LIST = (ROOT / "paper-a" / "data" / "reference"
               / "prose_targets.local.txt")


def targets() -> list[str]:
    if not TARGET_LIST.exists():
        return []
    return [ln.split("#")[0].strip()
            for ln in TARGET_LIST.read_text(encoding="utf-8").split('\n')
            if ln.split("#")[0].strip()]


def strip_markdown(text: str) -> str:
    # Horizontal rules first. "---" is a markdown rule, and the em-dash
    # pattern matches its second and third hyphens, so every section break in
    # a file counted as an em dash. Twelve of them in one draft.
    text = re.sub(r"(?m)^\s{0,3}([-*_])(?:\s*\1){2,}\s*$", " ", text)
    # Bracketed all-capitals placeholders such as [ARXIV] are template tokens
    # the author replaces before sending, not emphasis.
    text = re.sub(r"\[[A-Z][A-Z0-9_]{2,}\]", " ", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"(?m)^\s*\|.*\|\s*$", " ", text)      # tables
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s.*$", " ", text)  # headings
    # LIST MARKERS AND LEAD-IN LABELS. A brief written as "3. **Tokenization
    # and names.** An and Rudinger. One sentence, then point at the companion."
    # is a labelled list, and the label is a heading by another name. Left in,
    # it reads to the sentence splitter as a run of two-word sentences and the
    # forced-punchline check fires on the document's structure rather than on
    # its prose.
    text = re.sub(r"(?m)^\s{0,6}(?:\d+[.)]|[-*+])\s+", "", text)
    text = re.sub(r"(?m)^\s*\*\*[^*\n]{2,60}?\.?\*\*\s*", "", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    return text


def shared_formulas(raw: str, n: int = 7, floor: int = 0) -> list[tuple[int, str]]:
    """Phrases of n words appearing at the head or foot of many drafts.

    Restricted to the first and last 45 words of each draft, because that is
    where a template shows: the greeting-and-hook and the sign-off. A phrase
    repeated in the middle of several drafts is usually the finding itself,
    which is supposed to be the same in every one of them.
    """
    drafts = re.split(r"(?m)^# \d+ [-\u2013\u2014] ", raw)[1:]
    if len(drafts) < 4:
        return []
    # PROPORTIONAL, not a fixed count. Four drafts out of nine sharing a closing
    # formula is a template; four out of twenty-three is coincidence, and a
    # fixed floor calls both the same thing. More than a third is the line.
    floor = floor or max(4, len(drafts) // 3 + 1)
    seen: dict[str, set[int]] = {}
    for i, d in enumerate(drafts):
        # Stop at the sign-off. A signature block is identical by design, and
        # so is the salutation; neither is the template the check is after.
        cut = re.search(r"(?m)^\s*(?:Best|Regards|Sincerely|Thanks)\b", d)
        body = strip_markdown(d[:cut.start()] if cut else d)
        w = body.split()
        edges = w[:45] + w[-45:]
        for j in range(len(edges) - n):
            g = " ".join(edges[j:j + n]).lower().strip(" .,;:")
            if len(g) > 24:
                seen.setdefault(g, set()).add(i)
    # The competing-interest disclosure is required to be the same in every
    # first-contact email. A disclosure that varies from letter to letter is a
    # worse document, not a more human one, so it is not a template finding.
    # A sender's own name and status are not a template. They are who he is,
    # and varying them across letters would be worse writing, not better. What
    # the check is after is the sentence built AROUND them repeating.
    # WHO THE SENDER IS, in whatever form he writes it. A person introduces
    # himself the same way in every letter, and rewording it from letter to
    # letter to satisfy a checker reads as someone trying not to repeat
    # himself. What the check is for is the sentence built AROUND the
    # introduction repeating, and that is measured on what follows it.
    # THE AUTHOR'S OWN OPENER, which he wrote and asked to have used verbatim
    # in every letter. The humanizer skill states the rule this follows: a
    # writing sample from the author takes priority over the style rules, and
    # a deliberate repeated opening is not a template. Flagging it would mean
    # a checker overruling the person whose name is on the email.
    REQUIRED = ("david mao", "high-school senior", "high school student",
                "from pennsylvania", "recently been reading",
                "measurement validity", "audit reproducibility",
                "founding team", "recruiting software", "commercial interest",
                "company that builds", "builds ai recruiting", "none of its",
                "recruiting startup", "founding team at an", "industry this audits",
                "nothing of theirs", "systems were audited")
    hits = [(len(v), k) for k, v in seen.items()
            if len(v) >= floor and not any(r in k for r in REQUIRED)]
    hits.sort(reverse=True)
    # Keep the longest phrase of each overlapping family.
    out: list[tuple[int, str]] = []
    for cnt, g in hits:
        if not any(g in o for _, o in out):
            out.append((cnt, g))
    return out[:6]


def limits() -> dict[str, tuple[float, float]]:
    prof = json.loads(PROFILE.read_text(encoding="utf-8"))["per_1000_words"]
    out: dict[str, tuple[float, float]] = {}
    for k, v in prof.items():
        if k in BUDGET:
            out[k] = BUDGET[k]
        elif k == "sent_len_sd":
            out[k] = (v["median"] * 0.75, v["median"] * 0.55)
        elif k == "sent_len_mean":
            continue
        else:
            # A floor, because several of these appear zero times in all 24
            # papers. Without it a single "crucial" in a 400-word note would
            # be a hard failure, and a checker that fires on one word gets
            # switched off.
            out[k] = (max(v["p90"], 0.2), max(v["max"], 0.4))
    return out


def audit(path: pathlib.Path, lim: dict) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    got = measure(strip_markdown(raw))
    words = int(got["n_words"])
    print(f"\n  {path.relative_to(ROOT)}   ({words:,} words)")
    if words < 200:
        print("    too short to score")
        return []

    hard: list[str] = []
    for k, (warn, fail) in sorted(lim.items()):
        if k == "sent_len_sd" and path.name in CORRESPONDENCE:
            print(f"    skip  {k:<18}  correspondence; the baseline is drawn "
                  f"from papers")
            continue
        v = got.get(k, 0.0)
        src = "BUDGET" if k in BUDGET else "corpus"
        if k in LOWER_IS_WORSE:
            bad, worst = v < warn, v < fail
            rel = f"{v:6.2f}  (want >= {warn:.2f}, {src})"
        else:
            bad, worst = v > warn, v > fail
            rel = f"{v:6.2f}  (limit {warn:.2f}, {src})"
        if worst:
            print(f"    FAIL  {k:<18}{rel}")
            print(f"          -> {WHY.get(k, '')}")
            hard.append(f"{path.name}: {k}")
        elif bad:
            print(f"    warn  {k:<18}{rel}")
            print(f"          -> {WHY.get(k, '')}")
    shared = shared_formulas(raw)
    if shared:
        print(f"    FAIL  {'shared formula':<18}{shared[0][0]} drafts share an "
              f"opening or closing phrase")
        for cnt, g in shared:
            print(f"          {cnt}x  \u201c{g}\u201d")
        print("          -> a formula repeated across drafts is the template "
              "showing through;")
        print("             vary it, or cut it from the drafts that do not "
              "need it")
        hard.append(f"{path.name}: shared formula")
    if not hard:
        print("    within the range published prose occupies")
    return hard


def main() -> int:
    if not PROFILE.exists():
        print("  no style profile. Run:")
        print("    sh paper-a/src/_py.sh paper-a/src/style_corpus.py --write")
        return 1
    lim = limits()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    paths = ([ROOT / t for t in targets()] if "--all" in sys.argv or not args
             else [pathlib.Path(a) if pathlib.Path(a).is_absolute()
                   else ROOT / a for a in args])

    print("=" * 74)
    print("AI-TELL AUDIT  --  against 24 published papers, 220,264 words")
    print("=" * 74)
    hard: list[str] = []
    for p in paths:
        if p.exists():
            hard += audit(p, lim)
        else:
            print(f"\n  {p} does not exist")
    print("\n" + "-" * 74)
    if hard:
        print(f"  {len(hard)} feature(s) outside anything in the corpus:")
        for h in hard:
            print(f"    - {h}")
        return 1
    print("  every document reads within the range the corpus occupies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
