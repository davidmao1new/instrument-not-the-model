r"""Measure how published academic prose actually reads, so "sounds AI-generated"
becomes a number instead of an opinion.

WHY. Told that a document "sounds AI-generated", the useless response is to
rewrite it by feel and declare it fixed -- the same judgement that produced the
problem is the one grading the repair. The corpus in `lit/text/` is 26 papers
and roughly 316,000 words of published, peer-reviewed, human-written prose,
including Bertrand and Mullainathan in the AER and Benjamini and Hochberg in
JRSS-B. It is already on disk because every citation in this project was
verified against full text. That makes it a usable baseline.

This module reads that corpus and reports, per 1,000 words, the frequency of
the constructions that separate machine prose from published prose. The
companion `audit_ai_tells.py` scores a target document against these baselines.

WHAT IS MEASURED, and why each one earns its place. Every threshold in
`audit_ai_tells.py` comes from running this over the corpus; none is invented.

  em_dash            LLMs use the em dash as a general-purpose joint, often two
                     or three to a sentence. Published prose uses it sparingly
                     and prefers commas, parentheses and full stops.
  shout_caps         Emphasis by capitalising a word (IN ENGLISH, NOT read).
                     Published prose sets emphasis in italics or, far more
                     often, by word order. Acronyms are excluded.
  discourse_opener   Sentences beginning Crucially, Importantly, Notably,
                     Indeed, Moreover, Furthermore, Ultimately.
  not_x_but_y        "not a X but a Y", "isn't just X, it's Y" -- the
                     antithesis frame, which machine prose reaches for
                     constantly and journals use once a paper.
  triad              Three parallel items in a row ("clear, concise and
                     simple"). One is fine; a rate is a tic.
  reveal_colon       A colon used for a dramatic reveal mid-sentence rather
                     than to introduce a list or a quotation.
  hedge_stack        Two or more hedges in one clause ("may potentially
                     suggest").
  sent_len_sd        Burstiness. Human paragraphs mix a four-word sentence
                     with a forty-word one; machine prose regresses to a
                     uniform middle.

    sh paper-a/src/_py.sh paper-a/src/style_corpus.py            # report
    sh paper-a/src/_py.sh paper-a/src/style_corpus.py --write    # save profile

The profile is written to paper-a/data/reference/style_profile.json. Only the
statistics are stored, never the corpus text, which is copyrighted and is why
lit/ is not redistributed.
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = ROOT / "lit" / "text"
OUT = ROOT / "paper-a" / "data" / "reference" / "style_profile.json"

OPENERS = ("crucially", "importantly", "notably", "indeed", "moreover",
           "furthermore", "ultimately", "critically", "essentially",
           "fundamentally")

RX = {
    "em_dash": re.compile(r"—|--(?!-)"),
    # Capitalised tokens. Telling emphasis from an acronym is done in
    # `measure`, not here: the first attempt required the token to contain a
    # vowel, which flags API and OSF as shouting and would have taught anyone
    # reading the report to ignore it.
    "shout_caps": re.compile(r"(?<![A-Za-z])[A-Z]{2,}(?![A-Za-z])"),
    "not_x_but_y": re.compile(
        r"\bnot\s+(?:a|an|the|just|only|merely|simply)\b[^.;]{2,60}?\bbut\b|"
        r"\bisn'?t\s+(?:just|only|merely)\b[^.;]{2,60}?,\s*it'?s\b",
        re.I),
    # A colon introducing a single clause. Colons that introduce a list are
    # ordinary punctuation and are excluded in `measure`.
    "reveal_colon": re.compile(r"[a-z]{3,}:\s+[a-z]"),
    "hedge_stack": re.compile(
        r"\b(?:may|might|could|can|appears?\s+to|seems?\s+to|suggests?)\b"
        r"[^.;]{0,24}?\b(?:potentially|possibly|perhaps|arguably|likely|"
        r"tentatively|somewhat)\b", re.I),
}

TRIAD = re.compile(r"\b\w+,\s+\w+(?:\s+\w+)?,?\s+and\s+\w+", re.I)

# ---------------------------------------------------------------------------
# Wikipedia's "Signs of AI writing", via the humanizer skill (MIT). Curated
# observations, not my guesses. Punctuation alone missed all of these.

# §7 overused AI words. Technical senses are excluded where they collide with
# real usage in this field: "key" as in key-value cache, "gate" as in margin
# gate, "highlight" is rare enough to keep.
AI_WORDS = re.compile(
    r"\b(?:delve|delves|delving|crucial|pivotal|tapestry|testament|"
    r"underscore[sd]?|showcase[sd]?|garner(?:ed|s)?|intricate|intricacies|"
    r"interplay|vibrant|enduring|foster(?:ing|ed|s)?|align(?:ing|s)? with|"
    r"landscape of|realm of|myriad|plethora|leverage[sd]?|utilise[sd]?|"
    r"utilize[sd]?|robustly|seamless(?:ly)?|holistic)\b", re.I)

# §23 filler, and §3 throat-clearing openers. "It is important to note that"
# and "In this section we discuss" say nothing the next clause does not.
FILLER = re.compile(
    r"\b(?:it is important to note that|it is worth noting that|"
    r"it should be noted that|in order to(?= \w)|due to the fact that|"
    r"at this point in time|in the event that|has the ability to|"
    r"it is interesting to note|needless to say|as previously mentioned)\b",
    re.I)

# §28 announcing the next point instead of making it.
ANNOUNCE = re.compile(
    r"(?:^|(?<=[.!?]) )(?:in this (?:section|paper|note|document)|"
    r"this (?:section|answers|covers|explains|describes) (?:answers|in order|"
    r"the|what|how)|let'?s (?:dive|explore|look|break)|here'?s what|"
    r"what follows is|before (?:we|I) (?:begin|start)|"
    r"the following (?:section|table) (?:describes|explains|covers))",
    re.I)

# The same tic in its clipped form: a noun-phrase fragment standing as a
# sentence purely to introduce the question after it. "One question. Does the
# flip rate hold?" announces; "Does the flip rate hold?" asks. Restricted to
# fragments that actually precede an interrogative, so a short sentence used
# for emphasis is not caught.
ANNOUNCE_FRAG = re.compile(
    r"(?:^|(?<=[.!?]) )(?:[A-Z][a-z]*\s+){0,4}"
    r"(?:question|ask|point|thing|follow-up|request|note)\.\s+"
    r"(?:Is|Are|Does|Do|Did|Can|Could|Would|Will|Was|Were|Has|Have|What|Why|"
    r"How|Which|Whether)\b")

# §32 formulaic sayings: "X is the Y of Z" and its relatives.
SAYING = re.compile(
    r"\bis the (?:language|currency|architecture|backbone|bedrock|cornerstone|"
    r"lifeblood) of\b|\bbecomes? a trap\b|\bnot a \w+ but a mirror\b", re.I)

# §24 stacked qualifiers.
QUALIFIER = re.compile(
    r"\b(?:to be fair|it'?s also possible|could potentially|might arguably|"
    r"in some cases it may|this is an inference|it could be argued)\b", re.I)


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p for p in parts if len(p.split()) >= 3]


def measure(text: str) -> dict[str, float]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    n = max(len(words), 1)
    per_k = 1000.0 / n

    out: dict[str, float] = {}
    for name, rx in RX.items():
        if name in ("shout_caps", "reveal_colon"):
            continue
        out[name] = len(rx.findall(text)) * per_k

    # SHOUTING vs ACRONYM, decided by the document itself. A word written in
    # capitals for emphasis is a word the author also writes in lower case
    # somewhere else: ENGLISH beside English, NOT beside not. A genuine acronym
    # never appears lower-cased. This needs no dictionary and no allow-list,
    # and it does not fire on API, OSF, DOI, OLS or CC-BY.
    # Built from words that are NOT already all-capitals. Including them makes
    # every capitalised token match its own lower-cased form, so OSF and DOI
    # flagged themselves as shouting.
    lower = {w.lower() for w in words if not w.isupper()}
    shouts = [m.group(0) for m in RX["shout_caps"].finditer(text)
              if m.group(0).lower() in lower and len(m.group(0)) >= 2]
    out["shout_caps"] = len(shouts) * per_k

    # A colon that introduces a list is ordinary. One that introduces a single
    # clause is the reveal. Look ahead to the end of the sentence: two or more
    # commas, or a semicolon, means a list.
    reveals = 0
    for m in RX["reveal_colon"].finditer(text):
        tail = text[m.end():m.end() + 160]
        stop = re.search(r"(?<![A-Z])\.\s", tail)
        clause = tail[:stop.start()] if stop else tail
        if clause.count(",") < 2 and ";" not in clause:
            reveals += 1
    out["reveal_colon"] = reveals * per_k

    sents = sentences(text)
    out["triad"] = len(TRIAD.findall(text)) * per_k
    for name, rx in (("ai_words", AI_WORDS), ("filler", FILLER),
                     ("saying", SAYING), ("qualifier", QUALIFIER)):
        out[name] = len(rx.findall(text)) * per_k
    out["announce"] = (len(ANNOUNCE.findall(text))
                       + len(ANNOUNCE_FRAG.findall(text))) * per_k

    # §31 forced punchlines. One short sentence is emphasis. Two or more in a
    # row is the tic, and "One question." sitting alone is the case that
    # prompted this. Counted as runs, not as short sentences, because a short
    # sentence on its own is exactly what good prose uses for emphasis.
    runs, cur = 0, 0
    for s in sents:
        if len(s.split()) <= 5:
            cur += 1
            if cur == 2:
                runs += 1
        else:
            cur = 0
    out["fragment_run"] = runs * per_k

    # Paragraph-length variation is a real tell, and it is NOT measured here.
    # The corpus is text extracted from PDFs, which arrives with its paragraph
    # breaks flattened, so every paper scores the same sentinel and there is no
    # baseline to compare a draft against. Shipping it would mean a column of
    # numbers with nothing behind them.
    out["discourse_opener"] = sum(
        1 for s in sents
        if s.split()[0].strip(",").lower() in OPENERS) * per_k

    lens = [len(s.split()) for s in sents]
    out["sent_len_mean"] = statistics.mean(lens) if lens else 0.0
    out["sent_len_sd"] = statistics.pstdev(lens) if len(lens) > 1 else 0.0
    out["n_words"] = float(len(words))
    return out


def clean(raw: str) -> str:
    """Drop the parts of an extracted PDF that are not prose."""
    lines = []
    for ln in raw.split("\n"):
        s = ln.strip()
        if not s or len(s) < 25:
            continue
        # reference lines, tables, headers, page furniture
        if re.match(r"^\[?\d+\]?[.)]?\s", s) and len(s) < 200:
            continue
        if sum(c.isdigit() for c in s) > len(s) * 0.25:
            continue
        if s.count("  ") > 3:
            continue
        lines.append(s)
    return " ".join(lines)


def corpus_profile() -> dict:
    if not CORPUS.exists():
        raise SystemExit(f"no corpus at {CORPUS}")
    per_paper: dict[str, dict[str, float]] = {}
    for f in sorted(CORPUS.glob("*.txt")):
        text = clean(f.read_text(encoding="utf-8", errors="replace"))
        if len(text.split()) < 1500:
            continue
        per_paper[f.stem] = measure(text)

    keys = [k for k in next(iter(per_paper.values())) if k != "n_words"]
    summary = {}
    for k in keys:
        vals = sorted(v[k] for v in per_paper.values())
        summary[k] = {
            "median": statistics.median(vals),
            "p90": vals[int(0.9 * (len(vals) - 1))],
            "max": vals[-1],
            "mean": statistics.mean(vals),
        }
    return {
        "n_papers": len(per_paper),
        "n_words": int(sum(v["n_words"] for v in per_paper.values())),
        "per_1000_words": summary,
        "papers": {k: {m: round(x, 3) for m, x in v.items()}
                   for k, v in per_paper.items()},
    }


def main() -> int:
    prof = corpus_profile()
    print("=" * 74)
    print("STYLE PROFILE  --  published, human-written academic prose")
    print("=" * 74)
    print(f"\n  {prof['n_papers']} papers, {prof['n_words']:,} words\n")
    print(f"  {'feature':<20}{'median':>9}{'mean':>9}{'p90':>9}{'max':>9}")
    print("  " + "-" * 54)
    for k, v in prof["per_1000_words"].items():
        print(f"  {k:<20}{v['median']:>9.2f}{v['mean']:>9.2f}"
              f"{v['p90']:>9.2f}{v['max']:>9.2f}")
    print("\n  Rates are per 1,000 words, except sent_len_*, which are words.")

    if "--write" in sys.argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(prof, indent=1), encoding="utf-8")
        print(f"\n  wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
