"""The two-page outreach summary, as a PDF, with its numbers interpolated.

WHY THIS IS GENERATED. The first version of this summary was written by hand
and claimed 220,000 matched pairs against an artifact holding 31,468 -- a
seven-fold overstatement that would have gone out in ten cold emails to
researchers who audit measurement claims for a living. The paper has an
apparatus specifically to prevent that and the summary sat outside it.

So the summary reads B.EXPORT, which build_paper_v3.main() fills with the same
locals the paper interpolates. A number here cannot disagree with the paper
unless the paper is wrong.

THE PROSE IS DAVID'S. He rewrote it on 2026-08-13 and this file carries his
sentences verbatim, with every measurement in them replaced by the artifact it
came from. All fifteen matched what he had typed. Two did not survive as typed
and both are noted at the point of use: the test count, which was 355 when he
wrote it and moves whenever the suite grows, and the companion paper's figures,
which are read from paper-c's own released abstract rather than restated here.

If you edit the prose, edit it HERE. A number written as a literal in this file
is the exact failure the apparatus exists to catch.

    sh paper-a/src/_py.sh paper-a/src/build_summary.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as rl_canvas

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_paper_v3 as B  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

OUT = B.ROOT / "outreach" / "Mao_summary.pdf"
COMPANION = B.ROOT / "paper-c" / "releases" / "abstract_arxiv.txt"

PAGE_W, PAGE_H = LETTER
MX, MTOP, MBOT = 1.05 * inch, 0.95 * inch, 0.85 * inch
COL_W = PAGE_W - 2 * MX

BODY, LEAD = 9.8, 13.6
R, BLD, IT = "Times-Roman", "Times-Bold", "Times-Italic"
INK = (0.09, 0.09, 0.10)
MUTE = (0.42, 0.42, 0.45)


def pct(x, n=0):
    return "--" if x is None else f"{100 * x:.{n}f} %"


def word(n):
    """A small count as a word, the way the paper writes them."""
    return B.NUM.get(n, str(n))


def companion(pattern: str, what: str) -> tuple[str, ...]:
    """A number from the COMPANION paper's own released abstract.

    Paper A's EXPORT does not hold paper C's counts, and typing them here would
    be the 220,000 mistake with a different denominator. paper-c writes its
    abstract from its own artifacts at build time, so reading it back is the
    same guarantee one file further away. A pattern that stops matching is
    fatal rather than skipped: a silently dropped sentence in an outreach
    document is worse than a build that stops.
    """
    if not COMPANION.exists():
        sys.exit(f"{COMPANION.relative_to(B.ROOT)} missing -- build paper C "
                 f"first: sh paper-a/src/_py.sh paper-c/src/build_paper_c.py")
    m = re.search(pattern, COMPANION.read_text(encoding="utf-8"))
    if not m:
        sys.exit(f"paper C's abstract no longer states {what}. The summary "
                 f"claims it. Reconcile the two before sending this to anyone.")
    return m.groups()


class Doc:
    def __init__(self, path):
        self.c = rl_canvas.Canvas(str(path), pagesize=LETTER)
        self.y = PAGE_H - MTOP
        self.page = 1

    def _room(self, h):
        if self.y - h < MBOT:
            self.c.setFont(R, 8)
            self.c.setFillColorRGB(*MUTE)
            self.c.drawCentredString(PAGE_W / 2, MBOT - 22, str(self.page))
            self.c.showPage()
            self.page += 1
            self.y = PAGE_H - MTOP

    # ---- inline runs, so a paper title can be italic mid-sentence.
    # Markup is a single asterisk pair. Wrapping has to measure per run, since
    # Times-Italic and Times-Roman are not the same width.
    @staticmethod
    def _runs(text, font):
        out, ital = [], False
        for piece in text.split("*"):
            if piece:
                out.append((piece, IT if ital else font))
            ital = not ital
        return out

    def para(self, text, size=BODY, font=R, lead=LEAD, after=7.0, colour=INK):
        # Empty strings are dropped rather than laid out. A run boundary at a
        # space -- "*Title* shows" -- splits into a chunk ending and a chunk
        # beginning with the same space, and laying both out puts a visible
        # double space after every italic phrase.
        words = [(w, f)
                 for chunk, f in self._runs(text, font)
                 for w in chunk.split(" ") if w]
        lines, cur, cw = [], [], 0.0
        space = pdfmetrics.stringWidth(" ", font, size)
        for w, f in words:
            ww = pdfmetrics.stringWidth(w, f, size)
            if cur and cw + space + ww > COL_W:
                lines.append(cur)
                cur, cw = [(w, f)], ww
            else:
                cw += (space if cur else 0) + ww
                cur.append((w, f))
        if cur:
            lines.append(cur)
        self._room(min(len(lines), 3) * lead)
        self.c.setFillColorRGB(*colour)
        for ln in lines:
            self._room(lead)
            x = MX
            for i, (w, f) in enumerate(ln):
                self.c.setFont(f, size)
                if i:
                    x += space
                self.c.drawString(x, self.y - size, w)
                x += pdfmetrics.stringWidth(w, f, size)
            self.y -= lead
        self.y -= after

    def head(self, text, size=11.2, before=10.0):
        self.y -= before
        self._room(size * 1.5 + 2 * LEAD)
        self.c.setFillColorRGB(*INK)
        self.c.setFont(BLD, size)
        self.c.drawString(MX, self.y - size, text)
        self.y -= size * 1.55

    def rule(self, after=8.0):
        self._room(6)
        self.c.setStrokeColorRGB(0.80, 0.80, 0.83)
        self.c.setLineWidth(0.5)
        self.c.line(MX, self.y - 2, PAGE_W - MX, self.y - 2)
        self.y -= after

    def save(self):
        self.c.setFont(R, 8)
        self.c.setFillColorRGB(*MUTE)
        self.c.drawCentredString(PAGE_W / 2, MBOT - 22, str(self.page))
        self.c.save()


def main() -> int:
    B.main()
    E = B.EXPORT
    if not E:
        sys.exit("no EXPORT from build_paper_v3.main()")

    # THE OPERATING-POINT SENTENCE NEEDS BOTH NUMBERS. Quoting the 484 alone
    # -- a conversion-factor error at a 99.65 %-saturated operating point --
    # without the overstatement realised on a measured effect is what drew
    # "needs equations, absolute differences, and denominators" from the first
    # expert who read this. Both, in that order.
    _sc = json.loads((B.ROOT / "paper-a/data/delta_stability/reporting_scale.json")
                     .read_text(encoding="utf-8"))["summary"]
    # AND THE REVIEW CLAIM IS A LEDGER, NOT AN ADJECTIVE. "Seven rounds of
    # adversarial review" was unfalsifiable and also wrong: six rounds have
    # released artifacts. build_review_ledger.py counts them from disk.
    _LED = json.loads((B.ROOT / "paper-a/data/reference/review_ledger.json")
                      .read_text(encoding="utf-8"))
    _LED_ROUNDS = word(_LED["n_rounds_with_artifacts"]).capitalize()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = Doc(OUT)

    d.c.setFillColorRGB(*INK)
    d.c.setFont(BLD, 15.5)
    d.c.drawString(MX, d.y - 15.5, "The Instrument Is Not the Model")
    d.y -= 21
    d.c.setFont(R, 11)
    d.c.drawString(MX, d.y - 11,
                   "How much of an LLM hiring disparity comes from "
                   "unreported design choices")
    d.y -= 18
    d.c.setFont(R, 9)
    d.c.setFillColorRGB(*MUTE)
    d.c.drawString(MX, d.y - 9,
                   "David Mao  ·  Wissahickon High School, class of 2027  ·  "
                   "davidmao.xyz")
    d.y -= 13
    d.c.drawString(MX, d.y - 9, "Two-page summary.")
    d.y -= 16
    d.rule(10)

    d.head("Research question", before=2)
    d.para(
        "In recent years, organizations are significantly increasing their use "
        "of language models in their hiring processes. As a result, it has "
        "become crucial to audit the use of new technologies for bias and "
        "discrimination. For example, if you upload two résumés to the same "
        "model and only change the applicant's name, how differently do the "
        "models treat the two résumés?")
    d.para(
        "In New York City that difference carries legal weight. Local Law 144 "
        "makes it unlawful to use an automated employment decision tool that "
        "has not been audited for bias in the preceding year.")
    d.para(
        "The problem comes when that difference comes from dozens of choices "
        "nobody reports, such as which instruction wording was used, which "
        "names were drawn, which job was posted, which quantization was "
        "downloaded, how requests were batched, whether the intervals resample "
        "rows or name pairs. This paper holds the model fixed and varies only "
        "those choices.")

    d.head("Findings")
    d.para(
        f"Over {E['n_records']:,} matched pairs and {E['n_calls']:,} model "
        f"calls, on {word(E['n_panel'])} open-weight checkpoints and "
        f"{word(E['n_front'])} frontier APIs, I found that the instruction "
        f"wording moves the effect by {pct(E['ratio_lo'])} to "
        f"{pct(E['ratio_hi'])} of itself, across the {E['ratio_n']} "
        "model-by-posting cells where the effect is separable from zero. This "
        "included edits that did not change any word of the instruction at all "
        "and only its punctuation or line breaks.")
    d.para(
        "Choices made after the data is collected matter as much. Resampling "
        "rows rather than name pairs narrows intervals by "
        f"{E['resamp_lo']:.1f} to {E['resamp_hi']:.1f} times, because a first "
        "name appears in many rows and the rows are not independent. A fixed "
        "operating point misstates the log-odds-to-percentage-point conversion "
        f"by {E['jacobian_lo']:.1f} to {E['jacobian_hi']:.0f} times; on the "
        f"{word(_sc['n_distinguishable'])} checkpoints whose effect is "
        "distinguishable from zero, the overstatement actually incurred is "
        f"{_sc['realised_min']:.1f} to {_sc['realised_max']:.1f} times. Since "
        "each résumé is scored alone, the pairing itself is an analysis-time "
        "choice where re-pairing the same scores moves the statistic the field "
        "reports, while the mean paired difference stays algebraically "
        "invariant.")
    d.para(
        "The measurements reproduce bitwise only when request batching and "
        "key-value cache residency are controlled. These are usually described "
        "as deterministic at temperature zero, but the results show that they "
        "are not because of order dependence. No published audit reports any "
        f"of this. Of {E['n_audits']} LLM hiring audits read end to end, none "
        "fully reports its batching, cache policy or token matching, and none "
        f"of the {E['n_disp_applicable']} that vary the wording reports a "
        "dispersion statistic across them.")

    d.head("Recommendations")
    d.para(
        "A minimum reporting set including the wording set and the dispersion "
        "across it, the name list and its token statistics, the serving "
        "configuration, and the resampling unit, along with a screening rule "
        "for when a reported disparity should be trusted, whose false-positive "
        "rate is calibrated in closed form rather than asserted.")
    d.para(
        "An even stronger system is to report the mean paired difference, "
        "which is algebraically invariant to the pairing. This removes the "
        "auditor's degree of freedom when it comes to pairing rather than "
        "disclosing it.")

    # THE COMPANION'S NUMBERS COME FROM THE COMPANION. Paper A's EXPORT holds
    # the token-matched fractions because paper A computes them too, but the
    # "three of four" and the collapse to three name pairs are paper C's
    # results and are read from paper C's own released abstract.
    away, of = companion(r"further from zero on (\w+) of (\w+)",
                         "how many checkpoints the restriction moves")
    clusters, = companion(r"built from (\d+) independent name pairs",
                          "the size of the balanced grid")
    d.head("A companion paper")
    d.para(
        "*The Matched Pair Is Not Token-Matched* shows that a correspondence "
        "audit's central assumption — that the two prompts are identical apart "
        "from the name — is false in the units the model actually reads.")
    d.para(
        f"Only {pct(E['token_matched_lo'])} to {pct(E['token_matched_hi'])} of "
        "name pairs from the canonical Bertrand and Mullainathan list are "
        f"token-matched across {word(E['n_open_weight'])} tokenizers. "
        "Restricting to matched pairs moves the reported disparity further "
        f"from zero on {away} of {of} checkpoints. How far it moves the "
        "reported disparity cannot be determined because the token-matched "
        f"subset collapses to as few as {word(int(clusters))} independent "
        "first-name pairs, and three standard statistical procedures reach "
        "different conclusions about whether the result crosses the "
        "conventional 0.05 significance threshold. The paper reports all three "
        "standard statistical procedures that yield different conclusions for "
        "transparency.")

    # DERIVED, NOT TYPED. This said "490 tests" and the suite had already
    # grown past it inside a day; David's draft said 355 and the figure-port
    # tests had already taken it past that. Counting `def test_` understates
    # the collected total wherever a test is parametrised, so the prose says
    # "test functions", which is what is actually counted.
    n_tests = sum(
        f.read_text(encoding="utf-8").count("def test_")
        for f in (B.ROOT / "tests").glob("test_*.py"))

    d.head("How the work was checked")
    d.para(
        "Every number in both papers is interpolated from a released artifact "
        "at typesetting time and none is manually typed in prose. An audit "
        "script reads the typesetting source and fails the build if a "
        "measurement was typed into the prose rather than read from disk. A "
        "second audit checks the document against the artifacts for internal "
        "contradictions. A third fails if any artifact-guarded passage never "
        "fired, so a result cannot silently vanish. A fourth reads the "
        f"rendered geometry and fails if anything is drawn outside its column. "
        f"{n_tests} test functions pin claims the papers are not allowed to "
        f"make. {_LED_ROUNDS} rounds of adversarial critique raised "
        f"{_LED['totals']['examined']} findings against the paper's own "
        f"artifacts; {_LED['totals']['refuted']} did not survive a refutation "
        f"pass, {_LED['totals']['confirmed']} were confirmed, and "
        f"{_LED['totals']['serious']} of those were graded critical or major. "
        "A changelog records every claim that was amended and why, including "
        "several that were withdrawn.")

    d.head("What I am looking for")
    d.para(
        "I am a high-school senior and currently the sole author of this "
        "paper. The work is finished and self-checked, but it has not been "
        "read deeply by anyone in this field. I am looking for a senior "
        "collaborator who would critically evaluate it, tell me where it "
        "overreaches, and, if the work merits it, join the project as a "
        "co-author.")

    d.rule(8)
    d.para(
        "Competing interests. I am on the founding team at a company that "
        "builds AI recruiting software, which is a direct commercial interest "
        "in the class of system this paper audits. No system built by that "
        "company was audited and none of its data, models or internal "
        "documents was used; every claim about deployed practice is sourced to "
        "public documentation.",
        size=8.6, lead=11.6, colour=MUTE, after=5)
    d.para(
        "Generative AI. Large language models assisted in drafting parts of "
        "the prose of the current preprint, were used as tools for the "
        "analysis and typesetting code, and were run adversarially against the "
        "manuscript's own claims to audit the methodology and find errors in "
        "it. The research questions, the design, every experimental decision "
        "and every interpretation are mine, and I have verified each reported "
        "number against the artifact it is computed from.",
        size=8.6, lead=11.6, colour=MUTE, after=0)

    d.save()
    print(f"wrote {OUT.relative_to(B.ROOT)}  ({d.page} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
