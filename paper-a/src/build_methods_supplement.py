"""A methods supplement an external reviewer asked for, from the artifacts.

WHY THIS EXISTS. Reviewing the two-page summary on 2026-08-19, the reviewer
listed four things needed before any of the claims could be assessed:

  1. "A design table for each analysis: treatment, outcome, assignment unit,
     observation unit, block, target population, estimator, and uncertainty
     estimator."
  2. "Information on models, prompts, names, matching rule(s), retained
     (excluded) pairs."
  3. "Full sample breakdown by model, posting, prompt, and analysis cell."
  4. "Evidence for the pairing and resampling claims, including formulas,
     pairing rules, simulations or randomisation tests, coverage, and
     supporting citations."

He was right that none of it was assembled anywhere, and a referee would ask for
the same four. This document is those four, in his order, and every number in it
is read from the artifact that produced it or from the experiment module that
defines it -- including the twelve prompt strings, which are printed in full
rather than described, because the paper's central claim is about exact strings
and a described prompt is not the prompt.

    sh paper-a/src/_py.sh paper-a/src/build_methods_supplement.py
"""
from __future__ import annotations

import json
import pathlib
import sys

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as rl_canvas

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "paper-a" / "data"
OUT = ROOT / "paper-a" / "releases" / "Mao_methods_supplement.pdf"

PAGE_W, PAGE_H = LETTER
MX, MTOP, MBOT = 0.95 * inch, 0.9 * inch, 0.8 * inch
COL_W = PAGE_W - 2 * MX
R, BLD, IT, MONO = "Times-Roman", "Times-Bold", "Times-Italic", "Courier"
INK, MUTE = (0.09, 0.09, 0.10), (0.42, 0.42, 0.45)


def j(rel: str):
    p = D / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


class Doc:
    def __init__(self, path):
        self.c = rl_canvas.Canvas(str(path), pagesize=LETTER)
        self.y = PAGE_H - MTOP
        self.page = 1

    def _foot(self):
        self.c.setFont(R, 8)
        self.c.setFillColorRGB(*MUTE)
        self.c.drawCentredString(PAGE_W / 2, MBOT - 24, str(self.page))

    def _room(self, h):
        if self.y - h < MBOT:
            self._foot()
            self.c.showPage()
            self.page += 1
            self.y = PAGE_H - MTOP

    def wrap(self, text, font, size, width):
        words, lines, cur = text.split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if pdfmetrics.stringWidth(t, font, size) <= width:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    def para(self, text, size=9.4, font=R, lead=12.6, after=6.0, colour=INK,
             indent=0.0):
        lines = self.wrap(text, font, size, COL_W - indent)
        self._room(min(len(lines), 3) * lead)
        self.c.setFillColorRGB(*colour)
        for ln in lines:
            self._room(lead)
            self.c.setFont(font, size)
            self.c.drawString(MX + indent, self.y - size, ln)
            self.y -= lead
        self.y -= after

    def head(self, text, size=12.5, before=13.0):
        self.y -= before
        self._room(size * 1.6 + 3 * 12.6)
        self.c.setFillColorRGB(*INK)
        self.c.setFont(BLD, size)
        self.c.drawString(MX, self.y - size, text)
        self.y -= size * 1.7

    def sub(self, text, size=10.4, before=8.0):
        self.y -= before
        self._room(size * 1.6 + 2 * 12.6)
        self.c.setFillColorRGB(*INK)
        self.c.setFont(BLD, size)
        self.c.drawString(MX, self.y - size, text)
        self.y -= size * 1.55

    def mono(self, text, size=7.6, lead=9.4, indent=14.0, after=4.0):
        for ln in self.wrap(text, MONO, size, COL_W - indent):
            self._room(lead)
            self.c.setFont(MONO, size)
            self.c.setFillColorRGB(*INK)
            self.c.drawString(MX + indent, self.y - size, ln)
            self.y -= lead
        self.y -= after

    def row(self, cells, widths, size=8.2, lead=10.4, bold=False, after=0.0):
        font = BLD if bold else R
        stacks = [self.wrap(str(c), font, size, w - 6) for c, w in
                  zip(cells, widths)]
        h = max(len(s) for s in stacks) * lead
        self._room(h + 2)
        y0 = self.y
        x = MX
        for stack, w in zip(stacks, widths):
            yy = y0
            for ln in stack:
                self.c.setFont(font, size)
                self.c.setFillColorRGB(*INK)
                self.c.drawString(x, yy - size, ln)
                yy -= lead
            x += w
        self.y = y0 - h - after

    def rule(self, after=5.0):
        self._room(5)
        self.c.setStrokeColorRGB(0.78, 0.78, 0.82)
        self.c.setLineWidth(0.5)
        self.c.line(MX, self.y - 2, PAGE_W - MX, self.y - 2)
        self.y -= after

    def save(self):
        self._foot()
        self.c.save()


def main() -> int:
    import build_paper_v3 as B          # SHORT/TINY label maps only
    import experiment_delta_stability as E
    import occupations as OC
    import stimuli as ST

    design = j("reference/design_table.json")
    s2 = j("delta_stability/study2_v2.json")
    resamp = j("delta_stability/resampling_unit.json")
    pairfree = j("names/pairing_freedom.json")
    tbal = j("instrument/token_balanced_grid.json")
    occ = j("occupation/occupation_analysis.json")
    front = j("frontier/frontier_margin_analysis.json")
    fverd = j("frontier/frontier_verdict_analysis.json")
    if not (design and s2 and resamp and pairfree):
        sys.exit("missing artifacts; build the paper first")

    models = [m for m in B.SHORT if m in s2]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = Doc(OUT)

    # ---- title -------------------------------------------------------
    d.c.setFillColorRGB(*INK)
    d.c.setFont(BLD, 15.0)
    d.c.drawString(MX, d.y - 15, "The Instrument Is Not the Model")
    d.y -= 20
    d.c.setFont(BLD, 11.5)
    d.c.drawString(MX, d.y - 11.5, "Methods supplement")
    d.y -= 17
    d.c.setFont(R, 9)
    d.c.setFillColorRGB(*MUTE)
    d.c.drawString(MX, d.y - 9, "David Mao  ·  davidmao.xyz")
    d.y -= 15
    d.rule(9)

    # ================================================================
    d.head("1.  Design table, one row per analysis", before=6)
    d.para(
        "One row per analysis. The target-population column is worth reading "
        "first, because almost nothing here generalises to a "
        "population of applicants, it generalises to the instrument's own "
        "design space, and that is the argument rather than a caveat.")
    FIELDS = [("treatment", "Treatment"), ("outcome", "Outcome"),
              ("assignment", "Assignment unit"),
              ("observation", "Observation unit"), ("block", "Block"),
              ("population", "Target population"), ("estimator", "Estimator"),
              ("uncertainty", "Uncertainty estimator"), ("_n", "n")]
    for r in design["rows"]:
        d.sub(f"§{r['sec']}  {r['name']}", size=9.4, before=6)
        for k, label in FIELDS:
            if str(r.get(k, "")).strip():
                d.para(f"{label}.  {r[k]}", size=8.2, lead=10.2, after=0.8,
                       indent=10)
        d.para(f"Artifact.  {r['artifact']}", size=8.2, lead=10.2, after=3.0,
               indent=10, colour=MUTE)

    # ================================================================
    d.head("2.  Models, prompts, names, matching rule, retained pairs")

    d.sub("2.1  Models")
    d.para(
        "Four open-weight instruction-tuned checkpoints carry the behavioural "
        "results. Weights are pinned by file hash and served locally; the "
        "serving configuration is part of the experiment rather than an "
        "implementation detail, and §5.2 reports what happens when it is not "
        "controlled.")
    for m in models:
        d.para(f"·  {B.SHORT[m]}", size=8.6, lead=10.6, after=0.6, indent=10)
    _nf = len(front["models"]) if front else 0
    d.para(
        f"A further {_nf} frontier APIs return log probabilities and carry the "
        "replication in §4.7"
        + (f", plus one verdict-only arm ({fverd['model']}, "
           f"{fverd['n_pairs']} matched pairs) whose endpoint returns no "
           "next-token distribution to any caller." if fverd else "."),
        size=8.6, lead=10.6, indent=10)

    d.sub("2.2  The twelve prompts, in full")
    d.para(
        "Six semantic paraphrases (S) and six semantically null perturbations "
        "(N). The system message is shared; only the closing request differs "
        "across S, and across N nothing differs that a reader could act on. "
        "N1 is byte-identical to S1 by construction, which is what makes the "
        "null arm free: it costs no extra calls and it is the control that "
        "decides whether the effect is about meaning or about surface form.")
    d.para("System message, common to all twelve:", size=8.6, after=2.0)
    d.mono(E.VARIANTS["S1"]["system"])
    d.para("The six semantic paraphrases differ in the closing request:",
           size=8.6, after=2.0)
    for k in [f"S{i}" for i in range(1, 7)]:
        d.mono(f"{k}   {E.VARIANTS[k]['ask']}", after=1.0)
    d.para("The six null perturbations all use S1's text and change only this:",
           size=8.6, after=2.0)
    for k in [f"N{i}" for i in range(1, 7)]:
        d.mono(f"{k}   {E.VARIANTS[k].get('note', '')}", after=1.0)
    d.para(
        "The user message is assembled as: the job posting, then the "
        "candidate's name and résumé body, then the request. The perturbation "
        "is applied to that assembled string, so N3 and N5 act on the "
        "posting and the résumé as well as on the instruction.", size=8.6,
        lead=11.0)

    d.sub("2.3  Names")
    _fp = len(E.PAIRS)
    d.para(
        f"{_fp} matched first-name pairs, drawn from the Bertrand and "
        "Mullainathan (2004) appendix list as transcribed in `names.py`, "
        f"crossed with surnames to give {len(ST.NAME_GRID)} full-name pairs. "
        "Names are matched on gender within a pair and the pairing is fixed "
        "before any call is made. The full grid is in the released artifacts.")
    d.para("The twelve first-name pairs, White then Black:", size=8.6, after=2.0)
    for a, b in E.PAIRS:
        d.mono(f"{a.split()[0]:<12} {b.split()[0]}", after=0.4)

    d.sub("2.4  Résumé templates and postings")
    d.para(
        f"{len(E.TEMPLATES)} résumé templates spanning a strength range "
        f"({', '.join(sorted(E.TEMPLATES))}), and {len(OC.OCCUPATIONS)} job "
        f"postings ({', '.join(OC.OCCUPATIONS)}) chosen for gender-typing "
        "contrast rather than coverage. Every name pair sees every template "
        "under every wording, so the design is fully crossed and the "
        "resampling question in Part 4 is well posed.")

    d.sub("2.5  Token matching, and which pairs are excluded")
    if tbal:
        _mm = tbal.get("max_matching", {})
        _cl = _mm.get("female_first", 0) + _mm.get("male_first", 0)
        d.para(
            "A pair is TOKEN-MATCHED under a given tokenizer when the two "
            "names segment to the same number of tokens. The rule is "
            "arithmetic on the list and the tokenizer and needs no model call, "
            "so it is checkable before any data exists. It is a property of "
            "the pair and the tokenizer jointly, so a pair matched under one "
            "checkpoint can be unmatched under another.")
        d.para(
            "Restricting to matched pairs is what the companion paper "
            "measures. The restriction is severe: a grid balanced by "
            f"construction retains {tbal.get('n_pairs', '—')} rows built from "
            f"{_cl} independent first-name pairs, which is too small to carry "
            "an audit. That is the reason the companion reports three "
            "procedures that disagree rather than one that does not.")
    d.para(
        "NOTHING IS EXCLUDED FROM THE MAIN ANALYSES. The token-matched subset "
        "is a separate arm reported in the companion; the results in Part 3 "
        "use the full grid. No pair, cell or model is dropped anywhere for any "
        "reason, and no result in either paper is conditioned on a filter that "
        "is not named in its own section.")

    # ================================================================
    d.head("3.  Sample breakdown")
    _n_rec, _n_single, _n_calls = B.corpus_size()
    d.para(
        f"{_n_rec:,} matched pairs and {_n_calls:,} model calls in total. The "
        "unit of a row is a matched pair: one name pair, one wording, one "
        "template, one posting, one model. Each row is two scorings, one per "
        "résumé, because the two are never in the same prompt.")
    W = [88, 62, 62, 62, 62, 78]
    d.row(["model", "pairs/cell", "scorings", "wordings", "templates",
           "SD across wordings"], W, bold=True)
    d.rule(3)
    for m in models:
        ov, bi = s2[m]["overall"], s2[m]["binary"]
        d.row([B.SHORT[m], f"{ov['n']}", f"{bi['n']}",
               f"{len(s2[m]['per_variant'])}", f"{len(E.TEMPLATES)}",
               f"{s2[m]['ps_sd_across_wordings']:.4f}"], W)
    d.rule(4)
    d.para(
        "Every cell of every model is present; there are no missing "
        "combinations. The per-wording estimates behind the last column are "
        "released per model in `study2_v2.json` under `per_variant`, which is "
        "the breakdown you asked for and, as far as I can tell, the only "
        "per-wording set published in this literature.", size=8.8, lead=11.4)
    if occ:
        d.para(
            "By posting: the same design is run on each of "
            f"{len(OC.OCCUPATIONS)} postings, so the model-by-posting cell is "
            "the unit the headline dispersion ratio is computed over.",
            size=8.8, lead=11.4)

    # ================================================================
    d.head("4.  The pairing and resampling claims, with their evidence")

    d.sub("4.1  What the pairing claim does and does not say")
    d.para(
        "Each résumé is scored on its own. The two are never in the same prompt, "
        "because presented together the answer is a function of presentation "
        "order on every checkpoint we gated. A consequence we did not follow "
        "up until you pushed on it: nothing in the MEASUREMENT pairs the two "
        "résumés. The pairing happens in the analysis.")
    d.para(
        "You are right that this does not reach a design like yours. Where "
        "profiles are preconstructed and assigned inside a prespecified block "
        "(a vacancy, a firm, an employer), the match is fixed before any "
        "outcome is observed and re-pairing across blocks would break the "
        "design rather than exercise a degree of freedom. The claim reaches "
        "only audits that score independently and impose a pairing afterwards "
        "to compute a statistic that is a function of WHICH résumé beat which. "
        "§6.3 now says so and names the blocked design as the case it excludes.")
    _ps = pairfree["summary"]
    d.para("The randomisation test, and its result:", size=8.8, after=2.0)
    d.mono(f"n_perm            {pairfree['n_perm']:,} random re-pairings, "
           f"drawn within gender")
    d.mono(f"seed              {pairfree['seed']}")
    d.mono(f"statistic         probability of superiority")
    d.mono(f"SD over pairings  {_ps['min_perm_sd']:.4f} to "
           f"{_ps['max_perm_sd']:.4f} across {_ps['n_models']} models")
    d.mono(f"best-worst range  {_ps['min_best_worst_range']:.3f} to "
           f"{_ps['max_best_worst_range']:.3f}, by maximum-weight matching")
    d.mono(f"sign flips        {_ps['n_models_where_repairing_flips_sign']} of "
           f"{_ps['n_models']} models change sign under some re-pairing")
    d.para(
        "Two numbers, two different claims. The SD is what a random re-pairing "
        "costs, and it is small, an order of magnitude below the "
        "between-wording dispersion, and we report it as the small term it is. "
        "The best-worst range is what an adversarial re-pairing could buy, "
        "computable in a second from data the analyst already has. No honest "
        "researcher searches that space; the point is that no reader of a "
        "published audit can tell whether one did, because the pairing is "
        "never reported.", size=8.8, lead=11.4)
    d.para(
        "You are also right that the remedy is not novel. The mean paired "
        "difference is the difference of the two group means, so re-pairing "
        "cannot move it at all. §6.3 says exactly that. In a balanced design "
        "it is the ordinary difference in means, which is what a competent "
        "analyst already reports. The contribution is not the estimator; it is "
        "that some published audits report a PAIRWISE statistic instead, and "
        "that the choice between them is undisclosed.", size=8.8, lead=11.4)

    d.sub("4.2  The resampling unit")
    _rs = resamp["pooled_summary"]
    d.para(
        "The design is crossed: every name pair appears under every wording "
        "and every template. Pooling those rows and bootstrapping them "
        "independently treats repeated measurements of the same twelve names "
        "as independent draws.")
    d.mono(f"n_boot            {resamp['n_boot']:,} replicates "
           f"({resamp['n_boot_contrast']:,} for contrasts)")
    d.mono(f"seed              {resamp['seed']}")
    d.mono(f"comparison        percentile bootstrap over ROWS vs over "
           f"NAME PAIRS, same seed, same replicate count")
    d.mono(f"width ratio       {_rs['min_ratio']:.2f} to {_rs['max_ratio']:.2f} "
           f"across {_rs['n_models']} models")
    d.mono(f"consequence       {_rs['n_significant_iid']} of "
           f"{_rs['n_models']} effects appear distinguishable from zero under "
           f"the row bootstrap; {_rs['n_significant_clustered']} survive")
    _ct = resamp.get("contrasts") or {}
    if _ct:
        d.mono(f"paired contrasts  {_ct['n']} of them; ratio "
               f"{_ct['min']:.2f} to {_ct['max']:.2f}, median "
               f"{_ct['median']:.2f}; "
               f"{100 * _ct['frac_widen_over_10pct']:.0f}% widen by >10%")
    d.para(
        "The generalisation is about provenance, not about one unit. THE "
        "RESAMPLING UNIT FOLLOWS THE ASSIGNMENT PROCESS. Here names carry the "
        "treatment and the grid is crossed over them, so the name pair is the "
        "unit. In a design that assigns preconstructed profiles within a "
        "vacancy, the vacancy is the block and clustering there is what a "
        "correct interval requires, which is what the correspondence-audit "
        "literature has long done, and what your cluster-robust wild bootstrap "
        "does. Neither is right in the abstract. §6.1 now states the principle "
        "rather than the unit, and the finding it reports is that no audit we "
        "surveyed states its unit clearly enough for a reader to check it "
        "against the assignment process.", size=8.8, lead=11.4)

    d.sub("4.3  Coverage, and what is not claimed")
    d.para(
        "The intervals are percentile bootstrap, not studentised, and we do "
        "not claim nominal coverage for them. What is claimed is a COMPARISON "
        "between two estimators computed on the same data at the same "
        "replicate count and seed, which is the quantity the ratio above "
        "reports. Where an interval and an exact permutation null are both "
        "available for the same estimand they are required to agree, and one "
        "place they did not. A frozen-rank, unstratified bootstrap in the "
        "construct-validity check was found by that requirement, and "
        "corrected.", size=8.8, lead=11.4)

    d.sub("4.4  Supporting citations")
    for c in [
        "Bertrand, M. and Mullainathan, S. (2004). Are Emily and Greg More "
        "Employable than Lakisha and Jamal? American Economic Review 94(4), "
        "991–1013.",
        "Lippens, L. (2024). Computer says ‘no’: Exploring systemic bias in "
        "ChatGPT using an audit approach. Computers in Human Behavior: "
        "Artificial Humans 2, 100054.",
        "Steegen, S., Tuerlinckx, F., Gelman, A. and Vanpaemel, W. (2016). "
        "Increasing Transparency Through a Multiverse Analysis. Perspectives "
        "on Psychological Science 11(5), 702–712.",
        "Simonsohn, U., Simmons, J. P. and Nelson, L. D. (2020). "
        "Specification curve analysis. Nature Human Behaviour 4, 1208–1214.",
        "Sclar, M., Choi, Y., Tsvetkov, Y. and Suhr, A. (2024). Quantifying "
        "Language Models’ Sensitivity to Spurious Features in Prompt Design. "
        "ICLR 2024.",
    ]:
        d.para("·  " + c, size=8.2, lead=10.2, after=2.0, indent=10)
    d.para(
        "The correspondence-audit inference literature is where this "
        "supplement is thinnest and where I would most value a pointer, "
        "specifically on whether the matched-versus-unmatched question has a "
        "settled answer when the pairing is imposed after scoring rather than "
        "before.", size=8.8, lead=11.4, colour=MUTE)

    d.save()
    print(f"wrote {OUT.relative_to(ROOT)}  ({d.page} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
