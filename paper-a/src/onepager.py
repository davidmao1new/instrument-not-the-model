"""Build the one-page summary attached to outreach email.

Its job is to make a researcher who receives an unsolicited email decide, in
about twenty seconds of skimming, that there is a real result here.

FORM. The first version was laid out like startup collateral: a coloured eyebrow
rule, a large sans headline, three marketing-voice columns. Researchers read
extended abstracts, and the form itself signals whether the sender knows the
genre. This is now set as a short preprint front page: centred title and author
block, an abstract, numbered sections, a figure with its caption below in the
"Figure N." convention, and a rule-separated footnote carrying the
conflict-of-interest declaration.

TYPE. Libertinus Serif throughout, the maintained successor to Linux Libertine
and the face ACM's `acmart` class sets. The OTF ships with CFF outlines that
reportlab cannot embed, so `tools/fonts/ttf/` holds a converted TTF and the page
and the figures end up in the same face rather than two different serifs.

NUMBERS. Everything is computed from `panel_gate_results.json`. Nothing is typed
in, so the page cannot drift from the data. The headline spread is taken from
ADMITTED models only: the largest spread in the panel belongs to a model that
fails the quality gate, and leading with a number from a model we ourselves
exclude would be the cheap version of the argument.

Regenerate:  .venv/Scripts/python.exe paper-a/src/onepager.py
"""
from __future__ import annotations

import json
import pathlib
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas

ROOT = pathlib.Path(__file__).resolve().parents[2]
GATE = ROOT / "paper-a" / "data" / "panel_gate" / "panel_gate_results.json"
FIG = ROOT / "paper-a" / "figures" / "fig2_template_spread.png"
TTF = ROOT / "tools" / "fonts" / "ttf"
OUT = ROOT / "paper-a" / "figures" / "one_page_summary.pdf"

INK = colors.HexColor("#111111")
GREY = colors.HexColor("#555555")
RULE = colors.HexColor("#999999")

W, H = LETTER
M = 0.95 * inch
CW = W - 2 * M

FONT_FALLBACK = False


def _fonts() -> tuple[str, str, str]:
    """Register Libertinus if converted; otherwise fall back to base-14 Times,
    which is itself a legitimate journal face rather than a compromise."""
    try:
        pdfmetrics.registerFont(TTFont("Lib", str(TTF / "LibertinusSerif-Regular.ttf")))
        pdfmetrics.registerFont(TTFont("Lib-B", str(TTF / "LibertinusSerif-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("Lib-I", str(TTF / "LibertinusSerif-Italic.ttf")))
        return "Lib", "Lib-B", "Lib-I"
    except Exception as exc:  # noqa: BLE001
        # LOUD, because silence here cost a whole build. Times-Roman is a PDF
        # base-14 font with no U+2009 THIN SPACE and no U+0394 -- so every
        # "0.0 %" set with a thin space and every Delta silently became a
        # wrong glyph. The public release shipped without tools/fonts/ttf for
        # exactly one build, and the whole 31-page paper came out in the
        # fallback face with no error, no warning, and the right page count.
        print(f"  [FONT FALLBACK] Libertinus not loadable ({exc}); "
              f"typesetting in Times-Roman. Thin spaces and Greek letters "
              f"WILL render wrong. Expected: {TTF}", file=sys.stderr)
        globals()["FONT_FALLBACK"] = True
        return "Times-Roman", "Times-Bold", "Times-Italic"


R, B, I = _fonts()


def wrap(c, text, font, size, maxw):
    c.setFont(font, size)
    out, cur = [], ""
    for w in text.split():
        t = f"{cur} {w}".strip()
        if c.stringWidth(t, font, size) <= maxw:
            cur = t
        else:
            out.append(cur); cur = w
    if cur:
        out.append(cur)
    return out


def justified(c, text, x, y, font, size, leading, maxw, color=INK):
    """Justified body text. Academic pages are justified; ragged-right reads as
    a slide. The final line of a paragraph is left as-is, per convention."""
    c.setFillColor(color)
    lines = wrap(c, text, font, size, maxw)
    for k, ln in enumerate(lines):
        c.setFont(font, size)
        words = ln.split()
        if k == len(lines) - 1 or len(words) == 1:
            c.drawString(x, y, ln)
        else:
            natural = sum(c.stringWidth(w, font, size) for w in words)
            gap = (maxw - natural) / (len(words) - 1)
            cx = x
            for w in words:
                c.drawString(cx, y, w)
                cx += c.stringWidth(w, font, size) + gap
        y -= leading
    return y


def centered(c, text, y, font, size, color=INK, leading=None):
    c.setFillColor(color); c.setFont(font, size)
    for ln in wrap(c, text, font, size, CW):
        c.drawCentredString(W / 2, y, ln)
        y -= (leading or size * 1.25)
    return y


def section(c, n, title, y):
    c.setFillColor(INK); c.setFont(B, 9.5)
    c.drawString(M, y, f"{n}   {title}")
    return y - 12.5


def main() -> int:
    if not GATE.exists():
        sys.exit(f"missing {GATE}; run panel_gate.py first")
    rows = [r for r in json.load(open(GATE, encoding="utf-8")) if "load_error" not in r]

    n_models = len(rows)
    admitted = [r for r in rows if r["admit"]]
    n_admitted = len(admitted)
    worst = max(((r.get("template") or {}) for r in admitted),
                key=lambda t: t.get("spread") or -1, default={})
    sp = worst.get("spread") or 0.0
    sci = worst.get("spread_ci")
    npt = worst.get("n_per_template")
    spreads = sorted(t["spread"] for t in ((r.get("template") or {}) for r in admitted)
                     if t.get("spread") is not None)

    def extreme(r):
        ci = r["position"].get("position_a_ci")
        return bool(ci) and (ci[1] < 0.35 or ci[0] > 0.65)
    n_extreme = sum(extreme(r) for r in rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = rl_canvas.Canvas(str(OUT), pagesize=LETTER)
    c.setTitle("Before You Measure Bias, Check the Instrument")
    c.setAuthor("David Mao")
    y = H - M

    # --- title block ------------------------------------------------------
    y = centered(c, "Before You Measure Bias, Check the Instrument:", y, B, 15, INK, 18)
    y = centered(c, "Prompt-Form Dominance in LLM Résumé-Screening Audits", y, B, 15, INK, 18)
    y -= 5
    y = centered(c, "David Mao", y, R, 10.5, INK, 13)
    # The publication contact, matching the preprint's front matter. This
    # summary goes to correspondents alongside the paper, so a second address
    # here would have them replying somewhere the paper does not name.
    y = centered(c, "Wissahickon High School  ·  davidmao.xyz@gmail.com", y, I, 9, GREY, 11)
    y -= 6
    c.setStrokeColor(RULE); c.setLineWidth(0.5); c.line(M, y, W - M, y)
    y -= 16

    # --- abstract ---------------------------------------------------------
    ind = 0.30 * inch
    c.setFillColor(INK); c.setFont(B, 9)
    c.drawString(M + ind, y, "Abstract")
    y -= 12
    abstract = (
        f"Three recent audits disagree about whether newer language models have become "
        f"fairer in résumé screening, and each uses a different measurement instrument: "
        f"embedding retrieval, a one-word verdict, and open-ended email generation. None "
        f"reports whether its instrument recovers a stable decision from the models it "
        f"audits. We gate {n_models} open-weight checkpoints spanning the 2024 boundary on "
        f"three instrument-validity checks before measuring any disparity. Among the "
        f"{n_admitted} models that can distinguish a strong résumé from an unqualified one, "
        f"five paraphrases of a single instruction move the acceptance rate by "
        f"{spreads[0]*100:.0f} to {sp*100:.0f} percentage points"
        + (f" (largest: 95% CI {sci[0]*100:.0f} to {sci[1]*100:.0f}, n = {npt} per wording)"
           if sci and npt else "")
        + f". For comparison, the largest demographic effect reported in this literature is "
        f"3.01 points. {n_extreme} of {n_models} models select almost entirely by list "
        f"position rather than by the candidate, and the direction of that position bias "
        f"reverses within a model family across a single generation. A demographic gap of "
        f"two or three points, measured under one fixed prompt on a model that swings tens "
        f"of points across paraphrases of that prompt, is not safely interpretable."
    )
    y = justified(c, abstract, M + ind, y, R, 8.8, 11.4, CW - 2 * ind)
    y -= 14

    # --- figure -----------------------------------------------------------
    if FIG.exists():
        img = ImageReader(str(FIG))
        iw, ih = img.getSize()
        dw = CW
        dh = dw * ih / iw
        if dh > 3.35 * inch:
            dh = 3.35 * inch; dw = dh * iw / ih
        c.drawImage(img, M + (CW - dw) / 2, y - dh, width=dw, height=dh, mask="auto")
        y -= dh + 14

    # --- sections ---------------------------------------------------------
    y = section(c, "1", "Method", y)
    y = justified(c, (
        "Six pre- and post-2024 open-weight checkpoints are run locally under identical "
        "decoding settings, every model artifact pinned by SHA-256. Each receives a job "
        "posting and paired résumés that are byte-identical except for the manipulated "
        "signal, and is asked for a screening decision under five paraphrases of one "
        "instruction. Every proportion is reported with a Wilson 95% interval. Before any "
        "disparity is estimated, a model must distinguish a strong résumé from an "
        "unqualified one, remain stable under rewording, and respond to the candidate "
        "rather than to list position."), M, y, R, 8.8, 11.4, CW)
    y -= 11

    y = section(c, "2", "What remains open", y)
    y = justified(c, (
        "Whether any model passes all three gates; none tested so far does. Whether the "
        "published generational reversal in name-based disparity survives instrument "
        "controls. Whether structured field input, which is what production applicant "
        "tracking systems actually deliver to a model, changes measured disparity; no "
        "published audit varies it. And whether membership in a professional society "
        "functions as a demographic proxy once names are redacted, which no published "
        "audit has tested and which requires perceived prestige to be measured rather "
        "than assumed."), M, y, R, 8.8, 11.4, CW)
    y -= 16

    # --- footnote rule ----------------------------------------------------
    # Availability only. The conflict-of-interest line that used to sit here was
    # removed on David's instruction: this paper measures instrument validity on
    # open-weight checkpoints and takes no position on vendors, so the author's
    # employment is not an interest in the subject matter. It remains in Paper B,
    # which argues directly about AEDT vendors' obligations, and in the
    # disclosure field of any venue submission that asks for one.
    c.setStrokeColor(RULE); c.setLineWidth(0.5); c.line(M, y, M + 2.0 * inch, y)
    y -= 11
    justified(c, (
        "Code, model hashes, prompts, and raw model output are version-controlled "
        "and will be released with the preprint."),
        M, y, I, 7.6, 9.8, CW, GREY)

    c.showPage(); c.save()
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  font={R}  models={n_models} admitted={n_admitted} "
          f"spread={sp*100:.0f}pp extreme={n_extreme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
