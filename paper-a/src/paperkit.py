"""A minimal two-column academic typesetter on reportlab.

There is no LaTeX on this machine, and installing a TeX distribution to render
one paper is disproportionate. This module implements the parts of a
conference-paper layout the paper actually needs: a two-column flow with
automatic column and page breaks, headings, justified body text, figures that
span one or two columns, tables, footnotes, and a reference list.

Set in Libertinus Serif, the maintained successor to Linux Libertine and the
face ACM's `acmart` class uses, so the output sits in the same visual family as
the venues this is aimed at.

Design decisions that matter for the output looking right:
  - Justified body with the last line of each paragraph left ragged.
  - Paragraph indent rather than inter-paragraph space, which is what
    single-spaced two-column papers use.
  - Figures are placed at the top of the next column that can hold them, which
    is what a real float does, rather than inline where they were declared.
  - Widow control: a heading never renders as the last line of a column.
"""
from __future__ import annotations

import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Callable

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas

ROOT = pathlib.Path(__file__).resolve().parents[2]
TTF = ROOT / "tools" / "fonts" / "ttf"

INK = colors.HexColor("#111111")
GREY = colors.HexColor("#4a4a4a")
RULE = colors.HexColor("#b0b0b0")

PAGE_W, PAGE_H = LETTER
MARGIN_X = 0.75 * inch
MARGIN_TOP = 0.85 * inch
MARGIN_BOT = 0.80 * inch
GUTTER = 0.28 * inch
COL_W = (PAGE_W - 2 * MARGIN_X - GUTTER) / 2
BODY_SIZE = 9.2
BODY_LEAD = 11.3
INDENT = 0.16 * inch


FONT_FALLBACK = False


def register_fonts() -> tuple[str, str, str]:
    try:
        pdfmetrics.registerFont(TTFont("Lib", str(TTF / "LibertinusSerif-Regular.ttf")))
        pdfmetrics.registerFont(TTFont("Lib-B", str(TTF / "LibertinusSerif-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("Lib-I", str(TTF / "LibertinusSerif-Italic.ttf")))
        pdfmetrics.registerFontFamily("Lib", normal="Lib", bold="Lib-B", italic="Lib-I")
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


R, B, I = register_fonts()

# Inline markup recognised in body text. Deliberately tiny: bold and italic
# only, no nesting semantics beyond "bold wins", no attributes. Anything richer
# would be a markup language, and this file is a typesetter for one paper.
_TAG = re.compile(r"</?(b|i)>")


@dataclass
class Block:
    """One renderable unit. `height` is computed before placement so the flow
    can decide whether it fits in the remaining column.

    `split` is set on blocks that can break across a column boundary -- in
    practice paragraphs. It takes the height available and returns
    (head, tail) or None if no acceptable break exists. Blocks without it are
    atomic: a table, a figure, a heading.
    """
    draw: Callable
    height: float
    keep_with_next: bool = False
    span2: bool = False
    split: Callable | None = None
    # A hard break. Flushes any queued floats, then starts a fresh page before
    # the next block. Needed because an appendix cannot begin halfway down the
    # second column of the page the conclusion ended on.
    force_break: bool = False


class Paper:
    def __init__(self, path: pathlib.Path, title: str, author: str):
        self.path = path
        self.c = rl_canvas.Canvas(str(path), pagesize=LETTER)
        self.c.setTitle(title)
        self.c.setAuthor(author)
        self.blocks: list[Block] = []
        self.page = 1
        self.running_title = title

    # ---------------- measurement helpers ----------------
    def wrap(self, text: str, font: str, size: float, maxw: float,
             first_indent: float = 0.0) -> list[tuple[str, float]]:
        self.c.setFont(font, size)
        lines, cur, avail = [], "", maxw - first_indent
        for w in text.split():
            t = f"{cur} {w}".strip()
            if self.c.stringWidth(t, font, size) <= avail:
                cur = t
            else:
                lines.append((cur, avail)); cur = w; avail = maxw
        if cur:
            lines.append((cur, avail))
        return lines

    # ---------------- inline markup ----------------
    #
    # Body text may carry <b>...</b> and <i>...</i>. Bold lead-ins are the
    # device this paper uses to open a paragraph with its claim, and emphasis is
    # needed to mark a word being used as a term rather than as itself; without
    # them the tags were being typeset literally.
    #
    # Tagging is resolved at CHARACTER level and only then grouped into words.
    # Doing it at word level breaks on the common case where a tag closes
    # mid-word -- "<i>more</i>," would become the two words "more" and "," and
    # be set with a space between them.

    def _word_runs(self, text: str, base_font: str):
        """Split into words, each a list of (text, font) runs."""
        chars, stack, pos = [], [], 0

        def face():
            if "b" in stack:
                return B          # no bold-italic face is bundled; bold wins
            if "i" in stack:
                return I
            return base_font

        for m in _TAG.finditer(text):
            f = face()
            chars.extend((ch, f) for ch in text[pos:m.start()])
            tag = m.group(1)
            if m.group(0).startswith("</"):
                if tag in stack:
                    stack.remove(tag)
            else:
                stack.append(tag)
            pos = m.end()
        f = face()
        chars.extend((ch, f) for ch in text[pos:])

        words, cur = [], []
        for ch, f in chars:
            if ch.isspace():
                if cur:
                    words.append(cur)
                    cur = []
            else:
                cur.append((ch, f))
        if cur:
            words.append(cur)
        # merge adjacent same-font characters so drawing is one call per run
        out = []
        for w in words:
            runs = []
            for ch, f in w:
                if runs and runs[-1][1] == f:
                    runs[-1][0] += ch
                else:
                    runs.append([ch, f])
            out.append([(t, f) for t, f in runs])
        return out

    def _word_w(self, word, size):
        return sum(self.c.stringWidth(t, f, size) for t, f in word)

    def wrap_runs(self, text: str, font: str, size: float, maxw: float,
                  first_indent: float = 0.0):
        """Like wrap(), but returns lines of run-tagged words."""
        space = self.c.stringWidth(" ", font, size)
        lines, cur, curw, avail = [], [], 0.0, maxw - first_indent
        for w in self._word_runs(text, font):
            ww = self._word_w(w, size)
            add = ww if not cur else ww + space
            if curw + add <= avail or not cur:
                cur.append(w)
                curw += add
            else:
                lines.append((cur, avail))
                cur, curw, avail = [w], ww, maxw
        if cur:
            lines.append((cur, avail))
        return lines

    # ---------------- content constructors ----------------
    def para(self, text: str, indent: bool = True, size: float = BODY_SIZE,
             lead: float = BODY_LEAD, font: str = None, color=INK,
             space_after: float = 0.0, span2: bool = False):
        font = font or R
        w = (PAGE_W - 2 * MARGIN_X) if span2 else COL_W
        fi = INDENT if indent else 0.0
        lines = self.wrap_runs(text, font, size, w, fi)
        h = len(lines) * lead + space_after
        space_w = self.c.stringWidth(" ", font, size)

        def draw(c, x, y, _lines=lines, _fi=fi):
            c.setFillColor(color)
            for k, (words, avail) in enumerate(_lines):
                xx = x + (_fi if k == 0 else 0)
                widths = [sum(c.stringWidth(t, f, size) for t, f in wd)
                          for wd in words]
                last = (k == len(_lines) - 1) or len(words) == 1
                gap = space_w if last else (
                    (avail - sum(widths)) / (len(words) - 1))
                cx = xx
                for wd, ww in zip(words, widths):
                    for t, f in wd:
                        c.setFont(f, size)
                        c.drawString(cx, y, t)
                        cx += c.stringWidth(t, f, size)
                    cx += gap
                y -= lead

        # WIDOWS AND ORPHANS. A break leaves at least MIN_KEEP lines above and
        # carries at least MIN_KEEP below; anything tighter looks like a
        # mistake rather than a break. A two-line paragraph is never split.
        MIN_KEEP = 2

        def _make(_lines, _fi, _space_after):
            def _draw(c, x, y, _l=_lines, _f=_fi):
                draw(c, x, y, _l, _f)
            return Block(_draw, len(_lines) * lead + _space_after,
                         span2=span2, split=_splitter(_lines, _fi, _space_after))

        def _splitter(_lines, _fi, _space_after):
            def _split(avail):
                n_fit = int((avail - 0.01) // lead)
                if n_fit < MIN_KEEP or len(_lines) - n_fit < MIN_KEEP:
                    return None
                head = _make(_lines[:n_fit], _fi, 0.0)
                # the continuation never re-indents: it is the same paragraph
                tail = _make(_lines[n_fit:], 0.0, _space_after)
                return head, tail
            return _split

        self.blocks.append(_make(lines, fi, space_after))

    def heading(self, text: str, level: int = 1, span2: bool = False):
        """A heading, wrapped to its column.

        A HEADING THAT DOES NOT WRAP RUNS OUT OF ITS COLUMN, and nothing
        complains: drawString will happily paint past the measure. It never
        showed while every heading was short. Moving six subsections into an
        appendix promoted their titles from level 2 to level 1, and two of them
        bled -- "A Instrument validation, and one limitation it exposed" ran
        265 pt into a 242 pt column and over the gutter into the neighbouring
        text, and "F Multiplicity, and a verdict that moved without its data"
        ran 29 pt past the right margin. Both were visible on the page and
        invisible to every check.
        """
        size = {1: 10.6, 2: 9.6, 3: 9.2}[level]
        font = B if level < 3 else I
        pre = 9.0 if level == 1 else 7.0
        lead = size * 1.35
        width = (PAGE_W - 2 * MARGIN_X) if span2 else COL_W
        # wrap() returns (line, available_width) pairs, not bare strings.
        lines = [ln for ln, _w in self.wrap(text, font, size, width)] or [text]
        h = pre + lead * len(lines)

        def draw(c, x, y, _l=lines):
            c.setFillColor(INK); c.setFont(font, size)
            yy = y - pre
            for ln in _l:
                c.drawString(x, yy, ln)
                yy -= lead
        self.blocks.append(Block(draw, h, keep_with_next=True, span2=span2))

    def page_break(self):
        """Start the next block on a fresh page.

        Used once, to open the appendix. FAccT counts the body against a page
        limit and the appendix against nothing, so the boundary between them
        has to be a page boundary or the count is ambiguous.
        """
        self.blocks.append(Block(lambda c, x, y: None, 0.0, force_break=True))

    def figure(self, img_path: pathlib.Path, span2: bool = False,
               max_h: float = 3.4 * inch, space_after: float = 8.0):
        if not img_path.exists():
            return
        img = ImageReader(str(img_path))
        iw, ih = img.getSize()
        w = (PAGE_W - 2 * MARGIN_X) if span2 else COL_W
        dh = w * ih / iw
        if dh > max_h:
            dh = max_h; w = dh * iw / ih
        total = dh + space_after

        def draw(c, x, y, _w=w, _dh=dh):
            c.drawImage(img, x, y - _dh, width=_w, height=_dh, mask="auto")
        self.blocks.append(Block(draw, total, span2=span2))

    def table(self, headers: list[str], rows: list[list[str]],
              widths: list[float], caption: str = "", span2: bool = False,
              size: float = 8.0, lead: float = 10.2,
              space_after: float = 13.0):
        total_w = sum(widths)
        # A TABLE WIDER THAN ITS COLUMN RENDERS OVER THE NEIGHBOURING ONE, AND
        # NOTHING WARNS. This module takes explicit widths and, until now, drew
        # whatever it was given: a single-column table summing past COL_W puts
        # its right-hand columns and its caption on top of the other column's
        # text, which is unreadable and looks like a rendering bug rather than
        # a specification error. It happened twice, to Table 6 and Table 10,
        # both when a column was added without re-checking the budget.
        #
        # Failing the build is the right response. A caller that genuinely
        # needs more room has span2 available, and if span2 is not enough the
        # table needs fewer columns or a smaller font -- not silence.
        avail = (PAGE_W - 2 * MARGIN_X) if span2 else COL_W
        if total_w > avail + 0.5:
            raise ValueError(
                f"table widths sum to {total_w:.1f} pt but only {avail:.1f} pt "
                f"is available ({'full width' if span2 else 'one column'}); "
                f"headers={headers[:4]}... Either shrink the widths, pass "
                f"span2=True, or drop a column. Rendering it would overprint "
                f"the neighbouring column.")
        # A CELL WIDER THAN ITS COLUMN SPILLS INTO THE NEXT ONE, and the
        # width-sum check above cannot see it: the columns fit, the CONTENT
        # does not. drawString and drawRightString clip nothing, so a long
        # header or a long value is simply painted over its neighbour, and on
        # the last column over the facing text. That is what happened to
        # Table 2, whose headers were wider than the widths they were given
        # even though those widths summed comfortably inside the column.
        #
        # Every string is therefore measured against the space it has. The
        # tolerance is the 3 pt gutter right-aligned cells already use, so a
        # cell may touch its boundary but not cross it.
        PAD = 3.0

        def _fit_header(txt, wd):
            """Split a header over up to two lines, as a real table does.

            Hand-tuning column widths every time a header grows is what put
            Table 2's headers on top of the facing column. A header that does
            not fit is wrapped instead; only if a single WORD cannot fit does
            this give up and let the caller hear about it.
            """
            if self.c.stringWidth(txt, B, size) <= wd - PAD:
                return [txt]
            words = txt.split()
            for cut in range(len(words) - 1, 0, -1):
                a, b = " ".join(words[:cut]), " ".join(words[cut:])
                if (self.c.stringWidth(a, B, size) <= wd - PAD
                        and self.c.stringWidth(b, B, size) <= wd - PAD):
                    return [a, b]
            return None

        hdr_lines, bad = [], []
        for hd, wd in zip(headers, widths):
            txt = hd[1:] if hd.startswith(">") else hd
            fit = _fit_header(txt, wd)
            if fit is None:
                bad.append((f"header {txt!r}",
                            self.c.stringWidth(txt, B, size), wd))
                fit = [txt]
            hdr_lines.append(fit)
        n_hdr = max(len(x) for x in hdr_lines) if hdr_lines else 1
        for r_i, row in enumerate(rows):
            for cell, wd, hd in zip(row, widths, headers):
                w = self.c.stringWidth(str(cell), R, size)
                if w > wd - PAD:
                    bad.append((f"row {r_i} cell {str(cell)[:24]!r}", w, wd))
        if bad:
            worst = max(bad, key=lambda t: t[1] - t[2])
            raise ValueError(
                f"{len(bad)} table cell(s) are wider than their column and "
                f"would overprint the next one. Worst: {worst[0]} needs "
                f"{worst[1]:.1f} pt but its column is {worst[2]:.1f} pt. "
                f"headers={headers[:4]}... Widen that column (and narrow "
                f"another), shorten the text, or reduce `size`.")

        # Captions render through plain wrap()+drawString, which parses no
        # runs: a <i> tag reaching here prints as five characters, which is
        # exactly what happened when a caps-to-italics pass edited caption
        # strings. A caption is set in italic already, so the honest
        # rendering of emphasis markup here is to drop the tags.
        caption = re.sub(r"</?[bi]>", "", caption) if caption else caption
        cap_lines = self.wrap(caption, I, 7.6, total_w) if caption else []
        h = ((len(cap_lines) * 9.4 + 4 if caption else 0)
             + lead * (len(rows) + n_hdr) + 10 + space_after)

        def draw(c, x, y, _cap=cap_lines):
            yy = y
            if _cap:
                c.setFillColor(INK)
                for ln, _ in _cap:
                    c.setFont(I, 7.6); c.drawString(x, yy, ln); yy -= 9.4
                yy -= 4
            c.setStrokeColor(INK); c.setLineWidth(0.7)
            c.line(x, yy + 2, x + total_w, yy + 2)
            yy -= lead * 0.75
            c.setFillColor(INK); c.setFont(B, size)
            # Multi-line headers are bottom-aligned, so the rule underneath sits
            # a constant distance from the last line whatever the tallest
            # header does.
            for li in range(n_hdr):
                cx = x
                yline = yy - (n_hdr - 1 - 0) * 0 - li * (lead * 0.85)
                for hd, wd, lines in zip(headers, widths, hdr_lines):
                    pad_top = n_hdr - len(lines)
                    k = li - pad_top
                    if 0 <= k < len(lines):
                        if hd.startswith(">"):
                            c.drawRightString(cx + wd - 3, yline, lines[k])
                        else:
                            c.drawString(cx, yline, lines[k])
                    cx += wd
            yy -= (n_hdr - 1) * (lead * 0.85) + 3
            c.setStrokeColor(RULE); c.setLineWidth(0.4)
            c.line(x, yy, x + total_w, yy)
            yy -= lead * 0.85
            c.setFont(R, size)
            for row in rows:
                cx = x
                for cell, wd, hd in zip(row, widths, headers):
                    c.setFillColor(INK)
                    if hd.startswith(">"):
                        c.drawRightString(cx + wd - 3, yy, cell)
                    else:
                        c.drawString(cx, yy, cell)
                    cx += wd
                yy -= lead
            c.setStrokeColor(INK); c.setLineWidth(0.7)
            c.line(x, yy + lead - 3, x + total_w, yy + lead - 3)
        self.blocks.append(Block(draw, h, span2=span2))

    def space(self, h: float):
        self.blocks.append(Block(lambda c, x, y: None, h))

    def rule(self, width_frac: float = 1.0, space: float = 6.0):
        def draw(c, x, y):
            c.setStrokeColor(RULE); c.setLineWidth(0.5)
            c.line(x, y, x + COL_W * width_frac, y)
        self.blocks.append(Block(draw, space))

    # ---------------- title block, spans both columns ----------------
    def title_block(self, title_lines: list[str], author: str, affil: str,
                    abstract: str, keywords: str = "", email: str = ""):
        w = PAGE_W - 2 * MARGIN_X
        y = PAGE_H - MARGIN_TOP
        c = self.c
        c.setFillColor(INK)
        for ln in title_lines:
            c.setFont(B, 16.5)
            c.drawCentredString(PAGE_W / 2, y, ln)
            y -= 20
        y -= 6
        c.setFont(R, 11); c.drawCentredString(PAGE_W / 2, y, author); y -= 13
        c.setFont(I, 9); c.setFillColor(GREY)
        c.drawCentredString(PAGE_W / 2, y, affil)
        y -= 12 if email else 16
        if email:
            # Upright rather than italic: an address is a literal string and
            # italicising it invites transcription errors.
            c.setFont(R, 8.6)
            c.drawCentredString(PAGE_W / 2, y, email)
            y -= 15

        # Abstract inset, narrower than the text block, which is the convention.
        inset = 0.55 * inch
        aw = w - 2 * inset
        c.setFillColor(INK); c.setFont(B, 9.2)
        c.drawString(MARGIN_X + inset, y, "ABSTRACT"); y -= 12
        # The LAST line of a justified paragraph must be set flush left. The
        # earlier version tested `avail == aw`, which is true of every
        # continuation line including the last, so the final line of the
        # abstract was stretched across the full measure with visible rivers.
        # Same non-parsing path as captions: strip run markup rather than
        # print it. The abstract renders roman; emphasis does not survive,
        # and that is the correct price for a renderer this simple.
        abstract = re.sub(r"</?[bi]>", "", abstract)
        _ablines = self.wrap(abstract, R, 9.0, aw)
        for _k, (ln, avail) in enumerate(_ablines):
            c.setFont(R, 9.0)
            words = ln.split()
            if len(words) > 1 and avail == aw and _k < len(_ablines) - 1:
                nat = sum(c.stringWidth(t, R, 9.0) for t in words)
                gap = (avail - nat) / (len(words) - 1)
                cx = MARGIN_X + inset
                for t in words:
                    c.drawString(cx, y, t); cx += c.stringWidth(t, R, 9.0) + gap
            else:
                c.drawString(MARGIN_X + inset, y, ln)
            y -= 11.2
        if keywords:
            y -= 5
            c.setFont(B, 8.6); c.drawString(MARGIN_X + inset, y, "Keywords:  ")
            c.setFont(R, 8.6)
            c.drawString(MARGIN_X + inset + c.stringWidth("Keywords:  ", B, 8.6),
                         y, keywords)
            y -= 12
        y -= 8
        c.setStrokeColor(RULE); c.setLineWidth(0.5)
        c.line(MARGIN_X, y, PAGE_W - MARGIN_X, y)
        self._start_y = y - 14

    # ---------------- flow ----------------
    def _footer(self):
        self.c.setFont(R, 8); self.c.setFillColor(GREY)
        self.c.drawCentredString(PAGE_W / 2, MARGIN_BOT - 22, str(self.page))

    def render(self):
        """Two-column flow with deferred full-width floats.

        Spanning figures and tables are NOT drawn where they are declared. They
        are queued and placed at the top of the next page, and the text keeps
        flowing into both columns meanwhile. That is what a float does in a real
        typesetter, and it is the difference between a dense page and one where
        the second column is stranded empty because a figure forced a break.

        The other subtlety is where column two starts. On page 1 it begins below
        the title block; on a page carrying floats it begins below them. A single
        per-page content top, updated whenever floats are placed, handles both.
        """
        c = self.c
        col_x = [MARGIN_X, MARGIN_X + COL_W + GUTTER]
        bottom = MARGIN_BOT
        page1_top = getattr(self, "_start_y", PAGE_H - MARGIN_TOP)

        pending: list[Block] = []
        col_top = page1_top
        col, y = 0, col_top
        i = guard = 0

        def place_floats():
            """Draw queued floats at the current top; return the new content top."""
            nonlocal col_top, y
            yy = col_top
            placed = 0
            for blk in pending:
                # Keep at least a third of the page for text so a float page
                # never becomes a float-only page unless the float demands it.
                if yy - blk.height < bottom + 2.2 * inch and placed:
                    break
                if yy - blk.height < bottom:
                    break
                blk.draw(c, MARGIN_X, yy)
                yy -= blk.height
                placed += 1
            del pending[:placed]
            col_top = yy
            y = yy

        def new_page():
            nonlocal col, y, col_top
            self._footer(); c.showPage(); self.page += 1
            col_top = PAGE_H - MARGIN_TOP
            col, y = 0, col_top
            if pending:
                place_floats()
                col = 0

        # Floats queued before any text still belong at the top of page 1.
        while i < len(self.blocks):
            guard += 1
            if guard > 60000:
                raise RuntimeError("layout did not converge; a block is taller "
                                   "than a full column")
            blk = self.blocks[i]

            if blk.force_break:
                # Everything still queued belongs to the part being closed, so
                # drain it before the break rather than letting a body float
                # drift onto the appendix's first page.
                while pending:
                    new_page()
                    if not pending:
                        break
                if not (col == 0 and y >= col_top - 0.01):
                    new_page()
                i += 1
                continue

            if blk.span2:
                pending.append(blk)
                i += 1
                continue

            need = blk.height
            if blk.keep_with_next and i + 1 < len(self.blocks):
                nxt = self.blocks[i + 1]
                if not nxt.span2:
                    # A heading needs real text under it, not the promise of
                    # it. Reserving four lines is not enough on its own: the
                    # next block only USES that room if it fits whole or can
                    # legally split into it, and a five-line paragraph can do
                    # neither -- splitting after four would leave a single line
                    # behind, which the widow rule forbids, so the paragraph
                    # moves and the heading is stranded. That is exactly how
                    # §6.1's heading ended a column alone.
                    #
                    # So ask the question directly: after drawing this heading,
                    # would the next block put anything here? If not, reserve
                    # its full height and move them together.
                    room_after = y - blk.height - bottom
                    usable = (nxt.height <= room_after
                              or (nxt.split is not None
                                  and nxt.split(room_after) is not None))
                    need += min(nxt.height, 4 * BODY_LEAD) if usable \
                        else nxt.height

            if y - need < bottom:
                # A SPLITTABLE BLOCK BREAKS RATHER THAN MOVES. Only when there
                # is enough room to be worth breaking into: below that the
                # split would leave a line or two stranded, which is worse
                # than a slightly short column.
                room = y - bottom
                if blk.split and not blk.keep_with_next and room > 3 * BODY_LEAD:
                    parts = blk.split(room)
                    if parts:
                        head, tail = parts
                        head.draw(c, col_x[col], y)
                        self.blocks[i] = tail
                        if col == 0:
                            col, y = 1, col_top
                        else:
                            new_page()
                        continue
                if col == 0:
                    col, y = 1, col_top
                else:
                    new_page()
                continue

            blk.draw(c, col_x[col], y)
            y -= blk.height
            i += 1

        # Anything still queued gets its own page or pages.
        while pending:
            new_page()
            if y >= col_top - 0.01 and not pending:
                break

        self._footer()
        c.save()
        return self.path
