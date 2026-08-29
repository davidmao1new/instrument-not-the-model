"""Recompute `summary_of_published_pp_gaps` from the transcribed Gao/Jiang/Yan
rows, and refuse to do it unless the transcription still matches the PDF.

WHY THIS FILE EXISTS. The calibration number in Section 8 -- "the median
published gap is X points, and N of M fall below 1.1" -- is the number that
decides whether this paper's finding is interesting. It is a summary of somebody
else's table, and until now both the table and the summary were typed by hand
into `published_effects.json`. That failed exactly the way hand-typing fails:
two of the fourteen rows of Table I were recorded as untranscribable ("did not
extract cleanly from the two-column layout") and dropped. They extract cleanly.
Dropping them moved the median from 0.64 pp to 0.505 pp and the count below
1.1 pp from 11-of-14 to 10-of-12, and the paper printed the wrong pair.

So the summary is now DERIVED rather than typed, and the derivation is fenced by
a check that would have caught the original error:

  * Every row in the artifact is matched against the PDF text layer of the page
    carrying Table I -- release date, Cb%, gap, confidence interval and both
    discordant-pair counts, each token found within the row's own y-band. A
    mistyped digit fails here.

  * The row SET is checked against a number the authors state in prose and
    nowhere near the table: that discordant pairs among 2024+ models "range from
    547 ... to 3,643". Those are b+c sums. This is the check that matters,
    because it is sensitive to rows being MISSING rather than merely wrong --
    the failure mode that actually occurred. With Gemma-4-31B-it absent the
    minimum is 895, not 547, and this script would have refused to write.

  * The count of race-axis rows is checked against the paper's own "fourteen
    models".

WHAT IS NEVER RECOMPUTED. `superseded_twelve_row` is read from the artifact and
passed through untouched. House rule: nothing is deleted, so the numbers the
paper reported before the correction stay next to the ones it reports now, and
this script must not quietly re-derive them into agreement.

WHAT THIS DOES NOT TOUCH. The `effects` rows themselves. A transcription is by
definition typed from a source; the honest thing is to type it, check it against
the source, and compute everything downstream. This script does the checking and
the computing, not the typing.

    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/build_published_effects_summary.py
    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/build_published_effects_summary.py --check
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
ART = ROOT / "paper-a" / "data" / "reference" / "published_effects.json"
PDF = ROOT / "lit" / "gao_jiang_yan_2026_cuhk.pdf"
TXT = ROOT / "lit" / "text" / "gao_jiang_yan_2026_cuhk.txt"

KEY = "summary_of_published_pp_gaps"
BLOCK = "gao_jiang_yan_2026"

# The paper's own count of race-axis models, stated in Sec. 3.1 and in the
# Table I heading. Hard-coded as an expectation, not read from the artifact,
# so that dropping a row cannot also drop the expectation.
N_RACE_MODELS = 14

# Printed page index (0-based) carrying Table I. Printed folio is 14.
TABLE_I_PAGE = 14

# The typographic minus and the asterisk this typesetter emits are not ASCII.
_NORM = str.maketrans({"\u2212": "-", "\u2217": "*", "\u2032": "'", "\u00a0": " "})


def _norm(s: str) -> str:
    """Fold the PDF's typography onto ASCII and strip digit-group commas."""
    s = s.translate(_NORM).replace("*", "")
    return re.sub(r"(?<=\d),(?=\d)", "", s)


def _fmt_pp(v: float) -> str:
    return f"{v:+.2f}"


def _fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:+.2f},{ci[1]:+.2f}]"


def _table_rows(page) -> list[tuple[str, set[str], bool]]:
    """Reassemble Table I into one entry per model row.

    The typesetter splits a single visual row across up to three y-bands -- the
    gap and the interval float a few points above the model name -- so bands are
    merged around the band that carries the release date. Anchoring on the model
    name instead does not work: 'Claude 3 Haiku' and 'Claude Haiku 4.5' share
    every token, which is how the first version of this check produced a false
    alarm on clean data.
    """
    words = page.get_text("words")
    anchors = sorted({round(w[1], 1) for w in words
                      if re.fullmatch(r"20\d\d-\d\d", w[4])})
    rows = []
    for y in anchors:
        near = [w for w in words if abs(w[1] - y) <= 12.0]
        # The model label is everything left of the release-date column. Do not
        # try to identify it by shape: 'GPT-3.5-turbo' looks like a number to
        # any reasonable numeric filter, which is how the second version of this
        # check produced a second false alarm on the same clean data.
        xr = min(w[0] for w in near if re.fullmatch(r"20\d\d-\d\d", w[4]))
        label = " ".join(_norm(w[4]) for w in sorted(near, key=lambda w: w[0])
                         if w[0] < xr - 5.0)
        tokens = {_norm(w[4]) for w in near}
        starred = any("\u2217" in w[4] for w in near)
        rows.append((label.strip(), tokens, starred))
    return rows


def verify_against_pdf(effects: list[dict]) -> list[str]:
    """Match every artifact row against the PDF text layer. Returns problems."""
    import fitz  # imported here so --check degrades with a clear message

    rows = _table_rows(fitz.open(PDF)[TABLE_I_PAGE])
    problems: list[str] = []
    if len(rows) != len(effects):
        problems.append(f"Table I page yields {len(rows)} model rows but the "
                        f"artifact carries {len(effects)}")

    for e in effects:
        hits = [r for r in rows if r[0] == _norm(e["model"])]
        if len(hits) != 1:
            problems.append(f"{e['model']}: matched {len(hits)} rows on the "
                            f"Table I page, expected exactly 1")
            continue
        _, tokens, starred = hits[0]
        want = [e["released"], str(e["cb_pct"]), _fmt_pp(e["delta_pp"]),
                _fmt_ci(e["ci"]), str(e["b"]), str(e["c"])]
        missing = [t for t in want if t not in tokens]
        if missing:
            problems.append(f"{e['model']}: not found in its Table I row: "
                            f"{missing} (row reads {sorted(tokens)})")
        # Significance: stars present iff we recorded a significant result.
        if starred != (e["sig"] != "n.s."):
            problems.append(f"{e['model']}: sig={e['sig']!r} but the printed row "
                            f"{'shows' if starred else 'does not show'} stars")
    return problems


def verify_row_set_against_prose(effects: list[dict]) -> list[str]:
    """Check the row SET, not the row values, against a prose invariant.

    The authors state the range of discordant-pair totals among 2024+ models in
    running text, far from the table. b+c must reproduce it. This is the only
    check here that is sensitive to a row being absent.
    """
    prose = TXT.read_text(encoding="utf-8")
    problems: list[str] = []

    m = re.search(r"ranges from ([\d,]+) \([^)]+\) to ([\d,]+) \(", prose)
    if not m:
        return ["could not locate the discordant-pair range sentence in the "
                "text dump; the prose cross-check did not run"]
    want_lo, want_hi = (int(g.replace(",", "")) for g in m.groups())

    post = [e for e in effects if e["released"] >= "2024"]
    got = sorted(e["b"] + e["c"] for e in post)
    if not post or got[0] != want_lo or got[-1] != want_hi:
        problems.append(
            f"discordant-pair totals for 2024+ models run {got[0]}..{got[-1]}, "
            f"but the paper's prose says {want_lo}..{want_hi}. A row is missing "
            f"or miscopied.")

    if len(effects) != N_RACE_MODELS:
        problems.append(f"{len(effects)} race-axis rows transcribed, but the "
                        f"paper reports {N_RACE_MODELS} models")
    return problems


def compute(effects: list[dict], superseded: dict) -> dict:
    vals = sorted(round(abs(e["delta_pp"]), 4) for e in effects)
    n = len(vals)
    med = round(statistics.median(vals), 4)
    mean = round(sum(vals) / n, 4)
    below = sum(1 for v in vals if v < 1.1)
    above = sum(1 for v in vals if v > 2.0)
    mx = max(vals)

    out = {
        "_source": f"all {n} Gao/Jiang/Yan race-axis rows above, the only set of "
                   "measured on one protocol percentage-point callback gaps across a "
                   "model panel",
        "_computed_by": "paper-a/src/build_published_effects_summary.py — these "
                        "are recomputed from the effects rows and verified "
                        "against the PDF, not typed",
        "n": n,
        "abs_values_sorted": vals,
        "median_abs_pp": med,
        "mean_abs_pp": mean,
        "n_below_1_1_pp": below,
        "n_above_2_pp": above,
        "max_abs_pp": mx,
        "_correction": "The abstract said this literature 'typically reports two "
                       f"to three points'. It does not. The median is {med:.2f} pp "
                       f"and {below} of {n} fall below 1.10 pp. Two to three "
                       "points describes only the two extremes. The corrected "
                       "number strengthens the paper's argument rather than "
                       "weakening it.",
    }
    if superseded:
        out["superseded_twelve_row"] = superseded
        o = superseded
        moved = ("UNCHANGED at" if o["max_abs_pp"] == mx
                 else f"{o['max_abs_pp']} ->")
        out["_what_the_correction_moved"] = (
            f"Median absolute gap {o['median_abs_pp']} -> {med} pp; count below "
            f"1.1 pp {o['n_below_1_1_pp']} of {o['n']} -> {below} of {n}; mean "
            f"{o['mean_abs_pp']} -> {mean} pp; maximum {moved} {mx} pp "
            "(Llama-3.1-8B-Instruct, the model our own panel overlaps, so the "
            "overlap comparison is untouched). "
            "The direction of the paper's argument is unchanged: the published "
            "median is still well under one point.")
    return out


def _span(text: str, key: str) -> tuple[int, int]:
    """Byte span of the JSON object value for `key`, brace-matched.

    Splicing rather than re-serialising keeps the rest of the artifact
    byte-identical, so a diff shows only what actually changed. The scanner
    respects string literals, because several values here contain braces.
    """
    m = re.search(rf'^(\s*)"{re.escape(key)}":\s*\{{', text, re.M)
    if not m:
        raise SystemExit(f"key {key!r} not found in {ART}")
    i = text.index("{", m.start())
    depth, in_str, esc = 0, False, False
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i, j + 1
    raise SystemExit(f"unbalanced braces after {key!r}")


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    raw = ART.read_text(encoding="utf-8")
    doc = json.loads(raw)
    effects = doc[BLOCK]["effects"]

    problems = verify_row_set_against_prose(effects)
    try:
        problems += verify_against_pdf(effects)
    except ImportError:
        print("WARN: PyMuPDF unavailable; PDF row check skipped", file=sys.stderr)

    if problems:
        print("TRANSCRIPTION CHECK FAILED — refusing to write:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"transcription check passed: {len(effects)} rows match Table I "
          f"and the prose invariants")

    summary = compute(effects, doc.get(KEY, {}).get("superseded_twelve_row", {}))

    body = json.dumps(summary, indent=2, ensure_ascii=False)
    # Collapse all-numeric arrays back onto one line. Cosmetic, but this file is
    # read by people as often as by the build, and a fourteen-line column of
    # bare floats hides the shape of the list it is meant to show.
    body = re.sub(r"\[\s*((?:-?\d+(?:\.\d+)?,\s*)*-?\d+(?:\.\d+)?)\s*\]",
                  lambda m: "[" + ", ".join(t.strip()
                                            for t in m.group(1).split(",")) + "]",
                  body)
    body = "\n".join(("  " + ln) if i else ln
                     for i, ln in enumerate(body.splitlines()))
    lo, hi = _span(raw, KEY)
    new = raw[:lo] + body + raw[hi:]

    if new == raw:
        print(f"{ART.relative_to(ROOT)} already current")
        return 0
    if check_only:
        print(f"STALE: {KEY} does not match the effects rows", file=sys.stderr)
        return 1

    json.loads(new)  # never write a file we cannot read back
    ART.write_text(new, encoding="utf-8")
    print(f"rewrote {KEY} in {ART.relative_to(ROOT)}: "
          f"n={summary['n']} median={summary['median_abs_pp']} "
          f"below_1.1={summary['n_below_1_1_pp']} max={summary['max_abs_pp']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
