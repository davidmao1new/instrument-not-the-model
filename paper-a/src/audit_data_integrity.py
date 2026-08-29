"""Standing integrity audit over every raw record the paper rests on.

This is not a unit test of the code; it is a check on the DATA, run against
whatever is currently on disk. It answers the questions a reviewer would ask if
they were suspicious, and it is designed so that passing is informative rather
than vacuous.

WHAT IT CHECKS

  WEIGHTS      Every checkpoint's SHA256 against config.yaml, so no result
               traces to a file that was silently replaced. A truncated hash
               was padded to look full-length once in this project's history;
               a 64-hex-character assertion is a separate test, and this one
               recomputes the digests.

  DUPLICATES   A cell measured twice with DIFFERENT values. The analyses
               deduplicate by keeping the last row per cell, which is correct
               for a resumed run but would silently hide a genuine
               inconsistency, so the count and the magnitude are reported here.

  COVERAGE     Cells expected by the design that are absent, per file. A
               partially complete run is fine and must be visible.

  RANGE        Margins that are NaN, infinite, or implausibly large. A margin
               is a log-odds difference over a 100-token window, so anything
               past about 50 means the probability arithmetic underflowed.

  ERRORS       Rows carrying a non-empty error field, which the analyses drop.
               Study 1 lost 24 calls to timeouts and recorded the count
               nowhere; this makes that class of loss impossible to miss again.

  BALANCE      Missing cells must not be unbalanced by race. Differential
               attrition between the two arms of a matched pair would bias the
               contrast, and it is exactly the failure a paired design is
               supposed to be immune to.

    .venv/Scripts/python.exe paper-a/src/audit_data_integrity.py
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "paper-a" / "data"
CONFIG = ROOT / "paper-a" / "config.yaml"

MARGIN_SANE = 60.0
FAILURES: list[str] = []
NOTES: list[str] = []


def fail(msg):
    FAILURES.append(msg)
    print(f"  FAIL  {msg}")


def note(msg):
    NOTES.append(msg)
    print(f"  note  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


# --------------------------------------------------------------------------
def check_weights():
    print("\nWEIGHTS")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    entries = [(m["id"], m["file"], m.get("sha256"), m.get("bytes"))
               for m in cfg["models"]]
    entries += [(e["id"] + " (Q8)", e["file"], e.get("sha256"), None)
                for e in cfg.get("robustness_quantization", [])]
    for mid, rel, want, want_bytes in entries:
        p = ROOT / rel
        if not p.exists():
            note(f"{mid}: weights absent from disk, skipped")
            continue
        if not want or len(want) != 64:
            fail(f"{mid}: config SHA256 is not 64 hex characters")
            continue
        if want_bytes is not None and p.stat().st_size != want_bytes:
            fail(f"{mid}: size {p.stat().st_size} != config {want_bytes}")
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != want:
            fail(f"{mid}: SHA256 {got[:16]}... != config {want[:16]}...")
        else:
            ok(f"{mid}: SHA256 verified")


# --------------------------------------------------------------------------
# Studies whose missing cells come from the log-probability window rather than
# from a fault, and whose analysis is required to say so. The mapping is to the
# artifact that must carry the record, and to the key that must be present in
# it; if an analysis stops reporting its censoring, the downgrade stops too and
# the imbalance becomes a failure again.
CENSORING_ANALYSED = {
    "second_task": ("second_task/second_task_analysis.json", "censoring"),
    "frontier": ("frontier/frontier_margin_analysis.json", "n_censored"),
}


def _censoring_is_analysed(folder_name: str, model: str = None):
    """Text describing where the censoring is reported, or None.

    PER-MODEL, NOT PER-FILE. The old form substring-matched the key against
    the whole artifact, so ANY model's censoring record vouched for every
    model's -- and gpt-4.1's unmeasurable entry carried no race breakdown at
    all while the audit said it was "measured and reported". The downgrade
    now requires the specific model's entry to hold the race split.
    """
    entry = CENSORING_ANALYSED.get(folder_name)
    if not entry:
        return None
    rel, key = entry
    p = DATA / rel
    if not p.exists():
        return None
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if folder_name == "frontier" and model:
        rec = (blob.get("models") or {}).get(model)
        if not isinstance(rec, dict):
            return None
        if ("censored_white_only" in rec and "censored_black_only" in rec):
            return (f"measured and reported in {rel} "
                    f"(white-only {rec['censored_white_only']}, "
                    f"black-only {rec['censored_black_only']})")
        return None
    if key in json.dumps(blob)[:2_000_000]:
        return f"measured and reported in {rel}"
    return None


def scan(label, folder, pattern, key_fields, expected=None):
    folder = DATA / folder
    if not folder.exists():
        note(f"{label}: directory absent, skipped")
        return
    files = sorted(folder.glob(pattern))
    if not files:
        note(f"{label}: no files matching {pattern}, skipped")
        return
    print(f"\n{label.upper()}")
    for f in files:
        rows = list(st.read_jsonl(f))
        if not rows:
            note(f"{f.name}: empty")
            continue

        seen = defaultdict(list)
        n_err = 0
        bad_range = 0
        miss_by_race = Counter()
        complete = 0
        for r in rows:
            if r.get("error"):
                n_err += 1
            k = tuple(r.get(x) for x in key_fields)
            vals = []
            for side in ("white", "black"):
                m = r.get(f"{side}_margin")
                if m is None:
                    miss_by_race[side] += 1
                    continue
                if not math.isfinite(m) or abs(m) > MARGIN_SANE:
                    bad_range += 1
                vals.append(round(float(m), 12))
            if len(vals) == 2:
                complete += 1
                seen[k].append(tuple(vals))

        dup_cells = {k: v for k, v in seen.items() if len(v) > 1}
        disagreeing = {k: v for k, v in dup_cells.items() if len(set(v)) > 1}
        worst = 0.0
        for v in disagreeing.values():
            xs = [x[0] - x[1] for x in v]
            worst = max(worst, max(xs) - min(xs))

        line = (f"{f.name}: {len(rows)} rows, {complete} complete cells, "
                f"{len(seen)} distinct")
        if expected:
            line += f" of {expected} expected"
        print(f"  --    {line}")

        if bad_range:
            fail(f"{f.name}: {bad_range} margins non-finite or |m| > {MARGIN_SANE}")
        if n_err:
            note(f"{f.name}: {n_err} rows carry an error field")
        if disagreeing:
            fail(f"{f.name}: {len(disagreeing)} cells measured twice with "
                 f"DIFFERENT values, largest spread {worst:.4f} log-odds")
        elif dup_cells:
            note(f"{f.name}: {len(dup_cells)} cells re-measured identically "
                 f"(resumed run, harmless)")
        if miss_by_race and miss_by_race.get("white", 0) != miss_by_race.get("black", 0):
            # DIFFERENTIAL ATTRITION IS ALWAYS A THREAT. Whether it is an
            # INTEGRITY failure depends on whether the study's analysis
            # measures it. Two studies here lose cells to the log-probability
            # window rather than to a bug -- the second task and the frontier
            # arm -- and both report the loss, test it for imbalance by arm,
            # and bound what it could hide. Downgrading those to a note is only
            # legitimate while that stays true, so it is CHECKED rather than
            # assumed: the analysis artifact must carry a censoring record.
            # The model comes from the file name (margin_<model>.jsonl), so
            # the downgrade is vouched for by THAT model's record, not by
            # whichever model happened to carry one.
            _model = f.stem[len("margin_"):] if f.stem.startswith("margin_") \
                else f.stem
            handled = _censoring_is_analysed(folder.name, _model)
            msg = (f"{f.name}: missing cells UNBALANCED by race — "
                   f"white {miss_by_race.get('white',0)}, "
                   f"black {miss_by_race.get('black',0)}.")
            if handled:
                note(msg + f" Window censoring; {handled}")
            else:
                fail(msg + " A paired design cannot absorb differential "
                           "attrition.")
        elif miss_by_race:
            note(f"{f.name}: {sum(miss_by_race.values())} margins missing, "
                 f"balanced across arms")
        if expected and len(seen) < expected:
            # A CELL THAT RAN AND WAS CENSORED IS NOT A CELL THAT DID NOT RUN.
            # `seen` counts cells with BOTH margins, so it undercounts by the
            # censored ones, and calling those "not yet run" reads as an
            # incomplete experiment when the experiment finished. gpt-4.1 has
            # 432 rows and one usable cell: nothing is unrun there.
            n_rows_distinct = len({tuple(r.get(x) for x in key_fields)
                                   for r in rows})
            unrun = expected - n_rows_distinct
            censored = n_rows_distinct - len(seen)
            if unrun > 0:
                note(f"{f.name}: {unrun} of {expected} cells not yet run")
            if censored > 0:
                note(f"{f.name}: {censored} cells ran but yielded no margin "
                     f"(window censoring), leaving {len(seen)} of {expected}")


# --------------------------------------------------------------------------
def check_superseded_excluded():
    """Quarantined directories must not be reachable by any analysis glob."""
    print("\nQUARANTINE")
    def _quarantined(p: pathlib.Path) -> bool:
        return p.name.startswith("_") or "SUPERSEDED" in p.name

    # WHAT A TOP-LEVEL GLOB ACTUALLY REACHES, computed by globbing rather
    # than asserted. The analyses read the top level of each data folder, so
    # that is what is enumerated here; folders that are themselves
    # quarantined are not among the ones any analysis globs.
    #
    # The rule this replaces read:
    #
    #     reachable = [f for f in files if f.parent == DATA / q.parent.name]
    #
    # Every file in `files` came from q.glob, so every f.parent IS q, and the
    # condition asked whether the quarantine directory equals its own parent
    # -- false by construction. `reachable` was always empty, the failure
    # branch was unreachable code, and five directories printed "ok, not
    # reachable" on a comparison that could not have said anything else.
    reach: set[pathlib.Path] = set()
    for folder in DATA.iterdir():
        if folder.is_dir() and not _quarantined(folder):
            reach |= {p.resolve() for p in folder.glob("*.jsonl")}
            reach |= {p.resolve() for p in folder.glob("*.json")}
    reach |= {p.resolve() for p in DATA.glob("*.jsonl")}
    reach |= {p.resolve() for p in DATA.glob("*.json")}

    quarantined = [p for p in DATA.rglob("*")
                   if p.is_dir() and _quarantined(p)]
    for q in quarantined:
        files = list(q.glob("*.jsonl")) + list(q.glob("*.json"))
        reachable = [f for f in files if f.resolve() in reach]
        if reachable:
            fail(f"{q.relative_to(DATA)}: {len(reachable)} quarantined files "
                 f"sit where a top-level glob would find them")
        else:
            ok(f"{q.relative_to(DATA)}: {len(files)} files quarantined, "
               f"not reachable by a top-level glob")

    # THE LIST THE ANALYSES ACTUALLY FILTER BY MUST COVER WHAT IS HERE.
    # Directory layout is not what protects the corpus counts: two modules
    # walk the data tree recursively and then drop anything whose path
    # contains one of a hand-typed set of directory names. This audit
    # identifies a quarantine directory by a different rule -- leading "_",
    # or "SUPERSEDED" in the name -- and the two agree today only by
    # coincidence of naming. A directory quarantined as "_withdrawn" would
    # be reported safe here and walked into there.
    try:
        from analyze_corpus_size import QUAR as ANALYSIS_QUAR
    except Exception as exc:  # noqa: BLE001
        note(f"cannot read the analyses' quarantine list ({exc}); the "
             "directories above are unverified against what the paper "
             "builder actually filters")
    else:
        for q in quarantined:
            parts = set(q.relative_to(DATA).parts)
            if not parts & set(ANALYSIS_QUAR):
                fail(f"{q.relative_to(DATA)}: quarantined by this audit's "
                     "rule but NOT excluded by the list the analyses filter "
                     f"on ({', '.join(ANALYSIS_QUAR)}); the paper builder "
                     "walks the tree recursively and would read it")
        present = {p.name for p in DATA.rglob("*") if p.is_dir()}
        for name in ANALYSIS_QUAR:
            if name not in present:
                note(f"the analyses exclude {name!r}, which is not a "
                     "directory in the data tree")

    # A SUPERSEDED ARTIFACT LEFT AMONG LIVE ONES. This looked at DATA itself
    # and at one hardcoded folder name, so the same file in any other live
    # folder was silent. It asks the computed set instead, which is the set
    # that defines the risk.
    for stray in sorted(reach):
        if _quarantined(stray):
            note(f"{stray.relative_to(DATA)}: superseded artifact sits beside "
                 f"live ones; it is loaded by exact filename only")


# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("DATA INTEGRITY AUDIT")
    print("=" * 78)
    check_weights()
    scan("study 2 (delta stability)", "delta_stability", "delta_*.jsonl",
         ("model", "variant", "template", "pair"), expected=432)
    scan("study 4 (name grid)", "names", "names_*.jsonl",
         ("model", "variant", "template", "pair"))
    scan("study 5 (mechanism panel)", "mechanism_panel", "mech_*.jsonl",
         ("model", "mode", "cond", "template", "pair"))
    scan("study 3 (mechanism)", "mechanism", "mechanism_*.jsonl",
         ("model", "cond", "template", "pair"), expected=288)
    scan("study 6 (quantization)", "quantization", "delta_*.jsonl",
         ("model", "variant", "template", "pair"), expected=432)
    # THE STUDIES ADDED AFTER THIS FILE WAS WRITTEN, WHICH IT DID NOT COVER.
    # The occupation panel, the replicate arm, the second task and the frontier
    # arm all became load-bearing without ever passing a completeness or
    # duplication check. Each is a full 432-cell design per model, except the
    # replicate arm, whose row count is repeats x cells and so has no single
    # expected value.
    scan("study 7 (occupation)", "occupation", "occ_*.jsonl",
         ("model", "occupation", "variant", "template", "pair"), expected=864)
    # CONCURRENCY IS PART OF THE KEY HERE. Without it the two serving
    # configurations collapse onto one cell, the audit sees the same cell
    # measured twice with different values, and reports §5.2's headline result
    # -- that batching changes the number -- as an integrity failure.
    scan("study 8 (replicate)", "replicate", "rep_*.jsonl",
         ("model", "concurrency", "repeat", "variant", "template", "pair"))
    scan("second task", "second_task", "*_*.jsonl",
         ("model", "domain", "variant", "level", "pair"), expected=432)
    scan("frontier (margin arm)", "frontier", "margin_*.jsonl",
         ("model", "variant", "template", "pair"), expected=432)
    check_superseded_excluded()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S), {len(NOTES)} note(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"no integrity failures; {len(NOTES)} note(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
