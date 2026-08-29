"""Merge the halves of a split critique round into one result.

WHY A ROUND IS SPLIT. `critique_round.js` spawns one agent per lens and up to
ten refuters per finding. Run whole, that is close to ninety agents, and in
round 1 roughly half of them died before returning. A dead agent does not fail
the round -- it silently removes a lens or a refutation, so the paper comes out
looking better checked than it was. Two invocations of four lenses each land
reliably; this merges them and reports what each half contributed, so a missing
half is visible rather than invisible.

DEDUPLICATION. Two lenses can find the same defect from different directions --
a caption critic and a statistics critic both reaching the ratio with a null
denominator, say. Those are merged on (normalised quote, normalised defect
opening) and the surviving record keeps every lens that raised it, because a
defect two independent readers found is stronger evidence than one that only
one did, and the fix list should say so.

CONVERGENCE. The loop's stopping rule is zero CONFIRMED findings of severity
major or critical, across the merged round. Minor findings never run out and
are not a reason to keep going.

    C:/research-toolchain/venv/Scripts/python.exe \\
        paper-a/src/merge_critique_rounds.py v5 half1.json half2.json
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
OUT_DIR = ROOT / "paper-a" / "releases"
SERIOUS = ("critical", "major")


def norm(s: str) -> str:
    """Whitespace- and case-insensitive, punctuation-light key."""
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()[:160]


def main() -> int:
    if len(sys.argv) < 3:
        sys.exit("usage: merge_critique_rounds.py <round-name> <half.json> ...")
    name = sys.argv[1]
    halves = [pathlib.Path(p) for p in sys.argv[2:]]

    merged: dict[tuple, dict] = {}
    per_half = []
    n_examined = n_refuted = 0
    for h in halves:
        if not h.exists():
            print(f"  !! missing {h}")
            per_half.append(dict(file=h.name, present=False))
            continue
        d = json.loads(h.read_text(encoding="utf-8"))
        n_examined += d.get("n_examined", 0)
        n_refuted += d.get("refuted_count", 0)
        lenses = sorted({f.get("lens") for f in d.get("confirmed", [])})
        per_half.append(dict(file=h.name, present=True,
                             version=d.get("version"),
                             n_examined=d.get("n_examined"),
                             n_confirmed=d.get("n_confirmed"),
                             n_serious=d.get("n_serious"),
                             complete=d.get("complete"),
                             n_lenses_failed=d.get("n_lenses_failed"),
                             lenses_with_findings=lenses))
        for f in d.get("confirmed", []):
            k = (norm(f.get("quote")), norm(f.get("defect"))[:80])
            if k in merged:
                cur = merged[k]
                cur["lenses"] = sorted(set(cur["lenses"]) | {f.get("lens")})
                # keep the most severe reading of a defect two readers found
                order = {"critical": 0, "major": 1, "minor": 2}
                if order.get(f.get("severity"), 3) < order.get(cur["severity"], 3):
                    cur["severity"] = f["severity"]
                cur["n_independent_reports"] += 1
            else:
                # NAMESPACE THE ID BY ITS HALF. Each half numbers its findings
                # positionally -- STAT-3, SC-04, RW-3 -- so the two halves of a
                # round collide on 14 id strings that mean entirely different
                # defects. Dedup is by quote and defect, so nothing is dropped
                # here, but anything downstream that indexes by id silently
                # merges two unrelated findings: a verification pass keyed on
                # id reported one root cause for two of them. The raw label is
                # kept beside the qualified one so a report can still be traced
                # back to the half that produced it.
                # ... and a half can reuse an id WITHIN itself too (half 2 of
                # round 5 produced 56 findings under 50 distinct labels), so
                # the qualifier carries a sequence number rather than trusting
                # the label to be unique anywhere.
                _seq = len(merged) + 1
                merged[k] = dict(
                    id=f"{h.stem.split('_')[-1]}#{_seq}:{f.get('id')}",
                    id_within_half=f.get("id"), half=h.stem,
                    severity=f.get("severity", "minor"),
                    lenses=[f.get("lens")], n_independent_reports=1,
                    quote=f.get("quote"), defect=f.get("defect"),
                    evidence=f.get("evidence"), fix=f.get("fix"),
                    refuter_check=f.get("refuter_check"))

    order = {"critical": 0, "major": 1, "minor": 2}
    confirmed = sorted(merged.values(),
                       key=lambda f: (order.get(f["severity"], 3),
                                      -f["n_independent_reports"]))
    serious = [f for f in confirmed if f["severity"] in SERIOUS]

    out = {
        "round": name,
        "halves": per_half,
        "n_halves_present": sum(1 for h in per_half if h.get("present")),
        "n_examined": n_examined,
        "n_confirmed_before_dedup": sum(h.get("n_confirmed", 0) or 0
                                        for h in per_half if h.get("present")),
        "n_confirmed": len(confirmed),
        "n_serious": len(serious),
        "refuted_count": n_refuted,
        # A ROUND THAT DID NOT RUN HAS NOT CONVERGED. Every half must be
        # present, must report every lens, and must have examined something.
        # Zero findings from zero examinations is the signature of a failed
        # round, not of a clean paper.
        "n_halves_incomplete": sum(
            1 for h in per_half
            if h.get("present") and h.get("complete") is False),
        "n_halves_empty": sum(
            1 for h in per_half
            if h.get("present") and not (h.get("n_examined") or 0)),
        "complete": bool(
            per_half and all(h.get("present") for h in per_half)
            and all(h.get("complete") is not False for h in per_half)
            and all((h.get("n_examined") or 0) > 0 for h in per_half)),
        "converged": bool(
            len(serious) == 0
            and per_half and all(h.get("present") for h in per_half)
            and all(h.get("complete") is not False for h in per_half)
            and all((h.get("n_examined") or 0) > 0 for h in per_half)),
        "_stopping_rule": "zero confirmed findings of severity major or "
                          "critical across a COMPLETE merged round -- every "
                          "half present, every lens reporting, and something "
                          "actually examined. Minor findings never run out and "
                          "are not a reason to continue; an empty round is "
                          "not a reason to stop.",
        "confirmed": confirmed,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"critique_{name}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{name}: {out['n_halves_present']} of {len(halves)} halves present")
    for h in per_half:
        if h.get("present"):
            print(f"  {h['file']:<28} examined {h['n_examined']:>3}  "
                  f"confirmed {h['n_confirmed']:>3}  serious {h['n_serious']:>3}"
                  f"   lenses: {', '.join(h['lenses_with_findings']) or '-'}")
        else:
            print(f"  {h['file']:<28} MISSING")
    print()
    print(f"  examined {n_examined}, refuted {n_refuted}, "
          f"confirmed {len(confirmed)} after dedup "
          f"({out['n_confirmed_before_dedup']} before), serious {len(serious)}")
    dup = [f for f in confirmed if f["n_independent_reports"] > 1]
    if dup:
        print(f"  {len(dup)} found by more than one lens:")
        for f in dup:
            print(f"    [{f['severity']}] {'+'.join(f['lenses'])}: "
                  f"{str(f['quote'])[:70]}")
    print()
    if not out["complete"]:
        print(f"  ROUND INCOMPLETE \u2014 {out['n_halves_incomplete']} half(s) "
              f"had a failed lens, {out['n_halves_empty']} examined nothing. "
              f"No conclusion about convergence can be drawn; re-run.")
    elif out["converged"]:
        print("  CONVERGED")
    else:
        print(f"  NOT CONVERGED: {len(serious)} serious findings remain")
    print(f"\nwrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
