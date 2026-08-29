"""Is the critique loop converging, and when should it stop?

THE QUESTION THIS ANSWERS. Six rounds of adversarial critique have run against
this paper. Each found defects; several found defects introduced by the fixes
of the round before. The obvious worry is that the process never terminates --
that a large enough model pointed at a long enough paper will always produce
findings, so "no findings" is unreachable and the stopping rule is arbitrary.

That is an empirical question about this repository, and the data to answer it
are on disk: six rounds of confirmed findings, each with a severity, a quote
and a defect description.

WHAT IS MEASURED, and why each matters for the stopping decision:

  * findings per round, by severity. A converging process shows a falling
    SERIOUS count even if the total holds up, because minor findings are
    inexhaustible in any long document -- there is always another sentence
    that could be phrased more precisely.

  * NOVELTY. A finding that repeats an earlier round's defect is evidence the
    fix failed; a finding on text that did not exist before is evidence the
    round is auditing new work rather than the same work again. We measure
    overlap by defect-text similarity across rounds.

  * SELF-INFLICTION. How many of a round's findings are about text introduced
    by the previous round's fixes? This is the quantity that decides whether
    more rounds help: a process whose repairs generate as many defects as they
    remove has stopped converging and is just churning.

  * SEVERITY DRIFT. If the modal finding moves from "this number is wrong" to
    "this sentence could be clearer", the loop has exhausted what it can see.

The output is a table, not a verdict. The decision of when to stop is the
author's; this tells them what the trend actually is.

    sh paper-a/src/_py.sh paper-a/src/analyze_critique_convergence.py
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
REL = ROOT / "paper-a" / "releases"
OUT = ROOT / "paper-a" / "data" / "reference" / "critique_convergence.json"
SERIOUS = ("critical", "major")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())


def shingles(s: str, n: int = 6) -> set:
    w = norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    rounds = []
    for p in sorted(REL.glob("critique_round[0-9].json")):
        m = re.search(r"round(\d)", p.name)
        if not m:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        conf = d.get("confirmed", [])
        rounds.append(dict(
            round=int(m.group(1)), file=p.name,
            n_examined=d.get("n_examined"),
            n_confirmed=len(conf),
            n_serious=sum(1 for f in conf if f.get("severity") in SERIOUS),
            n_critical=sum(1 for f in conf if f.get("severity") == "critical"),
            n_major=sum(1 for f in conf if f.get("severity") == "major"),
            n_minor=sum(1 for f in conf if f.get("severity") == "minor"),
            findings=conf))
    rounds.sort(key=lambda r: r["round"])
    if len(rounds) < 2:
        sys.exit("need at least two merged rounds")

    # NOVELTY. For each round, what share of its findings restate a defect an
    # earlier round already confirmed? A repeat means the earlier fix missed;
    # a novel finding means this round is looking at something new.
    for i, r in enumerate(rounds):
        prior = [f for q in rounds[:i] for f in q["findings"]]
        prior_sh = [shingles(f.get("defect", "")) for f in prior]
        rep = 0
        for f in r["findings"]:
            sh = shingles(f.get("defect", ""))
            if any(jaccard(sh, ps) >= 0.30 for ps in prior_sh):
                rep += 1
        r["n_repeat_of_earlier"] = rep
        r["n_novel"] = r["n_confirmed"] - rep
        r["novelty_rate"] = (r["n_novel"] / r["n_confirmed"]
                             if r["n_confirmed"] else None)

    # SELF-INFLICTION. A finding whose defect text names the previous round's
    # vocabulary of repair -- "an earlier draft", "we wrote", "this session",
    # "the fix" -- is about text the loop itself produced.
    SELF = ("earlier draft", "earlier version", "previous version",
            "we wrote", "the fix", "introduced", "one paragraph after",
            "the correction", "same defect")
    for r in rounds:
        r["n_self_inflicted"] = sum(
            1 for f in r["findings"]
            if any(k in norm(f.get("defect", "")) for k in SELF))

    for r in rounds:
        r.pop("findings", None)

    first, last = rounds[0], rounds[-1]
    trend = dict(
        serious_first=first["n_serious"], serious_last=last["n_serious"],
        serious_change=last["n_serious"] - first["n_serious"],
        critical_last=last["n_critical"],
        minor_share_first=(first["n_minor"] / first["n_confirmed"]
                           if first["n_confirmed"] else None),
        minor_share_last=(last["n_minor"] / last["n_confirmed"]
                          if last["n_confirmed"] else None),
        novelty_last=last["novelty_rate"],
        self_inflicted_last=last["n_self_inflicted"],
    )

    out = dict(
        _what="Round-over-round behaviour of the adversarial critique loop.",
        _why="To decide whether more rounds are worth running, and on what "
             "evidence, rather than by feel.",
        _novelty_method="A finding repeats an earlier one when their defect "
                        "texts share 30 % of their 6-word shingles.",
        n_rounds=len(rounds), rounds=rounds, trend=trend)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'round':>6}{'examined':>10}{'confirmed':>11}{'critical':>10}"
          f"{'major':>8}{'minor':>8}{'novel':>8}{'self':>7}")
    print("-" * 68)
    for r in rounds:
        print(f"{r['round']:>6}{str(r['n_examined']):>10}{r['n_confirmed']:>11}"
              f"{r['n_critical']:>10}{r['n_major']:>8}{r['n_minor']:>8}"
              f"{r['n_novel']:>8}{r['n_self_inflicted']:>7}")
    print()
    print(f"  serious findings: {trend['serious_first']} -> "
          f"{trend['serious_last']}")
    if trend["minor_share_first"] is not None:
        print(f"  minor share:      {trend['minor_share_first']:.0%} -> "
              f"{trend['minor_share_last']:.0%}")
    if trend["novelty_last"] is not None:
        print(f"  novelty, last round: {trend['novelty_last']:.0%} of findings "
              "restate no earlier one")
    print(f"  self-inflicted, last round: {trend['self_inflicted_last']}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
