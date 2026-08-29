"""Which paragraphs did the build script decide not to print, and why?

WHY THIS EXISTS. Twice in one round the paper ran an experiment, wrote the
analysis, built the table, and then printed nothing -- because the paragraph was
wrapped in a guard on a DIFFERENT quantity that happened to be absent. §4.6's
headline was gated on a dispersion-to-noise ratio that is None precisely because
the noise floor is zero, which is the good case. The abstract clause for the
same study had the same guard and the same consequence.

That failure is invisible from the built PDF: a missing paragraph looks exactly
like a paragraph that was never written. It is visible from the source, because
every such guard is a conditional whose value can be recomputed after the build.
This walks the build script for guarded prose blocks, evaluates each guard's key
against the artifacts on disk, and reports every block whose guard is FALSE --
so a suppressed result has to be looked at and either restored or accepted.

WHAT IT CANNOT DO. It is a heuristic over source text, not an interpreter: it
finds `if <name>` and `if <name>.get("key")` guards wrapping P(...) or
paper.table(...) calls, and it will miss a guard written some other way. A
false negative here is a paragraph nobody checked, so the report prints the
count of guards it could not evaluate rather than pretending it saw everything.

    C:/research-toolchain/venv/Scripts/python.exe \\
        paper-a/src/audit_suppressed_prose.py
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
SRC = ROOT / "paper-a" / "src" / "build_paper_v3.py"
D = ROOT / "paper-a" / "data"

# name -> artifact path, mirroring build_paper_v3's loader table. Read from the
# script itself so the two cannot drift.
LOADER = re.compile(r'^\s*"(\w+)":\s*D\s*/\s*(.+?),\s*$')
ASSIGN = re.compile(r'^\s*(\w+)\s*=\s*load\("(\w+)"\)\s*$')
GUARD = re.compile(
    r'^(\s*)if\s+(?:(\w+)\s+and\s+)?(\w+)(?:\.get\(["\'](\w+)["\']\))?'
    r'(?:\s*and\s+.*)?:\s*$')


def artifact_paths() -> dict[str, pathlib.Path]:
    out = {}
    for line in SRC.read_text(encoding="utf-8").splitlines():
        m = LOADER.match(line)
        if m:
            expr = m.group(2).replace('"', "").replace("'", "")
            parts = [p.strip() for p in expr.split("/")]
            out[m.group(1)] = D.joinpath(*parts)
    return out


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines()
    paths = artifact_paths()

    # local variable name -> loader key
    var2key = {}
    for line in lines:
        m = ASSIGN.match(line)
        if m:
            var2key[m.group(1)] = m.group(2)

    loaded: dict[str, object] = {}
    for var, key in var2key.items():
        p = paths.get(key)
        if p and p.exists():
            try:
                loaded[var] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                loaded[var] = None
        else:
            loaded[var] = None

    suppressed, unevaluable, live = [], 0, 0
    for i, line in enumerate(lines):
        m = GUARD.match(line)
        if not m:
            continue
        indent, extra, var, key = m.groups()
        if var not in loaded:
            continue
        # does this guard wrap any prose or table?
        body = []
        for j in range(i + 1, min(i + 60, len(lines))):
            nxt = lines[j]
            if nxt.strip() and not nxt.startswith(indent + " "):
                break
            body.append(nxt)
        blob = "\n".join(body)
        if not re.search(r"\bP\(|paper\.table\(|\bH\(", blob):
            continue

        obj = loaded.get(var)
        # `extra` is the left operand of `if A and B:`. It only tells us
        # anything when it is ITSELF a loaded artifact. Guards like
        # `if nl and nlen:` build their left operand locally from a glob, and
        # treating "not in the loader table" as "absent" reported §4.4 as
        # suppressed when it prints -- a false positive in an audit whose whole
        # value is that its output gets read rather than skimmed.
        if extra in loaded and loaded[extra] is None:
            val = False
        elif obj is None:
            val = False
        elif key:
            try:
                val = bool(obj.get(key))
            except Exception:  # noqa: BLE001
                unevaluable += 1
                continue
        else:
            val = bool(obj)

        if val:
            live += 1
        else:
            why = (f"{var} is missing" if obj is None
                   else f"{var}[{key!r}] is falsy" if key
                   else f"{var} is falsy")
            head = next((b.strip()[:96] for b in body
                         if b.strip().startswith(("P(", "paper.table(", "H("))),
                        "")
            suppressed.append((i + 1, var, key, why, head))

    print(f"{'line':>6}  {'guard':<28}{'why':<34}first statement")
    print("-" * 110)
    for ln, var, key, why, head in suppressed:
        g = f"{var}.get({key!r})" if key else var
        print(f"{ln:>6}  {g:<28}{why:<34}{head}")
    if not suppressed:
        print("  (none: every artifact-guarded prose block fired)")
    print()
    print(f"  {live} guarded blocks printed, {len(suppressed)} suppressed, "
          f"{unevaluable} guards this audit could not evaluate")
    if suppressed:
        print()
        print("  A suppressed block is not automatically a bug -- some guard "
              "genuinely optional content. But twice in this project the "
              "guard was on a DIFFERENT quantity than the paragraph reported, "
              "and a completed study printed nothing. Each line above needs a "
              "human decision, not a default.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
