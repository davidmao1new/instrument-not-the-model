"""Cut a numbered release of the paper and snapshot what produced it.

WHY VERSIONS ARE CUT RATHER THAN OVERWRITTEN. The critique loop runs the paper
past readers who have never seen it, fixes what survives refutation, and runs it
again. Without a numbered artifact per round there is no way to answer the only
question that matters about such a loop -- is it converging? -- and no way for a
reader to see what a round changed.

Each release writes:
  paper-a/releases/paper_instrument_validity_<vN>.pdf   the built paper
  paper-a/releases/<vN>.json                            what produced it
The manifest records the SHA-256 of the PDF, the page count, the test count, the
consistency and integrity verdicts, and the critique result that triggered the
round, so a version can be tied to the findings it answers.

    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/release_version.py v4 \
        --note "first critique round" --critique <path to round json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
PDF = ROOT / "paper-a" / "figures" / "paper_instrument_validity_v3.pdf"
REL = ROOT / "paper-a" / "releases"
PY = sys.executable


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(*cmd) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version", help="v4, v5, ...")
    ap.add_argument("--note", default="")
    ap.add_argument("--critique", default="",
                    help="path to the critique-round result this answers")
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    if not re.fullmatch(r"v\d+", args.version):
        sys.exit(f"version must look like v4, got {args.version!r}")

    REL.mkdir(parents=True, exist_ok=True)
    out_pdf = REL / f"paper_instrument_validity_{args.version}.pdf"
    if out_pdf.exists():
        sys.exit(f"{out_pdf.name} already exists; versions are never "
                 f"overwritten. Cut the next number instead.")

    manifest: dict = {"version": args.version, "note": args.note}

    if not args.skip_checks:
        print("  rebuilding ...", flush=True)
        rc, out = run(PY, "paper-a/src/build_paper_v3.py")
        manifest["build_ok"] = rc == 0
        manifest["build_tail"] = out.strip().splitlines()[-1] if out.strip() else ""
        if rc != 0:
            sys.exit(f"build failed:\n{out}")

        print("  tests ...", flush=True)
        rc, out = run(PY, "-m", "pytest", "tests", "-q")
        m = re.search(r"(\d+) passed", out)
        manifest["tests_passed"] = int(m.group(1)) if m else None
        manifest["tests_ok"] = rc == 0
        if rc != 0:
            sys.exit(f"tests failed:\n{out[-3000:]}")

        print("  consistency audit ...", flush=True)
        rc, out = run(PY, "paper-a/src/audit_consistency.py")
        manifest["consistency_clean"] = "no inconsistencies found" in out
        manifest["consistency_tail"] = out.strip().splitlines()[-1]

    if not PDF.exists():
        sys.exit(f"no built PDF at {PDF}")

    import fitz  # noqa: PLC0415
    doc = fitz.open(PDF)
    manifest["pages"] = doc.page_count
    text = "\n".join(p.get_text() for p in doc)
    manifest["chars"] = len(text)
    a, b = text.find("ABSTRACT"), text.find("Keywords")
    manifest["abstract_words"] = len(text[a:b].split()) if a >= 0 < b else None

    out_pdf.write_bytes(PDF.read_bytes())
    manifest["pdf"] = out_pdf.name
    manifest["sha256"] = sha256(out_pdf)

    if args.critique:
        cp = pathlib.Path(args.critique)
        if cp.exists():
            c = json.loads(cp.read_text(encoding="utf-8"))
            manifest["critique_round"] = {
                k: c.get(k) for k in
                ("version", "n_examined", "n_confirmed", "n_serious", "converged")}

    (REL / f"{args.version}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nreleased {out_pdf.relative_to(ROOT)}")
    for k in ("pages", "abstract_words", "tests_passed", "consistency_clean"):
        if k in manifest:
            print(f"  {k:<18} {manifest[k]}")
    print(f"  sha256             {manifest['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
