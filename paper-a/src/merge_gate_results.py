"""Merge the per-stream panel-gate outputs into one results file.

The gate runs as two concurrent streams, one per llama.cpp backend, because the
Vulkan build uses the iGPU and the CPU build uses the CPU cores and the two can
genuinely run at once on this machine. Each stream writes its own JSON so the
two processes never race on a file.

This merges them back into `panel_gate_results.json`, which is what `figures.py`
and `onepager.py` read, ordered to match `config.yaml` so the figures are stable
across runs regardless of which stream finished first.

It also records which backend produced each row. That matters: the two backends
are numerically different implementations, so if a model's result ever depends on
which one ran it, that is a finding about reproducibility rather than a nuisance.
"""
from __future__ import annotations

import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper-a" / "data" / "panel_gate"
CONFIG = ROOT / "paper-a" / "config.yaml"
STREAMS = {"vulkan": "gate_vulkan.json", "cpu": "gate_cpu.json"}
MERGED = OUT_DIR / "panel_gate_results.json"


def main() -> int:
    order = [m["id"] for m in yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["models"]]
    rows: dict[str, dict] = {}
    seen_backends: dict[str, str] = {}

    for backend, fname in STREAMS.items():
        p = OUT_DIR / fname
        if not p.exists():
            print(f"  [missing] {fname}")
            continue
        data = json.load(open(p, encoding="utf-8"))
        for r in data:
            r["backend"] = backend
            if r["model"] in rows:
                print(f"  [warn] {r['model']} appears in both streams "
                      f"({seen_backends[r['model']]} and {backend}); keeping the first")
                continue
            rows[r["model"]] = r
            seen_backends[r["model"]] = backend
        print(f"  [{backend}] {len(data)} models from {fname}")

    if not rows:
        sys.exit("no stream output found; run panel_gate.py first")

    merged = [rows[m] for m in order if m in rows]
    merged += [r for m, r in rows.items() if m not in order]   # anything unexpected
    MERGED.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    print(f"\nwrote {MERGED.relative_to(ROOT)}  ({len(merged)}/{len(order)} models)")
    for r in merged:
        if "load_error" in r:
            print(f"  {r['model']:<28} [{r['backend']:<6}] LOAD FAILED")
            continue
        pos, t = r["position"], (r.get("template") or {})
        ci = pos.get("position_a_ci")
        ci_s = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "(no CI)"
        sci = t.get("spread_ci")
        sp = (f"{(t.get('spread') or 0)*100:.0f} [{sci[0]*100:.0f}-{sci[1]*100:.0f}]"
              if sci else f"{(t.get('spread') or 0)*100:.0f}")
        print(f"  {r['model']:<28} [{r['backend']:<6}] "
              f"posA={pos['position_a_rate']:.2f} {ci_s} n={pos.get('position_a_n')}  "
              f"spread={sp} pp  admit={r['admit']}")

    missing = [m for m in order if m not in rows]
    if missing:
        print(f"\nstill missing: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
