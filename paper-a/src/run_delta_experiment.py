"""Driver for the δ-stability experiment: server lifecycle per model, resumable.

Mirrors run_prestige.py. Runs only ADMITTED models, since a δ estimated from a
model that cannot separate a strong résumé from an unqualified one is not
measuring a screening decision.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import requests
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = ROOT / "paper-a" / "config.yaml"
GATE = ROOT / "paper-a" / "data" / "panel_gate" / "panel_gate_results.json"
EXP = ROOT / "paper-a" / "src" / "experiment_delta_stability.py"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
BACKENDS = {
    "vulkan": ROOT / "tools" / "llamacpp-vulkan" / "llama-server.exe",
    "cpu": ROOT / "tools" / "llamacpp-cpu" / "llama-server.exe",
}
VULKAN_BUDGET = 7_100_000_000


def start_server(server, model_path, port, threads, backend, wait_s=900):
    ngl = 99 if backend == "vulkan" else 0
    if backend == "vulkan" and model_path.stat().st_size > VULKAN_BUDGET:
        ngl = max(1, int(40 * (VULKAN_BUDGET / model_path.stat().st_size) * 0.80))
    proc = subprocess.Popen(
        [str(server), "-m", str(model_path), "-ngl", str(ngl), "-c", "4096",
         "--host", "127.0.0.1", "--port", str(port), "-t", str(threads),
         "--parallel", "4", "--no-webui"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early ({proc.returncode})")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health",
                            timeout=3).json().get("status") == "ok":
                return proc
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    proc.kill()
    raise RuntimeError(f"server never became healthy for {model_path.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="vulkan", choices=sorted(BACKENDS))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in cfg["models"]}
    admitted = [r["model"] for r in json.load(open(GATE, encoding="utf-8"))
                if r.get("admit")]
    wanted = {x.strip() for x in args.only.split(",")} if args.only else None
    todo = [m for m in admitted if wanted is None or m in wanted]

    print(f"[{args.backend}:{args.port}] delta-stability for {len(todo)} model(s)",
          flush=True)

    for mid in todo:
        path = ROOT / by_id[mid]["file"]
        if not path.exists():
            print(f"[skip] {mid}: weights missing", flush=True)
            continue
        print(f"\n{'='*66}\n{mid}\n{'='*66}", flush=True)
        try:
            proc = start_server(BACKENDS[args.backend], path, args.port,
                                args.threads, args.backend)
        except RuntimeError as e:
            print(f"  [LOAD FAILED] {e}", flush=True)
            continue
        try:
            subprocess.run([str(PY), str(EXP), "--model-label", mid,
                            "--port", str(args.port)], cwd=str(ROOT), check=False)
        finally:
            proc.kill(); proc.wait(timeout=30); time.sleep(3)

    print("\ndelta-stability pass complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
