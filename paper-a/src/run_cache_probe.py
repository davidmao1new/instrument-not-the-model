"""Driver for the cache-residual probe: start a server, run the probe, stop it.

Two checkpoints, chosen because they carry the LARGEST residual in Study 8's
concurrency-1 arm (91.7% and 94.4% bitwise agreement, against 97.2% for
llama-3.1). If prompt-cache reuse is the remaining source of disagreement, it
has the most room to show up there.

Reuses run_suite's server lifecycle -- the port pre-flight, the offload ladder
and the exact-filename guard -- so this probe cannot be served the wrong weights
by a mechanism the suite already defends against.

    .venv/Scripts/python.exe paper-a/src/run_cache_probe.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_suite as rs  # noqa: E402

MODELS = ["llama-2-7b-chat", "mistral-7b-instruct-v0.3"]
PORT = 8080
THREADS = 4


def main() -> int:
    cfg = yaml.safe_load(rs.CONFIG.read_text(encoding="utf-8"))
    log = rs.ROOT / "paper-a" / "data" / "logs" / "cache_probe.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    with log.open("a", encoding="utf-8") as lf:
        for label in MODELS:
            path = rs.weights_for(cfg, label)
            lf.write(f"\n=== {label} :: {path.name} ===\n")
            lf.flush()
            proc = rs.start_server(rs.BACKENDS["vulkan"], path, PORT, THREADS,
                                   "vulkan")
            rs.verify_serving(PORT, path)
            try:
                r = subprocess.run(
                    [str(rs.PY), str(rs.SRC / "probe_cache_residual.py"),
                     "--model-label", label, "--port", str(PORT)],
                    stdout=lf, stderr=subprocess.STDOUT, text=True)
                lf.write(f"--- exit {r.returncode} ---\n")
                lf.flush()
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(5)
    print("cache probe complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
