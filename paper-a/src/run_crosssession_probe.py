"""Driver: re-measure the cache-off cells in a FRESH server process.

The point of the probe is that the server is a new process, so this driver must
start one rather than reuse whatever is listening. See
probe_cache_crosssession.py for why the question matters.
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
    log = rs.ROOT / "paper-a" / "data" / "logs" / "crosssession.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    rc = 0
    with log.open("a", encoding="utf-8") as lf:
        for label in MODELS:
            path = rs.weights_for(cfg, label)
            lf.write(f"\n=== {label} :: {path.name} (fresh process) ===\n")
            lf.flush()
            proc = rs.start_server(rs.BACKENDS["vulkan"], path, PORT, THREADS,
                                   "vulkan")
            rs.verify_serving(PORT, path)
            try:
                r = subprocess.run(
                    [str(rs.PY), str(rs.SRC / "probe_cache_crosssession.py"),
                     "--model-label", label, "--port", str(PORT)],
                    stdout=lf, stderr=subprocess.STDOUT, text=True)
                lf.write(f"--- exit {r.returncode} ---\n")
                lf.flush()
                rc = rc or r.returncode
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(4)
    print(f"cross-session probe complete (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
