"""Re-run the tokenizer probes against the current condition set.

WHY. `condition_tokens_<model>.json` on disk covers D0-D7. It was written before
the position controls D8, D9 and D10 existed, and those three are the whole
reason the mechanism result can separate "a delimiter was destroyed" from "the
name moved". A paper that says every condition's token indices were verified
needs an artifact that covers every condition, so the probe is re-run rather
than the claim softened.

The probe only calls /tokenize and /detokenize, so it is cheap; the cost is
loading the weights.

    .venv/Scripts/python.exe paper-a/src/run_token_probes.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_suite as rs  # noqa: E402

MODELS = ["llama-3.1-8b-instruct", "llama-2-7b-chat",
          "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
PROBES = ["probe_name_length.py"]
# Probes that only make sense on the reference checkpoint; run separately.
REFERENCE_ONLY = ["probe_condition_tokens.py", "probe_token_identity.py",
                  "probe_tokenization.py"]
REFERENCE = "llama-3.1-8b-instruct"
PORT = 8080
THREADS = 4


def main() -> int:
    cfg = yaml.safe_load(rs.CONFIG.read_text(encoding="utf-8"))
    log = rs.ROOT / "paper-a" / "data" / "logs" / "token_probes.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    rc = 0
    with log.open("a", encoding="utf-8") as lf:
        for label in MODELS:
            path = rs.weights_for(cfg, label)
            lf.write(f"\n=== {label} :: {path.name} ===\n")
            lf.flush()
            proc = rs.start_server(rs.BACKENDS["vulkan"], path, PORT, THREADS,
                                   "vulkan")
            rs.verify_serving(PORT, path)
            try:
                probes = list(PROBES)
                if label == REFERENCE:
                    probes += REFERENCE_ONLY
                for probe in probes:
                    r = subprocess.run(
                        [str(rs.PY), str(rs.SRC / probe),
                         "--model-label", label, "--port", str(PORT)],
                        stdout=lf, stderr=subprocess.STDOUT, text=True)
                    lf.write(f"--- {probe} exit {r.returncode} ---\n")
                    lf.flush()
                    rc = rc or r.returncode
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(4)
    print(f"token probes complete (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
