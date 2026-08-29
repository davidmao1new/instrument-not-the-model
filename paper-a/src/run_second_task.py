"""Drive the second-task experiment across every checkpoint and both domains.

Serves one checkpoint at a time, runs both domains against it, stops the server
and moves on -- rather than serving each checkpoint twice -- because loading a
4 GB checkpoint costs more than the domain switch does.

ORDER IS DELIBERATE. Domains alternate which goes first across models, so that
if the run is stopped early the surviving data is not all one domain on the
early models. Within a model the domain order is fixed by the model's position
so the schedule is deterministic and a resume produces the same plan.

Resumable at the cell level: `experiment_second_task.py` reads back whatever it
already wrote and skips those cells, so this driver can be killed and restarted
without losing or duplicating work.

    C:/research-toolchain/venv/Scripts/python.exe paper-a/src/run_second_task.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import localserve as ls  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "paper-a" / "src"
PY = sys.executable
PORT = 8090

MODELS = ["llama-2-7b-chat", "llama-3.1-8b-instruct",
          "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.3"]
DOMAINS = ["housing", "moderation"]


def main() -> int:
    t0 = time.time()
    for i, label in enumerate(MODELS):
        order = DOMAINS if i % 2 == 0 else list(reversed(DOMAINS))
        print(f"\n{'=' * 78}\n[{i + 1}/{len(MODELS)}] {label}   "
              f"domains: {', '.join(order)}\n{'=' * 78}", flush=True)
        proc = None
        try:
            proc, weights = ls.start(label, port=PORT, parallel=1)
            print(f"  serving {weights.name}", flush=True)
            for dom in order:
                print(f"  --- {label} / {dom} ---", flush=True)
                r = subprocess.run(
                    [PY, str(SRC / "experiment_second_task.py"),
                     "--model-label", label, "--domain", dom,
                     "--port", str(PORT)],
                    cwd=str(ROOT))
                if r.returncode != 0:
                    print(f"  !! {label}/{dom} exited {r.returncode}; "
                          f"continuing to the next cell rather than aborting "
                          f"the whole schedule", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  !! could not serve {label}: {e}", flush=True)
        finally:
            ls.stop(proc)
        print(f"  elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
