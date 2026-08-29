"""Driver for the round-4 experiment suite: declarative plan, server lifecycle,
resumable, unattended.

WHY A PLAN TABLE RATHER THAN A SHELL SCRIPT. Three studies across six
checkpoints and two inference modes is nineteen model-runs. Written as a
sequence of commands, a scheduling mistake -- the wrong weights on the wrong
port, a model run twice, a model silently skipped -- is invisible until the
analysis produces something odd. Written as a table, the schedule is one object
that can be printed, diffed and checked before anything starts.

TWO STREAMS. The machine has one integrated GPU and four cores. A Vulkan server
and a CPU server can run concurrently without contending for the same resource;
two Vulkan servers cannot, because they share 7.4 GiB of GPU-visible memory.
Streams are therefore keyed to a backend and a port:

    STREAM V   vulkan, port 8080   ~3.3 s/cell
    STREAM C   cpu,    port 8081   ~15 s/cell

The slow stream gets the fewest cells. Whichever finishes first can be given
more work; nothing in the plan depends on the two finishing together.

    Start-Process powershell -ArgumentList '-Command','... run_suite.py --stream V'

`Start-Process` and not `nohup`: nohup does not detach from the Windows process
group, so an earlier run died with the terminal that launched it.

    .venv/Scripts/python.exe paper-a/src/run_suite.py --stream V --dry-run
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

import requests
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "paper-a" / "src"
CONFIG = ROOT / "paper-a" / "config.yaml"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
BACKENDS = {
    "vulkan": ROOT / "tools" / "llamacpp-vulkan" / "llama-server.exe",
    "cpu": ROOT / "tools" / "llamacpp-cpu" / "llama-server.exe",
}
VULKAN_BUDGET = 7_100_000_000

# --------------------------------------------------------------------------
# The plan.
#
# Each job is (study, model_label, weights_key, extra_args). `weights_key` names
# an entry in config.yaml; "<id>@q8" selects the Q8_0 file from
# robustness_quantization instead of the primary Q4_K_M file.
#
# Ordering within a stream is deliberate: the studies that answer the sharpest
# questions run first, so a stream stopped early still yields a result. On
# stream V that is the base-versus-instruct mechanism comparison; on stream C it
# is the name grid, which is the single biggest gap.
# --------------------------------------------------------------------------
JOBS = {
    # ----------------------------------------------------------------------
    # STREAM A. Everything, on the Vulkan backend, one job at a time.
    #
    # This replaces an earlier two-stream schedule that ran a Vulkan server and
    # a CPU server concurrently. Measured, that was SLOWER than one stream:
    #
    #     vulkan alone           ~4.0 s/cell      0.250 cells/s
    #     vulkan + cpu together   7.1 and 80 s/cell   0.155 cells/s combined
    #
    # The CPU server takes four threads that the Vulkan server needs for
    # sampling, grammar evaluation and the host side of every dispatch, so the
    # second stream costs the first more than it contributes. On this machine
    # concurrency across backends is a pessimisation, and the measurement is
    # recorded here rather than in a commit message because it is the kind of
    # thing that gets re-litigated.
    #
    # Order is by decisiveness, so a stream stopped early still answers the
    # sharpest question. The base-versus-instruct pair leads: it is the only
    # comparison in the panel that isolates instruction tuning from everything
    # else, and it is the hypothesis Study 3 could not test.
    # ----------------------------------------------------------------------
    "A": [
        ("mech-raw", "mistral-7b-v0.1-base", "mistral-7b-v0.1-base", []),
        ("mech-raw", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("mech-chat", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("mech-chat", "mistral-7b-v0.1-base", "mistral-7b-v0.1-base", []),
        ("mech-raw", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("mech-chat", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("mech-chat", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("mech-raw", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("names", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("names", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("names", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("names", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        ("mech-chat", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        ("mech-raw", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        ("coverage", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("coverage", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("coverage", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        ("coverage", "llama-2-7b-chat", "llama-2-7b-chat", []),
        # Q8_0 weights exceed the Vulkan budget and run partially offloaded.
        ("quant", "llama-2-7b-chat-q8", "llama-2-7b-chat@q8", []),
        ("quant", "mistral-7b-instruct-v0.1-q8", "mistral-7b-instruct-v0.1@q8", []),
        # 13B also runs partially offloaded, so it goes last.
        ("mech-chat", "llama-2-13b-chat", "llama-2-13b-chat", []),
        ("mech-raw", "llama-2-13b-chat", "llama-2-13b-chat", []),
        ("coverage", "llama-2-13b-chat", "llama-2-13b-chat", []),
        # T2_mid arm. Added after the first panel pass, because reconciling it
        # against Study 3 showed Study 3's headline delimiter effect was
        # concentrated in T2_mid -- the one template the first panel pass did
        # not carry. Running it completes the template factor so the question
        # "is the mechanism template-specific?" becomes answerable instead of
        # being decided by which resumes happened to be in each design.
        ("mech-chat-t2", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("mech-raw-t2", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("mech-chat-t2", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("mech-raw-t2", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("mech-chat-t2", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("mech-raw-t2", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("mech-chat-t2", "mistral-7b-v0.1-base", "mistral-7b-v0.1-base", []),
        ("mech-raw-t2", "mistral-7b-v0.1-base", "mistral-7b-v0.1-base", []),
        ("mech-chat-t2", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        ("mech-raw-t2", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        # Study 4 completed to all three templates, so Study 4 and Study 2 span
        # the same resume set and the name-variance estimate is not itself
        # conditional on a template choice.
        ("names-t2", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("names-t2", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("names-t2", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("names-t2", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        # Study 7. Closes G7, the paper's largest stated limitation: everything
        # else is measured on one job posting in one occupation. Two further
        # occupations, structurally matched to the first and chosen to span the
        # gender typing of the labour market.
        ("occupation", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("occupation", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("occupation", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("occupation", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        # Re-queued: this job's server exited at ngl=31 on the first pass and
        # the arm lost a model. start_server now backs the offload off until
        # the checkpoint actually loads.
        ("quant", "llama-2-7b-chat-q8", "llama-2-7b-chat@q8", []),
        # The 13B was missing from the T2 arm, so the template factor was
        # complete for five checkpoints and not for the sixth.
        ("mech-chat-t2", "llama-2-13b-chat", "llama-2-13b-chat", []),
        ("mech-raw-t2", "llama-2-13b-chat", "llama-2-13b-chat", []),
        # Study 8. The noise floor currently rests on an accidental replicate
        # and asserts a cause it has not tested. Running the identical cell
        # five times at concurrency 4 and at concurrency 1 tests whether
        # batching is responsible, and yields a proper variance estimate.
        ("replicate", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("replicate", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("replicate", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("replicate", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        # ------------------------------------------------------------------
        # D9 RE-MEASUREMENT. The base arm of the position conditions was
        # written while D9's definition was being corrected in the working
        # tree, and nothing recorded which definition produced which row.
        # These jobs re-measure exactly those cells into a separate file so
        # the original values can be compared rather than assumed. They run
        # last because everything else is already trustworthy.
        # ------------------------------------------------------------------
        ("d9recheck-chat", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("d9recheck-raw", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("d9recheck-chat", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("d9recheck-raw", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("d9recheck-chat", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("d9recheck-raw", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("d9recheck-chat", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        ("d9recheck-raw", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        ("d9recheck-chat", "mistral-7b-v0.1-base", "mistral-7b-v0.1-base", []),
        ("d9recheck-raw", "mistral-7b-v0.1-base", "mistral-7b-v0.1-base", []),
        ("d9recheck-chat", "llama-2-13b-chat", "llama-2-13b-chat", []),
        ("d9recheck-raw", "llama-2-13b-chat", "llama-2-13b-chat", []),
    ],
    "V": [
        # G2, decisive row first: identical pretraining, one differs by tuning.
        ("mech-raw", "mistral-7b-v0.1-base", "mistral-7b-v0.1-base", []),
        ("mech-raw", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("mech-chat", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("mech-raw", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("mech-chat", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("mech-chat", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        ("mech-raw", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        # G1, the name grid.
        ("names", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("names", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        ("names", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        # Scale control. 13B needs partial offload, so it goes last.
        ("mech-chat", "llama-2-13b-chat", "llama-2-13b-chat", []),
        ("mech-raw", "llama-2-13b-chat", "llama-2-13b-chat", []),
        ("mech-chat", "mistral-7b-v0.1-base", "mistral-7b-v0.1-base", []),
    ],
    "C": [
        ("names", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("mech-chat", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("mech-raw", "llama-2-7b-chat", "llama-2-7b-chat", []),
        # G3, quantization. Same design as Study 2, Q8_0 weights, separate dir.
        ("quant", "llama-2-7b-chat-q8", "llama-2-7b-chat@q8", []),
        ("quant", "mistral-7b-instruct-v0.1-q8", "mistral-7b-instruct-v0.1@q8", []),
    ],
    # Probe pass. Cheap (48 calls per model) and independent of the two main
    # streams, so it runs on whichever backend is free once they finish. It
    # exists because Study 2 turned up a 21-26% disagreement between the
    # grammar-constrained verdict and the summed-mass margin, and the paper
    # cannot report that number without knowing whether it is a property of the
    # instrument or a hole in the hand-written token sets.
    "P": [
        ("coverage", "llama-3.1-8b-instruct", "llama-3.1-8b-instruct", []),
        ("coverage", "mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1", []),
        ("coverage", "mistral-7b-instruct-v0.3", "mistral-7b-instruct-v0.3", []),
        ("coverage", "llama-2-7b-chat", "llama-2-7b-chat", []),
        ("coverage", "llama-2-13b-chat", "llama-2-13b-chat", []),
    ],
}

STREAM_BACKEND = {"A": ("vulkan", 8080), "V": ("vulkan", 8080),
                  "C": ("cpu", 8081), "P": ("vulkan", 8082)}

# --------------------------------------------------------------------------
# What "complete" means for each study, so a job that has nothing left to do is
# skipped WITHOUT loading its checkpoint.
#
# Loading a 4-8 GB model to discover there are zero cells to run costs one to
# two minutes. Fourteen already-complete jobs at the head of the plan therefore
# cost twenty minutes on every restart, which is how a spurious stall became a
# self-sustaining restart loop. The counts are derived from the experiment
# modules rather than typed, so they cannot drift from the designs they test.
# --------------------------------------------------------------------------
def _expectations():
    sys.path.insert(0, str(SRC))
    import stimuli as st
    from experiment_mechanism import CONDITIONS
    from experiment_delta_stability import PAIRS, VARIANTS
    nc, nv, nt = len(CONDITIONS), len(VARIANTS), len(st.TEMPLATES)
    npair, nmech, nold = len(st.NAME_GRID), len(st.MECH_GRID), len(PAIRS)
    # A base job runs the TWO extreme templates; its -t2 partner adds the
    # third and so completes the file. Giving them the same expectation made
    # the base job load a checkpoint to run zero cells, because the file was
    # incomplete overall while the arm that job owns was finished. On the 13B,
    # partially offloaded, that is five minutes twice over.
    n_base_t = len(__import__("stimuli").TEMPLATES_2)
    return {
        "names":         ("names", "names_{m}.jsonl", nv * n_base_t * npair),
        "names-t2":      ("names", "names_{m}.jsonl", nv * nt * npair),
        "mech-chat":     ("mechanism_panel", "mech_chat_{m}.jsonl", nc * n_base_t * nmech),
        "mech-chat-t2":  ("mechanism_panel", "mech_chat_{m}.jsonl", nc * nt * nmech),
        "mech-raw":      ("mechanism_panel", "mech_raw_{m}.jsonl", nc * n_base_t * nmech),
        "mech-raw-t2":   ("mechanism_panel", "mech_raw_{m}.jsonl", nc * nt * nmech),
        "quant":         ("quantization", "delta_{m}.jsonl", nv * nt * nold),
        "occupation":    ("occupation", "occ_{m}.jsonl", 2 * nv * nt * nold),
        "replicate":     ("replicate", "rep_{m}.jsonl", 2 * 5 * nt * nold),
        # one condition, the two base templates, over the mechanism grid
        # two conditions (D9 under test, D8 as the never-changed control),
        # the two base templates, over the mechanism grid
        "d9recheck-chat": ("mechanism_panel", "mech_d9recheck_chat_{m}.jsonl",
                           2 * n_base_t * nmech),
        "d9recheck-raw":  ("mechanism_panel", "mech_d9recheck_raw_{m}.jsonl",
                           2 * n_base_t * nmech),
    }


EXPECTED = _expectations()
# -t2 jobs now carry their own expectation, so they are no longer aliases for
# completion purposes. The map is kept because the watchdog groups by output
# file and needs to know which entries share one.
ALIAS = {"names-t2": "names", "mech-chat-t2": "mech-chat",
         "mech-raw-t2": "mech-raw"}
PROBE = {"coverage": ("instrument", "token_coverage_{m}.json")}


def job_is_complete(study: str, label: str) -> bool:
    from stimuli import read_jsonl
    # The job's OWN entry wins. Resolving through ALIAS first made a `-t2` job
    # inherit its base job's smaller expectation, so `names-t2` at 1500 cells
    # was judged complete against the base target of 1152 and skipped -- which
    # would have silently left every T2 arm unrun.
    key = study if study in EXPECTED else ALIAS.get(study, study)
    data = ROOT / "paper-a" / "data"
    if key in EXPECTED:
        folder, pat, exp = EXPECTED[key]
        f = data / folder / pat.format(m=label)
        if not f.exists():
            return False
        n = sum(1 for r in read_jsonl(f)
                if r.get("white_margin") is not None
                and r.get("black_margin") is not None)
        return n >= exp
    if study in PROBE:
        folder, pat = PROBE[study]
        return (data / folder / pat.format(m=label)).exists()
    return False

STUDIES = {
    "names": (SRC / "experiment_names.py", []),
    "mech-chat": (SRC / "experiment_mechanism_panel.py", ["--mode", "chat"]),
    "mech-raw": (SRC / "experiment_mechanism_panel.py", ["--mode", "raw"]),
    "quant": (SRC / "experiment_delta_stability.py",
              ["--out-dir", "paper-a/data/quantization"]),
    "coverage": (SRC / "probe_token_coverage.py", ["--n-cells", "48"]),
    "mech-chat-t2": (SRC / "experiment_mechanism_panel.py",
                     ["--mode", "chat", "--templates", "T2_mid"]),
    "mech-raw-t2": (SRC / "experiment_mechanism_panel.py",
                    ["--mode", "raw", "--templates", "T2_mid"]),
    "names-t2": (SRC / "experiment_names.py", ["--templates", "T2_mid"]),
    "occupation": (SRC / "experiment_occupation.py", []),
    "replicate": (SRC / "experiment_replicate.py", ["--repeats", "5"]),
    "d9recheck-chat": (SRC / "experiment_mechanism_panel.py",
                       ["--mode", "chat", "--conditions", "D8,D9",
                        "--templates", "T1_strong,T3_marginal",
                        "--out-suffix", "d9recheck"]),
    "d9recheck-raw": (SRC / "experiment_mechanism_panel.py",
                      ["--mode", "raw", "--conditions", "D8,D9",
                       "--templates", "T1_strong,T3_marginal",
                       "--out-suffix", "d9recheck"]),
}


def weights_for(cfg, key: str) -> pathlib.Path:
    if key.endswith("@q8"):
        base = key[:-3]
        for e in cfg.get("robustness_quantization", []):
            if e["id"] == base:
                return ROOT / e["file"]
        raise KeyError(f"no Q8_0 entry for {base}")
    for m in cfg["models"]:
        if m["id"] == key:
            return ROOT / m["file"]
    raise KeyError(key)


def port_is_free(port: int) -> bool:
    """True if nothing answers /health on `port`.

    A stale llama-server from a previous session survived a terminal closing and
    kept port 8080 bound while serving DIFFERENT weights. A new server launched
    onto that port fails to bind and exits, the health check then succeeds
    against the old process, and the run proceeds against the wrong checkpoint
    while looking entirely normal. verify_serving catches it after the fact;
    this catches it before anything is loaded.
    """
    try:
        requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
        return False
    except Exception:  # noqa: BLE001
        return True


def _try_start(server, model_path, port, threads, ngl, wait_s):
    proc = subprocess.Popen(
        [str(server), "-m", str(model_path), "-ngl", str(ngl), "-c", "4096",
         "--host", "127.0.0.1", "--port", str(port), "-t", str(threads),
         "--parallel", "4", "--no-webui"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early ({proc.returncode}) at ngl={ngl}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health",
                            timeout=3).json().get("status") == "ok":
                return proc
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    proc.kill()
    raise RuntimeError(f"never became healthy at ngl={ngl}")


def start_server(server, model_path, port, threads, backend, wait_s=1200):
    """Launch a server, backing off the GPU offload until it actually loads.

    WHY A LADDER RATHER THAN A FORMULA. The offload count was computed as
    `40 * (budget / size) * 0.8`, which assumes 40 layers and a fixed headroom
    fraction. For llama-2-7b-chat at Q8_0 -- 7.16 GB against a 7.42 GB
    GPU-visible budget -- that yields 31 of the model's 32 layers, leaving
    nothing for the KV cache or the compute buffers, and the server exited with
    status 1 before serving a single request. The job was skipped and the
    quantization arm lost a model, silently apart from one line in a log.

    A formula cannot know the layer count, the KV-cache size at this context
    length, or how much of the iGPU's shared memory the desktop is already
    using. Retrying with less offload can, and it degrades to CPU-only rather
    than to nothing.
    """
    if not port_is_free(port):
        raise RuntimeError(
            f"port {port} is already answering /health. A stale llama-server is "
            f"holding it; kill it before starting this stream. Refusing to "
            f"launch, because a failed bind would leave the old checkpoint "
            f"answering every call.")
    size = model_path.stat().st_size
    if backend != "vulkan":
        ladder = [0]
    elif size <= VULKAN_BUDGET:
        ladder = [99, 24, 12, 0]
    else:
        first = max(1, int(32 * (VULKAN_BUDGET / size) * 0.70))
        ladder = [first, int(first * 0.6), int(first * 0.3), 0]
    ladder = [n for i, n in enumerate(ladder) if n not in ladder[:i]]

    last = None
    for ngl in ladder:
        try:
            proc = _try_start(server, model_path, port, threads, ngl, wait_s)
            if ngl != ladder[0]:
                print(f"  [offload] fell back to -ngl {ngl} for {model_path.name}",
                      flush=True)
            return proc
        except RuntimeError as e:
            last = e
            print(f"  [retry] {model_path.name}: {e}", flush=True)
            time.sleep(4)
    raise RuntimeError(f"server never became healthy for {model_path.name}: {last}")


def verify_serving(port: int, expected: pathlib.Path) -> None:
    """Exact-filename check, stricter than the per-experiment prefix guard.

    The experiment scripts compare a normalised 14-character prefix of the model
    label against the served filename. That is enough to catch a stream serving
    an entirely different family, and NOT enough to catch Q8_0 weights being
    labelled Q4_K_M, or v0.1 being labelled v0.3, because the prefix matches in
    both cases. The driver knows exactly which file it asked for, so it checks
    the whole basename.
    """
    j = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=15).json()
    served = ""
    for key in ("data", "models"):
        if isinstance(j.get(key), list) and j[key]:
            e = j[key][0]
            served = e.get("id") or e.get("model") or e.get("name") or ""
            break
    if pathlib.Path(served).name != expected.name:
        raise RuntimeError(
            f"REFUSING: port {port} serves '{pathlib.Path(served).name}', "
            f"driver loaded '{expected.name}'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, choices=sorted(JOBS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on 'study/model'")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    backend, port = STREAM_BACKEND[args.stream]
    jobs = JOBS[args.stream]
    if args.only:
        jobs = [j for j in jobs if args.only in f"{j[0]}/{j[1]}"]

    print(f"[stream {args.stream}] backend={backend} port={port} "
          f"{len(jobs)} job(s)", flush=True)
    for i, (study, label, wkey, _) in enumerate(jobs, 1):
        p = weights_for(cfg, wkey)
        mark = "" if p.exists() else "   *** WEIGHTS MISSING ***"
        print(f"  {i:2d}. {study:<10} {label:<28} {p.name}{mark}", flush=True)
    if args.dry_run:
        return 0

    t_all = time.time()
    for i, (study, label, wkey, extra) in enumerate(jobs, 1):
        script, base_args = STUDIES[study]
        path = weights_for(cfg, wkey)
        if not path.exists():
            print(f"[skip] {study}/{label}: weights missing", flush=True)
            continue
        if job_is_complete(study, label):
            print(f"[skip] {i}/{len(jobs)} {study}/{label}: already complete, "
                  f"not loading the checkpoint", flush=True)
            continue
        print(f"\n{'='*70}\n[{i}/{len(jobs)}] {study}  {label}\n{'='*70}", flush=True)
        try:
            proc = start_server(BACKENDS[backend], path, port, args.threads, backend)
        except RuntimeError as e:
            print(f"  [LOAD FAILED] {e}", flush=True)
            continue
        try:
            verify_serving(port, path)
            subprocess.run(
                [str(PY), str(script), "--model-label", label,
                 "--port", str(port)] + base_args + extra,
                cwd=str(ROOT), check=False)
        except Exception as e:  # noqa: BLE001
            print(f"  [ABORTED] {e}", flush=True)
        finally:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3)

    print(f"\n[stream {args.stream}] complete in "
          f"{(time.time()-t_all)/3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
