"""Start a llama-server for one checkpoint, from wherever the binaries live.

WHY THIS EXISTS AND WHY IT IS NOT A REFACTOR OF run_suite.py. The repository
moved onto a Google Drive virtual filesystem between sessions. Data, code and
artifacts read and write there fine. Executables do not run there, and a GGUF
that a server memory-maps is on the wrong side of a network filesystem. So the
things that must EXECUTE now live on local disk and everything else stays in the
repository, which is the arrangement the paper's reproducibility statement has
to describe anyway: the checkpoint is identified by its SHA-256, not its path.

run_suite.py and panel_gate.py are left alone deliberately. They produced every
confirmatory number in the paper and they are covered by the test suite; editing
their path resolution to chase a filesystem move would put that at risk for no
scientific gain. New probes use this module instead.

RESOLUTION ORDER for a model file, first hit wins:
  1. $RESEARCH_MODEL_DIR/<basename from config.yaml>
  2. <repo>/<path from config.yaml>            (the original layout)
Server binary:
  1. $RESEARCH_LLAMA_DIR/llama-server.exe
  2. <repo>/tools/llamacpp-vulkan/llama-server.exe

THE HASH IS CHECKED EVERY TIME. config.yaml pins each checkpoint by SHA-256 and
the whole point of the pin is that a file found at a new path is still provably
the same weights. A mismatch raises rather than warns.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import time

import requests
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = ROOT / "paper-a" / "config.yaml"

_HASH_CACHE: dict[str, str] = {}


def _sha256(p: pathlib.Path) -> str:
    key = f"{p}|{p.stat().st_size}|{p.stat().st_mtime_ns}"
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    _HASH_CACHE[key] = h.hexdigest()
    return _HASH_CACHE[key]


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def model_entry(label: str) -> dict:
    cfg = load_config()
    for m in cfg["models"]:
        if m["id"] == label:
            return m
    raise KeyError(f"{label} is not a model id in config.yaml")


def resolve_weights(label: str, verify: bool = True) -> pathlib.Path:
    e = model_entry(label)
    rel = pathlib.Path(e["file"])
    candidates = []
    env = os.environ.get("RESEARCH_MODEL_DIR")
    if env:
        candidates.append(pathlib.Path(env) / rel.name)
    candidates.append(ROOT / rel)
    for c in candidates:
        if c.exists():
            if verify:
                got = _sha256(c)
                if got != e["sha256"]:
                    raise RuntimeError(
                        f"{label}: SHA-256 mismatch at {c}\n"
                        f"  config.yaml pins {e['sha256']}\n"
                        f"  file on disk is  {got}\n"
                        f"This is a different checkpoint. Refusing to serve it.")
            return c
    raise FileNotFoundError(
        f"{label}: no weights found. Looked in: "
        + ", ".join(str(c) for c in candidates))


def server_binary() -> pathlib.Path:
    env = os.environ.get("RESEARCH_LLAMA_DIR")
    candidates = []
    if env:
        candidates.append(pathlib.Path(env) / "llama-server.exe")
    candidates.append(ROOT / "tools" / "llamacpp-vulkan" / "llama-server.exe")
    candidates.append(ROOT / "tools" / "llamacpp-cpu" / "llama-server.exe")
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("no llama-server.exe found in "
                            + ", ".join(str(c) for c in candidates))


def port_is_free(port: int) -> bool:
    try:
        requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        return False
    except Exception:  # noqa: BLE001
        return True


def start(label: str, port: int = 8080, ctx: int = 4096, threads: int = 8,
          ngl: int = 99, parallel: int = 1, wait_s: int = 900):
    """Launch a server for `label` and block until /health is ok.

    `parallel=1` is half of what §5.2 established as the configuration under
    which the measurement reproduces bitwise, and it is the default here so a
    new script does not inherit the nondeterminism the paper documents.

    THE OTHER HALF IS NOT A SERVER FLAG. Prompt-cache reuse is disabled
    per-REQUEST, by sending `cache_prompt: false` in the body -- there is no
    `--no-prompt-cache` on this build, and passing one makes the server exit 1
    with no useful message. `probe_cache_crosssession.py` does it the right way
    and this docstring exists so the next caller does not rediscover it.
    """
    if not port_is_free(port):
        raise RuntimeError(f"port {port} already answers /health; kill the "
                           f"stale server before launching another.")
    weights = resolve_weights(label)
    args = [str(server_binary()), "-m", str(weights), "-ngl", str(ngl),
            "-c", str(ctx), "--host", "127.0.0.1", "--port", str(port),
            "-t", str(threads), "--parallel", str(parallel), "--no-webui"]
    ladder = [ngl, 24, 12, 0] if ngl == 99 else [ngl, 0]
    last = None
    for attempt in ladder:
        args[args.index("-ngl") + 1] = str(attempt)
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if proc.poll() is not None:
                last = f"exited with code {proc.returncode} at -ngl {attempt}"
                break
            try:
                if requests.get(f"http://127.0.0.1:{port}/health",
                                timeout=3).json().get("status") == "ok":
                    return proc, weights
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        else:
            proc.kill()
            last = f"did not become healthy within {wait_s}s at -ngl {attempt}"
        time.sleep(2)
    raise RuntimeError(f"could not serve {label}: {last}")


def stop(proc) -> None:
    if proc is None:
        return
    proc.kill()
    try:
        proc.wait(timeout=30)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(2)


def tokenize(port: int, text: str) -> list[int]:
    r = requests.post(f"http://127.0.0.1:{port}/tokenize",
                      json={"content": text}, timeout=120)
    r.raise_for_status()
    return r.json()["tokens"]
