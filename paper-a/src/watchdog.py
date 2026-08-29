"""Unattended supervisor for the experiment suite.

This process is the thing that has to survive everything the last three days
threw at the runs. In order of how much each one cost:

  MACHINE REBOOT        an unexpected shutdown at 16:04 killed a stream and it
                        stayed dead for half an hour until a human looked.
  SERVER FAILS TO LOAD  the Q8 checkpoint exited at load, the driver logged one
                        line, skipped the job, and the quantization arm silently
                        lost a model.
  STALE SERVER          a llama-server outlived its terminal, held the port, and
                        answered health checks with the WRONG weights.
  STALL                 a server that is alive but no longer producing, which no
                        exit code reports.
  TRUNCATED WRITE       an interrupted run leaves a partial final line.

The design principle is that the watchdog fixes what it can recognise and
ESCALATES what it cannot, rather than trying to be clever about novel failures.
Escalation means: write a machine-readable alert and exit non-zero, so whatever
is waiting on this process is woken and a human or an agent looks. Silent
best-effort recovery is how a run ends up complete and wrong.

    python watchdog.py            supervise until the plan is complete
    python watchdog.py --status   print the current state and exit
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stimuli as st  # noqa: E402
from experiment_mechanism import CONDITIONS  # noqa: E402
from experiment_delta_stability import PAIRS, VARIANTS  # noqa: E402
import run_suite  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "paper-a" / "data"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
STATUS = ROOT / "watchdog_status.json"
ALERT = ROOT / "watchdog_ALERT.json"

POLL_S = 120
# A stall is no cell growth AND no log growth. Cell growth alone is the wrong
# signal: coverage probes write JSON rather than JSONL, a 13B checkpoint takes
# minutes to load, and an already-complete job produces nothing at all. Watching
# only cells, the supervisor called those a stall, restarted, and the restart
# re-ran the very work that had produced no cells -- for longer than the stall
# window, so each restart guaranteed the next. Two hours were lost to that loop.
STALL_MIN = 45
MAX_RESTARTS = 8
MIN_FREE_GB = 4.0

# --------------------------------------------------------------------------
# Expected cell counts, derived from the modules rather than typed, so the
# completion test cannot drift from the design it is testing.
# --------------------------------------------------------------------------
N_COND = len(CONDITIONS)
N_VAR = len(VARIANTS)
N_TMPL = len(st.TEMPLATES)
N_NAMEPAIR = len(st.NAME_GRID)
N_MECHPAIR = len(st.MECH_GRID)
N_OLDPAIR = len(PAIRS)

# Single source of truth: run_suite owns the expectations, so the supervisor
# and the driver cannot disagree about what "complete" means.
EXPECTED = run_suite.EXPECTED
ALIAS = run_suite.ALIAS
PROBE = run_suite.PROBE

FAILURE_PATTERNS = [
    (re.compile(r"\[LOAD FAILED\]"), "server failed to load", True),
    (re.compile(r"\[ABORTED\]"), "job aborted mid-run", True),
    (re.compile(r"REFUSING"), "contamination guard fired", False),
    (re.compile(r"corrupt and is not the last line"), "corrupt data file", False),
    (re.compile(r"MemoryError|OSError|No space left"), "resource exhaustion", False),
]


def _count_matching(needle: str) -> int:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             f"Where-Object {{ $_.CommandLine -like '*{needle}*' }} | "
             "Measure-Object).Count"],
            capture_output=True, text=True, timeout=60)
        return int((out.stdout or "0").strip() or 0)
    except Exception:  # noqa: BLE001
        return 0


LOCK = ROOT / "watchdog.lock"
HEARTBEAT_STALE_S = 420


def another_watchdog_running() -> bool:
    """True if a live watchdog other than this process already holds the lock.

    WHY NOT A PROCESS-NAME COUNT. The first version counted python processes
    whose command line contained "watchdog.py". That is wrong twice over: any
    process that merely MENTIONS the filename matches, including a one-liner
    that imports this module, and two scheduled-task firings a moment apart
    each see the other and both exit, leaving nothing supervising. Both
    happened.

    A lock file carrying a PID and a heartbeat is deterministic. A stale lock
    -- from a process killed by a session teardown, which cannot clean up after
    itself -- is detected by the heartbeat going cold and is taken over rather
    than blocking recovery forever.
    """
    if not LOCK.exists():
        return False
    try:
        d = json.loads(LOCK.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if int(d.get("pid", -1)) == os.getpid():
        return False
    if time.time() - float(d.get("heartbeat", 0)) > HEARTBEAT_STALE_S:
        print(f"  [lock] taking over from pid {d.get('pid')}: heartbeat is "
              f"{(time.time()-float(d.get('heartbeat', 0)))/60:.0f} min old",
              flush=True)
        return False
    return _pid_alive(int(d.get("pid", -1)))


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue | "
             "Measure-Object).Count"],
            capture_output=True, text=True, timeout=60)
        return int((out.stdout or "0").strip() or 0) > 0
    except Exception:  # noqa: BLE001
        return False


def touch_lock():
    LOCK.write_text(json.dumps(dict(pid=os.getpid(), heartbeat=time.time())),
                    encoding="utf-8")


def suite_running() -> bool:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*run_suite*' } | "
             "Measure-Object).Count"],
            capture_output=True, text=True, timeout=60)
        return int((out.stdout or "0").strip() or 0) > 0
    except Exception:  # noqa: BLE001
        return False


def kill_all():
    for name in ("llama-server",):
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Stop-Process -Name {name} -Force -ErrorAction SilentlyContinue"],
                       capture_output=True, timeout=60)
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*run_suite*' -or "
         "$_.CommandLine -like '*experiment_*' } | "
         "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
        capture_output=True, timeout=60)
    time.sleep(5)


def newest_log() -> pathlib.Path | None:
    logs = sorted(ROOT.glob("logs_suite_A*.txt"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def start_suite() -> pathlib.Path:
    n = len(list(ROOT.glob("logs_suite_A*.txt"))) + 1
    log = ROOT / f"logs_suite_A{n}.txt"
    err = ROOT / f"logs_suite_A{n}.err"
    # NO PIPES. Two bugs lived on this line, in sequence, and both disabled the
    # supervisor while leaving it apparently alive:
    #
    #   1. `capture_output` was passed to subprocess.Popen, which does not
    #      accept it. The first recovery attempt raised TypeError and the
    #      watchdog exited.
    #   2. Switching to subprocess.run(capture_output=True) fixed the crash and
    #      introduced a deadlock. capture_output makes Python read the child's
    #      stdout and stderr until EOF, the launched suite inherits those pipe
    #      handles, and so the call does not return until the ENTIRE SUITE
    #      exits. The watchdog sat inside its own restart path for eleven
    #      minutes without writing a status line, supervising nothing.
    #
    # Both are avoided by giving the launcher no pipes to inherit and not
    # waiting on it at all. PowerShell's Start-Process already redirects the
    # suite's own output to files.
    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command",
         f"Start-Process -FilePath '{PY}' "
         f"-ArgumentList 'paper-a\\src\\run_suite.py','--stream','A','--threads','6' "
         f"-RedirectStandardOutput '{log.name}' -RedirectStandardError '{err.name}' "
         f"-WindowStyle Hidden"],
        cwd=str(ROOT), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    time.sleep(15)          # let the server begin loading before polling health
    return log


def count_complete(folder, pattern, model) -> int:
    p = DATA / folder / pattern.format(m=model)
    if not p.exists():
        return 0
    n = 0
    for r in st.read_jsonl(p):
        if r.get("white_margin") is not None and r.get("black_margin") is not None:
            n += 1
    return n


def plan_status():
    """One row per job in the plan, with completion against the design."""
    rows = []
    seen = set()
    for study, label, wkey, _ in run_suite.JOBS["A"]:
        base = ALIAS.get(study, study)
        if study in EXPECTED or base in EXPECTED:
            # a file is complete only when its LARGEST claimant is satisfied
            folder, pat, _ = EXPECTED.get(study, EXPECTED.get(base))
            exp = max(e for k, (f, p_, e) in EXPECTED.items()
                      if f == folder and p_ == pat)
            key = (folder, pat.format(m=label))
            if key in seen:
                continue
            seen.add(key)
            got = count_complete(folder, pat, label)
            rows.append(dict(study=base, model=label, got=got, expected=exp,
                             done=got >= exp))
        elif study in PROBE:
            folder, pat = PROBE[study]
            p = DATA / folder / pat.format(m=label)
            rows.append(dict(study=study, model=label, got=int(p.exists()),
                             expected=1, done=p.exists()))
    return rows


def newest_log_mtime() -> float:
    best = 0.0
    for p in ROOT.glob("logs_suite_A*.txt"):
        try:
            best = max(best, p.stat().st_mtime)
        except OSError:
            pass
    return best


def newest_data_mtime() -> float:
    best = 0.0
    for p in DATA.rglob("*.jsonl"):
        try:
            best = max(best, p.stat().st_mtime)
        except OSError:
            pass
    return best


def free_gb() -> float:
    try:
        import shutil
        return shutil.disk_usage(str(ROOT)).free / 1e9
    except Exception:  # noqa: BLE001
        return 999.0


def scan_failures(log: pathlib.Path, since: int) -> tuple[list[str], int]:
    if not log or not log.exists():
        return [], since
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    found = []
    for ln in lines[since:]:
        for pat, what, recoverable in FAILURE_PATTERNS:
            if pat.search(ln):
                found.append(f"{what}: {ln.strip()}"
                             + ("" if recoverable else "   [NOT AUTO-RECOVERABLE]"))
    return found, len(lines)


def escalate(reason, detail):
    ALERT.write_text(json.dumps(dict(
        reason=reason, detail=detail,
        when=time.strftime("%Y-%m-%d %H:%M:%S"),
        plan=plan_status()), indent=2), encoding="utf-8")
    print(f"\n!!! ESCALATION: {reason}\n{detail}\n"
          f"wrote {ALERT.name}; exiting non-zero so a supervisor is woken.",
          flush=True)


def write_status(**kw):
    STATUS.write_text(json.dumps(
        dict(when=time.strftime("%Y-%m-%d %H:%M:%S"), plan=plan_status(), **kw),
        indent=2), encoding="utf-8")


def print_status():
    rows = plan_status()
    done = sum(r["done"] for r in rows)
    print(f"{'study':<12}{'model':<28}{'cells':>16}  state")
    for r in rows:
        print(f"{r['study']:<12}{r['model']:<28}"
              f"{f'{r[chr(103)+chr(111)+chr(116)]}/{r[chr(101)+chr(120)+chr(112)+chr(101)+chr(99)+chr(116)+chr(101)+chr(100)]}':>16}"
              f"  {'DONE' if r['done'] else 'running/queued'}")
    tot_got = sum(r["got"] for r in rows if r["expected"] > 1)
    tot_exp = sum(r["expected"] for r in rows if r["expected"] > 1)
    print(f"\n{done}/{len(rows)} jobs complete; "
          f"{tot_got}/{tot_exp} cells ({100*tot_got/max(tot_exp,1):.1f}%)")
    return done == len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--poll", type=int, default=POLL_S)
    args = ap.parse_args()

    if args.status:
        return 0 if print_status() else 2

    if another_watchdog_running():
        print("another watchdog holds the lock and is alive; exiting quietly.",
              flush=True)
        return 0
    touch_lock()

    if ALERT.exists():
        ALERT.unlink()
    restarts = 0
    log = newest_log()
    seen_lines = 0
    last_progress = time.time()
    last_cells = sum(r["got"] for r in plan_status())
    last_logmt = newest_log_mtime()
    print(f"watchdog up. {len(run_suite.JOBS['A'])} jobs in the plan, "
          f"{last_cells} cells already complete.", flush=True)

    while True:
        rows = plan_status()
        cells = sum(r["got"] for r in rows)
        all_done = all(r["done"] for r in rows)

        # EITHER signal counts as progress.
        logmt = newest_log_mtime()
        if cells > last_cells or logmt > last_logmt:
            last_cells = cells
            last_logmt = logmt
            last_progress = time.time()

        if all_done:
            touch_lock()
            write_status(state="COMPLETE", restarts=restarts)
            print(f"\nALL {len(rows)} JOBS COMPLETE at "
                  f"{time.strftime('%H:%M:%S')} ({cells} cells).", flush=True)
            return 0

        # --- disk ---------------------------------------------------------
        if free_gb() < MIN_FREE_GB:
            escalate("disk nearly full",
                     f"{free_gb():.1f} GB free, below the {MIN_FREE_GB} GB floor")
            return 3

        # --- unrecoverable failures in the log ----------------------------
        found, seen_lines = scan_failures(log, seen_lines)
        hard = [f for f in found if "NOT AUTO-RECOVERABLE" in f]
        if hard:
            escalate("failure the watchdog will not guess at", "\n".join(hard))
            return 4
        for f in found:
            print(f"  [noted] {f}", flush=True)

        # --- process died -------------------------------------------------
        if not suite_running():
            restarts += 1
            if restarts > MAX_RESTARTS:
                escalate("restart loop",
                         f"restarted {restarts} times and the plan is still "
                         f"incomplete; something is wrong that restarting does "
                         f"not fix")
                return 5
            print(f"  [recover] suite not running; restart {restarts} at "
                  f"{time.strftime('%H:%M:%S')}", flush=True)
            kill_all()
            log = start_suite()
            seen_lines = 0
            last_progress = time.time()
            time.sleep(60)
            continue

        # --- alive but not producing --------------------------------------
        stalled_min = (time.time() - last_progress) / 60
        if stalled_min > STALL_MIN:
            restarts += 1
            if restarts > MAX_RESTARTS:
                escalate("stall loop", f"stalled {stalled_min:.0f} min after "
                                       f"{restarts} restarts")
                return 6
            print(f"  [recover] no cell AND no log activity for "
                  f"{stalled_min:.0f} min; restarting (restart {restarts})",
                  flush=True)
            kill_all()
            log = start_suite()
            seen_lines = 0
            last_progress = time.time()
            time.sleep(60)
            continue

        touch_lock()
        write_status(state="running", restarts=restarts, cells=cells,
                     stalled_min=round(stalled_min, 1),
                     jobs_done=sum(r["done"] for r in rows), jobs=len(rows))
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
