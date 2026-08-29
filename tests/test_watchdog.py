"""Tests for the supervisor.

A watchdog that crashes on its own recovery path is worse than no watchdog,
because the run looks supervised while nothing is watching it. That is exactly
what happened: `start_suite` passed `capture_output` to `subprocess.Popen`,
which does not accept it, so the very first recovery attempt raised TypeError
and the supervisor exited. The suite stayed down and nothing noticed.

These tests exercise the recovery paths without launching anything, by
substituting the subprocess layer. They would have caught that bug in under a
second.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import types

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "paper-a" / "src"
sys.path.insert(0, str(SRC))

import watchdog as wd  # noqa: E402

# Captured BEFORE any test monkeypatches subprocess.Popen. Reading the signature
# after patching would inspect the double and pass vacuously.
import inspect as _inspect
REAL_POPEN_PARAMS = set(_inspect.signature(subprocess.Popen.__init__).parameters)


# --------------------------------------------------------------------------
# The bug: every subprocess call must use an API that accepts its own arguments
# --------------------------------------------------------------------------
def test_start_suite_launches_without_inheriting_pipes(monkeypatch, tmp_path):
    """Two bugs lived on this line and both left the watchdog apparently alive.

    First `capture_output` was passed to Popen, which does not accept it, so the
    first recovery raised TypeError. Then `subprocess.run(capture_output=True)`
    fixed the crash and deadlocked instead: the launched suite inherits the
    pipes, so the call blocks until the suite exits and the supervisor stops
    supervising.

    The launcher must therefore (a) use an API that accepts its arguments and
    (b) hand the child NO pipes to inherit and NOT wait on it.
    """
    calls = []

    class FakePopen:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(wd.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(wd.time, "sleep", lambda s: None)
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    log = wd.start_suite()
    assert calls, "start_suite made no subprocess call"
    assert log.name.startswith("logs_suite_A")

    _, kwargs = calls[0]
    # REAL_POPEN_PARAMS is captured at import, before any monkeypatching, or
    # this assertion reads the signature of the double and passes vacuously.
    for k in kwargs:
        assert k in REAL_POPEN_PARAMS, f"subprocess.Popen does not accept {k!r}"
    assert kwargs.get("stdout") is subprocess.DEVNULL, \
        "the child must not inherit a stdout pipe: run() would block on it"
    assert kwargs.get("stderr") is subprocess.DEVNULL
    assert "capture_output" not in kwargs
    assert "timeout" not in kwargs, "Popen does not wait; a timeout is meaningless"


def test_start_suite_does_not_block(monkeypatch, tmp_path):
    """A launcher that waits for the thing it launched is a deadlock. Asserted
    by making the double record whether anything waited."""
    waited = []

    class FakePopen:
        def __init__(self, *a, **k):
            pass

        def wait(self, *a, **k):
            waited.append(True)

        def communicate(self, *a, **k):
            waited.append(True)
            return "", ""

    monkeypatch.setattr(wd.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(wd.time, "sleep", lambda s: None)
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    wd.start_suite()
    assert not waited, "start_suite waited on the suite it launched"


def test_kill_all_uses_a_callable_that_accepts_its_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(wd.subprocess, "run",
                        lambda *a, **k: calls.append((a, k)) or
                        types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(wd.time, "sleep", lambda s: None)
    wd.kill_all()
    assert len(calls) >= 2


def test_suite_running_survives_a_powershell_failure(monkeypatch):
    def boom(*a, **k):
        raise OSError("powershell missing")
    monkeypatch.setattr(wd.subprocess, "run", boom)
    assert wd.suite_running() is False      # must not raise


def test_suite_running_parses_a_count(monkeypatch):
    monkeypatch.setattr(
        wd.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="2\n", stderr=""))
    assert wd.suite_running() is True
    monkeypatch.setattr(
        wd.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="0\n", stderr=""))
    assert wd.suite_running() is False


# --------------------------------------------------------------------------
# Expected counts must come from the design, not from constants
# --------------------------------------------------------------------------
def test_expected_counts_are_derived_from_the_experiment_modules():
    from experiment_mechanism import CONDITIONS
    from experiment_delta_stability import VARIANTS, PAIRS
    import stimuli as st

    nb = len(st.TEMPLATES_2)
    # A base job owns the two extreme templates; its -t2 partner completes the
    # file. They must NOT share an expectation, or the base target silently
    # marks the -t2 job done and every T2 arm goes unrun.
    assert wd.EXPECTED["names"][2] == len(VARIANTS) * nb * len(st.NAME_GRID)
    assert wd.EXPECTED["names-t2"][2] == len(VARIANTS) * len(st.TEMPLATES) * len(st.NAME_GRID)
    assert wd.EXPECTED["names-t2"][2] > wd.EXPECTED["names"][2]
    assert wd.EXPECTED["mech-chat"][2] == len(CONDITIONS) * nb * len(st.MECH_GRID)
    assert wd.EXPECTED["mech-chat-t2"][2] == len(CONDITIONS) * len(st.TEMPLATES) * len(st.MECH_GRID)
    assert wd.EXPECTED["quant"][2] == len(VARIANTS) * len(st.TEMPLATES) * len(PAIRS)
    assert wd.EXPECTED["occupation"][2] == 2 * len(VARIANTS) * len(st.TEMPLATES) * len(PAIRS)
    assert wd.EXPECTED["replicate"][2] == 2 * 5 * len(st.TEMPLATES) * len(PAIRS)


def test_every_study_in_the_plan_has_an_expectation():
    import run_suite
    known = set(wd.EXPECTED) | set(wd.ALIAS) | set(wd.PROBE)
    for study, *_ in run_suite.JOBS["A"]:
        assert study in known, f"{study} is in the plan and has no expected count"


def test_t2_aliases_map_to_their_base_study():
    for alias, base in wd.ALIAS.items():
        assert base in wd.EXPECTED
        assert alias.endswith("-t2")


# --------------------------------------------------------------------------
# Escalation must be machine-readable and must not swallow the reason
# --------------------------------------------------------------------------
def test_escalate_writes_a_readable_alert(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "ALERT", tmp_path / "alert.json")
    monkeypatch.setattr(wd, "plan_status", lambda: [{"study": "x", "model": "y",
                                                     "got": 1, "expected": 2,
                                                     "done": False}])
    wd.escalate("disk nearly full", "3.1 GB free")
    d = json.loads((tmp_path / "alert.json").read_text(encoding="utf-8"))
    assert d["reason"] == "disk nearly full"
    assert "3.1 GB" in d["detail"]
    assert d["plan"][0]["study"] == "x"


def test_failure_patterns_flag_the_known_classes():
    lines = {
        "  [LOAD FAILED] server exited early (1)": True,
        "  [ABORTED] something": True,
        "REFUSING TO RUN: port 8080 serves the wrong file": True,
        "    96/576 cells  9.6 min": False,
    }
    for line, should_match in lines.items():
        hit = any(p.search(line) for p, _, _ in wd.FAILURE_PATTERNS)
        assert hit is should_match, line


def test_unrecoverable_patterns_are_marked_as_such():
    """A contamination guard firing must NOT be auto-restarted into; it means
    the wrong weights were being served and a human has to look."""
    by_text = {what: recoverable for _, what, recoverable in wd.FAILURE_PATTERNS}
    assert by_text["contamination guard fired"] is False
    assert by_text["corrupt data file"] is False
    assert by_text["server failed to load"] is True


# --------------------------------------------------------------------------
# Completion accounting
# --------------------------------------------------------------------------
def test_count_complete_ignores_incomplete_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "DATA", tmp_path)
    (tmp_path / "names").mkdir()
    p = tmp_path / "names" / "names_m.jsonl"
    p.write_text('{"white_margin": 1.0, "black_margin": 0.0}\n'
                 '{"white_margin": null, "black_margin": 0.0}\n'
                 '{"white_margin": 2.0, "black_margin": 1.0}\n', encoding="utf-8")
    assert wd.count_complete("names", "names_{m}.jsonl", "m") == 2


def test_count_complete_on_a_missing_file_is_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "DATA", tmp_path)
    assert wd.count_complete("names", "names_{m}.jsonl", "nope") == 0


# --------------------------------------------------------------------------
# The singleton lock. A process-name count was tried first and was wrong twice:
# it matched any process merely MENTIONING watchdog.py, and two scheduled-task
# firings a moment apart each saw the other and both exited, leaving nothing
# supervising.
# --------------------------------------------------------------------------
def test_no_lock_means_no_other_watchdog(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "LOCK", tmp_path / "w.lock")
    assert wd.another_watchdog_running() is False


def test_our_own_lock_does_not_block_us(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "LOCK", tmp_path / "w.lock")
    wd.touch_lock()
    assert wd.another_watchdog_running() is False


def test_a_live_foreign_lock_blocks(monkeypatch, tmp_path):
    import os, time as _t
    monkeypatch.setattr(wd, "LOCK", tmp_path / "w.lock")
    (tmp_path / "w.lock").write_text(
        json.dumps({"pid": os.getpid() + 1, "heartbeat": _t.time()}),
        encoding="utf-8")
    monkeypatch.setattr(wd, "_pid_alive", lambda pid: True)
    assert wd.another_watchdog_running() is True


def test_a_dead_foreign_lock_does_not_block(monkeypatch, tmp_path):
    import os, time as _t
    monkeypatch.setattr(wd, "LOCK", tmp_path / "w.lock")
    (tmp_path / "w.lock").write_text(
        json.dumps({"pid": os.getpid() + 1, "heartbeat": _t.time()}),
        encoding="utf-8")
    monkeypatch.setattr(wd, "_pid_alive", lambda pid: False)
    assert wd.another_watchdog_running() is False


def test_a_stale_heartbeat_is_taken_over(monkeypatch, tmp_path):
    """A watchdog killed by a session teardown cannot clean up its own lock.
    If a cold heartbeat blocked forever, recovery would never resume."""
    import os, time as _t
    monkeypatch.setattr(wd, "LOCK", tmp_path / "w.lock")
    (tmp_path / "w.lock").write_text(
        json.dumps({"pid": os.getpid() + 1,
                    "heartbeat": _t.time() - wd.HEARTBEAT_STALE_S - 60}),
        encoding="utf-8")
    monkeypatch.setattr(wd, "_pid_alive", lambda pid: True)   # alive but frozen
    assert wd.another_watchdog_running() is False


def test_a_corrupt_lock_does_not_block(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "LOCK", tmp_path / "w.lock")
    (tmp_path / "w.lock").write_text("not json", encoding="utf-8")
    assert wd.another_watchdog_running() is False


# --------------------------------------------------------------------------
# The pre-check that keeps a restart cheap. Defining job_is_complete and never
# calling it cost twenty-five minutes of checkpoint loading on every restart,
# which was longer than the stall window that triggered the restart.
# --------------------------------------------------------------------------
def test_run_suite_actually_calls_the_completion_precheck():
    import run_suite, inspect
    src = inspect.getsource(run_suite.main)
    assert "job_is_complete(" in src, \
        "run_suite.main never calls job_is_complete; every restart reloads " \
        "every already-finished checkpoint"


def test_job_is_complete_recognises_a_finished_job(monkeypatch, tmp_path):
    import run_suite
    monkeypatch.setattr(run_suite, "ROOT", tmp_path)
    d = tmp_path / "paper-a" / "data" / "quantization"
    d.mkdir(parents=True)
    exp = run_suite.EXPECTED["quant"][2]
    with (d / "delta_m.jsonl").open("w", encoding="utf-8") as fh:
        for _ in range(exp):
            fh.write('{"white_margin": 1.0, "black_margin": 0.0}\n')
    assert run_suite.job_is_complete("quant", "m") is True


def test_job_is_complete_is_false_when_short(monkeypatch, tmp_path):
    import run_suite
    monkeypatch.setattr(run_suite, "ROOT", tmp_path)
    d = tmp_path / "paper-a" / "data" / "quantization"
    d.mkdir(parents=True)
    with (d / "delta_m.jsonl").open("w", encoding="utf-8") as fh:
        fh.write('{"white_margin": 1.0, "black_margin": 0.0}\n')
    assert run_suite.job_is_complete("quant", "m") is False


def test_job_is_complete_is_false_when_absent(monkeypatch, tmp_path):
    import run_suite
    monkeypatch.setattr(run_suite, "ROOT", tmp_path)
    assert run_suite.job_is_complete("quant", "nope") is False


def test_a_t2_job_is_not_marked_complete_by_its_base_target(monkeypatch, tmp_path):
    """The regression that would have left every T2 arm unrun: `names-t2`
    resolved through ALIAS to `names`, inherited the smaller target, and a file
    holding only the two base templates was judged finished."""
    import run_suite
    monkeypatch.setattr(run_suite, "ROOT", tmp_path)
    d = tmp_path / "paper-a" / "data" / "names"
    d.mkdir(parents=True)
    base_target = run_suite.EXPECTED["names"][2]
    with (d / "names_m.jsonl").open("w", encoding="utf-8") as fh:
        for _ in range(base_target):
            fh.write('{"white_margin": 1.0, "black_margin": 0.0}\n')
    assert run_suite.job_is_complete("names", "m") is True
    assert run_suite.job_is_complete("names-t2", "m") is False
