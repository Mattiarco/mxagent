"""Una sola run alla volta: lock esclusivo (flock / msvcrt) preso da eval, run e verify."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mxagent.util import runlock

ROOT = Path(__file__).resolve().parents[1]

HOLD = """
import sys, time
from mxagent.util import runlock
runlock.acquire("prova-titolare")
print("held", __import__("os").getpid(), flush=True)
time.sleep(float(sys.argv[1]))
"""
TRY = """
from mxagent.util import runlock
try:
    l = runlock.acquire("prova-secondo")
    print("ok inherited" if l.inherited else "ok own", flush=True)
except runlock.LockBusy as e:
    print("busy", e, flush=True)
"""
# titolare che lancia un figlio (come il runner di una campagna con `mxagent run`)
PARENT = """
import subprocess, sys
from mxagent.util import runlock
runlock.acquire("prova-campagna")
r = subprocess.run([sys.executable, "-c", sys.argv[1]], capture_output=True, text=True, timeout=60)
print(r.stdout.strip(), r.stderr.strip()[-300:], flush=True)
"""


def _env(lock):
    env = {**os.environ, runlock.ENV_FILE: str(lock), "PYTHONPATH": str(ROOT)}
    env.pop(runlock.ENV_HELD, None)
    return env


def _hold(lock, seconds=60):
    p = subprocess.Popen([sys.executable, "-c", HOLD, str(seconds)], env=_env(lock),
                         stdout=subprocess.PIPE, text=True)
    word, pid = p.stdout.readline().split()
    assert word == "held"
    p.holder_pid = int(pid)  # su Windows il python del venv è un launcher: pid diverso
    return p


def _try(lock, env=None):
    r = subprocess.run([sys.executable, "-c", TRY], env=env or _env(lock), capture_output=True,
                       text=True, timeout=60)
    return r.stdout.strip()


def test_second_process_fails_while_first_holds_and_starts_after_it_dies(tmp_path):
    lock = tmp_path / "run.lock"
    p = _hold(lock)
    try:
        out = _try(lock)
        assert out.startswith("busy") and f"pid {p.holder_pid}" in out and "prova-titolare" in out
        assert "una alla volta" in out
    finally:
        p.kill()  # SIGKILL: il kernel rilascia il lock, nessun file da pulire
        p.wait()
    assert _try(lock) == "ok own"


def test_child_of_the_holder_is_not_blocked(tmp_path):
    lock = tmp_path / "run.lock"
    r = subprocess.run([sys.executable, "-c", PARENT, TRY], env=_env(lock), capture_output=True,
                       text=True, timeout=120)
    assert r.stdout.strip().startswith("ok inherited"), r.stdout + r.stderr


def test_held_variable_of_another_pid_does_not_bypass_the_lock(tmp_path):
    lock = tmp_path / "run.lock"
    p = _hold(lock)
    try:
        env = {**_env(lock), runlock.ENV_HELD: str(p.holder_pid + 12345)}
        assert _try(lock, env).startswith("busy")
    finally:
        p.kill()
        p.wait()


def test_stale_held_variable_takes_the_free_lock(tmp_path):
    """Il runner è morto: il job che ne eredita la variabile trova il lock libero e lo prende."""
    lock = tmp_path / "run.lock"
    env = {**_env(lock), runlock.ENV_HELD: "999999"}
    assert _try(lock, env) == "ok own"


def test_holder_info_and_reentrant_acquire(tmp_path, monkeypatch):
    monkeypatch.setenv(runlock.ENV_FILE, str(tmp_path / "run.lock"))
    a = runlock.acquire("mxagent eval --campaign x")
    assert runlock.acquire() is a  # stesso processo: già tenuto
    h = runlock.holder()
    assert h["pid"] == os.getpid() and h["cmd"] == "mxagent eval --campaign x" and h["since"]
    assert os.environ[runlock.ENV_HELD] == str(os.getpid())  # passato ai sottoprocessi
    runlock.release()
    assert runlock.ENV_HELD not in os.environ


@pytest.mark.parametrize("argv", [
    ["verify", "--task", str(ROOT / "benchmarks" / "b01_e2m1_decode")],
    ["run", "--task", str(ROOT / "benchmarks" / "b01_e2m1_decode"), "--mode", "baseline-a"],
    ["eval", "--campaign", str(ROOT / "configs" / "eval" / "smoke.yaml")],
])
def test_cli_commands_exit_when_another_run_holds_the_lock(argv, tmp_path, monkeypatch):
    from mxagent import cli
    lock = tmp_path / "run.lock"
    monkeypatch.setenv(runlock.ENV_FILE, str(lock))
    p = _hold(lock)
    try:
        with pytest.raises(SystemExit) as e:
            cli.main([*argv, *(["--out", str(tmp_path / "ev")] if argv[0] == "eval" else [])])
        msg = str(e.value.code)
        assert "un'altra run è in corso" in msg and f"pid {p.holder_pid}" in msg
    finally:
        p.kill()
        p.wait()


def test_dry_run_does_not_take_the_lock(tmp_path, monkeypatch, capsys):
    from mxagent import cli
    lock = tmp_path / "run.lock"
    monkeypatch.setenv(runlock.ENV_FILE, str(lock))
    p = _hold(lock)
    try:
        assert cli.main(["eval", "--campaign", str(ROOT / "configs" / "eval" / "smoke.yaml"),
                         "--dry-run", "--out", str(tmp_path / "ev")]) == 0
    finally:
        p.kill()
        p.wait()
