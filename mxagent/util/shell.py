"""Esecuzione di comandi esterni con timeout, log catturato e tempo misurato.

Ogni comando parte in un **process group proprio** (POSIX: ``start_new_session``).
Allo scadere del timeout l'intero gruppo riceve SIGTERM e, dopo ``grace`` secondi,
SIGKILL: muoiono anche i nipoti (JVM di sbt, firtool, g++ di Verilator), non solo il
figlio diretto. Anche a comando finito i processi rimasti nel gruppo vengono
terminati, salvo ``keep_group=True`` (``sbt --client``: il server persistente deve
sopravvivere fra una verifica e l'altra; lo chiude ``Verifier.close``).

Se il ``fork`` stesso fallisce (EAGAIN, ENOMEM: limite di processi o memoria) non si
solleva un'eccezione: il :class:`Result` ha ``spawn_error`` e il testo dell'errore in
``stderr``, che :mod:`mxagent.verify.infra` riconosce come errore di sistema.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .procs import group_alive, kill_group

#: process group dei comandi in corso (anche da thread diversi: campagna con --parallel)
_ACTIVE: set[int] = set()
_LOCK = threading.Lock()


def kill_active(grace: float = 30) -> list[int]:
    """Termina il process group di ogni comando ancora in corso (SIGTERM, poi SIGKILL):
    serve quando il processo principale viene interrotto mentre altri thread aspettano."""
    with _LOCK:
        groups = list(_ACTIVE)
    if os.name == "nt":
        for g in groups:
            _kill_tree_nt(g)
        return groups
    import signal
    for g in groups:  # prima SIGTERM a tutti, poi un'unica attesa
        try:
            os.killpg(g, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    t0 = time.monotonic()
    while time.monotonic() - t0 < grace and any(group_alive(g) for g in groups):
        with _LOCK:  # i gruppi raccolti dai loro thread escono da _ACTIVE
            groups = [g for g in groups if g in _ACTIVE]
        time.sleep(0.2)
    for g in groups:
        if group_alive(g):
            try:
                os.killpg(g, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    return groups


@dataclass
class Result:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False
    spawn_error: bool = False  # il processo non è nemmeno partito (fork/exec fallito)
    aborted: str = ""  # interrotto da ``watch`` prima del timeout: il motivo

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.aborted

    @property
    def log(self) -> str:
        return self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr


def which(name: str) -> str | None:
    return shutil.which(name)


def _kill_tree_nt(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False)


def run(cmd: list[str], cwd: str | Path | None = None, timeout: float | None = 600,
        env: dict[str, str] | None = None, grace: float = 30, keep_group: bool = False,
        watch: Callable[[float], str | None] | None = None, watch_every: float = 5) -> Result:
    """Esegue ``cmd``. Allo scadere di ``timeout`` manda SIGTERM al process group e
    aspetta ``grace`` secondi prima di SIGKILL: un ``mxagent run`` fa in tempo a scrivere
    ``result.json``, e nessun processo figlio resta vivo.

    ``watch(secondi trascorsi)`` viene chiamata ogni ``watch_every`` s: se restituisce un
    motivo il comando viene terminato come al timeout e il :class:`Result` ha ``aborted``
    (es. il server di ``sbt --client`` non è partito: inutile aspettare tutto il timeout)."""
    full_env = None if env is None else {**os.environ, **env}
    t0 = time.monotonic()
    exe = which(cmd[0]) or cmd[0]  # su Windows risolve .bat/.cmd (es. sbt)
    kw: dict = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True  # setsid: il pid del figlio è anche il pgid
    try:
        p = subprocess.Popen([exe, *cmd[1:]], cwd=cwd, env=full_env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, errors="replace", **kw)
    except FileNotFoundError as e:
        return Result(cmd, 127, "", str(e), time.monotonic() - t0)
    except OSError as e:  # fork/exec: BlockingIOError (EAGAIN), ENOMEM, EMFILE...
        return Result(cmd, 126, "", f"{type(e).__name__}: {e}", time.monotonic() - t0,
                      spawn_error=True)
    with _LOCK:
        _ACTIVE.add(p.pid)
    try:
        return _wait(p, cmd, t0, timeout, grace, keep_group, watch, watch_every)
    finally:
        with _LOCK:
            _ACTIVE.discard(p.pid)


def _communicate(p: subprocess.Popen, t0: float, timeout: float | None,
                 watch: Callable[[float], str | None] | None, every: float) -> tuple[str, str, str]:
    """(stdout, stderr, motivo di ``watch``); TimeoutExpired allo scadere di ``timeout``.
    Ripetere ``communicate`` dopo TimeoutExpired non perde output."""
    if watch is None:
        out, err = p.communicate(timeout=timeout)
        return out, err, ""
    while True:
        left = None if timeout is None else timeout - (time.monotonic() - t0)
        if left is not None and left <= 0:
            raise subprocess.TimeoutExpired(p.args, timeout)
        try:
            out, err = p.communicate(timeout=every if left is None else min(every, left))
            return out, err, ""
        except subprocess.TimeoutExpired:
            reason = watch(time.monotonic() - t0)
            if reason:
                return "", "", reason


def _stop(p: subprocess.Popen, grace: float) -> tuple[str, str]:
    if os.name == "nt":
        _kill_tree_nt(p.pid)
    else:
        kill_group(p.pid, grace, p)
    try:
        return p.communicate(timeout=grace + 5)
    except subprocess.TimeoutExpired:
        p.kill()
        return p.communicate()


def _wait(p: subprocess.Popen, cmd: list[str], t0: float, timeout: float | None, grace: float,
          keep_group: bool, watch: Callable[[float], str | None] | None = None,
          watch_every: float = 5) -> Result:
    try:
        out, err, reason = _communicate(p, t0, timeout, watch, watch_every)
        if reason:
            out, err = _stop(p, grace)
            return Result(cmd, -1, out or "", (err or "") + f"\n{reason}\n",
                          time.monotonic() - t0, aborted=reason)
        res = Result(cmd, p.returncode, out, err, time.monotonic() - t0)
    except subprocess.TimeoutExpired:
        out, err = _stop(p, grace)
        return Result(cmd, -1, out or "", err or "", time.monotonic() - t0, timed_out=True)
    except BaseException:
        # Ctrl-C / SIGTERM / SIGHUP (RunInterrupted) mentre si aspetta: il figlio è in una
        # sessione sua e non riceve il segnale del terminale, quindi lo si termina qui
        if os.name == "nt":
            _kill_tree_nt(p.pid)
        else:
            kill_group(p.pid, grace, p)  # come al timeout: tempo per scrivere result.json
        raise
    if not keep_group and os.name != "nt" and group_alive(p.pid):
        kill_group(p.pid, min(grace, 10))  # nipoti rimasti (es. JVM lanciata in background)
    return res
