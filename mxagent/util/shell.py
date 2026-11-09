"""Esecuzione di comandi esterni con timeout, log catturato e tempo misurato."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Result:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def log(self) -> str:
        return self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr


def which(name: str) -> str | None:
    return shutil.which(name)


def run(cmd: list[str], cwd: str | Path | None = None, timeout: float | None = 600,
        env: dict[str, str] | None = None, grace: float = 30) -> Result:
    """Esegue ``cmd``. Allo scadere di ``timeout`` manda SIGTERM e aspetta ``grace``
    secondi prima di SIGKILL: un ``mxagent run`` fa in tempo a scrivere ``result.json``."""
    full_env = None if env is None else {**os.environ, **env}
    t0 = time.monotonic()
    exe = which(cmd[0]) or cmd[0]  # su Windows risolve .bat/.cmd (es. sbt)
    try:
        p = subprocess.Popen([exe, *cmd[1:]], cwd=cwd, env=full_env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, errors="replace")
    except FileNotFoundError as e:
        return Result(cmd, 127, "", str(e), time.monotonic() - t0)
    try:
        out, err = p.communicate(timeout=timeout)
        return Result(cmd, p.returncode, out, err, time.monotonic() - t0)
    except subprocess.TimeoutExpired:
        p.terminate()
        try:
            out, err = p.communicate(timeout=grace)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
        return Result(cmd, -1, out or "", err or "", time.monotonic() - t0, timed_out=True)
