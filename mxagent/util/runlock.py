"""Una sola run alla volta sulla macchina: lock esclusivo su un file.

Una GPU e un Ollama: due run insieme si contendono il modello (il tempo di una run
dipende dall'altra) e sommano i thread delle JVM di sbt. ``mxagent eval`` (non
``--dry-run``), ``mxagent run`` e ``mxagent verify`` prendono il lock all'avvio; se è
già preso escono indicando pid e comando di chi lo tiene.

* il lock è ``flock`` (POSIX) o ``msvcrt.locking`` (Windows) sul file: lo rilascia il
  kernel quando il processo muore, anche per SIGKILL, quindi non restano lock orfani
  (il file resta, vuoto di significato finché nessuno lo tiene);
* il descrittore non è ereditabile (``O_CLOEXEC``): un server sbt sopravvissuto non
  tiene il lock;
* i sottoprocessi di chi tiene il lock (i job di una campagna) ereditano
  ``MXAGENT_LOCK_HELD=<pid>``: se il lock è tenuto proprio da quel pid partono senza
  bloccarsi; se quel processo è morto il lock è libero e lo prendono loro.

Percorso: ``MXAGENT_LOCK_FILE`` oppure ``~/.cache/mxagent/run.lock``.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ENV_FILE = "MXAGENT_LOCK_FILE"
ENV_HELD = "MXAGENT_LOCK_HELD"
#: Windows: il byte bloccato sta oltre il contenuto, che resta leggibile dagli altri
_NT_OFFSET = 2**30

_HELD: "Lock | None" = None  # tenuto per tutta la vita del processo


def lock_path() -> Path:
    p = os.environ.get(ENV_FILE)
    return Path(p).expanduser() if p else Path.home() / ".cache" / "mxagent" / "run.lock"


@dataclass
class Lock:
    path: Path
    fd: int
    inherited: bool = False  # lock del processo padre (job di una campagna): nessun fd


class LockBusy(RuntimeError):
    def __init__(self, path: Path, holder: dict):
        self.path, self.holder = path, holder
        who = (f"pid {holder.get('pid', '?')}: {holder.get('cmd', '?')}"
               + (f", dal {holder['since']}" if holder.get("since") else ""))
        super().__init__(f"un'altra run è in corso ({who}). Le run sono sempre una alla volta "
                         f"(una GPU e un Ollama): aspetta che finisca o fermala. Lock: {path}")


def _try_lock(fd: int) -> bool:
    if os.name == "nt":
        import msvcrt
        os.lseek(fd, _NT_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:  # BlockingIOError (EWOULDBLOCK/EAGAIN)
        return False


def holder(path: Path | None = None) -> dict:
    """pid, comando e ora di avvio scritti da chi tiene (o ha tenuto) il lock."""
    try:
        lines = (path or lock_path()).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    out = dict(ln.split("=", 1) for ln in lines if "=" in ln)
    if out.get("pid", "").isdigit():
        out["pid"] = int(out["pid"])
    return out


def _write_info(fd: int, cmd: str) -> None:
    info = (f"pid={os.getpid()}\ncmd={cmd}\nsince={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            ).encode("utf-8")
    os.lseek(fd, 0, os.SEEK_SET)
    if os.name != "nt":
        os.ftruncate(fd, 0)
    else:  # l'area bloccata (oltre _NT_OFFSET) non va toccata: si azzera solo il testo
        info = info.ljust(512, b" ")
    os.write(fd, info)
    os.fsync(fd)


def acquire(cmd: str | None = None, path: Path | None = None) -> Lock:
    """Prende il lock (non bloccante) e lo tiene fino alla fine del processo.
    ``LockBusy`` se lo tiene un altro processo che non è il padre di questa run."""
    global _HELD
    if _HELD is not None:
        return _HELD
    path = Path(path or lock_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NOINHERIT", 0), 0o644)
    cmd = cmd or " ".join(["mxagent", *sys.argv[1:]])
    if _try_lock(fd):
        _write_info(fd, cmd)
        _HELD = Lock(path, fd)
    else:
        os.close(fd)
        h = holder(path)
        parent = os.environ.get(ENV_HELD, "")
        if not (parent.isdigit() and h.get("pid") == int(parent)):
            raise LockBusy(path, h)
        _HELD = Lock(path, -1, inherited=True)  # job lanciato da chi tiene il lock
    os.environ[ENV_HELD] = str(h["pid"] if _HELD.inherited else os.getpid())
    return _HELD


def acquire_or_exit(cmd: str | None = None) -> Lock:
    try:
        return acquire(cmd)
    except LockBusy as e:
        raise SystemExit(f"mxagent: {e}") from None


def release() -> None:
    """Rilascia il lock prima della fine del processo (test)."""
    global _HELD
    if _HELD is None:
        return
    if not _HELD.inherited:
        os.close(_HELD.fd)  # chiudere il descrittore rilascia flock/msvcrt
    if os.environ.get(ENV_HELD) == str(os.getpid()):
        os.environ.pop(ENV_HELD, None)
    _HELD = None
