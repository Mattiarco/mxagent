"""Chiusura esplicita di una run: ``result.json`` ed evento ``end`` nel trace, sempre.

Una run può finire in quattro modi e ognuno lascia ``result.json``:

``done``         design corretto (bit-exact su tutti i vettori)
``failed``       nessun design corretto: iterazioni, anti-loop o budget esauriti
``error``        eccezione non prevista (bug o strumento): la run non è valutabile
``interrupted``  SIGINT/SIGTERM/SIGHUP (Ctrl-C, ``kill``, terminale chiuso senza tmux)

Il file si scrive in un ``finally``; i segnali diventano un'eccezione
(:class:`RunInterrupted`) così il ``finally`` gira anche quando la run viene
interrotta dall'esterno.
"""
from __future__ import annotations

import signal
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .recorder import Recorder

_SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP")  # SIGHUP: terminale chiuso (non esiste su Windows)


class RunInterrupted(BaseException):
    """Segnale ricevuto durante la run. BaseException: nessun ``except Exception``
    intermedio (es. quello che protegge lo stadio PPA) la può inghiottire."""

    def __init__(self, signame: str):
        super().__init__(signame)
        self.signame = signame


@contextmanager
def interruptible() -> Iterator[None]:
    """Trasforma SIGINT/SIGTERM/SIGHUP in :class:`RunInterrupted` (solo nel thread
    principale: altrove ``signal.signal`` non è permesso e il contesto non fa nulla)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old: dict[int, Any] = {}

    def handler(signum, _frame):
        # i segnali successivi (es. SIGHUP + SIGTERM) non devono interrompere la chiusura
        for s in old:
            signal.signal(s, signal.SIG_IGN)
        raise RunInterrupted(signal.Signals(signum).name)

    for name in _SIGNALS:
        s = getattr(signal, name, None)
        if s is None:
            continue
        try:
            old[s] = signal.signal(s, handler)
        except (ValueError, OSError):
            pass
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def llm_by_role(events: list[dict]) -> dict[str, dict[str, int]]:
    """Chiamate e token per ruolo, dagli eventi ``llm`` del trace (le baseline hanno un
    solo ruolo, ``coder``)."""
    out: dict[str, Counter] = {}
    for e in events:
        if e.get("kind") != "llm":
            continue
        c = out.setdefault(e.get("role") or "coder", Counter())
        tin, tout = (e.get("tokens") or [0, 0])[:2]
        c["calls"] += 1
        c["tokens_in"] += int(tin or 0)
        c["tokens_out"] += int(tout or 0)
        c["cached"] += bool(e.get("cached"))
    return {k: dict(v) for k, v in out.items()}


def ppa_attempts(ppa_out: dict | None) -> dict[str, int]:
    """Tentativi del PPA Optimizer per esito (accepted, rejected, functional_fail, ...)."""
    steps = ((ppa_out or {}).get("optimizer") or {}).get("steps") or []
    c = Counter(s.get("outcome") for s in steps)
    keys = ("accepted", "rejected", "functional_fail", "synth_fail", "no_code", "duplicate")
    return {"total": len(steps), **{k: c.get(k, 0) for k in keys}}


def write_result(rec: Recorder, result: dict[str, Any]) -> None:
    """``result.json`` + evento finale ``end`` (con i campi principali)."""
    rec.write_json("result.json", result)
    keys = ("status", "reason", "passed", "best_candidate", "wall_s", "llm_calls", "tokens_in",
            "tokens_out")
    rec.event("end", **{k: result.get(k) for k in keys})
