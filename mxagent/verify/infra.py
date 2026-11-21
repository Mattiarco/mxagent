"""Errori di sistema (risorse del sistema operativo, JVM) distinti dagli errori del codice.

Un candidato che non compila perché la JVM non riesce a creare un thread, o perché
``fork`` fallisce con EAGAIN, non dice nulla sul modello. Questi errori:

* producono un report con ``stage = "infra"``, mai ``compile``/``lint``/``sim``;
* fanno ripetere la verifica dello stesso candidato (``Verifier``), dopo aver
  terminato i processi figli rimasti;
* se persistono interrompono la run con :class:`InfraError`: status ``error``,
  reason ``infra: ...``. Il feedback non arriva mai al modello e non consuma
  iterazioni, reflect, dedup né anti-loop.

La lista dei pattern è qui e solo qui (:data:`INFRA_PATTERNS`).
"""
from __future__ import annotations

import re

#: (tipo, pattern) cercati nell'output di sbt/firtool/verilator/cocotb e nel testo
#: delle eccezioni Python. Solo errori del sistema o della JVM, mai del codice.
INFRA_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("native_thread", re.compile(r"unable to create (?:new )?native thread|"
                                 r"can't start new thread", re.I)),
    # EAGAIN: limite di processi/thread (ulimit -u, pids.max del cgroup)
    ("eagain", re.compile(r"Resource temporarily unavailable|\bEAGAIN\b|\[Errno 11\]|"
                          r"\berror=11\b|BlockingIOError")),
    ("fork", re.compile(r"fork: retry|\bcannot fork\b|fork failed|Cannot run program .*error=(?:11|12)",
                        re.I)),
    # ENOMEM e memoria della JVM
    ("enomem", re.compile(r"Cannot allocate memory|\[Errno 12\]|\bENOMEM\b|\berror=12\b")),
    ("jvm_oom", re.compile(r"OutOfMemoryError")),
    ("jvm_fatal", re.compile(r"A fatal error has been detected by the Java Runtime|"
                             r"Could not reserve enough space|"
                             r"Error occurred during initialization of VM|"
                             r"There is insufficient memory for the Java Runtime")),
    # descrittori di file e disco
    ("emfile", re.compile(r"Too many open files|\[Errno 24\]|\bEMFILE\b")),
    ("enospc", re.compile(r"No space left on device|\[Errno 28\]|\bENOSPC\b")),
)


def classify(text: str | None) -> str | None:
    """``"<tipo>: <riga>"`` del primo errore di sistema nel testo, altrimenti None."""
    if not text:
        return None
    for line in text.splitlines():
        for kind, pat in INFRA_PATTERNS:
            if pat.search(line):
                return f"{kind}: {' '.join(line.split())[:240]}"
    return None


def classify_exc(e: BaseException) -> str | None:
    """Come :func:`classify` per un'eccezione: ogni ``OSError`` non gestito è di sistema
    (fork, file, rete); per le altre conta il testo."""
    text = f"{type(e).__name__}: {e}"
    hit = classify(text)
    if hit:
        return hit
    if isinstance(e, OSError):
        return f"oserror: {text[:240]}"
    return None


class InfraError(RuntimeError):
    """Errore di sistema persistente: la run non è valutabile (status ``error``)."""

    def __init__(self, detail: str, where: str = "", report=None, attempts: int = 1):
        super().__init__(f"{detail}" + (f" ({where})" if where else ""))
        self.detail, self.where, self.report, self.attempts = detail, where, report, attempts


def infra_outcome(e: BaseException) -> dict | None:
    """Esito di result.json per un errore di sistema (InfraError o OSError), altrimenti None."""
    if isinstance(e, InfraError):
        return {"status": "error", "reason": f"infra: {e.detail}"[:300],
                "error": f"{type(e).__name__}: {e}"[:500], "infra_attempts": e.attempts}
    hit = classify_exc(e)  # ogni OSError, o un'eccezione il cui testo è un errore di sistema
    if hit:
        return {"status": "error", "reason": f"infra: {hit}"[:300],
                "error": f"{type(e).__name__}: {e}"[:500]}
    return None
