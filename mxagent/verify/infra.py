"""Errori di sistema (risorse del sistema operativo, JVM) distinti dagli errori del codice.

Un candidato che non compila perché la JVM non riesce a creare un thread, o perché
``fork`` fallisce con EAGAIN, non dice nulla sul modello. Questi errori:

* producono un report con ``stage = "infra"``, mai ``compile``/``lint``/``sim``;
* fanno ripetere la verifica dello stesso candidato (``Verifier``), dopo aver
  terminato i processi figli rimasti;
* se persistono interrompono la run con :class:`InfraError`: status ``error``,
  reason ``infra: ...``. Il feedback non arriva mai al modello e non consuma
  iterazioni, reflect, dedup né anti-loop.

La lista dei pattern è qui e solo qui (:data:`INFRA_PATTERNS`), insieme alla regola per
i fallimenti di sbt senza errori del sorgente (:func:`classify_build`).
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


#: client/server di sbt che non si parlano (server morto, non avviato, non risponde)
SBT_SERVER = re.compile(
    r"sbt server (?:could not|failed to|is not|did not)|server (?:failed to start|did not start|"
    r"is not responding|was shut down)|Could not connect to (?:the )?(?:sbt )?server|"
    r"Connection refused|Broken pipe|java\.net\.ConnectException|ServerAlreadyBootingException|"
    r"sbt server is already booting", re.I)
#: stadi di errore che riguardano il sorgente del candidato (vedi chisel.parse_sbt_log);
#: "sbt" è il ripiego per le righe [error] che non indicano nessun errore del codice
CODE_STAGES = ("scalac", "firtool", "elaboration")
#: codici d'uscita 128 + segnale dello script sbt quando la JVM muore (ABRT, BUS, KILL,
#: SEGV, TERM): firtool che abortisce dentro Chisel dà invece exit 1 con l'eccezione
SIGNAL_EXIT = {134: "SIGABRT", 135: "SIGBUS", 137: "SIGKILL", 139: "SIGSEGV", 143: "SIGTERM"}


def classify_build(log: str, errors: list[tuple[str, str]], returncode: int | None = None,
                   timed_out: bool = False) -> str | None:
    """Errore di sistema in una compilazione/elaborazione sbt fallita, altrimenti None.

    ``errors``: (stadio, messaggio) estratti dal log. Di sistema se:

    * il log o i messaggi contengono un pattern di :data:`INFRA_PATTERNS`;
    * il processo è terminato per un segnale (``returncode`` < 0 o 128 + segnale);
    * client e server di sbt non si parlano (:data:`SBT_SERVER`);
    * nessun errore riguarda il sorgente (nessuno stadio in :data:`CODE_STAGES`):
      sbt è fallito senza dire perché, es. il server è morto a metà comando e il
      log contiene solo ``[error] Total time: 1 s``. -> ``sbt_no_errors``.

    Un timeout resta un errore del candidato (un generatore che non termina).
    """
    hit = classify("\n".join([log or "", *(m for _, m in errors)]))
    if hit:
        return hit
    if timed_out:
        return None
    if returncode is not None and returncode < 0:
        return f"sbt_signal: processo terminato dal segnale {-returncode}"
    if returncode in SIGNAL_EXIT:
        return f"sbt_signal: exit {returncode} ({SIGNAL_EXIT[returncode]})"
    for line in (log or "").splitlines():
        if SBT_SERVER.search(line):
            return f"sbt_server: {' '.join(line.split())[:240]}"
    if not any(stage in CODE_STAGES for stage, _ in errors):
        tail = [ln.strip() for ln in (log or "").splitlines() if ln.strip()][-3:]
        return "sbt_no_errors: " + (" | ".join(tail)[:240] or "log vuoto")
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
