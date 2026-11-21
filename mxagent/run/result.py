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
import sys
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
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


def llm_calls_to_pass(events: list[dict], top: str | None = None) -> int | None:
    """Chiamate al modello fino al primo candidato del design di primo livello che
    passa, compresa quella che lo ha generato; tutti i ruoli (spec, planner, coder,
    fixer, reflector, integrate...). None se nessun candidato del top passa.

    Si contano gli eventi ``llm`` che PRECEDONO nel trace il primo evento ``verify``
    con ``passed`` del modulo ``top``: il trace è in ordine cronologico e ogni
    candidato è verificato dopo la chiamata che lo ha prodotto. I pass dei blocchi
    della decomposizione (``module`` diverso dal top) non chiudono il conteggio; le
    baseline non mettono ``module`` nei loro eventi ``verify`` (c'è solo il top).
    Le chiamate dopo il pass (PPA Optimizer) non sono contate."""
    n = 0
    for e in events:
        k = e.get("kind")
        if k == "llm":
            n += 1
        elif (k == "verify" and e.get("passed")
              and (top is None or e.get("module") in (None, top))):
            return n
    return None


MATCHED, PAIRED = "__baseline-b-matched__", "__baseline-b__"


def expected_cache_origins(run_dir: Path | str) -> list[Path]:
    """Run da cui una risposta in cache è ATTESA e non rende la run replayed: la run
    stessa (un tentativo precedente dello stesso job, es. dopo un errore di sistema) e,
    per un job ``baseline-b-matched`` di una campagna, la ``baseline-b`` appaiata (stesso
    task, seed e modelli, stessa campagna): matched ne ripete per costruzione le prime
    1 + max_fix_iters richieste."""
    d = Path(run_dir)
    out = [d]
    if MATCHED in d.name:
        out.append(d.with_name(d.name.replace(MATCHED, PAIRED)))
    return out


def same_run(origin: str | None, run_dir: Path) -> bool:
    """La provenienza ``origin`` (cartella assoluta registrata in cache) è ``run_dir``?
    Si confrontano campagna/runs/job (ultime tre componenti): regge lo spostamento
    della cartella della campagna, non confonde campagne diverse."""
    if not origin:
        return False
    a = [x for x in str(origin).replace("\\", "/").split("/") if x][-3:]
    b = [x for x in str(run_dir).replace("\\", "/").split("/") if x][-3:]
    return a == b


#: una run è "replayed" se TUTTE le sue chiamate al modello vengono dalla cache LLM e
#: almeno una viene da una run diversa da quelle attese (:func:`expected_cache_origins`):
#: il risultato ripete un'esecuzione precedente. Le risposte senza provenienza (voci di
#: cache scritte prima del campo ``origin``) contano come estranee. Le run con una parte
#: delle chiamate nuove non sono replayed (``llm_cached_fraction``, elencate a parte).
def retry_event(info: dict) -> dict[str, Any]:
    """Campi dell'evento ``infra_retry`` da un elemento di ``VerifyReport.infra_retries``."""
    return {"attempt": info.get("attempt"), "wait_s": info.get("wait_s"), "where": info.get("where"),
            "tag": info.get("tag"), "reason": info.get("detail"), "java": info.get("java"),
            "killed": info.get("killed")}


def infra_retry_fields(events: list[dict]) -> dict[str, Any]:
    """``infra_retries`` di result.json: un elemento per ogni nuovo tentativo di verifica
    dopo un errore di sistema (eventi ``infra_retry`` del trace)."""
    keys = ("attempt", "wait_s", "where", "tag", "reason", "java", "killed")
    return {"infra_retries": [{k: e.get(k) for k in keys} for e in events
                              if e.get("kind") == "infra_retry"]}


def llm_cache_fields(events: list[dict], expected: list[Path] | None = None) -> dict[str, Any]:
    """``llm_cached_calls``, ``llm_cached_expected``, ``llm_cached_fraction`` e
    ``replayed`` dagli eventi ``llm`` (``cache_origin`` = run che aveva la risposta)."""
    calls = [e for e in events if e.get("kind") == "llm"]
    cached = [e for e in calls if e.get("cached")]
    exp = [e for e in cached if any(same_run(e.get("cache_origin"), d) for d in expected or [])]
    return {"llm_cached_calls": len(cached), "llm_cached_expected": len(exp),
            "llm_cached_fraction": round(len(cached) / len(calls), 4) if calls else None,
            "replayed": bool(calls) and len(cached) == len(calls) and len(exp) < len(cached)}


def llm_result_fields(llm: Any, bud: dict) -> dict[str, Any]:
    """Backend, timeout configurato, tempo nel modello, timeout e costo (result.json)."""
    return {"llm_backend": getattr(llm, "backend", "scripted"),
            "llm_timeout_s": getattr(llm, "timeout", None),
            "llm_seconds": bud.get("llm_seconds"), "llm_timeouts": bud.get("timeouts", 0),
            "llm_timeout_seconds": bud.get("timeout_seconds", 0.0),
            "llm_failed_calls": bud.get("failed_calls", 0), "cost_usd": bud.get("cost_usd")}


def ppa_attempts(ppa_out: dict | None) -> dict[str, int]:
    """Tentativi del PPA Optimizer per esito (accepted, rejected, functional_fail, ...)."""
    steps = ((ppa_out or {}).get("optimizer") or {}).get("steps") or []
    c = Counter(s.get("outcome") for s in steps)
    keys = ("accepted", "rejected", "functional_fail", "synth_fail", "no_code", "duplicate")
    return {"total": len(steps), **{k: c.get(k, 0) for k in keys}}


def first_error(report: Any) -> str | None:
    """Primo errore del miglior candidato di una run non passata (notes.md del report):
    la prima firma d'errore del verifier, altrimenti la prima riga del feedback."""
    if report is None:
        return "no Scala code in the answer"
    if getattr(report, "passed", False):
        return None
    keys = getattr(report, "error_keys", None) or []
    if keys:
        return str(keys[0])[:300]
    for text in (getattr(report, "feedback", ""), getattr(report, "feedback_raw", "")):
        for ln in (text or "").splitlines():
            if ln.strip():
                return ln.strip()[:300]
    return None


def check_models(rec: Recorder, label: str, mismatches: dict[str, dict[str, str]]) -> None:
    """All'avvio: se il profilo dichiarato (nome della run) non corrisponde ai modelli
    effettivi dei ruoli, warning su stderr ed evento ``model_profile_mismatch``."""
    if not mismatches:
        return
    roles = ", ".join(f"{r}={d['actual']} (atteso {d['expected']})" for r, d in mismatches.items())
    print(f"WARNING: il profilo '{label}' non corrisponde ai ruoli: {roles}", file=sys.stderr)
    rec.event("model_profile_mismatch", profile=label, mismatches=mismatches)


def write_result(rec: Recorder, result: dict[str, Any]) -> None:
    """``result.json`` + evento finale ``end`` (con i campi principali)."""
    rec.write_json("result.json", result)
    keys = ("status", "reason", "passed", "best_candidate", "wall_s", "llm_calls", "tokens_in",
            "tokens_out")
    rec.event("end", **{k: result.get(k) for k in keys})
