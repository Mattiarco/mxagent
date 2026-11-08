"""Riconoscimento dei candidati duplicati (stesso sorgente a meno di commenti e spazi).

Un candidato già visto in questa run non si ricompila e non si risimula: il suo
esito è già noto. Esempio reale (b01, 2026-11-07): il 2° tentativo del PPA
Optimizer era identico al 1° e ha speso ~53 s di 32B più compilazione e simulazione.
"""
from __future__ import annotations

import hashlib
import re

# stringhe Scala ("...", con escape) | commento di blocco | commento di riga
_TOKENS = re.compile(r'"(?:\\.|[^"\\\n])*"|/\*.*?\*/|//[^\n]*', re.S)
_PKG = re.compile(r"^\s*package\s+[\w.]+\s*;?\s*$", re.M)

#: aggiunto al prompt quando la proposta è un duplicato (una sola rigenerazione)
DUPLICATE_NOTE = ("NOTE: the previous proposal was identical to an already-evaluated design "
                  "({same_as}: {outcome}); propose a structurally different idea.")


def normalize(code: str) -> str:
    """Sorgente senza commenti, righe vuote e spazi ridondanti; le stringhe restano intatte.
    Le righe ``package`` si tolgono: il verifier le sostituisce comunque."""
    code = _TOKENS.sub(lambda m: m.group(0) if m.group(0).startswith('"') else " ", code)
    code = _PKG.sub("", code)
    lines = (re.sub(r"[ \t]+", " ", ln).strip() for ln in code.splitlines())
    return "\n".join(ln for ln in lines if ln)


def source_hash(sources: dict[str, str]) -> str:
    """Hash di un candidato (tutti i file, in ordine di nome)."""
    h = hashlib.sha256()
    for name in sorted(sources):
        h.update(name.encode())
        h.update(b"\0")
        h.update(normalize(sources[name]).encode())
        h.update(b"\0")
    return h.hexdigest()


def retry_temperature(t: float) -> float:
    """+0.2, al massimo 0.9 (ma mai sotto la temperatura già usata)."""
    return max(t, min(0.9, round(t + 0.2, 3)))
