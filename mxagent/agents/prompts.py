"""Prompt versionati in ``prompts/`` con sostituzione ``{variabile}``.

Ogni run registra l'hash di ogni prompt usato: i risultati restano legati alla
versione esatta del testo.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent.parent.parent / "prompts"
_VAR = re.compile(r"\{(\w+)\}")


def load(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def render(name: str, **vars: str) -> str:
    """Sostituisce solo ``{nome}`` noti: le graffe del codice Scala restano intatte."""
    text = load(name)
    missing = {m for m in _VAR.findall(text) if m not in vars and _looks_var(m)}
    if missing:
        raise KeyError(f"prompt {name}: variabili mancanti {sorted(missing)}")
    return _VAR.sub(lambda m: str(vars[m[1]]) if m[1] in vars else m[0], text).strip()


def _looks_var(name: str) -> bool:
    return name.islower() and not name.isdigit()


def fingerprint(*names: str) -> dict[str, str]:
    return {n: hashlib.sha256(load(n).encode()).hexdigest()[:12] for n in names}
