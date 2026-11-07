"""Estrazione del codice Scala dalle risposte dell'LLM."""
from __future__ import annotations

import re

_FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)


def extract_scala(text: str, top: str | None = None) -> str | None:
    """Blocco di codice Scala più plausibile.

    Preferisce i blocchi ``scala``/``chisel``/senza linguaggio che definiscono
    ``class <top>``; altrimenti il più lungo che contiene una classe. Se non ci
    sono blocchi ma il testo è codice, restituisce il testo.
    """
    blocks = [(lang.lower(), body.strip()) for lang, body in _FENCE.findall(text)]
    blocks = [b for lang, b in blocks if lang in ("", "scala", "chisel", "sc") and b]
    if not blocks:
        t = text.strip()
        return t if re.search(r"\bclass\s+\w+", t) and "import chisel3" in t else None
    if top:
        pat = re.compile(rf"\b(class|object)\s+{re.escape(top)}\b")
        with_top = [b for b in blocks if pat.search(b)]
        if with_top:
            return max(with_top, key=len)
    with_class = [b for b in blocks if re.search(r"\bclass\s+\w+", b)]
    return max(with_class or blocks, key=len)
