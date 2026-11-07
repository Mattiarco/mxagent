"""Libreria di componenti verificati (``library/``).

Ogni componente che passa i test viene salvato con la sua specifica,
interfaccia, codice e provenienza. Usi:

* **riuso esatto**: stesso blocco del catalogo con gli stessi parametri ->
  il codice viene riverificato e riusato senza chiamare l'LLM;
* **esempi (RAG)**: i componenti più simili alla specifica corrente diventano
  esempi verificati nel prompt del Coder.

Anti-leakage: ``exclude_tasks`` esclude i componenti nati dal task in esame
(in valutazione la libreria parte vuota o leave-one-task-out).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .embed import Embedder


@dataclass
class Component:
    name: str  # nome del modulo Chisel
    kind: str  # "block:<key>" per i blocchi del catalogo, "task:<nome>" per i top-level
    params: dict
    spec: str
    interface: str
    code: str
    origin_task: str  # task della run che l'ha prodotto (per l'anti-leakage)
    model: str = ""
    run: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def id(self) -> str:
        h = hashlib.sha256(self.code.encode()).hexdigest()[:10]
        kind = re.sub(r"[^\w]", "_", self.kind)
        return f"{kind}__{self.name}__{h}"

    def doc(self) -> str:
        return f"{self.kind} {self.name}\n{self.spec}\n{self.interface}"


class Library:
    def __init__(self, root: str | Path, embedder: Embedder | None = None):
        self.root = Path(root)
        self.embedder = embedder or Embedder(None)
        self.last_method = "none"

    def _items(self) -> list[Component]:
        if not self.root.is_dir():
            return []
        out = []
        for p in sorted(self.root.glob("*.json")):
            try:
                out.append(Component(**json.loads(p.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def add(self, c: Component) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.root / f"{c.id}.json"
        p.write_text(json.dumps(asdict(c), indent=1), encoding="utf-8")
        return p

    def __len__(self) -> int:
        return len(self._items())

    def exact(self, kind: str, params: dict, exclude_tasks: set[str] = frozenset()) -> list[Component]:
        """Componenti dello stesso blocco con gli stessi parametri (più recenti prima)."""
        hits = [c for c in self._items() if c.kind == kind and c.params == params
                and c.origin_task not in exclude_tasks]
        return sorted(hits, key=lambda c: c.created, reverse=True)

    def search(self, query: str, k: int = 2, exclude_tasks: set[str] = frozenset(),
               min_score: float = 0.05) -> list[tuple[Component, float]]:
        items = [c for c in self._items() if c.origin_task not in exclude_tasks]
        scores, self.last_method = self.embedder.rank(query, [c.doc() for c in items])
        ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)
        seen, out = set(), []
        for c, s in ranked:
            if s < min_score or c.kind in seen:
                continue
            seen.add(c.kind)  # un esempio per tipo di componente
            out.append((c, s))
            if len(out) == k:
                break
        return out
