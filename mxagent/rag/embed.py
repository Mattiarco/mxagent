"""Similarità testuale: embedding locali Ollama (``/api/embed``) con ripiego TF-IDF.

Il ripiego lessicale rende il retrieval deterministico e disponibile anche senza
un modello di embedding installato; il metodo usato viene registrato nella run.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..llm.ollama import OllamaClient

_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokens(text: str) -> list[str]:
    out = []
    for t in _TOK.findall(text):
        # spezza CamelCase e snake_case: FixedToFP32 -> fixed, to, fp32
        for p in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+|[A-Z]+", t):
            out.append(p.lower())
    return out


def tfidf_rank(query: str, docs: list[str]) -> list[float]:
    toks = [Counter(tokens(d)) for d in docs]
    n = len(docs)
    df = Counter(t for c in toks for t in c)
    idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in df}

    def vec(c: Counter) -> dict[str, float]:
        return {t: (1 + math.log(v)) * idf.get(t, 1.0) for t, v in c.items()}

    q = vec(Counter(tokens(query)))
    qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
    scores = []
    for c in toks:
        d = vec(c)
        dn = math.sqrt(sum(v * v for v in d.values())) or 1.0
        scores.append(sum(q[t] * d.get(t, 0.0) for t in q) / (qn * dn))
    return scores


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class Embedder:
    """Embedding con Ollama; ``available`` è False se il modello non risponde."""

    def __init__(self, model: str | None, host: str | None = None):
        self.model = model
        self.client = OllamaClient(host, timeout=120)
        self.available = bool(model)
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self.available:
            return None
        todo = [t for t in texts if t not in self._cache]
        if todo:
            try:
                d = self.client._request("/api/embed", {"model": self.model, "input": todo})
                for t, e in zip(todo, d.get("embeddings", [])):
                    self._cache[t] = e
            except Exception:  # noqa: BLE001 - modello assente o server vecchio: ripiego TF-IDF
                self.available = False
                return None
        return [self._cache[t] for t in texts]

    def rank(self, query: str, docs: list[str]) -> tuple[list[float], str]:
        if not docs:
            return [], "none"
        embs = self.embed([query, *docs])
        if embs:
            return [cosine(embs[0], e) for e in embs[1:]], f"embed:{self.model}"
        return tfidf_rank(query, docs), "tfidf"
