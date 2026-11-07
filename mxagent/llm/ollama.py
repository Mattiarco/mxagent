"""Client HTTP minimale per Ollama (solo libreria standard).

In M1 serve a ``doctor`` (versione, modelli installati, context length);
la chat con output strutturato, retry e cache arriva in M3.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_HOST = "http://localhost:11434"


def normalize_host(host: str | None) -> str:
    host = host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    if host.count(":") == 1:  # senza porta
        host += ":11434"
    return host.rstrip("/")


def normalize_model(name: str) -> str:
    return name if ":" in name else name + ":latest"


class OllamaClient:
    def __init__(self, host: str | None = None, timeout: float = 30):
        self.host = normalize_host(host)
        self.timeout = timeout

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.host + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if data is None else "POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def version(self) -> str:
        return self._request("/api/version").get("version", "?")

    def models(self) -> list[str]:
        return [m["name"] for m in self._request("/api/tags").get("models", [])]

    def show(self, model: str) -> dict[str, Any]:
        """Metadati del modello: context length nativa, parametri, quantizzazione, digest."""
        d = self._request("/api/show", {"model": model})
        info = d.get("model_info", {}) or {}
        ctx = next((v for k, v in info.items() if k.endswith(".context_length")), None)
        det = d.get("details", {}) or {}
        return {"context_length": ctx, "parameter_size": det.get("parameter_size"),
                "quantization": det.get("quantization_level"), "family": det.get("family")}


__all__ = ["OllamaClient", "normalize_host", "normalize_model", "urllib"]
_ = urllib.error  # riesportato per chi intercetta gli errori di rete
