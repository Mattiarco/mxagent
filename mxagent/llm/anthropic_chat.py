"""Backend cloud: Claude via API Anthropic (SDK ufficiale ``anthropic``).

Serve da termine di confronto per Ollama: stessa interfaccia (:class:`ChatModel`),
stessi prompt, stessa estrazione del codice, stessa cache, stessi retry e timeout
(il ciclo è quello di :class:`~mxagent.llm.chat.BaseChat`; qui c'è solo UN tentativo).

La chiave si legge SOLO dall'ambiente (``ANTHROPIC_API_KEY``, letta dall'SDK):
non compare mai nel codice, nella config, nella cache o nei log.

Adattamenti confinati in questo modulo (la pipeline non cambia):

* messaggi: i messaggi ``system`` diventano il parametro ``system`` dell'API;
* ``num_predict`` -> ``max_tokens``; ``num_ctx`` non si invia (contesto fisso del modello);
* ``temperature`` NON si invia: con l'SDK 2.x (API ``2026-10-21``) i parametri di
  campionamento danno 400 su ogni modello, e i modelli recenti accettano solo il
  default (1.0). Il valore chiesto dal ruolo entra comunque nella chiave di cache;
* ``seed``: l'API non lo supporta. Resta nella chiave di cache, così N run con seed
  diversi fanno N chiamate distinte (e una run rieseguita è riprodotta dalla cache);
* output strutturato: lo schema Ollama (``format``) diventa ``output_config.format``
  quando è esprimibile (ogni oggetto con ``properties`` riceve
  ``additionalProperties: false``); se non lo è (es. ``params`` libero nel piano) si
  invia senza vincolo e il JSON si estrae dal testo con lo stesso ``parse_json``.
  Ogni ripiego è annotato in ``LLMResponse.notes``;
* ``thinking``: ``between_tools`` (default) = nessun ragionamento esteso; i token
  di ragionamento, se attivo, sono fatturati come output e contati come tali.
"""
from __future__ import annotations

import copy
import os
from typing import Any

from .chat import BaseChat, Budget, LLMResponse, LLMTimeout, TransientLLMError

#: prezzi API in USD per milione di token (input, output); stima del costo per chiamata
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}
#: lettura dalla cache dei prompt: 1/10 del prezzo di input; scrittura: 1.25x
CACHE_READ, CACHE_WRITE = 0.1, 1.25


def estimate_cost(model: str, tin: int, tout: int, cache_read: int = 0,
                  cache_write: int = 0) -> float | None:
    p = PRICES_USD_PER_MTOK.get(model)
    if p is None:
        return None
    pin, pout = p
    return (tin * pin + cache_read * pin * CACHE_READ + cache_write * pin * CACHE_WRITE
            + tout * pout) / 1e6


def strict_schema(schema: Any) -> dict | None:
    """Schema per ``output_config.format`` o None se non esprimibile.

    Ogni oggetto deve avere ``properties`` (riceve ``additionalProperties: false``);
    un oggetto libero (``{"type": "object"}`` senza proprietà) non è esprimibile."""
    if not isinstance(schema, dict):
        return None
    s = copy.deepcopy(schema)
    ok = True

    def walk(node: Any) -> None:
        nonlocal ok
        if isinstance(node, dict):
            if node.get("type") == "object":
                if not node.get("properties"):
                    ok = False
                    return
                node["additionalProperties"] = False
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(s)
    return s if ok else None


class AnthropicChat(BaseChat):
    backend = "anthropic"

    def __init__(self, cache_dir=".cache/llm", timeout: float = 120, max_retries: int = 2,
                 budget: Budget | None = None, *, thinking: str = "between_tools",
                 effort: str | None = None, client: Any = None):
        super().__init__(cache_dir, timeout, max_retries, budget)
        self.thinking, self.effort = thinking, effort
        import anthropic  # import pigro: chi usa solo Ollama non ne ha bisogno
        self._sdk = anthropic
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("backend anthropic: variabile d'ambiente ANTHROPIC_API_KEY "
                                   "non impostata")
            # retry e timeout li gestisce BaseChat (identici a Ollama): niente retry nell'SDK
            client = anthropic.Anthropic(max_retries=0, timeout=timeout)
        self.client = client

    def digest(self, model: str) -> str:
        return f"anthropic-api:{model}"  # i modelli API sono identificati dal nome

    def _payload(self, messages, *, model, temperature, num_ctx, num_predict, seed, format):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages
                if m["role"] != "system"]
        req: dict[str, Any] = {"model": model, "max_tokens": int(num_predict), "messages": msgs}
        if system:
            req["system"] = system
        if self.thinking != "omit":
            req["thinking"] = {"type": self.thinking}
        oc: dict[str, Any] = {}
        if self.effort:
            oc["effort"] = self.effort
        notes = []
        if format is not None:
            sch = strict_schema(format)
            if sch is not None:
                oc["format"] = {"type": "json_schema", "schema": sch}
            else:
                notes.append("format: schema not expressible as strict JSON schema; "
                             "sent unconstrained, JSON parsed from text")
        if oc:
            req["output_config"] = oc
        # chiave di cache: richiesta effettiva + ciò che il ruolo ha chiesto e l'API ignora
        return {"backend": "anthropic", "model": model, "request": req, "seed": seed,
                "temperature_requested": temperature, "num_ctx_requested": num_ctx,
                "notes": notes}

    def _request(self, payload: dict) -> LLMResponse:
        sdk = self._sdk
        req = payload["request"]
        notes = list(payload.get("notes") or [])
        try:
            try:
                resp = self.client.with_options(timeout=self.timeout).messages.create(**req)
            except sdk.BadRequestError as e:
                fmt = (req.get("output_config") or {}).get("format")
                if fmt is None:
                    raise
                # schema rifiutato dall'API: stessa richiesta senza vincolo di formato
                oc = {k: v for k, v in req["output_config"].items() if k != "format"}
                req2 = {k: v for k, v in req.items() if k != "output_config"}
                if oc:
                    req2["output_config"] = oc
                notes.append(f"format: rejected by the API ({str(e.message)[:120]}); "
                             "sent unconstrained, JSON parsed from text")
                resp = self.client.with_options(timeout=self.timeout).messages.create(**req2)
        except sdk.APITimeoutError as e:
            raise LLMTimeout("anthropic: request timed out") from e
        except (sdk.RateLimitError, sdk.InternalServerError, sdk.APIConnectionError) as e:
            raise TransientLLMError(f"anthropic: {type(e).__name__}") from e
        except sdk.APIStatusError as e:
            if e.status_code >= 500:  # 529 overloaded e simili
                raise TransientLLMError(f"anthropic HTTP {e.status_code}") from e
            raise RuntimeError(f"anthropic HTTP {e.status_code}: {str(e.message)[:300]}") from e
        text = "".join(getattr(b, "text", "") for b in resp.content if b.type == "text")
        u = resp.usage
        cr = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cw = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        tin, tout = int(u.input_tokens or 0), int(u.output_tokens or 0)
        return LLMResponse(text=text, model=payload["model"], prompt_tokens=tin + cr + cw,
                           completion_tokens=tout, done_reason=str(resp.stop_reason or ""),
                           cost_usd=estimate_cost(payload["model"], tin, tout, cr, cw),
                           notes=notes)

    def describe(self, roles: dict | None = None) -> dict:
        out = super().describe()
        out["sdk_version"] = getattr(self._sdk, "__version__", "?")
        out["params"] = {
            r: {"model": v.model, "max_tokens": v.num_predict,
                "temperature": "not sent: API default 1.0 (requested %s; sampling params are "
                               "rejected by API 2026-10-21 / current models)" % v.temperature,
                "seed": "not supported by the API (kept only in the cache key)",
                "num_ctx": "not sent (model context window)",
                "thinking": self.thinking, "effort": self.effort or "model default",
                "price_usd_per_mtok": PRICES_USD_PER_MTOK.get(v.model)}
            for r, v in (roles or {}).items()}
        out["structured_output"] = ("output_config.format (strict JSON schema) when expressible, "
                                    "otherwise unconstrained + parse_json")
        return out
