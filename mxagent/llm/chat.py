"""Chat con il modello: output strutturato, retry, timeout, cache, budget.

* Il ciclo di chiamata (cache, retry, timeout, budget) è uno solo, in :class:`BaseChat`;
  i backend (:class:`OllamaChat`, :class:`~mxagent.llm.anthropic_chat.AnthropicChat`)
  implementano solo UN tentativo di richiesta. La pipeline vede soltanto
  :class:`ChatModel`: non sa quale backend sta usando (:func:`make_chat`).
* Tutte le opzioni che influenzano l'output (modello + digest, messaggi,
  ``seed``, ``temperature``, ``num_ctx``, ``num_predict``, ``format``) entrano
  nella chiave di cache: rieseguire una run con la cache piena è deterministico
  e non chiama il modello.
* ``num_ctx`` è sempre esplicito: il default di Ollama è basso e tronca i
  prompt senza errore.
* Ogni chiamata ha un timeout (``llm.timeout_s``). Un tentativo scaduto si ripete
  come gli altri errori temporanei; se scadono tutti i tentativi la chiamata NON
  solleva: restituisce una risposta vuota con ``error="timeout"``, che la pipeline
  tratta come un candidato senza codice (tentativo fallito) e prosegue.
* Ogni chiamata è conteggiata nel :class:`Budget` condiviso dalla run; la
  stessa contabilità vale per baseline e sistema completo (confronto a pari costo).
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .ollama import OllamaClient


class BudgetExceeded(RuntimeError):
    """Budget della run esaurito; ``limit`` = max_llm_calls | max_tokens | max_wall_s."""

    def __init__(self, msg: str, limit: str = ""):
        super().__init__(msg)
        self.limit = limit


class LLMTimeout(Exception):
    """Un tentativo di chiamata è scaduto (sollevata dai backend, gestita da BaseChat)."""


class TransientLLMError(Exception):
    """Errore temporaneo (rete, 5xx, 429): il tentativo si ripete dopo un'attesa."""


@dataclass
class Budget:
    max_calls: int = 60
    max_tokens: int = 400_000
    max_wall_s: float = 1800
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0
    #: tentativi scaduti (anche quelli poi ripetuti con successo) e loro durata
    timeouts: int = 0
    timeout_seconds: float = 0.0
    #: chiamate finite in errore (tutti i tentativi scaduti): candidati persi
    failed_calls: int = 0
    #: costo stimato (solo backend a pagamento; None = nessuna stima)
    cost_usd: float | None = None
    started: float = field(default_factory=time.monotonic)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def wall(self) -> float:
        return time.monotonic() - self.started

    def check(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"chiamate LLM esaurite ({self.calls}/{self.max_calls})", "max_llm_calls")
        if self.tokens >= self.max_tokens:
            raise BudgetExceeded(f"token esauriti ({self.tokens}/{self.max_tokens})", "max_tokens")
        if self.wall >= self.max_wall_s:
            raise BudgetExceeded(f"tempo esaurito ({self.wall:.0f}/{self.max_wall_s:.0f}s)", "max_wall_s")

    def charge(self, r: "LLMResponse") -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.llm_seconds += r.seconds + sum(r.timeouts)
        self.timeouts += len(r.timeouts)
        self.timeout_seconds += sum(r.timeouts)
        self.failed_calls += bool(r.error)
        if r.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + r.cost_usd

    def to_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "tokens": self.tokens,
                "llm_seconds": round(self.llm_seconds, 1), "wall_seconds": round(self.wall, 1),
                "timeouts": self.timeouts, "timeout_seconds": round(self.timeout_seconds, 1),
                "failed_calls": self.failed_calls,
                "cost_usd": round(self.cost_usd, 4) if self.cost_usd is not None else None}


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False
    data: Any = None  # JSON già decodificato se è stato chiesto un ``format``
    done_reason: str = ""
    #: "" = risposta ricevuta; "timeout" = tutti i tentativi scaduti (text vuoto)
    error: str = ""
    #: durata (s) di ogni tentativo scaduto di questa chiamata
    timeouts: list[float] = field(default_factory=list)
    attempts: int = 1
    cost_usd: float | None = None
    #: adattamenti interni al backend per questa chiamata (es. formato JSON non vincolato)
    notes: list[str] = field(default_factory=list)
    #: cartella della run che ha ottenuto la risposta dal modello (salvata nella cache: una
    #: risposta letta dalla cache dice da quale run viene). None = voce senza provenienza
    origin: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, str]], *, model: str, temperature: float,
             num_ctx: int, num_predict: int, seed: int, format: Any = None,
             tag: str = "") -> LLMResponse: ...


class BaseChat:
    """Cache, budget, retry e timeout comuni a tutti i backend.

    Un backend implementa :meth:`_payload` (richiesta + chiave di cache),
    :meth:`_request` (UN tentativo: solleva :class:`LLMTimeout`,
    :class:`TransientLLMError` o ``RuntimeError`` per gli errori definitivi),
    :meth:`digest` e :meth:`describe`."""

    backend = "?"

    def __init__(self, cache_dir: str | Path | None = ".cache/llm", timeout: float = 300,
                 max_retries: int = 2, budget: Budget | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        #: run che usa questo client (cartella): finisce in ``origin`` delle risposte nuove
        self.origin: str | None = None
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.budget = budget

    # ------------------------------------------------------ da implementare
    def digest(self, model: str) -> str:
        raise NotImplementedError

    def _payload(self, messages, *, model, temperature, num_ctx, num_predict, seed,
                 format) -> dict:
        raise NotImplementedError

    def _request(self, payload: dict) -> LLMResponse:
        raise NotImplementedError

    def describe(self, roles: dict | None = None) -> dict:
        """Backend e parametri EFFETTIVI inviati al modello (per meta.json)."""
        return {"backend": self.backend, "timeout_s": self.timeout, "max_retries": self.max_retries}

    # ----------------------------------------------------------- comune
    def _key(self, payload: dict) -> str:
        blob = json.dumps({**payload, "digest": self.digest(payload["model"])}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages, *, model, temperature, num_ctx, num_predict, seed, format=None,
             tag=""):
        if self.budget:
            self.budget.check()
        payload = self._payload(messages, model=model, temperature=temperature, num_ctx=num_ctx,
                                num_predict=num_predict, seed=seed, format=format)
        key = self._key(payload)
        cfile = self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None
        if cfile and cfile.is_file():
            r = LLMResponse(**json.loads(cfile.read_text(encoding="utf-8")))
            r.cached, r.timeouts = True, []  # i timeout di allora non si ripetono
        else:
            r = self._call(payload)
            r.origin = self.origin  # la provenienza non entra nella chiave
            if cfile and not r.error:  # un timeout non si mette mai in cache
                cfile.parent.mkdir(parents=True, exist_ok=True)
                cfile.write_text(json.dumps(r.to_dict()), encoding="utf-8")
        if format is not None and r.data is None:
            r.data = parse_json(r.text)
        if self.budget:
            self.budget.charge(r)  # anche le risposte in cache contano: il costo è della run
        return r

    def _call(self, payload: dict) -> LLMResponse:
        """Fino a ``max_retries + 1`` tentativi. Un tentativo scaduto si ripete subito
        (il tempo è già passato); un errore temporaneo dopo un'attesa crescente."""
        last: Exception | None = None
        timeouts: list[float] = []
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                r = self._request(payload)
            except LLMTimeout as e:
                timeouts.append(round(time.monotonic() - t0, 2))
                last = e
                continue
            except TransientLLMError as e:
                last = e
                time.sleep(min(30, 2 ** attempt * 3))
                continue
            r.seconds = time.monotonic() - t0
            r.timeouts, r.attempts = timeouts, attempt + 1
            return r
        if isinstance(last, LLMTimeout):
            return LLMResponse(text="", model=payload["model"], error="timeout", timeouts=timeouts,
                               attempts=self.max_retries + 1, done_reason="timeout")
        raise RuntimeError(f"{self.backend} non risponde dopo {self.max_retries + 1} tentativi: {last}")


class OllamaChat(BaseChat):
    backend = "ollama"

    def __init__(self, host: str | None = None, cache_dir: str | Path | None = ".cache/llm",
                 timeout: float = 300, max_retries: int = 2, budget: Budget | None = None):
        super().__init__(cache_dir, timeout, max_retries, budget)
        self.client = OllamaClient(host, timeout=timeout)
        self._digests: dict[str, str] = {}

    def digest(self, model: str) -> str:
        if model not in self._digests:
            try:
                tags = self.client._request("/api/tags").get("models", [])
                self._digests.update({m["name"]: m.get("digest", "") for m in tags})
            except Exception:  # noqa: BLE001
                pass
            self._digests.setdefault(model, "")
        return self._digests[model]

    def _payload(self, messages, *, model, temperature, num_ctx, num_predict, seed, format):
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature, "num_ctx": num_ctx,
                               "num_predict": num_predict, "seed": seed}}
        if format is not None:
            payload["format"] = format
        return payload

    def _request(self, payload: dict) -> LLMResponse:
        try:
            d = self.client._request("/api/chat", payload)
        except urllib.error.HTTPError as e:
            if e.code < 500:
                body = e.read().decode(errors="replace")[:300]
                raise RuntimeError(f"Ollama HTTP {e.code}: {body}") from e
            raise TransientLLMError(f"Ollama HTTP {e.code}") from e
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as e:
            if isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError):
                raise LLMTimeout(str(e)) from e
            raise TransientLLMError(str(e)) from e
        return LLMResponse(
            text=d.get("message", {}).get("content", ""), model=payload["model"],
            prompt_tokens=int(d.get("prompt_eval_count") or 0),
            completion_tokens=int(d.get("eval_count") or 0),
            done_reason=d.get("done_reason", ""))

    def describe(self, roles: dict | None = None) -> dict:
        out = super().describe()
        out["host"] = self.client.host
        out["params"] = {r: {"model": v.model, "temperature": v.temperature, "num_ctx": v.num_ctx,
                             "num_predict": v.num_predict, "seed": "sent (per call)"}
                         for r, v in (roles or {}).items()}
        out["structured_output"] = "format = JSON schema (constrained decoding)"
        return out


def make_chat(llm_cfg, budget: Budget | None = None) -> BaseChat:
    """Client del backend della config (``llm.backend``): stessa interfaccia per tutti."""
    timeout = llm_cfg.effective_timeout_s
    if llm_cfg.backend == "anthropic":
        from .anthropic_chat import AnthropicChat
        return AnthropicChat(llm_cfg.cache_dir, timeout, llm_cfg.max_retries, budget,
                             thinking=llm_cfg.anthropic.thinking, effort=llm_cfg.anthropic.effort)
    return OllamaChat(llm_cfg.host, llm_cfg.cache_dir, timeout, llm_cfg.max_retries, budget=budget)


def model_digest(llm: Any, model: str) -> str:
    """Digest del modello per meta.json (``scripted`` per gli LLM finti dei test)."""
    return llm.digest(model) if isinstance(llm, BaseChat) else "scripted"


def describe_llm(llm: Any, roles: dict | None = None) -> dict:
    return llm.describe(roles) if isinstance(llm, BaseChat) else {"backend": "scripted"}


def parse_json(text: str) -> Any:
    """JSON da una risposta: intero testo, oppure il primo oggetto/array bilanciato."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_, close in (("{", "}"), ("[", "]")):
        start = text.find(open_)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    esc = (ch == "\\") and not esc
                    if ch == '"' and not esc:
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == open_:
                    depth += 1
                elif ch == close:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find(open_, start + 1)
    return None


#: risposta di ScriptedChat che simula una chiamata con tutti i tentativi scaduti
TIMEOUT = object()


class ScriptedChat:
    """LLM finto per i test: restituisce risposte predefinite in ordine
    (:data:`TIMEOUT` simula una chiamata scaduta)."""

    def __init__(self, replies: list[str], budget: Budget | None = None):
        self.replies = list(replies)
        self.budget = budget
        self.requests: list[dict] = []

    def chat(self, messages, *, model, temperature, num_ctx, num_predict, seed, format=None,
             tag=""):
        if self.budget:
            self.budget.check()
        self.requests.append({"messages": [dict(m) for m in messages], "model": model,
                              "temperature": temperature,
                              "seed": seed, "format": format, "tag": tag})
        text = self.replies.pop(0) if self.replies else ""
        if text is TIMEOUT:
            r = LLMResponse(text="", model=model, error="timeout", timeouts=[1.0], attempts=1,
                            done_reason="timeout")
        else:
            r = LLMResponse(text=text, model=model,
                            prompt_tokens=sum(len(m["content"]) for m in messages) // 4,
                            completion_tokens=len(text) // 4, seconds=0.0,
                            data=parse_json(text) if format is not None else None)
        if self.budget:
            self.budget.charge(r)
        return r
