"""Chat con Ollama (``/api/chat``): output strutturato, retry, cache, budget.

* Tutte le opzioni che influenzano l'output (modello + digest, messaggi,
  ``seed``, ``temperature``, ``num_ctx``, ``num_predict``, ``format``) entrano
  nella chiave di cache: rieseguire una run con la cache piena è deterministico
  e non chiama il modello.
* ``num_ctx`` è sempre esplicito: il default di Ollama è basso e tronca i
  prompt senza errore.
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


@dataclass
class Budget:
    max_calls: int = 60
    max_tokens: int = 400_000
    max_wall_s: float = 1800
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0
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
        self.llm_seconds += r.seconds

    def to_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "tokens": self.tokens,
                "llm_seconds": round(self.llm_seconds, 1), "wall_seconds": round(self.wall, 1)}


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

    def to_dict(self) -> dict:
        return asdict(self)


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, str]], *, model: str, temperature: float,
             num_ctx: int, num_predict: int, seed: int, format: Any = None,
             tag: str = "") -> LLMResponse: ...


class OllamaChat:
    def __init__(self, host: str | None = None, cache_dir: str | Path | None = ".cache/llm",
                 timeout: float = 600, max_retries: int = 2, budget: Budget | None = None):
        self.client = OllamaClient(host, timeout=timeout)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        self.budget = budget
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

    def _key(self, payload: dict) -> str:
        blob = json.dumps({**payload, "digest": self.digest(payload["model"])}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages, *, model, temperature, num_ctx, num_predict, seed, format=None,
             tag=""):
        if self.budget:
            self.budget.check()
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature, "num_ctx": num_ctx,
                               "num_predict": num_predict, "seed": seed}}
        if format is not None:
            payload["format"] = format
        key = self._key(payload)
        cfile = self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None
        if cfile and cfile.is_file():
            r = LLMResponse(**json.loads(cfile.read_text(encoding="utf-8")))
            r.cached = True
        else:
            r = self._call(payload)
            if cfile:
                cfile.parent.mkdir(parents=True, exist_ok=True)
                cfile.write_text(json.dumps(r.to_dict()), encoding="utf-8")
        if format is not None and r.data is None:
            r.data = parse_json(r.text)
        if self.budget:
            self.budget.charge(r)  # anche le risposte in cache contano: il costo è della run
        return r

    def _call(self, payload: dict) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                d = self.client._request("/api/chat", payload)
                return LLMResponse(
                    text=d.get("message", {}).get("content", ""), model=payload["model"],
                    prompt_tokens=int(d.get("prompt_eval_count") or 0),
                    completion_tokens=int(d.get("eval_count") or 0),
                    seconds=time.monotonic() - t0, done_reason=d.get("done_reason", ""))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = e
                if isinstance(e, urllib.error.HTTPError) and e.code < 500:
                    body = e.read().decode(errors="replace")[:300]
                    raise RuntimeError(f"Ollama HTTP {e.code}: {body}") from e
                time.sleep(min(30, 2 ** attempt * 3))
        raise RuntimeError(f"Ollama non risponde dopo {self.max_retries + 1} tentativi: {last}")


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


class ScriptedChat:
    """LLM finto per i test: restituisce risposte predefinite in ordine."""

    def __init__(self, replies: list[str], budget: Budget | None = None):
        self.replies = list(replies)
        self.budget = budget
        self.requests: list[dict] = []

    def chat(self, messages, *, model, temperature, num_ctx, num_predict, seed, format=None,
             tag=""):
        if self.budget:
            self.budget.check()
        self.requests.append({"messages": [dict(m) for m in messages], "model": model,
                              "seed": seed, "format": format, "tag": tag})
        text = self.replies.pop(0) if self.replies else ""
        r = LLMResponse(text=text, model=model, prompt_tokens=sum(len(m["content"]) for m in messages) // 4,
                        completion_tokens=len(text) // 4, seconds=0.0,
                        data=parse_json(text) if format is not None else None)
        if self.budget:
            self.budget.charge(r)
        return r
