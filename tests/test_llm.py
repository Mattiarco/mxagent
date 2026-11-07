"""Client Ollama contro un server HTTP finto (nessun modello vero)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from mxagent.llm.chat import Budget, BudgetExceeded, OllamaChat, parse_json


class FakeOllama(BaseHTTPRequestHandler):
    calls: list = []
    fail_next = 0

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": "m:1", "digest": "abc"}]})
        else:
            self._send(200, {"version": "x"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.calls.append(req)
        if FakeOllama.fail_next:
            FakeOllama.fail_next -= 1
            return self._send(500, {"error": "boom"})
        content = '{"ok": true}' if "format" in req else f"echo seed={req['options']['seed']}"
        self._send(200, {"message": {"role": "assistant", "content": content},
                         "prompt_eval_count": 100, "eval_count": 20, "done_reason": "stop"})


@pytest.fixture
def server():
    FakeOllama.calls, FakeOllama.fail_next = [], 0
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


KW = dict(model="m:1", temperature=0.2, num_ctx=4096, num_predict=256)


def test_chat_options_cache_and_budget(server, tmp_path):
    b = Budget(max_calls=10)
    llm = OllamaChat(server, cache_dir=tmp_path, budget=b)
    msgs = [{"role": "user", "content": "hi"}]
    r1 = llm.chat(msgs, seed=7, **KW)
    assert r1.text == "echo seed=7" and not r1.cached and (r1.prompt_tokens, r1.completion_tokens) == (100, 20)
    sent = FakeOllama.calls[0]
    assert sent["options"] == {"temperature": 0.2, "num_ctx": 4096, "num_predict": 256, "seed": 7}
    assert sent["stream"] is False
    r2 = llm.chat(msgs, seed=7, **KW)
    assert r2.cached and r2.text == r1.text and len(FakeOllama.calls) == 1  # dalla cache
    llm.chat(msgs, seed=8, **KW)  # seed diverso -> nuova chiamata
    assert len(FakeOllama.calls) == 2
    assert b.calls == 3 and b.tokens == 360  # anche le risposte in cache contano nel budget


def test_structured_output(server, tmp_path):
    llm = OllamaChat(server, cache_dir=None)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    r = llm.chat([{"role": "user", "content": "x"}], seed=0, format=schema, **KW)
    assert FakeOllama.calls[-1]["format"] == schema and r.data == {"ok": True}


def test_retry_on_server_error(server, monkeypatch):
    monkeypatch.setattr("mxagent.llm.chat.time.sleep", lambda s: None)
    FakeOllama.fail_next = 1
    r = OllamaChat(server, cache_dir=None, max_retries=2).chat([{"role": "user", "content": "x"}],
                                                               seed=1, **KW)
    assert r.text == "echo seed=1" and len(FakeOllama.calls) == 2


def test_budget_exceeded(server):
    b = Budget(max_calls=1)
    llm = OllamaChat(server, cache_dir=None, budget=b)
    llm.chat([{"role": "user", "content": "x"}], seed=1, **KW)
    with pytest.raises(BudgetExceeded):
        llm.chat([{"role": "user", "content": "x"}], seed=2, **KW)


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Sure! ```json\n{"a": {"b": "}"}}\n``` done') == {"a": {"b": "}"}}
    assert parse_json("[1, 2]") == [1, 2]
    assert parse_json("nothing here") is None
