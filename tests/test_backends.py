"""Scheda del blocco, timeout per chiamata, miglior candidato, backend Anthropic (mockato)
e script di confronto. Nessun modello vero, nessuna toolchain."""
import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from mxagent.agents.schemas import PLAN_SCHEMA, SPEC_SCHEMA
from mxagent.bench.subkernels import CATALOG, block_task
from mxagent.config import load_config
from mxagent.eval.stats import fisher_exact, mann_whitney_u
from mxagent.formats.cards import format_card
from mxagent.rag.library import Library
from mxagent.llm.chat import (TIMEOUT, BaseChat, Budget, LLMResponse, LLMTimeout, OllamaChat,
                              ScriptedChat, make_chat)
from mxagent.run import best as B
from mxagent.run.baselines import run_loop
from mxagent.verify.verifier import VerifyReport

from .test_agents import B01, B05, PLAN, cfg, code, orch
from .test_baselines import BAD, GOOD, ROLE, TASK, fake_verify

ROOT = Path(__file__).resolve().parent.parent


# =============================================================== 1. scheda del blocco
def _parent_markers(task) -> list[str]:
    first = next(ln for ln in task.spec_text.splitlines() if ln.strip() and not ln.startswith("#"))
    return [format_card(num=task.numerics_obj), "FORMAT MXFP4", "Numeric semantics required",
            "Accumulation:", first.strip()]


def test_every_block_has_its_own_card_without_parent_card():
    parent = format_card(num=B05.numerics_obj)
    for key in CATALOG:
        t = block_task(key, "X", numerics=B05.numerics)
        assert t.domain_text, key  # ogni blocco ha una scheda sua
        assert t.domain_text not in parent or len(t.domain_text) < len(parent)
        assert "FORMAT MXFP4" not in t.domain_text and "Numeric semantics" not in t.domain_text
    # la semantica numerica del padre che serve al blocco è esplicitata nella scheda
    f2f = block_task("fixed_to_fp32", "X", numerics={"out_subnormals": "flush"}).domain_text
    assert "subnormals flushed to zero" in f2f
    assert "subnormals kept" in block_task("fixed_to_fp32", "X", numerics=B05.numerics).domain_text


def test_block_prompts_never_contain_parent_card(tmp_path):
    # E2M1Mul sbaglia al primo colpo: prompt di generazione E di riflessione del blocco
    replies = [json.dumps(PLAN), code("E2M1Mul", "bug"), code("E2M1Mul"), code("AdderTree"),
               code("E8M0Pair"), code("FixedToFP32"), code("MXFP4Dot32")]
    o, llm, _ = orch(B05, cfg(), replies, tmp_path, library=Library(tmp_path / 'lib'))
    assert o.run().metrics["passed"]
    block_reqs = [r for r in llm.requests if r["tag"].startswith("block:")]
    tags = [r["tag"] for r in block_reqs]
    assert any(":reflect" in t for t in tags) and any(":gen" in t for t in tags)
    for r in block_reqs:
        text = "\n".join(m["content"] for m in r["messages"])
        for marker in _parent_markers(B05):
            assert marker not in text, (r["tag"], marker[:60])
    mul = next(r for r in block_reqs if "E2M1Mul" in r["tag"])["messages"][1]["content"]
    assert "All 16 codes:" in mul and "DOMAIN KNOWLEDGE" in mul
    f2f = next(r for r in block_reqs if "FixedToFP32" in r["tag"])["messages"][1]["content"]
    assert "Output rounding: RNE" in f2f
    # il top integrato (il padre stesso) ha la scheda del formato
    top = next(r for r in llm.requests if r["tag"].startswith("integrate:"))["messages"][1]["content"]
    assert "FORMAT MXFP4" in top


# ==================================================================== 2. timeout
class _Fake(BaseChat):
    backend = "fake"

    def __init__(self, plan, **kw):
        super().__init__(**kw)
        self.plan, self.n = list(plan), 0

    def digest(self, model):
        return "d"

    def _payload(self, messages, **kw):
        return {"model": kw["model"], "messages": messages, "seed": kw["seed"]}

    def _request(self, payload):
        self.n += 1
        step = self.plan.pop(0)
        if step == "timeout":
            raise LLMTimeout("slow")
        return LLMResponse(text=step, model=payload["model"], prompt_tokens=10, completion_tokens=5)


KW = dict(model="m", temperature=0.2, num_ctx=4096, num_predict=256)


def test_timeout_all_attempts_returns_error_and_is_not_cached(tmp_path):
    b = Budget()
    llm = _Fake(["timeout", "timeout"], cache_dir=tmp_path, timeout=1, max_retries=1, budget=b)
    r = llm.chat([{"role": "user", "content": "x"}], seed=1, **KW)
    assert r.error == "timeout" and r.text == "" and len(r.timeouts) == 2 and llm.n == 2
    assert b.timeouts == 2 and b.failed_calls == 1 and b.calls == 1
    assert not list(tmp_path.rglob("*.json"))  # niente cache per un timeout
    d = b.to_dict()
    assert d["timeouts"] == 2 and d["failed_calls"] == 1 and "timeout_seconds" in d


def test_timeout_then_success_is_retried_and_cached(tmp_path):
    llm = _Fake(["timeout", "ok"], cache_dir=tmp_path, timeout=1, max_retries=2, budget=Budget())
    r = llm.chat([{"role": "user", "content": "x"}], seed=1, **KW)
    assert r.text == "ok" and not r.error and len(r.timeouts) == 1 and r.attempts == 2
    r2 = llm.chat([{"role": "user", "content": "x"}], seed=1, **KW)
    assert r2.cached and r2.timeouts == [] and llm.n == 2


class _Slow(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"models": []}')

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        time.sleep(1.0)
        try:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"message": {"content": "late"}}')
        except OSError:
            pass


def test_ollama_socket_timeout_becomes_timeout_error():
    srv = HTTPServer(("127.0.0.1", 0), _Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = OllamaChat(f"http://127.0.0.1:{srv.server_port}", cache_dir=None, timeout=0.3,
                         max_retries=1)
        r = llm.chat([{"role": "user", "content": "x"}], seed=0, **KW)
        assert r.error == "timeout" and len(r.timeouts) == 2
    finally:
        srv.shutdown()


def test_timeout_in_full_run_is_a_failed_attempt_and_pipeline_continues(tmp_path):
    c = cfg(**{"agent.decompose": False})
    o, llm, _ = orch(B01, c, [TIMEOUT, code("E2M1Decode")], tmp_path,
                     library=Library(tmp_path / 'lib'))
    m = o.run().metrics
    assert m["passed"] and m["no_code"] == 1 and m["llm"]["timeouts"] == 1
    assert m["llm"]["failed_calls"] == 1
    ev = o.rec.events
    to = [e for e in ev if e["kind"] == "llm_timeout"]
    assert to and to[0]["error"] == "timeout" and to[0]["tag"].startswith("monolithic:E2M1Decode")
    assert "seconds" in to[0]
    assert any(e["kind"] == "verify" and e.get("llm_error") == "timeout" for e in ev)
    assert any(e["kind"] == "llm" and e["outcome"] == "timeout" for e in ev)
    res = json.loads((o.rec.root / "result.json").read_text())
    assert res["llm_timeouts"] == 1 and res["llm_failed_calls"] == 1 and "llm_seconds" in res


def test_timeout_in_baseline_does_not_crash(tmp_path):
    llm = ScriptedChat([TIMEOUT, GOOD])
    res = run_loop(TASK, load_config(ROOT / "configs" / "default.yaml",
                                     {"budget.max_fix_iters": 3, "synth.backends": []}),
                   "baseline-b", seed=0, role=ROLE, runs_dir=tmp_path, llm=llm,
                   verify_fn=fake_verify)
    assert res.metrics["passed"] and res.metrics["llm"]["timeouts"] == 1


def test_timeout_defaults_and_cli_override():
    c = load_config(ROOT / "configs" / "default.yaml")
    assert c.llm.backend == "ollama" and c.llm.effective_timeout_s == 300
    c = load_config(ROOT / "configs" / "default.yaml", {"llm.backend": "anthropic"})
    assert c.llm.effective_timeout_s == 120
    c = load_config(ROOT / "configs" / "default.yaml", {"llm.timeout_s": 42})
    assert c.llm.effective_timeout_s == 42
    with pytest.raises(ValueError):
        load_config(ROOT / "configs" / "default.yaml", {"llm.backend": "openai"})


# ============================================================= 3. miglior candidato
def _rep(tag, stage, rate=None, total=100):
    r = VerifyReport(tag, "t", stage, stage == "pass")
    if rate is not None:
        r.sim = {"pass_rate": rate, "total": total, "failed": round(total * (1 - rate))}
    return r


def test_rank_key_order_and_synth_only_for_ties():
    cs = [B.Candidate("c0", {}, _rep("c0", "compile"), seq=5),
          B.Candidate("lint", {}, _rep("lint", "lint"), seq=1),
          B.Candidate("s50a", {}, _rep("s50a", "sim", 0.5), seq=2),
          B.Candidate("s50b", {}, _rep("s50b", "sim", 0.5), seq=3),
          B.Candidate("s40", {}, _rep("s40", "sim", 0.4), seq=4)]
    synthed = []

    def synth(verilog, tag):
        synthed.append(tag)
        return {"ok": tag == "s50a", "transistors": 100}

    best = B.select_best(cs, synth)
    # (b) 0.5 > 0.4 > lint (0) ; fra i pari merito (c) decide la sintesi, non (d) la recenza
    assert best.tag == "s50a" and sorted(synthed) == ["s50a", "s50b"]
    # senza sintesi a parità vince il più recente (d)
    assert B.select_best([B.Candidate(c.tag, {}, c.report, c.seq) for c in cs]).tag == "s50b"
    # (a) prima di tutto: un candidato elaborato batte uno che non compila anche se più recente
    assert B.rank_key(cs[1]) > B.rank_key(cs[0])


def test_failed_full_run_writes_partial_best(tmp_path):
    c = cfg(**{"agent.decompose": False, "budget.max_fix_iters": 1, "agent.escape": False})
    o, llm, _ = orch(B01, c, [code("E2M1Decode", "bug1"), code("E2M1Decode", "bug2")], tmp_path,
                     library=Library(tmp_path / 'lib'))
    m = o.run().metrics
    assert not m["passed"] and m["solution"] == "partial"
    rep = json.loads((o.rec.root / "best" / "report.json").read_text())
    assert rep["solution"] == "partial" and rep["pass_rate"] == 0.5 and rep["elaborated"]
    assert rep["iteration"] == 2  # a parità (0.5) vince il più recente
    assert (o.rec.root / "best" / "E2M1Decode.scala").is_file()
    res = json.loads((o.rec.root / "result.json").read_text())
    assert res["solution"] == "partial" and res["best"]["tag"] == rep["tag"]


def test_passed_full_run_best_is_complete(tmp_path):
    o, _, _ = orch(B01, cfg(**{"agent.decompose": False}), [code("E2M1Decode")], tmp_path,
                   library=Library(tmp_path / 'lib'))
    m = o.run().metrics
    assert m["solution"] == "complete" and m["best"]["phase"] == "final"
    assert json.loads((o.rec.root / "result.json").read_text())["solution"] == "complete"


def test_baseline_best_partial(tmp_path):
    c = load_config(ROOT / "configs" / "default.yaml", {"budget.max_fix_iters": 1})
    res = run_loop(TASK, c, "baseline-b", seed=0, role=ROLE, runs_dir=tmp_path,
                   llm=ScriptedChat([BAD, BAD]), verify_fn=fake_verify,
                   synth_fn=lambda v, t: {"ok": True, "transistors": 10, "depth": 3})
    assert res.metrics["solution"] == "partial"
    rep = json.loads((res.run_dir / "best" / "report.json").read_text())
    assert rep["synth"]["ok"] is True and rep["synth"]["transistors"] == 10 and rep["iteration"] == 2
    assert "SOLUZIONE PARZIALE" in B.describe(res.metrics["best"], res.run_dir)


# ===================================================== 4. backend Anthropic (mockato)
class _Msgs:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **req):
        self.owner.reqs.append(req)
        nxt = self.owner.plan.pop(0) if self.owner.plan else "ok"
        if isinstance(nxt, Exception):
            raise nxt
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=nxt)],
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                                                     cache_read_input_tokens=0,
                                                     cache_creation_input_tokens=0),
                               stop_reason="end_turn")


class FakeAnthropic:
    def __init__(self, plan=()):
        self.plan, self.reqs, self.timeouts = list(plan), [], []
        self.messages = _Msgs(self)

    def with_options(self, timeout=None, **_):
        self.timeouts.append(timeout)
        return self


def _anth(plan=(), **kw):
    from mxagent.llm.anthropic_chat import AnthropicChat
    return AnthropicChat(None, 120, kw.pop("max_retries", 2), kw.pop("budget", None),
                         client=FakeAnthropic(plan), **kw)


def test_anthropic_request_mapping_and_cost():
    llm = _anth(["hello"], budget=Budget())
    r = llm.chat([{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}],
                 model="claude-sonnet-5-5", temperature=0.4, num_ctx=16384, num_predict=4096, seed=3)
    req = llm.client.reqs[0]
    assert req["system"] == "SYS" and req["messages"] == [{"role": "user", "content": "U"}]
    assert req["max_tokens"] == 4096 and "temperature" not in req and "seed" not in req
    assert req["thinking"] == {"type": "between_tools"} and llm.client.timeouts == [120]
    assert r.text == "hello" and (r.prompt_tokens, r.completion_tokens) == (1000, 200)
    assert r.cost_usd == pytest.approx((1000 * 2 + 200 * 10) / 1e6)
    assert llm.budget.cost_usd == pytest.approx(r.cost_usd)
    d = llm.describe({"coder": ROLE})
    assert d["backend"] == "anthropic" and "not sent" in d["params"]["coder"]["temperature"]
    assert "sk-" not in json.dumps(d)


def test_anthropic_structured_output_strict_or_unconstrained():
    llm = _anth(['{"strategy": "monolithic"}', '{"operation": "decode"}'])
    r = llm.chat([{"role": "user", "content": "p"}], model="claude-sonnet-5-5", temperature=0.2,
                 num_ctx=1, num_predict=256, seed=0, format=PLAN_SCHEMA)
    assert "output_config" not in llm.client.reqs[0]  # params libero: non esprimibile
    assert r.data == {"strategy": "monolithic"} and r.notes
    llm.chat([{"role": "user", "content": "s"}], model="claude-sonnet-5-5", temperature=0.2,
             num_ctx=1, num_predict=256, seed=0, format=SPEC_SCHEMA)
    fmt = llm.client.reqs[1]["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
    assert fmt["schema"]["properties"]["ports"]["items"]["additionalProperties"] is False


def test_anthropic_timeout_uses_common_retry_logic():
    import anthropic
    err = anthropic.APITimeoutError(request=None)
    llm = _anth([err, err, err], max_retries=2, budget=Budget())
    r = llm.chat([{"role": "user", "content": "x"}], model="claude-sonnet-5-5", temperature=0.2,
                 num_ctx=1, num_predict=64, seed=0)
    assert r.error == "timeout" and len(r.timeouts) == 3 and llm.budget.timeouts == 3


def test_anthropic_requires_env_key(monkeypatch):
    from mxagent.llm.anthropic_chat import AnthropicChat
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicChat(None, 120, 0, None)
    c = load_config(ROOT / "configs" / "default.yaml", {"llm.backend": "anthropic"})
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_chat(c.llm)


def test_full_pipeline_runs_unchanged_on_anthropic_backend(tmp_path):
    good = code("E2M1Decode")
    llm = _anth([good])
    from mxagent.agents.orchestrator import Orchestrator
    from .test_agents import FakeVerifier
    log = []
    from mxagent.config import LLM_ROLES
    c = cfg(**{"agent.decompose": False, "llm.backend": "anthropic",
               **{f"llm.roles.{r}.model": "claude-sonnet-5-5" for r in LLM_ROLES}})
    o = Orchestrator(B01, c, seed=1, runs_dir=tmp_path, llm=llm,
                     verifier_factory=lambda t, n: FakeVerifier(t, log), library=Library(tmp_path / 'lib'))
    m = o.run().metrics
    assert m["passed"] and m["solution"] == "complete"
    meta = json.loads((o.rec.root / "meta.json").read_text())
    assert meta["llm_backend"]["backend"] == "anthropic" and meta["model_digest"].startswith("anthropic")
    res = json.loads((o.rec.root / "result.json").read_text())
    assert res["llm_backend"] == "anthropic" and res["cost_usd"] > 0 and res["llm_timeout_s"] == 120


# ========================================================== 5. statistica e confronto
def test_fisher_and_mann_whitney():
    assert fisher_exact(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)  # tè di Fisher
    assert fisher_exact(10, 0, 0, 10) < 1e-4
    assert fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)
    mw = mann_whitney_u(range(10), range(10, 20))
    assert mw["u"] == 0 and mw["p"] < 0.001
    assert mann_whitney_u([1, 2, 3], [1, 2, 3])["p"] == pytest.approx(1.0)
    assert mann_whitney_u([], [1])["p"] is None


def _compare_module():
    spec = importlib.util.spec_from_file_location("compare_backends",
                                                  ROOT / "scripts" / "compare_backends.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_error_classification():
    cb = _compare_module()
    assert cb.classify({"stage": "no_code", "llm_error": "timeout"}) == ["timeout"]
    assert "unparsable" in cb.classify({"stage": "no_code"})[0]
    assert cb.classify({"stage": "compile", "error_keys": ["scalac:X:bad"]}) == ["Chisel/Scala compile error"]
    assert cb.classify({"stage": "compile", "error_keys": ["firtool:X:bad"]}) == ["elaboration error"]
    fb = ("FUNCTIONAL MISMATCH: 3/10\nPatterns:\n - every failing vector has a_scale=nan; 0/7 "
          "passing vectors do\n - 80% of failing vector has a=subnormal; 1/7 passing vectors do")
    assert cb.classify({"stage": "sim", "error_keys": ["mismatch:y"], "feedback": fb}) == \
        ["tests failed: NaN", "tests failed: subnormal"]
    fb2 = " - every failing vector has a_scale=max; 0/7 passing vectors do"
    assert cb.mismatch_cases(fb2) == ["scaling"]


def test_compare_report_from_runs(tmp_path):
    cb = _compare_module()
    for backend, model, passed in (("ollama", "q:14b", [1, 0, 0]), ("anthropic", "claude-sonnet-5-5", [1, 1, 1])):
        for seed, p in enumerate(passed):
            d = tmp_path / "runs" / cb.run_id("b01", "full", backend, model, seed)
            d.mkdir(parents=True)
            res = {"task": "b01", "mode": "full", "model": model, "seed": seed, "status": "done",
                   "passed": bool(p), "solution": "complete" if p else "partial",
                   "best": {"pass_rate": 1.0 if p else 0.5, "tests_passed": 16 if p else 8,
                            "tests_total": 16, "synth": {"ok": True, "transistors": 100}},
                   "candidates": 2, "llm_timeouts": 0, "wall_s": 10 + seed, "tokens_out": 50,
                   "llm_backend": backend}
            (d / "result.json").write_text(json.dumps(res))
            (d / "trace.json").write_text(json.dumps(
                [] if p else [{"kind": "verify", "stage": "compile", "passed": False,
                               "error_keys": ["scalac:X:e"]}]))
    cb.report(tmp_path)
    md = (tmp_path / "compare.md").read_text(encoding="utf-8")
    assert "Analisi degli errori" in md and "Chisel/Scala compile error" in md
    assert "p Fisher" in md
    rows = (tmp_path / "compare_summary.csv").read_text().splitlines()
    assert len(rows) == 3
