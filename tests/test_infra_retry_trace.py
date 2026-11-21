"""Ogni nuovo tentativo di verifica dopo un errore di sistema lascia un evento
``infra_retry`` nel trace (tentativo, attesa, motivo, JVM vive) e result.json ha
``infra_retries``; il modello non riceve mai quel feedback. Full e baseline."""
import json

import pytest

from mxagent.llm.chat import ScriptedChat
from mxagent.run.baselines import run_loop
from mxagent.verify.verifier import Verifier, VerifyReport, _InfraHit
from tests.test_baselines import BAD, CFG, GOOD, ROLE, TASK

DETAIL = "native_thread: java.lang.OutOfMemoryError: unable to create native thread"


class FlakyOnce(Verifier):
    """Primo tentativo: errore di sistema; poi passa se il codice contiene 'ok', altrimenti
    errore di compilazione vero (col suo feedback)."""

    def __init__(self, task, work, fails=1):
        super().__init__(task, work, infra_backoff=(0, 0, 0))
        self.fails, self.calls, self.cleanups = fails, 0, 0

    def _verify_once(self, sources, tag):
        self.calls += 1
        if self.calls <= self.fails:
            raise _InfraHit(DETAIL, "compile")
        code = sources[f"{self.task.top}.scala"]
        if "ok" in code:
            r = VerifyReport(tag, self.task.name, "pass", True)
            r.sim = {"pass_rate": 1.0}
            return r
        r = VerifyReport(tag, self.task.name, "compile", False)
        r.feedback = r.feedback_raw = "COMPILATION FAILED:\n[scalac] X.scala:3:5: REAL ERROR"
        r.error_keys = ["scalac:X.scala:REAL ERROR"]
        return r

    def kill_leftovers(self):
        self.cleanups += 1
        return [4242]


def _sent(llm) -> str:
    return "\n".join(m["content"] for r in llm.requests for m in r["messages"])


def _check(root, llm, n_retries):
    trace = json.loads((root / "trace.json").read_text())
    ev = [e for e in trace if e["kind"] == "infra_retry"]
    assert len(ev) == n_retries
    e = ev[0]
    assert e["attempt"] == 1 and e["wait_s"] == 0 and e["where"] == "compile"
    assert e["reason"] == DETAIL and e["killed"] == [4242] and isinstance(e["java"], list)
    res = json.loads((root / "result.json").read_text())
    assert len(res["infra_retries"]) == n_retries
    assert res["infra_retries"][0]["reason"] == DETAIL and res["infra_retries"][0]["tag"]
    sent = _sent(llm)
    assert "native thread" not in sent and "infra" not in sent.lower()
    return res


@pytest.mark.parametrize("mode", ["baseline-a", "baseline-b", "baseline-b-matched"])
def test_baselines_trace_retries_and_never_feed_them_back(tmp_path, mode):
    llm = ScriptedChat([BAD, GOOD, GOOD])
    v = FlakyOnce(TASK, tmp_path / "v")
    res = run_loop(TASK, CFG, mode, seed=0, role=ROLE, runs_dir=tmp_path, llm=llm, verifier=v,
                   run_id="r")
    r = _check(tmp_path / "r", llm, 1)
    if mode == "baseline-a":  # una sola generazione: il retry non la consuma
        assert not res.metrics["passed"] and v.calls == 2 and r["candidates"] == 1
        return
    assert res.metrics["passed"] and v.calls == 3  # 1 errore di sistema + 2 candidati veri
    assert r["candidates"] == 2 and r["reflect_rounds"] == 1  # nessuna iterazione consumata
    assert "REAL ERROR" in _sent(llm)  # il feedback vero arriva, quello di sistema no


def test_full_traces_retries_and_never_feeds_them_back(tmp_path):
    from tests.test_agents import B01, cfg, code
    from mxagent.agents.orchestrator import Orchestrator
    llm = ScriptedChat([code("E2M1Decode", "// bug"), code("E2M1Decode", "// ok")])
    v = FlakyOnce(B01, tmp_path / "v", fails=2)
    o = Orchestrator(B01, cfg(**{"agent.decompose": False}), seed=1, runs_dir=tmp_path, llm=llm,
                     verifier_factory=lambda t, n: v)
    m = o.run().metrics
    # 2 errori di sistema + 2 candidati + la verifica ripetuta del design finale (final/)
    assert m["passed"] and len(llm.requests) == 2 and v.calls == 5
    res = _check(o.rec.root, llm, 2)
    assert [r["attempt"] for r in res["infra_retries"]] == [1, 2]
    assert res["candidates"] == 2 and "REAL ERROR" in _sent(llm)


def test_no_retry_no_event(tmp_path):
    llm = ScriptedChat([GOOD])
    v = FlakyOnce(TASK, tmp_path / "v", fails=0)
    run_loop(TASK, CFG, "baseline-b", seed=0, role=ROLE, runs_dir=tmp_path, llm=llm, verifier=v,
             run_id="r")
    res = json.loads((tmp_path / "r" / "result.json").read_text())
    assert res["infra_retries"] == []
