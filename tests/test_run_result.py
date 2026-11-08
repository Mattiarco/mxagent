"""Chiusura esplicita delle run: result.json ed evento ``end`` in ogni caso
(successo, fallimento, eccezione, segnale, budget), con LLM e verifier finti."""
import json
import signal
from pathlib import Path

import pytest

from mxagent.agents.orchestrator import Orchestrator
from mxagent.bench import load_task
from mxagent.config import load_config
from mxagent.llm.chat import ScriptedChat
from mxagent.run.baselines import run_loop
from mxagent.run.result import RunInterrupted, interruptible, llm_by_role, ppa_attempts
from mxagent.verify.verifier import VerifyReport

ROOT = Path(__file__).resolve().parent.parent
B01 = load_task(ROOT / "benchmarks" / "b01_e2m1_decode")


def cfg(**over):
    base = {"agent.best_of_n": 1, "agent.spec_parser": False, "agent.decompose": False,
            "agent.library": False, "sim.sbt_mode": "batch", "agent.ppa": False,
            "synth.backends": [], "agent.escape": False}
    return load_config(ROOT / "configs" / "default.yaml", {**base, **over})


def code(mark):
    return f"```scala\nimport chisel3._\nclass E2M1Decode extends RawModule {{ {mark} }}\n```"


class Verifier:
    """'// ok' passa, '// boom' solleva ``exc``, il resto è un mismatch."""

    def __init__(self, task, exc=None):
        self.task, self.exc = task, exc

    def verify(self, sources, tag):
        src = sources[f"{self.task.top}.scala"]
        if "// boom" in src:
            raise self.exc
        ok = "// ok" in src
        r = VerifyReport(tag, self.task.name, "pass" if ok else "sim", ok)
        r.sim = {"pass_rate": 1.0 if ok else 0.5, "total": 16}
        r.error_keys = [] if ok else ["mismatch:y"]
        return r

    def close(self):
        pass


def run(tmp, replies, exc=None, **over):
    o = Orchestrator(B01, cfg(**over), seed=1, runs_dir=tmp, llm=ScriptedChat(replies),
                     verifier_factory=lambda t, n: Verifier(t, exc))
    return o


def load(o):
    res = json.loads((o.rec.root / "result.json").read_text(encoding="utf-8"))
    trace = json.loads((o.rec.root / "trace.json").read_text(encoding="utf-8"))
    return res, trace


def test_result_on_success(tmp_path):
    o = run(tmp_path, [code("// bug"), code("// ok")], **{"budget.max_fix_iters": 2})
    o.run()
    res, trace = load(o)
    assert res["status"] == "done" and res["reason"] == "pass" and res["passed"]
    assert res["pass_rate"] == 1.0 and res["n_vectors"] == 16
    assert res["iterations_to_pass"] == 2 and res["time_to_pass_s"] is not None
    assert res["best_candidate"] == "E2M1Decode_01"
    assert res["llm_calls"] == 2 and res["tokens_in"] > 0 and res["tokens_out"] > 0
    assert set(res["llm_by_role"]) == {"coder", "reflector"}
    assert sum(r["calls"] for r in res["llm_by_role"].values()) == 2
    assert trace[-1]["kind"] == "end" and trace[-1]["status"] == "done"
    assert o.states[-1] == "DONE"
    stops = [e for e in trace if e["kind"] == "loop_stop"]
    assert stops and stops[0]["reason"] == "pass" and stops[0]["iters_max"] == 2


def test_result_on_failure_with_reason(tmp_path):
    o = run(tmp_path, [code("// bug")] * 3, **{"budget.max_fix_iters": 2})
    o.run()
    res, trace = load(o)
    assert res["status"] == "failed" and res["reason"] == "max_iters" and not res["passed"]
    assert res["pass_rate"] == 0.5 and res["iterations_to_pass"] is None
    assert o.states[-1] == "FAILED"
    stop = [e for e in trace if e["kind"] == "loop_stop"][-1]
    assert stop["reason"] == "max_iters" and stop["iters_done"] == 2 == stop["iters_max"]


def test_result_written_when_exception_mid_run(tmp_path):
    o = run(tmp_path, [code("// bug"), code("// boom")], exc=RuntimeError("verilator crashed"),
            **{"budget.max_fix_iters": 3})
    with pytest.raises(RuntimeError, match="verilator crashed"):
        o.run()
    res, trace = load(o)
    assert res["status"] == "failed" and res["reason"] == "exception"
    assert "RuntimeError: verilator crashed" in res["error"]
    assert not res["passed"] and res["llm_calls"] == 2
    assert res["best_candidate"] is None  # nessun design ancora consolidato
    assert trace[-1]["kind"] == "end" and trace[-1]["status"] == "failed"
    assert not (o.rec.root / "metrics.json").exists()


def test_result_written_when_interrupted(tmp_path):
    o = run(tmp_path, [code("// boom")], exc=RunInterrupted("SIGHUP"))
    with pytest.raises(RunInterrupted):
        o.run()
    res, trace = load(o)
    assert res["status"] == "interrupted" and res["reason"] == "signal SIGHUP"
    assert "INTERRUPTED" in o.states and trace[-1]["kind"] == "end"


def test_real_sigint_becomes_interrupted(tmp_path):
    class SigVerifier(Verifier):
        def verify(self, sources, tag):
            signal.raise_signal(signal.SIGINT)
            return super().verify(sources, tag)

    o = Orchestrator(B01, cfg(), seed=1, runs_dir=tmp_path, llm=ScriptedChat([code("// ok")]),
                     verifier_factory=lambda t, n: SigVerifier(t))
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(RunInterrupted):
        o.run()
    assert signal.getsignal(signal.SIGINT) is before  # handler ripristinato
    res, _ = load(o)
    assert res["status"] == "interrupted" and res["reason"] == "signal SIGINT"


def test_budget_exhausted_is_recorded(tmp_path):
    o = run(tmp_path, [code("// bug")] * 5, **{"budget.max_fix_iters": 4, "budget.max_llm_calls": 2})
    o.run()
    res, trace = load(o)
    assert res["status"] == "failed" and res["reason"].startswith("budget")
    ev = [e for e in trace if e["kind"] == "budget_exhausted"]
    assert ev and ev[0]["limit"] == "max_llm_calls" and ev[0]["stage"] == "MONOLITHIC"


def test_baseline_writes_result_even_on_exception(tmp_path):
    c = cfg(**{"budget.max_fix_iters": 3})
    v = Verifier(B01, RuntimeError("sbt died"))
    with pytest.raises(RuntimeError):
        run_loop(B01, c, "baseline-b", seed=0, role=c.llm.roles["coder"], runs_dir=tmp_path,
                 llm=ScriptedChat([code("// bug"), code("// boom")]), verify_fn=v.verify,
                 run_id="r")
    res = json.loads((tmp_path / "r" / "result.json").read_text())
    assert res["status"] == "failed" and "sbt died" in res["error"] and res["llm_calls"] == 2
    assert res["llm_by_role"] == {"coder": {"calls": 2, "tokens_in": res["tokens_in"],
                                            "tokens_out": res["tokens_out"], "cached": 0}}

    ok = run_loop(B01, c, "baseline-b", seed=0, role=c.llm.roles["coder"], runs_dir=tmp_path,
                  llm=ScriptedChat([code("// ok")]), verify_fn=v.verify, run_id="ok")
    res = json.loads((ok.run_dir / "result.json").read_text())
    assert res["status"] == "done" and res["iterations_to_pass"] == 1


def test_interruptible_outside_main_thread_is_noop():
    import threading
    out = []

    def body():
        with interruptible():
            out.append("ran")
    t = threading.Thread(target=body)
    t.start()
    t.join()
    assert out == ["ran"]


def test_helpers():
    ev = [{"kind": "llm", "role": "coder", "tokens": [10, 5]},
          {"kind": "llm", "role": "optimizer", "tokens": [7, 3], "cached": True},
          {"kind": "verify"}, {"kind": "llm", "role": "coder", "tokens": [1, 1]}]
    assert llm_by_role(ev) == {"coder": {"calls": 2, "tokens_in": 11, "tokens_out": 6, "cached": 0},
                               "optimizer": {"calls": 1, "tokens_in": 7, "tokens_out": 3, "cached": 1}}
    out = {"optimizer": {"steps": [{"outcome": "accepted"}, {"outcome": "rejected"},
                                   {"outcome": "duplicate"}, {"outcome": "rejected"}]}}
    assert ppa_attempts(out) == {"total": 4, "accepted": 1, "rejected": 2, "functional_fail": 0,
                                 "synth_fail": 0, "no_code": 0, "duplicate": 1}
    assert ppa_attempts(None)["total"] == 0
