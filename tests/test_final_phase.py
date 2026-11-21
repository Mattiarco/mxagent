"""result.json esiste sempre, anche se la sintesi di best/ (fase finale, dopo l'esito)
fallisce, viene interrotta da un segnale o non finisce: run s2 su b05, arrivata a FAILED
(max_iters) e rimasta nella sintesi dei pari merito MXFP4Dot32_02/_09/_10 senza
result.json né metrics.json."""
import json
import time

import pytest

from mxagent.llm.chat import ScriptedChat
from mxagent.run import best as _best
from mxagent.run.baselines import run_loop
from mxagent.run.result import RunInterrupted
from tests.test_agents import B01, FakeVerifier, cfg, code
from tests.test_baselines import BAD, CFG, ROLE, TASK


def _full(tmp_path, synth):
    from mxagent.agents.orchestrator import Orchestrator
    c = cfg(**{"agent.decompose": False, "budget.max_fix_iters": 1, "synth.backends": ["generic"]})
    llm = ScriptedChat([code("E2M1Decode", "// bug")] * 6)
    log = []
    o = Orchestrator(B01, c, seed=1, runs_dir=tmp_path, llm=llm,
                     verifier_factory=lambda t, n: FakeVerifier(t, log), synth_fn=synth)
    return o


def _result(root):
    return json.loads((root / "result.json").read_text())


def test_full_exception_in_best_synthesis_still_writes_result(tmp_path):
    seen = []

    def synth(verilog, task, s, work, backends=None):
        res = _result(work.parents[1])  # provvisorio: c'è già, con l'esito
        seen.append((res["status"], res["final_phase"]))
        raise RuntimeError("yosys esploso")
    o = _full(tmp_path, synth)
    m = o.run().metrics
    res = _result(o.rec.root)
    assert seen and seen[0] == ("failed", "best")
    assert res["status"] == "failed" and res["reason"] == "max_iters" and not res["passed"]
    assert res["final_phase"] == "done" and (o.rec.root / "metrics.json").is_file()
    assert res["best"]["solution"] == "partial" and "yosys esploso" in res["best"]["synth"]["error"]
    assert m["stop"] == "max_iters"


def test_full_signal_during_best_synthesis_keeps_outcome_and_records_error(tmp_path):
    def synth(*a, **k):
        raise RunInterrupted("SIGTERM")
    o = _full(tmp_path, synth)
    o.run()
    res = _result(o.rec.root)
    assert res["status"] == "failed" and res["reason"] == "max_iters"  # esito già noto
    assert res["final_phase_error"].startswith("interrupted: SIGTERM")
    assert res["best"]["solution"] == "partial"  # scelto senza sintesi
    assert (o.rec.root / "metrics.json").is_file()
    trace = json.loads((o.rec.root / "trace.json").read_text())
    assert trace[-1]["kind"] == "end"


def test_full_crash_of_the_best_phase_is_in_final_phase_error(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise OSError("disco pieno a metà di best/")
    monkeypatch.setattr(_best, "write_best", broken)
    o = _full(tmp_path, lambda *a, **k: None)
    o.run()
    res = _result(o.rec.root)
    assert res["status"] == "failed" and "disco pieno" in res["final_phase_error"]
    assert res["best"]["solution"] == "none"


@pytest.mark.parametrize("exc,field", [(RuntimeError("yosys esploso"), None),
                                       (RunInterrupted("SIGTERM"), "interrupted: SIGTERM")])
def test_baseline_best_failure_still_writes_result(tmp_path, exc, field):
    seen = []

    def synth(verilog, tag):
        seen.append(_result(tmp_path / "r")["final_phase"])
        raise exc
    c = CFG.model_copy(deep=True)
    run_loop(TASK, c, "baseline-b", seed=0, role=ROLE, runs_dir=tmp_path,
             llm=ScriptedChat([BAD] * 4), verify_fn=_mismatch, run_id="r", synth_fn=synth)
    res = _result(tmp_path / "r")
    assert seen and seen[0] == "best"
    assert res["status"] == "failed" and res["final_phase"] == "done"
    assert (tmp_path / "r" / "metrics.json").is_file()
    if field:
        assert res["final_phase_error"].startswith(field)
    else:
        assert res["final_phase_error"] is None and "yosys esploso" in res["best"]["synth"]["error"]


def _mismatch(sources, tag):
    from mxagent.verify.verifier import VerifyReport
    r = VerifyReport(tag, TASK.name, "sim", False)
    r.sim = {"pass_rate": 0.5, "total": 16, "failed": 8}
    r.verilog = ["x.sv"]
    r.feedback = r.feedback_raw = "TEST FAILED"
    r.error_keys = ["mismatch"]
    return r


def test_best_synthesis_stops_at_the_deadline():
    calls = []
    cands = [_best.Candidate(f"c{i}", {}, _mismatch({}, f"c{i}"), seq=i) for i in range(3)]
    b = _best.select_best(cands, lambda v, t: calls.append(t) or {"ok": True},
                          deadline=time.monotonic() - 1)
    assert not calls and all(c.synth == {"ok": False, "skipped": "best_timeout"} for c in cands)
    assert b.tag == "c2"  # stesso criterio: il più recente fra i pari merito


def test_capped_timeout_never_exceeds_the_time_left():
    s = CFG.synth
    assert _best.capped(s, None) is s
    assert _best.capped(s, time.monotonic() + 100).timeout_s <= 100
    assert _best.capped(s, time.monotonic() + 10 ** 6).timeout_s == s.timeout_s
    assert CFG.synth.best_timeout_s == 600
