"""Confronto a pari budget: baseline-b-matched, llm_calls_to_pass, successo per k chiamate.

LLM, verifier e sintesi finti (nessuna toolchain, nessun modello).
"""
import json
from pathlib import Path

import pytest

from mxagent.eval.campaign import DEFAULT_MODES, Campaign, load_campaign
from mxagent.eval.report import calls_to_pass, success_by_calls
from mxagent.llm.chat import Budget, ScriptedChat
from mxagent.run.baselines import MODES, run_loop, settings_for
from mxagent.run.result import llm_calls_to_pass
from tests.test_agents import B05, PLAN, cfg as agents_cfg, code as agents_code, orch
from tests.test_baselines import BAD, CFG, GOOD, ROLE, TASK, fake_verify
from tests.test_final import code as final_code, make as final_make

ROOT = Path(__file__).resolve().parent.parent


def _result(run_dir: Path) -> dict:
    return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))


def _trace(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))


# ------------------------------------------------------- baseline-b-matched
def test_matched_mode_is_registered_and_derived_from_budget():
    assert "baseline-b-matched" in MODES
    c = CFG.model_copy(deep=True)
    c.budget.max_llm_calls = 37
    assert settings_for("baseline-b-matched", c).max_fix_iters == 36  # 1 + 36 = max_llm_calls
    assert settings_for("baseline-b", c).max_fix_iters == c.budget.max_fix_iters  # invariata
    b, m = settings_for("baseline-b", c), settings_for("baseline-b-matched", c)
    assert (b.feedback, b.prompt_set, b.history) == (m.feedback, m.prompt_set, m.history)


def test_matched_stops_at_max_llm_calls_not_at_7(tmp_path):
    c = CFG.model_copy(deep=True)
    c.budget.max_fix_iters, c.budget.max_llm_calls = 6, 20
    llm_b = ScriptedChat([BAD] * 30)
    rb = run_loop(TASK, c, "baseline-b", seed=0, role=ROLE, runs_dir=tmp_path / "b", llm=llm_b,
                  verify_fn=fake_verify)
    assert len(llm_b.requests) == 7 and rb.metrics["llm"]["calls"] == 7
    llm_m = ScriptedChat([BAD] * 30)
    rm = run_loop(TASK, c, "baseline-b-matched", seed=0, role=ROLE, runs_dir=tmp_path / "m",
                  llm=llm_m, verify_fn=fake_verify)
    assert len(llm_m.requests) == 20 and rm.metrics["llm"]["calls"] == 20
    assert not rm.metrics["passed"] and rm.metrics["max_fix_iters"] == 19
    res = _result(rm.run_dir)
    assert res["mode"] == "baseline-b-matched" and res["llm_calls"] == 20
    assert res["llm_calls_to_pass"] is None and res["max_fix_iters"] == 19
    # stessi prompt e log grezzi di B: il primo messaggio è identico
    assert llm_m.requests[0]["messages"] == llm_b.requests[0]["messages"]
    assert "TEST FAILED" in llm_m.requests[-1]["messages"][-1]["content"]


def test_matched_also_stops_on_the_shared_token_budget(tmp_path):
    llm = ScriptedChat([BAD] * 80, budget=Budget(max_calls=60, max_tokens=3000))
    r = run_loop(TASK, CFG, "baseline-b-matched", seed=0, role=ROLE, runs_dir=tmp_path, llm=llm,
                 verify_fn=fake_verify)
    assert r.metrics["stop"].startswith("budget") and "token" in r.metrics["stop"]
    assert len(llm.requests) < 60


def test_matched_passes_and_counts_calls_to_pass(tmp_path):
    llm = ScriptedChat([BAD] * 11 + [GOOD])
    r = run_loop(TASK, CFG, "baseline-b-matched", seed=0, role=ROLE, runs_dir=tmp_path, llm=llm,
                 verify_fn=fake_verify)
    assert r.metrics["passed"] and r.metrics["llm_calls_to_pass"] == 12
    assert _result(r.run_dir)["llm_calls_to_pass"] == 12


def test_sixty_turn_history_is_truncated_and_recorded(tmp_path):
    """~60 turni con contesto piccolo: nessun crash, sistema e primo messaggio utente
    sempre presenti, l'ultimo feedback sempre presente, troncamenti in result.json."""
    c = CFG.model_copy(deep=True)
    c.budget.max_llm_calls = 60
    role = ROLE.model_copy(update={"num_ctx": 4096, "num_predict": 1024})
    long_bad = BAD.replace("/* bug */", "/* bug */ " + "x" * 2000)
    llm = ScriptedChat([long_bad] * 60, budget=Budget(max_calls=60, max_tokens=10**9))
    r = run_loop(TASK, c, "baseline-b-matched", seed=0, role=role, runs_dir=tmp_path, llm=llm,
                 verify_fn=fake_verify)
    assert len(llm.requests) == 60
    first_user = llm.requests[0]["messages"][1]["content"]
    for req in llm.requests:
        msgs = req["messages"]
        assert msgs[0]["role"] == "system" and msgs[1]["content"] == first_user
        assert msgs[-1]["role"] == "user" and [m["role"] for m in msgs[2:]] == \
            ["assistant", "user"] * ((len(msgs) - 2) // 2)
    for req in llm.requests[1:]:
        assert "TEST FAILED" in req["messages"][-1]["content"]  # ultimo feedback tenuto
    m, res = r.metrics, _result(r.run_dir)
    assert m["history_truncations"] > 0 and m["history_truncated_calls"] > 0
    assert res["history_truncations"] == m["history_truncations"]
    assert res["history_truncated_calls"] == m["history_truncated_calls"]
    ev = [e for e in _trace(r.run_dir) if e["kind"] == "history_truncated"]
    assert len(ev) == m["history_truncated_calls"]
    assert sum(e["dropped_pairs"] for e in ev) == m["history_truncations"]
    assert max(len(q["messages"]) for q in llm.requests) < 20  # la storia non cresce senza limiti


# --------------------------------------------------- llm_calls_to_pass
def test_llm_calls_to_pass_on_synthetic_traces():
    ev = [{"kind": "llm"}, {"kind": "verify", "module": "Blk", "passed": True},
          {"kind": "llm"}, {"kind": "verify", "module": "Top", "passed": False},
          {"kind": "llm"}, {"kind": "verify", "module": "Top", "passed": True},
          {"kind": "llm"}, {"kind": "verify", "module": "Top", "passed": True}]  # PPA dopo il pass
    assert llm_calls_to_pass(ev, "Top") == 3  # il pass del blocco non chiude il conteggio
    assert llm_calls_to_pass(ev[:4], "Top") is None
    base = [{"kind": "llm"}, {"kind": "verify", "passed": False}, {"kind": "llm"},
            {"kind": "verify", "passed": True}]
    assert llm_calls_to_pass(base, "E2M1Decode") == 2  # baseline: verify senza module


def test_full_calls_to_pass_excludes_ppa_calls_after_pass(tmp_path):
    o, _ = final_make(tmp_path, [final_code("// bug"), final_code("// ok 100,10"),
                                 final_code("// ok 120,10"), final_code("// ok 70,10")],
                      **{"budget.max_fix_iters": 2})
    r = o.run()
    res = _result(r.run_dir)
    assert res["passed"] and res["llm_calls"] == 4  # 2 fino al pass + 2 del PPA Optimizer
    assert res["llm_calls_to_pass"] == 2 and r.metrics["llm_calls_to_pass"] == 2
    assert res["llm_to_pass"]["calls"] == 2  # coincide con lo snapshot del budget
    assert not [e for e in _trace(r.run_dir) if e["kind"] == "calls_to_pass_mismatch"]
    # ordine nel trace: la seconda chiamata precede il primo verify passato del top
    ev = _trace(r.run_dir)
    first_pass = next(i for i, e in enumerate(ev) if e["kind"] == "verify" and e.get("passed"))
    assert sum(e["kind"] == "llm" for e in ev[:first_pass]) == 2


def test_full_calls_to_pass_counts_every_role_of_a_decomposition(tmp_path):
    replies = [json.dumps(PLAN), agents_code("E2M1Mul"), agents_code("AdderTree"),
               agents_code("E8M0Pair"), agents_code("FixedToFP32"),
               agents_code("MXFP4Dot32", "// bug"), agents_code("MXFP4Dot32")]
    o, llm, _ = orch(B05, agents_cfg(), replies, tmp_path)
    m = o.run().metrics
    assert m["passed"] and m["strategy"] == "decompose" and len(llm.requests) == 7
    # piano + 4 blocchi (ognuno passa: non chiude il conteggio) + integrate fallito + reflect
    assert m["llm_calls_to_pass"] == 7 == m["llm_to_pass"]["calls"]


def test_failed_full_run_has_no_calls_to_pass(tmp_path):
    o, _ = final_make(tmp_path, [final_code("// bug")] * 3, **{"budget.max_fix_iters": 2})
    r = o.run()
    assert not r.metrics["passed"] and _result(r.run_dir)["llm_calls_to_pass"] is None


def test_report_rebuilds_calls_to_pass_from_trace_of_old_runs(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "trace.json").write_text(json.dumps([
        {"kind": "llm"}, {"kind": "verify", "module": "AdderTree", "passed": True},
        {"kind": "llm"}, {"kind": "llm"}, {"kind": "verify", "module": "MXFP4Dot32", "passed": True},
        {"kind": "llm"}]))
    (rd / "meta.json").write_text(json.dumps({"task": "b05_mxfp4_dot32"}))
    m = {"passed": True, "llm_to_pass": {"calls": 3}, "llm": {"calls": 4}}
    assert calls_to_pass(rd, {"passed": True}, m, ROOT / "benchmarks") == (3, "trace")
    assert calls_to_pass(rd, {"passed": True, "llm_calls_to_pass": 9}, m) == (9, "result")
    assert calls_to_pass(rd, {"passed": False}, {"passed": False}) == (None, None)
    (rd / "trace.json").unlink()
    assert calls_to_pass(rd, {"passed": True}, m) == (3, "budget")


# ---------------------------------------------------- successo per k
def _rec(task, mode, seed, passed, calls=None, models="m", status="done"):
    return {"task": task, "mode": mode, "seed": seed, "models": models, "status": status,
            "passed": passed, "calls_to_pass": calls if passed else None, "variant": "main"}


def test_success_by_calls_curve_on_fake_records():
    recs = [
        # task t1, 3 seed; full passa con 3, 9, 23 chiamate; B con 5; matched con 5 e 40
        _rec("t1", "full", 0, True, 3), _rec("t1", "full", 1, True, 9), _rec("t1", "full", 2, True, 23),
        _rec("t1", "baseline-b", 0, True, 5), _rec("t1", "baseline-b", 1, False),
        _rec("t1", "baseline-b", 2, False),
        _rec("t1", "baseline-b-matched", 0, True, 5), _rec("t1", "baseline-b-matched", 1, True, 40),
        _rec("t1", "baseline-b-matched", 2, False),
        # seed 3: full c'è, B in errore -> la chiave non è appaiata e non conta per nessuno
        _rec("t1", "full", 3, True, 2), _rec("t1", "baseline-b", 3, False, status="error"),
        _rec("t1", "baseline-b-matched", 3, False),
    ]
    modes = ["baseline-b", "baseline-b-matched", "full"]
    d = success_by_calls(recs, modes)
    assert d["pairs"] == 3 and d["missing_modes"] == [] and d["kmax"] == 60
    assert d["by_k"][7] == {"baseline-b": 1, "baseline-b-matched": 1, "full": 1}
    assert d["by_k"][15] == {"baseline-b": 1, "baseline-b-matched": 1, "full": 2}
    assert d["by_k"][30] == {"baseline-b": 1, "baseline-b-matched": 1, "full": 3}
    assert d["by_k"][60] == {"baseline-b": 1, "baseline-b-matched": 2, "full": 3}
    assert d["passed_any"] == {"baseline-b": 1, "baseline-b-matched": 2, "full": 3}
    full = d["curve"]["full"]
    assert len(full) == 60 and full[1] == 0 and full[2] == 1 and full[8] == 2 and full[22] == 3
    assert all(a <= b for a, b in zip(full, full[1:]))  # cumulata
    # una modalità senza run non svuota l'appaiamento
    d2 = success_by_calls([r for r in recs if r["mode"] != "baseline-b-matched"],
                          ["baseline-a", *modes])
    assert d2["missing_modes"] == ["baseline-a", "baseline-b-matched"] and d2["pairs"] == 3


# ------------------------------------------------------------- campagne
def test_campaign_modes_with_matched():
    base = {"name": "t", "tasks": ["b01_e2m1_decode"], "models": {"m": {"all": "x:1b"}}}
    assert Campaign.model_validate(base).modes == list(DEFAULT_MODES)  # non entra da sola
    c = Campaign.model_validate({**base, "modes": ["baseline_b_matched", "full"]})
    assert c.modes == ["baseline-b-matched", "full"]
    for f in ("main.yaml", "b04_b05.yaml"):
        assert "baseline-b-matched" in load_campaign(ROOT / "configs" / "eval" / f).modes


@pytest.mark.parametrize("mode", ["baseline-b", "baseline-b-matched"])
def test_baseline_modes_use_one_model(mode):
    from mxagent.eval.campaign import ModelSet
    ov = ModelSet(all="x:1b").overrides(mode)
    assert ov == {"llm.roles.coder.model": "x:1b"}
