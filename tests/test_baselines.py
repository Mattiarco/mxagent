"""Loop delle baseline con LLM e verifier finti (nessuna toolchain, nessun modello)."""
import json
from pathlib import Path

from mxagent.agents.extract import extract_scala
from mxagent.agents.prompts import render
from mxagent.bench import load_task
from mxagent.config import load_config
from mxagent.llm.chat import Budget, ScriptedChat
from mxagent.run.baselines import _fit_history, run_loop
from mxagent.verify.verifier import VerifyReport

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config(ROOT / "configs" / "default.yaml", {"budget.max_fix_iters": 3})
TASK = load_task(ROOT / "benchmarks" / "b01_e2m1_decode")
ROLE = CFG.llm.roles["coder"]

GOOD = "Here it is:\n```scala\nimport chisel3._\nclass E2M1Decode extends RawModule { /* ok */ }\n```"
BAD = "```scala\nimport chisel3._\nclass E2M1Decode extends RawModule { /* bug */ }\n```"


def fake_verify(sources, tag):
    code = next(iter(sources.values()))
    ok = "/* ok */" in code
    r = VerifyReport(tag, TASK.name, "pass" if ok else "sim", ok)
    r.sim = {"pass_rate": 1.0 if ok else 0.875}
    r.feedback = "STRUCTURED: a=subnormal"
    r.feedback_raw = "TEST FAILED: 2 of 16 test vectors mismatched."
    r.error_keys = [] if ok else ["mismatch:y"]
    return r


def test_baseline_b_fixes_with_raw_feedback(tmp_path):
    llm = ScriptedChat([BAD, BAD, GOOD])
    res = run_loop(TASK, CFG, "baseline-b", seed=5, role=ROLE, runs_dir=tmp_path, llm=llm,
                   verify_fn=fake_verify)
    m = res.metrics
    assert m["passed"] and m["pass_iter"] == 3 and m["iterations"] == 3 and m["repeated_errors"] == 1
    # il feedback dato all'LLM è quello grezzo, mai quello strutturato
    last_user = llm.requests[-1]["messages"][-1]["content"]
    assert "TEST FAILED: 2 of 16" in last_user and "STRUCTURED" not in last_user
    # storia multi-turn completa e seed diverso per iterazione
    assert len(llm.requests[-1]["messages"]) == 6
    assert [r["seed"] for r in llm.requests] == [5, 6, 7]
    # prompt ingenuo: niente scheda dei formati
    first = llm.requests[0]["messages"][1]["content"]
    assert "E2M1Decode" in first and "FlatIO" in first and "Numeric semantics" not in first
    d = res.run_dir
    for f in ("config.yaml", "meta.json", "trace.json", "metrics.json", "final/E2M1Decode.scala",
              "candidates/00/E2M1Decode.scala"):
        assert (d / f).is_file(), f
    meta = json.loads((d / "meta.json").read_text())
    assert meta["mode"] == "baseline-b" and len(meta["prompts"]) == 3
    assert m["llm"]["calls"] == 3


def test_baseline_a_single_shot(tmp_path):
    llm = ScriptedChat([BAD, GOOD])
    res = run_loop(TASK, CFG, "baseline-a", seed=0, role=ROLE, runs_dir=tmp_path, llm=llm,
                   verify_fn=fake_verify)
    assert not res.metrics["passed"] and res.metrics["iterations"] == 1 and len(llm.requests) == 1
    assert res.metrics["best_pass_rate"] == 0.875


def test_no_code_and_budget(tmp_path):
    llm = ScriptedChat(["I cannot do that.", BAD, BAD, BAD], budget=Budget(max_calls=2))
    res = run_loop(TASK, CFG, "baseline-b", seed=0, role=ROLE, runs_dir=tmp_path, llm=llm,
                   verify_fn=fake_verify)
    m = res.metrics
    assert not m["passed"] and m["stop"].startswith("budget") and m["no_code"] == 1
    assert "No ```scala code block" in llm.requests[1]["messages"][-1]["content"]


def test_history_truncation():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for i in range(10):
        msgs += [{"role": "assistant", "content": "x" * 3000}, {"role": "user", "content": "fb"}]
    dropped = _fit_history(msgs, num_ctx=4096, reserve=1024)
    assert dropped > 0 and msgs[1]["content"] == "task" and len(msgs) % 2 == 0


def test_extract_scala():
    assert "class E2M1Decode" in extract_scala(GOOD, "E2M1Decode")
    two = "```scala\nclass Helper\n```\ntext\n```scala\nclass Top extends Module\n```"
    assert extract_scala(two, "Top").startswith("class Top")
    assert extract_scala("no code", "Top") is None
    unterminated = "```scala\nimport chisel3._\nclass Top extends Module {"
    assert extract_scala(unterminated, "Top").endswith("{")


def test_prompt_render_keeps_scala_braces():
    out = render("baseline/fix", feedback="val x = Wire(UInt(4.W)) {a}")
    assert "{a}" in out and "{feedback}" not in out
