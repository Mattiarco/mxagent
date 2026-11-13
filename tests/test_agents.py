"""Sistema completo (M4) con LLM e verifier finti: logica della macchina a stati."""
import json
from pathlib import Path

from mxagent.agents.orchestrator import Orchestrator
from mxagent.agents.schemas import validate_plan
from mxagent.agents.solver import strip_classes
from mxagent.bench import load_task
from mxagent.config import load_config
from mxagent.llm.chat import ScriptedChat
from mxagent.rag.library import Library
from mxagent.verify.verifier import VerifyReport

ROOT = Path(__file__).resolve().parent.parent
B01 = load_task(ROOT / "benchmarks" / "b01_e2m1_decode")
B05 = load_task(ROOT / "benchmarks" / "b05_mxfp4_dot32")


def cfg(**over):
    base = {"agent.best_of_n": 1, "agent.spec_parser": False, "library.path": "unused",
            "sim.sbt_mode": "batch", "agent.ppa": False, "synth.backends": []}
    return load_config(ROOT / "configs" / "default.yaml", {**base, **over})


class FakeVerifier:
    """Passa se il file del top contiene '// ok'; altrimenti errore fisso o mismatch."""

    def __init__(self, task, log):
        self.task, self.log = task, log

    def verify(self, sources, tag):
        code = sources[f"{self.task.top}.scala"]
        self.log.append((self.task.top, tag, code))
        if "// ok" in code and "UInt(4)" not in code:
            r = VerifyReport(tag, self.task.name, "pass", True)
            r.sim = {"pass_rate": 1.0}
            return r
        if "syntax" in code:
            r = VerifyReport(tag, self.task.name, "compile", False)
            r.feedback = "COMPILATION FAILED:\n[scalac] X.scala:3:5: object Driver is not a member of package chisel3"
            r.feedback_raw = r.feedback
            r.error_keys = ["scalac:X.scala:object Driver is not a member"]
            return r
        r = VerifyReport(tag, self.task.name, "sim", False)
        r.sim = {"pass_rate": 0.5}
        r.feedback, r.feedback_raw = "FUNCTIONAL MISMATCH: structured", "TEST FAILED raw"
        r.error_keys = ["mismatch:y"]
        return r

    def close(self):
        pass


def code(top, mark="// ok"):
    return f"```scala\nimport chisel3._\nclass {top} extends RawModule {{ {mark}\n  val m = \"{mark}\" }}\n```"


def orch(task, c, replies, tmp, library=None):
    log = []
    llm = ScriptedChat(replies)
    o = Orchestrator(task, c, seed=1, runs_dir=tmp, llm=llm,
                     verifier_factory=lambda t, n: FakeVerifier(t, log), library=library)
    return o, llm, log


def test_monolithic_sanitizer_makes_chisel3_code_pass(tmp_path):
    bad = "```scala\nimport chisel3._\nclass E2M1Decode extends RawModule { val io = IO(new Bundle { val a = Input(UInt(4)) }) // ok\n}\n```"
    o, llm, log = orch(B01, cfg(**{"agent.decompose": False}), [bad], tmp_path)
    m = o.run().metrics
    assert m["passed"] and m["strategy"] == "monolithic" and m["sanitized_candidates"] == 1
    assert "UInt(4.W)" in log[0][2] and "FlatIO(" in log[0][2]
    # stessa risposta senza KB: il sanitizer non interviene e il candidato non passa
    o2, _, _ = orch(B01, cfg(**{"agent.decompose": False, "agent.kb": False,
                                "budget.max_fix_iters": 0}), [bad], tmp_path / "nokb")
    assert not o2.run().metrics["passed"]


PLAN = {"strategy": "decompose", "glue": "multiply, sum, convert",
        "blocks": [{"key": "e2m1_mul", "module": "E2M1Mul"},
                   {"key": "adder_tree", "module": "AdderTree", "params": {"n": 32, "w": 9}},
                   {"key": "e8m0_pair", "module": "E8M0Pair"},
                   {"key": "fixed_to_fp32", "module": "FixedToFP32", "params": {"w": 14, "frac": 2}}]}


def test_decompose_integrate_and_library_reuse(tmp_path):
    lib = Library(tmp_path / "lib")
    replies = [json.dumps(PLAN), code("E2M1Mul"), code("AdderTree"), code("E8M0Pair"),
               code("FixedToFP32"),
               "```scala\nimport chisel3._\nclass E2M1Mul extends RawModule {}\nclass MXFP4Dot32 extends RawModule { // ok\n}\n```"]
    o, llm, log = orch(B05, cfg(), replies, tmp_path / "r1", library=lib)
    m = o.run().metrics
    assert m["passed"] and m["strategy"] == "decompose"
    assert [b["module"] for b in m["blocks"]] == ["E2M1Mul", "AdderTree", "E8M0Pair", "FixedToFP32"]
    assert m["states"] == ["PLAN", "BLOCK", "BLOCK", "BLOCK", "BLOCK", "INTEGRATE", "FINAL", "DONE"]
    top_sources = log[-1][2]
    assert "class E2M1Mul" not in top_sources  # ridefinizione del submodulo rimossa
    integ = llm.requests[-1]["messages"][1]["content"]
    assert "VERIFIED SUBMODULES" in integ and "FixedToFP32" in integ and "y: Output(UInt(32.W))" in integ
    assert len(lib) == 5  # 4 blocchi + il top
    # candidates/<id>: blocchi col nome del modulo, top integrato con prefisso "int"
    cands = sorted(d.name for d in (o.rec.root / "candidates").iterdir())
    assert cands == ["AdderTree_00", "E2M1Mul_00", "E8M0Pair_00", "FixedToFP32_00", "MXFP4Dot32_int00"]

    # seconda run su un altro task (anti-leakage: origin diverso): i blocchi si riusano senza LLM
    other = B05.model_copy(update={"name": "b05_variant"})
    other.spec_text, other.root = B05.spec_text, B05.root
    o2, llm2, log2 = orch(other, cfg(), [json.dumps(PLAN), code("MXFP4Dot32")], tmp_path / "r2",
                          library=lib)
    m2 = o2.run().metrics
    assert m2["passed"] and m2["llm"]["calls"] == 2  # piano + top: nessun blocco rigenerato
    assert all(b["stop"] == "reused" for b in m2["blocks"])


def test_library_excludes_current_task(tmp_path):
    lib = Library(tmp_path / "lib")
    replies = [json.dumps(PLAN)] + [code(n) for n in ("E2M1Mul", "AdderTree", "E8M0Pair",
                                                       "FixedToFP32", "MXFP4Dot32")]
    orch(B05, cfg(), replies, tmp_path / "r1", library=lib)[0].run()
    # stesso task: i componenti nati da b05 non si possono riusare (anti-leakage)
    o, llm, _ = orch(B05, cfg(), replies, tmp_path / "r2", library=lib)
    m = o.run().metrics
    assert m["passed"] and not any(b["stop"] == "reused" for b in m["blocks"])


def test_escape_ladder_resample_escalate_stop(tmp_path):
    c = cfg(**{"agent.decompose": False, "budget.max_fix_iters": 12, "agent.escape_after_repeats": 1})
    replies = [code("E2M1Decode", f"syntax {i}") for i in range(12)]
    o, llm, _ = orch(B01, c, replies, tmp_path)
    m = o.run().metrics
    assert not m["passed"] and m["stop"] == "escape_stop" and m["escalated"]
    events = json.loads((o.rec.root / "trace.json").read_text())
    actions = [e["action"] for e in events if e["kind"] == "escape"]
    assert actions == ["resample", "escalate", "stop"]
    models = [r["model"] for r in llm.requests]
    assert c.llm.roles["escalate"].model in models  # dopo l'escalation si usa il modello grande
    # il fixer (modello piccolo) è usato per gli errori di compilazione prima dell'escalation
    assert c.llm.roles["fixer"].model in models
    # le trappole note della KB entrano nel prompt di riflessione
    refl = [r for r in llm.requests if r["tag"].endswith("reflect")]
    assert refl and "KNOWN CHISEL 6 PITFALLS" in refl[0]["messages"][1]["content"]


def test_fallback_to_monolithic_when_block_fails(tmp_path):
    c = cfg(**{"agent.max_block_iters": 0, "agent.escape": False})
    plan = {"strategy": "decompose", "blocks": [{"key": "e2m1_decode", "module": "Dec"}]}
    replies = [json.dumps(plan), code("Dec", "bug"), code("E2M1Decode")]
    o, _, _ = orch(B01, c, replies, tmp_path)
    m = o.run().metrics
    assert m["passed"] and m["states"] == ["PLAN", "BLOCK", "FALLBACK", "MONOLITHIC", "FINAL", "DONE"]


def test_structured_vs_raw_feedback_in_reflection(tmp_path):
    for structured, expect, avoid in ((True, "FUNCTIONAL MISMATCH: structured", "TEST FAILED raw"),
                                      (False, "TEST FAILED raw", "FUNCTIONAL MISMATCH")):
        c = cfg(**{"agent.decompose": False, "agent.structured_feedback": structured,
                   "budget.max_fix_iters": 1, "agent.escape": False})
        o, llm, _ = orch(B01, c, [code("E2M1Decode", "bug"), code("E2M1Decode")],
                         tmp_path / str(structured))
        assert o.run().metrics["passed"]
        p = llm.requests[1]["messages"][1]["content"]
        assert expect in p and avoid not in p and "DOMAIN KNOWLEDGE" in p


def test_plan_validation():
    p = validate_plan(PLAN, "MXFP4Dot32")
    assert p.strategy == "decompose" and len(p.blocks) == 4 and not p.problems
    bad = validate_plan({"strategy": "decompose", "blocks": [{"key": "warp_drive", "module": "X"}]}, "T")
    assert bad.strategy == "monolithic" and "blocco sconosciuto" in bad.problems[0]
    p = validate_plan({"strategy": "decompose", "blocks": [
        {"key": "adder_tree", "module": "T", "params": {"n": 16, "w": 9, "color": 3}},
        {"key": "adder_tree", "module": "T"}]}, "T")
    assert [b.module for b in p.blocks] == ["T_0", "T_1"] and p.blocks[0].params == {"n": 16, "w": 9}
    assert validate_plan(None, "T").strategy == "monolithic"


def test_strip_classes():
    src = "import chisel3._\nclass A extends RawModule {\n  val x = { 1 }\n}\nclass Top extends RawModule {}\n"
    out = strip_classes(src, {"A"})
    assert "class A" not in out and "class Top" in out
