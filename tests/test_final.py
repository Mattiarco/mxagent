"""Cartella final/ (design scelto, verifica rieseguita, PPA, motivo della selezione)
con LLM, verifier e sintesi finti che producono file Verilog veri."""
import hashlib
import json
from pathlib import Path

from mxagent.agents.orchestrator import Orchestrator
from mxagent.bench import load_task
from mxagent.config import load_config
from mxagent.llm.chat import ScriptedChat
from mxagent.synth.ppa import PPAReport
from mxagent.synth.yosys import SynthResult
from mxagent.verify.verifier import VerifyReport

ROOT = Path(__file__).resolve().parent.parent
B01 = load_task(ROOT / "benchmarks" / "b01_e2m1_decode")


def cfg(**over):
    base = {"agent.best_of_n": 1, "agent.spec_parser": False, "agent.decompose": False,
            "agent.library": False, "sim.sbt_mode": "batch", "agent.escape": False,
            "synth.backends": ["generic", "ecp5"], "synth.optimize_iters": 2}
    return load_config(ROOT / "configs" / "default.yaml", {**base, **over})


class FileVerifier:
    """Scrive <work>/<tag>/E2M1Decode.sv con un contenuto che dipende dal codice
    (come firtool: stesso sorgente -> stesso Verilog). '// ok A,D' passa."""

    def __init__(self, task, work: Path, log: list):
        self.task, self.work, self.log = task, work, log

    def verify(self, sources, tag):
        code = sources[f"{self.task.top}.scala"]
        self.log.append(tag)
        ok = "// ok" in code and "// bad" not in code
        r = VerifyReport(tag, self.task.name, "pass" if ok else "sim", ok)
        r.sim = {"pass_rate": 1.0 if ok else 0.5, "total": 16}
        d = self.work / tag
        d.mkdir(parents=True, exist_ok=True)
        sv = d / f"{self.task.top}.sv"
        body = code.split("// ok")[1].split()[0] if "// ok" in code else "x"
        sv.write_text(f"// generated\nmodule {self.task.top}(); // {body}\nendmodule\n")
        r.verilog = [str(sv)]
        return r

    def close(self):
        pass


def synth(verilog, task, c, work, backends=None):
    a, d = (float(x) for x in Path(verilog[0]).read_text().split("// ")[2].split()[0].split(","))
    rep = PPAReport(task.top)
    for b in backends or c.backends:
        rep.results[b] = (SynthResult(b, True, {"transistors": a, "depth": d}) if b == "generic"
                          else SynthResult(b, False, error=f"{b} not found in PATH", skipped=True))
    return rep


def code(tag):
    return f"IDEA: x.\n```scala\nimport chisel3._\nclass E2M1Decode extends RawModule {{ {tag}\n  val m = \"{tag}\" }}\n```"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make(tmp, replies, **over):
    log: list = []
    o = Orchestrator(B01, cfg(**over), seed=1, runs_dir=tmp / "runs", llm=ScriptedChat(replies),
                     verifier_factory=lambda t, n: FileVerifier(t, tmp / "verify" / n, log),
                     synth_fn=synth)
    return o, log


def test_final_dir_after_ppa_acceptance(tmp_path):
    o, log = make(tmp_path, [code("// bug"), code("// ok 100,10"),       # solve: 00 fallisce, 01 passa
                             code("// ok 120,10"), code("// ok 70,10")],  # ppa00 rifiutato, ppa01 accettato
                  **{"budget.max_fix_iters": 2})
    r = o.run()
    fin = r.run_dir / "final"
    assert sorted(p.name for p in fin.iterdir()) == ["E2M1Decode.scala", "E2M1Decode.sv", "ppa.json",
                                                    "report.json", "selection.json"]
    assert "70,10" in (fin / "E2M1Decode.scala").read_text()
    # la verifica è stata rieseguita sul design finale ed è l'ultima della run
    assert log[-1] == "E2M1Decode_final"
    rep = json.loads((fin / "report.json").read_text())
    assert rep["passed"] and rep["tag"] == "E2M1Decode_final"
    # il .sv in final/ coincide (hash) con quello dell'ultima verifica
    last = tmp_path / "verify" / "final" / "E2M1Decode_final" / "E2M1Decode.sv"
    assert sha(fin / "E2M1Decode.sv") == sha(last)
    sel = json.loads((fin / "selection.json").read_text())
    assert sel["selected"] == "E2M1Decode_ppa01" and sel["sv_matches_selected"]
    assert sel["sv_sha256"]["E2M1Decode.sv"] == sha(last)
    assert "PPA candidate accepted" in sel["reason"] and "16 vectors" in sel["reason"]
    by_id = {c["id"]: c for c in sel["candidates"]}
    assert by_id["E2M1Decode_00"]["outcome"] == "discarded" and "failed at sim" in by_id["E2M1Decode_00"]["reason"]
    assert by_id["E2M1Decode_01"]["outcome"] == "superseded"
    assert by_id["E2M1Decode_ppa00"]["outcome"] == "rejected" and "not better" in by_id["E2M1Decode_ppa00"]["reason"]
    assert by_id["E2M1Decode_ppa01"]["outcome"] == "selected"
    ppa = json.loads((fin / "ppa.json").read_text())
    assert list(ppa["backends"]) == ["generic", "ecp5", "vivado"]
    assert ppa["backends"]["ecp5"] == {"ok": False, "skipped": True, "error": "ecp5 not found in PATH"}
    res = json.loads((r.run_dir / "result.json").read_text())
    assert res["best_candidate"] == "E2M1Decode_ppa01" and res["final_dir"] == "final"
    assert o.states[-2:] == ["FINAL", "DONE"]


def test_final_dir_without_ppa_gain(tmp_path):
    o, log = make(tmp_path, [code("// ok 100,10"), code("// ok 100,10 // bad"), "no code"])
    r = o.run()
    sel = json.loads((r.run_dir / "final" / "selection.json").read_text())
    assert sel["selected"] == "E2M1Decode_00" and sel["sv_matches_selected"]
    assert sel["reason"].startswith("first candidate to pass") and "none accepted" in sel["reason"]
    outcomes = [(c["phase"], c["outcome"]) for c in sel["candidates"]]
    assert outcomes == [("solve", "selected"), ("ppa", "functional_fail"), ("ppa", "no_code")]


def test_failed_run_has_no_verified_final(tmp_path):
    o, log = make(tmp_path, [code(f"// bug {i}") for i in range(2)], **{"budget.max_fix_iters": 1})
    r = o.run()
    assert "E2M1Decode_final" not in log
    assert not (r.run_dir / "final" / "selection.json").exists()
    assert json.loads((r.run_dir / "result.json").read_text())["final_dir"] is None
