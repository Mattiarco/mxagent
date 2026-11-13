"""Sintesi e PPA (M5): parser su output reali, wrapper, regola di accettazione,
PPA Optimizer e stadio PPA dell'orchestratore con strumenti finti; flussi veri
(Yosys, nextpnr) solo dove gli strumenti sono installati."""
import json
import shutil
from pathlib import Path

import pytest

from mxagent.agents.orchestrator import Orchestrator
from mxagent.bench import load_task
from mxagent.config import load_config
from mxagent.llm.chat import ScriptedChat
from mxagent.synth import external, nextpnr, yosys
from mxagent.synth.ppa import PPAReport, accept, objective, synthesize
from mxagent.synth.wrapper import WRAP_TOP, wrapper
from mxagent.synth.yosys import SynthResult
from mxagent.verify.verifier import VerifyReport

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures"
B01 = load_task(ROOT / "benchmarks" / "b01_e2m1_decode")
B03 = load_task(ROOT / "benchmarks" / "b03_e8m0_to_fp32")
B05 = load_task(ROOT / "benchmarks" / "b05_mxfp4_dot32")


def cfg(**over):
    base = {"agent.best_of_n": 1, "agent.spec_parser": False, "agent.decompose": False,
            "agent.library": False, "sim.sbt_mode": "batch", "synth.backends": ["generic"]}
    return load_config(ROOT / "configs" / "default.yaml", {**base, **over})


# ---------------------------------------------------------------- parser
def test_parse_stat_generic_real_output():
    cells, tot = yosys.parse_stat((FX / "synth" / "stat_generic_b03.json").read_text())
    assert cells == {"$_NAND_": 8, "$_NOR_": 7, "$_NOT_": 1}
    assert tot == {"cells": 16, "transistors": 62}


def test_parse_ltp_real_output():
    assert yosys.parse_ltp((FX / "synth" / "ltp_b03.txt").read_text()) == 5
    assert yosys.parse_ltp("nothing") is None


def test_fpga_metrics():
    m = yosys.fpga_metrics("ecp5", {"LUT4": 2489, "CCU2C": 228, "PFUMX": 766, "L6MUX21": 414,
                                    "MULT18X18D": 32})
    assert m == {"luts": 2489, "carry": 228, "muxes": 1180, "dsp": 32, "bram": 0, "ffs": 0}
    m = yosys.fpga_metrics("xc7", {"LUT2": 206, "LUT6": 532, "CARRY4": 159, "MUXF7": 1, "FDRE": 3,
                                   "RAMB18E1": 2})
    assert m == {"luts": 738, "carry": 159, "muxes": 1, "dsp": 0, "bram": 2, "ffs": 3}


def test_fmax_statistics_and_slack():
    m = nextpnr.fmax_metrics([100.0, 125.0, 80.0], clock_ns=10.0)
    assert m["fmax_mhz"] == 100.0 and m["fmax_mean"] == pytest.approx(101.67)
    assert m["fmax_min"] == 80.0 and m["fmax_max"] == 125.0
    assert m["fmax_per_seed"] == [100.0, 125.0, 80.0] and m["seeds"] == 3
    assert m["slack_ns"] == pytest.approx(10.0 - 12.5)  # seed peggiore: 12.5 ns > 10 ns


def test_nextpnr_report_parser():
    rep = {"fmax": {"$glbnet$clk$TRELLIS_IO_IN": {"achieved": 20.406, "constraint": 100}},
           "utilization": {"TRELLIS_COMB": {"available": 83640, "used": 3092}}}
    assert nextpnr.parse_report(json.dumps(rep)) == pytest.approx(20.406)
    assert nextpnr.parse_report(json.dumps({"fmax": {}})) is None


def test_external_parsers_on_documented_formats():
    # estratti nel formato documentato (strumenti non installati: vedi external.py)
    sta = "  0.1234   0.1234 ^ y (out)\n           0.5678   data arrival time\n"
    assert external.parse_sta(sta) == pytest.approx(0.5678)
    util = "| Slice LUTs*             |  738 |     0 |     63400 |  1.16 |\n" \
           "| Slice Registers         |  560 |     0 |    126800 |  0.44 |\n"
    assert external.parse_vivado_util(util) == {"luts": 738, "ffs": 560}
    tim = "    WNS(ns)      TNS(ns)\n    -------      -------\n     -3.214      -50.000\n"
    assert external.parse_vivado_wns(tim) == pytest.approx(-3.214)


# --------------------------------------------------------------- wrapper
def test_wrapper_b05_widths_and_connections():
    w = wrapper(B05)
    assert f"module {WRAP_TOP}(input clk, input din, output dout);" in w
    # a_scale 8 + a 128 + b_scale 8 + b 128 = 272 bit di ingresso, y 32 bit
    assert "reg [271:0] sh;" in w and "{sh[270:0], din}" in w
    assert ".a_scale(sh[7:0])" in w and ".a(sh[135:8])" in w and ".b(sh[271:144])" in w
    assert "reg [31:0] q;" in w and "assign dout = ^q;" in w and ".clock" not in w


def test_wrapper_single_bit_input():
    w = wrapper(B03)  # x: 8 bit
    assert "reg [7:0] sh;" in w and ".x(sh[7:0])" in w


# -------------------------------------------------------------- accept
def test_accept_rule():
    c = cfg().synth  # adp, min_gain 1%
    assert accept((100, 10), (90, 10), c)[0]
    assert not accept((100, 10), (99.5, 10), c)[0]  # sotto min_gain
    assert accept((100, 10), (60, 15), c)[0]  # adp 1000 -> 900
    ca = cfg(**{"synth.objective": "area"}).synth
    ok, why = accept((100, 10), (60, 15), ca)
    assert not ok and "delay worsened" in why
    cd = cfg(**{"synth.objective": "delay"}).synth
    ok, why = accept((100, 10), (200, 8), cd)
    assert not ok and "area worsened" in why
    assert objective(100, 10, "adp") == 1000


def test_config_rejects_unknown_backend():
    with pytest.raises(Exception, match="sconosciuti"):
        cfg(**{"synth.backends": ["generic", "quartus"]})


# ---------------------------------------------- optimizer (strumenti finti)
class FakeVerifier:
    """Passa se il codice contiene '// ok'; '// bad' = mismatch funzionale."""

    def __init__(self, task):
        self.task = task

    def verify(self, sources, tag):
        code = sources[f"{self.task.top}.scala"]
        if "// ok" in code and "// bad" not in code:
            r = VerifyReport(tag, self.task.name, "pass", True)
            r.sim = {"pass_rate": 1.0}
            r.verilog = [f"/fake/{tag}/{self.task.top}.sv::{code.split('// ok')[1].split()[0]}"]
            return r
        r = VerifyReport(tag, self.task.name, "sim", False)
        r.feedback = "FUNCTIONAL MISMATCH: 3/16 test vectors fail"
        return r

    def close(self):
        pass


def fake_synth(verilog, task, c, work, backends=None):
    """Area e profondità codificate nel 'verilog' finto: '...::A,D'."""
    rep = PPAReport(task.top)
    a, d = (float(x) for x in verilog[0].split("::")[1].split(","))
    for b in backends or c.backends:
        if b == "generic":
            rep.results[b] = SynthResult("generic", True, {"transistors": a, "depth": d, "cells": a})
        else:
            rep.results[b] = SynthResult(b, False, error="non disponibile", skipped=True)
    return rep


def code(top, tag):
    return f"IDEA: try.\n```scala\nimport chisel3._\nclass {top} extends RawModule {{ {tag}\n  val m = \"{tag}\" }}\n```"


def test_orchestrator_ppa_accepts_only_verified_improvements(tmp_path):
    c = cfg(**{"synth.optimize_iters": 4, "synth.backends": ["generic", "ecp5"]})
    replies = [code("E2M1Decode", "// ok 100,10"),       # monolitico: passa (adp 1000)
               code("E2M1Decode", "// ok 90,9 // bad"),   # più piccolo ma sbagliato: rifiutato
               code("E2M1Decode", "// ok 120,10"),        # corretto ma peggiore: rifiutato
               code("E2M1Decode", "// ok 70,10"),         # corretto e migliore: accettato
               "no code here"]
    o = Orchestrator(B01, c, seed=1, runs_dir=tmp_path, llm=ScriptedChat(replies),
                     verifier_factory=lambda t, n: FakeVerifier(t), synth_fn=fake_synth)
    r = o.run()
    m = r.metrics
    assert m["passed"] and m["states"] == ["MONOLITHIC", "PPA", "FINAL", "DONE"]
    assert m["ppa"]["tried"] == 4 and m["ppa"]["accepted"] == 1
    assert m["ppa"]["gain"] == pytest.approx(0.3)
    assert m["ppa"]["initial"]["transistors"] == 100 and m["ppa"]["final"]["transistors"] == 70
    # metriche funzionali fotografate al pass (le chiamate PPA non contano)
    assert m["llm_to_pass"]["calls"] == 1 and m["llm"]["calls"] == 5
    ppa = json.loads((r.run_dir / "ppa.json").read_text())
    outcomes = [s["outcome"] for s in ppa["optimizer"]["steps"]]
    assert outcomes == ["functional_fail", "rejected", "accepted", "no_code"]
    assert ppa["initial"]["ecp5"]["skipped"] is True
    # il design finale passa su tutti i backend della config più vivado (skipped se assente)
    assert list(ppa["final"]) == ["generic", "ecp5", "vivado"] and ppa["final"]["generic"]["ok"]
    assert ppa["scope"] == {"loop": ["generic"], "final": ["generic", "ecp5", "vivado"]}
    scope = [e for e in json.loads((r.run_dir / "trace.json").read_text()) if e["kind"] == "ppa_scope"]
    assert scope and scope[0]["loop_backends"] == ["generic"]
    trace = json.loads((r.run_dir / "trace.json").read_text())
    stop = [e for e in trace if e["kind"] == "ppa_stop"]
    assert stop == [{**stop[0], "reason": "iters_exhausted", "iters_done": 4, "iters_max": 4,
                     "accepted": 1}]
    res = json.loads((r.run_dir / "result.json").read_text())
    assert res["status"] == "done" and res["best_candidate"] == "E2M1Decode_ppa02"
    assert res["ppa"]["attempts"] == {"total": 4, "accepted": 1, "rejected": 1, "functional_fail": 1,
                                      "synth_fail": 0, "no_code": 1, "duplicate": 0}
    assert res["llm_by_role"]["optimizer"]["calls"] == 4
    final = (r.run_dir / "final" / "E2M1Decode.scala").read_text()
    assert "70,10" in final
    assert "100,10" in (r.run_dir / "final_functional" / "E2M1Decode.scala").read_text()
    # il prompt PPA riporta la sintesi e i tentativi precedenti
    llm = [e for e in json.loads((r.run_dir / "trace.json").read_text()) if e["kind"] == "llm"]
    assert "transistors=100" in llm[1]["prompt"] and "functional_fail" in llm[2]["prompt"]


def test_orchestrator_ppa_disabled_keeps_report(tmp_path):
    c = cfg(**{"agent.ppa": False})
    o = Orchestrator(B01, c, seed=1, runs_dir=tmp_path,
                     llm=ScriptedChat([code("E2M1Decode", "// ok 100,10")]),
                     verifier_factory=lambda t, n: FakeVerifier(t), synth_fn=fake_synth)
    m = o.run().metrics
    assert m["states"] == ["MONOLITHIC", "PPA", "FINAL", "DONE"] and m["ppa"]["tried"] == 0
    assert m["ppa"]["final"]["transistors"] == 100


def test_ppa_tool_crash_does_not_fail_run(tmp_path):
    def boom(*a, **k):
        raise RuntimeError("yosys esploso")
    o = Orchestrator(B01, cfg(), seed=1, runs_dir=tmp_path,
                     llm=ScriptedChat([code("E2M1Decode", "// ok 100,10")]),
                     verifier_factory=lambda t, n: FakeVerifier(t), synth_fn=boom)
    m = o.run().metrics
    assert m["passed"] and m["ppa"]["initial"] is None


# ------------------------------------------------------ strumenti veri
needs_yosys = pytest.mark.skipif(not yosys.available(), reason="yosys non installato")


@needs_yosys
def test_yosys_generic_and_fpga_on_firtool_output(tmp_path):
    sv = [FX / "verilog" / "E2M1Decode.sv"]  # contiene un assignment pattern '{...}
    g = yosys.synth(sv, "E2M1Decode", "generic", tmp_path / "g")
    assert g.ok, g.error
    assert g.metrics["transistors"] > 0 and g.metrics["depth"] >= 1
    assert set(g.cells) <= {"$_NAND_", "$_NOR_", "$_NOT_"}
    e = yosys.synth(sv, "E2M1Decode", "ecp5", tmp_path / "e")
    assert e.ok and e.metrics["luts"] > 0


@needs_yosys
def test_yosys_reports_errors(tmp_path):
    bad = tmp_path / "bad.sv"
    bad.write_text("module X(input a, output y); assign y = ; endmodule\n")
    r = yosys.synth([bad], "X", "generic", tmp_path / "w")
    assert not r.ok and r.error


@pytest.mark.skipif(not nextpnr.available(), reason="nextpnr-ecp5 non installato")
def test_nextpnr_fmax_on_b03(tmp_path):
    c = cfg(**{"synth.pnr_seeds": [1]}).synth
    rep = synthesize([FX / "verilog" / "E8M0ToFP32.sv"], B03, c, tmp_path, backends=["nextpnr"])
    r = rep.results["nextpnr"]
    assert r.ok, r.error
    assert r.metrics["fmax_mhz"] > 0 and r.metrics["seeds"] == 1


def test_missing_tools_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    yosys.has_slang.cache_clear()
    wanted = ["generic", "ecp5", "xc7", "nextpnr", "opensta", "vivado"]
    rep = synthesize([FX / "verilog" / "E8M0ToFP32.sv"], B03, cfg().synth, tmp_path, backends=wanted)
    assert list(rep.results) == wanted  # nessun backend sparisce
    assert all(r.skipped and not r.ok and r.error for r in rep.results.values())
    assert "nextpnr-ecp5" in rep.results["nextpnr"].error and "PATH" in rep.results["nextpnr"].error
    summ = rep.summary()
    assert all(set(d) >= {"ok", "skipped", "error"} for d in summ.values())
    saved = json.loads((tmp_path / "ppa.json").read_text())
    assert list(saved["results"]) == wanted
    yosys.has_slang.cache_clear()


def test_backend_exception_becomes_error_entry(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("nextpnr segfault")
    monkeypatch.setattr(nextpnr, "pnr_ecp5", boom)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    yosys.has_slang.cache_clear()
    rep = synthesize([FX / "verilog" / "E8M0ToFP32.sv"], B03, cfg().synth, tmp_path,
                     backends=["generic", "nextpnr"])
    r = rep.results["nextpnr"]
    assert not r.ok and not r.skipped and "OSError: nextpnr segfault" in r.error
    assert rep.summary()["nextpnr"] == {"ok": False, "skipped": False, "error": r.error}
    yosys.has_slang.cache_clear()


def test_doctor_synth_lists_every_backend(monkeypatch):
    from mxagent.config import SYNTH_BACKENDS
    from mxagent.doctor import synth_backends
    monkeypatch.setattr(shutil, "which", lambda name: "/x/yosys" if name == "yosys" else None)
    checks = {c.name: c for c in synth_backends(["generic", "nextpnr"])}
    assert len(checks) == len(SYNTH_BACKENDS)
    assert checks["synth:generic [config]"].ok and not checks["synth:nextpnr [config]"].ok
    assert checks["synth:xc7"].ok and not checks["synth:vivado"].ok
    assert not any(c.required for c in checks.values())


def ppa_run(tmp_path, replies, **over):
    c = cfg(**{"synth.optimize_iters": 4, **over})
    o = Orchestrator(B01, c, seed=1, runs_dir=tmp_path, llm=ScriptedChat(replies),
                     verifier_factory=lambda t, n: FakeVerifier(t), synth_fn=fake_synth)
    r = o.run()
    trace = json.loads((r.run_dir / "trace.json").read_text())
    return r, [e for e in trace if e["kind"] == "ppa_stop"][0]


def test_ppa_plateau_stops_after_two_attempts_without_gain(tmp_path):
    # il caso b01: stesse metriche (rifiutato) due volte di fila -> stop, non 4 tentativi
    replies = [code("E2M1Decode", "// ok 100,10"), code("E2M1Decode", "// ok 100,10 // v2"),
               code("E2M1Decode", "// ok 100,10 // v3"), code("E2M1Decode", "// ok 70,10")]
    r, stop = ppa_run(tmp_path, replies)
    assert stop["reason"] == "plateau" and stop["iters_done"] == 2
    assert r.metrics["ppa"]["tried"] == 2 and r.metrics["llm"]["calls"] == 3
    res = json.loads((r.run_dir / "result.json").read_text())
    assert res["ppa"]["stop"] == "plateau"


def test_ppa_plateau_counter_resets_on_acceptance(tmp_path):
    replies = [code("E2M1Decode", "// ok 100,10"), code("E2M1Decode", "// ok 100,10 // v2"),
               code("E2M1Decode", "// ok 80,10"), code("E2M1Decode", "// ok 80,10 // v2"),
               code("E2M1Decode", "// ok 80,10 // v3")]
    r, stop = ppa_run(tmp_path, replies)
    assert stop["reason"] == "iters_exhausted" and stop["iters_done"] == 4
    ppa = json.loads((r.run_dir / "ppa.json").read_text())
    assert [s["outcome"] for s in ppa["optimizer"]["steps"]] == ["rejected", "accepted", "rejected",
                                                                 "rejected"]


def test_ppa_skipped_for_small_design_but_final_synthesis_done(tmp_path):
    r, stop = ppa_run(tmp_path, [code("E2M1Decode", "// ok 40,6")],
                      **{"synth.backends": ["generic", "ecp5"]})
    assert stop["reason"] == "too_small" and stop["cells"] == 40 and stop["min_cells"] == 50
    assert r.metrics["llm"]["calls"] == 1 and r.metrics["ppa"]["tried"] == 0  # nessuna chiamata optimizer
    ppa = json.loads((r.run_dir / "ppa.json").read_text())
    assert list(ppa["final"]) == ["generic", "ecp5", "vivado"] and ppa["final"]["generic"]["ok"]
    res = json.loads((r.run_dir / "result.json").read_text())
    assert res["ppa"]["stop"] == "too_small" and res["final_dir"] == "final"
    sel = json.loads((r.run_dir / "final" / "selection.json").read_text())
    assert "PPA optimizer skipped (too_small" in sel["reason"]
    # ppa_min_cells = 0: il loop parte comunque
    _, stop0 = ppa_run(tmp_path / "zero", [code("E2M1Decode", "// ok 40,6"), "no code"],
                       **{"synth.ppa_min_cells": 0, "synth.optimize_iters": 1})
    assert stop0["reason"] == "iters_exhausted"
