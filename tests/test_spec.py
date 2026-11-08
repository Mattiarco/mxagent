"""Spec strutturata: l'interfaccia viene dal benchmark, l'LLM compila solo i campi
descrittivi; le differenze diventano un evento ``spec_mismatch``."""
import json
from pathlib import Path

import pytest

from mxagent.agents.orchestrator import Orchestrator
from mxagent.agents.schemas import SPEC_SCHEMA, reconcile_spec
from mxagent.bench import load_task
from mxagent.config import load_config
from mxagent.llm.chat import ScriptedChat

ROOT = Path(__file__).resolve().parent.parent
BENCH = sorted((ROOT / "benchmarks").glob("b0[1-5]_*"))


def task_ports(t):
    return [(p.name, p.dir, p.type, p.total_width, p.elems, p.frac_bits) for p in t.ports]


def spec_ports(spec):
    return [(q["name"], q["direction"], q["type"], q["width"], q["elems"], q["frac_bits"])
            for q in spec["ports"]]


def llm_spec(t, **over):
    """Spec 'come la estrarrebbe l'LLM', corretta salvo ``over``."""
    d = {"operation": "other", "element_format": "e2m1", "block_size": 1, "output_format": "x",
         "latency_cycles": 0, "combinational": True,
         "ports": [{"name": p.name, "direction": p.dir, "type": p.type, "width": p.total_width,
                    "elems": p.elems, "meaning": f"llm says {p.name}"} for p in t.ports]}
    d.update(over)
    return d


@pytest.mark.parametrize("bench", BENCH, ids=lambda p: p.name)
def test_spec_has_exact_benchmark_ports(bench, tmp_path):
    t = load_task(bench)
    # LLM che sbaglia larghezze, perde una porta e ne inventa una
    bad = llm_spec(t, latency_cycles=1, ports=[{"name": t.ports[0].name, "direction": "out",
                                                "width": 999},
                                               {"name": "clk_en", "direction": "in", "width": 1}])
    c = load_config(ROOT / "configs" / "default.yaml",
                    {"agent.decompose": False, "agent.library": False, "sim.sbt_mode": "batch",
                     "budget.max_fix_iters": 0, "agent.best_of_n": 1, "synth.backends": []})
    o = Orchestrator(t, c, seed=0, runs_dir=tmp_path, llm=ScriptedChat([json.dumps(bad)]),
                     verifier_factory=lambda *a: None)
    spec = o.spec()
    saved = json.loads((o.rec.root / "spec.json").read_text())
    assert spec_ports(saved) == task_ports(t) == spec_ports(spec)
    assert saved["interface_source"] == "benchmark task.yaml"
    assert saved["latency_cycles"] == 0 and saved["combinational"] is True
    ev = [e for e in o.rec.events if e["kind"] == "spec_mismatch"]
    assert len(ev) == 1 and ev[0]["used"] == "benchmark"
    m = " | ".join(ev[0]["mismatches"])
    assert f"port {t.ports[0].name}: direction" in m and "width 999" in m
    assert "clk_en: not in the benchmark interface" in m and "latency_cycles 1 != 0" in m
    assert f"port {t.ports[-1].name}: missing" in m


@pytest.mark.parametrize("bench", BENCH, ids=lambda p: p.name)
def test_correct_extraction_has_no_mismatch(bench):
    t = load_task(bench)
    spec, mism = reconcile_spec(llm_spec(t), t)
    assert mism == [] and spec_ports(spec) == task_ports(t)
    # i campi descrittivi dell'LLM restano; il significato del benchmark ha la precedenza
    assert spec["operation"] == "other"
    for p, q in zip(t.ports, spec["ports"]):
        assert q["meaning"] == (p.doc or f"llm says {p.name}")


def test_b01_output_scale_and_signedness():
    t = load_task(ROOT / "benchmarks" / "b01_e2m1_decode")
    spec, _ = reconcile_spec({}, t)
    y = spec["ports"][1]
    assert y == {**y, "name": "y", "type": "sint", "width": 5, "frac_bits": 1, "signed": True}
    assert spec["output_scale"] == "y = value * 2^1 (i.e. value * 2)"
    b02 = load_task(ROOT / "benchmarks" / "b02_e2m1_mul")
    assert reconcile_spec({}, b02)[0]["output_scale"] == "p = value * 2^2 (i.e. value * 4)"


def test_vector_width_per_element_is_tolerated():
    t = load_task(ROOT / "benchmarks" / "b05_mxfp4_dot32")
    d = llm_spec(t)
    for q in d["ports"]:
        q.pop("elems")
        if q["name"] in ("a", "b"):
            q["width"] = 4  # larghezza di un elemento invece del totale
    assert reconcile_spec(d, t)[1] == []


def test_schema_requires_ports():
    assert "ports" in SPEC_SCHEMA["required"]
    assert set(SPEC_SCHEMA["properties"]) >= {"ports", "output_scale", "combinational"}
