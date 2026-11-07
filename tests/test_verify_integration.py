"""Verifier end-to-end con la toolchain vera (sbt, firtool, Verilator, cocotb).

Saltato dove la toolchain non c'è (es. Windows). Su Linux/WSL:
    MXAGENT_WORK=~/mxwork pytest -q tests/test_verify_integration.py
"""
import os
from pathlib import Path

import pytest

from mxagent.bench import load_task, reference_sources
from mxagent.util.shell import which
from mxagent.verify.sim import runner_available
from mxagent.verify.verifier import Verifier

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
MUT = ROOT / "tests" / "fixtures" / "mutants"

reason = runner_available() or (None if which("sbt") else "sbt non trovato")
pytestmark = pytest.mark.skipif(reason is not None, reason=reason or "")


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    base = os.environ.get("MXAGENT_WORK")
    return Path(base) / "pytest" if base else tmp_path_factory.mktemp("verify")


def _verify(work, task_name, files, tag):
    t = load_task(BENCH / task_name)
    src = {p.name: p.read_text() for p in files}
    return Verifier(t, work / task_name).verify(src, tag)


@pytest.mark.parametrize("name,n", [("b01_e2m1_decode", 16), ("b02_e2m1_mul", 256),
                                    ("b03_e8m0_to_fp32", 256)])
def test_references_pass(work, name, n):
    rep = _verify(work, name, reference_sources(load_task(BENCH / name)), "ref")
    assert rep.passed and rep.stage == "pass", rep.feedback
    assert rep.sim["total"] == n and rep.sim["failed"] == 0


@pytest.mark.parametrize("task,mutant,stage,check", [
    ("b01_e2m1_decode", "b01_subnormal", "sim", "a=subnormal"),
    ("b02_e2m1_mul", "b02_sign_or", "sim", "a=neg AND b=neg"),
    ("b03_e8m0_to_fp32", "b03_no_subnormal", "sim", "x=min"),
    ("b01_e2m1_decode", "b01_chisel3_api", "compile", "[scalac]"),
    ("b01_e2m1_decode", "b01_type_mismatch", "compile", "[elaboration]"),
    ("b01_e2m1_decode", "b01_comb_loop", "compile", "[firtool]"),
])
def test_mutants_are_caught(work, task, mutant, stage, check):
    rep = _verify(work, task, [MUT / f"{mutant}.scala"], mutant)
    assert not rep.passed and rep.stage == stage
    assert check in rep.feedback, rep.feedback
    assert rep.error_keys
