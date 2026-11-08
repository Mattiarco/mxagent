"""Pulizia di fine run e seed di cocotb."""
import json
from pathlib import Path

from mxagent.agents.orchestrator import Orchestrator
from mxagent.llm.chat import ScriptedChat
from mxagent.run.cleanup import clean_run
from mxagent.verify.sim import seed_env
from tests.test_run_result import B01, Verifier, cfg, code

KEEP = ["config.yaml", "trace.json", "final/E2M1Decode.sv", "final/E2M1Decode.scala",
        "sbt/build.sbt", "sbt/src/main/scala/cand_E2M1Decode_final/E2M1Decode.scala",
        "verify/mono/E2M1Decode_00/report.json", "verify/mono/E2M1Decode_00/verilog/E2M1Decode.sv",
        "verify/mono/E2M1Decode_00/sim/vectors_1.json", "verify/mono/E2M1Decode_00/sim/results_1.json",
        "verify/mono/E2M1Decode_00/sim/results.xml", "verify/mono/E2M1Decode_00/sim.log",
        "ppa/initial/generic/yosys.log"]
DROP = ["sbt/target/scala-2.13/classes/A.class", "sbt/project/target/x.jar",
        "sbt/project/project/y", "sbt/.bsp/sbt.json",
        "verify/mono/E2M1Decode_00/sim/VE2M1Decode", "verify/mono/E2M1Decode_00/sim/VE2M1Decode.o",
        "verify/mono/E2M1Decode_00/sim/libverilated.a", "verify/mono/E2M1Decode_00/sim/Vtop.mk",
        "verify/mono/E2M1Decode_00/sim/obj_dir/x.o", "wrap/sim_build/Vwrap"]


def make_tree(root: Path) -> None:
    for rel in KEEP + DROP:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * (10_000 if rel in DROP else 10))


def test_clean_run_keeps_data_and_removes_builds(tmp_path):
    make_tree(tmp_path)
    out = clean_run(tmp_path)
    assert all((tmp_path / k).is_file() for k in KEEP)
    assert not any((tmp_path / d).exists() for d in DROP)
    assert not (tmp_path / "sbt" / "target").exists()
    assert out["freed_bytes"] == 10_000 * len(DROP) and out["kept_bytes"] == 10 * len(KEEP)


def test_orchestrator_cleans_unless_configured(tmp_path):
    for keep in (False, True):
        o = Orchestrator(B01, cfg(**{"runs.keep_build_artifacts": keep}), seed=1,
                         runs_dir=tmp_path / str(keep), llm=ScriptedChat([code("// ok")]),
                         verifier_factory=lambda t, n: Verifier(t))
        make_tree(o.rec.root)
        o.run()
        trace = json.loads((o.rec.root / "trace.json").read_text())
        assert (o.rec.root / "sbt" / "target").exists() is keep
        kinds = [e["kind"] for e in trace]
        assert ("cleanup" in kinds) is not keep and kinds[-1] == "end"


def test_cocotb_seed_env():
    assert seed_env(7) == ["COCOTB_RANDOM_SEED=7", "RANDOM_SEED=7"]
