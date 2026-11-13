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


def test_clean_run_finds_nested_sbt_workspaces(tmp_path):
    """runs/verify/<task>/ e eval_wrap/ hanno il loro workspace sbt: si pulisce anche quello."""
    keep = ["sbt/build.sbt", "sbt/src/main/scala/cand_X_00/X.scala", "manual/X.sv",
            "eval_wrap/sbt/build.sbt", "eval_wrap/sim/results.json", "eval_wrap/wrap_report.json"]
    drop = ["sbt/target/a.class", "sbt/project/target/b.jar", "eval_wrap/sbt/target/c.jar",
            "eval_wrap/sbt/.bsp/sbt.json", "eval_wrap/sim/Vtop", "manual/sim/Vtop.o"]
    for rel in keep + drop:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    # una copia di build.sbt dentro target non è un workspace (e target sparisce comunque)
    (tmp_path / "sbt/target/streams/build.sbt").parent.mkdir(parents=True)
    (tmp_path / "sbt/target/streams/build.sbt").write_text("x")
    clean_run(tmp_path)
    assert all((tmp_path / k).is_file() for k in keep)
    assert not any((tmp_path / d).exists() for d in drop)


def test_campaign_runner_cleans_job_dirs(tmp_path):
    from mxagent.eval.campaign import Campaign, jobs
    from mxagent.eval.runner import Runner
    from mxagent.util.shell import Result

    c = Campaign(name="t", tasks=["b01_e2m1_decode"], modes=["baseline-a"], seeds=1,
                 models={"m": {"all": "x:1b"}}, ppa_backends=[])

    def fake(cmd, cwd, timeout):
        rd = Path(cmd[cmd.index("--runs-dir") + 1]) / cmd[cmd.index("--run-id") + 1]
        (rd / "sbt" / "target").mkdir(parents=True)
        (rd / "sbt" / "target" / "big.jar").write_bytes(b"x" * 1000)
        (rd / "metrics.json").write_text(json.dumps({"passed": False}))
        return Result(cmd, 1, "", "", 0.1)

    r = Runner(c, tmp_path / "ev", repo=Path(__file__).resolve().parent.parent, executor=fake)
    r.run(log=lambda s: None)
    rd = tmp_path / "ev" / "runs" / jobs(c, tmp_path / "ev")[0].id
    assert (rd / "metrics.json").is_file() and not (rd / "sbt" / "target").exists()
