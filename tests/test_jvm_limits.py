"""Thread delle JVM di sbt: -XX:ActiveProcessorCount (sim.jvm_cpus), shutdown del server
solo se esiste, ruolo delle JVM nella diagnosi, stima del picco per la campagna."""
from pathlib import Path

from mxagent.config import load_config
from mxagent.util import procs
from mxagent.verify import chisel

ROOT = Path(__file__).resolve().parent.parent


def test_sbt_env_limits_jvm_cpus_and_keeps_stack(monkeypatch):
    monkeypatch.delenv("SBT_OPTS", raising=False)
    env = chisel.sbt_env(4)
    assert env["SBT_OPTS"] == "-Xss8m -XX:ActiveProcessorCount=4" and env["NO_COLOR"] == "1"
    assert chisel.sbt_env(None)["SBT_OPTS"] == "-Xss8m"  # nessun limite: come prima
    monkeypatch.setenv("SBT_OPTS", "-Xss16m -XX:ActiveProcessorCount=2")
    assert chisel.sbt_env(4)["SBT_OPTS"] == "-Xss16m -XX:ActiveProcessorCount=2"  # vince l'utente


def test_default_config_limits_jvm_cpus():
    assert load_config(ROOT / "configs" / "default.yaml").sim.jvm_cpus == 4


def test_elaborate_passes_the_limit_to_sbt(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, cwd=None, timeout=None, env=None, keep_group=False, **k):
        seen.update(env=env, keep_group=keep_group)
        from mxagent.util.shell import Result
        return Result(cmd, 1, "", "[error] boom", 0.1)
    monkeypatch.delenv("SBT_OPTS", raising=False)
    monkeypatch.setattr(chisel, "run", fake_run)
    chisel.elaborate(tmp_path, "p", "Top", tmp_path / "out", mode="client", jvm_cpus=3)
    assert "-XX:ActiveProcessorCount=3" in seen["env"]["SBT_OPTS"] and seen["keep_group"]


def test_shutdown_only_if_a_server_exists(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(chisel, "run", lambda cmd, **k: calls.append((cmd, k)))
    assert chisel.shutdown_server(tmp_path) is None and not calls  # niente server: niente sbt
    act = chisel.active_file(tmp_path)
    act.parent.mkdir(parents=True)
    act.write_text("{}")
    chisel.shutdown_server(tmp_path, alive=False)  # server morto: active.json stantio tolto
    assert not calls and not act.exists()
    act.write_text("{}")
    chisel.shutdown_server(tmp_path, 4)
    assert calls[0][0][-1] == "shutdown"
    assert "-XX:ActiveProcessorCount=4" in calls[0][1]["env"]["SBT_OPTS"]


def test_jvm_roles_from_cmdline():
    def P(cmd):
        return procs.Proc(1, 1, 0, 50, "java", cmd)
    srv = P("java -Xss8m -XX:ActiveProcessorCount=4 -jar /usr/share/sbt/bin/sbt-launch.jar --detach-stdio")
    assert srv.role == "server" and srv.active_cpus == 4
    assert P("java -jar sbt-launch.jar --client runMain x").role == "client"
    assert P("java -Dsbt.supershell=false -jar sbt-launch.jar -batch runMain x").role == "batch"
    assert P("java -cp x sbt.ForkMain 4242").role == "forkmain"
    assert P("java -jar other.jar").role == "java" and P("java -jar other.jar").active_cpus is None


def test_peak_estimate_uses_jvm_cpus():
    full = procs.job_peak_threads(20)
    lim = procs.job_peak_threads(20, jvm_cpus=4)
    assert full == (2 * 20 + 60) + (20 + 30) + 15 + 20 + 10
    assert lim == (2 * 4 + 60) + (4 + 30) + 15 + 20 + 10 and lim < full
    assert procs.job_peak_threads(2, jvm_cpus=4) == procs.job_peak_threads(2)  # mai sopra ncpu
