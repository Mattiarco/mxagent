"""Errori di sistema (EAGAIN, native thread, memoria) distinti dagli errori del codice,
retry della verifica, esito "error / infra" uguale in tutte le modalità, pulizia dei
processi figli. Nessuna toolchain: verifier e LLM finti, processi Python come figli."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mxagent.bench import load_task
from mxagent.llm.chat import ScriptedChat
from mxagent.run.baselines import run_loop
from mxagent.util import procs, shell
from mxagent.verify import chisel
from mxagent.verify.chisel import ElabResult, SourceError
from mxagent.verify.infra import INFRA_PATTERNS, InfraError, classify, classify_exc, infra_outcome
from mxagent.verify.verifier import Verifier, VerifyReport, _InfraHit
from tests.test_baselines import BAD, CFG, GOOD, ROLE, TASK

ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------ classificazione
SYSTEM = [  # dai log delle run di prova su b05 e dai casi tipici
    "elaboration:None:OutOfMemoryError: unable to create native thread: possibly out of memory "
    "or process/resource limits reached",
    'IOException: Cannot run program "/usr/bin/infocmp": error=11, Resource temporarily unavailable',
    "BlockingIOError: [Errno 11] Resource temporarily unavailable",
    "java.lang.OutOfMemoryError: Java heap space",
    "OSError: [Errno 12] Cannot allocate memory",
    "bash: fork: retry: Resource temporarily unavailable",
    "There is insufficient memory for the Java Runtime Environment to continue.",
    "OSError: [Errno 24] Too many open files",
    "OSError: [Errno 28] No space left on device",
    "RuntimeError: can't start new thread",
    "pthread_create failed: EAGAIN",
]
CODE = [  # errori del codice generato: NON sono di sistema
    "[scalac] FixedToFp32.scala:5:10: ';' expected but '(' found.",
    "[scalac] X.scala:3:5: object Driver is not a member of package chisel3",
    "[scalac] X.scala:9:3: type mismatch; found: chisel3.UInt required: chisel3.SInt",
    "[elaboration] Top.scala:33: Invalid bit range [hi=0, lo=8]",
    "[firtool] Top.scala:12:7: error: sink \"io.y\" not fully initialized",
    "ChiselException: Connection between sink (UInt<4>) and source (SInt<5>) failed",
    "%Error: Top.sv:10:3: Cannot find file containing module: 'Foo'",
    "TEST FAILED: 2 of 16 test vectors mismatched.",
    "sim:timeout della simulazione (900s)",
]


@pytest.mark.parametrize("text", SYSTEM)
def test_system_errors_are_infra(text):
    assert classify(text) is not None


@pytest.mark.parametrize("text", CODE)
def test_code_errors_are_not_infra(text):
    assert classify(text) is None


def test_patterns_live_in_one_place_and_cover_the_kinds():
    kinds = {k for k, _ in INFRA_PATTERNS}
    assert {"native_thread", "eagain", "enomem", "jvm_oom", "emfile", "enospc"} <= kinds
    assert classify("ok\nunable to create native thread\n").startswith("native_thread:")
    assert classify_exc(BlockingIOError(11, "Resource temporarily unavailable")).startswith("eagain")
    assert classify_exc(FileNotFoundError(2, "x")).startswith("oserror")  # ogni OSError
    assert classify_exc(ValueError("bad")) is None
    out = infra_outcome(InfraError("eagain: boom", "compile", attempts=4))
    assert out["status"] == "error" and out["reason"].startswith("infra: eagain")
    assert infra_outcome(ValueError("x")) is None


# ------------------------------------------------------------ retry nel verifier
class Flaky(Verifier):
    """_verify_once che fallisce per errore di sistema ``fails`` volte, poi passa."""

    def __init__(self, tmp, fails, **kw):
        super().__init__(TASK, tmp, infra_backoff=(0, 0, 0), **kw)
        self.fails, self.calls, self.cleanups = fails, 0, 0

    def _verify_once(self, sources, tag):
        self.calls += 1
        if self.calls <= self.fails:
            raise _InfraHit("native_thread: unable to create native thread", "compile")
        r = VerifyReport(tag, TASK.name, "pass", True)
        r.sim = {"pass_rate": 1.0}
        return r

    def kill_leftovers(self):
        self.cleanups += 1
        return []


def test_transient_infra_error_is_retried(tmp_path):
    v = Flaky(tmp_path, fails=2)
    rep = v.verify({"E2M1Decode.scala": "x"}, "c00")
    assert rep.passed and v.calls == 3 and v.cleanups == 2
    assert [r["attempt"] for r in rep.infra_retries] == [1, 2]
    saved = json.loads((tmp_path / "c00" / "report.json").read_text())
    assert saved["stage"] == "pass" and len(saved["infra_retries"]) == 2


def test_persistent_infra_error_raises_with_infra_report(tmp_path):
    v = Flaky(tmp_path, fails=99)
    with pytest.raises(InfraError) as ei:
        v.verify({"E2M1Decode.scala": "x"}, "c00")
    assert v.calls == 4 and ei.value.attempts == 4  # 1 + 3 tentativi (backoff 10/30/60 s)
    rep = ei.value.report
    assert rep.stage == "infra" and not rep.passed and rep.feedback == "" and rep.score == (-1, 0.0)
    saved = json.loads((tmp_path / "c00" / "report.json").read_text())
    assert saved["stage"] == "infra" and saved["infra"].startswith("native_thread")


def test_default_backoff_is_10_30_60():
    assert Verifier(TASK, Path("unused")).infra_backoff == (10, 30, 60)


def test_elaboration_log_with_system_error_is_infra_not_compile(tmp_path, monkeypatch):
    """Il caso della run s2: l'elaborazione fallisce per 'unable to create native thread'."""
    log = ("[error] java.lang.OutOfMemoryError: unable to create native thread: possibly out of "
           "memory or process/resource limits reached")
    calls = []

    def fake_elab(work, pkg, top, out_dir, mode="batch", timeout=900, **kw):
        calls.append(1)
        return ElabResult(False, [SourceError("elaboration", "OutOfMemoryError: unable to create "
                                              "native thread")], [], 1.0, log)

    monkeypatch.setattr(chisel, "elaborate", fake_elab)
    monkeypatch.setattr(chisel, "prepare_workspace", lambda w: w)
    monkeypatch.setattr(chisel, "write_sources", lambda *a, **k: None)
    v = Verifier(TASK, tmp_path, infra_backoff=(0, 0))
    monkeypatch.setattr(v, "kill_leftovers", lambda: [])
    with pytest.raises(InfraError) as ei:
        v.verify({"E2M1Decode.scala": "class E2M1Decode"}, "c00")
    assert len(calls) == 3 and ei.value.where == "compile"
    # un errore di compilazione vero resta "compile" e non viene ripetuto
    calls.clear()
    monkeypatch.setattr(chisel, "elaborate", lambda *a, **k: (calls.append(1), ElabResult(
        False, [SourceError("scalac", "';' expected but '(' found.", "X.scala", 3, 5)], [], 1.0,
        "[error] X.scala:3:5: ';' expected but '(' found."))[1])
    rep = v.verify({"E2M1Decode.scala": "class E2M1Decode"}, "c01")
    assert rep.stage == "compile" and len(calls) == 1 and not rep.infra_retries


def test_spawn_failure_is_a_result_classified_as_infra(monkeypatch):
    def boom(*a, **k):
        raise BlockingIOError(11, "Resource temporarily unavailable")
    monkeypatch.setattr(subprocess, "Popen", boom)
    r = shell.run([sys.executable, "-c", "pass"])
    assert r.spawn_error and r.returncode == 126 and classify(r.log).startswith("eagain")


# ------------------------------------------------- stesso esito in ogni modalità
class InfraVerifier:
    def __init__(self, exc):
        self.exc, self.calls = exc, 0

    def verify(self, sources, tag):
        self.calls += 1
        raise self.exc

    def close(self):
        pass


EXCS = [InfraError("native_thread: unable to create native thread", "compile", attempts=4),
        BlockingIOError(11, "Resource temporarily unavailable")]


@pytest.mark.parametrize("exc", EXCS, ids=["infra_error", "blocking_io"])
@pytest.mark.parametrize("mode", ["baseline-a", "baseline-b", "baseline-b-matched"])
def test_baselines_end_with_error_infra_and_never_feed_it_back(tmp_path, mode, exc):
    llm = ScriptedChat([BAD, GOOD, GOOD])
    v = InfraVerifier(exc)
    with pytest.raises(type(exc)):
        run_loop(TASK, CFG, mode, seed=0, role=ROLE, runs_dir=tmp_path, llm=llm, verify_fn=v.verify,
                 run_id="r")
    res = json.loads((tmp_path / "r" / "result.json").read_text())
    assert res["status"] == "error" and res["reason"].startswith("infra: ")
    assert len(llm.requests) == 1 and v.calls == 1  # nessuna correzione: il modello non lo vede
    assert not (tmp_path / "r" / "metrics.json").exists()  # per la campagna: job "error"


@pytest.mark.parametrize("exc", EXCS, ids=["infra_error", "blocking_io"])
def test_full_ends_with_error_infra_like_the_baselines(tmp_path, exc):
    from tests.test_agents import B01, cfg, code
    from mxagent.agents.orchestrator import Orchestrator
    llm = ScriptedChat([code("E2M1Decode", "// bug"), code("E2M1Decode")])
    v = InfraVerifier(exc)
    o = Orchestrator(B01, cfg(**{"agent.decompose": False}), seed=1, runs_dir=tmp_path, llm=llm,
                     verifier_factory=lambda t, n: v)
    with pytest.raises(type(exc)):
        o.run()
    res = json.loads((o.rec.root / "result.json").read_text())
    assert res["status"] == "error" and res["reason"].startswith("infra: ")
    assert len(llm.requests) == 1 and v.calls == 1
    assert "INFRA_ERROR" in res["states"]


def test_cli_run_exits_3_on_infra(tmp_path, monkeypatch):
    from mxagent import cli

    def boom(*a, **k):
        raise InfraError("eagain: Resource temporarily unavailable", "compile")
    monkeypatch.setattr("mxagent.run.baselines.run_loop", boom)
    rc = cli.main(["run", "--task", str(ROOT / "benchmarks" / "b01_e2m1_decode"), "--mode",
                   "baseline-b", "--runs-dir", str(tmp_path)])
    assert rc == 3


# ------------------------------------------------------- pulizia dei processi
CHILD = r"""
import subprocess, sys, time
g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
open(sys.argv[1], "w").write(str(g.pid))
if sys.argv[2] == "hang":
    time.sleep(120)
"""


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True,
                             text=True).stdout
        return str(pid) in out
    return procs.alive(pid)


def _wait_dead(pid: int, s: float = 10) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < s:
        if not _alive(pid):
            return True
        time.sleep(0.2)
    return False


def test_timeout_kills_grandchildren(tmp_path):
    pidf = tmp_path / "pid"
    r = shell.run([sys.executable, "-c", CHILD, str(pidf), "hang"], timeout=3, grace=2)
    assert r.timed_out
    assert _wait_dead(int(pidf.read_text())), "il nipote è sopravvissuto al timeout"


@pytest.mark.skipif(os.name == "nt", reason="pulizia del gruppo a comando finito: solo POSIX")
def test_leftover_grandchildren_are_killed_after_normal_exit(tmp_path):
    pidf = tmp_path / "pid"
    r = shell.run([sys.executable, "-c", CHILD, str(pidf), "exit"], timeout=30, grace=2)
    assert r.ok
    assert _wait_dead(int(pidf.read_text()))
    # keep_group (sbt --client): il server persistente sopravvive
    pidf2 = tmp_path / "pid2"
    shell.run([sys.executable, "-c", CHILD, str(pidf2), "exit"], timeout=30, keep_group=True)
    pid2 = int(pidf2.read_text())
    assert _alive(pid2)
    procs.kill_pids([pid2], grace=2)
    assert _wait_dead(pid2)


def test_verification_that_times_out_leaves_no_children(tmp_path, monkeypatch):
    """Verifica vera (sbt sostituito da un processo che lancia un figlio e si blocca):
    dopo il timeout di elaborate non resta vivo nessun processo."""
    pidf = tmp_path / "pid"
    monkeypatch.setattr(chisel, "sbt_cmd", lambda mode, c: [sys.executable, "-c", CHILD,
                                                             str(pidf), "hang"])
    monkeypatch.setattr(chisel.elaborate, "__defaults__", ("batch", 3, chisel.DEFAULT_JVM_CPUS,
                                                         chisel.DEFAULT_SERVER_START_S))
    v = Verifier(TASK, tmp_path / "v", sbt_mode="batch", infra_backoff=())
    rep = v.verify({"E2M1Decode.scala": "import chisel3._\nclass E2M1Decode extends RawModule {}"},
                   "c00")
    v.close()
    assert rep.stage == "compile" and "timeout" in rep.feedback
    assert _wait_dead(int(pidf.read_text())), "processo figlio rimasto vivo dopo la verifica"


# ------------------------------------------------------------- controllo risorse
def _snap(**kw):
    base = dict(nproc_soft=100000, nproc_hard=100000, nofile_soft=65536, nofile_hard=65536,
                user_procs=50, user_threads=400, threads_max=1000000, cgroups=[], java=[],
                mem_total=128 * 2**30, mem_available=100 * 2**30, ncpu=20, orphans=[])
    base.update(kw)
    return procs.Snapshot(**base)


def test_preflight_thresholds(monkeypatch):
    monkeypatch.setattr(procs, "supported", lambda: True)  # come su Linux
    err, warn, est = procs.preflight(_snap())
    assert err == [] and est["need_tasks"] == procs.job_peak_threads(20) + procs.RESERVED_TASKS
    assert "parallel" not in est
    # TasksMax troppo basso anche per un solo job: 900 task liberi < 1 job + quota riservata
    err, _, est = procs.preflight(_snap(cgroups=[{"path": "/u", "max": "1200", "current": "300"}]))
    assert est["need_tasks"] > 900
    assert err and "per un job ne servono" in err[0] and "unable to create native thread" in err[0]
    assert "TasksMax" in err[0] and "--parallel" not in err[0]
    # con il limite che basta per un job: nessun errore
    err, _, _ = procs.preflight(_snap(cgroups=[{"path": "/u", "max": "4000", "current": "300"}]))
    assert err == []
    orphan = procs.Proc(4242, 1, 1000, 120, "java", "java -jar sbt-launch.jar", "/x/runs/r/sbt")
    err, _, _ = procs.preflight(_snap(orphans=[orphan]))
    assert err and "orfane" in err[0] and "4242" in err[0]
    _, warn, _ = procs.preflight(_snap(nofile_soft=1024))
    assert any("ulimit -n" in w for w in warn)
    _, warn, _ = procs.preflight(_snap(mem_available=2 * 2**30))
    assert any("per un job" in w for w in warn)


def test_orphans_are_judged_by_working_dir_not_parent(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "b05_s0"
    live = procs.Proc(10, 1, 1000, 3, "python", f"python -m mxagent run --runs-dir {tmp_path}/runs "
                      f"--run-id b05_s0", str(tmp_path))
    jvm_live = procs.Proc(11, 1, 1000, 90, "java", "java -jar sbt-launch.jar", str(run_dir / "sbt"))
    jvm_dead = procs.Proc(12, 1, 1000, 90, "java", "java -jar sbt-launch.jar",
                          str(tmp_path / "runs" / "old" / "sbt"))
    monkeypatch.setattr(procs, "under", lambda root, cwd: str(cwd).startswith(str(root)))
    assert [p.pid for p in procs.orphan_jvms([live, jvm_live, jvm_dead])] == [12]


def test_dry_run_shows_parallel_and_peak(capsys):
    from mxagent import cli
    assert cli.main(["eval", "--campaign", str(ROOT / "configs" / "eval" / "main.yaml"), "--dry-run",
                     "--parallel", "3", "--out", str(ROOT / ".pytest_cache" / "dry_main")]) == 0
    out = capsys.readouterr().out
    assert "parallel: 3 job insieme" in out and "picco stimato di processi+thread: 3 x" in out
