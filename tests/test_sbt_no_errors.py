"""sbt fallito senza errori del sorgente = errore di sistema (server morto, JVM uccisa,
client che non raggiunge il server): verifica ripetuta dopo il riavvio del server,
niente al modello; se persiste ``infra: sbt_no_errors``.

Il log di ``tests/data/infra/E8M0Pair_00_elaborate.log`` viene dalla run di prova s3 su
b05 (blocco E8M0Pair, primo candidato): il server sbt è morto a metà comando e il log
contiene solo la riga finale di sbt. Prima era finito come stage "compile" con
error_key "sbt:None:..." e il feedback "COMPILATION FAILED: [sbt] ?: ..." al modello."""
from pathlib import Path

import pytest

from mxagent.verify import chisel
from mxagent.verify.chisel import ElabResult, SourceError, parse_sbt_log
from mxagent.verify.infra import InfraError, classify_build
from mxagent.verify.verifier import Verifier
from tests.test_baselines import TASK

DATA = Path(__file__).resolve().parent / "data" / "infra"
E8M0_LOG = (DATA / "E8M0Pair_00_elaborate.log").read_text(encoding="utf-8")


def _errs(log):
    return [(e.stage, e.message) for e in parse_sbt_log(log)]


def test_real_log_of_E8M0Pair_00_is_sbt_no_errors():
    errs = parse_sbt_log(E8M0_LOG)
    assert [e.stage for e in errs] == ["sbt"] and errs[0].file is None  # com'era: "[sbt] ?:"
    hit = classify_build(E8M0_LOG, _errs(E8M0_LOG), returncode=1)
    assert hit.startswith("sbt_no_errors: ") and "Total time" in hit


@pytest.mark.parametrize("rc,kind", [(-9, "sbt_signal"), (137, "sbt_signal"), (134, "sbt_signal")])
def test_jvm_killed_by_a_signal_is_infra(rc, kind):
    log = "[error] X.scala:3:5: ';' expected but '(' found."  # anche con errori nel log
    assert classify_build(log, _errs(log), returncode=rc).startswith(kind)


def test_server_not_responding_is_infra():
    log = "[info] server was not detected. starting an instance\n[error] Connection refused"
    assert classify_build(log, _errs(log), returncode=1).startswith("sbt_server")


CANDIDATE_ERRORS = [  # errori veri del candidato: restano "compile" e vanno al modello
    "[error] /w/src/main/scala/cand_x/Top.scala:5:10: ';' expected but '(' found.",
    "[error] /w/src/main/scala/cand_x/Top.scala 33:68: Invalid bit range [hi=0, lo=8]",
    "[error] Top.scala:12:7: error: sink \"io.y\" not fully initialized",
    # eccezione di elaborazione senza posizione nel sorgente: è comunque del candidato
    "[error] chisel3.package$ChiselException: Connection between sink (UInt<4>) and source "
    "(SInt<5>) failed\n[error] Total time: 3 s",
    "[error] java.lang.StackOverflowError\n[error] Total time: 9 s",
]


@pytest.mark.parametrize("log", CANDIDATE_ERRORS)
def test_candidate_errors_are_not_infra(log):
    assert any(s in ("scalac", "firtool", "elaboration") for s, _ in _errs(log))
    assert classify_build(log, _errs(log), returncode=1) is None


def test_timeout_stays_a_candidate_error():
    assert classify_build("", [("sbt", "timeout dopo 900s")], returncode=-15, timed_out=True) is None


def _verifier(tmp_path, monkeypatch, logs, backoff=(0, 0, 0)):
    """Verifier con elaborate finto: un log per chiamata (l'ultimo si ripete)."""
    calls, cleanups = [], []

    def fake_elab(work, pkg, top, out_dir, mode="batch", timeout=900, **kw):
        log = logs[min(len(calls), len(logs) - 1)]
        calls.append(log)
        return ElabResult(False, parse_sbt_log(log) or [SourceError("sbt", "x")], [], 1.0, log, 1)
    monkeypatch.setattr(chisel, "elaborate", fake_elab)
    monkeypatch.setattr(chisel, "prepare_workspace", lambda w: w)
    monkeypatch.setattr(chisel, "write_sources", lambda *a, **k: None)
    v = Verifier(TASK, tmp_path, infra_backoff=backoff)
    monkeypatch.setattr(v, "kill_leftovers", lambda: cleanups.append(1) or [])
    return v, calls, cleanups


def test_dead_server_is_retried_after_restart_and_not_fed_back(tmp_path, monkeypatch):
    real = CANDIDATE_ERRORS[0]
    v, calls, cleanups = _verifier(tmp_path, monkeypatch, [E8M0_LOG, real])
    rep = v.verify({"E2M1Decode.scala": "class E2M1Decode"}, "E8M0Pair_00")
    assert len(calls) == 2 and len(cleanups) == 1  # server chiuso e JVM terminate, poi di nuovo
    assert rep.stage == "compile" and "';' expected" in rep.feedback  # il vero errore
    assert "Total time" not in rep.feedback and not any(k.startswith("sbt:") for k in rep.error_keys)
    assert rep.infra_retries[0]["detail"].startswith("sbt_no_errors")


def test_persistent_sbt_no_errors_ends_as_infra(tmp_path, monkeypatch):
    v, calls, _ = _verifier(tmp_path, monkeypatch, [E8M0_LOG])
    with pytest.raises(InfraError) as ei:
        v.verify({"E2M1Decode.scala": "class E2M1Decode"}, "E8M0Pair_00")
    assert len(calls) == 4 and ei.value.detail.startswith("sbt_no_errors")
    assert ei.value.report.stage == "infra" and ei.value.report.feedback == ""
    from mxagent.verify.infra import infra_outcome
    assert infra_outcome(ei.value)["reason"].startswith("infra: sbt_no_errors")
