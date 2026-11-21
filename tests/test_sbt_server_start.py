"""Server di ``sbt --client`` che non parte: il client aspetterebbe fino a ``sim.timeout_s``
(900 s) e il timeout finirebbe al modello come errore del candidato. Se ``active.json``
non compare entro ``sim.sbt_server_start_s`` il comando viene interrotto ed è un errore
di sistema (``sbt_server``): riavvio, retry, niente al modello."""
import sys
import time

import pytest

from mxagent.util import shell
from mxagent.util.shell import Result
from mxagent.verify import chisel
from mxagent.verify.chisel import ElabResult, SourceError
from mxagent.verify.infra import InfraError, classify_build
from mxagent.verify.verifier import Verifier
from tests.test_baselines import TASK

SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


def test_watch_aborts_the_command_before_the_timeout():
    t0 = time.monotonic()
    r = shell.run(SLEEP, timeout=60, grace=1, watch=lambda t: "stop" if t >= 0.5 else None,
                  watch_every=0.1)
    assert r.aborted == "stop" and not r.ok and not r.timed_out
    assert "stop" in r.log and time.monotonic() - t0 < 20


def test_watch_returning_none_lets_the_command_finish():
    r = shell.run([sys.executable, "-c", "print('ok')"], timeout=30, watch=lambda t: None,
                  watch_every=0.05)
    assert r.ok and r.stdout.strip() == "ok" and not r.aborted


def test_timeout_still_applies_with_watch():
    r = shell.run(SLEEP, timeout=0.5, grace=1, watch=lambda t: None, watch_every=0.1)
    assert r.timed_out and not r.aborted


def test_server_watch(tmp_path):
    assert chisel.server_watch(tmp_path, None) is None
    w = chisel.server_watch(tmp_path, 10)
    assert w(5) is None  # in avvio
    reason = w(10)
    assert reason.startswith(chisel.SERVER_NOT_STARTED)
    chisel.active_file(tmp_path).parent.mkdir(parents=True)
    chisel.active_file(tmp_path).write_text("{}")
    assert w(20) is None  # partito
    assert chisel.server_watch(tmp_path, 10) is None  # già vivo all'inizio: niente controllo


def test_elaborate_with_a_server_that_never_starts_is_sbt_server(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, cwd=None, timeout=None, env=None, keep_group=False, watch=None, **kw):
        seen["watch"] = watch
        reason = watch(240)
        return Result(cmd, -1, "", f"\n{reason}\n", 240.0, aborted=reason)
    monkeypatch.setattr(chisel, "run", fake_run)
    el = chisel.elaborate(tmp_path, "cand", "Top", tmp_path / "out", mode="client")
    assert seen["watch"] is not None and not el.ok and not el.timed_out
    hit = classify_build(el.log, [(e.stage, e.message) for e in el.errors], el.returncode,
                         el.timed_out)
    assert hit.startswith("sbt_server: ") and chisel.SERVER_NOT_STARTED in hit


def test_batch_mode_has_no_server_watch(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, watch=None, **kw):
        seen["watch"] = watch
        return Result(cmd, 1, "", "[error] Total time: 1 s", 1.0)
    monkeypatch.setattr(chisel, "run", fake_run)
    chisel.elaborate(tmp_path, "cand", "Top", tmp_path / "out", mode="batch")
    assert seen["watch"] is None


def _verifier(tmp_path, monkeypatch, n_fail):
    calls = []
    reason = f"{chisel.SERVER_NOT_STARTED} in 240s (no active.json)"

    def fake_elab(work, pkg, top, out_dir, mode="batch", timeout=900, **kw):
        calls.append(kw.get("server_start_s"))
        if len(calls) <= n_fail:
            return ElabResult(False, [SourceError("sbt", reason)], [], 240.0, f"\n{reason}\n", None)
        return ElabResult(False, [SourceError("scalac", "';' expected", "Top.scala", 3, 5)], [], 1.0,
                          "[error] Top.scala:3:5: ';' expected", 1)
    monkeypatch.setattr(chisel, "elaborate", fake_elab)
    monkeypatch.setattr(chisel, "prepare_workspace", lambda w: w)
    monkeypatch.setattr(chisel, "write_sources", lambda *a, **k: None)
    v = Verifier(TASK, tmp_path, sbt_mode="client", infra_backoff=(0, 0, 0), server_start_s=240)
    monkeypatch.setattr(v, "kill_leftovers", lambda: [])
    return v, calls


def test_server_not_started_is_retried_and_not_fed_back(tmp_path, monkeypatch):
    v, calls = _verifier(tmp_path, monkeypatch, n_fail=1)
    rep = v.verify({"Top.scala": "x"}, tag="c0")
    assert calls == [240, 240]
    assert chisel.SERVER_NOT_STARTED not in (rep.feedback or "")
    assert "';' expected" in rep.feedback  # l'errore vero del candidato arriva


def test_server_never_starting_ends_as_infra(tmp_path, monkeypatch):
    v, calls = _verifier(tmp_path, monkeypatch, n_fail=99)
    with pytest.raises(InfraError) as e:
        v.verify({"Top.scala": "x"}, tag="c0")
    assert e.value.detail.startswith("sbt_server: ") and len(calls) == 4
