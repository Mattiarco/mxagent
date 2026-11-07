"""La KB è verificata: ogni esempio sbagliato fallisce col messaggio previsto,
ogni esempio corretto elabora davvero (sbt + Chisel 6.6 + firtool)."""
import os
from pathlib import Path

import pytest

from mxagent.rag import kb
from mxagent.util.shell import which
from mxagent.verify import chisel

pytestmark = pytest.mark.skipif(not which("sbt") or os.name == "nt",
                                reason="richiede sbt su Linux (WSL o server)")


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    base = os.environ.get("MXAGENT_WORK")
    w = chisel.prepare_workspace((Path(base) / "kb" if base else tmp_path_factory.mktemp("kb")) / "sbt")
    yield w
    from mxagent.util.shell import run
    run(chisel.sbt_cmd("client", "shutdown"), cwd=w, timeout=60)


def _elab(ws, src, tag):
    chisel.write_sources(ws, f"kb_{tag}", {"Kb.scala": src})
    return chisel.elaborate(ws, f"kb_{tag}", "Kb", ws.parent / f"out_{tag}", mode="client")


@pytest.mark.parametrize("p", [p for p in kb.load() if p.right], ids=lambda p: p.id)
def test_right_example_elaborates(ws, p):
    r = _elab(ws, p.right, p.id + "_ok")
    assert r.ok, "\n".join(e.render() for e in r.errors)


@pytest.mark.parametrize("p", [p for p in kb.load() if p.wrong], ids=lambda p: p.id)
def test_wrong_example_fails_with_trigger(ws, p):
    r = _elab(ws, p.wrong, p.id + "_bad")
    assert not r.ok, f"{p.id}: l'esempio sbagliato compila"
    msg = "\n".join(e.render() for e in r.errors)
    assert kb.matching(msg, limit=99) and p in kb.matching(msg, limit=99), f"{p.id}: nessun trigger su\n{msg}"


@pytest.mark.parametrize("p", [p for p in kb.load() if p.wrong and p.autofix], ids=lambda p: p.id)
def test_autofix_repairs_wrong_example(ws, p):
    fixed = kb.sanitize(p.wrong)
    assert p.id in fixed.applied
    r = _elab(ws, fixed.code, p.id + "_fix")
    assert r.ok, "\n".join(e.render() for e in r.errors)
