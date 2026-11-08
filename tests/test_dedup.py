"""Candidati duplicati: normalizzazione del sorgente, nessuna riverifica, una
rigenerazione con temperatura più alta e nota esplicita nel prompt."""
import json

from mxagent.agents.dedup import normalize, retry_temperature, source_hash
from mxagent.agents.orchestrator import Orchestrator
from mxagent.llm.chat import ScriptedChat
from tests.test_synth import B01, FakeVerifier, cfg, fake_synth

SRC = """package foo.bar
import chisel3._

// decoder E2M1
class E2M1Decode extends RawModule {
  val a = IO(Input(UInt(4.W)))   /* ingresso */
  val y = IO(Output(SInt(5.W)))
  y := a.asSInt // TODO
}
"""

SAME = """import chisel3._
/* versione
   riformattata */
class   E2M1Decode extends RawModule {

    val a = IO(Input(UInt(4.W)))
	val y = IO(Output(SInt(5.W)))   // uscita
    y := a.asSInt
}
"""


def test_normalization_ignores_comments_and_whitespace():
    assert normalize(SRC) == normalize(SAME)
    assert source_hash({"E2M1Decode.scala": SRC}) == source_hash({"E2M1Decode.scala": SAME})
    # una differenza nel codice cambia l'hash
    other = SAME.replace("asSInt", "zext")
    assert source_hash({"E2M1Decode.scala": SRC}) != source_hash({"E2M1Decode.scala": other})
    # il contenuto delle stringhe conta, anche se contiene // o /*
    s1 = 'val m = "// ok 1" // c'
    s2 = 'val m = "// ok 2" // c'
    assert normalize(s1) == 'val m = "// ok 1"' and normalize(s1) != normalize(s2)
    # stesso codice in file diversi = candidato diverso
    assert source_hash({"A.scala": SRC}) != source_hash({"B.scala": SRC})


def test_retry_temperature():
    assert retry_temperature(0.3) == 0.5
    assert retry_temperature(0.8) == 0.9
    assert retry_temperature(1.0) == 1.0  # mai più bassa di quella già usata


def mk(top, tag, extra=""):
    """Come tests.test_synth.code, con commenti/spazi extra che non cambiano il codice."""
    return (f"IDEA: idea {tag}.\n```scala\nimport chisel3._\n{extra}class {top} extends RawModule "
            f"{{ {tag}\n  val m = \"{tag}\" }}\n```")


class LogVerifier(FakeVerifier):
    def __init__(self, task, log):
        super().__init__(task)
        self.log = log

    def verify(self, sources, tag):
        self.log.append(tag)
        return super().verify(sources, tag)


def run(tmp, replies, **over):
    log: list = []
    llm = ScriptedChat(replies)
    c = cfg(**{"synth.backends": ["generic"], **over})
    o = Orchestrator(B01, c, seed=1, runs_dir=tmp, llm=llm,
                     verifier_factory=lambda t, n: LogVerifier(t, log), synth_fn=fake_synth)
    r = o.run()
    trace = json.loads((r.run_dir / "trace.json").read_text())
    return r, llm, log, trace


def test_ppa_duplicate_not_verified_and_regenerated_once(tmp_path):
    replies = [mk("E2M1Decode", "// ok 100,10"),                          # solve: passa
               mk("E2M1Decode", "// ok 120,10"),                          # ppa00: rifiutato
               mk("E2M1Decode", "// ok 120,10", extra="// stessa idea\n\n"),  # ppa01: duplicato
               mk("E2M1Decode", "// ok 70,10")]                           # rigenerato: accettato
    r, llm, log, trace = run(tmp_path, replies, **{"synth.optimize_iters": 2})
    assert log == ["E2M1Decode_00", "E2M1Decode_ppa00", "E2M1Decode_ppa01", "E2M1Decode_final"]
    ppa = json.loads((r.run_dir / "ppa.json").read_text())["optimizer"]
    assert [(s["n"], s["outcome"]) for s in ppa["steps"]] == [(0, "rejected"), (1, "duplicate"),
                                                             (1, "accepted")]
    assert ppa["steps"][1]["tag"] is None and "identical to E2M1Decode_ppa00 (rejected)" in ppa["steps"][1]["detail"]
    # la rigenerazione: seed diverso, temperatura +0.2, nota e idee già provate nel prompt
    dup_req, regen = llm.requests[2], llm.requests[3]
    assert regen["tag"].endswith(":dedup") and regen["seed"] != dup_req["seed"]
    t0 = cfg().llm.roles["optimizer"].temperature
    assert regen["temperature"] == retry_temperature(t0) and dup_req["temperature"] == t0
    p = regen["messages"][1]["content"]
    assert "identical to an already-evaluated design" in p and "structurally different" in p
    assert "IDEAS ALREADY TRIED:\n- idea // ok 120,10." in p
    # nel trace il duplicato ha un evento "duplicate" e nessun "verify"
    dups = [e for e in trace if e["kind"] == "duplicate"]
    assert len(dups) == 1 and dups[0]["same_as"] == "E2M1Decode_ppa00"
    res = json.loads((r.run_dir / "result.json").read_text())
    assert res["duplicates"] == 1 and res["ppa"]["attempts"]["duplicate"] == 1
    assert res["best_candidate"] == "E2M1Decode_ppa01"
    stop = [e for e in trace if e["kind"] == "ppa_stop"][0]
    assert stop["iters_done"] == 2 and stop["duplicates"] == 1


def test_ppa_duplicate_of_starting_design(tmp_path):
    """Il caso b01: l'optimizer ripropone il design di partenza; anche la rigenerazione
    è un duplicato -> si passa al tentativo successivo senza verificare nulla."""
    start = mk("E2M1Decode", "// ok 100,10")
    replies = [start, start, start, mk("E2M1Decode", "// ok 110,10")]
    r, llm, log, trace = run(tmp_path, replies, **{"synth.optimize_iters": 2})
    assert log == ["E2M1Decode_00", "E2M1Decode_ppa01", "E2M1Decode_final"]
    ppa = json.loads((r.run_dir / "ppa.json").read_text())["optimizer"]
    assert [s["outcome"] for s in ppa["steps"]] == ["duplicate", "duplicate", "rejected"]
    assert "identical to E2M1Decode_00 (pass)" in ppa["steps"][0]["detail"]


def test_solver_duplicate_of_failed_candidate_not_reverified(tmp_path):
    bug = mk("E2M1Decode", "// bug")
    replies = [bug, mk("E2M1Decode", "// bug", extra="/* rifatto */\n"), mk("E2M1Decode", "// ok 100,10")]
    r, llm, log, trace = run(tmp_path, replies, **{"agent.ppa": False, "budget.max_fix_iters": 2,
                                                   "agent.escape": False})
    assert r.metrics["passed"]
    assert log == ["E2M1Decode_00", "E2M1Decode_01", "E2M1Decode_final"]
    assert llm.requests[2]["tag"].endswith("reflect:dedup")
    assert "identical to an already-evaluated design (E2M1Decode_00: sim)" in \
        llm.requests[2]["messages"][1]["content"]
    verifies = [e for e in trace if e["kind"] == "verify"]
    assert len(verifies) == 2  # il duplicato non ha un evento verify
    assert json.loads((r.run_dir / "result.json").read_text())["duplicates"] == 1
