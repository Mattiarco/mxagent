"""Verifier: parti testabili senza toolchain (parser su log reali, feedback, ddmin)."""
from pathlib import Path

import pytest

from mxagent.bench import KERNELS, load_task, reference_sources
from mxagent.bench.task import Task
from mxagent.formats.packing import pack, to_signed, unpack
from mxagent.verify import chisel, lint
from mxagent.verify.ddmin import complexity, minimize
from mxagent.verify.feedback import SimSummary, compare, patterns, pick_representatives
from mxagent.verify.verifier import Verifier

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "tests" / "fixtures" / "logs"
BENCH = ROOT / "benchmarks"


# ------------------------------------------------------------------ parser
def test_parse_scalac_error():
    errs = chisel.parse_sbt_log((LOGS / "b01_chisel3_api.log").read_text())
    assert len(errs) == 1
    e = errs[0]
    assert (e.stage, e.file, e.line, e.col) == ("scalac", "b01_chisel3_api.scala", 9, 24)
    assert e.message == "type mismatch; found : Int(4) required: chisel3.Width"


def test_parse_elaboration_error():
    errs = chisel.parse_sbt_log((LOGS / "b01_type_mismatch.log").read_text(), pkg="cand_b01_type_mismatch")
    assert len(errs) == 1
    e = errs[0]
    assert e.stage == "elaboration" and e.message.startswith("ChiselException: Connection between sink")
    assert (e.file, e.line) == ("b01_type_mismatch.scala", 13)


def test_parse_firtool_error():
    errs = chisel.parse_sbt_log((LOGS / "b01_comb_loop.log").read_text())
    assert len(errs) == 1
    e = errs[0]
    assert e.stage == "firtool" and e.message.startswith("detected combinational cycle")
    assert "class E2M1Decode" not in e.message and (e.file, e.line) == ("b01_comb_loop.scala", 7)


def test_parse_chisel6_elaboration_list():
    # log reale (campagna M4): il messaggio era finito vuoto ("Error:")
    errs = chisel.parse_sbt_log((LOGS / "chisel6_elab_list.log").read_text())
    assert len(errs) == 1
    e = errs[0]
    assert (e.stage, e.file, e.line, e.col) == ("elaboration", "AdderTree.scala", 33, 68)
    assert e.message == "Invalid bit range [hi=0, lo=8]"


def test_error_key_ignores_line_numbers():
    a = chisel.SourceError("scalac", "type mismatch at 12", "X.scala", 12)
    b = chisel.SourceError("scalac", "type mismatch at 40", "X.scala", 40)
    assert a.key() == b.key()


def test_snippet_and_line_offset(tmp_path):
    src = tmp_path / "A.scala"
    src.write_text(chisel.inject_package("package foo\nclass A {\n  val x = 1\n}\n", "cand_a"))
    assert src.read_text().startswith("package cand_a\n\nclass A")
    log = f"[error] {src}:4:7: not found: value y\n[error]   val x = 1\n[error]       ^\n"
    e = chisel.parse_sbt_log(log, tmp_path)[0]
    assert e.line == 2  # riga come nel sorgente scritto dall'LLM (senza package iniettato)
    assert ">   2 |   val x = 1" in e.snippet


def test_write_sources_cleans_previous(tmp_path):
    ws = chisel.prepare_workspace(tmp_path / "ws")
    assert (ws / "build.sbt").is_file() and (ws / "src/main/scala/mxagent/Elaborate.scala").is_file()
    chisel.write_sources(ws, "cand_1", {"A": "class A"})
    chisel.write_sources(ws, "cand_2", {"B.scala": "class B"})
    pkgs = sorted(p.name for p in (ws / "src/main/scala").iterdir())
    assert pkgs == ["cand_2", "mxagent"]


def test_parse_verilator(tmp_path):
    sv = tmp_path / "Top.sv"
    sv.write_text("module Top();\n  wire [3:0] a; // @[Foo.scala 12:5]\n  assign a = b;\nendmodule\n")
    log = ("%Warning-WIDTHTRUNC: " + str(sv) + ":3:12: Operator ASSIGNW expects 4 bits\n"
           "%Warning-UNUSEDSIGNAL: " + str(sv) + ":2:14: Signal is not used: 'a'\n"
           "%Error: Exiting due to 1 warning(s)\n")
    issues = lint.parse_verilator(log, [sv])
    assert len(issues) == 1
    i = issues[0]
    assert i.code == "WIDTHTRUNC" and i.blocking and i.scala_loc == "Foo.scala:12:5"


# ------------------------------------------------------------------ task
@pytest.mark.parametrize("name", ["b01_e2m1_decode", "b02_e2m1_mul", "b03_e8m0_to_fp32"])
def test_benchmarks_load(name):
    t = load_task(BENCH / name)
    assert t.golden in KERNELS and reference_sources(t) and t.spec_text
    v = Verifier(t, Path("unused"))
    vecs = v.vectors()
    assert vecs and all(s == "exhaustive" for s, _ in vecs)
    outs = {p.name for p in t.outputs}
    for _, x in vecs[:20]:
        y = v.golden(x)
        assert set(y) == outs
        for p in t.outputs:
            assert 0 <= y[p.name] < (1 << p.total_width)
    assert "FlatIO" in t.interface_text()


def test_b01_golden_matches_spec_examples():
    v = Verifier(load_task(BENCH / "b01_e2m1_decode"), Path("unused"))
    to = lambda a: to_signed(v.golden({"a": a})["y"], 5)  # noqa: E731
    assert [to(a) for a in (0x1, 0x2, 0x3, 0x7, 0xF, 0x8)] == [1, 2, 3, 12, -12, 0]


# ------------------------------------------------------------------ feedback
def _b02_or_mutant(v):
    """Modello Python del mutante b02_sign_or (segno con OR)."""
    mags = [0, 1, 2, 3, 4, 6, 8, 12]
    p = mags[v["a"] & 7] * mags[v["b"] & 7]
    if (v["a"] >> 3) | (v["b"] >> 3):
        p = -p
    return [p & 0x1FF]


def test_patterns_find_conjunction():
    t = load_task(BENCH / "b02_e2m1_mul")
    ver = Verifier(t, Path("unused"))
    vecs = [x for _, x in ver.vectors()]
    exp = [ver.golden(x) for x in vecs]
    got = [_b02_or_mutant(x) for x in vecs]
    mism = compare(t, vecs, exp, got, ["exhaustive"] * len(vecs))
    assert len(mism) == 49  # come misurato in simulazione col mutante vero
    pats = patterns(t, vecs, {m.index for m in mism})
    # 15 vettori con a, b negativi passano: prodotto zero (-0 == 0). La congiunzione
    # deve comunque battere i singoli "a=neg" / "b=neg" (79/207)
    assert pats[0] == "every failing vector has a=neg AND b=neg; 15/207 passing vectors do"
    reps = pick_representatives(t, mism, 3, complexity)
    assert len(reps) == 3 and all(r.bad_ports == ["p"] for r in reps)
    s = SimSummary(total=len(vecs), failed=len(mism), per_suite={"exhaustive": [256, 49]},
                   patterns=pats, examples=reps)
    txt = s.render(t)
    assert "49/256" in txt and "expected 1 [000000001] = +0.25" in txt


# ------------------------------------------------------------------ ddmin
VEC_TASK = Task.model_validate({
    "name": "fake_dot", "title": "x", "difficulty": 5, "top": "Dot", "golden": "e2m1_decode",
    "ports": [{"name": "s", "dir": "in", "type": "e8m0"},
              {"name": "a", "dir": "in", "type": "e2m1", "elems": 32},
              {"name": "y", "dir": "out", "type": "fp32"}]})


def test_ddmin_isolates_culprit_element():
    import random
    rng = random.Random(0)
    codes = [rng.randrange(16) for _ in range(32)]
    codes[5] = 0x1  # il colpevole: un subnormale in posizione 5
    for i in range(32):
        if i != 5 and codes[i] & 7 == 1:
            codes[i] = 0x2
    v = {"s": 130, "a": pack(codes, 4)}
    sims = []

    def still_fails(cands):  # "DUT" difettoso: sbaglia se l'elemento 5 è subnormale
        sims.append(len(cands))
        return [unpack(c["a"], 4, 32)[5] & 7 == 1 for c in cands]

    m, rounds = minimize(VEC_TASK, v, still_fails)
    assert unpack(m["a"], 4, 32) == tuple(0x1 if i == 5 else 0 for i in range(32))
    assert m["s"] == 127  # scala portata a 1
    assert complexity(VEC_TASK, m) == 1 and rounds == len(sims) <= 8
