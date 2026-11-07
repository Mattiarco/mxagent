"""KB e sanitizer senza toolchain."""
import re

from mxagent.rag import kb


def test_kb_loads_and_is_consistent():
    ps = kb.load()
    assert len(ps) >= 15
    for p in ps:
        assert p.rule and (p.right or p.wrong)
        for t in p.triggers:
            re.compile(t)
        if not p.silent and p.wrong:
            assert p.triggers, f"{p.id}: errore di compilazione senza trigger"
    assert "FlatIO" in kb.rules_text()


def test_matching_on_real_messages():
    m = kb.matching("[scalac] X.scala:7:24: type mismatch; found : Int(4) required: chisel3.Width")
    assert m[0].id == "width_literal"
    m = kb.matching("[scalac] X.scala:24:19: value reduceTree is not a member of IndexedSeq[chisel3.SInt]")
    assert m[0].id == "seq_reduce_tree"
    assert kb.matching("everything fine") == []


def test_sanitize_fixes_chisel3_style():
    code = """package foo
import chisel3._
import chisel3.Driver
class Top extends Module {
  val io = IO(new Bundle { val a = Input(UInt(4)); val b = Output(Bool()) })
  val c = 5.U(8)
  io.b := io.a(0).toBool && io.a.asUInt() =/= 0.U
  val t = Cat(io.a, io.a)
}
object Main extends App {
  chisel3.Driver.execute(args, () => new Top)
}
"""
    r = kb.sanitize(code)
    c = r.code
    assert "FlatIO(new Bundle" in c and "UInt(4.W)" in c and "5.U(8.W)" in c
    assert ".asBool" in c and "asUInt()" not in c and "object Main" not in c
    assert "import chisel3.Driver" not in c and "import chisel3.util._" in c
    for fix in ("no_main_object", "width_literal", "literal_width_apply", "as_without_parens",
                "to_bool", "flat_io", "import_util", "drop_stage_imports"):
        assert fix in r.applied, fix


def test_sanitize_is_idempotent_on_good_code():
    good = "import chisel3._\nclass A extends RawModule { val io = FlatIO(new Bundle { val y = Output(UInt(4.W)) }); io.y := 3.U(4.W) }\n"
    r = kb.sanitize(good)
    assert r.code == good and r.applied == []


def test_examples_render():
    p = next(x for x in kb.load() if x.id == "uint_sint_connect")
    ex = p.example()
    assert "WRONG:" in ex and "RIGHT:" in ex and "import" not in ex
