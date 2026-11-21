"""Task.interface_text(): le porte vanno nel prompt come codice Chisel 6 valido.

Prima le porte erano elencate come ``- x: Input(SInt(14.W)) -- ...`` e il modello
copiava la notazione nel Bundle (``val x: Input(...)`` -> scalac: ';' expected).
Ora il testo contiene la dichiarazione FlatIO pronta da copiare, con la semantica
in un commento. Il test di integrazione la compila ed elabora davvero.
"""
import os
import re
import textwrap
from pathlib import Path

import pytest

from mxagent.bench import load_task
from mxagent.bench.subkernels import CATALOG, block_task
from mxagent.util.shell import which

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"

BAD = re.compile(r"val\s+\w+\s*:\s*(Input|Output)\(")
OLD = re.compile(r"\b\w+\s*:\s*(Input|Output)\(")  # vecchia notazione "nome: Input("
HEAD = "val io = FlatIO(new Bundle {"


def _tasks():
    out = [(d.name, load_task(d)) for d in sorted(BENCH.iterdir()) if (d / "task.yaml").is_file()]
    out += [(f"block:{k}", block_task(k, "Blk_" + k)) for k in CATALOG]
    return out


TASKS = _tasks()
IDS = [n for n, _ in TASKS]


def extract_flatio(text: str) -> str:
    """Il blocco ``val io = FlatIO(new Bundle {...})`` così come compare nel prompt."""
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip() == HEAD)
    ind = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    j = next(k for k in range(i + 1, len(lines)) if lines[k] == ind + "})")
    return textwrap.dedent("\n".join(lines[i:j + 1]))


def test_catalog_has_fixed_to_fp32():
    assert "fixed_to_fp32" in CATALOG and "block:fixed_to_fp32" in IDS
    assert len([n for n in IDS if not n.startswith("block:")]) >= 8


@pytest.mark.parametrize("submodule", [False, True])
@pytest.mark.parametrize("name,t", TASKS, ids=IDS)
def test_interface_is_valid_flatio(name, t, submodule):
    text = t.interface_text(submodule=submodule)
    assert HEAD in text
    assert not BAD.search(text), text
    assert not OLD.search(text), text
    block = extract_flatio(text)
    for p in t.ports:  # una dichiarazione per porta, con tipo e commento
        m = re.search(rf"^\s*val {re.escape(p.name)}\s+= (\S+)\s+// (.*)$", block, re.M)
        assert m, (p.name, block)
        assert m.group(1) == p.chisel_type() and m.group(2) == p.doc_text()
    comments = {ln.index("//") for ln in block.splitlines() if "//" in ln}
    assert len(comments) == 1  # commenti allineati
    if t.timing.kind == "comb":
        assert "RawModule" in text
    else:
        assert f"fixed latency {t.timing.latency}" in text and "Module" in text
    if submodule:
        assert "do NOT redeclare" in text and f"Module(new {t.top})" in text


def test_fixed_to_fp32_example():
    text = block_task("fixed_to_fp32", "FixedToFp32").interface_text()
    assert text.splitlines()[0] == "Top module: class FixedToFp32, purely combinational (extend RawModule)."
    assert "      val x   = Input(SInt(14.W))   // two's complement signed integer in units of 2^-2" in text
    assert "      val y   = Output(UInt(32.W))  // IEEE-754 binary32 bit pattern" in text


def test_vector_packing_goes_in_comment():
    t = load_task(BENCH / "b05_mxfp4_dot32")
    line = next(ln for ln in t.interface_text().splitlines() if re.match(r"\s*val a\s+=", ln))
    code, comment = line.split("//", 1)
    assert "UInt(128.W)" in code and "packed" not in code
    assert "vector of 32 x 4-bit elements packed (element i at bits [4*i+3:4*i])" in comment


# ------------------------------------------------------------ integrazione
def _stub(cls: str, t, text: str) -> str:
    parent = "RawModule" if t.timing.kind == "comb" else "Module"
    outs = "\n".join(f"  io.{p.name} := DontCare" for p in t.outputs)
    # dontTouch: senza logica firtool eliminerebbe lo stub e non si vedrebbero le porte Verilog
    return (f"class {cls} extends {parent} {{\n{textwrap.indent(extract_flatio(text), '  ')}\n"
            f"{outs}\n  dontTouch(io)\n}}\n")


@pytest.mark.integration
@pytest.mark.skipif(not which("sbt"), reason="sbt non trovato")
def test_flatio_blocks_compile_and_elaborate(tmp_path):
    """Tutti i blocchi FlatIO (task + catalogo, top e sottomodulo) in UNA elaborazione:
    ogni stub ha le uscite a DontCare, un top li istanzia e ne pilota gli ingressi."""
    from mxagent.verify import chisel

    stubs, inst = [], []
    for i, (_, t) in enumerate(TASKS):
        for sub in (False, True):
            cls = f"Stub{i}{'s' if sub else 't'}"
            stubs.append(_stub(cls, t, t.interface_text(submodule=sub)))
            inst.append(f"  val u_{cls} = Module(new {cls})")
            inst += [f"  u_{cls}.io.{p.name} := DontCare" for p in t.inputs]
    src = ("import chisel3._\nimport chisel3.util._\n\n" + "\n".join(stubs)
           + "\nclass AllPorts extends Module {\n" + "\n".join(inst) + "\n}\n")
    base = Path(os.environ["MXAGENT_WORK"]) / "pytest_iface" if os.environ.get("MXAGENT_WORK") else tmp_path
    ws = chisel.prepare_workspace(base / "sbt")
    chisel.write_sources(ws, "iface_check", {"AllPorts.scala": src})
    el = chisel.elaborate(ws, "iface_check", "AllPorts", base / "verilog")
    assert el.ok, "\n".join(e.render() for e in el.errors) + "\n" + el.log[-3000:]
    sv = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in el.verilog)
    for i, (_, t) in enumerate(TASKS):  # FlatIO: nomi Verilog = nomi delle porte, senza io_
        m = re.search(rf"module Stub{i}t\((.*?)\);", sv, re.S)
        assert m, f"Stub{i}t mancante"
        header = re.sub(r"//[^\n]*", "", m.group(1))  # firtool raggruppa: "input [3:0] a, b"
        found = set(re.findall(r"\w+", header))
        assert {p.name for p in t.ports} <= found, (t.name, found)
    assert "io_" not in sv
