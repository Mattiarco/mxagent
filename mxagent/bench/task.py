"""Contratto di un benchmark: porte tipizzate, temporizzazione, golden kernel, vettori.

Un task vive in ``benchmarks/<nome>/``:

``task.yaml``      contratto (questo modulo lo valida)
``spec.md``        specifica in linguaggio naturale (va nei prompt)
``reference/``     design di riferimento scritto a mano (MAI visibile all'agente)

Il tipo di una porta decide come i valori vengono scomposti nel feedback
(segno/esponente/mantissa) e quanti bit occupa.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..formats.numerics import Numerics

#: larghezza in bit degli elementi per tipo (None = la dichiara la porta)
TYPE_WIDTH: dict[str, int | None] = {
    "e2m1": 4, "e8m0": 8, "e4m3": 8, "fp32": 32, "bf16": 16, "uint": None, "sint": None,
    "bool": 1,
}


class Port(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dir: Literal["in", "out"]
    type: Literal["e2m1", "e8m0", "e4m3", "fp32", "bf16", "uint", "sint", "bool"]
    width: int | None = None  # larghezza di UN elemento (obbligatoria per uint/sint)
    elems: int = Field(1, ge=1)  # >1: vettore impacchettato (elemento 0 negli LSB)
    frac_bits: int = 0  # sint/uint in virgola fissa: valore = intero * 2^-frac_bits
    doc: str = ""

    @model_validator(mode="after")
    def _width(self) -> "Port":
        fixed = TYPE_WIDTH[self.type]
        if fixed is None:
            if not self.width:
                raise ValueError(f"porta {self.name}: tipo {self.type} richiede width")
        elif self.width not in (None, fixed):
            raise ValueError(f"porta {self.name}: {self.type} è largo {fixed} bit")
        else:
            self.width = fixed
        return self

    @property
    def elem_width(self) -> int:
        assert self.width is not None
        return self.width

    @property
    def total_width(self) -> int:
        return self.elem_width * self.elems

    def chisel_type(self) -> str:
        t = "SInt" if self.type == "sint" and self.elems == 1 else "UInt"
        if self.type == "bool":
            t = "Bool"
            return f"{'Input' if self.dir == 'in' else 'Output'}(Bool())"
        return f"{'Input' if self.dir == 'in' else 'Output'}({t}({self.total_width}.W))"

    def describe(self) -> str:
        what = {"e2m1": "FP4 E2M1 code", "e8m0": "E8M0 scale code", "e4m3": "FP8 E4M3 code",
                "fp32": "IEEE-754 binary32 bit pattern", "bf16": "bfloat16 bit pattern",
                "uint": "unsigned integer", "sint": "two's complement signed integer",
                "bool": "1-bit flag"}[self.type]
        if self.type in ("uint", "sint") and self.frac_bits:
            what += f" in units of 2^-{self.frac_bits}"
        vec = (f", vector of {self.elems} x {self.elem_width}-bit elements packed "
               f"(element i at bits [{self.elem_width}*i+{self.elem_width - 1}:{self.elem_width}*i])"
               if self.elems > 1 else "")
        return f"{self.name}: {self.chisel_type()} -- {what}{vec}" + (f". {self.doc}" if self.doc else "")


class Timing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["comb", "pipe"] = "comb"
    latency: int = Field(0, ge=0)  # cicli fra ingresso e uscita (pipe)
    target_mhz: float | None = None

    @model_validator(mode="after")
    def _lat(self) -> "Timing":
        if self.kind == "comb" and self.latency:
            raise ValueError("un task combinatorio ha latenza 0")
        if self.kind == "pipe" and not self.latency:
            raise ValueError("un task pipe richiede latency >= 1")
        return self


class VectorPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exhaustive: bool = False  # tutto lo spazio degli ingressi (solo se piccolo)
    random: int = 0  # vettori casuali (generatore del kernel)
    corner: bool = True  # casi limite scelti a mano dal kernel


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    title: str
    difficulty: int = Field(ge=1, le=8)
    top: str  # nome del modulo Chisel/Verilog di primo livello
    timing: Timing = Timing()
    ports: list[Port]
    golden: str  # chiave in mxagent.bench.kernels.KERNELS
    numerics: dict[str, Any] = {}
    vectors: VectorPlan = VectorPlan()
    params: dict[str, Any] = {}  # parametri passati al kernel (es. formato, k)
    wrapper: Literal["cvxif"] | None = None  # M7: wrapper coprocessore generato dopo il pass

    # riempiti dal loader
    root: Path | None = Field(None, exclude=True)
    spec_text: str = Field("", exclude=True)

    @model_validator(mode="after")
    def _ports(self) -> "Task":
        names = [p.name for p in self.ports]
        if len(set(names)) != len(names):
            raise ValueError("nomi di porta duplicati")
        if {"clock", "reset"} & set(names):
            raise ValueError("clock e reset sono impliciti (Module), non vanno dichiarati")
        if not any(p.dir == "out" for p in self.ports):
            raise ValueError("serve almeno una porta di uscita")
        return self

    @property
    def inputs(self) -> list[Port]:
        return [p for p in self.ports if p.dir == "in"]

    @property
    def outputs(self) -> list[Port]:
        return [p for p in self.ports if p.dir == "out"]

    def port(self, name: str) -> Port:
        return next(p for p in self.ports if p.name == name)

    @property
    def numerics_obj(self) -> Numerics:
        return Numerics.from_dict(self.numerics)

    def interface_text(self) -> str:
        kind = ("purely combinational (use RawModule or ignore clock/reset)"
                if self.timing.kind == "comb" else
                f"pipelined Module with fixed latency {self.timing.latency} cycle(s): the output "
                f"for an input applied in cycle t must appear in cycle t+{self.timing.latency}; a new "
                "input is accepted every cycle")
        lines = [f"Top module: class {self.top}, {kind}.",
                 "Ports (declare them with FlatIO so Verilog names match exactly):"]
        lines += [f"  - {p.describe()}" for p in self.ports]
        return "\n".join(lines)


def load_task(path: str | Path) -> Task:
    root = Path(path)
    if root.is_file():
        root = root.parent
    data = yaml.safe_load((root / "task.yaml").read_text(encoding="utf-8"))
    t = Task.model_validate(data)
    t.root = root
    spec = root / "spec.md"
    t.spec_text = spec.read_text(encoding="utf-8") if spec.is_file() else ""
    return t


def reference_sources(task: Task) -> list[Path]:
    assert task.root is not None
    return sorted((task.root / "reference").glob("*.scala"))
