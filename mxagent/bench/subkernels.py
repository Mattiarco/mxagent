"""Catalogo dei blocchi MX per la decomposizione (Planner).

Il Planner non inventa sottomoduli liberi: sceglie blocchi da questo catalogo,
ciascuno con interfaccia fissata, golden e vettori di test. Così ogni
sottomodulo generato dall'LLM è verificato da solo (esaustivamente quando lo
spazio è piccolo) prima dell'integrazione. La logica di collegamento ("glue")
vive nel top-level e viene verificata dal test del task.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from ..formats import fp
from ..formats import golden as g
from ..formats.numerics import Numerics
from ..formats.packing import pack, to_signed, to_unsigned, unpack
from ..formats.specs import E2M1, FP32
from .kernels import KERNELS, Kernel, Vec


@dataclass(frozen=True)
class Block:
    key: str
    title: str
    doc: str  # descrizione funzionale (va nel prompt del Planner e del Coder)
    ports: Callable[[dict[str, Any]], list[dict[str, Any]]]
    kernel: Kernel
    defaults: dict[str, Any] = field(default_factory=dict)
    exhaustive: Callable[[dict[str, Any]], bool] = lambda p: False

    def params(self, p: dict[str, Any] | None) -> dict[str, Any]:
        out = {**self.defaults, **(p or {})}
        unknown = set(out) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.key}: parametri sconosciuti {sorted(unknown)}")
        return out


def _clog2(n: int) -> int:
    return max(1, math.ceil(math.log2(n))) if n > 1 else 1


# --------------------------------------------------------------- e2m1_mul
def _mul(v: Vec, num: Numerics, p: dict) -> Vec:
    return {"p": to_unsigned(g.e2m1_mul_fixed(v["a"], v["b"]), 9)}


# --------------------------------------------------------------- e2m1_decode
def _dec(v: Vec, num: Numerics, p: dict) -> Vec:
    return {"y": to_unsigned(g.e2m1_to_fixed(v["a"], 1), 5)}


# ------------------------------------------------------------ adder_tree
def _tree_w(p: dict) -> int:
    return p["w"] + _clog2(p["n"])


def _tree(v: Vec, num: Numerics, p: dict) -> Vec:
    s = sum(to_signed(x, p["w"]) for x in unpack(v["x"], p["w"], p["n"]))
    return {"y": to_unsigned(s, _tree_w(p))}


def _tree_rand(rng: random.Random, p: dict) -> Vec:
    lo, hi = -(1 << (p["w"] - 1)), (1 << (p["w"] - 1)) - 1
    mode = rng.random()
    if mode < 0.2:
        xs = [rng.choice([lo, hi]) for _ in range(p["n"])]
    elif mode < 0.4:
        xs = [rng.randint(-3, 3) for _ in range(p["n"])]
    else:
        xs = [rng.randint(lo, hi) for _ in range(p["n"])]
    return {"x": pack([to_unsigned(x, p["w"]) for x in xs], p["w"])}


def _tree_corner(p: dict) -> list[Vec]:
    lo, hi, n, w = -(1 << (p["w"] - 1)), (1 << (p["w"] - 1)) - 1, p["n"], p["w"]
    cases = [[hi] * n, [lo] * n, [0] * n, [hi, lo] * (n // 2) + [hi] * (n % 2),
             [-1] * n, [1] + [0] * (n - 1), [0] * (n - 1) + [lo]]
    return [{"x": pack([to_unsigned(x, w) for x in c], w)} for c in cases]


# ------------------------------------------------------------- e8m0_pair
def _pair(v: Vec, num: Numerics, p: dict) -> Vec:
    nan = int(v["a"] == 0xFF or v["b"] == 0xFF)
    e = 0 if nan else (v["a"] - 127) + (v["b"] - 127)
    return {"exp": to_unsigned(e, 10), "nan": nan}


# ---------------------------------------------------------- fixed_to_fp32
def _f2f(v: Vec, num: Numerics, p: dict) -> Vec:
    if v["nan"]:
        return {"y": FP32.canonical_nan}
    x = to_signed(v["x"], p["w"])
    e = to_signed(v["exp"], 10)
    val = fp.Value.of(x, e - p["frac"])
    if val.kind is fp.Kind.ZERO:
        return {"y": 0}
    return {"y": g.out_encode(val, num)}


def _f2f_rand(rng: random.Random, p: dict) -> Vec:
    w = p["w"]
    lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
    r = rng.random()
    x = rng.randint(lo, hi) if r < 0.7 else rng.choice([lo, hi, 1, -1, 0, 3, -3])
    e = rng.randint(-40, 40) if rng.random() < 0.5 else rng.randint(-300, 300)
    return {"x": to_unsigned(x, w), "exp": to_unsigned(e, 10), "nan": int(rng.random() < 0.03)}


def _f2f_corner(p: dict) -> list[Vec]:
    w, f = p["w"], p["frac"]
    out = []
    xs = [1, -1, 3, 5, 7, (1 << (w - 1)) - 1, -(1 << (w - 1)), 0]
    # bordi: overflow (2^128), normale massimo, minimo normale, subnormali e pareggi
    for x in xs:
        for e in (130 + f, 127 + f, 126 + f, -126 + f, -127 + f, -140 + f, -149 + f, -150 + f,
                  -151 + f, -160 + f, 0, -254, 254):
            out.append({"x": to_unsigned(x, w), "exp": to_unsigned(e, 10), "nan": 0})
    out.append({"x": 5, "exp": 0, "nan": 1})
    return out


# ---------------------------------------------------------------- fp32_exp
def _fexp(v: Vec, num: Numerics, p: dict) -> Vec:
    val = fp.decode(FP32, v["x"])
    special = int(val.kind in (fp.Kind.NAN, fp.Kind.INF))
    zero = int(val.kind is fp.Kind.ZERO)
    e = 0 if (special or zero) else val.mant.bit_length() - 1 + val.exp
    return {"e": to_unsigned(e, 9), "zero": zero, "special": special}


def _fexp_rand(rng: random.Random, p: dict) -> Vec:
    from .kernels_mx import _rand_f32
    return {"x": _rand_f32(rng)}


def _fexp_corner(p: dict) -> list[Vec]:
    return [{"x": x} for x in (0, 0x80000000, 1, 0x007FFFFF, 0x00800000, 0x3F800000, 0x7F7FFFFF,
                               0x7F800000, 0xFF800000, 0x7FC00000, 0x00400000, 0x80000001)]


# ---------------------------------------------------------------- max_tree
def _max(v: Vec, num: Numerics, p: dict) -> Vec:
    es = [to_signed(x, p["w"]) for x in unpack(v["e"], p["w"], p["n"])]
    val = [e for i, e in enumerate(es) if v["valid"] >> i & 1]
    return {"m": to_unsigned(max(val), p["w"]) if val else 0, "any": int(bool(val))}


def _max_rand(rng: random.Random, p: dict) -> Vec:
    lo, hi = -(1 << (p["w"] - 1)), (1 << (p["w"] - 1)) - 1
    es = [rng.randint(lo, hi) if rng.random() < 0.5 else rng.randint(-10, 10) for _ in range(p["n"])]
    valid = rng.getrandbits(p["n"]) if rng.random() < 0.8 else rng.choice([0, 1, 1 << (p["n"] - 1)])
    return {"e": pack([to_unsigned(e, p["w"]) for e in es], p["w"]), "valid": valid}


# --------------------------------------------------------- e2m1_quant_elem
def _qel(v: Vec, num: Numerics, p: dict) -> Vec:
    val = fp.decode(FP32, v["x"])
    if val.kind in (fp.Kind.NAN, fp.Kind.INF):
        return {"q": 0}
    sh = to_signed(v["shared"], 9)
    return {"q": fp.encode(E2M1, fp.scale_pow2(val, -sh), rounding=num.quant_rounding,
                           overflow=fp.Overflow.SATURATE)}


def _qel_rand(rng: random.Random, p: dict) -> Vec:
    sh = rng.randint(-127, 125) if rng.random() < 0.3 else rng.randint(-20, 20)
    r = rng.random()
    if r < 0.6:  # valore vicino alla scala: è il caso d'uso reale (|x| < 8 * 2^shared)
        x = math.ldexp(rng.uniform(-8, 8), sh)
    elif r < 0.8:
        grid = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7, 0.5, 1, 1.5, 2, 3, 4, 6]
        x = math.ldexp(rng.choice(grid) * rng.choice([1, -1]), sh)
    else:
        x = math.ldexp(rng.uniform(-1, 1), sh - rng.randint(1, 30))
    from .kernels_mx import _f32_round
    return {"x": _f32_round(x), "shared": to_unsigned(sh, 9)}


# ======================================================================
CATALOG: dict[str, Block] = {}


def _add(b: Block) -> None:
    CATALOG[b.key] = b
    KERNELS[f"sub:{b.key}"] = b.kernel


_add(Block(
    "e2m1_decode", "E2M1 decode",
    "Decodes one FP4 E2M1 code into its exact value as a signed integer in units of 1/2 "
    "(y = value*2, range [-12, 12]). Both zero codes give 0.",
    lambda p: [{"name": "a", "dir": "in", "type": "e2m1"},
               {"name": "y", "dir": "out", "type": "sint", "width": 5, "frac_bits": 1}],
    Kernel(_dec, space=lambda p: ({"a": a} for a in range(16))), exhaustive=lambda p: True))

_add(Block(
    "e2m1_mul", "E2M1 x E2M1 exact product",
    "Exact product of two E2M1 codes as a signed integer in units of 1/4 (p = a*b*4, range "
    "[-144, 144]). Any zero operand (+0 or -0) gives 0.",
    lambda p: [{"name": "a", "dir": "in", "type": "e2m1"}, {"name": "b", "dir": "in", "type": "e2m1"},
               {"name": "p", "dir": "out", "type": "sint", "width": 9, "frac_bits": 2}],
    Kernel(_mul, space=lambda p: ({"a": a, "b": b} for a, b in product(range(16), repeat=2))),
    exhaustive=lambda p: True))

_add(Block(
    "adder_tree", "Signed adder tree",
    "Exact sum of n signed w-bit integers packed in x (element i at bits [w*i+w-1 : w*i]). "
    "The output y is wide enough to never overflow: w + ceil(log2 n) bits.",
    lambda p: [{"name": "x", "dir": "in", "type": "sint", "width": p["w"], "elems": p["n"]},
               {"name": "y", "dir": "out", "type": "sint", "width": _tree_w(p)}],
    Kernel(_tree, corner=_tree_corner, random=_tree_rand), defaults={"n": 32, "w": 9}))

_add(Block(
    "e8m0_pair", "Combine two E8M0 scales",
    "Given two E8M0 scale codes a and b, outputs exp = (a-127) + (b-127) as a signed 10-bit "
    "integer (the exponent of the product of the two scales) and nan = 1 if a or b is 0xFF "
    "(then exp must be 0).",
    lambda p: [{"name": "a", "dir": "in", "type": "e8m0"}, {"name": "b", "dir": "in", "type": "e8m0"},
               {"name": "exp", "dir": "out", "type": "sint", "width": 10},
               {"name": "nan", "dir": "out", "type": "bool"}],
    Kernel(_pair, space=lambda p: ({"a": a, "b": b} for a, b in product(range(256), repeat=2))),
    exhaustive=lambda p: True))

_add(Block(
    "fixed_to_fp32", "Signed fixed-point x 2^exp to FP32 (RNE)",
    "Converts the exact value x * 2^(exp - frac) to IEEE binary32 with ONE round-to-nearest-even "
    "rounding. x is a signed w-bit integer, exp a signed 10-bit exponent offset, frac a constant. "
    "Results below 2^-126 must become binary32 SUBNORMALS (correctly rounded, not flushed); "
    "results that round to >= 2^128 become +/-Inf. An exact zero gives +0 (0x00000000). If nan=1 "
    "the output is the canonical NaN 0x7FC00000.",
    lambda p: [{"name": "x", "dir": "in", "type": "sint", "width": p["w"], "frac_bits": p["frac"]},
               {"name": "exp", "dir": "in", "type": "sint", "width": 10},
               {"name": "nan", "dir": "in", "type": "bool"},
               {"name": "y", "dir": "out", "type": "fp32"}],
    Kernel(_f2f, corner=_f2f_corner, random=_f2f_rand), defaults={"w": 14, "frac": 2}))

_add(Block(
    "fp32_exp", "FP32 exponent extraction",
    "For a binary32 input x outputs e = floor(log2(|x|)) as a signed 9-bit integer (for "
    "subnormals this depends on the leading one of the mantissa: e = msb(mantissa) - 149), "
    "zero = 1 for +/-0 and special = 1 for Inf/NaN. When zero or special is 1, e must be 0.",
    lambda p: [{"name": "x", "dir": "in", "type": "fp32"},
               {"name": "e", "dir": "out", "type": "sint", "width": 9},
               {"name": "zero", "dir": "out", "type": "bool"},
               {"name": "special", "dir": "out", "type": "bool"}],
    Kernel(_fexp, corner=_fexp_corner, random=_fexp_rand)))

_add(Block(
    "max_tree", "Masked signed maximum",
    "Maximum of the n signed w-bit integers in e whose bit in the mask `valid` is 1 "
    "(element i at bits [w*i+w-1 : w*i], valid bit i). any = 1 if at least one valid bit is set; "
    "if none is valid, m must be 0.",
    lambda p: [{"name": "e", "dir": "in", "type": "sint", "width": p["w"], "elems": p["n"]},
               {"name": "valid", "dir": "in", "type": "uint", "width": p["n"]},
               {"name": "m", "dir": "out", "type": "sint", "width": p["w"]},
               {"name": "any", "dir": "out", "type": "bool"}],
    Kernel(_max, random=_max_rand), defaults={"n": 32, "w": 9}))

_add(Block(
    "e2m1_quant_elem", "Quantize one FP32 to E2M1 given the shared exponent",
    "Computes the E2M1 code of x / 2^shared with round-to-nearest-even and saturation to +/-6.0 "
    "(sign preserved, also for results that round to zero: -tiny -> 0b1000). shared is a signed "
    "9-bit exponent in [-127, 125]. Ties go to the code with even mantissa bit. x is finite "
    "(NaN/Inf never reach this block; output 0 for them).",
    lambda p: [{"name": "x", "dir": "in", "type": "fp32"},
               {"name": "shared", "dir": "in", "type": "sint", "width": 9},
               {"name": "q", "dir": "out", "type": "e2m1"}],
    Kernel(_qel, random=_qel_rand)))


def block_task(key: str, top: str, params: dict[str, Any] | None = None,
               numerics: dict[str, Any] | None = None):
    """Task verificabile per un blocco del catalogo con nome di modulo ``top``."""
    from .task import Task

    b = CATALOG[key]
    p = b.params(params)
    t = Task.model_validate({
        "name": f"sub_{key}_{top}", "title": b.title, "difficulty": 2, "top": top,
        "ports": b.ports(p), "golden": f"sub:{key}", "params": p,
        "numerics": numerics or {},
        "vectors": {"exhaustive": b.exhaustive(p), "corner": True,
                    "random": 0 if b.exhaustive(p) else 3000},
    })
    t.spec_text = f"{b.title}.\n{b.doc}" + (f"\nParameters: {p}." if p else "")
    return t


def catalog_text() -> str:
    lines = []
    for b in CATALOG.values():
        d = f" (params: {b.defaults})" if b.defaults else ""
        lines.append(f"- {b.key}{d}: {b.doc}")
    return "\n".join(lines)
