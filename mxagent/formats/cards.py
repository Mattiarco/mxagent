"""Schede testuali dei formati, iniettate nei prompt (la conoscenza di dominio
non è lasciata all'LLM). Il testo è generato dalle costanti verificate di
:mod:`specs`, così prompt e golden model non possono divergere."""
from __future__ import annotations

from . import fp
from .numerics import DEFAULT, Numerics
from .packing import CONVENTION
from .specs import E2M1, E4M3, BlockFormat, MXFP4


def e2m1_table() -> str:
    rows = []
    for c in range(16):
        s, e, m = E2M1.fields(c)
        v = fp.decode(E2M1, c).to_float()
        rows.append(f"  {c:04b} (s={s} e={e:02b} m={m}) = {v:+g}")
    return "\n".join(rows)


def format_card(fmt: BlockFormat = MXFP4, num: Numerics = DEFAULT) -> str:
    if fmt.scale == "e8m0":
        scale = ("Shared scale E8M0: 8-bit UNSIGNED exponent, value = 2^(X-127) for X in 0..254 "
                 "(2^-127 .. 2^127); X = 0xFF is NaN; no zero, no Inf, no sign. "
                 "If a block scale is NaN every value of the block is NaN (OCP MX v1.0 §5.1).")
    else:
        mx = fp.decode(E4M3, 0x7E).to_float()
        scale = (f"Block scale FP8 E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa; subnormals when "
                 f"exponent field is 0; S.1111.111 is NaN; no Inf; max ±{mx:g}. Plus an optional FP32 "
                 "per-tensor scale.")
    return "\n".join([
        f"FORMAT {fmt.name.upper()}: blocks of k={fmt.k} elements, each FP4 E2M1, sharing one scale.",
        "Element FP4 E2M1 (OCP MX v1.0 §5.3.3): bit3 = sign, bits[2:1] = exponent (bias 1), "
        "bit0 = mantissa. exponent==0 -> subnormal: value = (-1)^s * 0.5 * m; otherwise "
        "(-1)^s * 2^(e-1) * (1 + m/2). No Inf, no NaN. Magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6. "
        "Code 1000 is -0 and must behave as zero.",
        "All 16 codes:", e2m1_table(),
        "Every E2M1 value is a multiple of 0.5 in [-6, 6]: in units of 1/2 it is an integer in "
        "[-12, 12] (SInt(5)). Every product of two E2M1 values is a multiple of 0.25 in [-36, 36]: "
        "in units of 1/4 it is an integer in [-144, 144] (SInt(9)). A sum of 32 such products is "
        "in [-4608, 4608] units of 1/4 (SInt(14)).",
        scale,
        CONVENTION,
        "Numeric semantics required (bit-exact):",
        num.describe(),
    ])


assert E2M1.width == 4  # la scheda presuppone nibble
