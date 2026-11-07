"""Kernel dei benchmark a blocchi: b04 (quantizzatore) e b05 (dot product).

I generatori casuali mescolano casi "realistici" (scale vicine a 1, tensori con
range dinamico ristretto) e casi estremi (scale ai bordi, subnormali, NaN/Inf,
cancellazioni), perché i bug degli LLM stanno quasi sempre lì.
"""
from __future__ import annotations

import math
import random
import struct
from typing import Any

from ..formats import golden as g
from ..formats.golden import MXBlock
from ..formats.numerics import Numerics
from ..formats.packing import pack, unpack
from ..formats.specs import MXFP4
from .kernels import KERNELS, Kernel, Vec

K = MXFP4.k


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


# ======================================================================
#                         b05: dot di due blocchi
# ======================================================================
def _dot(v: Vec, num: Numerics, p: dict[str, Any]) -> Vec:
    a = MXBlock(v["a_scale"], unpack(v["a"], 4, K))
    b = MXBlock(v["b_scale"], unpack(v["b"], 4, K))
    return {"y": g.mx_dot(a, b, num)}


def _rand_scale(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.70:
        return rng.randint(110, 144)
    if r < 0.85:
        return rng.randint(0, 254)
    if r < 0.95:
        return rng.choice([rng.randint(0, 6), rng.randint(248, 254)])
    return 0xFF


def _rand_elems(rng: random.Random) -> list[int]:
    density = rng.choice([1.0, 1.0, 0.5, 0.1])
    return [rng.randrange(16) if rng.random() < density else rng.choice([0, 8]) for _ in range(K)]


def _dot_random(rng: random.Random, p: dict[str, Any]) -> Vec:
    return {"a_scale": _rand_scale(rng), "a": pack(_rand_elems(rng), 4),
            "b_scale": _rand_scale(rng), "b": pack(_rand_elems(rng), 4)}


def _dot_vec(sa: int, ea: list[int], sb: int, eb: list[int]) -> Vec:
    return {"a_scale": sa, "a": pack(ea, 4), "b_scale": sb, "b": pack(eb, 4)}


def _one(i: int, code: int) -> list[int]:
    return [code if j == i else 0 for j in range(K)]


def _dot_corner(p: dict[str, Any]) -> list[Vec]:
    ones, six, nsix = [0x2] * K, [0x7] * K, [0xF] * K
    c = [
        _dot_vec(127, ones, 127, ones),             # 32.0
        _dot_vec(127, six, 127, six),               # 1152, massimo
        _dot_vec(127, six, 127, nsix),              # -1152
        _dot_vec(0xFF, ones, 127, ones),            # NaN
        _dot_vec(127, ones, 0xFF, ones),
        _dot_vec(0xFF, [0] * K, 0xFF, [0] * K),
        _dot_vec(254, six, 254, six),               # overflow -> Inf
        _dot_vec(254, nsix, 254, six),              # -Inf
        _dot_vec(254, _one(0, 2), 128, _one(0, 2)),  # 2^128 -> Inf (bordo)
        _dot_vec(254, _one(0, 7), 127, _one(0, 7)),  # 36 * 2^127 -> Inf
        _dot_vec(254, _one(0, 2), 127, _one(0, 2)),  # 2^127 normale
        _dot_vec(0, _one(0, 1), 0, _one(0, 1)),     # 0.25 * 2^-254 -> 0
        _dot_vec(0, ones, 127, ones),               # 32 * 2^-127 = 2^-122 normale
        _dot_vec(0, _one(0, 2), 127, _one(0, 2)),   # 2^-127 subnormale
        _dot_vec(0, _one(0, 2), 104, _one(0, 2)),   # 2^-150: pareggio -> 0
        _dot_vec(0, _one(0, 3), 104, _one(0, 2)),   # 1.5*2^-150 -> min subnormale
        _dot_vec(0, _one(0, 7), 105, _one(0, 7)),   # 36*2^-149 -> subnormale esatto
        _dot_vec(1, _one(0, 1), 1, _one(0, 1)),     # 2^-254 -> 0
        _dot_vec(127, [2, 0xA] + [0] * 30, 127, [2, 2] + [0] * 30),  # cancellazione -> +0
        _dot_vec(127, [8] * K, 127, ones),          # -0 * 1 -> +0
        _dot_vec(127, [0] * K, 127, [0] * K),
        _dot_vec(127, _one(31, 1), 127, _one(31, 1)),  # ultimo elemento (ordine dei bit)
        _dot_vec(127, _one(0, 1), 127, _one(0, 1)),    # primo elemento
    ]
    for i in (1, 7, 16, 30):
        c.append(_dot_vec(130, _one(i, 0x5), 120, _one(i, 0xD)))
    # sottonormali con riporto nell'arrotondamento verso il minimo normale
    for sa in range(0, 8):
        c.append(_dot_vec(sa, six, 0, six))
    return c


KERNELS["mx_dot"] = Kernel(_dot, corner=_dot_corner, random=_dot_random)


# ======================================================================
#                     b04: quantizzatore FP32 -> MXFP4
# ======================================================================
def _quant(v: Vec, num: Numerics, p: dict[str, Any]) -> Vec:
    q = g.quantize_mx(list(unpack(v["x"], 32, K)), MXFP4, num)
    return {"scale": q.scale, "q": pack(q.elems, 4)}


def _f32_round(x: float) -> int:
    try:
        return f32(x)
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def _rand_f32(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.05:
        return rng.choice([0, 0x80000000])
    if r < 0.12:
        return rng.randint(1, (1 << 23) - 1) | (rng.getrandbits(1) << 31)  # subnormale
    if r < 0.14:
        return rng.choice([0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001])
    return rng.getrandbits(32) & ~(0xFF << 23) | (rng.randint(1, 254) << 23)


def _quant_random(rng: random.Random, p: dict[str, Any]) -> Vec:
    mode = rng.random()
    if mode < 0.6:  # tensore realistico: range dinamico ristretto
        base = rng.randint(-125, 120)
        xs = []
        for _ in range(K):
            m = rng.uniform(-8, 8) if rng.random() < 0.9 else 0.0
            xs.append(_f32_round(math.ldexp(m, base)))
    elif mode < 0.75:  # valori sulla griglia E2M1 e sui pareggi
        X = rng.randint(-100, 100)
        grid = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7]
        xs = [_f32_round(math.ldexp(rng.choice(grid) * rng.choice([1, -1]), X)) for _ in range(K)]
        xs[rng.randrange(K)] = _f32_round(math.ldexp(rng.choice([4.0, 6.0, 7.0, 7.9]), X))
    elif mode < 0.97:
        xs = [_rand_f32(rng) for _ in range(K)]
    else:  # NaN/Inf in un blocco altrimenti normale
        xs = [_f32_round(rng.uniform(-4, 4)) for _ in range(K)]
        xs[rng.randrange(K)] = rng.choice([0x7F800000, 0xFF800000, 0x7FC00000])
    return {"x": pack(xs, 32)}


def _quant_corner(p: dict[str, Any]) -> list[Vec]:
    z = [0] * K
    vs = [
        z,
        [0x80000000] + z[1:],
        [0x7F7FFFFF] + z[1:],                                   # max FP32
        [f32(7.0), f32(-7.0)] + z[2:],                          # saturazione
        [f32(x) for x in (4.0, 2.5, 0.25, 0.75, 5.0, -0.25, 3.5, 1.25)] + z[8:],  # pareggi
        [f32(x) for x in (6.0, 5.0, 4.5, 3.0, 2.0, 1.75, 0.5, 0.125)] + z[8:],
        [1] + z[1:],                                            # min subnormale -> scala clampata
        [0x00400000, 0x00000001, 0x00200000] + z[3:],           # solo subnormali
        [f32(1e-40), f32(1.0)] + z[2:],                         # sticky su valori minuscoli
        [0x7FC00000] + [f32(1.0)] * (K - 1),
        [0xFF800000] + z[1:],
        [f32(2.0 ** -125)] * K,                                 # shared = -127 non clampato
        [f32(2.0 ** -126)] + z[1:],
        [f32(2.0 ** 127 * 1.5)] + [f32(1.0)] * (K - 1),
        [f32(float(i) - 16) for i in range(K)],                 # rampa
        [f32(x * 2 ** 20) for x in (1.0, 1.4, 1.5, 1.6, 2.9, 3.0, 3.1)] + z[7:],
    ]
    return [{"x": pack(x, 32)} for x in vs]


KERNELS["mx_quantize"] = Kernel(_quant, corner=_quant_corner, random=_quant_random)
