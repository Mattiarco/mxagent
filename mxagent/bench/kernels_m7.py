"""Kernel dei benchmark di M7.

b06  dot MXFP4 in pipeline: stesso kernel di b05 (``mx_dot``), cambia solo il
     contratto temporale del task.
b07  dot NVFP4: 16 x E2M1 per blocco, scala E4M3 ([NVIDIA] e arXiv 2509.25149
     Eq. 5: il prodotto di blocco usa le due scale E4M3; le scale per tensore
     si applicano dopo il GEMM e non fanno parte di questa unità).
b08  MAC MXFP4: y = acc_in + Dot(A, B) con un solo arrotondamento
     ([OCP-MX] §6.2, DotGeneral con un blocco).
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
from ..formats.specs import MXFP4, NVFP4
from .kernels import KERNELS, Kernel, Vec
from .kernels_mx import _dot_corner, _dot_random, _rand_f32, f32

KN = NVFP4.k  # 16


# ======================================================================
#                          b07: dot NVFP4
# ======================================================================
def _nv_dot(v: Vec, num: Numerics, p: dict[str, Any]) -> Vec:
    a = MXBlock(v["a_scale"], unpack(v["a"], 4, KN))
    b = MXBlock(v["b_scale"], unpack(v["b"], 4, KN))
    return {"y": g.nvfp4_dot(a, b, num)}


def _rand_e4m3_scale(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.60:  # positive normali intorno a 1 (exp 4..10 => 2^-3..2^3)
        return (rng.randint(4, 10) << 3) | rng.randrange(8)
    if r < 0.80:  # qualunque positivo finito (subnormali compresi)
        return rng.choice([c for c in range(0x7F)])
    if r < 0.92:  # negativo: il formato lo ammette, l'unità deve rispettarne il segno
        return 0x80 | rng.randrange(0x7F)
    return rng.choice([0x7F, 0xFF, 0x00, 0x80, 0x7E, 0x01])  # NaN, zeri, max, min sub


def _nv_elems(rng: random.Random) -> list[int]:
    density = rng.choice([1.0, 1.0, 0.5, 0.1])
    return [rng.randrange(16) if rng.random() < density else rng.choice([0, 8]) for _ in range(KN)]


def _nv_random(rng: random.Random, p: dict[str, Any]) -> Vec:
    return {"a_scale": _rand_e4m3_scale(rng), "a": pack(_nv_elems(rng), 4),
            "b_scale": _rand_e4m3_scale(rng), "b": pack(_nv_elems(rng), 4)}


def _nv(sa: int, ea: list[int], sb: int, eb: list[int]) -> Vec:
    return {"a_scale": sa, "a": pack(ea, 4), "b_scale": sb, "b": pack(eb, 4)}


def _one(i: int, code: int, k: int = KN) -> list[int]:
    return [code if j == i else 0 for j in range(k)]


ONE_E4M3 = 0x38  # 1.0 = 2^(7-7)


def _nv_corner(p: dict[str, Any]) -> list[Vec]:
    ones, six, nsix = [0x2] * KN, [0x7] * KN, [0xF] * KN
    c = [
        _nv(ONE_E4M3, ones, ONE_E4M3, ones),          # 16.0
        _nv(ONE_E4M3, six, ONE_E4M3, six),            # 576
        _nv(ONE_E4M3, six, ONE_E4M3, nsix),           # -576
        _nv(0x7E, six, 0x7E, six),                    # 448^2 * 576 (finito in FP32)
        _nv(0x7F, ones, ONE_E4M3, ones),              # NaN (S.1111.111)
        _nv(ONE_E4M3, ones, 0xFF, ones),              # NaN negativo
        _nv(0x7F, [0] * KN, ONE_E4M3, [0] * KN),      # NaN anche con elementi nulli
        _nv(0x00, six, ONE_E4M3, six),                # scala zero -> +0
        _nv(0x80, six, ONE_E4M3, six),                # scala -0 -> +0 (zero esatto)
        _nv(0xB8, ones, ONE_E4M3, ones),              # scala -1 -> -16
        _nv(0xB8, ones, 0xB8, ones),                  # (-1)(-1) -> +16
        _nv(0x01, _one(0, 1), 0x01, _one(0, 1)),      # 2^-9 * 2^-9 * 0.25 = 2^-20
        _nv(0x01, ones, 0x01, ones),                  # sottonormale E4M3 su entrambi
        _nv(0x07, six, 0x08, six),                    # max sub x min normale
        _nv(ONE_E4M3, [2, 0xA] + [0] * 14, ONE_E4M3, [2, 2] + [0] * 14),  # cancellazione -> +0
        _nv(ONE_E4M3, [8] * KN, ONE_E4M3, ones),     # -0 * 1 -> +0
        _nv(ONE_E4M3, _one(15, 1), ONE_E4M3, _one(15, 1)),  # ultimo elemento
        _nv(ONE_E4M3, _one(0, 1), ONE_E4M3, _one(0, 1)),    # primo elemento
        _nv(0x3B, [1, 3, 5, 7, 9, 11, 13, 15] * 2, 0x45, [2, 4, 6, 7, 10, 12, 14, 15] * 2),
    ]
    for i in (1, 7, 8, 14):
        c.append(_nv(0x40, _one(i, 0x5), 0x30, _one(i, 0xD)))
    # mantisse della scala: 1.001 .. 1.111 (prodotto di scale non potenza di 2)
    for m in range(1, 8):
        c.append(_nv(0x38 | m, [3] * KN, 0x38 | (8 - m if m < 8 else 1), [5] * KN))
    return c


KERNELS["nvfp4_dot"] = Kernel(_nv_dot, corner=_nv_corner, random=_nv_random)


# ======================================================================
#                          b08: MAC MXFP4
# ======================================================================
K = MXFP4.k


def _mac(v: Vec, num: Numerics, p: dict[str, Any]) -> Vec:
    a = MXBlock(v["a_scale"], unpack(v["a"], 4, K))
    b = MXBlock(v["b_scale"], unpack(v["b"], 4, K))
    return {"y": g.mx_mac([a], [b], num, acc_in=v["acc"])}


def _mac_random(rng: random.Random, p: dict[str, Any]) -> Vec:
    v = _dot_random(rng, p)
    r = rng.random()
    if r < 0.6:  # accumulatore vicino al dot: cancellazioni e arrotondamenti interessanti
        bits = g.mx_dot(MXBlock(v["a_scale"], unpack(v["a"], 4, K)),
                        MXBlock(v["b_scale"], unpack(v["b"], 4, K)))
        dot = struct.unpack("<f", struct.pack("<I", bits))[0]
        if math.isfinite(dot) and dot != 0.0:
            x = dot * rng.choice([-1.0, -1.0, 1.0]) * 2.0 ** rng.randint(-26, 26) * rng.uniform(0.5, 2)
            v["acc"] = f32(x) if math.isfinite(x) and abs(x) < 3.4e38 else _rand_f32(rng)
        else:
            v["acc"] = _rand_f32(rng)
    else:
        v["acc"] = _rand_f32(rng)
    return v


def _mac_corner(p: dict[str, Any]) -> list[Vec]:
    base = _dot_corner(p)
    accs = [f32(0.0), 0x80000000, f32(1.0), f32(-32.0), 0x7F800000, 0xFF800000, 0x7FC00000,
            0x00000001, 0x7F7FFFFF, 0xFF7FFFFF]
    out = []
    for i, v in enumerate(base):  # ogni corner del dot con un accumulatore diverso
        out.append({**v, "acc": accs[i % len(accs)]})
    ones = pack([0x2] * K, 4)
    out += [
        {"a_scale": 127, "a": ones, "b_scale": 127, "b": ones, "acc": f32(-32.0)},  # esatto -> +0
        {"a_scale": 127, "a": ones, "b_scale": 127, "b": ones, "acc": 0x80000000},  # -0 + 32
        {"a_scale": 127, "a": pack([0] * K, 4), "b_scale": 127, "b": ones, "acc": 0x80000000},  # -0 + 0 -> +0
        {"a_scale": 127, "a": pack([0] * K, 4), "b_scale": 127, "b": ones, "acc": f32(-1.5)},
        {"a_scale": 254, "a": pack([7] * K, 4), "b_scale": 254, "b": pack([7] * K, 4),
         "acc": 0xFF800000},  # -Inf + dot finito enorme -> -Inf
        {"a_scale": 127, "a": ones, "b_scale": 127, "b": ones, "acc": f32(2.0 ** 24)},  # assorbimento + RNE
        {"a_scale": 127, "a": pack([1] + [0] * 31, 4), "b_scale": 127, "b": pack([1] + [0] * 31, 4),
         "acc": f32(2.0 ** 23)},  # 2^23 + 0.25: sotto mezzo ulp -> 2^23
        {"a_scale": 127, "a": pack([3] + [0] * 31, 4), "b_scale": 126, "b": pack([2] + [0] * 31, 4),
         "acc": f32(2.0 ** 24 + 2)},  # 2^24+2 + 0.75: sotto mezzo ulp (2) -> 2^24+2
        {"a_scale": 127, "a": pack([2] + [0] * 31, 4), "b_scale": 127, "b": pack([2] + [0] * 31, 4),
         "acc": f32(2.0 ** 24)},  # 2^24 + 1: pareggio esatto -> pari (2^24)
        {"a_scale": 127, "a": pack([2] + [0] * 31, 4), "b_scale": 127, "b": pack([2] + [0] * 31, 4),
         "acc": f32(2.0 ** 24 + 2)},  # 2^24+2 + 1: pareggio -> pari (2^24+4)
        {"a_scale": 0, "a": pack([2] + [0] * 31, 4), "b_scale": 127, "b": pack([2] + [0] * 31, 4),
         "acc": 0x00400000},  # sottonormali: 2^-127 + 2^-127
    ]
    return out


KERNELS["mx_mac"] = Kernel(_mac, corner=_mac_corner, random=_mac_random)
