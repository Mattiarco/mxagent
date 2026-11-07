"""Motore di arrotondamento contro oracoli indipendenti (numpy float32/float16)."""
import math
import random
import struct
import warnings

import numpy as np
import pytest

from mxagent.formats import fp
from mxagent.formats.fp import Overflow, Rounding, Subnormals, Value
from mxagent.formats.specs import E2M1, E4M3, FP16, FP32


def f32_bits_np(x: float) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return int(np.array([x], dtype=np.float64).astype(np.float32).view(np.uint32)[0])


def f16_bits_np(x: float) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return int(np.array([x], dtype=np.float64).astype(np.float16).view(np.uint16)[0])


def value_of_double(x: float) -> Value:
    if x == 0:
        return Value.zero(1 if math.copysign(1, x) < 0 else 0)
    m, e = math.frexp(x)
    n = int(m * (1 << 53))
    return Value.of(n, e - 53)


def random_doubles(rng, n, emin, emax):
    out = []
    for _ in range(n):
        bits = rng.choice([1, 2, 3, 8, 24, 25, 26, 30, 53])  # ties e casi corti
        mant = rng.getrandbits(bits) | (1 << (bits - 1))
        x = math.ldexp(mant, rng.randint(emin, emax) - bits)
        out.append(-x if rng.random() < 0.5 else x)
    return out


def test_fp32_vs_numpy_random():
    rng = random.Random(1234)
    xs = random_doubles(rng, 40000, -160, 132)  # subnormali, underflow, overflow
    xs += [0.0, -0.0, 2.0 ** -149, 2.0 ** -150, 3 * 2.0 ** -151, 2.0 ** -126,
           float(np.finfo(np.float32).max), 2.0 ** 128 * (1 - 2 ** -25), 2.0 ** 128 * (1 - 2 ** -24)]
    for x in xs:
        assert fp.encode(FP32, value_of_double(x)) == f32_bits_np(x), x


def test_fp16_vs_numpy_random():
    rng = random.Random(99)
    for x in random_doubles(rng, 20000, -30, 20):
        assert fp.encode(FP16, value_of_double(x)) == f16_bits_np(x), x


def test_fp32_decode_all_classes():
    rng = random.Random(7)
    codes = [rng.getrandbits(32) for _ in range(20000)] + [0, 0x80000000, 0x7F800000,
                                                            0xFF800000, 0x7FC00000, 1, 0x00800000]
    for c in codes:
        v = fp.decode(FP32, c)
        x = struct.unpack("<f", struct.pack("<I", c))[0]
        if math.isnan(x):
            assert v.is_nan
        else:
            assert v.to_float() == x and str(v.to_float()) == str(x)
            assert fp.encode(FP32, v) == c


def test_rne_ties_and_modes():
    assert fp.encode(E2M1, Value.of(5, -1)) == 0x4  # 2.5 -> 2
    assert fp.encode(E2M1, Value.of(7, -1)) == 0x6  # 3.5 -> 4
    assert fp.encode(E2M1, Value.of(5)) == 0x6  # 5 -> 4 (codice 0b110, mantissa 0)
    assert fp.encode(E2M1, Value.of(1, -2)) == 0x0  # 0.25 -> 0
    assert fp.encode(E2M1, Value.of(3, -2)) == 0x2  # 0.75 -> 1.0
    assert fp.encode(E2M1, Value.of(5, -1), rounding=Rounding.RNA) == 0x5  # 2.5 -> 3
    assert fp.encode(E2M1, Value.of(11, -2), rounding=Rounding.RTZ) == 0x4  # 2.75 -> 2
    assert fp.encode(E2M1, Value.of(-1, -2)) == 0x8  # -0.25 -> -0
    assert fp.encode(E2M1, Value.of(7)) == 0x7 and fp.encode(E2M1, Value.of(-100)) == 0xF
    assert fp.encode(E2M1, Value.inf(1)) == 0xF
    with pytest.raises(ValueError):
        fp.encode(E2M1, Value.nan())


def brute_e2m1(x: float) -> int:
    """Oracolo: più vicino fra le 8 magnitudini, pareggi al codice pari, saturazione."""
    mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    s = 0x8 if math.copysign(1, x) < 0 else 0
    a = abs(x)
    if a >= 6.0:
        return s | 7
    best = min(range(8), key=lambda i: (abs(a - mags[i]), i & 1))
    return s | best


def test_e2m1_exhaustive_grid():
    for k in range(-8 * 64, 8 * 64 + 1):
        x = k / 64
        assert fp.encode(E2M1, value_of_double(x), overflow=Overflow.SATURATE) == brute_e2m1(x), x


def test_e4m3_saturate_and_ovf():
    assert fp.encode(E4M3, Value.of(464)) == 0x7E  # pareggio 448/480 -> 448 (pari)
    assert fp.encode(E4M3, Value.of(465)) == 0x7E  # 480 non esiste: Overflow INF -> SAT (no Inf)
    assert fp.encode(E4M3, Value.of(465), overflow=Overflow.NAN) == 0x7F
    assert fp.encode(E4M3, Value.inf(1), overflow=Overflow.SATURATE) == 0xFE
    assert fp.encode(E4M3, Value.inf(0), overflow=Overflow.NAN) == 0x7F
    assert fp.encode(E4M3, Value.nan()) == 0x7F
    for c in range(256):
        v = fp.decode(E4M3, c)
        if not v.is_nan:
            assert fp.encode(E4M3, v) == c


def test_e4m3_vs_ml_dtypes():
    ml = pytest.importorskip("ml_dtypes")
    rng = random.Random(5)
    for x in random_doubles(rng, 20000, -12, 9):
        if abs(x) > 448:
            continue  # ml_dtypes in overflow dà NaN (OVF), qui testiamo il range
        ref = int(np.array([x]).astype(ml.float8_e4m3fn).view(np.uint8)[0])
        assert fp.encode(E4M3, value_of_double(x)) == ref, x


def test_ftz_and_rtz_overflow():
    sub = Value.of(1, -140)
    assert fp.encode(FP32, sub, subnormals=Subnormals.FLUSH) == 0
    assert fp.encode(FP32, -sub, subnormals=Subnormals.FLUSH) == 0x80000000
    assert fp.encode(FP32, Value.of(1, 200), rounding=Rounding.RTZ) == 0x7F7FFFFF
    assert fp.encode(FP32, Value.of(1, 200), overflow=Overflow.SATURATE) == 0x7F7FFFFF
    assert fp.encode(FP32, Value.of(1, 200)) == 0x7F800000


def test_exact_arith():
    assert fp.add(Value.inf(0), Value.inf(1)).is_nan
    assert fp.mul(Value.zero(), Value.inf(0)).is_nan
    assert fp.add(Value.zero(1), Value.zero(1)) == Value.zero(1)
    assert fp.add(Value.zero(1), Value.zero(0)) == Value.zero(0)
    assert fp.add(Value.of(3, -1), Value.of(-3, -1)).kind is fp.Kind.ZERO
    assert fp.mul(Value.of(3, -1), Value.of(-3, -1)) == Value.of(-9, -2)
