"""Golden model: test esaustivi dove lo spazio è piccolo, random altrove, con
oracoli indipendenti (aritmetica float64 esatta + conversione numpy a float32)."""
import math
import random
import warnings

import numpy as np
import pytest

from mxagent.formats import golden as g
from mxagent.formats.golden import MXBlock
from mxagent.formats.numerics import Accumulation, Numerics, ZeroBlock
from mxagent.formats.specs import E2M1_MAGNITUDES, NVFP4, E4M3
from mxagent.formats import fp

NAN32 = 0x7FC00000


def e2m1_float(c):
    m = E2M1_MAGNITUDES[c & 7]
    return -m if c & 8 else m


def f32_bits(x: float) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return int(np.array([x], dtype=np.float64).astype(np.float32).view(np.uint32)[0])


def f32_from_bits(b: int) -> float:
    return float(np.array([b], dtype=np.uint32).view(np.float32)[0])


def oracle_dot(a: MXBlock, b: MXBlock) -> int:
    """float64 è esatto qui: Σ di 32 prodotti multipli di 1/4 (|Σ| <= 1152) per
    2^(ea+eb) con ea+eb in [-254, 254]; poi un solo arrotondamento numpy -> f32."""
    if a.scale == 0xFF or b.scale == 0xFF:
        return NAN32
    s = sum(e2m1_float(x) * e2m1_float(y) for x, y in zip(a.elems, b.elems))
    v = math.ldexp(s, (a.scale - 127) + (b.scale - 127))
    return f32_bits(v) if v != 0 else 0


def rand_block(rng, k=32, scale=None):
    return MXBlock(rng.randrange(256) if scale is None else scale,
                   tuple(rng.randrange(16) for _ in range(k)))


# ------------------------------------------------------------- 1 e 2
def test_e2m1_to_fixed_exhaustive():
    for c in range(16):
        assert g.e2m1_to_fixed(c) == int(e2m1_float(c) * 2)
        assert g.e2m1_to_fixed(c, frac_bits=3) == int(e2m1_float(c) * 8)
    assert {g.e2m1_to_fixed(c) for c in range(16)} == {-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12}


def test_e2m1_mul_exhaustive_16x16():
    for a in range(16):
        for b in range(16):
            assert g.e2m1_mul_fixed(a, b) == int(e2m1_float(a) * e2m1_float(b) * 4)
    # unità 1/4: ±36 -> ±144, serve SInt(9) (valore riportato nella scheda dei prompt)
    assert max(g.e2m1_mul_fixed(a, b) for a in range(16) for b in range(16)) == 144
    assert min(g.e2m1_mul_fixed(a, b) for a in range(16) for b in range(16)) == -144


def test_e2m1_bad_code():
    with pytest.raises(ValueError):
        g.e2m1_to_fixed(16)


# ----------------------------------------------------------------- 3
def test_e8m0_to_fp32_exhaustive_256():
    for c in range(255):
        assert g.e8m0_to_fp32(c) == f32_bits(2.0 ** (c - 127)), c
    assert g.e8m0_to_fp32(0xFF) == NAN32
    assert g.e8m0_to_fp32(0) == 0x00400000  # 2^-127 subnormale
    assert g.e8m0_to_fp32(1) == 0x00800000
    assert g.e8m0_to_fp32(254) == 0x7F000000
    ftz = Numerics.from_dict({"out_subnormals": "flush"})
    assert g.e8m0_to_fp32(0, ftz) == 0


# ----------------------------------------------------------------- 5
def test_dot_known_values():
    ones = MXBlock(127, (0x2,) * 32)
    assert g.mx_dot(ones, ones) == 0x42000000  # 32.0
    six = MXBlock(127, (0x7,) * 32)
    assert g.mx_dot(six, six) == f32_bits(1152.0)
    assert g.mx_dot(six, MXBlock(127, (0xF,) * 32)) == f32_bits(-1152.0)
    assert g.mx_dot(MXBlock(0xFF, (0x2,) * 32), ones) == NAN32
    assert g.mx_dot(ones, MXBlock(0xFF, (0,) * 32)) == NAN32
    big = MXBlock(254, (0x7,) * 32)
    assert g.mx_dot(big, big) == 0x7F800000  # overflow -> +Inf
    sat = Numerics.from_dict({"out_overflow": "saturate"})
    assert g.mx_dot(big, big, sat) == 0x7F7FFFFF
    tiny = MXBlock(0, (0x1,) + (0,) * 31)
    assert g.mx_dot(tiny, tiny) == 0  # 0.25 * 2^-254 -> underflow a +0
    # cancellazione esatta -> +0, anche con zeri negativi
    assert g.mx_dot(MXBlock(127, (0x2, 0xA) + (0,) * 30), ones) == 0
    assert g.mx_dot(MXBlock(127, (0x8,) * 32), ones) == 0


def test_dot_random_vs_oracle():
    rng = random.Random(2024)
    for _ in range(3000):
        a, b = rand_block(rng), rand_block(rng)
        # metà dei casi con scale vicine a 127 per esercitare i normali
        if rng.random() < 0.5:
            a = MXBlock(rng.randint(100, 154), a.elems)
            b = MXBlock(rng.randint(100, 154), b.elems)
        assert g.mx_dot(a, b) == oracle_dot(a, b)


def test_dot_all_scale_pairs():
    """Tutte le 256 x 256 coppie di scale con due pattern di elementi."""
    rng = random.Random(1)
    pats = [tuple(rng.randrange(16) for _ in range(32)), (0x1,) + (0,) * 31]
    for ea in pats:
        for sa in range(256):
            for sb in range(256):
                a, b = MXBlock(sa, ea), MXBlock(sb, pats[0])
                assert g.mx_dot(a, b) == oracle_dot(a, b), (sa, sb)


def test_dot_with_acc_in():
    rng = random.Random(3)
    for _ in range(2000):
        a, b = rand_block(rng, scale=rng.randint(110, 140)), rand_block(rng, scale=rng.randint(110, 140))
        acc = f32_bits(rng.uniform(-1e4, 1e4))
        s = sum(e2m1_float(x) * e2m1_float(y) for x, y in zip(a.elems, b.elems))
        exact = math.ldexp(s, a.scale + b.scale - 254) + f32_from_bits(acc)  # esatto in f64
        assert g.mx_dot(a, b, acc_in=acc) == (f32_bits(exact) if exact != 0 else 0)
    one = MXBlock(127, (0x2,) * 32)
    assert g.mx_dot(one, one, acc_in=0x7F800000) == 0x7F800000
    assert g.mx_dot(one, one, acc_in=0x7FC00001) == NAN32


# ----------------------------------------------------------------- 8
def test_mac_per_block_vs_numpy_sequential():
    rng = random.Random(11)
    num = Numerics.from_dict({"accumulation": "per_block"})
    for _ in range(300):
        n = rng.randint(1, 6)
        A = [rand_block(rng, scale=rng.randint(115, 140)) for _ in range(n)]
        B = [rand_block(rng, scale=rng.randint(115, 140)) for _ in range(n)]
        acc0 = f32_bits(rng.uniform(-100, 100))
        ref = np.float32(f32_from_bits(acc0))
        for a, b in zip(A, B):
            s = sum(e2m1_float(x) * e2m1_float(y) for x, y in zip(a.elems, b.elems))
            # acc + dot esatto (la somma in f64 è esatta in questo range), poi
            # un solo arrotondamento a f32 per blocco
            ref = np.float32(float(ref) + math.ldexp(s, a.scale + b.scale - 254))
        got = g.mx_mac(A, B, num, acc_in=acc0)
        assert got == (f32_bits(float(ref)) if ref != 0 else 0)


def test_mac_exact_single_rounding():
    rng = random.Random(12)
    for _ in range(300):
        n = rng.randint(1, 6)
        A = [rand_block(rng, scale=rng.randint(115, 140)) for _ in range(n)]
        B = [rand_block(rng, scale=rng.randint(115, 140)) for _ in range(n)]
        total = sum(math.ldexp(sum(e2m1_float(x) * e2m1_float(y) for x, y in zip(a.elems, b.elems)),
                               a.scale + b.scale - 254) for a, b in zip(A, B))
        # la somma in f64 è esatta se gli esponenti stanno in ~50 bit di range: 115..140 -> sì
        assert g.mx_mac(A, B) == (f32_bits(total) if total != 0 else 0)


def test_mac_modes_differ_somewhere():
    """Le due semantiche di accumulo devono essere davvero diverse (altrimenti
    il campo numerics sarebbe inutile)."""
    big, small = MXBlock(127 + 24, (0x2,) + (0,) * 31), MXBlock(127, (0x1,) + (0,) * 31)
    one = MXBlock(127, (0x2,) + (0,) * 31)
    A, B = [big, small, small], [one, one, one]  # 2^24 + 0.5 + 0.5
    pb = Numerics.from_dict({"accumulation": "per_block"})
    assert g.mx_mac(A, B) == f32_bits(2.0 ** 24 + 1)
    assert g.mx_mac(A, B, pb) == f32_bits(2.0 ** 24)


# ----------------------------------------------------------------- 4
def oracle_quantize(xs: list[float]) -> MXBlock:
    """Oracolo float: §6.3 con frexp e ricerca esaustiva del più vicino."""
    if any(math.isnan(x) or math.isinf(x) for x in xs):
        return MXBlock(0xFF, (0,) * 32)
    amax = max(abs(x) for x in xs)
    if amax == 0:
        return MXBlock(0, tuple(0x8 if math.copysign(1, x) < 0 else 0 for x in xs))
    shared = math.frexp(amax)[1] - 1 - 2
    if shared > 127:
        return MXBlock(0xFF, (0,) * 32)
    shared = max(shared, -127)
    from tests.test_fp import brute_e2m1
    return MXBlock(shared + 127, tuple(brute_e2m1(math.ldexp(x, -shared)) for x in xs))


def rand_f32(rng):
    kind = rng.random()
    if kind < 0.05:
        return 0.0
    if kind < 0.1:
        return math.ldexp(rng.randint(1, 1 << 23), -149)  # subnormale
    x = f32_from_bits(f32_bits(math.ldexp(rng.uniform(1, 2), rng.randint(-130, 127))))
    return -x if rng.random() < 0.5 else x


def test_quantize_random_vs_oracle():
    rng = random.Random(77)
    for _ in range(3000):
        mode = rng.random()
        if mode < 0.5:  # valori in un range dinamico ristretto, come nei tensori reali
            base = rng.randint(-120, 120)
            xs = [f32_from_bits(f32_bits(math.ldexp(rng.uniform(-8, 8), base))) for _ in range(32)]
        else:
            xs = [rand_f32(rng) for _ in range(32)]
        got = g.quantize_mx([f32_bits(x) for x in xs])
        assert got == oracle_quantize(xs), xs


def test_quantize_corner_cases():
    z = [0.0] * 32
    assert g.quantize_mx([f32_bits(x) for x in z]) == MXBlock(0, (0,) * 32)
    one = Numerics.from_dict({"quant_zero_block": "one"})
    assert g.quantize_mx([0] * 32, num=one).scale == 127
    neg0 = [0x80000000] + [0] * 31
    assert g.quantize_mx(neg0).elems[0] == 0x8
    nan = [NAN32] + [0x3F800000] * 31
    assert g.quantize_mx(nan) == MXBlock(0xFF, (0,) * 32)
    inf = [0xFF800000] + [0] * 31
    assert g.quantize_mx(inf).scale == 0xFF
    # max FP32: floor(log2) = 127 -> shared = 125
    mx = [0x7F7FFFFF] + [0] * 31
    assert g.quantize_mx(mx) == MXBlock(125 + 127, (0x7,) + (0,) * 31)
    # 7.0 -> shared 0, 7/1 -> satura a 6 (§6.3 passo 2)
    q = g.quantize_mx([f32_bits(7.0), f32_bits(-7.0)] + [0] * 30)
    assert q.scale == 127 and q.elems[:2] == (0x7, 0xF)
    # pareggi RNE dopo la scala: amax 4 -> X = 1; 2.5 -> 2, 0.25 -> 0, 0.75 -> 1, 5 -> 4
    xs = [4.0, 2.5, 0.25, 0.75, 5.0, -0.25] + [0.0] * 26
    assert g.quantize_mx([f32_bits(x) for x in xs]).elems[:6] == (0x6, 0x4, 0x0, 0x2, 0x6, 0x8)
    # scala che va in underflow: min subnormale FP32 2^-149 -> shared clampato a -127
    q = g.quantize_mx([1] + [0] * 31)
    assert q.scale == 0 and q.elems[0] == 0  # 2^-149 / 2^-127 = 2^-22 -> 0
    daz = Numerics.from_dict({"in_subnormals": "daz"})
    assert g.quantize_mx([0x00400000] + [0] * 31, num=daz) == MXBlock(0, (0,) * 32)
    with pytest.raises(ValueError):
        g.quantize_mx([0] * 31)


def test_quantize_dequantize_error_bound():
    """Proprietà: |dequant(quant(x)) - x| <= mezzo ulp E2M1 * X, salvo saturazione."""
    rng = random.Random(8)
    for _ in range(500):
        xs = [rng.uniform(-10, 10) for _ in range(32)]
        xs = [f32_from_bits(f32_bits(x)) for x in xs]
        q = g.quantize_mx([f32_bits(x) for x in xs])
        X = 2.0 ** (q.scale - 127)
        for x, c in zip(xs, q.elems):
            y = e2m1_float(c) * X
            if abs(x) / X <= 6:
                ulp = 0.5 if abs(x) / X < 2 else (1.0 if abs(x) / X < 4 else 2.0)
                assert abs(y - x) <= ulp / 2 * X + 1e-12


# ----------------------------------------------------------------- 7
def test_nvfp4_dot():
    rng = random.Random(21)
    for _ in range(2000):
        sa, sb = rng.randrange(0x7F), rng.randrange(0x7F)  # scale positive, no NaN
        a = MXBlock(sa, tuple(rng.randrange(16) for _ in range(16)))
        b = MXBlock(sb, tuple(rng.randrange(16) for _ in range(16)))
        s = sum(e2m1_float(x) * e2m1_float(y) for x, y in zip(a.elems, b.elems))
        ref = s * fp.decode(E4M3, sa).to_float() * fp.decode(E4M3, sb).to_float()  # esatto in f64
        assert g.nvfp4_dot(a, b) == (f32_bits(ref) if ref != 0 else 0)
    nanb = MXBlock(0x7F, (2,) * 16)
    assert g.nvfp4_dot(nanb, MXBlock(0x38, (2,) * 16)) == NAN32
    one = MXBlock(0x38, (0x2,) * 16)  # E4M3 0x38 = 1.0
    assert g.nvfp4_dot(one, one) == f32_bits(16.0)
    assert g.nvfp4_dot(one, one, tensor_scale_a=f32_bits(0.5), tensor_scale_b=f32_bits(0.25)) == f32_bits(2.0)
    with pytest.raises(NotImplementedError):
        g.quantize_mx([0] * 16, fmt=NVFP4)


# ------------------------------------------------------- elementwise
def test_elemwise_mul():
    rng = random.Random(4)
    for _ in range(300):
        a, b = rand_block(rng, scale=rng.randint(60, 200)), rand_block(rng, scale=rng.randint(60, 200))
        out = g.mx_elemwise_mul(a, b)
        for x, y, o in zip(a.elems, b.elems, out):
            v = e2m1_float(x) * e2m1_float(y) * 2.0 ** (a.scale + b.scale - 254)
            ref = f32_bits(v)
            if v == 0:  # segno dello zero: IEEE (xor dei segni)
                ref = 0x80000000 if ((x >> 3) ^ (y >> 3)) else 0
            assert o == ref


def test_numerics_roundtrip_and_validation():
    n = Numerics.from_dict({"accumulation": "per_block", "out_overflow": "saturate"})
    assert Numerics.from_dict(n.to_dict()) == n
    with pytest.raises(ValueError):
        Numerics.from_dict({"bogus": 1})
    with pytest.raises(ValueError):
        Numerics.from_dict({"accumulation": "wrong"})
    assert ZeroBlock.MIN.value == "min" and Accumulation.EXACT.value == "exact"
