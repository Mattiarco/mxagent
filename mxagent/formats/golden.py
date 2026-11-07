"""Golden model bit-exact dei kernel dei benchmark.

Indipendente dall'LLM, scritto a mano, solo aritmetica intera esatta
(:mod:`mxagent.formats.fp`). Ogni funzione prende e restituisce **codici
binari** (interi), esattamente come le porte dell'hardware.

Kernel (numerati come i benchmark):

1. :func:`e2m1_to_fixed`        E2M1 -> virgola fissa con segno
2. :func:`e2m1_mul_fixed`       E2M1 x E2M1 -> prodotto esatto in virgola fissa
3. :func:`e8m0_to_fp32`         scala E8M0 -> FP32 (NaN incluso)
4. :func:`quantize_mx`          vettore FP32 -> blocco MX (scala + elementi)
5. :func:`mx_dot`               dot di due blocchi MX -> FP32 (opz. + accumulatore)
6. (stessa funzione di 5, versione pipelined)
7. :func:`nvfp4_dot`            dot di due blocchi NVFP4 (scala E4M3, scala per tensore)
8. :func:`mx_mac`               acc += dot(A_j, B_j) su più blocchi ([OCP-MX] §6.2)
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import fp
from .fp import Kind, Value
from .numerics import (DEFAULT, Accumulation, InputSubnormals, Numerics, ScaleOverflow,
                       ZeroBlock)
from .specs import (E2M1, E4M3, E8M0_EMAX, E8M0_EMIN, E8M0_NAN, FORMATS, FP32, MXFP4, NVFP4,
                    BlockFormat, e8m0_decode, e8m0_exp)


# ---------------------------------------------------------------- utilità
def out_encode(v: Value, num: Numerics = DEFAULT) -> int:
    fmt = FORMATS[num.out_format]
    return fp.encode(fmt, v, rounding=num.out_rounding, overflow=num.out_overflow,
                     subnormals=num.out_subnormals)


def fp32_decode(bits: int, num: Numerics = DEFAULT) -> Value:
    v = fp.decode(FP32, bits)
    if num.in_subnormals is InputSubnormals.DAZ and v.kind is Kind.FINITE:
        _, ef, _ = FP32.fields(bits)
        if ef == 0:
            return Value.zero(v.sign)
    return v


def _check_codes(codes: Sequence[int], width: int, what: str) -> None:
    for c in codes:
        if not 0 <= c < (1 << width):
            raise ValueError(f"{what}: codice {c} fuori da {width} bit")


# ------------------------------------------------------------ 1. decode E2M1
def e2m1_to_fixed(code: int, frac_bits: int = 1) -> int:
    """Valore di un nibble E2M1 come intero con segno in unità ``2**-frac_bits``.

    Con ``frac_bits=1`` (minimo esatto) il range è [-12, 12]: serve un SInt(5).
    """
    _check_codes([code], 4, "E2M1")
    return fp.decode(E2M1, code).signed_int_at(-frac_bits)


# ------------------------------------------------------- 2. prodotto E2M1
def e2m1_mul_fixed(a: int, b: int) -> int:
    """Prodotto esatto di due E2M1 in unità 1/4: il valore sta in [-36, 36],
    quindi l'intero restituito sta in [-144, 144] (SInt(9))."""
    _check_codes([a, b], 4, "E2M1")
    return fp.mul(fp.decode(E2M1, a), fp.decode(E2M1, b)).signed_int_at(-2)


# ----------------------------------------------------------- 3. scala E8M0
def e8m0_to_fp32(code: int, num: Numerics = DEFAULT) -> int:
    """2^(code-127) in FP32; 0xFF -> NaN canonico. Il codice 0 (2^-127) è
    subnormale in FP32 (0x00400000), oppure +0 con ``out_subnormals=flush``."""
    _check_codes([code], 8, "E8M0")
    return out_encode(e8m0_decode(code), num)


# -------------------------------------------------------- 4. quantizzatore
@dataclass(frozen=True)
class MXBlock:
    scale: int  # codice della scala (E8M0 o E4M3)
    elems: tuple[int, ...]  # codici degli elementi


def _floor_log2(v: Value) -> int:
    return v.mant.bit_length() - 1 + v.exp


def quantize_mx(fp32_bits: Sequence[int], fmt: BlockFormat = MXFP4,
                num: Numerics = DEFAULT) -> MXBlock:
    """Conversione [OCP-MX] §6.3 di ``k`` valori FP32 in un blocco MX.

    1. X = 2^(floor(log2(max|V_i|)) - emax_elem)   (emax_elem = 2 per E2M1)
    2. P_i = quantize(V_i / X) con arrotondamento ``quant_rounding`` e
       saturazione al massimo normale preservando il segno.
    Casi limite secondo ``num`` (vedi :mod:`numerics`).
    """
    if fmt.scale != "e8m0":
        raise NotImplementedError("NVFP4 ha una scala per tensore: usa formats.nvfp4.quantize_nvfp4")
    if len(fp32_bits) != fmt.k:
        raise ValueError(f"servono {fmt.k} valori, ricevuti {len(fp32_bits)}")
    _check_codes(fp32_bits, 32, "FP32")
    vals = [fp32_decode(b, num) for b in fp32_bits]
    if any(v.kind in (Kind.NAN, Kind.INF) for v in vals):
        return MXBlock(E8M0_NAN, (0,) * fmt.k)
    finite = [v for v in vals if v.kind is Kind.FINITE]
    if not finite:
        if num.quant_zero_block is ZeroBlock.MIN:
            shared = E8M0_EMIN
        else:
            shared = 0
        # elementi: zeri con il loro segno
        return MXBlock(shared + 127, tuple(E2M1.pack(v.sign, 0, 0) for v in vals))
    amax_log2 = max(_floor_log2(v) for v in finite)
    shared = amax_log2 - fmt.elem_emax
    if shared > E8M0_EMAX:
        if num.quant_scale_overflow is ScaleOverflow.NAN:
            return MXBlock(E8M0_NAN, (0,) * fmt.k)
        shared = E8M0_EMAX
    shared = max(shared, E8M0_EMIN)
    elems = tuple(
        fp.encode(fmt.elem, fp.scale_pow2(v, -shared), rounding=num.quant_rounding,
                  overflow=fp.Overflow.SATURATE)
        for v in vals)
    return MXBlock(shared + 127, elems)


# ------------------------------------------------------------- 5. dot MX
def _scale_value(fmt: BlockFormat, code: int) -> Value:
    if fmt.scale == "e8m0":
        return e8m0_decode(code)
    return fp.decode(E4M3, code)


def block_dot_exact(fmt: BlockFormat, a: MXBlock, b: MXBlock) -> Value:
    """Dot esatto di due blocchi: X_A X_B Σ P_A,i P_B,i ([OCP-MX] §6.1), senza
    arrotondamenti. NaN se una scala è NaN; uno zero esatto è +0."""
    for blk, nm in ((a, "A"), (b, "B")):
        if len(blk.elems) != fmt.k:
            raise ValueError(f"blocco {nm}: servono {fmt.k} elementi")
        _check_codes(blk.elems, fmt.elem.width, f"elementi {nm}")
        _check_codes([blk.scale], 8, f"scala {nm}")
    xa, xb = _scale_value(fmt, a.scale), _scale_value(fmt, b.scale)
    if xa.is_nan or xb.is_nan:
        return Value.nan()
    # Σ in unità 1/4 (ogni prodotto E2M1 x E2M1 è multiplo di 1/4)
    unit = -2 * fmt.elem.mbits
    acc = sum(fp.mul(fp.decode(fmt.elem, p), fp.decode(fmt.elem, q)).signed_int_at(unit)
              for p, q in zip(a.elems, b.elems))
    s = Value.of(acc, unit)
    if s.kind is Kind.ZERO:
        return Value.zero(0)
    r = fp.mul(fp.mul(s, xa), xb)
    return Value.zero(0) if r.kind is Kind.ZERO else r


def mx_dot(a: MXBlock, b: MXBlock, num: Numerics = DEFAULT, acc_in: int | None = None,
           fmt: BlockFormat = MXFP4) -> int:
    """Dot di due blocchi MX -> codice FP32 (o ``num.out_format``).

    Con ``acc_in`` (codice FP32) calcola round(acc_in + dot) con un solo
    arrotondamento: fused, coerente con entrambe le modalità di accumulo per un
    singolo blocco.
    """
    d = block_dot_exact(fmt, a, b)
    if acc_in is not None:
        d = fp.add(fp32_decode(acc_in, num), d)
    return out_encode(d, num)


# ------------------------------------------------------------- 7. NVFP4
def nvfp4_dot(a: MXBlock, b: MXBlock, num: Numerics = DEFAULT,
              tensor_scale_a: int | None = None, tensor_scale_b: int | None = None) -> int:
    """Dot di due micro-blocchi NVFP4 (16 x E2M1, scala E4M3) -> FP32.

    Le scale per tensore (codici FP32, opzionali) moltiplicano il dot esatto
    prima dell'unico arrotondamento finale. La scala E4M3 è usata col suo
    segno (NVIDIA usa scale positive; un benchmark può vincolarle).
    """
    d = block_dot_exact(NVFP4, a, b)
    for ts in (tensor_scale_a, tensor_scale_b):
        if ts is not None:
            d = fp.mul(d, fp32_decode(ts, num))
    return out_encode(d, num)


# -------------------------------------------------------------- 8. MAC
def mx_mac(blocks_a: Sequence[MXBlock], blocks_b: Sequence[MXBlock], num: Numerics = DEFAULT,
           acc_in: int | None = None, fmt: BlockFormat = MXFP4) -> int:
    """DotGeneral [OCP-MX] §6.2: acc_in + Σ_j Dot(A_j, B_j) -> FP32."""
    if len(blocks_a) != len(blocks_b):
        raise ValueError("numero di blocchi diverso")
    out_fmt = FORMATS[num.out_format]
    acc = fp32_decode(acc_in, num) if acc_in is not None else Value.zero(0)
    for a, b in zip(blocks_a, blocks_b):
        d = block_dot_exact(fmt, a, b)
        acc = fp.add(acc, d)
        if num.accumulation is Accumulation.PER_BLOCK:
            acc = fp.decode(out_fmt, out_encode(acc, num))
    return out_encode(acc, num)


# ------------------------------------------------- moltiplicazione elementwise
def mx_elemwise_mul(a: MXBlock, b: MXBlock, num: Numerics = DEFAULT,
                    fmt: BlockFormat = MXFP4) -> tuple[int, ...]:
    """(X_A P_A,i) * (X_B P_B,i) per ogni i, arrotondato in FP32."""
    xa, xb = _scale_value(fmt, a.scale), _scale_value(fmt, b.scale)
    out = []
    for p, q in zip(a.elems, b.elems):
        v = fp.mul(fp.mul(fp.decode(fmt.elem, p), xa), fp.mul(fp.decode(fmt.elem, q), xb))
        out.append(out_encode(v, num))
    return tuple(out)
