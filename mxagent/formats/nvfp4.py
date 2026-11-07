"""Quantizzatore NVFP4 (FP32 -> blocchi da 16 x E2M1 con scala E4M3 + scala per tensore FP32).

Fonte primaria: NVIDIA, "Pretraining Large Language Models with NVFP4",
arXiv:2509.25149v2, Appendice B (letta il 2026-11-07):

  Eq. 1  s_enc = (6 * 448) / amax_x                  scala globale di codifica (amax sul tensore)
         s_dec = 1 / s_enc                           salvata in FP32 (decodifica dopo il GEMM)
  Eq. 2  s_dec,b = amax_b / 6                        scala di decodifica del blocco (16 valori)
  Eq. 3  s_dec,b,e4m3 = e4m3(s_dec,b * s_enc)        "Round-to-nearest-even is used ... in Equation 3"
         s_enc,b = 1 / (fp32(s_dec,b,e4m3) * s_dec)  scala locale di codifica
  Eq. 4  x_hat_i = q(x_i * s_enc,b)                  q = conversione a FP4 (E2M1)
  Eq. 5  dequantizzazione: s_dec,b,e4m3 di A e B nel prodotto di blocco, s_dec dopo il GEMM.

Scelte NON specificate dalla fonte (dichiarate qui e in ``NVFP4_CHOICES``):

* ogni operazione di Eq. 1-4 è eseguita in FP32 con arrotondamento RNE
  (la fonte dice solo che s_dec è "stored in FP32");
* Eq. 3: conversione a E4M3 con saturazione a ±448 (E4M3 non ha Inf; la fonte
  non nomina la saturazione; per il blocco con amax = amax_x il valore è 448);
* Eq. 4: q = RNE con saturazione a ±6 (la fonte raccomanda RNE per pesi e
  attivazioni, sez. 4; la saturazione non è nominata);
* amax_x = 0: s_enc = s_dec = 1 e tutti i blocchi a zero (Eq. 1 non è definita);
* s_enc non rappresentabile (amax_x minuscolo): saturato al massimo FP32;
* blocco con scala E4M3 = 0 (amax_b = 0 o troppo piccola): s_enc,b = 0, gli
  elementi diventano zeri con il segno dell'ingresso;
* NaN o Inf in ingresso: non supportati (ValueError).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from . import fp
from .fp import Kind, Value
from .specs import E2M1, E4M3, FP32, NVFP4

NVFP4_SOURCE = "arXiv:2509.25149v2, Appendix B, Eq. 1-4"
NVFP4_CHOICES = (
    "FP32 RNE for every operation of Eq. 1-4",
    "E4M3 scale: RNE with saturation to 448",
    "E2M1 elements: RNE with saturation to 6",
    "amax_x = 0: s_enc = s_dec = 1, all-zero blocks",
    "s_enc overflow: saturate to FP32 max",
    "zero E4M3 block scale: s_enc,b = 0, elements are signed zeros",
    "NaN/Inf inputs rejected",
)


def _frac(v: Value) -> Fraction:
    if v.kind is Kind.ZERO:
        return Fraction(0)
    if v.kind is not Kind.FINITE:
        raise ValueError("valore non finito")
    m = Fraction(v.mant) * (Fraction(2) ** v.exp)
    return -m if v.sign else m


def _value(x: Fraction, extra_bits: int = 400) -> Value:
    """Valore diadico che arrotonda come ``x`` in ogni formato con precisione
    molto minore di ``extra_bits``: troncamento + bit di sticky (mai un pareggio)."""
    if x == 0:
        return Value.zero(0)
    sign = 1 if x < 0 else 0
    a = abs(x)
    q, r = divmod(a.numerator << extra_bits, a.denominator)
    n = 2 * q + (1 if r else 0)
    return Value.of(-n if sign else n, -(extra_bits + 1))


def _round(fmt, x: Fraction, overflow=fp.Overflow.SATURATE) -> int:
    return fp.encode(fmt, _value(x), rounding=fp.Rounding.RNE, overflow=overflow)


def f32(x: Fraction) -> Fraction:
    """Arrotonda un razionale a FP32 (RNE, overflow saturato) e restituisce il valore."""
    return _frac(fp.decode(FP32, _round(FP32, x)))


@dataclass(frozen=True)
class NVFP4Tensor:
    scales: tuple[int, ...]  # codici E4M3, uno per blocco da 16
    elems: tuple[int, ...]  # codici E2M1
    tensor_scale: int  # s_dec in FP32 (codice)

    def dequantize(self) -> list[Fraction]:
        s_dec = _frac(fp.decode(FP32, self.tensor_scale))
        out = []
        for b, sc in enumerate(self.scales):
            s = _frac(fp.decode(E4M3, sc)) * s_dec
            out += [_frac(fp.decode(E2M1, e)) * s for e in self.elems[16 * b:16 * b + 16]]
        return out


def quantize_nvfp4(fp32_bits: Sequence[int]) -> NVFP4Tensor:
    k = NVFP4.k
    if not fp32_bits or len(fp32_bits) % k:
        raise ValueError(f"servono multipli di {k} valori, ricevuti {len(fp32_bits)}")
    vals = [fp.decode(FP32, b & 0xFFFFFFFF) for b in fp32_bits]
    if any(v.kind in (Kind.NAN, Kind.INF) for v in vals):
        raise ValueError("NaN/Inf in ingresso non supportati (scelta documentata)")
    xs = [_frac(v) for v in vals]
    amax = max(abs(x) for x in xs)
    if amax == 0:
        one = fp.encode(FP32, Value.of(1))
        return NVFP4Tensor((0,) * (len(xs) // k), tuple(E2M1.pack(v.sign, 0, 0) for v in vals), one)
    s_enc = f32(Fraction(6 * 448) / amax)  # Eq. 1 (saturato se non rappresentabile)
    s_dec = f32(1 / s_enc)
    scales, elems = [], []
    for b in range(len(xs) // k):
        blk, bv = xs[b * k:(b + 1) * k], vals[b * k:(b + 1) * k]
        amax_b = max(abs(x) for x in blk)
        s_dec_b = f32(amax_b / 6)  # Eq. 2
        code = _round(E4M3, f32(s_dec_b * s_enc))  # Eq. 3 (RNE, saturazione a 448)
        scales.append(code)
        sb = _frac(fp.decode(E4M3, code))
        if sb == 0:
            elems += [E2M1.pack(v.sign, 0, 0) for v in bv]
            continue
        s_enc_b = f32(1 / f32(sb * s_dec))  # scala locale di codifica
        for x, v in zip(blk, bv):
            y = f32(x * s_enc_b)
            e = _round(E2M1, y)  # Eq. 4 (RNE, saturazione a 6)
            if y == 0:
                e = E2M1.pack(v.sign, 0, 0)  # zero con il segno dell'ingresso
            elems.append(e)
    return NVFP4Tensor(tuple(scales), tuple(elems), fp.encode(FP32, _value(s_dec)))
