"""Costanti dei formati, verificate sulle fonti primarie.

Fonti (verificate il 2026-11-07):

[OCP-MX]  OCP Microscaling Formats (MX) Specification v1.0, Sep 2023.
          §5.1 semantica del blocco, §5.2 Tab. 1 formati concreti,
          §5.3.1 Tab. 2-3 FP8, §5.3.3 Tab. 5 FP4, §5.4.1 Tab. 7 E8M0,
          §6.1-6.3 dot product e conversione.
[NV-FP4]  NVIDIA Technical Blog, "Introducing NVFP4 for Efficient and Accurate
          Low-Precision Inference": micro-blocchi da 16 valori E2M1, scala
          E4M3 per blocco, scala FP32 di secondo livello per tensore.

Valori NON presenti nelle fonti sono marcati ``NON VERIFICATO`` e non vengono
usati dal golden model finché non lo sono.
"""
from __future__ import annotations

from dataclasses import dataclass

from .fp import FloatFormat, Value, decode

# ---------------------------------------------------------------- elementi
#: FP4 E2M1 — [OCP-MX] §5.3.3 Tab. 5: bias 1, nessun Inf/NaN,
#: max normal S.11.1 = ±6.0, min normal S.01.0 = ±1.0, unico subnormale S.00.1 = ±0.5
E2M1 = FloatFormat("e2m1", ebits=2, mbits=1, bias=1, specials="none")

#: FP6 — [OCP-MX] §5.3.2 Tab. 4 (non usati dai benchmark, inclusi per completezza)
E2M3 = FloatFormat("e2m3", ebits=2, mbits=3, bias=1, specials="none")
E3M2 = FloatFormat("e3m2", ebits=3, mbits=2, bias=3, specials="none")

#: FP8 E4M3 — [OCP-MX] §5.3.1 Tab. 2: bias 7, nessun Inf, NaN = S.1111.111,
#: max normal ±448, min normal ±2^-6, min subnormal ±2^-9
E4M3 = FloatFormat("e4m3", ebits=4, mbits=3, bias=7, specials="e4m3", canonical_nan=0x7F)
#: FP8 E5M2 — [OCP-MX] §5.3.1 Tab. 2: bias 15, IEEE-like (Inf e NaN)
E5M2 = FloatFormat("e5m2", ebits=5, mbits=2, bias=15, specials="ieee", canonical_nan=0x7E)

# ---------------------------------------------------------------- IEEE
#: binary32 (IEEE 754-2019). NaN canonico quiet positivo.
FP32 = FloatFormat("fp32", ebits=8, mbits=23, bias=127, specials="ieee",
                   canonical_nan=0x7FC0_0000)
BF16 = FloatFormat("bf16", ebits=8, mbits=7, bias=127, specials="ieee", canonical_nan=0x7FC0)
FP16 = FloatFormat("fp16", ebits=5, mbits=10, bias=15, specials="ieee", canonical_nan=0x7E00)

FORMATS: dict[str, FloatFormat] = {f.name: f for f in (E2M1, E2M3, E3M2, E4M3, E5M2,
                                                        FP32, BF16, FP16)}

# ---------------------------------------------------------------- scala E8M0
#: [OCP-MX] §5.4.1 Tab. 7: unsigned, bias 127, esponenti -127..127,
#: nessun Inf, nessuno zero, unico NaN = 0b11111111
E8M0_BIAS = 127
E8M0_NAN = 0xFF
E8M0_EMIN = -127
E8M0_EMAX = 127


def e8m0_decode(code: int) -> Value:
    code &= 0xFF
    if code == E8M0_NAN:
        return Value.nan()
    return Value.of(1, code - E8M0_BIAS)


def e8m0_exp(code: int) -> int | None:
    """Esponente non polarizzato, None se NaN."""
    code &= 0xFF
    return None if code == E8M0_NAN else code - E8M0_BIAS


def e8m0_from_exp(exp: int) -> int:
    if not E8M0_EMIN <= exp <= E8M0_EMAX:
        raise ValueError(f"esponente {exp} fuori dal range E8M0 [-127, 127]")
    return exp + E8M0_BIAS


# ---------------------------------------------------------------- MX concreti
@dataclass(frozen=True)
class BlockFormat:
    """Formato a blocchi: ``k`` elementi con una scala condivisa."""

    name: str
    elem: FloatFormat
    k: int
    scale: str  # "e8m0" | "e4m3"
    tensor_scale: bool = False  # scala FP32 per tensore (NVFP4)

    @property
    def elem_emax(self) -> int:
        """Esponente della massima potenza di due rappresentabile nell'elemento
        (usato in [OCP-MX] §6.3 passo 1). E2M1: 4 = 2^2 -> 2."""
        ef, _ = self.elem.max_field
        return ef - self.elem.bias


#: [OCP-MX] §5.2 Tab. 1: MXFP4 = E2M1, k = 32, scala E8M0 (8 bit)
MXFP4 = BlockFormat("mxfp4", E2M1, k=32, scale="e8m0")
#: [NV-FP4]: E2M1, micro-blocchi da 16, scala E4M3 + scala FP32 per tensore
NVFP4 = BlockFormat("nvfp4", E2M1, k=16, scale="e4m3", tensor_scale=True)

BLOCK_FORMATS: dict[str, BlockFormat] = {MXFP4.name: MXFP4, NVFP4.name: NVFP4}

#: magnitudini E2M1 per codice a 3 bit (deriva da E2M1, controllata nei test con Tab. 5)
E2M1_MAGNITUDES: tuple[float, ...] = tuple(decode(E2M1, c).to_float() for c in range(8))
