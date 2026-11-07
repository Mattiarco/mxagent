"""Semantica numerica esplicita di un task.

La spec OCP MX lascia implementation-defined la precisione interna del dot
product (§6.1), l'algoritmo dettagliato della scala (§4 "Not in scope", §6.3 è
solo "should") e la conversione dei NaN verso FP4 (§5.3.3). Senza fissare questi
punti "bit-exact" non ha un significato: ogni task li dichiara in un blocco
``numerics:`` letto sia dal golden model sia dai prompt.

I default seguono la spec dove è normativa e, per il resto, il codice di
riferimento OCP (microsoft/microxcaling, ``mx/mx_ops.py``), indicato come
``[MX-REF]``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any

from .fp import Overflow, Rounding, Subnormals


class Accumulation(str, Enum):
    #: somma esatta di tutti i prodotti (e dell'accumulatore in ingresso),
    #: un solo arrotondamento finale nel formato di uscita
    EXACT = "exact"
    #: per ogni blocco: dot esatto del blocco, poi acc = round(acc + dot_blocco)
    #: (un arrotondamento per blocco: tipico MAC con accumulatore FP32)
    PER_BLOCK = "per_block"


class ScaleOverflow(str, Enum):
    NAN = "nan"  # [MX-REF]: shared_exp > 127 -> scala NaN
    CLAMP = "clamp"  # shared_exp clampato a 127


class ScaleUnderflow(str, Enum):
    CLAMP = "clamp"  # [MX-REF]: shared_exp < -127 -> -127


class ZeroBlock(str, Enum):
    MIN = "min"  # [MX-REF]: amax = 0 -> log2(FP32_MIN_NORMAL) - emax = -128 -> clamp -> codice 0
    ONE = "one"  # scala 2^0 (codice 127)


class SpecialInput(str, Enum):
    #: NaN o Inf nel blocco -> scala NaN (0xFF), elementi a 0 (don't care: con
    #: scala NaN gli elementi sono ignorati, [OCP-MX] §5.1). Coincide con [MX-REF]
    #: per Inf (amax = Inf -> shared_exp > 127 -> NaN).
    NAN_SCALE = "nan_scale"


class InputSubnormals(str, Enum):
    KEEP = "keep"
    DAZ = "daz"  # denormals-are-zero sugli ingressi FP32


@dataclass(frozen=True)
class Numerics:
    # ---- uscita (FP32 di default, [OCP-MX] §6.2 "should be Float32")
    out_format: str = "fp32"
    out_rounding: Rounding = Rounding.RNE
    out_subnormals: Subnormals = Subnormals.KEEP
    out_overflow: Overflow = Overflow.INF
    accumulation: Accumulation = Accumulation.EXACT
    # ---- conversione verso MX ([OCP-MX] §6.3)
    quant_rounding: Rounding = Rounding.RNE  # "must support roundTiesToEven"
    quant_scale_overflow: ScaleOverflow = ScaleOverflow.NAN
    quant_scale_underflow: ScaleUnderflow = ScaleUnderflow.CLAMP
    quant_zero_block: ZeroBlock = ZeroBlock.MIN
    quant_special_input: SpecialInput = SpecialInput.NAN_SCALE
    in_subnormals: InputSubnormals = InputSubnormals.KEEP

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Numerics":
        d = dict(d or {})
        kw: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            raw = d.pop(f.name)
            typ = f.type if not isinstance(f.type, str) else _TYPES[f.name]
            kw[f.name] = typ(raw) if isinstance(typ, type) and issubclass(typ, Enum) else raw
        if d:
            raise ValueError(f"campi numerics sconosciuti: {sorted(d)}")
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(self).items()}

    def describe(self) -> str:
        """Testo per i prompt: regole numeriche che l'hardware deve rispettare."""
        acc = {
            Accumulation.EXACT: "all products are summed EXACTLY (wide fixed-point, no "
                                "intermediate rounding), then ONE final rounding",
            Accumulation.PER_BLOCK: "each block's dot product is computed exactly, then "
                                    "acc = round(acc + block_dot) once per block",
        }[self.accumulation]
        return "\n".join([
            f"- Output format: {self.out_format}, rounding {self.out_rounding.value.upper()}, "
            f"subnormal outputs {'kept' if self.out_subnormals is Subnormals.KEEP else 'flushed to zero'}, "
            f"overflow -> {self.out_overflow.value}.",
            f"- Accumulation: {acc}.",
            "- An exactly-zero dot product is +0. If any block scale is NaN, the output is "
            "the canonical NaN 0x7FC00000.",
            f"- Quantization to MX: element rounding {self.quant_rounding.value.upper()}, "
            "saturate to ±6.0; shared_exp = floor(log2(max|V|)) - 2; "
            f"shared_exp > 127 -> {self.quant_scale_overflow.value}, < -127 -> clamp to -127; "
            f"all-zero block -> scale {'code 0' if self.quant_zero_block is ZeroBlock.MIN else 'code 127'}; "
            "NaN/Inf in block -> scale 0xFF and elements 0.",
            f"- FP32 subnormal inputs: {'kept' if self.in_subnormals is InputSubnormals.KEEP else 'treated as zero (DAZ)'}.",
        ])


_TYPES = {
    "out_format": str, "out_rounding": Rounding, "out_subnormals": Subnormals,
    "out_overflow": Overflow, "accumulation": Accumulation, "quant_rounding": Rounding,
    "quant_scale_overflow": ScaleOverflow, "quant_scale_underflow": ScaleUnderflow,
    "quant_zero_block": ZeroBlock, "quant_special_input": SpecialInput,
    "in_subnormals": InputSubnormals,
}

DEFAULT = Numerics()
