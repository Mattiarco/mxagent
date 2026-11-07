"""Motore esatto per formati floating-point binari piccoli e grandi.

Tutti i valori finiti trattati in questo progetto sono *diadici*: ``n * 2**e``
con ``n`` ed ``e`` interi. Il golden model lavora quindi solo con interi Python
(precisione illimitata) e non usa mai ``float``: nessun doppio arrotondamento,
nessuna dipendenza dalla FPU della macchina.

Un formato è descritto da :class:`FloatFormat` (bit di esponente/mantissa,
bias, politica dei valori speciali). Le funzioni principali sono:

* :func:`decode` — codice intero → :class:`Value` (zero, finito, Inf, NaN);
* :func:`encode` — :class:`Value` → codice, con modo di arrotondamento,
  politica di overflow e gestione dei subnormali esplicite.

Le politiche dei valori speciali coprono i formati usati:

``ieee``     esponente tutto a 1: mantissa 0 = Inf, altrimenti NaN (FP32, BF16, FP16, E5M2)
``e4m3``     solo ``S.1111.111`` è NaN, nessun Inf (OCP FP8 E4M3, MX spec Tab. 2)
``none``     nessun Inf/NaN, tutte le codifiche sono finite (FP4 E2M1, FP6; MX §5.3.2-5.3.3)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Kind(str, Enum):
    ZERO = "zero"
    FINITE = "finite"
    INF = "inf"
    NAN = "nan"


@dataclass(frozen=True)
class Value:
    """Valore esatto: ``(-1)**sign * mant * 2**exp`` se ``kind`` è FINITE.

    Per ZERO e INF conta solo ``sign``; per NAN nulla. ``mant`` > 0 per i finiti.
    """

    kind: Kind
    sign: int = 0
    mant: int = 0
    exp: int = 0

    # ---------------------------------------------------------- costruttori
    @staticmethod
    def zero(sign: int = 0) -> "Value":
        return Value(Kind.ZERO, sign & 1)

    @staticmethod
    def inf(sign: int = 0) -> "Value":
        return Value(Kind.INF, sign & 1)

    @staticmethod
    def nan() -> "Value":
        return Value(Kind.NAN)

    @staticmethod
    def of(n: int, e: int = 0, sign: int | None = None) -> "Value":
        """``n * 2**e`` con ``n`` intero con segno; ``sign`` forza il segno dello zero."""
        if n == 0:
            return Value.zero(sign if sign is not None else 0)
        s = 1 if n < 0 else 0
        n = abs(n)
        tz = (n & -n).bit_length() - 1  # normalizza: mantissa dispari
        return Value(Kind.FINITE, s, n >> tz, e + tz)

    # --------------------------------------------------------------- query
    @property
    def is_nan(self) -> bool:
        return self.kind is Kind.NAN

    @property
    def is_finite(self) -> bool:
        return self.kind in (Kind.ZERO, Kind.FINITE)

    def signed_int_at(self, e: int) -> int:
        """Valore come intero in unità ``2**e`` (deve essere esatto)."""
        if self.kind is Kind.ZERO:
            return 0
        if self.kind is not Kind.FINITE:
            raise ValueError(f"{self} non è finito")
        shift = self.exp - e
        if shift < 0:
            raise ValueError(f"{self} non è rappresentabile in unità 2**{e}")
        v = self.mant << shift
        return -v if self.sign else v

    def to_float(self) -> float:
        """Solo per stampa/debug: può perdere precisione o andare in overflow."""
        if self.kind is Kind.NAN:
            return float("nan")
        if self.kind is Kind.INF:
            return float("-inf") if self.sign else float("inf")
        if self.kind is Kind.ZERO:
            return -0.0 if self.sign else 0.0
        import math

        v = math.ldexp(float(self.mant), self.exp)
        return -v if self.sign else v

    def __neg__(self) -> "Value":
        if self.kind is Kind.NAN:
            return self
        return Value(self.kind, self.sign ^ 1, self.mant, self.exp)


def add(a: Value, b: Value) -> Value:
    """Somma esatta (nessun arrotondamento). Inf-Inf = NaN; 0 + 0 segue IEEE (RNE)."""
    if a.is_nan or b.is_nan:
        return Value.nan()
    if a.kind is Kind.INF or b.kind is Kind.INF:
        if a.kind is Kind.INF and b.kind is Kind.INF and a.sign != b.sign:
            return Value.nan()
        return a if a.kind is Kind.INF else b
    if a.kind is Kind.ZERO and b.kind is Kind.ZERO:
        return Value.zero(a.sign & b.sign)  # -0 + -0 = -0, altrimenti +0
    if a.kind is Kind.ZERO:
        return b
    if b.kind is Kind.ZERO:
        return a
    e = min(a.exp, b.exp)
    return Value.of(a.signed_int_at(e) + b.signed_int_at(e), e)


def mul(a: Value, b: Value) -> Value:
    """Prodotto esatto. 0 * Inf = NaN."""
    if a.is_nan or b.is_nan:
        return Value.nan()
    s = a.sign ^ b.sign
    if a.kind is Kind.INF or b.kind is Kind.INF:
        if a.kind is Kind.ZERO or b.kind is Kind.ZERO:
            return Value.nan()
        return Value.inf(s)
    if a.kind is Kind.ZERO or b.kind is Kind.ZERO:
        return Value.zero(s)
    return Value(Kind.FINITE, s, a.mant * b.mant, a.exp + b.exp)


def scale_pow2(a: Value, k: int) -> Value:
    """``a * 2**k`` esatto."""
    if a.kind is not Kind.FINITE:
        return a
    return Value(Kind.FINITE, a.sign, a.mant, a.exp + k)


# ======================================================================
#                                formati
# ======================================================================
class Rounding(str, Enum):
    RNE = "rne"  # round to nearest, ties to even (IEEE roundTiesToEven)
    RNA = "rna"  # round to nearest, ties away from zero
    RTZ = "rtz"  # verso zero (troncamento)


class Overflow(str, Enum):
    INF = "inf"  # IEEE: oltre il massimo -> ±Inf (o NaN se il formato non ha Inf)
    SATURATE = "saturate"  # clamp a ±max finito (OCP "SAT")
    NAN = "nan"  # OCP "OVF" per E4M3: -> NaN


class Subnormals(str, Enum):
    KEEP = "keep"
    FLUSH = "flush"  # risultati subnormali -> zero con segno (FTZ)


@dataclass(frozen=True)
class FloatFormat:
    name: str
    ebits: int
    mbits: int
    bias: int
    specials: str  # "ieee" | "e4m3" | "none"
    canonical_nan: int | None = None  # codice emesso per NaN (None se il formato non ha NaN)

    # ------------------------------------------------------------ derivati
    @property
    def width(self) -> int:
        return 1 + self.ebits + self.mbits

    @property
    def emin(self) -> int:
        """Esponente minimo dei normali (anche scala dei subnormali)."""
        return 1 - self.bias

    @property
    def has_inf(self) -> bool:
        return self.specials == "ieee"

    @property
    def has_nan(self) -> bool:
        return self.specials in ("ieee", "e4m3")

    @property
    def max_field(self) -> tuple[int, int]:
        """(campo esponente, campo mantissa) del massimo finito."""
        top = (1 << self.ebits) - 1
        full = (1 << self.mbits) - 1
        if self.specials == "ieee":
            return top - 1, full
        if self.specials == "e4m3":
            return top, full - 1  # S.1111.111 è NaN
        return top, full

    @property
    def max_finite(self) -> Value:
        ef, mf = self.max_field
        return self._fields_value(0, ef, mf)

    @property
    def min_subnormal(self) -> Value:
        return Value.of(1, self.emin - self.mbits)

    def _fields_value(self, s: int, ef: int, mf: int) -> Value:
        if ef == 0:
            if mf == 0:
                return Value.zero(s)
            v = Value.of(mf, self.emin - self.mbits)
        else:
            v = Value.of((1 << self.mbits) | mf, ef - self.bias - self.mbits)
        return -v if s else v

    def fields(self, code: int) -> tuple[int, int, int]:
        """(segno, esponente, mantissa) di un codice."""
        code &= (1 << self.width) - 1
        return (code >> (self.ebits + self.mbits),
                (code >> self.mbits) & ((1 << self.ebits) - 1),
                code & ((1 << self.mbits) - 1))

    def pack(self, s: int, ef: int, mf: int) -> int:
        return (s << (self.ebits + self.mbits)) | (ef << self.mbits) | mf


def decode(fmt: FloatFormat, code: int) -> Value:
    s, ef, mf = fmt.fields(code)
    top = (1 << fmt.ebits) - 1
    if fmt.specials == "ieee" and ef == top:
        return Value.inf(s) if mf == 0 else Value.nan()
    if fmt.specials == "e4m3" and ef == top and mf == (1 << fmt.mbits) - 1:
        return Value.nan()
    return fmt._fields_value(s, ef, mf)


def _round_shift(n: int, shift: int, mode: Rounding, sign: int) -> int:
    """``n / 2**shift`` arrotondato a intero (n >= 0, shift > 0)."""
    q = n >> shift
    rem = n & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if mode is Rounding.RTZ or rem == 0:
        return q
    if rem > half:
        return q + 1
    if rem < half:
        return q
    # pareggio esatto
    if mode is Rounding.RNA:
        return q + 1
    return q + (q & 1)  # RNE


def encode(fmt: FloatFormat, v: Value, *, rounding: Rounding = Rounding.RNE,
           overflow: Overflow = Overflow.INF, subnormals: Subnormals = Subnormals.KEEP,
           nan_code: int | None = None) -> int:
    """Arrotonda il valore esatto ``v`` nel formato ``fmt`` e ne restituisce il codice.

    * Arrotondamento con range di esponente illimitato verso l'alto, poi verifica
      di overflow sul valore arrotondato (comportamento IEEE 754 §7.4).
    * ``overflow``: INF (default IEEE; se il formato non ha Inf ripiega su
      SATURATE), SATURATE (OCP SAT), NAN (OCP OVF per E4M3).
      In RTZ l'overflow IEEE dà il massimo finito, non Inf.
    * Ingresso ±Inf: Inf se il formato lo ha, altrimenti SAT -> ±max, OVF -> NaN
      (MX spec Tab. 3).
    * Ingresso NaN: ``nan_code`` (default ``fmt.canonical_nan``); per formati
      senza NaN è implementation-defined (MX §5.3.3) e va passato esplicitamente.
    * Valori sotto il minimo subnormale dopo l'arrotondamento -> zero con segno
      (MX §5.3.3). Con ``subnormals=FLUSH`` anche i risultati subnormali -> zero.
    """
    s = v.sign
    if v.kind is Kind.NAN:
        code = nan_code if nan_code is not None else fmt.canonical_nan
        if code is None:
            raise ValueError(f"{fmt.name} non ha NaN: serve nan_code esplicito")
        return code
    if v.kind is Kind.ZERO:
        return fmt.pack(s, 0, 0)
    if v.kind is Kind.INF:
        return _overflow_code(fmt, s, overflow, rounding, from_inf=True)

    n, e = v.mant, v.exp
    msb = n.bit_length() - 1
    E = max(msb + e, fmt.emin)  # esponente del risultato (clamp al range subnormale)
    qe = E - fmt.mbits  # esponente del quanto (ulp)
    shift = qe - e
    m = n << -shift if shift <= 0 else _round_shift(n, shift, rounding, s)
    if m >> (fmt.mbits + 1):  # riporto: 1.11..1 -> 10.00..0
        m >>= 1
        qe += 1
    if m == 0:
        return fmt.pack(s, 0, 0)
    E = qe + fmt.mbits
    if m >> fmt.mbits:  # normale
        ef, mf = E + fmt.bias, m & ((1 << fmt.mbits) - 1)
        mef, mmf = fmt.max_field
        if ef > mef or (ef == mef and mf > mmf):
            return _overflow_code(fmt, s, overflow, rounding, from_inf=False)
        return fmt.pack(s, ef, mf)
    # subnormale
    if subnormals is Subnormals.FLUSH:
        return fmt.pack(s, 0, 0)
    return fmt.pack(s, 0, m)


def _overflow_code(fmt: FloatFormat, s: int, overflow: Overflow, rounding: Rounding,
                   from_inf: bool) -> int:
    mef, mmf = fmt.max_field
    sat = fmt.pack(s, mef, mmf)
    if overflow is Overflow.NAN:
        if fmt.canonical_nan is None:
            raise ValueError(f"{fmt.name} non ha NaN: overflow=NAN non ammesso")
        return fmt.canonical_nan
    if overflow is Overflow.SATURATE or not fmt.has_inf:
        return sat
    # Overflow.INF in un formato con Inf
    if from_inf:
        return fmt.pack(s, (1 << fmt.ebits) - 1, 0)
    if rounding is Rounding.RTZ:
        return sat
    return fmt.pack(s, (1 << fmt.ebits) - 1, 0)


def round_value(fmt: FloatFormat, v: Value, **kw) -> Value:
    """Arrotonda ``v`` nel formato e restituisce il valore risultante."""
    return decode(fmt, encode(fmt, v, **kw))
