"""Scomposizione leggibile dei valori sulle porte (segno/esponente/mantissa) e
classificazione dei casi (zero, subnormale, NaN...) per il feedback."""
from __future__ import annotations

from ..bench.task import Port
from ..formats import fp
from ..formats.packing import to_signed, unpack
from ..formats.specs import BF16, E2M1, E4M3, FP32, e8m0_decode

_FMT = {"e2m1": E2M1, "e4m3": E4M3, "fp32": FP32, "bf16": BF16}


def _num(v: fp.Value) -> str:
    if v.kind is fp.Kind.NAN:
        return "NaN"
    if v.kind is fp.Kind.INF:
        return "-Inf" if v.sign else "+Inf"
    x = v.to_float()
    return f"{x:+.9g}" if x != 0 else ("-0" if v.sign else "+0")


def elem_class(port: Port, code: int) -> list[str]:
    """Etichette di classe per un singolo elemento."""
    t = port.type
    if t in _FMT:
        f = _FMT[t]
        v = fp.decode(f, code)
        s, e, m = f.fields(code)
        labels = ["neg"] if s and v.kind is not fp.Kind.NAN else []
        if v.kind is fp.Kind.NAN:
            return labels + ["nan"]
        if v.kind is fp.Kind.INF:
            return labels + ["inf"]
        if v.kind is fp.Kind.ZERO:
            return labels + ["zero"]
        if e == 0:
            return labels + ["subnormal"]
        if (e, m) == f.max_field:
            return labels + ["max"]
        return labels + ["normal"]
    if t == "e8m0":
        return ["nan"] if code == 0xFF else (["min"] if code == 0 else
                                             ["max"] if code == 254 else ["one"] if code == 127 else ["normal"])
    if t == "sint":
        x = to_signed(code, port.elem_width)
        return ["zero"] if x == 0 else (["neg"] if x < 0 else ["pos"])
    return ["zero"] if code == 0 else ["nonzero"]


def port_classes(port: Port, word: int) -> set[str]:
    if port.elems == 1:
        return set(elem_class(port, word))
    out: set[str] = set()
    for c in unpack(word, port.elem_width, port.elems):
        out.update(elem_class(port, c))
    return out


def describe_elem(port: Port, code: int) -> str:
    t = port.type
    w = port.elem_width
    if t in _FMT:
        f = _FMT[t]
        s, e, m = f.fields(code)
        return (f"0x{code:0{(w + 3) // 4}X} (s={s} e={e:0{f.ebits}b} m={m:0{f.mbits}b}"
                f"{'' if f.mbits <= 4 else f' [0x{m:X}]'} = {_num(fp.decode(f, code))})")
    if t == "e8m0":
        v = e8m0_decode(code)
        return f"0x{code:02X} ({'NaN' if v.is_nan else f'2^{code - 127}'})"
    if t == "sint":
        x = to_signed(code, w)
        real = f" = {x / (1 << port.frac_bits):+g}" if port.frac_bits else ""
        return f"{x} [{code:0{w}b}]{real}"
    if t == "bool":
        return str(code)
    real = f" = {code / (1 << port.frac_bits):g}" if port.frac_bits else ""
    return f"{code} [0x{code:X}]{real}"


def describe(port: Port, word: int | None, max_elems: int = 8) -> str:
    if word is None:
        return "X/Z (undefined)"
    if port.elems == 1:
        return describe_elem(port, word)
    codes = unpack(word, port.elem_width, port.elems)
    nz = [(i, c) for i, c in enumerate(codes) if c != 0]
    shown = ", ".join(f"[{i}]={describe_elem(port, c)}" for i, c in nz[:max_elems])
    more = f", ... (+{len(nz) - max_elems} more nonzero)" if len(nz) > max_elems else ""
    return f"{port.elems} elems, nonzero: {shown or 'none'}{more}"


def diff_bits(port: Port, exp: int, got: int | None) -> str:
    if got is None:
        return "output is X/Z"
    x = exp ^ got
    if not x:
        return ""
    bits = [i for i in range(port.total_width) if x >> i & 1]
    txt = f"{len(bits)} bit(s) differ: {bits if len(bits) <= 12 else str(bits[:12])[:-1] + ', ...]'}"
    if port.type in _FMT and port.elems == 1:
        f = _FMT[port.type]
        fe, fg = f.fields(exp), f.fields(got)
        parts = [n for n, a, b in zip(("sign", "exponent", "mantissa"), fe, fg) if a != b]
        txt += f" (fields: {', '.join(parts)})"
    return txt
