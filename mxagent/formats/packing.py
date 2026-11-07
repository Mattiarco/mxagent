"""Convenzioni di impacchettamento dei vettori sulle porte hardware.

Convenzione unica in tutto il progetto (iniettata nei prompt):
l'elemento ``i`` di un vettore di elementi larghi ``w`` bit occupa i bit
``[w*i + w-1 : w*i]`` della porta, cioè l'elemento 0 sta negli LSB.
"""
from __future__ import annotations

from collections.abc import Sequence


def pack(elems: Sequence[int], width: int) -> int:
    mask = (1 << width) - 1
    out = 0
    for i, e in enumerate(elems):
        if e & ~mask:
            raise ValueError(f"elemento {i} = {e} non sta in {width} bit")
        out |= e << (width * i)
    return out


def unpack(word: int, width: int, count: int) -> tuple[int, ...]:
    mask = (1 << width) - 1
    return tuple((word >> (width * i)) & mask for i in range(count))


def to_signed(x: int, width: int) -> int:
    x &= (1 << width) - 1
    return x - (1 << width) if x >> (width - 1) else x


def to_unsigned(x: int, width: int) -> int:
    if not -(1 << (width - 1)) <= x < (1 << width):
        raise ValueError(f"{x} non sta in {width} bit")
    return x & ((1 << width) - 1)


CONVENTION = ("Vector ports: element i of a vector of w-bit elements occupies bits "
              "[w*i + w-1 : w*i] (element 0 in the least significant bits).")
