"""Adattatori fra le porte di un task e il golden model.

Ogni kernel fornisce:

* ``fn(inputs, num, params) -> outputs``: dizionari ``porta -> intero`` così come
  stanno sui fili (unsigned; i SInt in complemento a due);
* ``space(params)``: tutti i vettori di ingresso (solo se lo spazio è piccolo);
* ``corner(params)``: casi limite scelti a mano;
* ``random(rng, params)``: un vettore casuale "interessante".
"""
from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any

from ..formats import golden as g
from ..formats.numerics import Numerics
from ..formats.packing import to_unsigned

Vec = dict[str, int]


@dataclass(frozen=True)
class Kernel:
    fn: Callable[[Vec, Numerics, dict[str, Any]], Vec]
    space: Callable[[dict[str, Any]], Iterator[Vec]] | None = None
    corner: Callable[[dict[str, Any]], list[Vec]] = lambda p: []
    random: Callable[[random.Random, dict[str, Any]], Vec] | None = None


# ---------------------------------------------------------------- b01
def _e2m1_decode(v: Vec, num: Numerics, p: dict[str, Any]) -> Vec:
    fb = p.get("frac_bits", 1)
    return {"y": to_unsigned(g.e2m1_to_fixed(v["a"], fb), p.get("out_width", 5))}


# ---------------------------------------------------------------- b02
def _e2m1_mul(v: Vec, num: Numerics, p: dict[str, Any]) -> Vec:
    return {"p": to_unsigned(g.e2m1_mul_fixed(v["a"], v["b"]), 9)}


# ---------------------------------------------------------------- b03
def _e8m0_to_fp32(v: Vec, num: Numerics, p: dict[str, Any]) -> Vec:
    return {"y": g.e8m0_to_fp32(v["x"], num)}


KERNELS: dict[str, Kernel] = {
    "e2m1_decode": Kernel(
        _e2m1_decode,
        space=lambda p: ({"a": a} for a in range(16)),
    ),
    "e2m1_mul": Kernel(
        _e2m1_mul,
        space=lambda p: ({"a": a, "b": b} for a, b in product(range(16), repeat=2)),
    ),
    "e8m0_to_fp32": Kernel(
        _e8m0_to_fp32,
        space=lambda p: ({"x": x} for x in range(256)),
    ),
}


from . import kernels_mx  # noqa: E402,F401  (registra b04/b05)
from . import subkernels  # noqa: E402,F401  (registra i blocchi del catalogo)
from . import kernels_m7  # noqa: E402,F401  (registra b07 NVFP4 e b08 MAC)


def get_kernel(name: str) -> Kernel:
    try:
        return KERNELS[name]
    except KeyError:
        raise KeyError(f"golden kernel sconosciuto: {name} (noti: {sorted(KERNELS)})") from None
