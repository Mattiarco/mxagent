"""Statistiche per il report (solo libreria standard, risultati esatti dove possibile)."""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Intervallo di confidenza di Wilson (95%) per una proporzione k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Stimatore non distorto di pass@k (Chen et al., 2021) con n campioni e c successi."""
    if k > n:
        raise ValueError("k > n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mcnemar_exact(b: int, c: int) -> float:
    """p-value bilaterale esatto di McNemar: b = solo A passa, c = solo B passa."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def median(xs: Iterable[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return statistics.median(v) if v else None


def iqr(xs: Iterable[float | None]) -> tuple[float, float] | None:
    v = sorted(x for x in xs if x is not None)
    if len(v) < 2:
        return None
    q = statistics.quantiles(v, n=4, method="inclusive")
    return (q[0], q[2])


def geomean(xs: Iterable[float | None]) -> float | None:
    v = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None
