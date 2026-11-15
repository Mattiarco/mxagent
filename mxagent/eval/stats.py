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


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """p-value bilaterale esatto di Fisher per la tabella 2x2 [[a, b], [c, d]]
    (es. successi/fallimenti di due configurazioni): somma delle tabelle con gli stessi
    margini e probabilità ipergeometrica <= quella osservata."""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2
    if n == 0:
        return 1.0

    def prob(x: int) -> float:
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)

    p0 = prob(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p for x in range(lo, hi + 1) if (p := prob(x)) <= p0 * (1 + 1e-9)))


def mann_whitney_u(xs: Iterable[float | None], ys: Iterable[float | None]) -> dict:
    """Test U di Mann-Whitney bilaterale (due campioni indipendenti).

    p-value con l'approssimazione normale, correzione per i pareggi e di continuità
    (adeguata da n ~ 8 per gruppo; con n = 10 per configurazione è il caso tipico).
    Restituisce U (del primo campione), z, p e le dimensioni; p = None se un gruppo è vuoto
    o tutti i valori coincidono."""
    x = [float(v) for v in xs if v is not None]
    y = [float(v) for v in ys if v is not None]
    n1, n2 = len(x), len(y)
    if not n1 or not n2:
        return {"u": None, "z": None, "p": None, "n1": n1, "n2": n2}
    allv = sorted(x + y)
    ranks: dict[float, float] = {}
    i = 0
    ties = 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[allv[i]] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    r1 = sum(ranks[v] for v in x)
    u1 = r1 - n1 * (n1 + 1) / 2
    n = n1 + n2
    mu = n1 * n2 / 2
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return {"u": u1, "z": 0.0, "p": None, "n1": n1, "n2": n2}
    z = (abs(u1 - mu) - 0.5) / math.sqrt(var)
    z = max(z, 0.0)
    p = math.erfc(z / math.sqrt(2))
    return {"u": u1, "z": round(z, 4), "p": min(1.0, p), "n1": n1, "n2": n2}


def median(xs: Iterable[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return statistics.median(v) if v else None


def iqr(xs: Iterable[float | None]) -> tuple[float, float] | None:
    v = sorted(x for x in xs if x is not None)
    if len(v) < 2:
        return None
    q = statistics.quantiles(v, n=4, method="inclusive")
    return (q[0], q[2])


def mean_std(xs: Iterable[float | None]) -> dict:
    """n, media e deviazione standard campionaria (None con meno di 2 valori)."""
    v = [float(x) for x in xs if x is not None]
    return {"n": len(v), "mean": statistics.fmean(v) if v else None,
            "std": statistics.stdev(v) if len(v) >= 2 else None}


def geomean(xs: Iterable[float | None]) -> float | None:
    v = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None
