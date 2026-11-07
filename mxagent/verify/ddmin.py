"""Riduzione dei vettori che falliscono (delta-debugging a round batch).

Dato un vettore che fallisce, si generano semplificazioni (elementi azzerati a
blocchi in stile ddmin, scale portate a 1, scalari a 0) e si simulano TUTTE in
un'unica chiamata. Si tiene la più semplice che fallisce ancora e si ripete.
Il costo è un round di simulazione per livello, non per candidato.
"""
from __future__ import annotations

from collections.abc import Callable

from ..bench.task import Port, Task
from ..formats.packing import pack, unpack

Vec = dict[str, int]
#: valore "neutro" di una porta scalare per tipo
NEUTRAL = {"e8m0": 127, "e4m3": 0x38, "fp32": 0, "bf16": 0}


def complexity(task: Task, v: Vec) -> int:
    """Numero di elementi/scalari non banali: misura di quanto un caso è "grande"."""
    n = 0
    for p in task.inputs:
        if p.elems > 1:
            n += sum(1 for c in unpack(v[p.name], p.elem_width, p.elems) if c)
        else:
            n += v[p.name] != NEUTRAL.get(p.type, 0)
    return n


def _chunks(idx: list[int], parts: int) -> list[list[int]]:
    k = max(1, len(idx) // parts)
    return [idx[i:i + k] for i in range(0, len(idx), k)]


def candidates(task: Task, v: Vec) -> list[Vec]:
    out: list[Vec] = []
    for p in task.inputs:
        if p.elems > 1:
            codes = list(unpack(v[p.name], p.elem_width, p.elems))
            nz = [i for i, c in enumerate(codes) if c]
            if len(nz) <= 1:
                continue
            for parts in (2, 4, 8, len(nz)):
                for ch in _chunks(nz, parts):
                    # complemento (tieni solo ch) e sottrazione (azzera ch)
                    for keep in (set(ch), set(nz) - set(ch)):
                        if not keep or len(keep) == len(nz):
                            continue
                        new = [c if i in keep else 0 for i, c in enumerate(codes)]
                        out.append({**v, p.name: pack(new, p.elem_width)})
        else:
            neutral = NEUTRAL.get(p.type, 0)
            if v[p.name] != neutral:
                out.append({**v, p.name: neutral})
    # dedup mantenendo l'ordine
    seen, uniq = set(), []
    for c in out:
        k = tuple(sorted(c.items()))
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def minimize(task: Task, v: Vec, still_fails: Callable[[list[Vec]], list[bool]],
             max_rounds: int = 8) -> tuple[Vec, int]:
    """Riduce ``v``; ``still_fails`` simula un batch e dice quali falliscono ancora.
    Restituisce (vettore minimo, round di simulazione usati)."""
    cur, rounds = v, 0
    while rounds < max_rounds:
        cands = [c for c in candidates(task, cur) if complexity(task, c) < complexity(task, cur)]
        if not cands:
            break
        rounds += 1
        flags = still_fails(cands)
        failing = [c for c, f in zip(cands, flags) if f]
        if not failing:
            break
        cur = min(failing, key=lambda c: complexity(task, c))
    return cur, rounds


__all__ = ["complexity", "candidates", "minimize", "Port"]
