"""Confronto con il golden e feedback strutturato e compresso per il Reflector.

Invece dei log grezzi l'LLM riceve:
* errori di compilazione già parsati (file, riga, messaggio, snippet);
* warning di lint rilevanti, riportati al sorgente Scala;
* per i mismatch: conteggi per suite, pattern sulle classi di ingresso che
  falliscono, e pochi vettori minimi con valori scomposti in campi e bit diversi.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ..bench.task import Task
from .values import describe, diff_bits, port_classes

Vec = dict[str, int]


@dataclass
class Mismatch:
    index: int
    suite: str
    inputs: Vec
    expected: Vec
    got: dict[str, int | None]
    minimized_from: int | None = None  # indice del vettore originale se ridotto

    @property
    def bad_ports(self) -> list[str]:
        return [k for k, e in self.expected.items() if self.got.get(k) != e]

    def to_dict(self) -> dict:
        return {"index": self.index, "suite": self.suite, "inputs": self.inputs,
                "expected": self.expected, "got": self.got, "minimized_from": self.minimized_from}


def compare(task: Task, vecs: list[Vec], expected: list[Vec], got_rows: list,
            suites: list[str]) -> list[Mismatch]:
    outs = [p.name for p in task.outputs]
    mism = []
    for i, (v, e, row) in enumerate(zip(vecs, expected, got_rows)):
        got = dict(zip(outs, row)) if row is not None else {k: None for k in outs}
        if any(got[k] != e[k] for k in outs):
            mism.append(Mismatch(i, suites[i], v, e, got))
    return mism


def patterns(task: Task, vecs: list[Vec], failing: set[int], max_items: int = 4) -> list[str]:
    """Classi di ingresso sovra-rappresentate nei fallimenti.

    Esempio: "every failing vector has a=subnormal; 0/14 passing vectors do".
    """
    if not failing or len(failing) == len(vecs):
        return []
    sets = [frozenset((p.name, c) for p in task.inputs for c in port_classes(p, v[p.name]))
            for v in vecs]
    nf, np_ = len(failing), len(vecs) - len(failing)

    def rates(keys: tuple) -> tuple[float, float, int]:
        f = sum(1 for i in failing if all(k in sets[i] for k in keys))
        p = sum(1 for i, s in enumerate(sets) if i not in failing and all(k in s for k in keys))
        return f / nf, (p / np_ if np_ else 0.0), p

    singles = {}
    for key in set().union(*(sets[i] for i in failing)):
        rf, rp, pc = rates((key,))
        if rf >= 0.5:
            singles[key] = (rf, rp, pc)
    cands = [((k,), *r) for k, r in singles.items()]
    # congiunzioni di due classi: catturano regole come "a negativo E b negativo"
    keys = sorted(singles)
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            if k1[0] == k2[0] and len(task.inputs) > 1:
                continue  # stessa porta: poco informativo
            rf, rp, pc = rates((k1, k2))
            best_single = max(singles[k1][0] - singles[k1][1], singles[k2][0] - singles[k2][1])
            if rf >= 0.5 and (rf - rp) > best_single + 0.05:
                cands.append(((k1, k2), rf, rp, pc))
    cands = [c for c in cands if c[1] - c[2] >= 0.4]
    cands.sort(key=lambda c: (c[1] - c[2], len(c[0])), reverse=True)
    out = []
    for keys_, rf, _, pc in cands[:max_items]:
        share = "every" if rf == 1 else f"{rf:.0%} of"
        cond = " AND ".join(f"{port}={cls}" for port, cls in keys_)
        out.append(f"{share} failing vector has {cond}; {pc}/{np_} passing vectors do")
    return out


def pick_representatives(task: Task, mism: list[Mismatch], k: int,
                         complexity) -> list[Mismatch]:
    """Fino a k mismatch semplici e diversi fra loro (classi e porte sbagliate diverse)."""
    ordered = sorted(mism, key=lambda m: (complexity(task, m.inputs), m.index))
    chosen, sigs = [], set()
    for m in ordered:
        sig = (tuple(sorted((p.name, tuple(sorted(port_classes(p, m.inputs[p.name]))))
                            for p in task.inputs)), tuple(m.bad_ports))
        if sig in sigs:
            continue
        sigs.add(sig)
        chosen.append(m)
        if len(chosen) == k:
            return chosen
    for m in ordered:  # riempi con i più semplici rimanenti
        if len(chosen) == k:
            break
        if m not in chosen:
            chosen.append(m)
    return chosen


def render_mismatch(task: Task, m: Mismatch, n: int) -> str:
    lines = [f"#{n} ({m.suite}{', minimized' if m.minimized_from is not None else ''})"]
    for p in task.inputs:
        lines.append(f"   in  {p.name} = {describe(p, m.inputs[p.name])}")
    for p in task.outputs:
        e, g = m.expected[p.name], m.got.get(p.name)
        if e == g:
            continue
        lines.append(f"   out {p.name}: expected {describe(p, e)}")
        lines.append(f"   {' ' * len(p.name)}      got      {describe(p, g)}")
        d = diff_bits(p, e, g)
        if d:
            lines.append(f"   {' ' * len(p.name)}      {d}")
    return "\n".join(lines)


def table_rows(task: Task, mism: list[Mismatch]) -> list[str]:
    """Una riga compatta per mismatch: valori interpretati, senza scomposizione in bit."""
    def val(p, x):
        if x is None:
            return "X"
        d = describe(p, x)
        m = re.search(r"= ([^)\]]+)\)?$", d)
        return f"{x:#x}" + (f" ({m.group(1).strip()})" if m else "")

    rows = []
    for m in mism:
        ins = ", ".join(f"{p.name}={val(p, m.inputs[p.name])}" for p in task.inputs)
        outs = "; ".join(f"{p.name}: {val(p, m.expected[p.name])} / {val(p, m.got.get(p.name))}"
                         for p in task.outputs if m.expected[p.name] != m.got.get(p.name))
        rows.append(f"{ins} -> {outs}")
    return rows


@dataclass
class SimSummary:
    total: int = 0
    failed: int = 0
    per_suite: dict[str, list[int]] = field(default_factory=dict)  # suite -> [tot, fail]
    patterns: list[str] = field(default_factory=list)
    examples: list[Mismatch] = field(default_factory=list)
    ddmin_rounds: int = 0
    table: list[str] = field(default_factory=list)  # tutti i fallimenti, se lo spazio è piccolo

    @property
    def pass_rate(self) -> float:
        return 1.0 - self.failed / self.total if self.total else 0.0

    def render(self, task: Task) -> str:
        suites = ", ".join(f"{s}: {t - f}/{t} pass" for s, (t, f) in self.per_suite.items())
        lines = [f"FUNCTIONAL MISMATCH: {self.failed}/{self.total} test vectors fail ({suites})."]
        if self.patterns:
            lines.append("Patterns:")
            lines += [f" - {p}" for p in self.patterns]
        lines.append(f"Smallest failing vectors ({len(self.examples)}):")
        lines += [render_mismatch(task, m, i + 1) for i, m in enumerate(self.examples)]
        if self.table:
            lines.append(f"All {len(self.table)} failing vectors (input -> expected / got):")
            lines += [f"  {r}" for r in self.table]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"total": self.total, "failed": self.failed, "pass_rate": round(self.pass_rate, 6),
                "per_suite": self.per_suite, "patterns": self.patterns,
                "examples": [m.to_dict() for m in self.examples], "ddmin_rounds": self.ddmin_rounds}
