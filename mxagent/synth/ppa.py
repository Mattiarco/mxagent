"""Report PPA multi-backend e regola di accettazione del PPA Optimizer.

Ogni numero viene da uno strumento eseguito davvero; un backend assente è
marcato ``skipped`` e non produce numeri.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..bench.task import Task
from ..config import SynthConfig
from . import external, nextpnr, yosys
from .yosys import SynthResult


@dataclass
class PPAReport:
    top: str
    results: dict[str, SynthResult] = field(default_factory=dict)

    def get(self, backend: str, metric: str) -> float | None:
        r = self.results.get(backend)
        return r.metrics.get(metric) if r and r.ok else None

    def primary(self, cfg: SynthConfig) -> tuple[float, float] | None:
        """(area, ritardo) della metrica primaria, None se non disponibile."""
        b = cfg.primary
        area = self.get(b, "transistors" if b == "generic" else "area")
        delay = self.get(b, "depth")
        if area is None or delay is None:
            return None
        return float(area), float(delay)

    def to_dict(self) -> dict:
        return {"top": self.top, "results": {k: v.to_dict() for k, v in self.results.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "PPAReport":
        return cls(d["top"], {k: SynthResult.from_dict(v) for k, v in d["results"].items()})

    def summary(self) -> dict:
        """Metriche compatte per metrics.json e le tabelle di M6."""
        return {k: (r.metrics if r.ok else {"error": r.error, "skipped": r.skipped})
                for k, r in self.results.items()}

    def render(self, max_cells: int = 10) -> str:
        lines = []
        for k, r in self.results.items():
            if r.skipped:
                lines.append(f"{k:8s} skipped: {r.error}")
                continue
            if not r.ok:
                lines.append(f"{k:8s} FAILED: {r.error}")
                continue
            m = ", ".join(f"{a}={b:g}" if isinstance(b, (int, float)) else f"{a}={b}"
                          for a, b in r.metrics.items())
            lines.append(f"{k:8s} {m}  ({r.seconds:.1f} s)")
            if k in ("generic", "liberty") and r.cells:
                top = sorted(r.cells.items(), key=lambda kv: -kv[1])[:max_cells]
                lines.append("         cells: " + ", ".join(f"{c}={n}" for c, n in top))
        return "\n".join(lines)


def objective(area: float, delay: float, kind: str) -> float:
    return {"area": area, "delay": delay, "adp": area * delay}[kind]


def accept(old: tuple[float, float], new: tuple[float, float], cfg: SynthConfig) -> tuple[bool, str]:
    """Accetta se l'obiettivo migliora di almeno ``min_gain``.

    Con obiettivo ``area`` il ritardo non può peggiorare di oltre ``tolerance``
    (e viceversa per ``delay``); con ``adp`` conta solo il prodotto.
    """
    o_old, o_new = objective(*old, cfg.objective), objective(*new, cfg.objective)
    gain = (o_old - o_new) / o_old if o_old else 0.0
    (a0, d0), (a1, d1) = old, new
    if cfg.objective == "area" and d1 > d0 * (1 + cfg.tolerance):
        return False, f"delay worsened {d0:g} -> {d1:g}"
    if cfg.objective == "delay" and a1 > a0 * (1 + cfg.tolerance):
        return False, f"area worsened {a0:g} -> {a1:g}"
    if gain <= 0:
        return False, f"{cfg.objective} {o_old:g} -> {o_new:g} ({-gain:+.1%}): not better than current"
    if gain < cfg.min_gain:
        return False, f"{cfg.objective} {o_old:g} -> {o_new:g} ({-gain:+.1%}): below min_gain {cfg.min_gain:.0%}"
    return True, f"{cfg.objective} {o_old:g} -> {o_new:g} ({-gain:+.1%})"


def synthesize(verilog: list[str | Path], task: Task, cfg: SynthConfig, work: Path,
               backends: list[str] | None = None) -> PPAReport:
    files = [Path(v) for v in verilog if str(v).endswith((".sv", ".v"))]
    rep = PPAReport(task.top)
    for b in backends or cfg.backends:
        w = work / b
        if b in yosys.FLOWS:
            rep.results[b] = yosys.synth(files, task.top, b, w, liberty=cfg.liberty, timeout=cfg.timeout_s)
        elif b == "nextpnr":
            rep.results[b] = nextpnr.pnr_ecp5(files, task, w, device=cfg.ecp5_device,
                                              package=cfg.ecp5_package, clock_ns=cfg.clock_ns,
                                              seeds=cfg.pnr_seeds, timeout=cfg.timeout_s)
        elif b == "opensta":
            rep.results[b] = external.opensta(files, task.top, w, cfg.liberty, timeout=cfg.timeout_s)
        elif b == "vivado":
            rep.results[b] = external.vivado(files, task, w, part=cfg.fpga_part, clock_ns=cfg.clock_ns,
                                             timeout=cfg.timeout_s)
        else:
            raise ValueError(f"backend sconosciuto {b}")
    work.mkdir(parents=True, exist_ok=True)
    (work / "ppa.json").write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")
    return rep
