"""Miglior candidato della run, anche quando nessun candidato passa tutti i test.

Criterio di ordinamento (in quest'ordine; ogni livello conta solo a parità dei precedenti):

(a) **elaborato**: il candidato compila ed elabora in Verilog (stadio del verifier oltre
    ``compile``: scalac, elaborazione Chisel e firtool riusciti);
(b) **test passati**: frazione dei vettori Verilator bit-exact contro il golden model
    (``pass_rate``; 1.0 = tutti, cioè soluzione completa). Un candidato fermo al lint
    non ha simulato nulla e vale 0;
(c) **sintesi Yosys riuscita** (backend primario, ``synth.primary``). Si sintetizzano
    solo i candidati ancora a pari merito dopo (a) e (b), al più ``max_synth`` (i più
    recenti); senza Yosys la sintesi è ``skipped`` per tutti e il livello non decide;
(d) **iterazione più recente** (``seq``: ordine di verifica nella run).

Una run superata ha già un design scelto e verificato di nuovo (``final/``, eventualmente
migliorato dal PPA Optimizer): quello è il miglior candidato, ``solution: complete``.
Altrimenti il migliore secondo (a)-(d) è ``solution: partial``; senza candidati
verificati ``solution: none``.

Uscita: ``runs/<id>/best/`` con i sorgenti Chisel, il Verilog emesso e ``report.json``
(test passati/totali, esito della sintesi con risorse e ritardo se disponibili,
iterazione in cui il candidato è stato prodotto).
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CRITERION = ("(a) elaborates to Verilog, (b) fraction of golden-model test vectors passed, "
             "(c) Yosys synthesis succeeded (primary backend), (d) most recent iteration")

SynthFn = Callable[[list[str], str], dict | None]  # (verilog, tag) -> summary del backend primario


@dataclass
class Candidate:
    tag: str
    sources: dict[str, str]
    report: Any  # VerifyReport
    seq: int  # ordine di verifica nella run (1 = primo candidato verificato)
    phase: str = ""  # monolithic | integrate | baseline-b ...
    origin: str = ""  # gen | reflect | resample | fix | ppa ...
    synth: dict | None = None  # summary del backend primario (ok, skipped, metriche)
    extra: dict = field(default_factory=dict)

    @property
    def elaborated(self) -> bool:
        return self.report is not None and self.report.stage != "compile"

    @property
    def pass_rate(self) -> float:
        return float((self.report.sim or {}).get("pass_rate", 0.0)) if self.report else 0.0

    @property
    def tests(self) -> tuple[int | None, int | None]:
        sim = (self.report.sim or {}) if self.report else {}
        total = sim.get("total")
        if total is None:
            return None, None
        return int(total) - int(sim.get("failed", 0)), int(total)

    @property
    def synth_ok(self) -> bool:
        return bool(self.synth and self.synth.get("ok"))


def rank_key(c: Candidate) -> tuple:
    """Chiave di ordinamento (più grande = migliore): vedi il docstring del modulo."""
    return (c.elaborated, c.pass_rate, c.synth_ok, c.seq)


def select_best(cands: list[Candidate], synth_fn: SynthFn | None = None,
                max_synth: int = 3) -> Candidate | None:
    """Miglior candidato secondo :func:`rank_key`; sintetizza (c) solo i pari merito."""
    cands = [c for c in cands if c.report is not None]
    if not cands:
        return None
    top = max((c.elaborated, c.pass_rate) for c in cands)
    tied = sorted((c for c in cands if (c.elaborated, c.pass_rate) == top),
                  key=lambda c: c.seq, reverse=True)
    if synth_fn is not None and top[0]:
        for c in tied[:max_synth]:
            if c.synth is None:
                try:
                    c.synth = synth_fn(list(c.report.verilog), c.tag)
                except Exception as e:  # noqa: BLE001 - la sintesi non deve far fallire la run
                    c.synth = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    return max(cands, key=rank_key)


def summary(c: Candidate | None, solution: str, primary: str = "generic") -> dict:
    """Campi del miglior candidato per result.json / metrics.json."""
    if c is None:
        return {"solution": "none", "tag": None, "criterion": CRITERION}
    passed, total = c.tests
    s = c.synth or {}
    return {"solution": solution, "tag": c.tag, "phase": c.phase, "origin": c.origin,
            "iteration": c.seq, "stage": c.report.stage, "elaborated": c.elaborated,
            "tests_passed": passed, "tests_total": total, "pass_rate": round(c.pass_rate, 6),
            "synth": {"backend": primary, "ok": bool(s.get("ok")) if s else None,
                      "skipped": s.get("skipped"), "error": s.get("error"),
                      **{k: v for k, v in s.items() if k not in ("ok", "skipped", "error")}}
            if s else None,
            "criterion": CRITERION}


def write_best(root: Path, c: Candidate | None, solution: str, primary: str = "generic",
               ranking: list[Candidate] | None = None) -> dict:
    """``<run>/best/``: sorgenti, Verilog, ``report.json``. Restituisce il summary."""
    out = Path(root) / "best"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    info = summary(c, solution, primary)
    if c is not None:
        for name, code in c.sources.items():
            (out / name).write_text(code, encoding="utf-8")
        copied = []
        for v in c.report.verilog:
            p = Path(v)
            if p.is_file():
                shutil.copyfile(p, out / p.name)
                copied.append(p.name)
        info["verilog"] = copied
    if ranking:
        info["ranking"] = [{"tag": x.tag, "phase": x.phase, "iteration": x.seq,
                            "stage": x.report.stage, "pass_rate": round(x.pass_rate, 6),
                            "synth_ok": x.synth_ok if x.synth is not None else None}
                           for x in sorted(ranking, key=rank_key, reverse=True)[:10]]
    import json
    (out / "report.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str),
                                     encoding="utf-8")
    return {k: v for k, v in info.items() if k != "ranking"}


def describe(info: dict | None, run_dir: Path | str) -> str:
    """Riga per lo stdout: soluzione completa o parziale, test, sintesi."""
    if not info or info.get("solution") == "none":
        return "miglior candidato: NESSUNO (nessun candidato verificato)"
    kind = {"complete": "SOLUZIONE COMPLETA", "partial": "SOLUZIONE PARZIALE"}[info["solution"]]
    t = (f"{info['tests_passed']}/{info['tests_total']} test" if info.get("tests_total")
         else f"stadio {info.get('stage')}, nessun test simulato")
    s = info.get("synth") or {}
    syn = ("sintesi ok" if s.get("ok") else "sintesi saltata" if s.get("skipped")
           else "sintesi fallita" if s else "sintesi non eseguita")
    return (f"miglior candidato: {kind} - {info['tag']} (iterazione {info['iteration']}), "
            f"{t}, {syn} -> {Path(run_dir) / 'best'}")
