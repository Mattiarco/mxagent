"""PPA Optimizer: migliora area/ritardo di un design già corretto.

Ogni candidato deve ripassare la verifica completa (stessi vettori, bit-exact)
prima di essere sintetizzato; viene accettato solo se la metrica primaria
migliora secondo ``synth.accept``. Il design di partenza resta il migliore
finché nessun candidato lo batte, quindi lo stadio PPA non può mai rendere
sbagliato un design corretto.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..bench.task import Task
from ..llm.chat import BudgetExceeded
from ..rag import kb
from ..synth.ppa import PPAReport, accept, objective
from ..verify.verifier import Verifier
from . import prompts
from .extract import extract_scala
from .solver import AgentContext, strip_classes

SynthFn = Callable[[list[str], str], PPAReport]  # (verilog, etichetta) -> report primario

_GOAL = {"area": "area", "delay": "logic depth (critical path)", "adp": "area x delay"}


@dataclass
class PPAStep:
    n: int
    outcome: str  # accepted | rejected | functional_fail | no_code | synth_fail
    detail: str
    primary: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return {"n": self.n, "outcome": self.outcome, "detail": self.detail,
                "primary": list(self.primary) if self.primary else None}


@dataclass
class PPAResult:
    initial: tuple[float, float] | None
    final: tuple[float, float] | None
    sources: dict[str, str]
    verilog: list[str]
    steps: list[PPAStep] = field(default_factory=list)
    stop: str = "iters"
    best_tag: str | None = None  # tag della verifica del candidato accettato per ultimo

    @property
    def accepted(self) -> int:
        return sum(s.outcome == "accepted" for s in self.steps)

    def to_dict(self, kind: str) -> dict:
        def obj(p):
            return round(objective(*p, kind), 3) if p else None
        gain = None
        if self.initial and self.final:
            o0, o1 = objective(*self.initial, kind), objective(*self.final, kind)
            gain = round((o0 - o1) / o0, 4) if o0 else None
        return {"objective": kind, "initial": list(self.initial) if self.initial else None,
                "final": list(self.final) if self.final else None,
                "objective_initial": obj(self.initial), "objective_final": obj(self.final),
                "gain": gain, "tried": len(self.steps), "accepted": self.accepted, "stop": self.stop,
                "best_tag": self.best_tag,
                "steps": [s.to_dict() for s in self.steps]}


class PPAOptimizer:
    def __init__(self, ctx: AgentContext, task: Task, verifier: Verifier, synth_fn: SynthFn, *,
                 extra_sources: dict[str, str] | None = None):
        self.ctx, self.task, self.verifier, self.synth_fn = ctx, task, verifier, synth_fn
        self.cfg = ctx.cfg.synth
        self.extra = extra_sources or {}
        self._seed = ctx.seed * 1000 + 7919

    def _prompt(self, code: str, rep: PPAReport, history: list[str]) -> str:
        c = self.cfg
        a, d = rep.primary(c) or (0, 0)
        obj_text = {"area": f"reduce area below {a:g} without increasing delay by more than "
                            f"{c.tolerance:.0%}",
                    "delay": f"reduce delay below {d:g} gate levels without increasing area by more "
                             f"than {c.tolerance:.0%}",
                    "adp": f"reduce area x delay below {a * d:g} (area {a:g}, delay {d:g})"}[c.objective]
        hist = ""
        if history:
            hist = "\nPREVIOUS OPTIMISATION ATTEMPTS:\n" + "\n".join(f"- {h}" for h in history) + "\n"
        return prompts.render("full/ppa", top=self.task.top, goal=_GOAL[c.objective],
                              spec=self.task.spec_text.strip(), interface=self.task.interface_text(),
                              domain=self.ctx.domain(self.task), code=code,
                              ppa=rep.render(), objective_text=obj_text, history=hist)

    def run(self, sources: dict[str, str], verilog: list[str], start: PPAReport) -> PPAResult:
        top_file = f"{self.task.top}.scala"
        best_src, best_v, best_rep = dict(sources), list(verilog), start
        best = start.primary(self.cfg)
        res = PPAResult(best, best, best_src, best_v)
        if best is None:
            res.stop = "no_primary_metric"
            self._stop(res)
            return res
        history: list[str] = []
        for i in range(self.cfg.optimize_iters):
            self._seed += 1
            try:
                r = self.ctx.chat("optimizer", self.ctx.system_prompt(),
                                  self._prompt(best_src[top_file], best_rep, history),
                                  seed=self._seed, tag=f"ppa:{self.task.top}:{i}")
            except BudgetExceeded as e:
                res.stop = f"budget: {e}"
                self.ctx.rec.event("budget_exhausted", limit=e.limit, detail=str(e), stage="PPA")
                break
            idea = re.search(r"IDEA:\s*(.+?)(?:\n\s*\n|```|$)", r.text, re.S)
            idea_s = idea.group(1).strip().replace("\n", " ")[:200] if idea else "(no idea given)"
            code = extract_scala(r.text, self.task.top)
            if code is None:
                step = PPAStep(i, "no_code", "no Scala code in the answer")
            else:
                if self.ctx.cfg.agent.kb:
                    code = kb.sanitize(code).code
                if self.extra:
                    code = strip_classes(code, {k.removesuffix(".scala") for k in self.extra})
                cand = {**self.extra, top_file: code}
                vr = self.verifier.verify(cand, f"{self.task.top}_ppa{i:02d}")
                self.ctx.rec.write_text(f"candidates/ppa{i:02d}/{top_file}", code)
                if not vr.passed:
                    lines = [ln.strip() for ln in vr.feedback.splitlines() if ln.strip()]
                    first = " ".join(lines[:2])[:240]  # intestazione + primo errore
                    step = PPAStep(i, "functional_fail", f"{vr.stage}: {first}")
                else:
                    pr = self.synth_fn(vr.verilog, f"ppa{i:02d}")
                    p = pr.primary(self.cfg)
                    if p is None:
                        errs = "; ".join(f"{k}: {v.error}" for k, v in pr.results.items() if not v.ok)
                        step = PPAStep(i, "synth_fail", errs[:300])
                    else:
                        ok, why = accept(best, p, self.cfg)
                        step = PPAStep(i, "accepted" if ok else "rejected", why, p)
                        if ok:
                            best, best_src, best_v, best_rep = p, cand, list(vr.verilog), pr
                            res.best_tag = vr.tag
            res.steps.append(step)
            history.append(f"attempt {i + 1}: {idea_s} -> {step.outcome}: {step.detail}")
            self.ctx.rec.event("ppa", module=self.task.top, n=i, outcome=step.outcome,
                               detail=step.detail, primary=list(step.primary) if step.primary else None)
        else:
            res.stop = "iters_exhausted"
        res.final, res.sources, res.verilog = best, best_src, best_v
        self._stop(res)
        return res

    def _stop(self, res: PPAResult) -> None:
        """Perché il loop è finito (iterazioni esaurite, budget, metrica assente)."""
        self.ctx.rec.event("ppa_stop", module=self.task.top, reason=res.stop,
                           iters_done=len(res.steps), iters_max=self.cfg.optimize_iters,
                           accepted=res.accepted)
