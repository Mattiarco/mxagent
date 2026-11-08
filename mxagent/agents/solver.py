"""Risoluzione di un singolo modulo: genera (best-of-N) -> verifica -> rifletti, con anti-loop.

Usato per i blocchi del catalogo, per il top-level integrato e per il caso
monolitico. Ogni chiamata all'LLM è a turno singolo con memoria compatta
(riassunto degli ultimi tentativi), quindi resta sempre dentro ``num_ctx``.
"""
from __future__ import annotations

import re
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..bench.task import Task
from ..config import Config, ModelRole
from ..formats.cards import format_card
from ..llm.chat import ChatModel
from ..rag import kb
from ..rag.library import Library
from ..run.recorder import Recorder
from ..verify.verifier import VerifyReport, Verifier
from . import prompts
from .extract import extract_scala


@dataclass
class AgentContext:
    cfg: Config
    llm: ChatModel
    rec: Recorder
    library: Library | None
    seed: int
    task_name: str  # task della run (anti-leakage della libreria)
    counters: dict[str, int] = field(default_factory=lambda: {"candidates": 0, "sanitized": 0,
                                                              "kb_hits": 0, "no_code": 0})
    escalated: bool = False

    def role(self, name: str) -> ModelRole:
        roles = self.cfg.llm.roles
        if not self.cfg.agent.routing:
            return roles["coder"]
        if self.escalated and name in ("coder", "fixer", "reflector"):
            return roles.get("escalate", roles["planner"])
        if name == "optimizer" and name not in roles:
            return roles["reflector"]
        return roles[name]

    def chat(self, role: str, system: str, user: str, *, seed: int, temperature: float | None = None,
             fmt: Any = None, tag: str = ""):
        r = self.role(role)
        resp = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                             model=r.model, temperature=r.temperature if temperature is None else temperature,
                             num_ctx=r.num_ctx, num_predict=r.num_predict, seed=seed, format=fmt, tag=tag)
        self.rec.event("llm", role=role, model=r.model, tag=tag, seed=seed, prompt=user,
                       response=resp.text, tokens=[resp.prompt_tokens, resp.completion_tokens],
                       seconds=round(resp.seconds, 2), cached=resp.cached)
        return resp

    def system_prompt(self) -> str:
        rules = kb.rules_text() if self.cfg.agent.kb else "- Use Chisel 6 syntax."
        return prompts.render("full/system", rules=rules)

    def domain(self, task: Task) -> str:
        if not self.cfg.agent.format_card:
            return ""
        return "DOMAIN KNOWLEDGE (verified, follow it exactly)\n" + format_card(num=task.numerics_obj) + "\n"


@dataclass
class Attempt:
    report: VerifyReport | None
    sources: dict[str, str]
    origin: str  # gen | reflect | resample | reuse
    diagnosis: str = ""


@dataclass
class SolveResult:
    module: str
    passed: bool
    best: Attempt | None
    attempts: list[Attempt]
    escapes: list[str]
    stop: str


def _score(a: Attempt) -> tuple:
    if a.report is None:
        return (-1, 0.0, 0)
    return (*a.report.score, -sum(len(c) for c in a.sources.values()))


def strip_classes(code: str, names: set[str]) -> str:
    """Rimuove dal codice le definizioni di classi già fornite (submoduli verificati)."""
    for n in names:
        m = re.search(rf"^\s*(?:class|object)\s+{re.escape(n)}\b[^{{]*\{{", code, re.M)
        while m:
            depth, j = 0, m.end() - 1
            while j < len(code):
                depth += {"{": 1, "}": -1}.get(code[j], 0)
                if depth == 0:
                    break
                j += 1
            code = code[:m.start()] + code[j + 1:]
            m = re.search(rf"^\s*(?:class|object)\s+{re.escape(n)}\b[^{{]*\{{", code, re.M)
    return code


class ModuleSolver:
    def __init__(self, ctx: AgentContext, task: Task, verifier: Verifier, *, kind: str,
                 max_iters: int, extra_sources: dict[str, str] | None = None,
                 gen_prompt: Callable[[], str] | None = None, hint: str = ""):
        self.ctx, self.task, self.verifier, self.kind = ctx, task, verifier, kind
        self.hint = hint
        self.max_iters = max_iters
        self.extra = extra_sources or {}
        self.gen_prompt = gen_prompt or self._coder_prompt
        self.attempts: list[Attempt] = []
        self.memory: list[str] = []
        self.escapes: list[str] = []
        # seed deterministico per modulo (hash() di Python è randomizzato fra processi)
        self._seed = ctx.seed * 1000 + zlib.crc32(task.top.encode()) % 997

    # ------------------------------------------------------------- prompt
    def _examples(self) -> str:
        lib, cfg = self.ctx.library, self.ctx.cfg
        if not (lib and cfg.agent.library and cfg.library.examples):
            return ""
        excl = {self.ctx.task_name} if cfg.library.exclude_current_task else set()
        hits = lib.search(self.task.spec_text + "\n" + self.task.interface_text(),
                          k=cfg.library.examples, exclude_tasks=excl)
        hits = [(c, s) for c, s in hits if c.name != self.task.top]
        if not hits:
            return ""
        self.ctx.rec.event("rag", module=self.task.top, method=lib.last_method,
                           hits=[(c.id, round(s, 3)) for c, s in hits])
        parts = ["VERIFIED EXAMPLES (similar components from the library; they pass all tests):"]
        for c, _ in hits:
            parts.append(f"// {c.kind}: {c.spec.splitlines()[0][:120]}\n```scala\n{c.code[:2500]}\n```")
        return "\n".join(parts) + "\n"

    def _coder_prompt(self) -> str:
        return prompts.render("full/coder", top=self.task.top, spec=self.task.spec_text.strip(),
                              interface=self.task.interface_text(), domain=self.ctx.domain(self.task),
                              examples=self._examples(),
                              hint=f"IMPLEMENTATION PLAN\n{self.hint}\n" if self.hint else "")

    def _next_seed(self) -> int:
        self._seed += 1
        return self._seed

    # -------------------------------------------------------- valutazione
    def _evaluate(self, text: str, origin: str, diagnosis: str = "") -> Attempt:
        ctx = self.ctx
        code = extract_scala(text, self.task.top)
        n = len(self.attempts)
        if code is None:
            ctx.counters["no_code"] += 1
            a = Attempt(None, {}, origin, diagnosis)
            self.ctx.rec.event("verify", module=self.task.top, n=n, origin=origin, stage="no_code")
            self.attempts.append(a)
            return a
        fixes: list[str] = []
        if ctx.cfg.agent.kb:
            s = kb.sanitize(code)
            code, fixes = s.code, s.applied
            ctx.counters["sanitized"] += bool(fixes)
        if self.extra:
            code = strip_classes(code, {_class_name(k) for k in self.extra})
        sources = {**self.extra, f"{self.task.top}.scala": code}
        ctx.counters["candidates"] += 1
        rep = self.verifier.verify(sources, f"{self.task.top}_{n:02d}")
        a = Attempt(rep, sources, origin, diagnosis)
        self.attempts.append(a)
        ctx.rec.save_candidate(ctx.counters["candidates"], {f"{self.task.top}.scala": code})
        ctx.rec.event("verify", module=self.task.top, n=n, origin=origin, stage=rep.stage,
                      passed=rep.passed, score=list(rep.score), sanitized=fixes,
                      error_keys=rep.error_keys, feedback=rep.feedback[:3000],
                      seconds=rep.seconds)
        self.memory.append(self._summary(a, n))
        return a

    def _summary(self, a: Attempt, n: int) -> str:
        r = a.report
        rate = r.sim.get("pass_rate") if r and r.sim else None
        what = f"{r.stage}" + (f", {rate:.1%} vectors pass" if rate is not None else "")
        key = (r.error_keys[0] if r and r.error_keys else "")[:110]
        diag = f" diagnosis: {a.diagnosis[:160]}" if a.diagnosis else ""
        return f"attempt {n + 1} ({a.origin}): {what}.{diag} {('error: ' + key) if key else ''}".strip()

    @property
    def best(self) -> Attempt | None:
        cands = [a for a in self.attempts if a.report is not None]
        return max(cands, key=_score) if cands else None

    # ---------------------------------------------------------- strategie
    def generate(self, n: int, temp_boost: float = 0.0, origin: str = "gen") -> None:
        system = self.ctx.system_prompt()
        user = self.gen_prompt()
        role = self.ctx.role("coder")
        for i in range(n):
            t = min(1.2, role.temperature + temp_boost + (0.15 * i if n > 1 else 0.0))
            r = self.ctx.chat("coder", system, user, seed=self._next_seed(), temperature=t,
                              tag=f"{self.kind}:{self.task.top}:{origin}{i}")
            a = self._evaluate(r.text, origin)
            if a.report and a.report.passed:
                return

    def reflect(self) -> None:
        best = self.best
        if best is None:  # nessun codice valido finora: rigenera
            self.generate(1, temp_boost=0.2, origin="gen")
            return
        rep = best.report
        assert rep is not None
        cfg = self.ctx.cfg
        fb = rep.feedback if cfg.agent.structured_feedback else rep.feedback_raw
        pit = ""
        if cfg.agent.kb:
            hits = kb.matching(rep.feedback + "\n" + rep.feedback_raw)
            if hits:
                self.ctx.counters["kb_hits"] += len(hits)
                pit = "\nKNOWN CHISEL 6 PITFALLS MATCHING THIS ERROR:\n" + "\n\n".join(p.example() for p in hits) + "\n"
        mem = ""
        if cfg.agent.memory_items and len(self.memory) > 1:
            mem = "\nPREVIOUS ATTEMPTS (do not repeat failed approaches):\n" + "\n".join(
                f"- {m}" for m in self.memory[-cfg.agent.memory_items:]) + "\n"
        code = best.sources[f"{self.task.top}.scala"]
        user = prompts.render("full/reflect", top=self.task.top, spec=self.task.spec_text.strip(),
                              interface=self.task.interface_text(), domain=self.ctx.domain(self.task),
                              code=code, feedback=fb, pitfalls=pit, memory=mem)
        role = "fixer" if rep.stage in ("compile", "lint") else "reflector"
        r = self.ctx.chat(role, self.ctx.system_prompt(), user, seed=self._next_seed(),
                          tag=f"{self.kind}:{self.task.top}:reflect")
        m = re.search(r"DIAGNOSIS:\s*(.+?)(?:\n\s*\n|```|$)", r.text, re.S)
        self._evaluate(r.text, "reflect", m.group(1).strip().replace("\n", " ") if m else "")

    def _stuck(self, since: int) -> str | None:
        cfg = self.ctx.cfg.agent
        reps = [a.report for a in self.attempts[since:] if a.report is not None]
        keys = [tuple(r.error_keys) for r in reps]
        k = cfg.escape_after_repeats
        if len(keys) >= k + 1 and keys[-1] and all(x == keys[-1] for x in keys[-(k + 1):]):
            return f"same error {k + 1} times in a row"
        if len(keys) >= 4 and keys[-1] == keys[-3] and keys[-2] == keys[-4] and keys[-1] != keys[-2]:
            return "oscillation between two errors"
        s = cfg.escape_after_stall
        tail = self.attempts[since:]
        if len(tail) > s:
            before = max((_score(a) for a in tail[:-s]), default=(-1,))
            after = max(_score(a) for a in tail[-s:])
            if after[:2] <= before[:2]:
                return f"no improvement in {s} attempts"
        return None

    # --------------------------------------------------------------- loop
    def solve(self) -> SolveResult:
        cfg = self.ctx.cfg
        stop = "max_iters"
        self.generate(cfg.agent.best_of_n)
        since, level = 0, 0
        it = 0
        while not (self.best and self.best.report and self.best.report.passed):
            if it >= self.max_iters:
                break
            if cfg.agent.escape:
                why = self._stuck(since)
                if why:
                    level += 1
                    action = {1: "resample", 2: "escalate"}.get(level, "stop")
                    if action == "escalate" and self.ctx.escalated:
                        action = "stop"
                    self.escapes.append(f"{why} -> {action}")
                    self.ctx.rec.event("escape", module=self.task.top, reason=why, action=action)
                    if action == "stop":
                        stop = "escape_stop"
                        break
                    if action == "escalate":
                        self.ctx.escalated = True
                    self.memory.append(f"(strategy change after: {why})")
                    since = len(self.attempts)
                    if action == "resample":
                        self.generate(cfg.agent.best_of_n, temp_boost=0.4, origin="resample")
                        it += 1
                        continue
            self.reflect()
            it += 1
        b = self.best
        passed = bool(b and b.report and b.report.passed)
        self.ctx.rec.event("loop_stop", module=self.task.top, solver=self.kind,
                           reason="pass" if passed else stop, iters_done=it, iters_max=self.max_iters,
                           attempts=len(self.attempts))
        return SolveResult(self.task.top, passed, b, self.attempts, self.escapes,
                           "pass" if passed else stop)


def _class_name(filename: str) -> str:
    return filename.rsplit("/", 1)[-1].removesuffix(".scala")
