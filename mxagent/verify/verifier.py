"""Verifier deterministico (nessun LLM).

    sbt/Chisel/firtool  ->  lint Verilator  ->  cocotb: esaustivi + corner + random
                        ->  confronto bit-exact col golden  ->  riduzione dei casi

Restituisce un :class:`VerifyReport` serializzabile, con il feedback testuale
già compresso per il Reflector e un punteggio per ordinare i candidati.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..bench.kernels import get_kernel
from ..bench.task import Task
from . import chisel, lint
from .ddmin import complexity, minimize
from .feedback import SimSummary, compare, patterns, pick_representatives, table_rows
from .sim import Simulator

Vec = dict[str, int]
STAGES = ("compile", "lint", "sim", "pass")


@dataclass
class VerifyReport:
    tag: str
    task: str
    stage: str  # dove si è fermato: compile | lint | sim | pass
    passed: bool
    compile: dict = field(default_factory=dict)
    lint: dict = field(default_factory=dict)
    sim: dict = field(default_factory=dict)
    feedback: str = ""
    #: feedback "da agente base": log grezzi degli strumenti (Baseline B e ablation)
    feedback_raw: str = ""
    seconds: dict[str, float] = field(default_factory=dict)
    verilog: list[str] = field(default_factory=list)
    error_keys: list[str] = field(default_factory=list)  # firme per l'anti-loop

    @property
    def score(self) -> tuple[int, float]:
        """Ordinamento dei candidati: stadio raggiunto, poi frazione di vettori ok."""
        return STAGES.index(self.stage), float(self.sim.get("pass_rate", 0.0))

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["score"] = list(self.score)
        return d

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class Verifier:
    def __init__(self, task: Task, work_dir: Path, *, sbt_mode: str = "batch",
                 random_vectors: int = 2000, seed: int = 0, max_examples: int = 3,
                 sim_timeout: float = 900, ddmin_rounds: int = 8, max_vectors: int = 200_000,
                 sbt_workspace: Path | None = None):
        self.task = task
        self.work = Path(work_dir)
        # workspace sbt condivisibile fra più Verifier della stessa run (un solo server JVM)
        self.sbt_ws = Path(sbt_workspace) if sbt_workspace else self.work / "sbt"
        self.sbt_mode = sbt_mode
        self.seed = seed
        self.n_random = random_vectors
        self.max_examples = max_examples
        self.sim_timeout = sim_timeout
        self.ddmin_rounds = ddmin_rounds
        self.max_vectors = max_vectors
        self.kernel = get_kernel(task.golden)
        self.num = task.numerics_obj
        self._vecs: list[tuple[str, Vec]] | None = None

    # ------------------------------------------------------------- vettori
    def golden(self, v: Vec) -> Vec:
        return self.kernel.fn(v, self.num, self.task.params)

    def vectors(self) -> list[tuple[str, Vec]]:
        """(suite, vettore): esaustivi, poi corner, poi random (deterministici dal seed)."""
        if self._vecs is not None:
            return self._vecs
        k, plan, p = self.kernel, self.task.vectors, self.task.params
        out: list[tuple[str, Vec]] = []
        if plan.exhaustive:
            if k.space is None:
                raise ValueError(f"il kernel {self.task.golden} non supporta test esaustivi")
            for v in k.space(p):
                out.append(("exhaustive", v))
                if len(out) > self.max_vectors:
                    raise ValueError("spazio troppo grande per un test esaustivo")
        if plan.corner:
            out += [("corner", v) for v in k.corner(p)]
        n_rand = plan.random or (0 if plan.exhaustive else self.n_random)
        if n_rand and k.random is not None:
            rng = random.Random(self.seed)
            out += [("random", k.random(rng, p)) for _ in range(n_rand)]
        names = [q.name for q in self.task.inputs]
        for _, v in out:
            if sorted(v) != sorted(names):
                raise ValueError(f"il kernel produce porte {sorted(v)}, il task dichiara {sorted(names)}")
        self._vecs = out
        return out

    def _rows(self, vecs: list[Vec]) -> list[list[int]]:
        names = [p.name for p in self.task.inputs]
        return [[v[n] for n in names] for v in vecs]

    # ------------------------------------------------------------- verifica
    def verify(self, sources: dict[str, str], tag: str) -> VerifyReport:
        t0 = time.monotonic()
        safe = re.sub(r"\W", "_", tag)
        pkg = f"cand_{safe}"
        job = self.work / safe
        job.mkdir(parents=True, exist_ok=True)
        rep = VerifyReport(tag, self.task.name, "compile", False)

        # 1. compilazione + elaborazione
        ws = chisel.prepare_workspace(self.sbt_ws)
        chisel.write_sources(ws, pkg, sources)
        el = chisel.elaborate(ws, pkg, self.task.top, job / "verilog", mode=self.sbt_mode)
        (job / "elaborate.log").write_text(el.log, encoding="utf-8")
        rep.compile, rep.seconds["compile"] = el.to_dict(), round(el.seconds, 2)
        if not el.ok:
            rep.error_keys = [e.key() for e in el.errors]
            rep.feedback = "COMPILATION FAILED:\n" + "\n".join(e.render() for e in el.errors[:5])
            rep.feedback_raw = raw_build_log(el.log)
            return self._done(rep, job, t0)
        rep.verilog = [str(p) for p in el.verilog]

        # 2. lint di sintetizzabilità
        rep.stage = "lint"
        li = lint.lint(el.verilog, self.task.top)
        (job / "lint.log").write_text(li.log, encoding="utf-8")
        rep.lint, rep.seconds["lint"] = li.to_dict(), round(li.seconds, 2)
        hard = [i for i in li.issues if i.severity == "error"]
        if hard:
            rep.error_keys = [f"lint:{i.code}:{i.message[:80]}" for i in hard]
            rep.feedback = "VERILATOR LINT ERRORS:\n" + "\n".join(i.render() for i in hard[:8])
            rep.feedback_raw = _tail(li.log, 40)
            return self._done(rep, job, t0)
        lint_fb = ""
        if li.blocking:  # si simula comunque, ma il candidato non può passare
            lint_fb = ("SYNTHESIZABILITY WARNINGS (must be fixed):\n"
                       + "\n".join(i.render() for i in li.blocking[:8]))

        # 3. simulazione
        rep.stage = "sim"
        sv = Simulator(self.task, el.verilog, job / "sim", timeout=self.sim_timeout)
        suites_vecs = self.vectors()
        suites = [s for s, _ in suites_vecs]
        vecs = [v for _, v in suites_vecs]
        run = sv.run(self._rows(vecs))
        rep.seconds["sim_build"] = round(sv.build_seconds, 2)
        rep.seconds["sim"] = round(run.seconds, 2)
        (job / "sim.log").write_text(run.log, encoding="utf-8")
        if not run.ok:
            rep.error_keys = [f"sim:{run.error}"]
            rep.sim = {"error": run.error, "pass_rate": 0.0}
            rep.feedback = f"SIMULATION ERROR: {run.error}\n" + _tail(run.log, 25)
            rep.feedback_raw = _tail(run.log, 40)
            return self._done(rep, job, t0)

        expected = [self.golden(v) for v in vecs]
        mism = compare(self.task, vecs, expected, run.outputs, suites)
        summ = SimSummary(total=len(vecs), failed=len(mism))
        for s in dict.fromkeys(suites):
            summ.per_suite[s] = [suites.count(s), sum(1 for m in mism if m.suite == s)]
        if mism:
            failing = {m.index for m in mism}
            summ.patterns = patterns(self.task, vecs, failing)
            reps = pick_representatives(self.task, mism, self.max_examples, complexity)
            summ.examples, summ.ddmin_rounds = self._minimize(sv, reps)
            if len(vecs) <= 64 and len(mism) <= 24:  # spazio piccolo: tutti i casi, compatti
                summ.table = table_rows(self.task, mism)
            rep.error_keys = [f"mismatch:{','.join(m.bad_ports)}:{p}" for m in summ.examples[:1]
                              for p in summ.patterns[:1]] or ["mismatch"]
        rep.sim = summ.to_dict()
        raw_lint = ("\n\nVerilator lint:\n" + "\n".join(
            ln for ln in li.log.splitlines() if ln.startswith("%"))[:3000]) if lint_fb else ""
        if mism:
            rep.feedback = summ.render(self.task) + ("\n\n" + lint_fb if lint_fb else "")
            rep.feedback_raw = raw_mismatches(self.task, mism, len(vecs)) + raw_lint
        elif lint_fb:
            rep.feedback = lint_fb
            rep.feedback_raw = raw_lint.strip()
            rep.error_keys = [f"lint:{i.code}" for i in li.blocking]
        else:
            rep.stage, rep.passed = "pass", True
            rep.feedback = rep.feedback_raw = f"ALL {summ.total} TEST VECTORS PASS (bit-exact)."
        return self._done(rep, job, t0)

    def _minimize(self, sv: Simulator, reps):
        """Delta-debugging dei mismatch rappresentativi (solo se gli ingressi sono vettori)."""
        if not any(p.elems > 1 for p in self.task.inputs) or self.ddmin_rounds <= 0:
            return reps, 0
        outs = [p.name for p in self.task.outputs]
        total = 0

        def still_fails(cands: list[Vec]) -> list[bool]:
            r = sv.run(self._rows(cands))
            if not r.ok:
                return [False] * len(cands)
            return [row is None or dict(zip(outs, row)) != self.golden(c)
                    for c, row in zip(cands, r.outputs)]

        out = []
        for m in reps:
            v, rounds = minimize(self.task, m.inputs, still_fails, self.ddmin_rounds)
            total += rounds
            if v is not m.inputs:
                r = sv.run(self._rows([v]))
                got = dict(zip(outs, r.outputs[0])) if r.ok and r.outputs[0] else {k: None for k in outs}
                from .feedback import Mismatch
                m = Mismatch(m.index, m.suite, v, self.golden(v), got, minimized_from=m.index)
            out.append(m)
        return out, total

    def close(self) -> None:
        """Chiude il server sbt (solo modalità client)."""
        ws = self.sbt_ws
        if self.sbt_mode == "client" and ws.is_dir():
            from ..util.shell import run as _run
            _run(chisel.sbt_cmd("client", "shutdown"), cwd=ws, timeout=60)

    def __enter__(self) -> "Verifier":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _done(self, rep: VerifyReport, job: Path, t0: float) -> VerifyReport:
        rep.seconds["total"] = round(time.monotonic() - t0, 2)
        rep.save(job / "report.json")
        return rep


def raw_build_log(log: str, n: int = 40) -> str:
    """Righe [error] del log sbt così come sono (quello che vede un agente base)."""
    errs = [ln for ln in log.splitlines() if ln.startswith("[error]")]
    text = "\n".join(errs[:n]) or _tail(log, n)
    # unica normalizzazione (equità): percorso -> nome file e righe riportate al
    # sorgente dell'LLM, togliendo le 2 righe del package iniettato dal verifier
    return re.sub(r"(?:[A-Za-z]:)?[\w./\\-]*/cand_\w+/(\w+\.scala):(\d+)",
                  lambda m: f"{m[1]}:{int(m[2]) - 2}", text)


def raw_mismatches(task: Task, mism, total: int, n: int = 5) -> str:
    """Stampa da testbench generico: interi esadecimali, nessuna scomposizione."""
    lines = [f"TEST FAILED: {len(mism)} of {total} test vectors mismatched. First {min(n, len(mism))}:"]
    for m in mism[:n]:
        ins = ", ".join(f"{k}=0x{v:X}" for k, v in m.inputs.items())
        exp = ", ".join(f"{k}=0x{v:X}" for k, v in m.expected.items())
        got = ", ".join(f"{k}={'X' if v is None else f'0x{v:X}'}" for k, v in m.got.items())
        lines.append(f"  inputs: {ins} | expected: {exp} | got: {got}")
    return "\n".join(lines)


def _tail(log: str, n: int) -> str:
    return "\n".join(log.strip().splitlines()[-n:])
