"""Sistema completo: macchina a stati esplicita.

    SPEC -> PLAN -> [BLOCK_i: riuso esatto | solve]* -> INTEGRATE -> PPA -> [WRAP] -> DONE
                 +-> MONOLITHIC (piano monolitico, o ripiego se un blocco/l'integrazione falliscono)

Lo stadio PPA (M5) parte solo dopo il pass e non cambia le metriche funzionali
(passed, time_to_pass): ottimizza area/ritardo e ogni candidato rifà la verifica.
Lo stadio WRAP (M7, solo per i task con ``wrapper: cvxif``) mette il design nel
coprocessore RISC-V CV-X-IF e ne esegue il test di protocollo.

Ogni transizione è un evento ``state`` nella trace. Il budget (chiamate, token,
tempo) è lo stesso delle baseline: il confronto è a pari costo.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from ..bench.subkernels import CATALOG, block_task, catalog_text
from ..bench.task import Task
from ..config import Config
from ..llm.chat import Budget, BudgetExceeded, ChatModel, OllamaChat, parse_json
from ..rag.embed import Embedder
from ..rag.library import Component, Library
from ..run.baselines import RunResult
from ..run.recorder import Recorder, new_run_id
from ..synth import yosys as _yosys
from ..synth.ppa import PPAReport, synthesize
from ..util.shell import run as _run
from ..verify import chisel
from ..verify.verifier import Verifier
from . import prompts
from .ppa import PPAOptimizer
from .schemas import PLAN_SCHEMA, SPEC_SCHEMA, Plan, validate_plan
from .solver import AgentContext, ModuleSolver, SolveResult

PROMPTS = ["full/system", "full/coder", "full/reflect", "full/spec", "full/planner", "full/integrate",
           "full/ppa"]


class Orchestrator:
    def __init__(self, task: Task, cfg: Config, *, seed: int, runs_dir: Path,
                 llm: ChatModel | None = None, verifier_factory=None,
                 library: Library | None = None, mode: str = "full", synth_fn=None,
                 run_id: str | None = None):
        self.task, self.cfg, self.seed, self.mode = task, cfg, seed, mode
        self.budget = Budget(cfg.budget.max_llm_calls, cfg.budget.max_tokens, cfg.budget.max_wall_s)
        coder = cfg.llm.roles["coder"].model
        self.rec = Recorder(Path(runs_dir) / (run_id or new_run_id(task.name, mode, coder, seed)))
        if llm is None:
            llm = OllamaChat(cfg.llm.host, cfg.llm.cache_dir, cfg.llm.timeout_s, cfg.llm.max_retries,
                             budget=self.budget)
        elif getattr(llm, "budget", None) is None:
            llm.budget = self.budget  # type: ignore[attr-defined]
        self.budget = getattr(llm, "budget", None) or self.budget
        self.llm = llm
        if library is None and cfg.agent.library:
            emb = cfg.llm.roles.get("embed")
            library = Library(cfg.library.path, Embedder(emb.model if emb else None, cfg.llm.host))
        self.library = library if cfg.agent.library else None
        self.ctx = AgentContext(cfg, llm, self.rec, self.library, seed, task.name)
        self._vf = verifier_factory or self._make_verifier
        self._synth = synth_fn or synthesize  # (verilog, task, cfg, work, backends) -> PPAReport
        self.sbt_ws = self.rec.root / "sbt"
        self.states: list[str] = []
        self._t_pass: float | None = None
        self._llm_at_pass: dict | None = None
        self.blocks: list[dict] = []

    # ------------------------------------------------------------ servizi
    def _make_verifier(self, task: Task, name: str) -> Verifier:
        s = self.cfg.sim
        return Verifier(task, self.rec.root / "verify" / name, sbt_mode=s.sbt_mode,
                        random_vectors=s.random_vectors, seed=s.seed, max_examples=s.max_examples,
                        sim_timeout=s.timeout_s, ddmin_rounds=s.ddmin_rounds,
                        sbt_workspace=self.sbt_ws)

    def _state(self, name: str, **data) -> None:
        self.states.append(name)
        self.rec.event("state", state=name, **data)

    # -------------------------------------------------------------- stati
    def spec(self) -> dict:
        if not self.cfg.agent.spec_parser:
            return {}
        self._state("SPEC")
        user = prompts.render("full/spec", spec=self.task.spec_text.strip(),
                              interface=self.task.interface_text())
        r = self.ctx.chat("planner", "You extract structured hardware specifications as JSON.",
                          user, seed=self.seed, fmt=SPEC_SCHEMA, tag="spec")
        data = r.data if isinstance(r.data, dict) else (parse_json(r.text) or {})
        self.rec.write_json("spec.json", data)
        return data

    def plan(self, structured: dict) -> Plan:
        if not self.cfg.agent.decompose:
            return Plan("monolithic", problems=["decomposizione disattivata (ablation)"])
        self._state("PLAN")
        user = prompts.render("full/planner", top=self.task.top, spec=self.task.spec_text.strip(),
                              interface=self.task.interface_text(),
                              structured=str(structured or "(not available)"), catalog=catalog_text())
        r = self.ctx.chat("planner", "You are a hardware architect. Answer with JSON only.", user,
                          seed=self.seed, fmt=PLAN_SCHEMA, tag="plan")
        p = validate_plan(r.data if r.data is not None else parse_json(r.text), self.task.top)
        self.rec.write_json("plan.json", p.to_dict())
        return p

    def _reuse(self, bt: Task, key: str, params: dict) -> SolveResult | None:
        lib, cfg = self.library, self.cfg
        if not (lib and cfg.library.reuse_exact):
            return None
        excl = {self.task.name} if cfg.library.exclude_current_task else set()
        for comp in lib.exact(f"block:{key}", params, excl)[:2]:
            code = comp.code.replace(f"class {comp.name}", f"class {bt.top}")
            v = self._vf(bt, bt.top)
            rep = v.verify({f"{bt.top}.scala": code}, f"{bt.top}_reuse")
            self.rec.event("reuse", module=bt.top, component=comp.id, passed=rep.passed)
            if rep.passed:
                from .solver import Attempt
                a = Attempt(rep, {f"{bt.top}.scala": code}, "reuse")
                return SolveResult(bt.top, True, a, [a], [], "reused")
        return None

    def _save(self, kind: str, name: str, params: dict, t: Task, code: str) -> None:
        if self.library is None or self.cfg.library.readonly:
            return
        self.library.add(Component(name=name, kind=kind, params=params, spec=t.spec_text,
                                   interface=t.interface_text(), code=code,
                                   origin_task=self.task.name, model=self.cfg.llm.roles["coder"].model,
                                   run=self.rec.root.name))

    def solve_block(self, b) -> SolveResult:
        bt = block_task(b.key, b.module, b.params, numerics=self.task.numerics)
        self._state("BLOCK", module=b.module, key=b.key, params=b.params)
        res = self._reuse(bt, b.key, b.params)
        if res is None:
            res = ModuleSolver(self.ctx, bt, self._vf(bt, b.module), kind="block",
                               max_iters=self.cfg.agent.max_block_iters).solve()
            if res.passed:
                self._save(f"block:{b.key}", b.module, b.params, bt,
                           res.best.sources[f"{b.module}.scala"])
        self.blocks.append({"module": b.module, "key": b.key, "params": b.params,
                            "passed": res.passed, "stop": res.stop,
                            "attempts": len(res.attempts), "escapes": res.escapes})
        return res

    def integrate(self, plan: Plan, solved: list[tuple]) -> SolveResult:
        self._state("INTEGRATE")
        extra, subs = {}, []
        for b, res in solved:
            extra[f"{b.module}.scala"] = res.best.sources[f"{b.module}.scala"]
            bt = block_task(b.key, b.module, b.params)
            subs.append(f"* {b.module} ({b.key}): {CATALOG[b.key].doc}\n  {bt.interface_text()}")

        def gen_prompt() -> str:
            return prompts.render("full/integrate", top=self.task.top, spec=self.task.spec_text.strip(),
                                  interface=self.task.interface_text(),
                                  domain=self.ctx.domain(self.task), submodules="\n".join(subs),
                                  glue=plan.glue or "(connect the submodules as needed)")

        return ModuleSolver(self.ctx, self.task, self._vf(self.task, "top"), kind="integrate",
                            max_iters=self.cfg.budget.max_fix_iters, extra_sources=extra,
                            gen_prompt=gen_prompt).solve()

    def monolithic(self, plan: Plan | None = None) -> SolveResult:
        self._state("MONOLITHIC")
        hint = plan.glue if plan and plan.strategy == "monolithic" else ""
        return ModuleSolver(self.ctx, self.task, self._vf(self.task, "mono"), kind="monolithic",
                            max_iters=self.cfg.budget.max_fix_iters, hint=hint).solve()

    def ppa(self, final: SolveResult) -> dict:
        """Report PPA del design corretto, PPA Optimizer, report del design finale."""
        s, top = self.cfg.synth, self.task.top
        best = final.best
        assert best is not None and best.report is not None
        self._state("PPA")
        work = self.rec.root / "ppa"
        t0 = time.monotonic()

        def primary(verilog: list[str], tag: str) -> PPAReport:
            return self._synth(verilog, self.task, s, work / tag, backends=[s.primary])

        final_backends = final_backend_list(s.backends)
        out: dict = {"tools": _yosys.tool_versions(),
                     "scope": {"loop": [s.primary], "final": final_backends}}
        # nel loop dell'optimizer si sintetizza solo la metrica primaria (velocità);
        # il design di partenza e quello finale passano su tutti i backend
        self.rec.event("ppa_scope", loop_backends=[s.primary], final_backends=final_backends,
                       note="optimizer loop uses only the primary backend; initial and final "
                            "designs are synthesized on all final_backends")
        start = primary(best.report.verilog, "initial")
        res = None
        if self.cfg.agent.ppa and s.optimize_iters > 0:
            extra = {k: v for k, v in best.sources.items() if k != f"{top}.scala"}
            opt = PPAOptimizer(self.ctx, self.task, self._vf(self.task, "ppa"), primary,
                               extra_sources=extra)
            res = opt.run(best.sources, best.report.verilog, start)
            out["optimizer"] = res.to_dict(s.objective)
        # report completo (tutti i backend) del design di partenza e, se cambiato, del finale
        others = [b for b in final_backends if b != s.primary]
        full0 = self._synth(best.report.verilog, self.task, s, work / "full_initial", backends=others)
        full0.results = {s.primary: start.results[s.primary], **full0.results}
        out["initial"] = full0.summary()
        self.final_ppa = full0
        if res and res.accepted:
            for name, code in best.sources.items():
                self.rec.write_text(f"final_functional/{name}", code)
            full1 = self._synth(res.verilog, self.task, s, work / "full_final", backends=final_backends)
            out["final"] = full1.summary()
            self.final_ppa = full1
            best.sources, best.report.verilog = res.sources, res.verilog
        else:
            out["final"] = out["initial"]
        out["seconds"] = round(time.monotonic() - t0, 1)
        self.rec.write_json("ppa.json", out)
        return out

    def _wrap(self, final: SolveResult) -> dict:
        """Coprocessore CV-X-IF attorno al design corretto e test di protocollo (M7)."""
        from ..wrap.cvxif import test_wrapper
        self._state("WRAP")
        best = final.best
        assert best is not None and best.report is not None
        try:
            r = test_wrapper(self.task, best.report.verilog, self.rec.root / "wrap",
                             seed=self.seed, sbt_mode="batch")
            out = {k: v for k, v in r.to_dict().items() if k != "errors"}
            out["errors"] = r.errors[:5]
        except Exception as e:  # noqa: BLE001 - errore di strumento: registrato, la run resta valida
            out = {"passed": False, "stage": "error", "errors": [f"{type(e).__name__}: {e}"]}
        self.rec.write_json("wrap.json", out)
        self.rec.event("wrap", **{k: v for k, v in out.items() if k != "errors"})
        return out

    def _ppa_safe(self, final: SolveResult) -> dict | None:
        """Lo stadio PPA non deve mai far fallire una run già corretta."""
        try:
            return self.ppa(final)
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001 - errore di strumento: registrato, run salva
            self.rec.event("ppa_error", error=f"{type(e).__name__}: {e}")
            return {"error": f"{type(e).__name__}: {e}"}

    # ---------------------------------------------------------------- run
    def run(self) -> RunResult:
        cfg, t0 = self.cfg, time.monotonic()
        self.rec.write_text("config.yaml", cfg.dump())
        coder = cfg.llm.roles["coder"].model
        digest = self.llm.digest(coder) if isinstance(self.llm, OllamaChat) else "scripted"
        self.rec.meta(task=self.task.name, mode=self.mode, seed=self.seed, model=coder,
                      model_digest=digest, roles={k: v.model_dump() for k, v in cfg.llm.roles.items()},
                      agent=cfg.agent.model_dump(), prompts=prompts.fingerprint(*PROMPTS),
                      library_size=len(self.library) if self.library else 0,
                      numerics=self.task.numerics_obj.to_dict())
        final: SolveResult | None = None
        plan = Plan("monolithic")
        stop, time_to_pass = "max_iters", None
        ppa_out: dict | None = None
        wrap_out: dict | None = None
        try:
            plan = self.plan(self.spec())
            if plan.strategy == "decompose":
                solved = []
                for b in plan.blocks:
                    res = self.solve_block(b)
                    if not res.passed:
                        self._state("FALLBACK", reason=f"blocco {b.module} non risolto")
                        break
                    solved.append((b, res))
                else:
                    final = self.integrate(plan, solved)
                    if not final.passed:
                        self._state("FALLBACK", reason="integrazione non riuscita")
                if final is None or not final.passed:
                    final = self.monolithic()
            else:
                final = self.monolithic(plan)
            stop = final.stop
            if final.passed:
                self._t_pass = time.monotonic()
                self._llm_at_pass = self.budget.to_dict()  # confronto a pari budget (M6)
            if final.passed and (self.cfg.agent.ppa or self.cfg.synth.backends):
                ppa_out = self._ppa_safe(final)
            if final.passed and self.task.wrapper == "cvxif":
                wrap_out = self._wrap(final)
        except BudgetExceeded as e:
            stop = f"budget: {e}"
        finally:
            if self.cfg.sim.sbt_mode == "client" and self.sbt_ws.is_dir():
                _run(chisel.sbt_cmd("client", "shutdown"), cwd=self.sbt_ws, timeout=60)

        passed = bool(final and final.passed)
        if passed:
            time_to_pass = self._t_pass - t0 if self._t_pass else time.monotonic() - t0
            self._state("DONE")
            self._save(f"task:{self.task.name}", self.task.top, {}, self.task,
                       final.best.sources[f"{self.task.top}.scala"])
        best = final.best if final else None
        if best:
            self.rec.save_final(best.sources, best.report.verilog if best.report else [])
        b = best.report if best else None
        c = self.ctx.counters
        metrics = {
            "task": self.task.name, "mode": self.mode, "model": coder, "seed": self.seed,
            "passed": passed, "stop": stop if not passed else "pass",
            "iterations": c["candidates"] + c["no_code"],
            "time_to_pass_s": round(time_to_pass, 1) if time_to_pass else None,
            "wall_s": round(time.monotonic() - t0, 1),
            "best_stage": b.stage if b else "no_code",
            "best_pass_rate": b.sim.get("pass_rate", 0.0) if b else 0.0,
            "strategy": plan.strategy, "plan_problems": plan.problems, "blocks": self.blocks,
            "states": self.states, "escapes": sum(len(x.get("escapes", [])) for x in self.blocks)
            + (len(final.escapes) if final else 0),
            "escalated": self.ctx.escalated, "no_code": c["no_code"],
            "sanitized_candidates": c["sanitized"], "kb_hits": c["kb_hits"],
            "llm": self.budget.to_dict(),
            "llm_to_pass": self._llm_at_pass,
            "ppa": _ppa_metrics(ppa_out, self.cfg.synth.primary, self.task.timing.target_mhz),
            "wrapper": wrap_out,
        }
        self.rec.write_json("metrics.json", metrics)
        return RunResult(self.rec.root, metrics, b, [a.report for a in (final.attempts if final else [])
                                                      if a.report])


def final_backend_list(backends: list[str]) -> list[str]:
    """Backend del report finale: quelli della config più Vivado, sempre presente
    (``skipped`` col motivo se non è nel PATH)."""
    return list(dict.fromkeys([*backends, "vivado"]))


def _ppa_metrics(out: dict | None, primary: str, target_mhz: float | None = None) -> dict | None:
    if not out:
        return None
    opt = out.get("optimizer") or {}
    m = {"initial": (out.get("initial") or {}).get(primary),
         "final": (out.get("final") or {}).get(primary),
         "gain": opt.get("gain"), "tried": opt.get("tried", 0), "accepted": opt.get("accepted", 0),
         "seconds": out.get("seconds")}
    if target_mhz:
        m["timing"] = timing_check((out.get("final") or {}).get("nextpnr"), target_mhz)
    return m


def timing_check(nextpnr: dict | None, target_mhz: float) -> dict:
    """Target di frequenza (b06): rispettato se la Fmax PEGGIORE sui seed lo raggiunge."""
    if not nextpnr or "fmax_min" not in nextpnr:
        return {"target_mhz": target_mhz, "fmax_mhz": None, "met": None}
    return {"target_mhz": target_mhz, "fmax_mhz": nextpnr["fmax_mhz"], "fmax_min": nextpnr["fmax_min"],
            "met": nextpnr["fmax_min"] >= target_mhz}


def run_full(task: Task, cfg: Config, *, seed: int, runs_dir: Path, **kw) -> RunResult:
    return Orchestrator(task, cfg, seed=seed, runs_dir=runs_dir, **kw).run()


__all__ = ["Orchestrator", "run_full", "asdict"]
