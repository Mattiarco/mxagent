"""Baseline per la valutazione (stesso modello, stesso verifier, stesso budget).

* **Baseline A** (``baseline-a``): single-shot con prompt ingenuo (specifica +
  interfaccia), una sola generazione, nessun feedback.
* **Baseline B** (``baseline-b``): agente generico generate -> compile -> test
  con i **log grezzi** degli strumenti e storia multi-turn completa (come un
  agente base o un ReChisel minimale), fino a successo, ``max_fix_iters`` o budget.

Nessuna conoscenza di dominio (scheda dei formati, regole Chisel 6, RAG,
decomposizione) entra nelle baseline: è ciò che il sistema completo aggiunge.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents import prompts
from ..agents.extract import extract_scala
from ..bench.task import Task
from ..config import Config, ModelRole
from ..llm.chat import Budget, BudgetExceeded, ChatModel, OllamaChat
from ..verify.verifier import VerifyReport, Verifier
from .recorder import Recorder, new_run_id

VerifyFn = Callable[[dict[str, str], str], VerifyReport]
MODES = ("baseline-a", "baseline-b")


@dataclass
class LoopSettings:
    max_fix_iters: int
    feedback: str = "raw"  # raw | structured
    prompt_set: str = "baseline"
    history: str = "full"  # full: storia multi-turn (troncata solo se supera num_ctx)


@dataclass
class RunResult:
    run_dir: Path
    metrics: dict[str, Any]
    best: VerifyReport | None = None
    reports: list[VerifyReport] = field(default_factory=list)


def settings_for(mode: str, cfg: Config) -> LoopSettings:
    if mode == "baseline-a":
        return LoopSettings(max_fix_iters=0)
    if mode == "baseline-b":
        return LoopSettings(max_fix_iters=cfg.budget.max_fix_iters, feedback="raw")
    raise ValueError(f"modalità sconosciuta: {mode} (note: {MODES})")


def _est_tokens(messages: list[dict[str, str]]) -> int:
    return sum(len(m["content"]) for m in messages) // 3 + 8 * len(messages)


def _fit_history(messages: list[dict[str, str]], num_ctx: int, reserve: int) -> int:
    """Rimuove le coppie (assistant, user) più vecchie dopo il primo messaggio
    utente finché la storia sta in ``num_ctx - reserve``. Restituisce quante ne ha tolte."""
    dropped = 0
    while _est_tokens(messages) > num_ctx - reserve and len(messages) > 4:
        del messages[2:4]
        dropped += 1
    return dropped


def run_loop(task: Task, cfg: Config, mode: str, *, seed: int, role: ModelRole,
             runs_dir: Path, llm: ChatModel | None = None, verify_fn: VerifyFn | None = None,
             verifier: Verifier | None = None, budget: Budget | None = None,
             run_id: str | None = None) -> RunResult:
    st = settings_for(mode, cfg)
    budget = budget or Budget(cfg.budget.max_llm_calls, cfg.budget.max_tokens, cfg.budget.max_wall_s)
    run_id = run_id or new_run_id(task.name, mode, role.model, seed)
    rec = Recorder(Path(runs_dir) / run_id)
    rec.write_text("config.yaml", cfg.dump())
    names = [f"{st.prompt_set}/system", f"{st.prompt_set}/task", f"{st.prompt_set}/fix"]
    if llm is None:
        llm = OllamaChat(cfg.llm.host, cfg.llm.cache_dir, cfg.llm.timeout_s, cfg.llm.max_retries,
                         budget=budget)
    elif getattr(llm, "budget", None) is None:
        llm.budget = budget  # type: ignore[attr-defined]
    budget = getattr(llm, "budget", None) or budget
    digest = llm.digest(role.model) if isinstance(llm, OllamaChat) else "scripted"
    rec.meta(task=task.name, mode=mode, seed=seed, model=role.model, model_digest=digest,
             role=role.model_dump(), prompts=prompts.fingerprint(*names),
             settings=st.__dict__, numerics=task.numerics_obj.to_dict())

    own_verifier = False
    if verify_fn is None:
        if verifier is None:
            verifier = Verifier(task, rec.root / "verify", sbt_mode=cfg.sim.sbt_mode,
                                random_vectors=cfg.sim.random_vectors, seed=cfg.sim.seed,
                                max_examples=cfg.sim.max_examples, sim_timeout=cfg.sim.timeout_s,
                                ddmin_rounds=cfg.sim.ddmin_rounds)
            own_verifier = True
        verify_fn = verifier.verify

    messages = [
        {"role": "system", "content": prompts.render(names[0])},
        {"role": "user", "content": prompts.render(names[1], spec=task.spec_text.strip(),
                                                    interface=task.interface_text())},
    ]
    t0 = time.monotonic()
    reports: list[VerifyReport] = []
    best: tuple[VerifyReport, dict[str, str]] | None = None
    stop, pass_iter, time_to_pass = "max_iters", None, None
    no_code = repeated = truncations = 0
    prev_keys: list[str] | None = None
    try:
        for it in range(st.max_fix_iters + 1):
            truncations += _fit_history(messages, role.num_ctx, role.num_predict)
            r = llm.chat(messages, model=role.model, temperature=role.temperature,
                         num_ctx=role.num_ctx, num_predict=role.num_predict,
                         seed=seed + it, tag=f"{mode}:{it}")
            rec.event("llm", iter=it, messages=messages[-1:] if it else messages,
                      response=r.text, tokens=[r.prompt_tokens, r.completion_tokens],
                      seconds=round(r.seconds, 2), cached=r.cached, done_reason=r.done_reason)
            code = extract_scala(r.text, task.top)
            if code is None:
                no_code += 1
                fb = "No ```scala code block was found in your answer."
                rec.event("verify", iter=it, stage="no_code", passed=False)
            else:
                sources = {f"{task.top}.scala": code}
                rec.save_candidate(it, sources)
                rep = verify_fn(sources, f"it{it:02d}")
                reports.append(rep)
                rec.event("verify", iter=it, stage=rep.stage, passed=rep.passed,
                          score=list(rep.score), seconds=rep.seconds, feedback=rep.feedback,
                          error_keys=rep.error_keys)
                if prev_keys is not None and rep.error_keys and rep.error_keys == prev_keys:
                    repeated += 1
                prev_keys = rep.error_keys
                if best is None or rep.score > best[0].score:
                    best = (rep, sources)
                if rep.passed:
                    stop, pass_iter, time_to_pass = "pass", it + 1, time.monotonic() - t0
                    break
                fb = rep.feedback_raw if st.feedback == "raw" else rep.feedback
            if it == st.max_fix_iters:
                break
            messages += [{"role": "assistant", "content": r.text},
                         {"role": "user", "content": prompts.render(names[2], feedback=fb)}]
    except BudgetExceeded as e:
        stop = f"budget: {e}"
    finally:
        if own_verifier and verifier is not None:
            verifier.close()

    if best:
        rec.save_final(best[1], best[0].verilog)
    b = best[0] if best else None
    metrics = {
        "task": task.name, "mode": mode, "model": role.model, "seed": seed,
        "passed": stop == "pass", "stop": stop,
        "iterations": len(reports) + no_code, "pass_iter": pass_iter,
        "time_to_pass_s": round(time_to_pass, 1) if time_to_pass else None,
        "wall_s": round(time.monotonic() - t0, 1),
        "best_stage": b.stage if b else "no_code",
        "best_pass_rate": b.sim.get("pass_rate", 0.0) if b else 0.0,
        "stages": [r.stage for r in reports],
        "no_code": no_code, "repeated_errors": repeated, "history_truncations": truncations,
        "llm": budget.to_dict(),
    }
    rec.write_json("metrics.json", metrics)
    return RunResult(rec.root, metrics, b, reports)
