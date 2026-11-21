"""Baseline per la valutazione (stesso modello, stesso verifier, stesso budget).

* **Baseline A** (``baseline-a``): single-shot con prompt ingenuo (specifica +
  interfaccia), una sola generazione, nessun feedback.
* **Baseline B** (``baseline-b``): agente generico generate -> compile -> test
  con i **log grezzi** degli strumenti e storia multi-turn completa (come un
  agente base o un ReChisel minimale), fino a successo, ``max_fix_iters`` o budget.
  Al più ``1 + budget.max_fix_iters`` chiamate (7 con la config di default):
  invariata, i risultati già raccolti restano validi.
* **Baseline B a pari budget** (``baseline-b-matched``): identica a B (prompt, log
  grezzi, storia multi-turn, verifier) ma continua finché passa o finché esaurisce
  lo stesso :class:`~mxagent.llm.chat.Budget` del sistema completo
  (``max_llm_calls``, ``max_tokens``, ``max_wall_s``): ``max_fix_iters`` è
  ``budget.max_llm_calls - 1``, cioè al più ``max_llm_calls`` chiamate. Risponde
  alla domanda "il framework vince solo perché prova più volte?".

Storia multi-turn: quando il prompt stimato supera ``num_ctx - num_predict``,
:func:`_fit_history` toglie le coppie (risposta, feedback) più vecchie; il messaggio
di sistema e il primo messaggio utente (specifica e interfaccia) restano sempre,
così come l'ultimo feedback. È la condizione reale della baseline con molti turni:
quante coppie sono state tolte e in quante chiamate finisce in ``metrics.json``,
``result.json`` e negli eventi ``history_truncated`` del trace.

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
from ..agents.solver import log_llm_call
from ..bench.task import Task
from ..config import Config, ModelRole, profile_label
from ..llm.chat import Budget, BudgetExceeded, ChatModel, describe_llm, make_chat, model_digest
from ..util import procs
from ..verify.infra import infra_outcome
from ..verify.verifier import VerifyReport, Verifier
from . import best as _best
from .cleanup import clean_run
from .recorder import Recorder, new_run_id
from .result import (RunInterrupted, check_models, first_error, interruptible, llm_by_role,
                     expected_cache_origins, infra_retry_fields, llm_cache_fields,
                     llm_calls_to_pass, llm_result_fields, retry_event, write_result)

VerifyFn = Callable[[dict[str, str], str], VerifyReport]
MODES = ("baseline-a", "baseline-b", "baseline-b-matched")


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
    if mode == "baseline-b-matched":
        # stesso tetto di chiamate del sistema completo: la prima generazione + correzioni;
        # token e tempo li ferma il Budget, condiviso con il flusso completo
        return LoopSettings(max_fix_iters=max(0, cfg.budget.max_llm_calls - 1), feedback="raw")
    raise ValueError(f"modalità sconosciuta: {mode} (note: {MODES})")


def _est_tokens(messages: list[dict[str, str]]) -> int:
    return sum(len(m["content"]) for m in messages) // 3 + 8 * len(messages)


def _fit_history(messages: list[dict[str, str]], num_ctx: int, reserve: int) -> int:
    """Rimuove le coppie (assistant, user) più vecchie dopo il primo messaggio
    utente finché la storia sta in ``num_ctx - reserve``. Restituisce quante ne ha tolte.

    Restano sempre il messaggio di sistema, il primo messaggio utente e l'ultima
    coppia (risposta precedente + ultimo feedback): se nemmeno quelli stanno nel
    contesto la chiamata parte comunque e il troncamento lo fa il backend."""
    dropped = 0
    while _est_tokens(messages) > num_ctx - reserve and len(messages) > 4:
        del messages[2:4]
        dropped += 1
    return dropped


def run_loop(task: Task, cfg: Config, mode: str, *, seed: int, role: ModelRole,
             runs_dir: Path, llm: ChatModel | None = None, verify_fn: VerifyFn | None = None,
             verifier: Verifier | None = None, budget: Budget | None = None,
             run_id: str | None = None, synth_fn: _best.SynthFn | None = None) -> RunResult:
    st = settings_for(mode, cfg)
    budget = budget or Budget(cfg.budget.max_llm_calls, cfg.budget.max_tokens, cfg.budget.max_wall_s)
    profile, mism = profile_label(cfg, mode)
    if role.model != cfg.llm.roles["coder"].model:  # modello passato a mano, diverso dalla config
        profile, mism = f"all-{role.model}", {}
    models_by_role = {"coder": role.model}
    run_id = run_id or new_run_id(task.name, mode, profile, seed)
    rec = Recorder(Path(runs_dir) / run_id)
    rec.write_text("config.yaml", cfg.dump())
    names = [f"{st.prompt_set}/system", f"{st.prompt_set}/task", f"{st.prompt_set}/fix"]
    if llm is None:  # backend della config (ollama | anthropic): stessa interfaccia
        llm = make_chat(cfg.llm, budget=budget)
    elif getattr(llm, "budget", None) is None:
        llm.budget = budget  # type: ignore[attr-defined]
    budget = getattr(llm, "budget", None) or budget
    if hasattr(llm, "origin"):  # provenienza delle risposte che questa run mette in cache
        llm.origin = str(rec.root.resolve())
    digest = model_digest(llm, role.model)
    rec.meta(task=task.name, mode=mode, seed=seed, model=role.model, model_profile=profile,
             models_by_role=models_by_role, model_digest=digest,
             llm_backend=describe_llm(llm, {"coder": role}),
             role=role.model_dump(), prompts=prompts.fingerprint(*names),
             settings=st.__dict__, numerics=task.numerics_obj.to_dict())
    check_models(rec, profile, mism)

    own_verifier = False
    if verify_fn is None:
        if verifier is None:
            verifier = Verifier(task, rec.root / "verify", sbt_mode=cfg.sim.sbt_mode,
                                random_vectors=cfg.sim.random_vectors, seed=cfg.sim.seed,
                                max_examples=cfg.sim.max_examples, sim_timeout=cfg.sim.timeout_s,
                                ddmin_rounds=cfg.sim.ddmin_rounds, jvm_cpus=cfg.sim.jvm_cpus)
            own_verifier = True
        if getattr(verifier, "on_infra_retry", "x") is None:  # evento infra_retry nel trace
            verifier.on_infra_retry = lambda info: rec.event("infra_retry", **retry_event(info))
        verify_fn = verifier.verify

    messages = [
        {"role": "system", "content": prompts.render(names[0])},
        {"role": "user", "content": prompts.render(names[1], spec=task.spec_text.strip(),
                                                    interface=task.interface_text())},
    ]
    t0 = time.monotonic()
    reports: list[VerifyReport] = []
    best: tuple[VerifyReport, dict[str, str]] | None = None
    stop, cands_to_pass, time_to_pass = "max_iters", None, None
    rounds = 0  # cicli di correzione: chiamate dopo la prima, ognuna col feedback della precedente
    no_code = repeated = truncations = truncated_calls = 0
    prev_keys: list[str] | None = None
    outcome: dict = {"status": "failed", "reason": "run non terminata"}
    cands: list[_best.Candidate] = []  # ogni candidato verificato, per best/
    best_info: dict | None = None
    t_end: list[float] = []  # fine del lavoro (prima della sintesi di best/)

    def write_res() -> None:
        """result.json con lo stato corrente (le variabili sono lette alla chiamata)."""
        b = best[0] if best else None
        sim = (b.sim if b else None) or {}
        bud = budget.to_dict()
        try:
            write_result(rec, {
                "task": task.name, "mode": mode, "model": role.model, "model_profile": profile,
                "models_by_role": models_by_role, "seed": seed, **outcome,
                "passed": stop == "pass", "pass_rate": sim.get("pass_rate", 0.0),
                "n_vectors": sim.get("total"), "best_stage": b.stage if b else "no_code",
                "best_candidate": b.tag if b else None,
                "first_error": None if stop == "pass" else first_error(b),
                "candidates": len(reports), "no_code": no_code,
                "candidates_to_pass": cands_to_pass, "reflect_rounds": rounds,
                "time_to_pass_s": round(time_to_pass, 1) if time_to_pass else None,
                "wall_s": round((t_end[0] if t_end else time.monotonic()) - t0, 1),
                "llm_calls": bud["calls"],
                "tokens_in": bud["prompt_tokens"], "tokens_out": bud["completion_tokens"],
                "llm_by_role": llm_by_role(rec.events), **llm_result_fields(llm, bud),
                "solution": (best_info or {}).get("solution", "none"), "best": best_info,
                "llm_calls_to_pass": llm_calls_to_pass(rec.events, task.top) if stop == "pass"
                else None,
                **llm_cache_fields(rec.events, expected_cache_origins(rec.root)),
                **infra_retry_fields(rec.events),
                "history_truncations": truncations, "history_truncated_calls": truncated_calls,
                "max_fix_iters": st.max_fix_iters,
                "duplicates": 0, "ppa": None})
        except Exception as e:  # noqa: BLE001 - non deve mascherare l'errore originale
            rec.event("result_error", error=f"{type(e).__name__}: {e}")

    with interruptible():
        try:
            try:
                for it in range(st.max_fix_iters + 1):
                    rounds = it
                    dropped = _fit_history(messages, role.num_ctx, role.num_predict)
                    if dropped:
                        truncations += dropped
                        truncated_calls += 1
                        rec.event("history_truncated", iter=it, dropped_pairs=dropped,
                                  total_dropped=truncations, kept_messages=len(messages),
                                  est_tokens=_est_tokens(messages), num_ctx=role.num_ctx)
                    r = llm.chat(messages, model=role.model, temperature=role.temperature,
                                 num_ctx=role.num_ctx, num_predict=role.num_predict,
                                 seed=seed + it, tag=f"{mode}:{it}")
                    log_llm_call(rec, r, role="coder", model=role.model, tag=f"{mode}:{it}",
                                 iter=it, messages=messages[-1:] if it else messages)
                    code = extract_scala(r.text, task.top)
                    if code is None:  # anche una chiamata scaduta (r.error): tentativo fallito
                        no_code += 1
                        fb = "No ```scala code block was found in your answer."
                        rec.event("verify", iter=it, stage="no_code", passed=False,
                                  **({"llm_error": r.error} if r.error else {}))
                    else:
                        sources = {f"{task.top}.scala": code}
                        if budget.wall >= budget.max_wall_s:  # niente verifiche oltre il tempo
                            raise BudgetExceeded(f"tempo esaurito ({budget.wall:.0f}/"
                                                 f"{budget.max_wall_s:.0f}s)", "max_wall_s")
                        rep = verify_fn(sources, f"it{it:02d}")
                        reports.append(rep)
                        cands.append(_best.Candidate(rep.tag, sources, rep, seq=len(reports),
                                                     phase=mode, origin="gen" if it == 0 else "fix"))
                        rec.save_candidate(rep.tag, sources, {
                            "phase": mode, "module": task.top, "n": it,
                            "origin": "gen" if it == 0 else "fix", "stage": rep.stage,
                            "passed": rep.passed, "pass_rate": (rep.sim or {}).get("pass_rate"),
                            "n_vectors": (rep.sim or {}).get("total")})
                        rec.event("verify", iter=it, stage=rep.stage, passed=rep.passed,
                                  score=list(rep.score), seconds=rep.seconds, feedback=rep.feedback,
                                  error_keys=rep.error_keys,
                                  **({"infra_retries": rep.infra_retries} if rep.infra_retries
                                     else {}))
                        if prev_keys is not None and rep.error_keys and rep.error_keys == prev_keys:
                            repeated += 1
                        prev_keys = rep.error_keys
                        if best is None or rep.score > best[0].score:
                            best = (rep, sources)
                        if rep.passed:
                            stop, cands_to_pass, time_to_pass = "pass", len(reports), time.monotonic() - t0
                            break
                        fb = rep.feedback_raw if st.feedback == "raw" else rep.feedback
                    if it == st.max_fix_iters:
                        break
                    messages += [{"role": "assistant", "content": r.text},
                                 {"role": "user", "content": prompts.render(names[2], feedback=fb)}]
            except BudgetExceeded as e:
                stop = f"budget: {e}"
                rec.event("budget_exhausted", limit=e.limit, detail=str(e), stage="loop")
            outcome = {"status": "done" if stop == "pass" else "failed", "reason": stop}
            if best:
                rec.save_final(best[1], best[0].verilog)
            b = best[0] if best else None
            t_end.append(time.monotonic())
            best_info = _baseline_best(rec, cfg, task, cands, synth_fn)
            metrics = {
                "task": task.name, "mode": mode, "model": role.model, "seed": seed,
                "passed": stop == "pass", "stop": stop,
                "iterations": len(reports) + no_code, "candidates_to_pass": cands_to_pass,
                "reflect_rounds": rounds,
                "time_to_pass_s": round(time_to_pass, 1) if time_to_pass else None,
                "wall_s": round(t_end[0] - t0, 1),
                "best_stage": b.stage if b else "no_code",
                "best_pass_rate": b.sim.get("pass_rate", 0.0) if b else 0.0,
                "solution": best_info.get("solution"), "best": best_info,
                "stages": [r.stage for r in reports],
                "no_code": no_code, "repeated_errors": repeated, "history_truncations": truncations,
                "history_truncated_calls": truncated_calls, "max_fix_iters": st.max_fix_iters,
                "llm_calls_to_pass": llm_calls_to_pass(rec.events, task.top) if stop == "pass"
                else None,
                "llm": budget.to_dict(),
                **llm_cache_fields(rec.events, expected_cache_origins(rec.root)),
            }
            rec.write_json("metrics.json", metrics)
            return RunResult(rec.root, metrics, b, reports)
        except RunInterrupted as e:
            outcome = {"status": "interrupted", "reason": f"signal {e.signame}"}
            raise
        except BaseException as e:
            # errore di sistema (verifica ripetuta invano, OSError): run non valutabile,
            # stesso esito del sistema completo
            outcome = (infra_outcome(e) if isinstance(e, Exception) else None) or {
                "status": "failed", "reason": "exception", "error": f"{type(e).__name__}: {e}"[:500]}
            if outcome["status"] == "error":
                rec.event("infra_error", reason=outcome["reason"], error=outcome["error"])
            raise
        finally:
            if own_verifier and verifier is not None:
                verifier.close()
            try:  # nessuna JVM resta viva nella cartella della run
                killed = procs.kill_pids([p.pid for p in procs.java_under(rec.root)])
                if killed:
                    rec.event("jvm_killed", pids=killed)
            except Exception as e:  # noqa: BLE001
                rec.event("cleanup_error", error=f"{type(e).__name__}: {e}")
            if not cfg.runs.keep_build_artifacts:
                try:
                    rec.event("cleanup", **clean_run(rec.root))
                except Exception as e:  # noqa: BLE001 - la pulizia non deve far fallire la run
                    rec.event("cleanup_error", error=f"{type(e).__name__}: {e}")
            write_res()


def _baseline_best(rec: Recorder, cfg: Config, task: Task, cands: list,
                   synth_fn: "_best.SynthFn | None") -> dict:
    """``best/`` delle baseline: stesso criterio del sistema completo (run/best.py).
    Non deve mai far fallire la run."""
    try:
        s = cfg.synth
        if synth_fn is None and s.backends:
            from ..synth.ppa import synthesize

            def synth_fn(verilog: list[str], tag: str) -> dict | None:
                rep = synthesize(verilog, task, s, rec.root / "best_synth" / tag,
                                 backends=[s.primary])
                return rep.summary().get(s.primary)
        b = _best.select_best(cands, synth_fn)
        sol = "complete" if b is not None and b.report.passed else "partial"
        info = _best.write_best(rec.root, b, sol, s.primary, ranking=cands)
    except Exception as e:  # noqa: BLE001
        rec.event("best_error", error=f"{type(e).__name__}: {e}")
        info = {"solution": "none", "error": f"{type(e).__name__}: {e}"[:300]}
    rec.event("best", **{k: v for k, v in info.items()
                         if k in ("solution", "tag", "iteration", "tests_passed", "tests_total",
                                  "stage")})
    return info
