#!/usr/bin/env python3
"""Confronto fra backend LLM (Ollama locale vs Claude via API) sulla stessa pipeline.

Scopo: verificare se la metodologia dipende dal modello e, se sì, quanto e perché.
Ogni configurazione (backend + modello) gira sugli STESSI scenari (task x modalità),
con la stessa config, gli stessi prompt, lo stesso verifier, lo stesso budget e n
seed (0..n-1). Ogni run è un sottoprocesso ``mxagent run`` (isolato e riprendibile:
una run con ``result.json`` già presente non si rifà). Il reference design di ogni
task è sintetizzato una volta (``reference/<task>/``) come termine di paragone PPA.

Uso::

    python scripts/compare_backends.py \\
        --configs ollama:qwen2.5-coder:14b,anthropic:claude-sonnet-5-5 \\
        --tasks b01_e2m1_decode,b03_e8m0_to_fp32,b05_mxfp4_dot32 --modes full --n 10 \\
        --out runs/compare_<data>
    python scripts/compare_backends.py --out runs/compare_<data> --report-only   # solo report

Uscite in ``--out``:

``runs/<id>/``         le run (result.json, metrics.json, trace.json, best/, ...)
``compare_runs.csv``   una riga per run
``compare_summary.csv`` metriche per backend/modello/scenario
``compare_errors.csv`` fallimenti per categoria (analisi degli errori)
``compare.md``         tabelle, test statistici (Fisher, Mann-Whitney) e analisi degli errori

Ogni run ha una libreria propria e vuota (``library.path`` per run): nessuna run impara
da un'altra, i risultati non dipendono dall'ordine né dal backend delle run precedenti.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mxagent.eval.stats import fisher_exact, mann_whitney_u, mean_std, wilson  # noqa: E402
from mxagent.util.shell import run as sh  # noqa: E402

DEFAULT_CONFIGS = "ollama:qwen2.5-coder:14b,anthropic:claude-sonnet-5-5"
DEFAULT_TASKS = "b01_e2m1_decode,b03_e8m0_to_fp32,b05_mxfp4_dot32"


# ------------------------------------------------------------------ job
def parse_configs(s: str) -> list[tuple[str, str]]:
    out = []
    for item in s.split(","):
        backend, _, model = item.strip().partition(":")
        if backend not in ("ollama", "anthropic") or not model:
            raise SystemExit(f"--configs: '{item}' non è backend:modello (backend ollama|anthropic)")
        out.append((backend, model))
    return out


def label(backend: str, model: str) -> str:
    return re.sub(r"[^\w.-]", "-", f"{backend}-{model}")


def run_id(task: str, mode: str, backend: str, model: str, seed: int) -> str:
    return f"{task}_{mode}_{label(backend, model)}_s{seed}"


def jobs(a) -> list[dict]:
    out = []
    for task in a.tasks.split(","):
        for mode in a.modes.split(","):
            for backend, model in parse_configs(a.configs):
                for seed in range(a.seed_start, a.seed_start + a.n):
                    out.append({"task": task, "mode": mode, "backend": backend, "model": model,
                                "seed": seed, "id": run_id(task, mode, backend, model, seed)})
    return out


def command(a, j: dict, out: Path) -> list[str]:
    sets = [f"library.path={(out / 'libraries' / j['id']).as_posix()}", *a.set]
    cmd = [sys.executable, "-m", "mxagent", "run", "--task", str(Path(a.bench_dir) / j["task"]),
           "--mode", j["mode"], "--backend", j["backend"], "--model", j["model"],
           "--seed", str(j["seed"]), "--config", a.config, "--runs-dir", str(out / "runs"),
           "--run-id", j["id"], "--set", *sets]
    if a.llm_timeout:
        cmd += ["--llm-timeout", str(a.llm_timeout)]
    return cmd


def execute(a, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    js = jobs(a)
    (out / "compare.json").write_text(json.dumps({
        "configs": a.configs, "tasks": a.tasks, "modes": a.modes, "n": a.n,
        "seed_start": a.seed_start, "config": a.config, "set": a.set,
        "llm_timeout": a.llm_timeout, "jobs": len(js)}, indent=1), encoding="utf-8")
    if not a.no_reference:
        for task in a.tasks.split(","):
            ref = out / "reference" / task
            if (ref / "ppa.json").is_file():
                continue
            r = sh([sys.executable, "-m", "mxagent", "synth", "--task", str(Path(a.bench_dir) / task),
                    "--config", a.config, "--backends", a.ref_backends, "--out", str(ref),
                    "--tag", "compare_ref"], cwd=ROOT, timeout=a.job_timeout)
            print(f"reference {task}: {'ok' if (ref / 'ppa.json').is_file() else 'ERRORE'}"
                  + ("" if (ref / "ppa.json").is_file() else f" ({(r.stderr or r.stdout)[-200:]})"))
    todo = [j for j in js if not _finished(out / "runs" / j["id"])]
    print(f"{len(js)} run nel confronto, {len(todo)} da eseguire")
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    for i, j in enumerate(todo, 1):
        cmd = command(a, j, out)
        if a.dry_run:
            print(" ".join(cmd))
            continue
        r = sh(cmd, cwd=ROOT, timeout=a.job_timeout, grace=120)
        (logs / f"{j['id']}.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        res = _read(out / "runs" / j["id"] / "result.json") or {}
        print(f"[{i}/{len(todo)}] {j['id']}: rc={r.returncode}"
              + (" TIMEOUT" if r.timed_out else "")
              + f" status={res.get('status')} solution={res.get('solution')} ({r.seconds:.0f} s)")


# -------------------------------------------------------------- lettura
def _infra_error(res: dict) -> bool:
    """Run finita per un errore di infrastruttura (eccezione: rete, autenticazione, bug) o
    interrotta: non è un esito del modello, resta fuori dalle statistiche e si riesegue."""
    return res.get("status") == "interrupted" or res.get("reason") == "exception"


def _finished(run_dir: Path) -> bool:
    res = _read(run_dir / "result.json")
    return bool(res) and not _infra_error(res)


def _read(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


#: classi di ingresso (verify/values.py) -> categoria del caso che fallisce
CASE_CLASSES = {"nan": "NaN", "inf": "Inf", "subnormal": "subnormal", "zero": "zero",
                "max": "max/overflow", "min": "min scale", "neg": "sign"}


def mismatch_cases(feedback: str) -> list[str]:
    """Tipi di caso dei vettori che falliscono, dai pattern del feedback strutturato
    (``every failing vector has a_scale=nan; ...``). Porte di scala -> ``scaling``."""
    cases = set()
    for line in (feedback or "").splitlines():
        if "failing vector has" not in line:
            continue
        cond = line.split("failing vector has", 1)[1].split(";", 1)[0]
        for port, cls in re.findall(r"(\w+)=(\w+)", cond):
            if cls in ("nan", "inf"):
                cases.add(CASE_CLASSES[cls])
            elif "scale" in port.lower() or "shared" in port.lower() or "exp" in port.lower():
                cases.add("scaling")
            elif cls in CASE_CLASSES:
                cases.add(CASE_CLASSES[cls])
    return sorted(cases) or ["no dominant input class"]


def classify(ev: dict) -> list[str]:
    """Categoria di un candidato fallito (evento ``verify`` del trace)."""
    stage = ev.get("stage")
    if stage == "no_code":
        return ["timeout" if ev.get("llm_error") == "timeout" else "unparsable output (no Scala block)"]
    keys = ev.get("error_keys") or []
    k0 = keys[0] if keys else ""
    if stage == "compile":
        if k0.startswith("scalac:"):
            return ["Chisel/Scala compile error"]
        if k0.startswith(("elaboration:", "firtool:")):
            return ["elaboration error"]
        return ["build error (sbt)"]
    if stage == "lint":
        return ["Verilator lint error"]
    if stage == "sim":
        if k0.startswith("sim:"):
            return ["simulation error"]
        return [f"tests failed: {c}" for c in mismatch_cases(ev.get("feedback", ""))]
    return []


def load(out: Path) -> tuple[list[dict], list[dict], dict]:
    meta = _read(out / "compare.json") or {}
    refs = {}
    for p in (out / "reference").glob("*/ppa.json") if (out / "reference").is_dir() else []:
        d = _read(p) or {}
        g = (d.get("results") or {}).get("generic") or {}
        refs[p.parent.name] = (g.get("metrics") or {}) if g.get("ok") else {}
    rows, errors = [], []
    for rd in sorted((out / "runs").glob("*")) if (out / "runs").is_dir() else []:
        res = _read(rd / "result.json")
        if not res:
            continue
        m = _read(rd / "metrics.json") or {}
        trace = _read(rd / "trace.json") or []
        mt = _read(rd / "meta.json") or {}
        best = res.get("best") or {}
        bsyn = best.get("synth") or {}
        fin = ((res.get("ppa") or {}).get("final") or {}).get("generic") or {}
        backend = (mt.get("llm_backend") or {}).get("backend") or res.get("llm_backend")
        row = {
            "run": rd.name, "infra_error": _infra_error(res), "error": res.get("error"), "task": res.get("task"), "mode": res.get("mode"),
            "backend": backend, "model": res.get("model"), "seed": res.get("seed"),
            "status": res.get("status"), "reason": res.get("reason"),
            "passed": bool(res.get("passed")), "solution": res.get("solution"),
            "best_pass_rate": best.get("pass_rate", res.get("pass_rate")),
            "best_tests_passed": best.get("tests_passed"), "best_tests_total": best.get("tests_total"),
            "best_stage": best.get("stage", res.get("best_stage")),
            "best_iteration": best.get("iteration"),
            "candidates": res.get("candidates"), "candidates_to_pass": res.get("candidates_to_pass"),
            "reflect_rounds": res.get("reflect_rounds"),
            "llm_calls": res.get("llm_calls"), "llm_timeouts": res.get("llm_timeouts"),
            "llm_failed_calls": res.get("llm_failed_calls"),
            "llm_seconds": res.get("llm_seconds"), "wall_s": res.get("wall_s"),
            "time_to_pass_s": res.get("time_to_pass_s"),
            "tokens_in": res.get("tokens_in"), "tokens_out": res.get("tokens_out"),
            "cost_usd": res.get("cost_usd"),
            "best_synth_ok": bsyn.get("ok") if bsyn else None,
            "transistors": bsyn.get("transistors", fin.get("transistors")),
            "depth": bsyn.get("depth", fin.get("depth")),
            "strategy": m.get("strategy"),
        }
        rows.append(row)
        plan = _read(rd / "plan.json") or {}
        if any("JSON assente" in p for p in plan.get("problems") or []):
            errors.append({**_key(row), "category": "unparsable output (plan JSON)", "count": 1})
        for ev in trace:
            if ev.get("kind") == "verify" and not ev.get("passed"):
                for cat in classify(ev):
                    errors.append({**_key(row), "category": cat, "count": 1})
    return rows, errors, {"meta": meta, "refs": refs}


def _key(r: dict) -> dict:
    return {"run": r["run"], "task": r["task"], "mode": r["mode"], "backend": r["backend"],
            "model": r["model"]}


# --------------------------------------------------------------- report
def _ms(xs, fmt="{:.2f}") -> str:
    d = mean_std(xs)
    if d["mean"] is None:
        return "-"
    return fmt.format(d["mean"]) + (f" ± {fmt.format(d['std'])}" if d["std"] is not None else "")


def _p(p) -> str:
    return "-" if p is None else (f"{p:.3g}" if p >= 1e-4 else "<1e-4")


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["mode"], r["backend"], r["model"])].append(r)
    out = []
    for (task, mode, backend, model), allr in sorted(groups.items()):
        rs = [r for r in allr if not r["infra_error"]]
        n = len(rs)
        k = sum(r["passed"] for r in rs)
        lo, hi = wilson(k, n)
        succ = [r for r in rs if r["passed"]]
        syn = [r["best_synth_ok"] for r in rs if r["best_synth_ok"] is not None]
        out.append({
            "task": task, "mode": mode, "backend": backend, "model": model, "runs": n,
            "infra_errors": len(allr) - n,
            "complete": k, "partial": sum(r["solution"] == "partial" for r in rs),
            "success_rate": round(k / n, 4) if n else 0.0,
            "success_ci95": f"[{lo:.2f}, {hi:.2f}]",
            "best_pass_rate": _ms([r["best_pass_rate"] for r in rs], "{:.3f}"),
            "candidates_to_pass": _ms([r["candidates_to_pass"] for r in succ], "{:.1f}"),
            "reflect_rounds": _ms([r["reflect_rounds"] for r in rs], "{:.1f}"),
            "timeouts": sum(r["llm_timeouts"] or 0 for r in rs),
            "failed_calls": sum(r["llm_failed_calls"] or 0 for r in rs),
            "wall_s": _ms([r["wall_s"] for r in rs], "{:.0f}"),
            "llm_s": _ms([r["llm_seconds"] for r in rs], "{:.0f}"),
            "tokens_in": _ms([r["tokens_in"] for r in rs], "{:.0f}"),
            "tokens_out": _ms([r["tokens_out"] for r in rs], "{:.0f}"),
            "cost_usd": _ms([r["cost_usd"] for r in rs], "{:.3f}"),
            "synth_ok": f"{sum(syn)}/{len(syn)}" if syn else "-",
            "transistors": _ms([r["transistors"] for r in rs], "{:.0f}"),
            "depth": _ms([r["depth"] for r in rs], "{:.1f}"),
        })
    return out


def tests(rows: list[dict]) -> list[dict]:
    """Per scenario, ogni coppia di configurazioni: Fisher sul tasso di successo,
    Mann-Whitney su test passati del miglior candidato, tempo e token."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not r["infra_error"]:
            by[(r["task"], r["mode"])][(r["backend"], r["model"])].append(r)
    out = []
    for (task, mode), cfgs in sorted(by.items()):
        keys = sorted(cfgs)
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                A, B = cfgs[ka], cfgs[kb]
                sa, sb = sum(r["passed"] for r in A), sum(r["passed"] for r in B)
                d = {"task": task, "mode": mode, "a": f"{ka[0]}:{ka[1]}", "b": f"{kb[0]}:{kb[1]}",
                     "success_a": f"{sa}/{len(A)}", "success_b": f"{sb}/{len(B)}",
                     "fisher_p": fisher_exact(sa, len(A) - sa, sb, len(B) - sb)}
                for f in ("best_pass_rate", "wall_s", "tokens_out", "candidates"):
                    mw = mann_whitney_u([r[f] for r in A], [r[f] for r in B])
                    d[f"mw_{f}_p"] = mw["p"]
                out.append(d)
    return out


def render(rows, summ, tst, errors, ctx) -> str:
    meta = ctx["meta"]
    L = ["# Confronto fra backend LLM", "",
         f"Configurazioni: `{meta.get('configs', '?')}` · scenari: task `{meta.get('tasks', '?')}`, "
         f"modalità `{meta.get('modes', '?')}` · n = {meta.get('n', '?')} run per configurazione "
         f"(seed {meta.get('seed_start', 0)}..). Stessa config (`{meta.get('config', '?')}`), stessi "
         "prompt, verifier, budget; ogni run con libreria propria e vuota.", "",
         "Soluzione **completa** = tutti i vettori bit-exact; **parziale** = miglior candidato "
         "secondo (a) elaborazione in Verilog, (b) test passati, (c) sintesi Yosys, "
         "(d) iterazione più recente (`mxagent/run/best.py`).", "",
         "Le run finite per un errore di infrastruttura (eccezione, interruzione) sono escluse "
         "dalle metriche (colonna *err. infra*) e vengono rieseguite alla ripresa.", "",
         "## Metriche per backend/modello/scenario", "",
         "| task | mode | backend | modello | run | err. infra | complete | parziali | successo (CI95) | "
         "test passati best (μ±σ) | candidati fino al pass | timeout | tempo totale s | "
         "token in | token out | costo $ | sintesi ok | transistor | profondità |",
         "|" + "---|" * 19]
    for s in summ:
        L.append(f"| {s['task']} | {s['mode']} | {s['backend']} | {s['model']} | {s['runs']} | "
                 f"{s['infra_errors']} | {s['complete']} | {s['partial']} | {s['success_rate']:.0%} {s['success_ci95']} | "
                 f"{s['best_pass_rate']} | {s['candidates_to_pass']} | {s['timeouts']} | "
                 f"{s['wall_s']} | {s['tokens_in']} | {s['tokens_out']} | {s['cost_usd']} | "
                 f"{s['synth_ok']} | {s['transistors']} | {s['depth']} |")
    refs = ctx.get("refs") or {}
    if refs:
        L += ["", "Reference design (Yosys generic): " + ", ".join(
            f"{t}: {m.get('transistors', '-')} transistor, profondità {m.get('depth', '-')}"
            for t, m in sorted(refs.items()))]
    L += ["", "## Test statistici", "",
          "Fisher esatto (bilaterale) sul tasso di successo; Mann-Whitney U (bilaterale, "
          "approssimazione normale con correzione per i pareggi) su test passati del miglior "
          "candidato, tempo totale, token in uscita e candidati verificati. p < 0.05 indica una "
          "differenza fra i modelli non spiegabile dalla sola variabilità fra seed.", "",
          "| task | mode | A | B | successi A | successi B | p Fisher | p MW test passati | "
          "p MW tempo | p MW token out | p MW candidati |", "|" + "---|" * 11]
    for t in tst:
        L.append(f"| {t['task']} | {t['mode']} | {t['a']} | {t['b']} | {t['success_a']} | "
                 f"{t['success_b']} | {_p(t['fisher_p'])} | {_p(t['mw_best_pass_rate_p'])} | "
                 f"{_p(t['mw_wall_s_p'])} | {_p(t['mw_tokens_out_p'])} | {_p(t['mw_candidates_p'])} |")
    if not tst:
        L.append("(serve più di una configurazione per scenario)")
    # analisi degli errori
    L += ["", "## Analisi degli errori", "",
          "Ogni candidato fallito (evento `verify` del trace) è classificato per categoria: "
          "errore di compilazione Chisel/Scala, errore di elaborazione (Chisel/firtool), lint, "
          "errore di simulazione, test falliti con il tipo di caso dominante nei vettori che "
          "falliscono (NaN, Inf, subnormali, scaling, zero, segno, overflow: dai pattern del "
          "feedback strutturato), timeout del modello, output non parsabile (nessun blocco "
          "Scala; piano JSON non valido). Percentuali sul totale dei fallimenti della "
          "configurazione nello scenario.", ""]
    agg = defaultdict(Counter)
    infra = {r["run"] for r in rows if r["infra_error"]}
    for e in errors:
        if e["run"] in infra:
            continue
        agg[(e["task"], e["mode"], f"{e['backend']}:{e['model']}")][e["category"]] += e["count"]
    cats = sorted({c for cnt in agg.values() for c in cnt})
    if agg:
        cfgs = sorted({k[2] for k in agg})
        for task_mode in sorted({k[:2] for k in agg}):
            L += [f"### {task_mode[0]} ({task_mode[1]})", "",
                  "| categoria | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs)]
            for c in cats:
                cells = []
                for cf in cfgs:
                    cnt = agg.get((*task_mode, cf), Counter())
                    tot = sum(cnt.values())
                    cells.append(f"{cnt[c]} ({cnt[c] / tot:.0%})" if tot and cnt[c] else "-")
                if any(x != "-" for x in cells):
                    L.append(f"| {c} | " + " | ".join(cells) + " |")
            L.append("")
        L += ["Lettura: una configurazione che concentra i fallimenti in errori di compilazione o "
              "output non parsabile ha un problema di *forma* (sintassi Chisel 6, formato della "
              "risposta); una che fallisce su test con casi NaN/Inf/subnormali/scaling ha un "
              "problema di *semantica numerica*. Se la distribuzione delle categorie è simile fra "
              "i modelli e cambia solo il volume, la metodologia è robusta rispetto al modello; "
              "se cambia la categoria dominante, la dipendenza è nel tipo di errore che il "
              "feedback non riesce a correggere."]
    else:
        L.append("Nessun fallimento registrato.")
    infra_rows = [r for r in rows if r["infra_error"]]
    if infra_rows:
        L += ["", "## Errori di infrastruttura (esclusi dalle metriche)", ""]
        L += [f"- `{r['run']}`: {str(r.get('error') or r.get('reason'))[:200]}" for r in infra_rows]
    return "\n".join(L) + "\n"


def report(out: Path) -> None:
    rows, errors, ctx = load(out)
    if not rows:
        raise SystemExit(f"nessuna run in {out / 'runs'}")
    summ, tst = summarize(rows), tests(rows)
    for name, data in (("compare_runs.csv", rows), ("compare_summary.csv", summ),
                       ("compare_tests.csv", tst)):
        if data:
            with (out / name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    cnt = Counter((e["task"], e["mode"], e["backend"], e["model"], e["category"]) for e in errors)
    with (out / "compare_errors.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "mode", "backend", "model", "category", "count"])
        for k, v in sorted(cnt.items()):
            w.writerow([*k, v])
    (out / "compare.md").write_text(render(rows, summ, tst, errors, ctx), encoding="utf-8")
    print(f"report: {out / 'compare.md'} ({len(rows)} run)")
    for s in summ:
        print(f"  {s['task']:22s} {s['mode']:10s} {s['backend']:9s} {s['model']:22s} "
              f"err.infra {s['infra_errors']}  successo {s['complete']}/{s['runs']}  parziali {s['partial']}  "
              f"test best {s['best_pass_rate']}  timeout {s['timeouts']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", default=DEFAULT_CONFIGS,
                   help=f"backend:modello separati da virgola (default {DEFAULT_CONFIGS})")
    p.add_argument("--tasks", default=DEFAULT_TASKS, help="task (cartelle in --bench-dir)")
    p.add_argument("--modes", default="full", help="full,baseline-a,baseline-b")
    p.add_argument("--n", type=int, default=10, help="run ripetute per configurazione (seed)")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                   help="override comuni a tutte le run (es. budget.max_wall_s=1800)")
    p.add_argument("--llm-timeout", type=float, default=None,
                   help="timeout per chiamata (default: quello del backend)")
    p.add_argument("--bench-dir", default="benchmarks")
    p.add_argument("--ref-backends", default="generic", help="sintesi dei reference design")
    p.add_argument("--no-reference", action="store_true", help="non sintetizzare i reference")
    p.add_argument("--job-timeout", type=float, default=2700,
                   help="timeout di una run (s): SIGTERM, la run scrive comunque result.json")
    p.add_argument("--out", default=None, help="default runs/compare_<data>")
    p.add_argument("--dry-run", action="store_true", help="stampa i comandi senza eseguirli")
    p.add_argument("--report-only", action="store_true", help="rifà solo il report")
    a = p.parse_args(argv)
    out = Path(a.out or ROOT / "runs" / f"compare_{date.today():%Y%m%d}")
    if not a.report_only:
        execute(a, out)
        if a.dry_run:
            return 0
    report(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
