"""Report di una campagna: solo numeri letti dai file prodotti dalle run.

Uscite in ``<campagna>/``: ``report.md`` (tabelle), ``runs.csv`` (una riga per
job), ``summary.json`` (tutti gli aggregati). Job mancanti o finiti in errore
sono elencati e non contano come fallimenti del modello.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .campaign import Campaign, Job, jobs
from .stats import geomean, mcnemar_exact, mean_std, median, pass_at_k, wilson


def _read(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _metrics(d: dict | None, backend: str) -> dict:
    r = ((d or {}).get("results") or {}).get(backend) or {}
    return (r.get("metrics") or {}) if r.get("ok") else {}


def _ppa_primary(d: dict | None) -> dict | None:
    """Metriche ``generic`` da un ppa.json di ``mxagent synth``."""
    if not d:
        return None
    g = (d.get("results") or {}).get("generic") or {}
    return g.get("metrics") if g.get("ok") else None


def load_records(out: Path) -> tuple[Campaign, list[dict], dict[str, dict]]:
    meta = _read(out / "campaign.json")
    if not meta:
        raise SystemExit(f"{out}/campaign.json assente: non è la cartella di una campagna")
    c = Campaign.model_validate(meta["campaign"])
    status: dict[str, dict] = {}
    if (out / "index.jsonl").is_file():
        for ln in (out / "index.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                e = json.loads(ln)
                status[e["job"]] = e
    recs = []
    for j in jobs(c, out):
        st = status.get(j.id, {})
        r: dict[str, Any] = {"job": j.id, "task": j.task, "mode": j.mode, "variant": j.variant,
                             "models": j.models, "seed": j.seed, "status": st.get("status", "missing")}
        if r["status"] == "done":
            rd = out / st["run_dir"]
            m = _read(rd / "metrics.json") or {}
            meta_r = _read(rd / "meta.json") or {}
            llm = m.get("llm_to_pass") or m.get("llm") or {}
            es = _read(rd / "eval_synth" / "ppa.json")
            q = _ppa_primary(es)
            npr = ((es or {}).get("results") or {}).get("nextpnr") or {}
            wr = _read(rd / "eval_wrap" / "wrap_report.json")
            x7, e5 = _metrics(es, "xc7"), _metrics(es, "ecp5")
            res = _read(rd / "result.json") or {}
            r.update({
                "passed": bool(m.get("passed")), "stop": m.get("stop"),
                "best_stage": m.get("best_stage"), "best_pass_rate": m.get("best_pass_rate"),
                "iterations": m.get("iterations"), "time_to_pass_s": m.get("time_to_pass_s"),
                "wall_s": m.get("wall_s"), "calls_to_pass": llm.get("calls") if m.get("passed") else None,
                "tokens_to_pass": llm.get("tokens") if m.get("passed") else None,
                "calls_total": (m.get("llm") or {}).get("calls"),
                "model": m.get("model"), "model_digest": meta_r.get("model_digest"),
                "transistors": q.get("transistors") if q else None, "depth": q.get("depth") if q else None,
                "fmax_min": (npr.get("metrics") or {}).get("fmax_min") if npr.get("ok") else None,
                "fmax_mhz": (npr.get("metrics") or {}).get("fmax_mhz") if npr.get("ok") else None,
                "adp": q["transistors"] * q["depth"] if q and q.get("transistors") and q.get("depth")
                else None,
                "luts_xc7": x7.get("luts"), "ffs_xc7": x7.get("ffs"),
                "luts_ecp5": e5.get("luts"), "ffs_ecp5": e5.get("ffs"),
                "tokens_total": (m.get("llm") or {}).get("tokens"),
                "duplicates": res.get("duplicates"),
                "wrapper_passed": wr.get("passed") if wr else None,
            })
            p = m.get("ppa") or {}
            if p:
                ini = p.get("initial") or {}
                r.update({"ppa_tried": p.get("tried"), "ppa_accepted": p.get("accepted"),
                          "ppa_gain": p.get("gain"), "transistors_pre_opt": ini.get("transistors"),
                          "depth_pre_opt": ini.get("depth")})
        elif st:
            r["error"] = st.get("error", "")
            r["result_status"] = st.get("result_status")
        recs.append(r)
    refs = {t: _ppa_primary(_read(out / "reference" / t / "ppa.json")) for t in c.tasks}
    return c, recs, refs


# ------------------------------------------------------------- aggregati
def _rate(rs: list[dict]) -> dict:
    done = [r for r in rs if r["status"] == "done"]
    k, n = sum(r["passed"] for r in done), len(done)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def _fmt_rate(a: dict, ci: bool = False) -> str:
    if not a["n"]:
        return "–"
    s = f"{a['k']}/{a['n']}"
    if ci:
        s += f" ({a['rate']:.0%}, CI {a['ci95'][0]:.0%}–{a['ci95'][1]:.0%})"
    return s


def _f(x: float | None, fmt: str = "{:.1f}") -> str:
    return "–" if x is None else fmt.format(x)


def _pairs(recs: list[dict], a: dict, b: dict) -> dict:
    """Confronto appaiato per (task, seed, modelli) fra due gruppi di job."""
    def key(r):
        return (r["task"], r["seed"], r["models"])

    def sel(f):
        return {key(r): r["passed"] for r in recs
                if r["status"] == "done" and all(r[k] == v for k, v in f.items())}
    A, B = sel(a), sel(b)
    common = sorted(set(A) & set(B))
    only_a = sum(A[k] and not B[k] for k in common)
    only_b = sum(B[k] and not A[k] for k in common)
    both = sum(A[k] and B[k] for k in common)
    return {"pairs": len(common), "only_a": only_a, "only_b": only_b, "both": both,
            "rate_a": sum(A[k] for k in common) / len(common) if common else None,
            "rate_b": sum(B[k] for k in common) / len(common) if common else None,
            "p_mcnemar": mcnemar_exact(only_a, only_b)}


def _task_meta(c: Campaign, name: str) -> dict:
    from ..bench.task import load_task
    try:
        t = load_task(Path(c.benchmarks_dir) / name)
    except (FileNotFoundError, OSError):
        return {}
    return {"target_mhz": t.timing.target_mhz, "wrapper": t.wrapper}


def summarize(c: Campaign, recs: list[dict], refs: dict[str, dict]) -> dict:
    main = [r for r in recs if r["variant"] == "main"]
    S: dict[str, Any] = {"campaign": c.name, "digest": c.digest()}
    cnt = Counter(r["status"] for r in recs)
    S["jobs"] = {"expected": len(recs), **cnt}
    S["models_seen"] = sorted({(r["models"], r.get("model"), r.get("model_digest"))
                               for r in recs if r["status"] == "done"}, key=str)

    # successo per (modelli, modalità, task) e totale
    succ: dict = defaultdict(dict)
    for ms in c.models:
        for mode in c.modes:
            rs = [r for r in main if r["models"] == ms and r["mode"] == mode]
            succ[ms][mode] = {"total": _rate(rs),
                              "tasks": {t: _rate([r for r in rs if r["task"] == t]) for t in c.tasks}}
            # pass@k medio sui task (k <= seed completati per task)
            pk = {}
            for k in (1, 3, 5, 10):
                vals = []
                for t in c.tasks:
                    a = succ[ms][mode]["tasks"][t]
                    if a["n"] >= k:
                        vals.append(pass_at_k(a["n"], a["k"], k))
                if vals and len(vals) == len(c.tasks):
                    pk[f"pass@{k}"] = sum(vals) / len(vals)
            succ[ms][mode]["pass_at_k"] = pk
    S["success"] = succ

    # confronti appaiati
    S["paired"] = {}
    if "full" in c.modes:
        for ms in c.models:
            for base in [m for m in c.modes if m != "full"]:
                S["paired"][f"{ms}: full vs {base}"] = _pairs(
                    main, {"mode": "full", "models": ms}, {"mode": base, "models": ms})

    # efficienza sulle run passate
    eff: dict = defaultdict(dict)
    for ms in c.models:
        for mode in c.modes:
            ok = [r for r in main if r["models"] == ms and r["mode"] == mode and r.get("passed")]
            eff[ms][mode] = {"n": len(ok),
                             "calls_to_pass": median(r["calls_to_pass"] for r in ok),
                             "tokens_to_pass": median(r["tokens_to_pass"] for r in ok),
                             "time_to_pass_s": median(r["time_to_pass_s"] for r in ok),
                             "iterations": median(r["iterations"] for r in ok)}
    S["efficiency"] = eff

    # dove si fermano le run fallite
    fail: dict = defaultdict(dict)
    for ms in c.models:
        for mode in c.modes:
            bad = [r for r in main if r["models"] == ms and r["mode"] == mode
                   and r["status"] == "done" and not r["passed"]]
            fail[ms][mode] = {"stage": dict(Counter(r["best_stage"] for r in bad)),
                              "budget": sum(str(r.get("stop", "")).startswith("budget") for r in bad)}
    S["failures"] = fail

    # ablation: ogni variante contro full/main sugli stessi (task, seed, modelli)
    abl = {}
    for name in c.ablations.variants:
        rs = [r for r in recs if r["variant"] == name]
        tasks = sorted({r["task"] for r in rs})
        mss = sorted({r["models"] for r in rs})
        ref = [r for r in main if r["mode"] == "full" and r["task"] in tasks and r["models"] in mss
               and any(x["seed"] == r["seed"] for x in rs)]
        pr = _pairs([*rs, *ref], {"variant": name}, {"variant": "main", "mode": "full"})
        abl[name] = {"overrides": c.ablations.variants[name], "tasks": tasks, "models": mss,
                     "variant": _rate(rs), "full": _rate(ref), "paired": pr}
    S["ablations"] = abl

    # qualità dei risultati (sintesi generic) rispetto al riferimento
    qor: dict = defaultdict(dict)
    for ms in c.models:
        for mode in c.modes:
            ok = [r for r in main if r["models"] == ms and r["mode"] == mode and r.get("passed")]
            ra, rd = [], []
            for r in ok:
                ref = refs.get(r["task"])
                if ref and r.get("transistors"):
                    ra.append(r["transistors"] / ref["transistors"])
                    rd.append(r["depth"] / ref["depth"] if ref.get("depth") else None)
            q = {"n": len(ra), "area_ratio_geomean": geomean(ra), "depth_ratio_geomean": geomean(rd)}
            if mode == "full":
                q["optimizer_tried"] = sum(r.get("ppa_tried") or 0 for r in ok)
                q["optimizer_accepted"] = sum(r.get("ppa_accepted") or 0 for r in ok)
                q["optimizer_improved_runs"] = sum(1 for r in ok if (r.get("ppa_gain") or 0) > 0)
                q["optimizer_gain_median"] = median(r.get("ppa_gain") for r in ok if r.get("ppa_gain"))
            qor[ms][mode] = q
    S["qor"] = qor

    # vincoli di M7: target di frequenza (task con timing.target_mhz) e wrapper CV-X-IF
    tasks = {t: _task_meta(c, t) for t in c.tasks}
    m7: dict = defaultdict(dict)
    for ms in c.models:
        for mode in c.modes:
            ok = [r for r in main if r["models"] == ms and r["mode"] == mode and r.get("passed")]
            tm = [r for r in ok if tasks[r["task"]].get("target_mhz")]
            wr = [r for r in ok if tasks[r["task"]].get("wrapper")]
            m7[ms][mode] = {
                "timing_n": len(tm),
                "timing_met": sum(1 for r in tm if r.get("fmax_min") is not None
                                  and r["fmax_min"] >= tasks[r["task"]]["target_mhz"]),
                "timing_measured": sum(1 for r in tm if r.get("fmax_min") is not None),
                "wrapper_n": len(wr), "wrapper_passed": sum(1 for r in wr if r.get("wrapper_passed")),
                "wrapper_tested": sum(1 for r in wr if r.get("wrapper_passed") is not None)}
    S["m7"] = m7
    S["stats"] = cell_stats(c, main)
    S["task_meta"] = tasks
    S["references"] = refs
    S["errors"] = [{"job": r["job"], "status": r["status"], "error": r.get("error", ""),
                    "result_status": r.get("result_status")}
                   for r in recs if r["status"] in ("error", "timeout")]
    return S


#: metriche con media e deviazione standard: (chiave, su quali run, etichetta)
STAT_METRICS = [
    ("best_pass_rate", "done", "pass rate vettori"),
    ("iterations", "passed", "iterazioni al pass"),
    ("time_to_pass_s", "passed", "tempo al pass (s)"),
    ("tokens_to_pass", "passed", "token al pass"),
    ("tokens_total", "done", "token totali"),
    ("adp", "passed", "ADP (transistori x profondità)"),
    ("luts_xc7", "passed", "LUT xc7"),
    ("ffs_xc7", "passed", "FF xc7"),
    ("luts_ecp5", "passed", "LUT4 ecp5"),
    ("ffs_ecp5", "passed", "FF ecp5"),
    ("fmax_mhz", "passed", "Fmax ECP5 (MHz)"),
]


def cell_stats(c: Campaign, main: list[dict]) -> list[dict]:
    """Una riga per (task, modelli, modalità): pass@1 / pass@k e media ± dev. std.

    Per task e non aggregate: area, LUT ed Fmax di task diversi non si mediano.
    """
    rows = []
    for t in c.tasks:
        for ms in c.models:
            for mode in c.modes:
                rs = [r for r in main if r["task"] == t and r["models"] == ms and r["mode"] == mode
                      and r["status"] == "done"]
                ok = [r for r in rs if r.get("passed")]
                n, k = len(rs), len(ok)
                row: dict[str, Any] = {"task": t, "models": ms, "mode": mode, "runs": n, "passed": k,
                                       "pass@1": pass_at_k(n, k, 1) if n else None}
                for kk in (3, 5, 10):
                    row[f"pass@{kk}"] = pass_at_k(n, k, kk) if n >= kk else None
                for key, over, _ in STAT_METRICS:
                    row[key] = mean_std(r.get(key) for r in (ok if over == "passed" else rs))
                rows.append(row)
    return rows


def _ms(d: dict, fmt: str) -> str:
    if not d or d.get("mean") is None:
        return "–"
    s = fmt.format(d["mean"])
    return s + (f" ± {fmt.format(d['std'])}" if d.get("std") is not None else "")


# ---------------------------------------------------------------- output
def render(c: Campaign, S: dict) -> str:
    L: list[str] = [f"# Report della campagna `{c.name}`", ""]
    j = S["jobs"]
    L += [f"Generato {datetime.now():%Y-%m-%d %H:%M} da `mxagent report`; campagna {S['digest']}.",
          "Ogni numero è letto dai file delle run (metrics.json, ppa.json); nulla è stimato.", "",
          f"Job: {j['expected']} previsti, {j.get('done', 0)} completati, "
          f"{j.get('error', 0)} in errore, {j.get('timeout', 0)} in timeout, "
          f"{j.get('missing', 0)} mancanti. Seed per cella: {c.seeds}. "
          f"Libreria: {c.library.policy}.", ""]
    if j.get("done", 0) < j["expected"]:
        L += ["> **Campagna incompleta**: i tassi sono calcolati sulle sole run completate.", ""]
    L += ["Modelli:", ""]
    for ms, model, dig in S["models_seen"]:
        L.append(f"- `{ms}`: {model} (digest {str(dig)[:12]})")
    L.append("")

    L += ["## Successo (bit-exact su tutti i vettori)", ""]
    for ms in c.models:
        L += [f"### `{ms}`", "", "| task | " + " | ".join(c.modes) + " |",
              "|---|" + "---|" * len(c.modes)]
        for t in c.tasks:
            L.append(f"| {t} | " + " | ".join(_fmt_rate(S["success"][ms][m]["tasks"][t])
                                              for m in c.modes) + " |")
        L.append("| **totale** | " + " | ".join(_fmt_rate(S["success"][ms][m]["total"], ci=True)
                                                for m in c.modes) + " |")
        pk = [(m, S["success"][ms][m]["pass_at_k"]) for m in c.modes]
        keys = sorted({k for _, d in pk for k in d}, key=lambda s: int(s.split("@")[1]))
        if keys:
            L += ["", "| pass@k (media sui task) | " + " | ".join(c.modes) + " |",
                  "|---|" + "---|" * len(c.modes)]
            for k in keys:
                L.append(f"| {k} | " + " | ".join(_f(d.get(k), "{:.2f}") for _, d in pk) + " |")
        L.append("")

    if S["paired"]:
        L += ["## Confronto appaiato (stessi task e seed, McNemar esatto)", "",
              "| confronto | coppie | tasso A | tasso B | solo A | solo B | p |", "|---|---|---|---|---|---|---|"]
        for name, p in S["paired"].items():
            L.append(f"| {name} | {p['pairs']} | {_f(p['rate_a'], '{:.0%}')} | {_f(p['rate_b'], '{:.0%}')} "
                     f"| {p['only_a']} | {p['only_b']} | {p['p_mcnemar']:.3g} |")
        L.append("")

    L += ["## Efficienza (mediane sulle run passate; chiamate e token fino al pass)", "",
          "| modelli | modalità | passate | chiamate LLM | token (k) | tempo (s) | iterazioni |",
          "|---|---|---|---|---|---|---|"]
    for ms in c.models:
        for m in c.modes:
            e = S["efficiency"][ms][m]
            tok = e["tokens_to_pass"] / 1000 if e["tokens_to_pass"] is not None else None
            L.append(f"| {ms} | {m} | {e['n']} | {_f(e['calls_to_pass'])} | {_f(tok)} "
                     f"| {_f(e['time_to_pass_s'], '{:.0f}')} | {_f(e['iterations'])} |")
    L += ["", "## Dove si fermano le run fallite (fase del miglior candidato)", "",
          "| modelli | modalità | compile | lint | sim | no_code | fermate dal budget |", "|---|---|---|---|---|---|---|"]
    for ms in c.models:
        for m in c.modes:
            f = S["failures"][ms][m]
            st = f["stage"]
            L.append(f"| {ms} | {m} | {st.get('compile', 0)} | {st.get('lint', 0)} | {st.get('sim', 0)} "
                     f"| {st.get('no_code', 0)} | {f['budget']} |")
    L.append("")

    if S["ablations"]:
        L += ["## Ablation (sistema completo senza un meccanismo; appaiate con `full`)", "",
              "| variante | override | task | variante | full | Δ | solo variante | solo full | p |",
              "|---|---|---|---|---|---|---|---|---|"]
        for name, a in S["ablations"].items():
            p = a["paired"]
            d = (p["rate_a"] - p["rate_b"]) if p["rate_a"] is not None and p["rate_b"] is not None else None
            ov = ", ".join(f"{k}={v}" for k, v in a["overrides"].items())
            L.append(f"| {name} | `{ov}` | {len(a['tasks'])} | {_fmt_rate(a['variant'])} | {_fmt_rate(a['full'])} "
                     f"| {_f(d, '{:+.0%}')} | {p['only_a']} | {p['only_b']} | {p['p_mcnemar']:.3g} |")
        L.append("")

    L += ["## Qualità dei risultati (Yosys generic, design passati)", "",
          "Rapporti rispetto al riferimento scritto a mano (media geometrica; < 1 = meglio del riferimento).", "",
          "| task | transistori rif. | profondità rif. |", "|---|---|---|"]
    for t, ref in S["references"].items():
        L.append(f"| {t} | {_f(ref.get('transistors') if ref else None, '{:.0f}')} "
                 f"| {_f(ref.get('depth') if ref else None, '{:.0f}')} |")
    L += ["", "| modelli | modalità | design | area / rif. | profondità / rif. | PPA opt.: tentativi / accettati / run migliorate / guadagno mediano |",
          "|---|---|---|---|---|---|"]
    for ms in c.models:
        for m in c.modes:
            q = S["qor"][ms][m]
            opt = "–"
            if m == "full":
                opt = (f"{q['optimizer_tried']} / {q['optimizer_accepted']} / {q['optimizer_improved_runs']} / "
                       f"{_f(q['optimizer_gain_median'], '{:.1%}')}")
            L.append(f"| {ms} | {m} | {q['n']} | {_f(q['area_ratio_geomean'], '{:.2f}')} "
                     f"| {_f(q['depth_ratio_geomean'], '{:.2f}')} | {opt} |")
    L.append("")

    tm = {t: m for t, m in S.get("task_meta", {}).items() if m.get("target_mhz") or m.get("wrapper")}
    if tm:
        L += ["## Vincoli di sistema (M7, design passati)", "",
              "Target di frequenza: Fmax ECP5 peggiore sui seed di nextpnr >= target. "
              "Wrapper: test di protocollo CV-X-IF con il MAC generato.", "",
              "| modelli | modalità | target rispettato / misurati / passati | wrapper ok / testati / passati |",
              "|---|---|---|---|"]
        for ms in c.models:
            for m in c.modes:
                x = S["m7"][ms][m]
                L.append(f"| {ms} | {m} | {x['timing_met']} / {x['timing_measured']} / {x['timing_n']} "
                         f"| {x['wrapper_passed']} / {x['wrapper_tested']} / {x['wrapper_n']} |")
        L.append("")

    if S.get("stats"):
        L += ["## Media ± deviazione standard per task, configurazione e modello", "",
              "Run completate; iterazioni, tempo, token al pass e metriche del design solo sulle run "
              "passate. Sintesi del design finale (`eval_synth`): ADP = transistori × profondità "
              "(Yosys generic), LUT/FF di Yosys, Fmax mediana di nextpnr. Le stesse righe sono in "
              "`stats.csv`.", "",
              "| task | modelli | modalità | passate | pass@1 | pass@5 | pass@10 | pass rate vettori "
              "| iterazioni | tempo (s) | token al pass (k) | ADP | LUT xc7 | FF xc7 | Fmax (MHz) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in S["stats"]:
            tok = r["tokens_to_pass"]
            tokk = {**tok, "mean": tok["mean"] / 1000 if tok["mean"] is not None else None,
                    "std": tok["std"] / 1000 if tok["std"] is not None else None}
            L.append(f"| {r['task']} | {r['models']} | {r['mode']} | {r['passed']}/{r['runs']} "
                     f"| {_f(r['pass@1'], '{:.2f}')} | {_f(r['pass@5'], '{:.2f}')} | {_f(r['pass@10'], '{:.2f}')} "
                     f"| {_ms(r['best_pass_rate'], '{:.3f}')} | {_ms(r['iterations'], '{:.1f}')} "
                     f"| {_ms(r['time_to_pass_s'], '{:.0f}')} | {_ms(tokk, '{:.1f}')} "
                     f"| {_ms(r['adp'], '{:.0f}')} | {_ms(r['luts_xc7'], '{:.0f}')} "
                     f"| {_ms(r['ffs_xc7'], '{:.0f}')} | {_ms(r['fmax_mhz'], '{:.1f}')} |")
        L.append("")

    if S["errors"]:
        L += ["## Errori di infrastruttura (esclusi dai tassi)", ""]
        L += [f"- `{e['job']}` {e['status']}"
              + (f" (result.json: {e['result_status']})" if e.get("result_status") else "")
              + f": {e['error'][:200]}" for e in S["errors"]]
        L.append("")
    return "\n".join(L)


CSV_FIELDS = ["job", "task", "mode", "variant", "models", "seed", "status", "passed", "stop",
              "best_stage", "best_pass_rate", "iterations", "calls_to_pass", "tokens_to_pass",
              "calls_total", "time_to_pass_s", "wall_s", "model", "model_digest", "transistors",
              "depth", "transistors_pre_opt", "depth_pre_opt", "ppa_tried", "ppa_accepted", "ppa_gain",
              "fmax_min", "fmax_mhz", "adp", "luts_xc7", "ffs_xc7", "luts_ecp5", "ffs_ecp5",
              "tokens_total", "duplicates", "wrapper_passed"]
STATS_CSV_FIELDS = ["task", "models", "mode", "runs", "passed", "pass@1", "pass@3", "pass@5", "pass@10"]


def write_report(out: Path) -> dict:
    c, recs, refs = load_records(out)
    S = summarize(c, recs, refs)
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    with (out / "stats.csv").open("w", newline="", encoding="utf-8") as f:
        cols = STATS_CSV_FIELDS + [f"{k}_{s}" for k, _, _ in STAT_METRICS for s in ("n", "mean", "std")]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in S["stats"]:
            w.writerow({**{k: r[k] for k in STATS_CSV_FIELDS},
                        **{f"{k}_{s}": r[k][s] for k, _, _ in STAT_METRICS for s in ("n", "mean", "std")}})
    (out / "report.md").write_text(render(c, S), encoding="utf-8")
    return S


__all__ = ["Job", "load_records", "summarize", "render", "write_report"]
