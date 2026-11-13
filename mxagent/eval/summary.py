"""Report per run di una campagna: ``summary.csv``, ``summary.md``, ``notes.md``,
``pass_rate.png``. Ogni numero viene da ``result.json``, ``index.jsonl`` o dalla
sintesi ``eval_synth/ppa.json``; nessuna statistica inferenziale.

Con un solo seed per combinazione (``seeds: 1``) è l'unico report: niente pass@k,
intervalli di confidenza o test, perché con n = 1 non hanno senso.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .campaign import Campaign, jobs

#: colonne di summary.csv (e della tabella per run di summary.md)
FIELDS = ["task", "config", "profile", "seed", "status", "passed", "pass_rate", "n_vectors",
          "candidates_to_pass", "reflect_rounds", "time_to_pass_s", "wall_s",
          "llm_calls", "tokens_in", "tokens_out",
          "luts_xc7", "ffs_xc7", "luts_ecp5", "ffs_ecp5", "fmax_mean_mhz", "adp_generic",
          "ppa_gain_pct", "ppa_stop", "best_stage", "reason", "first_error", "job"]


def _read(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, ValueError):
        return None


def _metrics(es: dict | None, backend: str) -> dict:
    r = ((es or {}).get("results") or {}).get(backend) or {}
    return (r.get("metrics") or {}) if r.get("ok") else {}


def config_name(mode: str) -> str:
    return mode.replace("-", "_")  # baseline-a -> baseline_a, come nel comando


def run_rows(out: Path, c: Campaign) -> list[dict]:
    """Una riga per job della campagna, eseguito o no."""
    index: dict[str, dict] = {}
    if (out / "index.jsonl").is_file():
        for ln in (out / "index.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                e = json.loads(ln)
                index[e["job"]] = e
    rows = []
    for j in jobs(c, out):
        st = index.get(j.id) or {}
        rd = out / "runs" / j.id
        res = _read(rd / "result.json") if st.get("run_dir") else None
        row: dict[str, Any] = {k: None for k in FIELDS}
        row.update(job=j.id, task=j.task, config=config_name(j.mode), profile=j.models, seed=j.seed)
        ist = st.get("status", "missing")
        if ist in ("missing", "skipped_budget"):
            row.update(status=ist, passed=False,
                       reason="non eseguita" + (" (budget di campagna esaurito)"
                                               if ist == "skipped_budget" else ""))
            rows.append(row)
            continue
        res = res or {}
        # done/failed/interrupted dalla run; error/timeout se la run non ha scritto result.json
        status = res.get("status") or ist
        if ist in ("error", "timeout") and status in ("done", "failed"):
            status = ist  # result.json c'è ma il processo è finito male: conta l'indice
        ppa = res.get("ppa") or {}
        gain = ppa.get("gain")
        es = _read(rd / "eval_synth" / "ppa.json")
        g, x7, e5, npr = (_metrics(es, b) for b in ("generic", "xc7", "ecp5", "nextpnr"))
        row.update({
            "status": status, "passed": bool(res.get("passed")),
            "pass_rate": res.get("pass_rate"), "n_vectors": res.get("n_vectors"),
            "candidates_to_pass": res.get("candidates_to_pass"),
            "reflect_rounds": res.get("reflect_rounds"),
            "time_to_pass_s": res.get("time_to_pass_s"), "wall_s": res.get("wall_s"),
            "llm_calls": res.get("llm_calls"), "tokens_in": res.get("tokens_in"),
            "tokens_out": res.get("tokens_out"),
            "luts_xc7": x7.get("luts"), "ffs_xc7": x7.get("ffs"),
            "luts_ecp5": e5.get("luts"), "ffs_ecp5": e5.get("ffs"),
            "fmax_mean_mhz": npr.get("fmax_mean"),
            "adp_generic": g["transistors"] * g["depth"] if g.get("transistors") and g.get("depth")
            else None,
            "ppa_gain_pct": round(100 * gain, 2) if j.mode == "full" and gain is not None else None,
            "ppa_stop": ppa.get("stop") if j.mode == "full" else None,
            "best_stage": res.get("best_stage"),
            "reason": res.get("reason") or st.get("error"),
            "first_error": res.get("first_error") or (st.get("error") if ist != "done" else None),
        })
        rows.append(row)
    return rows


def _med(xs) -> float | None:
    v = [float(x) for x in xs if x is not None]
    return statistics.median(v) if v else None


def group_rows(c: Campaign, rows: list[dict]) -> list[dict]:
    """Per configurazione e profilo: passate / eseguite e mediane (nessun intervallo)."""
    out = []
    for prof in c.models:
        for mode in c.modes:
            rs = [r for r in rows if r["profile"] == prof and r["config"] == config_name(mode)]
            ran = [r for r in rs if r["status"] not in ("missing", "skipped_budget")]
            ok = [r for r in ran if r["passed"]]
            out.append({
                "config": config_name(mode), "profile": prof, "runs": len(rs), "executed": len(ran),
                "passed": len(ok),
                "median_wall_s": _med(r["wall_s"] for r in ran),
                "median_time_to_pass_s": _med(r["time_to_pass_s"] for r in ok),
                "median_tokens": _med((r["tokens_in"] or 0) + (r["tokens_out"] or 0)
                                      for r in ran if r["tokens_in"] is not None),
            })
    return out


def _f(x, fmt="{:.0f}") -> str:
    return "–" if x is None or x == "" else fmt.format(x)


def render(c: Campaign, rows: list[dict], groups: list[dict], png: str | None) -> str:
    n = c.seeds
    L = [f"# Campagna {c.name}", ""]
    if n == 1:
        L += ["> **n = 1 seed per combinazione.** I risultati sono indicativi e **non "
              "statisticamente significativi**: una sola run per (task, configurazione, profilo). "
              "Non si calcolano pass@k né intervalli di confidenza.", ""]
    else:
        L += [f"> {n} seed per combinazione; qui solo i valori per run e le mediane "
              "(statistiche in report.md).", ""]
    budget = f", budget di campagna {c.campaign_budget_s} s" if c.campaign_budget_s else ""
    L += [f"Task: {', '.join(c.tasks)}. Configurazioni: {', '.join(config_name(m) for m in c.modes)}. "
          f"Profili: {', '.join(c.models)}. Seed: {c.seed_start}"
          + (f"–{c.seed_start + n - 1}" if n > 1 else "") + f"{budget}.", ""]
    L += ["## Riepilogo per configurazione e profilo", "",
          "| configurazione | profilo | passate / eseguite | non eseguite | mediana tempo totale (s) "
          "| mediana tempo al pass (s) | mediana token (in+out) |",
          "|---|---|---|---|---|---|---|"]
    for g in groups:
        L.append(f"| {g['config']} | {g['profile']} | {g['passed']} / {g['executed']} "
                 f"| {g['runs'] - g['executed']} | {_f(g['median_wall_s'])} "
                 f"| {_f(g['median_time_to_pass_s'])} | {_f(g['median_tokens'])} |")
    L += ["", "## Run", "",
          "Tempi in secondi. LUT/FF: Yosys (`synth_xilinx`, `synth_ecp5`); Fmax: media di nextpnr-ecp5 "
          "sui seed; ADP: transistori × profondità di Yosys generic. Sintesi del design finale delle "
          "run passate (`eval_synth`). Guadagno PPA: obiettivo iniziale → finale, solo `full`.", "",
          "| task | config | profilo | status | pass | pass rate | cand. al pass | reflect | t pass "
          "| t totale | chiamate | token in | token out | LUT/FF xc7 | LUT/FF ecp5 | Fmax (MHz) | ADP "
          "| PPA |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ppa = "–"
        if r["config"] == "full" and r["status"] not in ("missing", "skipped_budget"):
            ppa = (f"{r['ppa_gain_pct']:+.1f}%" if r["ppa_gain_pct"] is not None else "–") + (
                f" ({r['ppa_stop']})" if r["ppa_stop"] else "")
        L.append(
            f"| {r['task']} | {r['config']} | {r['profile']} | {r['status']} "
            f"| {'sì' if r['passed'] else 'no'} | {_f(r['pass_rate'], '{:.3f}')} "
            f"| {_f(r['candidates_to_pass'])} | {_f(r['reflect_rounds'])} | {_f(r['time_to_pass_s'])} "
            f"| {_f(r['wall_s'])} | {_f(r['llm_calls'])} | {_f(r['tokens_in'])} | {_f(r['tokens_out'])} "
            f"| {_f(r['luts_xc7'])}/{_f(r['ffs_xc7'])} | {_f(r['luts_ecp5'])}/{_f(r['ffs_ecp5'])} "
            f"| {_f(r['fmax_mean_mhz'], '{:.1f}')} | {_f(r['adp_generic'])} | {ppa} |")
    L.append("")
    if png:
        L += [f"![pass rate dei vettori]({png})", ""]
    else:
        L += ["(grafico non generato: matplotlib non installato, `pip install -e .[eval]`)", ""]
    L += ["Le run non passate sono descritte in `notes.md`; tutti i valori sono in `summary.csv`.", ""]
    return "\n".join(L)


def render_notes(rows: list[dict]) -> str:
    L = ["# Run non passate", "",
         "Stadio di fallimento del miglior candidato (compile, lint, sim, no_code) e primo errore.", ""]
    bad = [r for r in rows if not r["passed"]]
    if not bad:
        L.append("Nessuna: tutte le run eseguite sono passate.")
    for r in bad:
        L.append(f"## {r['task']} · {r['config']} · {r['profile']} · s{r['seed']}")
        L.append("")
        if r["status"] in ("missing", "skipped_budget"):
            L += [f"- non eseguita: {r['reason']}", ""]
            continue
        L.append(f"- status: `{r['status']}`" + (f" — {r['reason']}" if r["reason"] else ""))
        L.append(f"- stadio: `{r['best_stage'] or '–'}`, pass rate {_f(r['pass_rate'], '{:.3f}')}")
        L.append(f"- primo errore: `{(r['first_error'] or '–')[:300]}`")
        L.append("")
    return "\n".join(L)


def plot_pass_rate(c: Campaign, rows: list[dict], path: Path) -> bool:
    """Barre del pass rate dei vettori per task × configurazione, una serie per profilo."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    cats = [(t, config_name(m)) for t in c.tasks for m in c.modes]
    profs = list(c.models)
    w = 0.8 / max(1, len(profs))
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(cats)), 4))
    for k, p in enumerate(profs):
        xs, ys, miss = [], [], []
        for i, (t, cf) in enumerate(cats):
            rs = [r for r in rows if r["task"] == t and r["config"] == cf and r["profile"] == p
                  and r["status"] not in ("missing", "skipped_budget")]
            x = i + (k - (len(profs) - 1) / 2) * w
            if rs and rs[0]["pass_rate"] is not None:
                xs.append(x)
                ys.append(rs[0]["pass_rate"])
            else:
                miss.append(x)
        bars = ax.bar(xs, ys, w, label=p)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
        for x in miss:
            ax.text(x, 0.02, "n/e", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(range(len(cats)), [f"{t.split('_')[0]}\n{cf}" for t, cf in cats], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("pass rate dei vettori (miglior candidato)")
    ax.set_title(f"{c.name} — n = {c.seeds} seed per combinazione" + (" (indicativo)" if c.seeds == 1 else ""))
    ax.legend(title="profilo", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def write_summary(out: Path, c: Campaign) -> dict:
    rows = run_rows(out, c)
    groups = group_rows(c, rows)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    png = "pass_rate.png" if plot_pass_rate(c, rows, out / "pass_rate.png") else None
    (out / "summary.md").write_text(render(c, rows, groups, png), encoding="utf-8")
    (out / "notes.md").write_text(render_notes(rows), encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"rows": rows, "groups": groups, "png": png, "jobs": {"expected": len(rows), **counts}}


__all__ = ["FIELDS", "run_rows", "group_rows", "write_summary"]
