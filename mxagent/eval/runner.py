"""Esecuzione di una campagna: un sottoprocesso ``mxagent run`` per job.

* isolamento: un crash o un timeout di un job non ferma la campagna;
* ripresa: ``index.jsonl`` (append-only) registra l'esito di ogni job; una
  campagna rilanciata salta i job già completati (``--retry-errors`` rifà
  quelli finiti in errore o timeout);
* sintesi: ogni design passato (di qualunque modalità) viene sintetizzato con
  ``ppa_backends`` per confrontare la qualità dei risultati; il riferimento di
  ogni task è sintetizzato una volta in ``reference/<task>/``.

Gli errori di infrastruttura (crash, timeout) sono tenuti distinti dai
fallimenti del modello e riportati come tali nel report.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from ..util.shell import Result, run
from .campaign import Campaign, Job, jobs

Exec = Callable[[list[str], Path, float], Result]


def _default_exec(cmd: list[str], cwd: Path, timeout: float) -> Result:
    return run(cmd, cwd=cwd, timeout=timeout)


class Runner:
    def __init__(self, campaign: Campaign, out_dir: Path, *, repo: Path | None = None,
                 executor: Exec | None = None, python: str | None = None):
        self.c, self.out = campaign, Path(out_dir)
        self.repo = Path(repo or Path.cwd())
        self.exec = executor or _default_exec
        self.python = python or sys.executable
        self.index_path = self.out / "index.jsonl"
        self._lock = threading.Lock()

    # ------------------------------------------------------------- indice
    def status(self) -> dict[str, dict]:
        """Ultimo esito per job."""
        st: dict[str, dict] = {}
        if self.index_path.is_file():
            for ln in self.index_path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    e = json.loads(ln)
                    st[e["job"]] = e
        return st

    def _append(self, entry: dict) -> None:
        with self._lock:
            with self.index_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --------------------------------------------------------------- job
    def command(self, j: Job) -> list[str]:
        return [self.python, "-m", "mxagent", "run",
                "--task", str(Path(self.c.benchmarks_dir) / j.task), "--mode", j.mode,
                "--config", self.c.config, "--seed", str(j.seed),
                "--runs-dir", str(self.out / "runs"), "--run-id", j.id,
                *(["--set", *j.set_args()] if j.overrides else [])]

    def timeout(self) -> float:
        if self.c.job_timeout_s:
            return float(self.c.job_timeout_s)
        from ..config import load_config
        cfg = load_config(self.repo / self.c.config, dict(self.c.set))
        return 2.0 * cfg.budget.max_wall_s + 1800

    def run_job(self, j: Job) -> dict:
        run_dir = self.out / "runs" / j.id
        t0 = time.monotonic()
        r = self.exec(self.command(j), self.repo, self.timeout())
        (self.out / "logs").mkdir(parents=True, exist_ok=True)
        (self.out / "logs" / f"{j.id}.log").write_text(r.log, encoding="utf-8")
        has_metrics = (run_dir / "metrics.json").is_file()
        # exit 1 = run completata ma non passata: è un esito, non un errore
        if r.timed_out:
            status = "timeout"
        elif has_metrics and r.returncode in (0, 1):
            status = "done"
        else:
            status = "error"
        entry = {"job": j.id, "status": status, "returncode": r.returncode,
                 "seconds": round(time.monotonic() - t0, 1), "run_dir": f"runs/{j.id}",
                 "finished": datetime.now().isoformat(timespec="seconds")}
        if status != "done":
            entry["error"] = " | ".join(r.log.strip().splitlines()[-3:])[:500]
            res = run_dir / "result.json"  # interrupted / failed per eccezione: lo dice la run
            if res.is_file():
                try:
                    rj = json.loads(res.read_text(encoding="utf-8"))
                    entry["result_status"] = rj.get("status")
                    entry["error"] = (rj.get("error") or rj.get("reason") or "") + " | " + entry["error"]
                except (OSError, ValueError):
                    pass
        elif self.c.ppa_backends:
            entry["ppa"] = self.synth_run(j, run_dir)
        self._append(entry)
        return entry

    def _task(self, name: str):
        from ..bench.task import load_task
        return load_task(self.repo / self.c.benchmarks_dir / name)

    def _backends(self, task) -> list[str]:
        b = list(self.c.ppa_backends)
        if task.timing.target_mhz and "nextpnr" not in b:
            b.append("nextpnr")  # b06: il target di frequenza si misura con place & route
        return b

    def synth_run(self, j: Job, run_dir: Path) -> str:
        m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if not m.get("passed"):
            return "not_passed"
        task = self._task(j.task)
        out = run_dir / "eval_synth"
        if (out / "ppa.json").is_file():
            status = "cached"
        else:
            r = self.exec([self.python, "-m", "mxagent", "synth", "--run", str(run_dir),
                           "--bench-dir", self.c.benchmarks_dir, "--config", self.c.config,
                           "--backends", ",".join(self._backends(task)), "--out", str(out)],
                          self.repo, self.timeout())
            status = "ok" if (out / "ppa.json").is_file() else f"error: {r.log.strip()[-200:]}"
        if task.wrapper == "cvxif" and not (run_dir / "eval_wrap" / "wrap_report.json").is_file():
            self.exec([self.python, "-m", "mxagent", "wrap", "--run", str(run_dir),
                       "--bench-dir", self.c.benchmarks_dir, "--out", str(run_dir / "eval_wrap"),
                       "--seed", str(j.seed)], self.repo, self.timeout())
        return status

    def synth_references(self) -> dict[str, str]:
        res = {}
        for t in self.c.tasks:
            out = self.out / "reference" / t
            if (out / "ppa.json").is_file():
                res[t] = "cached"
                continue
            r = self.exec([self.python, "-m", "mxagent", "synth",
                           "--task", str(Path(self.c.benchmarks_dir) / t), "--config", self.c.config,
                           "--backends", ",".join(self._backends(self._task(t))), "--out", str(out),
                           "--tag", "eval_ref"], self.repo, self.timeout())
            res[t] = "ok" if (out / "ppa.json").is_file() else f"error: {r.log.strip()[-200:]}"
        return res

    # --------------------------------------------------------------- run
    def pending(self, retry_errors: bool = False) -> list[Job]:
        st = self.status()
        todo = []
        for j in jobs(self.c, self.out):
            s = st.get(j.id, {}).get("status")
            if s == "done" or (s in ("error", "timeout") and not retry_errors):
                continue
            todo.append(j)
        return todo

    def run(self, *, parallel: int = 1, retry_errors: bool = False, limit: int | None = None,
            log: Callable[[str], None] = print) -> list[dict]:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "campaign.json").write_text(json.dumps(
            {"campaign": self.c.model_dump(mode="json"), "digest": self.c.digest(),
             "python": self.python, "host": os.uname().nodename if hasattr(os, "uname") else "",
             "written": datetime.now().isoformat(timespec="seconds")}, indent=1), encoding="utf-8")
        if self.c.ppa_backends:
            log(f"sintesi dei riferimenti: {self.synth_references()}")
        todo = self.pending(retry_errors)[:limit]
        total = len(jobs(self.c, self.out))
        log(f"{len(todo)} job da eseguire ({total} nella campagna, {parallel} in parallelo)")
        done: list[dict] = []

        def one(j: Job) -> dict:
            e = self.run_job(j)
            with self._lock:
                done.append(e)
                log(f"[{len(done)}/{len(todo)}] {j.id}: {e['status']} ({e['seconds']} s)")
            return e

        if parallel <= 1:
            for j in todo:
                one(j)
        else:
            with ThreadPoolExecutor(parallel) as ex:
                list(ex.map(one, todo))
        return done
