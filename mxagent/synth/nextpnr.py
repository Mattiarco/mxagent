"""Place & route su Lattice ECP5 con nextpnr: frequenza massima misurata.

Il DUT è inserito nel wrapper registrato (``wrapper.py``), sintetizzato con
``synth_ecp5`` e piazzato/instradato con un seed per volta. La Fmax di un
singolo seed è rumorosa: si riportano mediana, minimo e massimo.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from ..bench.task import Task
from ..util.shell import run, which
from . import yosys
from .wrapper import WRAP_TOP, wrapper


def available() -> bool:
    return which("nextpnr-ecp5") is not None and yosys.available()


def parse_report(text: str) -> float | None:
    """Fmax (MHz) del clock del wrapper dal file ``--report`` di nextpnr."""
    d = json.loads(text)
    fm = d.get("fmax") or {}
    vals = [v.get("achieved") for v in fm.values() if isinstance(v, dict)]
    vals = [float(v) for v in vals if v is not None]
    return min(vals) if vals else None


def fmax_metrics(fmax: list[float], clock_ns: float) -> dict:
    """Statistiche sui seed. Lo slack è rispetto a ``clock_ns`` sul seed PEGGIORE;
    ``crit_ns`` (mediana) è il cammino registro -> DUT -> registro del wrapper, cioè
    il ritardo combinatorio del modulo più clock-to-q e setup."""
    med, worst = statistics.median(fmax), min(fmax)
    return {"fmax_mhz": round(med, 2), "fmax_mean": round(statistics.fmean(fmax), 2),
            "fmax_min": round(worst, 2), "fmax_max": round(max(fmax), 2),
            "fmax_per_seed": [round(f, 2) for f in fmax],
            "crit_ns": round(1000.0 / med, 3), "clock_ns": clock_ns,
            "slack_ns": round(clock_ns - 1000.0 / worst, 3), "seeds": len(fmax)}


def pnr_ecp5(files: list[Path], task: Task, work: Path, *, device: str = "85k",
             package: str = "CABGA381", clock_ns: float = 10.0, seeds: list[int] | None = None,
             timeout: float = 1800) -> yosys.SynthResult:
    if not available():
        return yosys.SynthResult("nextpnr", False, error="nextpnr-ecp5 non trovato nel PATH",
                                 skipped=True)
    work.mkdir(parents=True, exist_ok=True)
    wrap = work / "wrap.sv"
    wrap.write_text(wrapper(task), encoding="utf-8")
    s = yosys.synth([*files, wrap], WRAP_TOP, "ecp5", work / "synth", timeout=timeout,
                    netlist="wrap.json")
    netlist = work / "synth" / "wrap.json"
    res = yosys.SynthResult("nextpnr", False, seconds=s.seconds, log=s.log)
    if not s.ok:
        res.error = f"sintesi del wrapper: {s.error}"
        return res
    fmax: list[float] = []
    util: dict = {}
    for seed in seeds or [1]:
        rep = work / f"report_s{seed}.json"
        rep.unlink(missing_ok=True)
        r = run(["nextpnr-ecp5", f"--{device}", "--package", package, "--json", str(netlist.resolve()),
                 "--freq", f"{1000.0 / clock_ns:.3f}", "--seed", str(seed), "--report", str(rep.resolve()),
                 "--timing-allow-fail", "-q", "-l", str((work / f"pnr_s{seed}.log").resolve())],
                cwd=work, timeout=timeout)
        res.seconds = round(res.seconds + r.seconds, 2)
        if not rep.is_file():
            tail = [ln for ln in r.log.splitlines() if "ERROR" in ln] or r.log.splitlines()[-3:]
            res.error = f"nextpnr seed {seed}: " + " | ".join(tail)[:500]
            continue
        txt = rep.read_text()
        f = parse_report(txt)
        if f is not None:
            fmax.append(f)
        util = json.loads(txt).get("utilization", util)
    if not fmax:
        res.error = res.error or "nessuna Fmax nel report di nextpnr"
        return res
    res.metrics = fmax_metrics(fmax, clock_ns)
    res.wrapped = True
    res.cells = {k: int(v["used"]) for k, v in util.items() if isinstance(v, dict) and v.get("used")}
    res.ok = True
    return res
