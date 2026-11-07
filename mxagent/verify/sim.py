"""Stadio 3: simulazione con Verilator + cocotb.

Il modello Verilator viene compilato una sola volta per candidato
(:class:`Simulator`); ogni chiamata a :meth:`Simulator.run` lancia cocotb in un
sottoprocesso con un file di vettori, così il delta-debugging può fare molti
round di simulazione senza ricompilare. Un crash o un timeout del simulatore
diventano errori strutturati, non eccezioni.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from ..bench.task import Task
from ..util.shell import run

ENTRY = [sys.executable, "-m", "mxagent.verify.sim_entry"]


@dataclass
class SimRun:
    ok: bool
    outputs: list[list[int | None] | None]
    error: str = ""
    seconds: float = 0.0
    log: str = ""


class Simulator:
    def __init__(self, task: Task, sv_files: list[Path], build_dir: Path, timeout: float = 900,
                 simulator: str = "verilator"):
        self.task = task
        self.sv_files = sv_files
        self.build_dir = build_dir
        self.timeout = timeout
        self.simulator = simulator
        self.built = False
        self.build_seconds = 0.0
        self._n = 0

    def _call(self, action: str, extra: list[str]) -> tuple[bool, str, float]:
        args = [*ENTRY, action, "--top", self.task.top, "--build-dir", str(self.build_dir),
                "--simulator", self.simulator, *extra]
        r = run(args, timeout=self.timeout)
        if r.timed_out:
            return False, f"timeout della simulazione ({self.timeout:.0f}s)", r.seconds
        return r.ok, r.log, r.seconds

    def build(self) -> tuple[bool, str]:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        ok, log, sec = self._call("build", ["--sources", *[str(p) for p in self.sv_files]])
        self.built, self.build_seconds = ok, sec
        return ok, log

    def run(self, vectors: list[list[int]]) -> SimRun:
        if not self.built:
            ok, log = self.build()
            if not ok:
                return SimRun(False, [], "compilazione Verilator fallita", self.build_seconds, log)
        self._n += 1
        vec_file = self.build_dir / f"vectors_{self._n}.json"
        res_file = self.build_dir / f"results_{self._n}.json"
        res_file.unlink(missing_ok=True)
        t = self.task
        vec_file.write_text(json.dumps({
            "kind": t.timing.kind, "latency": t.timing.latency,
            "inputs": [p.name for p in t.inputs], "outputs": [p.name for p in t.outputs],
            "vectors": vectors}), encoding="utf-8")
        ok, log, sec = self._call("test", ["--vectors", str(vec_file), "--results", str(res_file)])
        if not res_file.is_file():
            return SimRun(False, [], "la simulazione non ha prodotto risultati (crash?)", sec, log)
        outs = json.loads(res_file.read_text(encoding="utf-8"))["outputs"]
        return SimRun(True, outs, "", sec, log)


def runner_available() -> str | None:
    """None se cocotb + simulatore sono utilizzabili, altrimenti il motivo."""
    try:
        import cocotb  # noqa: F401
    except ImportError:
        return "cocotb non installato"
    from ..util.shell import which
    if not which("verilator"):
        return "verilator non trovato"
    if os.name == "nt":
        return "cocotb+Verilator richiede Linux (usa WSL o il server)"
    return None
