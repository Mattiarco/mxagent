"""Registro di una run: ``runs/<id>/`` con tutto ciò che serve a riprodurla.

    config.yaml     configurazione effettiva (dopo override)
    meta.json       task, modalità, seed, versioni toolchain, digest dei modelli, hash dei prompt
    trace.json      eventi in ordine: chiamate LLM (prompt, risposta, token) e verifiche
    candidates/NN/  sorgenti di ogni iterazione
    final/          sorgenti e Verilog del miglior candidato
    metrics.json    metriche della run (dai tool, mai stimate); solo se la run arriva in fondo
    result.json     esito SEMPRE presente: done | failed | interrupted (vedi run/result.py)
"""
from __future__ import annotations

import json
import platform
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import __version__
from .. import versions as V


def new_run_id(task: str, mode: str, model: str, seed: int) -> str:
    m = re.sub(r"[^\w.-]", "-", model)
    return f"{datetime.now():%Y%m%d-%H%M%S}_{task}_{mode}_{m}_s{seed}"


class Recorder:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        self.t0 = time.monotonic()

    def write_json(self, name: str, data: Any) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def event(self, kind: str, **data: Any) -> None:
        self.events.append({"t": round(time.monotonic() - self.t0, 2), "kind": kind, **data})
        self.write_json("trace.json", self.events)  # sempre su disco: una run interrotta resta leggibile

    def meta(self, **data: Any) -> None:
        self.write_json("meta.json", {
            "mxagent": __version__, "python": platform.python_version(), "host": platform.node(),
            "toolchain": {"chisel": V.CHISEL_VERSION, "firtool": V.FIRTOOL_VERSION,
                          "scala": V.SCALA_VERSION, "sbt": V.SBT_VERSION},
            "started": datetime.now().isoformat(timespec="seconds"), **data})

    def save_candidate(self, it: int, sources: dict[str, str]) -> None:
        for name, code in sources.items():
            self.write_text(f"candidates/{it:02d}/{name}", code)

    def save_final(self, sources: dict[str, str], verilog: list[str]) -> None:
        for name, code in sources.items():
            self.write_text(f"final/{name}", code)
        for v in verilog:
            p = Path(v)
            if p.is_file():
                (self.root / "final").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, self.root / "final" / p.name)
