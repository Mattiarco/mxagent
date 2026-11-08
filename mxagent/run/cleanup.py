"""Pulizia di fine run: via gli artefatti di build, restano i dati della run.

Una run di b01 occupava 43 MB, quasi tutti in ``sbt/target`` (jar e classi) e
negli oggetti di Verilator (``sim/*.o``, ``*.a``, eseguibili). Si cancellano:

* ``sbt/target``, ``sbt/project/target``, ``sbt/project/project``, ``sbt/.bsp``;
* nelle cartelle di simulazione (``sim/`` e ``sim_build/``) tutto tranne vettori,
  risultati e log (``.json``, ``.xml``, ``.log``, ``.txt``).

Restano log, report, vettori, risultati, sorgenti e ``.sv``. Si disattiva con
``runs.keep_build_artifacts: true``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

SBT_DIRS = ("target", "project/target", "project/project", ".bsp")
SIM_DIRS = ("sim", "sim_build")
KEEP_SUFFIXES = {".json", ".xml", ".log", ".txt"}


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _remove(p: Path) -> int:
    n = _size(p)
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        p.unlink(missing_ok=True)
    return n - (_size(p) if p.exists() else 0)


def build_artifacts(root: Path) -> list[Path]:
    """File e cartelle da cancellare in ``root`` (la cartella della run)."""
    out = [root / "sbt" / d for d in SBT_DIRS if (root / "sbt" / d).exists()]
    for sim in (d for d in root.rglob("*") if d.is_dir() and d.name in SIM_DIRS):
        if any(sim.is_relative_to(o) for o in out):
            continue
        for f in sim.iterdir():
            if f.is_dir() or f.suffix.lower() not in KEEP_SUFFIXES:
                out.append(f)
    return out


def clean_run(root: Path) -> dict:
    """Cancella gli artefatti di build; restituisce i byte liberati e quelli rimasti."""
    root = Path(root)
    freed = sum(_remove(p) for p in build_artifacts(root))
    return {"freed_bytes": freed, "kept_bytes": _size(root)}
