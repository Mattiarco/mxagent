"""Campagna di valutazione: YAML validato -> lista deterministica di job.

Un job è (task, modalità, variante, insieme di modelli, seed). La variante è
``main`` per le modalità principali o il nome di un'ablation. Il confronto è a
pari budget: tutti i job usano lo stesso ``budget`` della config di base.

Politica della libreria (anti-leakage, indipendenza dall'ordine):

``fresh``   ogni run parte da una libreria vuota tutta sua (default): il riuso
            avviene solo dentro la run, i risultati non dipendono dall'ordine
            dei task né dai job paralleli.
``frozen``  libreria costruita prima (``library.path``) e in sola lettura; i
            componenti del task in esame sono comunque esclusi.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..run.baselines import MODES

ALL_MODES = (*MODES, "full")
LLM_ROLES = ("planner", "coder", "fixer", "reflector", "escalate", "optimizer")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSet(_Strict):
    """``all`` assegna un modello a ogni ruolo; ``roles`` li sovrascrive uno per uno.

    Le baseline usano un solo modello: ``baseline`` (default: quello del coder).
    """

    all: str | None = None
    roles: dict[str, str] = {}
    baseline: str | None = None

    @field_validator("roles")
    @classmethod
    def _roles(cls, v: dict[str, str]) -> dict[str, str]:
        bad = set(v) - set(LLM_ROLES)
        if bad:
            raise ValueError(f"ruoli sconosciuti {sorted(bad)}; disponibili {LLM_ROLES}")
        return v

    @model_validator(mode="after")
    def _some(self) -> "ModelSet":
        if not self.all and not self.roles:
            raise ValueError("insieme di modelli vuoto: serve all o roles")
        return self

    def overrides(self, mode: str) -> dict[str, Any]:
        if mode != "full":
            m = self.baseline or self.roles.get("coder") or self.all
            return {"llm.roles.coder.model": m}
        out = {f"llm.roles.{r}.model": self.all for r in LLM_ROLES} if self.all else {}
        out.update({f"llm.roles.{r}.model": m for r, m in self.roles.items()})
        return out


class Ablations(_Strict):
    tasks: list[str] = []  # default: tutti i task della campagna
    models: list[str] = []  # default: tutti gli insiemi di modelli
    seeds: int | None = None  # default: come la campagna
    variants: dict[str, dict[str, Any]] = {}


class LibraryPolicy(_Strict):
    policy: str = "fresh"
    path: str | None = None

    @model_validator(mode="after")
    def _ok(self) -> "LibraryPolicy":
        if self.policy not in ("fresh", "frozen"):
            raise ValueError("library.policy: fresh | frozen")
        if self.policy == "frozen" and not self.path:
            raise ValueError("library.policy=frozen richiede library.path")
        return self


class Campaign(_Strict):
    name: str
    config: str = "configs/default.yaml"
    benchmarks_dir: str = "benchmarks"
    tasks: list[str]
    modes: list[str] = list(ALL_MODES)
    seeds: int = Field(10, ge=1)
    seed_start: int = 0
    models: dict[str, ModelSet]
    ablations: Ablations = Ablations()
    library: LibraryPolicy = LibraryPolicy()
    set: dict[str, Any] = {}  # override globali della config
    ppa_backends: list[str] = ["generic"]  # sintesi dei design passati (tutte le modalità)
    job_timeout_s: int | None = None  # default: 2 x budget.max_wall_s + 1800

    @field_validator("modes")
    @classmethod
    def _modes(cls, v: list[str]) -> list[str]:
        bad = set(v) - set(ALL_MODES)
        if bad:
            raise ValueError(f"modalità sconosciute {sorted(bad)}; disponibili {ALL_MODES}")
        return v

    @model_validator(mode="after")
    def _refs(self) -> "Campaign":
        for t in self.ablations.tasks:
            if t not in self.tasks:
                raise ValueError(f"ablations.tasks: {t} non è fra i task della campagna")
        for m in self.ablations.models:
            if m not in self.models:
                raise ValueError(f"ablations.models: {m} non è fra gli insiemi di modelli")
        if "main" in self.ablations.variants:
            raise ValueError("'main' è riservato alle modalità principali")
        return self

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True)
                              .encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Job:
    task: str
    mode: str
    variant: str  # main | nome dell'ablation
    models: str
    seed: int
    overrides: tuple[tuple[str, Any], ...]

    @property
    def id(self) -> str:
        return f"{self.task}__{self.mode}__{self.variant}__{self.models}__s{self.seed}"

    def set_args(self) -> list[str]:
        return [f"{k}={json.dumps(v) if not isinstance(v, str) else v}" for k, v in self.overrides]


def load_campaign(path: str | Path) -> Campaign:
    return Campaign.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def jobs(c: Campaign, out_dir: Path) -> list[Job]:
    """Job della campagna in ordine deterministico (seed esterno: risultati parziali bilanciati)."""
    out: list[Job] = []

    def lib(job_id: str) -> dict[str, Any]:
        if c.library.policy == "frozen":
            return {"library.path": c.library.path, "library.readonly": True}
        return {"library.path": str((out_dir / "libs" / job_id).as_posix()), "library.readonly": False}

    def add(task: str, mode: str, variant: str, mset: str, seed: int, extra: dict[str, Any]) -> None:
        ov = {**c.set, **c.models[mset].overrides(mode), **extra}
        probe = Job(task, mode, variant, mset, seed, ())
        ov.update(lib(probe.id) if mode == "full" else {})
        out.append(Job(task, mode, variant, mset, seed, tuple(sorted(ov.items()))))

    seeds = range(c.seed_start, c.seed_start + c.seeds)
    for s in seeds:
        for mset in c.models:
            for t in c.tasks:
                for mode in c.modes:
                    add(t, mode, "main", mset, s, {})
    ab = c.ablations
    ab_seeds = range(c.seed_start, c.seed_start + (ab.seeds or c.seeds))
    for s in ab_seeds:
        for mset in ab.models or list(c.models):
            for t in ab.tasks or c.tasks:
                for name, ov in ab.variants.items():
                    add(t, "full", name, mset, s, ov)
    return out
