"""Configurazione YAML validata (pydantic). Una run salva sempre la config
effettiva risolta in ``runs/<id>/config.yaml``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .formats.numerics import Numerics


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelRole(_Strict):
    """Modello Ollama per un ruolo (routing: piccolo per sintassi, grande per piano/bug)."""

    model: str
    temperature: float = Field(0.2, ge=0, le=2)
    num_ctx: int = Field(16384, ge=2048)
    num_predict: int = Field(4096, ge=128)


class LLMConfig(_Strict):
    host: str | None = None  # default: $OLLAMA_HOST o http://localhost:11434
    seed: int = 0
    timeout_s: float = 600
    max_retries: int = 2
    cache_dir: str = ".cache/llm"
    roles: dict[str, ModelRole]

    @field_validator("roles")
    @classmethod
    def _required_roles(cls, v: dict[str, ModelRole]) -> dict[str, ModelRole]:
        missing = {"planner", "coder", "fixer", "reflector"} - set(v)
        if missing:
            raise ValueError(f"ruoli LLM mancanti: {sorted(missing)}")
        return v

    def models(self) -> list[str]:
        return list(dict.fromkeys(r.model for r in self.roles.values()))


class BudgetConfig(_Strict):
    """Budget per run: identico fra baseline e sistema completo (confronto a pari costo)."""

    max_llm_calls: int = 60
    max_tokens: int = 400_000
    max_wall_s: int = 1800
    best_of_n: int = 3
    max_fix_iters: int = 6


class SimConfig(_Strict):
    simulator: str = "verilator"
    sbt_mode: str = "client"  # client = server sbt persistente (~1 s a caldo), batch = a freddo
    max_examples: int = 3  # vettori minimi riportati nel feedback
    ddmin_rounds: int = 8
    random_vectors: int = 2000
    seed: int = 0
    timeout_s: int = 900


SYNTH_BACKENDS = ("generic", "liberty", "ecp5", "xc7", "nextpnr", "opensta", "vivado")


class SynthConfig(_Strict):
    """Sintesi e PPA (M5). Vedi docs/ppa.md per cosa misura ogni backend."""

    # backend del report finale; quelli assenti sulla macchina vengono saltati e segnalati
    backends: list[str] = ["generic", "ecp5", "xc7", "nextpnr"]
    primary: str = "generic"  # metrica che guida il PPA Optimizer: generic | liberty
    liberty: str | None = None  # file .lib (liberty e opensta)
    fpga_part: str = "xc7a100tcsg324-1"  # solo vivado
    ecp5_device: str = "85k"
    ecp5_package: str = "CABGA381"
    clock_ns: float = Field(10.0, gt=0)
    pnr_seeds: list[int] = [1, 2, 3]
    timeout_s: int = 1800
    # PPA Optimizer: obiettivo, iterazioni, soglie di accettazione
    objective: str = "adp"  # area | delay | adp (area x delay)
    optimize_iters: int = Field(4, ge=0)
    min_gain: float = Field(0.01, ge=0)  # miglioramento relativo minimo per accettare
    tolerance: float = Field(0.02, ge=0)  # peggioramento ammesso sull'altra metrica

    @field_validator("backends")
    @classmethod
    def _known(cls, v: list[str]) -> list[str]:
        bad = set(v) - set(SYNTH_BACKENDS)
        if bad:
            raise ValueError(f"backend di sintesi sconosciuti {sorted(bad)}; disponibili {SYNTH_BACKENDS}")
        return v

    @field_validator("primary")
    @classmethod
    def _primary(cls, v: str) -> str:
        if v not in ("generic", "liberty"):
            raise ValueError("synth.primary: generic | liberty")
        return v

    @field_validator("objective")
    @classmethod
    def _objective(cls, v: str) -> str:
        if v not in ("area", "delay", "adp"):
            raise ValueError("synth.objective: area | delay | adp")
        return v


class AgentConfig(_Strict):
    """Meccanismi del sistema completo; ognuno si spegne per le ablation."""

    decompose: bool = True  # Planner + blocchi del catalogo verificati singolarmente
    spec_parser: bool = True
    kb: bool = True  # regole Chisel 6 nel prompt + sanitizer + trappole richiamate dagli errori
    format_card: bool = True  # conoscenza di dominio (formati, semantica numerica) nei prompt
    structured_feedback: bool = True  # False: log grezzi come la Baseline B
    library: bool = True  # riuso esatto + esempi verificati (RAG)
    best_of_n: int = Field(3, ge=1, le=8)
    escape: bool = True  # anti-loop: ricampiona / modello più grande / rinuncia
    routing: bool = True  # False: un solo modello (coder) per tutti i ruoli
    memory_items: int = Field(4, ge=0, le=12)  # tentativi riassunti nel prompt del Reflector
    escape_after_repeats: int = Field(2, ge=1)  # stesso errore N volte di fila -> escape
    escape_after_stall: int = Field(3, ge=1)  # N iterazioni senza miglioramento -> escape
    max_block_iters: int = Field(4, ge=0)  # correzioni per blocco del catalogo
    ppa: bool = True  # PPA Optimizer dopo il pass (non cambia passed/time_to_pass)


class LibraryConfig(_Strict):
    path: str = "library"
    reuse_exact: bool = True
    examples: int = Field(2, ge=0, le=5)
    exclude_current_task: bool = True  # anti-leakage
    readonly: bool = False  # libreria congelata (valutazione): niente aggiunte durante la run


class Config(_Strict):
    llm: LLMConfig
    agent: AgentConfig = AgentConfig()
    library: LibraryConfig = LibraryConfig()
    budget: BudgetConfig = BudgetConfig()
    sim: SimConfig = SimConfig()
    synth: SynthConfig = SynthConfig()
    numerics: dict[str, Any] = {}
    runs_dir: str = "runs"

    @field_validator("numerics")
    @classmethod
    def _valid_numerics(cls, v: dict[str, Any]) -> dict[str, Any]:
        Numerics.from_dict(v)  # solleva ValueError se non valido
        return v

    @property
    def numerics_obj(self) -> Numerics:
        return Numerics.from_dict(self.numerics)

    def dump(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for dotted, value in (overrides or {}).items():
        cur = data
        *parents, last = dotted.split(".")
        for p in parents:
            cur = cur.setdefault(p, {})
        cur[last] = value
    return Config.model_validate(data)
