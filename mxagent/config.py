"""Configurazione YAML validata (pydantic). Una run salva sempre la config
effettiva risolta in ``runs/<id>/config.yaml``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .formats.numerics import Numerics


#: ruoli LLM generativi (``embed`` è a parte: resta sempre il modello di embedding)
LLM_ROLES = ("planner", "coder", "fixer", "reflector", "escalate", "optimizer")


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
    #: profili di modello: nome -> {ruolo: modello}, con ``all`` per tutti i ruoli LLM
    profiles: dict[str, dict[str, str]] = {}
    #: profilo applicato (``--model-profile``); None = ruoli come scritti nella config
    profile: str | None = None

    @field_validator("roles")
    @classmethod
    def _required_roles(cls, v: dict[str, ModelRole]) -> dict[str, ModelRole]:
        missing = {"planner", "coder", "fixer", "reflector"} - set(v)
        if missing:
            raise ValueError(f"ruoli LLM mancanti: {sorted(missing)}")
        return v

    @field_validator("profiles")
    @classmethod
    def _profile_roles(cls, v: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
        for name, p in v.items():
            bad = set(p) - {*LLM_ROLES, "all"}
            if bad:
                raise ValueError(f"profilo {name}: ruoli sconosciuti {sorted(bad)}")
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


class RunsConfig(_Strict):
    """Cartelle delle run."""

    #: False: a fine run si cancellano sbt/target e gli oggetti di Verilator (run/cleanup.py)
    keep_build_artifacts: bool = False


class Config(_Strict):
    llm: LLMConfig
    agent: AgentConfig = AgentConfig()
    library: LibraryConfig = LibraryConfig()
    budget: BudgetConfig = BudgetConfig()
    sim: SimConfig = SimConfig()
    synth: SynthConfig = SynthConfig()
    numerics: dict[str, Any] = {}
    runs_dir: str = "runs"
    runs: RunsConfig = RunsConfig()

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


def profile_roles(profiles: dict[str, dict[str, str]], name: str) -> dict[str, str]:
    """Modello di ogni ruolo LLM secondo il profilo ``name`` (``embed`` escluso)."""
    if name not in profiles:
        raise ValueError(f"profilo di modello sconosciuto: {name}; disponibili {sorted(profiles)}")
    p = profiles[name]
    out = {r: p.get(r, p.get("all")) for r in LLM_ROLES}
    missing = [r for r, m in out.items() if not m]
    if missing:
        raise ValueError(f"profilo {name}: nessun modello per {missing} (usa 'all' o elencali)")
    return out  # type: ignore[return-value]


def apply_profile(data: dict[str, Any], name: str) -> None:
    """Imposta nel YAML grezzo il modello di tutti i ruoli LLM secondo il profilo.

    Temperature e contesto restano quelli del ruolo; un ruolo assente (es. ``optimizer``)
    viene creato con le impostazioni del ``reflector``."""
    llm = data.setdefault("llm", {})
    roles = llm.setdefault("roles", {})
    for r, m in profile_roles(llm.get("profiles") or {}, name).items():
        base = roles.get(r) or dict(roles.get("reflector") or {})
        roles[r] = {**base, "model": m}
    llm["profile"] = name


def effective_models(cfg: "Config", mode: str = "full") -> dict[str, str]:
    """Modello che ogni ruolo usa davvero nella run (``models_by_role``).

    Le baseline usano solo il ``coder``; con ``agent.routing: false`` tutti i ruoli LLM
    del sistema completo usano il coder; ``optimizer`` assente ripiega sul reflector."""
    roles = cfg.llm.roles
    if mode != "full":
        return {"coder": roles["coder"].model}
    out = {}
    for r in LLM_ROLES:
        if not cfg.agent.routing:
            out[r] = roles["coder"].model
        elif r in roles:
            out[r] = roles[r].model
        elif r == "optimizer":
            out[r] = roles["reflector"].model
        elif r == "escalate":
            out[r] = roles["planner"].model
    if "embed" in roles:
        out["embed"] = roles["embed"].model
    return out


def profile_label(cfg: "Config", mode: str = "full") -> tuple[str, dict[str, dict[str, str]]]:
    """Etichetta del profilo (nome della run, meta.json) e differenze rispetto al profilo
    dichiarato.

    Con un profilo dichiarato l'etichetta è quel nome e ``mismatches`` elenca i ruoli
    che usano un altro modello (es. un ``--set`` successivo). Senza profilo l'etichetta
    è il primo profilo della config che coincide coi ruoli effettivi, ``all-<modello>``
    se tutti i ruoli usano lo stesso modello, altrimenti ``custom``."""
    eff = {r: m for r, m in effective_models(cfg, mode).items() if r != "embed"}
    profiles = cfg.llm.profiles
    if cfg.llm.profile:
        exp = profile_roles(profiles, cfg.llm.profile)
        mism = {r: {"expected": exp[r], "actual": m} for r, m in eff.items() if exp[r] != m}
        return cfg.llm.profile, mism
    for name in profiles:
        try:
            exp = profile_roles(profiles, name)
        except ValueError:
            continue
        if all(exp[r] == m for r, m in eff.items()):
            return name, {}
    models = set(eff.values())
    return (f"all-{models.pop()}" if len(models) == 1 else "custom"), {}


def load_config(path: str | Path, overrides: dict[str, Any] | None = None,
                profile: str | None = None) -> Config:
    """Config YAML + profilo di modello + override puntuali (in quest'ordine: un
    ``--set llm.roles.X.model=...`` vince sul profilo, e viene segnalato come mismatch)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    overrides = dict(overrides or {})
    name = profile or overrides.pop("llm.profile", None) or (data.get("llm") or {}).get("profile")
    if name:
        apply_profile(data, name)
    for dotted, value in overrides.items():
        cur = data
        *parents, last = dotted.split(".")
        for p in parents:
            cur = cur.setdefault(p, {})
        cur[last] = value
    return Config.model_validate(data)
