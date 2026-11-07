"""JSON schema per l'output strutturato (``format`` di Ollama) e validazione del piano."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..bench.subkernels import CATALOG

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": ["decode", "multiply", "scale_convert", "quantize",
                                                  "dot_product", "mac", "other"]},
        "element_format": {"type": "string"},
        "scale_format": {"type": "string"},
        "block_size": {"type": "integer"},
        "output_format": {"type": "string"},
        "accumulation": {"type": "string", "enum": ["exact", "rounded", "none"]},
        "latency_cycles": {"type": "integer"},
        "corner_cases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["operation", "element_format", "block_size", "output_format", "latency_cycles"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "enum": ["monolithic", "decompose"]},
        "blocks": {"type": "array", "items": {
            "type": "object",
            "properties": {"key": {"type": "string", "enum": sorted(CATALOG)},
                           "module": {"type": "string"},
                           "params": {"type": "object"},
                           "purpose": {"type": "string"}},
            "required": ["key", "module"]}},
        "glue": {"type": "string"},
    },
    "required": ["strategy"],
}


@dataclass
class PlannedBlock:
    key: str
    module: str
    params: dict[str, Any]
    purpose: str = ""


@dataclass
class Plan:
    strategy: str
    blocks: list[PlannedBlock] = field(default_factory=list)
    glue: str = ""
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"strategy": self.strategy, "glue": self.glue, "problems": self.problems,
                "blocks": [b.__dict__ for b in self.blocks]}


def validate_plan(data: Any, top: str) -> Plan:
    """Piano valido o ripiego monolitico (con i motivi registrati)."""
    if not isinstance(data, dict) or data.get("strategy") not in ("monolithic", "decompose"):
        return Plan("monolithic", problems=["piano non valido: JSON assente o strategia sconosciuta"])
    if data["strategy"] == "monolithic":
        return Plan("monolithic", glue=str(data.get("glue", "")))
    blocks, probs, names = [], [], set()
    for b in data.get("blocks") or []:
        if not isinstance(b, dict):
            continue
        key = b.get("key")
        if key not in CATALOG:
            probs.append(f"blocco sconosciuto: {key}")
            continue
        mod = re.sub(r"\W", "", str(b.get("module") or "")) or "".join(p.title() for p in key.split("_"))
        if not mod[0].isupper():
            mod = mod[0].upper() + mod[1:]
        if mod == top or mod in names:
            mod = f"{mod}_{len(names)}"
        raw = b.get("params") if isinstance(b.get("params"), dict) else {}
        defaults = CATALOG[key].defaults
        params = {k: int(v) for k, v in raw.items() if k in defaults and isinstance(v, (int, float))}
        dropped = set(raw) - set(params)
        if dropped:
            probs.append(f"{key}: parametri ignorati {sorted(dropped)}")
        try:
            CATALOG[key].params(params)
        except ValueError as e:
            probs.append(str(e))
            continue
        names.add(mod)
        blocks.append(PlannedBlock(key, mod, params, str(b.get("purpose", ""))))
    if not blocks:
        return Plan("monolithic", problems=probs + ["nessun blocco valido: monolitico"])
    return Plan("decompose", blocks, str(data.get("glue", "")), probs)
