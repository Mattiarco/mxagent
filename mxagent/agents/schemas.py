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
        "combinational": {"type": "boolean"},
        "output_scale": {"type": "string"},
        "ports": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"},
                           "direction": {"type": "string", "enum": ["in", "out"]},
                           "type": {"type": "string"},
                           "width": {"type": "integer"},
                           "elems": {"type": "integer"},
                           "meaning": {"type": "string"}},
            "required": ["name", "direction", "width"]}},
        "corner_cases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["operation", "element_format", "block_size", "output_format", "latency_cycles",
                 "ports"],
}


def port_scale(p) -> str | None:
    """Scala di una porta in virgola fissa: ``y = value * 2^f``."""
    if p.type in ("sint", "uint") and p.frac_bits:
        return f"{p.name} = value * 2^{p.frac_bits} (i.e. value * {2 ** p.frac_bits})"
    return None


def reconcile_spec(data: Any, task) -> tuple[dict, list[str]]:
    """Spec strutturata con l'interfaccia ESATTA del benchmark.

    Porte (nome, direzione, tipo, larghezze, scala) e temporizzazione vengono da
    ``task.yaml``; dall'LLM si tengono solo i campi descrittivi (operazione,
    formati, casi limite, significato delle porte). Ogni differenza fra ciò che
    l'LLM ha estratto e il benchmark è restituita in ``mismatches``.
    """
    spec = dict(data) if isinstance(data, dict) else {}
    mism: list[str] = []
    llm_ports = {str(q.get("name")): q for q in spec.get("ports") or [] if isinstance(q, dict)}
    ports = []
    for p in task.ports:
        q = llm_ports.pop(p.name, None)
        if q is None:
            mism.append(f"port {p.name}: missing in the extracted spec")
        else:
            want = {"direction": p.dir, "width": p.total_width}
            if "elems" in q:
                want["elems"] = p.elems
            for k, v in want.items():
                got = q.get(k)
                if got != v and not (k == "width" and got in (p.elem_width,) and p.elems > 1):
                    mism.append(f"port {p.name}: {k} {got!r} != {v!r}")
        ports.append({"name": p.name, "direction": p.dir, "type": p.type, "width": p.total_width,
                      "elem_width": p.elem_width, "elems": p.elems, "frac_bits": p.frac_bits,
                      "signed": p.type == "sint", "scale": port_scale(p),
                      "meaning": p.doc or (str(q.get("meaning", "")) if q else "")})
    mism += [f"port {n}: not in the benchmark interface" for n in llm_ports]
    latency = task.timing.latency if task.timing.kind == "pipe" else 0
    if "latency_cycles" in spec and spec["latency_cycles"] != latency:
        mism.append(f"latency_cycles {spec['latency_cycles']!r} != {latency}")
    comb = task.timing.kind == "comb"
    if "combinational" in spec and spec["combinational"] != comb:
        mism.append(f"combinational {spec['combinational']!r} != {comb}")
    scales = [port_scale(p) for p in task.outputs if port_scale(p)]
    spec.update(ports=ports, latency_cycles=latency, combinational=comb,
                output_scale="; ".join(scales) if scales else (spec.get("output_scale") or None),
                interface_source="benchmark task.yaml")
    return spec, mism

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
