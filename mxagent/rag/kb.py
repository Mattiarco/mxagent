"""Base di conoscenza delle trappole di Chisel 6 (``knowledge/chisel6_pitfalls.yaml``).

Tre usi, tutti deterministici:

* :func:`rules_text` — scheda compatta di regole inserita nel prompt del Coder;
* :func:`matching` — voci richiamate dai messaggi d'errore del verifier
  (con esempio sbagliato/corretto) per il Reflector;
* :func:`sanitize` — correzioni automatiche sicure applicate al codice prima di
  sbt (API Chisel 3, ``IO`` -> ``FlatIO``, oggetti ``main``...).

Ogni voce è verificata da ``tests/test_kb_integration.py`` con sbt e firtool veri.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

KB_FILE = Path(__file__).resolve().parent.parent.parent / "knowledge" / "chisel6_pitfalls.yaml"


@dataclass(frozen=True)
class Pitfall:
    id: str
    rule: str
    triggers: tuple[str, ...] = ()
    autofix: tuple[tuple[str, str], ...] = ()
    wrong: str = ""
    right: str = ""
    silent: bool = False

    def example(self) -> str:
        out = [f"[{self.id}] {self.rule}"]
        if self.wrong:
            out.append("WRONG:\n" + _body(self.wrong))
        if self.right:
            out.append("RIGHT:\n" + _body(self.right))
        return "\n".join(out)


def _body(src: str) -> str:
    return "\n".join(ln for ln in src.strip().splitlines() if not ln.startswith("import "))


@lru_cache(maxsize=4)
def load(path: str | None = None) -> tuple[Pitfall, ...]:
    data = yaml.safe_load(Path(path or KB_FILE).read_text(encoding="utf-8"))
    out = []
    for d in data:
        out.append(Pitfall(
            id=d["id"], rule=d["rule"], triggers=tuple(d.get("triggers", [])),
            autofix=tuple(tuple(x) for x in d.get("autofix", [])),
            wrong=d.get("wrong", ""), right=d.get("right", ""), silent=bool(d.get("silent"))))
    ids = [p.id for p in out]
    if len(set(ids)) != len(ids):
        raise ValueError("id duplicati nella KB")
    return tuple(out)


def rules_text() -> str:
    return "\n".join(f"- {p.rule}" for p in load())


def matching(error_text: str, limit: int = 3) -> list[Pitfall]:
    hits = []
    for p in load():
        if any(re.search(t, error_text) for t in p.triggers):
            hits.append(p)
    return hits[:limit]


# ----------------------------------------------------------------- sanitize
_MAIN_OBJ = re.compile(r"^object\s+\w+\s+extends\s+App\b|^object\s+\w+\s*\{[^\n]*$", re.M)


def _drop_main_objects(code: str) -> tuple[str, bool]:
    """Rimuove ``object X extends App {...}`` e oggetti con ``def main`` (bilanciando le graffe)."""
    changed = False
    out, i = [], 0
    for m in re.finditer(r"^object\s+\w+[^\n{]*\{", code, re.M):
        start = m.start()
        if start < i:
            continue
        depth, j = 0, m.end() - 1
        while j < len(code):
            if code[j] == "{":
                depth += 1
            elif code[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = code[start:j + 1]
        if re.search(r"extends\s+App|def\s+main\s*\(|ChiselStage|Driver\.|emitVerilog|emitSystemVerilog", body):
            out.append(code[i:start])
            i = j + 1
            changed = True
    out.append(code[i:])
    return "".join(out), changed


@dataclass
class SanitizeResult:
    code: str
    applied: list[str] = field(default_factory=list)


def sanitize(code: str) -> SanitizeResult:
    applied = []
    code, ch = _drop_main_objects(code)
    if ch:
        applied.append("no_main_object")
    new = re.sub(r"^\s*import\s+chisel3\.(Driver|stage\._|stage\.ChiselStage|iotesters\.\S*)\s*$",
                 "", code, flags=re.M)
    new = re.sub(r"^\s*import\s+circt\.stage\.\S+\s*$", "", new, flags=re.M)
    if new != code:
        applied.append("drop_stage_imports")
        code = new
    for p in load():
        for pat, rep in p.autofix:
            new = re.sub(pat, rep, code)
            if new != code:
                applied.append(p.id)
                code = new
    if "chisel3.util" not in code and re.search(
            r"\b(Cat|Log2|MuxCase|MuxLookup|Fill|Reverse|PriorityEncoder|Mux1H|log2Ceil|log2Up|PopCount)\b", code):
        code = re.sub(r"(import\s+chisel3\._\s*\n)", r"\1import chisel3.util._\n", code, count=1)
        applied.append("import_util")
    if "import chisel3._" not in code:
        code = "import chisel3._\n" + code
        applied.append("import_chisel3")
    return SanitizeResult(code, applied)
