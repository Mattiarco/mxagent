"""Stadio 2: lint di sintetizzabilità con Verilator (prima della simulazione).

Classifica i warning: quelli che indicano hardware sbagliato (latch, loop
combinatori, troncamenti, multi-driver) bloccano; gli stilistici no. Ogni
warning viene riportato al sorgente Scala tramite i locatori ``// @[File.scala l:c]``
che firtool lascia nel Verilog.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..util.shell import run, which

#: codici Verilator che indicano un difetto di sintetizzabilità o funzionale
BLOCKING = {"LATCH", "NOLATCH", "COMBDLY", "UNOPTFLAT", "UNOPT", "ALWCOMBORDER", "MULTIDRIVEN",
            "WIDTHTRUNC", "WIDTH", "CASEINCOMPLETE", "CASEOVERLAP", "INFINITELOOP", "UNDRIVEN"}
#: rumore tipico del Verilog emesso da firtool
IGNORED = {"DECLFILENAME", "UNUSEDSIGNAL", "UNUSEDPARAM", "UNUSED", "PINCONNECTEMPTY",
           "GENUNNAMED", "VARHIDDEN", "SYNCASYNCNET", "PROCASSINIT", "WIDTHEXPAND", "WIDTHXZEXPAND"}

_MSG = re.compile(r"^%(?P<sev>Warning|Error)(?:-(?P<code>[A-Z0-9_]+))?:\s*"
                  r"(?:(?P<file>(?:[A-Za-z]:)?[^:\s]+\.s?v):(?P<line>\d+):(?:(?P<col>\d+):)?)?"
                  r"\s*(?P<msg>.*)$")
_LOC = re.compile(r"@\[(?P<file>[\w./\\-]+\.scala)\s+(?P<line>\d+)(?::(?P<col>\d+))?")


@dataclass
class LintIssue:
    severity: str  # error | warning
    code: str
    message: str
    sv_file: str | None = None
    sv_line: int | None = None
    scala_loc: str | None = None  # "File.scala:12:5" se ricavabile
    blocking: bool = False

    def render(self) -> str:
        loc = self.scala_loc or (f"{self.sv_file}:{self.sv_line}" if self.sv_file else "?")
        return f"[verilator {self.code or self.severity}] {loc}: {self.message}"


@dataclass
class LintResult:
    ok: bool
    issues: list[LintIssue] = field(default_factory=list)
    seconds: float = 0.0
    log: str = ""

    @property
    def blocking(self) -> list[LintIssue]:
        return [i for i in self.issues if i.blocking]

    def to_dict(self) -> dict:
        return {"ok": self.ok, "issues": [asdict(i) for i in self.issues], "seconds": round(self.seconds, 2)}


def scala_locator(sv_path: Path, line: int) -> str | None:
    """Locatore Scala più vicino (sulla riga o fino a 3 righe sopra)."""
    try:
        lines = sv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for ln in range(line, max(0, line - 4), -1):
        if 0 < ln <= len(lines):
            m = _LOC.search(lines[ln - 1])
            if m:
                return f"{Path(m['file']).name}:{m['line']}" + (f":{m['col']}" if m["col"] else "")
    return None


def parse_verilator(log: str, sv_files: list[Path]) -> list[LintIssue]:
    by_name = {p.name: p for p in sv_files}
    out: list[LintIssue] = []
    for raw in log.splitlines():
        m = _MSG.match(raw.strip())
        if not m:
            continue
        code = m["code"] or ""
        if m["sev"] == "Warning" and code in IGNORED:
            continue
        if m["msg"].startswith("Exiting due to"):
            continue
        sv = Path(m["file"]).name if m["file"] else None
        line = int(m["line"]) if m["line"] else None
        loc = scala_locator(by_name[sv], line) if sv in by_name and line else None
        sev = "error" if m["sev"] == "Error" else "warning"
        out.append(LintIssue(sev, code, m["msg"].strip(), sv, line, loc,
                             blocking=sev == "error" or code in BLOCKING))
    return out


def lint(sv_files: list[Path], top: str, timeout: float = 300) -> LintResult:
    if not which("verilator"):
        return LintResult(False, [LintIssue("error", "TOOL", "verilator non trovato", blocking=True)])
    cmd = ["verilator", "--lint-only", "-Wall", "-Wno-fatal", "--top-module", top,
           *[str(p) for p in sv_files]]
    r = run(cmd, timeout=timeout)
    issues = parse_verilator(r.log, sv_files)
    if r.timed_out:
        issues.append(LintIssue("error", "TIMEOUT", f"lint oltre {timeout}s", blocking=True))
    ok = not any(i.blocking for i in issues) and (r.ok or not r.log.strip())
    return LintResult(ok, issues, r.seconds, r.log)
