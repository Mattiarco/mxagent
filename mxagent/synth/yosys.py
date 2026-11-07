"""Sintesi logica con Yosys: area e profondità logica misurate dallo strumento.

Flussi (``flow``):

``generic``  ``synth -flatten`` + ``abc -g cmos2``: porte NAND/NOR/NOT.
             area = transistori stimati (``stat -tech cmos``),
             ritardo = profondità logica in livelli di porta (``ltp -noff``).
             Indipendente dalla tecnologia e veloce: guida il PPA Optimizer.
``liberty``  come generic ma mappato su una libreria .lib (``abc -liberty``);
             area = ``stat -liberty`` nelle unità della libreria (di solito um^2).
``ecp5``     ``synth_ecp5``: LUT4, CCU2C (carry), PFUMX/L6MUX21, MULT18X18D, FF.
``xc7``      ``synth_xilinx -family xc7``: LUT1-6, CARRY4, MUXF7/8, DSP48E1, FF.

Il Verilog emesso da firtool usa costrutti SystemVerilog (assignment pattern
su array impacchettati) che ``read_verilog -sv`` non accetta: si usa il
frontend slang (plugin incluso in OSS CAD Suite) e, se manca, ``read_verilog``.
"""
from __future__ import annotations

import functools
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..util.shell import run, which

FLOWS = ("generic", "liberty", "ecp5", "xc7")


@dataclass
class SynthResult:
    backend: str
    ok: bool
    metrics: dict[str, float] = field(default_factory=dict)
    cells: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    error: str = ""
    skipped: bool = False  # strumento non disponibile: nessun numero, non è un fallimento
    log: str = ""  # percorso del log completo

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SynthResult":
        return cls(**d)


def available() -> bool:
    return which("yosys") is not None


@functools.lru_cache(maxsize=1)
def tool_versions() -> dict[str, str]:
    """Versioni degli strumenti di sintesi (per meta/ppa.json: riproducibilità)."""
    out = {}
    for name, cmd in {"yosys": ["yosys", "-V"], "nextpnr-ecp5": ["nextpnr-ecp5", "--version"],
                      "sta": ["sta", "-version"], "vivado": ["vivado", "-version"]}.items():
        if which(cmd[0]):
            r = run(cmd, timeout=60)
            out[name] = (r.log.strip().splitlines() or ["?"])[0][:120]
    out["slang"] = str(has_slang())
    return out


@functools.lru_cache(maxsize=1)
def has_slang() -> bool:
    if not available():
        return False
    return run(["yosys", "-m", "slang", "-q", "-p", "help read_slang"], timeout=60).ok


def _q(p: Path | str) -> str:
    return '"' + str(p).replace('"', '\\"') + '"'


def frontend(files: list[str], top: str) -> tuple[list[str], list[str]]:
    """(argomenti extra di yosys, comandi di lettura). ``files`` sono percorsi
    relativi senza spazi: read_slang non toglie le virgolette."""
    if has_slang():
        return ["-m", "slang"], [f"read_slang {' '.join(files)} --top {top}"]
    return [], [f"read_verilog -sv {f}" for f in files] + [f"hierarchy -check -top {top}"]


def script(flow: str, top: str, liberty: str | None = None, netlist: str | None = None) -> list[str]:
    """Comandi dopo la lettura; le uscite (stat.json, ltp.txt, netlist) sono
    relative alla cartella di lavoro."""
    stat, ltp = "stat.json", "ltp.txt"
    if flow == "generic":
        body = [f"synth -flatten -top {top}", "abc -g cmos2", "opt_clean",
                f"tee -q -o {stat} stat -json -tech cmos"]
    elif flow == "liberty":
        if not liberty:
            raise ValueError("flusso liberty: serve synth.liberty")
        lib = _q(liberty)
        body = [f"synth -flatten -top {top}", f"dfflibmap -liberty {lib}", f"abc -liberty {lib}",
                "opt_clean", f"tee -q -o {stat} stat -json -liberty {lib}"]
        if netlist:
            body.append(f"write_verilog -noattr {netlist}")  # netlist per OpenSTA
    elif flow == "ecp5":
        body = [f"synth_ecp5 -top {top}" + (f" -json {netlist}" if netlist else ""),
                f"tee -q -o {stat} stat -json"]
    elif flow == "xc7":
        body = [f"synth_xilinx -family xc7 -flatten -top {top}", f"tee -q -o {stat} stat -json"]
    else:
        raise ValueError(f"flusso sconosciuto {flow}")
    return body + [f"tee -q -o {ltp} ltp -noff"]


# --------------------------------------------------------------- parsing
_LTP = re.compile(r"Longest topological path in \S+ \(length=(\d+)\)")


def parse_stat(text: str) -> tuple[dict[str, int], dict[str, float]]:
    """(celle per tipo, totali) da ``stat -json``."""
    d = json.loads(text)
    des = d.get("design") or next(iter(d.get("modules", {}).values()), {})
    cells = {k.lstrip("\\"): int(v) for k, v in (des.get("num_cells_by_type") or {}).items()}
    tot: dict[str, float] = {"cells": int(des.get("num_cells", sum(cells.values())))}
    if "estimated_num_transistors" in des:
        tot["transistors"] = int(str(des["estimated_num_transistors"]).rstrip("+"))
    if "area" in des:
        tot["area"] = float(des["area"])
    return cells, tot


def parse_ltp(text: str) -> int | None:
    m = _LTP.search(text)
    return int(m.group(1)) if m else None


def _sum(cells: dict[str, int], pattern: str) -> int:
    return sum(v for k, v in cells.items() if re.fullmatch(pattern, k))


def fpga_metrics(flow: str, cells: dict[str, int]) -> dict[str, float]:
    if flow == "ecp5":
        return {"luts": _sum(cells, r"LUT4"), "carry": _sum(cells, r"CCU2C"),
                "muxes": _sum(cells, r"PFUMX|L6MUX21"), "dsp": _sum(cells, r"MULT18X18D|ALU54B"),
                "ffs": _sum(cells, r"TRELLIS_FF")}
    if flow == "xc7":
        return {"luts": _sum(cells, r"LUT[1-6]"), "carry": _sum(cells, r"CARRY4"),
                "muxes": _sum(cells, r"MUXF[78]"), "dsp": _sum(cells, r"DSP48E1"),
                "ffs": _sum(cells, r"FD[A-Z]*")}
    return {}


def errors_of(log: str) -> str:
    lines = [ln.strip() for ln in log.splitlines() if "ERROR" in ln]
    if lines:
        return " | ".join(dict.fromkeys(lines))[:600]
    tail = log.strip().splitlines()
    return tail[-1][:300] if tail else "yosys fallito senza messaggi"


# ------------------------------------------------------------------ run
def synth(files: list[Path], top: str, flow: str, work: Path, *, liberty: str | None = None,
          timeout: float = 1800, netlist: str | None = None) -> SynthResult:
    """Sintetizza ``files`` in ``work``; ``netlist`` (nome di file) salva la
    netlist in ``work`` (JSON per ecp5, Verilog per liberty)."""
    if not available():
        return SynthResult(flow, False, error="yosys non trovato nel PATH", skipped=True)
    if flow == "liberty" and not liberty:
        return SynthResult(flow, False, error="synth.liberty non impostato", skipped=True)
    work.mkdir(parents=True, exist_ok=True)
    for f in ("stat.json", "ltp.txt"):
        (work / f).unlink(missing_ok=True)
    src = work / "src"
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir()
    names = []
    for f in map(Path, files):
        if not f.is_file():
            return SynthResult(flow, False, error=f"file non trovato: {f}")
        shutil.copyfile(f, src / f.name)
        names.append(f"src/{f.name}")
    margs, read = frontend(names, top)
    ys = work / "synth.ys"
    ys.write_text("\n".join(read + script(flow, top, liberty, netlist)) + "\n", encoding="utf-8")
    log = work / "yosys.log"
    r = run(["yosys", *margs, "-q", "-l", str(log.resolve()), "-s", str(ys.resolve())],
            cwd=work, timeout=timeout)
    res = SynthResult(flow, False, seconds=round(r.seconds, 2), log=str(log))
    if r.timed_out:
        res.error = f"timeout dopo {timeout:.0f} s"
        return res
    if not r.ok or not (work / "stat.json").is_file():
        full = log.read_text(errors="replace") if log.is_file() else r.log
        res.error = errors_of(full or r.log)
        return res
    cells, tot = parse_stat((work / "stat.json").read_text())
    res.cells, res.metrics = cells, dict(tot)
    depth = parse_ltp((work / "ltp.txt").read_text()) if (work / "ltp.txt").is_file() else None
    if depth is not None:
        res.metrics["depth"] = depth
    res.metrics.update(fpga_metrics(flow, cells))
    res.ok = True
    return res
