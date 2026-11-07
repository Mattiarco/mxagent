"""Backend opzionali: OpenSTA (con una libreria .lib) e Vivado in batch.

ATTENZIONE: nessuno dei due strumenti era installato sulla macchina di
sviluppo, quindi questi flussi NON sono stati validati con uno strumento vero;
lo sono solo i parser, su estratti dei formati documentati. Si attivano solo se
``sta`` / ``vivado`` sono nel PATH e vengono segnalati come saltati altrimenti.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..bench.task import Task
from ..util.shell import run, which
from . import yosys
from .wrapper import WRAP_TOP, wrapper


# ---------------------------------------------------------------- OpenSTA
def parse_sta(text: str) -> float | None:
    """Ritardo massimo (unità della libreria, di solito ns) da ``report_checks``."""
    vals = [float(m) for m in re.findall(r"(-?\d+(?:\.\d+)?)\s+data arrival time", text)]
    return max(vals) if vals else None


def opensta(files: list[Path], top: str, work: Path, liberty: str | None,
            timeout: float = 1800) -> yosys.SynthResult:
    if which("sta") is None:
        return yosys.SynthResult("opensta", False, error="sta (OpenSTA) non trovato nel PATH",
                                 skipped=True)
    if not liberty:
        return yosys.SynthResult("opensta", False, error="synth.liberty non impostato", skipped=True)
    work.mkdir(parents=True, exist_ok=True)
    s = yosys.synth(files, top, "liberty", work / "synth", liberty=liberty, timeout=timeout,
                    netlist="netlist.v")
    net = (work / "synth" / "netlist.v").resolve()
    res = yosys.SynthResult("opensta", False, seconds=s.seconds, log=s.log)
    if not s.ok:
        res.error = f"sintesi liberty: {s.error}"
        return res
    tcl = work / "sta.tcl"
    tcl.write_text("\n".join([
        f"read_liberty {liberty}", f"read_verilog {net}", f"link_design {top}",
        "create_clock -name vclk -period 1000",
        "set_input_delay 0 -clock vclk [all_inputs]",
        "set_output_delay 0 -clock vclk [all_outputs]",
        "report_checks -path_delay max -digits 4", "exit", ""]), encoding="utf-8")
    r = run(["sta", "-no_splash", "-exit", str(tcl.resolve())], cwd=work, timeout=timeout)
    (work / "sta.log").write_text(r.log, encoding="utf-8")
    res.seconds = round(res.seconds + r.seconds, 2)
    d = parse_sta(r.log)
    if d is None:
        res.error = "nessun 'data arrival time' nell'output di OpenSTA"
        return res
    res.metrics = {"delay": d, "area": s.metrics.get("area", 0.0)}
    res.ok = True
    return res


# ----------------------------------------------------------------- Vivado
def parse_vivado_util(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    pats = {"luts": r"(?:Slice|CLB) LUTs\*?", "ffs": r"(?:Slice|CLB) Registers",
            "carry": r"CARRY[48]", "dsp": r"DSPs", "muxes": r"F7 Muxes"}
    for key, pat in pats.items():
        m = re.search(rf"^\|\s*{pat}\s*\|\s*(\d+)", text, re.M)
        if m:
            out[key] = int(m.group(1))
    return out


def parse_vivado_wns(text: str) -> float | None:
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if "WNS(ns)" in ln:
            for nxt in lines[i + 1:i + 5]:
                m = re.match(r"\s*(-?\d+\.\d+)", nxt)
                if m:
                    return float(m.group(1))
    return None


def vivado(files: list[Path], task: Task, work: Path, *, part: str, clock_ns: float,
           timeout: float = 3600) -> yosys.SynthResult:
    if which("vivado") is None:
        return yosys.SynthResult("vivado", False, error="vivado non trovato nel PATH", skipped=True)
    work.mkdir(parents=True, exist_ok=True)
    wrap = work / "wrap.sv"
    wrap.write_text(wrapper(task), encoding="utf-8")
    srcs = " ".join(str(Path(f).resolve()) for f in [*files, wrap])
    tcl = work / "run.tcl"
    tcl.write_text("\n".join([
        f"read_verilog -sv {{{srcs}}}",
        f"synth_design -top {WRAP_TOP} -part {part} -mode out_of_context",
        f"create_clock -period {clock_ns} -name clk [get_ports clk]",
        "opt_design", "place_design", "route_design",
        "report_utilization -file util.rpt", "report_timing_summary -file timing.rpt", ""]),
        encoding="utf-8")
    r = run(["vivado", "-mode", "batch", "-nojournal", "-source", str(tcl.resolve())], cwd=work,
            timeout=timeout)
    (work / "vivado.log").write_text(r.log, encoding="utf-8")
    res = yosys.SynthResult("vivado", False, seconds=round(r.seconds, 2), log=str(work / "vivado.log"))
    if not (work / "timing.rpt").is_file():
        res.error = " | ".join(ln for ln in r.log.splitlines() if "ERROR" in ln)[:500] or "vivado fallito"
        return res
    wns = parse_vivado_wns((work / "timing.rpt").read_text(errors="replace"))
    util = parse_vivado_util((work / "util.rpt").read_text(errors="replace")) \
        if (work / "util.rpt").is_file() else {}
    if wns is None:
        res.error = "WNS non trovato in timing.rpt"
        return res
    crit = clock_ns - wns
    # le risorse includono i registri del wrapper (shift register e uscite)
    res.metrics = {**util, "wns_ns": wns, "crit_ns": round(crit, 3),
                   "fmax_mhz": round(1000.0 / crit, 2) if crit > 0 else 0.0}
    res.ok = True
    return res
