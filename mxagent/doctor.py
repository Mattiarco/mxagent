"""``mxagent doctor``: verifica dell'ambiente con istruzioni di installazione senza sudo.

Ogni controllo produce un :class:`Check` (obbligatorio o facoltativo). I
facoltativi mancanti disattivano solo la fase corrispondente (es. Vivado).
Con ``--warmup`` elabora il modulo Smoke: scarica Chisel/firtool nelle cache e
verifica che il Verilog sia prodotto proprio da firtool |FIRTOOL_VERSION|.
"""
from __future__ import annotations

import importlib
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import versions as V
from .util.shell import run, which

REPO = Path(__file__).resolve().parent.parent
SETUP = "bash scripts/setup_server.sh"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    hint: str = ""
    required: bool = True


def _ver(text: str, pattern: str) -> tuple[int, ...] | None:
    m = re.search(pattern, text)
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def is_arm64(machine: str | None = None) -> bool:
    return (machine or platform.machine()).lower() in ("aarch64", "arm64")


def _oss_hint() -> str:
    arch = "arm64" if is_arm64() else "x64"
    return (f"OSS CAD Suite (yosys, verilator, nextpnr, sta): {SETUP}  oppure scarica "
            f"oss-cad-suite-linux-{arch}-<data>.tgz da github.com/YosysHQ/oss-cad-suite-build")


# ------------------------------------------------------------------ controlli
def check_python() -> list[Check]:
    ok = sys.version_info >= (3, 10)
    out = [Check("python", ok, platform.python_version(), "serve Python >= 3.10")]
    for mod, req, hint in [("numpy", True, "pip install -e ."), ("yaml", True, "pip install -e ."),
                           ("pydantic", True, "pip install -e ."),
                           ("cocotb", True, "pip install -e '.[sim]'"),
                           ("matplotlib", False, "pip install -e '.[eval]' (solo per i grafici)")]:
        try:
            m = importlib.import_module(mod)
            out.append(Check(f"py:{mod}", True, getattr(m, "__version__", "ok"), required=req))
        except ImportError:
            out.append(Check(f"py:{mod}", False, "non installato", hint, required=req))
    return out


def check_java() -> Check:
    if not which("java"):
        return Check("java", False, "non trovato", f"JDK {V.JDK_RECOMMENDED}: {SETUP}")
    r = run(["java", "-version"], timeout=60)
    m = re.search(r'version "(\d+)(?:\.(\d+))?', r.log)
    major = int(m.group(1)) if m else 0
    if major == 1 and m and m.group(2):
        major = int(m.group(2))
    lo, hi = V.JDK_RANGE
    return Check("java", lo <= major <= hi,
                 f"Java {major}" + ("" if major == V.JDK_RECOMMENDED else f" (consigliato {V.JDK_RECOMMENDED})"),
                 f"serve JDK {lo}-{hi}: {SETUP}")


def check_sbt() -> Check:
    found = which("sbt")
    return Check("sbt", bool(found), f"{found} (il progetto fissa sbt {V.SBT_VERSION} in "
                 "chisel/project/build.properties)" if found else "non trovato", f"{SETUP}")


def check_firtool(machine: str | None = None, env: dict | None = None) -> Check | None:
    """firtool esplicito (CHISEL_FIRTOOL_PATH) o assente su aarch64.
    firtool-resolver NON controlla la versione del binario indicato: lo si fa qui."""
    env = os.environ if env is None else env
    path = env.get("CHISEL_FIRTOOL_PATH")
    want = f"CIRCT firtool-{V.FIRTOOL_VERSION}"
    if path:
        exe = Path(path) / ("firtool.exe" if os.name == "nt" else "firtool")
        if not exe.is_file():
            return Check("firtool", False, f"{exe} non esiste",
                         "CHISEL_FIRTOOL_PATH deve essere la CARTELLA che contiene firtool")
        r = run([str(exe), "--version"], timeout=60)
        got = next((ln.strip() for ln in r.log.splitlines() if ln.strip().startswith("CIRCT ")),
                   "versione non leggibile")
        return Check("firtool", got == want, f"{got} da CHISEL_FIRTOOL_PATH",
                     f"serve {want}; su aarch64 lo compila {SETUP}")
    if is_arm64(machine):
        return Check("firtool", False, f"Chisel {V.CHISEL_VERSION} non distribuisce firtool per "
                     "aarch64 e CHISEL_FIRTOOL_PATH non è impostata",
                     f"{SETUP} lo compila dai sorgenti CIRCT {V.FIRTOOL_VERSION} (una volta sola)")
    return None  # x86_64/Windows: lo scarica Chisel, verificato da --warmup


def check_verilator() -> list[Check]:
    if not which("verilator"):
        return [Check("verilator", False, "non trovato", _oss_hint())]
    r = run(["verilator", "--version"], timeout=60)
    v = _ver(r.log, r"Verilator (\d+\.\d+)")
    out = [Check("verilator", bool(v) and v >= V.VERILATOR_MIN, f"{r.log.strip()[:60]}",
                 f"serve Verilator >= {'.'.join(map(str, V.VERILATOR_MIN))}: {_oss_hint()}")]
    try:
        import cocotb  # noqa: PLC0415

        cv = _ver(cocotb.__version__, r"(\d+\.\d+)")
        need = V.COCOTB_VERILATOR_MIN.get(cv[0]) if cv else None
        if need and v:
            out.append(Check("cocotb+verilator", v >= need,
                             f"cocotb {cocotb.__version__} richiede Verilator >= "
                             f"{need[0]}.{need[1]:03d}, trovato {v[0]}.{v[1]:03d}",
                             "aggiorna OSS CAD Suite oppure usa cocotb 1.9.x"))
    except ImportError:
        pass
    return out


def check_tool(name: str, cmd: list[str], hint: str, required: bool = True,
               pattern: str = r"(\d+\.\d+(?:\.\d+)?)") -> Check:
    if not which(cmd[0]):
        return Check(name, False, "non trovato", hint, required)
    r = run(cmd, timeout=60)
    m = re.search(pattern, r.log)
    return Check(name, True, (m.group(1) if m else r.log.strip()[:50]) + f"  ({which(cmd[0])})",
                 hint, required)


def check_slang(hint: str) -> Check:
    """Il Verilog di firtool richiede il frontend slang per essere letto da Yosys."""
    from .synth.yosys import has_slang

    if not which("yosys"):
        return Check("yosys-slang", False, "yosys assente", hint, required=False)
    ok = has_slang()
    return Check("yosys-slang", ok, "plugin caricato" if ok else
                 "assente: read_verilog non legge alcuni costrutti SV di firtool", hint, required=False)


def check_gpu() -> Check:
    if not which("nvidia-smi"):
        return Check("GPU NVIDIA", False, "nvidia-smi assente: Ollama userà la CPU (lento)",
                     "serve il driver NVIDIA (lato amministratore)", required=False)
    r = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader"], timeout=30)
    gpus = [g.strip() for g in r.stdout.splitlines() if g.strip()]
    return Check("GPU NVIDIA", r.ok and bool(gpus), "; ".join(gpus) or r.log[-200:], required=False)


def check_ollama(host: str | None, models: list[str]) -> list[Check]:
    from .llm.ollama import OllamaClient, normalize_model

    cli = OllamaClient(host, timeout=10)
    try:
        installed = cli.models()
        version = cli.version()
    except Exception as e:  # noqa: BLE001
        return [Check("ollama", False, f"non raggiungibile su {cli.host}: {str(e)[:100]}",
                      "avvia `ollama serve` (anche in tmux) o imposta OLLAMA_HOST / llm.host")]
    out = [Check("ollama", True, f"{cli.host}, v{version}, {len(installed)} modelli")]
    have = {normalize_model(n) for n in installed}
    for m in dict.fromkeys(models):
        ok = normalize_model(m) in have
        detail = "non installato"
        if ok:
            try:
                i = cli.show(m)
                detail = (f"{i['parameter_size']} {i['quantization']}, "
                          f"context nativo {i['context_length']}")
            except Exception as e:  # noqa: BLE001
                detail = f"installato (show fallito: {e})"
        out.append(Check(f"modello {m}", ok, detail, f"ollama pull {m}"))
    return out


def warmup(timeout: float = 1800) -> Check:
    """Elabora Smoke: popola le cache e verifica la versione di firtool nel Verilog."""
    chisel = REPO / "chisel"
    if not which("sbt"):
        return Check("warm-up Chisel", False, "sbt assente", SETUP)
    cmd = "runMain mxagent.Elaborate mxagent.Smoke build/Smoke"
    r = run(["sbt", "-batch", cmd], cwd=chisel, timeout=timeout)
    sv = chisel / "build" / "Smoke" / "Smoke.sv"
    if not r.ok or not sv.is_file():
        lines = r.log.strip().splitlines()
        # la prima riga d'errore significativa, non il riepilogo finale di sbt
        errs = [ln for ln in lines if ("[error]" in ln or "Exception" in ln or "error:" in ln)
                and "Total time" not in ln and "(Compile / " not in ln]
        log = chisel / "build" / "warmup.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(r.log, encoding="utf-8")
        return Check("warm-up Chisel", False, f"elaborazione fallita in {r.seconds:.0f}s: "
                     f"{(errs or lines or ['?'])[0].strip()}",
                     f"log completo in {log}; per rilanciare: cd chisel && sbt -batch '{cmd}'")
    head = sv.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    want = f"firtool-{V.FIRTOOL_VERSION}"
    return Check("warm-up Chisel", want in head, f"{head.strip()} ({r.seconds:.0f}s)",
                 f"il Verilog deve venire da {want}")


def run_checks(host: str | None, models: list[str], do_warmup: bool = False,
               skip_llm: bool = False) -> list[Check]:
    checks = check_python()
    checks += [check_java(), check_sbt()]
    fir = check_firtool()
    if fir:
        checks.append(fir)
    checks += check_verilator()
    oss = _oss_hint()
    checks += [
        check_tool("g++", ["g++", "--version"], "compilatore C++ per Verilator"),
        check_tool("make", ["make", "--version"], "GNU make per Verilator/cocotb"),
        check_tool("yosys", ["yosys", "-V"], oss, required=False, pattern=r"Yosys ([\d.+]+)"),
        check_tool("iverilog", ["iverilog", "-V"], oss, required=False,
                   pattern=r"version (\d+\.\d+)"),
        check_tool("nextpnr-ecp5", ["nextpnr-ecp5", "--version"], oss, required=False,
                   pattern=r"Version ([^\s)]+)"),
        check_slang(oss),
        check_tool("OpenSTA", ["sta", "-version"],
                   "facoltativo, non incluso in OSS CAD Suite: compilare OpenSTA (parallaxsw/OpenSTA) "
                   "e impostare synth.liberty", required=False, pattern=r"(\d+\.\d+\.\d+)"),
        check_tool("vivado", ["vivado", "-version"], "facoltativo: source settings64.sh di Vivado",
                   required=False, pattern=r"v(\d{4}\.\d)"),
        check_gpu(),
    ]
    if not skip_llm:
        checks += check_ollama(host, models)
    if do_warmup:
        checks.append(warmup())
    return checks


def synth_backends(backends: list[str], liberty: str | None = None) -> list[Check]:
    """Disponibilità di ogni backend di sintesi (``mxagent doctor --synth``).

    Un backend non disponibile non ferma le run: compare nel ppa.json come ``skipped``.
    I backend elencati in ``synth.backends`` sono marcati ``[config]``.
    """
    from .config import SYNTH_BACKENDS

    yos = which("yosys")
    lib = f"synth.liberty={liberty}" if liberty else "synth.liberty non impostato"
    need: dict[str, tuple[bool, str, str]] = {
        "generic": (bool(yos), "yosys (abc -g cmos2)", _oss_hint()),
        "liberty": (bool(yos and liberty), f"yosys + .lib ({lib})",
                    "imposta synth.liberty a un file .lib (es. Nangate45, sky130)"),
        "ecp5": (bool(yos), "yosys synth_ecp5", _oss_hint()),
        "xc7": (bool(yos), "yosys synth_xilinx", _oss_hint()),
        "nextpnr": (bool(yos and which("nextpnr-ecp5")), "yosys + nextpnr-ecp5", _oss_hint()),
        "opensta": (bool(which("sta") and liberty), f"sta + .lib ({lib})",
                    "compila OpenSTA (parallaxsw/OpenSTA) e imposta synth.liberty"),
        "vivado": (bool(which("vivado")), "vivado nel PATH", "source <Vivado>/settings64.sh"),
    }
    out = []
    for b in SYNTH_BACKENDS:
        ok, what, hint = need[b]
        tag = " [config]" if b in backends else ""
        out.append(Check(f"synth:{b}{tag}", ok, ("disponibile: " if ok else "mancante: ") + what,
                         hint, required=False))
    return out


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        mark = "OK " if c.ok else ("ERR" if c.required else " - ")
        lines.append(f"  [{mark}] {c.name:<18} {c.detail}")
        if not c.ok and c.hint:
            lines.append(f"        -> {c.hint}")
    bad = [c.name for c in checks if c.required and not c.ok]
    lines.append("")
    lines.append("ambiente pronto" if not bad else f"mancano componenti obbligatori: {', '.join(bad)}")
    return "\n".join(lines)
