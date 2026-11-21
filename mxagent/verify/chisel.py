"""Stadio 1 del verifier: compilazione Scala + elaborazione Chisel + firtool.

Ogni candidato viene messo in un *package* proprio dentro un workspace sbt
(copia di ``chisel/``). Il package è iniettato dal verifier: le righe
``package ...`` scritte dall'LLM vengono sostituite, così i nomi non collidono.

Modalità sbt:

``batch``   ``sbt -batch`` a freddo per ogni job (robusto, ~15-30 s)
``client``  ``sbt --client``: server JVM persistente e compilazione incrementale
            (pochi secondi); il workspace è riusato e va usato da un job alla volta.

Gli errori sono restituiti già strutturati (:class:`SourceError`): file, riga,
colonna, messaggio, snippet del sorgente.
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..util.shell import Result, run

CHISEL_TEMPLATE = Path(__file__).resolve().parent.parent.parent / "chisel"
PKG_RE = re.compile(r"^\s*package\s+[\w.]+\s*;?\s*$", re.M)


@dataclass
class SourceError:
    stage: str  # scalac | elaboration | firtool | sbt
    message: str
    file: str | None = None
    line: int | None = None
    col: int | None = None
    snippet: str = ""

    def render(self) -> str:
        loc = f"{self.file}:{self.line}" + (f":{self.col}" if self.col else "") if self.file else "?"
        out = f"[{self.stage}] {loc}: {self.message}"
        if self.snippet:
            out += "\n" + self.snippet
        return out

    def key(self) -> str:
        """Firma per riconoscere errori ripetuti (anti-loop): senza numeri di riga."""
        return f"{self.stage}:{self.file}:{re.sub(r'[0-9]+', '#', self.message)[:120]}"


@dataclass
class ElabResult:
    ok: bool
    errors: list[SourceError] = field(default_factory=list)
    verilog: list[Path] = field(default_factory=list)
    seconds: float = 0.0
    log: str = ""
    returncode: int | None = None
    timed_out: bool = False

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": [asdict(e) for e in self.errors],
                "verilog": [str(p) for p in self.verilog], "seconds": round(self.seconds, 2),
                "returncode": self.returncode}


# ----------------------------------------------------------------- workspace
def prepare_workspace(work: Path) -> Path:
    """Copia il progetto sbt (build.sbt, project/, Elaborate.scala) se manca."""
    work.mkdir(parents=True, exist_ok=True)
    for rel in ("build.sbt", "project/build.properties", "src/main/scala/mxagent/Elaborate.scala"):
        dst = work / rel
        src = CHISEL_TEMPLATE / rel
        if not dst.is_file() or dst.read_bytes() != src.read_bytes():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    return work


def inject_package(code: str, pkg: str) -> str:
    code = PKG_RE.sub("", code, count=0).lstrip("\n")
    return f"package {pkg}\n\n{code}"


def write_sources(work: Path, pkg: str, sources: dict[str, str], clean: bool = True) -> Path:
    """Scrive i sorgenti del candidato in ``src/main/scala/<pkg>/``.

    Con ``clean`` rimuove gli altri candidati: un candidato rotto non deve
    impedire la compilazione del successivo (stesso progetto sbt).
    """
    gen_root = work / "src" / "main" / "scala"
    if clean:
        for d in gen_root.iterdir() if gen_root.is_dir() else []:
            if d.is_dir() and d.name != "mxagent":
                shutil.rmtree(d)
    d = gen_root / pkg
    d.mkdir(parents=True, exist_ok=True)
    for name, code in sources.items():
        if not name.endswith(".scala"):
            name += ".scala"
        (d / Path(name).name).write_text(inject_package(code, pkg), encoding="utf-8")
    return d


# ------------------------------------------------------------------- parser
_SCALAC = re.compile(
    r"^\[error\]\s+(?:--\s*\[E\d+\].*?)?(?P<file>(?:[A-Za-z]:)?[^\s:]*?\.scala):(?P<line>\d+):"
    r"(?P<col>\d+):\s*(?P<msg>.*)$")
_FIRTOOL = re.compile(
    r"(?P<file>[\w./\\-]+\.scala):(?P<line>\d+):(?P<col>\d+):\s*error:\s*(?P<msg>.*)$")
_FRAME = re.compile(r"\bat\s+(?P<where>[\w.$<>]+)\((?P<file>\w+\.scala):(?P<line>\d+)\)")
_CHISEL_LOC = re.compile(r"@\[(?:[\w./\\-]*/)?(?P<file>\w+\.scala)[ :](?P<line>\d+):(?P<col>\d+)\]")
_EXC = re.compile(r"(?P<cls>(?:[\w$]+\.)+[\w$]*(?:Exception|Error))\b(?::\s*(?P<msg>.*))?")
# Chisel 6 raccoglie gli errori di elaborazione in una lista:
#   [error] src/main/scala/pkg/Top.scala 33:68: Invalid bit range [hi=0, lo=8]
_CHISEL_LIST = re.compile(
    r"^\[error\]\s+(?P<file>(?:[A-Za-z]:)?[^\s:]*?\.scala)\s+(?P<line>\d+):(?P<col>\d+):\s*(?P<msg>.*)$")


def _strip(line: str) -> str:
    return re.sub(r"^\[(?:error|info|warn)\]\s?", "", line)


def _snippet(src_dir: Path | None, file: str | None, line: int | None, ctx: int = 2) -> str:
    if not (src_dir and file and line):
        return ""
    p = src_dir / Path(file).name
    if not p.is_file():
        return ""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    lo, hi = max(1, line - ctx), min(len(lines), line + ctx)
    return "\n".join(f"{'>' if i == line else ' '}{i:4d} | {lines[i - 1]}" for i in range(lo, hi + 1))


def _line_offset(src_dir: Path | None, file: str | None) -> int:
    """Righe aggiunte dall'iniezione del package (per riportare le righe dell'LLM)."""
    return 2 if src_dir and file and (src_dir / Path(file).name).is_file() else 0


def parse_sbt_log(log: str, src_dir: Path | None = None, pkg: str | None = None) -> list[SourceError]:
    """Estrae errori strutturati dal log di sbt (scalac, elaborazione Chisel, firtool)."""
    lines = log.splitlines()
    errs: list[SourceError] = []
    seen: set[tuple] = set()

    def add(e: SourceError) -> None:
        k = (e.stage, e.file, e.line, e.message)
        if k not in seen:
            seen.add(k)
            if e.file:
                e.file = Path(e.file).name
            errs.append(e)

    # --- scalac
    for i, ln in enumerate(lines):
        m = _SCALAC.match(ln.strip())
        if not m:
            continue
        msg = m["msg"].strip()
        if msg.startswith("error:"):  # firtool riporta i locatori Scala con lo stesso formato
            add(SourceError("firtool", msg[len("error:"):].strip(), m["file"], int(m["line"]),
                            int(m["col"])))
            continue
        # le righe seguenti (found/required...) completano il messaggio; la riga di
        # codice citata da scalac è indentata di almeno 3 spazi e si scarta
        extra = []
        for nxt in lines[i + 1:i + 6]:
            t = _strip(nxt.strip())
            if (_SCALAC.match(nxt.strip()) or not t.strip() or t.strip().startswith("^")
                    or re.match(r"\s*\w+ errors? found", t)):
                break
            if not nxt.startswith("[error]   "):
                extra.append(t.strip())
        if extra:
            msg += " " + " ".join(extra)
        add(SourceError("scalac", " ".join(msg.split()), m["file"], int(m["line"]), int(m["col"])))
    if errs:
        return _finish(errs, src_dir)

    # --- firtool (errori con locatore Scala)
    for ln in lines:
        m = _FIRTOOL.search(_strip(ln))
        if m:
            add(SourceError("firtool", m["msg"].strip(), m["file"], int(m["line"]), int(m["col"])))
    if errs:
        return _finish(errs, src_dir)

    # --- lista di errori di elaborazione (Chisel 6)
    for ln in lines:
        m = _CHISEL_LIST.match(ln.strip())
        if m:
            add(SourceError("elaboration", " ".join(m["msg"].split()), m["file"], int(m["line"]),
                            int(m["col"])))
    if errs:
        return _finish(errs, src_dir)

    # --- eccezioni di elaborazione Chisel
    for i, ln in enumerate(lines):
        t = _strip(ln).strip()
        m = _EXC.search(t)
        if not m or t.startswith("at "):
            continue
        if "sbt.internal" in m["cls"] or m["cls"].startswith("java.lang.RuntimeException: Nonzero"):
            continue
        msg = f"{re.split(r'[.$]', m['cls'])[-1]}: {m['msg'] or ''}".strip()
        loc = _CHISEL_LOC.search(t)
        file = line = col = None
        if loc:
            file, line, col = loc["file"], int(loc["line"]), int(loc["col"])
        else:  # primo frame dello stack nel package del candidato
            for nxt in lines[i + 1:i + 60]:
                f = _FRAME.search(nxt)
                if f and (pkg is None or f["where"].startswith(pkg + ".")):
                    file, line = f["file"], int(f["line"])
                    break
        add(SourceError("elaboration", msg, file, line, col))
        break
    if not errs and re.search(r"\[error\]", log):
        tail = [_strip(x) for x in lines if x.startswith("[error]")][-5:]
        add(SourceError("sbt", " | ".join(t.strip() for t in tail if t.strip())))
    return _finish(errs, src_dir)


def _finish(errs: list[SourceError], src_dir: Path | None) -> list[SourceError]:
    for e in errs:
        if e.line and e.file:
            off = _line_offset(src_dir, e.file)
            e.snippet = _snippet(src_dir, e.file, e.line)
            e.line -= off  # righe come le ha scritte l'LLM (senza il package iniettato)
            e.snippet = "\n".join(
                re.sub(r"^([> ])\s*(\d+)", lambda m: f"{m[1]}{int(m[2]) - off:4d}", s)
                for s in e.snippet.splitlines())
    return errs


# --------------------------------------------------------------- esecuzione
#: CPU che vede ogni JVM di sbt (``-XX:ActiveProcessorCount``; ``sim.jvm_cpus``). La JVM
#: dimensiona su questo numero i thread del GC, dei compilatori JIT, del ForkJoinPool e
#: dei pool di sbt/zinc/Scala: con tutti i core di una macchina grande (20 sul DGX Spark)
#: un server sbt per run tiene un centinaio di thread. Il Verilog prodotto non cambia.
DEFAULT_JVM_CPUS = 4


def sbt_env(jvm_cpus: int | None = DEFAULT_JVM_CPUS) -> dict[str, str]:
    """Ambiente dei comandi sbt: ``SBT_OPTS`` (default ``-Xss8m``, Chisel ricorre in
    profondità) più ``-XX:ActiveProcessorCount`` se non già presente. Vale per il
    server di ``--client`` (avviato dal primo comando del workspace) e per ``-batch``."""
    opts = os.environ.get("SBT_OPTS", "-Xss8m")
    if jvm_cpus and "ActiveProcessorCount" not in opts:
        opts = f"{opts} -XX:ActiveProcessorCount={int(jvm_cpus)}".strip()
    return {"SBT_OPTS": opts, "NO_COLOR": "1"}


def active_file(work: Path) -> Path:
    """Scritto dal server sbt di ``work`` all'avvio, cancellato allo shutdown."""
    return Path(work) / "project" / "target" / "active.json"


def shutdown_server(work: Path, jvm_cpus: int | None = DEFAULT_JVM_CPUS, alive: bool | None = None,
                    timeout: float = 60) -> Result | None:
    """``sbt --client shutdown`` solo se un server c'è: senza server il client ne
    avvierebbe uno nuovo solo per spegnerlo. ``alive`` (JVM vive nel workspace, se
    note) ha la precedenza su ``active.json``, che resta se il server muore di colpo:
    in quel caso il file viene tolto, così il comando successivo parte pulito."""
    act = active_file(work)
    if alive is None:
        alive = act.is_file()
    if not alive:
        act.unlink(missing_ok=True)
        return None
    return run(sbt_cmd("client", "shutdown"), cwd=work, timeout=timeout, env=sbt_env(jvm_cpus))


def sbt_cmd(mode: str, command: str) -> list[str]:
    if mode == "client":
        return ["sbt", "--client", command]
    return ["sbt", "-batch", "-Dsbt.supershell=false", command]


def elaborate(work: Path, pkg: str, top: str, out_dir: Path, mode: str = "batch",
              timeout: float = 900, jvm_cpus: int | None = DEFAULT_JVM_CPUS) -> ElabResult:
    """Compila ed elabora ``<pkg>.<top>``; emette SystemVerilog in ``out_dir``."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    target = os.path.relpath(out_dir, work).replace("\\", "/") if _same_drive(out_dir, work) else str(out_dir)
    cmd = f"runMain mxagent.Elaborate {pkg}.{top} {target}"
    # client: il server sbt (JVM persistente) deve sopravvivere al comando; lo chiude
    # Verifier.close. batch: a fine comando nessun processo del gruppo resta vivo.
    r: Result = run(sbt_cmd(mode, cmd), cwd=work, timeout=timeout,
                    env=sbt_env(jvm_cpus), keep_group=mode == "client")
    log = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", r.log)
    src_dir = work / "src" / "main" / "scala" / pkg
    verilog = sorted([*out_dir.rglob("*.sv"), *out_dir.rglob("*.v")])
    if r.timed_out:
        return ElabResult(False, [SourceError("sbt", f"timeout dopo {timeout:.0f}s")], [], r.seconds, log,
                          r.returncode, timed_out=True)
    if not r.ok or not verilog:
        errs = parse_sbt_log(log, src_dir, pkg) or [SourceError("sbt", "elaborazione fallita senza "
                                                               "errori riconoscibili")]
        return ElabResult(False, errs, [], r.seconds, log, r.returncode)
    return ElabResult(True, [], verilog, r.seconds, log, r.returncode)


def _same_drive(a: Path, b: Path) -> bool:
    return os.path.splitdrive(a.resolve())[0] == os.path.splitdrive(b.resolve())[0]
