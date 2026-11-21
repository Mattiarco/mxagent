"""Processi e limiti di risorse (Linux, da ``/proc`` e dal cgroup; nessuna dipendenza).

Serve a tre cose: terminare i processi rimasti (JVM di sbt, firtool, g++) a fine
verifica e a fine run, controllare le risorse prima di una campagna
(``mxagent eval``, ``mxagent doctor --resources``) e diagnosticare gli errori
EAGAIN / "unable to create native thread". Su sistemi senza ``/proc`` le funzioni
restituiscono valori vuoti.
"""
from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

PROC = Path("/proc")


@dataclass
class Proc:
    pid: int
    ppid: int
    uid: int
    threads: int
    name: str
    cmdline: str
    cwd: str = ""

    @property
    def is_java(self) -> bool:
        return self.name == "java" or "/java " in f"{self.cmdline} " or self.cmdline.startswith("java ")

    @property
    def is_sbt(self) -> bool:
        c = self.cmdline
        return self.is_java and ("sbt-launch" in c or "xsbt.boot" in c or "sbt.ForkMain" in c
                                 or "-Dsbt." in c or "sbt.server" in c)

    @property
    def role(self) -> str:
        """server (di ``--client``), client (JVM thin client), batch, forkmain o java."""
        c = self.cmdline
        if not self.is_java:
            return self.name
        if "sbt.ForkMain" in c:
            return "forkmain"
        if not self.is_sbt:
            return "java"
        if "--detach-stdio" in c or "--server" in c:
            return "server"
        if "--client" in c or "sbt.client" in c or "NetworkClient" in c:
            return "client"
        return "batch"

    @property
    def active_cpus(self) -> int | None:
        """``-XX:ActiveProcessorCount`` della JVM (None se assente)."""
        import re
        m = re.search(r"-XX:ActiveProcessorCount=(\d+)", self.cmdline)
        return int(m[1]) if m else None


def _status(pid: int) -> dict[str, str]:
    out = {}
    try:
        for ln in (PROC / str(pid) / "status").read_text().splitlines():
            k, _, v = ln.partition(":")
            out[k] = v.strip()
    except OSError:
        pass
    return out


def procs(uid: int | None = None) -> list[Proc]:
    """Processi visibili (di ``uid`` se dato)."""
    if not PROC.is_dir():
        return []
    out = []
    for d in PROC.iterdir():
        if not d.name.isdigit():
            continue
        st = _status(int(d.name))
        if not st:
            continue
        try:
            p_uid = int(st.get("Uid", "-1").split()[0])
        except ValueError:
            continue
        if uid is not None and p_uid != uid:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmd = ""
        try:
            cwd = os.readlink(d / "cwd")
        except OSError:
            cwd = ""
        out.append(Proc(int(d.name), int(st.get("PPid", "0") or 0), p_uid,
                        int(st.get("Threads", "1") or 1), st.get("Name", ""), cmd, cwd))
    return out


def under(path: Path, cwd: str) -> bool:
    try:
        base = str(Path(path).resolve())
    except OSError:
        return False
    return bool(cwd) and (cwd == base or cwd.startswith(base.rstrip("/") + "/"))


def java_under(path: Path) -> list[Proc]:
    """JVM con directory di lavoro dentro ``path`` (server sbt di un workspace)."""
    me = os.getuid() if hasattr(os, "getuid") else None
    return [p for p in procs(me) if p.is_java and under(path, p.cwd)]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    st = _status(pid)
    return st.get("State", "").split()[:1] != ["Z"]  # zombie = già terminato


def kill_pids(pids: list[int], grace: float = 10) -> list[int]:
    """SIGTERM, ``grace`` secondi, poi SIGKILL. Restituisce i pid terminati."""
    pids = [p for p in pids if p != os.getpid() and alive(p)]
    if not pids or os.name == "nt":
        return []
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    t0 = time.monotonic()
    while time.monotonic() - t0 < grace and any(alive(p) for p in pids):
        time.sleep(0.2)
    for p in pids:
        if alive(p):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
    return pids


def kill_group(pgid: int, grace: float = 10, proc=None) -> None:
    """SIGTERM al process group, ``grace`` secondi, poi SIGKILL (POSIX). ``proc`` (il
    ``Popen`` capogruppo) viene raccolto durante l'attesa: uno zombie conta ancora come
    membro del gruppo e farebbe aspettare tutto ``grace``."""
    if os.name == "nt":
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    t0 = time.monotonic()
    while time.monotonic() - t0 < grace:
        if proc is not None:
            proc.poll()
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def group_alive(pgid: int) -> bool:
    if os.name == "nt":
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ------------------------------------------------------------------ limiti
def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def cgroup_pids() -> list[dict]:
    """``pids.max`` / ``pids.current`` del cgroup del processo e dei suoi antenati (v2),
    o del controller ``pids`` (v1). Il limite effettivo è il più basso."""
    out = []
    cg = _read(PROC / "self" / "cgroup") or ""
    for ln in cg.splitlines():
        parts = ln.split(":", 2)
        if len(parts) != 3:
            continue
        ctrl, rel = parts[1], parts[2]
        roots = [Path("/sys/fs/cgroup")] if ctrl == "" else (
            [Path("/sys/fs/cgroup/pids")] if "pids" in ctrl.split(",") else [])
        for root in roots:
            d = root / rel.lstrip("/")
            while True:
                mx = _read(d / "pids.max")
                if mx is not None:
                    out.append({"path": str(d), "max": mx, "current": _read(d / "pids.current")})
                if d == root or d.parent == d:
                    break
                d = d.parent
    return out


def meminfo() -> dict[str, int]:
    out = {}
    for ln in (_read(PROC / "meminfo") or "").splitlines():
        k, _, v = ln.partition(":")
        try:
            out[k] = int(v.split()[0]) * 1024
        except (ValueError, IndexError):
            pass
    return out


@dataclass
class Snapshot:
    nproc_soft: int | None
    nproc_hard: int | None
    nofile_soft: int | None
    nofile_hard: int | None
    user_procs: int
    user_threads: int
    threads_max: int | None
    cgroups: list[dict]
    java: list[Proc]
    mem_total: int | None
    mem_available: int | None
    ncpu: int
    orphans: list[Proc] = field(default_factory=list)

    @property
    def pids_limit(self) -> int | None:
        """Limite effettivo sui task (processi + thread) dell'utente: il minimo fra
        ulimit -u e i pids.max numerici del cgroup."""
        vals = [self.nproc_soft] if self.nproc_soft not in (None, -1) else []
        vals += [int(c["max"]) for c in self.cgroups if str(c["max"]).isdigit()]
        return min(vals) if vals else None

    @property
    def pids_used(self) -> int | None:
        cur = [int(c["current"]) for c in self.cgroups
               if str(c.get("current") or "").isdigit() and str(c["max"]).isdigit()]
        return max(cur) if cur else self.user_threads


def _rlimit(name: str) -> tuple[int | None, int | None]:
    try:
        import resource
        s, h = resource.getrlimit(getattr(resource, name))
        inf = resource.RLIM_INFINITY
        return (-1 if s == inf else s), (-1 if h == inf else h)
    except (ImportError, AttributeError, ValueError, OSError):
        return None, None


def _arg(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def live_roots(ps: list[Proc]) -> list[Path]:
    """Cartelle coperte da processi ``mxagent`` vivi: la run di ogni ``mxagent run``
    (``--runs-dir``/``--run-id``) e ``runs/verify`` di verify/synth/wrap."""
    roots = []
    for p in ps:
        argv = p.cmdline.split()
        if "mxagent" not in p.cmdline or p.pid == os.getpid():
            continue
        base = Path(p.cwd) if p.cwd else Path.cwd()
        if "run" in argv:
            rd, rid = _arg(argv, "--runs-dir"), _arg(argv, "--run-id")
            roots.append(base / rd / rid if rd and rid else base / (rd or "runs"))
        if any(c in argv for c in ("verify", "synth", "wrap")):
            roots.append(Path(os.environ.get("MXAGENT_WORK") or base / "runs" / "verify"))
            if _arg(argv, "--run"):
                roots.append(base / _arg(argv, "--run"))
    return roots


def orphan_jvms(ps: list[Proc]) -> list[Proc]:
    """JVM di sbt che nessun processo mxagent vivo sta usando. Il server di
    ``sbt --client`` si stacca dal client per progetto (ppid 1 anche a run attiva):
    conta la cartella di lavoro, non il genitore."""
    roots = live_roots(ps)
    return [p for p in ps if p.is_sbt and p.pid != os.getpid()
            and not any(under(r, p.cwd) for r in roots)]


def snapshot() -> Snapshot:
    uid = os.getuid() if hasattr(os, "getuid") else None
    ps = procs(uid)
    java = [p for p in ps if p.is_java]
    orphans = orphan_jvms(ps)
    tm = _read(PROC / "sys" / "kernel" / "threads-max")
    m = meminfo()
    ns, nh = _rlimit("RLIMIT_NPROC")
    fs, fh = _rlimit("RLIMIT_NOFILE")
    return Snapshot(ns, nh, fs, fh, len(ps), sum(p.threads for p in ps),
                    int(tm) if tm and tm.isdigit() else None, cgroup_pids(), java,
                    m.get("MemTotal"), m.get("MemAvailable"), os.cpu_count() or 1, orphans)


# --------------------------------------------------------- stima e controllo
#: thread di picco stimati per job (vedi :func:`job_peak_threads`)
def job_peak_threads(ncpu: int, sbt_mode: str = "client", jvm_cpus: int | None = None) -> int:
    """Stima prudente dei task (processi + thread) di picco di UN job.

    * server sbt (JVM): GC paralleli e compilatori JIT ~ c, pool di sbt/zinc e
      Chisel ~ 60 -> 2*c + 60, con c = ``jvm_cpus`` (``-XX:ActiveProcessorCount``,
      ``sim.jvm_cpus``) o tutti i core se non limitato;
    * client sbt (``--client``, JVM o sbtn) o JVM ``-batch`` a freddo: c + 30;
    * firtool (``--mlir-disable-threading``) e il processo Python della run: ~ 15;
    * Verilator: build con ``make``/g++ (fino a ncpu processi) e cocotb: ncpu + 10.
    """
    c = min(ncpu, jvm_cpus) if jvm_cpus else ncpu
    jvm = 2 * c + 60
    client = c + 30
    return jvm + client + 15 + ncpu + 10


#: task riservati fuori dai job: Ollama (thread di llama.cpp, CUDA, HTTP), tmux,
#: shell, runner della campagna, sintesi finale (yosys/nextpnr)
RESERVED_TASKS = 1024


def supported() -> bool:
    """Il controllo delle risorse richiede /proc (Linux)."""
    return os.name != "nt" and PROC.is_dir()


def preflight(parallel: int, snap: Snapshot | None = None,
              jvm_cpus: int | None = None) -> tuple[list[str], list[str], dict]:
    """Controllo prima di una campagna: (problemi bloccanti, avvisi, stima)."""
    s = snap or snapshot()
    per_job = job_peak_threads(s.ncpu, jvm_cpus=jvm_cpus)
    need = parallel * per_job + RESERVED_TASKS
    est = {"parallel": parallel, "per_job_tasks": per_job, "reserved_tasks": RESERVED_TASKS,
           "jvm_cpus": jvm_cpus,
           "need_tasks": need, "ncpu": s.ncpu, "pids_limit": s.pids_limit, "pids_used": s.pids_used}
    errors, warns = [], []
    if not supported():
        warns.append("controllo delle risorse non disponibile su questo sistema (serve /proc)")
        return errors, warns, est
    lim, used = s.pids_limit, s.pids_used or 0
    if lim is not None and lim - used < need:
        errors.append(
            f"limite di processi/thread troppo basso: disponibili {lim - used} task (limite {lim}, "
            f"in uso {used}); per parallel={parallel} ne servono ~{need} "
            f"({parallel} x {per_job} di picco per job: JVM di sbt server+client, firtool, "
            f"Verilator/g++; più {RESERVED_TASKS} per Ollama, tmux, sintesi). Oltre il limite la "
            "JVM fallisce con 'unable to create native thread' e fork con EAGAIN (Errno 11). "
            "Alza `ulimit -u` / TasksMax del cgroup o riduci --parallel.")
    if s.orphans:
        errors.append(
            f"{len(s.orphans)} JVM di sbt orfane (lanciate da run non più attive): "
            + ", ".join(f"pid {p.pid} ({p.threads} thread, cwd {p.cwd or '?'})" for p in s.orphans[:8])
            + ". Occupano thread e memoria; terminale con `mxagent doctor --resources --kill-orphans` "
            "(o `kill <pid>`).")
    if s.nofile_soft not in (None, -1) and s.nofile_soft < 4096:
        warns.append(f"ulimit -n = {s.nofile_soft}: una JVM di sbt apre centinaia di jar; "
                     "consigliato >= 4096")
    if s.mem_available is not None:
        need_mem = parallel * 4 * 2**30  # ~4 GiB per job: heap di sbt server + client + Verilator
        if s.mem_available < need_mem:
            warns.append(f"memoria disponibile {s.mem_available / 2**30:.1f} GiB < "
                         f"{need_mem / 2**30:.0f} GiB stimati per parallel={parallel} "
                         "(~4 GiB per job; Ollama tiene il modello in memoria a parte)")
    return errors, warns, est


def render_snapshot(s: Snapshot) -> str:
    def gib(x):
        return "?" if x is None else f"{x / 2**30:.1f} GiB"

    def lim(x):
        return "?" if x is None else ("unlimited" if x == -1 else str(x))
    L = [f"ulimit -u (processi+thread): soft {lim(s.nproc_soft)}, hard {lim(s.nproc_hard)}",
         f"ulimit -n (file aperti):     soft {lim(s.nofile_soft)}, hard {lim(s.nofile_hard)}",
         f"kernel.threads-max: {lim(s.threads_max)}   cpu: {s.ncpu}",
         f"processi dell'utente: {s.user_procs}, thread (task): {s.user_threads}"]
    for c in s.cgroups:
        L.append(f"cgroup pids: max {c['max']}, current {c['current']}  ({c['path']})")
    if not s.cgroups:
        L.append("cgroup pids: nessun limite trovato")
    L.append(f"limite effettivo sui task: {lim(s.pids_limit)}, in uso: {s.pids_used}")
    L.append(f"memoria: totale {gib(s.mem_total)}, disponibile {gib(s.mem_available)}")
    L.append(f"JVM dell'utente: {len(s.java)}" + (f" (orfane di sbt: {len(s.orphans)})" if s.java else ""))
    for p in s.java:
        tag = " ORFANA" if p in s.orphans else ""
        cpus = f" cpu {p.active_cpus}" if p.active_cpus else ""
        L.append(f"  pid {p.pid} ppid {p.ppid} {p.role} thread {p.threads}{cpus}{tag} "
                 f"cwd {p.cwd or '?'}  {p.cmdline[:140]}")
    return "\n".join(L)
