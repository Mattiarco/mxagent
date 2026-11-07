"""Wrapper RISC-V CV-X-IF (M7): elaborazione, programma di test, simulazione, verifica.

Il wrapper (``chisel/src/main/scala/mxagent/cvxif/MxCvxif.scala``) è scritto a
mano, è fisso e istanzia il MAC di b08 come ``ExtModule``: si elabora una volta
sola e si simula insieme al Verilog di qualunque MAC verificato (generato
dall'agente o di riferimento). Il programma di test:

* esegue ogni vettore del golden b08 (corner + casuali) come sequenza
  mx.lda x4, mx.ldb x4, mx.scale, mx.mac e controlla rd col golden ``mx_mac``;
* inserisce istruzioni da rifiutare (altri opcode, funct7/funct3 non validi);
* inserisce istruzioni uccise (commit_kill) che non devono cambiare lo stato;
* varia il ritardo degli operandi (rs_valid), la posizione del commit (anche
  prima dell'handshake di issue) e la backpressure su result_ready.

Ogni controllo è contro le regole di CV-X-IF v1.0.0 (docs/cvxif.md).
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..bench.kernels import get_kernel
from ..bench.task import Task
from ..formats.packing import unpack
from ..util.shell import run
from ..verify import chisel

WRAPPER_SRC = chisel.CHISEL_TEMPLATE / "src" / "main" / "scala" / "mxagent" / "cvxif" / "MxCvxif.scala"
TOP = "MxCvxifCoprocessor"
PKG = "mxagent.cvxif"
OPCODE = 0b0001011
LDA, LDB, SCALE, MAC = 0, 1, 2, 3
ID_W = 4


def r_type(f7: int, rs2: int, rs1: int, f3: int, rd: int, opcode: int = OPCODE) -> int:
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | opcode


@dataclass
class WrapReport:
    passed: bool
    stage: str  # elaborate | build | sim | check | pass
    instructions: int = 0
    macs: int = 0
    killed: int = 0
    rejected: int = 0
    cycles: int = 0
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------- elaborazione
def elaborate_wrapper(work: Path, sbt_mode: str = "batch",
                      src: Path | None = None) -> tuple[list[Path], str]:
    """SystemVerilog del wrapper (senza il MAC); cache per contenuto del sorgente.
    ``src`` sostituisce il wrapper (test di mutazione)."""
    import hashlib
    code = (src or WRAPPER_SRC).read_text(encoding="utf-8")
    key = hashlib.sha256(code.encode()).hexdigest()[:12]
    out = work / f"wrapper_{key}"
    done = sorted(out.glob("*.sv")) if out.is_dir() else []
    if done:
        return done, ""
    ws = chisel.prepare_workspace(work / "sbt")
    chisel.write_sources(ws, PKG, {"MxCvxif.scala": code})
    el = chisel.elaborate(ws, PKG, TOP, out, mode=sbt_mode)
    if not el.ok:
        return [], "; ".join(e.render() for e in el.errors)
    # l'ExtModule non deve produrre un modulo MXFP4Mac vuoto
    return [p for p in el.verilog if p.stem != "MXFP4Mac"], ""


# ----------------------------------------------------------------- programma
def program(task: Task, seed: int = 0, n_random: int = 150) -> dict:
    """Istruzioni con id, temporizzazione e attese (stato architetturale simulato)."""
    rng = random.Random(seed)
    k = get_kernel(task.golden)
    num = task.numerics_obj
    vecs = list(k.corner(task.params)) + [k.random(rng, task.params) for _ in range(n_random)]
    a_buf, b_buf, scales = [0] * 4, [0] * 4, [127, 127]
    instrs: list[dict] = []

    def emit(f7: int, rs0: int, rs1: int, *, legal: bool = True, kill: bool = False,
             raw: int | None = None) -> dict:
        rd = rng.randrange(1, 32)
        ins = {"instr": raw if raw is not None else r_type(f7, rng.randrange(32), rng.randrange(32), 0, rd),
               "rs0": rs0 & 0xFFFFFFFF, "rs1": rs1 & 0xFFFFFFFF, "legal": legal, "kill": kill,
               "op": f7 if legal else None, "rd": (raw >> 7) & 31 if raw is not None else rd,
               "id": len(instrs) % (1 << ID_W), "rs_delay": rng.choice([0, 0, 0, 1, 2, 4]),
               "commit_offset": rng.choice([0, 0, 1, 2, 3, 6]), "expect": None}
        instrs.append(ins)
        return ins

    def maybe_noise() -> None:
        r = rng.random()
        if r < 0.08:  # istruzione da rifiutare
            raw = rng.choice([
                r_type(0, 1, 2, 0, 3, 0b0110011),           # add (opcode OP)
                r_type(rng.randint(4, 127), 1, 2, 0, 3),    # funct7 non valido
                r_type(MAC, 1, 2, rng.randint(1, 7), 3),    # funct3 non valido
                r_type(0, 1, 2, 0, 3, 0b0101011),           # custom-1
            ])
            emit(0, rng.getrandbits(32), rng.getrandbits(32), legal=False, raw=raw)
        elif r < 0.16:  # istruzione uccisa: non deve cambiare lo stato
            op = rng.choice([LDA, LDB, SCALE, MAC])
            emit(op, rng.getrandbits(32), rng.randrange(4), kill=True)

    for v in vecs:
        ea, eb = unpack(v["a"], 32, 4), unpack(v["b"], 32, 4)
        for j in rng.sample(range(4), 4):
            maybe_noise()
            emit(LDA, ea[j], j)
            a_buf[j] = ea[j]
            maybe_noise()
            emit(LDB, eb[j], j)
            b_buf[j] = eb[j]
        maybe_noise()
        emit(SCALE, v["a_scale"], v["b_scale"])
        scales = [v["a_scale"], v["b_scale"]]
        maybe_noise()
        m = emit(MAC, v["acc"], rng.getrandbits(32))
        cur = {"acc": v["acc"], "a_scale": scales[0], "b_scale": scales[1],
               "a": sum(c << (32 * i) for i, c in enumerate(a_buf)),
               "b": sum(c << (32 * i) for i, c in enumerate(b_buf))}
        m["expect"] = k.fn(cur, num, task.params)["y"]
    return {"seed": seed, "id_width": ID_W, "instrs": instrs}


# ------------------------------------------------------------------- verifica
def check(prog: dict, obs: dict) -> list[str]:
    errs: list[str] = []
    ins = prog["instrs"]
    if obs.get("timeout"):
        errs.append(f"timeout: {len(obs['issues'])}/{len(ins)} istruzioni emesse, "
                    f"{len(obs['results'])} result")
    for o in obs["issues"]:
        e = ins[o["n"]]
        want = (int(e["legal"]), int(e["legal"] and e["op"] == MAC),
                (1 if e["op"] == MAC else 3) if e["legal"] else 0)
        got = (o["accept"], o["writeback"], o["register_read"])
        if got != want:
            errs.append(f"issue #{o['n']} id={e['id']} instr={e['instr']:#010x}: "
                        f"(accept, writeback, register_read) = {got}, attesi {want}")
    expected = [e for e in ins if e["legal"] and not e["kill"]]
    res = obs["results"]
    if len(res) != len(expected):
        errs.append(f"{len(res)} transazioni di result, attese {len(expected)} "
                    "(una per istruzione accettata e non uccisa)")
    for n, (r, e) in enumerate(zip(res, expected)):
        problems = []
        if r["id"] != e["id"]:
            problems.append(f"id {r['id']} (atteso {e['id']})")
        if r["we"] != int(e["op"] == MAC):
            problems.append(f"we {r['we']}")
        if e["op"] == MAC:
            if r["rd"] != e["rd"]:
                problems.append(f"rd {r['rd']} (atteso {e['rd']})")
            if r["data"] != e["expect"]:
                problems.append(f"data {r['data'] if r['data'] is None else hex(r['data'])} "
                                f"(atteso {e['expect']:#010x})")
        if problems:
            errs.append(f"result {n} (istruzione id={e['id']} op={e['op']}): " + ", ".join(problems))
        if len(errs) > 20:
            break
    return errs


def test_wrapper(task: Task, mac_verilog: list[str | Path], work: Path, *, seed: int = 0,
                 n_random: int = 150, sbt_mode: str = "batch", timeout: float = 1800,
                 wrapper_src: Path | None = None) -> WrapReport:
    import time
    t0 = time.monotonic()
    work.mkdir(parents=True, exist_ok=True)
    wrap_sv, err = elaborate_wrapper(work.parent / "_cvxif_wrapper", sbt_mode, wrapper_src)
    if not wrap_sv:
        return WrapReport(False, "elaborate", errors=[err or "elaborazione del wrapper fallita"],
                          seconds=round(time.monotonic() - t0, 1))
    prog = program(task, seed, n_random)
    pf = work / "program.json"
    pf.write_text(json.dumps(prog), encoding="utf-8")
    rf = work / "results.json"
    rf.unlink(missing_ok=True)
    entry = [sys.executable, "-m", "mxagent.verify.sim_entry"]
    srcs = [str(p) for p in [*wrap_sv, *mac_verilog] if str(p).endswith((".sv", ".v"))]
    b = run([*entry, "build", "--top", TOP, "--build-dir", str(work / "sim"), "--sources", *srcs],
            timeout=timeout)
    (work / "build.log").write_text(b.log, encoding="utf-8")
    rep = WrapReport(False, "build", instructions=len(prog["instrs"]),
                     macs=sum(e["op"] == MAC and e["legal"] and not e["kill"] for e in prog["instrs"]),
                     killed=sum(e["kill"] for e in prog["instrs"]),
                     rejected=sum(not e["legal"] for e in prog["instrs"]))
    if not b.ok:
        rep.errors = [ln for ln in b.log.splitlines() if "%Error" in ln][:5] or [b.log[-400:]]
        rep.seconds = round(time.monotonic() - t0, 1)
        return rep
    t = run([*entry, "test", "--top", TOP, "--build-dir", str(work / "sim"),
             "--test-module", "mxagent.wrap.tb_cvxif",
             "--env", f"MXAGENT_PROGRAM={pf.resolve()}", f"MXAGENT_RESULTS={rf.resolve()}"],
            timeout=timeout)
    (work / "sim.log").write_text(t.log, encoding="utf-8")
    if not rf.is_file():
        rep.stage, rep.errors = "sim", [t.log[-600:]]
        rep.seconds = round(time.monotonic() - t0, 1)
        return rep
    obs = json.loads(rf.read_text(encoding="utf-8"))
    rep.cycles = obs["cycles"]
    rep.errors = check(prog, obs)
    rep.stage = "pass" if not rep.errors else "check"
    rep.passed = not rep.errors
    rep.seconds = round(time.monotonic() - t0, 1)
    (work / "wrap_report.json").write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")
    return rep
