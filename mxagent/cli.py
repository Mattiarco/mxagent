"""CLI headless di mxagent.

    mxagent doctor [--warmup] [--synth] [--config configs/default.yaml]
    mxagent golden card   [--format mxfp4|nvfp4]
    mxagent golden dot    --a-scale 127 --a 2,2,... --b-scale 127 --b 2,2,...
    mxagent golden quant  --x 1.0,-2.5,...
    mxagent verify --task benchmarks/b01_e2m1_decode [--sources X.scala]
    mxagent run    --task benchmarks/b01_e2m1_decode [--mode full|baseline-a|baseline-b]
    mxagent synth  --task benchmarks/b05_mxfp4_dot32 [--sources X.scala] | --run runs/<id>
    mxagent wrap   --task benchmarks/b08_mxfp4_mac [--sources X.scala] | --run runs/<id>
    mxagent eval   --campaign configs/eval/main.yaml [--dry-run] [--parallel N]
    mxagent eval   --tasks b04_mxfp4_quantize,b05_mxfp4_dot32 --configs full,baseline_a,baseline_b \
                   --model-profiles all14b,all32b --seeds 0 --campaign-budget-s 21600 \
                   --out runs/eval_<data> [--dry-run]
    mxagent report runs/eval_<data>   (oppure --dir)
"""
from __future__ import annotations

import argparse
import struct
import sys

from . import __version__

def _ints(s: str) -> list[int]:
    return [int(x, 0) for x in s.replace(" ", "").split(",") if x]


def _block(scale: int, elems: list[int], k: int):
    from .formats.golden import MXBlock

    if len(elems) == 1:
        elems = elems * k
    if len(elems) != k:
        raise SystemExit(f"servono {k} elementi (o uno solo, replicato), ricevuti {len(elems)}")
    return MXBlock(scale, tuple(elems))


def _f32(bits: int) -> str:
    return f"0x{bits:08X} ({struct.unpack('<f', struct.pack('<I', bits))[0]!r})"


def _clean(path, keep: bool) -> None:
    """Cancella gli artefatti di build (sbt/target, oggetti di Verilator) sotto ``path``."""
    from pathlib import Path

    from .run.cleanup import clean_run

    if keep or not Path(path).is_dir():
        return
    r = clean_run(Path(path))
    if r["freed_bytes"]:
        print(f"pulizia {path}: {r['freed_bytes'] / 1e6:.1f} MB di build rimossi "
              f"({r['kept_bytes'] / 1e6:.1f} MB restano)")


def cmd_doctor(a: argparse.Namespace) -> int:
    from .config import load_config
    from .doctor import format_checks, run_checks

    cfg = load_config(a.config)
    if a.synth:
        from .doctor import synth_backends
        print(f"mxagent {__version__} - backend di sintesi (synth.backends = {cfg.synth.backends})\n")
        checks = synth_backends(cfg.synth.backends, cfg.synth.liberty)
        print("\n".join(format_checks(checks).splitlines()[:-2]))  # senza "ambiente pronto"
        miss = [c.name for c in checks if not c.ok and "[config]" in c.name]
        if miss:
            print(f"backend in config non disponibili (saranno 'skipped' nel ppa.json): {', '.join(miss)}")
        return 0
    checks = run_checks(cfg.llm.host, cfg.llm.models(), do_warmup=a.warmup, skip_llm=a.skip_llm)
    print(f"mxagent {__version__} - controllo dell'ambiente\n")
    print(format_checks(checks))
    return 0 if all(c.ok for c in checks if c.required) else 1


def cmd_golden(a: argparse.Namespace) -> int:
    from .formats import golden as g
    from .formats.cards import format_card
    from .formats.numerics import Numerics
    from .formats.specs import BLOCK_FORMATS

    fmt = BLOCK_FORMATS[a.format]
    num = Numerics.from_dict(dict(kv.split("=", 1) for kv in a.numerics))
    if a.what == "card":
        print(format_card(fmt, num))
    elif a.what == "dot":
        A, B = _block(a.a_scale, _ints(a.a), fmt.k), _block(a.b_scale, _ints(a.b), fmt.k)
        r = g.mx_dot(A, B, num, fmt=fmt) if fmt.scale == "e8m0" else g.nvfp4_dot(A, B, num)
        print(_f32(r))
    elif a.what == "quant":
        xs = [float(x) for x in a.x.split(",")]
        if len(xs) == 1:
            xs *= fmt.k
        if len(xs) > fmt.k:
            raise SystemExit(f"al massimo {fmt.k} valori")
        xs += [0.0] * (fmt.k - len(xs))  # blocco completato con zeri
        bits = [struct.unpack("<I", struct.pack("<f", x))[0] for x in xs]
        q = g.quantize_mx(bits, fmt, num)
        print(f"scale = 0x{q.scale:02X}  elems = {' '.join(f'{e:X}' for e in q.elems)}")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    """Verifica deterministica di sorgenti Chisel contro un benchmark."""
    import os
    from pathlib import Path

    from .bench.task import load_task, reference_sources
    from .verify.verifier import Verifier

    if a.block:
        import json as _json

        from .bench.subkernels import block_task
        task = block_task(a.block, a.top or a.block, _json.loads(a.params or "{}"))
    else:
        task = load_task(a.task)
    files = [Path(p) for p in a.sources] or reference_sources(task)
    if not files:
        raise SystemExit("nessun sorgente: passa --sources o aggiungi reference/*.scala al task")
    sources = {p.name: p.read_text(encoding="utf-8") for p in files}
    work = Path(a.work or os.environ.get("MXAGENT_WORK") or "runs/verify")
    mode = a.sbt_mode or ("batch" if os.name == "nt" else "client")
    with Verifier(task, work / task.name, sbt_mode=mode, seed=a.seed,
                  random_vectors=a.random) as v:
        rep = v.verify(sources, a.tag)
        if a.keep_server:
            v.sbt_mode = "batch"  # evita lo shutdown in __exit__
    # col server sbt ancora acceso non si toccano i suoi target
    _clean(work / task.name, a.keep_build or a.keep_server)
    print(f"[{task.name}] stage={rep.stage} passed={rep.passed} "
          f"score={rep.score} tempi={rep.seconds}")
    print(rep.feedback)
    return 0 if rep.passed else 1


def cmd_run(a: argparse.Namespace) -> int:
    import json
    import os
    from pathlib import Path

    from .bench.task import load_task
    from .config import load_config
    from .run.baselines import MODES, run_loop

    if a.mode not in (*MODES, "full"):
        raise SystemExit(f"modalità {a.mode}: disponibili {MODES + ('full',)}")
    from .config import LLM_ROLES

    over = {}
    if a.model and a.model_profile:
        raise SystemExit("--model e --model-profile sono alternativi")
    if a.model:  # scorciatoia per un profilo uniforme: tutti i ruoli LLM, embed escluso
        over = {f"llm.roles.{r}.model": a.model for r in LLM_ROLES}
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        import yaml as _yaml
        over[k] = _yaml.safe_load(v)
    try:
        cfg = load_config(a.config, over, profile=a.model_profile)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    task = load_task(a.task)
    runs = Path(a.runs_dir or os.environ.get("MXAGENT_RUNS") or cfg.runs_dir)
    if a.mode == "full":
        from .agents.orchestrator import run_full
        res = run_full(task, cfg, seed=a.seed, runs_dir=runs, run_id=a.run_id)
    else:
        res = run_loop(task, cfg, a.mode, seed=a.seed, role=cfg.llm.roles["coder"], runs_dir=runs,
                       run_id=a.run_id)
    m = res.metrics
    print(json.dumps({k: m[k] for k in ("task", "mode", "model", "seed", "passed", "stop",
                                        "iterations", "best_stage", "best_pass_rate",
                                        "wall_s", "llm") if k in m}, indent=1))
    if m.get("ppa"):
        print("ppa:", json.dumps(m["ppa"]))
    print(f"run salvata in {res.run_dir}")
    return 0 if m["passed"] else 1


def cmd_synth(a: argparse.Namespace) -> int:
    """PPA di un design: riferimento/sorgenti di un task (verificati prima) o finale di una run."""
    import json
    import os
    from pathlib import Path

    from .bench.task import load_task, reference_sources
    from .config import load_config
    from .synth.ppa import synthesize
    from .synth.yosys import tool_versions

    over = {"synth.backends": a.backends.split(",")} if a.backends else {}
    cfg = load_config(a.config, over)
    if a.run:
        run = Path(a.run)
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        task = load_task(Path(a.bench_dir) / meta["task"])
        verilog = sorted(str(p) for p in (run / "final").glob("*.sv"))
        if not verilog:
            raise SystemExit(f"{run}/final non contiene Verilog (run non passata?)")
        out = Path(a.out or run / "synth")
    else:
        from .verify.verifier import Verifier
        if not a.task:
            raise SystemExit("serve --task o --run")
        task = load_task(a.task)
        files = [Path(p) for p in a.sources] or reference_sources(task)
        sources = {p.name: p.read_text(encoding="utf-8") for p in files}
        work = Path(os.environ.get("MXAGENT_WORK") or "runs/verify") / task.name
        mode = "batch" if os.name == "nt" else "client"
        with Verifier(task, work, sbt_mode=mode, random_vectors=a.random) as v:
            rep = v.verify(sources, a.tag)
        _clean(work, a.keep_build)  # il Verilog emesso (.sv) resta
        if not rep.passed:  # si sintetizza solo un design corretto
            print(rep.feedback)
            raise SystemExit(f"[{task.name}] il design non passa la verifica ({rep.stage})")
        verilog = rep.verilog
        out = Path(a.out or work / a.tag / "synth")
    rep = synthesize(verilog, task, cfg.synth, out)
    data = {"task": task.name, "top": task.top, "tools": tool_versions(), **rep.to_dict()}
    (out / "ppa.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"[{task.name}] {task.top}  (report: {out / 'ppa.json'})")
    print(rep.render())
    return 0 if all(r.ok or r.skipped for r in rep.results.values()) else 1


def cmd_wrap(a: argparse.Namespace) -> int:
    """Coprocessore CV-X-IF attorno al MAC di b08 (riferimento, sorgenti o run) e suo test."""
    import json
    import os
    from pathlib import Path

    from .bench.task import load_task, reference_sources
    from .wrap.cvxif import test_wrapper

    mode = "batch" if os.name == "nt" else "client"
    if a.run:
        run_dir = Path(a.run)
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        task = load_task(Path(a.bench_dir) / meta["task"])
        verilog = sorted(str(p) for p in (run_dir / "final").glob("*.sv"))
        work = Path(a.out or run_dir / "wrap")
    else:
        from .verify.verifier import Verifier
        task = load_task(a.task)
        files = [Path(p) for p in a.sources] or reference_sources(task)
        base = Path(os.environ.get("MXAGENT_WORK") or "runs/verify") / task.name
        with Verifier(task, base, sbt_mode=mode, random_vectors=a.random) as v:
            rep = v.verify({p.name: p.read_text(encoding="utf-8") for p in files}, a.tag)
        _clean(base, a.keep_build)
        if not rep.passed:
            print(rep.feedback)
            raise SystemExit(f"[{task.name}] il MAC non passa la verifica ({rep.stage})")
        verilog = rep.verilog
        work = Path(a.out or base / a.tag / "wrap")
    if task.wrapper != "cvxif":
        raise SystemExit(f"{task.name} non dichiara wrapper: cvxif")
    r = test_wrapper(task, verilog, work, seed=a.seed, n_random=a.programs, sbt_mode="batch")
    _clean(work, a.keep_build)
    print(f"[{task.name}] CV-X-IF: {r.stage} passed={r.passed} istruzioni={r.instructions} "
          f"mac={r.macs} uccise={r.killed} rifiutate={r.rejected} cicli={r.cycles} ({r.seconds} s)")
    for e in r.errors[:10]:
        print("  -", e)
    return 0 if r.passed else 1


def _seeds(s: str) -> list[int]:
    """``0``, ``0,1,2`` o ``0-4``: seed consecutivi."""
    if "-" in s:
        lo, hi = (int(x) for x in s.split("-", 1))
        out = list(range(lo, hi + 1))
    else:
        out = _ints(s)
    if not out or out != list(range(out[0], out[0] + len(out))):
        raise SystemExit(f"--seeds {s}: servono seed consecutivi (es. 0, 0,1,2 o 0-4)")
    return out


def flag_campaign(a: argparse.Namespace, name: str):
    """Campagna dai flag di ``mxagent eval`` (senza YAML)."""
    from .eval.campaign import Campaign

    if not (a.tasks and a.configs and a.model_profiles):
        raise SystemExit("servono --campaign oppure --tasks, --configs e --model-profiles")
    seeds = _seeds(a.seeds)
    return Campaign.model_validate({
        "name": name, "config": a.config, "tasks": a.tasks.split(","),
        "modes": a.configs.split(","), "seeds": len(seeds), "seed_start": seeds[0],
        "models": {p: {"profile": p} for p in a.model_profiles.split(",")},
        # libreria in sola lettura: nessuna run impara da un'altra della stessa campagna
        "library": {"policy": "frozen", "path": a.library},
        "set": {"budget.max_wall_s": a.max_wall_s, "budget.max_llm_calls": a.max_llm_calls,
                "budget.max_tokens": a.max_tokens, "synth.optimize_iters": a.ppa_iters,
                "library.exclude_current_task": True},
        "ppa_backends": a.synth_backends.split(","),
        # SIGTERM al job oltre max_wall_s + 15 min (sintesi finale): result.json resta
        "job_timeout_s": a.max_wall_s + 900,
        "campaign_budget_s": a.campaign_budget_s, "warmup": not a.no_warmup})


def describe_campaign(c, r, todo: list) -> str:
    """Testo del dry-run: job in ordine, budget e stima del tempo (limite superiore)."""
    from .eval.campaign import jobs

    cfg = r.base_config()
    b, s = cfg.budget, cfg.synth
    js = jobs(c, r.out)
    L = [f"campagna {c.name} ({c.digest()}): {len(js)} job, {len(todo)} da eseguire", "",
         f"budget per run: max_wall_s {b.max_wall_s} s, max_llm_calls {b.max_llm_calls}, "
         f"max_tokens {b.max_tokens}",
         f"PPA (solo full): al più {s.optimize_iters} tentativi, plateau dopo {s.ppa_patience}, "
         f"saltato sotto {s.ppa_min_cells} celle; baseline senza PPA",
         f"sintesi finale delle run passate: {', '.join(c.ppa_backends)}",
         f"timeout per job: {r.timeout():.0f} s (SIGTERM, poi SIGKILL dopo 120 s)",
         "budget di campagna: " + (f"{c.campaign_budget_s} s ({c.campaign_budget_s / 3600:.1f} h), "
                                   "poi skipped_budget" if c.campaign_budget_s else "nessuno"),
         f"libreria: {c.library.policy}" + (f" ({c.library.path}, sola lettura)" if c.library.path else ""),
         "warm-up dei modelli a ogni cambio di profilo (fuori dal conteggio): "
         + ("sì" if c.warmup else "no"), "", "ordine di esecuzione:"]
    done = {j.id for j in js} - {j.id for j in todo}
    for i, j in enumerate(js, 1):
        L.append(f"  {i:2d}. {j.task:22s} {j.mode:11s} {j.models:8s} s{j.seed}"
                 + ("   (già fatto)" if j.id in done else ""))
    n = len(todo)
    worst = n * b.max_wall_s
    L += ["", f"stima: ogni run si ferma a max_wall_s, quindi al più {n} x {b.max_wall_s} s = "
              f"{worst / 3600:.1f} h di run, più la sintesi finale delle run passate "
              f"(minuti per run). È un limite superiore, non una previsione."]
    if c.campaign_budget_s and worst >= c.campaign_budget_s:
        L.append(f"il limite superiore raggiunge il budget di campagna "
                 f"({c.campaign_budget_s / 3600:.1f} h): le ultime run potrebbero finire in "
                 "skipped_budget (prima i profili/task in testa all'elenco)")
    if todo:
        L += ["", "primo comando: " + " ".join(r.command(todo[0]))]
    return "\n".join(L)


def cmd_eval(a: argparse.Namespace) -> int:
    """Esegue (o riprende) una campagna di valutazione."""
    from datetime import date
    from pathlib import Path

    from .eval.campaign import load_campaign
    from .eval.report import write_report
    from .eval.runner import Runner

    if a.campaign:
        c = load_campaign(a.campaign)
        if a.campaign_budget_s:
            c = c.model_copy(update={"campaign_budget_s": a.campaign_budget_s})
        out = Path(a.out or Path("evals") / c.name)
    else:
        out = Path(a.out or Path("runs") / f"eval_{date.today():%Y%m%d}")
        c = flag_campaign(a, out.name)
    r = Runner(c, out)
    if a.dry_run:
        print(describe_campaign(c, r, r.pending(a.retry_errors)))
        return 0
    r.run(parallel=a.parallel, retry_errors=a.retry_errors, limit=a.limit)
    S = write_report(out)
    print(f"report: {out / 'summary.md'}  (job: {S['jobs']})")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from pathlib import Path

    from .eval.report import write_report

    d = Path(a.path or a.dir or "")
    if not str(a.path or a.dir or ""):
        raise SystemExit("serve la cartella della campagna: mxagent report runs/eval_<data>")
    S = write_report(d)
    print(f"report: {d / 'summary.md'}  (job: {S['jobs']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mxagent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"mxagent {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="verifica toolchain, modelli Ollama e cache")
    d.add_argument("--config", default="configs/default.yaml")
    d.add_argument("--warmup", action="store_true", help="elabora un modulo di prova (popola le cache)")
    d.add_argument("--skip-llm", action="store_true", help="non controllare Ollama (solo toolchain HW)")
    d.add_argument("--synth", action="store_true", help="elenca i backend di sintesi disponibili/mancanti")
    d.set_defaults(func=cmd_doctor)

    gp = sub.add_parser("golden", help="interroga il golden model")
    gp.add_argument("what", choices=["card", "dot", "quant"])
    gp.add_argument("--format", default="mxfp4", choices=["mxfp4", "nvfp4"])
    gp.add_argument("--numerics", nargs="*", default=[], metavar="KEY=VAL")
    gp.add_argument("--a-scale", type=lambda s: int(s, 0), default=127)
    gp.add_argument("--b-scale", type=lambda s: int(s, 0), default=127)
    gp.add_argument("--a", default="2")
    gp.add_argument("--b", default="2")
    gp.add_argument("--x", default="1.0")
    gp.set_defaults(func=cmd_golden)

    vp = sub.add_parser("verify", help="verifica deterministica (compile, lint, cocotb) di un design")
    vp.add_argument("--task", help="cartella del benchmark")
    vp.add_argument("--block", help="in alternativa a --task: blocco del catalogo (es. fixed_to_fp32)")
    vp.add_argument("--top", help="nome del modulo per --block")
    vp.add_argument("--params", help="parametri JSON per --block")
    vp.add_argument("--sources", nargs="*", default=[], help="file .scala (default: il riferimento)")
    vp.add_argument("--tag", default="manual")
    vp.add_argument("--work", default=None, help="cartella di lavoro (default $MXAGENT_WORK o runs/verify)")
    vp.add_argument("--sbt-mode", default=None, choices=["batch", "client"],
                    help="client = server sbt persistente (default su Linux), batch = a freddo")
    vp.add_argument("--keep-server", action="store_true", help="lascia acceso il server sbt")
    vp.add_argument("--keep-build", action="store_true",
                    help="non cancellare sbt/target e gli oggetti di Verilator")
    vp.add_argument("--seed", type=int, default=0)
    vp.add_argument("--random", type=int, default=2000)
    vp.set_defaults(func=cmd_verify)

    rp = sub.add_parser("run", help="esegue l'agente su un benchmark")
    rp.add_argument("--task", required=True)
    rp.add_argument("--mode", default="full", help="full | baseline-a | baseline-b")
    rp.add_argument("--set", nargs="*", metavar="KEY=VAL",
                    help="override della config, es. agent.decompose=false (ablation)")
    rp.add_argument("--model", default=None,
                    help="un solo modello per tutti i ruoli LLM (embed escluso)")
    rp.add_argument("--model-profile", default=None,
                    help="profilo di modello della config: all14b | all32b | mixed")
    rp.add_argument("--config", default="configs/default.yaml")
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--runs-dir", default=None, help="default $MXAGENT_RUNS o runs_dir della config")
    rp.add_argument("--run-id", default=None, help="nome della cartella della run (default: generato)")
    rp.set_defaults(func=cmd_run)

    sp = sub.add_parser("synth", help="sintesi e PPA (Yosys, nextpnr, OpenSTA/Vivado se presenti)")
    sp.add_argument("--task", help="cartella del benchmark (sintetizza il riferimento o --sources)")
    sp.add_argument("--sources", nargs="*", default=[], help="file .scala (default: il riferimento)")
    sp.add_argument("--run", help="cartella di una run: sintetizza final/*.sv")
    sp.add_argument("--bench-dir", default="benchmarks", help="dove cercare il task di una --run")
    sp.add_argument("--backends", default=None, help="es. generic,ecp5,nextpnr (default: config)")
    sp.add_argument("--config", default="configs/default.yaml")
    sp.add_argument("--out", default=None, help="cartella dei risultati")
    sp.add_argument("--tag", default="synth")
    sp.add_argument("--random", type=int, default=2000)
    sp.add_argument("--keep-build", action="store_true", help="non cancellare gli artefatti di build")
    sp.set_defaults(func=cmd_synth)

    wp = sub.add_parser("wrap", help="wrapper coprocessore RISC-V CV-X-IF del MAC (b08) e suo test")
    wp.add_argument("--task", default="benchmarks/b08_mxfp4_mac")
    wp.add_argument("--sources", nargs="*", default=[], help="MAC in Chisel (default: il riferimento)")
    wp.add_argument("--run", help="cartella di una run passata: usa final/*.sv")
    wp.add_argument("--bench-dir", default="benchmarks")
    wp.add_argument("--out", default=None)
    wp.add_argument("--tag", default="wrap")
    wp.add_argument("--seed", type=int, default=0)
    wp.add_argument("--programs", type=int, default=150, help="vettori casuali oltre ai corner")
    wp.add_argument("--random", type=int, default=2000)
    wp.add_argument("--keep-build", action="store_true", help="non cancellare gli artefatti di build")
    wp.set_defaults(func=cmd_wrap)

    ep = sub.add_parser("eval", help="campagna di valutazione (baseline, sistema completo, ablation)")
    ep.add_argument("--campaign", default=None, help="YAML della campagna (es. configs/eval/main.yaml)")
    ep.add_argument("--out", default=None,
                    help="cartella della campagna (default evals/<nome> o runs/eval_<data>)")
    g = ep.add_argument_group("campagna senza YAML")
    g.add_argument("--tasks", help="es. b04_mxfp4_quantize,b05_mxfp4_dot32")
    g.add_argument("--configs", help="full,baseline_a,baseline_b")
    g.add_argument("--model-profiles", help="es. all14b,all32b (llm.profiles della config)")
    g.add_argument("--seeds", default="0", help="0 | 0,1,2 | 0-4")
    g.add_argument("--config", default="configs/default.yaml")
    g.add_argument("--max-wall-s", type=int, default=1800)
    g.add_argument("--max-llm-calls", type=int, default=40)
    g.add_argument("--max-tokens", type=int, default=300000)
    g.add_argument("--ppa-iters", type=int, default=2, help="tentativi massimi del PPA (full)")
    g.add_argument("--library", default="library", help="libreria (sola lettura)")
    g.add_argument("--synth-backends", default="generic,ecp5,xc7,nextpnr",
                   help="sintesi finale delle run passate")
    g.add_argument("--no-warmup", action="store_true", help="niente warm-up dei modelli")
    ep.add_argument("--campaign-budget-s", type=int, default=None,
                    help="tempo massimo della campagna; oltre, i job rimanenti sono skipped_budget")
    ep.add_argument("--parallel", type=int, default=1, help="job in parallelo (Ollama serializza su GPU)")
    ep.add_argument("--retry-errors", action="store_true", help="rifà i job finiti in errore/timeout")
    ep.add_argument("--limit", type=int, default=None, help="esegue al più N job (prove)")
    ep.add_argument("--dry-run", action="store_true", help="elenca i job senza eseguirli")
    ep.set_defaults(func=cmd_eval)

    pp = sub.add_parser("report", help="report di una campagna (anche incompleta)")
    pp.add_argument("path", nargs="?", default=None, help="cartella della campagna")
    pp.add_argument("--dir", default=None, help="come path (forma precedente)")
    pp.set_defaults(func=cmd_report)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
