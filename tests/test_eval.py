"""Valutazione (M6): statistiche, espansione dei job, runner con esecutore
finto (nessun LLM né toolchain), report su risultati sintetici noti."""
import json
from pathlib import Path

import pytest

from mxagent.eval.campaign import Campaign, jobs
from mxagent.eval.report import write_report
from mxagent.eval.runner import Runner
from mxagent.eval.stats import geomean, mcnemar_exact, pass_at_k, wilson
from mxagent.util.shell import Result

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ stats
def test_wilson_known_values():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = wilson(0, 10)
    assert lo == 0 and hi == pytest.approx(0.2775, abs=1e-4)
    assert wilson(0, 0) == (0.0, 1.0)


def test_pass_at_k():
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(10, 3, 5) == pytest.approx(1 - 21 / 252)
    assert pass_at_k(10, 0, 5) == 0.0 and pass_at_k(10, 6, 5) == 1.0
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(3, 3) == 1.0 and mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) / 1024)


def test_geomean():
    assert geomean([1, 4]) == pytest.approx(2.0) and geomean([None]) is None


# --------------------------------------------------------------- campagna
def camp(**over):
    d = {"name": "t", "config": "configs/default.yaml", "tasks": ["b01_e2m1_decode", "b03_e8m0_to_fp32"],
         "seeds": 3, "models": {"m": {"all": "x:1b"}},
         "ablations": {"tasks": ["b03_e8m0_to_fp32"], "variants": {"no_kb": {"agent.kb": False}}},
         "ppa_backends": ["generic"], "job_timeout_s": 60}
    d.update(over)
    return Campaign.model_validate(d)


def test_job_expansion(tmp_path):
    c = camp()
    js = jobs(c, tmp_path)
    assert len(js) == 2 * 3 * 3 + 1 * 3  # main + ablation
    assert len({j.id for j in js}) == len(js)
    base = next(j for j in js if j.mode == "baseline-b")
    assert dict(base.overrides) == {"llm.roles.coder.model": "x:1b"}  # baseline: un modello, niente libreria
    full = next(j for j in js if j.mode == "full" and j.variant == "main")
    ov = dict(full.overrides)
    assert ov["llm.roles.reflector.model"] == "x:1b" and ov["library.readonly"] is False
    assert full.id in ov["library.path"]  # libreria fresca per run
    abl = next(j for j in js if j.variant == "no_kb")
    assert dict(abl.overrides)["agent.kb"] is False and abl.task == "b03_e8m0_to_fp32"
    assert "agent.kb=false" in abl.set_args()


def test_frozen_library_and_validation(tmp_path):
    c = camp(library={"policy": "frozen", "path": "lib0"})
    full = next(j for j in jobs(c, tmp_path) if j.mode == "full")
    assert dict(full.overrides)["library.path"] == "lib0" and dict(full.overrides)["library.readonly"]
    with pytest.raises(Exception):
        camp(modes=["baseline-c"])
    with pytest.raises(Exception):
        camp(ablations={"tasks": ["b99"], "variants": {}})
    with pytest.raises(Exception):
        camp(library={"policy": "frozen"})


def test_shipped_campaigns_are_valid(tmp_path):
    from mxagent.eval.campaign import load_campaign
    for f in (ROOT / "configs" / "eval").glob("*.yaml"):
        c = load_campaign(f)
        assert jobs(c, tmp_path)
    c = load_campaign(ROOT / "configs" / "eval" / "b04_b05.yaml")
    js = jobs(c, tmp_path)
    assert len(js) == 120 and {j.task for j in js} == {"b04_mxfp4_quantize", "b05_mxfp4_dot32"}


# ------------------------------------------------- runner + report (finti)
# esiti: (task, mode, variant, seed) -> passed; transistori del design passato
OUTCOME = {
    ("b01_e2m1_decode", "full"): [True, True, True],
    ("b01_e2m1_decode", "baseline-a"): [False, False, True],
    ("b01_e2m1_decode", "baseline-b"): [True, False, False],
    ("b03_e8m0_to_fp32", "full"): [True, True, False],
    ("b03_e8m0_to_fp32", "baseline-a"): [False, False, False],
    ("b03_e8m0_to_fp32", "baseline-b"): [False, True, False],
    ("b03_e8m0_to_fp32", "no_kb"): [False, False, False],
}


class FakeExec:
    """Simula `mxagent run` / `mxagent synth` scrivendo i file che produrrebbero."""

    def __init__(self, crash: set[str] = frozenset()):
        self.calls, self.crash = [], set(crash)

    def __call__(self, cmd, cwd, timeout):
        self.calls.append(cmd)
        args = cmd[3:]
        if cmd[3] == "synth":
            out = Path(args[args.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            t = 100 if "--task" in args else 80
            (out / "ppa.json").write_text(json.dumps({"results": {
                "generic": {"ok": True, "metrics": {"transistors": t, "depth": 10}},
                "xc7": {"ok": True, "metrics": {"luts": t // 10, "ffs": 0}},
                "nextpnr": {"ok": True, "metrics": {"fmax_mhz": 200.0, "fmax_min": 190.0}}}}))
            return Result(cmd, 0, "", "", 0.1)
        run_id = args[args.index("--run-id") + 1]
        if run_id in self.crash:
            rd = Path(args[args.index("--runs-dir") + 1]) / run_id
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "result.json").write_text(json.dumps({"status": "interrupted",
                                                        "reason": "signal SIGHUP"}))
            return Result(cmd, 2, "", "Traceback ...\nRuntimeError: boom", 0.1)
        task, mode, variant, _, s = run_id.split("__")
        seed = int(s[1:])
        passed = OUTCOME[(task, variant if variant != "main" else mode)][seed]
        rd = Path(args[args.index("--runs-dir") + 1]) / run_id
        rd.mkdir(parents=True, exist_ok=True)
        m = {"task": task, "mode": mode, "passed": passed, "stop": "pass" if passed else "max_iters",
             "best_stage": "pass" if passed else "sim", "iterations": 3, "time_to_pass_s": 10.0 if passed else None,
             "wall_s": 12.0, "model": "x:1b", "llm": {"calls": 5, "tokens": 5000}}
        if mode == "full":
            m["llm_to_pass"] = {"calls": 3, "tokens": 3000}
            m["ppa"] = {"initial": {"transistors": 90, "depth": 10}, "tried": 2, "accepted": 1, "gain": 0.1}
        (rd / "metrics.json").write_text(json.dumps(m))
        (rd / "meta.json").write_text(json.dumps({"model_digest": "abc123"}))
        return Result(cmd, 0 if passed else 1, "", "", 0.1)


def test_runner_resume_errors_and_report(tmp_path):
    c = camp()
    crash = "b01_e2m1_decode__baseline-a__main__m__s2"
    ex = FakeExec(crash={crash})
    r = Runner(c, tmp_path, repo=ROOT, executor=ex, python="py")
    done = r.run(log=lambda s: None)
    assert len(done) == 21
    st = r.status()
    assert st[crash]["status"] == "error" and "boom" in st[crash]["error"]
    assert st[crash]["result_status"] == "interrupted" and "signal SIGHUP" in st[crash]["error"]
    assert st["b01_e2m1_decode__full__main__m__s0"]["ppa"] == "ok"
    assert st["b03_e8m0_to_fp32__baseline-a__main__m__s0"].get("ppa") == "not_passed"
    # ripresa: nulla da rifare, salvo gli errori su richiesta
    assert r.pending() == [] and [j.id for j in r.pending(retry_errors=True)] == [crash]
    n_calls = len(ex.calls)
    r.run(log=lambda s: None)
    assert len(ex.calls) == n_calls  # riferimenti in cache, nessun job

    S = write_report(tmp_path)
    assert S["jobs"] == {"expected": 21, "done": 20, "error": 1}
    succ = S["success"]["m"]
    assert succ["full"]["total"]["k"] == 5 and succ["full"]["total"]["n"] == 6
    assert succ["baseline-a"]["total"]["n"] == 5  # il crash non è un fallimento del modello
    p = S["paired"]["m: full vs baseline-b"]
    assert (p["pairs"], p["only_a"], p["only_b"], p["both"]) == (6, 3, 0, 2)
    assert p["p_mcnemar"] == pytest.approx(0.25)
    assert S["efficiency"]["m"]["full"]["calls_to_pass"] == 3  # chiamate fino al pass, non totali
    assert S["efficiency"]["m"]["baseline-b"]["calls_to_pass"] == 5
    a = S["ablations"]["no_kb"]
    assert a["variant"]["k"] == 0 and a["full"]["k"] == 2 and a["paired"]["only_b"] == 2
    q = S["qor"]["m"]["full"]
    assert q["area_ratio_geomean"] == pytest.approx(0.8) and q["optimizer_accepted"] == 5
    assert S["failures"]["m"]["baseline-a"]["stage"] == {"sim": 5}
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Campagna incompleta" in md and crash in md and "| b01_e2m1_decode | 0/2 | 1/3 | 3/3 |" in md
    rows = (tmp_path / "runs.csv").read_text().splitlines()
    assert len(rows) == 22 and rows[0].startswith("job,task,mode")
    # media ± dev. std per (task, modelli, modalità)
    st = {(r["task"], r["mode"]): r for r in S["stats"]}
    assert len(S["stats"]) == 2 * 3  # 2 task x 3 modalità x 1 insieme di modelli
    b01 = st[("b01_e2m1_decode", "full")]
    assert (b01["runs"], b01["passed"], b01["pass@1"], b01["pass@5"]) == (3, 3, 1.0, None)
    assert b01["adp"] == {"n": 3, "mean": 800.0, "std": 0.0}
    assert b01["luts_xc7"]["mean"] == 8 and b01["fmax_mhz"]["mean"] == 200.0
    assert b01["time_to_pass_s"] == {"n": 3, "mean": 10.0, "std": 0.0}
    b03a = st[("b03_e8m0_to_fp32", "baseline-a")]
    assert b03a["passed"] == 0 and b03a["adp"] == {"n": 0, "mean": None, "std": None}
    assert b03a["tokens_total"] == {"n": 3, "mean": 5000.0, "std": 0.0}  # su tutte le run, non solo le passate
    assert "## Media ± deviazione standard" in md and "800 ± 0" in md
    assert "result.json: interrupted" in md
    srows = (tmp_path / "stats.csv").read_text().splitlines()
    assert len(srows) == 7 and "adp_mean" in srows[0] and "pass@1" in srows[0]


def test_report_on_partial_campaign(tmp_path):
    c = camp()
    r = Runner(c, tmp_path, repo=ROOT, executor=FakeExec(), python="py")
    r.run(limit=4, log=lambda s: None)
    S = write_report(tmp_path)
    assert S["jobs"]["done"] == 4 and S["jobs"]["missing"] == 17


# ------------------------------------------------- campagna da flag, budget, warm-up
FLAGS = ["eval", "--tasks", "b04_mxfp4_quantize,b05_mxfp4_dot32",
         "--configs", "full,baseline_a,baseline_b", "--model-profiles", "all14b,all32b",
         "--seeds", "0", "--campaign-budget-s", "21600"]


def test_flag_campaign_dry_run(tmp_path, capsys, monkeypatch):
    from mxagent.cli import main
    monkeypatch.chdir(ROOT)
    assert main([*FLAGS, "--out", str(tmp_path / "eval_x"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "12 job, 12 da eseguire" in out and "21600 s (6.0 h)" in out
    assert "max_wall_s 1800 s, max_llm_calls 40, max_tokens 300000" in out
    assert "al più 2 tentativi" in out and "12 x 1800 s = 6.0 h" in out
    order = [ln.split()[1:4] for ln in out.splitlines() if ln.strip()[:3].strip(". ").isdigit()]
    # prima tutto all14b, poi all32b; dentro ogni profilo prima b04 poi b05
    assert [o[2] for o in order] == ["all14b"] * 6 + ["all32b"] * 6
    assert [o[0][:3] for o in order[:6]] == ["b04"] * 3 + ["b05"] * 3
    assert [o[1] for o in order[:3]] == ["full", "baseline-a", "baseline-b"]
    assert "llm.profile=all14b" in out and "library.readonly=true" in out
    assert not (tmp_path / "eval_x").exists()  # il dry-run non scrive nulla


def test_flag_campaign_jobs_overrides(tmp_path):
    from mxagent.cli import flag_campaign
    c = flag_campaign(_parse([*FLAGS, "--out", str(tmp_path / "e")]), "e")
    assert c.campaign_budget_s == 21600 and c.warmup and c.job_timeout_s == 2700
    js = jobs(c, tmp_path)
    full = dict(next(j for j in js if j.mode == "full").overrides)
    base = dict(next(j for j in js if j.mode == "baseline-a").overrides)
    assert full["llm.profile"] == "all14b" and full["library.readonly"] is True
    assert full["synth.optimize_iters"] == 2 and full["budget.max_llm_calls"] == 40
    assert base["llm.profile"] == "all14b" and "library.path" not in base


def _parse(argv):
    """Argomenti di ``mxagent eval`` dal parser vero, senza eseguire la campagna."""
    from mxagent import cli
    captured = {}

    def fake(a):
        captured["a"] = a
        return 0
    orig = cli.cmd_eval
    cli.cmd_eval = fake
    try:
        cli.main(argv)
    finally:
        cli.cmd_eval = orig
    return captured["a"]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_campaign_budget_skips_remaining_jobs_and_warmup_is_free(tmp_path):
    from mxagent.cli import flag_campaign
    c = flag_campaign(_parse([*FLAGS[:-2], "--campaign-budget-s", "250", "--out", "x"]), "x")
    c = c.model_copy(update={"ppa_backends": []})
    clock, warm = Clock(), []

    def ex(cmd, cwd, timeout):
        clock.t += 100  # ogni job dura 100 s
        rd = Path(cmd[cmd.index("--runs-dir") + 1]) / cmd[cmd.index("--run-id") + 1]
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "metrics.json").write_text(json.dumps({"passed": False}))
        return Result(cmd, 1, "", "", 100)

    def wu(models):
        clock.t += 10_000  # il warm-up non conta nel budget
        warm.append(sorted(set(models.values())))
        return {m: {"ok": True, "seconds": 1} for m in set(models.values())}

    r = Runner(c, tmp_path, repo=ROOT, executor=ex, warmup=wu, clock=clock)
    log = []
    r.run(log=log.append)
    st = r.status()
    order = [j.id for j in jobs(c, tmp_path)]
    assert [st[j]["status"] for j in order] == ["done"] * 3 + ["skipped_budget"] * 9
    assert "budget di campagna esaurito" in st[order[3]]["error"]
    assert warm == [["nomic-embed-text", "qwen2.5-coder:14b"]]  # solo all14b è partito
    assert (tmp_path / "warmup.jsonl").is_file()
    # ripresa senza budget: i job skipped_budget si eseguono, con warm-up del 32b
    r2 = Runner(c.model_copy(update={"campaign_budget_s": None}), tmp_path, repo=ROOT, executor=ex,
                warmup=wu, clock=clock)
    assert len(r2.pending()) == 9
    r2.run(log=log.append)
    assert all(e["status"] == "done" for e in r2.status().values())
    assert warm[-1] == ["nomic-embed-text", "qwen2.5-coder:32b"]


def test_shell_timeout_terminates(tmp_path):
    import sys
    from mxagent.util.shell import run
    r = run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1, grace=5)
    assert r.timed_out and r.seconds < 20
