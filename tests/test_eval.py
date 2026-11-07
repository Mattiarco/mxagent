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
            (out / "ppa.json").write_text(json.dumps({"results": {"generic": {
                "ok": True, "metrics": {"transistors": t, "depth": 10}}}}))
            return Result(cmd, 0, "", "", 0.1)
        run_id = args[args.index("--run-id") + 1]
        if run_id in self.crash:
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


def test_report_on_partial_campaign(tmp_path):
    c = camp()
    r = Runner(c, tmp_path, repo=ROOT, executor=FakeExec(), python="py")
    r.run(limit=4, log=lambda s: None)
    S = write_report(tmp_path)
    assert S["jobs"]["done"] == 4 and S["jobs"]["missing"] == 17
