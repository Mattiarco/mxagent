"""Run replayed dalla cache LLM e task esclusi dai totali (report_exclude).

Cache vera di BaseChat con un backend finto; campagne finte scritte su disco.
"""
import json
from pathlib import Path

import pytest

from mxagent.eval.campaign import Campaign, jobs, load_campaign
from mxagent.eval.report import write_report
from mxagent.llm.chat import BaseChat, Budget, LLMResponse
from mxagent.run.baselines import run_loop
from mxagent.run.result import llm_cache_fields
from tests.test_baselines import BAD, CFG, GOOD, ROLE, TASK, fake_verify

ROOT = Path(__file__).resolve().parent.parent


class FakeBackend(BaseChat):
    """Backend finto: risposte in ordine; la cache (cache_dir) è quella vera di BaseChat."""
    backend = "fake"

    def __init__(self, replies, cache_dir, max_calls=60):
        super().__init__(cache_dir=cache_dir, budget=Budget(max_calls=max_calls))
        self.replies, self.requests = list(replies), 0

    def digest(self, model):
        return "d0"

    def _payload(self, messages, **kw):
        return {"model": kw["model"], "messages": messages, "seed": kw["seed"]}

    def _request(self, payload):
        self.requests += 1
        return LLMResponse(text=self.replies.pop(0), model=payload["model"], prompt_tokens=10,
                           completion_tokens=10)


def _res(run_dir: Path) -> dict:
    return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))


def _run(tmp, name, cache_dir, replies=(BAD, GOOD), max_calls=60):
    llm = FakeBackend(replies, cache_dir, max_calls)
    r = run_loop(TASK, CFG, "baseline-b", seed=3, role=ROLE, runs_dir=tmp, llm=llm,
                 verify_fn=fake_verify, run_id=name)
    return r, llm


def test_second_identical_run_is_replayed(tmp_path):
    cache = tmp_path / "cache"
    r1, l1 = _run(tmp_path, "first", cache)
    r2, l2 = _run(tmp_path, "again", cache)
    assert l1.requests == 2 and l2.requests == 0  # la seconda non chiama mai il modello
    a, b = _res(r1.run_dir), _res(r2.run_dir)
    assert (a["replayed"], a["llm_cached_calls"], a["llm_cached_fraction"]) == (False, 0, 0.0)
    assert (b["replayed"], b["llm_cached_calls"], b["llm_cached_fraction"]) == (True, 2, 1.0)
    assert b["passed"] and b["llm_by_role"]["coder"]["cached"] == 2
    assert r2.metrics["replayed"] is True
    # senza cache (llm.cache_dir = None) nessuna run è mai replayed
    r3, l3 = _run(tmp_path, "nocache", None)
    assert l3.requests == 2 and _res(r3.run_dir)["replayed"] is False


def test_partial_cache_is_not_replayed(tmp_path):
    cache = tmp_path / "cache"
    _run(tmp_path, "first", cache, replies=(BAD,), max_calls=1)  # solo la prima in cache
    # prima chiamata dalla cache, la seconda va al modello
    r, llm = _run(tmp_path, "second", cache, replies=(GOOD,))
    res = _res(r.run_dir)
    assert llm.requests == 1 and res["passed"]
    assert (res["replayed"], res["llm_cached_calls"], res["llm_cached_fraction"]) == (False, 1, 0.5)
    assert llm_cache_fields([{"kind": "llm", "cached": True}, {"kind": "llm", "cached": False}]) == \
        {"llm_cached_calls": 1, "llm_cached_fraction": 0.5, "replayed": False}
    assert llm_cache_fields([]) == {"llm_cached_calls": 0, "llm_cached_fraction": None,
                                    "replayed": False}


def test_no_llm_cache_flag_and_campaign_policies(tmp_path):
    from mxagent.config import load_config
    assert load_config(ROOT / "configs" / "default.yaml", {"llm.cache_dir": None}).llm.cache_dir is None
    base = {"name": "t", "tasks": ["b01_e2m1_decode"], "seeds": 1, "models": {"m": {"all": "x"}}}
    shared = Campaign.model_validate(base)
    assert all("llm.cache_dir" not in dict(j.overrides) for j in jobs(shared, tmp_path))
    off = Campaign.model_validate({**base, "llm_cache": "off"})
    assert all(dict(j.overrides)["llm.cache_dir"] is None for j in jobs(off, tmp_path))
    assert "llm.cache_dir=null" in jobs(off, tmp_path)[0].set_args()
    camp = Campaign.model_validate({**base, "llm_cache": "campaign"})
    assert all(dict(j.overrides)["llm.cache_dir"].endswith("/llm_cache") for j in jobs(camp, tmp_path))
    with pytest.raises(Exception):
        Campaign.model_validate({**base, "llm_cache": "sometimes"})
    import yaml
    assert Campaign.model_validate({**base, **yaml.safe_load("llm_cache: off")}).llm_cache == "off"
    nc = load_campaign(ROOT / "configs" / "eval" / "b01_full_nocache.yaml")
    assert nc.llm_cache == "off" and nc.modes == ["full"]
    assert all(dict(j.overrides)["llm.cache_dir"] is None for j in jobs(nc, tmp_path))
    # report_exclude e llm_cache al default non cambiano l'hash delle campagne già eseguite
    ex = Campaign.model_validate({**base, "report_exclude": {"b01_e2m1_decode": "card_leak"}})
    assert ex.digest() == shared.digest() != off.digest()
    with pytest.raises(Exception):
        Campaign.model_validate({**base, "report_exclude": {"b99": "x"}})
    assert load_campaign(ROOT / "configs" / "eval" / "main.yaml").report_exclude == \
        {"b01_e2m1_decode": "card_leak"}


def test_cli_no_llm_cache_sets_cache_dir_none(monkeypatch, tmp_path):
    from mxagent import cli
    seen = {}

    def fake_run_loop(task, cfg, mode, **kw):
        seen["cache_dir"] = cfg.llm.cache_dir
        raise SystemExit(0)

    monkeypatch.setattr("mxagent.run.baselines.run_loop", fake_run_loop)
    with pytest.raises(SystemExit):
        cli.main(["run", "--task", str(ROOT / "benchmarks" / "b01_e2m1_decode"), "--mode",
                  "baseline-b", "--no-llm-cache", "--runs-dir", str(tmp_path)])
    assert seen["cache_dir"] is None


# ------------------------------------------------ report su campagna finta
TASKS = ["b01_e2m1_decode", "b03_e8m0_to_fp32", "b05_mxfp4_dot32"]
MODES = ["baseline-b", "full"]
# (task, mode) -> [(passed, calls_to_pass, replayed)] per seed 0..2
OUT = {
    ("b01_e2m1_decode", "full"): [(True, 3, True), (True, 4, True), (True, 5, True)],
    ("b01_e2m1_decode", "baseline-b"): [(False, None, False)] * 3,
    ("b03_e8m0_to_fp32", "full"): [(True, 9, False), (False, None, False), (True, 20, False)],
    ("b03_e8m0_to_fp32", "baseline-b"): [(True, 6, False), (False, None, False), (False, None, False)],
    ("b05_mxfp4_dot32", "full"): [(True, 23, False), (False, None, False), (False, None, False)],
    ("b05_mxfp4_dot32", "baseline-b"): [(False, None, False)] * 3,
}


def _campaign_dir(tmp: Path, **extra) -> Path:
    c = Campaign.model_validate({"name": "fake", "tasks": TASKS, "modes": MODES, "seeds": 3,
                                 "models": {"m": {"all": "x:1b"}}, "ppa_backends": [], **extra})
    (tmp / "campaign.json").write_text(json.dumps({"campaign": c.model_dump(mode="json")}))
    idx = []
    for j in jobs(c, tmp):
        passed, calls, replayed = OUT[(j.task, j.mode)][j.seed]
        rd = tmp / "runs" / j.id
        rd.mkdir(parents=True)
        (rd / "metrics.json").write_text(json.dumps({
            "task": j.task, "mode": j.mode, "passed": passed, "best_stage": "pass" if passed else "sim",
            "iterations": 1, "model": "x:1b", "llm": {"calls": 60, "tokens": 1}}))
        (rd / "result.json").write_text(json.dumps({
            "task": j.task, "mode": j.mode, "status": "done" if passed else "failed",
            "passed": passed, "llm_calls_to_pass": calls, "replayed": replayed,
            "llm_cached_calls": 16 if replayed else 0, "llm_cached_fraction": 1.0 if replayed else 0.0}))
        idx.append({"job": j.id, "status": "done", "run_dir": f"runs/{j.id}"})
    (tmp / "index.jsonl").write_text("\n".join(json.dumps(e) for e in idx))
    return tmp


def test_replayed_runs_are_excluded_listed_and_warned(tmp_path, capsys):
    S = write_report(_campaign_dir(tmp_path))
    assert "WARNING: 3 run replayed" in capsys.readouterr().err
    assert S["replayed"]["n"] == 3 and S["replayed"]["by_cell"] == {"b01_e2m1_decode / full": 3}
    assert {(r["task"], r["mode"], r["seed"]) for r in S["replayed"]["runs"]} == \
        {("b01_e2m1_decode", "full", s) for s in range(3)}
    tot = S["success"]["m"]
    assert (tot["full"]["total"]["k"], tot["full"]["total"]["n"]) == (3, 6)  # replayed fuori
    assert (tot["baseline-b"]["total"]["k"], tot["baseline-b"]["total"]["n"]) == (1, 9)
    assert S["success"]["m"]["full"]["tasks"]["b01_e2m1_decode"]["n"] == 0
    # appaiamento: le chiavi di b01 perdono la run full replayed e escono per tutti
    cm = S["calls_matched"]["m"]
    assert cm["pairs"] == 6 and cm["by_k"][7] == {"baseline-b": 1, "full": 0}
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "3 run replayed" in md and "## Run replayed" in md
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").count("(replayed)") == 3


def test_report_exclude_moves_b01_to_its_own_section(tmp_path, capsys):
    S = write_report(_campaign_dir(tmp_path, report_exclude={"b01_e2m1_decode": "card_leak"}))
    assert S["tasks_main"] == ["b03_e8m0_to_fp32", "b05_mxfp4_dot32"]
    tot = S["success"]["m"]
    assert (tot["full"]["total"]["k"], tot["full"]["total"]["n"]) == (3, 6)
    assert (tot["baseline-b"]["total"]["k"], tot["baseline-b"]["total"]["n"]) == (1, 6)
    assert "b01_e2m1_decode" not in tot["full"]["tasks"]
    ex = S["excluded"]["b01_e2m1_decode"]
    assert ex["reason"] == "card_leak" and ex["replayed"] == 3
    assert ex["success"]["m"]["full"]["n"] == 0  # tutte replayed
    assert (ex["success_with_replayed"]["m"]["full"]["k"],
            ex["success_with_replayed"]["m"]["full"]["n"]) == (3, 3)
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## b01: card contiene la soluzione" in md
    assert "Esclusi dai totali: `b01_e2m1_decode` (card contiene la soluzione)" in md
    main_table = md.split("## Successo (bit-exact")[1].split("##")[1]
    assert "b01_e2m1_decode" not in main_table
    # controllo: la tabella su tutte le run comprende b01 e le replayed
    allr = S["calls_matched_all"]["m"]
    assert allr["pairs"] == 9 and allr["by_k"][7] == {"baseline-b": 1, "full": 3}
    curve = (tmp_path / "calls_curve.csv").read_text().splitlines()
    assert curve[0] == "scope,models,mode,k,passed,pairs"
    assert any(ln.startswith("all_runs,m,full,7,3,9") for ln in curve)
    assert any(ln.startswith("main,m,full,30,3,6") for ln in curve)  # 9, 20, 23 chiamate
    assert any(ln.startswith("main,m,full,15,1,6") for ln in curve)


def test_report_exclude_from_cli_on_an_old_campaign(tmp_path, capsys):
    d = _campaign_dir(tmp_path)  # campagna senza report_exclude in campaign.json
    from mxagent import cli
    assert cli.main(["report", str(d), "--exclude", "b01_e2m1_decode:card_leak"]) == 0
    S = json.loads((d / "summary.json").read_text())
    assert S["tasks_main"] == ["b03_e8m0_to_fp32", "b05_mxfp4_dot32"]
    assert S["report_exclude"] == {"b01_e2m1_decode": "card_leak"}
    with pytest.raises(SystemExit):
        write_report(d, {"b99_nope": "x"})
