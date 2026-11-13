"""Profili di modello (--model-profile): risoluzione, etichetta della run, mismatch."""
import json

import pytest

from mxagent.agents.orchestrator import Orchestrator
from mxagent.config import LLM_ROLES, effective_models, load_config, profile_label
from mxagent.eval.campaign import ModelSet
from mxagent.llm.chat import ScriptedChat
from mxagent.run.baselines import run_loop
from tests.test_run_result import B01, ROOT, Verifier, cfg, code

DEFAULT = ROOT / "configs" / "default.yaml"
Q14, Q32 = "qwen2.5-coder:14b", "qwen2.5-coder:32b"


@pytest.mark.parametrize("name,model", [("all14b", Q14), ("all32b", Q32)])
def test_uniform_profiles_set_every_llm_role(name, model):
    c = load_config(DEFAULT, profile=name)
    assert {r: c.llm.roles[r].model for r in LLM_ROLES} == dict.fromkeys(LLM_ROLES, model)
    assert c.llm.roles["embed"].model == "nomic-embed-text"
    assert c.llm.roles["reflector"].temperature == 0.2  # resto del ruolo invariato
    assert profile_label(c) == (name, {})
    assert effective_models(c)["embed"] == "nomic-embed-text"


def test_mixed_profile_and_default_config_are_labelled_mixed():
    mixed = load_config(DEFAULT, profile="mixed")
    assert mixed.llm.roles["coder"].model == Q14 and mixed.llm.roles["reflector"].model == Q32
    # la config di default senza profilo coincide con "mixed": il nome della run lo dice
    assert profile_label(load_config(DEFAULT)) == ("mixed", {})


def test_profile_from_overrides_and_set_wins_with_mismatch():
    c = load_config(DEFAULT, {"llm.profile": "all14b", "llm.roles.reflector.model": Q32})
    assert c.llm.profile == "all14b" and c.llm.roles["reflector"].model == Q32
    label, mism = profile_label(c)
    assert label == "all14b" and mism == {"reflector": {"expected": Q14, "actual": Q32}}


def test_routing_off_uses_coder_for_every_role():
    c = load_config(DEFAULT, {"agent.routing": False})
    assert set(effective_models(c).values()) == {Q14, "nomic-embed-text"}
    assert profile_label(c) == ("all14b", {})
    label, mism = profile_label(load_config(DEFAULT, {"agent.routing": False}, profile="mixed"))
    assert label == "mixed" and set(mism) == {"planner", "reflector", "escalate", "optimizer"}


def test_unknown_profile_and_custom_labels():
    with pytest.raises(ValueError, match="sconosciuto"):
        load_config(DEFAULT, profile="all70b")
    one = load_config(DEFAULT, {f"llm.roles.{r}.model": "qwen2.5-coder:7b" for r in LLM_ROLES})
    assert profile_label(one) == ("all-qwen2.5-coder:7b", {})
    assert profile_label(load_config(DEFAULT, {"llm.roles.planner.model": "x:1b"}))[0] == "custom"
    # baseline: un solo ruolo (coder)
    assert profile_label(load_config(DEFAULT, profile="all32b"), "baseline-b") == ("all32b", {})


def test_run_dir_meta_and_result_carry_the_profile(tmp_path):
    o = Orchestrator(B01, cfg(**{"llm.profile": "all14b"}), seed=0, runs_dir=tmp_path,
                     llm=ScriptedChat([code("// ok")]), verifier_factory=lambda t, n: Verifier(t))
    o.run()
    root = o.rec.root
    assert root.name.endswith("_b01_e2m1_decode_full_all14b_s0")
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    res = json.loads((root / "result.json").read_text(encoding="utf-8"))
    expect = {**dict.fromkeys(LLM_ROLES, Q14), "embed": "nomic-embed-text"}
    assert meta["model_profile"] == res["model_profile"] == "all14b"
    assert meta["models_by_role"] == res["models_by_role"] == expect
    trace = json.loads((root / "trace.json").read_text(encoding="utf-8"))
    assert "model_profile_mismatch" not in [e["kind"] for e in trace]


def test_mismatch_is_warned_and_traced(tmp_path, capsys):
    c = cfg(**{"llm.profile": "all14b", "llm.roles.reflector.model": Q32})
    o = Orchestrator(B01, c, seed=0, runs_dir=tmp_path, llm=ScriptedChat([code("// ok")]),
                     verifier_factory=lambda t, n: Verifier(t))
    o.run()
    trace = json.loads((o.rec.root / "trace.json").read_text(encoding="utf-8"))
    ev = [e for e in trace if e["kind"] == "model_profile_mismatch"]
    assert ev and ev[0]["mismatches"]["reflector"]["actual"] == Q32
    assert "WARNING" in capsys.readouterr().err


def test_baseline_run_is_labelled_by_profile(tmp_path):
    c = cfg(**{"llm.profile": "all32b"})
    r = run_loop(B01, c, "baseline-a", seed=0, role=c.llm.roles["coder"], runs_dir=tmp_path,
                 llm=ScriptedChat([code("// ok")]), verify_fn=Verifier(B01).verify)
    assert r.run_dir.name.endswith("_baseline-a_all32b_s0")
    res = json.loads((r.run_dir / "result.json").read_text(encoding="utf-8"))
    assert res["model_profile"] == "all32b" and res["models_by_role"] == {"coder": Q32}


def test_campaign_model_set_profile():
    ms = ModelSet(profile="all14b")
    assert ms.overrides("full") == {"llm.profile": "all14b"}
    assert ms.overrides("baseline-a") == {"llm.profile": "all14b"}
    c = load_config(DEFAULT, ms.overrides("baseline-b"))
    assert c.llm.roles["coder"].model == Q14
