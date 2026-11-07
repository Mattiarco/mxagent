"""Test del framework: config, CLI golden, doctor (senza dipendere dai tool installati)."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from mxagent import cli, doctor
from mxagent.config import load_config
from mxagent.formats.numerics import Accumulation

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "configs" / "default.yaml"


def test_default_config_loads():
    cfg = load_config(DEFAULT)
    assert cfg.llm.roles["coder"].num_ctx >= 8192
    assert cfg.numerics_obj.accumulation is Accumulation.EXACT
    assert "qwen2.5-coder:32b" in cfg.llm.models()


def test_config_overrides_and_validation(tmp_path):
    cfg = load_config(DEFAULT, {"numerics.accumulation": "per_block", "budget.best_of_n": 5})
    assert cfg.numerics_obj.accumulation is Accumulation.PER_BLOCK and cfg.budget.best_of_n == 5
    with pytest.raises(ValidationError):
        load_config(DEFAULT, {"numerics.accumulation": "bogus"})
    with pytest.raises(ValidationError):
        load_config(DEFAULT, {"budget.typo": 1})
    bad = tmp_path / "c.yaml"
    bad.write_text("llm: {roles: {coder: {model: x}}}\n")
    with pytest.raises(ValidationError, match="ruoli LLM mancanti"):
        load_config(bad)


def test_config_dump_roundtrip(tmp_path):
    cfg = load_config(DEFAULT)
    p = tmp_path / "out.yaml"
    p.write_text(cfg.dump(), encoding="utf-8")
    assert load_config(p) == cfg


def test_cli_golden(capsys):
    assert cli.main(["golden", "dot"]) == 0
    assert "0x42000000" in capsys.readouterr().out  # 32 x (1.0 * 1.0)
    assert cli.main(["golden", "dot", "--a-scale", "0xFF"]) == 0
    assert "0x7FC00000" in capsys.readouterr().out
    assert cli.main(["golden", "quant", "--x", "7.0"]) == 0
    assert "scale = 0x7F" in capsys.readouterr().out
    # meno di k valori: completati con zeri. amax 7 -> X = 1; 7 satura a 6, -2.5 -> -2
    assert cli.main(["golden", "quant", "--x", "1.0,-2.5,7.0,0.25,0.75,5.0"]) == 0
    assert "scale = 0x7F  elems = 2 C 7 0 2 6 0" in capsys.readouterr().out
    assert cli.main(["golden", "card"]) == 0
    out = capsys.readouterr().out
    assert "E2M1" in out and "SInt(9)" in out and "0x7FC00000" in out
    assert cli.main(["golden", "dot", "--format", "nvfp4", "--a-scale", "0x38",
                     "--b-scale", "0x38"]) == 0
    assert "0x41800000" in capsys.readouterr().out  # 16.0


def test_cli_all_commands_registered(capsys):
    import pytest
    for cmd in ("doctor", "golden", "verify", "run", "synth", "eval", "report"):
        with pytest.raises(SystemExit) as e:
            cli.main([cmd, "--help"])
        assert e.value.code == 0


def test_firtool_check_logic(tmp_path):
    assert doctor.check_firtool("x86_64", {}) is None
    c = doctor.check_firtool("aarch64", {})
    assert c is not None and not c.ok
    c = doctor.check_firtool("x86_64", {"CHISEL_FIRTOOL_PATH": str(tmp_path)})
    assert c is not None and not c.ok and "non esiste" in c.detail


def test_ollama_unreachable_is_reported():
    checks = doctor.check_ollama("http://127.0.0.1:9", ["m"])
    assert len(checks) == 1 and not checks[0].ok and checks[0].required


def test_format_checks():
    txt = doctor.format_checks([doctor.Check("a", True, "x"),
                                doctor.Check("b", False, "y", "fai z"),
                                doctor.Check("c", False, "w", required=False)])
    assert "mancano componenti obbligatori: b" in txt and "-> fai z" in txt
