"""M7: benchmark b06-b08, quantizzatore NVFP4, wrapper CV-X-IF.

I test d'integrazione (riferimenti, wrapper con Verilator, mutanti) girano solo
dove c'è la toolchain (Linux)."""
import random
import struct
from fractions import Fraction
from pathlib import Path

import pytest

from mxagent.agents.orchestrator import timing_check
from mxagent.bench import load_task
from mxagent.bench.kernels import get_kernel
from mxagent.formats.nvfp4 import NVFP4_CHOICES, quantize_nvfp4
from mxagent.formats.packing import pack, unpack
from mxagent.wrap import cvxif

ROOT = Path(__file__).resolve().parent.parent
B06 = load_task(ROOT / "benchmarks" / "b06_mxfp4_dot32_pipe")
B07 = load_task(ROOT / "benchmarks" / "b07_nvfp4_dot16")
B08 = load_task(ROOT / "benchmarks" / "b08_mxfp4_mac")


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def val(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# ------------------------------------------------------------- contratti
def test_task_contracts():
    assert B06.timing.kind == "pipe" and B06.timing.latency == 3 and B06.timing.target_mhz == 30
    assert "fixed latency 3" in B06.interface_text()
    assert [p.total_width for p in B07.inputs] == [8, 64, 8, 64]
    assert B08.wrapper == "cvxif" and B08.inputs[0].name == "acc"


# ------------------------------------------------------------------ b07
def _e4m3(c: int) -> Fraction:
    s = -1 if c & 0x80 else 1
    e, m = (c >> 3) & 0xF, c & 7
    return s * (Fraction(m, 8) * Fraction(1, 64) if e == 0 else (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7))


def _e2m1(c: int) -> Fraction:
    s = -1 if c & 8 else 1
    e, m = (c >> 1) & 3, c & 1
    return s * (Fraction(m, 2) if e == 0 else (1 + Fraction(m, 2)) * Fraction(2) ** (e - 1))


def test_b07_result_is_always_exact_in_fp32():
    k, num = get_kernel("nvfp4_dot"), B07.numerics_obj
    rng = random.Random(3)
    vecs = k.corner({}) + [k.random(rng, {}) for _ in range(1500)]
    for v in vecs:
        y = k.fn(v, num, {})["y"]
        if (v["a_scale"] & 0x7F) == 0x7F or (v["b_scale"] & 0x7F) == 0x7F:
            assert y == 0x7FC00000
            continue
        ex = _e4m3(v["a_scale"]) * _e4m3(v["b_scale"]) * sum(
            _e2m1(x) * _e2m1(z) for x, z in zip(unpack(v["a"], 4, 16), unpack(v["b"], 4, 16)))
        assert Fraction(val(y)) == ex and (ex != 0 or y == 0)


def test_b07_known_values():
    k, num = get_kernel("nvfp4_dot"), B07.numerics_obj
    one = pack([2] * 16, 4)
    assert k.fn({"a_scale": 0x38, "a": one, "b_scale": 0x38, "b": one}, num, {})["y"] == f32(16.0)
    assert k.fn({"a_scale": 0xB8, "a": one, "b_scale": 0x38, "b": one}, num, {})["y"] == f32(-16.0)
    assert k.fn({"a_scale": 0x80, "a": one, "b_scale": 0x38, "b": one}, num, {})["y"] == 0  # -0 scala -> +0
    six = pack([7] * 16, 4)
    assert k.fn({"a_scale": 0x7E, "a": six, "b_scale": 0x7E, "b": six}, num, {})["y"] == f32(448.0 * 448 * 576)


# ------------------------------------------------------------------ b08
def _mac(acc: int, ea: list[int], eb: list[int], sa: int = 127, sb: int = 127) -> int:
    k = get_kernel("mx_mac")
    return k.fn({"acc": acc, "a_scale": sa, "a": pack(ea, 4), "b_scale": sb, "b": pack(eb, 4)},
                B08.numerics_obj, {})["y"]


def test_b08_single_rounding():
    one = [2] + [0] * 31  # 1.0
    assert _mac(f32(2.0 ** 24), one, one) == f32(2.0 ** 24)            # pareggio -> pari
    assert _mac(f32(2.0 ** 24 + 2), one, one) == f32(2.0 ** 24 + 4)    # pareggio -> pari
    # dot = 0.25 (non rappresentabile vicino a 2^23 + ...): 2^23 + 0.25 -> 2^23
    q = [1] + [0] * 31
    assert _mac(f32(2.0 ** 23), q, q) == f32(2.0 ** 23)
    # doppio arrotondamento evitato: acc + dot esatto, non acc + round(dot)
    assert _mac(f32(-32.0), [2] * 32, [2] * 32) == 0
    assert _mac(0x80000000, [0] * 32, [2] * 32) == 0                    # -0 + +0 = +0
    assert _mac(0xFF800000, [7] * 32, [7] * 32, 254, 254) == 0xFF800000 # -Inf + finito
    assert _mac(0x7FC00001, one, one) == 0x7FC00000                     # NaN canonico


def test_b08_tiny_negative_sum_keeps_sign():
    # acc = 0, dot = -2^-254 * 0.25: troppo piccolo, arrotonda a -0
    assert _mac(0, [0x9] + [0] * 31, [1] + [0] * 31, 0, 0) == 0x80000000


# ---------------------------------------------------------- NVFP4 quant
def test_nvfp4_quantizer_follows_appendix_b():
    t = quantize_nvfp4([f32(1.0)] * 16)
    assert t.scales == (0x7E,)  # il blocco con amax_x usa la scala massima E4M3 (448)
    assert t.elems == (7,) * 16  # 1.0 * s_enc,b = 6 -> massimo E2M1
    assert all(abs(x - 1) < Fraction(1, 1000) for x in t.dequantize())
    assert len(NVFP4_CHOICES) >= 5


def test_nvfp4_quantizer_properties():
    rng = random.Random(7)
    for _ in range(30):
        xs = [rng.gauss(0, 1) * 10 ** rng.uniform(-3, 3) for _ in range(64)]
        t = quantize_nvfp4([f32(x) for x in xs])
        assert len(t.scales) == 4 and len(t.elems) == 64
        amax = max(abs(val(f32(x))) for x in xs)
        top = max(range(64), key=lambda i: abs(val(f32(xs[i]))))
        assert t.scales[top // 16] == 0x7E and t.elems[top] & 7 == 7
        deq = t.dequantize()
        for x, d in zip(xs, deq):
            # errore entro mezzo passo E2M1 (<= 1 a fondo scala 6) della scala del blocco
            assert abs(Fraction(val(f32(x))) - d) <= Fraction(amax) / 6 * Fraction(1, 1) + Fraction(1, 10 ** 30)
        assert all(0 <= s < 0x7F for s in t.scales)  # scale positive e finite


def test_nvfp4_quantizer_edge_cases():
    z = quantize_nvfp4([0, 0x80000000] * 8)
    assert z.scales == (0,) and z.elems == (0, 8) * 8
    with pytest.raises(ValueError):
        quantize_nvfp4([0x7FC00000] + [0] * 15)
    with pytest.raises(ValueError):
        quantize_nvfp4([0] * 15)
    # un blocco tutto nullo accanto a uno pieno: scala 0, elementi zero con segno
    t = quantize_nvfp4([f32(3.0)] * 16 + [0x80000000] * 16)
    assert t.scales[1] == 0 and t.elems[16:] == (8,) * 16


# ---------------------------------------------------------- CV-X-IF (logica)
def test_cvxif_encoding_and_program():
    assert cvxif.r_type(3, 0, 5, 0, 7) == (3 << 25) | (5 << 15) | (7 << 7) | 0b0001011
    p = cvxif.program(B08, seed=1, n_random=5)
    ins = p["instrs"]
    assert [e["id"] for e in ins[:20]] == [i % 16 for i in range(20)]
    macs = [e for e in ins if e["op"] == cvxif.MAC and e["legal"] and not e["kill"]]
    assert all(e["expect"] is not None for e in macs)
    assert any(e["kill"] for e in ins) and any(not e["legal"] for e in ins)


def _perfect_obs(prog):
    ins = prog["instrs"]
    issues = [{"n": i, "id": e["id"], "cycle": i, "accept": int(e["legal"]),
               "writeback": int(e["legal"] and e["op"] == cvxif.MAC),
               "register_read": (1 if e["op"] == cvxif.MAC else 3) if e["legal"] else 0}
              for i, e in enumerate(ins)]
    results = [{"cycle": 0, "id": e["id"], "data": e["expect"] if e["op"] == cvxif.MAC else 0,
                "rd": e["rd"], "we": int(e["op"] == cvxif.MAC), "hartid": 0}
               for e in ins if e["legal"] and not e["kill"]]
    return {"issues": issues, "results": results, "cycles": 1, "timeout": False}


def test_cvxif_check_detects_protocol_violations():
    prog = cvxif.program(B08, seed=2, n_random=3)
    obs = _perfect_obs(prog)
    assert cvxif.check(prog, obs) == []
    killed = next(e for e in prog["instrs"] if e["kill"])
    bad = dict(obs, results=obs["results"] + [{"cycle": 0, "id": killed["id"], "data": 0, "rd": 0,
                                               "we": 0, "hartid": 0}])
    assert any("transazioni di result" in e for e in cvxif.check(prog, bad))
    bad = _perfect_obs(prog)
    i = next(n for n, r in enumerate(bad["results"]) if r["we"])
    bad["results"][i]["data"] ^= 1
    assert any("data" in e for e in cvxif.check(prog, bad))
    bad = _perfect_obs(prog)
    rej = next(o for o in bad["issues"] if not prog["instrs"][o["n"]]["legal"])
    rej["accept"] = 1
    assert any("accept" in e for e in cvxif.check(prog, bad))


def test_timing_check():
    assert timing_check({"fmax_mhz": 37.0, "fmax_min": 30.4}, 30)["met"] is True
    assert timing_check({"fmax_mhz": 37.0, "fmax_min": 29.9}, 30)["met"] is False
    assert timing_check(None, 30)["met"] is None


# ------------------------------------------------------ integrazione (WSL)
def _toolchain():
    from mxagent.verify.sim import runner_available
    from mxagent.util.shell import which
    return runner_available() is None and which("sbt") is not None


needs_tools = pytest.mark.skipif(not _toolchain(), reason="serve sbt + Verilator + cocotb (Linux)")


@needs_tools
def test_m7_references_and_wrapper(tmp_path):
    from mxagent.bench.task import reference_sources
    from mxagent.verify.verifier import Verifier
    reps = {}
    for t in (B06, B07, B08):
        with Verifier(t, tmp_path / t.name, sbt_mode="batch", random_vectors=300) as v:
            reps[t.name] = v.verify({p.name: p.read_text() for p in reference_sources(t)}, "ref")
        assert reps[t.name].passed, reps[t.name].feedback
    r = cvxif.test_wrapper(B08, reps[B08.name].verilog, tmp_path / "wrap", n_random=20)
    assert r.passed, r.errors
    for m in sorted((ROOT / "tests" / "fixtures" / "cvxif_mutants").glob("*.scala")):
        rm = cvxif.test_wrapper(B08, reps[B08.name].verilog, tmp_path / m.stem, n_random=20,
                                wrapper_src=m)
        assert not rm.passed and rm.stage == "check", m.stem


@needs_tools
def test_b06_wrong_latency_is_caught(tmp_path):
    from mxagent.bench.task import reference_sources
    from mxagent.verify.verifier import Verifier
    src = reference_sources(B06)[0].read_text()
    mutant = src.replace("io.y := r(res)", "io.y := res")  # latenza 2 invece di 3
    assert mutant != src
    with Verifier(B06, tmp_path, sbt_mode="batch", random_vectors=200) as v:
        rep = v.verify({"MXFP4Dot32Pipe.scala": mutant}, "lat2")
    assert rep.stage == "sim" and not rep.passed
