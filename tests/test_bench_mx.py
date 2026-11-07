"""Kernel b04/b05 e catalogo dei blocchi: coerenza fra golden e contratti delle porte."""
import random
from pathlib import Path

import pytest

from mxagent.bench import load_task
from mxagent.bench.subkernels import CATALOG, block_task, catalog_text
from mxagent.formats.packing import pack, to_unsigned
from mxagent.verify.verifier import Verifier

ROOT = Path(__file__).resolve().parent.parent


def _check_task(t, n=300):
    v = Verifier(t, Path("unused"), random_vectors=n)
    vecs = v.vectors()
    assert vecs
    for _, x in vecs[:n]:
        assert set(x) == {p.name for p in t.inputs}
        for p in t.inputs:
            assert 0 <= x[p.name] < (1 << p.total_width), p.name
        y = v.golden(x)
        assert set(y) == {p.name for p in t.outputs}
        for p in t.outputs:
            assert 0 <= y[p.name] < (1 << p.total_width), p.name
    # deterministico dal seed
    assert [x for _, x in Verifier(t, Path("u"), random_vectors=n).vectors()][:50] == \
           [x for _, x in vecs][:50]
    return vecs


@pytest.mark.parametrize("name", ["b04_mxfp4_quantize", "b05_mxfp4_dot32"])
def test_block_benchmarks(name):
    t = load_task(ROOT / "benchmarks" / name)
    vecs = _check_task(t)
    suites = {s for s, _ in vecs}
    assert suites == {"corner", "random"}


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_catalog_blocks(key):
    t = block_task(key, "Blk")
    _check_task(t)
    assert key in catalog_text()


def test_dot_kernel_spot_values():
    t = load_task(ROOT / "benchmarks" / "b05_mxfp4_dot32")
    v = Verifier(t, Path("u"))
    x = {"a_scale": 127, "a": pack([2] * 32, 4), "b_scale": 127, "b": pack([2] * 32, 4)}
    assert v.golden(x) == {"y": 0x42000000}


def test_fixed_to_fp32_block_matches_dot():
    """Il blocco di conversione composto con somma e scale deve dare il dot del task."""
    rng = random.Random(3)
    dot = Verifier(load_task(ROOT / "benchmarks" / "b05_mxfp4_dot32"), Path("u"))
    conv = Verifier(block_task("fixed_to_fp32", "C"), Path("u"))
    from mxagent.formats.golden import e2m1_mul_fixed
    for _ in range(500):
        x = dot.kernel.random(rng, {})
        ea = [(x["a"] >> 4 * i) & 15 for i in range(32)]
        eb = [(x["b"] >> 4 * i) & 15 for i in range(32)]
        s = sum(e2m1_mul_fixed(a, b) for a, b in zip(ea, eb))
        nan = int(x["a_scale"] == 255 or x["b_scale"] == 255)
        e = 0 if nan else x["a_scale"] + x["b_scale"] - 254
        y = conv.golden({"x": to_unsigned(s, 14), "exp": to_unsigned(e, 10), "nan": nan})
        assert y == dot.golden(x)
