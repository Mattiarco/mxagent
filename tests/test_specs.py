"""Costanti contro le tabelle della spec OCP MX v1.0 (valori ricopiati dal PDF)."""
import pytest

from mxagent.formats import fp
from mxagent.formats.specs import (E2M1, E2M3, E3M2, E4M3, E5M2, MXFP4, NVFP4, e8m0_decode,
                                   E2M1_MAGNITUDES)

F = fp.decode


def val(fmt, code):
    return F(fmt, code).to_float()


def test_e2m1_table5():
    # §5.3.3 Tab. 5
    assert E2M1.bias == 1 and not E2M1.has_inf and not E2M1.has_nan
    assert val(E2M1, 0b0000) == 0.0 and val(E2M1, 0b1000) == 0.0
    assert str(val(E2M1, 0b1000)) == "-0.0"
    assert val(E2M1, 0b0111) == 6.0 and val(E2M1, 0b1111) == -6.0  # max normal S 11 1
    assert val(E2M1, 0b0010) == 1.0  # min normal S 01 0
    assert val(E2M1, 0b0001) == 0.5  # max = min subnormal S 00 1
    assert E2M1_MAGNITUDES == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    assert E2M1.max_finite.to_float() == 6.0


def test_fp6_table4():
    assert val(E2M3, 0b011111) == 7.5 and val(E2M3, 0b001000) == 1.0
    assert val(E2M3, 0b000111) == 0.875 and val(E2M3, 0b000001) == 0.125
    assert val(E3M2, 0b011111) == 28.0 and val(E3M2, 0b000100) == 0.25
    assert val(E3M2, 0b000011) == 0.1875 and val(E3M2, 0b000001) == 0.0625


def test_e4m3_table2():
    assert E4M3.bias == 7 and not E4M3.has_inf
    assert F(E4M3, 0b0_1111_111).is_nan and F(E4M3, 0b1_1111_111).is_nan
    assert val(E4M3, 0b0_1111_110) == 448.0
    assert val(E4M3, 0b0_0001_000) == 2.0 ** -6
    assert val(E4M3, 0b0_0000_111) == 2.0 ** -6 * 0.875
    assert val(E4M3, 0b0_0000_001) == 2.0 ** -9
    assert val(E4M3, 0b0_1111_000) == 256.0  # esponente 1111 è normale in E4M3
    nans = [c for c in range(256) if F(E4M3, c).is_nan]
    assert nans == [0x7F, 0xFF]


def test_e5m2_table2():
    assert F(E5M2, 0b0_11111_00).kind is fp.Kind.INF
    assert all(F(E5M2, 0b0_11111_00 | m).is_nan for m in (1, 2, 3))
    assert val(E5M2, 0b0_11110_11) == 57344.0
    assert val(E5M2, 0b0_00000_01) == 2.0 ** -16


def test_e8m0_table7():
    assert e8m0_decode(0xFF).is_nan
    assert e8m0_decode(0).to_float() == 2.0 ** -127
    assert e8m0_decode(127).to_float() == 1.0
    assert e8m0_decode(254).to_float() == 2.0 ** 127
    assert all(not e8m0_decode(c).is_nan and e8m0_decode(c).kind is fp.Kind.FINITE
               for c in range(255))  # nessuno zero


def test_block_formats():
    assert (MXFP4.k, MXFP4.scale, MXFP4.elem) == (32, "e8m0", E2M1)  # Tab. 1
    assert MXFP4.elem_emax == 2  # 4 = 2^2, §6.3
    assert (NVFP4.k, NVFP4.scale, NVFP4.tensor_scale) == (16, "e4m3", True)


@pytest.mark.parametrize("code", range(16))
def test_e2m1_roundtrip(code):
    assert fp.encode(E2M1, F(E2M1, code)) == code
