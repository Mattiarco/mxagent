"""Formati MX/NVFP4: costanti verificate, aritmetica esatta e golden model."""
from .fp import FloatFormat, Kind, Overflow, Rounding, Subnormals, Value, decode, encode
from .golden import (MXBlock, e2m1_mul_fixed, e2m1_to_fixed, e8m0_to_fp32, mx_dot,
                     mx_elemwise_mul, mx_mac, nvfp4_dot, quantize_mx)
from .numerics import DEFAULT, Accumulation, Numerics
from .specs import BF16, E2M1, E4M3, FP32, MXFP4, NVFP4, BlockFormat

__all__ = [
    "FloatFormat", "Kind", "Overflow", "Rounding", "Subnormals", "Value", "decode", "encode",
    "MXBlock", "e2m1_mul_fixed", "e2m1_to_fixed", "e8m0_to_fp32", "mx_dot", "mx_elemwise_mul",
    "mx_mac", "nvfp4_dot", "quantize_mx", "DEFAULT", "Accumulation", "Numerics",
    "BF16", "E2M1", "E4M3", "FP32", "MXFP4", "NVFP4", "BlockFormat",
]
