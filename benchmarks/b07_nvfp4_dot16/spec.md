Design a purely combinational unit that computes the dot product of two NVFP4 micro-blocks
(NVIDIA NVFP4: 16 FP4 E2M1 elements with one FP8 E4M3 block scale) and returns it as an
IEEE-754 binary32 number. The per-tensor FP32 scales of NVFP4 are applied elsewhere (after
the GEMM) and are not part of this unit.

- Block A = scale `a_scale` (FP8 E4M3 code) and 16 E2M1 elements packed in `a`; block B
  likewise. Element i occupies bits [4*i+3 : 4*i].
- E4M3 (OCP FP8): 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits. Exponent field 0
  encodes subnormals (value = (-1)^s * m/8 * 2^-6), otherwise (-1)^s * (1 + m/8) * 2^(e-7).
  There is no infinity; S.1111.111 (0x7F and 0xFF) is NaN; the maximum magnitude is 448
  (0x7E). The scale is used with its sign and both zeros (0x00, 0x80) are valid.
- Result: y = a_scale * b_scale * sum_i (A_i * B_i), computed exactly and rounded once to
  binary32 (round-to-nearest-even). Note: with these ranges the exact result always fits
  binary32 exactly (at most 20 significant bits, magnitude between 2^-20 and 448*448*576),
  so no rounding, overflow or subnormal output can actually occur.
- An exactly zero result is +0 (0x00000000), also from negative zeros, zero scales or
  cancellation.
- If a_scale or b_scale is NaN the output is the canonical NaN 0x7FC00000.
