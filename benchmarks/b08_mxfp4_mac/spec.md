Design a purely combinational multiply-accumulate unit for MXFP4 (OCP MX v1.0 section 6.2,
DotGeneral with one block pair): y = acc + Dot(A, B), with a SINGLE rounding to IEEE-754
binary32. After it passes, the unit is wrapped as a RISC-V coprocessor (CORE-V eXtension
Interface); the wrapper is generated automatically and is not part of this task.

- `acc` is a binary32 bit pattern (any value: zeros, subnormals, normals, infinities, NaN).
- Block A = shared scale `a_scale` (E8M0) and 32 FP4 E2M1 elements packed in `a`;
  block B likewise. Element i occupies bits [4*i+3 : 4*i].
- Dot(A, B) = 2^(a_scale-127) * 2^(b_scale-127) * sum_i (A_i * B_i), where the sum is
  EXACT (multiple of 0.25, fits a signed 14-bit integer in units of 1/4). Dot(A, B) can be
  far outside the binary32 range (scales up to 2^127 each, down to 2^-127 each).
- y = round_RNE(acc + Dot(A, B)): the exact sum of acc and the exact dot product is rounded
  ONCE (no intermediate rounding of the dot product). Subnormal results are kept (never
  flushed); results that round to 2^128 or more give +Inf / -Inf.
- An exactly zero result is +0 (0x00000000) (round-to-nearest: x + (-x) = +0, -0 + +0 = +0).
  A NONZERO exact sum too small to be represented rounds to a zero that keeps its sign
  (a tiny negative sum gives -0, 0x80000000).
- NaN: if a_scale or b_scale is 0xFF, or acc is NaN, y is the canonical NaN 0x7FC00000.
  If acc is +/-Inf (and no NaN), y = acc (the dot product is always finite).
