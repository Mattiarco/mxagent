Design a purely combinational quantizer that converts a vector of 32 IEEE-754 binary32
values into one MXFP4 block (OCP MX v1.0 section 6.3).

- Input `x`: 32 binary32 values, element i at bits [32*i+31 : 32*i].
- Shared exponent: shared = floor(log2(max_i |x_i|)) - 2, computed over the nonzero
  finite inputs (2 is the exponent of 4.0, the largest power of two in E2M1).
  Binary32 subnormal inputs are valid values (their floor(log2) depends on the leading
  one of the mantissa). Clamp shared to [-127, 127]. Output `scale` = shared + 127.
- If all inputs are zero, shared = -127 (scale = 0x00) and every element is a zero
  with the sign of its input.
- If ANY input is NaN or +/-Inf: scale = 0xFF and all elements are 0.
- Element i: `q` code i (bits [4*i+3 : 4*i]) = E2M1 encoding of x_i / 2^shared, rounded
  to nearest with ties to even (even mantissa bit) and saturated to +/-6.0. The sign is
  always preserved, also when the magnitude rounds to zero (a tiny negative input gives
  0b1000).
