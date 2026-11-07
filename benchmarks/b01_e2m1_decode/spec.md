Design a purely combinational decoder that converts one FP4 E2M1 element into an
exact signed fixed-point number.

- Input `a` (4 bits) is an E2M1 code: bit 3 is the sign, bits 2..1 the exponent
  (bias 1), bit 0 the mantissa.
- Output `y` (SInt, 5 bits) is the value of `a` multiplied by 2, so that every E2M1
  value becomes an integer: 0.5 -> 1, 1.0 -> 2, 1.5 -> 3, 6.0 -> 12, -6.0 -> -12.
- Both zero codes (0b0000 = +0 and 0b1000 = -0) must produce 0.
