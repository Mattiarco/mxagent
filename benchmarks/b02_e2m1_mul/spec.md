Design a purely combinational multiplier of two FP4 E2M1 elements that produces the
EXACT product as a signed fixed-point number (no rounding).

- Inputs `a` and `b` (4 bits each) are E2M1 codes.
- Output `p` (SInt, 9 bits) is a*b multiplied by 4. Every product of two E2M1
  values is a multiple of 0.25 in [-36, 36], so p is an integer in [-144, 144].
  Examples: 0.5*0.5 -> 1, 1.5*1.5 -> 9, 6*6 -> 144, -6*6 -> -144.
- Any product with a zero operand (+0 or -0) must be 0.
