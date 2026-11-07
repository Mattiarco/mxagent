Design a purely combinational converter from an E8M0 shared scale to IEEE-754
binary32.

- Input `x` (8 bits) is an E8M0 code: value 2^(x-127) for x in 0..254; x = 0xFF is NaN.
- Output `y` (32 bits) is the exact binary32 bit pattern of 2^(x-127).
  - For x = 0xFF output the canonical quiet NaN 0x7FC00000.
  - For x = 0 the value 2^-127 is a binary32 SUBNORMAL: output 0x00400000
    (subnormals are kept, never flushed to zero).
  - Every other x is a normal number with zero mantissa.
