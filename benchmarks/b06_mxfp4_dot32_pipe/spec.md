Design a PIPELINED unit that computes the dot product of two MXFP4 blocks (OCP MX v1.0
section 6.1) and returns it as an IEEE-754 binary32 number. It must reach the target
clock frequency on the FPGA flow (Lattice ECP5, nextpnr): split the logic into balanced
pipeline stages.

- Timing: Chisel `Module` (implicit clock and reset). The inputs sampled at a rising clock
  edge produce their result on `y` exactly 3 clock cycles later; a new input is accepted
  every cycle (fully pipelined, no stalls, no valid signals). Reset does not need to clear
  the pipeline.
- Block A = shared scale `a_scale` (E8M0) and 32 FP4 E2M1 elements packed in `a`;
  block B likewise (`b_scale`, `b`). Element i occupies bits [4*i+3 : 4*i].
- Result: Dot(A, B) = 2^(a_scale-127) * 2^(b_scale-127) * sum_i (A_i * B_i).
- The sum of the 32 element products must be computed EXACTLY (each product is a
  multiple of 0.25, the sum fits a signed 14-bit integer in units of 1/4), then the
  scaled result is rounded ONCE to binary32 with round-to-nearest-even.
- Results smaller than 2^-126 must be produced as correctly rounded binary32 subnormals
  (never flushed to zero). Results that round to 2^128 or more give +Inf / -Inf.
- An exactly zero result is +0 (0x00000000), also when it comes from negative zeros or
  from cancellation.
  A NONZERO result too small to be represented rounds to a zero that keeps its sign
  (a tiny negative result gives -0, 0x80000000).
- If a_scale or b_scale is 0xFF (NaN) the output is the canonical NaN 0x7FC00000.
