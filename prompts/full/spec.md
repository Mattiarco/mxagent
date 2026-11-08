Extract a structured specification from this hardware request.

REQUEST
{spec}

INTERFACE
{interface}

Return JSON with these fields:
- operation: one of "decode", "multiply", "scale_convert", "quantize", "dot_product", "mac", "other"
- element_format: e.g. "e2m1"; scale_format: e.g. "e8m0", "e4m3" or "none"
- block_size: integer (1 if not a block operation)
- output_format: e.g. "fp32", "fixed", "e2m1+e8m0"
- accumulation: "exact" or "rounded" or "none"
- latency_cycles: integer (0 = combinational); combinational: true or false
- ports: one entry per port of the INTERFACE: name, direction ("in"/"out"), type (e.g.
  "e2m1", "sint", "fp32"), width (total bits), elems (1 if not a vector), meaning
  (what the value represents, including its unit or scale)
- output_scale: how the output encodes the value, e.g. "y = value * 2" (empty if not scaled)
- corner_cases: list of short strings (the special cases the design must handle)
