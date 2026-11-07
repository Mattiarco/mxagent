The Chisel module `{top}` below is functionally correct: it passes every test vector bit-exactly. Rewrite it to reduce {goal} while keeping EXACTLY the same behaviour on every input and the same interface.

SPECIFICATION
{spec}

INTERFACE
{interface}

{domain}
CURRENT CODE (verified correct)
```scala
{code}
```

SYNTHESIS RESULT OF THE CURRENT CODE (Yosys, gate level: NAND/NOR/NOT; area = estimated transistors, delay = logic depth in gate levels)
{ppa}

OBJECTIVE: {objective_text}

Hardware techniques that often help in this kind of unit:
- An FP4 E2M1 value has only 8 magnitudes: replace generic multipliers of decoded values with a small lookup table (16x16 products fit in a table, or decode once and use shifts and adds).
- Keep every intermediate signal at the minimum width that holds its exact range; avoid widening to 32 bits early.
- Share logic: one normalisation / leading-zero count instead of several, compute a common exponent once.
- Use balanced reduction trees (reduceTree) rather than linear chains to cut logic depth.
- Prefer comparisons and muxes on small fields over arithmetic on wide values.
{history}
Answer in this format:
IDEA: one or two sentences describing the optimisation.
Then the complete Scala source of `{top}` in a single ```scala code block (the whole module, not a diff).
