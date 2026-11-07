Plan the hardware implementation of `{top}`.

SPECIFICATION
{spec}

INTERFACE
{interface}

STRUCTURED SPEC
{structured}

AVAILABLE VERIFIED BUILDING BLOCKS (each one will be generated and verified on its own before integration):
{catalog}

Decide:
- "monolithic" if the design is small enough to write directly in one module (a few lines of logic, e.g. a lookup or a single conversion);
- "decompose" otherwise: choose the building blocks from the list above that the top module will instantiate (each block once, even if the top instantiates it many times), give each a CamelCase module name and its parameters (only the parameters listed for that block), and describe in "glue" how the top connects them and which small logic stays in the top.

Return JSON: {"strategy": "monolithic" | "decompose", "blocks": [{"key": block key, "module": ModuleName, "params": {...}, "purpose": short text}], "glue": text}
