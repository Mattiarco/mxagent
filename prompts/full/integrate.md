Write the top-level Chisel module `{top}` by instantiating the already VERIFIED submodules listed below.

SPECIFICATION
{spec}

INTERFACE OF {top} (mandatory)
{interface}

{domain}
VERIFIED SUBMODULES (already written and tested: do NOT redefine them, just instantiate them with Module(new Name) and connect their io fields; they all use FlatIO and are purely combinational RawModules):
{submodules}

PLAN FOR THE GLUE LOGIC
{glue}

Write ONLY the class `{top}`. Start with `import chisel3._` and `import chisel3.util._`.
Return the complete Scala source in a single ```scala code block.
