"""Wrapper registrato per misurare la frequenza massima di un modulo.

I benchmark sono (per ora) combinatori e hanno centinaia di bit di I/O, più dei
pin di un FPGA. Il wrapper:

* carica gli ingressi da una catena di shift register alimentata da un solo pin;
* registra tutte le uscite e ne fa lo XOR verso un pin.

Il cammino misurato è quindi registro -> DUT -> registro, cioè esattamente il
ritardo combinatorio del modulo (più setup/clock-to-q). Nessuna uscita è
inosservata, quindi la sintesi non può eliminare logica del DUT. I conteggi di
risorse si prendono dalla sintesi del DUT da solo, non del wrapper.
"""
from __future__ import annotations

from ..bench.task import Task

WRAP_TOP = "mxagent_ppa_wrap"


def wrapper(task: Task) -> str:
    ins = [(p.name, p.total_width) for p in task.inputs]
    outs = [(p.name, p.total_width) for p in task.outputs]
    w_in, w_out = sum(w for _, w in ins), sum(w for _, w in outs)
    lines = [f"// generato da mxagent.synth.wrapper per {task.top} ({task.name})",
             f"module {WRAP_TOP}(input clk, input din, output dout);",
             f"  reg [{w_in - 1}:0] sh;"]
    lines.append("  always @(posedge clk) sh <= " +
                 (f"{{sh[{w_in - 2}:0], din}};" if w_in > 1 else "din;"))
    conns, lo = [], 0
    for name, w in ins:
        conns.append(f".{name}(sh[{lo + w - 1}:{lo}])")
        lo += w
    for name, w in outs:
        lines.append(f"  wire [{w - 1}:0] o_{name};")
        conns.append(f".{name}(o_{name})")
    if task.timing.kind == "pipe":  # Module Chisel: clock e reset impliciti
        conns = [".clock(clk)", ".reset(1'b0)"] + conns
    lines.append(f"  {task.top} dut({', '.join(conns)});")
    cat = ", ".join(f"o_{n}" for n, _ in reversed(outs))
    lines += [f"  reg [{w_out - 1}:0] q;",
              f"  always @(posedge clk) q <= {{{cat}}};",
              "  assign dout = ^q;",
              "endmodule", ""]
    return "\n".join(lines)
