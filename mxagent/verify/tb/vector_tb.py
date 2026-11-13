"""Testbench cocotb generico: applica vettori e registra le uscite.

Non confronta nulla: il confronto con il golden model avviene in Python
(mxagent.verify.compare), così il feedback ha tutte le informazioni.

Variabili d'ambiente:
  MXAGENT_VECTORS  JSON {"kind": "comb"|"pipe", "latency": L,
                         "inputs": ["a", ...], "outputs": ["y", ...],
                         "vectors": [[a0, ...], ...]}   (interi, anche > 64 bit)
  MXAGENT_RESULTS  JSON di uscita {"outputs": [[y0, ...] | null, ...]}

Compatibile con cocotb 1.9 e 2.x.
"""
from __future__ import annotations

import json
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


def _read(sig):
    try:
        return int(sig.value)
    except (ValueError, TypeError):  # X/Z
        return None


def _clock(sig, period_ns: int = 10):
    try:
        return Clock(sig, period_ns, unit="ns")  # cocotb 2.x
    except TypeError:
        return Clock(sig, period_ns, units="ns")  # cocotb 1.9


@cocotb.test()
async def run_vectors(dut):
    with open(os.environ["MXAGENT_VECTORS"], encoding="utf-8") as f:
        spec = json.load(f)
    ins = [getattr(dut, n) for n in spec["inputs"]]
    outs = [getattr(dut, n) for n in spec["outputs"]]
    vecs = spec["vectors"]
    lat = int(spec.get("latency", 0))
    results: list = [None] * len(vecs)
    clk = getattr(dut, "clock", None)
    rst = getattr(dut, "reset", None)

    if spec["kind"] == "comb":
        if clk is not None:
            clk.value = 0
        if rst is not None:
            rst.value = 0
        for i, v in enumerate(vecs):
            for s, x in zip(ins, v):
                s.value = x
            await Timer(1, "ns")
            results[i] = [_read(o) for o in outs]
    else:
        cocotb.start_soon(_clock(clk).start())
        rst.value = 1
        for s in ins:
            s.value = 0
        for _ in range(3):
            await RisingEdge(clk)
        rst.value = 0
        # ingressi pilotati sul fronte di discesa del ciclo c; l'uscita relativa
        # all'ingresso j si legge sul fronte di discesa del ciclo j + L, prima di
        # pilotare il nuovo ingresso (vedi docs/verifier.md)
        for c in range(len(vecs) + lat):
            await FallingEdge(clk)
            j = c - lat
            if j >= 0:
                results[j] = [_read(o) for o in outs]
            if c < len(vecs):
                for s, x in zip(ins, vecs[c]):
                    s.value = x

    with open(os.environ["MXAGENT_RESULTS"], "w", encoding="utf-8") as f:
        json.dump({"outputs": results}, f)
