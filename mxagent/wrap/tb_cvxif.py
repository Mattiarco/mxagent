"""Testbench cocotb del coprocessore CV-X-IF: fa la parte della CPU.

Esegue un programma (MXAGENT_PROGRAM, JSON preparato da ``mxagent.wrap.cvxif``) e
registra ciò che osserva su issue e result (MXAGENT_RESULTS); il confronto con
le attese avviene in Python. Regole del protocollo rispettate dalla CPU simulata
(CV-X-IF v1.0.0):

* issue in ordine di programma, id incrementali (wrap-around); con
  X_ISSUE_REGISTER_SPLIT = 0 register_valid = issue_valid e stessi id/hartid;
  rs_valid può salire durante la transazione;
* ogni istruzione (accettata o rifiutata) riceve un commit, al più presto nel
  ciclo in cui inizia la sua transazione di issue (anche prima dell'handshake);
  commit in ordine, al più uno per ciclo;
* un kill riguarda l'istruzione e le più recenti: dopo un kill la CPU non avvia
  nuovi issue finché il kill non è stato inviato (flush);
* result_ready con backpressure casuale.
"""
from __future__ import annotations

import json
import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


def _clock(sig, period_ns: int = 10):
    try:
        return Clock(sig, period_ns, unit="ns")  # cocotb 2.x
    except TypeError:
        return Clock(sig, period_ns, units="ns")  # cocotb 1.9


def _i(sig) -> int | None:
    try:
        return int(sig.value)
    except (ValueError, TypeError):
        return None


@cocotb.test()
async def run_program(dut):
    prog = json.loads(open(os.environ["MXAGENT_PROGRAM"], encoding="utf-8").read())
    instrs = prog["instrs"]
    rng = random.Random(prog["seed"])
    idw = prog["id_width"]
    clk = dut.clock
    cocotb.start_soon(_clock(clk).start())

    for name in ("compressed_valid", "compressed_req_instr", "compressed_req_hartid",
                 "issue_valid", "issue_req_instr", "issue_req_hartid", "issue_req_id",
                 "register_valid", "register_hartid", "register_id", "register_rs_0",
                 "register_rs_1", "register_rs_valid", "commit_valid", "commit_hartid",
                 "commit_id", "commit_commit_kill", "result_ready"):
        getattr(dut, name).value = 0
    dut.reset.value = 1
    for _ in range(3):
        await RisingEdge(clk)
    dut.reset.value = 0

    issues: list[dict] = []
    results: list[dict] = []
    commits: list[tuple[int, int, int]] = []  # (ciclo minimo, id, kill) in ordine FIFO
    i, cycle, start = 0, 0, None
    flush_wait = False  # un kill da inviare blocca nuovi issue
    idle_tail = 0
    max_cycles = 60 * len(instrs) + 2000
    timeout = False

    while True:
        if cycle >= max_cycles:
            timeout = True
            break
        done_issuing = i >= len(instrs)
        if done_issuing and not commits:
            idle_tail += 1
            if idle_tail > 60:
                break
        await FallingEdge(clk)
        # ---------------------------------------------------------- issue
        issuing = not done_issuing and not flush_wait
        if issuing:
            ins = instrs[i]
            if start is None:
                start = cycle
                if not ins["kill"]:
                    commits.append((cycle + ins["commit_offset"], ins["id"], 0))
            dut.issue_valid.value = 1
            dut.issue_req_instr.value = ins["instr"]
            dut.issue_req_id.value = ins["id"]
            dut.register_valid.value = 1
            dut.register_id.value = ins["id"]
            dut.register_rs_0.value = ins["rs0"]
            dut.register_rs_1.value = ins["rs1"]
            dut.register_rs_valid.value = 0b11 if cycle - start >= ins["rs_delay"] else 0
        else:
            dut.issue_valid.value = 0
            dut.register_valid.value = 0
            dut.register_rs_valid.value = 0
        # --------------------------------------------------------- commit
        sent = None
        if commits and commits[0][0] <= cycle:
            sent = commits[0]
            dut.commit_valid.value = 1
            dut.commit_id.value = sent[1]
            dut.commit_commit_kill.value = sent[2]
        else:
            dut.commit_valid.value = 0
        # --------------------------------------------------------- result
        ready = 1 if rng.random() > 0.3 else 0
        dut.result_ready.value = ready

        await ReadOnly()
        hs = issuing and _i(dut.issue_ready) == 1
        if hs:
            issues.append({"n": i, "id": instrs[i]["id"], "cycle": cycle,
                           "accept": _i(dut.issue_resp_accept),
                           "writeback": _i(dut.issue_resp_writeback),
                           "register_read": _i(dut.issue_resp_register_read)})
        if _i(dut.result_valid) == 1:
            idle_tail = 0
            if ready:
                results.append({"cycle": cycle, "id": _i(dut.result_id), "data": _i(dut.result_data),
                                "rd": _i(dut.result_rd), "we": _i(dut.result_we),
                                "hartid": _i(dut.result_hartid)})
        await RisingEdge(clk)
        cycle += 1
        if sent is not None:
            commits.pop(0)
            if sent[2]:
                flush_wait = False
        if hs:
            ins = instrs[i]
            if ins["kill"]:
                commits.append((cycle + ins["commit_offset"], ins["id"], 1))
                flush_wait = True
            i, start = i + 1, None

    with open(os.environ["MXAGENT_RESULTS"], "w", encoding="utf-8") as f:
        json.dump({"issues": issues, "results": results, "cycles": cycle, "timeout": timeout,
                   "id_width": idw}, f)
