"""Punto d'ingresso in sottoprocesso per il runner cocotb (isolamento e timeout).

    python -m mxagent.verify.sim_entry build --top T --build-dir D --sources a.sv ...
    python -m mxagent.verify.sim_entry test  --top T --build-dir D --vectors v.json --results r.json
"""
from __future__ import annotations

import argparse
import os
import sys


def get_runner(sim: str):
    try:
        from cocotb_tools.runner import get_runner as gr  # cocotb 2.x
    except ImportError:
        from cocotb.runner import get_runner as gr  # cocotb 1.9
    return gr(sim)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["build", "test"])
    p.add_argument("--top", required=True)
    p.add_argument("--build-dir", required=True)
    p.add_argument("--simulator", default="verilator")
    p.add_argument("--sources", nargs="*", default=[])
    p.add_argument("--vectors")
    p.add_argument("--results")
    p.add_argument("--test-module", default="mxagent.verify.tb.vector_tb")
    p.add_argument("--env", nargs="*", default=[], metavar="KEY=VAL",
                   help="variabili per il testbench (oltre a vettori e risultati)")
    a = p.parse_args()
    runner = get_runner(a.simulator)
    if a.action == "build":
        build_args = ["-Wno-fatal", "-Wno-lint", "-Wno-style"] if a.simulator == "verilator" else []
        runner.build(sources=a.sources, hdl_toplevel=a.top, build_dir=a.build_dir,
                     build_args=build_args, timescale=("1ns", "1ps"), always=True)
        return 0
    env = dict(kv.split("=", 1) for kv in a.env)
    if a.vectors:
        env["MXAGENT_VECTORS"] = os.path.abspath(a.vectors)
    if a.results:
        env["MXAGENT_RESULTS"] = os.path.abspath(a.results)
    os.environ.update(env)
    # build e test sono in processi diversi: la lingua va ripetuta (il runner non la ricorda)
    runner.test(hdl_toplevel=a.top, hdl_toplevel_lang="verilog", test_module=a.test_module,
                build_dir=a.build_dir, test_dir=a.build_dir, extra_env=env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
