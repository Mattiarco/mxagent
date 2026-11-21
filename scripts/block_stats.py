#!/usr/bin/env python3
"""Statistiche dei candidati di un blocco nelle run di una campagna (o di una run).

    python scripts/block_stats.py evals/b05_flatio_fix --module FixedToFp32

Legge ogni ``<run>/verify/<Modulo>/<tag>/report.json`` (il nome del modulo lo sceglie
il planner: il confronto ignora maiuscole). Per run e in totale: candidati generati,
quanti compilano, quanti arrivano alla simulazione, quanti passano; poi gli errori
dei candidati falliti raggruppati per tipo (stadio + messaggio senza numeri).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def error_kind(rep: dict) -> str:
    if rep["stage"] == "sim":
        return "sim: mismatch sui vettori"
    keys = rep.get("error_keys") or []
    if not keys:
        return f"{rep['stage']}: (nessuna firma)"
    # "scalac:File.scala:';' expected but '(' found." -> "scalac: ';' expected ..."
    k = keys[0]
    m = re.match(r"(\w+):[^:]*\.scala:(.*)$", k) or re.match(r"(\w+):(?:None:)?(.*)$", k)
    return f"{m.group(1)}: {m.group(2).strip()}" if m else k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--module", default="FixedToFp32")
    a = ap.parse_args()
    want = a.module.lower()
    per_run: dict[str, Counter] = {}
    errors: Counter = Counter()
    for f in sorted(a.root.rglob("verify/*/*/report.json")):
        if f.parent.parent.name.lower() != want:
            continue
        rep = json.loads(f.read_text(encoding="utf-8"))
        run = f.parent.parent.parent.parent
        c = per_run.setdefault(str(run.relative_to(a.root)) or ".", Counter())
        c["candidati"] += 1
        c["compilano"] += rep["stage"] != "compile"
        c["in simulazione"] += rep["stage"] in ("sim", "pass")
        c["passano"] += bool(rep["passed"])
        if not rep["passed"]:
            errors[error_kind(rep)] += 1
    if not per_run:
        raise SystemExit(f"nessun report per il modulo {a.module} sotto {a.root}")
    cols = ["candidati", "compilano", "in simulazione", "passano"]
    tot = sum(per_run.values(), Counter())
    w = max(len(r) for r in [*per_run, "TOTALE"])
    print(f"{'run'.ljust(w)}  " + "  ".join(cols))
    for r, c in [*per_run.items(), ("TOTALE", tot)]:
        print(f"{r.ljust(w)}  " + "  ".join(str(c[k]).rjust(len(k)) for k in cols))
    print("\nerrori residui per tipo:")
    for k, n in errors.most_common():
        print(f"  {n:4d}  {k}")


if __name__ == "__main__":
    main()
