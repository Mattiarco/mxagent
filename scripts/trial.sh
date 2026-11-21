#!/usr/bin/env bash
# Prova di UNA run (le run sono sempre una alla volta: una GPU e un Ollama), con la
# diagnosi delle risorse prima, durante e dopo, e un riepilogo unico da incollare.
#
#   bash scripts/trial.sh [TASK] [PROFILO] [SEED] [MODALITÀ]
#   (default: benchmarks/b05_mxfp4_dot32 all14b 2 full)
#
# Cache LLM disattivata (--llm-cache off): ogni chiamata va al modello. Uscita in
# $OUT (default runs/trial_<data>): check_before.txt, check_during_<k>.txt,
# check_after.txt, watch.tsv (campioni ogni 5 s), s<seed>/ (la run), s<seed>.log,
# summary.txt (stampato anche a schermo). Durata: fino a ~45 min (budget 1800 s più la
# scelta di best/). Se un'altra run o una campagna è in corso, la run esce subito
# (lock di mxagent.util.runlock) e il riepilogo lo dice.
set -u
TASK=${1:-benchmarks/b05_mxfp4_dot32}
PROF=${2:-all14b}
SEED=${3:-2}
MODE=${4:-full}
OUT=${OUT:-runs/trial_$(date +%Y%m%d_%H%M%S)}
PY=${PYTHON:-python}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

echo "prova: 1 run, $TASK, $MODE, $PROF, seed $SEED, cache LLM off -> $OUT"
bash "$here/check_resources.sh" > "$OUT/check_before.txt" 2>&1

mxagent run --task "$TASK" --mode "$MODE" --model-profile "$PROF" --seed "$SEED" --llm-cache off \
  --runs-dir "$OUT" --run-id "s$SEED" > "$OUT/s$SEED.log" 2>&1 &
rpid=$!
echo "run avviata: pid $rpid (log $OUT/s$SEED.log)"

bash "$here/watch_resources.sh" "$OUT/watch.tsv" 5 "$rpid" &
wpid=$!
# "durante": a 3 minuti (server sbt avviato, verifiche in corso), poi ogni 10 minuti
(
  sleep 180; k=1
  while kill -0 "$rpid" 2>/dev/null; do
    bash "$here/check_resources.sh" > "$OUT/check_during_$k.txt" 2>&1
    k=$((k + 1)); sleep 600
  done
) &
dpid=$!

wait "$rpid"; code=$?
wait "$wpid" 2>/dev/null
kill "$dpid" 2>/dev/null; wait "$dpid" 2>/dev/null
sleep 5  # le JVM terminate a fine run hanno il tempo di uscire
bash "$here/check_resources.sh" > "$OUT/check_after.txt" 2>&1
echo "codice di uscita: $code" > "$OUT/exit_codes.txt"
pgrep -u "$(id -u)" -a java > "$OUT/java_after.txt" 2>/dev/null || true

"$PY" - "$OUT" <<'PYEOF' | tee "$OUT/summary.txt"
import json, sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
print(f"== riepilogo {out}")
print((out / "exit_codes.txt").read_text().strip())
ready = True
for log in sorted(out.glob("s*.log")):
    text = log.read_text(errors="replace")
    if "un'altra run è in corso" in text:
        ready = False
        print("RUN NON AVVIATA: " + text.strip()[-400:])
runs = sorted(p for p in out.iterdir() if p.is_dir() and p.name.startswith("s"))
if not runs:
    ready = False
    print("nessuna cartella di run in", out)
for rd in runs:
    print(f"\n-- run {rd.name}")
    tr = rd / "trace.json"
    ev = json.loads(tr.read_text()) if tr.is_file() else []
    res = rd / "result.json"
    r = json.loads(res.read_text()) if res.is_file() else None
    if r is None:
        ready = False
        print("result.json: MANCANTE")
    else:
        print(f"result.json: status {r.get('status')}, reason {r.get('reason')}, passed {r.get('passed')}, "
              f"wall_s {r.get('wall_s')}, llm_calls {r.get('llm_calls')}, cached {r.get('llm_cached_calls')}, "
              f"final_phase_error {r.get('final_phase_error')}")
        if r.get("status") == "error":
            ready = False
    print("stati:", [e.get("state") for e in ev if e.get("kind") == "state"])
    print("end:", any(e.get("kind") == "end" for e in ev), " metrics.json:", (rd / "metrics.json").is_file())
    retries = [e for e in ev if e.get("kind") == "infra_retry"]
    print(f"eventi infra_retry: {len(retries)}")
    for e in retries:
        print(f"  tentativo {e.get('attempt')} attesa {e.get('wait_s')} s, {e.get('where')}, {e.get('tag')}: "
              f"{str(e.get('reason'))[:160]}  java vive: {e.get('java')}")
    blocks = defaultdict(lambda: [0, 0, 0])
    infra = []
    for rep in sorted((rd / "verify").glob("*/*/report.json")):
        d = json.loads(rep.read_text())
        b = blocks[rep.parent.parent.name]
        if d.get("stage") == "infra":
            infra.append(f"{rep.parent.parent.name}/{rep.parent.name}: {d.get('infra')}")
            continue
        b[0] += 1
        b[1] += d.get("stage") != "compile"
        b[2] += bool(d.get("passed"))
        keys = [k for k in d.get("error_keys", []) if k.startswith("sbt:")]
        if keys:
            print(f"  ATTENZIONE {rep.parent.parent.name}/{rep.parent.name}: error_key {keys[0][:120]}")
    for m, (n, c, p) in blocks.items():
        print(f"  blocco {m}: candidati {n}, compilano {c}, passano {p}")
    print(f"report stage infra: {len(infra)}" + "".join(f"\n  {x}" for x in infra))
    if infra:
        ready = False

w = out / "watch.tsv"
if w.is_file():
    rows = [ln.split("\t") for ln in w.read_text().splitlines()]
    head, data = rows[0], rows[1:]
    col = {h: i for i, h in enumerate(head)}

    def mx(name, fn=max):
        vals = [int(r[col[name]]) for r in data if r[col[name]].lstrip("-").isdigit()]
        return fn(vals) if vals else None
    print(f"\n-- picchi da watch.tsv ({len(data)} campioni ogni 5 s)")
    for h in ("user_tasks", "cgroup_pids", "java_n", "java_threads", "firtool", "verilator", "cc1plus",
              "make", "python", "yosys", "other", "ollama_threads", "committed_mb"):
        print(f"  {h:15s} max {mx(h)}")
    print(f"  {'mem_avail_mb':15s} min {mx('mem_avail_mb', min)}")
    per_role = defaultdict(int)
    for r in data:
        for item in r[col["java_detail"]].split():
            _, role, th = item.split(":")
            per_role[role] = max(per_role[role], int(th))
    print("  thread massimi per JVM, per ruolo:", dict(per_role) or "nessuna JVM vista")
java_after = (out / "java_after.txt").read_text().strip()
print("\nJVM rimaste alla fine:", java_after or "nessuna")
print("\ncriterio 'pronta' (nessun errore di sistema non recuperato, result.json presente):",
      "SÌ" if ready else "NO")
PYEOF

for f in "$OUT"/check_before.txt "$OUT"/check_during_1.txt "$OUT"/check_after.txt; do
  [ -f "$f" ] && { echo; echo "######## $(basename "$f")"; cat "$f"; }
done
