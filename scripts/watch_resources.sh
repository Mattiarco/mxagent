#!/usr/bin/env bash
# Campiona ogni INTERVAL secondi i task (processi + thread) dell'utente e chi li usa:
# JVM (server sbt, client, batch) con i loro thread, firtool, Verilator, g++/make,
# Python, Yosys, Ollama; memoria disponibile e memoria impegnata (Committed_AS).
#
#   bash scripts/watch_resources.sh OUT.tsv [INTERVAL] [PID...]
#
# Si ferma quando tutti i PID dati sono terminati (o con Ctrl-C / kill). Una riga per
# campione, separata da tab; la colonna java_detail elenca pid:ruolo:thread di ogni JVM.
set -u
out=${1:?uso: watch_resources.sh OUT.tsv [INTERVAL] [PID...]}
iv=${2:-5}
shift $(( $# >= 2 ? 2 : 1 ))
watch_pids=("$@")
me=$(id -un)
uid=$(id -u)

pids_cur() {  # pids.current del cgroup del processo (v2 o v1)
  local line path
  line=$(grep -E '^0::|:pids:' /proc/self/cgroup | head -1)
  path=${line##*:}
  for f in "/sys/fs/cgroup$path/pids.current" "/sys/fs/cgroup/pids$path/pids.current"; do
    [ -r "$f" ] && { cat "$f"; return; }
  done
  echo NA
}

role() {  # ruolo di una JVM dalla riga di comando
  local c
  c=$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null)
  case "$c" in
    *sbt.ForkMain*) echo forkmain ;;
    *--detach-stdio*|*" --server"*) echo server ;;
    *sbt-launch*--client*|*sbt.client*|*NetworkClient*) echo client ;;
    *sbt-launch*|*xsbt.boot*) echo batch ;;
    *) echo java ;;
  esac
}

printf 'time\tuser_tasks\tcgroup_pids\tjava_n\tjava_threads\tfirtool\tverilator\tcc1plus\tmake\tpython\tyosys\tother\tollama_threads\tmem_avail_mb\tcommitted_mb\tjava_detail\n' > "$out"
while :; do
  ts=$(date +%H:%M:%S)
  # thread per comando del processo (nlwp = thread del processo; il nome dei singoli
  # thread non serve: la JVM li rinomina "GC Thread#0", "C2 CompilerThread" ...)
  read -r total firtool verilator cc1 mk py ys other < <(ps -u "$me" -o nlwp=,comm= 2>/dev/null | awk '
    {n=$1; c=$2; t+=n}
    c=="java"{next}
    c~/^firtool/{f+=n; next}
    c~/^(verilator|verilator_bin)/{v+=n; next}
    c~/^(cc1plus|g\+\+|c\+\+|ld|as|collect2)/{g+=n; next}
    c=="make"{m+=n; next}
    c~/^python/{p+=n; next}
    c~/^yosys/{y+=n; next}
    {o+=n}
    END{printf "%d %d %d %d %d %d %d %d\n", t, f, v, g, m, p, y, o}')
  jn=0; jt=0; det=""
  for pid in $(pgrep -u "$uid" -x java 2>/dev/null); do
    th=$(awk '/^Threads:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    [ -z "$th" ] && continue
    jn=$((jn + 1)); jt=$((jt + th))
    det="$det$pid:$(role "$pid"):$th "
  done
  ot=0
  for pid in $(pgrep -x ollama 2>/dev/null); do
    th=$(awk '/^Threads:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    ot=$((ot + ${th:-0}))
  done
  ma=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
  ca=$(awk '/^Committed_AS:/{print int($2/1024)}' /proc/meminfo)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ts" "$total" "$(pids_cur)" \
    "$jn" "$jt" "$firtool" "$verilator" "$cc1" "$mk" "$py" "$ys" "$other" "$ot" "$ma" "$ca" "${det% }" >> "$out"
  if [ ${#watch_pids[@]} -gt 0 ]; then
    alive=0
    for p in "${watch_pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break
  fi
  sleep "$iv"
done
