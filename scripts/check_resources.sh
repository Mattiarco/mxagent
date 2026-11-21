#!/usr/bin/env bash
# Diagnosi degli errori di sistema durante le verifiche (Errno 11 / EAGAIN,
# "unable to create native thread"): limiti di processi e thread, cgroup, memoria,
# JVM vive (anche server `sbt --client` orfani), Ollama.
#
#   bash scripts/check_resources.sh            # solo lettura, non termina nulla
#
# Nessuna dipendenza oltre a coreutils/procps. La stessa diagnosi, con la stima per
# --parallel, è `mxagent doctor --resources --parallel N`.
set -u
me=$(id -un)
uid=$(id -u)
hr() { printf '\n== %s\n' "$1"; }

hr "limiti (shell corrente)"
echo "ulimit -u (processi+thread): soft $(ulimit -Su)  hard $(ulimit -Hu)"
echo "ulimit -n (file aperti):     soft $(ulimit -Sn)  hard $(ulimit -Hn)"
echo "kernel.threads-max: $(cat /proc/sys/kernel/threads-max 2>/dev/null || echo ?)"
echo "kernel.pid_max:     $(cat /proc/sys/kernel/pid_max 2>/dev/null || echo ?)"
echo "cpu: $(nproc 2>/dev/null || echo ?)"

hr "cgroup (limite pids: vale il più basso lungo la catena)"
while IFS=: read -r _ ctrl path; do
  if [ -z "$ctrl" ]; then root=/sys/fs/cgroup; elif [[ ",$ctrl," == *",pids,"* ]]; then root=/sys/fs/cgroup/pids; else continue; fi
  d="$root$path"
  while [ -n "$d" ] && [ "$d" != "/" ]; do
    [ -r "$d/pids.max" ] && echo "pids.max $(cat "$d/pids.max")  pids.current $(cat "$d/pids.current" 2>/dev/null)  ($d)"
    [ "$d" = "$root" ] && break
    d=$(dirname "$d")
  done
done < /proc/self/cgroup
if command -v systemctl >/dev/null 2>&1; then
  echo "systemd user slice: $(systemctl show "user-$uid.slice" -p TasksMax -p TasksCurrent 2>/dev/null | tr '\n' ' ')"
fi

hr "processi e thread dell'utente $me"
echo "processi: $(ps -u "$me" --no-headers 2>/dev/null | wc -l)"
echo "thread (task, contano per ulimit -u e pids.max): $(ps -L -u "$me" --no-headers 2>/dev/null | wc -l)"
echo "thread per comando (somma di nlwp):"
ps -u "$me" -o nlwp=,comm= 2>/dev/null | awk '{n[$2]+=$1; p[$2]++} END{for (c in n) printf "  %-16s %6d thread in %3d processi\n", c, n[c], p[c]}' | sort -k2 -nr | head -12
echo "primi 10 per numero di thread:"
ps -u "$me" -o pid=,ppid=,nlwp=,etime=,comm= --sort=-nlwp 2>/dev/null | head -10 | \
  awk '{printf "  pid %-8s ppid %-8s thread %-5s età %-12s %s\n", $1, $2, $3, $4, $5}'

hr "JVM dell'utente (sbt: server di --client, -batch, ForkMain)"
found=0
for pid in $(pgrep -u "$uid" -x java 2>/dev/null); do
  found=1
  cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || echo ?)
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  nlwp=$(ps -o nlwp= -p "$pid" 2>/dev/null | tr -d ' ')
  etime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
  rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  kind=java; grep -qE 'sbt-launch|xsbt.boot|-Dsbt\.' <<<"$cmd" && kind=sbt
  role=""
  if [ "$kind" = sbt ]; then
    case "$cmd" in
      *sbt.ForkMain*) role=forkmain ;; *--detach-stdio*|*" --server"*) role=server ;;
      *--client*|*sbt.client*|*NetworkClient*) role=client ;; *) role=batch ;;
    esac
  fi
  cpus=$(sed -n 's/.*-XX:ActiveProcessorCount=\([0-9]*\).*/\1/p' <<<"$cmd")
  maps=$(wc -l < "/proc/$pid/maps" 2>/dev/null || echo ?)
  orphan=""
  # orfana se nessun `mxagent run` vivo ha la cartella della run sopra il cwd della JVM
  if [ "$kind" = sbt ]; then
    orphan=" ORFANA?"
    for rp in $(pgrep -u "$uid" -f "mxagent run" 2>/dev/null); do
      args=$(tr '\0' ' ' < "/proc/$rp/cmdline" 2>/dev/null)
      rid=$(sed -n 's/.*--run-id \([^ ]*\).*/\1/p' <<<"$args")
      [ -n "$rid" ] && [[ "$cwd" == *"/$rid/"* || "$cwd" == *"/$rid" ]] && orphan=""
    done
  fi
  echo "  pid $pid ppid $ppid $kind${role:+/$role} thread $nlwp cpu ${cpus:-tutte} mappe $maps rss $((rss / 1024)) MiB età $etime$orphan  cwd $cwd"
done
[ $found = 0 ] && echo "  nessuna"
echo "server sbt --client (file active.json dei workspace sotto runs/ ed evals/):"
find runs evals -path '*project/target/active.json' 2>/dev/null | head -20 | sed 's/^/  /' || true

hr "memoria"
free -h 2>/dev/null || grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo
# pthread_create restituisce EAGAIN anche se non riesce a mappare lo stack del thread
# (overcommit stretto, vm.max_map_count): non solo per ulimit -u / pids.max
echo "vm.overcommit_memory: $(cat /proc/sys/vm/overcommit_memory 2>/dev/null || echo ?) (0 euristico, 1 sempre, 2 stretto)"
grep -E '^(CommitLimit|Committed_AS):' /proc/meminfo | sed 's/^/  /'
echo "vm.max_map_count: $(cat /proc/sys/vm/max_map_count 2>/dev/null || echo ?)"
echo "OOM killer recente (dmesg, se leggibile):"
dmesg 2>/dev/null | grep -iE 'out of memory|oom-kill|killed process' | tail -5 | sed 's/^/  /' || true

hr "Ollama"
if pgrep -x ollama >/dev/null 2>&1; then
  for pid in $(pgrep -x ollama); do
    echo "  pid $pid utente $(ps -o user= -p "$pid") thread $(ps -o nlwp= -p "$pid" | tr -d ' ') rss $(( $(ps -o rss= -p "$pid") / 1024 )) MiB"
  done
  echo "  (se l'utente è $me i suoi thread contano nello stesso limite delle JVM)"
  command -v ollama >/dev/null && ollama ps 2>/dev/null | sed 's/^/  /'
else
  echo "  ollama non in esecuzione"
fi

hr "configurazione JVM/sbt"
echo "SBT_OPTS=${SBT_OPTS:-'(non impostata: mxagent usa -Xss8m)'}"
echo "sim.jvm_cpus in configs/default.yaml: $(grep -E '^\s*jvm_cpus:' configs/default.yaml 2>/dev/null | awk '{print $2}') (-XX:ActiveProcessorCount delle JVM di sbt)"
if command -v sbt >/dev/null 2>&1; then
  sbt_dir=$(dirname "$(readlink -f "$(command -v sbt)")")
  echo "sbt: $(command -v sbt); thin client nativo (sbtn): $(ls "$sbt_dir"/sbtn* 2>/dev/null | tr '\n' ' ' || true)"
  echo "  (senza sbtn per questa architettura, sbt --client avvia una JVM client per ogni comando)"
fi
echo "JAVA_TOOL_OPTIONS=${JAVA_TOOL_OPTIONS:-'(non impostata)'}"
echo "sim.sbt_mode in configs/default.yaml: $(grep -E '^\s*sbt_mode:' configs/default.yaml 2>/dev/null | awk '{print $2}')"
