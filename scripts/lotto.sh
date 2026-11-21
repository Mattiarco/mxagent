#!/usr/bin/env bash
# Lancia un lotto di seed con scripts/compare_backends.py, un (seed, kernel) alla volta,
# e non avvia nuovi job dopo STOP_H ore: nessuna run viene interrotta a metà e il
# lancio resta entro ~12 h. Rilanciato con gli stessi argomenti riprende da dove era
# rimasto (le run con result.json non si rifanno).
#
#   KERNELS=b01_e2m1_decode,... bash scripts/lotto.sh CARTELLA PRIMO_SEED ULTIMO_SEED \
#       -- [argomenti di compare_backends.py; --set va per ultimo]
#
# Variabili: KERNELS (obbligatoria), STOP_H (default 11: dopo 11 h nessun job nuovo),
#            SEED0 (primo seed della cartella, per compare.json; default 10), PYTHON.
set -u
[ $# -ge 4 ] && [ "$4" = "--" ] || { echo "uso: KERNELS=... $0 CARTELLA PRIMO ULTIMO -- [args]"; exit 2; }
OUT=$1; S0=$2; S1=$3; shift 4
KERNELS=${KERNELS:?imposta KERNELS=task1,task2,...}
STOP_H=${STOP_H:-11}
SEED0=${SEED0:-10}
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
log="$OUT/lotto_s${S0}-${S1}.log"
start=$(date +%s)
echo "== $(date '+%F %T') lotto $OUT seed $S0-$S1, kernel $KERNELS, stop dopo ${STOP_H} h" | tee -a "$log"
for s in $(seq "$S0" "$S1"); do
  for t in ${KERNELS//,/ }; do
    el=$(( $(date +%s) - start ))
    if [ "$el" -ge $(( STOP_H * 3600 )) ]; then
      echo "== $(date '+%F %T') STOP dopo $((el / 60)) min: prossimo job $t seed $s." \
           "Rilancia lo stesso comando per riprendere." | tee -a "$log"
      exit 0
    fi
    "$PY" scripts/compare_backends.py --tasks "$t" --seed-start "$s" --n 1 --out "$OUT" "$@" 2>&1 | tee -a "$log"
  done
done
"$PY" scripts/compare_backends.py --tasks "$KERNELS" --seed-start "$SEED0" --n $(( S1 - SEED0 + 1 )) \
  --out "$OUT" "$@" 2>&1 | tee -a "$log"
echo "== $(date '+%F %T') lotto $S0-$S1 finito in $(( ($(date +%s) - start) / 60 )) min" | tee -a "$log"
