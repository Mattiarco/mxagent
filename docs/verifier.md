# Verifier deterministico

Il verifier non usa LLM. Riceve i sorgenti Chisel di un candidato e un benchmark
(`benchmarks/<task>/task.yaml`) e restituisce un `VerifyReport` (JSON in
`<work>/<task>/<tag>/report.json`) con stadio raggiunto, punteggio e feedback compresso.

```
sorgenti .scala ──▶ 1. sbt: scalac + elaborazione Chisel + firtool ──▶ SystemVerilog
                    2. verilator --lint-only -Wall (sintetizzabilità)
                    3. cocotb + Verilator: esaustivi / corner / random
                    4. confronto bit-exact col golden model (Python)
                    5. riduzione dei casi (pattern, rappresentativi, delta-debugging)
```

## Stadi ed errori

| stadio | strumento | errore strutturato |
|---|---|---|
| `compile` | scalac | file, riga, colonna, messaggio, snippet (righe come le ha scritte l'LLM) |
| `compile` | elaborazione Chisel | eccezione + primo frame nel package del candidato |
| `compile` | firtool | messaggio con locatore Scala (es. loop combinatori) |
| `lint` | Verilator | codice warning + locatore Scala ricavato da `// @[File.scala l:c]` |
| `sim` | cocotb | conteggi per suite, pattern, vettori minimi scomposti in campi |

I warning di lint bloccanti (latch, loop, troncamenti, multi-driver...) non fermano
la simulazione, ma impediscono di passare: l'LLM riceve sia il problema di
sintetizzabilità sia lo stato funzionale.

## Package e workspace

Ogni candidato finisce in `src/main/scala/cand_<tag>/` di un workspace sbt copiato
da `chisel/`; la riga `package` è iniettata dal verifier (quella dell'LLM viene
sostituita) e i numeri di riga negli errori sono riportati al sorgente originale.
Il workspace contiene un solo candidato alla volta: uno rotto non blocca i successivi.

Modalità sbt (misurate in WSL, Ubuntu 26.04, 12 core):

| modalità | compilazione + elaborazione |
|---|---|
| `batch` (sbt a freddo) | ~7 s |
| `client` (server persistente, default su Linux) | ~1 s a caldo, ~7 s il primo job |

## Protocollo di temporizzazione del testbench

`mxagent/verify/tb/vector_tb.py` è generico: legge i vettori da JSON, pilota le
porte e scrive le uscite; il confronto avviene in Python.

- `comb`: per ogni vettore si pilotano gli ingressi, si attende 1 ns, si leggono le uscite.
- `pipe` con latenza L: reset per 3 cicli, poi a ogni fronte di **discesa** del ciclo
  `c` si leggono le uscite (appartengono all'ingresso `c - L`) e si pilota l'ingresso
  `c`. Un nuovo ingresso a ogni ciclo, quindi il test copre anche il throughput.

## Feedback

Esempio reale (mutante con il segno del prodotto calcolato con OR):

```
FUNCTIONAL MISMATCH: 49/256 test vectors fail (exhaustive: 207/256 pass).
Patterns:
 - every failing vector has a=neg AND b=neg; 15/207 passing vectors do
Smallest failing vectors (3):
#1 (exhaustive)
   in  a = 0x9 (s=1 e=00 m=1 = -0.5)
   in  b = 0x9 (s=1 e=00 m=1 = -0.5)
   out p: expected 1 [000000001] = +0.25
          got      -1 [111111111] = -0.25
          8 bit(s) differ: [1, 2, 3, 4, 5, 6, 7, 8]
```

Per ingressi vettoriali (blocchi da 32 elementi) i casi vengono ridotti con
delta-debugging a round batch: tutte le semplificazioni di un livello (elementi
azzerati a blocchi, scale portate a 1) sono simulate in un'unica chiamata,
senza ricompilare il modello Verilator.

`error_keys` contiene firme degli errori senza numeri di riga: servono all'anti-loop
(M4) per riconoscere errori ripetuti od oscillazioni.
