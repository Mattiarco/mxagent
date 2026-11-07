# Sintesi e PPA (M5)

Il package `mxagent/synth/` misura area e ritardo dei design **con strumenti
eseguiti davvero**: ogni numero nei report viene da un file prodotto da Yosys,
nextpnr, OpenSTA o Vivado. Un backend assente sulla macchina viene marcato
`skipped` e non produce numeri.

## Backend

| backend | strumento | metriche | note |
|---|---|---|---|
| `generic` | Yosys `synth -flatten` + `abc -g cmos2` | `transistors` (stima `stat -tech cmos`), `depth` (livelli di porta, `ltp -noff`), `cells` | indipendente dalla tecnologia, veloce; **metrica primaria** del PPA Optimizer |
| `liberty` | Yosys + `abc -liberty <.lib>` | `area` (unità della libreria), `depth` | serve `synth.liberty` (es. Nangate45, sky130) |
| `ecp5` | Yosys `synth_ecp5` | `luts` (LUT4), `carry` (CCU2C), `muxes` (PFUMX+L6MUX21), `dsp`, `depth` | |
| `xc7` | Yosys `synth_xilinx -family xc7` | `luts` (LUT1–6), `carry` (CARRY4), `muxes` (MUXF7/8), `dsp`, `depth` | `cells` include i buffer di I/O |
| `nextpnr` | Yosys + nextpnr-ecp5 (LFE5U-85F, CABGA381) | `fmax_mhz` (mediana su `pnr_seeds`), `fmax_min`, `fmax_max`, `crit_ns` | wrapper registrato, vedi sotto |
| `opensta` | Yosys (liberty) + OpenSTA | `delay` (data arrival time), `area` | **non validato**: OpenSTA non era installato |
| `vivado` | Vivado batch, out-of-context, place & route | `luts`, `ffs`, `carry`, `dsp`, `wns_ns`, `fmax_mhz` | **non validato**: Vivado non era installato |

Per OpenSTA e Vivado sono testati solo i parser, su estratti dei formati
documentati. Il primo run con lo strumento vero va controllato a mano.

**Frontend.** Il Verilog emesso da firtool 1.62 contiene costrutti SystemVerilog
(assignment pattern `'{...}` su array impacchettati) che `read_verilog -sv` di
Yosys rifiuta. Si usa quindi il plugin **slang** (`read_slang`), incluso in OSS
CAD Suite. Se il plugin manca si ripiega su `read_verilog`, e `mxagent doctor`
lo segnala.

**Wrapper per la Fmax.** I benchmark sono combinatori e hanno centinaia di bit
di I/O, più dei pin dell'FPGA. Il wrapper ([wrapper.py](../mxagent/synth/wrapper.py)):
- carica gli ingressi da una catena di shift register alimentata da un pin;
- registra tutte le uscite e le riduce con XOR su un pin.

Il cammino misurato è quindi registro → DUT → registro. Siccome tutte le uscite
sono osservate, la sintesi non può eliminare logica del DUT. Le risorse si
riportano dalla sintesi del DUT da solo; la Fmax viene dal wrapper.

## PPA dei riferimenti

Dati di `mxagent synth --task benchmarks/<b> --backends generic,ecp5,xc7,nextpnr`:
- strumenti: Yosys 0.70+344, nextpnr-0.11.1-88, OSS CAD Suite;
- macchina: WSL Ubuntu, 2026-11-07;
- Fmax: mediana su 3 seed, con min–max tra parentesi.

| task | transistori | profondità | ECP5 LUT4 / CCU2C / DSP | XC7 LUT / CARRY4 | Fmax ECP5 (MHz) |
|---|---:|---:|---|---|---|
| b01 E2M1 decode | 114 | 8 | 8 / 3 / 0 | 9 / 2 | 299.3 (291.0–301.2) |
| b02 E2M1 mul | 584 | 20 | 16 / 5 / 1 | 26 / 4 | 104.1 (88.3–108.7) |
| b03 E8M0 → FP32 | 62 | 5 | 5 / 0 / 0 | 3 / 0 | 285.6 (282.8–319.3) |
| b04 quantizzatore MXFP4 | 59 748 | 140 | 12 435 / 1 197 / 0 | 5 910 / 583 | 15.4 (15.2–15.6) |
| b05 dot MXFP4 (k=32) | 44 610 | 148 | 2 489 / 228 / 32 | 1 651 / 159 | 20.9 (20.4–21.1) |
| b06 dot MXFP4 in pipeline (3 stadi) | 46 048 | 81 | 2 136 / 239 / 32 (99 FF) | 1 693 / 162 | 37.0 (30.4–38.8) |
| b07 dot NVFP4 (k=16) | 27 134 | 136 | 895 / 96 / 18 | 806 / 74 | 22.3 (19.2–22.6) |
| b08 MAC MXFP4 | 160 726 | 873 | 8 416 / 1 467 / 32 | 8 048 / 645 | 8.6 (8.1–8.7) |

Per b06 la profondità è quella dello stadio più lungo, perché `ltp -noff` si
ferma ai registri. Il target di b06 (30 MHz) è rispettato dal riferimento su
tutti e tre i seed.

Il riferimento di b08 è lento perché somma su una finestra a virgola fissa da
282 bit. È scritto per essere dimostrabilmente corretto, e lascia molto margine
al PPA Optimizer.

I riferimenti sono scritti per essere chiari, non ottimizzati, e servono come
punto di confronto per i design generati. Per esempio b05 usa 32 moltiplicatori
generici: su ECP5 Yosys li mappa su 32 blocchi DSP, mentre sul generico diventano
logica.

Tempi di sintesi su b05 (CPU WSL): generic 73 s, ecp5 3 s, xc7 128 s,
nextpnr (3 seed) 45 s.

## PPA Optimizer

Lo stadio `PPA` parte **solo dopo il pass** ([orchestrator.py](../mxagent/agents/orchestrator.py),
[ppa.py](../mxagent/agents/ppa.py)):

1. Sintesi primaria (`synth.primary`, default `generic`) del design corretto.
2. Fino a `synth.optimize_iters` passi. A ogni passo l'LLM (ruolo `optimizer`,
   se manca si usa `reflector`) riceve:
   - specifica, interfaccia e scheda di dominio;
   - il codice migliore finora;
   - il report di sintesi, con la ripartizione delle celle;
   - l'obiettivo;
   - i tentativi precedenti con il loro esito.
3. Ogni candidato rifà la **verifica completa** sugli stessi vettori, bit-exact.
   Un candidato sbagliato viene scartato (`functional_fail`) e non viene sintetizzato.
4. Un candidato corretto viene sintetizzato e accettato solo se l'obiettivo
   (`synth.objective`: `area`, `delay` o `adp` = area × profondità) migliora di
   almeno `min_gain`:
   - con `area`, la profondità non può peggiorare di oltre `tolerance`;
   - con `delay`, l'area non può peggiorare di oltre `tolerance`.
5. Il design di partenza resta il migliore finché nessun candidato lo batte,
   quindi lo stadio PPA **non può rendere sbagliato un design corretto**.
6. Report completo (tutti i `synth.backends`) del design di partenza e, se è
   cambiato, di quello finale.

Lo stadio PPA non cambia le metriche funzionali.
- `passed`, `time_to_pass_s` e `llm_to_pass` (il budget al momento del pass)
  vengono fissati prima dello stadio. Il confronto con le baseline a pari budget
  in M6 usa questi valori.
- `llm` comprende anche le chiamate dell'optimizer.
- `metrics.json["ppa"]` riporta la metrica primaria iniziale e finale, il
  guadagno e i tentativi fatti e accettati.
- In `ppa.json` ci sono tutti i backend e ogni passo dell'optimizer.

Limite: con un design decomposto l'optimizer riscrive solo il top-level; i
sottomoduli verificati restano invariati.

Ablation: `--set agent.ppa=false` tiene solo il report di sintesi, senza
ottimizzazione. Per sintetizzare il design finale di una baseline si usa
`mxagent synth --run <run>`.
