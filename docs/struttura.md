# Struttura del progetto

Mappa delle cartelle e dei file di **mxagent**, con il ruolo di ciascuno.
Per installare ed eseguire il progetto vedi [DEPLOYMENT.md](../DEPLOYMENT.md).

## Panoramica

```
mxagent/
├── README.md              descrizione del progetto, formati, uso, benchmark, roadmap
├── DEPLOYMENT.md          guida all'installazione e all'esecuzione
├── LICENSE                licenza MIT
├── Makefile               scorciatoie: setup, test, doctor, warmup, docker, clean
├── pyproject.toml         pacchetto Python, dipendenze, extra [sim,eval,dev], comando `mxagent`
├── .gitignore
│
├── mxagent/               codice sorgente Python (pacchetto e CLI)
├── chisel/                progetto sbt condiviso (Scala 2.13.14, Chisel 6.6.0)
├── benchmarks/            8 benchmark: specifica, contratto delle porte, riferimento
├── prompts/               prompt versionati degli agenti e delle baseline
├── knowledge/             base di conoscenza sulle trappole di Chisel 6 (per il RAG)
├── configs/               configurazioni YAML delle run e delle campagne
├── scripts/               installazione della toolchain senza sudo
├── docker/                immagine Docker riproducibile
├── tests/                 test pytest e fixture
└── docs/                  documentazione tecnica dei sottosistemi
```

Il flusso tra le parti principali:

```
benchmarks/<task>/spec.md + task.yaml
        │
        ▼
mxagent/agents  ──prompt (prompts/, knowledge/, formats/cards)──►  Ollama (mxagent/llm)
        │  sorgenti Chisel candidati
        ▼
mxagent/verify  ── chisel/ (sbt + firtool) → Verilator lint → cocotb ──►  confronto con mxagent/formats/golden
        │  feedback strutturato (verify/feedback) ─► di nuovo agli agenti, finché il design non passa
        ▼
mxagent/synth   ── Yosys / nextpnr ──► PPA  ──► PPA Optimizer (agents/ppa)
        │
        ▼
runs/<id>/  (mxagent/run/recorder)   ──►  mxagent/eval: campagne e report
```

## `mxagent/` — codice sorgente

| percorso | contenuto |
|---|---|
| `__main__.py`, `cli.py` | CLI headless: sottocomandi `doctor`, `golden`, `verify`, `run`, `synth`, `wrap`, `eval`, `report` |
| `config.py` | configurazione YAML validata con pydantic; ogni run salva la config effettiva |
| `versions.py` | versioni bloccate della toolchain, unica fonte per setup, doctor e Dockerfile |
| `doctor.py` | `mxagent doctor`: controllo dell'ambiente con istruzioni di installazione |

### `mxagent/formats/` — formati numerici e golden model

| file | contenuto |
|---|---|
| `specs.py` | costanti di MXFP4 (OCP MX v1.0) e NVFP4, ciascuna con la fonte |
| `fp.py` | motore esatto per formati floating-point binari |
| `numerics.py` | semantica numerica esplicita di un task (accumulo, arrotondamento, subnormali, overflow) |
| `golden.py` | golden model bit-exact dei kernel, solo aritmetica intera |
| `nvfp4.py` | quantizzatore NVFP4 (FP32 → blocchi da 16 × E2M1 con scala E4M3) |
| `packing.py` | convenzioni di impacchettamento dei vettori sulle porte hardware |
| `cards.py` | schede testuali dei formati, iniettate nei prompt |

### `mxagent/agents/` — sistema agentico (`--mode full`)

| file | contenuto |
|---|---|
| `orchestrator.py` | macchina a stati del sistema completo: spec parser, planner, decomposizione, integrazione |
| `solver.py` | risoluzione di un modulo: genera (best-of-N) → verifica → rifletti, con anti-loop ed escalation |
| `ppa.py` | PPA Optimizer: migliora area/ritardo di un design già corretto |
| `prompts.py` | caricamento dei prompt da `prompts/` con sostituzione delle variabili |
| `schemas.py` | JSON schema per l'output strutturato di Ollama e validazione del piano |
| `extract.py` | estrazione del codice Scala dalle risposte dell'LLM |

### `mxagent/llm/` — client del modello

| file | contenuto |
|---|---|
| `ollama.py` | client HTTP minimale per Ollama (solo libreria standard) |
| `chat.py` | chat con output strutturato, retry, cache delle risposte e controllo del budget |

### `mxagent/rag/` — conoscenza e riuso

| file | contenuto |
|---|---|
| `kb.py` | base di conoscenza delle trappole di Chisel 6 (`knowledge/chisel6_pitfalls.yaml`) |
| `library.py` | libreria di componenti già verificati (`library/`), riusati dall'agente |
| `embed.py` | similarità testuale: embedding Ollama con ripiego TF-IDF |

### `mxagent/verify/` — verifier deterministico (nessun LLM)

| file | contenuto |
|---|---|
| `verifier.py` | orchestrazione dei tre stadi e report di verifica |
| `chisel.py` | stadio 1: compilazione Scala, elaborazione Chisel, firtool → Verilog |
| `lint.py` | stadio 2: lint di sintetizzabilità con Verilator |
| `sim.py`, `sim_entry.py` | stadio 3: simulazione Verilator + cocotb in un sottoprocesso isolato |
| `tb/vector_tb.py` | testbench cocotb generico: applica i vettori e registra le uscite |
| `feedback.py` | confronto con il golden e feedback strutturato e compresso per il Reflector |
| `ddmin.py` | riduzione dei vettori che falliscono (delta-debugging) |
| `values.py` | scomposizione leggibile dei valori sulle porte (segno, esponente, mantissa) |

### `mxagent/synth/` — sintesi e PPA

| file | contenuto |
|---|---|
| `yosys.py` | sintesi logica con Yosys (generic, ECP5, Xilinx 7): area e profondità |
| `nextpnr.py` | place & route su Lattice ECP5: frequenza massima |
| `wrapper.py` | wrapper registrato per misurare la frequenza di un modulo |
| `external.py` | backend facoltativi: OpenSTA e Vivado in batch (non validati) |
| `ppa.py` | report PPA multi-backend e regola di accettazione del PPA Optimizer |

### `mxagent/bench/` — benchmark

| file | contenuto |
|---|---|
| `task.py` | contratto di un benchmark: porte tipizzate, temporizzazione, golden kernel, vettori |
| `kernels.py` | adattatori fra le porte di un task e il golden model (b01–b03) |
| `kernels_mx.py` | kernel dei benchmark a blocchi: b04 (quantizzatore), b05 (dot product) |
| `kernels_m7.py` | kernel di b06–b08 |
| `subkernels.py` | catalogo dei blocchi MX usati dal Planner per la decomposizione |

### Altri package

| percorso | contenuto |
|---|---|
| `run/baselines.py` | baseline A (single-shot) e B (loop generico), stesso verifier e budget |
| `run/recorder.py` | registro di una run: `runs/<id>/` con tutto ciò che serve a riprodurla |
| `eval/campaign.py` | campagna: YAML validato → lista deterministica di job |
| `eval/runner.py` | esecuzione della campagna, un sottoprocesso `mxagent run` per job, riprendibile |
| `eval/report.py` | report della campagna (`report.md`, `runs.csv`, `summary.json`) |
| `eval/stats.py` | statistiche: intervalli di Wilson, pass@k, test di McNemar |
| `wrap/cvxif.py` | coprocessore RISC-V CV-X-IF attorno al MAC di b08 |
| `wrap/tb_cvxif.py` | testbench cocotb del coprocessore (fa la parte della CPU) |
| `util/shell.py` | esecuzione di comandi esterni con timeout e log |

## `chisel/` — progetto sbt

| percorso | contenuto |
|---|---|
| `build.sbt`, `project/build.properties` | progetto sbt condiviso; versioni bloccate, controllate da `doctor` |
| `src/main/scala/mxagent/Elaborate.scala` | punto d'ingresso per l'elaborazione dei moduli in Verilog |
| `src/main/scala/mxagent/cvxif/MxCvxif.scala` | wrapper CV-X-IF scritto a mano (non generato dall'LLM) |
| `generated/` | *(runtime)* sorgenti prodotti dall'agente, aggiunti alla build |

## `benchmarks/` — gli 8 task

Ogni benchmark ha la stessa forma:

```
benchmarks/bNN_<nome>/
├── task.yaml      contratto delle porte, temporizzazione, numerics, vettori di test
├── spec.md        specifica in linguaggio naturale (finisce nei prompt)
└── reference/     implementazione Chisel scritta a mano (mai visibile all'agente)
```

| cartella | task | riferimento |
|---|---|---|
| `b01_e2m1_decode` | decoder E2M1 → virgola fissa | `E2M1Decode.scala` |
| `b02_e2m1_mul` | moltiplicatore E2M1 × E2M1 esatto | `E2M1Mul.scala` |
| `b03_e8m0_to_fp32` | scala E8M0 → FP32 | `E8M0ToFP32.scala` |
| `b04_mxfp4_quantize` | quantizzatore 32 × FP32 → blocco MXFP4 | `MXFP4Quantize.scala` |
| `b05_mxfp4_dot32` | dot product di due blocchi MXFP4 → FP32 | `MXFP4Dot32.scala` |
| `b06_mxfp4_dot32_pipe` | b05 in pipeline (latenza 3) | `MXFP4Dot32Pipe.scala` |
| `b07_nvfp4_dot16` | dot product NVFP4 | `NVFP4Dot16.scala` |
| `b08_mxfp4_mac` | MAC MXFP4 (poi coprocessore CV-X-IF) | `MXFP4Mac.scala` |

## `prompts/` e `knowledge/`

| percorso | contenuto |
|---|---|
| `prompts/full/` | prompt del sistema completo: `system`, `spec`, `planner`, `coder`, `reflect`, `integrate`, `ppa` |
| `prompts/baseline/` | prompt delle baseline: `system`, `task`, `fix` |
| `knowledge/chisel6_pitfalls.yaml` | errori tipici degli LLM con Chisel 6.6; ogni voce ha una regola, un esempio sbagliato e uno corretto, entrambi verificati dai test |

L'hash dei prompt viene registrato in `meta.json` di ogni run.

## `configs/`

| file | contenuto |
|---|---|
| `default.yaml` | configurazione della tesi: modelli per ruolo, meccanismi, budget, simulazione, sintesi |
| `local_cpu.yaml` | sviluppo su CPU/WSL con modelli piccoli |
| `eval/main.yaml` | campagna principale (750 job) |
| `eval/smoke.yaml` | prova rapida dell'harness |

## `scripts/` e `docker/`

| file | contenuto |
|---|---|
| `scripts/setup_server.sh` | installa nella home, senza sudo: JDK 17, sbt, OSS CAD Suite, firtool (aarch64), venv, modelli, warm-up |
| `docker/Dockerfile` | immagine Ubuntu 24.04 che esegue lo stesso script; Ollama resta sull'host |

## `tests/`

| percorso | contenuto |
|---|---|
| `test_fp.py`, `test_specs.py`, `test_golden.py` | motore FP, costanti dei formati, golden model (anche esaustivi) |
| `test_verify_unit.py`, `test_verify_integration.py` | verifier: logica e integrazione con sbt/Verilator/cocotb |
| `test_llm.py` | client Ollama contro un server HTTP finto |
| `test_agents.py`, `test_baselines.py` | macchina a stati del sistema completo e baseline, con LLM e verifier finti |
| `test_kb.py`, `test_kb_integration.py` | base di conoscenza: senza toolchain, e con sbt/Chisel (gli esempi sbagliati falliscono, quelli corretti elaborano) |
| `test_synth.py` | sintesi e parser dei report |
| `test_bench_mx.py`, `test_m7.py` | benchmark a blocchi, b06–b08, NVFP4, CV-X-IF |
| `test_eval.py` | harness di valutazione e statistiche |
| `test_framework.py` | configurazione, CLI `golden`, `doctor` (senza toolchain) |
| `fixtures/mutants/`, `fixtures/cvxif_mutants/` | design volutamente sbagliati: il verifier deve rilevarli |
| `fixtures/logs/`, `fixtures/verilog/`, `fixtures/synth/` | log, Verilog e report di sintesi di riferimento per i parser |
| `fixtures/decomposed_b05/` | esempio di b05 decomposto in blocchi |

I test che richiedono la toolchain vengono saltati automaticamente se questa manca.

## `docs/`

| file | contenuto |
|---|---|
| `struttura.md` | questo documento |
| `agent.md` | sistema completo: solve di un modulo, decomposizione, KB, libreria, ablation |
| `verifier.md` | verifier deterministico e feedback |
| `ppa.md` | sintesi multi-backend e PPA Optimizer |
| `eval.md` | campagne di valutazione e report |
| `cvxif.md` | coprocessore RISC-V CV-X-IF |

## Cartelle generate a runtime (non versionate)

| cartella | contenuto |
|---|---|
| `runs/<id>/` | una run: `config.yaml`, `meta.json`, `trace.json`, `candidates/`, `final/`, `verify/`, `metrics.json`, `ppa.json` |
| `evals/<campagna>/` | una campagna: `runs/`, `reference/`, `index.jsonl`, `campaign.json`, `report.md`, `runs.csv`, `summary.json` |
| `library/` | componenti verificati riusabili |
| `.cache/llm/` | cache delle risposte del modello |
| `chisel/target/`, `chisel/generated/`, `chisel/build/` | artefatti di sbt e sorgenti generati |
