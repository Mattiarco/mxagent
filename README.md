# mxagent

Pipeline agentica **interamente locale** (LLM via [Ollama](https://ollama.com)) che
genera unità aritmetiche **MXFP4 / NVFP4** in **Chisel 6**, le verifica in modo
**bit-exact** contro un golden model Python e ne misura area e timing.

Ispirato a [ReChisel](https://github.com/niujuxin/ReChisel) (Niu et al., DAC 2025,
arXiv:2505.19734), di cui riprende l'idea del ciclo di *reflection* su feedback di
compilazione e simulazione, specializzandolo sull'aritmetica a microscaling.

> **Stato:** milestone **M1–M7** implementate (toolchain, golden model, verifier
> deterministico, baseline, sistema completo con decomposizione, KB, RAG, best-of-N,
> anti-loop, sintesi multi-backend e PPA Optimizer, harness di valutazione, NVFP4,
> coprocessore RISC-V CV-X-IF); benchmark 1–8.
> Il resto del sistema arriva per milestone, vedi [Roadmap](#roadmap).

## Formati e semantica numerica

Le costanti sono in [`mxagent/formats/specs.py`](mxagent/formats/specs.py), ciascuna
con la sezione della fonte da cui è stata verificata:

| | elementi | blocco | scala |
|---|---|---|---|
| **MXFP4** (OCP MX v1.0, Tab. 1) | FP4 E2M1: bias 1, ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}, nessun Inf/NaN | 32 | E8M0: 2^(X−127), 0xFF = NaN |
| **NVFP4** (NVIDIA) | FP4 E2M1 | 16 | FP8 E4M3 (max 448, NaN = S.1111.111) + FP32 per tensore |

La spec OCP lascia *implementation-defined* la precisione interna del dot product
(§6.1) e i dettagli della scala (§6.3). Per avere un "bit-exact" ben definito ogni
task dichiara un blocco `numerics:` ([`numerics.py`](mxagent/formats/numerics.py)):
accumulo esatto o per blocco, arrotondamento, subnormali, overflow, casi limite
del quantizzatore. Lo stesso testo finisce nei prompt (`mxagent golden card`).

Il golden model ([`golden.py`](mxagent/formats/golden.py)) usa solo aritmetica
intera esatta (ogni valore è `n·2^e`): niente `float`, niente doppi arrotondamenti.
È validato contro tre oracoli indipendenti: le tabelle della spec, numpy
(float32/float16) e `ml_dtypes` (E4M3), con test esaustivi dove lo spazio è piccolo
(16×16 coppie E2M1, 256 scale, 256×256 coppie di scale nel dot).

Il quantizzatore NVFP4 ([`nvfp4.py`](mxagent/formats/nvfp4.py)) segue le Eq. 1–4
dell'Appendice B di arXiv:2509.25149 (NVIDIA). La fonte non specifica:
- arrotondamento e saturazione delle scale E4M3 e degli elementi E2M1;
- il caso amax = 0;
- la precisione dei passaggi intermedi.

Queste scelte sono dichiarate in `NVFP4_CHOICES`.

## Installazione (server Linux headless, senza sudo)

```bash
git clone <repo> mxagent && cd mxagent
bash scripts/setup_server.sh --models qwen2.5-coder:14b,qwen2.5-coder:32b
source ~/.local/mxagent/env.sh        # oppure --add-to-bashrc
mxagent doctor
```

Lo script è idempotente e installa in `~/.local/mxagent`: JDK 17 e sbt (Coursier),
OSS CAD Suite (Yosys, Verilator, nextpnr, Icarus), firtool 1.62.1 (compilato dai
sorgenti solo su aarch64), un venv Python con mxagent e cocotb; infine elabora un
modulo di prova per popolare le cache (dopo funziona offline). Ollama deve girare
sulla macchina con la GPU (`ollama serve`); se è altrove: `OLLAMA_HOST=...`.

Versioni bloccate ([`mxagent/versions.py`](mxagent/versions.py)): Scala 2.13.14,
sbt 1.10.1, Chisel 6.6.0, firtool 1.62.1. Vivado è facoltativo: se `vivado` è nel
PATH viene usato in batch mode.

Docker (senza Vivado; Ollama resta sull'host):

```bash
docker build -f docker/Dockerfile -t mxagent .
docker run --rm -it --add-host=host.docker.internal:host-gateway mxagent doctor
```

## Uso

```bash
mxagent doctor [--warmup]                       # ambiente, modelli, cache
mxagent golden card                             # scheda MXFP4 iniettata nei prompt
mxagent golden dot --a 2 --b 7 --a-scale 130    # dot di due blocchi -> FP32
mxagent golden quant --x 1.0,-2.5,7.0,...       # FP32 -> blocco MXFP4
mxagent verify --task benchmarks/b02_e2m1_mul  # verifica il riferimento (o --sources X.scala)
mxagent run --task benchmarks/b01_e2m1_decode --mode baseline-b --model qwen2.5-coder:7b --seed 0
mxagent run --task benchmarks/b01_e2m1_decode --model-profile all14b   # 14B per tutti i ruoli
mxagent synth --task benchmarks/b05_mxfp4_dot32  # PPA del riferimento (verificato prima)
mxagent synth --run runs/<id> --backends generic,ecp5,nextpnr   # PPA del design di una run
mxagent wrap                                    # coprocessore CV-X-IF del MAC (b08) e test di protocollo
mxagent eval --campaign configs/eval/main.yaml  # campagna (baseline, full, ablation), riprendibile
mxagent report --dir evals/main                 # tabelle, CI di Wilson, McNemar, PPA
make test                                       # test (quelli d'integrazione richiedono la toolchain)
```

Il verifier è descritto in [docs/verifier.md](docs/verifier.md), la sintesi e il PPA
Optimizer in [docs/ppa.md](docs/ppa.md), la valutazione in [docs/eval.md](docs/eval.md), il
coprocessore RISC-V in [docs/cvxif.md](docs/cvxif.md).

Ogni `mxagent run` produce `runs/<id>/` (o `$MXAGENT_RUNS/<id>/`):

| file | contenuto |
|---|---|
| `config.yaml` | configurazione effettiva |
| `meta.json` | task, modalità, seed, digest del modello, hash dei prompt, versioni toolchain |
| `trace.json` | ogni chiamata LLM (prompt, risposta, token, tempo) e ogni verifica |
| `candidates/NN/` | sorgenti di ogni iterazione |
| `final/` | miglior candidato: sorgenti Chisel e Verilog emesso |
| `verify/` | log e `report.json` del verifier per iterazione |
| `metrics.json` | esito, iterazioni, tempo fino alla correttezza, token, errori ripetuti, PPA |
| `ppa.json`, `ppa/` | sintesi del design corretto e di quello ottimizzato, passi del PPA Optimizer |
| `final_functional/` | design corretto prima del PPA Optimizer (solo se questo l'ha migliorato) |

Le risposte del modello sono in cache (`.cache/llm`, chiave = modello + digest +
messaggi + opzioni): una run rieseguita con gli stessi parametri è identica.

Modelli: `--model-profile all14b | all32b | mixed` imposta il modello di tutti i
ruoli LLM (`embed` escluso) secondo `llm.profiles` della config; `--model X` è la
scorciatoia per un profilo uniforme. Il nome della run e `meta.json` riportano il
profilo (`..._full_all14b_s0`), `result.json` i modelli effettivi in `models_by_role`.
La config di default senza profilo è `mixed` (14B per coder/fixer, 32B per gli altri).

Modalità: `full` (sistema completo, [docs/agent.md](docs/agent.md)), `baseline-a`
(single-shot, prompt ingenuo) e `baseline-b` (loop generico con log grezzi e storia
completa). Le ablation si fanno con `--set agent.<meccanismo>=false`.

## Benchmark

| | task | spazio di test |
|---|---|---|
| b01 | decoder E2M1 → virgola fissa | esaustivo (16) |
| b02 | moltiplicatore E2M1 × E2M1 esatto | esaustivo (256) |
| b03 | scala E8M0 → FP32 (NaN, subnormale 2^-127) | esaustivo (256) |
| b04 | quantizzatore 32 × FP32 → blocco MXFP4 (scala + RNE + saturazione) | corner + 3000 random |
| b05 | dot product di due blocchi MXFP4 → FP32 (somma esatta, un arrotondamento) | corner + 4000 random |
| b06 | b05 in pipeline (latenza 3, un ingresso per ciclo), target 30 MHz su ECP5 | corner + 4000 random |
| b07 | dot NVFP4: 16 × E2M1, scale E4M3 (subnormali, segno, NaN) → FP32 | corner + 4000 random |
| b08 | MAC MXFP4: acc + dot con un solo arrotondamento; poi coprocessore CV-X-IF | corner + 5000 random |

Ogni task ha `task.yaml` (contratto delle porte), `spec.md` (va nei prompt) e
`reference/` (scritto a mano, mai visibile all'agente).

## Roadmap

| | contenuto | stato |
|---|---|---|
| M1 | toolchain, formati, golden model + test esaustivi | ✅ |
| M2 | verifier deterministico (sbt/firtool, Verilator, cocotb, delta-debugging), benchmark 1–3 | ✅ |
| M3 | client Ollama, loop base (= Baseline B) | ✅ |
| M4 | spec parser, planner, coder, reflector, RAG, best-of-N, escape | ✅ |
| M5 | sintesi Yosys (generic, ECP5, Xilinx 7) + nextpnr, PPA optimizer; OpenSTA/Vivado opzionali (non validati) | ✅ |
| M6 | harness di valutazione, ablation, report | ✅ (harness; campagna sul server) |
| M7 | b06–b08, quantizzatore NVFP4 (golden), coprocessore RISC-V CV-X-IF | ✅ |

## Riferimenti

- OCP Microscaling Formats (MX) Specification v1.0, 2023.
- NVIDIA, *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*.
- Niu et al., *ReChisel*, DAC 2025 — <https://github.com/niujuxin/ReChisel>.
- F. Zaruba, L. Benini, *The Cost of Application-Class Processing*, IEEE TVLSI, 2019.

Licenza MIT.
