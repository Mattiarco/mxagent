# Deployment guide — mxagent

Questa guida spiega come installare, verificare ed eseguire **mxagent** su una
macchina nuova, fino a riprodurre la campagna di valutazione della tesi.
Per la descrizione del sistema vedi il [README](README.md); per l'organizzazione
delle cartelle vedi [docs/struttura.md](docs/struttura.md).

Indice:

1. [Architettura del deployment](#1-architettura-del-deployment)
2. [Requisiti](#2-requisiti)
3. [Scenario A — server Linux (consigliato)](#3-scenario-a--server-linux-consigliato)
4. [Scenario B — Docker](#4-scenario-b--docker)
5. [Scenario C — solo golden model e test, senza toolchain](#5-scenario-c--solo-golden-model-e-test-senza-toolchain)
6. [Verifica dell'installazione](#6-verifica-dellinstallazione)
7. [Primo utilizzo](#7-primo-utilizzo)
8. [Riprodurre la valutazione della tesi](#8-riprodurre-la-valutazione-della-tesi)
9. [Configurazione](#9-configurazione)
10. [Aggiornamento e disinstallazione](#10-aggiornamento-e-disinstallazione)
11. [Risoluzione dei problemi](#11-risoluzione-dei-problemi)

---

## 1. Architettura del deployment

mxagent è un'applicazione a riga di comando, senza servizi di rete propri. I
componenti a runtime sono tre:

```
┌──────────────────────────── macchina di lavoro (Linux) ───────────────────────────┐
│                                                                                   │
│  mxagent (Python 3.12, venv)                                                      │
│    ├─ agenti: Planner, Coder, Fixer, Reflector, PPA Optimizer ──HTTP──┐           │
│    ├─ verifier: sbt + Chisel 6.6 + firtool 1.62.1 → Verilator + cocotb│           │
│    └─ sintesi: Yosys (generic/ECP5/Xilinx 7), nextpnr-ecp5            │           │
│                 [opz.] OpenSTA, Vivado                                │           │
└───────────────────────────────────────────────────────────────────────┼───────────┘
                                                                        ▼
                                          Ollama (porta 11434), sulla macchina con la GPU
                                          (può essere la stessa macchina o un'altra)
```

- Tutti gli strumenti hardware (JDK, sbt, OSS CAD Suite, firtool, venv) vengono
  installati **nella home dell'utente**, senza `sudo`, da
  [`scripts/setup_server.sh`](scripts/setup_server.sh).
- **Ollama** è l'unico servizio esterno. mxagent lo raggiunge via HTTP
  all'indirizzo `$OLLAMA_HOST` (default `http://localhost:11434`).
- Dopo il primo setup le cache di sbt/Chisel/firtool sono popolate e la parte
  hardware funziona **offline**. Serve rete solo per il setup e per scaricare i
  modelli con Ollama.

## 2. Requisiti

### Sistema operativo e architettura

| | supportato |
|---|---|
| Linux x86_64 | ✅ (firtool viene scaricato già compilato) |
| Linux aarch64 (es. NVIDIA DGX Spark) | ✅ (firtool viene compilato dai sorgenti, vedi sotto) |
| Windows | tramite **WSL 2** (Ubuntu), come Linux x86_64 |
| macOS | non supportato dallo script di setup |

### Software già presente sulla macchina

`bash`, `curl`, `tar`, `gzip`, `git`, `g++` e `make` (Verilator compila il
modello C++ della simulazione). Tutto il resto lo installa lo script.

Solo su **aarch64**, per compilare firtool: un compilatore C/C++ (gcc/g++ ≥ 7.4
o clang ≥ 5) e circa **20 GB** di spazio temporaneo. La compilazione richiede
tipicamente 20–60 minuti.

### Risorse

| risorsa | indicazione |
|---|---|
| disco (toolchain + cache) | alcuni GB in `~/.local/mxagent` e `~/.cache/coursier` |
| disco (modelli Ollama) | ~9 GB per `qwen2.5-coder:14b`, ~20 GB per `qwen2.5-coder:32b` (quantizzazione di default) |
| GPU | consigliata per Ollama; con `qwen2.5-coder:32b` serve una GPU con memoria sufficiente a contenere il modello (≈ 20 GB+). Senza GPU Ollama usa la CPU ed è molto lento |
| CPU / RAM | la verifica (sbt, Verilator) e la sintesi (Yosys, nextpnr) girano su CPU; più core accelerano `--parallel` nelle campagne |

### Versioni bloccate

Unica fonte: [`mxagent/versions.py`](mxagent/versions.py). Lo script di setup e
`mxagent doctor` le leggono da lì.

| componente | versione |
|---|---|
| Python | 3.12 (venv gestito da `uv`) |
| JDK | 17 (accettati 11–21) |
| Scala / sbt | 2.13.14 / 1.10.1 |
| Chisel | 6.6.0 |
| firtool (CIRCT) | 1.62.1 — un firtool diverso cambia il Verilog emesso |
| Verilator | ≥ 5.6 (≥ 5.36 con cocotb 2.x) |
| Yosys, nextpnr-ecp5, Icarus | ultima release di OSS CAD Suite |

### Modelli Ollama

Con la configurazione di default ([`configs/default.yaml`](configs/default.yaml)):

| modello | ruoli |
|---|---|
| `qwen2.5-coder:32b` | planner, reflector, escalate, optimizer |
| `qwen2.5-coder:14b` | coder, fixer |
| `nomic-embed-text` | embedding per il RAG (se manca, ripiego TF-IDF) |

Per la configurazione di sviluppo su CPU ([`configs/local_cpu.yaml`](configs/local_cpu.yaml))
servono invece `qwen2.5-coder:7b`, `qwen2.5-coder:14b` e `nomic-embed-text`.

---

## 3. Scenario A — server Linux (consigliato)

È lo scenario usato per la tesi: un server Linux headless raggiunto via SSH.

### 3.1 Ottenere il codice

Dal repository:

```bash
git clone https://github.com/Mattiarco/mxagent.git
cd mxagent
```

oppure dall'archivio consegnato:

```bash
unzip mxagent.zip && cd mxagent
```

### 3.2 Installare Ollama (sulla macchina con la GPU)

Se Ollama è già installato e in esecuzione, passa al punto 3.3.

Con permessi di amministratore si usa l'installer ufficiale (<https://ollama.com/download>).
**Senza sudo** (x86_64; su aarch64 sostituisci `amd64` con `arm64`):

```bash
mkdir -p ~/.local/mxagent/ollama
curl -fL https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C ~/.local/mxagent/ollama
tmux new -d -s ollama '~/.local/mxagent/ollama/bin/ollama serve'
export PATH="$HOME/.local/mxagent/ollama/bin:$PATH"
```

`ollama serve` deve restare acceso per tutto il lavoro: per questo va in `tmux`
o `screen`, oppure in un servizio systemd se si hanno i permessi.

### 3.3 Installare la toolchain

```bash
bash scripts/setup_server.sh \
    --models qwen2.5-coder:14b,qwen2.5-coder:32b,nomic-embed-text \
    --add-to-bashrc
source ~/.local/mxagent/env.sh
```

Lo script è **idempotente**: se si interrompe, basta rilanciarlo e riprende
saltando ciò che è già installato. In ordine:

1. Coursier → JDK 17 e sbt 1.10.1;
2. OSS CAD Suite (Yosys, Verilator, nextpnr-ecp5, Icarus) in `~/.local/mxagent/oss-cad-suite`;
3. firtool 1.62.1 — solo su aarch64, compilato dai sorgenti con verifica sha256;
4. `env.sh`, che imposta `JAVA_HOME`, `PATH` e, se serve, `CHISEL_FIRTOOL_PATH`;
5. venv Python 3.12 con `uv` e installazione di mxagent con gli extra `[sim,eval,dev]`;
6. `ollama pull` dei modelli indicati con `--models`, se Ollama è raggiungibile;
7. warm-up: elabora un modulo di prova, popola le cache e controlla la versione di firtool.

Opzioni utili:

| opzione | effetto |
|---|---|
| `--prefix DIR` | installa in `DIR` invece di `~/.local/mxagent` (es. su `/scratch`) |
| `--models a,b,c` | modelli da scaricare con `ollama pull` |
| `--add-to-bashrc` | aggiunge `source <prefix>/env.sh` a `~/.bashrc` |
| `--skip-warmup` | salta l'elaborazione di prova |
| `--firtool-path DIR` | usa un firtool 1.62.1 già compilato (cartella del binario) |
| `--build-firtool` | compila firtool anche su x86_64 |
| `--keep-build` | conserva la cartella di build di firtool |

Variabili d'ambiente dello script: `MXAGENT_BUILD_DIR` (dove compilare firtool,
~20 GB), `MXAGENT_BUILD_JOBS` (job paralleli), `OSS_CAD_SUITE_URL` (forza una
release precisa di OSS CAD Suite), `MXAGENT_JDK`, `CC`/`CXX`.

In alternativa: `make setup` equivale a `bash scripts/setup_server.sh` senza opzioni.

### 3.4 Ollama su un'altra macchina

Se la GPU è su un host diverso da quello dove gira mxagent:

```bash
# sull'host con la GPU: Ollama deve ascoltare anche sulle interfacce esterne
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# sulla macchina di mxagent
export OLLAMA_HOST=http://<host-gpu>:11434
```

Se la porta 11434 non è esposta in rete, usa un tunnel SSH:

```bash
ssh -N -L 11434:localhost:11434 utente@host-gpu   # poi OLLAMA_HOST=http://localhost:11434
```

L'host si può fissare anche nella configurazione (`llm.host` in `configs/*.yaml`).

### 3.5 Vivado e OpenSTA (facoltativi)

Non sono necessari: i backend di sintesi validati sono Yosys (generic, ECP5,
Xilinx 7) e nextpnr. Per abilitare gli altri:

- **Vivado**: `source /opt/Xilinx/Vivado/<versione>/settings64.sh` prima di
  usare mxagent; `vivado` nel `PATH` viene rilevato e usato in batch mode;
- **OpenSTA**: compilare `parallaxsw/OpenSTA`, mettere `sta` nel `PATH` e
  impostare `synth.liberty` con una libreria `.lib` (es. Nangate45, sky130).

Entrambi i backend sono **non validati** (vedi [docs/ppa.md](docs/ppa.md)).

---

## 4. Scenario B — Docker

L'immagine usa lo stesso script di setup su Ubuntu 24.04. Vivado non è incluso
(non è redistribuibile) e **Ollama resta sull'host**.

```bash
docker build -f docker/Dockerfile -t mxagent .      # oppure: make docker
```

Sull'host, Ollama deve accettare connessioni dal container:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Esecuzione (l'entrypoint è `mxagent`, il comando di default è `doctor`):

```bash
docker run --rm -it --add-host=host.docker.internal:host-gateway \
    -v "$PWD/runs:/work/runs" mxagent doctor

docker run --rm -it --add-host=host.docker.internal:host-gateway \
    -v "$PWD/runs:/work/runs" mxagent \
    run --task benchmarks/b02_e2m1_mul --mode full --seed 0
```

- Nel container `OLLAMA_HOST` vale `http://host.docker.internal:11434`. Per un
  Ollama remoto: `-e OLLAMA_HOST=http://<host>:11434`.
- Monta come volumi le cartelle dei risultati, altrimenti si perdono alla fine
  del container: `-v "$PWD/runs:/work/runs"` per le run,
  `-v "$PWD/evals:/work/evals"` per le campagne.

---

## 5. Scenario C — solo golden model e test, senza toolchain

Per consultare il golden model o eseguire i test che non richiedono
sbt/Verilator/Ollama (ad esempio su un portatile, per ispezionare il codice):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
mxagent golden dot --a 2 --b 7 --a-scale 130    # atteso: 0x44C00000 (1536.0)
make test-fast
```

I test d'integrazione che richiedono la toolchain vengono **saltati**
automaticamente (`skipped`): non è un errore.

---

## 6. Verifica dell'installazione

```bash
source ~/.local/mxagent/env.sh
mxagent doctor              # toolchain + Ollama + modelli della config
mxagent doctor --warmup     # in più: elaborazione Chisel di prova
make test                   # test completi (quelli d'integrazione usano la toolchain)
```

`mxagent doctor` stampa una riga per controllo:

- `[OK ]` — tutto a posto;
- `[ERR]` — componente **obbligatorio** mancante o di versione sbagliata. Sotto
  compare `->` con il comando per risolvere. Il comando esce con codice 1;
- `[ - ]` — componente **facoltativo** assente (Yosys, nextpnr, OpenSTA, Vivado,
  GPU): si disattiva solo la fase corrispondente.

Obbligatori: Python ≥ 3.10 con `numpy`, `pyyaml`, `pydantic`, `cocotb`; Java;
sbt; firtool 1.62.1 (solo dove va fornito a mano); Verilator; `g++`; `make`;
Ollama raggiungibile con i modelli della config.

Per controllare un'altra configurazione: `mxagent doctor --config configs/local_cpu.yaml`.
Per controllare solo la toolchain hardware: `mxagent doctor --skip-llm`.

Controllo end-to-end consigliato, che non usa l'LLM:

```bash
mxagent verify --task benchmarks/b02_e2m1_mul    # verifica il riferimento: deve passare
mxagent synth  --task benchmarks/b05_mxfp4_dot32 # sintesi e PPA del riferimento
```

---

## 7. Primo utilizzo

```bash
# una run del sistema completo su un benchmark
mxagent run --task benchmarks/b02_e2m1_mul --mode full --seed 0

# baseline, forzando un modello per tutti i ruoli
mxagent run --task benchmarks/b01_e2m1_decode --mode baseline-b --model qwen2.5-coder:14b

# ablation: spegnere un meccanismo
mxagent run --task benchmarks/b05_mxfp4_dot32 --set agent.kb=false

# PPA del design prodotto da una run
mxagent synth --run runs/<id> --backends generic,ecp5,nextpnr

# coprocessore RISC-V CV-X-IF attorno al MAC (b08)
mxagent wrap
mxagent wrap --run runs/<id-di-una-run-b08>
```

Ogni run scrive `runs/<id>/` (o `$MXAGENT_RUNS/<id>/`) con configurazione
effettiva, trace delle chiamate LLM, candidati, design finale (Chisel e
Verilog), report del verifier, metriche e PPA. Il contenuto dei file è descritto
nel [README](README.md#uso).

Le risposte dell'LLM sono in cache in `.cache/llm`: rieseguire una run con gli
stessi parametri dà lo stesso risultato senza richiamare il modello.

---

## 8. Riprodurre la valutazione della tesi

La campagna principale è [`configs/eval/main.yaml`](configs/eval/main.yaml):
8 benchmark × 3 modalità × 2 modelli × 10 seed, più 9 ablation su 3 benchmark,
per un totale di **750 job**.

```bash
# 0. prova dell'harness su CPU (un task, un seed, modelli piccoli)
mxagent eval --campaign configs/eval/smoke.yaml

# 1. elenco dei job, senza eseguirli
mxagent eval --campaign configs/eval/main.yaml --dry-run

# 2. esecuzione: conviene lanciarla in tmux, è lunga
tmux new -s eval
mxagent eval --campaign configs/eval/main.yaml --parallel 2

# 3. ripresa dopo un'interruzione: stesso comando, i job completati vengono saltati
mxagent eval --campaign configs/eval/main.yaml
mxagent eval --campaign configs/eval/main.yaml --retry-errors   # rifà errori/timeout

# 4. report (funziona anche su una campagna parziale)
mxagent report --dir evals/main
```

Il report produce in `evals/main/`:

- `report.md` — tabelle di successo con intervalli di Wilson, pass@k, confronti
  appaiati con test di McNemar, efficienza, ablation, PPA;
- `runs.csv` — una riga per job;
- `summary.json` — gli stessi numeri in formato leggibile da programma.

Ogni job sta in `evals/main/runs/<job>/`. `index.jsonl` registra lo stato di
ogni job e `campaign.json` la campagna con il suo hash. I dettagli sono in
[docs/eval.md](docs/eval.md).

Note sulla riproducibilità:

- il seed viene passato a Ollama e le risposte vengono messe in cache;
- `meta.json` di ogni run registra il digest del modello, l'hash dei prompt e le
  versioni della toolchain. Un modello con digest diverso (per esempio
  aggiornato con `ollama pull`) può dare risultati diversi;
- `--parallel N` sovrappone LLM e verifica/sintesi; Ollama serializza comunque
  le richieste sulla stessa GPU.

---

## 9. Configurazione

### File

| file | uso |
|---|---|
| `configs/default.yaml` | configurazione della tesi: modelli per ruolo, meccanismi dell'agente, budget, simulazione, sintesi, semantica numerica |
| `configs/local_cpu.yaml` | sviluppo su CPU/WSL: modelli piccoli, budget di tempo più largo |
| `configs/eval/main.yaml` | campagna di valutazione della tesi |
| `configs/eval/smoke.yaml` | prova rapida dell'harness |

Si sceglie con `--config` e si sovrascrive al volo con `--set chiave=valore`
(es. `--set budget.max_llm_calls=30 synth.clock_ns=20`).

### Variabili d'ambiente

| variabile | significato | default |
|---|---|---|
| `OLLAMA_HOST` | indirizzo del server Ollama | `http://localhost:11434` |
| `MXAGENT_RUNS` | cartella delle run | `runs` (`runs_dir` della config) |
| `MXAGENT_WORK` | cartella di lavoro di `verify`/`synth`/`wrap` | `runs/verify` |
| `CHISEL_FIRTOOL_PATH` | cartella del binario firtool 1.62.1 (impostata da `env.sh` su aarch64) | — |
| `JAVA_HOME`, `PATH`, `COURSIER_CACHE` | impostate da `env.sh` | — |

### Cartelle create a runtime

Escluse da git (vedi `.gitignore`):

| cartella | contenuto |
|---|---|
| `runs/` | output delle run e dei comandi `verify`/`synth`/`wrap` |
| `evals/` | campagne di valutazione |
| `library/` | libreria di componenti verificati riusati dall'agente |
| `.cache/llm/` | cache delle risposte dell'LLM |
| `chisel/target/`, `chisel/generated/`, `chisel/build/` | build sbt e sorgenti generati |
| `~/.local/mxagent/` | toolchain, venv, `env.sh` |
| `~/.cache/coursier/` | dipendenze Maven/Scala, riusate offline |

---

## 10. Aggiornamento e disinstallazione

Aggiornamento del codice:

```bash
git pull
source ~/.local/mxagent/env.sh
uv pip install --python ~/.local/mxagent/venv/bin/python -e '.[sim,eval,dev]'   # o rilancia setup_server.sh
mxagent doctor
```

Se cambiano le versioni in `mxagent/versions.py`, rilancia
`bash scripts/setup_server.sh`.

Pulizia degli artefatti di build: `make clean`.

Disinstallazione completa:

```bash
rm -rf ~/.local/mxagent ~/.cache/coursier
# rimuovi la riga "source ~/.local/mxagent/env.sh" da ~/.bashrc
ollama rm qwen2.5-coder:14b qwen2.5-coder:32b nomic-embed-text   # facoltativo
```

---

## 11. Risoluzione dei problemi

| sintomo | causa probabile | soluzione |
|---|---|---|
| `mxagent: command not found` | `env.sh` non caricato | `source ~/.local/mxagent/env.sh` (o `--add-to-bashrc` nel setup) |
| `doctor`: `ollama ... non raggiungibile` | `ollama serve` spento o host sbagliato | avvia `ollama serve` (in tmux); controlla `OLLAMA_HOST`; da Docker, Ollama deve ascoltare su `0.0.0.0` |
| `doctor`: `modello X` in errore | modello non scaricato | `ollama pull X` |
| `doctor`: firtool in errore su aarch64 | firtool non compilato o versione diversa | rilancia `setup_server.sh` (la build riprende), oppure `--firtool-path` |
| compilazione di firtool: spazio insufficiente | servono ~20 GB | `MXAGENT_BUILD_DIR=/scratch/... bash scripts/setup_server.sh` |
| compilazione di firtool: manca il compilatore | niente gcc/clang | senza sudo: `conda install -c conda-forge gxx`, poi rilancia |
| `'uint64_t' was not declared` (GCC ≥ 15) | header mancante in LLVM 18 | già gestito dallo script (`-include cstdint`); verifica di usare lo script aggiornato |
| URL di OSS CAD Suite non trovato | limite delle API GitHub o release senza asset | imposta `OSS_CAD_SUITE_URL` con l'URL di un `.tgz` preciso |
| warm-up Chisel fallito | rete assente al primo avvio o JDK fuori range | controlla la rete per il primo setup, `java -version` (11–21); log indicato da `doctor` |
| `cocotb+verilator` in errore | Verilator troppo vecchio per la versione di cocotb | aggiorna OSS CAD Suite (rimuovi `~/.local/mxagent/oss-cad-suite` e rilancia il setup) |
| run lentissime | Ollama su CPU | controlla `nvidia-smi` e la riga `GPU NVIDIA` di `doctor`; in alternativa usa `configs/local_cpu.yaml` |
| run interrotte per budget | limiti in `budget:` | alza `--set budget.max_wall_s=... budget.max_llm_calls=...` |
| risultati diversi tra due macchine | digest del modello o versione di firtool diversi | confronta `meta.json` delle due run |
