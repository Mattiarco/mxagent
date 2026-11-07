# Valutazione (M6)

```bash
mxagent eval --campaign configs/eval/main.yaml --dry-run   # elenco e numero dei job
mxagent eval --campaign configs/eval/main.yaml             # esegue o riprende
mxagent eval --campaign configs/eval/main.yaml --retry-errors
mxagent report --dir evals/main                            # report, anche di una campagna parziale
```

## Campagna

Un YAML ([main.yaml](../configs/eval/main.yaml)) dichiara:
- task;
- modalità (`baseline-a`, `baseline-b`, `full`);
- numero di seed;
- insiemi di modelli;
- ablation;
- politica della libreria;
- backend di sintesi per i design passati.

Ogni job è una combinazione (task, modalità, variante, modelli, seed) e gira
come sottoprocesso `mxagent run` in `<campagna>/runs/<job>/`.

La campagna principale ha 750 job:
- 8 task × 3 modalità × 2 modelli × 10 seed = 480 job principali;
- 9 ablation × 3 task × 1 modello × 10 seed = 270 job di ablation.

**Equità del confronto.**
- **Stesso budget:** tutti i job usano lo stesso `budget` della config di base
  (chiamate LLM, token, tempo).
- **Stesso modello:** con `all: <modello>` baseline e sistema completo usano lo
  stesso modello. Il routing piccolo/grande e il passaggio al modello più grande
  dell'anti-loop sono quindi neutri in questa campagna. Per studiarli serve un
  insieme di modelli con `roles:` diversi; la baseline usa allora `baseline:`
  (default: il modello del coder).
- **Costo fino al pass:** le chiamate dello stadio PPA, che avviene dopo il pass,
  non contano. Il report usa `llm_to_pass` per il sistema completo e `llm` per le
  baseline, che si fermano al pass.
- **Niente leakage:** con la politica `fresh`, la default, ogni run del sistema
  completo parte da una libreria vuota tutta sua. Il riuso avviene solo dentro la
  run, quindi i risultati non dipendono dall'ordine dei task né dal
  parallelismo. Con `frozen` si usa una libreria costruita prima, in sola
  lettura; i componenti del task in esame sono comunque esclusi.
- **Riproducibilità:**
  - il seed è passato a Ollama;
  - le risposte sono in cache;
  - `meta.json` di ogni run registra il digest del modello, l'hash dei prompt e
    le versioni degli strumenti;
  - `campaign.json` registra la campagna e il suo hash.

## Runner

[runner.py](../mxagent/eval/runner.py):
- **Indice:** `index.jsonl` è append-only e registra per ogni job:
  - lo stato: `done`, `error` o `timeout`;
  - la durata;
  - un estratto dell'errore.

  Una campagna rilanciata salta i job completati.
- **Errori di infrastruttura:** crash e timeout del sottoprocesso sono tenuti
  distinti dai fallimenti del modello. Il report li elenca e li esclude dai tassi.
- **Sintesi:** ogni design passato, di qualunque modalità, viene sintetizzato
  con `ppa_backends` (default `generic`) in `runs/<job>/eval_synth/`. Il
  riferimento di ogni task viene sintetizzato una volta, in `reference/<task>/`.
- **Parallelismo:** `--parallel N` esegue N job insieme. Ogni job ha il suo
  workspace sbt. Ollama serializza le richieste sulla stessa GPU, quindi il
  guadagno sta soprattutto nella sovrapposizione fra LLM e verifica/sintesi.

## Report

[report.py](../mxagent/eval/report.py) scrive `report.md`, `runs.csv` (una riga
per job) e `summary.json`. Ogni numero viene da `metrics.json`, `meta.json` e
`ppa.json` delle run.

| sezione | contenuto |
|---|---|
| Successo | k/n per task e modalità, totale con intervallo di Wilson al 95%, pass@k (stimatore non distorto, Chen et al. 2021) mediato sui task |
| Confronto appaiato | `full` contro ogni baseline sulle stesse coppie (task, seed): run vinte solo da A o solo da B, p-value esatto di McNemar |
| Efficienza | mediane sulle run passate: chiamate LLM e token fino al pass, tempo, iterazioni |
| Fallimenti | fase del miglior candidato delle run fallite (compile / lint / sim / nessun codice), run fermate dal budget |
| Ablation | ogni variante contro `full` sugli stessi task e seed: tassi, Δ, McNemar |
| Vincoli di sistema (M7) | b06: target di frequenza rispettato (Fmax ECP5 peggiore sui seed ≥ target); b08: test di protocollo CV-X-IF superato col MAC generato |
| Qualità dei risultati | transistori e profondità (Yosys generic) dei design passati rispetto al riferimento (media geometrica), tentativi e guadagno del PPA Optimizer |
| Errori | job in errore o timeout, esclusi dai tassi |

Con 10 seed per cella gli intervalli sono larghi: per esempio 5/10 dà un
intervallo di Wilson del 24–76%. Le conclusioni vanno tratte soprattutto dai
totali e dai confronti appaiati.

## Prova su CPU

[smoke.yaml](../configs/eval/smoke.yaml) esegue b01 con un seed, le tre modalità
e un'ablation, usando `configs/local_cpu.yaml`. Serve a controllare l'harness
end-to-end, non a misurare.
