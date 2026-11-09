# Valutazione (M6)

```bash
mxagent eval --campaign configs/eval/main.yaml --dry-run   # elenco e numero dei job
mxagent eval --campaign configs/eval/main.yaml             # esegue o riprende
mxagent eval --campaign configs/eval/main.yaml --retry-errors
mxagent report evals/main                                  # report, anche di una campagna parziale
```

## Campagna da riga di comando (un seed, budget rigidi)

```bash
mxagent eval --tasks b04_mxfp4_quantize,b05_mxfp4_dot32   --configs full,baseline_a,baseline_b --model-profiles all14b,all32b --seeds 0   --campaign-budget-s 21600 --out runs/eval_<data> [--dry-run]
mxagent report runs/eval_<data>    # summary.md, summary.csv, notes.md, pass_rate.png
```

Senza YAML la campagna usa:
- per run `max_wall_s` 1800, `max_llm_calls` 40, `max_tokens` 300000 (`--max-wall-s`,
  `--max-llm-calls`, `--max-tokens`); oltre `max_wall_s` non parte più né una chiamata
  LLM né una verifica, e lo stop è in `result.json` (`reason: "budget: ..."`);
- PPA solo nelle run `full`, al più `--ppa-iters` (2) tentativi con l'arresto
  anticipato `plateau`; le baseline non hanno PPA. Ogni run passata, di qualunque
  modalità, ha la sintesi finale su `--synth-backends` (`eval_synth/ppa.json`);
- libreria `frozen` in sola lettura (`--library`, default `library/`) ed
  `exclude_current_task: true`: nessuna run impara da un'altra della campagna;
- profili di modello (`--model-profiles`) da `llm.profiles` della config;
- esecuzione in sequenza nell'ordine profilo → task → modalità, così se il budget
  finisce resta un profilo completo;
- warm-up dei modelli in Ollama prima del primo job di ogni profilo, fuori dal
  conteggio del tempo (`warmup.jsonl`; `--no-warmup` per toglierlo);
- budget di campagna (`--campaign-budget-s`): somma dei tempi dei job (sintesi
  inclusa, warm-up escluso). Esaurito il budget, i job rimanenti sono registrati in
  `index.jsonl` come `skipped_budget` senza eseguirli; il job in corso finisce.
  Rilanciando lo stesso comando i job `skipped_budget` vengono eseguiti;
- timeout per job `max_wall_s` + 900 s: SIGTERM (la run scrive `result.json` come
  `interrupted`), SIGKILL dopo altri 120 s.

Il dry-run stampa i job nell'ordine di esecuzione, i budget e un limite superiore del
tempo (numero di run × `max_wall_s`), non una previsione.

`mxagent report <cartella>` scrive sempre, da `result.json`, `index.jsonl` ed
`eval_synth/ppa.json` di ogni run ([summary.py](../mxagent/eval/summary.py)):
- `summary.csv` e `summary.md`: una riga per run (task, configurazione, profilo,
  `status`, pass e pass rate dei vettori, `candidates_to_pass`, `reflect_rounds`,
  tempo al pass e totale, chiamate e token in/out, LUT/FF xc7 ed ecp5, Fmax media di
  nextpnr, ADP generic, guadagno PPA per `full`) e una tabella per configurazione e
  profilo (passate/eseguite, mediane di tempo e token);
- `pass_rate.png`: pass rate dei vettori per task × configurazione × profilo
  (richiede matplotlib: `pip install -e .[eval]`);
- `notes.md`: per ogni run non passata stadio di fallimento e primo errore
  (`first_error` di `result.json`).

Con un solo seed (`seeds: 1`) il report dichiara n = 1 e non calcola pass@k,
intervalli di confidenza né test; con più seed si aggiungono `report.md`,
`stats.csv`, `runs.csv` e `summary.json` descritti sotto.

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
  Se la run ha lasciato `result.json` (es. `interrupted` per SIGHUP), il suo
  stato è riportato in `result_status`.
- **Sintesi:** ogni design passato, di qualunque modalità, viene sintetizzato
  con `ppa_backends` (default `generic`) in `runs/<job>/eval_synth/`. Il
  riferimento di ogni task viene sintetizzato una volta, in `reference/<task>/`.
- **Parallelismo:** `--parallel N` esegue N job insieme. Ogni job ha il suo
  workspace sbt. Ollama serializza le richieste sulla stessa GPU, quindi il
  guadagno sta soprattutto nella sovrapposizione fra LLM e verifica/sintesi.

## Report

[report.py](../mxagent/eval/report.py) scrive `report.md`, `runs.csv` (una riga
per job), `stats.csv` (una riga per task, modelli e modalità) e `summary.json`. Ogni numero viene da `metrics.json`, `meta.json` e
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
| Media ± dev. std | per task, modelli e modalità: pass@1/5/10; pass rate dei vettori e token totali (tutte le run); iterazioni, tempo e token al pass, ADP, LUT/FF xc7 ed Fmax ECP5 (run passate). Per task e non aggregate: area ed Fmax di task diversi non si mediano. Le stesse righe, con LUT/FF ecp5, sono in `stats.csv` |
| Errori | job in errore o timeout, esclusi dai tassi |

Con 10 seed per cella gli intervalli sono larghi: per esempio 5/10 dà un
intervallo di Wilson del 24–76%. Le conclusioni vanno tratte soprattutto dai
totali e dai confronti appaiati.

## Prova su CPU

[smoke.yaml](../configs/eval/smoke.yaml) esegue b01 con un seed, le tre modalità
e un'ablation, usando `configs/local_cpu.yaml`. Serve a controllare l'harness
end-to-end, non a misurare.
