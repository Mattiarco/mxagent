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
- modalità (`baseline-a`, `baseline-b`, `baseline-b-matched`, `full`);
- numero di seed;
- insiemi di modelli;
- ablation;
- politica della libreria;
- backend di sintesi per i design passati.

Ogni job è una combinazione (task, modalità, variante, modelli, seed) e gira
come sottoprocesso `mxagent run` in `<campagna>/runs/<job>/`.

La campagna principale ha 910 job:
- 8 task × 4 modalità × 2 modelli × 10 seed = 640 job principali;
- 9 ablation × 3 task × 1 modello × 10 seed = 270 job di ablation.

**Equità del confronto.**
- **Stesso budget, non stesso numero di tentativi.** Tutti i job ricevono lo
  stesso `budget` della config di base (`max_llm_calls`, `max_tokens`,
  `max_wall_s`) come *tetto*. Quanti tentativi una modalità fa davvero dipende però
  dal suo ciclo:

  | modalità | si ferma a | chiamate al più (default) |
  |---|---|---|
  | `baseline-a` | una generazione | 1 |
  | `baseline-b` | pass, `1 + budget.max_fix_iters` chiamate o budget | 7 |
  | `baseline-b-matched` | pass o budget (`max_fix_iters = max_llm_calls - 1`) | 60 |
  | `full` | pass o budget (`max_llm_calls`, token, tempo) | 60 |

  Quindi `full` contro `baseline-b` **non** è a parità di tentativi: `full` può
  fare fino a 60 chiamate, B al più 7. Il confronto a pari costo è fra `full` e
  `baseline-b-matched` (stesso Budget, stessi limiti) oppure, a posteriori, la
  sezione del report "Successo a parità di chiamate" (run passate con al più k
  chiamate, k = 7, 15, 30, 60, sugli stessi job). `baseline-b` resta invariata
  perché i risultati già raccolti restino validi.

  Cosa resta diverso anche con `baseline-b-matched`: la verifica (compilazione e
  simulazione) non è nel budget, salvo il tempo; i token per chiamata sono
  diversi (`full`: prompt con scheda di dominio, regole, esempi; B: storia che
  cresce a ogni turno fino a `num_ctx`), quindi `max_tokens` può scattare a numeri
  di chiamate diversi (lo stop è in `result.json`, `reason: "budget: ..."`); la
  baseline B tiene la storia multi-turn
  completa e, quando supera `num_ctx`, toglie le coppie (risposta, feedback) più
  vecchie, tenendo sempre sistema, primo messaggio utente e ultimo feedback. Quante
  coppie sono state tolte e in quante chiamate è in `history_truncations` e
  `history_truncated_calls` di `result.json`/`metrics.json` (eventi
  `history_truncated` nel trace): con ~60 turni il troncamento è la condizione
  normale della baseline e va riportato.
- **Stesso modello:** con `all: <modello>` baseline e sistema completo usano lo
  stesso modello. Il routing piccolo/grande e il passaggio al modello più grande
  dell'anti-loop sono quindi neutri in questa campagna. Per studiarli serve un
  insieme di modelli con `roles:` diversi; la baseline usa allora `baseline:`
  (default: il modello del coder).
- **Costo fino al pass:** `llm_calls_to_pass` in `result.json` è il numero di
  chiamate al modello fino al primo candidato del design di primo livello che
  passa, compresa quella che lo ha generato, contando tutti i ruoli (spec,
  planner, coder, fixer, reflector, integrate). I pass dei blocchi della
  decomposizione non chiudono il conteggio; le chiamate dopo il pass (PPA
  Optimizer) non contano. Si ricava dagli eventi `llm` del trace che precedono il
  primo evento `verify` passato del top; `None` se la run non passa. `llm_calls`
  resta il totale della run. Per le run precedenti al campo il report lo ricostruisce
  da `trace.json` con la stessa funzione (`calls_to_pass_src` in `runs.csv`).
- **Niente leakage:** con la politica `fresh`, la default, ogni run del sistema
  completo parte da una libreria vuota tutta sua. Il riuso avviene solo dentro la
  run, quindi i risultati non dipendono dall'ordine dei task né dal
  parallelismo. Con `frozen` si usa una libreria costruita prima, in sola
  lettura; i componenti del task in esame sono comunque esclusi.
- **Riproducibilità e cache:**
  - il seed è passato a Ollama;
  - le risposte sono in cache (`llm.cache_dir`, default `.cache/llm`), con chiave
    richiesta completa + seed + digest del modello. Rieseguire un job già eseguito
    (anche in un'altra campagna sulla stessa macchina) riprende le risposte: la run
    dura secondi e **non è un risultato nuovo**. `result.json` registra
    `llm_cached_calls`, `llm_cached_fraction` e `replayed` (vero se *tutte* le
    chiamate della run vengono dalla cache). Il report esclude le run replayed dai
    tassi principali, le elenca in una sezione a parte e avvisa (su stderr e in
    testa a `report.md`/`summary.md`) senza fallire. Le run con solo una parte
    delle chiamate in cache restano nei totali e sono elencate con la frazione;
  - per rieseguire senza cache: `mxagent run --no-llm-cache` (equivale a
    `--set llm.cache_dir=null`), oppure nella campagna `llm_cache: off` o
    `mxagent eval --llm-cache off`. Con `llm_cache: campaign` la cache è
    `<campagna>/llm_cache`: campagne diverse non condividono risposte. Il default
    `shared` è il comportamento di prima. La cache esistente non viene cancellata;
  - `meta.json` di ogni run registra il digest del modello, l'hash dei prompt e
    le versioni degli strumenti;
  - `campaign.json` registra la campagna e il suo hash.

- **Task esclusi dal totale (`report_exclude`).** La campagna può togliere dei task
  dai totali principali con un motivo, es. `report_exclude: {b01_e2m1_decode:
  card_leak}` in [main.yaml](../configs/eval/main.yaml): in `full` la scheda di
  dominio contiene la tabella dei 16 codici E2M1, che per b01 è la soluzione,
  mentre le baseline non la ricevono. Il task ha una sezione sua nel report ("b01:
  card contiene la soluzione") e il report dice su quali task sono calcolati i
  totali. Per una campagna già eseguita: `mxagent report --dir <campagna>
  --exclude b01_e2m1_decode:card_leak` (si somma a `report_exclude`). Non cambia i
  job né l'hash della campagna.

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
| Successo a parità di chiamate | per k = 7, 15, 30, 60: run passate con `llm_calls_to_pass` ≤ k, per modalità, sugli stessi job appaiati (task, seed, modelli); curva completa per k = 1..kmax in `calls_curve.csv` |
| Efficienza | mediane sulle run passate: chiamate LLM (`llm_calls_to_pass`) e token fino al pass, tempo, iterazioni |
| Fallimenti | fase del miglior candidato delle run fallite (compile / lint / sim / nessun codice), run fermate dal budget |
| Ablation | ogni variante contro `full` sugli stessi task e seed: tassi, Δ, McNemar |
| Vincoli di sistema (M7) | b06: target di frequenza rispettato (Fmax ECP5 peggiore sui seed ≥ target); b08: test di protocollo CV-X-IF superato col MAC generato |
| Qualità dei risultati | transistori e profondità (Yosys generic) dei design passati rispetto al riferimento (media geometrica), tentativi e guadagno del PPA Optimizer |
| Media ± dev. std | per task, modelli e modalità: pass@1/5/10; pass rate dei vettori e token totali (tutte le run); iterazioni, tempo e token al pass, ADP, LUT/FF xc7 ed Fmax ECP5 (run passate). Per task e non aggregate: area ed Fmax di task diversi non si mediano. Le stesse righe, con LUT/FF ecp5, sono in `stats.csv` |
| Successo a parità di chiamate, tutte le run | stessa tabella con task esclusi e run replayed compresi: solo controllo |
| `<task>: <motivo>` | per ogni task in `report_exclude`: tassi propri, senza e con le run replayed |
| Run replayed | quante e quali (task, modalità, seed), escluse dai tassi; run con cache parziale e frazione |
| Errori | job in errore o timeout, esclusi dai tassi |

Con 10 seed per cella gli intervalli sono larghi: per esempio 5/10 dà un
intervallo di Wilson del 24–76%. Le conclusioni vanno tratte soprattutto dai
totali e dai confronti appaiati.

## Prova su CPU

[smoke.yaml](../configs/eval/smoke.yaml) esegue b01 con un seed, le tre modalità
e un'ablation, usando `configs/local_cpu.yaml`. Serve a controllare l'harness
end-to-end, non a misurare.
