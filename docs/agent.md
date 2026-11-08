# Sistema completo (`--mode full`)

```
                 ┌──────────┐   ┌─────────┐
 spec.md ───────▶│   SPEC   │──▶│  PLAN   │── monolithic ─────────────────────┐
 task.yaml       │ (JSON)   │   │ (JSON)  │                                   │
                 └──────────┘   └────┬────┘                                   ▼
                                     │ decompose                       ┌────────────┐
                                     ▼                                 │ MONOLITHIC │
                     ┌──────────────────────────────┐   fallimento ──▶│   solve    │
                     │ BLOCK i (catalogo MX)        │────────────────▶└─────┬──────┘
                     │  riuso esatto dalla libreria │                       │
                     │  oppure solve (golden blocco)│                       │
                     └──────────────┬───────────────┘                       │
                                    ▼                                       │
                             ┌─────────────┐   fallimento                   │
                             │  INTEGRATE  │──────────────▶ MONOLITHIC      │
                             │ top + blocchi│                               │
                             └──────┬──────┘                                │
                                    ▼                                       ▼
                                  PPA ◀─────────────────────────────────────┘
                     (sintesi + PPA Optimizer, solo dopo il pass: docs/ppa.md)
                                    ▼
                              DONE | FAILED
                     (componenti verificati salvati in library/)
```

## Fine della run: `result.json`

Ogni run termina con uno stato esplicito, scritto in un `try/finally`, e lascia
sempre `result.json` e un ultimo evento `end` nel trace:

| `status` | quando | `reason` |
|---|---|---|
| `done` | design bit-exact su tutti i vettori | `pass` |
| `failed` | nessun design corretto | `max_iters`, `escape_stop`, `budget: ...` |
| `failed` | eccezione non prevista | `exception` (dettaglio in `error`) |
| `interrupted` | SIGINT, SIGTERM o SIGHUP (es. terminale chiuso senza tmux) | `signal SIGHUP` |

`result.json` contiene:
- task, modalità, modello, seed;
- `passed`, `pass_rate`, `n_vectors` e `best_candidate` (tag della verifica);
- iterazioni e tempo fino alla correttezza;
- chiamate e token, in totale e per ruolo (`llm_by_role`, dagli eventi `llm` del trace);
- le metriche PPA iniziali e finali per backend, il guadagno e i tentativi per esito
  (accettati, rifiutati, falliti, duplicati).

Le baseline scrivono lo stesso file, con `ppa: null`.

Ogni uscita da un loop lascia un evento con il motivo:
- `loop_stop` (solver): `pass`, `max_iters` o `escape_stop`, con `iters_done` e `iters_max`;
- `ppa_stop` (optimizer): `iters_exhausted`, `budget: ...` o `no_primary_metric`;
- `budget_exhausted`: `limit` è `max_llm_calls`, `max_tokens` o `max_wall_s`, e
  `stage` è lo stato in cui il budget è finito.

`metrics.json` resta la fonte per `mxagent report` e si scrive solo se la run
arriva alla fine. Una run senza `metrics.json` ma con `result.json` in stato
`interrupted` o `failed` è stata interrotta o è andata in errore.

## Solve di un modulo (`mxagent/agents/solver.py`)

1. **Generazione best-of-N**: N campioni dal Coder (seed e temperature diversi).
   Prompt: regole Chisel 6 (KB verificata), scheda del formato e semantica numerica,
   specifica, interfaccia, fino a 2 esempi verificati dalla libreria (RAG).
2. **Sanitizer** deterministico (KB): API Chisel 3 → Chisel 6, `IO` → `FlatIO`,
   oggetti `main` rimossi, `import chisel3.util._` se serve.
3. **Verifier** (nessun LLM): compile → lint → cocotb → confronto bit-exact.
4. **Ordinamento**: stadio raggiunto, frazione di vettori corretti, lunghezza del codice.
5. **Riflessione** sempre sul *miglior* candidato (ritorno automatico al migliore), con
   feedback strutturato, trappole della KB richiamate dall'errore e memoria compatta
   degli ultimi tentativi. **Routing**: errori di compilazione/lint → `fixer`
   (modello piccolo); errori funzionali → `reflector` (modello grande).
6. **Anti-loop**: stesso errore ripetuto, oscillazione A/B/A/B o nessun miglioramento
   per K tentativi → *resample* (N campioni nuovi, temperatura +0.4) → *escalate*
   (modello `escalate` per tutti i ruoli) → *stop* (si tiene il migliore).

Ogni chiamata è a turno singolo: il contesto non cresce con le iterazioni.

## Decomposizione

Il Planner sceglie blocchi dal catalogo (`mxagent/bench/subkernels.py`), ognuno con
interfaccia fissa, golden e vettori: ogni sottomodulo generato è verificato da solo
(esaustivamente se lo spazio è piccolo) prima dell'integrazione. Un piano non valido
(JSON rotto, blocchi inesistenti, parametri sconosciuti) diventa monolitico, con i
motivi registrati in `plan.json`.

| blocco | test |
|---|---|
| `e2m1_decode`, `e2m1_mul` | esaustivi (16, 256) |
| `e8m0_pair` | esaustivo (65 536) |
| `adder_tree`, `fixed_to_fp32`, `fp32_exp`, `max_tree`, `e2m1_quant_elem` | corner + 3000 random |

Il catalogo è collaudato da una versione di b05 decomposta a mano
(`tests/fixtures/decomposed_b05/`): ogni blocco passa il suo golden e il top composto
passa il golden del task.

## Base di conoscenza Chisel 6

`knowledge/chisel6_pitfalls.yaml`: ogni voce ha regola, esempio sbagliato/corretto,
trigger sui messaggi d'errore e (se sicura) correzione automatica.
`tests/test_kb_integration.py` compila ogni esempio con Chisel 6.6 + firtool veri:
gli sbagliati devono fallire col messaggio previsto, i corretti devono elaborare,
le correzioni automatiche devono riparare gli sbagliati. La verifica ha già eliminato
due regole false scritte "a memoria" (es. `-x` su `UInt` compila in Chisel 6.6).

## Libreria e anti-leakage

`library/` (o `library.path`): ogni componente verificato, con task d'origine.
Con `exclude_current_task` i componenti nati dal task in esame non vengono mai
riusati né mostrati come esempi. In valutazione la libreria parte vuota.

## Ablation

Ogni meccanismo ha un interruttore in `agent:` (vedi `configs/default.yaml`):
`decompose`, `spec_parser`, `kb`, `format_card`, `structured_feedback`, `library`,
`best_of_n`, `escape`, `routing`. Da riga di comando:

```bash
mxagent run --task benchmarks/b05_mxfp4_dot32 --mode full --set agent.decompose=false
```
