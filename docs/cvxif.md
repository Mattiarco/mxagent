# Coprocessore RISC-V CV-X-IF (M7)

Il MAC MXFP4 di b08, generato dall'agente o di riferimento, viene messo in un
coprocessore con interfaccia **CORE-V eXtension Interface (CV-X-IF) v1.0.0**:
- fonte: OpenHW Group, release ratificata il 2024-05-22;
- testo letto da docs.openhwgroup.org/projects/openhw-group-core-v-xif, pagina
  `x_ext.html`, il 2026-11-07.

Il wrapper è scritto a mano e non viene generato dall'LLM:
[MxCvxif.scala](../chisel/src/main/scala/mxagent/cvxif/MxCvxif.scala).
Istanzia il MAC per nome come `ExtModule`, quindi lo stesso wrapper elaborato si
simula con il Verilog di qualunque MAC che rispetti il contratto di b08.

```bash
mxagent wrap                              # MAC di riferimento: verifica, poi test del coprocessore
mxagent wrap --sources MyMac.scala        # un MAC qualsiasi
mxagent wrap --run runs/<id>              # il MAC finale di una run
```

Nel sistema completo, per i task con `wrapper: cvxif`, lo stadio `WRAP` segue il
pass e registra l'esito in `wrap.json` e in `metrics.json["wrapper"]`. Il report
di M6 lo riporta per ogni modalità, comprese le baseline.

## Parametri dell'interfaccia

| parametro | valore |
|---|---|
| X_NUM_RS | 2 |
| X_ID_WIDTH | 4 |
| X_RFR_WIDTH / X_RFW_WIDTH | 32 (XLEN = 32) |
| X_NUM_HARTS / X_HARTID_WIDTH | 1 / 1 |
| X_DUALREAD / X_DUALWRITE | 0 / 0 |
| X_ISSUE_REGISTER_SPLIT | 0: gli operandi arrivano con l'issue |
| X_MISA | 0 |

I pacchetti SystemVerilog della specifica (`issue_req`, `issue_resp`, `register`,
`commit`, `result`) sono esposti come segnali piatti `<segnale>_<campo>`, per
esempio `issue_resp_register_read` o `commit_commit_kill`. L'interfaccia
compressed risponde sempre `accept = 0`. Le interfacce memory non esistono in
v1.0.0.

## Istruzioni (opcode custom-0 `0001011`, formato R, funct3 = 0)

| funct7 | istruzione | effetto | write-back |
|---|---|---|---|
| 0 | `mx.lda rs1, rs2` | A[8j .. 8j+7] = rs1 (8 × E2M1), j = rs2[1:0] | no |
| 1 | `mx.ldb rs1, rs2` | come sopra, per B | no |
| 2 | `mx.scale rs1, rs2` | a_scale = rs1[7:0], b_scale = rs2[7:0] (E8M0) | no |
| 3 | `mx.mac rd, rs1` | rd = rs1 + Dot(A, B), un solo arrotondamento binary32 | sì |

Ogni altra istruzione viene rifiutata subito (`accept = 0`). Un dot product da 32
elementi richiede quindi 10 istruzioni: 4 `lda`, 4 `ldb`, `scale`, `mac`.

## Microarchitettura e regole del protocollo

Il coprocessore ha una sola istruzione in volo (`issue_ready = 0` finché è occupato).
Rispetta le regole della specifica in questo modo:
- **Accettazione:** un'istruzione si accetta solo con gli operandi necessari
  validi (`rs_valid`); prima si tiene basso `issue_ready`. `register_read` vale
  `01` per `mac` e `11` per le altre.
- **Nessuno stato speculativo:** buffer A/B e scale cambiano solo dopo il commit
  con `commit_kill = 0`, quindi un kill non ha nulla da annullare.
- **Un risultato per istruzione:** ogni istruzione accettata e non uccisa produce
  esattamente una transazione di result, anche senza write-back (`we = 0`). Le
  istruzioni rifiutate o uccise non producono result, e non ci sono result
  speculativi.
- **Commit:**
  - un commit può arrivare nello stesso ciclo dell'issue;
  - può arrivare prima del suo handshake, mentre il coprocessore aspetta gli
    operandi o è ancora occupato con l'istruzione precedente: viene registrato;
  - il confronto fra id tiene conto del wrap-around: un commit di un id ≥ al
    proprio rende l'istruzione non speculativa, un kill di un id ≤ al proprio la
    uccide;
- **Result:** `result_valid` resta alto finché la CPU non risponde con `result_ready`.

## Test di protocollo

[tb_cvxif.py](../mxagent/wrap/tb_cvxif.py) simula la CPU e
[cvxif.py](../mxagent/wrap/cvxif.py) prepara il programma e controlla le uscite.

**Programma.** Ogni vettore del golden di b08 (corner più casuali) diventa la
sequenza lda×4, ldb×4 (in ordine casuale), scale, mac. Nel programma sono
inserite istruzioni da rifiutare (altri opcode, funct7 o funct3 non validi) e
istruzioni uccise.

**Comportamento della CPU simulata:**
- ritardo casuale degli operandi;
- commit a distanza casuale dall'inizio dell'issue, anche prima dell'handshake;
- flush dopo un kill;
- backpressure casuale su `result_ready`.

**Cosa si controlla:**
- per ogni istruzione, `accept`, `writeback` e `register_read`;
- numero, ordine e id dei result;
- `we` e `rd`;
- il valore di ogni `mac` contro il golden `mx_mac`, sullo stato architetturale
  atteso: le istruzioni uccise non devono modificarlo.

**Risultati misurati** (MAC di riferimento, seed 0):
- 2275 istruzioni: 196 MAC bit-exact, 154 uccise, 161 rifiutate;
- 9218 cicli.

**Il test trova i bug** (mutanti in `tests/fixtures/cvxif_mutants/`):
- **Scrittura dei buffer prima del commit:** risultati MAC sbagliati.
- **Result anche per un'istruzione uccisa:** 934 result invece di 860, con id
  fuori ordine.
- **Primo bug reale:** la prima versione del wrapper registrava i commit
  anticipati solo da fermo. Il test si bloccava dopo 16 istruzioni, il bug è
  stato corretto ed è documentato nel codice.

**Limiti:**
- Il throughput è di un'istruzione alla volta.
- Il modello di CPU è un modello di test scritto seguendo la specifica, non un
  core reale. L'integrazione con un core, per esempio CV32E40X, non è stata
  provata.
