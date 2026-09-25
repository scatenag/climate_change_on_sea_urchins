# 0007 — `split_date` è un campo di `ResponseSpec`, non una costante di modulo

## Contesto

`common.py::SPLIT_DATE` era una costante di modulo (`_SPLIT_CONFIG = "2016-06-01"`), usata
ovunque nel pacchetto come confine pre/post. Era una decisione volutamente rimandata (ADR
0000, voce 5): non è un valore di sito o di sorgente dati, è una stima di changepoint
specifica della serie EC50 di Livorno (rank-based; vedi ADR-0001 per perché non coincide con
la stima QLR/AR(1) di `changepoint.py` sulla stessa serie).

Con V2.2 il caso diventa più di uno studio (finestre temporali diverse, celle ambientali
spostate, e in futuro una seconda serie di risposta reale). Una costante di modulo condivisa
non può funzionare: un secondo studio erediterebbe il changepoint stimato su un caso diverso.

## Alternative valutate

- **Lasciarla come costante di `common.py`**: scartata — è esattamente il problema che questo
  ADR risolve.
- **Un campo a livello di `StudySpec`**: scartata — il confine pre/post è una proprietà
  *della serie di risposta*, non dello studio nel suo insieme. Uno studio con due serie di
  risposta avrebbe due changepoint potenzialmente diversi.
- **Un campo di `ResponseSpec`, obbligatorio, validato contro l'intervallo coperto dalla
  serie caricata**: scelta adottata.

## Decisione

`ResponseSpec.split_date` (stringa ISO) è obbligatorio. `study_spec.py` non lo valida contro
i dati (quel modulo non legge mai `data/`): la validazione — che la data cada nell'intervallo
temporale effettivamente coperto dalla serie di risposta caricata — avviene in `common.py`,
al momento in cui `SPLIT_DATE` diventa un valore utilizzabile, con un messaggio esplicito che
rifiuta il caricamento invece di produrre più avanti un confronto pre/post con un lato vuoto.

Per Livorno: `2016-06-01`, invariato.

## Conseguenze

Il commento esplicito accanto a `SPLIT_DATE` in `common.py`, richiesto da ADR-0001 e mai
scritto, è ora superfluo nella forma originale — il valore non vive più lì, vive nella
specifica, con il rimando ad ADR-0001 scritto direttamente nel commento YAML. La domanda
posta da ADR-0001 (perché `changepoint.py` non coincide con `SPLIT_DATE`) resta valida e non
è toccata da questo ADR.
