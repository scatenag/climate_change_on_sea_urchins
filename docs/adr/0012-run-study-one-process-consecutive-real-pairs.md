# 0012 — `ccsu-run-study`: un processo per studio, differenze solo fra mesi consecutivi reali

## Contesto

La pipeline di Livorno (`ccsu-run-pipeline`) conosce un solo studio, quello che `common` carica
all'importazione. Il comando generico (M1.5) deve far girare uno studio di formato 2 a partire da una
cartella di ingressi (il CSV della risposta e la SST scaricata) e dare, oltre ai risultati, il numero di
test dichiarato e la provenienza. La CCF di Livorno sulle differenze prime differenzia la serie *imputata* e
scarta soltanto i mesi in cui è imputato il mese *t*: 48 delle 163 differenze tenute (29%) hanno il mese
*t−1* imputato, cioè non sono osservazioni (misura del 15/10, `drafts/`).

## Decisione

- **La CCF generica usa solo coppie di mesi consecutivi entrambi reali** (misura presente e non imputata),
  qualunque imputazione lo studio dichiari. L'imputazione resta nel dataset, dichiarata, per chi la usa
  altrove; la pipeline di Livorno non cambia (decisione del proprietario, 16/10).
- Statistica come in D3: Spearman fra la differenza prima del driver (l'anomalia mensile della SST, vedi sotto) a *t−k* e quella della risposta a *t*,
  ritardi 0..`analysis.max_lag_months`, correzione Benjamini-Hochberg sui ritardi eseguiti. Un ritardo con meno
  di 10 coppie non è eseguito: compare in `run.json` con la ragione, non sparisce.
- **`analysis.max_lag_months`** nella specifica (predefinito 12, da 1 a 60, campi ignoti rifiutati come ovunque):
  il numero di test dichiarato (`max_lag_months + 1`) viene da lì. Livorno lo dichiara (12) e un test lo tiene
  uguale a `common.TAU_MAX`. Il nucleo del calcolo è `mhw_analysis.ccf_at_lags` (i ritardi vengono dal
  chiamante); `compute_ccf` ne è l'involucro con `TAU_MAX`, stesso risultato.
- **Un processo per studio.** Il comando legge lo studio, gli ingressi in una cartella e scrive in una
  cartella: `ccf.csv`, `dataset/` (le tabelle del costruttore e il suo manifest), `progress.json` e `run.json`.
  `progress.json` si riscrive in modo atomico a ogni passo (`load_study`, `read_response`, `read_sst`,
  `build_dataset`, `correlate`, `write_results`) e un altro processo può leggerlo in qualunque momento; un
  fallimento riporta passo e ragione, e non lascia risultati.
- Un solo driver in questa versione, l'anomalia mensile della SST (revisione del 16/10, sotto); le metriche delle ondate di calore entrano con M1.9.
- Rifiuti con la ragione: studio di formato 1, sorgente diversa dal CSV, più di un sito o di una risposta,
  `sst_daily` assente dall'ambiente.
- `run.json` porta: hash dello studio e degli ingressi, il manifest del download, parametri, test dichiarati/eseguiti/
  non eseguiti, copertura (mesi esclusi), note del CSV, versioni e commit del codice. Nessun percorso della home.

## Revisione del 16/10: il driver è l'anomalia mensile della SST, non la temperatura

La prima versione usava la temperatura mensile. La simulazione per M1.6 (`drafts/misura-m1-6-potenza/potenza2.py`,
282 mesi della fixture, circa metà dei mesi mancanti, differenze fra mesi consecutivi reali, BH su 13 ritardi)
ha mostrato che la stagionalità della temperatura grezza produce associazioni che non sono del driver:

- **Risposta senza alcun legame con la SST ma con un proprio ciclo stagionale** (50% della varianza): almeno un
  ritardo con BH<0.05 nell'**83%** delle simulazioni con mancanti indipendenti e nel **97%** con mancanti in
  sequenze (in media da 5.5 a 7.5 ritardi significativi); sull'anomalia il 6.3% e il 5.3%.
- **Con un legame vero** (ritardo 3, generato sulla grezza, CCF sulla grezza) i ritardi significativi sono in media da
  4.4 a 9.4, a seconda della quota di segnale e dei mancanti: l'utente leggerebbe come associazioni gli artefatti
  stagionali.
- **Generata e analizzata sull'anomalia**, con 70 coppie (mancanti indipendenti) il ritardo vero è ritrovato
  (massimo di |r| sul ritardo, segno atteso, BH<0.05) nel 34%, 97% e 100% dei casi con segnale al 20%, 50%, 80% della
  varianza; con 105 coppie (sequenze) nel 57%, 99% e 100%.
- **Il prezzo.** Se la dipendenza passa dalla sola parte stagionale della temperatura (generata sulla grezza, CCF
  sull'anomalia) la CCF non la vede: rilevazione 0.02–0.08. È un limite da dichiarare, non un difetto: la parte
  stagionale della temperatura non si distingue da qualunque altro ciclo stagionale della risposta.

Decisione del proprietario (16/10): il driver della CCF generica è l'**anomalia mensile della SST rispetto alla
climatologia dichiarata nello studio** (`anomaly_mean` di `mhw_monthly`, già scritta dal costruttore), e la serie a
verità nota di M1.6 si genera sull'anomalia. `ccf.csv` e `run.json` riportano il driver «SST anomaly» e la linea di
base. Un mese escluso dal costruttore per giorni mancanti nella SST non ha anomalia (stessa regola). La temperatura
resta nel dataset. La CCF **cerca solo l'associazione non stagionale**, e una dipendenza che passa dalla sola parte
stagionale non la vede: va detto a chi usa lo strumento (aiuto del comando, README).

**Limite registrato:** con una risposta indipendente dalla SST, la quota di simulazioni con almeno un ritardo
BH<0.05 sull'anomalia è sopra il 5% (6.2–6.6% con mancanti indipendenti, 7.4–7.9% in sequenze). Causa misurata, non
supposta, in ADR-0000 voce 20. Nessun rimedio ora.

## Limite accettato per M1: il comando gira dove `common` ha caricato Livorno

`common` carica lo studio di Livorno all'importazione (`DATA`, `RESULTS`, `SPLIT_DATE`, la linea di base
delle ondate, `TAU_MAX`) e l'installazione è editabile dal repository. Il comando funziona quindi solo da una
copia del repository con `data/` ed `examples/`; la separazione completa (D2) è rinviata
(ADR-0000, voce 19). Per non calcolare uno studio nuovo con i valori di Livorno senza errore:

- un test statico percorre il codice raggiungibile da `run_study.run_study` e vieta i riferimenti a quelle
  costanti e a `config`, e richiede che la catena passi per il costruttore, il lettore del CSV, il rilevamento
  delle ondate, `ccf_at_lags`, `aggregate_period` e `impute_series`;
- un test fa girare lo stesso studio sintetico in due processi, con `CCSU_STUDY` non impostato e con un altro
  studio di formato 1 caricato (altra linea di base, altra data di divisione, altro `max_lag`), e richiede
  risultati identici byte per byte.

Entrambi sono stati visti fallire su un sabotaggio (un riferimento a `common.DATA`; una linea di base che
dipende da `common.SPLIT_DATE` letta senza nome visibile, che solo il secondo prende).

## Conseguenze

- La CCF generica e quella di Livorno differiscono per costruzione sulle differenze prime; nessun numero
  pubblicato cambia. La CCF di Livorno con le sole coppie reali, misurata a parte, non cambia alcuna
  conclusione (nessun ritardo con p<0.05 nell'una o nell'altra).
- Con circa metà dei mesi mancanti restano mediamente 70 coppie (105 se i mancanti sono in sequenze): basta a
  ritrovare un ritardo noto con BH se il segnale è forte (misura per M1.6, `drafts/misura-m1-6-potenza/`).
