# 0000 — Decisioni rimandate

## Stato

Vivo — raccoglie le decisioni volutamente non ancora prese. Non anticiparle: quando una
viene sciolta, va rimossa da qui e, se non banale, diventa un ADR proprio.

## Perché questo file

`CLAUDE.md` vieta di aggirare le decisioni architetturali già prese ma anche di anticipare
quelle rimandate. Questo file è l'elenco di queste ultime, con il contesto minimo per capire
perché sono aperte e cosa serve per chiuderle. Fonte: sezione "Questioni aperte" di
`docs/roadmap/STATO.md`.

---

## 1. Tempistica delle modifiche al manoscritto

**Domanda:** il manoscritto è già stato sottomesso? Le tre modifiche al testo individuate —
i numeri della dose termica, la frase in §3.5, la sezione Data availability — vanno applicate
nelle bozze già in circolazione oppure devono rientrare prima dell'invio definitivo.

**Contesto:** le modifiche nascono da un disallineamento tra quanto il codice produce
attualmente e quanto il testo del manoscritto dichiara. Finché lo stato della sottomissione
non è chiaro, non è possibile stabilire se si tratta di una correzione su un documento ancora
in bozza o di un errata da gestire con la rivista.

---

## 2. Seconda serie di risposta per V2.2

**Domanda:** quale serie di risposta biologica usare per validare il passo V2.2 (prova sul
campo) della roadmap.

**Contesto:** nessuna delle candidate valutate copre l'intero ventennio della serie EC50
attuale — righting response, volume delle uova, contenuto proteico e NRRT partono tutte da
zero punti storici. La scelta condiziona quanto V2.2 potrà davvero validare l'astrazione
introdotta in V2.1.

---

## 3. Nome definitivo del progetto

**Domanda:** quale sarà il nome definitivo del progetto, e quando avverrà la rinomina del
repository.

**Contesto:** il repository porta ancora il nome del caso di studio originale
(`climate_change_on_sea_urchins`), mentre la roadmap prevede che il progetto diventi uno
strumento generico oltre quel singolo caso. Rinominare il repository prima che l'astrazione
sia effettiva rischierebbe di promettere una genericità non ancora raggiunta.

---

## 4. Destino delle analisi in R

**Domanda:** le analisi in R (attualmente solo DLNM, dopo che SEA e mixed-effects sono stati
portati in Python) restano una dipendenza opzionale del progetto, oppure vengono anch'esse
portate in Python.

**Contesto:** il porting di SEA e mixed-effects in Python ha già ridotto la superficie R a un
solo modulo (`scripts/mhw_lag_analysis.R`, DLNM). Non è ancora deciso se valga la pena
completare il porting per eliminare la dipendenza da R, o se mantenerla come opzionale sia
sufficiente per gli obiettivi della versione corrente.

---

## 6. `label` non sanificato per l'uso nei nomi di file

**Domanda:** `ResponseSpec.label` (`v2.1/response-abstraction`) è una stringa di
visualizzazione usata anche per costruire nomi di file e altre identità di output (etichette
di riga/colonna, campi `"variable"`/`"series"`, chiavi di dizionario). Un secondo caso
potrebbe dichiarare un `label` con spazi o caratteri non adatti a un nome di file (es.
"Righting response (s)"). Va sanificato prima dell'uso in un nome di file? Con quale regola?

**Contesto:** per il caso Livorno non si pone: `label == "EC50"`, già un nome di file valido,
quindi non blocca questo passo (`dist_EC50.csv` resta identico). Rimandata a quando arriva un
secondo caso reale (V2.2) che esponga il problema concretamente, invece di indovinare una
regola di sanificazione senza un caso su cui verificarla.

---

## 7. Finestra sull'intero record resa esplicita

**Domanda:** quando uno studio dichiara finestre temporali (`StudySpec.windows`), anche la
finestra sull'intero record dovrebbe diventare una finestra dichiarata, con la propria
directory `results/<study_id>/<window_id>/`, invece di restare implicita.

**Contesto:** V2.2 (PR 5a/5b) fa percorrere a una sola esecuzione della pipeline tutte le
finestre dichiarate, con i risultati in directory sorelle `results/<study_id>/<window_id>/`.
Uno studio senza finestre (Livorno, oggi) resta com'è: risultati in `results/<study_id>/`,
senza sottodirectory, che concettualmente è una finestra implicita sull'intero record.
Formalizzarla adesso sposterebbe i file appena spostati da ADR-0008 e produrrebbe deriva nel
golden master senza alcun guadagno per un caso che non ha finestre da confrontare. Ma in uno
studio che dichiara finestre, l'intero record è il termine di paragone naturale della tabella
di confronto: lasciarlo implicito, in una directory di forma diversa dalle sorelle, è
un'asimmetria da sciogliere quando arriva il primo studio con finestre.

---

## 8. Scelte scientifiche scritte nel codice invece che nella specifica

**Domanda:** quando e come portare nella specifica di studio (`study.yaml`) le scelte sui dati
che oggi vivono solo nel codice. Per l'invariante 6 di `CLAUDE.md` una trasformazione non
leggibile nel file di studio è un difetto. Emerse leggendo i moduli per le finestre temporali
(V2.2, 5b), rimandate perché portarle nella specifica cambia lo schema per ogni studio e non
serve ai prerequisiti di V2.2.

**Contesto, una voce per scelta:**
- **Imputazione della risposta**: media mobile centrata di `IMPUTE_WINDOW_MONTHS = 12` mesi con
  `IMPUTE_MIN_PERIODS = 3` (`common.impute_response`, condivisa da `scripts/build_dataset.py`).
  È applicata **due volte**: una da `build_dataset.py` sui valori reali, una di nuovo da
  `common.load_data()` sulla serie già imputata. La seconda passata riempie 8 mesi che la
  prima non raggiunge (2007-06, 2007-07, 2007-09, 2007-12, 2009-08, 2009-09, 2022-11,
  2023-06). Le finestre riproducono le due passate per restare equivalenti al percorso
  sull'intero record, ma se la doppia passata sia voluta non è deciso.
- **`forecast.py`**: addestramento da `2016-01-01`, fisso nel codice e diverso da
  `split_date` (`2016-06-01`). Sembra un residuo del vecchio punto di taglio di gennaio, come
  era successo al confronto pre/post del controllo negativo (`negative_control.py`, che lo
  conserva ora esplicitamente come `PUBLISHED_SPLIT_DATE`).
- **`correlations.py`**: media mobile centrata a 12 mesi applicata a *tutti* i valori della
  risposta, reali compresi, prima della decomposizione.
- **Medie mobili centrate con `min_periods` basso** (annotate il 1/10/2026, mentre si
  correggevano i riempimenti ai bordi delle serie): vicino al primo e all'ultimo mese la
  finestra è incompleta e il valore si calcola da un lato solo, fino a un solo mese con
  `min_periods=1`. Sono la media a 12 mesi di `correlations.py` qui sopra (`min_periods=1`), il
  trend a 25 mesi di `regime_shift.py` (`min_periods=8`), il trend della temperatura a 25 mesi
  di `mhw_robustness.py` (`min_periods=12`) e l'imputazione della risposta (`min_periods=3`,
  prima voce). Non sono riempimenti di valori mancanti ma, sugli ultimi mesi della serie, si
  comportano in modo simile: scelte di metodo, da decidere insieme alla media di
  `correlations.py`.
- **`mhw_lag_extra.py`**: un'osservazione della risposta è associata a un evento se dista
  meno di 20 giorni dalla data attesa (`_nearest_ec50`).

(`YEAR_MIN`/`YEAR_MAX` di `mhw_lag_annual.py` non è qui: si risolve in V2.2 5b-4 con la regola
degli anni completi.)

---

## 9. Convenzione della data di una rottura

**Domanda:** una data di rottura indica l'ultimo periodo prima del cambiamento o il primo
dopo? Oggi i moduli non la usano in modo uniforme, e allinearli cambierebbe valori salvati.

**Contesto:**
- `regime_shift.py`: `pettitt()` restituisce l'indice dell'**ultima** osservazione prima del
  cambiamento, e quella data viene salvata (`break`, `break_date`, `break_year`). Per la
  risposta di Livorno risulta 2016-05. `split_date` e il manoscritto indicano invece il
  **primo** mese dopo, 2016-06, con lo stesso p del Pettitt mensile. Dal 29/9 il testo del
  verdict dichiara entrambe le date, ma i campi salvati restano sulla convenzione
  «ultimo prima».
- Nello stesso modulo le medie `pre_mean`/`post_mean` tagliano un elemento prima della
  statistica di Pettitt: `x[:k]` e `x[k:]` mettono l'ultimo mese prima del cambiamento (2016-05)
  nel periodo dopo, mentre il segmento della statistica è `x[:k+1]`. Vale anche per le rotture
  annuali MHW e ambientali in `regime_shift_changepoints.csv`.
- La distanza fra due rotture dipende dalla convenzione quando una è mensile e l'altra
  annuale. Confrontando gli ultimi periodi prima, MHW 2013 contro risposta 2016-05, sono 3 anni
  (`exposure_precedes_response_years`). Confrontando i primi periodi dopo, MHW 2014 contro
  risposta 2016-06, sono 2 anni.
- `changepoint.py` (QLR/AR(1)) riporta la propria `break_date` dall'indice di rottura
  dell'algoritmo; quale delle due convenzioni segua non è stato verificato.

Allineare significa scegliere una convenzione per tutti i moduli, correggere le medie di
`regime_shift` e aggiornare i valori salvati e il riferimento del golden master: una modifica
di risultati, da decidere esplicitamente.
