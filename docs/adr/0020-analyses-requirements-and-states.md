# 0020 — Che cosa serve a ogni analisi e quali analisi consente uno studio (M1.7)

**PROVVISORIA** nei contenuti del registro (i requisiti sono quelli delle implementazioni di oggi; M1.9 li rivede analisi per analisi
quando la sposta nel comando) e in attesa della conferma del proprietario insieme alla decisione dopo MA.5 su quali analisi di
Livorno entrano nel motore generico. Reversibile, nessun effetto su Livorno.

## Contesto

`ccsu-run-study` esegue una sola analisi, la CCF fra l'anomalia mensile della SST e la risposta. La pipeline di Livorno ne ha altre
quindici (più il DLNM in R) che leggono ciascuna cose diverse: le cinque variabili ambientali, le determinazioni individuali, i
controlli negativi, la SST giornaliera, la data di cesura, il rame. Uno studio nuovo che non ha il pH non può avere la speciazione
del rame né il forecast, e la pagina deve dirlo con il motivo invece di tacere o di fallire a metà (COME_SI_LAVORA §2: un errore non
si filtra in silenzio).

## Decisione

- **Un registro dichiarativo** (`analyses.py`: nomi e regole, nessun import di analisi, nessuno stato di `common`, nessuna
  scrittura): per ogni analisi, `Needs` (variabili, SST giornaliera, catalogo delle ondate, determinazioni individuali, controlli,
  cesura, contaminante, R), i file che scrive e, dove il codice fissa una scelta che uno studio dovrebbe dichiarare, una nota
  (soglia di 24 °C della termica; scenari e anni di addestramento del forecast). Diciassette voci: le quindici analisi di
  `pipeline._MODULES` (senza `mhw_detection`, che nel motore è il costruttore del dataset), la CCF generica `ccf_sst_anomaly` e il
  DLNM (R). Campi ignoti e nomi di variabile sbagliati sono errori.
- **Tre stati**: `runs` (il comando la esegue: solo la CCF generica), `not_applicable` (manca qualcosa allo studio: i motivi, in
  ordine fisso), `pending` (lo studio ha quanto serve, il comando non la esegue ancora: M1.9). Un'analisi spenta non scrive nulla.
- **`ccsu-run-study` scrive `analyses.json`** (formato 1: stato, motivi, nota e file di ogni analisi) insieme a `ccf.csv` e
  `run.json`; lo toglie a ogni avvio come gli altri. La pagina dei risultati lo legge e mostra la tabella «Other analyses of this
  tool»: dentro la stessa PR, e con le stesse parole del file (stesso studio, stesso motivo: criterio 3, motore e interfaccia).
- **I motivi non contengono parole scritte dallo studio** (il nome del contaminante non viene ripetuto): niente Markdown dal testo
  dell'utente. Il pH, la salinità, l'ossigeno e la CO₂ sono variabili mensili che il catalogo non ha ancora (M1.8): i loro id sono
  quelli previsti (`ph_monthly`, …), e finché non sono nel catalogo il motivo lo dice; quando M1.8 li aggiunge, la frase sparisce da sola
  (un test lo verifica). **La temperatura mensile è la media mensile della SST giornaliera** che il costruttore fa: la fornisce `sst_daily`.
- **Non si dichiara un minimo di mesi.** La riga di M1.7 lo prevedeva, ma nessun modulo ha una guardia che fermi l'intera analisi:
  i minimi sono dentro le singole analisi (una colonna restituita com'è, un ritardo saltato, un test non eseguito). Un valore
  dichiarato qui sarebbe inventato; il numero di test eseguiti e non eseguiti resta in `run.json` per la CCF e in ogni analisi
  quando M1.9 la sposta.

## Come sono stati trovati i requisiti

Togliendo **un elemento alla volta** dalla fixture congelata di Livorno (una colonna ambientale, le determinazioni individuali, i
controlli, la SST giornaliera, le tabelle delle ondate) e rieseguendo ogni modulo (150 prove, fuori da git). Con il pH tolto
falliscono `timeseries`, `period_split`, `correlations`, `cu_speciation`, `regime_shift`, `forecast`; senza salinità gli stessi tranne
il forecast; senza O₂ e CO₂ `timeseries`, `period_split`, `correlations`, `regime_shift`. Senza determinazioni individuali falliscono
`period_split`, `changepoint`, `negative_control`; senza le colonne dei controlli `negative_control`; senza SST giornaliera
`thermal_legacy`. Gli errori dovuti a `load_data` (la colonna della temperatura mensile, i file delle ondate) non diventano
requisiti di un'analisi che quella grandezza non usa: `timeseries`, `cu_speciation`, `thermal_legacy`, `changepoint`,
`negative_control` non dichiarano le ondate. `stationarity` e `mhw_analysis` si adattano alle variabili che trovano (togliendone una
girano), e il registro lo dice in una nota. La cesura è dichiarata dove il cuore dell'analisi è il confronto prima/dopo
(`period_split`, `cu_speciation`, `negative_control`); `timeseries` e `correlations` hanno solo parte delle uscite legate ad essa e lo
dicono in una nota.

Guardie (`tests/test_analyses.py`): ogni modulo della pipeline ha una voce e nessuna voce senza modulo; **ogni file del golden
master appartiene a una e una sola analisi** (65 file: 62 dei moduli Python, 3 del DLNM); le variabili dichiarate coprono i nomi delle
colonne ambientali presenti nel codice del modulo; ciò che il modulo legge (ondate, determinazioni individuali, SST giornaliera,
controlli) coincide con ciò che il registro dichiara.

## Alternative

- **La dichiarazione dentro ogni modulo** (una costante `REQUIRES`): più vicina al codice, ma importare quindici moduli pesanti
  (statsmodels, scikit-learn) per costruire una tabella, e farlo nel processo dell'app, è inutile; le guardie sul sorgente danno lo
  stesso legame.
- **Due stati** (gira / non gira): non distingue «non applicabile a questo studio» da «applicabile ma non ancora nel comando», che
  è proprio ciò che il ricercatore deve sapere.
- **Dichiarare i requisiti generici di ogni analisi come saranno in M1.9** (ad esempio `timeseries` su tutte le variabili dichiarate):
  è una scelta di metodo per analisi, del proprietario; qui si dichiara ciò che oggi il codice legge.

## Conseguenze

- Per ora ogni studio nuovo vede una sola analisi eseguita e le altre «possibile, non eseguita» o spente con il motivo: è lo stato
  vero dello strumento. M1.8 aggiunge le variabili (e fa sparire un motivo per volta), M1.9 sposta le analisi e le fa passare da
  `pending` a `runs`.
- Una nuova analisi aggiunta alla pipeline senza una voce del registro fa fallire i test.
