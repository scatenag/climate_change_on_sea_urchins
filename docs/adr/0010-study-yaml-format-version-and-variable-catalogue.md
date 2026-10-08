# 0010 — `study.yaml` ha una versione del formato; l'ambiente si nomina dal catalogo delle variabili

## Contesto

La milestone M1 (`docs/roadmap/MILESTONE-M1.md`) porta lo strumento a chi ha solo i propri dati: lo
studio non lo scrive più chi conosce il codice, e un file di studio può arrivare da una versione
precedente dello strumento o da una successiva. Lo `study.yaml` di Livorno (V2.1) dichiara dataset
Copernicus per esteso, richiede `split_date` e `data_dir`, e conosce una sola sorgente di risposta
(il foglio Google). Per uno studio nuovo servono: una sorgente CSV, `split_date` facoltativo,
l'imputazione dichiarata, il contaminante dichiarato, e l'ambiente nominato da un catalogo
verificato invece che da nomi di dataset scritti a mano dall'utente (il CO₂ in Pascal è nato così:
un'unità dichiarata in nessun punto).

## Alternative valutate

- **Un solo formato, esteso con campi facoltativi.** Scartata: un file nuovo e uno vecchio non si
  distinguerebbero, e «la versione che lo legge non lo conosce» non avrebbe un messaggio.
- **Un file di studio diverso per ogni tipo di studio.** Scartata: due schemi da mantenere per lo
  stesso oggetto.
- **`format_version` esplicito, con il formato 1 che resta quello di Livorno (chiave assente = 1) e un
  formato 2 con regole proprie, validate nello stesso modello.** Scelta adottata.

## Decisione

- `format_version` è un intero; assente vale 1. Le versioni lette sono in
  `study_spec.SUPPORTED_FORMATS`; una versione sconosciuta è rifiutata nominandola e nominando quelle
  supportate (mai interpretata «alla meglio»).
- **Formato 1** (Livorno): invariato. `data_dir` e `split_date` obbligatori, ambiente per dataset,
  sorgente di risposta solo `google_sheet`.
- **Formato 2**: ambiente come elenco di `{catalog_id: …}` verificato contro `catalog.py` (id ignoto
  rifiutato con l'elenco di quelli noti; ogni sito deve cadere nel dominio del prodotto); sorgente
  `csv` con `file` che è solo un nome di file (mai un percorso), granularità per prova o già
  aggregata, intervallo di confidenza facoltativo ma a coppie; `split_date` facoltativo (senza, le
  analisi pre/post sono spente con il motivo); `imputation` assente salvo dichiarazione (D8);
  `contaminant` dichiarato (D7); `data_dir` facoltativo.
- **Catalogo delle variabili** (`catalog.py`): dati dichiarativi validati (campi extra rifiutati, nessun
  campo eseguibile né percorso): provider, dataset del prodotto reprocessato e di quello che lo
  segue, variabile, profondità, cadenza, unità nativa e di analisi, conversione nominata, dominio.
  Parte con la sola SST giornaliera (fetta verticale); le variabili mensili una alla volta (M1.8).
- Un modello nuovo rifiuta i campi che non conosce; `StudySpec` ora fa lo stesso a livello di
  radice (un refuso come `split_dat` era ignorato in silenzio). I sotto-modelli del formato 1 già
  esistenti restano come sono.
- Uno studio di formato 2 si carica e si valida, ma la pipeline non lo esegue ancora: `common.py` lo
  dice esplicitamente invece di fallire più avanti. Lo sbloccherà `ccsu-run-study` (M1.5).

## Conseguenze

- Il caricamento di file non fidati (YAML sicuro, `data_dir` e sorgenti remote ignorati con avviso) è
  M1.10; questo formato prepara il terreno (nome di file nudo, nessun campo eseguibile) ma non lo
  realizza.
- Il catalogo registra l'id corretto del dataset analysis-forecast giornaliero, diverso da quello che
  `scripts/fetch_copernicus_daily.py` chiama «fallback» e che non esiste nel catalogo Copernicus.
- Lo `study.yaml` di Livorno dichiara ora l'imputazione che il codice applica (`passes: 2`) e il
  contaminante; finché il costruttore del dataset (M1.4) non legge la dichiarazione, un test la tiene
  uguale al codice.
