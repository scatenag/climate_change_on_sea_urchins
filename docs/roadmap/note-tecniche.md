# Note tecniche — magazzino di dettaglio per versione

> **Questo file non è il piano.** Il piano è `METODO_E_FASI.md`; lo stato è `STATO.md`.
> Questo è il materiale tecnico di dettaglio: si consulta **all'inizio della versione a cui
> serve**, e si rivede in quel momento. Molte cose scritte qui saranno superate da ciò che
> avrai imparato nel frattempo: trattale come ipotesi di lavoro, non come specifiche.
>
> Convenzione dei branch: `v<versione>/<slug>`, es. `v2.1/spec-schema`.

Repo di partenza: `scatenag/climate_change_on_sea_urchins` (v1.3.0).

---

## 1. Diagnosi dello stato attuale

Cosa blocca concretamente la generalizzazione. Riferimenti a file reali.

| # | Blocco | Dove | Impatto |
|---|---|---|---|
| B1 | `config.py` alla radice, fuori dal package, importato con `sys.path.insert` dagli script | `config.py`, `scripts/fetch_*.py` | Non installabile in modo pulito; un solo sito possibile |
| B2 | `load_data()` a firma fissa (4-tuple), path hardcoded, imputazione EC50 rolling-12 cablata, backfill temperatura da `sst_daily` cablato | `src/.../common.py` | Ogni nuovo asse dati rompe la firma; le scelte statistiche sono invisibili nella config |
| B3 | Vocabolario colonne globale: `ENV_COLS`, `ALL_COLS`, `MHW_COLS` | `src/.../common.py` | Nessun modo di avere due dataset con variabili diverse nello stesso processo |
| B4 | `results/` è un namespace piatto (`corr_all.csv`, `forecast_mean.csv`, ...) | `results/` | **Blocco strutturale per V3**: N casi collidono sugli stessi filename |
| B5 | Moduli di analisi = `run()` senza argomenti, side-effect su disco, leggono da `load_data()` | tutti i moduli in `src/` | Non componibili, non parallelizzabili, non testabili per-caso |
| B6 | `dashboard.py` 2910 righe, codice Streamlit a livello di modulo, hack `del sys.modules` in `app.py` + `RERUN_LOCK` globale | `src/.../dashboard.py`, `app.py`, `__init__.py` | Il lock serializza tutte le sessioni (throughput = 1 utente); con N casi peggiora. È anche il punto più fragile agli occhi di un reviewer |
| B7 | Stringa `"EC50"` cablata: 229 occorrenze nel dashboard, ~150 nel resto | ovunque | Il rename è meccanico ma va fatto una volta sola e bene |
| B8 | Analisi domain-specific senza dichiarazione dei requisiti (`cu_speciation` richiede pH/T/S + endpoint metallo; `mhw_*`/`thermal_legacy` richiedono SST giornaliera) | `src/.../cu_speciation.py` ecc. | Su un caso diverso o crashano o producono numeri privi di senso |
| B9 | Ingestion cablata su MEDSEA + Google Sheets, dataset ID / START / END costanti | `scripts/fetch_copernicus*.py` | Nessuna copertura fuori dal Mediterraneo, nessuna cadenza diversa da mensile/giornaliera |
| B10 | Nessun metadato di unità di misura in nessun punto della pipeline | ovunque | Sintomo concreto: il caveat CO₂ nel README (vedi 1.1) |

### 1.1 Bug da chiudere subito: unità del CO₂

Il README dichiara le unità di `spco2` non verificate, con valori 31–58 invece dei ~380–450 µatm
attesi. **`spco2` di Copernicus Marine è documentato in Pascal.** Conversione: ×9.8692 (1 atm = 101325 Pa).

Verifica eseguita sui dati del repo:

```
2003: 399.4 µatm    2013: 398.5 µatm    2023: 428.3 µatm
trend 2003-2025: +1.65 µatm/anno (p = 3.8e-08)
```

Range e pendenza corretti per pCO₂ superficiale mediterraneo. Il cross-check con `data.csv`
tornava a 0.99 semplicemente perché anche la serie originale era in Pa. Va corretto nel
branch `v1.1/co2-units` (sezione 3) **prima** di qualunque refactor: cambia i valori assoluti
di `data_extended.csv` e quindi il golden master.

Nota scientifica: la correzione è moltiplicativa e costante, quindi correlazioni di Spearman,
CCF e changepoint **non cambiano**. Cambiano medie, trend assoluti e — attenzione —
`cu_speciation.py`, che usa pH e non CO₂ per il sistema carbonatico, quindi dovrebbe essere
invariato: verificalo esplicitamente.

---

## 2. Architettura target (V4)

L'astrazione centrale è la **specifica di studio** dichiarativa (YAML), e il **caso** come
tupla `(risposta, sito, finestra, contesto ambientale)`.

```yaml
# study.yaml — esempio
study:
  id: mediterranean-sentinels-2026
  description: Confronto multi-sito di sensibilità gametica

sites:
  - id: livorno-sud
    lat: 43.4278
    lon: 10.3956
    bbox_delta: 0.1
    depth: [0, 10]

responses:
  - id: plividus-cu-ec50
    source: {type: google_sheet, sheet_id: "..."}
    column_map: {date: DATE, value: EC50, ci_low: LL, ci_high: UL, n: pos}
    unit: mg/L
    direction: higher_is_better      # valore alto = organismo meno sensibile
    taxon: {name: Paracentrotus lividus, aphia_id: 124316}
    endpoint: {code: EC50, effect: mortality, exposure_h: 72}
    chemical: {name: copper, cas: "7440-50-8"}
    protocol: {standard: "UNI EN ISO 17244:2015", lab: ISPRA-Livorno}

environment:
  - id: medsea-bgc
    provider: copernicus_marine
    variables:
      - {name: pH,   dataset: cmems_mod_med_bgc-car_my_4.2km_P1M-m, var: ph,    unit: "1"}
      - {name: CO2,  dataset: cmems_mod_med_bgc-co2_my_4.2km_P1M-m, var: spco2, unit: Pa}
    cadence: P1M

windows:
  - {id: full, start: 2003-01, end: 2026-06}

cases:
  - {id: livorno-ec50-full, response: plividus-cu-ec50, site: livorno-sud,
     window: full, environment: [medsea-phy, medsea-bgc]}

analyses:
  - {plugin: lagged_correlation, params: {max_lag: 12, method: spearman}}
  - {plugin: period_split,       params: {split: auto}}

inference:
  family: study                 # ambito su cui si controlla la molteplicità
  multiplicity: benjamini_hochberg
  alpha: 0.05
  null_model: {type: block_bootstrap, block: 12, n: 2000}
  preregistered: true           # se true, il piano è congelato: hash in manifest

comparisons:
  - {plugin: forest_plot,        params: {effect: peak_lag_rho}}
  - {plugin: random_effects_meta, params: {effect: peak_lag_rho, moderators: [latitude, taxon]}}
```

Interfacce da introdurre (in ordine di importanza):

```python
class EnvProvider(Protocol):
    id: str
    def fetch(self, site: SiteSpec, variables: list[VariableSpec],
              window: WindowSpec, cadence: str) -> pd.DataFrame: ...
    def describe(self) -> ProviderMetadata: ...   # per la provenienza

class ResponseSource(Protocol):
    def load(self, spec: ResponseSpec) -> ResponseFrame: ...
    # ResponseFrame tidy: datetime, value, ci_low, ci_high, n, flag, unit

class Analysis(Protocol):
    name: str
    requires: Requirements   # variabili, cadenza, n minimo, serve incertezza?, serve daily?
    def run(self, ds: AlignedDataset, **params) -> AnalysisResult: ...

@dataclass
class AnalysisResult:
    analysis: str; case_id: str; params: dict
    effects: pd.DataFrame     # CONTRATTO: estimate, ci_low, ci_high, n, scale,
                              #            direction_adjusted, n_tests
    tables: dict[str, pd.DataFrame]
    provenance: Provenance    # hash input, versioni, timestamp
```

**Il contratto sugli effetti (`effects`) è la decisione architetturale più importante
dell'intero piano.** Senza una forma standardizzata e comparabile degli effect size,
V3 (confronto tra tuple) e V4 (confronto tra organismi) sono impossibili — resterebbero
una collezione di grafici affiancati. Scrivilo come ADR-0002 e non derogare.

Registry dei plugin via entry points, così che terzi possano aggiungere analisi
**senza forkare**:

```toml
[project.entry-points."<pkg>.analyses"]
lagged_correlation = "<pkg>.analyses.lagged_correlation:LaggedCorrelation"
```

---

## 3. V1.1 — Messa in sicurezza (prima di toccare qualsiasi cosa)

Questa fase non aggiunge funzionalità. Salta questa fase e i refactor successivi
cambieranno silenziosamente risultati già pubblicati.

### Branch `v1.1/golden-master`

**Obiettivo.** Congelare il comportamento numerico attuale come test di regressione.

**Cosa fare.**
- Copia `results/` in `tests/fixtures/golden/v1.3.0/`.
- `tests/test_golden_master.py`: esegue `ccsu-run-pipeline` su un dataset committato e
  confronta ogni CSV/JSON con il golden, con tolleranza esplicita (`rtol=1e-9` per i
  deterministici, tolleranza dichiarata per i moduli con seed casuale).
- Fissa i seed ovunque ci sia stocasticità (bootstrap, RF/GB, CCM, surrogati). Se un
  modulo non è riproducibile, quello è il primo bug da chiudere.
- Marker `@pytest.mark.golden` per poterli lanciare da soli.

**Accettazione.** `pytest -m golden` verde due esecuzioni consecutive; nessun file di
`results/` diverso dal golden.

**Non fare.** Non cambiare nessun algoritmo in questo branch.

### Branch `v1.1/co2-units`

**Obiettivo.** Chiudere il caveat unità CO₂ (sezione 1.1).

**Cosa fare.**
- In `scripts/fetch_copernicus*.py`, converti `spco2` da Pa a µatm (×9.8692) al momento
  del parsing, con la costante nominata e commentata.
- Aggiorna `cross_check_co2()` in `build_dataset.py`: il confronto con `data.csv` va
  fatto **dopo** aver applicato la stessa conversione alla serie storica.
- Aggiungi `tests/test_data_quality.py::test_co2_in_expected_uatm_range` (350–500 µatm).
- Aggiorna README (togli il caveat, scrivi la risoluzione), `paper.md`, e rigenera il
  golden master in un commit separato e ben etichettato.
- Verifica esplicitamente che `cu_speciation` non cambi (usa pH, non CO₂).

**Accettazione.** Nuovo golden approvato da te commit-per-commit; test di range verde;
README senza caveat aperti.

### Branch `v1.1/dev-tooling`

**Obiettivo.** Infrastruttura che rende visibile la qualità del processo.

**Cosa fare.**
- `ruff` (lint+format), `mypy` in modalità permissiva sui moduli nuovi, `pre-commit`.
- Copertura test con soglia minima in CI (parti dal valore attuale, poi solo in salita).
- `CHANGELOG.md` (Keep a Changelog), `docs/adr/` con ADR-0001 = "perché generalizziamo".
- `CLAUDE.md` con le regole invarianti della sezione 0.
- `.github/ISSUE_TEMPLATE/`, `CODE_OF_CONDUCT.md`, `GOVERNANCE.md` (anche minimale:
  chi decide, come si propone un cambiamento, come si diventa maintainer).
- **Apri una issue pubblica per ogni branch di questo piano**, con label per fase, e
  lavora sempre via PR collegata alla issue. Costo: 20 minuti. Valore: trasforma uno
  sviluppo solo-e-privato in un processo aperto documentato, che è oggi un criterio
  editoriale esplicito (vedi sezione 6).
- Tag release `v1.3.0` su GitHub con note.

**Accettazione.** CI verde con lint+type+test+coverage; issue aperte; release taggata.

---

## 4. V2 — Il caso diventa configurazione

### Branch `v2.1/spec-schema`

**Obiettivo.** Sostituire le costanti globali con una specifica validata.

**Cosa fare.**
- Modelli `pydantic` v2: `SiteSpec`, `ResponseSpec`, `VariableSpec`, `EnvSpec`,
  `WindowSpec`, `StudySpec`. Dentro il package, non alla radice.
- Loader YAML + validazione + messaggi d'errore utili (riga/campo).
- Esporta lo schema come JSON Schema in `docs/schema/study.schema.json` (permette
  autocompletamento negli editor e validazione da parte di terzi).
- CLI `<pkg> validate study.yaml`.
- `config.py` alla radice diventa uno **shim deprecato** che legge
  `examples/livorno_paracentrotus/study.yaml` ed emette `DeprecationWarning`.
- ADR-0002: contratto sugli effect size (scrivilo ora, anche se lo implementi in V3).

**Accettazione.** Golden master verde; `config.py` non contiene più valori, solo il
caricamento; `<pkg> validate` fallisce in modo leggibile su YAML malformato.

**Non fare.** Non toccare ancora i moduli di analisi.

### Branch `v2.1/response-abstraction`

**Obiettivo.** Eliminare `"EC50"` come identità cablata.

**Cosa fare.**
- Nome colonna canonico interno: `response`. Il mapping dal nome sorgente avviene al
  caricamento tramite `ResponseSpec.column_map`. I CSV in `data/` **non si toccano**.
- `ResponseSpec` porta: `label` (per la UI), `unit`, `direction`
  (`higher_is_better` / `lower_is_better`), `taxon`, `endpoint`, `chemical`, `protocol`.
- Sostituzione meccanica delle 229 occorrenze nel dashboard con `spec.label` /
  `spec.unit`. Fallo con uno script + revisione, non a mano.
- `direction` va usato subito: introduci `signed_effect()` che normalizza il segno,
  altrimenti in V3 confronterai mele con pere senza accorgertene.

**Accettazione.** `grep -c '"EC50"' src/` → 0 fuori da `examples/`; golden verde;
dashboard identico a schermo con la config di Livorno.

### Branch `v2.1/provider-adapters`

**Obiettivo.** Ingestion pluggabile.

**Cosa fare.**
- `EnvProvider` Protocol + `CopernicusMarineProvider` (rifattorizza gli script esistenti)
  + `LocalCsvProvider` (per chi ha già i dati) + `ErddapProvider` (via `erddapy`).
- `ResponseSource` + `GoogleSheetSource`, `CsvSource`, `ExcelSource`.
- Credenziali **solo** da variabili d'ambiente / `.netrc`, mai da file di config.
  Documenta in `docs/CREDENTIALS.md`.
- Retry con backoff, cache locale su disco con chiave = hash(richiesta), gestione
  esplicita del fallback multiyear → analysisforecast già presente.
- Ogni provider espone `describe()` per la provenienza (dataset ID, versione prodotto,
  data di download, DOI del prodotto).

**Accettazione.** Il caso Livorno gira attraverso i provider senza differenze numeriche;
un test con `LocalCsvProvider` su dati sintetici passa senza credenziali (importante:
la CI non ha accesso a Copernicus).

### Branch `v2.1/dataset-builder`

**Obiettivo.** Sostituire `load_data()` e `build_dataset.py`.

**Cosa fare.**
- `AlignedDataset`: allineamento su indice temporale comune con regole **dichiarate**
  nello spec, non cablate: `resample`, `imputation` (`none` / `rolling_mean` /
  `interpolate`, con finestra), `gap_policy`, `min_coverage`.
- L'imputazione rolling-12 attuale diventa un'opzione esplicita del caso Livorno, non
  un comportamento di default invisibile. **Questa è una modifica sostanziale alla
  trasparenza metodologica del tool** — mettila in evidenza nella documentazione.
- Il backfill temperatura da `sst_daily` diventa una regola `fill_from` dichiarata.
- `df_full` / `df_real` diventano viste (`ds.observed` / `ds.with_imputed`) invece di
  due oggetti separati che si possono confondere.
- Shim `load_data()` retrocompatibile con `DeprecationWarning`.

**Accettazione.** Golden verde; nessuna scelta statistica presente solo nel codice
(controllo: leggendo `study.yaml` si deve poter ricostruire ogni trasformazione).

### Branch `v2.1/dashboard-modularization`

**Obiettivo.** Rendere il dashboard sopravvivibile a V3.

**Cosa fare.**
- Da `dashboard.py` (2910 righe) a package `dashboard/` con `tabs/<nome>.py`, una
  funzione `render(ctx)` per tab, nessun codice Streamlit a livello di modulo.
- **Rimuovi l'hack `del sys.modules` + `RERUN_LOCK`.** Con il codice dentro funzioni e
  `st.cache_data`/`st.cache_resource` usati correttamente, la causa radice (ri-esecuzione
  del corpo del modulo) sparisce. Se dopo il refactor il lock serve ancora, documenta
  in un ADR perché — ma verifica prima che non sia un sintomo risolto.
- Etichette, unità e titoli tutti da `ResponseSpec`.

**Accettazione.** Nessun file oltre ~400 righe; due sessioni concorrenti servite senza
serializzazione; screenshot pre/post identici.

**Nota.** Questo è il branch più a rischio di rompere cose in silenzio. Fallo con calma
e verifica tab per tab.

### Branch `v2.2/second-case-validation`

**Obiettivo.** Dimostrare la genericità invece di dichiararla.

**Cosa fare.**
- Due casi che non dipendono da nessuno:
  1. **Finestre temporali diverse** sulla stessa serie: stabilità dei risultati.
  2. **Celle ambientali spostate** (50, 200, 500 km): test di falsificazione. Se la
     correlazione regge con l'ambiente di un altro bacino, non è ambientale. Va conservato
     come controllo negativo permanente dello strumento.
- Almeno una **seconda serie di risposta**. Il criterio di scelta non è la rilevanza
  scientifica del caso ma quanto è *strutturalmente diversa* da EC50: altra unità di misura,
  altro endpoint, possibilmente altra direzione, magari altra cadenza. Una serie noiosa ma
  diversa nella forma mette alla prova il progetto più di una interessante ma identica.
- Sposta il caso attuale in `examples/livorno_paracentrotus/` (specifica + dati + notebook di
  riferimento) e aggiungi una cartella per ogni caso nuovo.
- Job CI che esegue tutti gli esempi.

**Vincolo da tenere presente fin d'ora**, senza farne un compito: prima o poi arriverà un
sito che la griglia ambientale non risolve (lagune, aree costiere confinate, acque interne).
Non progettare "il sito" assumendo che sia sempre una cella di modello marino: quando
succederà dovrà essere possibile una fonte in situ o una cella surrogata con avviso esplicito.

**Accettazione.** Almeno tre esempi in CI; `docs/ADAPTING.md` riscritto: la sezione 3 attuale
("la parte onesta: non c'è scorciatoia") deve poter essere cancellata.

> Rilascio `v2.0.0` qui, con CHANGELOG e nota di migrazione.

---

## 5. V3 — Molti casi, un confronto

### Branch `v3.1/case-model`

- `CaseSpec` + `case_id` = hash deterministico dello spec normalizzato.
- `results/<study_id>/<case_id>/<analysis>/…` — risolve B4.
- `manifest.json` per caso: hash degli input, versione del pacchetto, versioni delle
  dipendenze, timestamp, spec completa. È ciò che rende la riproducibilità
  **verificabile** invece che dichiarata.
- CLI `<pkg> cases list|show`.

### Branch `v3.1/analysis-plugin-api`

- `Analysis` Protocol + `Requirements` (variabili richieste, cadenza minima, n minimo,
  serve incertezza, serve serie giornaliera).
- Registry via entry points.
- Porta 3 moduli come pilota: `correlations`, `period_split`, `stationarity`.
- Skip esplicito e **loggato** quando i requisiti non sono soddisfatti: mai fallire in
  silenzio, mai produrre un numero su input inadeguati.

### Branch `v3.1/port-remaining-analyses`

- Porta il resto. Classifica ognuno:
  - **core generici**: `timeseries`, `correlations`, `period_split`, `stationarity`,
    `forecast`, lagged/CCF (da `mhw_analysis`).
  - **plugin opzionali domain-specific**: `cu_speciation` (richiede endpoint su metallo
    + pH/T/S), `mhw_detection`/`mhw_*`/`thermal_legacy` (richiedono SST giornaliera e
    hanno senso solo per stressor termico), `regime_shift`.
- I domain-specific vanno in `<pkg>.contrib.*` o in un pacchetto separato installabile
  come extra. Un tool generico che carica di default la speciazione del rame non è
  generico.

### Branch `v3.1/runner-cache`

- Orchestratore su N casi: sequenziale, poi parallelo (`joblib`/`concurrent.futures`).
- Cache content-addressed: se `hash(spec + dati + versione codice)` è invariato, salta.
  Con 20 casi × 13 analisi la differenza è tra secondi e mezz'ora.
- CLI `<pkg> run study.yaml [--case ...] [--analysis ...] [--force]`.

### Branch `v3.1/effect-contract`

- Implementa il contratto ADR-0002: ogni analisi restituisce `effects` con
  `estimate, ci_low, ci_high, n, scale, unit, direction_adjusted, n_tests`.
- `scale ∈ {raw, z, log_ratio, rank, percent_per_decade}`.
- Armonizzazione del segno via `ResponseSpec.direction`.
- Test: nessun `AnalysisResult` senza `effects` conforme (validalo in `__post_init__`).

### Branch `v3.2/inference-ledger`

**Questo è il branch che distingue il tool da un generatore di p-value.**

- **Ledger di molteplicità**: il runner somma `n_tests` di tutte le analisi della
  famiglia dichiarata, applica BH-FDR (e Bonferroni come riferimento), e produce
  `inference_ledger.json` con: quanti test eseguiti, quanti attesi significativi per
  caso, quanti osservati, quanti sopravvivono alla correzione.
- **Servizio di null model** condiviso: block bootstrap, phase randomization/IAAFT,
  surrogati AR-matched. Ogni analisi può chiedere una distribuzione nulla calibrata
  sulla propria serie.
- **Calcolatore di potenza**: dati n, autocorrelazione, rumore e cadenza, qual è il
  |ρ| minimo rilevabile all'80% di potenza (via simulazione). Da riportare **accanto a
  ogni risultato nullo**, così un "nessun effetto" diventa informativo invece che muto.
- Nel dashboard: p grezzo e p corretto sempre affiancati, mai il solo grezzo.

Motivazione da mettere nella documentazione: nel case study di Livorno il segnale
MHW→EC50 a lag 1 anno mostrava il 3.7% di test con p<0.05 contro un 4.9% atteso per caso,
e non sopravviveva a BH-FDR. Un tool che permette di generare N tuple × M lag senza
contabilizzare la molteplicità **produce quel tipo di falso positivo per costruzione**.
Il ledger non è una feature accessoria: è la ragione per cui questo tool è pubblicabile
e un for-loop sulle analisi no.

### Branch `v3.2/comparison-layer`

- Tabella effetti allineata cross-case.
- Forest plot con CI.
- Meta-analisi a effetti casuali (DerSimonian-Laird + REML): stima aggregata, τ², I², Q.
- Meta-regressione con moderatori (latitudine, specie, protocollo, lunghezza serie,
  laboratorio): risponde a "l'effetto varia sistematicamente con cosa?".
- Test di eterogeneità **prima** dell'aggregazione: se I² è alto, il tool deve dirlo
  a voce alta e sconsigliare la stima pooled.

### Branch `v3.2/dashboard-multicase`

- Selettore di caso; viste di confronto; export della tabella effetti.
- Il dashboard resta un consumatore di artefatti precalcolati: non spostare calcolo
  pesante a runtime.

> Rilascio `v3.0.0`.

---

## 6. V4 — Meta-tool (proposta, da rivedere quando ci arriverai)

Hai chiesto di suggerire. Ecco cosa secondo me rende V4 un contributo scientifico e non
solo un'astrazione in più. In ordine di valore.

### 6.1 Motore multiverse / specification curve — `v4/multiverse`

Il vero problema del confronto tra tuple non è tecnico, è epistemico: ogni tupla
moltiplica i gradi di libertà dell'analista. La risposta metodologicamente corretta
esiste già in letteratura (specification curve analysis, multiverse analysis) ma
**non esiste un'implementazione per serie ambiente↔biologia**.

- Dichiarazione esplicita dei gradi di libertà nello YAML (lag, detrending, aggregazione,
  split year, trasformazione, esclusione outlier).
- Esecuzione dell'intera griglia, curva delle specificazioni ordinata per effetto.
- Test dell'effetto mediano contro un nulla permutazionale sulla **stessa griglia**.
- File di **preregistrazione**: hash del piano di analisi congelato prima di guardare i
  risultati; il manifest registra se un'analisi era pre-registrata o esplorativa, e il
  report le separa visivamente.

Questo è il pezzo che rende il tool citabile come metodo, non solo come software.

### 6.2 Confronto tra organismi e risposte di natura diversa — `v4/cross-organism`

La tua intuizione, resa operativa. Il problema: EC50 in mg/L, indice gonadico
adimensionale, scope for growth in J/h/g non sono confrontabili. Serve:

- **Normalizzazione**: z-score rispetto a una baseline dichiarata, log-response ratio,
  variazione percentuale per decade, o rank-based. La scelta va dichiarata, non imposta.
- **Armonizzazione della direzione** (già in V3): alcune risposte crescono con lo stress.
- **Metadati di comparabilità**: due EC50 con protocolli diversi non sono la stessa
  quantità. Il tool deve segnalare quando si stanno confrontando casi con
  `protocol`/`endpoint`/`exposure` divergenti — warning, non blocco.

### 6.3 Modello gerarchico cross-case — `v4/hierarchical-model`

Invece di N analisi separate + un forest plot, un modello multilivello con effetti
casuali per sito e specie (partial pooling). Risponde a una domanda che nessuna analisi
per-caso può porre: *esiste una relazione comune ambiente→sensibilità, e quanto varia
tra popolazioni?* Implementazione con `bambi`/`numpyro` come dipendenza opzionale.
Dato il tuo interesse per i metodi bayesiani, è probabilmente il pezzo più divertente
e quello che dà il risultato scientificamente più nuovo.

### 6.4 Vocabolari controllati e unità — `v4/vocabularies-units`

Qui la "I" di FAIR smette di essere una dichiarazione:

- Unità con `pint` + UCUM, validate a ogni confine della pipeline. (Il bug CO₂/Pa non
  sarebbe potuto esistere.)
- Taxon → WoRMS AphiaID.
- Endpoint/effect → vocabolario controllato ECOTOX (US EPA), che è lo standard de facto
  ed è pubblicamente documentato.
- Sostanza → CAS / DTXSID (CompTox Dashboard).
- Metodo → codice standard (OECD/ISO/UNI).
- Export come Frictionless Data Package.

Bonus concreto: con l'allineamento a ECOTOX diventa possibile un provider che **importa
serie da ECOTOX Knowledgebase**, aprendo il tool a chiunque abbia dati là dentro. È il
tipo di integrazione con l'ecosistema esistente che viene valutata positivamente.

### 6.5 Provider oltre il marino — `v4/providers-beyond-marine`

ERA5 via `cdsapi`, ERDDAP, EMODnet, catalogo STAC/`intake`. Serve a rendere vera
l'affermazione "generico": finché tutti i provider sono marini, il tool è un tool marino
con un'interfaccia astratta.

### 6.6 Generatore di report — `v4/report-generator`

`<pkg> report study.yaml` → documento Quarto/Jinja parametrico con metodi e risultati
popolati dai manifest. Effetto collaterale prezioso: il paper non può più divergere dal
codice, perché i numeri li scrive il codice.

### 6.7 Documentazione e rilascio — `v4/docs-release`

Sito mkdocs/Sphinx, API reference, tutorial eseguibili, pagina "Design & trade-offs"
(che è dove vanno riassunti gli ADR), governance, rilascio `v4.0.0` + Zenodo.

---

## 7. Nota sulla pubblicazione (non è una priorità)

La priorità è l'utilità dello strumento, non la sede editoriale. Vale però la pena sapere
che **quasi tutto ciò che rende un tool utile coincide con ciò che lo rende pubblicabile**:
storia di sviluppo pubblica con issue e pull request, rilasci versionati, decisioni
architetturali documentate, riuso da parte di qualcuno che non sei tu, e un contributo
concettuale riassumibile in una frase.

Se lavori come descritto in `METODO_E_FASI.md` — issue, PR, ADR, changelog, casi d'esempio
eseguiti in CI — quel materiale si accumula da solo, senza costi aggiuntivi. Il giorno in cui
volessi tornare su una rivista di software, il lavoro sarà già fatto. Se quel giorno non
arriva, non avrai perso niente: sono le stesse pratiche che ti permettono di riprendere il
progetto dopo tre mesi di pausa.

L'unica cosa da non fare è il contrario: sviluppare in privato e pubblicare alla fine.

## 8. Rinominare, con attenzione

Il nome `climate_change_on_sea_urchins` è inservibile per un tool generico, ma:

- **Rinomina il repo, non ricrearlo**: GitHub mantiene redirect e, soprattutto, la storia
  pubblica (che ora conta come criterio editoriale).
- **Il DOI Zenodo dei rilasci esistenti resta valido**; il DOI concept continua a puntare
  al repo rinominato. Verifica dopo il rename.
- **L'URL Streamlit cambia**: aggiorna README, `paper.md`, e i link nel materiale
  supplementare pubblicato. Il branch `sartori-2023-supplement` e il tag
  `v0.1.0-sartori-2023` restano il riferimento stabile per il paper del 2023 — hai già
  usato questo pattern, riusalo.
- Mantieni per una minor version un package shim `climate_change_on_sea_urchins` che
  reimporta dal nuovo nome con `DeprecationWarning`.

Nomi possibili (verifica disponibilità su PyPI): `sentinel-lag`, `biolag`, `ecolag`,
`envresponse`. Preferisci qualcosa che descriva la relazione (ambiente→risposta ritardata)
e non il dominio marino, altrimenti tra due anni sei di nuovo qui.

---

## 9. Ordine di esecuzione consigliato

```
v1.1/golden-master        ← non saltare
v1.1/co2-units                  ← chiude un caveat pubblico aperto
v1.1/dev-tooling          ← attiva il processo aperto DA SUBITO
v2.1/spec-schema
v2.1/response-abstraction
v2.1/provider-adapters
v2.1/dataset-builder
v2.1/dashboard-modularization   ← il più rischioso, isolalo
v2.2/second-case-validation     ← qui si scopre se V2 funziona davvero
                                        [release v2.0.0]
v3.1/case-model
v3.1/analysis-plugin-api
v3.1/port-remaining-analyses
v3.1/runner-cache
v3.1/effect-contract
v3.2/inference-ledger           ← il cuore scientifico
v3.2/comparison-layer
v3.2/dashboard-multicase
                                        [release v3.0.0]
v4/vocabularies-units
v4/cross-organism
v4/multiverse
v4/hierarchical-model
v4/providers-beyond-marine
v4/report-generator
v4/docs-release
                                        [release v4.0.0 + Zenodo + submission]
```

Se il tempo è poco: **V1.1 + V2 completa + V3.1/V3.2 essenziali** è il
sottoinsieme minimo che produce qualcosa di difendibile. Il resto è espansione.

---

## 10. Punti aperti da decidere prima di iniziare

1. Il case study di Livorno resta nello stesso repo come esempio, o si separa? (Consiglio:
   resta — è il test di regressione ed è l'evidenza di impatto.)
2. Quale secondo caso reale, e chi lo esegue? È la decisione con più impatto sull'esito
   editoriale, e va presa ora perché condiziona il design.
3. Streamlit resta l'unica interfaccia, o CLI + API diventano il prodotto primario e il
   dashboard un consumatore fra tanti? (Consiglio: la seconda.)
4. Le analisi domain-specific (`cu_speciation`, `thermal_legacy`, `regime_shift`) restano
   nel core, in `contrib`, o in un pacchetto separato?
5. Retrocompatibilità: per quante versioni mantenere gli shim `config.py` / `load_data()`?
6. R: `mhw_lag_analysis.R` resta una dipendenza esterna opzionale, o si porta tutto in
   Python (DLNM ha equivalenti parziali)? Una dipendenza R rende l'installazione più
   fragile e complica la CI e Binder.
