# Stato del progetto

> Si aggiorna alla fine di **ogni** sessione di lavoro, prima del merge.
> Tenerlo corto: se supera una pagina, sposta il dettaglio in un ADR o in una issue.

**Ultimo aggiornamento:** 25 settembre 2026

---

## Dove siamo

**Versione corrente:** V2 · Il caso diventa configurazione — passo **V2.1 Specifica e
astrazione della risposta**, in corso

**Obiettivo del passo:** poter modificare qualunque cosa sapendo subito se un risultato
scientifico è cambiato. Da fine agosto si è aggiunto un obiettivo collegato: il pacchetto è
citato in un lavoro come mezzo di verifica indipendente, quindi deve riprodurre i numeri
pubblicati in modo verificabile.

**Fatto finora**
- `SPLIT_DATE` esplicito al 2016-06-01, propagato a tutti i moduli
- modulo `changepoint.py` con procedura QLR/AR(1), primaria sulla serie mensile, riusata su
  controllo negativo e metrica MHW annuale
- VIF per finestra in `thermal_legacy`, più sensibilità alla soglia (Tabella S2)
- ordinamento deterministico della sequenza grezza per (data, ID)
- **unità del CO₂ corrette** (`spco2` era in Pascal, fattore ×9.8692) — non tocca
  correlazioni/changepoint/stress index/speciazione/forecast, verificato con un confronto
  controllato dati-pre-fix vs dati-correnti
- quattro moduli nuovi per la sezione di riproduzione dei numeri del lavoro: controllo
  negativo (`negative_control.py`), contrasto pre/post per singola prova (`period_split.py`
  esteso), sensibilità alla soglia termica (`thermal_legacy.py` esteso); il changepoint sulla
  metrica MHW annuale (`mhw_annual_changepoint.py`) resta come modulo ma non è più citato nel
  lavoro — il paragrafo che descriveva è stato tolto dal manoscritto
- `tests/test_paper_values.py` + fixture congelata `tests/fixtures/paper_mpb_2026/`: riesegue
  i quattro moduli sopra sui dati congelati, non su `data/` né su `results/` precalcolato
- `requirements-lock.txt`: ambiente esatto verificato riproducibile (rigenerato results/,
  confrontato byte-per-byte)
- repository ripulito (notebook in `notebooks/`, vendored code in `legacy/`, paper JOSS
  abbandonato tolto dal repo, file orfani rimossi) e README rivisto
- **release v1.5.0 pubblicata e archiviata**: tag su GitHub, DOI concept Zenodo
  `10.5281/zenodo.19352308` (risolve sempre sull'ultima versione), merge in `main` fatto
- memoria di progetto: `CLAUDE.md`, `docs/roadmap/`, sei ADR nuovi (0001-0006) su decisioni già
  prese durante la v1.5.0
- **rete di sicurezza numerica su tutta la pipeline** (branch `v1.6.0/golden-master`):
  `tests/test_golden_master.py` riesegue l'intera pipeline (16 moduli Python + DLNM in R
  quando disponibile) sulla stessa fixture congelata di `test_paper_values.py` e confronta
  tutti i 64 file di `results/` contro un riferimento congelato
  (`tests/fixtures/results_v1_5_0/`), con tolleranze dichiarate per categoria (deterministico,
  bootstrap/RNG seedato, ottimizzatore MLE, e il caso noto della sensibilità al riscalamento in
  `ccf_results_prewhitened.csv`). Riproducibilità misurata: **esatta** (scarto relativo 0.0) su
  due run indipendenti, incluso R. Verificato con sabotaggio: un valore alterato a mano fa
  fallire il test indicando file e campo esatti, ripristinato torna verde.

**V1.1 chiusa** — entrambe le parti rimaste (memoria di progetto, rete di sicurezza) sono
fatte, poi due bug scoperti dal primo run reale su GitHub Actions: (1) una riga di
`.gitignore` (`trend_*.csv`, senza ancoraggio di percorso) scartava in silenzio 6 file da
`git add tests/fixtures/results_v1_5_0/` — il riferimento congelato era incompleto,
intercettato dal test di copertura stesso; (2) due tolleranze del golden master erano tarate
solo sulla mia macchina (scarto 0.0 lì) e troppo strette su macchina diversa
(`ccf_results_prewhitened.csv` e le colonne `r_arima`/`p_arima` di `robustness_severe_ccf.csv`,
entrambe dalla stessa ricerca a griglia ARIMA) — nuovo livello di tolleranza dedicato
(`ARIMA_FIT`), esplicitamente un controllo di errore grossolano, non di regressione sottile.
**Corretto su `main` con un fast-forward diretto, senza PR** — errore di processo segnalato
dall'utente il 22/9: da allora ogni merge in `main` passa da una pull request, comprese le
correzioni urgenti della CI (`CLAUDE.md` "Cose da non fare mai", aggiornato).

**V2.1 "Specifica e astrazione della risposta" — chiusa.** Sette PR, tutte fuse su `main`:
#5, #6 (specifica dichiarativa), #7 (convergenza ARIMA), #8, #10, #11, #12 (astrazione della
serie di risposta), #13 (questo CHANGELOG). Prossimo passo da decidere: V2.2 "Prova sul
campo", o il resto di V2.1 rimasto esplicitamente fuori (dashboard, adattatori di sorgente,
costruttore del dataset) — vedi `note-tecniche.md` §4.

**`v2.1/study-spec` (#6) + `docs/golden-master-issue-4-link` (#5)**: modelli pydantic
(`SiteSpec`, `ResponseSpec` con `ResponseSourceSpec`/`ResponseAggregationSpec` separati,
`VariableSpec`, `WindowSpec`, `StudySpec`), `examples/livorno_paracentrotus/study.yaml`,
`config.py` invertito a lettore della specifica, `docs/schema/study.schema.json` +
`ccsu-validate-study`. Correzioni sui campi rispetto alla prima bozza: `unit` `ug/L` non
`mg/L` (bug dashboard, issue #3); `adverse_direction: decrease|increase` non `direction`; `n`
rinominato `n_assays`. Commento CO2 corretto (config.py + study.yaml): Copernicus dichiara
correttamente `spco2` in Pa, l'errore di lettura era della pipeline — due occorrenze residue
della stessa imprecisione restano in `README.md`/`dashboard.py`, da correggere insieme
all'issue #3.

**`fix/arima-convergence` (#7)**: `_best_arima_order` scarta i candidati che non convergono
invece di scegliere il migliore per AIC a prescindere — corregge un bug reale
(`mhw_severe_intensity` e `mhw_days` avevano l'ottimo AIC su un fit non convergente).
**Scoperta durante la verifica**: anche scartando i non convergenti, la selezione dell'ordine
non è riproducibile fra macchine — due run CI consecutivi, stesso codice, hanno scelto ordini
diversi per `mhw_days` (45% mesi a zero: quali candidati convergono affatto dipende
dall'hardware/BLAS del runner). Golden master aggiornato di conseguenza:
`ccf_results_prewhitened.csv`, `prewhitening_diagnostics.json` e la parte `diagnostics` di
`robustness_severe_ccf_note.json` (nuovo file) sono ora in `STRUCTURAL_ONLY_FILES` — solo
colonne/righe/posizione dei NaN, mai i valori numerici del fit. `ARIMA_FIT` (la tolleranza
larga introdotta il 21/9) è ritirata. `mhw_severe_intensity`: braccio ARIMA marcato
`not_applicable` (confuso con lo spostamento strutturale 2016 di EC50, confermato
empiricamente: 10/13 p<0.05 crollano a 1/13 togliendo lo spostamento); resta il braccio con la
differenza prima. `tests/test_mhw_analysis.py` (nuovo): copre fit/filter/correlate con ordine
fissato (1,0,1), indipendente dal problema di selezione. I quattro nuovi p<0.05 di `mhw_days`
non sopravvivono a Bonferroni (soglia 0.00056, minimo 0.0052) e non sono riportati da nessuna
parte (non riproducibili). **La issue è la [#9](https://github.com/scatenag/climate_change_on_sea_urchins/issues/9)**,
non #4 come indicato nei riferimenti di codice durante la sessione — #4 non è mai esistita su
GitHub, era solo un segnaposto di conversazione; tutti i riferimenti nel codice sono stati
corretti prima della fusione.

**Astrazione della serie di risposta (#8, #10, #11, #12)**: `EC50` non è più un'identità
cablata in nessun punto di `src/` fuori da `common.py` (l'unico posto autorizzato a conoscere
il letterale su disco) e `dashboard.py` (branch separato, non iniziato).
- `ResponseSpec.label` (campo nuovo): identità di visualizzazione per gli output, distinta da
  `id`. Per Livorno `label == "EC50"`.
- `common.py`: `RESPONSE_COL`/`IMPUTED_COL` — valgono `"response"`/`"response_imputed"` dal
  #11 (prima `"EC50"`/`"EC50_imputed"`, come passo transitorio) — non un alias:
  `load_data()` rinomina le colonne fisse su disco in queste costanti al confine, non duplica
  la colonna; `load_ec50_raw()`/`load_ec50_monthly()` estendono il confine unico di lettura a
  `data/ec50_raw.csv`/`ec50_sheets.csv`, letti direttamente prima da `changepoint.py`,
  `period_split.py` e (#12, chiude il punto) `negative_control.py` (che non tocca la colonna
  di risposta, solo i controlli negativi, ma violava lo stesso confine).
- `config.py` esporta `RESPONSE_SPEC`; `pipeline.py` la carica una volta e la passa come
  `response=` ai **10 moduli** con identità di output: `timeseries`, `correlations`,
  `stationarity`, `regime_shift`, `period_split`, `cu_speciation`, `thermal_legacy`,
  `forecast`, `mhw_analysis`, `mhw_lag_extra`. Ogni identità (etichette riga/colonna, campi
  `"variable"`/`"series"`, chiavi di dizionario per-variabile, nomi file, nomi di colonna
  derivati come `EC50_bio`/`EC50_forecast`/`EC50_pred`) costruita da `response.label`, mai da
  `RESPONSE_COL`. Chiavi di schema fisse mai derivate da `RESPONSE_COL` in partenza (es.
  `regime_shift.py`: `"ec50_regime_shift"`) lasciate come sono: non a rischio di deriva,
  ridenominarle avrebbe forzato un aggiornamento del riferimento non richiesto.
- `mhw_analysis.py`: la mascheratura dei mesi imputati (`target == "EC50" and "EC50_imputed"
  in df.columns`, duplicata in tre punti — prewhitening, differenza prima, livelli grezzi; il
  terzo trovato durante il lavoro, non nominato esplicitamente ma identico, incluso per
  coerenza) è ora `_mask_imputed(df, target, values)`: maschera quando esiste
  `f"{target}_imputed"`, indipendentemente dal nome del target. 4 test nuovi in
  `tests/test_mhw_analysis.py`.
- **Il flip di `RESPONSE_COL`/`IMPUTED_COL` (#11) ha trovato due fughe reali**, esattamente
  come doveva: `timeseries.py` non era mai stato migrato (dimenticanza nell'audit originale
  dei moduli con identità di output — `trend_EC50.csv` sarebbe silenziosamente diventato
  `trend_response.csv`); `period_split.py`'s `dist_EC50.csv` aveva il nome file già corretto
  ma non l'intestazione di colonna interna. Entrambi corretti prima di fondere.
- `docs/adr/0000-decisioni-rimandate.md`: aggiunta voce 6, `label` non sanificato per l'uso nei
  nomi di file (rimandata a un secondo caso reale, V2.2).
- **Golden master: 66/66, zero deriva** in ogni passo — `label == "EC50"` per Livorno,
  indipendentemente da cosa vale `RESPONSE_COL` internamente.

**V2.2, prerequisiti e piano concordato (24/9)** — sequenza di sette PR:
1. `CCSU_STUDY` (#15, **fusa**). 2. Indirection dei risultati (`common.results_dir()`, #16,
**fusa**) + `forecast.py` al confine (oggi legge `RESULTS.parent / "data"`). 3. `SPLIT_DATE` e
`data_dir` nella specifica (ADR 0000 #5) — **[#17](https://github.com/scatenag/climate_change_on_sea_urchins/pull/17),
aperta 25/9, CI in corso**. Ha trovato e corretto un bug reale scoperto nel farlo, non previsto dal
piano originale: `mhw_detection.py` legava i percorsi di `data/` una sola volta a import-time da
`common.ROOT`, invisibile al monkeypatch dei fixture — ogni golden master leggeva/scriveva
silenziosamente i file reali invece della fixture congelata, rimasto verde solo perché il
rilevamento MHW è deterministico e i dati reali non erano cambiati nel frattempo (dettagli:
`tests/test_data_boundary.py`, `tests/conftest.py`, ADR-0007 per `split_date`). **PR #17 fusa
25/9** (confronto richiesto degli `mhw_*.csv`/`sst_daily.csv` fixture-vs-reale: byte-identici,
nessuna deriva del riferimento). 4. Spostamento di Livorno in `results/<study_id>/` con
`git mv` (golden master indipendente dal percorso) + ADR — **fatto (ADR-0008), #18 fusa
25/9**. `results_dir()` restituisce ora `results/<study_id>/`; i 64 file esistenti
spostati; `README.md` (4 link) e `tests/test_thermal_legacy_vif.py` (costruiva il percorso da
sé, lo stesso tipo di problema del punto 3) aggiornati di conseguenza.
5. Meccanismo delle finestre temporali, **spezzato in due PR** (25/9). Le finestre si
dichiarano in `StudySpec.windows`, **non** con una variabile d'ambiente: una sola esecuzione le
percorre tutte, risultati in `results/<study_id>/<window_id>/` (directory sorelle lette dalla
tabella di confronto); Livorno, senza finestre, resta in `results/<study_id>/` (finestra
implicita, ADR-0000 voce 7). MHW e climatologia: una volta sull'intero record, in `data_dir`,
prima del ciclo sulle finestre; solo `mhw_detection` scrive in `data_dir`.
**5a** `results=` parametro di `run()` al posto della costante `RESULTS`, nessun cambio di
comportamento — **fatta, PR aperta**; il golden master ora verifica per modulo chi scrive in
`data_dir`. **5b** `window=` parametro di `run()`, con due vincoli: (1) un modulo che riceve una
finestra e non sa applicarla fallisce con errore esplicito, mai la ignora; (2) regola di
default: i dati si costruiscono sull'intero record, la finestra seleziona i punti su cui si
calcola la statistica — ritardi e dosi cumulate possono usare la storia prima dell'inizio
della finestra; detrend, fit, decomposizioni e changepoint solo sui punti della finestra. Per
ogni modulo documentare nel codice cosa è costruzione e cosa è statistica; dove non è ovvio,
fermarsi e chiedere. Da disambiguare in `thermal_legacy.py`: `WINDOWS`/`window=` sono già la
finestra di dose in mesi, concetto diverso. 6. Script di
download per studio + sei `study.yaml` con coordinate segnaposto — **il download lo fa l'utente in
locale**, dopo aver scelto su mappa le celle in mare; istantanee committate con manifest di
provenienza, la CI non tocca mai la rete. 7. Celle spostate: 50/200/500 km in **due direzioni**
(lungo costa e verso il largo), sei studi.
- Finestra senza `SPLIT_DATE`: analisi pre/post saltate e registrate **in un campo dell'output dello
  studio**, non solo nel log.
- Termine di paragone sito↔cella: tre valori per variabile — serie grezze, destagionalizzate,
  destagionalizzate **e detrendizzate** (tendenza lineare); il terzo è il riferimento per la
  specificità spaziale. Più quota di mesi MHW in comune.
- Tabella di confronto **descrittiva** (nessun test nuovo). Metriche concordate:
  - A, solo serie di risposta (finestre; per le celle identiche per costruzione e servono a
    controllare quell'identità): calo pre/post % della serie mensile; data e p del changepoint
    QLR/AR(1) mensile; **controllo negativo**: test di tendenza sull'intero record e differenza
    di livello pre/post (serie che deve restare inerte: se reagisce, il guasto è nella catena di
    calcolo).
  - B, dipendenti dall'ambiente: Spearman sui trend EC50~temperatura e EC50~pH; CCF prime
    differenze `mhw_peak_intensity`→EC50, r e p al **lag 2 mesi** + n. lag p<0.05 su 13; thermal
    legacy finestra 24 mesi, rho detrendizzato e p parziale (**dose cumulata a 24 mesi**: grandezza
    diversa dal lag CCF, da distinguere nella tabella con l'unità); quota geochimica
    (letteratura) da `cu_speciation`; `total_mhw_days` lag **1 anno**, rho detrendizzato e p.
  - **Numerosità accanto a ogni metrica**: mesi con dato reale (risposta), coppie effettivamente
    correlate (ambiente).
  - Escluse: braccio ARIMA (issue #9, differenze del runner scambiate per distanza), Granger,
    forecast.

**Issue aperte:**
- [#3](https://github.com/scatenag/climate_change_on_sea_urchins/issues/3) — dashboard
  `mg/L`→`ug/L` (23 occorrenze, fattore 1000) + le due occorrenze residue dell'attribuzione
  CO2 sbagliata in `README.md`/`dashboard.py`
- [#9](https://github.com/scatenag/climate_change_on_sea_urchins/issues/9) — selezione
  dell'ordine ARIMA non deterministica fra macchine; per V3.1 (vedi sopra)

**Convenzioni di lavoro** (vedi `CLAUDE.md`): niente chiamate all'API GitHub con la
credenziale git salvata; `gh` CLI autenticato via device-flow è stato esplicitamente
autorizzato per creare/fondere PR quando richiesto esplicitamente (usato per PR #5, #6, #7,
#10, #11, #12 — ogni fusione confermata dall'utente volta per volta, mai una standing
authorization). Ogni merge in `main` passa da PR — l'assistente apre e verifica la CI,
l'utente decide quando fondere.

---

## Questioni aperte

- Il manoscritto è già stato sottomesso? Le tre modifiche al testo — numeri della dose
  termica, frase in §3.5, Data availability — vanno in bozze oppure prima dell'invio.
- Quale seconda serie di risposta per validare V2.2. Nessuna candidata copre il ventennio:
  righting response, volume di uova, contenuto proteico e NRRT partono da zero punti. Non
  tenerla sul percorso critico (vedi `METODO_E_FASI.md` §V2.2 punto 5).
- Nome definitivo del progetto e momento della rinomina del repository (vedi
  `note-tecniche.md` §8).
- Se le analisi in R restano una dipendenza opzionale o vengono portate in Python.

---

## Da non dimenticare

- **L'auto-update non si tocca.** È una funzionalità voluta ed è ciò che il lavoro rivendica
  nelle conclusioni. I trigger a schedule girano solo sul branch di default: per congelare
  basta un tag, non serve fermare né isolare niente.
- **La sequenza grezza delle prove non è un oggetto ben definito**: 59 delle 295 condividono
  la data, per via della registrazione a livello di mese. Con ordinamenti diversi F varia fra
  326 e 380 e la rottura salta fra giugno e settembre 2016. Risolto fissando l'ordinamento,
  ma è il motivo per cui la serie mensile è l'analisi primaria.
- `SPLIT_DATE` è giugno 2016 mentre il modulo changepoint riporta settembre: la stima
  rank-based cade a giugno e i risultati sono insensibili alla scelta entro l'intervallo
  (calo 43.0% contro 43.2%). Chi legge il codice senza il paper vede solo un'incoerenza:
  serve ancora un commento esplicito in `common.py` accanto a `SPLIT_DATE` — non fatto.
- **Riga ID 224 del foglio EC50 (2020-01, controllo negativo `14, 1, 14`): non è un errore di
  trascrizione.** Confermato da Sartori (settembre 2026): sono due saggi distinti. Il modulo
  `negative_control.py` lo lascia uncorretto di conseguenza; non riaprire la domanda.
- `results/ccf_results_prewhitened.csv` non è invariante a un riscalamento lineare degli
  input (scoperto verificando il fix CO₂): l'ottimizzatore MLE dell'ARIMA di prewhitening
  converge in modo leggermente diverso a scale diverse. Nessun cambio di significatività
  osservato, ma va documentato con tolleranza dedicata quando si costruisce il golden master
  (prossimo passo) — non standardizzare la serie per "risolverlo" senza deciderlo prima.
- Il dashboard aggira il sistema di import di Python e usa un lock globale che serializza
  tutte le sessioni: prima di rifattorizzarlo, capire se la causa è ancora presente (vedi
  `note-tecniche.md` B6 — il refactor di V2.1/V3.3 dovrebbe farla sparire, verificarlo).
- Le analisi specifiche del caso (speciazione del rame, ondate di calore, legacy termica)
  hanno senso solo su un endpoint da metalli con pH, temperatura e salinità e serie
  giornaliera: non sono generiche (vedi `note-tecniche.md` B8, destinate a `contrib.*` in V3.1).
- **Il paper promette che la pipeline è un modello adattabile da altri programmi di
  monitoraggio.** Oggi non lo è: `ADAPTING.md` dice che cambiare sito sono quattro costanti,
  cambiare indicatore no. È l'argomento più forte a favore di V2.
- Le colonne del foglio sorgente EC50 `pos`/`neg` **non sono controlli**: sono le
  semi-ampiezze (asimmetriche) dell'intervallo di confidenza (`UL-EC50`, `EC50-LL`). Vedi
  `note-dati-sorgente.md` — la specifica V2.1 deve portare `ci_low`/`ci_high` distinti, mai
  dedurli dai nomi delle colonne.
- **L'argomento `DATA_DIR` di `scripts/mhw_lag_analysis.R` (#17) è verificato solo a mano**
  (output byte-identico sui dati reali): nessuna CI di PR lo esercita, solo il workflow
  schedulato su `main`. Se lo si tocca di nuovo, riverificare a mano allo stesso modo.
- **`scripts/*.py` (fetch/build/explore) conserva ancora percorsi risolti a import-time da
  `ROOT`/`DATA`**, lo stesso pattern corretto in `mhw_detection.py` in #17 — lasciato così
  perché fuori dal pacchetto installabile (CLAUDE.md) e nessun fixture li monkeypatcha, quindi
  non a rischio nello stesso modo. Se `scripts/` dovesse mai entrare nel pacchetto o in un
  fixture di test, va rifatto lo stesso controllo.
- **Lo spostamento di `results/` in `results/<study_id>/` (ADR-0008) ha reso stale
  `scripts/make_regime_shift_figure.py`, `make_mhw_lag_annual_figure.py`,
  `make_speciation_figure.py`, `make_thermal_legacy_figure.py`, `build_narrative_notebook.py`**:
  costruiscono ancora `ROOT / "results"` da soli e non troveranno più i file di Livorno.
  Lasciati intatti (fuori dal pacchetto, nessun test li esercita) — se servono di nuovo per
  produrre figure del manoscritto, vanno aggiornati a `results/livorno-paracentrotus/` (o meglio,
  a `common.RESULTS`) prima di rilanciarli.

---

## Versioni

| Versione | Passo | Titolo | Stato |
|---|---|---|---|
| **V1** · Il caso | V1.1 | Messa in sicurezza | fatta |
| **V2** · Il caso diventa configurazione | V2.1 | Specifica e astrazione della risposta | in corso |
| | V2.2 | Prova sul campo | da iniziare |
| **V3** · Molti casi, un confronto | V3.1 | Analisi intercambiabili | da iniziare |
| | V3.2 | Confronto e inferenza | da iniziare |
| | V3.3 | Installabile e usabile da altri | da iniziare |
| **V4** · Meta-tool | — | Organismi e risposte di natura diversa, vocabolari, multiverso | rimandata |
| **V5** · Istanza condivisa | — | Multi-utente | rimandata |
