# 0021 — Le variabili mensili del catalogo, una alla volta: il pH per primo (M1.8)

**PROVVISORIA** per due scelte (marcate sotto); reversibile, nessun effetto su Livorno.

## Contesto

Il catalogo aveva la sola SST giornaliera (ADR-0010). Uno studio nuovo che vuole la speciazione del rame, il forecast o le
correlazioni fra variabili ambientali ha bisogno di variabili mensili: salinità, ossigeno, pH, CO₂ (la temperatura è la media
mensile della SST, ADR-0020). La riga M1.8 le vuole una alla volta, ciascuna con il comando e l'app insieme. Questa PR porta la
parte comune e la prima variabile, il pH, perché è quella che fa sparire un motivo dalla tabella delle analisi spente.

## Decisione

- **Voce di catalogo `ph_monthly`** uguale a ciò che `scripts/fetch_copernicus.py` scarica per Livorno: prodotto
  `cmems_mod_med_bgc-car_my_4.2km_P1M-m` (e `…_anfc_…` come prodotto che lo segue), variabile `ph`, profondità 0–10 m, cadenza
  mensile; un test legge lo script e vuole l'uguaglianza. **Il dominio è quello del prodotto biogeochimico** (longitudine da −5.54,
  non −6.0 come la fisica), letto dal catalogo Copernicus il 20/11/2026: un sito nel Tirreno è dentro, uno a ovest di −5.54 no.
  Ogni voce dichiara il nome della propria colonna (`column`: `Temperature`, `pH`): è il nome nel file della serie e nel dataset.
- **Copertura di un prodotto mensile.** Il catalogo Copernicus non dà minimo e massimo del tempo ma l'elenco dei mesi (`values`):
  la copertura è il primo e l'ultimo (pH: 1999-01 → 2026-08, versione 202511). Per un prodotto giornaliero resta il minimo e il massimo.
- **`download.fetch_monthly`** (e `fetch_series`, che sceglie per cadenza): la media sulle celle di mare della scatola e poi sulla
  profondità, un valore al mese, dal mese di `start` all'ultimo mese del prodotto; un mese senza valore resta mancante, mai riempito
  (stessa regola della SST); la conversione nominata nel catalogo si applica una volta sola. I controlli prima della richiesta lunga
  (credenziali, dominio, copertura, un primo mese di prova, terra/mare) sono gli stessi della SST (`_open_product`). Il file si
  chiama come l'id (`ph_monthly.csv`, `ph_monthly.manifest.json`; quelli della SST restano `sst_daily.*`). `ccsu-download-sst` ha
  `--variable`.
- **Il dataset** (`build_dataset(variables=…)`) unisce la variabile alla tabella mensile accanto alla temperatura, sui mesi della
  griglia; i mesi che la variabile non ha restano mancanti; file di un'altra cadenza, con date che non sono il primo del mese, con
  mesi doppi o senza nessun mese dello studio sono rifiutati con il motivo; `coverage.variables` dice che cosa copre e quanti mesi
  hanno un valore.
- **`ccsu-run-study`** legge un file per ogni variabile mensile che lo studio elenca (`<id>.csv` nella cartella degli ingressi;
  se manca, fallisce al passo di lettura dicendo il file), ne scrive gli hash e i manifest in `run.json` (`variable_manifests`).
  **Non è un passo in più** (la lettura sta in `read_sst`). La CCF non cambia: il driver resta l'anomalia della SST; la variabile
  entra nel dataset e fa cambiare lo stato delle analisi (`analyses.json`).
- **App.** Nel passo 1 un multiselect («Monthly variables to download with the temperature (optional)», nessuna scelta di
  partenza): le scelte si scaricano nello stesso invio, con le stesse credenziali, una richiesta dopo l'altra; un sito fuori dal dominio
  di una variabile scelta è rifiutato prima di ogni richiesta; la copertura del prodotto di ogni variabile si legge prima di scaricare
  la SST (il catalogo è pubblico): se finisce prima del periodo, errore prima di scaricare; se **comincia dopo** (il pH dal 1999, la
  baseline suggerita da 1991), la variabile parte dal primo mese del suo prodotto e la pagina lo dice, i mesi prima restano senza
  valore e nessuno è inventato; un errore di una variabile non lascia nulla (neppure la SST); cambiare
  la scelta dopo il download dice che i dati sono per gli ingressi precedenti. Il passo 2 mette le variabili nell'`environment`
  dello studio e ne copia i file (anche per la serie di prova); la pagina dei risultati dice le variabili del dataset e quanti mesi
  hanno un valore. I conteggi di `coverage.variables` sono quelli dei mesi dello studio
  (`months_with_value`, `months_without_value`); i mesi del file fuori dallo studio sono in `months_outside_the_study`.
- **Verifica sul servizio vero (non ancora eseguita).** `scripts/verify_download.py` (workflow `verify_download.yml`, a mano o con l'etichetta) scarica il
  pH di Livorno e lo confronta con la colonna `pH` di `data/env_copernicus.csv` sui mesi comuni: **fallisce per una differenza
  grossolana (≥ 0.02)**, conta e riporta i mesi che differiscono di più di 1e-4 (la versione della rianalisi può non essere la stessa
  dei mesi più vecchi del file: PROVVISORIA, la soglia si stringe dopo la prima misura) o per meno di 12 mesi in comune. Il cubo mensile, i tempi
  del toolbox (primo del mese) e la richiesta di un solo mese non sono provati con il toolbox vero: la prova è il workflow con
  l'etichetta `verifica-download` (a mano, del proprietario), e fino ad allora il pezzo non è chiuso.
- **PROVVISORIA: nessuna coda analysis-forecast.** Il file di Livorno aggiunge ai mesi del prodotto reprocessato quelli del prodotto
  che lo segue; qui, come per la SST, si scarica solo il prodotto reprocessato e la serie finisce dove finisce il prodotto (la voce
  del catalogo registra l'id del seguente). Aggiungere la coda è una scelta che cambia la fine della serie.

## La salinità (20/11)

`salinity_monthly` segue lo stesso schema, senza codice nuovo oltre alla voce e alla verifica: prodotto
`cmems_mod_med_phy-sal_my_4.2km_P1M-m` (e `…_anfc_…`), variabile `so`, profondità 0–10 m, colonna `Salinity`, unità `0.001` (come la dà
il catalogo Copernicus), **dominio fisico, uguale a quello della temperatura** (longitudine da −6.0: un sito a ovest di −5.54 ha la
salinità e non il pH). Uguale a ciò che scarica `scripts/fetch_copernicus.py` (un test, per ogni voce, vuole l'uguaglianza). Lo script
di verifica ha un caso per ogni variabile mensile (`MONTHLY_CASES`), con il proprio limite grossolano, scelto **sotto la mediana
della variazione da un mese al successivo** in `data/env_copernicus.csv` (pH 0.0276 → 0.02; salinità 0.0549 → 0.04; misura del
20/11), così che anche uno sfasamento di un mese sia visto: PROVVISORIO come il resto della verifica, non eseguita sul servizio vero.
Restano O₂ e CO₂ (la CO₂ con la conversione Pa → µatm).

## L'ossigeno (20/11)

`o2_monthly`, ancora la stessa forma: prodotto `cmems_mod_med_bgc-bio_my_4.2km_P1M-m` (e `…_anfc_…`), variabile `o2`, profondità 0–10 m,
colonna `O2`, unità `mmol m-3` (che è µmol/L, come dice lo script di Livorno), **dominio biogeochimico** (lo stesso del pH: un test lo
vuole uguale). Caso di verifica con limite grossolano 4.0, sotto la mediana della variazione mese-mese (5.79) di `data/env_copernicus.csv`.
Resta la CO₂.

## Cosa non è in questa PR

La CO₂ (una PR a sé, M1.8; porta la conversione Pa → µatm nominata, `config.CO2_PA_TO_UATM`; la CO₂ può non avere profondità e va provata sul servizio); la temperatura 0–10 m come voce a sé (oggi la fornisce la SST: la
scelta di quale temperatura entra nelle analisi è del proprietario, e `analyses.VARIABLE_SOURCES` si aggiorna se cambia).

## Alternative

- **Una voce per tutte le variabili mensili in una PR**: la riga di M1.8 le vuole una alla volta, per vedere ogni voce uguale a
  quella di Livorno e ogni serie uguale al file.
- **Riempire un mese mancante** (interpolazione o ripetizione): rifiutato, COME_SI_LAVORA §2.
- **Un passo in più in `ccsu-run-study`** (`read_variables`): cambierebbe l'avanzamento che l'app e i test leggono per un lavoro di
  millisecondi.
