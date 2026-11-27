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
- **Verifica sul servizio vero (non ancora eseguita).** `scripts/verify_download.py` (workflow `verify_download.yml`, a mano o con
  l'etichetta) scarica ogni variabile mensile di Livorno e la confronta con la sua colonna di `data/env_copernicus.csv` in due
  intervalli (sotto, «La verifica in due intervalli»). Il cubo mensile, i tempi del toolbox (primo del mese) e la richiesta di un
  solo mese non sono provati con il toolbox della pipeline: la prova è il workflow, lanciato dal proprietario, e fino ad allora il
  pezzo non è chiuso.
- **PROVVISORIA: nessuna coda analysis-forecast.** Il file di Livorno aggiunge ai mesi del prodotto reprocessato quelli del prodotto
  che lo segue; qui, come per la SST, si scarica solo il prodotto reprocessato e la serie finisce dove finisce il prodotto (la voce
  del catalogo registra l'id del seguente). Aggiungere la coda è una scelta che cambia la fine della serie.

## La salinità (20/11)

`salinity_monthly` segue lo stesso schema, senza codice nuovo oltre alla voce e alla verifica: prodotto
`cmems_mod_med_phy-sal_my_4.2km_P1M-m` (e `…_anfc_…`), variabile `so`, profondità 0–10 m, colonna `Salinity`, unità `0.001` (come la dà
il catalogo Copernicus), **dominio fisico, uguale a quello della temperatura** (longitudine da −6.0: un sito a ovest di −5.54 ha la
salinità e non il pH). Uguale a ciò che scarica `scripts/fetch_copernicus.py` (un test, per ogni voce, vuole l'uguaglianza). Lo script
di verifica ha un caso per ogni variabile mensile (`MONTHLY_CASES`).
Restano O₂ e CO₂ (la CO₂ con la conversione Pa → µatm).

## L'ossigeno (20/11)

`o2_monthly`, ancora la stessa forma: prodotto `cmems_mod_med_bgc-bio_my_4.2km_P1M-m` (e `…_anfc_…`), variabile `o2`, profondità 0–10 m,
colonna `O2`, unità `mmol m-3` (che è µmol/L, come dice lo script di Livorno), **dominio biogeochimico** (lo stesso del pH: un test lo
vuole uguale).
Resta la CO₂.

## La CO₂ (20/11)

`co2_monthly`: prodotto `cmems_mod_med_bgc-co2_my_4.2km_P1M-m` (e `…_anfc_…`), variabile `spco2`, colonna `CO2`, dominio biogeochimico.
**La conversione è nel catalogo, con nome**: il prodotto dà la pressione parziale in pascal (`unit_native: Pa`), le analisi usano i µatm
(`unit_analysis: µatm`), fattore `1e6 / 101325` (1 atm = 101325 Pa), applicato una volta sola da `fetch_monthly` e scritto nel manifest;
un test vuole il fattore uguale a `config.CO2_PA_TO_UATM`, la costante che gli script di Livorno applicano. È la correzione dell'errore
del 10/9 (il valore letto come µatm), che qui non si ripete perché l'unità è dichiarata in un punto solo. Il prodotto è di superficie e
non ha una coordinata di profondità: la voce dichiara comunque 0–10 m, perché è ciò che lo script di Livorno chiede al toolbox (e
uguale a ciò che lo script scarica è il criterio). Il proprietario ha verificato sul servizio vero (27/11, toolbox 2.5.2) che il
dataset non ha l'asse della profondità e che la richiesta senza profondità funziona. La richiesta **con** la profondità, quella che fa
questa pipeline, è quella del job di aggiornamento: il 5/10 ha scaricato così il dataset `…_my_…` (toolbox 2.5.0, run 37319831363) e il
5/11, con il toolbox della serie della pipeline (2.3.0, run 42042082130), il dataset `…_anfc_…` (il `…_my_…` finiva ad agosto). Il
`…_my_…` con la serie 2.3 non è ancora provato: lo prova il workflow. Con la CO₂ tutte le variabili mensili di Livorno sono nel
catalogo.

## La verifica in due intervalli (28/11)

**Misura del proprietario sul servizio vero** (27/11, toolbox 2.5.2, fuori da questa pipeline; box di `scripts/fetch_copernicus.py`,
43.4278 N 10.3956 E ±0.1°, 0–10 m): i cinque dataset riproducono `data/env_copernicus.csv` fino a maggio 2023, con differenza
mediana di circa 1e-5; da giugno 2023 a ottobre 2026 (così nella misura; il file nel repository arriva a settembre 2026) i valori
differiscono, con massimo assoluto di 4.6 per l'O₂ e di 20.3 µatm per la CO₂. L'indagine sulla causa è riferita al proprietario; dati e
file di Livorno non cambiano.

La soglia si applica al **massimo**, non alla mediana. Sui file grezzi del download di Livorno dell'11/3/2026 (locali, non versionati) la
riduzione di questa pipeline (`download._monthly_series`, con la conversione del catalogo) riproduce il file fino a 2023-05 con
massimo 6e-5 (la CO₂, che ha il maggiore; misura del 28/11): è una prova della riduzione, non del servizio di oggi né del toolbox.

**Decisione del proprietario:** il confronto è diviso in due intervalli, ciascuno con il proprio esito, così un fallimento si legge
per intervallo. Fino a 2023-05 la soglia è **1e-3** per ogni variabile; da 2023-06 è larga. In ogni intervallo servono almeno 12 mesi
in comune; i mesi che differiscono di più di 1e-4 sono contati e riportati per intervallo.

**PROVVISORIA: i valori della soglia larga.** Sono la variazione più grande da un mese al successivo di `data/env_copernicus.csv`,
arrotondata (misura del 28/11: pH 0.0751 → 0.075; salinità 0.7401 → 0.74; O₂ 22.99 → 23; CO₂ 99.58 → 100). Fermano un errore
grossolano (un'unità, per esempio la CO₂ in pascal, o un'altra variabile), **non** uno sfasamento di un mese: quello lo vede il primo
intervallo, che passa per lo stesso codice. Un errore che riguardasse soltanto i mesi da giugno 2023 e restasse sotto la soglia larga
passa. Sostituiscono i limiti del 20/11 sull'intera serie (pH 0.02, salinità 0.04, O₂ 4.0, CO₂ 22.0), che avrebbero fatto fallire
l'O₂ per la differenza che il proprietario ha misurato.

## Cosa non è in questa PR

La temperatura 0–10 m come voce a sé (oggi la fornisce la SST: la
scelta di quale temperatura entra nelle analisi è del proprietario, e `analyses.VARIABLE_SOURCES` si aggiorna se cambia).

## Alternative

- **Una voce per tutte le variabili mensili in una PR**: la riga di M1.8 le vuole una alla volta, per vedere ogni voce uguale a
  quella di Livorno e ogni serie uguale al file.
- **Riempire un mese mancante** (interpolazione o ripetizione): rifiutato, COME_SI_LAVORA §2.
- **Un passo in più in `ccsu-run-study`** (`read_variables`): cambierebbe l'avanzamento che l'app e i test leggono per un lavoro di
  millisecondi.
