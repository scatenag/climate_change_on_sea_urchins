# Il percorso fin qui

> Cosa è successo dal consolidamento della versione citata nel lavoro (settembre 2026) a oggi:
> le fasi, le decisioni con il loro ADR, e gli errori silenziosi trovati, ciascuno con la causa
> e la regola di lavoro che ne è nata. Lo stato corrente è in [`STATO.md`](STATO.md), le regole
> in [`../COME_SI_LAVORA.md`](../COME_SI_LAVORA.md).
>
> Si aggiorna quando si chiude una fase o si trova un errore silenzioso nuovo. Non è un diario
> di sessione: il dettaglio sta nei commit, nelle PR e nel `CHANGELOG.md`.

---

## 1. Le fasi

| Periodo | Fase | Cosa ha prodotto | PR / commit |
|---|---|---|---|
| fino al 4/9/2026 | **V1, il caso** | analisi di un sito e di una serie di risposta (EC50 di *P. lividus* al largo di Livorno), dashboard, aggiornamento automatico dei dati; v1.4.0 | fino a `v1.4.0` |
| 10–14/9 | **Consolidamento della versione del lavoro** | unità del CO₂ corrette, una sola esecuzione ufficiale di `results/`, fixture congelata con il test dei valori del lavoro, ambiente bloccato, tag `v1.5.0` e DOI Zenodo | `7d4b75d`, `cb333cd`, `v1.5.0`; ADR-0001…0006 |
| 18–22/9 | **V1.1, messa in sicurezza** | golden master su tutta la pipeline (66 file), memoria di progetto (`CLAUDE.md`, `STATO.md`, ADR) | `799908c`, `62a1d94`, #5 |
| 22–24/9 | **V2.1, specifica e astrazione della risposta** | `study.yaml` validato con pydantic, `config.py` lettore della specifica, la risposta non si chiama più EC50 nel codice di analisi | #6, #7, #8, #10–#13 |
| 24–28/9 | **Prerequisiti di V2.2** | studio scelto con `CCSU_STUDY`, un solo risolutore dei percorsi, `data_dir`/`split_date`/climatologia nella specifica, `results/<study_id>/`, `results=` e `window=` parametri di `run()`, finestre temporali nei primi quattro moduli | #15–#20, #22, #25; ADR-0007, ADR-0008 |
| 28–29/9 | **Igiene degli output e dei dati** | nessuna identità, conteggio o conclusione scritti a mano negli output; il file per prova committato dall'aggiornamento | #26, #27 |
| 29/9–2/10 | **Il branch del lavoro** | il dashboard citato gira da `paper/mpb-2026` a codice congelato, con dati aggiornati ogni giorno da un job dedicato; due app pubbliche | #28, #30, #32, #36; ADR-0009 |
| 1–2/10 | **Bordi delle serie e preparazione unica dei dati** | mesi senza SST mancanti, nessun riempimento oltre i bordi, dashboard e job con le stesse funzioni; portate anche sul branch del lavoro e ricalcolate | #33, #34, #35; branch `3d41358`, `3f738c0` |
| 3–8/10 | **Protezione completa di `main`** | deploy key con scrittura per il job di aggiornamento, segreto in un environment limitato a `main`, token in sola lettura; ruleset attivato dopo il primo push reale con la chiave (5/10): PR obbligatoria senza approvazioni, controllo `test`, nessuna eccezione per le persone | #46; ruleset 24719143 |
| 2/10 | **Chiusura e cambio di direzione** | inventario, protezione parziale del repository (ruleset contro cancellazione e force push, cancellazione automatica dei branch fusi), questa memoria, definizione della milestone M1 | [`MILESTONE-M1.md`](MILESTONE-M1.md) |

Prima del 22/9 i commit entravano su `main` anche senza PR. Dal 22/9 ogni modifica passa da una
PR (vedi §3, errore 13).

---

## 2. Decisioni principali

| Decisione | Dove |
|---|---|
| `split_date` (giugno 2016) resta il confine pre/post; la stima QLR/AR(1) (settembre) è una conferma indipendente, non si riconciliano | [ADR-0001](../adr/0001-split-date-vs-changepoint-september.md) |
| La serie mensile è l'analisi primaria del changepoint; la sequenza per prova è secondaria (solo l'anno è stabile) | [ADR-0002](../adr/0002-monthly-series-primary-changepoint.md) |
| La sequenza per prova si ordina per (data, ID) | [ADR-0003](../adr/0003-raw-sequence-ordering-datetime-id.md) |
| I valori del lavoro si verificano su una fixture congelata, mai su `data/` | [ADR-0004](../adr/0004-frozen-fixture-for-paper-values-test.md) |
| `requirements.txt` permissivo, `requirements-lock.txt` esatto per la release | [ADR-0005](../adr/0005-requirements-lock-vs-loose.md) |
| L'aggiornamento automatico non si sospende; i numeri si congelano con un tag | [ADR-0006](../adr/0006-auto-update-never-pauses-freeze-via-tag.md) |
| `split_date` è un campo della serie di risposta, validato sui dati | [ADR-0007](../adr/0007-split-date-in-response-spec.md) |
| `results/` è per studio: `results/<study_id>/` | [ADR-0008](../adr/0008-results-per-study-directory.md) |
| Il dashboard citato gira da un branch a codice congelato, con dati aggiornati | [ADR-0009](../adr/0009-paper-branch-frozen-dashboard-code.md) |
| Decisioni volutamente rimandate | [ADR-0000](../adr/0000-decisioni-rimandate.md) |

Decisioni prese senza un ADR proprio, registrate nel `CHANGELOG.md`: le finestre temporali si
dichiarano nella specifica e una sola esecuzione le percorre tutte; un modulo che non sa
applicare una finestra non gira per quella finestra (rifiuto esplicito, mai la finestra
ignorata); la regola (a)/(b)/(c) delle finestre; i file del prewhitening ARIMA confrontati nel
golden master solo nella struttura.

---

## 3. Errori silenziosi trovati, e le regole nate da loro

Un errore è *silenzioso* quando il codice gira, i test passano e il risultato sbagliato arriva
in un output pubblico. Per ciascuno: cosa succedeva, perché, come è emerso, quale regola ne è
nata.

### 1. Il CO₂ in Pascal (corretto il 10/9, `7d4b75d`)
- **Cosa.** `spco2` di Copernicus è in Pascal; la pipeline lo trattava come µatm. Valori 31–58
  invece di circa 400, con una nota nel README che dichiarava l'unità «non verificata».
- **Perché.** L'unità non era dichiarata in nessun punto della pipeline. Il controllo incrociato
  con la serie storica dava rapporto 0.99 perché anche quella era in Pascal: due fonti con lo
  stesso errore si confermavano a vicenda.
- **Regola.** Ogni variabile porta la sua unità e la sua conversione, dichiarate e nominate in
  un punto solo; un intervallo di plausibilità fisica la controlla a ogni aggiornamento. Un
  accordo fra due fonti non prova niente se possono condividere l'errore.

### 2. EC50 in mg/L nel dashboard (issue #3, corretto il 24/9, #14)
- **Cosa.** 23 etichette in mg/L invece di µg/L sull'app pubblica: un fattore 1000.
- **Perché.** Etichette e unità scritte a mano nell'interfaccia, invece di venire dalla
  specifica della risposta.
- **Regola.** Etichetta e unità di una serie vengono dalla specifica, mai da un letterale.

### 3. Il Granger vuoto (11/9, `9b0db03`, `7b5fd5e`)
- **Cosa.** Per circa tre settimane il pannello Granger del dashboard è stato vuoto, senza
  nessun messaggio d'errore.
- **Perché.** Una versione nuova di statsmodels aveva tolto un argomento; senza limite di
  versione l'ambiente del job l'aveva installata. Il modulo catturava l'eccezione e scriveva un
  segnaposto `{"error": …}`, e il dashboard scartava in silenzio le voci con errore.
- **Regola.** Un errore non si filtra in silenzio: si mostra dove si guarda il risultato.
  L'ambiente della versione citata è bloccato (ADR-0005).

### 4. Percorsi risolti all'import (trovato il 25/9, #17)
- **Cosa.** `mhw_detection.py` calcolava i percorsi dei file una volta sola, all'import. Le
  fixture dei test reindirizzano i percorsi dopo l'import: il golden master leggeva la SST
  reale e sovrascriveva i file MHW reali, mentre il resto della pipeline usava la fixture.
- **Perché.** Un valore derivato dal confine dei dati e legato a un nome di modulo resta fermo
  anche quando il confine cambia.
- **Regola.** I percorsi si risolvono al momento della chiamata, dal confine unico
  (`common.DATA`, `common.results_dir()`); un test statico vieta di costruirli altrove, e il
  golden master verifica che `data/` e `results/` reali restino intatti.

### 5. Il golden master verde per coincidenza (21–25/9)
- **Cosa.** Tre volte un test è rimasto verde senza verificare quello che diceva: (a) il golden
  master dell'errore 4 passava perché la SST reale coincideva ancora con quella della fixture;
  (b) `test_mhw_analysis.py` leggeva `data/` reale dopo #17 ed è diventato rosso solo quando
  l'aggiornamento del 25/9 ha aggiunto un mese (#20); (c) una regola di `.gitignore` senza
  ancoraggio escludeva in silenzio 6 file dal riferimento congelato (`62a1d94`), e due
  tolleranze tarate su una sola macchina erano troppo strette su un'altra.
- **Perché.** Un test che gira sui dati vivi, o su una sola macchina, è verde finché i dati o la
  macchina non cambiano, non perché il codice è giusto.
- **Regola.** I test girano su fixture congelate, mai sui dati vivi. Un test nuovo va visto
  fallire su un sabotaggio prima di considerarlo valido. Un riferimento congelato ha un test di
  copertura (tutti i file attesi presenti).

### 6. Mesi senza SST trattati come mesi senza ondate (corretto il 1/10, #33)
- **Cosa.** Oltre la copertura della SST giornaliera, `load_data()` metteva a 0 le metriche
  MHW, cioè «nessuna ondata di calore», in piena estate; tre analisi poi riempivano in avanti.
  Sulla fixture congelata il Granger MHW→O₂ aveva 6 p grezzi sotto 0.05 invece di 3, per un
  salto inventato dallo 0 di luglio 2026.
- **Perché.** Riempire con 0 un valore mancante sembra neutro, ma per un conteggio di eventi 0 è
  un'osservazione.
- **Regola.** Un dato mancante resta mancante.
- **Seguito trovato il 2/10.** La copertura della SST non è ferma solo per il ritardo del
  prodotto: `scripts/fetch_copernicus_daily.py` ha la data di fine scritta a mano
  (`END = "2026-06-30"`), mentre il prodotto multiyear arriva al 31/8/2026. È la seconda volta
  (a luglio la stessa data era indietro di due anni, `09b928a`). Issue #45; corretta su `main`
  dalla #44 (fine letta dal catalogo, test che fallisce se torna un letterale); sul branch del
  lavoro su richiesta esplicita.

### 7. Riempimenti ai bordi delle serie (corretto il 1/10, #34)
- **Cosa.** `interpolate`, `ffill` e `bfill` prolungavano le serie oltre l'ultimo mese
  osservato: il forecast si addestrava con il valore di maggio ripetuto a giugno e luglio come
  se fosse misurato.
- **Perché.** Le funzioni di riempimento di pandas, ai bordi, estrapolano.
- **Regola.** Un dato mancante resta mancante anche ai bordi. Ogni correzione ha un test con tre
  mesi mancanti in coda.

### 8. Testi con conclusioni fisse (corretto il 29/9, #27)
- **Cosa.** Il verdetto del regime shift diceva sempre «AR(1) not rising» mentre il valore
  calcolato accanto era tau +0.14, p=0.017; una distanza di tre anni era presentata come
  «accumulation lag» senza dire se la rottura MHW fosse significativa; il titolo di una figura
  affermava la tesi che il lavoro confuta. La guardia RSPTEST ha trovato 21 punti in cui
  l'etichetta EC50 era scritta a mano negli output; il rovesciamento di `RESPONSE_COL` (#11)
  aveva già trovato due moduli mai migrati.
- **Perché.** Testi scritti quando i numeri dicevano un'altra cosa, mai rigenerati.
- **Regola.** Negli output nessuna identità, nessun conteggio, nessuna data e nessuna conclusione
  scritti a mano: tutto viene dai dati o dalla specifica. Guardia:
  `tests/test_response_label_leaks.py`.

### 9. La doppia preparazione dei dati (corretto il 1/10, #35)
- **Cosa.** Il dashboard preparava i dati per conto suo (aggregazione, una passata di
  imputazione invece di due, unione MHW, lisciatura delle correlazioni). I valori ricalcolati
  dal vivo differivano da quelli del job fino a 0.03 nelle correlazioni e 1.1 µg/L nel forecast.
- **Perché.** Due implementazioni della stessa grandezza divergono, e le correzioni vanno
  applicate due volte a mano.
- **Regola.** Una sola implementazione per ogni grandezza, condivisa da pipeline e dashboard.
  Resta una copia del forecast nel dashboard (scarto al più 3e-6 µg/L, misure dell'1–2/10/2026), issue #39.

### 10. L'instabilità dei ranghi nel prewhitening (#7, #33, issue #9)
- **Cosa.** La selezione dell'ordine ARIMA del prewhitening cambia fra due run della CI sullo
  stesso codice; anche a ordine fissato la parte MA trasforma le sequenze di mesi a zero in
  residui quasi pari (fino a 1e-17), che Spearman ordina secondo l'aritmetica della macchina.
- **Perché.** Il metodo è numericamente mal posto su un driver a eventi intermittenti.
- **Regola.** Un numero che dipende dalla macchina non si riporta come valore esatto: si riporta
  l'esito stabile alla perturbazione e fra le scelte equivalenti, con la misura (ADR-0000 voce 10,
  8/10/2026: la griglia da 39 test dà 0 significativi in tutti i casi quasi equivalenti per AIC). Il golden
  master confronta quei file solo nella struttura; il calcolo si verifica con parametri fissi su
  un driver sintetico continuo.

### 11. Il file per prova mai committato (corretto il 28/9, #26)
- **Cosa.** L'aggiornamento automatico rigenerava `data/ec50_raw.csv` ma non lo committava: i
  `results/` pubblicati non si potevano riprodurre dai `data/` pubblicati.
- **Regola.** Si committa ciò da cui si calcola. Un controllo verifica che la serie mensile sia
  l'aggregazione della serie per prova.

### 12. I commit dell'aggiornamento mai testati (corretto il 25/9, #22)
- **Cosa.** I commit dell'aggiornamento portano `[skip ci]`: proprio i commit con dati nuovi non
  venivano mai testati, e `main` è rimasto rosso senza che nessuno lo vedesse.
- **Regola.** I test girano dopo ogni aggiornamento (`workflow_run`) e un fallimento apre una
  issue che menziona il proprietario.

### 13. Un push diretto su `main` (21/9, `62a1d94`)
- **Cosa.** Una correzione urgente della CI è entrata su `main` con un push diretto da una
  sessione dell'assistente, senza PR. Il registro di GitHub mostra un evento `push`, non una
  fusione.
- **Regola.** Ogni modifica a `main` passa da una PR, comprese le correzioni urgenti. Dall'8/10 la
  impone GitHub (ruleset 24719143: PR obbligatoria e controllo `test` verde, con la deploy key
  del job di aggiornamento come unica eccezione); fino ad allora era una convenzione.
  [Prima del 22/9 i commit entravano anche senza PR.]

### 14. Copie di lavoro sparite (fino al 2/10)
- **Cosa.** Le copie di lavoro del branch del lavoro stavano nella cartella temporanea della
  sessione, svuotata al cambio di data; la seconda volta tre commit erano in *detached HEAD* e
  sono stati recuperati per hash.
- **Regola.** Copie di lavoro in un percorso persistente, ogni commit su un branch con nome.

### 15. Un'operazione in blocco che non aveva fatto niente (2/10)
- **Cosa.** In zsh una variabile non quotata non si spezza in parole: un elenco di 28 branch
  passato a `git push --delete` è diventato un solo argomento, e non è stato cancellato niente,
  senza un errore evidente.
- **Regola.** Dopo un'operazione in blocco si rilegge lo stato risultante prima di dichiararla
  riuscita.

### 16. Le ondate di calore di un altro sito (luglio 2026, `09b928a`)
- **Cosa.** Per mesi le statistiche MHW sono state calcolate sulla SST di coordinate sbagliate
  (zona di La Spezia), mentre il resto dei dati era del sito giusto.
- **Perché.** Il rilevamento MHW stava fuori dalla pipeline automatica e non è stato rieseguito
  dopo la correzione delle coordinate.
- **Regola.** Tutto ciò che dipende dalle coordinate o dai dati passa dalla stessa pipeline, nello
  stesso ordine; il rilevamento MHW è il primo passo di `ccsu-run-pipeline`.
- **Versioni.** La correzione (`09b928a`, 17/7) è contenuta in v1.4.0 e in v1.5.0: la versione
  citata nel lavoro non ne è toccata.
