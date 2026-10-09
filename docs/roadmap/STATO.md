# Stato del progetto

> Si aggiorna alla fine di **ogni** sessione, prima della fusione. Una pagina: il dettaglio sta in
> [`PERCORSO.md`](PERCORSO.md) (fasi, decisioni, errori), [`MILESTONE-M1.md`](MILESTONE-M1.md)
> (milestone in corso), [`../COME_SI_LAVORA.md`](../COME_SI_LAVORA.md) (regole), negli ADR e
> nelle issue.

**Ultimo aggiornamento:** 8 ottobre 2026

## Dove siamo

**V2 · Il caso diventa configurazione**, con la **milestone M1 · Lo strumento generico da capo a
fondo**: un'app generica, un percorso per i propri dati, una demo registrata. Il lavoro sulle
finestre di V2.2 è parcheggiato (sotto).
Definizione fissata dal proprietario; decisioni D1–D9 approvate il 2/10; piano riordinato come
fetta verticale (riga di comando prima, interfaccia dopo), approvato con la fusione di
`MILESTONE-M1.md`. Nessuna scadenza. M1.1 (formato 2 e catalogo con la sola SST) approvata dal proprietario; M1.2 (download della SST) in PR.

## Stato attuale

- **`main`**: CI verde; golden master su 66 file di `results/livorno-paracentrotus/`.
- **Due app pubbliche.** Sviluppo, da `main`: <https://climate-response-explorer.streamlit.app>.
  Lavoro, da `paper/mpb-2026`: <https://climate-change-on-sea-urchins.streamlit.app>.
- **`paper/mpb-2026`**: testa `88af03a` (8/10): `d090d1d` porta la correzione della data di fine della
  SST della #44 con il suo test (richiesta esplicita del proprietario, ADR-0009; suite del branch 132
  test verdi), e il job con `force_rebuild` e Copernicus attivo (run 37773760510, verde) ha scaricato
  la SST di luglio e agosto 2026 (62 giorni) e ricalcolato `results/` e figure. Sopra: v1.5.0 più µg/L,
  testi dai valori, titoli delle figure, `renv.lock`, versioni bloccate, correzioni #33/#34/#35.
  **Verifiche dell'8/10.** I file di `results/` committati dal job coincidono con una misura
  indipendente in locale (stessi dati) entro il rumore già noto: parti ARIMA (Ljung-Box fino a 6e-6) e
  forecast (3.4e-7 µg/L); i dati `sst_daily` e `mhw_*` sono identici byte per byte. L'app pubblicata
  ricalcola dal vivo correlazioni (scarto 0.0), CCF sulle differenze prime (0.0) e forecast (al più
  8.8e-7 µg/L, copia del forecast nel dashboard, #39) identici ai file del job.
  **Effetto della SST nuova sui risultati mostrati.** Entra un evento MHW (7/7–31/8, severo, 56
  giorni); il 2026 annuale passa da 1 evento/34 giorni a 2/93 (resta contato come anno intero).
  Nessuna correlazione e nessun p<0.05 della CCF grezza cambia significatività; nella CCF sulle
  differenze prime O2 e salinità al lag 0 passano da p 0.050 a 0.040 (3→5 test sotto 0.05 su 78).
  Dose termica: la finestra a 36 mesi supera Bonferroni nel test sui ranghi (0.059→0.037, non nel
  parziale), quella a 24 mesi si rafforza (0.0081→0.0032). Forecast finale dello scenario peggiore
  12.7→11.8 µg/L; pre/post delle MHW +167%→+180% per l'intensità. DLNM (R): i profili cumulativi
  cambiano di meno di 0.25 e i ritardi con IC95% che esclude 0 restano 4 (il riempimento a 0 delle
  metriche MHW è ancora nello script, #38). Il confronto A/B è in `drafts/`, fuori da git.
- **Job.** `update_ec50.yml` (EC50 ogni giorno, Copernicus il 5 del mese) spinge su `main`;
  `update_paper_branch.yml` spinge su `paper/mpb-2026` e ha l'input `force_rebuild`; `tests.yml`
  gira anche dopo ogni aggiornamento e apre una issue se fallisce; `validate_data.yml` ogni giorno.
- **Protezione.** `main`: due ruleset attivi. Il primo (24362986, anche per
  `sartori-2023-supplement`): niente cancellazione né force push. Il secondo (24719143, attivato l'8/10):
  PR obbligatoria con 0 approvazioni e controllo `test` obbligatorio (non «aggiornato a main»); unica
  eccezione le deploy key (`DeployKey`, sempre), nessuna per persone o amministratori. Il segreto
  `UPDATE_EC50_DEPLOY_KEY` sta solo nell'environment `auto-update-main` (limitato a `main`,
  dichiarato solo da `update_ec50.yml`); il token di quel job è in sola lettura. Il ruleset è stato
  attivato dopo il primo push reale con la chiave: l'aggiornamento Copernicus del 5/10 ha spinto
  `43f35dd..170ce7a` via SSH. **Da verificare** al primo push del job con il ruleset attivo (si
  ha solo quando arrivano dati nuovi). `paper/mpb-2026`: protezione classica, niente cancellazione
  né force push. I branch delle PR fuse si cancellano da soli.
- **Branch**: `main`, `paper/mpb-2026`, `sartori-2023-supplement` (riferimento del lavoro del
  2023, tag `v0.1.0-sartori-2023`).

## Lavoro parcheggiato

| Elemento | Stato in cui resta | Per riprenderlo |
|---|---|---|
| Finestre temporali nei moduli non migrati (5b-2, 5b-3, 5b-4) | Infrastruttura e quattro moduli fatti (#25). Gli altri non dichiarano `SUPPORTS_WINDOW` e per una finestra non girano: `window.json` li registra come non eseguiti, mai con la finestra ignorata | 5b-2 `timeseries`, `correlations` (partire da `compute_matrices`, #35), `cu_speciation`, `negative_control`; 5b-3 `mhw_analysis`, `mhw_robustness`, `mhw_lag_extra`; 5b-4 `mhw_lag_annual` (regola degli anni completi: per Livorno 2004–2025) e `regime_shift`. Proporre di marcare l'anno parziale in `mhw_detection`. Regola (a)/(b)/(c) in `tests/test_windows.py` |
| Prewhitening ARIMA | Confrontato solo nella struttura; braccio ARIMA di `mhw_severe_intensity` non applicabile; diagnosi nella #9 | V3.1: procedura deterministica, con un ADR |
| Valori di §3.6 e tag `v1.5.1` | In attesa dei valori di §3.6 del manoscritto finale | Spostare in `test_paper_values.py` il confronto pre/post alla data pubblicata di `negative_control`; tag `v1.5.1`; se §3.6 usa il dato corretto della prova 224, rilascio con dati corretti e DOI nuovo nelle bozze |
| `paper/mpb-2026` | Riceve solo dati nuovi e correzioni di affermazioni false | Su richiesta esplicita: prima su `main`, poi sul branch in un commit a sé, `force_rebuild`, verifica dell'app (ADR-0009). La data di fine della SST (#45) è già sul branch (8/10). Altri candidati: copia del forecast (#39), riempimento a 0 nel DLNM (#38), `update_ec50.yml` di v1.5.0 rimasto sul branch |
| Logo dell'app di sviluppo (#32) | In attesa del parere del proprietario | — |

## Da fare

- Primo push del job `update_ec50.yml` con il ruleset attivo: verificare che sia passato (succede
  solo quando arrivano dati nuovi).
- Dopo la prossima rielaborazione dei dati, riverificare che l'app del lavoro coincida con i file del
  job con lo stesso controllo (CSV di correlazioni, CCF e forecast calcolati dal vivo).

## In attesa del proprietario

- Rilettura e fusione della #37 (questa memoria); poi: `CLAUDE.md` locale sostituito dalla
  versione breve, `PASSAGGIO.md` cancellato, avvio della fetta verticale.
- Parte sulla seconda scheda del foglio in `note-dati-sorgente.md`: fuori finché non è chiarito
  se si può descrivere.

## Issue aperte

#9 prewhitening (V3.1) · #38 riempimento a 0 nel DLNM · #39 copia del forecast nel dashboard · #40
unità del CO₂ in `fetch_copernicus.py` · #41 interpolazione dei buchi della SST · #42 rimandi a
documenti non versionati · #43 percorsi risolti all'import negli script (tutte, tranne la #9, difetti
noti che non bloccano M1). Chiuse l'8/10: #23 (esercitazione) e #45 (data di fine della SST, corretta
su `main` con la #44 e sul branch del lavoro). Difetto noto non ancora in una issue: l'id del dataset
di ripiego in `fetch_copernicus_daily.py` (`…phy-temp_anfc…`) non esiste nel catalogo (quello giusto è
`…phy-tem_anfc_4.2km_P1D-m`); il ripiego non è mai stato usato.

## Prossimo passo

**M1.3 Sorgente CSV della risposta** (M1.2, download della SST, è in PR; manca la verifica a mano con le credenziali: Livorno, serie uguale a `data/sst_daily.csv` nei giorni comuni, tempo misurato). La regola
per i mesi incompleti della SST (M1.4) è approvata: un giorno mancante rende mancante il mese.
