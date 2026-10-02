# Stato del progetto

> Si aggiorna alla fine di **ogni** sessione, prima della fusione. Una pagina: il dettaglio sta in
> [`PERCORSO.md`](PERCORSO.md) (fasi, decisioni, errori), [`MILESTONE-M1.md`](MILESTONE-M1.md)
> (milestone in corso), [`../COME_SI_LAVORA.md`](../COME_SI_LAVORA.md) (regole), negli ADR e
> nelle issue.

**Ultimo aggiornamento:** 2 ottobre 2026

## Dove siamo

**V2 · Il caso diventa configurazione**, con la **milestone M1 · Lo strumento generico da capo a
fondo**: un'app generica, un percorso per i propri dati, una demo registrata. Il lavoro sulle
finestre di V2.2 è parcheggiato (sotto).
Definizione fissata dal proprietario; decisioni D1–D9 e piano dei pezzi **proposti, in attesa di
approvazione**. Nessun codice della milestone è ancora scritto.

## Stato attuale

- **`main`**: CI verde; golden master su 66 file di `results/livorno-paracentrotus/`.
- **Due app pubbliche.** Sviluppo, da `main`: <https://climate-response-explorer.streamlit.app>.
  Lavoro, da `paper/mpb-2026`: <https://climate-change-on-sea-urchins.streamlit.app>.
- **`paper/mpb-2026`**: testa `3f738c0`, ricalcolo forzato del 2/10 con il codice `3d41358`
  (v1.5.0 più µg/L, testi dai valori, titoli delle figure, `renv.lock`, versioni bloccate, e le
  correzioni #33/#34/#35). Verificato il 2/10: job verde (run 36979357234); l'app pubblicata
  ricalcola dal vivo correlazioni e CCF sulle differenze prime identiche ai file del job (scarto
  0.0) e il forecast entro 5.5e-7 µg/L; lettura dal vivo del foglio riuscita.
- **Job.** `update_ec50.yml` (EC50 ogni giorno, Copernicus il 5 del mese) spinge su `main`;
  `update_paper_branch.yml` spinge su `paper/mpb-2026` e ha l'input `force_rebuild`; `tests.yml`
  gira anche dopo ogni aggiornamento e apre una issue se fallisce; `validate_data.yml` ogni giorno.
- **Protezione.** Ruleset su `main` e `sartori-2023-supplement`: niente cancellazione né force
  push, nessuna eccezione. `paper/mpb-2026`: protezione classica, idem. **Su `main` PR e
  controlli non sono obbligatori**: la regola «sempre una PR» è solo una convenzione. I branch
  delle PR fuse si cancellano da soli.
- **Branch**: `main`, `paper/mpb-2026`, `sartori-2023-supplement` (riferimento del lavoro del
  2023, tag `v0.1.0-sartori-2023`).

## Lavoro parcheggiato

| Elemento | Stato in cui resta | Per riprenderlo |
|---|---|---|
| Finestre temporali nei moduli non migrati (5b-2, 5b-3, 5b-4) | Infrastruttura e quattro moduli fatti (#25). Gli altri non dichiarano `SUPPORTS_WINDOW` e per una finestra non girano: `window.json` li registra come non eseguiti, mai con la finestra ignorata | 5b-2 `timeseries`, `correlations` (partire da `compute_matrices`, #35), `cu_speciation`, `negative_control`; 5b-3 `mhw_analysis`, `mhw_robustness`, `mhw_lag_extra`; 5b-4 `mhw_lag_annual` (regola degli anni completi: per Livorno 2004–2025) e `regime_shift`. Proporre di marcare l'anno parziale in `mhw_detection`. Regola (a)/(b)/(c) in `tests/test_windows.py` |
| Prewhitening ARIMA | Confrontato solo nella struttura; braccio ARIMA di `mhw_severe_intensity` non applicabile; diagnosi nella #9 | V3.1: procedura deterministica, con un ADR |
| Valori di §3.6 e tag `v1.5.1` | In attesa dei valori di §3.6 del manoscritto finale | Spostare in `test_paper_values.py` il confronto pre/post alla data pubblicata di `negative_control`; tag `v1.5.1`; se §3.6 usa il dato corretto della prova 224, rilascio con dati corretti e DOI nuovo nelle bozze |
| `paper/mpb-2026` | Riceve solo dati nuovi e correzioni di affermazioni false | Su richiesta esplicita: prima su `main`, poi sul branch in un commit a sé, `force_rebuild`, verifica dell'app (ADR-0009). Candidati: copia del forecast, riempimento a 0 nel DLNM, data di fine della SST, `update_ec50.yml` di v1.5.0 rimasto sul branch |
| Logo dell'app di sviluppo (#32) | In attesa del parere del proprietario | — |

## In attesa del proprietario

- Approvazione di D1–D9 e del piano di M1 (`MILESTONE-M1.md` §5–§6).
- **Protezione completa di `main`** (PR obbligatoria, controllo `test` verde, nessuna eccezione
  per le persone). `update_ec50.yml` oggi spinge su `main` con `GITHUB_TOKEN`: con la protezione
  attiva gli serve un'eccezione. Proposta: una deploy key con permesso di scrittura, unica voce
  fra le eccezioni del ruleset, usata solo dal checkout di quel job; alternative: l'app GitHub
  Actions fra le eccezioni, o una PR automatica (che col `GITHUB_TOKEN` non fa partire i
  controlli). Ordine: chiave e segreto, PR che cambia il checkout del job, poi il ruleset.
- Versionare, aggiornare o archiviare i file non versionati di `docs/roadmap/`.
- Chiudere la #23 (esercitazione del 25/9, mai chiusa).

## Issue aperte

[#9](https://github.com/scatenag/climate_change_on_sea_urchins/issues/9) (prewhitening, V3.1) ·
[#23](https://github.com/scatenag/climate_change_on_sea_urchins/issues/23) (esercitazione, da
chiudere). Difetti noti che non bloccano M1, da aprire come issue: data di fine della SST scritta a
mano, riempimento a 0 nel DLNM, copia del forecast nel dashboard, attribuzione errata dell'unità
del CO₂ in `scripts/fetch_copernicus.py`, interpolazione silenziosa dei buchi della SST
giornaliera, riferimenti nel codice a documenti di `docs/roadmap/` non versionati.

## Prossimo passo

Con il piano approvato: **M1.1 Catalogo delle variabili**.
