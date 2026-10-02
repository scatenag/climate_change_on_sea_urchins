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
Definizione fissata dal proprietario; decisioni D1–D9 approvate il 2/10; piano riordinato come
fetta verticale (riga di comando prima, interfaccia dopo), approvato con la fusione di
`MILESTONE-M1.md`. Nessuna scadenza. Nessun codice della milestone è ancora scritto.

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
  push, nessuna eccezione. `paper/mpb-2026`: protezione classica, idem. **Protezione completa di
  `main` approvata, in corso**: deploy key con scrittura e segreto `UPDATE_EC50_DEPLOY_KEY`
  creati il 2/10; la #46 fa spingere `update_ec50.yml` con la chiave; dopo la sua fusione si
  attiva il ruleset (PR obbligatoria senza approvazioni, controllo `test` verde, unica eccezione
  le deploy key; quelle di Streamlit sono in sola lettura). Il primo push reale con la chiave è
  l'aggiornamento Copernicus del 5/10: da controllare. I branch delle PR fuse si cancellano da
  soli.
- **Branch**: `main`, `paper/mpb-2026`, `sartori-2023-supplement` (riferimento del lavoro del
  2023, tag `v0.1.0-sartori-2023`).

## Lavoro parcheggiato

| Elemento | Stato in cui resta | Per riprenderlo |
|---|---|---|
| Finestre temporali nei moduli non migrati (5b-2, 5b-3, 5b-4) | Infrastruttura e quattro moduli fatti (#25). Gli altri non dichiarano `SUPPORTS_WINDOW` e per una finestra non girano: `window.json` li registra come non eseguiti, mai con la finestra ignorata | 5b-2 `timeseries`, `correlations` (partire da `compute_matrices`, #35), `cu_speciation`, `negative_control`; 5b-3 `mhw_analysis`, `mhw_robustness`, `mhw_lag_extra`; 5b-4 `mhw_lag_annual` (regola degli anni completi: per Livorno 2004–2025) e `regime_shift`. Proporre di marcare l'anno parziale in `mhw_detection`. Regola (a)/(b)/(c) in `tests/test_windows.py` |
| Prewhitening ARIMA | Confrontato solo nella struttura; braccio ARIMA di `mhw_severe_intensity` non applicabile; diagnosi nella #9 | V3.1: procedura deterministica, con un ADR |
| Valori di §3.6 e tag `v1.5.1` | In attesa dei valori di §3.6 del manoscritto finale | Spostare in `test_paper_values.py` il confronto pre/post alla data pubblicata di `negative_control`; tag `v1.5.1`; se §3.6 usa il dato corretto della prova 224, rilascio con dati corretti e DOI nuovo nelle bozze |
| `paper/mpb-2026` | Riceve solo dati nuovi e correzioni di affermazioni false | Su richiesta esplicita: prima su `main`, poi sul branch in un commit a sé, `force_rebuild`, verifica dell'app (ADR-0009). La data di fine della SST (#45) verrà richiesta dopo la misura dell'effetto sull'app citata. Altri candidati: copia del forecast (#39), riempimento a 0 nel DLNM (#38), `update_ec50.yml` di v1.5.0 rimasto sul branch |
| Logo dell'app di sviluppo (#32) | In attesa del parere del proprietario | — |

## In attesa del proprietario

- Rilettura e fusione della #37 (questa memoria); poi: `CLAUDE.md` locale sostituito dalla
  versione breve, `PASSAGGIO.md` cancellato, avvio della fetta verticale.
- Fusione della #44 (data di fine della SST su `main`), prima del 5/10 perché l'aggiornamento
  mensile porti luglio e agosto 2026.
- Richiesta esplicita della stessa correzione su `paper/mpb-2026`, dopo la misura dell'effetto
  sull'app citata (mesi di SST e di ondate in più, risultati mostrati che cambiano).
- Parte sulla seconda scheda del foglio in `note-dati-sorgente.md`: fuori finché non è chiarito
  se si può descrivere.
- Chiudere la #23 (esercitazione del 25/9).

## Issue aperte

#9 prewhitening (V3.1) · #23 esercitazione, da chiudere · #38 riempimento a 0 nel DLNM · #39 copia
del forecast nel dashboard · #40 unità del CO₂ in `fetch_copernicus.py` · #41 interpolazione dei
buchi della SST · #42 rimandi a documenti non versionati · #43 percorsi risolti all'import negli
script · #45 data di fine della SST (corretta su `main` dalla #44, aperta per il branch del lavoro).

## Prossimo passo

**M1.1 Specifica formato 2 e catalogo con la sola SST**, primo pezzo della fetta verticale.
