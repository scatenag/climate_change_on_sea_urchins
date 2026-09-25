# 0008 — `results/` è per studio, non condiviso

## Contesto

`common.results_dir(study_id)` (introdotta in #16) era già l'unico punto che risolve dove
vivono i risultati, ma restituiva sempre `ROOT / "results"` — `study_id` non era ancora usato.
Con un secondo studio eseguibile (V2.2), due studi che condividono `results/` si
sovrascriverebbero a vicenda a ogni run del pipeline: gli stessi nomi di file
(`ccf_results.csv`, `forecast_mean.csv`, ...) non portano l'identità dello studio che li ha
prodotti.

## Alternative valutate

- **Nome file con prefisso `study_id`** (es. `livorno-paracentrotus_ccf_results.csv`):
  scartata — richiede toccare ogni punto in cui un modulo costruisce un nome di file in
  output, non solo il confine in `common.py`; il nome derivato da `response.label` (V2.1,
  invariante CLAUDE.md #4) andrebbe combinato con lo `study_id`, aumentando la superficie di
  errore per un guadagno minore rispetto alla directory.
- **Directory per studio, `results/<study_id>/`**: scelta adottata — un solo cambiamento
  (`results_dir()`), nessun modulo a valle deve sapere che esiste più di uno studio.

## Decisione

`results_dir(study_id)` restituisce `ROOT / "results" / study_id`. I 64 file esistenti di
Livorno spostati con `git mv` in `results/livorno-paracentrotus/` (nessun contenuto
modificato). Il golden master (`tests/conftest.py::golden_pipeline_results`) non è toccato da
questo spostamento: i suoi fixture assegnano `RESULTS` a una directory temporanea per ogni
modulo via `monkeypatch`, indipendentemente da cosa calcolerebbe `results_dir()` nella
run reale — è la stessa proprietà di indipendenza dal percorso già sfruttata per la PR di
`results_dir()` (#16).

## Conseguenze

- `README.md`: i quattro link a file specifici sotto `results/` (sezione riproduzione dei
  numeri del lavoro) aggiornati al nuovo percorso; il link alla directory `results/` stessa
  resta invariato, descrive ancora correttamente il contenitore di primo livello.
- `tests/test_thermal_legacy_vif.py` costruiva il percorso da sé (`ROOT / "results"`) invece
  di passare da `common.RESULTS` — con lo spostamento sarebbe silenziosamente passato a
  `pytest.skip()` invece di fallire o testare qualcosa: corretto per usare `common.RESULTS`,
  la stessa classe di problema (non lo stesso bug) di quello trovato in `mhw_detection.py`
  nella PR precedente.
- `scripts/make_regime_shift_figure.py`, `make_mhw_lag_annual_figure.py`,
  `make_speciation_figure.py`, `make_thermal_legacy_figure.py`, `build_narrative_notebook.py`
  costruiscono ancora `ROOT / "results"` da soli: con questo spostamento smettono di trovare i
  file di Livorno. Lasciati intatti, non corretti in questa PR — sono fuori dal pacchetto
  installabile (CLAUDE.md), nessun test li esercita, e lo stesso trattamento (documentare,
  non correggere) è stato applicato al pattern gemello in `scripts/*.py` per `data/` nella PR
  precedente. Annotato in `docs/roadmap/STATO.md`.
- Il workflow (`update_ec50.yml`) e lo script R (`mhw_lag_analysis.R`) non richiedono modifiche:
  entrambi già risolvono `RESULTS_DIR` da `common.RESULTS` a runtime, e `git add results/`
  nello step di commit intercetta l'intero albero indipendentemente dalla sua forma interna.
