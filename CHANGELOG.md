# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `tests/test_golden_master.py`: regression test for the entire `results/` directory (64
  files), not just the ~15 manuscript-cited numbers `test_paper_values.py` covers. Reruns the
  full pipeline (all 16 Python modules, plus the R DLNM script when available) against the
  same frozen fixture `test_paper_values.py` uses, and compares every output file against a
  frozen reference (`tests/fixtures/results_v1_5_0/`) with per-file tolerances declared and
  justified by category (closed-form, seeded bootstrap/RNG, iterative MLE optimizer, and the
  known ARIMA-prewhitening scale-sensitivity case) rather than one uniform tolerance. Marked
  `@pytest.mark.golden` (reruns the whole pipeline, ~2-3 min) so it can be run standalone with
  `pytest -m golden`, separate from the fast default suite. Verified with a sabotage test:
  corrupting one reference value makes the test fail naming that exact file/field; restoring
  it returns to green.
- `tests/conftest.py::golden_pipeline_results`: generalizes the existing `paper_results`
  fixture from 4 modules to the full pipeline (`pipeline._MODULES`).
- **V2.1, "specifica e astrazione della risposta" (branch `v2.1/study-spec`)**: declarative
  `examples/livorno_paracentrotus/study.yaml`, validated by new pydantic models in
  `src/climate_change_on_sea_urchins/study_spec.py` (`SiteSpec`, `ResponseSpec` — split into
  a per-trial `ResponseSourceSpec` and a separate `ResponseAggregationSpec`, since the raw
  source and the monthly-aggregated series are two different representations, not one —
  `VariableSpec`, `WindowSpec`, `StudySpec`). `config.py` inverted to read this file instead
  of hardcoding site/source values as module constants, re-exporting the exact same names
  (`SITE_LAT`, `SITE_LON`, `SITE_NAME`, `BBOX_DELTA`, `EC50_SHEET_ID`, `EC50_EXPORT_URL`) so
  none of the six files that import them changed. `CO2_PA_TO_UATM` deliberately stays a
  hardcoded physical constant, not part of the spec. New `ccsu-validate-study` console script
  and `docs/schema/study.schema.json` (JSON Schema export). New direct dependencies:
  `pydantic>=2,<3`, `PyYAML>=6,<7` (both were already present transitively via Streamlit, now
  declared). Verified with the golden master: zero drift, no new tolerances needed.
- **V2.1, response-series abstraction (branch `v2.1/response-abstraction`, in progress)**:
  `ResponseSpec` gains a `label` field (display identity for outputs -- for Livorno, `"EC50"`).
  `common.py` gains `RESPONSE_COL`/`IMPUTED_COL` (indirection constants, value `"EC50"`/
  `"EC50_imputed"` for now -- not an alias: `load_data()` renames the CSVs' fixed on-disk
  columns into these at the load boundary, so every downstream module reads the response
  column through the constant, never the literal) and `load_ec50_raw()`/`load_ec50_monthly()`
  (extending the single data-reading boundary to `data/ec50_raw.csv`/`ec50_sheets.csv`, which
  `changepoint.py` and `period_split.py` read directly before this). `config.py` exports
  `RESPONSE_SPEC`; `pipeline.py` loads it once and passes it as `response=` to the 7 modules
  whose output carries the response's identity (`correlations`, `stationarity`,
  `regime_shift`, `period_split`, `cu_speciation`, `thermal_legacy`, `forecast`) -- every
  row/column label, `"variable"`/`"series"` field, per-variable dict key, output filename and
  derived column name in those modules' output now comes from `response.label`, never from
  the internal `RESPONSE_COL`. `docs/adr/0000-decisioni-rimandate.md` #6: `label` isn't
  sanitized for filename use yet, deferred to V2.2's second case. Golden master: 65/65, zero
  drift (Livorno's `label == RESPONSE_COL == "EC50"` today, so every output is byte-identical).
  Not yet done: the MHW-family modules (`mhw_analysis.py`, `mhw_robustness.py`,
  `mhw_lag_extra.py`, `mhw_lag_annual.py`) and the final step (flipping `RESPONSE_COL`/
  `IMPUTED_COL`'s values to `"response"`/`"response_imputed"`) -- deliberately deferred until
  after `fix/arima-convergence` merges, since both branches touch `mhw_analysis.py`.

### Fixed

- `.gitignore`'s `trend_*.csv` rule (unanchored, matches any directory depth) was silently
  dropping 6 files from `git add tests/fixtures/results_v1_5_0/` — the committed golden-master
  reference shipped incomplete; caught by the coverage-guard test in `test_golden_master.py` on
  the first real CI run. Fixed with a scoped negation, not by removing the original rule.
- Two `test_golden_master.py` tolerances were set from single-machine measurements (both showed
  exactly 0 relative difference locally) and turned out far too tight on a different machine:
  `ccf_results_prewhitened.csv` and `robustness_severe_ccf.csv`'s `r_arima`/`p_arima` columns
  (the latter mis-categorized as fully deterministic — missed that it shares the same ARIMA
  prewhitening call). New `ARIMA_FIT` tolerance tier (rtol=0.5, atol=0.02, explicitly a
  gross-error check only) plus per-column tolerance overrides for files that mix a stable and
  an ARIMA-derived column.

## [1.5.0] - 2026-09-14

The version cited in Sartori, Scatena, Gaion et al. (submitted, *Marine Pollution Bulletin*) as
an independent means of verifying its published numbers.

### Fixed

- CO2 units corrected: Copernicus's `spco2` is Pascal, not microatmospheres (commit `7d4b75d`,
  2026-09-10) — conversion applied at ingestion via `config.CO2_PA_TO_UATM`. `results/` had not
  been regenerated since that fix landed (it was regenerated in three separate, inconsistent
  runs predating the seed work below); this release's single official run (see "vendemmia" below)
  is the first to reflect it consistently. Verified via a controlled before/after comparison
  (current data vs. data restored to just before `7d4b75d`, both with current code/seeds):
  differs only where expected (CO2 means/SD/trend/distributions), identical elsewhere
  (correlations, both changepoint analyses, the regime-shift stress index, copper speciation,
  forecast) to floating-point noise.
- Deterministic ordering of the raw EC50 sequence by (Datetime, ID) before the QLR/AR(1)
  changepoint — commit `cb333cd`.

### Added

- `negative_control.py`: new module reproducing manuscript section 3.6 (assay negative-control
  series) — trend, pre/post level and dispersion, and a QLR/AR(1) changepoint search (reuses
  `changepoint.qlr_ar1_changepoint`, not reimplemented). `data/ec50_raw.csv` gained three new
  columns (`ctrl_neg_rep1/2/3`, via `scripts/fetch_ec50.py`) to carry the source data. Also adds
  a data-quality check flagging single-replicate outliers (>3 pooled-SD from the other two
  replicates of the same trial) — flags trial 224 (2020-01-01); confirmed by D. Sartori (2026-09-14)
  to be two distinct trials, not a data-entry error, so the value stands uncorrected.
- `period_split.py`: adds `results/period_contrast_raw.json`, the manuscript's section 3.1
  pre/post contrast computed on the 295 individual EC50 trials (`data/ec50_raw.csv`), alongside
  the existing monthly-series contrast.
- `thermal_legacy.py`: adds `run_threshold_sensitivity()` / `results/thermal_threshold_sensitivity.csv`
  (manuscript Table S2) — the same detrended/partial tests swept over 22–26°C at the fixed
  24-month window, with 24°C (the a-priori primary threshold) marked as such, not presented on
  equal footing with the robustness-check values. `run()`'s existing computation/output is
  unchanged.
- `mhw_annual_changepoint.py`: new module applying `qlr_ar1_changepoint` to the annual MHW
  exposure metric. Investigated for the manuscript's section 3.5 2nd paragraph, but no (metric,
  year-range) variant tried reproduced the reference values (phi, break year, bootstrap p, CI90)
  together; that paragraph was removed from the manuscript as a result (D. Sartori, 2026-09-14).
  `results/mhw_annual_changepoint.json` keeps every variant tried, flagged
  `cited_in_manuscript: false`, as a record of what was investigated.
- The single official pipeline run ("vendemmia") this release's `results/` reflects — regenerated
  once from current code, current seeds, and the data vintage cut on the dates below, superseding
  three earlier, inconsistent runs.
- `tests/fixtures/paper_mpb_2026/`: frozen snapshot of `data/*.csv` this vintage was computed
  from. `tests/test_paper_values.py` (via the new `tests/conftest.py`) now re-runs the four
  manuscript-value modules above against this fixture instead of reading precomputed `results/`,
  so it no longer depends on the live, auto-updating `data/` and stays green as the real series
  grows past this vintage.
- `requirements-lock.txt`: `pip freeze` of the exact environment this vintage was verified
  reproducible in (re-running the pipeline against it changes nothing in `results/`).

### Changed

- README badge and `CITATION.cff` now point to the Zenodo **concept DOI** (10.5281/zenodo.19352308)
  instead of a version-specific one (10.5281/zenodo.22304864, the v1.4.0 record) — the concept
  DOI always resolves to the latest release, so it will not need updating again.
- Repo root decluttered ahead of the release: `analysis.ipynb` and `narrative_sentinel_regime_shift.ipynb`
  moved to `notebooks/`; `marineHeatWaves.py` (vendored, unused reference implementation) moved
  to `legacy/`. `paper.md`/`paper.bib` (the abandoned JOSS submission's source) removed from the
  repository entirely — kept locally under `drafts/` (already git-ignored), not tracked going
  forward. Three paper-only illustrations (`figures/fig0_pipeline.png`, `fig1_timeseries.png`,
  `fig2_app_screenshot.png`) removed with them; the four analysis figures `dashboard.py` loads at
  runtime (`fig_cu_speciation_decomposition.png`, `fig_thermal_legacy.png`, `fig_regime_shift.png`,
  `fig_mhw_lag_annual.png`) stay tracked in `figures/` — an initial pass removed those too, caught
  and reverted before merge.
- Removed obsolete/superseded tracked files (recoverable from git history/old releases, not kept
  locally): `docs/MHW_ANALYSIS_RESEARCH.md` (unreferenced research notes predating the current
  detrended/robustness analyses, moved to `drafts/`); `results/forecast_bio_{bad,good,mean}.csv`
  and `results/mhw_lag_correlations.csv` (outputs of the frozen `legacy/analysis_2023_exploratory.ipynb`,
  not produced by any current module); `assets/sea_urchin.png` (superseded by
  `sea_urchin_transparent.png`, the only one `dashboard.py` uses).

## [1.4.0] - 2026-09-04

### Added

- Explicit `SPLIT_DATE` set to 2016-06-01, propagated to all modules.
- New `changepoint.py` module with a QLR (Quandt-Andrews) breakpoint estimation procedure on
  AR(1) residuals.
- VIF (Variance Inflation Factor) per window in the thermal-legacy analysis.

### Changed

- Updated the Zenodo DOI to the v1.4.0 record (10.5281/zenodo.22304864).
