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

### Changed

- **ARIMA order selection discards candidates that don't converge, and the golden master
  stops pinning numeric values it can't reproduce (issue #4, branch `fix/arima-convergence`)**.
  - `_best_arima_order()` (`mhw_analysis.py`) used to pick the lowest-AIC candidate
    regardless of whether the optimizer actually converged. `warnings.simplefilter("ignore")`
    was silencing statsmodels' own `ConvergenceWarning` along with everything else. It now
    captures warnings instead of blanket-ignoring them, and discards any candidate that
    raises `ConvergenceWarning` or whose own `mle_retvals` says it didn't converge, before
    comparing AIC (`_fit_arima_if_converged()`, shared with a new `order` parameter on
    `compute_ccf_prewhitened()` that skips the search entirely when a fixed order is passed).
  - This fixes a real bug — a discarded fit's residuals are unreliable in a way AIC alone
    doesn't reveal — but does **not** make order selection itself reproducible across
    machines. Confirmed directly: two consecutive CI runs of the identical branch (no code
    change between them) picked different orders for `mhw_days` on GitHub's own hosted
    runners. For a near-degenerate driver (`mhw_days`: 45% zero months; `mhw_severe_intensity`:
    94%), *which* candidates converge at all depends on the runner's hardware/BLAS path, which
    can change which order wins by AIC even after discarding non-convergent ones. No
    Bonferroni-survival check or manuscript comparison was performed on the resulting numbers
    for this reason: they are not a property of the code and data alone, so they are not
    reported here or anywhere else. A deterministic order-selection procedure is left to
    V3.1 (see issue #4).
  - Consequently, `tests/test_golden_master.py` no longer compares the fitted values in
    `results/ccf_results_prewhitened.csv`, `results/prewhitening_diagnostics.json`, or
    `results/robustness_severe_ccf_note.json`'s `diagnostics` block against the frozen
    reference — the previous `ARIMA_FIT` tolerance tier (rtol=0.5, atol=0.02, introduced
    2026-09-21) is retired, and no numeric tolerance replaces it for these three: a new
    `STRUCTURAL_ONLY_FILES` set checks columns, row count, which driver/lag combinations
    exist, and where NaN falls, never the fitted numbers themselves (see the module
    docstring). `tests/test_mhw_analysis.py` (new) separately covers the fit/filter/
    correlate computation on its own, with a forced fixed order, under `LOOSE` tolerance.
  - `mhw_robustness.py::run_severe_ccf`'s ARIMA-prewhitened arm for `mhw_severe_intensity` is
    marked `not_applicable`: `results/robustness_severe_ccf.csv`'s `r_arima`/`p_arima` columns
    are always `NaN` (unconditionally, so still `TIGHT`-tolerance-safe), with the reason (and
    diagnostics) written to the new `results/robustness_severe_ccf_note.json`. Reason: 13 of
    the driver's 17 nonzero months fall after the 2016-06 EC50 regime shift (`SPLIT_DATE`);
    the filter is estimated on the driver alone and doesn't remove that shift from the target,
    so the residual correlation is confounded by it rather than reflecting a lagged response —
    confirmed by re-running with the shift removed from EC50 (pre/post demeaned): significant
    lags collapse from 10/13 to 1/13 at the order convergent on this machine. The
    first-differenced arm (`r_diff`/`p_diff`) is unaffected and is the retained result for
    this driver.

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
