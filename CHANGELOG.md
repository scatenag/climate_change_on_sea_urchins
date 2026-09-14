# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
  to `legacy/`. `paper.md`/`paper.bib`/`figures/` (the abandoned JOSS submission's source) removed
  from the repository entirely — kept locally under `drafts/` (already git-ignored), not tracked
  going forward.

## [1.4.0] - 2026-09-04

### Added

- Explicit `SPLIT_DATE` set to 2016-06-01, propagated to all modules.
- New `changepoint.py` module with a QLR (Quandt-Andrews) breakpoint estimation procedure on
  AR(1) residuals.
- VIF (Variance Inflation Factor) per window in the thermal-legacy analysis.

### Changed

- Updated the Zenodo DOI to the v1.4.0 record (10.5281/zenodo.22304864).
