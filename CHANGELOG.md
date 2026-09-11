# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `negative_control.py`: new module reproducing manuscript section 3.6 (assay negative-control
  series) — trend, pre/post level and dispersion, and a QLR/AR(1) changepoint search (reuses
  `changepoint.qlr_ar1_changepoint`, not reimplemented). `data/ec50_raw.csv` gained three new
  columns (`ctrl_neg_rep1/2/3`, via `scripts/fetch_ec50.py`) to carry the source data. Also adds
  a data-quality check flagging single-replicate outliers (>3 pooled-SD from the other two
  replicates of the same trial) — found and flags a real one (trial 224, 2020-01-01) that turned
  out to explain part of a Spearman-p discrepancy against the manuscript; that discrepancy is
  recorded as OPEN, pending a decision on the source data, not silently resolved.
- `period_split.py`: adds `results/period_contrast_raw.json`, the manuscript's section 3.1
  pre/post contrast computed on the 295 individual EC50 trials (`data/ec50_raw.csv`), alongside
  the existing monthly-series contrast.
- `thermal_legacy.py`: adds `run_threshold_sensitivity()` / `results/thermal_threshold_sensitivity.csv`
  (manuscript Table S2) — the same detrended/partial tests swept over 22–26°C at the fixed
  24-month window, with 24°C (the a-priori primary threshold) marked as such, not presented on
  equal footing with the robustness-check values. `run()`'s existing computation/output is
  unchanged.
- `mhw_annual_changepoint.py`: new module applying `qlr_ar1_changepoint` to the annual MHW
  exposure metric (manuscript section 3.5, 2nd paragraph). **Unresolved**: no (metric,
  year-range) variant tried reproduces the manuscript's reference values (phi, break year,
  bootstrap p, CI90) together; `results/mhw_annual_changepoint.json` records every variant tried
  side by side with the reference, explicitly as an open discrepancy — not resolved or guessed
  at here, pending review.
- `tests/test_paper_values.py`: new test file, one invariant per section above, each with a
  declared tolerance (or, for the unresolved changepoint case, asserting the discrepancy stays
  documented rather than silently passing).

### Fixed

- Deterministic ordering of the raw EC50 sequence by (Datetime, ID) before the QLR/AR(1)
  changepoint — commit `cb333cd`. Not yet included in a release tag.

## [1.4.0] - 2026-09-04

### Added

- Explicit `SPLIT_DATE` set to 2016-06-01, propagated to all modules.
- New `changepoint.py` module with a QLR (Quandt-Andrews) breakpoint estimation procedure on
  AR(1) residuals.
- VIF (Variance Inflation Factor) per window in the thermal-legacy analysis.

### Changed

- Updated the Zenodo DOI to the v1.4.0 record (10.5281/zenodo.22304864).
