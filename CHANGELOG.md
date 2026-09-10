# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
