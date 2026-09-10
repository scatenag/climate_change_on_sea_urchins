# Changelog

Tutte le modifiche rilevanti a questo progetto sono documentate in questo file.

Il formato segue [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- Ordinamento deterministico della sequenza grezza dei valori EC50 per (Datetime, ID) prima
  del changepoint QLR/AR(1) — commit `cb333cd`. Non ancora incluso in un tag di release.

## [1.4.0] - 2026-09-04

### Added

- `SPLIT_DATE` esplicito al 2016-06-01, propagato a tutti i moduli che prima calcolavano o
  assumevano la data di split in modo implicito.
- Nuovo modulo `changepoint.py` con procedura di stima del punto di rottura QLR
  (Quandt-Andrews) su residui AR(1).
- VIF (Variance Inflation Factor) per finestra nell'analisi di thermal legacy.

### Changed

- Aggiornato il DOI Zenodo al record v1.4.0 (10.5281/zenodo.22304864).
