# Adapting this framework to a new site, species, or indicator

This pipeline was built around one case study — *Paracentrotus lividus* gamete EC50 in the
North Tyrrhenian Sea — but the architecture separates "what site/data" from "how to process it", so
most of the case-study-specific detail lives in a small number of well-defined places. This
guide is organised by how much work each kind of change actually requires.

## 1. Just edit `config.py` (minutes)

- **A different oceanographic monitoring site**: change `SITE_LAT`, `SITE_LON`, `SITE_NAME`,
  `BBOX_DELTA` in [`config.py`](../config.py). All three Copernicus download scripts
  (`scripts/fetch_copernicus*.py`) and the dashboard read these values; nothing else needs to
  change for the physical-oceanography side of the pipeline.
- **A different bioassay spreadsheet with the same column schema** (`ID, DATE, EC50, UL, LL,
  pos, neg`): change `EC50_SHEET_ID` in `config.py`. Both `scripts/fetch_ec50.py` and the live
  fetch in the dashboard read from this single constant.

## 2. Touch a few files, no new architecture (a few hours)

- **A different set of environmental variables** (add/remove a Copernicus variable beyond SST,
  salinity, O2, pH, CO2): update `ENV_COLS` in
  [`src/climate_change_on_sea_urchins/common.py`](../src/climate_change_on_sea_urchins/common.py)
  and the corresponding download/parsing logic in `scripts/fetch_copernicus*.py`.
- **A different MHW detection sensitivity**: `scripts/detect_mhw.py` exposes `CLIM_START`,
  `CLIM_END`, and `PCTILE` (climatology baseline window and percentile threshold) as
  module-level constants.
- **A different bioassay schema** (different column names or a different toxicant/endpoint
  than copper EC50): the parsing logic is isolated in `scripts/fetch_ec50.py::fetch_raw()`, and
  the confidence-interval computation in `scripts/build_dataset.py`.

## 3. Genuine adaptation work (the honest part)

Swapping *P. lividus* EC50 for an entirely different biological indicator (a different species,
a different toxicological endpoint, a different unit) is **not** a configuration change. The
string `"EC50"` is used as a column name and UI label across roughly a dozen files —
`dashboard.py` (tab labels, chart titles, tooltips), every module in
`src/climate_change_on_sea_urchins/`, and `scripts/mhw_lag_analysis.R`. There is no shortcut
around this: a research group adapting the framework to a different indicator should expect to:

1. Rename the indicator column consistently from `data_extended.csv`/`data_ec50_ci.csv`
   onward, updating `ALL_COLS` in `src/climate_change_on_sea_urchins/common.py`.
2. Update the statistical assumptions where they are indicator-specific (e.g. the rolling-mean
   imputation window in `common.py::load_data()`, the ARDL/CCF lag range `TAU_MAX` in the same
   file, chosen for *P. lividus*'s ~2-month gametogenesis cycle).
3. Update `dashboard.py` labels/titles and the R script's variable names.

This is a deliberate trade-off: the pipeline's *stages* (ingest → align → analyse → present →
archive, see Figure 1 of the JOSS paper) are generic, but the statistical choices inside each
stage are tuned to this species and this stressor. We see this document, rather than a
fully-parameterised abstraction layer, as the more honest and more maintainable way to support
reuse — see [CONTRIBUTING.md](../CONTRIBUTING.md) for how to propose further generalisation.
