# Climate Change on Sea Urchins 🦔

[![Tests](https://github.com/scatenag/climate_change_on_sea_urchins/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/scatenag/climate_change_on_sea_urchins/actions/workflows/tests.yml)
[![Data validated](https://github.com/scatenag/climate_change_on_sea_urchins/actions/workflows/validate_data.yml/badge.svg?branch=main)](https://github.com/scatenag/climate_change_on_sea_urchins/actions/workflows/validate_data.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/scatenag/climate_change_on_sea_urchins/main?labpath=analysis.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://climate-change-on-sea-urchins.streamlit.app)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20924982.svg)](https://doi.org/10.5281/zenodo.20924982)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Coming from Sartori et al. (2023)?** The original supplementary material is preserved in the
> [`sartori-2023-supplement`](https://github.com/scatenag/climate_change_on_sea_urchins/tree/sartori-2023-supplement)
> branch and archived at tag
> [`v0.1.0-sartori-2023`](https://github.com/scatenag/climate_change_on_sea_urchins/releases/tag/v0.1.0-sartori-2023).

Open-source framework for FAIR climate-ecotoxicology research: integrating Copernicus Marine
data with biological sentinel monitoring in the Mediterranean Sea.

Applied to a 23-year study on the impact of **Marine Heatwaves** on gamete sensitivity of
*Paracentrotus lividus* (North Tyrrhenian Sea, off Livorno, 43.43°N 10.40°E); originally developed as
supplementary material for:
> *"Increased sensitivity of sea urchin larvae to metal toxicity as a consequence of the past two decades of Climate Change and Ocean Acidification in the Mediterranean Sea."*
> Davide Sartori, Guido Scatena, Cristina Andra Vrinceanu, Andrea Gaion — *Marine Pollution Bulletin* 194 (2023), 115274. https://doi.org/10.1016/j.marpolbul.2023.115274

Beyond this specific case study, the pipeline architecture (config-driven site selection,
Copernicus ingestion, event detection, lagged causal analysis, scenario forecasting) is built
to be retargeted — e.g. to another sentinel species or bioassay endpoint, another stressor
already tracked as a candidate predictor (ocean acidification, via the pH/CO₂ series), or any
other site within Copernicus Marine's global coverage. See [`docs/ADAPTING.md`](docs/ADAPTING.md)
for what that actually requires.

See [`CITATION.cff`](CITATION.cff) for citation metadata and [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to run tests and contribute.

## Installation

This repository is a proper, installable Python package (`pyproject.toml`, `src/` layout):

```bash
git clone https://github.com/scatenag/climate_change_on_sea_urchins.git
cd climate_change_on_sea_urchins
pip install -e .
```

This installs the `climate_change_on_sea_urchins` package (analysis pipeline + dashboard)
and two console scripts:

```bash
ccsu-run-pipeline   # re-run the full statistical pipeline, populating results/
ccsu-dashboard      # launch the Streamlit dashboard (equivalent to streamlit run app.py)
```

Or use it as a library:

```python
from climate_change_on_sea_urchins import load_data
df_full, df_real, events, monthly = load_data()
```

## Contents

### Code
| File / Folder | Description |
|---|---|
| [`pyproject.toml`](pyproject.toml) | Package metadata, dependencies, console-script entry points |
| [`src/climate_change_on_sea_urchins/`](src/climate_change_on_sea_urchins/) | The installable package: 6 analysis modules + `common.py` (shared data loading) + `pipeline.py` (orchestrates all 6) + `dashboard.py` (the Streamlit app) |
| [`config.py`](config.py) | Single source of truth for site coordinates and the EC50 data-source URL — edit this to adapt the framework (see [`docs/ADAPTING.md`](docs/ADAPTING.md)) |
| [`app.py`](app.py) | Thin entry point (`import climate_change_on_sea_urchins.dashboard`) — kept so `streamlit run app.py` and the existing Streamlit Community Cloud deployment work unchanged |
| [`scripts/`](scripts/) | Data download scripts (Copernicus Marine, EC50 from Google Sheets) — standalone, not part of the installable package since they require Copernicus credentials |
| [`marineHeatWaves.py`](marineHeatWaves.py) | Vendored MHW detection library (Hobday et al. 2016) |
| [`analysis.ipynb`](analysis.ipynb) | Exploratory notebook (illustration only — the tested, reusable code is the package above) — launch via Binder badge above |

### Data
| File | Description |
|---|---|
| [`data/data_extended.csv`](data/data_extended.csv) | Main dataset: monthly environmental + EC50 data, 2003–2025 (276 months) |
| [`data/data_ec50_ci.csv`](data/data_ec50_ci.csv) | EC50 values with 95% CI bounds and imputation flag |
| [`data/mhw_events.csv`](data/mhw_events.csv) | Marine Heatwave event catalogue (129 events, Hobday 2016 classification) |
| [`data/mhw_monthly.csv`](data/mhw_monthly.csv) | Monthly aggregated MHW metrics |
| [`data/mhw_annual.csv`](data/mhw_annual.csv) | Annual aggregated MHW metrics |
| [`data/env_copernicus.csv`](data/env_copernicus.csv) | Raw monthly env. variables fetched from Copernicus Marine |
| [`data/sst_daily.csv`](data/sst_daily.csv) | Raw daily SST fetched from Copernicus Marine (for MHW detection) |
| [`data/ec50_sheets.csv`](data/ec50_sheets.csv) | Raw EC50 export from Google Sheets |
| [`data/data.csv`](data/data.csv) | Original 2003–2022 dataset from the Sartori et al. (2023) supplement — kept as a validation reference for `scripts/build_dataset.py` |
| [`results/`](results/) | Pre-computed analysis outputs: correlations, stationarity tests, forecasts, R model outputs |

## Streamlit dashboard

The interactive dashboard is deployed at the link above (Streamlit badge).
To run locally, after [installing the package](#installation):

```bash
ccsu-dashboard
# equivalently: streamlit run app.py
```

## Running the notebook

Click the **Binder** badge above to run the notebook interactively in the browser — no installation required.

To run locally (the notebook additionally needs `jupyter` and `pmdarima`):

```bash
pip install -e ".[notebook]"
jupyter notebook analysis.ipynb
```

## Data provenance & sources

Site: **43.4278°N, 10.3956°E** (off Livorno, North Tyrrhenian Sea), ±0.1° bounding box, surface
layer (0–10 m) — see [`config.py`](config.py) for the single source of truth.

### Environmental variables (Copernicus Marine Service)

Fetched via the [`copernicusmarine`](https://pypi.org/project/copernicusmarine/) Python toolbox
(not a plain REST endpoint) — see [`scripts/fetch_copernicus.py`](scripts/fetch_copernicus.py) /
[`scripts/fetch_copernicus_daily.py`](scripts/fetch_copernicus_daily.py) for the exact calls.

| Variable | Product | Dataset ID | CMEMS variable | Unit |
|---|---|---|---|---|
| Temperature | [Mediterranean Sea Physics Reanalysis — MEDSEA_MULTIYEAR_PHY_006_004](https://data.marine.copernicus.eu/product/MEDSEA_MULTIYEAR_PHY_006_004/description) | `cmems_mod_med_phy-temp_my_4.2km_P1M-m` | `thetao` | °C |
| Salinity | same product | `cmems_mod_med_phy-sal_my_4.2km_P1M-m` | `so` | PSU |
| O₂ | [Mediterranean Sea Biogeochemistry Reanalysis — MEDSEA_MULTIYEAR_BGC_006_008](https://data.marine.copernicus.eu/product/MEDSEA_MULTIYEAR_BGC_006_008/description) | `cmems_mod_med_bgc-bio_my_4.2km_P1M-m` | `o2` | mmol/m³ |
| pH | same product | `cmems_mod_med_bgc-car_my_4.2km_P1M-m` | `ph` | total scale |
| CO₂ | same product | `cmems_mod_med_bgc-co2_my_4.2km_P1M-m` | `spco2` | µatm |
| Daily SST (MHW detection only) | Physics Reanalysis, daily resolution | `cmems_mod_med_phy-temp_my_4.2km_P1D-m` | `thetao` | °C |

Months not yet folded into the multiyear reanalysis are backfilled from the equivalent
`MEDSEA_ANALYSISFORECAST` near-real-time product (`..._anfc_...` dataset IDs, same variables) —
see the fallback IDs in the fetch scripts.

> ⚠️ **CO₂ unit caveat** (flagged directly in `scripts/fetch_copernicus.py`): the CO₂ column in
> the original 2003–2022 dataset (`data/data.csv`, from Sartori et al. 2023) has values ~33–58
> with no recorded unit, while Copernicus's `spco2` is surface pCO₂ in µatm (typically ~380–450
> in the Mediterranean) — these are plausibly *different quantities*.
> [`scripts/build_dataset.py`](scripts/build_dataset.py) cross-checks the overlap period before
> merging; treat the merged CO₂ series with this caveat in mind for anything beyond internal
> trend analysis.

### Marine heatwave detection

Hobday et al. (2016) method — 90th-percentile threshold on an 11-day moving-window daily
climatology (2003–2012 baseline), 5-day minimum event duration, ≤2-day gaps merged. Vendored
reference implementation in [`marineHeatWaves.py`](marineHeatWaves.py); the production detection
run uses the equivalent, explicitly-parameterized reimplementation in
[`scripts/detect_mhw.py`](scripts/detect_mhw.py).

### EC50 bioassay (biological sentinel data)

*Paracentrotus lividus* fertilization/embryo-toxicity assay (metal toxicity endpoint), collected
and maintained by **ISPRA** (Italian National Institute for Environmental Protection and
Research), published as a public Google Sheets export
(`config.py:EC50_EXPORT_URL` → `https://docs.google.com/spreadsheets/d/<sheet-id>/export?format=csv`,
raw columns `ID, DATE, EC50, UL, LL, pos, neg`).

## Automation & data validation

| Workflow | Trigger / frequency | What it does |
|---|---|---|
| [`tests.yml`](.github/workflows/tests.yml) | Every push/PR to `main` | Code-correctness test suite (`pytest tests/`) |
| [`update_ec50.yml`](.github/workflows/update_ec50.yml) | Daily 06:00 UTC (EC50) + monthly on the 5th at 07:00 UTC (Copernicus env data) + manual | Re-fetches EC50/Copernicus data, rebuilds the merged dataset, reruns the Python analysis pipeline, commits+pushes anything that changed |
| [`validate_data.yml`](.github/workflows/validate_data.yml) | Daily 06:30 UTC + on push touching `data/`/`results/` + manual | Runs [`tests/test_data_quality.py`](tests/test_data_quality.py) — see below |

**What the "Data validated" badge certifies, and what it doesn't:** it reflects that the current
`data/` and `results/` files pass automated checks for internal consistency — values within
physically plausible ranges, no duplicate/out-of-order timestamps, EC50 confidence intervals
bracketing their point estimate, marine heatwave event geometry self-consistent (start ≤ peak ≤
end, duration meeting the Hobday minimum), and p-values/correlation coefficients from the
analysis pipeline within their valid mathematical range. It does **not** certify the deeper
scientific accuracy of the upstream Copernicus reanalysis or of the EC50 bioassay itself — those
remain the responsibility of the original data providers (CMEMS, ISPRA) and standard scientific
peer review. See [`tests/test_data_quality.py`](tests/test_data_quality.py) for the exact checks.
