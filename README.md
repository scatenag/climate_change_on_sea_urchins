# Climate Change on Sea Urchins 🦔

[![Tests](https://github.com/scatenag/climate_change_on_sea_urchins/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/scatenag/climate_change_on_sea_urchins/actions/workflows/tests.yml)
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
*Paracentrotus lividus* (Ligurian Sea, off Livorno, 43.43°N 10.40°E); originally developed as
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

## Data sources

| Data | Source |
|---|---|
| Monthly SST, Salinity, O₂, pH, CO₂ | Copernicus Marine Service — MEDSEA_MULTIYEAR |
| Daily SST (for MHW detection) | Copernicus Marine Service — MEDSEA_MULTIYEAR daily |
| Monthly EC50 bioassay | ISPRA — Italian National Institute for Environmental Protection and Research |
