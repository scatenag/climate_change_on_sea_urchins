# Climate Change on Sea Urchins 🦔

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/scatenag/climate_change_on_sea_urchins/analysis-2025?labpath=analysis.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://climate-change-on-sea-urchins.streamlit.app)

Study on the impact of climate change and **Marine Heatwaves** on gamete sensitivity of *Paracentrotus lividus* in the Ligurian Sea (Gulf of La Spezia, 44.1°N 9.8°E).

This repository is the supplementary material for:
> *"Increased sensitivity of marine invertebrates to metal toxicity in the past two decades linked to Climate Change and Ocean Acidification: revelations from a natural population of sea urchins in the Mediterranean Sea."*
> Guido Scatena, Andrea Gaion, Andrea David Sartori — ISPRA

## Contents

### Code
| File / Folder | Description |
|---|---|
| [`app.py`](app.py) | Streamlit dashboard (reads pre-computed CSVs, no live analysis) |
| [`analysis/`](analysis/) | Modular Python analysis scripts (`run_all.py` executes the full pipeline) |
| [`scripts/`](scripts/) | Data download scripts (Copernicus Marine, EC50 from Google Sheets) |
| [`marineHeatWaves.py`](marineHeatWaves.py) | Vendored MHW detection library (Hobday et al. 2016) |
| [`analysis.ipynb`](analysis.ipynb) | Exploratory notebook — launch via Binder badge above |

### Data
| File | Description |
|---|---|
| [`data_extended.csv`](data_extended.csv) | Main dataset: monthly environmental + EC50 data, 2003–2025 (276 months) |
| [`data_ec50_ci.csv`](data_ec50_ci.csv) | EC50 values with 95% CI bounds and imputation flag |
| [`mhw_events.csv`](mhw_events.csv) | Marine Heatwave event catalogue (129 events, Hobday 2016 classification) |
| [`mhw_monthly.csv`](mhw_monthly.csv) | Monthly aggregated MHW metrics |
| [`mhw_annual.csv`](mhw_annual.csv) | Annual aggregated MHW metrics |
| [`sea_results.csv`](sea_results.csv) | Superposed Epoch Analysis output (R) |
| [`dlnm_results.csv`](dlnm_results.csv) | Distributed Lag Non-linear Model predictions + CI (R) |
| [`dlnm_lag_profile.csv`](dlnm_lag_profile.csv) | DLNM cumulative response profile over lags (R) |
| [`mixed_effects_predictions.csv`](mixed_effects_predictions.csv) | Mixed-effects model predictions (R) |
| [`results/`](results/) | Pre-computed outputs: correlations, stationarity tests, forecasts |

## Running the notebook

Click the **Binder** badge above to run the notebook interactively in the browser — no installation required.

To run locally:

```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

## Streamlit dashboard

The interactive dashboard is deployed at the link above (Streamlit badge).
To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data sources

| Data | Source |
|---|---|
| Monthly SST, Salinity, O₂, pH, CO₂ | Copernicus Marine Service — MEDSEA_MULTIYEAR |
| Daily SST (for MHW detection) | Copernicus Marine Service — MEDSEA_MULTIYEAR daily |
| Monthly EC50 bioassay | ISPRA — Istituto Superiore per la Protezione e la Ricerca Ambientale |
