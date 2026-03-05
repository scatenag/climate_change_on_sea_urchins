# Climate Change on Sea Urchins 🦔

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/scatenag/climate_change_on_sea_urchins/analysis-2025?labpath=analysis.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://climate-change-on-sea-urchins.streamlit.app)

Study on the impact of climate change and **Marine Heatwaves** on gamete sensitivity of *Paracentrotus lividus* in the Ligurian Sea (Gulf of La Spezia, 44.1°N 9.8°E).

This repository is the supplementary material for:
> *"Increased sensitivity of marine invertebrates to metal toxicity in the past two decades linked to Climate Change and Ocean Acidification: revelations from a natural population of sea urchins in the Mediterranean Sea."*
> Davide Sartori, Guido Scatena, Cristina Vrinceanu, Andrea Gaion — ISPRA

## Contents

| File / Folder | Description |
|---|---|
| [`analysis.ipynb`](analysis.ipynb) | Main statistical analysis notebook |
| [`data_extended.csv`](data_extended.csv) | Monthly environmental + EC50 data, 2003–2025 |
| [`mhw_events.csv`](mhw_events.csv) | Marine Heatwave event catalogue |
| [`mhw_monthly.csv`](mhw_monthly.csv) | Monthly MHW metrics |
| [`results/`](results/) | Pre-computed outputs (correlations, forecasts, stationarity…) |
| [`app.py`](app.py) | Streamlit dashboard (reads pre-computed CSVs) |
| [`analysis/`](analysis/) | Modular Python analysis scripts |
| [`scripts/`](scripts/) | Data download scripts (Copernicus, EC50) |

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
