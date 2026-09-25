# Climate Change on Sea Urchins 🦔

[![Tests](https://github.com/scatenag/climate_change_on_sea_urchins/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/scatenag/climate_change_on_sea_urchins/actions/workflows/tests.yml)
[![Data validated](https://github.com/scatenag/climate_change_on_sea_urchins/actions/workflows/validate_data.yml/badge.svg?branch=main)](https://github.com/scatenag/climate_change_on_sea_urchins/actions/workflows/validate_data.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/scatenag/climate_change_on_sea_urchins/main?labpath=notebooks/analysis.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://climate-change-on-sea-urchins.streamlit.app)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19352308.svg)](https://doi.org/10.5281/zenodo.19352308)
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

**Jump to:** [Installation](#installation) · [Contents](#contents) · [Dashboard](#streamlit-dashboard) ·
[Notebook](#running-the-notebook) · [Data provenance](#data-provenance--sources) ·
[Automation & validation](#automation--data-validation) ·
[Reproducing the manuscript](#reproducing-the-manuscripts-published-numbers)

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
| [`src/climate_change_on_sea_urchins/`](src/climate_change_on_sea_urchins/) | The installable package: `mhw_detection`, `timeseries`, `period_split`, `correlations`, `stationarity`, `mhw_analysis`, `mhw_lag_extra` (SEA + mixed-effects, Python port of the former R analyses), `mhw_robustness` (5-method robustness battery), `thermal_legacy` (+ its threshold sensitivity sweep, Table S2), `changepoint` (QLR/AR(1) breakpoint), `negative_control` (assay QC-series check), `mhw_lag_annual`, `mhw_annual_changepoint` (exploratory, currently unresolved — see below), `forecast` + `common.py` (shared data loading) + `pipeline.py` (orchestrates all of the above) + `dashboard.py` (the Streamlit app) |
| [`config.py`](config.py) | Single source of truth for site coordinates and the EC50 data-source URL — edit this to adapt the framework (see [`docs/ADAPTING.md`](docs/ADAPTING.md)) |
| [`app.py`](app.py) | Thin entry point (`import climate_change_on_sea_urchins.dashboard`) — kept so `streamlit run app.py` and the existing Streamlit Community Cloud deployment work unchanged |
| [`scripts/`](scripts/) | Data download scripts (Copernicus Marine, EC50 from Google Sheets) — standalone, not part of the installable package since they require Copernicus credentials |
| [`tests/`](tests/) | `pytest` suite: code correctness (`test_pipeline.py`), data/results plausibility (`test_data_quality.py`), and manuscript-value reproduction (`test_paper_values.py` + `conftest.py`'s fixture — see below) |
| [`requirements-lock.txt`](requirements-lock.txt) | `pip freeze` of the exact environment the current release's `results/` was verified reproducible in — `requirements.txt` stays the loose, hand-maintained set actually used for installs |
| [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) | End-to-end notebook: re-executes MHW detection and the full package pipeline (CCF, stationarity, Granger, SARIMAX forecast) live, from the committed data snapshot — launch via Binder badge above |
| [`notebooks/narrative_sentinel_regime_shift.ipynb`](notebooks/narrative_sentinel_regime_shift.ipynb) | Story-driven walkthrough of the regime-shift analysis; generated by `scripts/build_narrative_notebook.py`, not hand-edited |
| [`legacy/marineHeatWaves.py`](legacy/marineHeatWaves.py) | Vendored reference implementation (Hobday et al. 2016); not currently invoked — MHW detection actually runs via [`mhw_detection.py`](src/climate_change_on_sea_urchins/mhw_detection.py)'s own implementation of the same algorithm, as part of `ccsu-run-pipeline` |
| [`legacy/analysis_2023_exploratory.ipynb`](legacy/analysis_2023_exploratory.ipynb) | Superseded exploratory notebook behind the original Sartori et al. (2023) analysis — kept for reference, not maintained |

### Data
| File | Description |
|---|---|
| [`data/data_extended.csv`](data/data_extended.csv) | Main dataset: monthly environmental + EC50 data, 2003–present (updated monthly; see the auto-update workflow below — figures here are a snapshot, not refreshed on every data update) |
| [`data/data_ec50_ci.csv`](data/data_ec50_ci.csv) | EC50 values with 95% CI bounds and imputation flag |
| [`data/mhw_events.csv`](data/mhw_events.csv) | Marine Heatwave event catalogue (Hobday 2016 classification) |
| [`data/mhw_monthly.csv`](data/mhw_monthly.csv) | Monthly aggregated MHW metrics |
| [`data/mhw_annual.csv`](data/mhw_annual.csv) | Annual aggregated MHW metrics |
| [`data/env_copernicus.csv`](data/env_copernicus.csv) | Raw monthly env. variables fetched from Copernicus Marine |
| [`data/sst_daily.csv`](data/sst_daily.csv) | Raw daily SST fetched from Copernicus Marine (for MHW detection) |
| [`data/ec50_sheets.csv`](data/ec50_sheets.csv) | EC50 aggregated to monthly (Google Sheets export, one row per calendar month) |
| [`data/ec50_raw.csv`](data/ec50_raw.csv) | EC50 at full bioassay resolution, one row per trial (295), sorted by (Datetime, ID) for reproducible ordering — used by `changepoint.py`'s ordinal-sequence check and the section 3.1/3.6 manuscript-reproduction outputs below. Also carries the assay's negative-control replicates (`ctrl_neg_rep1/2/3`, present on 232 of 295 trials) |
| [`data/data.csv`](data/data.csv) | Original 2003–2022 dataset from the Sartori et al. (2023) supplement — kept as a validation reference for `scripts/build_dataset.py` |
| [`results/`](results/) | Pre-computed analysis outputs: correlations, stationarity tests, forecasts, R model outputs, and the manuscript-reproduction outputs covered [below](#reproducing-the-manuscripts-published-numbers) — regenerated in one official run per release (see `pipeline.py`), not accumulated incrementally |

## Streamlit dashboard

The interactive dashboard is deployed at the link above (Streamlit badge).
To run locally, after [installing the package](#installation):

```bash
ccsu-dashboard
# equivalently: streamlit run app.py
```

## Running the notebook

Click the **Binder** badge above to run `notebooks/analysis.ipynb` interactively in the browser — no
installation required. It reruns MHW detection and the full statistical pipeline (same code as
`ccsu-run-pipeline`) from the data already committed to this repository; raw ingestion from
Copernicus Marine and the ISPRA Google Sheet needs private credentials that Binder doesn't have,
so the notebook starts from those committed snapshots rather than re-downloading them. R-based
analyses (SEA, DLNM, mixed-effects) are not included — Binder's Python kernel has no R.

To run locally (the notebook additionally needs `jupyter`):

```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/analysis.ipynb
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

> ℹ️ **CO₂ unit note**: Copernicus's `spco2` is delivered in Pascal, and correctly documented
> as such — its CF `standard_name`, `surface_partial_pressure_of_carbon_dioxide_in_sea_water`,
> is associated with Pascal in the CF vocabulary. The error was this pipeline's, which read the
> value as µatm. Confirmed by an automated cross-check
> ([`scripts/build_dataset.py::cross_check_co2`](scripts/build_dataset.py), also enforced in
> [`tests/test_data_quality.py`](tests/test_data_quality.py)) against the original 2003–2022
> series (`data/data.csv`, Sartori et al. 2023), whose CO₂ column turned out to be in the same
> raw unit. Both are converted to µatm using the factor `1e6 / 101325` (see
> [`config.py`](config.py)`::CO2_PA_TO_UATM`) — Copernicus at ingestion, the original series only
> in memory for the cross-check, never on disk. Post-conversion the two agree closely over their
> 19-year overlap (ratio 0.99 ± 0.01) and read ~370–450 µatm, in line with typical Mediterranean
> surface pCO₂.

### Marine heatwave detection

Hobday et al. (2016) method — 90th-percentile threshold on an 11-day moving-window daily
climatology (2003–2012 baseline), 5-day minimum event duration, ≤2-day gaps merged. Vendored
reference implementation in [`marineHeatWaves.py`](legacy/marineHeatWaves.py); the production detection
run uses the equivalent, explicitly-parameterized reimplementation in
[`mhw_detection.py`](src/climate_change_on_sea_urchins/mhw_detection.py), run automatically as
the first step of `ccsu-run-pipeline` so the event catalogue can never drift out of sync with
`data/sst_daily.csv` again (see `git log` around 2026-07 for why this matters).

### EC50 bioassay (biological sentinel data)

*Paracentrotus lividus* fertilization/embryo-toxicity assay (metal toxicity endpoint), collected
and maintained by **[ISPRA](https://www.isprambiente.gov.it/)** (Italian National Institute for
Environmental Protection and Research), published as a public Google Sheets export
(`config.py:EC50_EXPORT_URL` → `https://docs.google.com/spreadsheets/d/<sheet-id>/export?format=csv`,
raw columns `ID, DATE, EC50, UL, LL, pos, neg`).

## Automation & data validation

| Workflow | Trigger / frequency | What it does |
|---|---|---|
| [`tests.yml`](.github/workflows/tests.yml) | Every push/PR to `main` | Code-correctness test suite (`pytest tests/`) |
| [`update_ec50.yml`](.github/workflows/update_ec50.yml) | Daily 06:00 UTC (EC50) + monthly on the 5th at 07:00 UTC (Copernicus env data) + manual | Re-fetches EC50/Copernicus data, rebuilds the merged dataset, reruns the Python analysis pipeline **and** the R DLNM analysis (`scripts/mhw_lag_analysis.R`), commits+pushes anything that changed |
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

## Reproducing the manuscript's published numbers

The analysis published in Sartori, Scatena, Gaion et al. (*Marine Pollution Bulletin*,
submitted) corresponds to package release **v1.5.0** (see the DOI badge above). The live
dashboard and the tip of this repository reflect the most recent data available and will keep
diverging from that specific release as the underlying series grows via the automated monthly
update — see [`tests/fixtures/paper_mpb_2026/`](tests/fixtures/paper_mpb_2026/) below for the
frozen snapshot the release's own numbers are checked against.

The manuscript points to this package as an independent means of verifying its results.
[`tests/test_paper_values.py`](tests/test_paper_values.py) re-runs the relevant analysis modules
(via [`tests/conftest.py`](tests/conftest.py)) against that frozen data snapshot — not the live
`data/`, and not precomputed `results/` — and checks, with a declared tolerance, that they
reproduce the published values for:

| Section | Output | What it checks |
|---|---|---|
| 3.1 (trial-level pre/post contrast) | [`results/period_contrast_raw.json`](results/livorno-paracentrotus/period_contrast_raw.json) | n/mean/SD/median/Mann-Whitney on the 295 individual EC50 determinations, split at `SPLIT_DATE` |
| 3.6 (negative control) | [`results/negative_control.json`](results/livorno-paracentrotus/negative_control.json) | Trend, pre/post level and dispersion, and an independent QLR/AR(1) changepoint search on the assay's own negative-control series — plus a data-quality check for single-replicate outliers |
| Table S2 (thermal threshold sensitivity) | [`results/thermal_threshold_sensitivity.csv`](results/livorno-paracentrotus/thermal_threshold_sensitivity.csv) | Same detrended/partial tests as the primary 24°C thermal-legacy analysis, swept over 22–26°C |

Not every published number is reproduced exactly — where a check surfaced a real discrepancy
(a stale split-date carried over from an earlier draft, an unresolved metric definition), the
corresponding output records the discrepancy explicitly rather than silently matching it. See
each output's own `note`/`status` fields for details.

For exact numerical reproduction (not just the pattern/tolerance `test_paper_values.py` checks),
use [`requirements-lock.txt`](requirements-lock.txt) — the frozen environment this release's own
`results/` was generated and verified in — alongside the frozen data fixture above.

[`results/mhw_annual_changepoint.json`](results/livorno-paracentrotus/mhw_annual_changepoint.json) applied the same
changepoint procedure to the annual MHW exposure metric, investigated for a manuscript
paragraph that no (metric, year-range) variant tried ended up reproducing — that paragraph was
removed from the manuscript as a result. The output is kept (`cited_in_manuscript: false`) as a
record of what was tried, but is no longer part of the reproduction check above.
