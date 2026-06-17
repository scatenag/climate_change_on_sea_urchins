"""
Tests for the climate_change_on_sea_urchins analysis pipeline.

These tests verify that:
- Core data files are present and well-formed
- The shared data-loading utility returns expected structure
- Pre-computed results exist and have non-trivial content
- Key scientific constants are correctly defined
- Site/data-source configuration is centralized in config.py (no drift)
- The analysis layer works on data shaped like, but not equal to, the
  real dataset — see docs/ADAPTING.md
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.common import load_data, ENV_COLS, ALL_COLS, MHW_COLS, SPLIT_YEAR, TAU_MAX
from config import SITE_LAT, SITE_LON, SITE_NAME, EC50_EXPORT_URL


def _load_module(path: Path):
    """Import a script module by file path without executing its __main__ block."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_split_year():
    assert SPLIT_YEAR == "2016"


def test_tau_max():
    assert TAU_MAX == 12


# ---------------------------------------------------------------------------
# Input data files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname", [
    "data_extended.csv",
    "data_ec50_ci.csv",
    "mhw_events.csv",
    "mhw_monthly.csv",
    "mhw_annual.csv",
])
def test_input_files_exist(fname):
    assert (ROOT / fname).exists(), f"Missing input file: {fname}"


def test_data_extended_columns():
    df = pd.read_csv(ROOT / "data_extended.csv")
    required = ["Datetime", "EC50"] + ENV_COLS
    for col in required:
        assert col in df.columns, f"Missing column '{col}' in data_extended.csv"


def test_data_extended_length():
    df = pd.read_csv(ROOT / "data_extended.csv")
    assert len(df) >= 276, "data_extended.csv should have at least 276 monthly rows"


def test_mhw_events_columns():
    df = pd.read_csv(ROOT / "mhw_events.csv")
    for col in ["start_date", "end_date", "peak_date", "duration_days",
                "intensity_max", "category"]:
        assert col in df.columns, f"Missing column '{col}' in mhw_events.csv"


def test_mhw_events_count():
    df = pd.read_csv(ROOT / "mhw_events.csv")
    assert len(df) >= 100, "Expected at least 100 MHW events"


# ---------------------------------------------------------------------------
# load_data() — shared utility
# ---------------------------------------------------------------------------

def test_load_data_returns_four_objects():
    result = load_data()
    assert len(result) == 4


def test_df_full_has_mhw_columns():
    df_full, _, _, _ = load_data()
    for col in MHW_COLS:
        assert col in df_full.columns, f"Missing MHW column '{col}' in df_full"


def test_df_real_only_real_measurements():
    _, df_real, _, _ = load_data()
    assert "EC50_imputed" in df_real.columns
    assert df_real["EC50_imputed"].sum() == 0, \
        "df_real must contain only non-imputed EC50 measurements"


def test_df_real_non_empty():
    _, df_real, _, _ = load_data()
    assert len(df_real) >= 100, "Expected at least 100 real EC50 measurements"


def test_df_full_no_missing_mhw():
    df_full, _, _, _ = load_data()
    for col in MHW_COLS:
        assert df_full[col].isna().sum() == 0, \
            f"Column '{col}' should have no NaN values (filled with 0)"


def test_events_dataframe_not_empty():
    _, _, events, _ = load_data()
    assert len(events) > 0


def test_monthly_dataframe_has_mhw_cols():
    _, _, _, monthly = load_data()
    assert "mhw_days" in monthly.columns


# ---------------------------------------------------------------------------
# Pre-computed results (produced by analysis/run_all.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname", [
    "ccf_results.csv",
    "corr_all.csv",
    "corr_pre.csv",
    "corr_post.csv",
    "ardl_response.csv",
    "forecast_bad.csv",
    "forecast_mean.csv",
    "forecast_good.csv",
    "stationarity_results.json",
    "granger_results.json",
])
def test_results_files_exist(fname):
    path = ROOT / "results" / fname
    assert path.exists(), (
        f"Missing results file: results/{fname}. "
        "Run `python analysis/run_all.py` to generate it."
    )


def test_ccf_results_has_lag_column():
    df = pd.read_csv(ROOT / "results" / "ccf_results.csv")
    assert "lag" in df.columns


def test_forecast_bad_longer_than_training():
    df = pd.read_csv(ROOT / "results" / "forecast_bad.csv")
    assert len(df) >= 12, "Forecast should cover at least 12 future months"


# ---------------------------------------------------------------------------
# Configuration and reusability (docs/ADAPTING.md)
#
# These guard against regressions where site/data-source constants drift
# back to being hardcoded in individual scripts instead of read from the
# single config.py source of truth.
# ---------------------------------------------------------------------------

def test_config_site_values_sane():
    assert -90 <= SITE_LAT <= 90
    assert -180 <= SITE_LON <= 180
    assert isinstance(SITE_NAME, str) and SITE_NAME


def test_config_ec50_url_well_formed():
    assert EC50_EXPORT_URL.startswith("https://docs.google.com/spreadsheets/")


@pytest.mark.parametrize("script", [
    "fetch_copernicus.py",
    "fetch_copernicus_update.py",
    "fetch_copernicus_daily.py",
])
def test_copernicus_scripts_use_config_site(script):
    # Checked statically (not imported): these scripts import copernicusmarine
    # and xarray, which are intentionally excluded from requirements.txt/CI
    # (see the note in requirements.txt) since they're only needed to run the
    # download scripts locally, not to test or run the dashboard.
    src = (ROOT / "scripts" / script).read_text()
    assert "from config import SITE_LAT, SITE_LON" in src
    assert f"SITE_LAT={SITE_LAT}" not in src.replace(" ", "")


def test_fetch_ec50_uses_config_url():
    mod = _load_module(ROOT / "scripts" / "fetch_ec50.py")
    assert mod.EXPORT_URL == EC50_EXPORT_URL


def test_app_imports_site_from_config():
    # app.py is a Streamlit script (executes dashboard logic and live network
    # calls at import time) — checked statically rather than imported.
    src = (ROOT / "app.py").read_text()
    assert "from config import" in src
    assert "SITE_LAT" in src and "EC50_EXPORT_URL" in src


# ---------------------------------------------------------------------------
# Reusability smoke test — synthetic "different site" data
#
# load_data() must not hardcode anything about the real Livorno dataset
# beyond the documented column names (ENV_COLS/ALL_COLS) and SPLIT_YEAR.
# This proves the claim in docs/ADAPTING.md tier 1/2: a structurally
# identical dataset from a different site, period, or length loads and
# processes correctly. It deliberately does NOT claim to test tier 3
# (a genuinely different biological indicator), which docs/ADAPTING.md
# is explicit about requiring real engineering work.
# ---------------------------------------------------------------------------

def test_load_data_on_synthetic_different_site_dataset(tmp_path, monkeypatch):
    import analysis.common as common

    n = 48
    dates = pd.date_range("2010-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(42)

    extended = pd.DataFrame({
        "Datetime": dates,
        "O2": rng.uniform(190, 230, n),
        "CO2": rng.uniform(380, 430, n),
        "Temperature": 15 + 5 * np.sin(np.arange(n) / 6),
        "Salinity": rng.uniform(36, 38, n),
        "pH": rng.uniform(7.9, 8.2, n),
        "EC50": rng.uniform(20, 60, n),
    })
    n_real = 40
    ci = pd.DataFrame({
        "Datetime": dates,
        "EC50_imputed": [False] * n_real + [True] * (n - n_real),
        "EC50_ci_upper": extended["EC50"] + 2,
        "EC50_ci_lower": extended["EC50"] - 2,
    })
    monthly = pd.DataFrame({
        "Datetime": dates,
        "mhw_days": rng.integers(0, 10, n),
        "mhw_peak_intensity": rng.random(n) * 2,
        "mhw_cum_intensity": rng.random(n) * 10,
    })
    events = pd.DataFrame({
        "start_date": pd.date_range("2010-02-01", periods=5, freq="120D"),
        "end_date":   pd.date_range("2010-02-08", periods=5, freq="120D"),
        "peak_date":  pd.date_range("2010-02-04", periods=5, freq="120D"),
        "duration_days": [7, 8, 6, 9, 7],
        "intensity_max": [1.1, 1.3, 0.9, 1.6, 1.2],
        "category": ["Moderate"] * 5,
    })

    extended.to_csv(tmp_path / "data_extended.csv", index=False)
    ci.to_csv(tmp_path / "data_ec50_ci.csv", index=False)
    monthly.to_csv(tmp_path / "mhw_monthly.csv", index=False)
    events.to_csv(tmp_path / "mhw_events.csv", index=False)

    monkeypatch.setattr(common, "ROOT", tmp_path)

    df_full, df_real, loaded_events, loaded_monthly = common.load_data()

    assert len(df_full) == n
    assert len(df_real) == n_real
    assert set(ENV_COLS).issubset(df_full.columns)
    assert len(loaded_events) == 5
    assert "mhw_days" in loaded_monthly.columns
