"""
Tests for the climate_change_on_sea_urchins analysis pipeline.

These tests verify that:
- Core data files are present and well-formed
- The shared data-loading utility returns expected structure
- Pre-computed results exist and have non-trivial content
- Key scientific constants are correctly defined
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.common import load_data, ENV_COLS, ALL_COLS, MHW_COLS, SPLIT_YEAR, TAU_MAX


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
