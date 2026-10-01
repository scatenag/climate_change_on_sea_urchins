"""
A month without data stays without data, at the edges of every series: no
fill may extend a series beyond its last (or before its first) observed
month. Environmental variables from the monthly reanalysis and the response
itself end at different months; three missing months at the end, as for the
SST on 2026-09-30 (tests/test_mhw_missing_sst.py covers the MHW metrics).
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

from climate_change_on_sea_urchins import common
from climate_change_on_sea_urchins.common import RESPONSE_COL
from climate_change_on_sea_urchins.correlations import extract_trends
from climate_change_on_sea_urchins.forecast import build_monthly_series
from climate_change_on_sea_urchins.mhw_analysis import compute_ccf_prewhitened, compute_granger

N_MISSING = 3


def _synthetic(n=120, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    driver = np.zeros(n)
    for i in range(1, n):
        driver[i] = 0.5 * driver[i - 1] + rng.normal()
    df = pd.DataFrame({
        "Datetime": pd.date_range("2010-01-01", periods=n, freq="MS"),
        "mhw_peak_intensity": driver,
        "Temperature": 20 + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(size=n),
        "pH": 8.1 - 0.001 * t + 0.01 * rng.normal(size=n),
        RESPONSE_COL: 40 - 0.1 * t + rng.normal(size=n),
    })
    df[f"{RESPONSE_COL}_imputed"] = False
    return df


def _with_missing_tail(df, col):
    out = df.copy()
    out.loc[out.index[-N_MISSING:], col] = np.nan
    return out


def _dashboard_function(name):
    """One top-level function of dashboard.py, compiled on its own and
    without its decorators: importing the module would start the app."""
    src = (Path(common.__file__).parent / "dashboard.py").read_text()
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    ns = {"pd": pd, "np": np, "stats": stats, "seasonal_decompose": seasonal_decompose,
          "SPLIT_DATE": pd.Timestamp("2015-01-01")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "dashboard.py", "exec"), ns)
    return ns[name]


def test_trends_stay_missing_beyond_last_observed_month():
    df = _with_missing_tail(_synthetic(), "Temperature").set_index("Datetime")
    trend = extract_trends(df, ["Temperature"])["Temperature"]
    assert trend.iloc[-N_MISSING:].isna().all()
    assert trend.iloc[:-N_MISSING].notna().all()


def test_forecast_series_ends_at_last_observed_environment():
    df = _with_missing_tail(_synthetic(), "Temperature")
    monthly = build_monthly_series(df, df, 0)
    assert monthly["Datetime"].max() == df["Datetime"].iloc[-N_MISSING - 1]


def test_forecast_series_ends_at_last_real_response():
    df = _synthetic()
    df_real = df.iloc[:-N_MISSING]
    monthly = build_monthly_series(df_real, df, 0)
    assert monthly["Datetime"].max() == df_real["Datetime"].max()


def test_prewhitened_target_months_without_data_form_no_pair():
    df = _with_missing_tail(_synthetic(), "Temperature")
    got, _ = compute_ccf_prewhitened(df, "mhw_peak_intensity", ["Temperature"], order=(1, 0, 1))
    for _, row in got.iterrows():
        # pairs (driver t, target t+lag) with the target observed up to
        # N - N_MISSING - 1; a fill would give N - lag
        assert row["n"] == len(df) - N_MISSING - row["lag"]


def test_granger_target_months_without_data_dropped():
    df = _synthetic()
    got = compute_granger(_with_missing_tail(df, "Temperature"), "mhw_peak_intensity", ["Temperature"])
    expected = compute_granger(df.iloc[:-N_MISSING], "mhw_peak_intensity", ["Temperature"])
    assert got == expected


def test_dashboard_forecast_series_ends_at_last_observed():
    df = _with_missing_tail(_synthetic(), "Temperature").rename(columns={RESPONSE_COL: "EC50"})
    monthly = _dashboard_function("_fc_build_monthly")(df, df, 0)
    assert monthly["Datetime"].max() == df["Datetime"].iloc[-N_MISSING - 1]
    df_real = df.iloc[:-2 * N_MISSING]
    monthly = _dashboard_function("_fc_build_monthly")(df_real, df, 0)
    assert monthly["Datetime"].max() == df_real["Datetime"].max()
