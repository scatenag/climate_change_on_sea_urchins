"""
A month without SST has no MHW catalogue: its MHW metrics are missing, not
zero. The daily SST arrives months after the response series, so at every
update the last months of data_extended.csv are beyond mhw_monthly.csv's
coverage; filling them with 0 would enter them in every analysis as months
without heatwaves (in the frozen fixture: July 2026).
"""
import ast
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_change_on_sea_urchins import common
from climate_change_on_sea_urchins.common import RESPONSE_COL
from climate_change_on_sea_urchins.forecast import build_monthly_series
from climate_change_on_sea_urchins.mhw_analysis import compute_ccf_prewhitened, compute_granger
from climate_change_on_sea_urchins.mhw_robustness import run_ccm, run_severe_ccf, run_wavelet_coherence

FIXTURE_DATA = common.Path(__file__).parent / "fixtures" / "paper_mpb_2026" / "data"
MHW_COLS = ["mhw_days", "mhw_peak_intensity", "mhw_cum_intensity"]


@pytest.fixture
def loaded(monkeypatch, tmp_path):
    # A copy of the fixture with the MHW catalogue ending two months earlier:
    # three months beyond the SST coverage, as in the real data (see below).
    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DATA, data)
    monthly = pd.read_csv(data / "mhw_monthly.csv", parse_dates=["Datetime"]).iloc[:-2]
    monthly.to_csv(data / "mhw_monthly.csv", index=False)
    monkeypatch.setattr(common, "DATA", data)
    df_full, *_ = common.load_data()
    return df_full, monthly


def test_three_months_beyond_sst(loaded):
    df_full, monthly = loaded
    assert (df_full["Datetime"] > monthly["Datetime"].max()).sum() == N_MISSING


def test_mhw_metrics_missing_beyond_sst(loaded):
    df_full, monthly = loaded
    beyond = df_full["Datetime"] > monthly["Datetime"].max()
    assert df_full.loc[beyond, MHW_COLS].isna().all().all()


def test_mhw_metrics_unchanged_within_sst(loaded):
    df_full, monthly = loaded
    within = df_full.merge(monthly[["Datetime"] + MHW_COLS], on="Datetime",
                           suffixes=("", "_monthly"))
    assert len(within) == len(monthly)
    for c in MHW_COLS:
        pd.testing.assert_series_equal(within[c], within[f"{c}_monthly"],
                                       check_names=False, check_dtype=False)


# ── Downstream: no analysis fills the missing months back in ────────────────
# Three missing months at the end, not one: the frozen fixture has one month
# beyond the SST coverage, the real data had three on 2026-09-30.

N_MISSING = 3


def _synthetic(n=120, seed=0):
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


def _with_missing_tail(df, col="mhw_peak_intensity"):
    out = df.copy()
    out.loc[out.index[-N_MISSING:], col] = np.nan
    return out


def _pairs_without_filling(n_months, lag):
    """Pairs (driver month t, response month t+lag) when the driver's last
    N_MISSING months are missing: the response months beyond the driver
    still pair with earlier driver months. A fill would give n_months - lag."""
    return min(n_months - N_MISSING, n_months - lag)


def test_prewhitened_ccf_uses_no_month_without_driver():
    df = _synthetic()
    got, _ = compute_ccf_prewhitened(_with_missing_tail(df), "mhw_peak_intensity",
                                     ["Temperature"], order=(1, 0, 1))
    assert got["spearman_r"].notna().all()
    for _, row in got.iterrows():
        assert row["n"] == _pairs_without_filling(len(df), row["lag"])


def test_granger_ignores_months_without_driver():
    df = _synthetic()
    got = compute_granger(_with_missing_tail(df), "mhw_peak_intensity", ["Temperature"])
    expected = compute_granger(df.iloc[:-N_MISSING], "mhw_peak_intensity", ["Temperature"])
    assert got == expected


def test_forecast_regressor_not_filled_beyond_driver():
    df = _with_missing_tail(_synthetic())
    lag = 2
    monthly = build_monthly_series(df, df, lag)
    last_driver = df.loc[df["mhw_peak_intensity"].notna(), "Datetime"].max()
    assert monthly["Datetime"].max() == last_driver + pd.DateOffset(months=lag)
    assert monthly["mhw_lagged"].notna().all()


def _with_events(df):
    """Severe/Extreme events peaking every 7 months across the whole record,
    catalogue included beyond the SST coverage would be impossible, so only
    within it."""
    peaks = df["Datetime"].iloc[::7]
    return pd.DataFrame({"category": "Severe", "peak_date": peaks.values,
                         "intensity_max": np.linspace(1, 3, len(peaks))})


def _robustness_frame():
    df = _synthetic()
    df["mhw_days"] = np.abs(df["mhw_peak_intensity"]) * 5
    return df


def test_severe_ccf_uses_no_month_without_catalogue():
    df = _with_missing_tail(_with_missing_tail(_robustness_frame()), "mhw_days")
    got, _ = run_severe_ccf(df, _with_events(df.iloc[:-N_MISSING]), "R")
    for _, row in got.iterrows():
        assert row["n"] == _pairs_without_filling(len(df), row["lag"])


def test_ccm_ignores_months_without_driver():
    df = _robustness_frame()
    got = run_ccm(_with_missing_tail(df), "R")
    expected = run_ccm(df.iloc[:-N_MISSING], "R")
    pd.testing.assert_frame_equal(got, expected)


def test_wavelet_ignores_months_without_driver():
    df = _robustness_frame()
    got = run_wavelet_coherence(_with_missing_tail(df), n_surrogates=5)
    expected = run_wavelet_coherence(df.iloc[:-N_MISSING], n_surrogates=5)
    assert got == expected


def _dashboard_function(name):
    """One top-level function of dashboard.py, compiled on its own: importing
    the module would start the Streamlit app."""
    src = (Path(common.__file__).parent / "dashboard.py").read_text()
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"pd": pd, "np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "dashboard.py", "exec"), ns)
    return ns[name]


def test_dashboard_forecast_regressor_not_filled_beyond_driver():
    df = _with_missing_tail(_synthetic()).rename(columns={RESPONSE_COL: "EC50"})
    lag = 2
    monthly = _dashboard_function("_fc_build_monthly")(df, df, lag)
    last_driver = df.loc[df["mhw_peak_intensity"].notna(), "Datetime"].max()
    assert monthly["Datetime"].max() == last_driver + pd.DateOffset(months=lag)
    assert monthly["mhw_lagged"].notna().all()
