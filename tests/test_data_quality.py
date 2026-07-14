"""
Data quality / plausibility validation.

Distinct from test_pipeline.py (which checks that files exist and have the
right shape): these tests check whether the *values* in data/ and results/
are physically/statistically plausible, so that a broken Copernicus fetch,
a unit-conversion bug, or a corrupted CI run fails CI loudly instead of
silently shipping bad numbers to the dashboard.

This is what the "data validated" badge in the README reflects (see
.github/workflows/validate_data.yml) — it certifies internal consistency
(values in plausible ranges, no gaps, cross-fields agree with each other),
not the deeper scientific accuracy of the upstream Copernicus reanalysis or
the biological EC50 assay itself, which no automated check can certify.

Range bounds below are derived empirically from the observed 2003-2025
record (min/max with a wide margin) rather than from literature values —
call out PHYSICAL_BOUNDS explicitly if/when literature-sourced bounds
become available (see project discussion with A. Gaion, 2026-07).
"""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent

# (min, max) — generous margin around the observed 2003-2025 range, wide
# enough to tolerate genuine future extremes (continued warming, a record
# MHW) without tolerating unit/decoding errors (e.g. a Kelvin/Celsius mixup,
# a percentage stored as a fraction).
PHYSICAL_BOUNDS = {
    "Temperature": (5.0, 32.0),      # °C, sea surface
    "Salinity":    (36.5, 39.5),     # PSU
    "pH":          (7.6, 8.3),       # total scale
    "O2":          (190.0, 290.0),   # dissolved oxygen, model units
    "CO2":         (15.0, 80.0),     # model units
    "EC50":        (1.0, 100.0),     # mg/L, sea urchin fertilization bioassay
}

MHW_CATEGORIES = {"Moderate", "Strong", "Severe", "Extreme"}
MHW_MIN_DURATION_DAYS = 5  # Hobday et al. (2016) criterion, hardcoded as MIN_DAYS in scripts/detect_mhw.py


def _read(fname):
    return pd.read_csv(ROOT / "data" / fname, parse_dates=["Datetime"]
                        if fname != "mhw_events.csv" else None)


# ---------------------------------------------------------------------------
# Structural / schema checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname", ["data_extended.csv", "data_ec50_ci.csv", "mhw_monthly.csv"])
def test_no_duplicate_timestamps(fname):
    df = _read(fname)
    dupes = df["Datetime"][df["Datetime"].duplicated()]
    assert dupes.empty, f"Duplicate Datetime values in {fname}: {dupes.tolist()}"


@pytest.mark.parametrize("fname", ["data_extended.csv", "data_ec50_ci.csv", "mhw_monthly.csv"])
def test_datetime_monotonic_increasing(fname):
    df = _read(fname)
    assert df["Datetime"].is_monotonic_increasing, \
        f"{fname} is not sorted by Datetime — downstream lag/rolling logic assumes order"


def test_data_extended_monthly_cadence():
    df = _read("data_extended.csv")
    gaps = df["Datetime"].diff().dropna()
    # Month-to-month gaps should be 28-31 days; anything else means a
    # skipped or duplicated month slipped through the merge.
    bad = gaps[(gaps.dt.days < 27) | (gaps.dt.days > 32)]
    assert bad.empty, f"Non-monthly gaps found in data_extended.csv Datetime: {bad.tolist()}"


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col", list(PHYSICAL_BOUNDS.keys()))
def test_data_extended_within_physical_bounds(col):
    df = _read("data_extended.csv")
    lo, hi = PHYSICAL_BOUNDS[col]
    values = df[col].dropna()
    assert not values.empty, f"Column '{col}' has no non-NaN values at all"
    out_of_range = values[(values < lo) | (values > hi)]
    assert out_of_range.empty, (
        f"{col}: {len(out_of_range)} value(s) outside plausible range "
        f"[{lo}, {hi}]: {out_of_range.tolist()}"
    )


def test_mhw_monthly_days_within_calendar_bounds():
    df = _read("mhw_monthly.csv")
    assert (df["mhw_days"] >= 0).all(), "mhw_days cannot be negative"
    assert (df["mhw_days"] <= 31).all(), "mhw_days cannot exceed days in a month"


def test_mhw_monthly_intensity_non_negative():
    df = _read("mhw_monthly.csv")
    assert (df["mhw_peak_intensity"].dropna() >= 0).all(), \
        "mhw_peak_intensity is an above-threshold anomaly and cannot be negative"


def test_mhw_events_duration_meets_hobday_minimum():
    df = pd.read_csv(ROOT / "data" / "mhw_events.csv")
    assert (df["duration_days"] >= MHW_MIN_DURATION_DAYS).all(), (
        f"Found MHW event(s) shorter than the Hobday (2016) {MHW_MIN_DURATION_DAYS}-day "
        "minimum — the detection algorithm itself should never emit these"
    )
    assert (df["duration_days"] <= 200).all(), \
        "Found an implausibly long (>200 day) MHW event — check for a merge/detection bug"


def test_mhw_events_category_is_known_value():
    df = pd.read_csv(ROOT / "data" / "mhw_events.csv")
    bad = set(df["category"].unique()) - MHW_CATEGORIES
    assert not bad, f"Unexpected MHW category value(s): {bad}"


# ---------------------------------------------------------------------------
# Cross-field consistency
# ---------------------------------------------------------------------------

def test_ec50_ci_brackets_point_estimate_for_real_measurements():
    df = _read("data_ec50_ci.csv")
    real = df[df["EC50_imputed"] == False].dropna(subset=["EC50", "EC50_ci_lower", "EC50_ci_upper"])
    bad = real[~((real["EC50_ci_lower"] <= real["EC50"]) & (real["EC50"] <= real["EC50_ci_upper"]))]
    assert bad.empty, (
        f"{len(bad)} real EC50 measurement(s) fall outside their own confidence interval: "
        f"{bad['Datetime'].dt.strftime('%Y-%m').tolist()}"
    )


def test_ec50_n_at_least_one_for_real_measurements():
    df = _read("data_ec50_ci.csv")
    real = df[df["EC50_imputed"] == False]
    assert (real["EC50_n"].dropna() >= 1).all(), \
        "A 'real' (non-imputed) EC50 row must be backed by at least one raw measurement"


def test_mhw_events_dates_ordered():
    df = pd.read_csv(ROOT / "data" / "mhw_events.csv", parse_dates=["start_date", "end_date", "peak_date"])
    assert (df["start_date"] <= df["peak_date"]).all(), "An MHW event's peak precedes its start"
    assert (df["peak_date"] <= df["end_date"]).all(), "An MHW event's peak follows its end"


def test_mhw_events_year_matches_start_date():
    df = pd.read_csv(ROOT / "data" / "mhw_events.csv", parse_dates=["start_date"])
    mismatched = df[df["year"] != df["start_date"].dt.year]
    assert mismatched.empty, f"event_id(s) with year != start_date's year: {mismatched['event_id'].tolist()}"


# ---------------------------------------------------------------------------
# Pipeline integrity — precomputed results should be mathematically sane
# regardless of what the input data says (catches bugs in the analysis
# code itself, not just bad inputs)
# ---------------------------------------------------------------------------

def test_stationarity_pvalues_are_probabilities():
    results = json.loads((ROOT / "results" / "stationarity_results.json").read_text())
    for r in results:
        if "error" in r:
            continue
        assert 0.0 <= r["adf_p"] <= 1.0, f"{r['variable']}: adf_p={r['adf_p']} not in [0,1]"
        if r.get("kpss_p") is not None:
            assert 0.0 <= r["kpss_p"] <= 1.0, f"{r['variable']}: kpss_p={r['kpss_p']} not in [0,1]"


def test_granger_pvalues_are_probabilities():
    granger = json.loads((ROOT / "results" / "granger_results.json").read_text())
    for var, lag_ps in granger.items():
        for lag, p in lag_ps.items():
            assert 0.0 <= p <= 1.0, f"Granger p-value for {var} lag {lag} = {p}, not in [0,1]"


@pytest.mark.parametrize("fname", ["corr_all.csv", "corr_pre.csv", "corr_post.csv"])
def test_correlation_matrices_bounded(fname):
    df = pd.read_csv(ROOT / "results" / fname, index_col=0)
    numeric = df.select_dtypes("number")
    out_of_range = numeric[(numeric < -1.0001) | (numeric > 1.0001)].stack()
    assert out_of_range.empty, f"{fname}: correlation value(s) outside [-1,1]: {out_of_range.to_dict()}"


@pytest.mark.parametrize("fname", [
    "ccf_results.csv", "ccf_results_diff.csv", "ccf_results_prewhitened.csv",
])
def test_ccf_results_bounded(fname):
    df = pd.read_csv(ROOT / "results" / fname)
    assert df["spearman_r"].between(-1.0001, 1.0001).all(), f"{fname}: spearman_r outside [-1,1]"
    assert df["p_value"].between(-0.0001, 1.0001).all(), f"{fname}: p_value outside [0,1]"
    assert (df["n"] > 0).all(), f"{fname}: non-positive sample size 'n'"
