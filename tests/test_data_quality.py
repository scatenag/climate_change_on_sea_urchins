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

import numpy as np
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
MHW_MIN_DURATION_DAYS = 5  # Hobday et al. (2016) criterion, hardcoded as MIN_DAYS in mhw_detection.py


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


# ---------------------------------------------------------------------------
# Provenance cross-check — CO2 unit consistency
#
# scripts/build_dataset.py::cross_check_co2() prints this same comparison
# but only when someone runs the script locally, so its result (currently
# ratio 0.99 +/- 0.01, see README "CO2 unit note") was never actually
# enforced anywhere. Turning it into an assertion means a future data
# refresh that silently breaks this consistency (e.g. a Copernicus product
# change) fails CI instead of only being visible to whoever happens to
# re-run build_dataset.py locally and read its stdout.
# ---------------------------------------------------------------------------

def test_co2_cross_check_ratio_near_one():
    env = pd.read_csv(ROOT / "data" / "env_copernicus.csv", parse_dates=["Datetime"])
    orig = pd.read_csv(ROOT / "data" / "data.csv")
    orig.columns = orig.columns.str.strip()
    orig = orig.rename(columns={"CO2_Con": "CO2"})
    orig["Datetime"] = pd.to_datetime(orig["date"], dayfirst=True)

    merged = pd.merge(
        env[["Datetime", "CO2"]],
        orig[["Datetime", "CO2"]].rename(columns={"CO2": "CO2_orig"}),
        on="Datetime", how="inner",
    ).dropna()
    assert not merged.empty, "No overlap between Copernicus and original CO2 series to cross-check"

    ratio = (merged["CO2"] / merged["CO2_orig"]).mean()
    assert 0.9 <= ratio <= 1.1, (
        f"Copernicus/original CO2 ratio drifted to {ratio:.2f} (expected ~1.0) — "
        "the two series are no longer consistent, see README 'CO2 unit note'"
    )


# ---------------------------------------------------------------------------
# Provenance cross-check — MHW catalogue must be reproducible from sst_daily.csv
#
# data/mhw_events.csv / mhw_monthly.csv / mhw_annual.csv are committed data,
# not just derived-on-the-fly results, so nothing stopped them from silently
# drifting out of sync with data/sst_daily.csv if the two were ever updated
# separately. That happened for months in this project's history (a
# site-coordinate fix in March 2026 regenerated sst_daily.csv but not the
# MHW catalogue derived from it, discovered only in July 2026 — see project
# history / drafts/nuova pubblicazione/refresh_dati_copernicus_2026-07-18.md).
# mhw_detection.run() is deterministic and is now the first step of
# ccsu-run-pipeline, so this should never recur, but this test makes the
# invariant explicit and catches it immediately (rather than months later)
# if it ever does.
# ---------------------------------------------------------------------------

def test_mhw_monthly_matches_recomputation_from_sst_daily():
    from climate_change_on_sea_urchins.mhw_detection import (
        compute_climatology, detect_events, to_monthly, CLIM_START, CLIM_END,
    )

    sst = pd.read_csv(ROOT / "data" / "sst_daily.csv", parse_dates=["Datetime"])
    clim = compute_climatology(sst, CLIM_START, CLIM_END)
    _events, daily = detect_events(sst, clim)
    recomputed = to_monthly(daily).set_index("Datetime")

    committed = _read("mhw_monthly.csv").set_index("Datetime")
    common_idx = recomputed.index.intersection(committed.index)
    assert len(common_idx) > 0, "No overlapping months between recomputed and committed mhw_monthly.csv"

    diff = (recomputed.loc[common_idx, "mhw_days"] - committed.loc[common_idx, "mhw_days"]).abs()
    bad_months = diff[diff > 0]
    assert bad_months.empty, (
        f"{len(bad_months)} month(s) where mhw_monthly.csv's mhw_days does not match what "
        f"mhw_detection.py recomputes from the current data/sst_daily.csv: "
        f"{[d.strftime('%Y-%m') for d in bad_months.index[:10]]}"
        f"{' ...' if len(bad_months) > 10 else ''}. "
        "This means data/mhw_events.csv/mhw_monthly.csv/mhw_annual.csv are stale relative to "
        "data/sst_daily.csv — run `ccsu-run-pipeline` (which regenerates them first) and commit "
        "the result."
    )


# ---------------------------------------------------------------------------
# Copper speciation decomposition — the geochemical (ocean-acidification)
# correction must stay small, otherwise the "the decline is biological, not a
# water-chemistry artifact" conclusion silently changes. Also guards the
# carbonate-system implementation against unit/formula regressions.
# ---------------------------------------------------------------------------

def test_cu_speciation_carbonate_ion_plausible():
    df = pd.read_csv(ROOT / "results" / "cu_speciation_decomposition.csv")
    co3_umol = df["CO3"] * 1e6
    assert co3_umol.between(150.0, 300.0).all(), (
        f"Carbonate ion outside plausible NW-Mediterranean surface range "
        f"[150,300] µmol/kg: min={co3_umol.min():.0f}, max={co3_umol.max():.0f}. "
        "Check the carbonate-system constants / total-alkalinity estimate in cu_speciation.py."
    )


def test_cu_speciation_correction_is_small():
    df = pd.read_csv(ROOT / "results" / "cu_speciation_decomposition.csv")
    # Per-month amplification varies with monthly pH; wide bound catches only
    # gross bugs (e.g. a factor-of-two error), not real seasonal pH swings.
    for col in ["fCu_amplification", "fCu_amplification_lit"]:
        assert df[col].between(0.75, 1.40).all(), (
            f"{col} outside [0.75,1.40] — implausible free-Cu²⁺ amplification, likely a bug."
        )
    # What the decomposition actually uses: the era-mean amplification stays close
    # to 1 because the realized pH change between eras is only ~0.01 units.
    s = json.loads((ROOT / "results" / "cu_speciation_summary.json").read_text())
    for key in ["fCu_amplification_post_carbonate", "fCu_amplification_post_literature"]:
        assert 0.95 <= s[key] <= 1.10, f"{key}={s[key]:.3f} — era-mean amplification too far from 1."


def test_cu_speciation_decline_is_mostly_biological():
    s = json.loads((ROOT / "results" / "cu_speciation_summary.json").read_text())
    assert abs(s["geochemical_share_literature_pct"]) < 15.0, (
        f"Ocean-acidification (Cu speciation) now explains "
        f"{s['geochemical_share_literature_pct']:.1f}% of the EC50 decline (was ~3%). "
        "The paper's central claim that the sensitization is genuinely biological "
        "assumes this share is small — re-examine before shipping."
    )
    assert s["biological_residual_mannwhitney_p"] < 0.01, (
        "Speciation-corrected EC50 no longer declines significantly pre/post-2016; "
        "the biological-sensitization result does not hold on current data."
    )


# ---------------------------------------------------------------------------
# Thermal-legacy hypothesis test — integrity guards. This module deliberately
# reports a NEGATIVE/HONEST result (thermal dose does not beat a plain time
# trend); these checks guard the arithmetic and the "detrending weakens the
# correlation" co-trend phenomenon, not a scientific claim.
# ---------------------------------------------------------------------------

def test_thermal_legacy_dose_positive_and_finite():
    df = pd.read_csv(ROOT / "results" / "thermal_legacy.csv")
    dose_cols = [c for c in df.columns if c.startswith("dose_")]
    assert dose_cols, "no thermal-dose columns in thermal_legacy.csv"
    for c in dose_cols:
        assert np.isfinite(df[c]).all() and (df[c] >= 0).all(), (
            f"{c}: thermal dose (degree-days) must be finite and non-negative"
        )


def test_thermal_legacy_verdict_and_cotrend():
    s = json.loads((ROOT / "results" / "thermal_legacy_summary.json").read_text())
    assert s["verdict"] in {
        "supported_after_detrending",
        "consistent_but_not_separable_from_trend",
    }, f"unexpected verdict: {s['verdict']}"
    for row in s["per_window"]:
        for k in ("raw_spearman_r", "detrended_spearman_r"):
            assert -1.0001 <= row[k] <= 1.0001, f"{k} outside [-1,1]"
    # The whole point: removing the time trend weakens the strongest raw
    # correlation. If this ever fails, the co-trend framing needs revisiting.
    strongest = min(s["per_window"], key=lambda r: r["raw_p"])
    assert abs(strongest["detrended_spearman_r"]) < abs(strongest["raw_spearman_r"]), (
        "Detrending did NOT weaken the strongest raw correlation — re-examine the "
        "co-trend interpretation in thermal_legacy.py."
    )
