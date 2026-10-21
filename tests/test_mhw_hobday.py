"""The heatwave climatology and events of the generic engine, faithful to Hobday et al. (2016) (ADR-0019), against the reference
implementation vendored in legacy/marineHeatWaves.py with its standard parameters: the daily threshold and the seasonal climatology for
all 366 days of the year, and the events (start, end, category) one by one, on Livorno's frozen SST and on synthetic series with
leap years, a series that starts in the middle of a year, and a baseline of 30 years.

The pipeline of Livorno (mhw_detection.py) is NOT this: its percentile is taken on the values of one day of the year and smoothed over
11 days, and its events tolerate gaps inside a run. Livorno's published numbers come from it and stay as they are; what each does
differently on Livorno's data is measured in drafts/ and summarised in ADR-0019."""
from __future__ import annotations

import ast
import importlib.util
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_change_on_sea_urchins import dataset_builder, mhw_detection, mhw_hobday

pytestmark = pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0:DeprecationWarning")   # from the vendored reference

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "climate_change_on_sea_urchins"
FIXTURE = ROOT / "tests" / "fixtures" / "paper_mpb_2026" / "data"


@pytest.fixture(scope="module")
def legacy():
    if int(np.__version__.split(".")[0]) >= 2:
        if os.environ.get("CI"):
            pytest.fail("the vendored reference needs NumPy 1: the comparison would be skipped in silence")
        pytest.skip("the vendored reference uses np.NaN, removed in NumPy 2")
    spec = importlib.util.spec_from_file_location("mhw_legacy", ROOT / "legacy" / "marineHeatWaves.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference(legacy, df, y0, y1):
    t = np.array([d.toordinal() for d in df["Datetime"].dt.date])
    mhw, clim = legacy.detect(t, df["Temperature"].to_numpy(float).copy(), climatologyPeriod=[y0, y1])
    return mhw, clim


def _synthetic(start, end, seed=0, trend=0.02, amp=6.0, noise=0.6, spells=()):
    days = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    x = np.arange(len(days))
    ar = np.zeros(len(days))
    for i in range(1, len(days)):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, noise)
    temp = 18 + amp * np.sin(2 * np.pi * (days.dayofyear.to_numpy() - 110) / 365.25) + trend * x / 365.25 + ar
    for a, b, h in spells:
        temp[(days >= a) & (days <= b)] += h
    return pd.DataFrame({"Datetime": days, "Temperature": temp})


def _compare(legacy, df, y0, y1):
    clim = mhw_hobday.compute_climatology(df, y0, y1)
    mhw, rc = _reference(legacy, df, y0, y1)
    doy = mhw_hobday.calendar_doy(df["Datetime"])
    by_doy = clim.set_index("doy")
    np.testing.assert_allclose(by_doy.loc[doy, "threshold_smooth"].to_numpy(), rc["thresh"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(by_doy.loc[doy, "climatology_smooth"].to_numpy(), rc["seas"], rtol=0, atol=1e-9)
    events, daily = mhw_hobday.detect_events(df, clim)
    got = [(e["start_date"].date(), e["end_date"].date(), e["category"]) for e in events]
    want = [(date.fromordinal(s), date.fromordinal(e), c) for s, e, c in zip(mhw["time_start"], mhw["time_end"], mhw["category"])]
    assert got == want
    return events, daily, mhw


# --- the threshold and the events are the reference's -----------------------------------------------------

def test_livornos_threshold_and_events_are_the_references(legacy):
    df = pd.read_csv(FIXTURE / "sst_daily.csv", parse_dates=["Datetime"])[["Datetime", "Temperature"]]
    events, daily, mhw = _compare(legacy, df, 2003, 2012)
    assert len(events) == mhw["n_events"] == 78                     # the repository's own detection finds 109 on the same data
    assert int(daily["in_event"].sum()) == int(sum(mhw["duration"]))


@pytest.mark.parametrize("start, end, y0, y1, seed", [
    ("1985-01-01", "2024-12-31", 1991, 2020, 1),        # 30 years of baseline, ten leap years in it
    ("2001-03-17", "2019-09-02", 2003, 2012, 2),        # the series starts and ends in the middle of a year
    ("2000-01-01", "2012-12-31", 2001, 2010, 3),        # a baseline that ends before the series does
])
def test_synthetic_series_give_the_references_threshold_and_events(legacy, start, end, y0, y1, seed):
    df = _synthetic(start, end, seed=seed, spells=[("2005-07-01", "2005-07-12", 3.0), ("2010-02-10", "2010-02-14", 3.5)])
    events, daily, mhw = _compare(legacy, df, y0, y1)
    assert len(events) >= 3


def test_a_baseline_with_no_leap_year_still_gives_a_threshold_for_every_day(legacy):
    df = _synthetic("2000-01-01", "2012-12-31", seed=4)
    clim = mhw_hobday.compute_climatology(df, 2005, 2007)
    assert len(clim) == 366 and clim["threshold_smooth"].notna().all() and clim["climatology_smooth"].notna().all()
    _compare(legacy, df, 2005, 2007)                                   # and it is the reference's, 29 February interpolated


def test_the_feb_29_value_lies_between_the_two_days_around_it_before_smoothing(legacy):
    df = _synthetic("1990-01-01", "2019-12-31", seed=5)
    clim = mhw_hobday.compute_climatology(df, 1990, 2019, smooth_width=1).set_index("doy")      # no smoothing: the raw interpolation
    assert clim.loc[60, "threshold"] == pytest.approx(0.5 * clim.loc[59, "threshold"] + 0.5 * clim.loc[61, "threshold"])


def test_the_threshold_is_exceeded_on_about_a_tenth_of_the_baseline_days():
    df = _synthetic("1991-01-01", "2020-12-31", seed=6, trend=0.0)
    clim = mhw_hobday.compute_climatology(df, 1991, 2020)
    daily = mhw_hobday.detect_events(df, clim)[1]
    share = daily["exceeds_threshold"].mean()
    assert 0.085 < share < 0.115, share           # the pipeline's own detection gives 13% on Livorno's 10-year baseline


def test_the_threshold_is_above_the_climatology_every_day():
    df = _synthetic("1991-01-01", "2020-12-31", seed=7)
    clim = mhw_hobday.compute_climatology(df, 1991, 2020)
    assert (clim["threshold_smooth"] > clim["climatology_smooth"]).all() and (clim["baseline_diff"] >= 0.01).all()


# --- the rules of the events, which the pipeline's detection does not follow -------------------------------

def _spiked(runs):
    """Noise around a constant, with `runs` = [(start index, length)] of days far above it (a deterministic seed)."""
    days = pd.date_range("1990-01-01", "2019-12-31", freq="D")
    temp = 15.0 + np.random.default_rng(11).normal(0, 0.5, len(days))
    for s, n in runs:
        temp[s:s + n] += 20
    return pd.DataFrame({"Datetime": days, "Temperature": temp})


def _near_spikes(events):
    """The events around the spikes (index 10000 is 2017-05-19); the noise alone makes none that long, and this says so if it does."""
    return [e for e in events if e["end_date"] >= pd.Timestamp("2017-05-01") and e["start_date"] <= pd.Timestamp("2017-06-15")]


def test_two_short_exceedances_with_a_one_day_gap_are_not_an_event(legacy):
    df = _spiked([(10000, 3), (10004, 3)])           # 3 + gap of 1 + 3: neither run reaches 5 days
    events, _, mhw = _compare(legacy, df, 1990, 2019)
    assert _near_spikes(events) == []
    pipeline_events = mhw_detection.detect_events(df, mhw_detection.compute_climatology(df, 1990, 2019))[0]
    assert len(_near_spikes(pipeline_events)) == 1    # the pipeline's tolerates the gap inside a run, so 3 + 1 + 3 days are an event for it


def test_two_events_of_five_days_with_a_gap_of_two_are_one_event_and_with_a_gap_of_three_two(legacy):
    joined = _near_spikes(_compare(legacy, _spiked([(10000, 5), (10007, 5)]), 1990, 2019)[0])
    apart = _near_spikes(_compare(legacy, _spiked([(10000, 5), (10008, 5)]), 1990, 2019)[0])
    assert len(joined) == 1 and joined[0]["duration_days"] == 12 and len(apart) == 2


def test_a_run_of_exactly_four_days_is_not_an_event(legacy):
    assert _near_spikes(_compare(legacy, _spiked([(10000, 4)]), 1990, 2019)[0]) == []
    assert len(_near_spikes(_compare(legacy, _spiked([(10000, 5)]), 1990, 2019)[0])) == 1


def _flat_clim(threshold=10.0, seasonal=9.0):
    return pd.DataFrame({"doy": np.arange(1, 367), "climatology": seasonal, "threshold": threshold,
                         "climatology_smooth": seasonal, "threshold_smooth": threshold, "baseline_diff": threshold - seasonal})


def test_a_temperature_equal_to_the_threshold_does_not_exceed_it():
    days = pd.date_range("2010-01-01", periods=30, freq="D")
    temp = np.full(30, 8.0)
    temp[5:12] = 10.0                                        # seven days exactly at the threshold
    temp[20:27] = 10.5                                       # seven days above it
    events, daily = mhw_hobday.detect_events(pd.DataFrame({"Datetime": days, "Temperature": temp}), _flat_clim())
    assert [(e["start_date"].day, e["end_date"].day) for e in events] == [(21, 27)]
    assert int(daily["exceeds_threshold"].sum()) == 7


def test_a_series_with_a_gap_or_a_missing_value_is_refused_not_filled():
    days = pd.date_range("2000-01-01", "2012-12-31", freq="D")
    df = pd.DataFrame({"Datetime": days, "Temperature": 15.0 + np.random.default_rng(0).normal(0, 1, len(days))})
    for bad in (df.drop(index=[100, 101]), df.assign(Temperature=df["Temperature"].where(df.index != 50))):
        with pytest.raises(ValueError):
            mhw_hobday.compute_climatology(bad, 2001, 2010)
        with pytest.raises(ValueError):
            mhw_hobday.detect_events(bad, _flat_clim())


def test_a_baseline_outside_the_series_is_refused():
    df = _synthetic("2000-01-01", "2005-12-31")
    with pytest.raises(ValueError):
        mhw_hobday.compute_climatology(df, 2010, 2015)


def test_the_metrics_are_computed_from_the_references_threshold_and_climatology(legacy):
    df = pd.read_csv(FIXTURE / "sst_daily.csv", parse_dates=["Datetime"])[["Datetime", "Temperature"]]
    mhw, rc = _reference(legacy, df, 2003, 2012)
    clim = mhw_hobday.compute_climatology(df, 2003, 2012)
    events, daily = mhw_hobday.detect_events(df, clim)
    temp = df["Temperature"].to_numpy()
    above = np.maximum(temp - rc["thresh"], 0)
    np.testing.assert_allclose(daily["intensity"], above, rtol=0, atol=1e-9)
    np.testing.assert_allclose(daily["anomaly"], temp - rc["seas"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(daily["intensity_norm"], np.maximum((temp - rc["seas"]) / (rc["thresh"] - rc["seas"]), 0), rtol=0, atol=1e-9)
    for e, a, b in zip(events, mhw["index_start"], mhw["index_end"]):
        assert e["duration_days"] == b - a + 1
        assert e["intensity_max"] == pytest.approx(above[a:b + 1].max(), abs=1e-9)
        assert e["intensity_cumulative"] == pytest.approx(above[a:b + 1].sum(), abs=1e-8)
    assert [e["event_id"] for e in events] == list(range(1, len(events) + 1))
    assert (daily.loc[daily["in_event"], "event_id"].astype(int).between(1, len(events))).all() and daily.loc[~daily["in_event"], "event_id"].isna().all()


def test_the_scale_of_the_category_is_never_below_a_hundredth_of_a_degree():
    days = pd.date_range("2000-01-01", "2012-12-31", freq="D")
    df = pd.DataFrame({"Datetime": days, "Temperature": 15.0})              # threshold = climatology
    clim = mhw_hobday.compute_climatology(df, 2001, 2010)
    assert (clim["baseline_diff"] == 0.01).all()


def test_a_baseline_that_does_not_cover_every_day_of_the_year_is_refused():
    df = _synthetic("2001-01-01", "2001-06-30", seed=3)
    with pytest.raises(ValueError, match="every day"):
        mhw_hobday.compute_climatology(df, 2001, 2001)


def test_a_baseline_outside_the_years_of_the_series_is_refused_not_cut_short(legacy):
    df = _synthetic("2003-01-01", "2012-12-31", seed=3)
    for y0, y1 in [(2005, 2015), (2000, 2008), (2008, 2005)]:
        with pytest.raises(ValueError, match="not inside"):
            mhw_hobday.compute_climatology(df, y0, y1)


# --- the daily table is the one the builder aggregates ---------------------------------------------------------

def test_the_daily_table_has_the_columns_the_monthly_aggregation_needs():
    df = _synthetic("2003-01-01", "2012-12-31", seed=8, spells=[("2008-07-01", "2008-07-12", 4.0)])
    clim = mhw_hobday.compute_climatology(df, 2003, 2012)
    events, daily = mhw_hobday.detect_events(df, clim)
    monthly = mhw_detection.to_monthly(daily)
    assert {"mhw_days", "mhw_peak_intensity", "mhw_cum_intensity", "anomaly_mean"} <= set(monthly.columns)
    assert monthly["mhw_days"].sum() == daily["in_event"].sum() == sum(e["duration_days"] for e in events)


# --- the builder uses it ----------------------------------------------------------------------------------------

def test_the_builder_builds_its_heatwave_catalogue_with_the_hobday_climatology():
    sst = pd.read_csv(FIXTURE / "sst_daily.csv", parse_dates=["Datetime"])
    months = pd.date_range("2003-01-01", "2026-06-01", freq="MS")
    response = pd.DataFrame({"Datetime": months, "value": 1.0, "ci_upper": np.nan, "ci_lower": np.nan, "n": 1})
    ds = dataset_builder.build_dataset(sst=sst, response=response, imputation=None, baseline=(2003, 2012))
    clim = mhw_hobday.compute_climatology(sst[["Datetime", "Temperature"]], 2003, 2012)
    events, daily = mhw_hobday.detect_events(sst[["Datetime", "Temperature"]], clim)
    assert len(ds.mhw_events) == len(events) == 78
    assert ds.mhw_events["start_date"].tolist() == [e["start_date"] for e in events]
    fixture = pd.read_csv(FIXTURE / "mhw_events.csv")
    assert len(fixture) == 109 and len(ds.mhw_events) != len(fixture)       # NOT the pipeline's catalogue: Livorno's numbers stay on the old detection
    assert ds.coverage["mhw"]["method"].startswith("Hobday et al. 2016") and ds.coverage["mhw"]["baseline"] == [2003, 2012]


# --- guards -------------------------------------------------------------------------------------------------------

def test_the_module_reads_no_state_of_the_loaded_study():
    names = set()
    for node in ast.walk(ast.parse((PKG / "mhw_hobday.py").read_text())):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    assert not names & {"DATA", "RESULTS", "SPLIT_DATE", "WINDOWS", "CLIM_START", "CLIM_END", "MHW_CLIM_START", "MHW_CLIM_END", "config", "environ"}


def test_the_pipelines_own_detection_is_unchanged_by_this_piece():
    # the golden master holds the numbers; this holds the reason: the parameters Livorno's published results come from
    assert (mhw_detection.PCTILE, mhw_detection.MIN_DAYS, mhw_detection.MAX_GAP) == (90, 5, 2)
    src = (PKG / "mhw_detection.py").read_text()
    assert "rolling(11, center=True, min_periods=1)" in src and 'baseline.groupby("doy")' in src
