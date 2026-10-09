"""The dataset builder (M1.4): from the daily SST and a monthly response series to the monthly table and
the heatwave catalogue a study is analysed on.

The monthly temperature is the mean of the daily SST. A month with any missing day is missing (decision of
2026-10-03): a partial mean is distorted by the seasonal cycle, and it would fall on the months the lags use
most. The rule fires on an incomplete last month and on an incomplete first month (a period that starts
mid-month), and the months it excludes are in the coverage reported with the results. The response is
imputed only as the study declares (absent unless declared), and every number comes from the functions
Livorno's pipeline uses: the heatwave catalogue from the fixture's SST is the fixture's catalogue, and the
response of the sheet, with Livorno's declared imputation, is the one data/ and load_data() hold.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_change_on_sea_urchins import common
from climate_change_on_sea_urchins.dataset_builder import (
    DatasetError, build_dataset, monthly_temperature, write_dataset,
)
from climate_change_on_sea_urchins.study_spec import ImputationSpec

FIXTURE = Path(__file__).parent / "fixtures" / "paper_mpb_2026" / "data"
BASELINE = (2003, 2004)          # the synthetic SST is short; Livorno's tests use its own (2003, 2012)


def _sst(start, end, base=15.0, amp=8.0, seed=0, hot=()):
    days = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    t = np.asarray(base + amp * np.sin(2 * np.pi * (days.dayofyear - 110) / 365.25) + rng.normal(0, 0.3, len(days)))
    for a, b, k in hot:                                            # heat spells: +k degrees from a to b
        t[(days >= a) & (days <= b)] += k
    return pd.DataFrame({"Datetime": days, "Temperature": t})


def _response(months, values, n=None):
    idx = pd.to_datetime(months)
    return pd.DataFrame({"Datetime": idx, "value": values, "ci_upper": np.nan, "ci_lower": np.nan,
                         "n": n if n is not None else [1 if not np.isnan(v) else 0 for v in values]})


IMP = ImputationSpec(method="centered_rolling_mean", window_months=12, min_periods=3, passes=1)


# --- the monthly temperature and the rule of incomplete months -------------------------------------

def test_a_complete_month_is_the_mean_of_its_days():
    sst = _sst("2004-01-01", "2004-03-31")
    t, report = monthly_temperature(sst)
    jan = sst[sst.Datetime.dt.month == 1].Temperature.mean()
    assert t.set_index("Datetime").loc[pd.Timestamp("2004-01-01"), "Temperature"] == pytest.approx(jan)
    assert report["incomplete_months"] == []


def test_an_incomplete_last_month_is_missing_and_reported():
    t, report = monthly_temperature(_sst("2004-01-01", "2004-03-10"))
    s = t.set_index("Datetime")["Temperature"]
    assert s[pd.Timestamp("2004-01-01")] == s[pd.Timestamp("2004-01-01")]          # present
    assert np.isnan(s[pd.Timestamp("2004-03-01")])
    assert report["incomplete_months"] == [{"month": "2004-03", "days_present": 10, "days_expected": 31}]


def test_an_incomplete_first_month_is_missing_and_reported():
    t, report = monthly_temperature(_sst("2004-01-15", "2004-03-31"))
    s = t.set_index("Datetime")["Temperature"]
    assert np.isnan(s[pd.Timestamp("2004-01-01")]) and not np.isnan(s[pd.Timestamp("2004-02-01")])
    assert report["incomplete_months"] == [{"month": "2004-01", "days_present": 17, "days_expected": 31}]


def test_one_missing_day_in_the_middle_makes_the_month_missing():
    sst = _sst("2004-01-01", "2004-03-31")
    sst.loc[sst.Datetime == "2004-02-10", "Temperature"] = np.nan
    t, report = monthly_temperature(sst)
    assert np.isnan(t.set_index("Datetime").loc[pd.Timestamp("2004-02-01"), "Temperature"])
    assert [m["month"] for m in report["incomplete_months"]] == ["2004-02"]
    assert report["incomplete_months"][0]["days_present"] == 28


def test_both_ends_incomplete_at_once():
    _, report = monthly_temperature(_sst("2004-01-15", "2004-03-10"))
    assert [m["month"] for m in report["incomplete_months"]] == ["2004-01", "2004-03"]


# --- building --------------------------------------------------------------------------------------

def _build(sst, resp, imputation=IMP, baseline=BASELINE):
    return build_dataset(sst=sst, response=resp, imputation=imputation, baseline=baseline)


def _long_sst(end="2008-12-31", **kw):
    return _sst("2003-01-01", end, hot=[("2005-07-01", "2005-07-20", 4.0)], **kw)


def test_the_grid_covers_both_series_and_nothing_is_invented():
    resp = _response(["2004-03-01", "2004-05-01"], [10.0, 14.0])
    ds = _build(_long_sst(), resp, imputation=None)
    m = ds.monthly
    assert m["Datetime"].iloc[0] == pd.Timestamp("2003-01-01") and m["Datetime"].iloc[-1] == pd.Timestamp("2008-12-01")
    assert m["response"].notna().sum() == 2                                     # no imputation declared: nothing filled
    assert m.loc[m["response"].isna(), "response_imputed"].all()
    assert not m.loc[m["response"].notna(), "response_imputed"].any()


def test_the_declared_imputation_fills_as_declared_and_is_flagged():
    vals = [10.0, np.nan, 12.0, 13.0, np.nan, 15.0, 16.0, 17.0]
    resp = _response(pd.date_range("2004-01-01", periods=8, freq="MS"), vals)
    ds = _build(_long_sst(), resp, imputation=IMP)
    m = ds.monthly.set_index("Datetime")
    assert m.loc["2004-02-01", "response_imputed"] and not np.isnan(m.loc["2004-02-01", "response"])
    assert not m.loc["2004-01-01", "response_imputed"] and m.loc["2004-01-01", "response"] == 10.0
    assert np.isnan(m.loc["2004-02-01", "response_ci_upper"])                   # no interval for an imputed month
    assert np.isnan(m.loc["2004-02-01", "response_n"]) and m.loc["2004-01-01", "response_n"] == 1   # a month with no measurement has no count


def test_two_passes_fill_what_one_pass_leaves():
    vals = [10.0, np.nan, np.nan, np.nan, 20.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 30.0]
    resp = _response(pd.date_range("2004-01-01", periods=12, freq="MS"), vals)
    one = _build(_long_sst(), resp, imputation=IMP).monthly
    two = _build(_long_sst(), resp, imputation=IMP.model_copy(update={"passes": 2})).monthly
    assert one["response"].isna().sum() > two["response"].isna().sum()


def test_the_declared_fill_reaches_only_months_near_observations():
    # The centered rolling mean fills within half a window of the observations, edges included (as
    # Livorno's does, docs/adr/0000 item 8); far from any observation the month stays missing.
    resp = _response(pd.date_range("2004-01-01", periods=6, freq="MS"), [1, 2, 3, 4, 5, 6.0])
    m = _build(_long_sst(), resp).monthly.set_index("Datetime")
    assert np.isnan(m.loc["2008-12-01", "response"]) and np.isnan(m.loc["2003-01-01", "response"])
    assert m.loc["2004-06-01", "response"] == 6.0


def test_months_with_an_incomplete_sst_are_missing_in_the_table_and_in_the_catalogue():
    sst = _sst("2003-01-01", "2008-02-14", hot=[("2005-07-01", "2005-07-20", 4.0), ("2008-02-01", "2008-02-14", 5.0)])
    ds = _build(sst, _response(["2004-03-01"], [10.0]), imputation=None)
    assert np.isnan(ds.monthly.set_index("Datetime").loc["2008-02-01", "Temperature"])
    assert pd.Timestamp("2008-02-01") not in set(ds.mhw_monthly["Datetime"])
    assert [m["month"] for m in ds.coverage["temperature"]["incomplete_months"]] == ["2008-02"]
    assert [m["month"] for m in ds.coverage["mhw"]["months_excluded_incomplete_sst"]] == ["2008-02"]


def test_an_incomplete_first_month_is_reported_in_the_coverage():
    sst = _sst("2002-12-15", "2008-12-31", hot=[("2005-07-01", "2005-07-20", 4.0)])
    ds = _build(sst, _response(["2004-03-01"], [10.0]), imputation=None)
    assert [m["month"] for m in ds.coverage["temperature"]["incomplete_months"]] == ["2002-12"]


def test_a_heat_spell_is_found():
    ds = _build(_long_sst(), _response(["2004-03-01"], [10.0]), imputation=None)
    assert len(ds.mhw_events) >= 1
    assert (ds.mhw_events["start_date"].astype(str).str[:4] == "2005").any()


def test_a_gap_inside_the_sst_stops_the_detection_with_the_reason():
    sst = _long_sst()
    sst.loc[sst.Datetime.between("2005-02-01", "2005-02-05"), "Temperature"] = np.nan
    with pytest.raises(DatasetError, match=r"2005-02-01.*missing day"):
        _build(sst, _response(["2004-03-01"], [10.0]))


def test_a_baseline_the_sst_does_not_cover_is_refused():
    with pytest.raises(DatasetError, match="baseline"):
        _build(_sst("2005-01-01", "2008-12-31"), _response(["2006-03-01"], [10.0]), baseline=(2003, 2012))


def test_a_baseline_that_ends_after_the_sst_is_refused_too():
    with pytest.raises(DatasetError, match="baseline"):
        _build(_sst("2003-01-01", "2008-12-31"), _response(["2006-03-01"], [10.0]), baseline=(2003, 2010))


def test_an_empty_response_is_refused():
    with pytest.raises(DatasetError, match="response"):
        _build(_long_sst(), _response([], []))


# --- coverage and files ----------------------------------------------------------------------------

def test_coverage_counts_the_response():
    resp = _response(pd.date_range("2004-01-01", periods=8, freq="MS"), [10.0, np.nan, 12.0, 13.0, np.nan, 15.0, 16.0, 17.0])
    c = _build(_long_sst(), resp).coverage["response"]
    assert c["months_with_value"] == 6 and c["months_imputed"] >= 2
    assert c["first"] == "2004-01" and c["last"] == "2004-08"


def test_written_files_and_manifest(tmp_path):
    ds = _build(_long_sst(), _response(["2004-03-01"], [10.0]))
    paths = write_dataset(ds, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dataset.manifest.json", "mhw_annual.csv", "mhw_events.csv", "mhw_monthly.csv", "monthly.csv"]
    m = json.loads(paths["manifest"].read_text())
    assert m["coverage"]["temperature"]["rule"].startswith("a month with any missing")
    assert m["imputation"]["passes"] == 1 and m["baseline"] == [2003, 2004]
    assert set(m["files"]) == {"monthly.csv", "mhw_events.csv", "mhw_monthly.csv", "mhw_annual.csv"}
    again = pd.read_csv(paths["monthly"], parse_dates=["Datetime"])
    assert list(again.columns) == ["Datetime", "Temperature", "response", "response_imputed", "response_ci_upper", "response_ci_lower", "response_n"]


def test_no_imputation_declared_is_written_as_such(tmp_path):
    ds = _build(_long_sst(), _response(["2004-03-01"], [10.0]), imputation=None)
    m = json.loads(write_dataset(ds, tmp_path)["manifest"].read_text())
    assert m["imputation"] is None


# --- Livorno: the same numbers as the pipeline's own ----------------------------------------------

@pytest.fixture(scope="module")
def livorno():
    sst = pd.read_csv(FIXTURE / "sst_daily.csv", parse_dates=["Datetime"])
    sheet = pd.read_csv(FIXTURE / "ec50_sheets.csv", parse_dates=["Datetime"]).rename(
        columns={"EC50": "value", "EC50_ci_upper": "ci_upper", "EC50_ci_lower": "ci_lower", "EC50_n": "n"})
    return sst, sheet


def test_livornos_heatwave_catalogue_is_the_fixtures(livorno):
    sst, sheet = livorno
    ds = _build(sst, sheet, imputation=None, baseline=(2003, 2012))
    for ours, theirs, kw in [(ds.mhw_events, "mhw_events.csv", dict(parse_dates=["start_date", "end_date", "peak_date"])),
                             (ds.mhw_monthly, "mhw_monthly.csv", dict(parse_dates=["Datetime"])),
                             (ds.mhw_annual, "mhw_annual.csv", {})]:
        ref = pd.read_csv(FIXTURE / theirs, **kw)
        pd.testing.assert_frame_equal(ours.reset_index(drop=True), ref, check_exact=False, rtol=1e-12, atol=1e-12)


def test_livornos_response_with_one_declared_pass_is_data_ec50_ci(livorno):
    sst, sheet = livorno
    ds = _build(sst, sheet, imputation=IMP.model_copy(update={"passes": 1}), baseline=(2003, 2012))
    ref = pd.read_csv(FIXTURE / "data_ec50_ci.csv", parse_dates=["Datetime"]).set_index("Datetime")
    # Livorno's grid also runs one month past the SST (July 2026, from the environment table): compare on
    # the builder's months, all of which must be in the reference.
    got = ds.monthly.set_index("Datetime")
    assert got.index.isin(ref.index).all() and len(ref) - len(got) <= 1
    ref = ref.loc[got.index]
    np.testing.assert_allclose(got["response"], ref["EC50"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["response_ci_upper"], ref["EC50_ci_upper"], rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(got["response_ci_lower"], ref["EC50_ci_lower"], rtol=0, atol=1e-12, equal_nan=True)
    assert got["response_imputed"].tolist() == ref["EC50_imputed"].tolist()


def test_livornos_declared_imputation_two_passes_is_what_load_data_holds(livorno, monkeypatch):
    sst, sheet = livorno
    monkeypatch.setattr(common, "DATA", FIXTURE)
    monkeypatch.setattr(common, "ROOT", FIXTURE.parent)
    df_full = common.load_data()[0].set_index("Datetime")
    ds = _build(sst, sheet, imputation=IMP.model_copy(update={"passes": 2}), baseline=(2003, 2012))
    got = ds.monthly.set_index("Datetime")["response"]
    assert got.index.isin(df_full.index).all()
    np.testing.assert_allclose(got, df_full.loc[got.index, common.RESPONSE_COL], rtol=0, atol=1e-12, equal_nan=True)


def test_impute_response_is_unchanged_by_the_parametrised_version():
    v = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, 6.0, 7.0, np.nan, 9.0, 10.0, 11.0, 12.0, np.nan, 14.0])
    old = v.fillna(v.rolling(window=12, min_periods=3, center=True).mean())
    pd.testing.assert_series_equal(common.impute_response(v), old, check_exact=True)
    pd.testing.assert_series_equal(common.impute_series(v, 12, 3), old, check_exact=True)
