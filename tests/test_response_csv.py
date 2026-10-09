"""The CSV response source (M1.3): a response series in a file the user provides, read strictly.

The format is declared in the study (delimiter, decimal separator, date format), never guessed: a
European CSV (';' and decimal commas, 31/01/2020) must not be read wrongly in silence, so a file that does
not match its declaration is refused, naming the line, the column, what was found and what the file looks
like. A missing measurement stays missing. One test per class of error.
"""
import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from climate_change_on_sea_urchins import common
from climate_change_on_sea_urchins.response_csv import (
    ResponseCsvError, read_response, read_response_text, suggest_format,
)
from climate_change_on_sea_urchins.study_spec import StudySpec, StudySpecError, load_study

FIXTURE = Path(__file__).parent / "fixtures" / "paper_mpb_2026" / "data"


def _response(**source_overrides):
    """A ResponseSpec (format 2, csv source, per-trial values aggregated by month)."""
    source = {"type": "csv", "file": "response.csv", "temporal_resolution": "month", "granularity": "per_trial",
              "column_map": {"date": "date", "value": "value"}}
    aggregation = source_overrides.pop("_aggregation", {"period": "month", "method": "mean", "count_field": "n_assays"})
    column_map = source_overrides.pop("column_map", None)
    source.update(source_overrides)
    if column_map:
        source["column_map"] = column_map
    d = {"format_version": 2, "id": "s", "description": "d", "sites": [{"id": "x", "lat": 43.4, "lon": 10.4, "name": "n", "bbox_delta": 0.1}],
         "mhw_climatology": {"baseline_start_year": 2003, "baseline_end_year": 2012},
         "environment": [{"catalog_id": "sst_daily"}],
         "responses": [{"id": "r", "label": "R", "nature": "index", "adverse_direction": "decrease", "unit": "1",
                        "source": source, **({"aggregation": aggregation} if aggregation else {})}]}
    return StudySpec.model_validate(d).responses[0]


CSV = textwrap.dedent("""\
    date,value
    2020-01-05,10
    2020-01-20,14
    2020-03-02,8.5
    """)


# --- the declared format -------------------------------------------------------------------------

def test_defaults_are_iso_comma_and_point():
    src = _response().source
    assert (src.delimiter, src.decimal, src.date_format) == (",", ".", "%Y-%m-%d")


def test_a_european_file_is_read_when_it_is_declared():
    text = "data;valore\n31/01/2020;12,5\n15/01/2020;7,5\n02/03/2020;8,25\n"
    r = _response(delimiter=";", decimal=",", date_format="%d/%m/%Y", column_map={"date": "data", "value": "valore"})
    data = read_response_text(text, r)
    assert data.series["Datetime"].tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-01")]
    assert data.series["value"].tolist() == pytest.approx([10.0, 8.25])
    assert data.series["n"].tolist() == [2, 1]


@pytest.mark.parametrize("bad", [
    {"delimiter": "x"}, {"decimal": ";"}, {"date_format": "%Q"}, {"date_format": "%d/%m"}, {"date_format": "$(rm)"},
    {"temporal_resolution": "day", "date_format": "%Y-%m"},
])
def test_invalid_format_declarations_are_refused(bad):
    with pytest.raises((StudySpecError, ValueError)):
        _response(**bad)


def test_the_delimiter_cannot_also_be_the_decimal_separator():
    with pytest.raises((StudySpecError, ValueError), match="decimal"):
        _response(delimiter=",", decimal=",")


# --- one test per class of error ------------------------------------------------------------------

def _problems(text, **kw):
    with pytest.raises(ResponseCsvError) as e:
        read_response_text(text, _response(**kw))
    return str(e.value)


def test_empty_file():
    assert "empty" in _problems("")


def test_header_only_has_no_data():
    assert "no data rows" in _problems("date,value\n")


def test_missing_column_names_what_the_file_has():
    msg = _problems("when,value\n2020-01-01,1\n")
    assert "'date'" in msg and "when" in msg and "value" in msg


def test_duplicate_header_names():
    assert "twice" in _problems("date,value,value\n2020-01-01,1,2\n")


def test_semicolon_file_declared_as_comma_is_recognised_not_misread():
    msg = _problems("date;value\n2020-01-01;1\n2020-01-02;2\n")
    assert "';'" in msg and "delimiter" in msg


def test_wrong_number_of_fields_names_the_line():
    msg = _problems("date,value\n2020-01-01,1\n2020-01-02,2,3\n")
    assert "line 3" in msg and "3 fields" in msg


def test_unparsable_date_names_line_column_value_and_what_the_dates_look_like():
    msg = _problems("date,value\n31/01/2020,1\n15/02/2020,2\n")
    assert "line 2" in msg and "'date'" in msg and "'31/01/2020'" in msg and "%d/%m/%Y" in msg


def test_ambiguous_dates_are_said_to_be_ambiguous():
    fmt = suggest_format("date,value\n01/02/2020,1\n03/04/2020,2\n", date_column="date", value_column="value")
    assert set(fmt["date_formats"]) == {"%d/%m/%Y", "%m/%d/%Y"} and fmt["date_ambiguous"] is True


def test_a_date_with_a_time_is_refused_not_truncated():
    assert "line 2" in _problems("date,value\n2020-01-05 10:30,1\n")


def test_non_numeric_value():
    msg = _problems("date,value\n2020-01-05,abc\n")
    assert "line 2" in msg and "'value'" in msg and "'abc'" in msg and "not a number" in msg


def test_decimal_comma_in_a_point_file_is_recognised():
    msg = _problems('date,value\n2020-01-05,"12,5"\n')
    assert "line 2" in msg and "decimal" in msg and "','" in msg


def test_decimal_point_in_a_comma_file_is_recognised():
    msg = _problems("date;value\n2020-01-05;12.5\n", delimiter=";", decimal=",")
    assert "line 2" in msg and "decimal" in msg


@pytest.mark.parametrize("token", ["nan", "inf", "-inf", "Infinity", "1e999"])
def test_not_a_finite_number(token):
    assert "finite" in _problems(f"date,value\n2020-01-05,{token}\n")


def test_not_utf8():
    with pytest.raises(ResponseCsvError, match="UTF-8"):
        read_response_text("date,value\n2020-01-05,1\n".encode("latin-1").replace(b"2020", b"\xe92020"), _response())


def test_a_utf8_bom_is_accepted():
    data = read_response_text(b"\xef\xbb\xbfdate,value\n2020-01-05,1\n", _response())
    assert len(data.series) == 1


def test_too_large_a_file_is_refused():
    big = "date,value\n" + "2020-01-05,1\n" * 300_000
    assert "rows" in _problems(big)


def test_problems_are_collected_and_capped():
    rows = "".join(f"2020-01-05,x{i}\n" for i in range(50))
    msg = _problems("date,value\n" + rows)
    assert "50 problems" in msg and msg.count("line ") <= 21


# --- missing measurements and the confidence interval ---------------------------------------------

def test_an_empty_value_stays_missing_and_is_counted():
    data = read_response_text("date,value\n2020-01-05,\n2020-01-20,4\n2020-02-03,\n", _response())
    assert data.notes["missing_values"] == 2
    s = data.series.set_index("Datetime")
    assert s.loc[pd.Timestamp("2020-01-01"), "value"] == 4 and s.loc[pd.Timestamp("2020-01-01"), "n"] == 1
    assert np.isnan(s.loc[pd.Timestamp("2020-02-01"), "value"]) and s.loc[pd.Timestamp("2020-02-01"), "n"] == 0


def test_months_without_any_row_are_not_invented():
    data = read_response_text(CSV, _response())
    assert data.series["Datetime"].tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-01")]


def test_confidence_interval_columns_come_in_the_series():
    text = "date,value,lo,hi\n2020-01-05,10,8,13\n2020-01-20,14,12,15\n"
    r = _response(column_map={"date": "date", "value": "value", "ci_low": "lo", "ci_high": "hi"})
    s = read_response_text(text, r).series
    assert s["ci_upper"].notna().all() and (s["ci_upper"] >= s["value"]).all() and (s["ci_lower"] <= s["value"]).all()


def test_a_value_outside_its_own_interval_is_refused():
    r = dict(column_map={"date": "date", "value": "value", "ci_low": "lo", "ci_high": "hi"})
    msg = _problems("date,value,lo,hi\n2020-01-05,10,11,13\n", **r)
    assert "line 2" in msg and "interval" in msg


def test_without_an_interval_there_is_none_made_up():
    s = read_response_text(CSV, _response()).series
    assert s["ci_upper"].isna().all() and s["ci_lower"].isna().all()


def test_control_columns_are_kept_with_the_observation():
    r = _response(control_columns=["c1", "c2"])
    data = read_response_text("date,value,c1,c2\n2020-01-05,10,3,\n", r)
    assert data.observations[["c1", "c2"]].iloc[0].isna().tolist() == [False, True]


# --- aggregated input -----------------------------------------------------------------------------

def test_already_aggregated_rows_pass_through():
    r = _response(granularity="aggregated", _aggregation=None)
    data = read_response_text("date,value\n2020-01-15,5\n2020-02-15,6\n", r)
    assert data.series["value"].tolist() == [5, 6]
    assert data.series["Datetime"].tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]


def test_two_aggregated_rows_in_the_same_period_are_refused():
    r = dict(granularity="aggregated", _aggregation=None)
    msg = _problems("date,value\n2020-01-05,5\n2020-01-20,6\n", **r)
    assert "line 2" in msg and "line 3" in msg and "same" in msg


# --- the same series as Livorno's ----------------------------------------------------------------

def test_livorno_trials_exported_and_read_back_give_the_same_monthly_series(tmp_path):
    raw = pd.read_csv(FIXTURE / "ec50_raw.csv", parse_dates=["Datetime"]).sort_values(["Datetime", "ID"])
    expected = pd.read_csv(FIXTURE / "ec50_sheets.csv", parse_dates=["Datetime"])
    for delimiter, decimal, fmt, render in [(",", ".", "%Y-%m-%d", lambda d: d.strftime("%Y-%m-%d")),
                                            (";", ",", "%d/%m/%Y", lambda d: d.strftime("%d/%m/%Y"))]:
        out = pd.DataFrame({"when": raw["Datetime"].map(render), "ec50": raw["EC50"]})
        path = tmp_path / "response.csv"
        out.to_csv(path, index=False, sep=delimiter, decimal=decimal)
        r = _response(delimiter=delimiter, decimal=decimal, date_format=fmt, column_map={"date": "when", "value": "ec50"})
        got = read_response(path, r).series
        assert got["Datetime"].tolist() == expected["Datetime"].tolist()
        np.testing.assert_allclose(got["value"], expected["EC50"], rtol=0, atol=1e-12)
        assert got["n"].tolist() == expected["EC50_n"].tolist()


def test_the_file_is_read_through_the_boundary_by_name_only(tmp_path):
    (tmp_path / "response.csv").write_text(CSV)
    data = read_response(tmp_path / "response.csv", _response())
    assert len(data.series) == 2


# --- the aggregation is the one implementation ---------------------------------------------------

def _old_aggregate_monthly(raw):
    """Frozen copy of common.aggregate_monthly as it was before the neutral core was extracted."""
    raw = raw.copy()
    raw["DATE"] = pd.to_datetime(raw["DATE"], dayfirst=False)
    raw["Datetime"] = raw["DATE"].dt.to_period("M").dt.to_timestamp()
    raw["hw_upper"] = raw["UL"] - raw["EC50"]
    raw["hw_lower"] = raw["EC50"] - raw["LL"]
    agg = raw.groupby("Datetime").agg(EC50=("EC50", "mean"), EC50_std=("EC50", "std"), EC50_n=("EC50", "count"),
                                       mean_hw_upper=("hw_upper", "mean"), mean_hw_lower=("hw_lower", "mean")).reset_index()
    agg["se"] = agg["EC50_std"] / np.sqrt(agg["EC50_n"])
    agg["EC50_ci_upper"] = agg["EC50"] + np.maximum(agg["mean_hw_upper"], 1.96 * agg["se"].fillna(0))
    agg["EC50_ci_lower"] = agg["EC50"] - np.maximum(agg["mean_hw_lower"], 1.96 * agg["se"].fillna(0))
    return agg[["Datetime", "EC50", "EC50_ci_upper", "EC50_ci_lower", "EC50_n"]].sort_values("Datetime").reset_index(drop=True)


def _synthetic_trials(seed=0):
    rng = np.random.default_rng(seed)
    n = 300
    ec = rng.uniform(5, 60, n)
    dates = pd.to_datetime("2003-01-01") + pd.to_timedelta(rng.integers(0, 365 * 20, n), unit="D")
    raw = pd.DataFrame({"DATE": dates.strftime("%d-%b-%y"), "EC50": ec, "UL": ec + rng.uniform(1, 9, n), "LL": ec - rng.uniform(1, 9, n)})
    raw.loc[rng.choice(n, 20, replace=False), ["UL", "LL"]] = np.nan
    raw.loc[rng.choice(n, 10, replace=False), "EC50"] = np.nan
    return raw


def test_aggregate_monthly_is_unchanged_by_extracting_the_neutral_core():
    raw = _synthetic_trials()
    pd.testing.assert_frame_equal(common.aggregate_monthly(raw), _old_aggregate_monthly(raw), check_exact=True)


def test_the_neutral_core_gives_the_same_numbers_under_neutral_names():
    raw = _synthetic_trials(1)
    old = _old_aggregate_monthly(raw)
    new = common.aggregate_period(raw.assign(DATE=pd.to_datetime(raw["DATE"], format="%d-%b-%y")),
                                  date="DATE", value="EC50", ci_low="LL", ci_high="UL")
    assert list(new.columns) == ["Datetime", "value", "ci_upper", "ci_lower", "n"]
    for a, b in zip(["EC50", "EC50_ci_upper", "EC50_ci_lower", "EC50_n"], ["value", "ci_upper", "ci_lower", "n"]):
        pd.testing.assert_series_equal(old[a].reset_index(drop=True), new[b].reset_index(drop=True), check_names=False, check_exact=True)


def test_through_the_csv_with_intervals_the_series_equals_livornos_aggregation(tmp_path):
    raw = _synthetic_trials(2).dropna(subset=["EC50", "UL", "LL"])
    raw["DATE"] = pd.to_datetime(raw["DATE"], format="%d-%b-%y")
    csv = raw.assign(DATE=raw["DATE"].dt.strftime("%Y-%m-%d"))
    path = tmp_path / "r.csv"; csv.to_csv(path, index=False)
    r = _response(column_map={"date": "DATE", "value": "EC50", "ci_low": "LL", "ci_high": "UL"})
    got = read_response(path, r).series
    ref = common.aggregate_monthly(raw.assign(DATE=raw["DATE"].dt.strftime("%d-%b-%y")))
    np.testing.assert_allclose(got["value"], ref["EC50"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["ci_upper"], ref["EC50_ci_upper"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["ci_lower"], ref["EC50_ci_lower"], rtol=0, atol=1e-12)
    assert got["n"].tolist() == ref["EC50_n"].tolist()
