"""ADR-0014: the climatology baseline proposed to a new study (1991-2020), the checks on a baseline the user changes,
and where the download starts. The coverage of the product comes from the catalogue, never from a date in the code."""
import ast
import datetime as dt
import re
from pathlib import Path

import pytest

from climate_change_on_sea_urchins import baseline
from climate_change_on_sea_urchins.download import Coverage

PKG = Path(baseline.__file__).parent


def cov(start, end):
    return Coverage(start=dt.date.fromisoformat(start), end=dt.date.fromisoformat(end), version="x")


def test_the_default_is_proposed_when_the_product_covers_it():
    s = baseline.suggest_baseline(cov("1987-01-01", "2026-08-31"))
    assert (s.start, s.end) == (1991, 2020) and s.note is None


def test_a_product_that_starts_later_gives_the_first_thirty_covered_years():
    s = baseline.suggest_baseline(cov("1993-01-01", "2026-08-31"))
    assert (s.start, s.end) == (1993, 2022)
    assert "1991" in s.note and "does not cover" in s.note


def test_a_product_that_starts_mid_year_begins_at_its_first_full_year():
    s = baseline.suggest_baseline(cov("1993-03-01", "2026-08-31"))
    assert (s.start, s.end) == (1994, 2023)


def test_a_product_shorter_than_thirty_years_gives_the_whole_covered_period_with_a_warning():
    s = baseline.suggest_baseline(cov("2003-01-01", "2026-08-31"))
    assert (s.start, s.end) == (2003, 2025)             # 2026 is not complete
    assert "shorter than 30 years" in s.note


def test_a_product_that_ends_before_2020_is_not_given_years_it_does_not_have():
    s = baseline.suggest_baseline(cov("1987-01-01", "2018-06-30"))       # covers 1987-2017 in full: 31 years
    assert (s.start, s.end) == (1987, 2016) and "does not cover" in s.note


def test_a_baseline_outside_the_covered_years_is_an_error():
    errors, _ = baseline.check_baseline(1980, 2005, cov("1987-01-01", "2026-08-31"))
    assert any("before" in e for e in errors)
    errors, _ = baseline.check_baseline(2000, 2030, cov("1987-01-01", "2026-08-31"))
    assert any("after" in e for e in errors)
    errors, _ = baseline.check_baseline(2010, 2005, cov("1987-01-01", "2026-08-31"))
    assert any("start" in e and "end" in e for e in errors)


def test_a_baseline_shorter_than_ten_years_is_an_error_because_the_threshold_would_be_the_data_themselves():
    for start, end in ((2003, 2003), (2003, 2004), (2003, 2011)):
        errors, _ = baseline.check_baseline(start, end, cov("1987-01-01", "2026-08-31"))
        assert any("at least 10" in e for e in errors), (start, end)
    errors, _ = baseline.check_baseline(2003, 2012, cov("1987-01-01", "2026-08-31"))
    assert not errors                                 # Livorno's own baseline is ten years


def test_a_short_baseline_and_one_that_reaches_the_last_year_are_warned_not_refused():
    errors, warnings = baseline.check_baseline(2003, 2012, cov("1987-01-01", "2026-08-31"))
    assert not errors and any("15" in w for w in warnings)
    errors, warnings = baseline.check_baseline(1995, 2025, cov("1987-01-01", "2026-08-31"))
    assert not errors and any("last" in w for w in warnings)
    errors, warnings = baseline.check_baseline(1991, 2020, cov("1987-01-01", "2026-08-31"))
    assert not errors and not warnings


def test_the_download_starts_at_the_baseline_or_earlier_for_the_lags():
    c = cov("1987-01-01", "2026-08-31")
    assert baseline.download_start(1991, c)[0] == dt.date(1991, 1, 1)
    assert baseline.download_start(1991, c, first_measurement_year=2003)[0] == dt.date(1991, 1, 1)
    start, note = baseline.download_start(1991, c, first_measurement_year=1988, max_lag_months=12)
    assert start == dt.date(1987, 1, 1)                  # one year before the first measurement
    start, note = baseline.download_start(1991, c, first_measurement_year=1985, max_lag_months=12)
    assert start == dt.date(1987, 1, 1) and "first day" in note   # cut at the product's start, said


def test_no_coverage_date_is_written_in_the_code_except_the_default():
    tree = ast.parse((PKG / "baseline.py").read_text())
    years = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and 1900 <= node.value <= 2100:
            years.append(node.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r"\b(19|20)\d\d\b", node.value):
            assert len(node.value) > 100, node.value        # a year in a string only inside a docstring, never as data
    assert sorted(years) == [1991, 2020]          # DEFAULT_BASELINE and nothing else
