"""The comparison and the expectations of scripts/verify_download.py (the manual check of M1.2 against the
real service), tested on synthetic outputs: the check must flag what the milestone says it must flag."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_download.py"
spec = importlib.util.spec_from_file_location("verify_download", SCRIPT)
vd = importlib.util.module_from_spec(spec); spec.loader.exec_module(vd)

PISA = ("error: site 'Pisa' (lat 43.72, lon 10.45): the grid cells around it are land in the product (sst_daily); "
        "the nearest sea cell is at lat 43.7, lon 10.3, about 11 km away. Move the site to the sea, or to that cell.")
SEA = ("error: site 'Mar Nero' (lat 43.0, lon 31.0): no sea cell of the product (sst_daily) within about 111 km. The point is "
       "inland, or in a sea the product does not cover although it lies inside the bounding box of its domain; the "
       "product's land/sea mask cannot tell the two apart (both are non-sea cells), so this check does not choose.")


def _write(path, days, values):
    pd.DataFrame({"Datetime": pd.to_datetime(days), "Temperature": values}).to_csv(path, index=False)


def test_compare_uses_only_days_with_a_value_in_both(tmp_path):
    _write(tmp_path / "a.csv", ["2024-01-01", "2024-01-02", "2024-01-03"], [10.0, None, 12.0])
    _write(tmp_path / "b.csv", ["2024-01-01", "2024-01-02", "2024-01-03"], [10.00004, 99.0, 12.0])
    r = vd.compare_series(tmp_path / "a.csv", tmp_path / "b.csv")
    assert r["common_days"] == 2 and r["max_abs_diff"] == pytest.approx(4e-5)


def test_distance_is_read_from_the_message():
    assert vd.nearest_sea_km(PISA) == 11
    assert vd.nearest_sea_km(SEA) is None


def _ok_results():
    return {"livorno": {"returncode": 0, "stderr": "", "stdout": "", "seconds": 60.0},
            "pisa": {"returncode": 1, "stderr": PISA, "stdout": "", "seconds": 5.0},
            "mar_nero": {"returncode": 1, "stderr": SEA, "stdout": "", "seconds": 5.0}}


GOOD = {"common_days": 8000, "max_abs_diff": 2e-5}


def test_all_three_as_expected():
    assert vd.check(_ok_results(), GOOD) == []


def test_a_livorno_difference_above_the_limit_is_flagged():
    assert any("Livorno" in f for f in vd.check(_ok_results(), {"common_days": 8000, "max_abs_diff": 3e-4}))


def test_the_black_sea_returning_data_is_flagged():
    r = _ok_results(); r["mar_nero"].update(returncode=0, stderr="")
    assert any("RETURNED DATA" in f for f in vd.check(r, GOOD))


def test_pisa_with_the_wrong_distance_or_the_wrong_kind_of_refusal_is_flagged():
    r = _ok_results(); r["pisa"]["stderr"] = PISA.replace("about 11 km", "about 70 km")
    assert any("Pisa" in f for f in vd.check(r, GOOD))
    r = _ok_results(); r["pisa"]["stderr"] = SEA
    assert any("not as a land cell" in f for f in vd.check(r, GOOD))


def test_the_black_sea_with_a_message_that_chooses_is_flagged():
    r = _ok_results(); r["mar_nero"]["stderr"] = "error: it is outside the area the product covers"
    assert any("another message" in f for f in vd.check(r, GOOD))


# --- the monthly variables against data/env_copernicus.csv (M1.8) ------------------------------------------------------

CASES = vd.MONTHLY_CASES


def _monthly(path, months, values, column="pH"):
    pd.DataFrame({"Datetime": pd.to_datetime(months), column: values}).to_csv(path, index=False)


def test_the_comparison_can_name_the_column_and_counts_the_months_that_differ(tmp_path):
    _monthly(tmp_path / "a.csv", ["2024-01-01", "2024-02-01", "2024-03-01"], [8.10, 8.11, None])
    _monthly(tmp_path / "b.csv", ["2024-01-01", "2024-02-01", "2024-03-01"], [8.10, 8.1102, 8.2])
    r = vd.compare_series(tmp_path / "a.csv", tmp_path / "b.csv", column="pH")
    assert r["common_days"] == 2 and r["max_abs_diff"] == pytest.approx(2e-4) and r["over_1e-4"] == 1


def test_the_cases_are_the_catalogue_variables_at_livorno_from_the_start_of_the_series():
    assert list(CASES) == ["ph", "salinity"]
    for key, case in CASES.items():
        assert (case["lat"], case["lon"]) == (43.4278, 10.3956) and case["start"] == "2003-01-01" and case["name"] == f"{key}_livorno"
    assert (CASES["ph"]["variable"], CASES["ph"]["column"]) == ("ph_monthly", "pH")
    assert (CASES["salinity"]["variable"], CASES["salinity"]["column"]) == ("salinity_monthly", "Salinity")


def test_the_gross_limits_are_below_the_median_change_from_one_month_to_the_next_so_a_month_of_shift_is_seen():
    # measured on data/env_copernicus.csv, 2026-11-21: the median |month-to-month change| is 0.0276 for the pH and 0.0549 for the salinity
    assert CASES["ph"]["max_diff"] == 0.02 and CASES["salinity"]["max_diff"] == 0.04 and vd.MIN_COMMON_MONTHS == 12


@pytest.mark.parametrize("key", ["ph", "salinity"])
def test_a_difference_just_below_the_gross_limit_passes_and_at_the_limit_fails_and_too_few_common_months_fail(key):
    case, ok = CASES[key], {"returncode": 0, "stderr": ""}
    good = {"common_days": 270, "max_abs_diff": case["max_diff"] - 1e-4, "over_1e-4": 3}
    assert vd.check_monthly(case, ok, good) == []
    assert vd.check_monthly(case, ok, {**good, "max_abs_diff": case["max_diff"]})
    assert vd.check_monthly(case, ok, {**good, "common_days": 12}) == [] and vd.check_monthly(case, ok, {**good, "common_days": 11})
    assert any(case["column"] in f and "failed" in f for f in vd.check_monthly(case, {"returncode": 1, "stderr": "boom"}, None))
    assert any("no common" in f for f in vd.check_monthly(case, ok, {"common_days": 0, "max_abs_diff": None, "over_1e-4": 0}))


def _fake_run_case(values):
    def run_case(case, out):
        folder = out / case["name"]; folder.mkdir(parents=True, exist_ok=True)
        ok = {"returncode": 0, "stderr": "", "stdout": "", "seconds": 1.0}
        if case["name"] == "livorno":
            _write(folder / "sst_daily.csv", ["2024-01-01"], [10.0])
            return ok
        for c in CASES.values():
            if case["name"] == c["name"]:
                _monthly(folder / f"{c['variable']}.csv", pd.date_range("2003-01-01", periods=24, freq="MS"), values[c["column"]], column=c["column"])
                return ok
        return {"returncode": 1, "stderr": PISA if case["name"] == "pisa" else SEA, "stdout": "", "seconds": 1.0}
    return run_case


def _main(tmp_path, monkeypatch, values, third=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    months = pd.date_range("2003-01-01", periods=24, freq="MS")
    _write(tmp_path / "sst.csv", ["2024-01-01"], [10.0])
    pd.DataFrame({"Datetime": months, "pH": [8.10] * 24, "Salinity": [38.0] * 24}).to_csv(tmp_path / "env.csv", index=False)
    monkeypatch.setattr(vd, "run_case", _fake_run_case(values))
    args = [str(tmp_path / "out"), str(tmp_path / "sst.csv")] + ([str(tmp_path / "env.csv")] if third else [])
    return vd.main(args), tmp_path / "out"


def test_main_downloads_every_monthly_variable_when_given_the_reference_and_a_gross_difference_fails_the_check(tmp_path, monkeypatch):
    import json
    code, out = _main(tmp_path / "a", monkeypatch, {"pH": [8.10] * 24, "Salinity": [38.0] * 24})
    assert code == 0 and "Monthly pH of Livorno" in (out / "summary.md").read_text() and "Monthly Salinity of Livorno" in (out / "summary.md").read_text()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["monthly"]["ph"]["common_days"] == 24 and summary["monthly"]["salinity"]["common_days"] == 24
    code, out2 = _main(tmp_path / "b", monkeypatch, {"pH": [8.10] * 24, "Salinity": [38.5] * 24})       # only the salinity is off
    failures = json.loads((out2 / "summary.json").read_text())["failures"]
    assert code == 1 and any("Salinity" in f for f in failures) and not any("pH" in f for f in failures)


def test_without_the_third_argument_no_monthly_variable_is_downloaded_and_the_old_check_is_unchanged(tmp_path, monkeypatch):
    code, out = _main(tmp_path, monkeypatch, {"pH": [8.10] * 24, "Salinity": [38.0] * 24}, third=False)
    assert code == 0 and not (out / "ph_livorno").exists() and not (out / "salinity_livorno").exists() and "Monthly" not in (out / "summary.md").read_text()
