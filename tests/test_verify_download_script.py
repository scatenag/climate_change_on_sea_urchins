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
    assert list(CASES) == ["ph", "salinity", "o2", "co2"]
    for key, case in CASES.items():
        assert (case["lat"], case["lon"]) == (43.4278, 10.3956) and case["start"] == "2003-01-01" and case["name"] == f"{key}_livorno"
    assert (CASES["ph"]["variable"], CASES["ph"]["column"]) == ("ph_monthly", "pH")
    assert (CASES["salinity"]["variable"], CASES["salinity"]["column"]) == ("salinity_monthly", "Salinity")
    assert (CASES["o2"]["variable"], CASES["o2"]["column"]) == ("o2_monthly", "O2")
    assert (CASES["co2"]["variable"], CASES["co2"]["column"]) == ("co2_monthly", "CO2")


def test_the_limits_are_one_strict_until_2023_05_and_one_wide_per_variable_from_2023_06():
    # the wide limits are the largest |month-to-month change| of data/env_copernicus.csv, measured 2026-11-28
    # (pH 0.0751, salinity 0.7401, oxygen 22.99, CO2 99.58)
    assert vd.MAX_DIFF_UNTIL_SPLIT == 1e-3 and vd.SPLIT_MONTH == "2023-06-01" and vd.MIN_COMMON_MONTHS == 12
    assert {k: c["max_diff_after"] for k, c in CASES.items()} == {"ph": 0.075, "salinity": 0.74, "o2": 23.0, "co2": 100.0}


def test_the_comparison_is_split_with_may_2023_in_the_first_interval_and_june_2023_in_the_second(tmp_path):
    months = ["2023-04-01", "2023-05-01", "2023-06-01", "2023-07-01", "2023-08-01"]
    _monthly(tmp_path / "a.csv", months, [8.10, 8.10, 8.10, 8.10, 8.10])
    _monthly(tmp_path / "b.csv", months, [8.10, 8.1005, 8.13, 8.10, 8.10])
    r = vd.compare_intervals(tmp_path / "a.csv", tmp_path / "b.csv", "pH")
    assert list(r) == ["until_2023_05", "from_2023_06"]
    first, second = r["until_2023_05"], r["from_2023_06"]
    assert (first["common_days"], second["common_days"]) == (2, 3)
    assert first["max_abs_diff"] == pytest.approx(5e-4) and second["max_abs_diff"] == pytest.approx(0.03)
    assert (first["over_1e-4"], second["over_1e-4"]) == (1, 1)


def _interval(n=40, diff=0.0, over=0):
    return {"common_days": n, "max_abs_diff": diff if n else None, "over_1e-4": over,
            "downloaded_days": 285, "reference_days": 285, "downloaded_end": "2026-09-01"}


@pytest.mark.parametrize("key", ["ph", "salinity", "o2", "co2"])
def test_each_interval_has_its_own_limit_and_its_own_failure(key):
    case, ok = CASES[key], {"returncode": 0, "stderr": ""}
    wide = case["max_diff_after"]
    good = {"until_2023_05": _interval(245, 0.99e-3, 3), "from_2023_06": _interval(40, wide * 0.99, 40)}
    assert vd.check_monthly(case, ok, good) == []
    f = vd.check_monthly(case, ok, {**good, "until_2023_05": _interval(245, 1e-3)})
    assert len(f) == 1 and case["column"] in f[0] and "until 2023-05" in f[0]
    f = vd.check_monthly(case, ok, {**good, "from_2023_06": _interval(40, wide)})
    assert len(f) == 1 and case["column"] in f[0] and "from 2023-06" in f[0]
    both = vd.check_monthly(case, ok, {"until_2023_05": _interval(245, 1.0e3), "from_2023_06": _interval(40, 1.0e3)})
    assert len(both) == 2 and "until 2023-05" in both[0] and "from 2023-06" in both[1]


@pytest.mark.parametrize("interval,label", [("until_2023_05", "until 2023-05"), ("from_2023_06", "from 2023-06")])
def test_each_interval_needs_twelve_common_months(interval, label):
    case, ok = CASES["ph"], {"returncode": 0, "stderr": ""}
    good = {"until_2023_05": _interval(), "from_2023_06": _interval()}
    assert vd.check_monthly(case, ok, {**good, interval: _interval(12)}) == []
    for n in (11, 0):
        f = vd.check_monthly(case, ok, {**good, interval: _interval(n)})
        assert len(f) == 1 and label in f[0] and f"only {n} common months" in f[0]


def test_a_monthly_download_that_failed_or_gave_no_series_is_a_failure():
    case = CASES["ph"]
    assert any("pH: download failed" in f for f in vd.check_monthly(case, {"returncode": 1, "stderr": "boom"}, None))
    assert any("pH: no series" in f for f in vd.check_monthly(case, {"returncode": 0, "stderr": ""}, None))


MONTHS = pd.date_range("2021-06-01", periods=48, freq="MS")     # 24 months up to 2023-05 and 24 from 2023-06
REFERENCE = {"pH": 8.10, "Salinity": 38.0, "O2": 240.0, "CO2": 400.0}


def _values(**changes):
    """The reference values on every month; changes[column] = (index of the month, amount added there)."""
    out = {c: [v] * 48 for c, v in REFERENCE.items()}
    for column, (i, amount) in changes.items():
        out[column][i] += amount
    return out


def _fake_run_case(values):
    def run_case(case, out):
        folder = out / case["name"]; folder.mkdir(parents=True, exist_ok=True)
        ok = {"returncode": 0, "stderr": "", "stdout": "", "seconds": 1.0}
        if case["name"] == "livorno":
            _write(folder / "sst_daily.csv", ["2024-01-01"], [10.0])
            return ok
        for c in CASES.values():
            if case["name"] == c["name"]:
                _monthly(folder / f"{c['variable']}.csv", MONTHS, values[c["column"]], column=c["column"])
                return {**ok, "seconds": {"pH": 2.0, "Salinity": 3.0, "O2": 4.0, "CO2": 5.0}[c["column"]]}
        return {"returncode": 1, "stderr": PISA if case["name"] == "pisa" else SEA, "stdout": "", "seconds": 1.0}
    return run_case


def _inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write(tmp_path / "sst.csv", ["2024-01-01"], [10.0])
    pd.DataFrame({"Datetime": MONTHS, **REFERENCE}).to_csv(tmp_path / "env.csv", index=False)


def _main(tmp_path, monkeypatch, values, third=True):
    _inputs(tmp_path)
    monkeypatch.setattr(vd, "run_case", _fake_run_case(values))
    args = [str(tmp_path / "out"), str(tmp_path / "sst.csv")] + ([str(tmp_path / "env.csv")] if third else [])
    return vd.main(args), tmp_path / "out"


def test_main_downloads_every_monthly_variable_and_reports_each_interval(tmp_path, monkeypatch):
    import json
    # the differences the owner measured after May 2023 (oxygen 4.6, CO2 20.3) are inside the wide limits
    code, out = _main(tmp_path, monkeypatch, _values(O2=(30, 4.6), CO2=(40, 20.3)))
    assert code == 0
    text = (out / "summary.md").read_text()
    for column, seconds in (("pH", 2.0), ("Salinity", 3.0), ("O2", 4.0), ("CO2", 5.0)):
        assert f"### Monthly {column} of Livorno against `data/env_copernicus.csv` (M1.8)" in text
        assert f"- download time: **{seconds} s**" in text
    for wide in ("0.075", "0.74", "23.0", "100.0"):
        assert f"| {wide} |" in text
    section = text[text.index("### Monthly O2"):text.index("### Monthly CO2")]
    assert "| until 2023-05 | 24 | 0.0 | 0.001 | 0 |" in section
    assert "| from 2023-06 | 24 | " in section and " | 23.0 | 1 |" in section
    monthly = json.loads((out / "summary.json").read_text())["monthly"]
    assert list(monthly) == ["ph", "salinity", "o2", "co2"]
    for key in monthly:
        assert [monthly[key][i]["common_days"] for i in ("until_2023_05", "from_2023_06")] == [24, 24]
    assert monthly["o2"]["from_2023_06"]["max_abs_diff"] == pytest.approx(4.6)
    assert monthly["o2"]["until_2023_05"]["max_abs_diff"] == 0.0


@pytest.mark.parametrize("change,column,failed,silent", [
    ({"Salinity": (23, 2e-3)}, "Salinity", "until 2023-05", "from 2023-06"),       # the last month of the first interval
    ({"Salinity": (24, 0.8)}, "Salinity", "from 2023-06", "until 2023-05"),        # the first month of the second
])
def test_main_fails_naming_the_variable_and_the_interval(tmp_path, monkeypatch, change, column, failed, silent):
    import json
    code, out = _main(tmp_path, monkeypatch, _values(**change))
    failures = json.loads((out / "summary.json").read_text())["failures"]
    assert code == 1 and len(failures) == 1 and failures[0].startswith(f"{column}, {failed}:") and silent not in failures[0]
    assert f"- FAIL: {failures[0]}" in (out / "summary.md").read_text()


def test_without_the_third_argument_no_monthly_variable_is_downloaded_and_the_old_check_is_unchanged(tmp_path, monkeypatch):
    code, out = _main(tmp_path, monkeypatch, _values(), third=False)
    assert code == 0 and "Monthly" not in (out / "summary.md").read_text()
    assert not any((out / c["name"]).exists() for c in CASES.values())


def test_a_monthly_download_that_failed_is_said_in_the_summary_and_fails_the_check(tmp_path, monkeypatch):
    import json
    _inputs(tmp_path)
    good = _fake_run_case(_values())
    monkeypatch.setattr(vd, "run_case", lambda case, out: {"returncode": 1, "stderr": "boom", "stdout": "", "seconds": 2.0} if case["name"] == "salinity_livorno" else good(case, out))
    code = vd.main([str(tmp_path / "out"), str(tmp_path / "sst.csv"), str(tmp_path / "env.csv")])
    text = (tmp_path / "out" / "summary.md").read_text()
    assert code == 1 and "Monthly Salinity of Livorno (M1.8)" in text and "no comparison (download failed)" in text
    assert any("Salinity: download failed" in f for f in json.loads((tmp_path / "out" / "summary.json").read_text())["failures"])
