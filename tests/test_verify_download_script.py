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
