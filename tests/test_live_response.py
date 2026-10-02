"""
The dashboard reads the response live from the source sheet, but must
prepare it exactly as the update job does: the same monthly aggregation,
the same imputation passes, the same merge with the environmental and MHW
data (common.load_data(ec50_monthly=...)). Two preparations of the same
quantity were why the app's live values differed from the job's.
"""
import ast
import shutil
from pathlib import Path

import pandas as pd
import pytest

from climate_change_on_sea_urchins import common

FIXTURE_DATA = Path(__file__).parent / "fixtures" / "paper_mpb_2026" / "data"
DASHBOARD = Path(common.__file__).parent / "dashboard.py"


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DATA, data)
    monkeypatch.setattr(common, "DATA", data)
    return data


def _sheet_monthly(data):
    return pd.read_csv(data / "ec50_sheets.csv", parse_dates=["Datetime"])


def test_live_path_reproduces_the_job_when_the_sheet_is_unchanged(data_dir):
    job = common.load_data()
    live = common.load_data(ec50_monthly=_sheet_monthly(data_dir))
    for got, expected in zip(live, job):
        pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True))


def test_live_path_includes_a_month_the_job_has_not_seen(data_dir):
    sheet = _sheet_monthly(data_dir)
    new_month = sheet["Datetime"].max() + pd.DateOffset(months=3)
    extra = pd.DataFrame({"Datetime": [new_month], "EC50": [20.0], "EC50_ci_upper": [22.0],
                          "EC50_ci_lower": [18.0], "EC50_n": [1]})
    df_full, df_real, *_ = common.load_data(ec50_monthly=pd.concat([sheet, extra]))
    row = df_real[df_real["Datetime"] == new_month]
    assert len(row) == 1 and row[common.RESPONSE_COL].iloc[0] == 20.0
    assert df_full["Datetime"].max() >= new_month


def test_aggregation_is_one_implementation():
    # the monthly aggregation of the sheet lives in common.py only
    from climate_change_on_sea_urchins.common import aggregate_monthly  # noqa: F401
    tree = ast.parse(DASHBOARD.read_text())
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not defined & {"_aggregate_ec50", "load_env_data"}
    for script in ("fetch_ec50.py", "build_dataset.py"):
        tree = ast.parse((Path(__file__).parents[1] / "scripts" / script).read_text())
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert not defined & {"aggregate_monthly", "impute_ec50"}, script


def test_dashboard_prepares_data_with_load_data():
    tree = ast.parse(DASHBOARD.read_text())
    calls = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "load_data" in calls
    src = DASHBOARD.read_text()
    assert ".rolling(window=12, min_periods=3" not in src, "the dashboard imputes the response itself"


def test_dashboard_correlations_are_the_jobs():
    # one implementation of the trend correlations: correlations.compute_matrices
    tree = ast.parse(DASHBOARD.read_text())
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not defined & {"_extract_trends", "_spearman_matrix"}
    calls = {getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "compute_matrices" in calls
