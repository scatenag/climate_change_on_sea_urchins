"""
The dashboard reads EC50 live from the source sheet, but must prepare it
exactly as the update job does: the same monthly aggregation, the same
imputation passes, the same merge with the environmental and MHW data
(common.load_data(ec50_monthly=...)), and the same correlations
(correlations.compute_matrices). Two preparations of the same quantity were
why the app's live values differed from the job's. Ported from main.
"""
import ast
import shutil
from pathlib import Path

import pandas as pd
import pytest

from climate_change_on_sea_urchins import common

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "paper_mpb_2026"
DASHBOARD = Path(common.__file__).parent / "dashboard.py"
SCRIPTS = Path(__file__).parents[1] / "scripts"


@pytest.fixture
def root(monkeypatch, tmp_path):
    shutil.copytree(FIXTURE_ROOT / "data", tmp_path / "data")
    monkeypatch.setattr(common, "ROOT", tmp_path)
    return tmp_path


def test_live_path_reproduces_the_job_when_the_sheet_is_unchanged(root):
    job = common.load_data()
    live = common.load_data(ec50_monthly=common.load_ec50_monthly())
    for got, expected in zip(live, job):
        pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True))


def test_live_path_includes_a_month_the_job_has_not_seen(root):
    sheet = common.load_ec50_monthly()
    new_month = sheet["Datetime"].max() + pd.DateOffset(months=3)
    extra = pd.DataFrame({"Datetime": [new_month], "EC50": [20.0], "EC50_ci_upper": [22.0],
                          "EC50_ci_lower": [18.0], "EC50_n": [1]})
    df_full, df_real, *_ = common.load_data(ec50_monthly=pd.concat([sheet, extra]))
    row = df_real[df_real["Datetime"] == new_month]
    assert len(row) == 1 and row["EC50"].iloc[0] == 20.0
    assert df_full["Datetime"].max() >= new_month


def _defined(path):
    return {n.name for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)}


def test_each_preparation_step_is_one_implementation():
    assert not _defined(DASHBOARD) & {"_aggregate_ec50", "load_env_data",
                                      "_extract_trends", "_spearman_matrix"}
    for script in ("fetch_ec50.py", "build_dataset.py"):
        assert not _defined(SCRIPTS / script) & {"aggregate_monthly", "impute_ec50"}, script


def test_dashboard_uses_the_jobs_functions():
    tree = ast.parse(DASHBOARD.read_text())
    calls = {getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert {"load_data", "aggregate_monthly", "compute_matrices"} <= calls
    assert ".rolling(window=12, min_periods=3" not in DASHBOARD.read_text(), \
        "the dashboard imputes EC50 itself"
