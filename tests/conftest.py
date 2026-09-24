"""
Shared fixtures for tests/test_paper_values.py and tests/test_golden_master.py,
both built on the same frozen data snapshot in tests/fixtures/paper_mpb_2026/data/
-- no second fixture is created (see docs/adr/0004).

No analysis module is modified: each module's own ROOT/RESULTS module-level
path constants (imported from common.py) are monkeypatched for the duration
of a fixture, exactly the way any other caller would set them, then
restored. Frozen fixture in, throwaway output out; the real project tree
(data/, results/) is untouched either way.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "paper_mpb_2026"


@pytest.fixture(scope="session")
def paper_results(tmp_path_factory):
    """Runs the four manuscript-value modules only (negative_control,
    period_split, thermal_legacy, mhw_annual_changepoint) -- see
    test_paper_values.py. Output goes to an unrelated throwaway dir (these
    four modules never read RESULTS.parent, so it doesn't need to be nested
    under a data/ sibling the way golden_pipeline_results below does)."""
    from climate_change_on_sea_urchins import (
        common, mhw_annual_changepoint, negative_control, period_split,
        thermal_legacy,
    )

    results_dir = tmp_path_factory.mktemp("paper_values_results")

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(common, "ROOT", FIXTURE_ROOT)
        for mod in (negative_control, period_split, thermal_legacy, mhw_annual_changepoint):
            if hasattr(mod, "ROOT"):
                mp.setattr(mod, "ROOT", FIXTURE_ROOT)
            mp.setattr(mod, "RESULTS", results_dir)

        negative_control.run()
        period_split.run()
        thermal_legacy.run()
        mhw_annual_changepoint.run()
    finally:
        mp.undo()

    return results_dir


@pytest.fixture(scope="session")
def golden_pipeline_results(tmp_path_factory):
    """Runs the full Python pipeline (pipeline.main(), all 16 modules) plus
    the R DLNM script (if Rscript/dlnm are available) against the same
    frozen fixture data -- see tests/test_golden_master.py.

    mhw_detection.run() (the pipeline's first step) regenerates
    data/mhw_events.csv/mhw_monthly.csv/mhw_annual.csv from
    data/sst_daily.csv, so unlike paper_results above this needs a *copy*
    of the fixture's data/, not the committed fixture read in place -- the
    committed fixture must stay untouched between runs.

    RESULTS is nested as ROOT/"results" to mirror the real layout; the R
    script, run against the copied tree, also writes there by default.
    """
    from climate_change_on_sea_urchins import common, pipeline

    tmp_root = tmp_path_factory.mktemp("golden_pipeline")
    shutil.copytree(FIXTURE_ROOT / "data", tmp_root / "data")
    results_dir = tmp_root / "results"
    results_dir.mkdir()

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(common, "ROOT", tmp_root)
        for _label, mod in pipeline._MODULES:
            if hasattr(mod, "ROOT"):
                mp.setattr(mod, "ROOT", tmp_root)
            if hasattr(mod, "RESULTS"):
                mp.setattr(mod, "RESULTS", results_dir)

        pipeline.main()
    finally:
        mp.undo()

    r_script = shutil.which("Rscript")
    if r_script is not None:
        scripts_dir = tmp_root / "scripts"
        scripts_dir.mkdir()
        shutil.copy(REPO_ROOT / "scripts" / "mhw_lag_analysis.R", scripts_dir)
        subprocess.run(
            [r_script, str(scripts_dir / "mhw_lag_analysis.R")],
            check=True, capture_output=True, text=True,
        )
        # else: leave the 3 dlnm_*.csv files absent -- test_golden_master.py
        # skips their comparison when Rscript/dlnm aren't available, matching
        # the project's "R is an optional dependency" stance.

    return results_dir
