"""
Shared fixtures for tests/test_paper_values.py and tests/test_golden_master.py,
both built on the same frozen data snapshot in tests/fixtures/paper_mpb_2026/data/
-- no second fixture is created (see docs/adr/0004).

No analysis module is modified: each module's own ROOT/DATA/RESULTS
module-level path constants (imported from common.py) are monkeypatched for
the duration of a fixture, exactly the way any other caller would set them,
then restored. Frozen fixture in, throwaway output out; the real project
tree (data/, results/) is untouched either way -- golden_pipeline_results
below actively verifies that last claim (see its own docstring for why).
"""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "paper_mpb_2026"


def _snapshot_tree(path: Path) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) per file under `path`, relative paths as keys.
    Cheap enough to call before/after a ~2 min pipeline run; changes mtime
    even when a write produces byte-identical content (e.g. re-running a
    deterministic detection algorithm on unchanged input, as
    mhw_detection.py's own import-time-path bug did silently, every run,
    until fixed -- see golden_pipeline_results below)."""
    if not path.is_dir():
        return {}
    return {
        str(p.relative_to(path)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in path.rglob("*") if p.is_file()
    }


@pytest.fixture(scope="session")
def paper_results(tmp_path_factory):
    """Runs the four manuscript-value modules only (negative_control,
    period_split, thermal_legacy, mhw_annual_changepoint) -- see
    test_paper_values.py. Output goes to an unrelated throwaway dir (these
    four modules never write outside RESULTS)."""
    from climate_change_on_sea_urchins import (
        common, mhw_annual_changepoint, negative_control, period_split,
        thermal_legacy,
    )

    results_dir = tmp_path_factory.mktemp("paper_values_results")
    fixture_data = FIXTURE_ROOT / "data"

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(common, "ROOT", FIXTURE_ROOT)
        mp.setattr(common, "DATA", fixture_data)
        for mod in (negative_control, period_split, thermal_legacy, mhw_annual_changepoint):
            if hasattr(mod, "ROOT"):
                mp.setattr(mod, "ROOT", FIXTURE_ROOT)
            if hasattr(mod, "DATA"):
                mp.setattr(mod, "DATA", fixture_data)
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
    script, run against the copied tree, also writes there.

    --- Why this fixture snapshots and re-checks the REAL data/ and
    results/ directories, not just the copied ones ---

    Until 2026-09-25, mhw_detection.py computed its input/output paths
    (SST_PATH, OUT_EVENTS, OUT_MONTHLY, OUT_ANNUAL) once at IMPORT time from
    common.ROOT, into separate module-level constants. Patching
    mhw_detection.ROOT below (as this fixture already did) had no effect on
    those already-bound paths: every golden-master run silently read the
    REAL project's data/sst_daily.csv and OVERWROTE the REAL project's
    data/mhw_events.csv, data/mhw_monthly.csv, data/mhw_annual.csv, while
    the rest of the pipeline (correctly redirected) used the frozen
    fixture's own static copies of those three files instead -- so the
    downstream numbers this test suite actually checked were never wrong,
    but the golden master's own claim of exercising mhw_detection.py
    against the frozen fixture was false, and every run wrote to the
    developer's real data/ for no reason.

    Nobody noticed for weeks because MHW detection is deterministic and
    data/sst_daily.csv did not change in that window (verified by hand,
    2026-09-25: byte-identical to the fixture's own copy) -- so `git status`
    stayed clean every time. A change to sst_daily.csv on either side, or a
    single `git diff`, would have caught it; neither happened. This is the
    same class of "stayed green while checking the wrong thing" failure
    this project has now hit four times (see CHANGELOG/STATO.md). Fixed by
    routing all data/ access through common.py's DATA constant and
    functions, resolved fresh at call time, never cached into a
    module-level constant elsewhere (see mhw_detection.py's own docstring,
    and tests/test_data_boundary.py, which regression-tests the fix
    directly rather than relying on this snapshot alone).

    This snapshot+recheck is the general-purpose guard: it fails loudly, by
    itself, if ANY module in the pipeline -- this one or a future one --
    writes into the real data/ or results/ during a golden-master run,
    regardless of the specific mechanism.
    """
    from climate_change_on_sea_urchins import common, pipeline

    tmp_root = tmp_path_factory.mktemp("golden_pipeline")
    tmp_data = tmp_root / "data"
    shutil.copytree(FIXTURE_ROOT / "data", tmp_data)
    results_dir = tmp_root / "results"
    results_dir.mkdir()

    real_data_before    = _snapshot_tree(REPO_ROOT / "data")
    real_results_before = _snapshot_tree(REPO_ROOT / "results")

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(common, "ROOT", tmp_root)
        mp.setattr(common, "DATA", tmp_data)
        for _label, mod in pipeline._MODULES:
            if hasattr(mod, "ROOT"):
                mp.setattr(mod, "ROOT", tmp_root)
            if hasattr(mod, "DATA"):
                mp.setattr(mod, "DATA", tmp_data)
            if hasattr(mod, "RESULTS"):
                mp.setattr(mod, "RESULTS", results_dir)

        pipeline.main()
    finally:
        mp.undo()

    assert _snapshot_tree(REPO_ROOT / "data") == real_data_before, (
        "golden_pipeline_results wrote into the REAL data/ directory -- the fixture's "
        "DATA redirection did not reach every module. See this fixture's own docstring "
        "for the bug this check exists to catch (it happened once already, silently)."
    )
    assert _snapshot_tree(REPO_ROOT / "results") == real_results_before, (
        "golden_pipeline_results wrote into the REAL results/ directory -- the fixture's "
        "RESULTS redirection did not reach every module."
    )

    r_script = shutil.which("Rscript")
    if r_script is not None:
        scripts_dir = tmp_root / "scripts"
        scripts_dir.mkdir()
        shutil.copy(REPO_ROOT / "scripts" / "mhw_lag_analysis.R", scripts_dir)
        subprocess.run(
            [r_script, str(scripts_dir / "mhw_lag_analysis.R"), str(tmp_data), str(results_dir)],
            check=True, capture_output=True, text=True,
        )
        # else: leave the 3 dlnm_*.csv files absent -- test_golden_master.py
        # skips their comparison when Rscript/dlnm aren't available, matching
        # the project's "R is an optional dependency" stance.

    return results_dir
