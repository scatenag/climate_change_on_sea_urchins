"""
Shared fixture for tests/test_paper_values.py: runs the four manuscript-
value modules (negative_control, period_split, thermal_legacy,
mhw_annual_changepoint) once per test session against the frozen data
snapshot in tests/fixtures/paper_mpb_2026/data/, writing their output to a
throwaway pytest tmp directory -- never data/ or results/.

No analysis module is modified: each module's own ROOT/RESULTS module-level
path constants (imported from common.py) are monkeypatched for the
duration of this fixture, exactly the way any other caller would set them,
then restored. Frozen fixture in, throwaway output out; the real project
tree is untouched either way.
"""
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "paper_mpb_2026"


@pytest.fixture(scope="session")
def paper_results(tmp_path_factory):
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
