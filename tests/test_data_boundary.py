"""
Regression tests for the "module-level path bound at import time, not
call time" bug class -- found in mhw_detection.py on 2026-09-25 (see
tests/conftest.py::golden_pipeline_results' docstring for the full story)
and the reason this file exists as its own dedicated guard, not folded
into test_golden_master.py.

The shape of the bug: a module computed SST_PATH/OUT_EVENTS/OUT_MONTHLY/
OUT_ANNUAL once, at import time, from common.ROOT. Redirecting
common.ROOT/common.DATA afterward (via monkeypatch, exactly how
tests/conftest.py's fixtures redirect every module to a frozen fixture
instead of the real project data/) had NO effect on those already-bound
paths -- the module kept silently reading and writing the real project's
data/ regardless of what the fixture intended. The golden master stayed
green throughout, not because the redirection worked, but because MHW
detection is deterministic and the real data/sst_daily.csv had not
changed since the frozen fixture was taken: re-running the same
deterministic algorithm on unchanged real input reproduced the exact
committed reference bytes, so no test ever showed a diff, and `git status`
stayed clean because the "wrong" write produced content identical to what
was already there. Nothing about that outcome was guaranteed by the code;
it was a coincidence of timing that a single day of new auto-updated data,
on either side, would have ended.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

from climate_change_on_sea_urchins import common, mhw_detection

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "src" / "climate_change_on_sea_urchins"


def test_no_module_builds_the_data_path_itself():
    """The other half of test_results_dir.py's search, for data/ instead of
    results/ -- this is what would have caught mhw_detection.py's bug by
    static search alone, before ever running it. common.py is the one place
    allowed to write ROOT / "data" (it's what DATA is derived from)."""
    offenders = []
    for path in SRC.glob("*.py"):
        if path.name == "common.py":
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"""ROOT\s*/\s*["']data["']""", code):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert not offenders, "data/ path built outside common.DATA:\n" + "\n".join(offenders)


def _synthetic_sst(n_days: int = 40) -> pd.DataFrame:
    """A tiny, deliberately-tiny SST series: far too short to produce any
    MHW event (detect_events requires MIN_DAYS=5 consecutive exceedances,
    and this synthetic climatology baseline is degenerate with 0 baseline
    years), and with a date range that shares nothing with the real
    project's data/sst_daily.csv (2003-present). If mhw_detection.run()
    ever silently fell back to reading the real file instead of this one,
    the assertions below would fail on shape/date-range, not by accident."""
    dates = pd.date_range("2099-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"Datetime": dates, "Temperature": 15.0})


def test_mhw_detection_reads_and_writes_only_the_patched_data_dir(tmp_path, monkeypatch):
    """The direct regression test: point mhw_detection at a tmp directory
    with synthetic content, run it, and prove (a) it produced output from
    the SYNTHETIC input, not the real project's, and (b) the real
    data/*.csv files were not touched. Fails exactly the way it would have
    failed before the 2026-09-25 fix, if re-run against the pre-fix code:
    (a) would pass by accident (the real data/ happens to also start
    earlier and be much longer, so no crash), but (b) would fail -- the
    real data/mhw_events.csv etc. would show a changed mtime.
    """
    _synthetic_sst().to_csv(tmp_path / "sst_daily.csv", index=False)

    real_data_snapshot = {
        p.name: p.stat().st_mtime_ns for p in (REPO_ROOT / "data").glob("mhw_*.csv")
    }

    monkeypatch.setattr(common, "DATA", tmp_path)
    mhw_detection.run()

    # (a) Output reflects the synthetic input, in the patched directory.
    monthly = pd.read_csv(tmp_path / "mhw_monthly.csv", parse_dates=["Datetime"])
    assert len(monthly) <= 2, "expected ~40 days of synthetic data to span at most 2 months"
    assert monthly["Datetime"].min().year == 2099
    assert not (tmp_path / "mhw_events.csv").exists(), "40 flat-temperature days must not detect any MHW event"

    # (b) The real project's own MHW catalogue is untouched.
    real_data_after = {
        p.name: p.stat().st_mtime_ns for p in (REPO_ROOT / "data").glob("mhw_*.csv")
    }
    assert real_data_after == real_data_snapshot, (
        "mhw_detection.run() wrote into the REAL data/ directory despite common.DATA "
        "being patched to a tmp directory -- the exact bug this test exists to catch."
    )


def test_a_module_level_path_bound_at_import_time_would_not_be_caught_by_patching_common():
    """Not a regression test of mhw_detection.py itself (covered above) --
    a minimal, self-contained demonstration of WHY the bug shape is
    dangerous, for whoever next adds a module that touches data/. If this
    assertion ever fails, the language's import semantics changed, not
    this project."""
    class _BuggyModule:
        """Mimics the pre-fix mhw_detection.py: binds a path from DATA at
        "import time" (i.e. construction time here), not at call time."""
        def __init__(self, data_dir):
            self.DATA = data_dir
            self.SST_PATH = data_dir / "sst_daily.csv"  # bound now, not looked up later

    mod = _BuggyModule(Path("/original/data"))
    mod.DATA = Path("/patched/data")  # a test fixture "redirecting" it, same as monkeypatch would
    assert mod.SST_PATH == Path("/original/data/sst_daily.csv"), (
        "if this ever equals /patched/data/sst_daily.csv, binding a path once at "
        "construction/import time and reading it later has stopped being stale after "
        "reassigning the name it was built from -- which is not how Python works, so "
        "this documents an invariant, not a live risk."
    )


def test_every_test_that_redirects_root_also_redirects_data():
    """Since common.DATA stopped being derived from common.ROOT (it comes
    from the spec's data_dir), redirecting ROOT alone no longer redirects
    what load_data() reads. tests/test_mhw_analysis.py's fixture did exactly
    that and silently read the REAL data/ instead of the frozen fixture --
    green until the 2026-09-25 auto-update added one EC50 month (n 163 ->
    164 at lag 0). The search above covers src/ only; this one covers the
    tests that redirect it."""
    offenders = []
    for path in (REPO_ROOT / "tests").glob("*.py"):
        code = path.read_text()
        if re.search(r"""setattr\(\s*common\s*,\s*["']ROOT["']""", code) and \
           not re.search(r"""setattr\(\s*common\s*,\s*["']DATA["']""", code):
            offenders.append(path.name)
    assert not offenders, "redirects common.ROOT but not common.DATA: " + ", ".join(offenders)
