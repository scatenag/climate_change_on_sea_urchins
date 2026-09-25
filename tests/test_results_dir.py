"""
Every reader and writer of results/ goes through one resolver,
common.results_dir(study_id), so that moving results under a per-study
directory is a change to that function alone.
"""
import re
from pathlib import Path

import config
from climate_change_on_sea_urchins import common

SRC = Path(__file__).parent.parent / "src" / "climate_change_on_sea_urchins"


def test_results_is_the_resolver_applied_to_the_selected_study():
    assert common.RESULTS == common.results_dir(config.STUDY_ID)


def test_no_module_builds_the_results_path_itself():
    offenders = []
    for path in SRC.glob("*.py"):
        if path.name == "common.py":
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"""/\s*["']results["']""", code) or "RESULTS.parent" in code:
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert not offenders, "results/ path built outside common.results_dir():\n" + "\n".join(offenders)


def test_no_analysis_module_reads_the_results_constant():
    """Analysis modules receive where to write as run()'s `results`
    parameter (CLAUDE.md invariant #3), never from common.RESULTS: one
    pipeline execution will write several windows' results into sibling
    directories, which a single process-wide constant cannot express.
    dashboard.py is a reader of the selected study's results, not a
    run() module, so it keeps reading the constant."""
    offenders = []
    for path in SRC.glob("*.py"):
        if path.name in ("common.py", "dashboard.py"):
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"\bRESULTS\b", code):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert not offenders, "module reads common.RESULTS instead of its results parameter:\n" + "\n".join(offenders)


def test_every_pipeline_module_that_writes_results_takes_results_as_a_parameter():
    import inspect
    from climate_change_on_sea_urchins import pipeline
    missing = [
        label for label, mod in pipeline._MODULES
        if label != "mhw_detection"  # writes data_dir only, never results
        and "results" not in inspect.signature(mod.run).parameters
    ]
    assert not missing, f"run() without a results parameter: {missing}"
