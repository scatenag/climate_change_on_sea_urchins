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
