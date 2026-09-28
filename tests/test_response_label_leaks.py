"""
The case's response identity must never leak into outputs through code.

V2.1's acceptance criterion searched the source for "EC50" as a standalone
string literal, so it could not see EC50 inside longer texts or inside
output keys (e.g. a note, or a column named skill_mhw_to_ec50). This test
does not depend on how the code is written: it runs the whole pipeline --
every module, the R DLNM script when available, and one temporal window --
on a copy of the frozen fixture, with the response's label changed to
RSPTEST, and requires that no file under results/ mention EC50 in any form
(any case, with or without a separator), in its name or its content.

Module outputs may contain only (the rule, docs/roadmap/STATO.md): method
descriptions true for any study, values computed in that run, identities
derived from the spec. Case-specific text (manuscript references, results of
past investigations, people, decisions) belongs in the case's documentation,
next to its study.yaml, or in ADRs.
"""
import datetime as dt
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_DATA = REPO_ROOT / "tests" / "fixtures" / "paper_mpb_2026" / "data"
LEAK = re.compile(r"ec[\s_\-]*50", re.IGNORECASE)


@pytest.fixture(scope="module")
def relabeled_results(tmp_path_factory):
    import config
    from climate_change_on_sea_urchins import common, pipeline
    from climate_change_on_sea_urchins.study_spec import WindowSpec

    tmp_root = tmp_path_factory.mktemp("relabeled")
    tmp_data = tmp_root / "data"
    shutil.copytree(FIXTURE_DATA, tmp_data)
    results = tmp_root / "results"
    results.mkdir()
    relabeled = config.RESPONSE_SPEC.model_copy(update={"label": "RSPTEST"})

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(common, "ROOT", tmp_root)
        mp.setattr(common, "DATA", tmp_data)
        for _label, mod in pipeline._MODULES:
            if hasattr(mod, "ROOT"):
                mp.setattr(mod, "ROOT", tmp_root)
            if hasattr(mod, "DATA"):
                mp.setattr(mod, "DATA", tmp_data)
        # Both the explicit path (pipeline.main passes config.RESPONSE_SPEC)
        # and the fallback (default_response_spec() reads config.RESPONSE_SPEC).
        mp.setattr(config, "RESPONSE_SPEC", relabeled)
        pipeline.main(results=results)
        window = WindowSpec(id="w2010-2020", start=dt.date(2010, 1, 1), end=dt.date(2020, 12, 31))
        pipeline.run_window(window, results / window.id, response=relabeled, provenance={})
    finally:
        mp.undo()

    r_script = shutil.which("Rscript")
    if r_script is not None:
        scripts_dir = tmp_root / "scripts"
        scripts_dir.mkdir()
        shutil.copy(REPO_ROOT / "scripts" / "mhw_lag_analysis.R", scripts_dir)
        subprocess.run(
            [r_script, str(scripts_dir / "mhw_lag_analysis.R"), str(tmp_data), str(results)],
            check=True, capture_output=True, text=True,
        )
    return results


@pytest.mark.golden
def test_no_output_mentions_ec50_when_the_label_is_different(relabeled_results):
    files = sorted(p for p in relabeled_results.rglob("*") if p.is_file())
    assert len(files) > 50, "the relabeled pipeline produced suspiciously few outputs"
    assert any("RSPTEST" in p.read_text() for p in files), "the relabeled label never reached an output"

    leaks = []
    for p in files:
        rel = p.relative_to(relabeled_results)
        if LEAK.search(str(rel)):
            leaks.append(f"{rel}: in the file name")
        for n, line in enumerate(p.read_text().splitlines(), 1):
            for m in LEAK.finditer(line):
                lo = max(m.start() - 60, 0)
                leaks.append(f"{rel}:{n}: ...{line[lo:m.end() + 60]}...")
    assert not leaks, f"{len(leaks)} EC50 leak(s) with label RSPTEST:\n" + "\n".join(leaks)
