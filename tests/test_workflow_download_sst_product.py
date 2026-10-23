"""The manual workflow that downloads the case study's SST from the first day of the product (use of the repository's
Copernicus secrets): it must be manual only, write nothing to the repository, give the secrets to one step, read the first
day from the catalogue and publish the series as an artifact."""
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "download_sst_product.yml"


@pytest.fixture(scope="module")
def wf():
    return yaml.safe_load(WORKFLOW.read_text())


def _steps(wf):
    return wf["jobs"]["download"]["steps"]


def test_it_runs_only_by_hand(wf):
    triggers = wf.get("on", wf.get(True))
    assert set(triggers) == {"workflow_dispatch"}


def test_it_cannot_write_to_the_repository(wf):
    assert wf["permissions"] == {"contents": "read"}
    checkout = next(s for s in _steps(wf) if str(s.get("uses", "")).startswith("actions/checkout"))
    assert checkout["with"]["persist-credentials"] is False
    text = WORKFLOW.read_text()
    for forbidden in ("git commit", "git push", "git add", "gh pr", "ssh-key", "deploy"):
        assert forbidden not in text, forbidden


def test_the_secrets_reach_one_step_only_and_as_environment_of_that_step(wf):
    holders = [s for s in _steps(wf) if "secrets." in yaml.safe_dump(s)]
    assert len(holders) == 1 and holders[0]["name"] == "Download the series"
    assert set(holders[0]["env"]) == {"COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD"}
    assert "secrets." not in yaml.safe_dump({k: v for k, v in wf.items() if k != "jobs"})


def test_the_first_day_is_read_from_the_catalogue_not_written_in_the_workflow(wf):
    import re
    run = next(s for s in _steps(wf) if s.get("name") == "Download the series")["run"]
    assert "coverage_of" in run and "--start" in run
    assert not re.search(r"\b(19|20)\d\d-\d\d-\d\d\b", run)


def test_the_series_is_published_as_an_artifact_even_if_the_download_fails_halfway(wf):
    up = next(s for s in _steps(wf) if str(s.get("uses", "")).startswith("actions/upload-artifact"))
    assert up["with"]["name"] == "sst-full-product" and up["with"]["path"] == "sst-full-product" and up["if"] == "always()"
    assert "retention-days" in up["with"]
