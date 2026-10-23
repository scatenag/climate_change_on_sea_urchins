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


def test_it_is_one_job_with_nothing_set_at_job_level(wf):
    assert set(wf["jobs"]) == {"download"}
    job = wf["jobs"]["download"]
    assert not {"env", "permissions", "environment", "services", "container", "outputs", "needs"} & set(job)


def test_it_cannot_write_to_the_repository(wf):
    assert wf["permissions"] == {"contents": "read"}
    checkout = next(s for s in _steps(wf) if str(s.get("uses", "")).startswith("actions/checkout"))
    assert checkout["uses"] == "actions/checkout@v4" and checkout["with"] == {"persist-credentials": False}
    for s in _steps(wf):
        if "uses" in s:
            assert s["uses"].split("@")[0] in {"actions/checkout", "actions/setup-python", "actions/upload-artifact"} and "@v" in s["uses"]
    text = WORKFLOW.read_text().lower()
    for forbidden in ("git commit", "git push", "git add", "git  ", "gh pr", "gh api", "ssh-key", "deploy", "curl", "wget"):
        assert forbidden not in text, forbidden


def test_the_secrets_reach_one_step_only_and_as_environment_of_that_step(wf):
    dump = yaml.safe_dump(wf)                         # the parsed workflow: comments are not in it
    assert dump.count("secrets") == 2                 # the username and the password, nothing else (no toJSON(secrets), no secrets['x'])
    holders = [s for s in _steps(wf) if "secrets" in yaml.safe_dump(s)]
    assert len(holders) == 1 and holders[0]["name"] == "Download the series"
    assert holders[0]["env"] == {"COPERNICUSMARINE_SERVICE_USERNAME": "${{ secrets.COPERNICUSMARINE_SERVICE_USERNAME }}",
                                 "COPERNICUSMARINE_SERVICE_PASSWORD": "${{ secrets.COPERNICUSMARINE_SERVICE_PASSWORD }}"}
    run = holders[0]["run"].replace('echo "First day of the product: $START"', "")      # the one line that prints, a date
    # the toolbox reads the two variables by itself: the script never names, prints or exports them
    for forbidden in ("COPERNICUSMARINE", "set -x", "echo", "printenv", "env ", "export", "GITHUB_ENV", "login", "${{"):
        assert forbidden not in run, forbidden


def test_the_first_day_is_read_from_the_catalogue_not_written_in_the_workflow(wf):
    import re
    run = next(s for s in _steps(wf) if s.get("name") == "Download the series")["run"]
    assert "coverage_of(catalog.get('sst_daily')).start.isoformat()" in run and '"--start", sys.argv[1]' in run
    assert "--end" not in run and not re.search(r"(19|20)\d\d-?\d\d-?\d\d", run)


def test_the_site_is_the_case_studys_not_numbers_written_in_the_workflow(wf):
    run = next(s for s in _steps(wf) if s.get("name") == "Download the series")["run"]
    for expected in ('"--lat", str(config.SITE_LAT)', '"--lon", str(config.SITE_LON)', '"--bbox-delta", str(config.BBOX_DELTA)', '"--name", config.SITE_NAME'):
        assert expected in run, expected
    assert "check=True" in run and '"--out", "sst-full-product"' in run
    assert run.count("subprocess.run(") == 1 and run.count("ccsu-download-sst") == 1        # one command, and it is the download


def test_the_series_is_published_as_an_artifact_even_if_the_download_fails_halfway(wf):
    up = next(s for s in _steps(wf) if str(s.get("uses", "")).startswith("actions/upload-artifact"))
    assert up["uses"] == "actions/upload-artifact@v4"
    assert up["with"] == {"name": "sst-full-product", "path": "sst-full-product", "if-no-files-found": "warn", "retention-days": 14}
    assert up["if"] == "always()"
