"""scripts/verify_home_clean.py (the manual check of MA.2 against the real toolbox) end to end without the network:
the check must see a write to the home directory, a cache, the working directory, or the runner's real home, and say
what and where; and the credentials must be gone from the environment when the client is built."""
import importlib.util
import datetime as dt
import os
import sys
from pathlib import Path

import pytest

from test_download import FakeClient, PASSWORD, USER

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("verify_home_clean", ROOT / "scripts" / "verify_home_clean.py")
vh = importlib.util.module_from_spec(spec); spec.loader.exec_module(vh)


class WritingClient(FakeClient):
    """A client that writes a file where `target` says, as a toolbox with a catalogue cache would."""
    target: Path | None = None

    def coverage_of(self, variable):
        if self.target is not None:
            self.target.parent.mkdir(parents=True, exist_ok=True)
            self.target.write_text("cache")
        return super().coverage_of(variable)


def _dirs(tmp_path):
    fh, fc, rh = tmp_path / "fresh_home", tmp_path / "fresh_cwd", tmp_path / "real_home"
    for d in (fh, fc, rh):
        d.mkdir()
    return fh, fc, rh


def _check(tmp_path, client):
    fh, fc, rh = _dirs(tmp_path)
    return vh.run_check((USER, PASSWORD), lambda: client, fresh_home=fh, fresh_cwd=fc, real_home=rh), (fh, fc, rh)


def test_a_toolbox_that_writes_nothing_is_clean(tmp_path):
    report, _ = _check(tmp_path, FakeClient())
    assert report["clean"] and report["days_downloaded"] == 10
    assert "clean" in vh.summary(report)


@pytest.mark.parametrize("where, key", [("fh", "fresh_home"), ("fc", "fresh_cwd"), ("rh", "real_home")])
def test_a_write_in_any_of_the_three_places_is_reported_with_what_and_where(tmp_path, where, key):
    fh, fc, rh = _dirs(tmp_path)
    base = {"fh": fh / ".copernicusmarine" / "cache.json", "fc": fc / "catalogue.cache", "rh": rh / ".copernicusmarine" / "catalogue.json"}[where]
    client = WritingClient(); client.target = base
    report = vh.run_check((USER, PASSWORD), lambda: client, fresh_home=fh, fresh_cwd=fc, real_home=rh)
    assert not report["clean"] and report["written"][key] and report["written"][key][0]["what"] == "created"
    text = vh.summary(report)
    assert "FAIL" in text and base.name in text


def test_a_change_of_an_existing_file_is_a_write_too(tmp_path):
    fh, fc, rh = _dirs(tmp_path)
    existing = rh / ".copernicusmarine" / "x.json"
    existing.parent.mkdir(); existing.write_text("old")
    client = WritingClient(); client.target = existing
    report = vh.run_check((USER, PASSWORD), lambda: client, fresh_home=fh, fresh_cwd=fc, real_home=rh)
    assert report["written"]["real_home"][0]["what"] == "changed"


def test_a_new_top_level_name_in_the_real_home_is_reported(tmp_path):
    fh, fc, rh = _dirs(tmp_path)
    client = WritingClient(); client.target = rh / ".copernicusmarine-credentials"        # a file, not under a watched name
    report = vh.run_check((USER, PASSWORD), lambda: client, fresh_home=fh, fresh_cwd=fc, real_home=rh)
    assert any(".copernicusmarine-credentials" in r["path"] for r in report["written"]["real_home"])


def test_main_takes_the_credentials_out_of_the_environment_before_the_client_is_built(tmp_path, monkeypatch):
    seen = {}
    from climate_change_on_sea_urchins import download
    class Spy(FakeClient):
        pass
    def factory():
        seen["env"] = {k: v for k, v in os.environ.items() if k.startswith("COPERNICUSMARINE")}
        seen["home"] = os.environ["HOME"]
        return Spy()
    monkeypatch.setattr(download, "CopernicusClient", factory)
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", USER)
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", PASSWORD)
    real_home = tmp_path / "real_home"; real_home.mkdir()
    monkeypatch.setenv("HOME", str(real_home))                 # never the user's own home in a test
    monkeypatch.chdir(tmp_path)
    code = vh.main([str(tmp_path / "out")])
    assert code == 0 and seen["env"] == {} and seen["home"] != str(real_home)
    report = (tmp_path / "out" / "home-check.md").read_text()
    assert USER not in report and PASSWORD not in report


def test_main_refuses_to_run_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
    assert vh.main([str(tmp_path / "out")]) == 2


def test_main_fails_when_the_toolbox_writes_in_the_home_directory(tmp_path, monkeypatch):
    from climate_change_on_sea_urchins import download

    class Writes(FakeClient):
        def coverage_of(self, variable):
            (Path(os.environ["HOME"]) / ".copernicusmarine").mkdir(exist_ok=True)
            (Path(os.environ["HOME"]) / ".copernicusmarine" / "catalogue-cache.json").write_text("x")
            return super().coverage_of(variable)

    monkeypatch.setattr(download, "CopernicusClient", Writes)
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", USER)
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", PASSWORD)
    real_home = tmp_path / "real_home"; real_home.mkdir()
    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.chdir(tmp_path)
    assert vh.main([str(tmp_path / "out")]) == 1
    assert "catalogue-cache.json" in (tmp_path / "out" / "home-check.md").read_text()
