"""MA.2: step 1 of the user's own data -- site, period and credentials, and the download of the sea-surface temperature.
Run through streamlit.testing from app.py with the Copernicus client replaced by the test client of tests/test_download.py
(a numpy cube in place of the toolbox): no network, no credentials, no variable of the environment. The real toolbox is
checked by scripts/verify_home_clean.py in the manual workflow."""
from __future__ import annotations

import ast
import datetime as dt
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import climate_change_on_sea_urchins as ccsu
from climate_change_on_sea_urchins import app_entry, dataset_builder, step_site
from climate_change_on_sea_urchins.download import Coverage
from test_download import FakeClient, PASSWORD, USER

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "climate_change_on_sea_urchins"
FIXTURE_SST = ROOT / "tests" / "fixtures" / "paper_mpb_2026" / "data" / "sst_daily.csv"
COVERAGE_2003 = Coverage(start=dt.date(2003, 1, 1), end=dt.date(2026, 9, 30), version="202511")


@dataclass
class RecordingClient(FakeClient):
    """The test client, recording the credentials it receives, whether the global lock was held, and serving the
    frozen fixture SST (the same value in every cell, so the daily mean is the fixture's)."""
    credentials_seen: list = field(default_factory=list)
    lock_held: list = field(default_factory=list)
    series: pd.Series | None = None

    def subset(self, variable, *, credentials, start, end, **kw):
        self.credentials_seen.append((credentials.username, credentials.password))
        self.lock_held.append(ccsu.RERUN_LOCK.locked())
        cube = super().subset(variable, credentials=credentials, start=start, end=end, **kw)
        if self.series is not None and not self.fail_with:
            days = pd.to_datetime(cube.times.astype("datetime64[ns]"))
            for i, d in enumerate(days):
                cube.values[i] = np.where(np.isnan(cube.values[i]), np.nan, self.series.get(d, np.nan))
        return cube


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for var in [v for v in os.environ if v.startswith("COPERNICUSMARINE")]:
        monkeypatch.delenv(var)
    monkeypatch.setattr(app_entry, "SESSIONS_ROOT", tmp_path / "sessions")
    monkeypatch.setattr(step_site, "acquisition_available", lambda: True)


@pytest.fixture
def client(monkeypatch):
    c = RecordingClient(coverage=COVERAGE_2003)
    monkeypatch.setattr(step_site, "make_client", lambda: c)
    return c


@pytest.fixture
def fixture_client(client):
    sst = pd.read_csv(FIXTURE_SST, parse_dates=["Datetime"]).set_index("Datetime")["Temperature"]
    client.series = sst
    client.coverage = Coverage(start=dt.date(2003, 1, 1), end=sst.index.max().date(), version="202511")
    return client


def _open():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    next(b for b in at.button if "own data" in b.label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _by_label(at, label):
    return next(t for t in at.text_input if t.label == label)


def _fill(at, *, lat="43.4278", lon="10.3956", name="Test site", start=2003, end=2012, first_year="", user=USER, password=PASSWORD):
    _by_label(at, "Latitude (degrees north)").set_value(lat)
    _by_label(at, "Longitude (degrees east)").set_value(lon)
    _by_label(at, "Name of the site").set_value(name)
    next(n for n in at.number_input if n.label == "Baseline: first year").set_value(start)
    next(n for n in at.number_input if n.label == "Baseline: last year").set_value(end)
    _by_label(at, "First year of your measurements (optional)").set_value(first_year)
    _by_label(at, "Copernicus username").set_value(user)
    _by_label(at, "Copernicus password").set_value(password)
    return at


def _submit(at):
    next(b for b in at.button if "Download the sea-surface" in b.label).click().run()
    return at


def _walk(node):
    for child in node.children.values():
        yield child
        if hasattr(child, "children"):
            yield from _walk(child)


def _text(at):
    """Everything the page shows, from the elements' protocol messages (what the browser receives). The value a test
    typed into an input is not in them: streamlit.testing does not emulate the form clearing its fields on submit."""
    return "\n".join(str(e.proto) for tree in (at.main, at.sidebar) for e in _walk(tree) if hasattr(e, "proto"))


def _errors(at):
    return " ".join(e.value for e in at.error).lower()


def _ctx(at):
    return at.session_state[app_entry.CONTEXT_KEY]


def _files_text(folder: Path) -> str:
    return "\n".join(p.read_bytes().decode("utf-8", "replace") for p in folder.rglob("*") if p.is_file())


# --- the coverage and the baseline proposed -----------------------------------------------------------------

def test_the_coverage_is_read_from_the_catalogue_and_the_baseline_proposed_from_it(client):
    at = _open()
    assert ("coverage", "sst_daily") in client.calls
    assert "2003-01-01" in _text(at) and "2026-09-30" in _text(at)
    start = next(n for n in at.number_input if n.label == "Baseline: first year").value
    end = next(n for n in at.number_input if n.label == "Baseline: last year").value
    assert (start, end) == (2003, 2025) and "shorter than 30 years" in _text(at)      # the product covers 23 years only


def test_a_product_covering_1991_to_2020_proposes_exactly_that(client):
    client.coverage = Coverage(start=dt.date(1987, 1, 1), end=dt.date(2026, 8, 31), version="202511")
    at = _open()
    assert [n.value for n in at.number_input if n.label.startswith("Baseline")] == [1991, 2020]


def test_nothing_about_the_site_is_filled_in_for_the_user(client):
    at = _open()
    assert _by_label(at, "Latitude (degrees north)").value == "" and _by_label(at, "Longitude (degrees east)").value == ""
    assert _by_label(at, "Name of the site").value == ""


def test_without_the_acquisition_extra_the_step_says_so(monkeypatch):
    monkeypatch.setattr(step_site, "acquisition_available", lambda: False)
    at = _open()
    assert "acquisition" in _text(at) and not [b for b in at.button if "Download the sea-surface" in b.label]


def test_a_failure_to_read_the_catalogue_is_shown_with_a_way_to_retry(monkeypatch):
    class Down(FakeClient):
        def coverage_of(self, variable):
            raise RuntimeError("no network")
    monkeypatch.setattr(step_site, "make_client", lambda: Down())
    at = _open()
    assert not at.exception and "catalogue" in _text(at) and any("Try again" in b.label for b in at.button)


# --- the inputs --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lat, lon, name, expect", [
    ("", "10.4", "x", "latitude"),
    ("43.4", "", "x", "longitude"),
    ("43,4", "10.4", "x", "dot"),
    ("abc", "10.4", "x", "not a number"),
    ("95", "10.4", "x", "between"),
    ("43.4", "10.4", "  ", "name"),
])
def test_an_input_that_is_missing_or_wrong_is_refused_with_the_reason_and_nothing_is_downloaded(client, lat, lon, name, expect):
    at = _fill(_open(), lat=lat, lon=lon, name=name)
    _submit(at)
    assert not at.exception and expect in _errors(at)
    assert not [c for c in client.calls if c[0] == "subset"] and client.credentials_seen == []


def test_a_site_outside_the_domain_is_refused_before_any_download_call(client):
    at = _submit(_fill(_open(), lat="60.0", lon="10.0"))
    assert "outside the domain" in _text(at)
    assert not [c for c in client.calls if c[0] == "subset"] and client.credentials_seen == []


def test_a_baseline_the_product_does_not_cover_is_refused_before_any_download_call(client):
    at = _submit(_fill(_open(), start=1980, end=2005))
    assert "before the first year" in _text(at)
    assert not [c for c in client.calls if c[0] == "subset"]


def test_a_land_cell_and_a_sea_the_product_does_not_cover_are_told_apart(client):
    client.sea = lambda lat, lon: False
    at = _submit(_fill(_open()))
    assert "inland, or in a sea the product does not cover" in _text(at)
    client.sea = lambda lat, lon: lat < 43.0                       # sea only to the south of the box
    at = _submit(_fill(_open()))
    assert "land in the product" in _text(at) and "km away" in _text(at)


def test_a_missing_credential_is_refused_by_the_download_with_a_plain_message(client):
    at = _submit(_fill(_open(), password=""))
    assert not at.exception and "username and password are required" in _text(at)


# --- the download ------------------------------------------------------------------------------------------

def test_the_series_and_the_manifest_are_written_in_the_session_folder_with_the_duration(fixture_client):
    at = _submit(_fill(_open()))
    assert not at.exception, [e.value for e in at.exception]
    inputs = _ctx(at).workdir / "inputs"
    series = pd.read_csv(inputs / "sst_daily.csv", parse_dates=["Datetime"])
    manifest = json.loads((inputs / "sst_daily.manifest.json").read_text())
    assert series["Datetime"].min() == pd.Timestamp("2003-01-01") and len(series) == manifest["n_days"]
    assert manifest["timing"]["download_seconds"] >= 0 and manifest["site"]["name"] == "Test site"
    assert "seconds" in _text(at) and "days" in _text(at)


def test_the_download_starts_at_the_baseline_or_earlier_for_the_first_measurement(client):
    client.coverage = Coverage(start=dt.date(1987, 1, 1), end=dt.date(2026, 8, 31), version="202511")
    _submit(_fill(_open(), start=1991, end=2020, first_year=""))
    assert min(c[5] for c in client.calls if c[0] == "subset") == dt.date(1991, 1, 1)
    client.calls.clear()
    _submit(_fill(_open(), start=1991, end=2020, first_year="1988"))
    assert min(c[5] for c in client.calls if c[0] == "subset") == dt.date(1987, 1, 1)


def test_the_credentials_reach_the_download_as_the_users_own(fixture_client):
    _submit(_fill(_open()))
    assert set(fixture_client.credentials_seen) == {(USER, PASSWORD)}


def test_the_global_lock_is_not_held_during_the_download(fixture_client):
    _submit(_fill(_open()))
    assert fixture_client.lock_held and not any(fixture_client.lock_held)


def test_the_view_shows_the_heatwaves_the_dataset_builder_finds(fixture_client):
    at = _submit(_fill(_open()))
    sst = pd.read_csv(FIXTURE_SST, parse_dates=["Datetime"])
    expected = len(dataset_builder.environment_dataset(sst, (2003, 2012)).mhw_events)
    assert f"{expected} marine heatwave" in _text(at)


def test_a_failure_of_the_view_is_shown_not_hidden(fixture_client):
    fixture_client.series = fixture_client.series.copy()
    fixture_client.series[pd.Timestamp("2010-06-15")] = np.nan                # a gap inside the daily SST
    at = _submit(_fill(_open()))
    assert not at.exception and "gap" in _text(at)


def test_changing_the_inputs_after_a_download_says_the_data_are_for_the_earlier_inputs(fixture_client):
    at = _submit(_fill(_open()))
    _by_label(at, "Name of the site").set_value("Another name").run()
    assert "earlier inputs" in _text(at)


def test_another_session_does_not_see_the_download(fixture_client):
    a = _submit(_fill(_open()))
    b = _open()
    assert app_entry.CONTEXT_KEY in b.session_state and _ctx(a).workdir != _ctx(b).workdir
    assert "Test site" not in _text(b) and not (_ctx(b).workdir / "inputs").exists()


# --- the credentials: nowhere -----------------------------------------------------------------------------

def _assert_nowhere(at, capsys, caplog, secret):
    out, err = capsys.readouterr()
    where = {
        "files of the session folder": _files_text(_ctx(at).workdir),
        "session state": repr(dict(at.session_state.filtered_state)),
        "the context": repr(_ctx(at)),
        "rendered text": _text(at),
        "environment": "\n".join(os.environ.values()),
        "stdout": out, "stderr": err, "log": caplog.text,
    }
    leaks = [k for k, v in where.items() if secret in v]
    assert not leaks, f"credential found in: {leaks}"


def test_the_credentials_are_in_no_file_no_state_no_message_no_environment_no_log(fixture_client, capsys, caplog):
    at = _submit(_fill(_open()))
    assert fixture_client.credentials_seen                                        # the download did run
    for secret in (PASSWORD, USER):
        _assert_nowhere(at, capsys, caplog, secret)


def test_an_error_that_contains_the_credentials_is_masked_everywhere(client, capsys, caplog):
    client.fail_with = f"authentication failed for {USER} with {PASSWORD}"
    at = _submit(_fill(_open()))
    assert not at.exception and "failed" in _text(at) and "***" in _text(at)
    for secret in (PASSWORD, USER):
        _assert_nowhere(at, capsys, caplog, secret)


def test_an_unexpected_error_inside_the_download_is_masked_too(client, capsys, caplog, monkeypatch):
    def boom(*a, **k):
        raise ValueError(f"unexpected {PASSWORD}")
    monkeypatch.setattr(step_site, "download_sst", boom)
    at = _submit(_fill(_open()))
    assert not at.exception and "Unexpected error" in _text(at)
    _assert_nowhere(at, capsys, caplog, PASSWORD)


def _calls(path):
    tree = ast.parse(path.read_text())
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def test_the_credentials_form_is_a_password_field_inside_a_form_that_clears_on_submit():
    src = (PKG / "step_site.py").read_text()
    tree = ast.parse(src)
    kwargs = lambda call: {k.arg: ast.literal_eval(k.value) for k in call.keywords if isinstance(k.value, ast.Constant)}
    password_fields = [c for c in _calls(PKG / "step_site.py") if isinstance(c.func, ast.Attribute) and c.func.attr == "text_input"
                       and kwargs(c).get("type") == "password"]
    forms = [c for c in _calls(PKG / "step_site.py") if isinstance(c.func, ast.Attribute) and c.func.attr == "form"]
    assert len(password_fields) == 1 and not [k for k in password_fields[0].keywords if k.arg == "key"]   # no state key
    assert len(forms) == 1 and kwargs(forms[0]).get("clear_on_submit") is True


@pytest.mark.parametrize("forbidden", [r"\blogin\b", r"\bos\.environ\b", r"\bgetenv\b", r"st\.cache", r"session_state\[[^\]]*(pass|user|cred)",
                                       r"\bprint\(", r"logging", r"\.netrc"])
def test_the_step_never_reads_the_environment_logs_caches_or_logs_in(forbidden):
    code = re.sub(r"#.*", "", (PKG / "step_site.py").read_text())
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    assert not re.search(forbidden, code, re.IGNORECASE), forbidden
