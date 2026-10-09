"""Download of the daily SST for a site (M1.2): the catalogue decides what is downloaded and where it ends,
the credentials are the user's and never leave the call, a site that cannot work is refused with the reason,
and a missing day stays missing.

All logic is tested here against a client the tests provide (a numpy cube in place of the Copernicus
toolbox), so it runs in CI, which has neither copernicusmarine nor xarray. The thin adapter to the real
toolbox is tested separately with a stand-in toolbox, where xarray is available.
"""
import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from climate_change_on_sea_urchins import catalog, download
from climate_change_on_sea_urchins.download import (
    Coverage, Credentials, Cube, DownloadError, SiteError, fetch_daily_sst, write_result,
)
from climate_change_on_sea_urchins.study_spec import SiteSpec

USER, PASSWORD = "mario.rossi-user", "s3cr3t-pa55-w0rd"
CREDS = Credentials(USER, PASSWORD)
LIVORNO = SiteSpec(id="s1", lat=43.4278, lon=10.3956, name="Livorno Sud", bbox_delta=0.1)
CLOCK = lambda: dt.datetime(2026, 10, 9, 12, 0, tzinfo=dt.timezone.utc)


def _cube(times, lats, lons, depths=(0.5, 2.0), fill=None, land=()):
    """Cube with value = 10 + day index + depth index; cells in `land` (lat index, lon index) are NaN."""
    t, d, la, lo = len(times), len(depths), len(lats), len(lons)
    values = np.empty((t, d, la, lo))
    for i in range(t):
        for k in range(d):
            values[i, k] = 10.0 + i + k
    for (a, b) in land:
        values[:, :, a, b] = np.nan
    if fill is not None:
        values = fill(values)
    return Cube(times=np.array(times, dtype="datetime64[D]"), depths=np.array(depths),
                lats=np.array(lats, float), lons=np.array(lons, float), values=values, dataset_version="202511")


@dataclass
class FakeClient:
    """Stands in for the Copernicus toolbox. `sea` says which grid points are sea (a function of lat, lon)."""
    coverage: Coverage = Coverage(start=dt.date(1987, 1, 1), end=dt.date(2026, 8, 31), version="202511")
    sea: callable = lambda lat, lon: True
    missing_days: tuple = ()
    calls: list = field(default_factory=list)
    fail_with: str | None = None
    step: float = 0.0625

    def coverage_of(self, variable):
        self.calls.append(("coverage", variable.id))
        return self.coverage

    def subset(self, variable, *, lat_min, lat_max, lon_min, lon_max, start, end, credentials):
        self.calls.append(("subset", lat_min, lat_max, lon_min, lon_max, start, end))
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        days = pd.date_range(start, end, freq="D")
        lats = np.arange(lat_min, lat_max + 1e-9, self.step)
        lons = np.arange(lon_min, lon_max + 1e-9, self.step)
        cube = _cube(list(days.values.astype("datetime64[D]")), lats, lons)
        for a, la in enumerate(lats):
            for b, lo in enumerate(lons):
                if not self.sea(la, lo):
                    cube.values[:, :, a, b] = np.nan
        for i, day in enumerate(days.date):
            if day in self.missing_days:
                cube.values[i] = np.nan
        return cube


def _fetch(client=None, site=LIVORNO, **kw):
    kw.setdefault("start", dt.date(2024, 1, 1))
    kw.setdefault("end", dt.date(2024, 1, 10))
    return fetch_daily_sst(site, credentials=CREDS, client=client or FakeClient(), now=CLOCK, **kw)


# --- credentials: the user's, both required, never read from anywhere else --------------------------

def test_credentials_never_show_in_repr_or_str():
    for text in (repr(CREDS), str(CREDS), f"{CREDS}", f"{[CREDS]}"):
        assert USER not in text and PASSWORD not in text


@pytest.mark.parametrize("user,password", [("", PASSWORD), (USER, ""), (None, PASSWORD), (USER, None), ("  ", PASSWORD)])
def test_both_credentials_are_required(user, password):
    client = FakeClient()
    with pytest.raises(DownloadError, match="username and password"):
        fetch_daily_sst(LIVORNO, start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 2),
                        credentials=Credentials(user, password), client=client, now=CLOCK)
    assert client.calls == []


def test_credentials_are_not_taken_from_the_environment(monkeypatch):
    # In a shared process the environment belongs to every session: a missing credential is an error,
    # never silently replaced by whatever the process happens to hold.
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "someone-else")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "their-password")
    with pytest.raises(DownloadError, match="username and password"):
        fetch_daily_sst(LIVORNO, start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 2),
                        credentials=Credentials("", ""), client=FakeClient(), now=CLOCK)


def test_an_error_from_the_toolbox_never_carries_the_credentials():
    client = FakeClient(fail_with=f"authentication failed for {USER} with password {PASSWORD}")
    with pytest.raises(DownloadError) as e:
        _fetch(client)
    text = f"{e.value} {e.value.args}"
    assert USER not in text and PASSWORD not in text
    # no chained exception either: a traceback would print the original message
    assert e.value.__cause__ is None and e.value.__context__ is None


def test_toolbox_error_is_still_informative():
    with pytest.raises(DownloadError, match="authentication failed"):
        _fetch(FakeClient(fail_with=f"authentication failed for {USER}"))


# --- the site: outside the domain, land cell, outside the product ----------------------------------

def test_a_site_outside_the_catalogue_domain_is_refused_before_any_request():
    client = FakeClient()
    cape_town = SiteSpec(id="c", lat=-33.9, lon=18.4, name="Cape Town", bbox_delta=0.1)
    with pytest.raises(SiteError, match=r"Cape Town.*outside"):
        _fetch(client, site=cape_town)
    assert client.calls == []


def test_a_land_cell_is_told_apart_and_the_nearest_sea_is_named():
    # Sea only west of lon 10.25: the site's box (10.2956 to 10.4956) is all land, the sea is ~15 km away.
    client = FakeClient(sea=lambda lat, lon: lon < 10.25)
    with pytest.raises(SiteError) as e:
        _fetch(client)
    msg = str(e.value)
    assert "land" in msg and "nearest" in msg and "km" in msg
    assert "outside the area" not in msg
    assert not any(c[0] == "subset" and c[6] > c[5] for c in client.calls), "no multi-day download for a refused site"


def test_a_point_in_the_domain_box_but_not_in_the_product_is_told_apart():
    client = FakeClient(sea=lambda lat, lon: False)
    with pytest.raises(SiteError) as e:
        _fetch(client)
    msg = str(e.value)
    assert "outside the area" in msg and "bounding box" in msg
    assert "land cell" not in msg


def test_the_two_site_messages_differ():
    land = pytest.raises(SiteError)
    with land as a:
        _fetch(FakeClient(sea=lambda lat, lon: lon < 10.25))
    with pytest.raises(SiteError) as b:
        _fetch(FakeClient(sea=lambda lat, lon: False))
    assert str(a.value) != str(b.value)


# --- the period: the end is the product's, the start is checked -----------------------------------

def test_without_an_end_the_series_ends_where_the_product_ends():
    client = FakeClient(coverage=Coverage(dt.date(1987, 1, 1), dt.date(2024, 1, 5), "202511"))
    result = _fetch(client, end=None)
    assert result.series["Datetime"].iloc[-1] == pd.Timestamp("2024-01-05")
    assert result.manifest["period"]["obtained"]["end"] == "2024-01-05"
    assert result.manifest["period"]["requested"]["end"] is None


def test_an_end_beyond_the_product_is_cut_and_both_are_recorded():
    client = FakeClient(coverage=Coverage(dt.date(1987, 1, 1), dt.date(2024, 1, 5), "202511"))
    result = _fetch(client, end=dt.date(2024, 1, 10))
    assert result.series["Datetime"].iloc[-1] == pd.Timestamp("2024-01-05")
    assert result.manifest["period"]["requested"]["end"] == "2024-01-10"
    assert result.manifest["period"]["product_coverage"] == {"start": "1987-01-01", "end": "2024-01-05"}


def test_a_start_before_the_product_is_refused():
    with pytest.raises(DownloadError, match="1987-01-01"):
        _fetch(start=dt.date(1980, 1, 1))


def test_a_period_entirely_after_the_product_is_refused():
    with pytest.raises(DownloadError, match="2026-08-31"):
        _fetch(start=dt.date(2026, 10, 1), end=dt.date(2026, 10, 5))


# --- the series -----------------------------------------------------------------------------------

def test_daily_value_is_the_spatial_mean_per_depth_then_the_depth_mean():
    # Cube values are 10 + day + depth index everywhere (no land): the mean over the box is that value,
    # and the depth mean of depth indices 0 and 1 adds 0.5.
    result = _fetch()
    expected = [10.0 + i + 0.5 for i in range(10)]
    assert result.series["Temperature"].tolist() == pytest.approx(expected)
    assert result.series["Datetime"].iloc[0] == pd.Timestamp("2024-01-01")


def test_land_cells_inside_the_box_are_left_out_of_the_mean():
    client = FakeClient(sea=lambda lat, lon: lon >= 10.35)   # the western part of the box is land
    result = _fetch(client)
    assert result.series["Temperature"].notna().all()
    assert result.manifest["cells"]["sea_with_data"] < result.manifest["cells"]["in_box"]


def test_a_missing_day_stays_missing_and_is_reported():
    client = FakeClient(missing_days=(dt.date(2024, 1, 4), dt.date(2024, 1, 5)))
    result = _fetch(client)
    s = result.series.set_index("Datetime")["Temperature"]
    assert len(s) == 10
    assert s[pd.Timestamp("2024-01-04")] != s[pd.Timestamp("2024-01-04")]    # NaN
    assert np.isnan(s[pd.Timestamp("2024-01-05")])
    assert s[pd.Timestamp("2024-01-03")] == pytest.approx(12.5) and s[pd.Timestamp("2024-01-06")] == pytest.approx(15.5)
    assert result.manifest["missing_days"]["count"] == 2
    assert result.manifest["missing_days"]["days"] == ["2024-01-04", "2024-01-05"]


# --- provenance and files -----------------------------------------------------------------------

def test_manifest_records_the_product_and_how_the_data_were_obtained():
    m = _fetch().manifest
    sst = catalog.get("sst_daily")
    assert m["catalog_id"] == "sst_daily"
    assert m["dataset"] == {"id": sst.dataset_multiyear, "version": "202511", "variable": "thetao",
                            "depth_m": [0.0, 5.0]}
    assert m["site"]["lat"] == 43.4278 and m["site"]["bbox_delta"] == 0.1
    assert m["downloaded_at"] == "2026-10-09T12:00:00+00:00"
    assert m["period"]["obtained"] == {"start": "2024-01-01", "end": "2024-01-10"}


def test_written_files_match_their_hash_and_hold_no_credentials(tmp_path):
    result = _fetch()
    paths = write_result(result, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sst_daily.csv", "sst_daily.manifest.json"]
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["files"]["sst_daily.csv"]["sha256"] == hashlib.sha256(paths["csv"].read_bytes()).hexdigest()
    for p in paths.values():
        text = p.read_text()
        assert USER not in text and PASSWORD not in text


def test_csv_has_the_layout_the_pipeline_reads(tmp_path):
    paths = write_result(_fetch(), tmp_path)
    df = pd.read_csv(paths["csv"], parse_dates=["Datetime"])
    assert list(df.columns) == ["Datetime", "Temperature"]
    assert len(df) == 10


def test_the_end_logic_agrees_with_the_update_jobs_script(monkeypatch):
    # The update job (scripts/fetch_copernicus_daily.py) has its own coverage_end until the job moves to
    # ccsu-run-study (M1.9). Same catalogue description, same answer.
    import importlib.util, sys, types
    from pathlib import Path
    for name in ("copernicusmarine", "xarray"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    script = Path(__file__).resolve().parent.parent / "scripts" / "fetch_copernicus_daily.py"
    spec = importlib.util.spec_from_file_location("fetch_daily_for_agreement", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    ns = types.SimpleNamespace
    ms = lambda d: dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp() * 1000
    desc = ns(products=[ns(datasets=[ns(versions=[ns(label="202511", parts=[ns(services=[
        ns(service_name="arco-time-series", variables=[ns(short_name="thetao", coordinates=[
            ns(coordinate_id="time", minimum_value=ms("1987-01-01"), maximum_value=ms("2026-08-31"))])])])])])])])
    assert download.coverage_from_description(desc, "thetao").end.isoformat() == module.coverage_end(desc, "thetao")
