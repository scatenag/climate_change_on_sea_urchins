"""The adapter to the Copernicus toolbox (CopernicusClient), against a stand-in toolbox that writes a real
NetCDF file. Needs xarray and h5netcdf (extra `acquisition`): skipped where they are not installed, as in
the CI -- the logic around the adapter is tested in test_download.py, which runs everywhere.

What matters here: the credentials reach `subset()` as arguments and nowhere else; `login` is never used
(the stand-in toolbox has none, so a call would fail the test); no credentials file or other file appears
in the home directory; the process environment is left as it was; the NetCDF becomes the Cube the logic
expects.
"""
import datetime as dt
import os
import types

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("h5netcdf")

from climate_change_on_sea_urchins import catalog
from climate_change_on_sea_urchins.download import Credentials, CopernicusClient, fetch_daily_sst
from climate_change_on_sea_urchins.study_spec import SiteSpec

SITE = SiteSpec(id="s", lat=43.4278, lon=10.3956, name="Livorno Sud", bbox_delta=0.1)
USER, PASSWORD = "mario.rossi-user", "s3cr3t-pa55-w0rd"


class StandInToolbox:
    """Has subset() and describe() only: no login, no credentials file handling."""
    __version__ = "0.0-test"

    def __init__(self):
        self.subset_calls = []

    def describe(self, dataset_id, disable_progress_bar=True):
        ns = types.SimpleNamespace
        ms = lambda d: dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp() * 1000
        return ns(products=[ns(datasets=[ns(versions=[ns(label="202511", parts=[ns(services=[
            ns(service_name="arco-time-series", variables=[ns(short_name="thetao", coordinates=[
                ns(coordinate_id="time", minimum_value=ms("1987-01-01"), maximum_value=ms("2026-08-31"))])])])])])])])

    def subset(self, **kw):
        self.subset_calls.append(kw)
        days = np.arange(np.datetime64(kw["start_datetime"]), np.datetime64(kw["end_datetime"]) + 1, dtype="datetime64[D]")
        lats = np.arange(kw["minimum_latitude"], kw["maximum_latitude"] + 1e-9, 0.0625)
        lons = np.arange(kw["minimum_longitude"], kw["maximum_longitude"] + 1e-9, 0.0625)
        depths = np.array([0.5, 2.0])
        data = np.ones((len(days), len(depths), len(lats), len(lons))) * 15.0
        data[:, 1] += 1.0
        ds = xr.Dataset({"thetao": (("time", "depth", "latitude", "longitude"), data)},
                        coords={"time": days.astype("datetime64[ns]"), "depth": depths, "latitude": lats, "longitude": lons})
        ds.to_netcdf(os.path.join(kw["output_directory"], kw["output_filename"]), engine="h5netcdf")


def test_credentials_go_to_subset_as_arguments_and_nowhere_else(tmp_path, monkeypatch):
    home = tmp_path / "home"; home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for k in [k for k in os.environ if k.startswith("COPERNICUSMARINE")]:
        monkeypatch.delenv(k)
    env_before = dict(os.environ)

    toolbox = StandInToolbox()
    client = CopernicusClient(toolbox=toolbox, xarray_module=xr)
    result = fetch_daily_sst(SITE, start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 3),
                             credentials=Credentials(USER, PASSWORD), client=client)

    assert toolbox.subset_calls, "the toolbox was never called"
    for kw in toolbox.subset_calls:
        assert kw["username"] == USER and kw["password"] == PASSWORD
        assert "credentials_file" not in kw
        assert kw["dataset_id"] == catalog.get("sst_daily").dataset_multiyear
        assert kw["dataset_version"] == "202511", "the version whose coverage was read must be the one requested"
        assert (kw["minimum_depth"], kw["maximum_depth"]) == (0.0, 5.0)
    assert dict(os.environ) == env_before, "the process environment was changed"
    assert list(home.iterdir()) == [], "something was written in the home directory"
    assert result.series["Temperature"].tolist() == pytest.approx([15.5, 15.5, 15.5])
    assert result.manifest["versions"]["copernicusmarine"] == "0.0-test"
    assert USER not in str(result.manifest) and PASSWORD not in str(result.manifest)


def test_the_series_comes_from_the_catalogue_dataset_and_the_toolbox_coverage(monkeypatch):
    toolbox = StandInToolbox()
    client = CopernicusClient(toolbox=toolbox, xarray_module=xr)
    result = fetch_daily_sst(SITE, start=dt.date(2026, 8, 30), end=None, credentials=Credentials(USER, PASSWORD), client=client)
    assert result.manifest["period"]["obtained"]["end"] == "2026-08-31"
    assert result.manifest["dataset"]["version"] == "202511"


def test_the_standin_has_no_login():
    assert not hasattr(StandInToolbox(), "login")
