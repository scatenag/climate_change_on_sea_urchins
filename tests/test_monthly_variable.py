"""M1.8: the monthly variables of the catalogue, one at a time. First the pH (`ph_monthly`).

The catalogue entry is what scripts/fetch_copernicus.py downloads today for Livorno (read from the script's source,
without importing it); the monthly series is the mean over the sea cells of the box and over depth 0-10 m, a month
without a value stays missing, a conversion named in the catalogue is applied once; the product's coverage is read from the
`values` list the Copernicus catalogue gives for a monthly time axis. All against a client the tests provide, so it
runs in CI, which has neither copernicusmarine nor xarray."""
import ast
import datetime as dt
import hashlib
import json
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_change_on_sea_urchins import catalog, download
from climate_change_on_sea_urchins.download import Coverage, Credentials, Cube, DownloadError, SiteError, fetch_series, write_result
from climate_change_on_sea_urchins.study_spec import SiteSpec

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "fetch_copernicus.py"
CREDS = Credentials("mario.rossi-user", "s3cr3t-pa55-w0rd")
LIVORNO = SiteSpec(id="s1", lat=43.4278, lon=10.3956, name="Livorno Sud", bbox_delta=0.1)
CLOCK = lambda: dt.datetime(2026, 11, 20, 12, 0, tzinfo=dt.timezone.utc)


# --- the catalogue entry is what Livorno's script downloads ---------------------------------------------------------

def _script():
    consts, datasets = {}, None
    for node in ast.parse(SCRIPT.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Constant):
                consts[node.targets[0].id] = node.value.value
            elif node.targets[0].id == "DATASETS":
                datasets = ast.literal_eval(node.value)
    return consts, {d["name"]: d for d in datasets}


# catalogue id -> (the script's name for it, its variable, the column, the unit)
ENTRIES = {"ph_monthly": ("carbonate", "ph", "pH", "1"), "salinity_monthly": ("salinity", "so", "Salinity", "0.001")}


@pytest.mark.parametrize("catalog_id", list(ENTRIES))
def test_the_entry_is_what_the_livorno_script_downloads(catalog_id):
    name, variable, column, unit = ENTRIES[catalog_id]
    consts, datasets = _script()
    d, entry = datasets[name], catalog.get(catalog_id)
    assert entry.dataset_multiyear == d["dataset_id"] and entry.dataset_analysis_forecast == d["fallback_id"]
    assert [entry.variable] == d["variables"] == [variable]
    assert (entry.depth_min_m, entry.depth_max_m) == (consts["DEPTH_MIN"], consts["DEPTH_MAX"]) == (0.0, 10.0)
    assert entry.cadence == "P1M" and entry.provider == "copernicus_marine" and entry.conversion is None
    assert entry.column == column and entry.unit_native == entry.unit_analysis == unit


def test_the_domain_of_the_biogeochemical_product_is_its_own_not_the_physics_one():
    ph, sal, sst = catalog.get("ph_monthly"), catalog.get("salinity_monthly"), catalog.get("sst_daily")
    assert (ph.domain.lat_min, ph.domain.lat_max) == (30.1875, 45.97916793823242)
    assert (ph.domain.lon_min, ph.domain.lon_max) == (-5.541666507720947, 36.29166793823242)      # read from the Copernicus catalogue, 2026-11-20
    assert sal.domain == sst.domain                                                                 # the physics products share their box (2026-11-20)
    assert catalog.covers(ph, 43.4278, 10.3956) and catalog.covers(sst, 43.0, -5.8) and catalog.covers(sal, 43.0, -5.8) and not catalog.covers(ph, 43.0, -5.8)


def test_every_entry_names_its_column_and_the_columns_do_not_collide():
    columns = [v.column for v in catalog.CATALOG.values()]
    assert catalog.get("sst_daily").column == "Temperature" and len(columns) == len(set(columns))
    assert catalog.monthly_ids() == ["ph_monthly", "salinity_monthly"]


# --- the coverage of a monthly product -------------------------------------------------------------------------

def _ms(day):
    return dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() * 1000


def _description(time_coordinate, variable="ph"):
    ns = types.SimpleNamespace
    service = ns(service_name="arco-time-series", variables=[ns(short_name=variable, coordinates=[ns(coordinate_id="time", **time_coordinate)])])
    return ns(products=[ns(datasets=[ns(versions=[ns(label="202511", parts=[ns(name="default", services=[service])])])])])


def test_a_monthly_product_gives_its_months_as_a_list_and_the_coverage_is_its_first_and_last():
    values = [_ms("1999-01-01"), _ms("1999-02-01"), _ms("2026-08-01"), _ms("2026-07-01")]
    cov = download.coverage_from_description(_description(dict(minimum_value=None, maximum_value=None, values=values)), "ph")
    assert (cov.start, cov.end, cov.version) == (dt.date(1999, 1, 1), dt.date(2026, 8, 1), "202511")


def test_a_daily_product_keeps_its_minimum_and_maximum():
    cov = download.coverage_from_description(_description(dict(minimum_value=_ms("1987-01-01"), maximum_value=_ms("2026-09-30"), values=None), "thetao"), "thetao")
    assert (cov.start, cov.end) == (dt.date(1987, 1, 1), dt.date(2026, 9, 30))


def test_a_time_axis_with_neither_is_refused_with_the_reason():
    with pytest.raises(DownloadError, match="no time coverage"):
        download.coverage_from_description(_description(dict(minimum_value=None, maximum_value=None, values=None)), "ph")


# --- the download ---------------------------------------------------------------------------------------------------

@dataclass
class FakeMonthlyClient:
    coverage: Coverage = Coverage(start=dt.date(1999, 1, 1), end=dt.date(2026, 8, 1), version="202511")
    sea: callable = lambda lat, lon: True
    missing_months: tuple = ()
    calls: list = field(default_factory=list)
    fail_with: str | None = None
    step: float = 0.0625

    def coverage_of(self, variable):
        self.calls.append(("coverage", variable.id))
        return self.coverage

    def subset(self, variable, *, lat_min, lat_max, lon_min, lon_max, start, end, credentials, dataset_version=None):
        self.calls.append(("subset", variable.id, start, end, dataset_version))
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        months = pd.date_range(start, end, freq="MS")
        lats = np.arange(lat_min, lat_max + 1e-9, self.step)
        lons = np.arange(lon_min, lon_max + 1e-9, self.step)
        depths = (0.5, 3.0, 8.0)
        values = np.empty((len(months), len(depths), len(lats), len(lons)))
        for i in range(len(months)):
            for k in range(len(depths)):
                values[i, k] = 8.0 + 0.01 * i + 0.001 * k
        for a, la in enumerate(lats):
            for b, lo in enumerate(lons):
                if not self.sea(la, lo):
                    values[:, :, a, b] = np.nan
        for i, m in enumerate(months):
            if m.date() in self.missing_months:
                values[i] = np.nan
        return Cube(times=np.array(months.values.astype("datetime64[D]")), depths=np.array(depths), lats=lats, lons=lons, values=values, dataset_version="202511")


def _fetch(client=None, site=LIVORNO, **kw):
    kw.setdefault("start", dt.date(2024, 1, 1))
    kw.setdefault("end", dt.date(2024, 6, 1))
    kw.setdefault("catalog_id", "ph_monthly")
    return fetch_series(site, credentials=CREDS, client=client or FakeMonthlyClient(), now=CLOCK, **kw)


def test_the_series_is_one_value_per_month_named_as_the_catalogue_says():
    r = _fetch()
    assert list(r.series.columns) == ["Datetime", "pH"] and len(r.series) == 6
    assert r.series["Datetime"].tolist() == list(pd.date_range("2024-01-01", "2024-06-01", freq="MS"))
    first = 8.0 + 0.001 * np.mean([0, 1, 2])                          # the mean over the depths of the first month
    assert r.series["pH"].iloc[0] == pytest.approx(first) and r.series["pH"].iloc[5] == pytest.approx(first + 0.05)


def test_a_start_inside_a_month_starts_at_that_month_and_the_request_is_recorded_as_given():
    r = _fetch(start=dt.date(2024, 2, 17))
    assert r.series["Datetime"].iloc[0] == pd.Timestamp("2024-02-01") and r.manifest["period"]["requested"]["start"] == "2024-02-17"
    assert r.manifest["period"]["obtained"] == {"start": "2024-02-01", "end": "2024-06-01"}


def test_land_cells_are_left_out_and_a_missing_month_stays_missing_and_is_reported():
    r = _fetch(FakeMonthlyClient(sea=lambda la, lo: la < 43.5, missing_months=(dt.date(2024, 3, 1), dt.date(2024, 5, 1))))
    assert r.series["pH"].isna().tolist() == [False, False, True, False, True, False]
    assert r.manifest["missing_months"] == {"count": 2, "months": ["2024-03", "2024-05"], "truncated": False} and r.manifest["n_months"] == 6


def test_the_manifest_says_what_the_series_is():
    m = _fetch().manifest
    assert m["kind"] == "monthly_variable" and m["catalog_id"] == "ph_monthly"
    assert m["dataset"] == {"id": "cmems_mod_med_bgc-car_my_4.2km_P1M-m", "version": "202511", "variable": "ph", "depth_m": [0.0, 10.0]}
    assert m["period"]["product_coverage"] == {"start": "1999-01-01", "end": "2026-08-01", "version": "202511"}
    assert "n_days" not in m and m["site"]["name"] == "Livorno Sud" and m["conversion"] is None


def test_without_an_end_the_series_ends_with_the_last_month_of_the_product_and_an_end_beyond_it_is_cut():
    client = FakeMonthlyClient(coverage=Coverage(start=dt.date(1999, 1, 1), end=dt.date(2024, 4, 1), version="202511"))
    assert _fetch(client, end=None).series["Datetime"].iloc[-1] == pd.Timestamp("2024-04-01")
    cut = _fetch(client, end=dt.date(2025, 1, 1))
    assert cut.series["Datetime"].iloc[-1] == pd.Timestamp("2024-04-01") and cut.manifest["period"]["requested"]["end"] == "2025-01-01"


def test_a_start_before_the_product_and_one_after_it_are_refused():
    with pytest.raises(DownloadError, match="before the product's first"):
        _fetch(start=dt.date(1998, 12, 1))
    with pytest.raises(DownloadError, match="after the product's last"):
        _fetch(start=dt.date(2026, 9, 1), end=None)


def test_a_conversion_named_in_the_catalogue_is_applied_once(monkeypatch):
    entry = catalog.CATALOG["ph_monthly"]
    doubled = entry.model_copy(update={"conversion": catalog.Conversion(name="test doubling", factor=2.0)})
    monkeypatch.setitem(catalog.CATALOG, "ph_monthly", doubled)
    r = _fetch()
    assert r.series["pH"].iloc[0] == pytest.approx(2 * (8.0 + 0.001))
    assert r.manifest["conversion"] == {"name": "test doubling", "factor": 2.0}


def test_a_site_outside_the_domain_of_this_product_is_refused_before_any_request():
    client = FakeMonthlyClient()
    west = SiteSpec(id="w", lat=43.0, lon=-5.8, name="West", bbox_delta=0.1)            # inside the physics product, outside the BGC one
    with pytest.raises(SiteError, match=r"West.*outside"):
        _fetch(client, site=west)
    assert client.calls == []


def test_a_land_cell_is_refused_with_the_nearest_sea_named():
    with pytest.raises(SiteError, match="land"):
        _fetch(FakeMonthlyClient(sea=lambda la, lo: la > 43.7))


def test_an_error_of_the_toolbox_never_carries_the_credentials():
    with pytest.raises(DownloadError) as e:
        _fetch(FakeMonthlyClient(fail_with="authentication failed for mario.rossi-user with password s3cr3t-pa55-w0rd"))
    text = f"{e.value} {e.value.args}"
    assert "mario.rossi-user" not in text and "s3cr3t-pa55-w0rd" not in text and e.value.__cause__ is None and e.value.__context__ is None


def test_both_credentials_are_required_and_nothing_is_requested_without_them():
    client = FakeMonthlyClient()
    with pytest.raises(DownloadError, match="username and password"):
        fetch_series(LIVORNO, catalog_id="ph_monthly", start=dt.date(2024, 1, 1), credentials=Credentials("u", ""), client=client, now=CLOCK)
    assert client.calls == []


def test_the_downloaded_version_is_requested_explicitly():
    client = FakeMonthlyClient()
    _fetch(client)
    assert all(c[4] == "202511" for c in client.calls if c[0] == "subset")


def test_files_are_named_after_the_catalogue_id_and_the_sst_keeps_its_names(tmp_path):
    paths = write_result(_fetch(), tmp_path)
    assert paths["csv"].name == "ph_monthly.csv" and paths["manifest"].name == "ph_monthly.manifest.json"
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["files"] == {"ph_monthly.csv": {"sha256": hashlib.sha256(paths["csv"].read_bytes()).hexdigest(), "rows": 6}}
    assert pd.read_csv(paths["csv"], parse_dates=["Datetime"])["pH"].round(6).tolist() == _fetch().series["pH"].round(6).tolist()
    assert download.SERIES_CSV == "sst_daily.csv" and download.MANIFEST_JSON == "sst_daily.manifest.json"


def test_the_dispatcher_sends_the_daily_sst_to_its_own_function(monkeypatch):
    seen = []
    monkeypatch.setattr(download, "fetch_daily_sst", lambda *a, **k: seen.append(("daily", k["catalog_id"])) or "daily")
    assert fetch_series(LIVORNO, catalog_id="sst_daily", start=dt.date(2024, 1, 1), credentials=CREDS, client=None) == "daily" and seen == [("daily", "sst_daily")]


def test_the_command_downloads_a_monthly_variable_with_the_variable_option(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "mario.rossi-user")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "s3cr3t-pa55-w0rd")
    download.main(["--lat", "43.4278", "--lon", "10.3956", "--name", "Livorno Sud", "--variable", "ph_monthly", "--start", "2024-01-01",
                   "--end", "2024-06-01", "--out", str(tmp_path)], client=FakeMonthlyClient())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ph_monthly.csv", "ph_monthly.manifest.json"]
    out = capsys.readouterr().out
    assert "6 months 2024-01-01 to 2024-06-01" in out and "0 missing" in out and "s3cr3t" not in out


def test_the_command_still_downloads_the_daily_sst_by_default(tmp_path, monkeypatch):
    from test_download import FakeClient
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "mario.rossi-user")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "s3cr3t-pa55-w0rd")
    download.main(["--lat", "43.4278", "--lon", "10.3956", "--start", "2024-01-01", "--end", "2024-01-05", "--out", str(tmp_path)], client=FakeClient())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sst_daily.csv", "sst_daily.manifest.json"]


def test_the_mean_is_over_the_cells_per_depth_and_then_over_depth_not_over_every_value():
    # two cells, two depths: one cell has no value at the deeper one. Per depth: (8 + 12) / 2 = 10 and 8; then 9. Over every value: 9.33.
    values = np.array([[[[8.0, 12.0]], [[8.0, np.nan]]]])                      # (time 1, depth 2, lat 1, lon 2)
    cube = Cube(times=np.array(["2024-01-01"], dtype="datetime64[D]"), depths=np.array([0.5, 5.0]), lats=np.array([43.4]),
                lons=np.array([10.3, 10.4]), values=values, dataset_version="202511")
    s = download._monthly_series(cube, dt.date(2024, 1, 1), dt.date(2024, 1, 1), 1.0)
    assert s.iloc[0] == pytest.approx(9.0)


def test_a_cube_with_two_times_in_a_month_is_a_download_error_not_a_crash():
    cube = Cube(times=np.array(["2024-01-01", "2024-01-15"], dtype="datetime64[D]"), depths=np.array([0.5]), lats=np.array([43.4]),
                lons=np.array([10.4]), values=np.ones((2, 1, 1, 1)), dataset_version="202511")
    with pytest.raises(DownloadError, match="more than one time in 2024-01"):
        download._monthly_series(cube, dt.date(2024, 1, 1), dt.date(2024, 1, 1), 1.0)


def test_a_daily_variable_is_refused_by_the_monthly_download():
    with pytest.raises(DownloadError, match="not a monthly variable"):
        download.fetch_monthly(LIVORNO, catalog_id="sst_daily", start=dt.date(2024, 1, 1), credentials=CREDS, client=FakeMonthlyClient(), now=CLOCK)
