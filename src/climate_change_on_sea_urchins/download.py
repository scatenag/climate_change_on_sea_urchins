"""Download of the daily SST for a site, from the variable catalogue (M1.2).

What is downloaded, and where the series ends, come from the catalogue (catalog.py) and from the
Copernicus catalogue's coverage of the product, never from a date written in the code (the daily SST
stopped at a hand-written end date twice, issue #45). The user's credentials are arguments of the calls
that need them: they are never read from the environment, never written to a file, never part of an
error message, and `copernicusmarine login` (which writes them to `~/.copernicusmarine`) is never used
(docs/roadmap/MILESTONE-M1.md, section 3).

The logic here works on a `Cube` (a numpy block of values) obtained from a `client`. The real client,
`CopernicusClient`, is a thin adapter to the Copernicus toolbox, imported only when used; tests and the
CI provide their own client, because the CI has neither copernicusmarine nor xarray.

A site that cannot work is refused with the reason, before any multi-day download: outside the
catalogue's domain; a land cell (the nearest sea cell is named); a point inside the domain's bounding box
but in no sea the product covers. A day without data stays missing: no interpolation (a missing day makes
the month missing, decision of 2026-10-03; the dataset builder applies it, M1.4).

The update job keeps its own script (scripts/fetch_copernicus_daily.py) until it moves to ccsu-run-study
(M1.9); a test keeps the two readings of the product's coverage equal.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from . import catalog
from .catalog import CatalogVariable

PROBE_DELTA_DEG = 1.0     # half-width of the box used to tell a land cell from a point outside the product
SERIES_CSV = "sst_daily.csv"
MANIFEST_JSON = "sst_daily.manifest.json"
MAX_LISTED_MISSING_DAYS = 200


class DownloadError(Exception):
    """A download that cannot be completed. The message never contains the credentials."""


class SiteError(DownloadError):
    """The site cannot be served by the product."""


@dataclass(frozen=True)
class Credentials:
    username: str | None
    password: str | None

    def __repr__(self) -> str:      # never shown, not even by a log line that prints the object
        return "Credentials(username=***, password=***)"

    __str__ = __repr__


@dataclass(frozen=True)
class Coverage:
    """The time range of the dataset version the toolbox downloads, as the Copernicus catalogue declares
    it, and the coverage of the dataset's other versions (recorded, never merged into this one)."""
    start: dt.date
    end: dt.date
    version: str | None = None
    other_versions: dict = field(default_factory=dict)


@dataclass
class Cube:
    """Values of one variable on a block: (time, depth, latitude, longitude); NaN where there is no sea
    or no value."""
    times: np.ndarray
    depths: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    values: np.ndarray
    dataset_version: str | None = None


class Client(Protocol):
    def coverage_of(self, variable: CatalogVariable) -> Coverage: ...

    def subset(self, variable: CatalogVariable, *, lat_min: float, lat_max: float, lon_min: float,
               lon_max: float, start: dt.date, end: dt.date, credentials: Credentials,
               dataset_version: str | None = None) -> Cube: ...


@dataclass
class Result:
    series: pd.DataFrame
    manifest: dict


# ── credentials and errors ────────────────────────────────────────────────────────────────────────

def _check_credentials(c: Credentials) -> None:
    ok = all(isinstance(v, str) and v.strip() for v in (c.username, c.password))
    if not ok:
        raise DownloadError(
            "Both the Copernicus username and password are required; they are not read from the "
            "environment or from any file.")


def _mask(text: str, c: Credentials) -> str:
    for secret in (c.username, c.password):
        if isinstance(secret, str) and secret:
            text = text.replace(secret, "***")
    return text


def _call(func: Callable, c: Credentials, what: str):
    """Run a client call; any failure becomes a DownloadError with the credentials masked. The error is
    raised outside the `except` block so no chained exception can carry the original message."""
    message = None
    try:
        return func()
    except DownloadError:
        raise
    except Exception as e:
        message = f"{type(e).__name__}: {e}"
    raise DownloadError(f"{what} failed: {_mask(message, c)}")


# ── the product's coverage ────────────────────────────────────────────────────────────────────────

def _ms_to_date(ms: float) -> dt.date:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date()


def _version_coverage(version, variable: str):
    """(start, end) in ms of `variable` in one dataset version: the stretch every service of it covers
    (latest start, earliest end); None if the version does not list it."""
    starts, ends = [], []
    for part in version.parts:
        for service in part.services:
            for var in service.variables:
                if var.short_name != variable:
                    continue
                for coord in var.coordinates:
                    if coord.coordinate_id == "time":
                        if coord.minimum_value is not None:
                            starts.append(coord.minimum_value)
                        if coord.maximum_value is not None:
                            ends.append(coord.maximum_value)
    return (max(starts), min(ends)) if starts and ends else None


def coverage_from_description(description, variable: str) -> Coverage:
    """Coverage of `variable` from a copernicusmarine.describe() result, for the dataset version that
    `subset()` downloads: the toolbox takes the first version of the dataset's list unless one is forced
    (it is, below, with the label read here). A merge over versions would let an old version that ended
    earlier cut the series without warning. The other versions are returned as `other_versions`."""
    chosen, others = None, {}
    for product in description.products:
        for dataset in product.datasets:
            for position, version in enumerate(dataset.versions):
                span = _version_coverage(version, variable)
                if span is None:
                    continue
                if position == 0:
                    chosen = (getattr(version, "label", None), span)
                else:
                    others[getattr(version, "label", str(position))] = {
                        "start": _ms_to_date(span[0]).isoformat(), "end": _ms_to_date(span[1]).isoformat()}
    if chosen is None:
        raise DownloadError(f"the Copernicus catalogue reports no time coverage for {variable!r} "
                            "in the version that would be downloaded")
    label, (lo, hi) = chosen
    return Coverage(start=_ms_to_date(lo), end=_ms_to_date(hi), version=label, other_versions=others)


# ── the site ─────────────────────────────────────────────────────────────────────────────────────

def _sea_cells(cube: Cube) -> np.ndarray:
    """Boolean (lat, lon): the cell has a value at some time and depth of the cube."""
    return np.isfinite(cube.values).any(axis=(0, 1))


def _distance_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _site_problem(probe: Cube, site, variable: CatalogVariable) -> str | None:
    """None when the probe shows sea inside the site's own box (so an empty first day was a gap in the
    data, not land); otherwise the reason the site cannot be served."""
    sea = _sea_cells(probe)
    la, lo = np.meshgrid(probe.lats, probe.lons, indexing="ij")
    in_box = (np.abs(la - site.lat) <= site.bbox_delta + 1e-9) & (np.abs(lo - site.lon) <= site.bbox_delta + 1e-9)
    if (sea & in_box).any():
        return None
    where = f"{site.name!r} (lat {site.lat}, lon {site.lon})"
    if sea.any():
        d = np.array([[_distance_km(site.lat, site.lon, float(a), float(b)) for b in probe.lons] for a in probe.lats])
        d = np.where(sea, d, np.inf)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        return (f"site {where}: the grid cells around it are land in the product ({variable.id}); the nearest sea "
                f"cell is at lat {probe.lats[i]:.3f}, lon {probe.lons[j]:.3f}, about {d[i, j]:.0f} km away. "
                "Move the site to the sea, or to that cell.")
    km = PROBE_DELTA_DEG * 111
    return (f"site {where}: no sea cell of the product ({variable.id}) within about {km:.0f} km. The point is "
            "inland, or in a sea the product does not cover although it lies inside the bounding box of its "
            "domain; the product's land/sea mask cannot tell the two apart (both are non-sea cells), so this "
            "check does not choose.")


# ── the download ─────────────────────────────────────────────────────────────────────────────────

def _daily_series(cube: Cube, start: dt.date, end: dt.date) -> pd.Series:
    """Spatial mean per depth, then the mean over depth (as scripts/fetch_copernicus_daily.py does); a day
    without any value is NaN, never filled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)        # all-NaN slices are the missing days
        per_depth = np.nanmean(cube.values, axis=(2, 3))
        daily = np.nanmean(per_depth, axis=1)
    s = pd.Series(daily, index=pd.DatetimeIndex(cube.times.astype("datetime64[ns]")))
    return s.reindex(pd.date_range(start, end, freq="D"))


def _package_version() -> str:
    try:
        return metadata.version("climate_change_on_sea_urchins")
    except metadata.PackageNotFoundError:
        return "unknown"


def fetch_daily_sst(site, *, start: dt.date, end: dt.date | None = None, credentials: Credentials,
                    client: Client, catalog_id: str = "sst_daily",
                    now: Callable[[], dt.datetime] | None = None) -> Result:
    """Download and reduce the daily SST of `site`. `end` None means up to the product's last day; an
    `end` beyond it is cut there, and both are recorded in the manifest."""
    now = now or (lambda: dt.datetime.now(dt.timezone.utc))
    _check_credentials(credentials)
    variable = catalog.get(catalog_id)
    d = variable.domain
    if not catalog.covers(variable, site.lat, site.lon):
        raise SiteError(
            f"site {site.name!r} (lat {site.lat}, lon {site.lon}) is outside the domain of the "
            f"{variable.id!r} product (lat {d.lat_min:.2f} to {d.lat_max:.2f}, lon {d.lon_min:.2f} to {d.lon_max:.2f}).")

    coverage = _call(lambda: client.coverage_of(variable), credentials, "reading the product's coverage")
    if start < coverage.start:
        raise DownloadError(f"the period starts on {start}, before the product's first day {coverage.start}.")
    if start > coverage.end:
        raise DownloadError(f"the period starts on {start}, after the product's last day {coverage.end}.")
    obtained_end = coverage.end if end is None else min(end, coverage.end)
    if obtained_end < start:
        raise DownloadError(f"the period ends on {end}, before it starts on {start}.")

    box = dict(lat_min=site.lat - site.bbox_delta, lat_max=site.lat + site.bbox_delta,
               lon_min=site.lon - site.bbox_delta, lon_max=site.lon + site.bbox_delta)
    # One day first: a site that cannot work is refused before a multi-year download.
    ver = coverage.version
    first = _call(lambda: client.subset(variable, start=start, end=start, credentials=credentials,
                                        dataset_version=ver, **box),
                  credentials, "downloading the first day")
    if not _sea_cells(first).any():
        wide = dict(lat_min=site.lat - PROBE_DELTA_DEG, lat_max=site.lat + PROBE_DELTA_DEG,
                    lon_min=site.lon - PROBE_DELTA_DEG, lon_max=site.lon + PROBE_DELTA_DEG)
        probe = _call(lambda: client.subset(variable, start=start, end=start, credentials=credentials,
                                            dataset_version=ver, **wide),
                      credentials, "probing the surroundings of the site")
        problem = _site_problem(probe, site, variable)
        if problem:
            raise SiteError(problem)

    cube = _call(lambda: client.subset(variable, start=start, end=obtained_end, credentials=credentials,
                                       dataset_version=ver, **box),
                 credentials, "downloading the series")
    series = _daily_series(cube, start, obtained_end)
    missing = [x.date().isoformat() for x in series.index[series.isna()]]
    sea = _sea_cells(cube)
    versions = {"climate_change_on_sea_urchins": _package_version()}
    versions.update(getattr(client, "versions", lambda: {})())

    manifest = {
        "format": 1,
        "kind": "daily_sst",
        "catalog_id": variable.id,
        "dataset": {"id": variable.dataset_multiyear, "version": coverage.version,
                    "variable": variable.variable, "depth_m": [variable.depth_min_m, variable.depth_max_m]},
        "site": {"id": site.id, "name": site.name, "lat": site.lat, "lon": site.lon, "bbox_delta": site.bbox_delta},
        "period": {
            "requested": {"start": start.isoformat(), "end": end.isoformat() if end else None},
            "obtained": {"start": start.isoformat(), "end": obtained_end.isoformat()},
            "product_coverage": {"start": coverage.start.isoformat(), "end": coverage.end.isoformat(),
                                 "version": coverage.version},
            "other_versions_coverage": coverage.other_versions,
        },
        "cells": {"in_box": int(sea.size), "sea_with_data": int(sea.sum())},
        "n_days": int(len(series)),
        "missing_days": {"count": len(missing), "days": missing[:MAX_LISTED_MISSING_DAYS],
                         "truncated": len(missing) > MAX_LISTED_MISSING_DAYS},
        "downloaded_at": now().isoformat(),
        "versions": versions,
    }
    df = pd.DataFrame({"Datetime": series.index, "Temperature": series.to_numpy(float)})
    return Result(series=df, manifest=manifest)


def write_result(result: Result, out_dir: str | Path) -> dict[str, Path]:
    """Write the series and its manifest (with the series' hash) into `out_dir`: nothing else."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / SERIES_CSV
    result.series.to_csv(csv, index=False)
    manifest = json.loads(json.dumps(result.manifest))
    manifest["files"] = {SERIES_CSV: {"sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
                                       "rows": int(len(result.series))}}
    path = out / MANIFEST_JSON
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"csv": csv, "manifest": path}


# ── the real client ──────────────────────────────────────────────────────────────────────────────

class CopernicusClient:
    """Adapter to the Copernicus Marine toolbox. The credentials go to `subset()` as arguments: with
    them given the toolbox neither reads nor writes a credentials file."""

    def __init__(self, toolbox=None, xarray_module=None):
        if toolbox is None:
            import copernicusmarine as toolbox     # extra `acquisition`; not installed in the CI
        if xarray_module is None:
            import xarray as xarray_module
        self._tb = toolbox
        self._xr = xarray_module

    def versions(self) -> dict:
        return {"copernicusmarine": getattr(self._tb, "__version__", "unknown")}

    def coverage_of(self, variable: CatalogVariable) -> Coverage:
        description = self._tb.describe(dataset_id=variable.dataset_multiyear, disable_progress_bar=True)
        return coverage_from_description(description, variable.variable)

    def subset(self, variable, *, lat_min, lat_max, lon_min, lon_max, start, end, credentials,
               dataset_version=None) -> Cube:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._tb.subset(
                dataset_id=variable.dataset_multiyear, dataset_version=dataset_version,
                variables=[variable.variable],
                minimum_latitude=lat_min, maximum_latitude=lat_max,
                minimum_longitude=lon_min, maximum_longitude=lon_max,
                minimum_depth=variable.depth_min_m, maximum_depth=variable.depth_max_m,
                start_datetime=start.isoformat(), end_datetime=end.isoformat(),
                output_filename="cube.nc", output_directory=tmp, overwrite=True,
                disable_progress_bar=True,
                username=credentials.username, password=credentials.password,
            )
            with self._xr.open_dataset(Path(tmp) / "cube.nc", engine="h5netcdf") as ds:
                da = ds[variable.variable]
                if "depth" not in da.dims:
                    da = da.expand_dims("depth")
                da = da.transpose("time", "depth", "latitude", "longitude")
                return Cube(times=np.asarray(da["time"].values).astype("datetime64[D]"),
                            depths=np.asarray(da["depth"].values), lats=np.asarray(da["latitude"].values),
                            lons=np.asarray(da["longitude"].values), values=np.asarray(da.values, dtype=float),
                            dataset_version=ds.attrs.get("product_version") or ds.attrs.get("version"))


# ── command line ─────────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None, client: Client | None = None) -> None:
    """`ccsu-download-sst`: download the daily SST of a point into a folder (series + manifest).
    The credentials are asked for here (or taken from this process's own environment), then passed to
    the download as arguments; `copernicusmarine login` is never used."""
    import argparse
    import getpass
    import os
    import sys
    from .study_spec import SiteSpec

    ap = argparse.ArgumentParser(prog="ccsu-download-sst", description=main.__doc__)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--name", default="site")
    ap.add_argument("--bbox-delta", type=float, default=0.1)
    ap.add_argument("--start", type=dt.date.fromisoformat, required=True)
    ap.add_argument("--end", type=dt.date.fromisoformat, default=None, help="default: the product's last day")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    user = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") or input("Copernicus username: ")
    password = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD") or getpass.getpass("Copernicus password: ")
    site = SiteSpec(id="site", lat=args.lat, lon=args.lon, name=args.name, bbox_delta=args.bbox_delta)
    try:
        result = fetch_daily_sst(site, start=args.start, end=args.end, credentials=Credentials(user, password),
                                 client=client or CopernicusClient())
    except DownloadError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)
    paths = write_result(result, args.out)
    m = result.manifest
    print(f"{m['n_days']} days {m['period']['obtained']['start']} to {m['period']['obtained']['end']}, "
          f"{m['missing_days']['count']} missing; written to {paths['csv']} and {paths['manifest']}")
