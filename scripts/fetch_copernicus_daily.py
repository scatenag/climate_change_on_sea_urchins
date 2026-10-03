"""
Download daily SST from Copernicus Marine for Marine Heatwave detection.

marineHeatWaves.detect() requires daily data — monthly data is NOT sufficient.

Site: off Livorno / North Tyrrhenian Sea (43.4278°N, 10.3956°E), surface layer
Period: 2003-01-01 → the last day the multiyear dataset covers, read from the Copernicus
catalogue at every run (product_end() below). Never a date written in the code: a literal end
date froze the series twice, two years behind in 2026-07 (commit 09b928a) and at 2026-06-30
afterwards. The analysis-forecast fallback is downloaded up to the same end, so its forecast
days never enter the series as observations.

Output: data/sst_daily.csv
        Columns: Datetime (daily), Temperature

Dataset: cmems_mod_med_phy-temp_my_4.2km_P1D-m (daily temperature)
Fallback: cmems_mod_med_phy-temp_anfc_4.2km_P1D-m
"""

import datetime as dt
import os
import sys
import copernicusmarine
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SITE_LAT, SITE_LON, BBOX_DELTA

DEPTH_MIN = 0.0
DEPTH_MAX = 5.0    # surface only for SST
START = "2003-01-01"

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).parent.parent / "data" / "sst_daily.csv"

DATASET_ID          = "cmems_mod_med_phy-temp_my_4.2km_P1D-m"
DATASET_ID_FALLBACK = "cmems_mod_med_phy-temp_anfc_4.2km_P1D-m"
RAW_FILE = RAW_DIR / "raw_temperature_daily.nc"


def coverage_end(description, variable: str) -> str:
    """Last day (ISO date) for which `variable` is covered, from a
    copernicusmarine.describe() result. When the dataset's services report different
    ends, the earliest is used: the stretch every service covers."""
    ends = []
    for product in description.products:
        for dataset in product.datasets:
            for version in dataset.versions:
                for part in version.parts:
                    for service in part.services:
                        for var in service.variables:
                            if var.short_name != variable:
                                continue
                            for coord in var.coordinates:
                                if coord.coordinate_id == "time" and coord.maximum_value is not None:
                                    ends.append(coord.maximum_value)
    if not ends:
        raise RuntimeError(f"the catalogue reports no time coverage for {variable!r}")
    # The catalogue gives time bounds in milliseconds since the epoch.
    end = dt.datetime.fromtimestamp(min(ends) / 1000, dt.timezone.utc)
    return end.date().isoformat()


def product_end(dataset_id: str = DATASET_ID) -> str:
    """Last day covered by the multiyear daily dataset, as the Copernicus catalogue
    declares it now."""
    description = copernicusmarine.describe(dataset_id=dataset_id, disable_progress_bar=True)
    return coverage_end(description, "thetao")


def download():
    if RAW_FILE.exists():
        print(f"Daily SST file already exists: {RAW_FILE}. Skipping download.")
        return

    if not (
        os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
        and os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    ):
        raise RuntimeError(
            "COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD "
            "not set. Without them copernicusmarine falls back to an interactive "
            "prompt that hangs/aborts on a non-interactive shell (e.g. CI)."
        )

    end = product_end()
    print(f"Multiyear daily SST covers up to {end} (Copernicus catalogue)")

    for dataset_id in [DATASET_ID, DATASET_ID_FALLBACK]:
        try:
            print(f"Downloading daily SST from {dataset_id} ({START} → {end}) ...")
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=["thetao"],
                minimum_longitude=SITE_LON - BBOX_DELTA,
                maximum_longitude=SITE_LON + BBOX_DELTA,
                minimum_latitude=SITE_LAT - BBOX_DELTA,
                maximum_latitude=SITE_LAT + BBOX_DELTA,
                minimum_depth=DEPTH_MIN,
                maximum_depth=DEPTH_MAX,
                start_datetime=START,
                end_datetime=end,
                output_filename="raw_temperature_daily.nc",
                output_directory=str(RAW_DIR),
                force_download=False,
            )
            if not RAW_FILE.exists():
                raise RuntimeError(f"subset() returned but {RAW_FILE} was not created")
            print(f"Saved to {RAW_FILE}")
            return
        except Exception as e:
            print(f"Failed ({dataset_id}): {e}")

    raise RuntimeError("Could not download daily SST from any dataset.")


def nc_to_daily_series() -> pd.DataFrame:
    ds = xr.open_dataset(RAW_FILE, engine="h5netcdf")
    da = ds["thetao"]

    # Spatial mean over bounding box
    spatial_dims = [d for d in da.dims if d in ("latitude", "longitude", "lat", "lon")]
    if spatial_dims:
        da = da.mean(dim=spatial_dims)

    # Depth mean over surface layer
    depth_dims = [d for d in da.dims if d in ("depth", "elevation")]
    if depth_dims:
        da = da.mean(dim=depth_dims)

    series = da.to_series()
    series.index = pd.to_datetime(series.index)
    series.name = "Temperature"

    # Ensure complete daily index (fill gaps with interpolation)
    full_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq="D")
    series = series.reindex(full_idx)
    n_gaps = series.isna().sum()
    if n_gaps > 0:
        print(f"  Interpolating {n_gaps} missing daily values...")
        series = series.interpolate(method="time")

    df = series.reset_index()
    df.columns = ["Datetime", "Temperature"]
    return df.sort_values("Datetime").reset_index(drop=True)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    download()
    df = nc_to_daily_series()
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df)} daily rows to {OUT_PATH}")
    print(f"  Period: {df['Datetime'].min().date()} → {df['Datetime'].max().date()}")
    print(f"  Temperature range: {df['Temperature'].min():.2f} – {df['Temperature'].max():.2f} °C")


if __name__ == "__main__":
    main()
