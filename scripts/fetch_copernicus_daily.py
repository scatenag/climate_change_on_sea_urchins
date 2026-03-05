"""
Download daily SST from Copernicus Marine for Marine Heatwave detection.

marineHeatWaves.detect() requires daily data — monthly data is NOT sufficient.

Site: Golfo di La Spezia / Mar Ligure (~44.1°N, 9.8°E), surface layer
Period: 2003-01-01 → 2024-04-30

Output: data/sst_daily.csv
        Columns: Datetime (daily), Temperature

Dataset: cmems_mod_med_phy-tem_my_4.2km_P1D-m (daily temperature)
Fallback: cmems_mod_med_phy-tem_anfc_4.2km_P1D-m
"""

import copernicusmarine
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

SITE_LAT = 44.1
SITE_LON = 9.8
BBOX_DELTA = 0.1
DEPTH_MIN = 0.0
DEPTH_MAX = 5.0    # surface only for SST
START = "2003-01-01"
END   = "2024-04-30"

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).parent.parent / "data" / "sst_daily.csv"

DATASET_ID          = "cmems_mod_med_phy-tem_my_4.2km_P1D-m"
DATASET_ID_FALLBACK = "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m"
RAW_FILE = RAW_DIR / "raw_temperature_daily.nc"


def download():
    if RAW_FILE.exists():
        print(f"Daily SST file already exists: {RAW_FILE}. Skipping download.")
        return

    for dataset_id in [DATASET_ID, DATASET_ID_FALLBACK]:
        try:
            print(f"Downloading daily SST from {dataset_id} ...")
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
                end_datetime=END,
                output_filename="raw_temperature_daily.nc",
                output_directory=str(RAW_DIR),
                force_download=False,
            )
            print(f"Saved to {RAW_FILE}")
            return
        except Exception as e:
            print(f"Failed ({dataset_id}): {e}")

    raise RuntimeError("Could not download daily SST from any dataset.")


def nc_to_daily_series() -> pd.DataFrame:
    ds = xr.open_dataset(RAW_FILE)
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
