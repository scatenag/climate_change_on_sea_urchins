"""
Incremental Copernicus update — downloads only months not yet in env_copernicus.csv.

Intended for use in CI (GitHub Actions, monthly schedule).
Reads the existing CSV, finds the last date, and fetches only new months.
Much faster than fetch_copernicus.py which downloads the full period.

Usage:
    python scripts/fetch_copernicus_update.py

Authentication via env vars:
    COPERNICUSMARINE_SERVICE_USERNAME
    COPERNICUSMARINE_SERVICE_PASSWORD
"""

import io
import tempfile
from pathlib import Path
from datetime import date

import copernicusmarine
import xarray as xr
import pandas as pd
import numpy as np

# ── Site config (must match fetch_copernicus.py) ──────────────────────────────
SITE_LAT   = 43.4278
SITE_LON   = 10.3956
BBOX_DELTA = 0.1
DEPTH_MIN  = 0.0
DEPTH_MAX  = 10.0

OUT_PATH = Path(__file__).parent.parent / "data" / "env_copernicus.csv"

DATASETS = [
    {"name": "temperature", "my_id": "cmems_mod_med_phy-temp_my_4.2km_P1M-m",
     "anfc_id": "cmems_mod_med_phy-tem_anfc_4.2km_P1M-m",
     "variable": "thetao", "column": "Temperature"},
    {"name": "salinity",    "my_id": "cmems_mod_med_phy-sal_my_4.2km_P1M-m",
     "anfc_id": "cmems_mod_med_phy-sal_anfc_4.2km_P1M-m",
     "variable": "so",      "column": "Salinity"},
    {"name": "oxygen",      "my_id": "cmems_mod_med_bgc-bio_my_4.2km_P1M-m",
     "anfc_id": "cmems_mod_med_bgc-bio_anfc_4.2km_P1M-m",
     "variable": "o2",      "column": "O2"},
    {"name": "ph",          "my_id": "cmems_mod_med_bgc-car_my_4.2km_P1M-m",
     "anfc_id": "cmems_mod_med_bgc-car_anfc_4.2km_P1M-m",
     "variable": "ph",      "column": "pH"},
    {"name": "co2",         "my_id": "cmems_mod_med_bgc-co2_my_4.2km_P1M-m",
     "anfc_id": "cmems_mod_med_bgc-co2_anfc_4.2km_P1M-m",
     "variable": "spco2",   "column": "CO2"},
]


def fetch_variable(cfg: dict, start: str, end: str) -> pd.Series:
    """Fetch a single variable for the given period; tries MY first, then ANFC."""
    for dataset_id in [cfg["my_id"], cfg["anfc_id"]]:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fname = f"tmp_{cfg['name']}.nc"
                copernicusmarine.subset(
                    dataset_id=dataset_id,
                    variables=[cfg["variable"]],
                    minimum_longitude=SITE_LON - BBOX_DELTA,
                    maximum_longitude=SITE_LON + BBOX_DELTA,
                    minimum_latitude=SITE_LAT - BBOX_DELTA,
                    maximum_latitude=SITE_LAT + BBOX_DELTA,
                    minimum_depth=DEPTH_MIN,
                    maximum_depth=DEPTH_MAX,
                    start_datetime=start,
                    end_datetime=end,
                    output_filename=fname,
                    output_directory=tmp,
                    force_download=True,
                )
                ds  = xr.open_dataset(f"{tmp}/{fname}", engine="h5netcdf")
                da  = ds[cfg["variable"]]
                for dim in [d for d in da.dims if d in ("latitude","longitude","lat","lon","depth","elevation")]:
                    da = da.mean(dim=dim)
                s = da.to_series()
                s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
                s.name = cfg["column"]
                print(f"  [{cfg['name']}] {dataset_id}: {len(s)} months fetched")
                return s.sort_index()
        except Exception as e:
            print(f"  [{cfg['name']}] {dataset_id} failed: {e}")
    raise RuntimeError(f"Could not fetch {cfg['name']}")


def main():
    # Read existing CSV to find the last date
    if not OUT_PATH.exists():
        print("env_copernicus.csv not found — run fetch_copernicus.py for initial download.")
        raise SystemExit(1)

    existing = pd.read_csv(OUT_PATH, parse_dates=["Datetime"])
    last_date = existing["Datetime"].max()
    # Start from the month after the last one in the file
    start_ts  = last_date + pd.DateOffset(months=1)
    # End: current month (Copernicus has ~1 month lag, but try anyway)
    end_ts    = pd.Timestamp(date.today().strftime("%Y-%m-01"))

    if start_ts >= end_ts:
        print(f"env_copernicus.csv is already up to {last_date.strftime('%Y-%m')} — nothing to do.")
        return

    start_str = start_ts.strftime("%Y-%m-%d")
    end_str   = end_ts.strftime("%Y-%m-%d")
    print(f"Fetching new Copernicus data: {start_str} → {end_str}\n")

    new_series = []
    for cfg in DATASETS:
        try:
            s = fetch_variable(cfg, start_str, end_str)
            new_series.append(s)
        except RuntimeError as e:
            print(f"  WARNING: {e} — column will be NaN for new months")

    if not new_series:
        print("No new data fetched.")
        return

    new_df = pd.concat(new_series, axis=1).reset_index().rename(columns={"index": "Datetime"})
    # Ensure all expected columns exist
    for cfg in DATASETS:
        if cfg["column"] not in new_df.columns:
            new_df[cfg["column"]] = np.nan

    # Filter to only truly new months
    new_df = new_df[new_df["Datetime"] > last_date]
    if new_df.empty:
        print("No months newer than existing data — done.")
        return

    updated = pd.concat([existing, new_df], ignore_index=True)
    updated = updated.sort_values("Datetime").drop_duplicates(subset=["Datetime"])
    updated.to_csv(OUT_PATH, index=False)
    print(f"\n✓ Appended {len(new_df)} new month(s) to {OUT_PATH}")
    print(f"  New range: {updated['Datetime'].min().strftime('%Y-%m')} → "
          f"{updated['Datetime'].max().strftime('%Y-%m')}")


if __name__ == "__main__":
    main()
