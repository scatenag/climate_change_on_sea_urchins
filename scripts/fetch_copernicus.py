"""
Download monthly environmental variables from Copernicus Marine Service.

Site: Golfo di La Spezia / Mar Ligure (43.4278°N, 10.3956°E), depth 0–10m
      Bounding box ±0.1° averaged over marine grid cells only (4.2km model).
Period: 2003-01-01 → present (full period, not just update)

Output: data/env_copernicus.csv
        Columns: Datetime, Temperature, Salinity, O2, pH, CO2

Requirements:
    pip install copernicusmarine xarray netcdf4

Authentication (run once interactively):
    copernicusmarine login

Dataset IDs (MEDSEA_MULTIYEAR_PHY_006_004 + MEDSEA_MULTIYEAR_BGC_006_008):
    Temperature: cmems_mod_med_phy-temp_my_4.2km_P1M-m  (var: thetao, °C)
    Salinity:    cmems_mod_med_phy-sal_my_4.2km_P1M-m   (var: so, PSU)
    O2:          cmems_mod_med_bgc-bio_my_4.2km_P1M-m   (var: o2, mmol/m³ = µmol/L)
    pH:          cmems_mod_med_bgc-car_my_4.2km_P1M-m   (var: ph)
    CO2:         cmems_mod_med_bgc-co2_my_4.2km_P1M-m   (var: spco2 µatm)

Fallback datasets (MEDSEA_ANALYSISFORECAST) for months not yet in multiyear:
    Temperature: cmems_mod_med_phy-tem_anfc_4.2km_P1M-m
    Salinity:    cmems_mod_med_phy-sal_anfc_4.2km_P1M-m
    O2:          cmems_mod_med_bgc-bio_anfc_4.2km_P1M-m
    pH:          cmems_mod_med_bgc-car_anfc_4.2km_P1M-m
    CO2:         cmems_mod_med_bgc-co2_anfc_4.2km_P1M-m

NOTE on CO2 units:
    The original data.csv CO2 column has values ~33–58 (unit unknown).
    Copernicus provides surface pCO2 (spco2) in µatm; Mediterranean values
    are typically ~380–450 µatm. These are likely DIFFERENT quantities.
    A cross-check against the original data overlap period is MANDATORY
    before using this column. See build_dataset.py for the check.
"""

import copernicusmarine
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# ── Site configuration ────────────────────────────────────────────────────────
SITE_LAT = 43.4278
SITE_LON = 10.3956
BBOX_DELTA = 0.1          # ±0.1° bounding box around the site
DEPTH_MIN = 0.0
DEPTH_MAX = 10.0
START = "2003-01-01"
END   = "2026-03-31"   # MEDSEA_MULTIYEAR is updated with ~1 month lag; extend as needed

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).parent.parent / "data" / "env_copernicus.csv"

# ── Dataset definitions ────────────────────────────────────────────────────────
DATASETS = [
    {
        "name": "temperature",
        "dataset_id": "cmems_mod_med_phy-temp_my_4.2km_P1M-m",
        "fallback_id": "cmems_mod_med_phy-tem_anfc_4.2km_P1M-m",
        "variables": ["thetao"],
        "filename": "raw_temperature.nc",
    },
    {
        "name": "salinity",
        "dataset_id": "cmems_mod_med_phy-sal_my_4.2km_P1M-m",
        "fallback_id": "cmems_mod_med_phy-sal_anfc_4.2km_P1M-m",
        "variables": ["so"],
        "filename": "raw_salinity.nc",
    },
    {
        "name": "oxygen",
        "dataset_id": "cmems_mod_med_bgc-bio_my_4.2km_P1M-m",
        "fallback_id": "cmems_mod_med_bgc-bio_anfc_4.2km_P1M-m",
        "variables": ["o2"],
        "filename": "raw_o2.nc",
    },
    {
        "name": "carbonate",
        "dataset_id": "cmems_mod_med_bgc-car_my_4.2km_P1M-m",
        "fallback_id": "cmems_mod_med_bgc-car_anfc_4.2km_P1M-m",
        "variables": ["ph"],
        "filename": "raw_carbonate.nc",
    },
    {
        "name": "co2",
        "dataset_id": "cmems_mod_med_bgc-co2_my_4.2km_P1M-m",
        "fallback_id": "cmems_mod_med_bgc-co2_anfc_4.2km_P1M-m",
        "variables": ["spco2"],
        "filename": "raw_co2.nc",
    },
]


def download_dataset(cfg: dict) -> Path:
    """Download a single dataset, with fallback if the primary fails."""
    out_file = RAW_DIR / cfg["filename"]
    if out_file.exists():
        print(f"  [{cfg['name']}] Already exists, skipping download.")
        return out_file

    for dataset_id in [cfg["dataset_id"], cfg.get("fallback_id")]:
        if dataset_id is None:
            continue
        try:
            print(f"  [{cfg['name']}] Downloading {dataset_id} ...")
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=cfg["variables"],
                minimum_longitude=SITE_LON - BBOX_DELTA,
                maximum_longitude=SITE_LON + BBOX_DELTA,
                minimum_latitude=SITE_LAT - BBOX_DELTA,
                maximum_latitude=SITE_LAT + BBOX_DELTA,
                minimum_depth=DEPTH_MIN,
                maximum_depth=DEPTH_MAX,
                start_datetime=START,
                end_datetime=END,
                output_filename=cfg["filename"],
                output_directory=str(RAW_DIR),
                force_download=False,
            )
            print(f"  [{cfg['name']}] Saved to {out_file}")
            return out_file
        except Exception as e:
            print(f"  [{cfg['name']}] Failed ({dataset_id}): {e}")

    raise RuntimeError(f"Could not download {cfg['name']} from any dataset.")


def nc_to_monthly_series(nc_path: Path, var_name: str) -> pd.Series:
    """
    Extract a monthly time series for a variable at the study site.
    Averages spatially over the bounding box and vertically over the depth layer.
    """
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    da = ds[var_name]

    # Average over spatial dimensions
    spatial_dims = [d for d in da.dims if d in ("latitude", "longitude", "lat", "lon")]
    if spatial_dims:
        da = da.mean(dim=spatial_dims)

    # Average over depth if present
    depth_dims = [d for d in da.dims if d in ("depth", "elevation")]
    if depth_dims:
        da = da.mean(dim=depth_dims)

    # Convert to pandas Series indexed by time
    series = da.to_series()
    series.index = pd.to_datetime(series.index).to_period("M").to_timestamp()
    series.name = var_name
    return series.sort_index()


def build_env_dataframe() -> pd.DataFrame:
    """Download all datasets and assemble a single monthly DataFrame."""
    series_list = []

    for cfg in DATASETS:
        nc_path = download_dataset(cfg)
        for var in cfg["variables"]:
            try:
                s = nc_to_monthly_series(nc_path, var)
                series_list.append(s)
                print(f"  [{cfg['name']}] Extracted {var}: {len(s)} months")
            except Exception as e:
                print(f"  [{cfg['name']}] Could not extract {var}: {e}")

    df = pd.concat(series_list, axis=1)
    df.index.name = "Datetime"
    df = df.reset_index()

    # Rename to match analysis.ipynb column names
    rename_map = {
        "thetao": "Temperature",
        "so": "Salinity",
        "o2": "O2",
        "ph": "pH",
        "spco2": "CO2",   # ⚠️ unit: µatm — cross-check against original data.csv required
    }
    df = df.rename(columns=rename_map)

    # Add a note column for CO2 unit tracking
    print("\n⚠️  CO2 (spco2) unit: µatm. Original data.csv CO2 values are ~33–58 (unit unknown).")
    print("    Cross-check required before using CO2 column. See build_dataset.py.\n")

    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Copernicus monthly data for site ({SITE_LAT}°N, {SITE_LON}°E)")
    print(f"Period: {START} → {END}, depth: {DEPTH_MIN}–{DEPTH_MAX}m\n")

    df = build_env_dataframe()
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df)} monthly rows to {OUT_PATH}")
    print(f"  Period: {df['Datetime'].min().date()} → {df['Datetime'].max().date()}")
    print(df.describe())


if __name__ == "__main__":
    main()
