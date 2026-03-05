"""Shared data loading for all analysis modules."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SPLIT_YEAR = "2016"
TAU_MAX    = 12

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_full   — all months (EC50 includes rolling-mean imputations)
        df_real   — only months with real EC50 bioassay measurements
        mhw_events
        mhw_monthly
    """
    data    = pd.read_csv(ROOT / "data_extended.csv",  parse_dates=["Datetime"])
    ci_df   = pd.read_csv(ROOT / "data_ec50_ci.csv",   parse_dates=["Datetime"])
    monthly = pd.read_csv(ROOT / "mhw_monthly.csv",    parse_dates=["Datetime"])
    events  = pd.read_csv(ROOT / "mhw_events.csv",
                          parse_dates=["start_date","end_date","peak_date"])

    # Merge MHW monthly metrics
    df = data.merge(
        monthly[["Datetime","mhw_days","mhw_peak_intensity","mhw_cum_intensity"]],
        on="Datetime", how="left"
    )
    df["mhw_days"]            = df["mhw_days"].fillna(0)
    df["mhw_peak_intensity"]  = df["mhw_peak_intensity"].fillna(0)
    df["mhw_cum_intensity"]   = df["mhw_cum_intensity"].fillna(0)

    # Imputation flag
    df = df.merge(ci_df[["Datetime","EC50_imputed","EC50_ci_upper","EC50_ci_lower"]],
                  on="Datetime", how="left")
    df["EC50_imputed"] = df["EC50_imputed"].fillna(True)

    # Rolling-mean impute EC50 for df_full (mirrors original notebook approach)
    df["EC50"] = df["EC50"].fillna(
        df["EC50"].rolling(window=12, min_periods=3, center=True).mean()
    )

    # Fill Temperature gaps (after Copernicus monthly ends) from daily SST monthly averages.
    # Without this, 2024 rows show only Jan–Apr (winter avg ~14°C), breaking trend analysis.
    sst_path = ROOT / "data" / "sst_daily.csv"
    if sst_path.exists():
        sst = pd.read_csv(sst_path, parse_dates=["Datetime"])
        sst["month"] = sst["Datetime"].dt.to_period("M").dt.to_timestamp()
        sst_monthly = sst.groupby("month")["Temperature"].mean().reset_index()
        sst_monthly.columns = ["Datetime", "Temperature_sst"]
        df = df.merge(sst_monthly, on="Datetime", how="left")
        mask = df["Temperature"].isna() & df["Temperature_sst"].notna()
        df.loc[mask, "Temperature"] = df.loc[mask, "Temperature_sst"]
        df.drop(columns=["Temperature_sst"], inplace=True)

    df_full = df.copy()
    df_real = df[~df["EC50_imputed"]].copy().reset_index(drop=True)

    return df_full, df_real, events, monthly


ENV_COLS  = ["O2", "CO2", "Temperature", "Salinity", "pH"]
ALL_COLS  = ENV_COLS + ["EC50"]
MHW_COLS  = ["mhw_peak_intensity", "mhw_days"]
