"""Shared data loading for all analysis modules."""
import re
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# EC50 pre/post regime-shift boundary. Full date, not just a year: the
# manuscript's changepoint (QLR/AR(1), see changepoint.py) lands mid-year,
# not on January 1st, so every module below needs the exact date, not the
# implicit "1 Jan of this year" that an earlier, year-only version of this
# constant produced.
_SPLIT_CONFIG = "2016-06-01"


def _parse_split_date(value):
    if re.fullmatch(r"\d{4}", str(value)):
        warnings.warn(
            "SPLIT_YEAR/split configuration given as a bare year is "
            "deprecated; use a full date (e.g. '2016-06-01'). Falling back "
            "to January 1st of that year.",
            DeprecationWarning, stacklevel=3,
        )
        return pd.Timestamp(f"{value}-01-01")
    return pd.Timestamp(value)


SPLIT_DATE = _parse_split_date(_SPLIT_CONFIG)
SPLIT_YEAR = str(SPLIT_DATE.year)  # kept for callers that only need the year (e.g. axis labels)
TAU_MAX    = 12

def aggregate_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple within-month measurements to a single monthly value.
    The one implementation: scripts/fetch_ec50.py (the update job) and the
    dashboard's live read of the sheet both call it.

    Strategy:
    - EC50: arithmetic mean of all measurements in the month
    - CI: use the mean of individual half-widths (UL-EC50, EC50-LL),
          then compute standard error across replicates and take
          the wider of the two as the final CI bound.
    """
    raw = raw.copy()
    raw["DATE"] = pd.to_datetime(raw["DATE"], dayfirst=False)
    # Normalize to first-of-month
    raw["Datetime"] = raw["DATE"].dt.to_period("M").dt.to_timestamp()

    # Half-widths from individual bioassay CI
    raw["hw_upper"] = raw["UL"] - raw["EC50"]
    raw["hw_lower"] = raw["EC50"] - raw["LL"]

    agg = raw.groupby("Datetime").agg(
        EC50=("EC50", "mean"),
        EC50_std=("EC50", "std"),
        EC50_n=("EC50", "count"),
        mean_hw_upper=("hw_upper", "mean"),
        mean_hw_lower=("hw_lower", "mean"),
    ).reset_index()

    # Standard error across replicates
    agg["se"] = agg["EC50_std"] / np.sqrt(agg["EC50_n"])

    # Final CI: use the larger of (propagated bioassay CI) vs (replicate SE * 1.96)
    agg["EC50_ci_upper"] = agg["EC50"] + np.maximum(
        agg["mean_hw_upper"], 1.96 * agg["se"].fillna(0)
    )
    agg["EC50_ci_lower"] = agg["EC50"] - np.maximum(
        agg["mean_hw_lower"], 1.96 * agg["se"].fillna(0)
    )

    result = agg[["Datetime", "EC50", "EC50_ci_upper", "EC50_ci_lower", "EC50_n"]].copy()
    result = result.sort_values("Datetime").reset_index(drop=True)
    return result



def impute_ec50(monthly_full: pd.DataFrame, ec50: pd.DataFrame) -> pd.DataFrame:
    """
    Merge EC50 into the full monthly grid.
    Months without bioassay data are filled with a 12-month centered rolling mean.
    CI bounds are NaN for imputed months (flag: EC50_imputed=True).
    The one implementation: scripts/build_dataset.py (the update job) and
    load_data(ec50_monthly=...) (the dashboard's live read) both call it.
    """
    df = pd.merge(monthly_full, ec50, on="Datetime", how="left")

    df["EC50_imputed"] = df["EC50"].isna()

    # Fill missing EC50 with 12-month centered rolling mean (same as original notebook)
    df["EC50"] = df["EC50"].fillna(
        df["EC50"].rolling(window=12, min_periods=3, center=True).mean()
    )

    # CI bounds remain NaN for imputed months
    return df



def _build_from_monthly(data: pd.DataFrame, ec50_monthly: pd.DataFrame
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """data_extended.csv and data_ec50_ci.csv as scripts/build_dataset.py
    builds them, from a monthly EC50 aggregate (aggregate_monthly) read
    elsewhere than data/ -- the environmental columns are data_extended's."""
    env = data.drop(columns=["EC50"])
    start = min(env["Datetime"].min(), ec50_monthly["Datetime"].min())
    end = max(env["Datetime"].max(), ec50_monthly["Datetime"].max())
    grid = pd.DataFrame({"Datetime": pd.date_range(start=start, end=end, freq="MS")})
    built = impute_ec50(pd.merge(grid, env, on="Datetime", how="left"), ec50_monthly)
    data = built[list(env.columns) + ["EC50"]]
    ci_df = built[["Datetime", "EC50", "EC50_ci_upper", "EC50_ci_lower", "EC50_n", "EC50_imputed"]]
    return data, ci_df


def load_mhw_annual() -> pd.DataFrame:
    """Annual MHW metrics, data/mhw_annual.csv (written by mhw_detection)."""
    return pd.read_csv(ROOT / "data" / "mhw_annual.csv")


def load_ec50_monthly() -> pd.DataFrame:
    """The monthly EC50 aggregate as the update job last saved it,
    data/ec50_sheets.csv (aggregate_monthly of the source sheet)."""
    return pd.read_csv(ROOT / "data" / "ec50_sheets.csv", parse_dates=["Datetime"])


def load_data(ec50_monthly: pd.DataFrame | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_full   — all months (EC50 includes rolling-mean imputations)
        df_real   — only months with real EC50 bioassay measurements
        mhw_events
        mhw_monthly

    With `ec50_monthly` (aggregate_monthly of the source sheet, read live by
    the dashboard): EC50 is rebuilt from it exactly as the update job builds
    data/ (scripts/build_dataset.py), then prepared as below -- the same
    values as the job's when the sheet has not changed
    (tests/test_live_response.py).
    """
    data    = pd.read_csv(ROOT / "data" / "data_extended.csv",  parse_dates=["Datetime"])
    ci_df   = pd.read_csv(ROOT / "data" / "data_ec50_ci.csv",   parse_dates=["Datetime"])
    if ec50_monthly is not None:
        data, ci_df = _build_from_monthly(data, ec50_monthly)
    monthly = pd.read_csv(ROOT / "data" / "mhw_monthly.csv",    parse_dates=["Datetime"])
    events  = pd.read_csv(ROOT / "data" / "mhw_events.csv",
                          parse_dates=["start_date","end_date","peak_date"])

    # Merge MHW monthly metrics. Months beyond the SST coverage (the daily SST
    # arrives months after the response) have no MHW catalogue: their metrics
    # stay missing, never 0, which would mean "no heatwave"
    # (tests/test_mhw_missing_sst.py).
    df = data.merge(
        monthly[["Datetime","mhw_days","mhw_peak_intensity","mhw_cum_intensity"]],
        on="Datetime", how="left"
    )

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
