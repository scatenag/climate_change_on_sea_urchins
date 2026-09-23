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

# Canonical response-series column names. Every module downstream of
# load_data() reads the response column through these two constants, never
# through the literal "EC50"/"EC50_imputed" -- an indirection, not an alias:
# there is exactly one column in the DataFrame either way, so a module that
# hasn't migrated yet cannot silently diverge from one that has by writing
# to the "other" copy.
#
# Values stay "EC50"/"EC50_imputed" -- today's real column names in
# data/data_extended.csv and data/data_ec50_ci.csv -- until every module
# that reads them has migrated from the literal to the constant. Only then
# do the two values change to "response"/"response_imputed" in one step
# (V2.1 response-abstraction, note-tecniche.md sec 4); the .rename() calls
# below decouple the on-disk CSV column names (which never change) from
# what the rest of the code calls the column, so that flip is the *only*
# change that step needs.
RESPONSE_COL = "EC50"
IMPUTED_COL  = "EC50_imputed"

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_full   — all months (response includes rolling-mean imputations)
        df_real   — only months with real response bioassay measurements
        mhw_events
        mhw_monthly
    """
    data    = pd.read_csv(ROOT / "data" / "data_extended.csv",  parse_dates=["Datetime"])
    data    = data.rename(columns={"EC50": RESPONSE_COL})
    ci_df   = pd.read_csv(ROOT / "data" / "data_ec50_ci.csv",   parse_dates=["Datetime"])
    ci_df   = ci_df.rename(columns={"EC50_imputed": IMPUTED_COL})
    monthly = pd.read_csv(ROOT / "data" / "mhw_monthly.csv",    parse_dates=["Datetime"])
    events  = pd.read_csv(ROOT / "data" / "mhw_events.csv",
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
    df = df.merge(ci_df[["Datetime",IMPUTED_COL,"EC50_ci_upper","EC50_ci_lower"]],
                  on="Datetime", how="left")
    df[IMPUTED_COL] = df[IMPUTED_COL].fillna(True)

    # Rolling-mean impute the response for df_full (mirrors original notebook approach)
    df[RESPONSE_COL] = df[RESPONSE_COL].fillna(
        df[RESPONSE_COL].rolling(window=12, min_periods=3, center=True).mean()
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
    df_real = df[~df[IMPUTED_COL]].copy().reset_index(drop=True)

    return df_full, df_real, events, monthly


ENV_COLS  = ["O2", "CO2", "Temperature", "Salinity", "pH"]
ALL_COLS  = ENV_COLS + [RESPONSE_COL]
MHW_COLS  = ["mhw_peak_intensity", "mhw_days"]


def load_ec50_raw() -> pd.DataFrame:
    """The per-trial (single-bioassay) raw sequence, data/ec50_raw.csv --
    one row per determination, not yet aggregated to any period (distinct
    from load_data()'s monthly df_full/df_real). Every column from the CSV
    is preserved (ID, the negative-control replicate columns, etc.); only
    the response value column is renamed to RESPONSE_COL, so callers never
    reference the literal "EC50".

    Sorted by (Datetime, ID), not Datetime alone: ~110 of 295 rows share a
    Datetime (many determinations record only the month), and that tie
    order changes downstream statistics enough to matter (see
    changepoint.py's module docstring) -- ID (the source sheet's row order)
    is what makes the sequence reproducible.
    """
    raw = pd.read_csv(ROOT / "data" / "ec50_raw.csv", parse_dates=["Datetime"])
    raw = raw.rename(columns={"EC50": RESPONSE_COL})
    if "ID" not in raw.columns:
        raise ValueError(
            "data/ec50_raw.csv is missing the ID column -- re-run "
            "scripts/fetch_ec50.py. Many determinations share a Datetime "
            "(month-only dates), so ID (the source sheet's row order) is "
            "required to give the ordinal sequence a reproducible order; "
            "sorting by Datetime alone is not enough."
        )
    return raw.sort_values(["Datetime", "ID"]).reset_index(drop=True)


def load_ec50_monthly() -> pd.DataFrame:
    """The monthly-aggregated response series as fetched directly from the
    source, data/ec50_sheets.csv -- distinct from load_data()'s df_full/
    df_real, which merge in environmental variables and MHW metrics on top
    of it. Its dates are unique (one row per month), unlike ec50_raw.csv's.
    Response value column renamed to RESPONSE_COL, same as load_ec50_raw().
    """
    monthly = pd.read_csv(ROOT / "data" / "ec50_sheets.csv", parse_dates=["Datetime"])
    monthly = monthly.rename(columns={"EC50": RESPONSE_COL})
    return monthly.sort_values("Datetime").reset_index(drop=True)
