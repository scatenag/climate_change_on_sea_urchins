"""
Fetch EC50 data from Google Sheets and aggregate to monthly time series.

Input:  Google Sheets (public export URL)
        Columns: ID, DATE, EC50, UL, LL, pos, neg

Output: data/ec50_sheets.csv
        Columns: Datetime, EC50, EC50_ci_upper, EC50_ci_lower, EC50_n
"""

import pandas as pd
import numpy as np
from pathlib import Path

SHEET_ID = "1e0-16D84ehRyotSC2BH9e9YqAnZksbv4gZDFgyPki8g"
EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

OUT_PATH = Path(__file__).parent.parent / "data" / "ec50_sheets.csv"


def fetch_raw() -> pd.DataFrame:
    print(f"Downloading EC50 data from Google Sheets...")
    df = pd.read_csv(EXPORT_URL)
    df.columns = df.columns.str.strip()
    print(f"  → {len(df)} rows, columns: {list(df.columns)}")
    return df


def aggregate_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple within-month measurements to a single monthly value.

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


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = fetch_raw()
    monthly = aggregate_monthly(raw)
    monthly.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(monthly)} monthly rows to {OUT_PATH}")
    print(f"  Period: {monthly['Datetime'].min().date()} → {monthly['Datetime'].max().date()}")
    print(f"  EC50 range: {monthly['EC50'].min():.2f} – {monthly['EC50'].max():.2f}")
    print(monthly.head())


if __name__ == "__main__":
    main()
