"""
Fetch EC50 data from Google Sheets and aggregate to monthly time series.

Input:  Google Sheets (public export URL)
        Columns: ID, DATE, EC50, UL, LL, pos, neg, SRT DATE,
                 Replica I/II/III CTRL negativo (malformed)

Output: data/ec50_sheets.csv
        Columns: Datetime, EC50, EC50_ci_upper, EC50_ci_lower, EC50_n
        data/ec50_raw.csv
        Columns: Datetime, ID, EC50, ctrl_neg_rep1, ctrl_neg_rep2,
        ctrl_neg_rep3 — one row per bioassay determination (full
        resolution, not aggregated to month), used by changepoint.py's
        ordinal-sequence representation. Only the MONTH is recorded for many
        determinations, so a large fraction of rows share a Datetime; ID
        (the sheet's own row order) is kept specifically to give that
        ordinal sequence a well-defined, reproducible order -- sort by
        (Datetime, ID), never by Datetime alone (ties would otherwise
        resolve arbitrarily, and differently across pandas versions/runs).
        The three ctrl_neg_rep* columns are the assay's negative-control
        replicates (percent malformed larvae out of 100 examined) — fields
        of the trial, not a separate series; used by negative_control.py.
        They are NaN for trials where the validity criterion's supporting
        data was not recorded (present for 232 of 295 trials) -- these NaNs
        are informative and must not be filled.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EC50_EXPORT_URL as EXPORT_URL
from climate_change_on_sea_urchins.common import aggregate_monthly

OUT_PATH     = Path(__file__).parent.parent / "data" / "ec50_sheets.csv"
RAW_OUT_PATH = Path(__file__).parent.parent / "data" / "ec50_raw.csv"


def fetch_raw() -> pd.DataFrame:
    print(f"Downloading EC50 data from Google Sheets...")
    df = pd.read_csv(EXPORT_URL)
    df.columns = df.columns.str.strip()
    print(f"  → {len(df)} rows, columns: {list(df.columns)}")
    return df


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = fetch_raw()

    monthly = aggregate_monthly(raw)
    monthly.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(monthly)} monthly rows to {OUT_PATH}")
    print(f"  Period: {monthly['Datetime'].min().date()} → {monthly['Datetime'].max().date()}")
    print(f"  EC50 range: {monthly['EC50'].min():.2f} – {monthly['EC50'].max():.2f}")
    print(monthly.head())

    # Full-resolution per-determination cache (one row per bioassay, real
    # DATE not normalized to first-of-month) — same fetch, no extra network
    # call, kept in sync automatically since this script already runs daily.
    # Sorted by (Datetime, ID): many rows share a Datetime (month-only
    # dates), so ID (the sheet's own row order) is the tiebreaker that
    # makes this order reproducible -- sorting by Datetime alone leaves
    # same-date rows in an order that depends on pandas' sort implementation
    # and is not guaranteed stable across versions.
    #
    # ctrl_neg_rep1/2/3: the three negative-control replicates, kept as
    # trial fields (same Datetime/ID as the determination itself) rather
    # than a separate cache, since they have no independent timeline of
    # their own -- see negative_control.py.
    per_assay = raw.copy()
    per_assay["Datetime"] = pd.to_datetime(per_assay["DATE"], dayfirst=False)
    per_assay = per_assay.rename(columns={
        "Replica I CTRL negativo (malformed)":   "ctrl_neg_rep1",
        "Replica II CTRL negativo (malformed)":  "ctrl_neg_rep2",
        "Replica III CTRL negativo (malformed)": "ctrl_neg_rep3",
    })
    per_assay = per_assay[[
        "Datetime", "ID", "EC50", "ctrl_neg_rep1", "ctrl_neg_rep2", "ctrl_neg_rep3",
    ]].sort_values(["Datetime", "ID"]).reset_index(drop=True)
    per_assay.to_csv(RAW_OUT_PATH, index=False)
    print(f"Saved {len(per_assay)} per-determination rows to {RAW_OUT_PATH}")


if __name__ == "__main__":
    main()
