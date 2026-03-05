"""
Marine Heatwave Detection — Hobday et al. (2016) method.

No external MHW library required — manual implementation adapted from
the climate-change-ecotoxicology sister project.

Input:  data/sst_daily.csv  (Datetime, Temperature)
Output: mhw_events.csv      — event catalog
        mhw_monthly.csv     — monthly aggregated metrics (for analysis.ipynb)
        mhw_annual.csv      — annual block statistics

Hobday criteria:
  - Threshold:     90th percentile of daily climatology (11-day moving window)
  - Min duration:  5 consecutive days above threshold
  - Gap allowance: ≤2 days gap merges two events
  - Categories:    Moderate / Strong / Severe / Extreme
  - Climatology baseline: configurable (default 2003–2012, first 10 years)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT         = Path(__file__).parent.parent
SST_PATH     = ROOT / "data" / "sst_daily.csv"
OUT_EVENTS   = ROOT / "mhw_events.csv"
OUT_MONTHLY  = ROOT / "mhw_monthly.csv"
OUT_ANNUAL   = ROOT / "mhw_annual.csv"

CLIM_START = 2003   # climatology baseline period
CLIM_END   = 2012
PCTILE     = 90
MIN_DAYS   = 5
MAX_GAP    = 2


# ── 1. Climatology & threshold ────────────────────────────────────────────────

def compute_climatology(df: pd.DataFrame, clim_start: int, clim_end: int) -> pd.DataFrame:
    """
    Compute daily climatology (mean) and 90th-percentile threshold
    over the baseline period, then apply an 11-day circular moving average.
    Returns a DataFrame indexed by day-of-year (1–366).
    """
    baseline = df[df["Datetime"].dt.year.between(clim_start, clim_end)].copy()
    baseline["doy"] = baseline["Datetime"].dt.dayofyear

    clim = (
        baseline.groupby("doy")["Temperature"]
        .agg(climatology="mean", threshold=lambda x: np.percentile(x, PCTILE))
        .reset_index()
    )

    # 11-day circular smoothing
    for col in ["climatology", "threshold"]:
        vals = clim[col].values
        padded = np.concatenate([vals[-5:], vals, vals[:5]])
        smoothed = pd.Series(padded).rolling(11, center=True, min_periods=1).mean().values[5:-5]
        clim[col + "_smooth"] = smoothed

    # Baseline difference (used for category normalization)
    clim["baseline_diff"] = clim["threshold_smooth"] - clim["climatology_smooth"]
    clim["baseline_diff"] = clim["baseline_diff"].clip(lower=0.01)  # avoid division by zero
    return clim


# ── 2. Event detection ────────────────────────────────────────────────────────

def detect_events(df: pd.DataFrame, clim: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    """
    Detect MHW events and annotate each day.
    Returns (events list, daily DataFrame with flags).
    """
    df = df.copy()
    df["doy"] = df["Datetime"].dt.dayofyear

    df = df.merge(
        clim[["doy", "climatology_smooth", "threshold_smooth", "baseline_diff"]],
        on="doy", how="left",
    )

    df["anomaly"]            = df["Temperature"] - df["climatology_smooth"]
    df["intensity"]          = (df["Temperature"] - df["threshold_smooth"]).clip(lower=0)
    df["intensity_norm"]     = (df["anomaly"] / df["baseline_diff"]).clip(lower=0)
    df["exceeds_threshold"]  = df["Temperature"] > df["threshold_smooth"]
    df["in_event"]           = False
    df["event_id"]           = pd.NA

    events = []
    in_event = False
    event_indices = []
    gap_days = 0

    for idx, row in df.iterrows():
        if row["exceeds_threshold"]:
            if not in_event:
                in_event = True
                event_indices = [idx]
                gap_days = 0
            else:
                event_indices.append(idx)
                gap_days = 0
        else:
            if in_event:
                gap_days += 1
                if gap_days <= MAX_GAP:
                    event_indices.append(idx)
                else:
                    ev = df.loc[event_indices]
                    if len(ev) >= MIN_DAYS:
                        events.append(_build_event(ev, len(events) + 1))
                        df.loc[event_indices, "in_event"] = True
                        df.loc[event_indices, "event_id"] = len(events)
                    in_event = False
                    event_indices = []
                    gap_days = 0

    # Handle event still open at end of series
    if in_event and len(event_indices) >= MIN_DAYS:
        ev = df.loc[event_indices]
        events.append(_build_event(ev, len(events) + 1))
        df.loc[event_indices, "in_event"] = True
        df.loc[event_indices, "event_id"] = len(events)

    return events, df


def _build_event(ev: pd.DataFrame, event_id: int) -> dict:
    peak_idx = ev["intensity"].idxmax()
    max_norm = ev["intensity_norm"].max()

    if max_norm >= 4.0:
        category = "Extreme"
    elif max_norm >= 3.0:
        category = "Severe"
    elif max_norm >= 2.0:
        category = "Strong"
    else:
        category = "Moderate"

    return {
        "event_id":             event_id,
        "start_date":           ev["Datetime"].iloc[0],
        "end_date":             ev["Datetime"].iloc[-1],
        "peak_date":            ev.loc[peak_idx, "Datetime"],
        "duration_days":        len(ev),
        "intensity_max":        ev["intensity"].max(),
        "intensity_mean":       ev["intensity"].mean(),
        "intensity_cumulative": ev["intensity"].sum(),
        "intensity_norm_max":   max_norm,
        "category":             category,
        "year":                 ev["Datetime"].iloc[0].year,
        "start_month":          ev["Datetime"].iloc[0].month,
    }


# ── 3. Monthly aggregation ────────────────────────────────────────────────────

def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily MHW flags to monthly metrics."""
    daily = daily.copy()
    daily["Datetime_month"] = daily["Datetime"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        daily.groupby("Datetime_month")
        .agg(
            mhw_days          =("in_event",  "sum"),
            mhw_peak_intensity=("intensity", "max"),
            mhw_cum_intensity =("intensity", "sum"),
            anomaly_mean      =("anomaly",   "mean"),
        )
        .reset_index()
        .rename(columns={"Datetime_month": "Datetime"})
    )
    monthly["mhw_peak_intensity"] = monthly["mhw_peak_intensity"].fillna(0)
    return monthly


# ── 4. Annual block statistics ────────────────────────────────────────────────

def to_annual(events: list) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    ev_df = pd.DataFrame(events)
    annual = (
        ev_df.groupby("year")
        .agg(
            event_count       =("event_id",             "count"),
            total_mhw_days    =("duration_days",         "sum"),
            mean_duration     =("duration_days",         "mean"),
            max_intensity     =("intensity_max",         "max"),
            mean_intensity    =("intensity_mean",        "mean"),
            cum_intensity_sum =("intensity_cumulative",  "sum"),
        )
        .reset_index()
    )
    # Fill years with zero events
    all_years = pd.DataFrame({"year": range(ev_df["year"].min(), ev_df["year"].max() + 1)})
    annual = all_years.merge(annual, on="year", how="left").fillna(0)
    return annual


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not SST_PATH.exists():
        raise FileNotFoundError(
            f"Daily SST file not found: {SST_PATH}\n"
            "Run: python scripts/fetch_copernicus_daily.py"
        )

    print(f"Loading daily SST from {SST_PATH}...")
    df = pd.read_csv(SST_PATH, parse_dates=["Datetime"])
    print(f"  {len(df)} days, {df['Datetime'].min().date()} → {df['Datetime'].max().date()}")

    print(f"\nComputing climatology (baseline {CLIM_START}–{CLIM_END}, p{PCTILE})...")
    clim = compute_climatology(df, CLIM_START, CLIM_END)
    print(f"  Climatology range: {clim['climatology_smooth'].min():.2f}–{clim['climatology_smooth'].max():.2f}°C")
    print(f"  Threshold range:   {clim['threshold_smooth'].min():.2f}–{clim['threshold_smooth'].max():.2f}°C")

    print(f"\nDetecting MHW events (min {MIN_DAYS} days, gap ≤{MAX_GAP} days)...")
    events, daily = detect_events(df, clim)
    print(f"  → {len(events)} events detected")

    if events:
        ev_df = pd.DataFrame(events)
        categories = ev_df["category"].value_counts().to_dict()
        print(f"  Categories: {categories}")
        print(f"  Duration: {ev_df['duration_days'].min()}–{ev_df['duration_days'].max()} days")
        print(f"  Intensity (max): {ev_df['intensity_max'].max():.2f}°C above threshold")

        ev_df.to_csv(OUT_EVENTS, index=False)
        print(f"\n  Saved: {OUT_EVENTS}")

    monthly = to_monthly(daily)
    monthly.to_csv(OUT_MONTHLY, index=False)
    print(f"  Saved: {OUT_MONTHLY} ({len(monthly)} months)")
    print(f"  MHW months: {(monthly['mhw_days'] > 0).sum()} / {len(monthly)}")

    annual = to_annual(events)
    if not annual.empty:
        annual.to_csv(OUT_ANNUAL, index=False)
        print(f"  Saved: {OUT_ANNUAL} ({len(annual)} years)")

    print("\nDone.")


if __name__ == "__main__":
    main()
