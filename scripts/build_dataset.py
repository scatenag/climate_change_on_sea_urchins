"""
Build the final analysis dataset from Copernicus + Google Sheets EC50.

Inputs:
    data/env_copernicus.csv   → Datetime, Temperature, Salinity, O2, pH, CO2
    data/ec50_sheets.csv      → Datetime, EC50, EC50_ci_upper, EC50_ci_lower, EC50_n

Outputs:
    data_extended.csv         → same schema as original data.csv (for analysis.ipynb)
    data_ec50_ci.csv          → EC50 CI bounds for all months (incl. imputed months)

Run AFTER:
    python scripts/fetch_ec50.py
    python scripts/fetch_copernicus.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
ENV_PATH  = ROOT / "data" / "env_copernicus.csv"
EC50_PATH = ROOT / "data" / "ec50_sheets.csv"
ORIG_PATH = ROOT / "data.csv"   # original — kept as validation reference only

OUT_DATA = ROOT / "data_extended.csv"
OUT_CI   = ROOT / "data_ec50_ci.csv"


def load_inputs():
    env = pd.read_csv(ENV_PATH, parse_dates=["Datetime"])
    ec50 = pd.read_csv(EC50_PATH, parse_dates=["Datetime"])
    orig = pd.read_csv(ORIG_PATH)
    orig.columns = orig.columns.str.strip()
    orig.rename(columns={"CO2_Con": "CO2"}, inplace=True)
    orig["Datetime"] = pd.to_datetime(orig["date"], dayfirst=True)
    orig = orig[["Datetime", "O2", "CO2", "Temperature", "Salinity", "pH", "EC50"]]
    return env, ec50, orig


def cross_check_co2(env: pd.DataFrame, orig: pd.DataFrame):
    """
    Compare CO2 values in the overlap period between Copernicus and original data.
    Prints statistics to help decide whether the CO2 columns are compatible.
    """
    merged = pd.merge(
        env[["Datetime", "CO2"]],
        orig[["Datetime", "CO2"]].rename(columns={"CO2": "CO2_orig"}),
        on="Datetime",
        how="inner",
    ).dropna()

    if merged.empty:
        print("⚠️  No overlapping CO2 data to cross-check.")
        return

    ratio = merged["CO2"] / merged["CO2_orig"]
    print("\n── CO2 Cross-Check (Copernicus spco2 µatm vs original) ──────────────")
    print(f"  Overlap period: {merged['Datetime'].min().date()} → {merged['Datetime'].max().date()}")
    print(f"  Copernicus CO2 (µatm): mean={merged['CO2'].mean():.1f}, range=[{merged['CO2'].min():.1f}, {merged['CO2'].max():.1f}]")
    print(f"  Original CO2:          mean={merged['CO2_orig'].mean():.1f}, range=[{merged['CO2_orig'].min():.1f}, {merged['CO2_orig'].max():.1f}]")
    print(f"  Ratio (Cop/Orig):      mean={ratio.mean():.2f}, std={ratio.std():.2f}")
    print("─" * 67)

    if abs(ratio.mean() - 1.0) < 0.05:
        print("  ✓ Units appear compatible (ratio ≈ 1).")
    elif abs(ratio.mean() - 10.0) < 2.0:
        print("  ⚠️  Ratio ≈ 10. Copernicus CO2 may need to be divided by 10.")
    else:
        print(f"  ⚠️  Ratio = {ratio.mean():.1f}. Manual unit inspection required.")
    print()


def impute_ec50(monthly_full: pd.DataFrame, ec50: pd.DataFrame) -> pd.DataFrame:
    """
    Merge EC50 into the full monthly grid.
    Months without bioassay data are filled with a 12-month centered rolling mean.
    CI bounds are NaN for imputed months (flag: EC50_imputed=True).
    """
    df = pd.merge(monthly_full, ec50, on="Datetime", how="left")

    df["EC50_imputed"] = df["EC50"].isna()

    # Fill missing EC50 with 12-month centered rolling mean (same as original notebook)
    df["EC50"] = df["EC50"].fillna(
        df["EC50"].rolling(window=12, min_periods=3, center=True).mean()
    )

    # CI bounds remain NaN for imputed months
    return df


def main():
    print("Building dataset from Copernicus + Google Sheets EC50...\n")

    env, ec50, orig = load_inputs()

    # Cross-check CO2 units
    cross_check_co2(env, orig)

    # Build a complete monthly time index covering the full period
    start = min(env["Datetime"].min(), ec50["Datetime"].min())
    end   = max(env["Datetime"].max(), ec50["Datetime"].max())
    full_index = pd.DataFrame(
        {"Datetime": pd.date_range(start=start, end=end, freq="MS")}
    )
    print(f"Full period: {start.date()} → {end.date()} ({len(full_index)} months)")

    # Merge environmental data (left join on full monthly index)
    df = pd.merge(full_index, env, on="Datetime", how="left")

    # Check coverage
    n_missing_env = df[["Temperature", "Salinity", "O2", "pH", "CO2"]].isna().any(axis=1).sum()
    if n_missing_env > 0:
        print(f"⚠️  {n_missing_env} months with missing environmental data (will be NaN in output)")

    # Merge and impute EC50
    df = impute_ec50(df, ec50)

    # ── Output 1: data_extended.csv (same schema as data.csv) ─────────────────
    data_extended = df[["Datetime", "O2", "CO2", "Temperature", "Salinity", "pH", "EC50"]].copy()
    data_extended.to_csv(OUT_DATA, index=False)
    print(f"\nSaved {len(data_extended)} rows to {OUT_DATA}")

    # ── Output 2: data_ec50_ci.csv (CI bounds for plotting) ───────────────────
    ec50_ci = df[["Datetime", "EC50", "EC50_ci_upper", "EC50_ci_lower", "EC50_n", "EC50_imputed"]].copy()
    ec50_ci.to_csv(OUT_CI, index=False)
    print(f"Saved {len(ec50_ci)} rows to {OUT_CI}")

    # ── Validation: compare with original data.csv on overlap ─────────────────
    overlap = pd.merge(
        data_extended,
        orig.rename(columns={c: c + "_orig" for c in orig.columns if c != "Datetime"}),
        on="Datetime",
        how="inner",
    )
    if not overlap.empty:
        print(f"\n── Validation vs original data.csv (overlap: {len(overlap)} months) ──")
        for col in ["Temperature", "Salinity", "O2", "pH"]:
            orig_col = col + "_orig"
            if orig_col in overlap.columns:
                diff = (overlap[col] - overlap[orig_col]).abs()
                print(f"  {col}: max_diff={diff.max():.4f}, mean_diff={diff.mean():.4f}")
        print()


if __name__ == "__main__":
    main()
