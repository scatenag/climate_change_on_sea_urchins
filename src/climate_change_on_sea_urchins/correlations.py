"""
Spearman correlation matrices — All / Pre-2016 / Post-2016.

Mirrors the original notebook approach:
  1. EC50 filled with rolling mean (window=12, centered) — same as notebook cell 7
  2. Seasonal decomposition (multiplicative, period=12) on each variable
  3. Spearman r computed on the TREND components — same as notebook cells 25-27

MHW columns (mhw_peak_intensity, mhw_days) are included raw (no decomposition,
as they are already summary metrics without strong seasonality).

Outputs: results/corr_all.csv, results/corr_pre.csv, results/corr_post.csv
         results/corr_pval_all.csv, results/corr_pval_pre.csv, results/corr_pval_post.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from .common import load_data, default_results_dir, ALL_COLS, MHW_COLS, RESPONSE_COL, SPLIT_DATE, default_response_spec


def extract_trends(df: pd.DataFrame, cols: list[str], period: int = 12) -> pd.DataFrame:
    """
    Run seasonal decomposition on each column and return a DataFrame of trend components.
    Uses multiplicative model with extrapolate_trend='freq' to avoid edge NaNs.
    Columns with fewer than 2*period non-NaN values are returned as-is.
    """
    trend_df = pd.DataFrame(index=df.index)
    for col in cols:
        series = df[col].copy()
        valid = series.dropna()
        if len(valid) < 2 * period:
            trend_df[col] = series
            continue
        try:
            dec = seasonal_decompose(
                series.interpolate("linear"),
                model="multiplicative",
                period=period,
                extrapolate_trend="freq",
                two_sided=False,
            )
            trend_df[col] = dec.trend.values
        except Exception:
            trend_df[col] = series
    return trend_df


def spearman_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (r_matrix, p_matrix) for a list of columns."""
    n = len(cols)
    r_mat = np.eye(n)
    p_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            a = df[cols[i]].dropna()
            b = df[cols[j]].dropna()
            common = a.index.intersection(b.index)
            if len(common) < 5:
                r, p = np.nan, np.nan
            else:
                r, p = stats.spearmanr(a[common], b[common])
            r_mat[i, j] = r_mat[j, i] = r
            p_mat[i, j] = p_mat[j, i] = p

    r_df = pd.DataFrame(r_mat, index=cols, columns=cols)
    p_df = pd.DataFrame(p_mat, index=cols, columns=cols)
    return r_df, p_df


def run(response=None, results=None):
    results = results if results is not None else default_results_dir()
    if response is None:
        response = default_response_spec()
    label = response.label  # display identity for the response row/column below

    df, _, _, _ = load_data()

    df_work = df.set_index("Datetime")

    # Mirror notebook cell 7: apply rolling mean to ALL response values (not
    # just NaN). The original notebook smooths the full series before
    # decomposition, which reduces measurement noise and produces
    # meaningful trend correlations.
    df_work[RESPONSE_COL] = df_work[RESPONSE_COL].rolling(
        window=12, min_periods=1, center=True
    ).mean()

    pre_mask  = df_work.index <  SPLIT_DATE
    post_mask = df_work.index >= SPLIT_DATE

    env_cols = ALL_COLS   # O2, CO2, Temperature, Salinity, pH, response
    mhw_cols = MHW_COLS   # mhw_peak_intensity, mhw_days
    all_cols  = env_cols + mhw_cols

    for period, mask in [("all", slice(None)), ("pre", pre_mask), ("post", post_mask)]:
        subset = df_work[mask]

        # Trend components for env variables
        trend_env = extract_trends(subset, env_cols)

        # MHW columns raw (no decomposition)
        trend_mhw = subset[mhw_cols].copy()

        combined = pd.concat([trend_env, trend_mhw], axis=1)

        r_df, p_df = spearman_matrix(combined, all_cols)
        # Output identity: the response's row/column name is its display
        # label, never the internal RESPONSE_COL (see common.py).
        r_df = r_df.rename(index={RESPONSE_COL: label}, columns={RESPONSE_COL: label})
        p_df = p_df.rename(index={RESPONSE_COL: label}, columns={RESPONSE_COL: label})
        r_df.to_csv(results / f"corr_{period}.csv")
        p_df.to_csv(results / f"corr_pval_{period}.csv")

    print(f"✓ correlations: trend-based Spearman matrices saved (all/pre/post, {len(all_cols)} vars)")


if __name__ == "__main__":
    run()
