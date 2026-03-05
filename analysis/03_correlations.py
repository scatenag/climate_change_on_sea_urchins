"""
Spearman correlation matrices — All / Pre-2016 / Post-2016.
Outputs: results/corr_all.csv, results/corr_pre.csv, results/corr_post.csv
         results/corr_pval_all.csv, results/corr_pval_pre.csv, results/corr_pval_post.csv
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from common import load_data, RESULTS, ALL_COLS, MHW_COLS, SPLIT_YEAR


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


def run():
    df, df_real, events, _ = load_data()

    # Use df_full but swap imputed EC50 with real only — replace imputed rows with NaN
    df_corr = df.copy()
    df_corr.loc[df_corr["EC50_imputed"], "EC50"] = np.nan

    pre  = df_corr[df_corr["Datetime"] <  SPLIT_YEAR + "-01-01"]
    post = df_corr[df_corr["Datetime"] >= SPLIT_YEAR + "-01-01"]

    cols = ALL_COLS + MHW_COLS

    for label, subset in [("all", df_corr), ("pre", pre), ("post", post)]:
        r_df, p_df = spearman_matrix(subset, cols)
        r_df.to_csv(RESULTS / f"corr_{label}.csv")
        p_df.to_csv(RESULTS / f"corr_pval_{label}.csv")

    print(f"✓ correlations: matrices saved for {len(cols)} variables (all/pre/post)")


if __name__ == "__main__":
    run()
