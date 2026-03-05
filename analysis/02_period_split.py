"""
Pre/Post 2016 statistical comparison.
Outputs: results/kruskal_stats.json, results/period_means.csv
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from common import load_data, RESULTS, ALL_COLS, MHW_COLS, SPLIT_YEAR

def run():
    df, df_real, events, _ = load_data()
    df = df.dropna(subset=ALL_COLS)

    pre  = df[df["Datetime"] <  SPLIT_YEAR + "-01-01"]
    post = df[df["Datetime"] >= SPLIT_YEAR + "-01-01"]

    # Use real EC50 only for EC50 tests; full series for env vars
    pre_ec50  = df_real[df_real["Datetime"] <  SPLIT_YEAR + "-01-01"]["EC50"]
    post_ec50 = df_real[df_real["Datetime"] >= SPLIT_YEAR + "-01-01"]["EC50"]

    results = {}
    for col in ALL_COLS:
        if col == "EC50":
            a, b = pre_ec50.dropna(), post_ec50.dropna()
        else:
            a, b = pre[col].dropna(), post[col].dropna()

        kw  = stats.kruskal(a, b)
        mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
        results[col] = {
            "kruskal_stat": float(kw.statistic),
            "kruskal_p":    float(kw.pvalue),
            "mannwhitney_stat": float(mwu.statistic),
            "mannwhitney_p":    float(mwu.pvalue),
            "n_pre":  int(len(a)),
            "n_post": int(len(b)),
            "mean_pre":  float(a.mean()),
            "mean_post": float(b.mean()),
            "median_pre":  float(a.median()),
            "median_post": float(b.median()),
        }

    (RESULTS / "kruskal_stats.json").write_text(json.dumps(results, indent=2))

    # Period means table
    rows = []
    for col in ALL_COLS + MHW_COLS:
        if col not in df.columns:
            continue
        all_val = df[col].mean() if col != "EC50" else df_real["EC50"].mean()
        pre_val = pre[col].mean() if col != "EC50" else pre_ec50.mean()
        post_val = post[col].mean() if col != "EC50" else post_ec50.mean()
        rows.append({"variable": col, "mean_all": all_val,
                     "mean_pre": pre_val, "mean_post": post_val,
                     "change_pct": 100*(post_val - pre_val)/abs(pre_val) if pre_val else np.nan})
    pd.DataFrame(rows).to_csv(RESULTS / "period_means.csv", index=False)

    # Distribution data for boxplots (Streamlit)
    for col in ALL_COLS:
        src = df_real if col == "EC50" else df
        out = src[["Datetime", col]].copy()
        out["period"] = np.where(out["Datetime"] < SPLIT_YEAR + "-01-01",
                                 f"2003–{int(SPLIT_YEAR)-1}", f"{SPLIT_YEAR}–2025")
        out.to_csv(RESULTS / f"dist_{col}.csv", index=False)

    print(f"✓ period_split: stats saved for {len(ALL_COLS)} variables")
    ec50_res = results["EC50"]
    print(f"  EC50 KW p={ec50_res['kruskal_p']:.2e}  "
          f"pre_mean={ec50_res['mean_pre']:.2f}  post_mean={ec50_res['mean_post']:.2f}")

if __name__ == "__main__":
    run()
