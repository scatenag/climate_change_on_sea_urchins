"""
Time series decomposition, trend extraction, PCA anomaly detection.
Outputs: results/trend_*.csv, results/pca_anomaly.csv
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import pairwise_distances
from common import load_data, RESULTS, ALL_COLS, SPLIT_YEAR

def decompose_series(df, col):
    s = df.set_index("Datetime")[col].dropna()
    result = seasonal_decompose(s, model="multiplicative", period=12, extrapolate_trend="freq")
    return result.trend, result.seasonal, result.resid

def run():
    df, df_real, events, monthly = load_data()
    df = df.dropna(subset=ALL_COLS)

    # ── Decompose full series ──────────────────────────────────────────────
    trends = {}
    for col in ALL_COLS:
        trend, seasonal, resid = decompose_series(df, col)
        trends[col] = trend
        out = pd.DataFrame({"Datetime": trend.index, "trend": trend.values,
                            "seasonal": seasonal.values, "residual": resid.values})
        out.to_csv(RESULTS / f"trend_{col}.csv", index=False)

    trend_df = pd.DataFrame(trends).reset_index().rename(columns={"Datetime": "Datetime"})
    trend_df.to_csv(RESULTS / "trends_all.csv", index=False)

    # ── Pre/Post split decompositions ──────────────────────────────────────
    for label, mask in [("pre", df["Datetime"] < SPLIT_YEAR + "-01-01"),
                        ("post", df["Datetime"] >= SPLIT_YEAR + "-01-01")]:
        sub = df[mask].dropna(subset=ALL_COLS)
        rows = []
        for col in ALL_COLS:
            try:
                t, s, r = decompose_series(sub, col)
                for dt, tv in zip(t.index, t.values):
                    rows.append({"Datetime": dt, "variable": col, "trend": tv})
            except Exception:
                pass
        pd.DataFrame(rows).to_csv(RESULTS / f"trends_{label}.csv", index=False)

    # ── PCA anomaly detection ──────────────────────────────────────────────
    X = df[ALL_COLS].values
    tscv = TimeSeriesSplit(n_splits=20, test_size=11)
    rec_errors = []
    rec_dates  = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        scaler = StandardScaler().fit(X_train)
        Xt = scaler.transform(X_test)
        Xtr = scaler.transform(X_train)
        pca = PCA(n_components=0.7).fit(Xtr)
        Xrec = pca.inverse_transform(pca.transform(Xt))
        err = np.linalg.norm(Xt - Xrec, axis=1)
        rec_errors.extend(err)
        rec_dates.extend(df["Datetime"].iloc[test_idx].tolist())

    anomaly = pd.DataFrame({"Datetime": rec_dates, "reconstruction_error": rec_errors})
    anomaly = anomaly.groupby("Datetime").mean().reset_index().sort_values("Datetime")
    anomaly.to_csv(RESULTS / "pca_anomaly.csv", index=False)

    print(f"✓ timeseries: {len(df)} months decomposed, PCA anomaly saved")

if __name__ == "__main__":
    run()
