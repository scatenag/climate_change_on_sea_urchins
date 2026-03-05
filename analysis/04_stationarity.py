"""
ADF + KPSS stationarity tests for all variables.
Output: results/stationarity_results.json
"""
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from common import load_data, RESULTS, ALL_COLS, MHW_COLS


def test_series(series: pd.Series, name: str) -> dict:
    s = series.dropna()
    if len(s) < 10:
        return {"variable": name, "n": len(s), "error": "insufficient data"}

    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(s, autolag="AIC")

    try:
        kpss_stat, kpss_p, _, kpss_crit = kpss(s, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p, kpss_crit = np.nan, np.nan, {}

    # Interpretation: stationary if ADF p<0.05 AND KPSS p>0.05
    adf_stationary  = bool(adf_p < 0.05)
    kpss_stationary = bool(kpss_p > 0.05) if not np.isnan(kpss_p) else None
    conclusion = "stationary" if (adf_stationary and kpss_stationary) else \
                 "non-stationary" if (not adf_stationary and not kpss_stationary) else \
                 "uncertain"

    return {
        "variable":        name,
        "n":               int(len(s)),
        "adf_stat":        float(adf_stat),
        "adf_p":           float(adf_p),
        "adf_stationary":  adf_stationary,
        "kpss_stat":       float(kpss_stat) if not np.isnan(kpss_stat) else None,
        "kpss_p":          float(kpss_p)    if not np.isnan(kpss_p)    else None,
        "kpss_stationary": kpss_stationary,
        "conclusion":      conclusion,
    }


def run():
    df, df_real, _, _ = load_data()

    # For EC50 use real measurements only (no imputed values)
    results = []
    for col in ALL_COLS + MHW_COLS:
        if col not in df.columns:
            continue
        src = df_real if col == "EC50" else df
        results.append(test_series(src[col], col))

    (RESULTS / "stationarity_results.json").write_text(json.dumps(results, indent=2))

    print(f"✓ stationarity: {len(results)} variables tested")
    for r in results:
        if "error" not in r:
            print(f"  {r['variable']:25s}  ADF p={r['adf_p']:.3f}  KPSS p={r['kpss_p']:.3f}  → {r['conclusion']}")


if __name__ == "__main__":
    run()
