"""
MHW causal analysis:
  - CCF (mhw_peak_intensity → each variable), lags 0–12
  - Granger causality (mhw_peak_intensity → each variable)
  - ARDL cumulative response EC50 ~ MHW(t-k)

Outputs:
    results/ccf_results.csv
    results/granger_results.json
    results/ardl_response.csv
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import ccf as sm_ccf, grangercausalitytests
from statsmodels.tsa.ardl import ARDL
from common import load_data, RESULTS, ALL_COLS, MHW_COLS, TAU_MAX


# ── 1. CCF ────────────────────────────────────────────────────────────────────

def compute_ccf(df: pd.DataFrame, driver: str, targets: list[str]) -> pd.DataFrame:
    """
    For each target variable: Spearman r between driver(t-k) and target(t)
    at each lag k in 0..TAU_MAX.
    Uses only real EC50 measurements for EC50; full series for others.
    """
    rows = []
    mhw = df[driver].values

    for target in targets:
        src = df[target].values
        for lag in range(0, TAU_MAX + 1):
            if lag == 0:
                x, y = mhw, src
            else:
                x, y = mhw[:-lag], src[lag:]
            # Drop NaN pairs
            mask = ~(np.isnan(x) | np.isnan(y))
            xa, ya = x[mask], y[mask]
            if len(xa) >= 10:
                r, p = stats.spearmanr(xa, ya)
            else:
                r, p = np.nan, np.nan
            rows.append(dict(variable=target, lag=lag, spearman_r=r, p_value=p, n=int(mask.sum())))

    return pd.DataFrame(rows)


# ── 2. Granger causality ──────────────────────────────────────────────────────

def compute_granger(df: pd.DataFrame, driver: str, targets: list[str]) -> dict:
    """
    Granger test: does driver Granger-cause target?
    Returns dict of {variable: {lag: p_value}} for lags 1..TAU_MAX.
    Uses first-differenced series for non-stationary variables.
    """
    results = {}
    x = df[driver].ffill().bfill()

    for target in targets:
        y = df[target].ffill().bfill()
        # Difference both to help stationarity
        data = pd.concat([y.diff(), x.diff()], axis=1).dropna()
        data.columns = ["y", "x"]
        if len(data) < TAU_MAX * 3:
            results[target] = {}
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc = grangercausalitytests(data[["y", "x"]], maxlag=TAU_MAX, verbose=False)
            results[target] = {
                lag: float(gc[lag][0]["ssr_ftest"][1])
                for lag in range(1, TAU_MAX + 1)
            }
        except Exception as e:
            results[target] = {"error": str(e)}

    return results


# ── 3. ARDL cumulative response EC50 ~ MHW ───────────────────────────────────

def compute_ardl(df_real: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Fit ARDL(p, q) for EC50 ~ MHW_peak_intensity using real EC50 measurements only.
    Builds a monthly-indexed series with NaN for missing months, then fits ARDL
    on the merged series. Reports cumulative impulse response (approximate).
    """
    # Monthly index from full series
    idx = df_full["Datetime"].dt.to_period("M")
    mhw_monthly = df_full.set_index(idx)["mhw_peak_intensity"]

    ec50_real = df_real.copy()
    ec50_real["period"] = ec50_real["Datetime"].dt.to_period("M")
    ec50_monthly = ec50_real.set_index("period")["EC50"]

    # Align on common monthly periods
    common = mhw_monthly.index.intersection(ec50_monthly.index)
    y = ec50_monthly[common]
    x = mhw_monthly[common]
    data = pd.DataFrame({"EC50": y, "MHW": x}).dropna()

    if len(data) < 20:
        print("  ARDL: insufficient aligned data")
        return pd.DataFrame()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARDL(data["EC50"], lags=3, exog=data[["MHW"]], order={"MHW": TAU_MAX})
            fit   = model.fit(cov_type="HC1")

        # Extract MHW lag coefficients
        rows = []
        for lag in range(0, TAU_MAX + 1):
            coef_name = f"MHW.L{lag}" if lag > 0 else "MHW"
            # statsmodels ARDL may name differently; search
            matches = [n for n in fit.params.index if f"MHW" in n and
                       (f".L{lag}" in n if lag > 0 else not any(f".L{k}" in n for k in range(1, TAU_MAX+1)))]
            if matches:
                c = float(fit.params[matches[0]])
                se = float(fit.bse[matches[0]])
            else:
                c, se = np.nan, np.nan
            rows.append(dict(lag=lag, coef=c, se=se))

        df_out = pd.DataFrame(rows)
        # Cumulative response
        df_out["cumul"] = df_out["coef"].cumsum()
        df_out["n_obs"] = int(len(data))
        return df_out

    except Exception as e:
        print(f"  ARDL failed: {e}")
        return pd.DataFrame()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    df, df_real, _, _ = load_data()

    # For CCF: use real EC50 (NaN for imputed), full series for others
    df_ccf = df.copy()
    df_ccf.loc[df_ccf["EC50_imputed"], "EC50"] = np.nan

    driver  = "mhw_peak_intensity"
    targets = [c for c in ALL_COLS if c in df_ccf.columns]

    # 1. CCF
    ccf_df = compute_ccf(df_ccf, driver, targets)
    ccf_df.to_csv(RESULTS / "ccf_results.csv", index=False)

    # Summary: best lag per variable
    best = (ccf_df.dropna(subset=["spearman_r"])
            .sort_values("p_value")
            .groupby("variable")
            .first()
            .reset_index()[["variable", "lag", "spearman_r", "p_value"]])
    print("✓ CCF — best lag per variable:")
    for _, row in best.iterrows():
        print(f"  {row['variable']:25s}  lag={int(row['lag']):2d}  r={row['spearman_r']:+.3f}  p={row['p_value']:.4f}")

    # 2. Granger
    granger = compute_granger(df, driver, targets)
    (RESULTS / "granger_results.json").write_text(json.dumps(granger, indent=2))
    print(f"\n✓ Granger causality saved for {len(granger)} variables")
    for var, lag_ps in granger.items():
        if isinstance(lag_ps, dict) and lag_ps and "error" not in lag_ps:
            min_p = min(lag_ps.values())
            best_lag = min(lag_ps, key=lag_ps.get)
            sig = "***" if min_p < 0.001 else "**" if min_p < 0.01 else "*" if min_p < 0.05 else "n.s."
            print(f"  MHW → {var:20s}  best lag={best_lag}  min_p={min_p:.4f}  {sig}")

    # 3. ARDL
    ardl_df = compute_ardl(df_real, df)
    if not ardl_df.empty:
        ardl_df.to_csv(RESULTS / "ardl_response.csv", index=False)
        print(f"\n✓ ARDL response saved ({len(ardl_df)} lags)")


if __name__ == "__main__":
    run()
