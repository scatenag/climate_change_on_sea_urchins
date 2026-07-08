"""
MHW causal analysis:
  - CCF (mhw_peak_intensity → each variable), lags 0–12, raw + two
    stationarity-robust variants (see CCF robustness review)
  - Granger causality (mhw_peak_intensity → each variable)
  - ARDL cumulative response EC50 ~ MHW(t-k)

Outputs:
    results/ccf_results.csv                 (raw levels — Method A, kept for comparison)
    results/ccf_results_diff.csv            (first differences — Method C, primary robust result)
    results/ccf_results_prewhitened.csv     (ARIMA pre-whitening — Method E, cross-check)
    results/prewhitening_diagnostics.json   (ARIMA order/AIC + Ljung-Box per driver)
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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from .common import load_data, RESULTS, ALL_COLS, MHW_COLS, TAU_MAX


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


def difference_series(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    First-difference each column (month-over-month change), removing a
    linear/stochastic trend before correlating. Mirrors the practice used in
    the 2023 Sartori et al. supplement (analysis.ipynb: differencing EC50 and
    Salinity after the KPSS test flagged them non-stationary, before the
    causal-discovery step) and the differencing already applied to both
    series in compute_granger() below.
    """
    out = df.copy()
    for col in cols:
        out[col] = out[col].diff()
    return out


def _best_arima_order(series: np.ndarray, max_p: int = 3, max_q: int = 3):
    """Grid-search ARIMA(p,0,q) by AIC. d=0: MHW driver series are at worst
    borderline-stationary (ADF rejects a unit root; KPSS is ambiguous), so no
    further differencing is imposed — matches the orders used in the CCF
    robustness review (e.g. peak intensity: ARIMA(2,0,1))."""
    best_aic, best_order, best_fit = np.inf, None, None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ARIMA(series, order=(p, 0, q)).fit()
                if fit.aic < best_aic:
                    best_aic, best_order, best_fit = fit.aic, (p, 0, q), fit
            except Exception:
                continue
    return best_order, best_fit


def compute_ccf_prewhitened(df: pd.DataFrame, driver: str, targets: list[str],
                             tau_max: int = TAU_MAX) -> tuple[pd.DataFrame, dict]:
    """
    Box-Jenkins pre-whitening CCF (Method E in the CCF robustness review):
    fit the best ARIMA(p,0,q) (by AIC) to the driver, apply that SAME fitted
    filter to each target series (ARIMA(...).filter(), not a fresh fit per
    target), and cross-correlate the two residual series. Ljung-Box on the
    driver's own residuals (lags 6/12/24) checks the filter actually left
    white noise before the CCF is trusted.

    `df` must be the full continuous (imputed) frame — EC50 residuals are
    restricted to real (non-imputed) months only *after* filtering, so the
    filter itself always sees an unbroken monthly series.
    """
    driver_full = df[driver].ffill().bfill().values
    order, driver_fit = _best_arima_order(driver_full)
    if driver_fit is None:
        return pd.DataFrame(), {}

    driver_resid = driver_fit.resid
    lb = acorr_ljungbox(driver_resid, lags=[6, 12, 24], return_df=True)
    diagnostics = {
        "driver": driver,
        "order": list(order),
        "aic": float(driver_fit.aic),
        "ljung_box_p": {int(lag): float(p) for lag, p in zip(lb.index, lb["lb_pvalue"])},
        "white_noise": bool((lb["lb_pvalue"] > 0.05).all()),
    }

    rows = []
    for target in targets:
        target_full = df[target].ffill().bfill().values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                target_filtered = ARIMA(target_full, order=order).filter(driver_fit.params)
        except Exception:
            continue
        target_resid = target_filtered.resid

        if target == "EC50" and "EC50_imputed" in df.columns:
            target_resid = np.where(df["EC50_imputed"].values, np.nan, target_resid)

        for lag in range(0, tau_max + 1):
            if lag == 0:
                xa, ya = driver_resid, target_resid
            else:
                xa, ya = driver_resid[:-lag], target_resid[lag:]
            mask = ~(np.isnan(xa) | np.isnan(ya))
            if mask.sum() >= 10:
                r, p = stats.spearmanr(xa[mask], ya[mask])
            else:
                r, p = np.nan, np.nan
            rows.append(dict(driver=driver, variable=target, lag=lag,
                              spearman_r=r, p_value=p, n=int(mask.sum())))

    return pd.DataFrame(rows), diagnostics


def _print_best_lags(ccf_df: pd.DataFrame, label: str) -> None:
    best = (ccf_df.dropna(subset=["spearman_r"])
            .sort_values("p_value")
            .groupby("variable")
            .first()
            .reset_index()[["variable", "lag", "spearman_r", "p_value"]])
    print(f"✓ CCF ({label}) — best lag per variable:")
    for _, row in best.iterrows():
        print(f"  {row['variable']:25s}  lag={int(row['lag']):2d}  r={row['spearman_r']:+.3f}  p={row['p_value']:.4f}")


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

    # 1a. CCF — raw levels (Method A: current draft method). Flagged in the CCF
    #     robustness review as confounded by shared non-stationary trend (all 13
    #     lags come out significant with no peak — the signature of spurious
    #     correlation, not a localized biological effect). Kept only for comparison.
    ccf_df = compute_ccf(df_ccf, driver, targets)
    ccf_df.to_csv(RESULTS / "ccf_results.csv", index=False)
    _print_best_lags(ccf_df, "raw levels")

    # 1b. CCF — first differences (Method C: primary robust result). Differencing
    #     is applied to the full continuous series first (so month-over-month
    #     deltas are never taken across a real-data gap), and the real-EC50-only
    #     mask is re-applied afterwards, matching method A's restriction.
    df_diff = difference_series(df, [driver] + targets)
    df_diff.loc[df["EC50_imputed"].values, "EC50"] = np.nan
    ccf_diff_df = compute_ccf(df_diff, driver, targets)
    ccf_diff_df.to_csv(RESULTS / "ccf_results_diff.csv", index=False)
    _print_best_lags(ccf_diff_df, "first differences")

    # 1c. CCF — ARIMA pre-whitening (Method E), cross-checked against the two
    #     other MHW driver metrics (all vs EC50 only, to keep runtime reasonable).
    prewhiten_parts, diagnostics_all = [], {}
    for alt_driver in ["mhw_peak_intensity", "mhw_days", "mhw_cum_intensity"]:
        if alt_driver not in df.columns:
            continue
        pw_targets = targets if alt_driver == driver else ["EC50"]
        pw_df, diag = compute_ccf_prewhitened(df, alt_driver, pw_targets)
        if not pw_df.empty:
            prewhiten_parts.append(pw_df)
            diagnostics_all[alt_driver] = diag

    if prewhiten_parts:
        pw_all = pd.concat(prewhiten_parts, ignore_index=True)
        pw_all.to_csv(RESULTS / "ccf_results_prewhitened.csv", index=False)
        (RESULTS / "prewhitening_diagnostics.json").write_text(json.dumps(diagnostics_all, indent=2))
        print("\n✓ CCF (ARIMA pre-whitening) — best lag, driver → EC50:")
        for alt_driver, diag in diagnostics_all.items():
            sub = pw_all[(pw_all["driver"] == alt_driver) & (pw_all["variable"] == "EC50")]
            sub = sub.dropna(subset=["spearman_r"]).sort_values("p_value")
            if not sub.empty:
                b = sub.iloc[0]
                lb_ok = "residuals white noise" if diag["white_noise"] else "residual autocorrelation remains"
                print(f"  {alt_driver:22s} ARIMA{tuple(diag['order'])}  lag={int(b['lag']):2d}  "
                      f"r={b['spearman_r']:+.3f}  p={b['p_value']:.4f}  ({lb_ok})")

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
