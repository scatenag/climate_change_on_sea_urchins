"""
EC50 forecast 2025–2040 — three MHW scenarios (bad / mean / good).
Uses SARIMAX with MHW peak intensity as exogenous regressor at the optimal lag.
EC50 is interpolated to a regular monthly grid for modelling.

Outputs:
    results/forecast_bad.csv
    results/forecast_mean.csv
    results/forecast_good.csv
    results/forecast_meta.json
"""
import json
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
from common import load_data, RESULTS, TAU_MAX


FORECAST_YEARS = 15   # cover up to current_year + 15


def find_optimal_lag(df_real: pd.DataFrame, df_full: pd.DataFrame) -> int:
    """
    Spearman r between mhw_peak_intensity(t-k) and real EC50(t).
    Looks up exact month in df_full for lagged MHW value (same method as explore script).
    """
    best_lag, best_r = 2, 0.0  # default to 2 months (known peak from exploration)
    for lag in range(0, TAU_MAX + 1):
        mhw_vals, ec50_vals = [], []
        for _, row in df_real.iterrows():
            target = row["Datetime"] - pd.DateOffset(months=lag)
            diffs  = (df_full["Datetime"] - target).abs()
            idx    = diffs.idxmin()
            if diffs[idx] <= pd.Timedelta("20 days"):
                mhw_vals.append(df_full.loc[idx, "mhw_peak_intensity"])
                ec50_vals.append(row["EC50"])
        xa, ya = np.array(mhw_vals), np.array(ec50_vals)
        mask = ~(np.isnan(xa) | np.isnan(ya))
        if mask.sum() < 10:
            continue
        r, _ = stats.spearmanr(xa[mask], ya[mask])
        if abs(r) > best_r:
            best_r, best_lag = abs(r), lag
    return best_lag


def build_monthly_series(df_real: pd.DataFrame, df_full: pd.DataFrame,
                          opt_lag: int) -> pd.DataFrame:
    """
    Create a regular monthly DataFrame aligned on df_full dates,
    with EC50 interpolated (linear) for missing months and lagged MHW.
    """
    monthly = df_full[["Datetime", "mhw_peak_intensity"]].copy().sort_values("Datetime")
    if opt_lag > 0:
        monthly["mhw_lagged"] = monthly["mhw_peak_intensity"].shift(opt_lag)
    else:
        monthly["mhw_lagged"] = monthly["mhw_peak_intensity"]

    # Merge real EC50 measurements; interpolate gaps
    ec50_monthly = (df_real[["Datetime", "EC50"]]
                    .set_index("Datetime")
                    .reindex(monthly.set_index("Datetime").index))
    monthly = monthly.set_index("Datetime")
    monthly["EC50"] = ec50_monthly["EC50"].interpolate("linear")
    monthly = monthly.dropna(subset=["EC50", "mhw_lagged"]).reset_index()
    return monthly


def project_mhw(mhw_annual: pd.DataFrame, n_months: int,
                scenario: str, last_year: int) -> pd.Series:
    """
    Linear trend on annual max intensity, scaled per scenario,
    seasonalised with a summer peak.
    """
    scale = {"bad": 1.3, "mean": 1.0, "good": 0.7}[scenario]
    years, intensity = mhw_annual["year"].values, mhw_annual["max_intensity"].values
    m, b, *_ = stats.linregress(years, intensity)

    future_dates = pd.date_range(
        start=pd.Timestamp(f"{last_year + 1}-01-01"),
        periods=n_months, freq="MS",
    )
    yr_frac = future_dates.year.values + future_dates.month.values / 12.0
    trend   = np.clip(m * yr_frac + b, 0, None)
    seasonal = 1.0 + 0.5 * np.sin(2 * np.pi * (future_dates.month.values - 8) / 12)
    return pd.Series((trend * seasonal * scale), index=future_dates)


def run():
    df, df_real, _, _ = load_data()

    mhw_annual_path = RESULTS.parent / "mhw_annual.csv"
    mhw_annual = pd.read_csv(mhw_annual_path) if mhw_annual_path.exists() else pd.DataFrame()

    # Optimal lag
    opt_lag = find_optimal_lag(df_real, df)
    print(f"  Optimal MHW→EC50 lag: {opt_lag} months")

    # Build regular monthly series
    monthly = build_monthly_series(df_real, df, opt_lag)
    monthly = monthly.set_index("Datetime").asfreq("MS")

    ec50_series = monthly["EC50"]
    mhw_hist    = monthly[["mhw_lagged"]]

    last_date  = df["Datetime"].max()
    last_year  = last_date.year
    n_months   = FORECAST_YEARS * 12

    meta = {"optimal_lag": int(opt_lag), "scenarios": {}}

    for scenario in ["bad", "mean", "good"]:
        mhw_future_s = project_mhw(mhw_annual, n_months, scenario, last_year)
        mhw_future   = pd.DataFrame({"mhw_lagged": mhw_future_s})

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    ec50_series,
                    exog=mhw_hist,
                    order=(1, 0, 1),
                    seasonal_order=(1, 0, 1, 12),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False)
                fc  = fit.get_forecast(steps=n_months, exog=mhw_future)
            fc_mean = np.asarray(fc.predicted_mean)
            fc_ci   = np.asarray(fc.conf_int(alpha=0.05))
            fc_lo, fc_hi = fc_ci[:, 0], fc_ci[:, 1]
        except Exception as e:
            print(f"  SARIMAX ({scenario}): {e} — using OLS trend")
            slope, intercept, *_ = stats.linregress(
                np.arange(len(ec50_series)), ec50_series.values)
            fc_mean = intercept + slope * (np.arange(n_months) + len(ec50_series))
            fc_std  = np.std(ec50_series.values - (intercept + slope * np.arange(len(ec50_series))))
            scale   = {"bad": 1.0, "mean": 0.0, "good": -1.0}[scenario]
            fc_mean = fc_mean + scale * 0.5 * fc_std   # separate scenarios slightly
            fc_lo   = fc_mean - 1.96 * fc_std
            fc_hi   = fc_mean + 1.96 * fc_std

        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=n_months, freq="MS",
        )
        out = pd.DataFrame({
            "Datetime":           future_dates,
            "EC50_forecast":      fc_mean,
            "CI_lower":           fc_lo,
            "CI_upper":           fc_hi,
            "scenario":           scenario,
            "mhw_intensity_proj": mhw_future_s.values,
        })
        out.to_csv(RESULTS / f"forecast_{scenario}.csv", index=False)
        meta["scenarios"][scenario] = {
            "final_mean":     float(fc_mean[-1]),
            "final_ci_lower": float(fc_lo[-1]),
            "final_ci_upper": float(fc_hi[-1]),
        }

    (RESULTS / "forecast_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"✓ forecast: 3 scenarios × {n_months} months ({last_year+1}–{last_year+FORECAST_YEARS})")
    for sc, v in meta["scenarios"].items():
        print(f"  {sc}: EC50 final = {v['final_mean']:.2f} [{v['final_ci_lower']:.2f}, {v['final_ci_upper']:.2f}]")


if __name__ == "__main__":
    run()
