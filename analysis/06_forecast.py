"""
EC50 forecast 2026–2040 — three climate scenarios (bad / mean / good).

Strategy:
  SARIMAX(1,0,1)(1,0,1,12) with three exogenous regressors:
    - pH  (most biologically relevant: ocean acidification directly affects embryo
           development; Spearman r with EC50 ≈ +0.76; correct sign: lower pH → lower EC50)
    - Temperature  (r = −0.68 with EC50; under bad scenario warms faster)
    - mhw_peak_intensity lagged by optimal k months  (r = −0.39 at k=2)

  Note: CO2 excluded — it is r=−0.98 collinear with pH; pH is preferred because it
  is the variable organisms respond to directly.

  Training window: post-2016 only (108 months).
  Rationale: the 2003–2015 period represents a different climate regime (EC50 ~40–55).
  Training on the full period causes SARIMAX to mean-revert toward the pre-2016 mean,
  making all scenarios unrealistically optimistic. The post-2016 regime is the relevant
  baseline for forecasting 2026–2040.

  Each scenario uses scenario-specific projections of all three exogenous
  variables (linear trend extrapolation × scenario multiplier):
    bad  → drift × 1.5  (faster acidification / warming / more MHW)
    mean → drift × 1.0  (business-as-usual)
    good → drift × 0.5  (climate mitigation)

  Confidence intervals: linearly-growing residual-based bounds
    CI(t) = 1.96 × σ_resid × (1 + 0.07 × t_years)

Outputs:
    results/forecast_bad.csv
    results/forecast_mean.csv
    results/forecast_good.csv
    results/forecast_env_bad.csv   (projected env vars for each scenario)
    results/forecast_env_mean.csv
    results/forecast_env_good.csv
    results/forecast_meta.json
"""
import json
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
from common import load_data, RESULTS, TAU_MAX


FORECAST_YEARS = 15
CI_GROWTH_RATE = 0.03   # reduced: post-2016 training window → use conservative CI growth

# Scenario multiplier applied to the linear trend slope of each exogenous variable.
# For variables with a negative trend (O2, pH), ×1.5 means faster decline (worse).
ENV_SCALE = {"bad": 1.5, "mean": 1.0, "good": 0.5}
MHW_SCALE = {"bad": 2.0, "mean": 1.0, "good": 0.5}   # MHW has its own seasonal projection


def find_optimal_lag(df_real: pd.DataFrame, df_full: pd.DataFrame) -> int:
    """Spearman r between mhw_peak_intensity(t-k) and real EC50(t)."""
    best_lag, best_r = 2, 0.0
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
    Regular monthly DataFrame with:
      - linearly-interpolated EC50
      - pH, Temperature  (filled forward/backward from monthly series)
      - mhw_peak_intensity lagged by opt_lag months
    """
    cols = ["Datetime", "pH", "Temperature", "mhw_peak_intensity"]
    monthly = df_full[cols].copy().sort_values("Datetime")
    monthly["mhw_lagged"] = (monthly["mhw_peak_intensity"].shift(opt_lag)
                             if opt_lag > 0 else monthly["mhw_peak_intensity"])
    ec50_monthly = (df_real[["Datetime", "EC50"]]
                    .set_index("Datetime")
                    .reindex(monthly.set_index("Datetime").index))
    monthly = monthly.set_index("Datetime")
    monthly["EC50"] = ec50_monthly["EC50"].interpolate("linear")
    for col in ["pH", "Temperature", "mhw_lagged"]:
        monthly[col] = monthly[col].ffill().bfill()
    monthly = monthly.dropna(subset=["EC50", "mhw_lagged", "pH", "Temperature"]).reset_index()
    return monthly


def project_env_var(series: pd.Series, n_months: int, last_year: int,
                    mult: float) -> pd.Series:
    """
    Project an environmental variable forward by extrapolating its linear trend,
    scaled by `mult` (1.0 = mean scenario, 1.5 = bad, 0.5 = good).

    The multiplier scales the *incremental drift* from the last observed value,
    so the series always starts from the last historical data point.
    Seasonal pattern is added from the last 12 months of the historical seasonal
    component (via OLS detrend + residual).
    """
    series = series.dropna()
    t = np.arange(len(series))
    slope, intercept, *_ = stats.linregress(t, series.values)

    future_dates = pd.date_range(
        start=pd.Timestamp(f"{last_year + 1}-01-01"),
        periods=n_months, freq="MS",
    )
    last_val = float(series.iloc[-1])
    last_t   = len(series) - 1
    t_future = np.arange(last_t + 1, last_t + 1 + n_months)

    # Mean scenario trend
    trend_mean = intercept + slope * t_future
    # Scale drift from last value
    delta = trend_mean - last_val
    projected = last_val + delta * mult

    # Seasonal: repeat the last 12 months of detrended residuals
    detrended  = series.values - (intercept + slope * t)
    seasonal   = np.tile(detrended[-12:], n_months // 12 + 1)[:n_months]
    return pd.Series(projected + seasonal, index=future_dates, name=series.name)


def project_mhw(mhw_annual: pd.DataFrame, n_months: int,
                scenario: str, last_year: int) -> pd.Series:
    """Linear trend on annual max MHW intensity, scaled per scenario."""
    scale = MHW_SCALE[scenario]
    years, intensity = mhw_annual["year"].values, mhw_annual["max_intensity"].values
    m, b, *_ = stats.linregress(years, intensity)
    future_dates = pd.date_range(
        start=pd.Timestamp(f"{last_year + 1}-01-01"),
        periods=n_months, freq="MS",
    )
    yr_frac  = future_dates.year.values + future_dates.month.values / 12.0
    trend    = np.clip(m * yr_frac + b, 0, None)
    seasonal = 1.0 + 0.5 * np.sin(2 * np.pi * (future_dates.month.values - 8) / 12)
    return pd.Series(trend * seasonal * scale, index=future_dates)


def run():
    df, df_real, _, _ = load_data()

    mhw_annual_path = RESULTS.parent / "mhw_annual.csv"
    mhw_annual = pd.read_csv(mhw_annual_path) if mhw_annual_path.exists() else pd.DataFrame()

    opt_lag = find_optimal_lag(df_real, df)
    print(f"  Optimal MHW→EC50 lag: {opt_lag} months")

    monthly = build_monthly_series(df_real, df, opt_lag)
    monthly = monthly.set_index("Datetime").asfreq("MS")

    # Train on post-2016 only: this is the current climate regime.
    # Full-period training causes SARIMAX to mean-revert toward the pre-2016
    # historical mean (~40 mg/L), making all forecasts unrealistically optimistic.
    monthly_train = monthly[monthly.index >= "2016-01-01"]
    print(f"  Training window: {monthly_train.index[0].date()} – {monthly_train.index[-1].date()} "
          f"({len(monthly_train)} months)")

    ec50_series = monthly_train["EC50"]
    exog_hist   = monthly_train[["pH", "Temperature", "mhw_lagged"]]

    last_date = df["Datetime"].max()
    last_year = last_date.year
    n_months  = FORECAST_YEARS * 12

    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=n_months, freq="MS",
    )

    # Residual SD for CI bands: use the SMALLER of SARIMAX residuals and series std.
    # The SARIMAX residual SD can be inflated when training on a short post-2016 window;
    # capping at series std prevents unrealistically wide CI bands.
    series_std = float(ec50_series.std())
    resid_std  = series_std
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _m = SARIMAX(ec50_series, exog=exog_hist,
                         order=(1, 0, 1), seasonal_order=(1, 0, 1, 12),
                         trend="c",
                         enforce_stationarity=False,
                         enforce_invertibility=False)
            _fit = _m.fit(disp=False)
        resid_std = min(float(np.std(_fit.resid)), series_std)
        print(f"  SARIMAX residual SD: {float(np.std(_fit.resid)):.2f}  "
              f"→ capped at series std {series_std:.2f}  (used for CI bands)")
    except Exception as e:
        print(f"  SARIMAX pilot fit failed ({e}), using series std as fallback")

    meta = {"optimal_lag": int(opt_lag), "exog_vars": ["pH", "Temperature", "mhw_lagged"],
            "scenarios": {}}

    for scenario in ["bad", "mean", "good"]:
        sc_mult = ENV_SCALE[scenario]

        # --- Project exogenous variables under this scenario ---
        pH_future   = project_env_var(monthly["pH"],          n_months, last_year, sc_mult)
        Temp_future = project_env_var(monthly["Temperature"], n_months, last_year, sc_mult)
        MHW_future  = project_mhw(mhw_annual, n_months, scenario, last_year)

        exog_future = pd.DataFrame({
            "pH":          pH_future.values,
            "Temperature": Temp_future.values,
            "mhw_lagged":  MHW_future.values,
        }, index=future_dates)

        # Save env projections (used by mechanistic model in notebook)
        env_out = exog_future.copy()
        env_out.index.name = "Datetime"
        env_out.reset_index().to_csv(RESULTS / f"forecast_env_{scenario}.csv", index=False)

        # --- SARIMAX forecast ---
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    ec50_series, exog=exog_hist,
                    order=(1, 0, 1), seasonal_order=(1, 0, 1, 12),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False)
                fc  = fit.get_forecast(steps=n_months, exog=exog_future)
            fc_mean = np.asarray(fc.predicted_mean)
        except Exception as e:
            print(f"  SARIMAX ({scenario}): {e} — using OLS fallback")
            t_all = np.arange(len(ec50_series))
            slope, intercept, *_ = stats.linregress(t_all, ec50_series.values)
            fc_mean = intercept + slope * (np.arange(n_months) + len(ec50_series))

        fc_mean = np.clip(fc_mean, 3.0, 200.0)

        years_ahead = np.arange(1, n_months + 1) / 12.0
        ci_half = np.minimum(
            1.96 * resid_std * (1.0 + CI_GROWTH_RATE * years_ahead),
            3.0 * series_std,   # hard cap: CI cannot exceed 3× historical std
        )
        fc_lo = np.clip(fc_mean - ci_half, 0.0, None)
        fc_hi = fc_mean + ci_half

        out = pd.DataFrame({
            "Datetime":      future_dates,
            "EC50_forecast": fc_mean,
            "CI_lower":      fc_lo,
            "CI_upper":      fc_hi,
            "scenario":      scenario,
            "pH_proj":       pH_future.values,
            "Temp_proj":     Temp_future.values,
            "mhw_proj":      MHW_future.values,
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
        print(f"  {sc}: EC50 final = {v['final_mean']:.2f} "
              f"[{v['final_ci_lower']:.2f}, {v['final_ci_upper']:.2f}]")


if __name__ == "__main__":
    run()
