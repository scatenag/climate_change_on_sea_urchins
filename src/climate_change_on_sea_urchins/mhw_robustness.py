"""
Robustness battery for the MHW -> EC50 lag hypothesis: a set of methodologically
independent checks beyond the primary CCF/Granger/SEA/DLNM suite, added after
the 2026-07 site-coordinate bug investigation to give the dashboard an honest,
reproducible answer to "is there really no way to find this relationship?"
rather than a one-off exploration that lives only in a chat transcript.

Methods (each answers the same question from a different angle):
  1. Severe/Extreme-only driver CCF — only intense events count, not all MHWs.
  2. Direct summer (JJA) temperature anomaly vs EC50 — bypasses MHW detection
     entirely, tests the more fundamental thermal-stress hypothesis.
  3. Random Forest / Gradient Boosting on all lagged MHW features at once,
     scored with blocked time-series cross-validation — a nonlinear,
     multivariate check for combined/interaction effects a pairwise CCF
     would miss.
  4. Convergent Cross Mapping (Sugihara et al. 2012) — nonlinear dynamical-
     systems causality, doesn't assume linearity or Granger's precedence
     logic.
  5. Wavelet coherence in the 1-8 month band, with circular-shift surrogate
     significance — checks for time-localized coupling that a whole-record
     linear correlation could average away.

Outputs (results/):
    results/robustness_severe_ccf.csv
    results/robustness_summer_temp.csv
    results/robustness_ml_importance.csv
    results/robustness_ml_cv_r2.json
    results/robustness_ccm.csv
    results/robustness_wavelet.json
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from .common import load_data, RESULTS, TAU_MAX
from .mhw_analysis import compute_ccf, difference_series, compute_ccf_prewhitened

RNG_SEED = 0


# ── 1. Severe/Extreme-only driver ───────────────────────────────────────────

def run_severe_ccf(df_full: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    sev = events[events["category"].isin(["Severe", "Extreme"])].copy()
    sev["peak_month"] = pd.to_datetime(sev["peak_date"]).dt.to_period("M").dt.to_timestamp()
    sev_monthly = sev.groupby("peak_month")["intensity_max"].max().rename("mhw_severe_intensity")

    df2 = df_full.merge(sev_monthly, left_on="Datetime", right_index=True, how="left")
    df2["mhw_severe_intensity"] = df2["mhw_severe_intensity"].fillna(0.0)

    driver, targets = "mhw_severe_intensity", ["EC50"]
    df_ccf = df2.copy()
    df_ccf.loc[df_ccf["EC50_imputed"], "EC50"] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = compute_ccf(df_ccf, driver, targets)
        df_diff = difference_series(df2, [driver] + targets)
        df_diff.loc[df2["EC50_imputed"].values, "EC50"] = np.nan
        diff = compute_ccf(df_diff, driver, targets)
        pw, _diag = compute_ccf_prewhitened(df2, driver, targets)

    raw = raw.rename(columns={"spearman_r": "r_raw", "p_value": "p_raw"})
    diff = diff.rename(columns={"spearman_r": "r_diff", "p_value": "p_diff"})
    pw = pw.rename(columns={"spearman_r": "r_arima", "p_value": "p_arima"})
    out = raw.merge(diff[["lag", "r_diff", "p_diff"]], on="lag") \
             .merge(pw[["lag", "r_arima", "p_arima"]], on="lag", how="left")
    return out[["lag", "r_raw", "p_raw", "r_diff", "p_diff", "r_arima", "p_arima", "n"]]


# ── 2. Direct summer temperature anomaly ────────────────────────────────────

def run_summer_temp(df_full: pd.DataFrame, df_real: pd.DataFrame) -> pd.DataFrame:
    d = df_full.sort_values("Datetime").reset_index(drop=True).copy()
    d["temp_trend"] = d["Temperature"].rolling(window=25, center=True, min_periods=12).mean()
    d["temp_detrended"] = d["Temperature"] - d["temp_trend"]
    jja = d[d["Datetime"].dt.month.isin([6, 7, 8])].groupby(
        d["Datetime"].dt.year)["temp_detrended"].mean().rename("jja_temp_anom")

    rows = []
    for _, r in df_real.iterrows():
        dt = r["Datetime"]
        for ref_year in [dt.year, dt.year - 1]:
            ref_date = pd.Timestamp(f"{ref_year}-07-15")
            lag_months = (dt.year - ref_date.year) * 12 + (dt.month - ref_date.month)
            if 0 <= lag_months <= 12 and ref_year in jja.index:
                rows.append(dict(lag=lag_months, EC50=r["EC50"], jja_anom=jja.loc[ref_year]))
    lag_df = pd.DataFrame(rows)

    out = []
    for lag in range(13):
        sub = lag_df[lag_df.lag == lag]
        if len(sub) >= 10:
            r, p = stats.spearmanr(sub["jja_anom"], sub["EC50"])
            out.append(dict(lag=lag, n=len(sub), r=r, p=p))
    return pd.DataFrame(out)


# ── 3. ML: all lags at once, out-of-sample ──────────────────────────────────

def run_ml_battery(df_full: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df_full.sort_values("Datetime").reset_index(drop=True)
    feat = pd.DataFrame({"Datetime": d["Datetime"]})
    for lag in range(0, TAU_MAX + 1):
        feat[f"mhw_peak_lag{lag}"] = d["mhw_peak_intensity"].shift(lag)
        feat[f"mhw_days_lag{lag}"] = d["mhw_days"].shift(lag)
    feat["time_index"] = (d["Datetime"] - d["Datetime"].min()).dt.days / 365.25
    feat["month_sin"] = np.sin(2 * np.pi * d["Datetime"].dt.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * d["Datetime"].dt.month / 12)

    real_mask = ~d["EC50_imputed"]
    Xr = feat.drop(columns=["Datetime"])[real_mask.values].reset_index(drop=True)
    yr = d["EC50"][real_mask.values].reset_index(drop=True)
    Xr = Xr.dropna()
    yr = yr.loc[Xr.index].reset_index(drop=True)
    Xr = Xr.reset_index(drop=True)

    mhw_cols = [c for c in Xr.columns if c.startswith("mhw_")]
    trend_cols = ["time_index", "month_sin", "month_cos"]

    def cv_r2(X, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            if len(train_idx) < 20 or len(test_idx) < 5:
                continue
            m = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0)
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores.append(r2_score(y.iloc[test_idx], m.predict(X.iloc[test_idx])))
        return scores

    s_trend = cv_r2(Xr[trend_cols], yr)
    s_full = cv_r2(Xr[trend_cols + mhw_cols], yr)

    rf = RandomForestRegressor(n_estimators=500, max_depth=4, random_state=RNG_SEED, min_samples_leaf=5)
    rf.fit(Xr[trend_cols + mhw_cols], yr)
    pi = permutation_importance(rf, Xr[trend_cols + mhw_cols], yr, n_repeats=100,
                                 random_state=RNG_SEED, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=trend_cols + mhw_cols).sort_values(ascending=False)
    imp_df = imp.reset_index()
    imp_df.columns = ["feature", "permutation_importance"]

    summary = dict(
        n_obs=int(len(Xr)),
        cv_r2_trend_only=[float(s) for s in s_trend],
        cv_r2_trend_only_mean=float(np.mean(s_trend)),
        cv_r2_with_mhw=[float(s) for s in s_full],
        cv_r2_with_mhw_mean=float(np.mean(s_full)),
        mhw_helps_out_of_sample=bool(np.mean(s_full) > np.mean(s_trend)),
    )
    return imp_df, summary


# ── 4. Convergent Cross Mapping ──────────────────────────────────────────────

def run_ccm(df_full: pd.DataFrame) -> pd.DataFrame:
    import skccm as ccm
    from skccm.utilities import train_test_split

    d = df_full.sort_values("Datetime").reset_index(drop=True)
    x = d["mhw_peak_intensity"].values.astype(float)
    y = d["EC50"].values.astype(float)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    E, tau = 3, 1
    X1 = ccm.Embed(x).embed_vectors_1d(tau, E)
    X2 = ccm.Embed(y).embed_vectors_1d(tau, E)
    x1tr, x1te, x2tr, x2te = train_test_split(X1, X2, percent=0.75)

    model = ccm.CCM()
    lib_lens = np.arange(10, min(len(x1tr), len(x2tr)), 8)
    model.fit(x1tr, x2tr)
    model.predict(x1te, x2te, lib_lengths=lib_lens)
    sc1, sc2 = model.score()

    return pd.DataFrame({
        "lib_length": lib_lens,
        "skill_mhw_to_ec50": sc1,   # EC50's manifold recovering MHW info -> evidence MHW drives EC50
        "skill_ec50_to_mhw": sc2,   # reverse direction, should be weaker if causality is one-directional
    })


# ── 5. Wavelet coherence with surrogate significance ────────────────────────

def run_wavelet_coherence(df_full: pd.DataFrame, n_surrogates: int = 100) -> dict:
    import pycwt as wavelet

    d = df_full.sort_values("Datetime").reset_index(drop=True)
    x = d["mhw_peak_intensity"].values.astype(float)
    y = d["EC50"].values.astype(float)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    mother = wavelet.Morlet(6)
    WCT, _aWCT, _coi, freq, _sig = wavelet.wct(x, y, 1, sig=False, mother=mother)
    period = 1 / freq
    band_mask = (period >= 1) & (period <= 8)
    observed = float(WCT[band_mask, :].mean())

    rng = np.random.default_rng(RNG_SEED)
    n = len(y)
    null_stats = []
    for _ in range(n_surrogates):
        shift = rng.integers(12, n - 12)
        y_shift = np.roll(y, shift)
        WCT_s, *_ = wavelet.wct(x, y_shift, 1, sig=False, mother=mother)
        null_stats.append(float(WCT_s[band_mask, :].mean()))
    null_stats = np.array(null_stats)
    p_value = float((null_stats >= observed).mean())

    return dict(
        band_months=[1, 8], observed_mean_coherence=observed,
        null_mean=float(null_stats.mean()), null_sd=float(null_stats.std()),
        n_surrogates=n_surrogates, p_value=p_value,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    df_full, df_real, events, monthly = load_data()

    print("── Robustness battery: 5 independent checks on the MHW->EC50 lag hypothesis ──")

    severe = run_severe_ccf(df_full, events)
    severe.to_csv(RESULTS / "robustness_severe_ccf.csv", index=False)
    print(f"✓ 1/5 Severe/Extreme-only CCF ({len(events[events.category.isin(['Severe','Extreme'])])} events)")

    summer = run_summer_temp(df_full, df_real)
    summer.to_csv(RESULTS / "robustness_summer_temp.csv", index=False)
    print("✓ 2/5 Direct summer-temperature CCF")

    imp_df, ml_summary = run_ml_battery(df_full)
    imp_df.to_csv(RESULTS / "robustness_ml_importance.csv", index=False)
    (RESULTS / "robustness_ml_cv_r2.json").write_text(json.dumps(ml_summary, indent=2))
    print(f"✓ 3/5 ML battery (RF/GBM, {ml_summary['n_obs']} obs, "
          f"MHW {'helps' if ml_summary['mhw_helps_out_of_sample'] else 'does NOT help'} out-of-sample)")

    try:
        ccm_df = run_ccm(df_full)
        ccm_df.to_csv(RESULTS / "robustness_ccm.csv", index=False)
        print("✓ 4/5 Convergent Cross Mapping")
    except ImportError:
        print("⚠ 4/5 Convergent Cross Mapping skipped (skccm not installed)")

    try:
        wct = run_wavelet_coherence(df_full)
        (RESULTS / "robustness_wavelet.json").write_text(json.dumps(wct, indent=2))
        print(f"✓ 5/5 Wavelet coherence (1-8mo band, surrogate p={wct['p_value']:.3f})")
    except ImportError:
        print("⚠ 5/5 Wavelet coherence skipped (pycwt not installed)")

    print("✓ mhw_robustness complete")


if __name__ == "__main__":
    run()
