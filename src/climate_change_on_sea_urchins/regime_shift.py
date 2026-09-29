"""
Regime-shift analysis: when and how did the wild sentinel population change state?

Documents the 2016 shift in copper EC50 and frames it against three things:

 (1) A multifactorial environmental stress index — PC1 of the DESEASONALISED
     monthly anomalies of the five climate-change-sensitive variables
     (T, S, CO2, O2, pH). One coordinated axis (warming + acidification +
     deoxygenation) captures most of their common variance, quantifying
     A. Gaion's "the stress is the combination of parameters" intuition.

 (2) Marine-heatwave EXPOSURE. MHW metrics have their OWN regime shift, and it
     PRECEDES the biological collapse: MHW days / cumulative intensity jump
     ~2013, EC50 collapses ~2016 — a ~3-year, population-scale accumulation lag.
     This is NOT the acute 2-month lag earlier analyses (and an independent 2025
     experiment) reject; it is a multi-year integration by long-lived wild adults.

 (3) Early-warning signals of a dynamical critical transition (rolling variance
     and lag-1 autocorrelation / "critical slowing down"). Reported HONESTLY:
     the canonical precursors are NOT present (absolute variance falls, AR(1)
     does not rise significantly). So this is a documented REGIME SHIFT, not a
     demonstrated tipping point — a distinction a reviewer will demand.

All correlations of the stress index with EC50 are co-trended and cannot prove
causation; the contribution here is timing/state-change description plus the
exclusion of alternatives (see cu_speciation, thermal_legacy), not a causal claim.

Method: Pettitt non-parametric changepoint; PCA (SVD) on deseasonalised anomalies;
EWS via Kendall-tau trend of rolling variance and AR(1) on detrended residuals.

Outputs:
    results/regime_shift_changepoints.csv  — Pettitt break + p per series
    results/regime_shift_summary.json      — stress index, exposure→response lag, EWS verdict
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

from .common import load_data, default_results_dir, SPLIT_YEAR, RESPONSE_COL, default_response_spec, load_mhw_annual

ENV = ["Temperature", "Salinity", "CO2", "O2", "pH"]

# Significance threshold of this module: for the early-warning trends and
# for calling a Pettitt break significant in the verdict text.
ALPHA = 0.05
# Sign of each variable along the climate-change stress axis (stress increases
# with warming, salinification, rising CO2; falling O2, falling pH).
STRESS_SIGN = {"Temperature": 1, "Salinity": 1, "CO2": 1, "O2": -1, "pH": -1}


def pettitt(x):
    """Pettitt (1979) non-parametric single-changepoint test.
    Returns (break_index, approx_two_sided_p)."""
    x = np.asarray(x, float)
    n = len(x)
    r = stats.rankdata(x)
    U = np.array([2 * r[:k].sum() - k * (n + 1) for k in range(1, n + 1)])
    K = np.abs(U)
    k = int(np.argmax(K))
    p = 2.0 * np.exp(-6.0 * K[k] ** 2 / (n ** 3 + n ** 2))
    return k, float(min(p, 1.0))


def _deseasonalise(df, cols):
    """Subtract each variable's month-of-year mean (anomalies)."""
    out = df[["Datetime"]].copy()
    m = df["Datetime"].dt.month
    for c in cols:
        clim = df.groupby(m)[c].transform("mean")
        out[c] = df[c] - clim
    return out


def _stress_index(df_full):
    """PC1 of deseasonalised env anomalies, oriented to increase with time/stress."""
    d = df_full.dropna(subset=ENV).copy()
    anom = _deseasonalise(d, ENV)
    # orient each variable so 'more stress' is +, then z-score
    Z = anom[ENV].mul([STRESS_SIGN[c] for c in ENV], axis=1)
    Z = (Z - Z.mean()) / Z.std()
    u, s, vt = np.linalg.svd(Z.values, full_matrices=False)
    pc1 = u[:, 0] * s[0]
    t = (d["Datetime"] - d["Datetime"].min()).dt.days.values
    if np.corrcoef(pc1, t)[0, 1] < 0:
        pc1, vt = -pc1, -vt
    var_expl = float((s ** 2 / (s ** 2).sum())[0])
    loadings = {c: float(l) for c, l in zip(ENV, vt[0])}
    idx = pd.DataFrame({"Datetime": d["Datetime"].values, "stress_pc1": pc1})
    return idx, var_expl, loadings


def _ews(ec_series):
    """Rolling-variance and AR(1) trend on long-detrended EC50 residuals."""
    trend = ec_series.rolling(25, center=True, min_periods=8).mean()
    resid = (ec_series - trend).dropna()
    W = 30
    rv = resid.rolling(W).var().dropna()
    ar1 = resid.rolling(W).apply(lambda z: pd.Series(z).autocorr(lag=1), raw=False).dropna()
    tv, pv = stats.kendalltau(np.arange(len(rv)), rv.values)
    ta, pa = stats.kendalltau(np.arange(len(ar1)), ar1.values)
    return {
        "variance_kendall_tau": float(tv), "variance_p": float(pv),
        "ar1_kendall_tau": float(ta), "ar1_p": float(pa),
    }


def run(response=None, results=None):
    results = results if results is not None else default_results_dir()
    if response is None:
        response = default_response_spec()
    label = response.label  # display identity for the response series below

    df_full, df_real, _, _ = load_data()
    split_year = int(SPLIT_YEAR)

    rows = []

    # --- response changepoint (monthly real measurements) ---
    r = df_real.dropna(subset=[RESPONSE_COL]).reset_index(drop=True)
    k, p = pettitt(r[RESPONSE_COL].values)
    ec50_break = r["Datetime"].iloc[k]
    rows.append({"series": label, "break_date": ec50_break.date().isoformat(),
                 "break_year": int(ec50_break.year), "p_value": p,
                 "pre_mean": float(r[RESPONSE_COL][:k].mean()), "post_mean": float(r[RESPONSE_COL][k:].mean())})

    # --- MHW exposure changepoints (annual) ---
    ann = load_mhw_annual()
    mhw_break_year = mhw_break_p = mhw_first_after = None
    for c in ["total_mhw_days", "cum_intensity_sum", "max_intensity"]:
        s = ann[c].dropna()
        kk, pp = pettitt(s.values)
        by = int(ann["year"].iloc[kk])
        if c == "total_mhw_days":
            mhw_break_year, mhw_break_p = by, pp
            mhw_first_after = int(ann["year"].iloc[kk + 1]) if kk + 1 < len(ann) else None
        rows.append({"series": f"MHW_{c}", "break_date": f"{by}", "break_year": by,
                     "p_value": pp, "pre_mean": float(s.iloc[:kk].mean()),
                     "post_mean": float(s.iloc[kk:].mean())})

    # --- environmental drivers (annual means) changepoints ---
    ann_env = df_full.assign(year=df_full["Datetime"].dt.year).groupby("year")[ENV].mean()
    for c in ENV:
        s = ann_env[c].dropna()
        kk, pp = pettitt(s.values)
        rows.append({"series": c, "break_date": f"{int(s.index[kk])}",
                     "break_year": int(s.index[kk]), "p_value": pp,
                     "pre_mean": float(s.iloc[:kk].mean()), "post_mean": float(s.iloc[kk:].mean())})

    pd.DataFrame(rows).to_csv(results / "regime_shift_changepoints.csv", index=False)

    # --- multifactorial stress index ---
    stress_idx, var_expl, loadings = _stress_index(df_full)
    stress_idx.to_csv(results / "regime_shift_stress_index.csv", index=False)

    # --- early-warning signals ---
    ews = _ews(r.set_index("Datetime")[RESPONSE_COL])
    csd = (ews["variance_kendall_tau"] > 0 and ews["variance_p"] < ALPHA
           and ews["ar1_kendall_tau"] > 0 and ews["ar1_p"] < ALPHA)

    exposure_lag = (int(ec50_break.year) - mhw_break_year) if mhw_break_year else None

    # Every sentence of the verdict is generated from values computed in this
    # run: a fixed conclusion ("NOT present", "BEFORE the collapse") could
    # contradict the values written right next to it on other data. Each
    # break's p is always reported; a distance in years between two breaks
    # is written only when BOTH are significant at ALPHA -- otherwise a
    # non-significant break would read as an established event and the
    # distance as a lag.
    #
    # Date convention: pettitt() returns the index of the LAST observation
    # before the change, and that is the date saved in break/break_date/
    # break_year; the text also gives the first observation after it (e.g.
    # the response's last month before 2016-05, first month after 2016-06 --
    # the one split_date names). The distance in years compares the last
    # period before each break, like exposure_precedes_response_years.
    # Whether to align every module on one convention is deferred:
    # docs/adr/0000, item 9.
    resp_sig = p < ALPHA
    first_after = r["Datetime"].iloc[k + 1] if k + 1 < len(r) else None
    parts = [
        f"Pettitt break in {label}: last month before the change "
        f"{ec50_break:%Y-%m}, first month after "
        f"{first_after:%Y-%m}" if first_after is not None else
        f"Pettitt break in {label}: last month before the change {ec50_break:%Y-%m}",
    ]
    parts[0] += f" (p={p:.1e}, {'significant' if resp_sig else 'not significant'} at {ALPHA:g})."
    if mhw_break_year is None:
        parts.append("No MHW-exposure break was computed.")
    else:
        mhw_sig = mhw_break_p < ALPHA
        after = f", first year after {mhw_first_after}" if mhw_first_after is not None else ""
        parts.append(
            f"MHW exposure (total MHW days): last year before the change {mhw_break_year}{after} "
            f"(p={mhw_break_p:.2g}, {'significant' if mhw_sig else 'not significant'} at {ALPHA:g})."
        )
        if resp_sig and mhw_sig:
            if exposure_lag > 0:
                parts.append(f"Comparing the last period before each break, the MHW-exposure break "
                             f"is {exposure_lag} yr earlier than the {label} break.")
            elif exposure_lag < 0:
                parts.append(f"Comparing the last period before each break, the MHW-exposure break "
                             f"is {-exposure_lag} yr later than the {label} break.")
            else:
                parts.append(f"Comparing the last period before each break, both fall in the same year.")
        else:
            parts.append("No distance between the two breaks is reported: at least one is not significant.")
    parts.append(
        f"PC1 explains {var_expl * 100:.0f}% of the variance of the deseasonalised "
        "T/S/CO2/O2/pH anomalies. Critical-slowing-down early-warning signals (rolling "
        f"variance AND lag-1 autocorrelation both rising, Kendall p<{ALPHA:g}): "
        f"{'detected' if csd else 'not detected'} "
        f"(variance tau={ews['variance_kendall_tau']:+.2f}, p={ews['variance_p']:.2g}; "
        f"AR(1) tau={ews['ar1_kendall_tau']:+.2f}, p={ews['ar1_p']:.2g})."
    )
    verdict = " ".join(parts)
    # Generic key names (an output key is a schema the dashboard reads); the
    # response's identity is recorded as a value, never in a key.
    summary = {
        "response_label": label,
        "response_regime_shift": {"break": ec50_break.date().isoformat(), "p": p,
                                  "pre_mean": float(r[RESPONSE_COL][:k].mean()),
                                  "post_mean": float(r[RESPONSE_COL][k:].mean())},
        "mhw_exposure_break_year": mhw_break_year,
        "exposure_precedes_response_years": exposure_lag,
        "multifactorial_stress_index": {
            "pc1_variance_explained": var_expl,
            "pc1_loadings_stress_oriented": loadings,
            "note": f"PC1 of deseasonalised T/S/CO2/O2/pH anomalies. Its correlation with "
                    f"{label} is co-trended, not causal.",
        },
        "early_warning_signals": ews,
        "critical_slowing_down_detected": bool(csd),
        "verdict": verdict,
    }
    with (results / "regime_shift_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ regime_shift: {label} break {ec50_break.date()} (p={p:.1e}); MHW exposure "
          f"break {mhw_break_year} (~{exposure_lag} yr earlier); stress PC1={var_expl*100:.0f}%; "
          f"critical slowing down: {'YES' if csd else 'NOT detected'}")


if __name__ == "__main__":
    run()
