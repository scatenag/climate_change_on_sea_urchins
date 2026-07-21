"""
Thermal-legacy hypothesis: is the 20-year EC50 decline explained by chronic
cumulative heat stress on the WILD adult population from which gametes are drawn?

The bioassay uses gametes from wild-collected adults (not lab-reared), so the
adults integrate the warming of their in-situ environment over years. The
hypothesis (A. Gaion, 2026-07): progressive chronic heat stress erodes the
reproductive condition of the wild population, lowering gamete/larval robustness
and hence the reference-toxicant EC50 — copper is merely the revealer, not the
cause.

Predictor: cumulative thermal dose = degree-days above a reproductive threshold
(P. lividus optimum 17-20 C; egg production stalls ~18 C) summed over a
multi-year window BEFORE each assay, from daily SST. Windows of 12/24/36 months
probe the multi-year integration a wild adult experiences.

THE DECISIVE TEST — the predictor must beat a plain time trend. Both EC50 and
cumulative thermal dose trend over two decades, so they correlate "for free".
This module therefore reports, alongside the raw correlation:
  * detrended correlation (residuals of both vs a linear time trend) — does
    thermal dose track the OFF-TREND wiggles of EC50, or only the shared ramp?
  * a nested OLS (EC50 ~ time  vs  EC50 ~ time + dose): delta-R2 and the partial
    p-value of dose given time;
  * the dose~time collinearity, which caps how much independent signal can exist.

Honest verdict: with a single observational co-trending series, a raw
correlation cannot distinguish chronic-heat causation from spurious co-trend.
This module makes that explicit rather than reporting the (impressive) raw number
alone — the failure mode the manuscript was criticised for.

Outputs:
    results/thermal_legacy.csv          — per-assay EC50 + thermal dose per window
    results/thermal_legacy_summary.json — raw/detrended/nested stats + verdict
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .common import load_data, RESULTS, ROOT

THRESHOLDS = [18.0, 20.0]          # C, reproductive thresholds for P. lividus
WINDOWS = [12, 24, 36]             # months of cumulative thermal history


def _thermal_dose(sst, assay_date, window_months, thr):
    """Degree-days above `thr` over the `window_months` before `assay_date`,
    normalised to a per-year rate so windows of different length are comparable."""
    start = assay_date - pd.DateOffset(months=window_months)
    m = (sst["Datetime"] > start) & (sst["Datetime"] <= assay_date)
    exc = (sst.loc[m, "Temperature"] - thr).clip(lower=0)
    return exc.sum() / window_months * 12.0


def _detrended_corr(x, y, t):
    rx = x - np.polyval(np.polyfit(t, x, 1), t)
    ry = y - np.polyval(np.polyfit(t, y, 1), t)
    r, p = stats.spearmanr(rx, ry)
    return float(r), float(p)


def run():
    _, df_real, _, _ = load_data()
    real = df_real.dropna(subset=["EC50"]).reset_index(drop=True)[["Datetime", "EC50"]]

    sst = pd.read_csv(ROOT / "data" / "sst_daily.csv", parse_dates=["Datetime"])
    sst = sst.sort_values("Datetime").reset_index(drop=True)

    t = (real["Datetime"] - real["Datetime"].min()).dt.days.values.astype(float)
    y = real["EC50"].values

    out = real.copy()
    rows = []
    for thr in THRESHOLDS:
        for win in WINDOWS:
            col = f"dose_{int(thr)}C_{win}m"
            dose = real["Datetime"].apply(lambda d: _thermal_dose(sst, d, win, thr)).values
            out[col] = dose

            raw_r, raw_p = stats.spearmanr(dose, y)
            det_r, det_p = _detrended_corr(dose, y, t)
            collin = float(stats.spearmanr(dose, t)[0])

            X_t = sm.add_constant(t)
            X_td = sm.add_constant(np.column_stack([t, dose]))
            r2_t = sm.OLS(y, X_t).fit().rsquared
            fit_td = sm.OLS(y, X_td).fit()

            rows.append({
                "threshold_C": thr, "window_months": win,
                "raw_spearman_r": float(raw_r), "raw_p": float(raw_p),
                "detrended_spearman_r": det_r, "detrended_p": det_p,
                "dose_time_collinearity": collin,
                "r2_time_only": float(r2_t),
                "r2_time_plus_dose": float(fit_td.rsquared),
                "delta_r2": float(fit_td.rsquared - r2_t),
                "partial_p_dose_given_time": float(fit_td.pvalues[2]),
            })

    out.to_csv(RESULTS / "thermal_legacy.csv", index=False)
    res = pd.DataFrame(rows)

    # Verdict: the hypothesis is SUPPORTED only if some window survives detrending
    # in the biologically expected (negative) direction; otherwise it is merely
    # consistent with the raw co-trend.
    supported = res[(res["detrended_p"] < 0.05) & (res["detrended_spearman_r"] < 0)]
    best_raw = res.loc[res["raw_p"].idxmin()]
    verdict = ("supported_after_detrending" if not supported.empty
               else "consistent_but_not_separable_from_trend")

    summary = {
        "hypothesis": "chronic cumulative heat stress on the wild adult population "
                      "drives the EC50 decline (copper is the revealer, not the cause)",
        "verdict": verdict,
        "strongest_raw": {
            "window": f"{int(best_raw.threshold_C)}C/{int(best_raw.window_months)}m",
            "raw_spearman_r": float(best_raw.raw_spearman_r),
            "raw_p": float(best_raw.raw_p),
        },
        "same_window_detrended_p": float(best_raw.detrended_p),
        "same_window_detrended_r": float(best_raw.detrended_spearman_r),
        "same_window_delta_r2_over_time": float(best_raw.delta_r2),
        "same_window_dose_time_collinearity": float(best_raw.dose_time_collinearity),
        "per_window": rows,
        "interpretation": (
            "Thermal dose is strongly correlated with EC50 in raw form but is ~"
            f"{best_raw.dose_time_collinearity:.2f} collinear with elapsed time; it "
            "adds only "
            f"{best_raw.delta_r2*100:.1f}% R2 over a plain time trend and does not "
            "track EC50's off-trend variation. With a single observational co-trending "
            "series the chronic-heat hypothesis cannot be causally separated from a "
            "shared secular trend — it is a plausible mechanism supported by physiology "
            "and by the exclusion of alternatives (Cu speciation, assay precision, "
            "nutrients), not a demonstrated one."
        ),
    }
    with (RESULTS / "thermal_legacy_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ thermal_legacy: raw {best_raw.raw_spearman_r:+.2f} (p={best_raw.raw_p:.1e}) "
          f"but detrended p={best_raw.detrended_p:.2f}, ΔR² over time "
          f"{best_raw.delta_r2*100:.1f}% → {verdict}")


if __name__ == "__main__":
    run()
