"""
Thermal-legacy hypothesis: is the 20-year EC50 decline explained by chronic
cumulative heat stress on the WILD adult population from which gametes are drawn?

The bioassay uses gametes from wild-collected adults (not lab-reared), so the
adults integrate the warming of their in-situ environment over years. The
hypothesis (A. Gaion, 2026-07): progressive chronic heat stress erodes the
reproductive condition of the wild population, lowering gamete/larval robustness
and hence the reference-toxicant EC50 — copper is merely the revealer, not the
cause.

Predictor: cumulative thermal dose = degree-days above 24 C summed over a
multi-year window BEFORE each assay, from daily SST. The single 24 C threshold
(D. Sartori, 2026-07-30, replacing two earlier ad hoc 18/20 C proxies with no
clear literature basis) is the chronic exposure level at which P. lividus
gametogenesis has been experimentally shown to collapse: adults held at 24 C
for 6 weeks show gonadal index falling from 5.14 to 1.19 and near-total loss of
germ cells (Amato et al. 2025, J. Mar. Sci. Eng. 13, 2293). Gallo et al. 2023
(Biomolecules 13, 1216) is corroborating but at different temperatures/duration
(17/23/28 C, 7-day ACUTE exposure): significant biomarker shifts and a modest
egg-viability drop (99.6%->~95%) from 23 C, i.e. thermal sensitivity emerging
in that range rather than a direct test of 24 C — cite accordingly, don't
conflate the two as both pinpointing 24 C.

Windows of 12/24/36/48/60 months probe the multi-year integration a wild adult
experiences (60 months ~ the oldest cohort age in the bioassay population).

THE DECISIVE TEST — the predictor must beat a plain time trend. Both EC50 and
cumulative thermal dose trend over two decades, so they correlate "for free".
This module therefore reports, alongside the raw correlation:
  * detrended correlation (residuals of both vs a linear time trend) — does
    thermal dose track the OFF-TREND wiggles of EC50, or only the shared ramp?
  * a nested OLS (EC50 ~ time  vs  EC50 ~ time + dose): delta-R2 and the partial
    p-value of dose given time;
  * the dose~time collinearity, which caps how much independent signal can exist;
  * BH-FDR and Bonferroni correction of the 5 windows, applied to BOTH the
    rank-based (Spearman) and the parametric (OLS partial) test — a window is
    only "robust" if it survives Bonferroni on both; surviving on the rank
    test alone is a fragile, method-dependent result, not a cross-validated
    one (same lag-family-correction principle as the CCF/Granger panels
    elsewhere in this pipeline, extended with a second, independent test).

Honest verdict: with a single observational co-trending series, a raw
correlation cannot distinguish chronic-heat causation from spurious co-trend.
This module makes that explicit rather than reporting the (impressive) raw number
alone — the failure mode the manuscript was criticised for. Verified against our
own data (2026-07-30, re-checked the same day after an initial Spearman-only pass
overstated the 36-month window — see git history): only the 24-month window is
robust to both the rank and the parametric test; 12- and 36-month windows clear
the rank test but fail the parametric one (fragile, not cross-validated); 48/60
month windows clear neither. Dose-time collinearity climbs monotonically with
window length (0.54->0.80), consistent with a real, narrow-window effect being
progressively swamped by shared trend, not an arbitrary cutoff.

Outputs:
    results/thermal_legacy.csv          — per-assay EC50 + thermal dose per window
    results/thermal_legacy_summary.json — raw/detrended/nested stats + verdict
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .common import load_data, RESULTS, ROOT

THRESHOLD_C = 24.0                 # C, chronic gametogenesis-blocking threshold
                                    # for P. lividus (Amato et al. 2025)
WINDOWS = [12, 24, 36, 48, 60]      # months of cumulative thermal history


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
    for win in WINDOWS:
        col = f"dose_{int(THRESHOLD_C)}C_{win}m"
        dose = real["Datetime"].apply(lambda d: _thermal_dose(sst, d, win, THRESHOLD_C)).values
        out[col] = dose

        raw_r, raw_p = stats.spearmanr(dose, y)
        det_r, det_p = _detrended_corr(dose, y, t)
        collin = float(stats.spearmanr(dose, t)[0])

        X_t = sm.add_constant(t)
        X_td = sm.add_constant(np.column_stack([t, dose]))
        r2_t = sm.OLS(y, X_t).fit().rsquared
        fit_td = sm.OLS(y, X_td).fit()

        rows.append({
            "window_months": win,
            "raw_spearman_r": float(raw_r), "raw_p": float(raw_p),
            "detrended_spearman_r": det_r, "detrended_p": det_p,
            "dose_time_collinearity": collin,
            "r2_time_only": float(r2_t),
            "r2_time_plus_dose": float(fit_td.rsquared),
            "delta_r2": float(fit_td.rsquared - r2_t),
            "dose_coef_given_time": float(fit_td.params[2]),
            "partial_p_dose_given_time": float(fit_td.pvalues[2]),
        })

    out.to_csv(RESULTS / "thermal_legacy.csv", index=False)
    res = pd.DataFrame(rows)

    # BH-FDR and Bonferroni correction across the 5 windows — non-independent
    # tests of the same hypothesis, so no single best-looking window can be
    # reported alone (same principle as the CCF/Granger lag-family corrections
    # elsewhere in this pipeline). Applied to BOTH the rank-based detrended
    # Spearman correlation AND the parametric OLS partial p-value: a window
    # that only clears the rank test but not the parametric one (or vice
    # versa) is a fragile, method-dependent result, not a cross-validated one
    # (caught 2026-07-30 when the initial Spearman-only 36-month "survivor"
    # turned out to fail the OLS partial test even uncorrected, p=0.072).
    res["p_fdr"] = multipletests(res["detrended_p"], method="fdr_bh")[1]
    res["p_bonferroni"] = multipletests(res["detrended_p"], method="bonferroni")[1]
    res["partial_p_fdr"] = multipletests(res["partial_p_dose_given_time"], method="fdr_bh")[1]
    res["partial_p_bonferroni"] = multipletests(res["partial_p_dose_given_time"], method="bonferroni")[1]
    rows = res.to_dict("records")

    correct_sign = (res["detrended_spearman_r"] < 0) & (res["dose_coef_given_time"] < 0)
    rank_survives = correct_sign & (res["p_bonferroni"] < 0.05)
    ols_survives = correct_sign & (res["partial_p_bonferroni"] < 0.05)

    robust = sorted(int(w) for w in res.loc[rank_survives & ols_survives, "window_months"])
    suggestive = sorted(int(w) for w in res.loc[rank_survives & ~ols_survives, "window_months"])
    not_surviving = sorted(int(w) for w in res["window_months"] if w not in robust and w not in suggestive)

    if robust:
        verdict = "supported_all_windows" if not (suggestive or not_surviving) else "supported_narrow_window"
    elif suggestive:
        verdict = "suggestive_not_cross_validated"
    else:
        verdict = "consistent_but_not_separable_from_trend"

    def _fmt(ws):
        return ", ".join(f"{w}m" for w in ws) if ws else "none"

    summary = {
        "hypothesis": "chronic cumulative heat stress (degree-days above 24C, the "
                      "gametogenesis-blocking threshold per Amato et al. 2025) on the "
                      "wild adult population drives the EC50 decline (copper is the "
                      "revealer, not the cause)",
        "threshold_C": THRESHOLD_C,
        "windows_months": WINDOWS,
        "verdict": verdict,
        "windows_robust": robust,
        "windows_suggestive_rank_only": suggestive,
        "windows_not_surviving": not_surviving,
        "per_window": rows,
        "interpretation": (
            f"Windows robust to BOTH the rank-based (Spearman, detrended) and the "
            f"parametric (OLS partial) Bonferroni-corrected test, in the biologically "
            f"expected (negative) direction ({_fmt(robust)}): a genuine, "
            f"cross-validated effect of chronic heat dose on EC50 beyond the shared "
            f"trend. Windows significant on the rank test alone but NOT corroborated "
            f"by the parametric partial test ({_fmt(suggestive)}) are fragile/"
            f"method-dependent and should be treated as suggestive, not established. "
            f"Windows surviving neither test ({_fmt(not_surviving)}) have dose-time "
            "collinearity high enough that the detrended residuals are mostly noise. "
            "Only claim the robust window(s) as a finding; report the suggestive ones, "
            "if any, explicitly as unconfirmed by the cross-check."
        ),
    }
    with (RESULTS / "thermal_legacy_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ thermal_legacy (24C threshold, {len(WINDOWS)} windows): "
          f"robust(both tests)={_fmt(robust)}  suggestive(rank-only)={_fmt(suggestive)}  "
          f"not-surviving={_fmt(not_surviving)} → {verdict}")


if __name__ == "__main__":
    run()
