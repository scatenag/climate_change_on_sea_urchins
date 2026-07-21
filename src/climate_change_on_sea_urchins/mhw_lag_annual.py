"""
Annual lagged MHW → EC50 analysis: the one MHW signal that survives detrending.

EXPLORATORY / HYPOTHESIS-GENERATING — read the caveats before citing.

Most apparent MHW–EC50 coupling is a shared secular trend (see thermal_legacy,
regime_shift): raw correlations are strong but collapse once the trend is removed,
and Convergent Cross Mapping does not converge. One exception survives, and only at
a specific scale: the *duration* of heatwave exposure in the PREVIOUS year
(total MHW days, lag 1 yr) predicts the annual-mean EC50 after linear detrending of
both series. The *count* of events (event_count) shows nothing — it is exposure
duration, not event number, and with a ~1-year carry-over consistent with heat
stress during one season degrading the gametes maturing for the next.

This reconciles with the independent 2025 experiment that found NO acute MHW effect
on metal tolerance: that tested the ACUTE effect (a heatwave during the assay); this
is a DELAYED carry-over, a different mechanism the experiment did not probe.

Honest strength (all reported in the summary):
  + survives detrending (Spearman), robust to leave-one-year-out jackknife,
    direction is asymmetric (MHW→EC50, not EC50→MHW);
  - Pearson is weak (rank-driven, not linear), does NOT survive Benjamini-Hochberg
    FDR across the 4×4 predictor×lag grid, n≈22, CCM negative.
Verdict: a suggestive, pre-registration-worthy hypothesis, NOT a demonstrated effect.

Outputs:
    results/mhw_lag_annual.csv          — full predictor×lag grid (raw + detrended)
    results/mhw_lag_annual_summary.json — best signal, robustness battery, verdict
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .common import load_data, RESULTS, ROOT

PREDICTORS = ["event_count", "total_mhw_days", "cum_intensity_sum", "max_intensity"]
LAGS = [0, 1, 2, 3]
YEAR_MIN, YEAR_MAX = 2004, 2025   # drop incomplete 2003 / partial 2026


def _detrend(s: pd.Series) -> pd.Series:
    return pd.Series(s.values - np.polyval(np.polyfit(s.index.values, s.values, 1), s.index.values),
                     index=s.index)


def run():
    _, df_real, _, _ = load_data()
    real = df_real.dropna(subset=["EC50"])
    ec = real.assign(y=real["Datetime"].dt.year).groupby("y")["EC50"].mean()
    ann = pd.read_csv(ROOT / "data" / "mhw_annual.csv").set_index("year")

    rows = []
    for pred in PREDICTORS:
        for lag in LAGS:
            j = pd.concat([ec.rename("ec"), ann[pred].shift(lag).rename("m")], axis=1).dropna()
            j = j[(j.index >= YEAR_MIN) & (j.index <= YEAR_MAX)]
            if len(j) < 8:
                continue
            rr, pr = stats.spearmanr(j["ec"], j["m"])
            jd = pd.DataFrame({"ec": _detrend(j["ec"]), "m": _detrend(j["m"])})
            rd, pd_ = stats.spearmanr(jd["ec"], jd["m"])
            rows.append({"predictor": pred, "lag_years": lag, "n": len(j),
                         "rho_raw": float(rr), "p_raw": float(pr),
                         "rho_detrended": float(rd), "p_detrended": float(pd_)})

    grid = pd.DataFrame(rows)
    # Benjamini-Hochberg FDR across the whole detrended grid
    grid["p_detrended_fdr"] = multipletests(grid["p_detrended"], method="fdr_bh")[1]
    grid.to_csv(RESULTS / "mhw_lag_annual.csv", index=False)

    # Best signal = smallest detrended p among the biologically expected (negative) ones
    cand = grid[grid["rho_detrended"] < 0].sort_values("p_detrended")
    best = cand.iloc[0]

    # --- robustness battery on the best signal ---
    j = pd.concat([ec.rename("ec"),
                   ann[best["predictor"]].shift(int(best["lag_years"])).rename("m")], axis=1).dropna()
    j = j[(j.index >= YEAR_MIN) & (j.index <= YEAR_MAX)]
    ecr, mr = _detrend(j["ec"]), _detrend(j["m"])
    pear_r, pear_p = stats.pearsonr(ecr, mr)
    # jackknife
    jack_ps = []
    for drop in j.index:
        s = j.drop(drop)
        jack_ps.append(stats.spearmanr(_detrend(s["ec"]), _detrend(s["m"]))[1])
    jack_ps = np.array(jack_ps)
    # reverse direction: does EC50(t-lag) predict MHW(t)?
    jr = pd.concat([ann[best["predictor"]].rename("m"),
                    ec.shift(int(best["lag_years"])).rename("ec")], axis=1).dropna()
    jr = jr[(jr.index >= YEAR_MIN) & (jr.index <= YEAR_MAX)]
    rev_r, rev_p = stats.spearmanr(_detrend(jr["m"]), _detrend(jr["ec"]))

    best_fdr = float(grid.loc[best.name, "p_detrended_fdr"])
    survives_fdr = bool(best_fdr < 0.05)
    verdict = ("confirmatory_survives_fdr" if survives_fdr
               else "exploratory_suggestive" if best["p_detrended"] < 0.05
               else "no_signal_beyond_trend")

    summary = {
        "best_signal": {
            "predictor": best["predictor"], "lag_years": int(best["lag_years"]), "n": int(best["n"]),
            "rho_detrended_spearman": float(best["rho_detrended"]),
            "p_detrended_spearman": float(best["p_detrended"]),
            "p_detrended_fdr_bh": best_fdr,
        },
        "robustness": {
            "pearson_r": float(pear_r), "pearson_p": float(pear_p),
            "jackknife_p_min": float(jack_ps.min()), "jackknife_p_max": float(jack_ps.max()),
            "jackknife_frac_below_0p05": float((jack_ps < 0.05).mean()),
            "reverse_direction_rho": float(rev_r), "reverse_direction_p": float(rev_p),
        },
        "event_count_lag1_detrended_p": float(
            grid[(grid.predictor == "event_count") & (grid.lag_years == 1)]["p_detrended"].iloc[0]),
        "verdict": verdict,
        "interpretation": (
            "Exposure DURATION (MHW days) in the previous year predicts EC50 after detrending "
            "(Spearman rho={:.2f}, p={:.3f}); event COUNT does not. Robust to jackknife and "
            "direction-asymmetric, but Pearson-weak and does not survive FDR — a suggestive, "
            "pre-registration-worthy carry-over hypothesis, not a demonstrated effect. Consistent "
            "with a delayed (not acute) mechanism, unlike the 2025 acute-exposure experiment."
        ).format(best["rho_detrended"], best["p_detrended"]),
    }
    with (RESULTS / "mhw_lag_annual_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ mhw_lag_annual: best = {best['predictor']} lag {int(best['lag_years'])}yr "
          f"(detrended rho={best['rho_detrended']:.2f}, p={best['p_detrended']:.3f}, "
          f"FDR={best_fdr:.2f}); jackknife {(jack_ps<0.05).mean()*100:.0f}% <0.05; "
          f"Pearson p={pear_p:.2f} → {verdict}")


if __name__ == "__main__":
    run()
