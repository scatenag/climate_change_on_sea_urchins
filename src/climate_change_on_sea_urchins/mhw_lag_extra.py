"""
Python port of the SEA and mixed-effects corroborating analyses previously
computed only in scripts/mhw_lag_analysis.R (Superposed Epoch Analysis,
event-based mixed-effects model). DLNM stays in R (scripts/mhw_lag_analysis.R)
since there is no comparably mature Python cross-basis implementation; that
script now only computes DLNM and is triggered automatically by the same
GitHub Action that refreshes the data (see .github/workflows/).

Two deliberate differences from the original R script, both bringing this
module in line with the rest of the Python pipeline and the paper's own
stated methodology (only real bioassays used for hypothesis tests):
  1. Uses only real (non-imputed) EC50 measurements, not the rolling-mean
     gap-filled series in data_extended.csv — matches mhw_analysis.py's
     convention and avoids the imputed series' artificial smoothness
     inflating apparent temporal structure.
  2. The mixed-effects model's season/time covariates are computed from the
     actual EC50 observation date, not the triggering event's end date (the
     original R script held both fixed at the event's end date across all 12
     lag_post_end rows, which leaks the secular EC50 trend into the
     lag_post_end coefficient for events whose 12-month follow-up window
     crosses into the next calendar year — found and confirmed during the
     2026-07 site-coordinate bug investigation).

Outputs (results/, same filenames the dashboard already reads):
    results/sea_results.csv
    results/mixed_effects_predictions.csv
    results/mixed_effects_summary.json
"""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from .common import load_data, RESULTS

LAG_MIN, LAG_MAX = -6, 12
N_BOOT = 999
RNG_SEED = 42


# ── A. Superposed Epoch Analysis ────────────────────────────────────────────

def _nearest_ec50(dates: pd.DatetimeIndex, obs_dates: pd.Series, obs_ec50: pd.Series) -> np.ndarray:
    obs_dates = obs_dates.values
    out = np.full(len(dates), np.nan)
    for i, d in enumerate(dates):
        diffs = np.abs((obs_dates - np.datetime64(d)) / np.timedelta64(1, "D"))
        idx = np.where(diffs < 20)[0]
        if len(idx):
            out[i] = obs_ec50.values[idx[0]]
    return out


def run_sea(df_real: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    lags = np.arange(LAG_MIN, LAG_MAX + 1)
    obs_dates, obs_ec50 = df_real["Datetime"], df_real["EC50"]

    rows = []
    for _, ev in events.iterrows():
        dates = [ev["peak_date"] + pd.DateOffset(months=int(l)) for l in lags]
        vals = _nearest_ec50(pd.DatetimeIndex(dates), obs_dates, obs_ec50)
        for lag, v in zip(lags, vals):
            rows.append((lag, v))
    epoch_df = pd.DataFrame(rows, columns=["lag", "EC50"])

    composite = (
        epoch_df.groupby("lag")["EC50"]
        .agg(n="count", mean_ec50="mean", sd_EC50="std")
        .reset_index()
    )
    composite["se_EC50"]  = composite["sd_EC50"] / np.sqrt(composite["n"])
    composite["ci_lower"] = composite["mean_ec50"] - 1.96 * composite["se_EC50"]
    composite["ci_upper"] = composite["mean_ec50"] + 1.96 * composite["se_EC50"]

    rng = np.random.default_rng(RNG_SEED)
    n_events = len(events)
    boot_means = np.full((N_BOOT, len(lags)), np.nan)
    for b in range(N_BOOT):
        rand_peaks = rng.choice(obs_dates.values, size=n_events, replace=False)
        for j, lag in enumerate(lags):
            dates = pd.DatetimeIndex(rand_peaks) + pd.DateOffset(months=int(lag))
            vals = _nearest_ec50(dates, obs_dates, obs_ec50)
            boot_means[b, j] = np.nanmean(vals)

    boot_p = []
    for j in range(len(lags)):
        obs = composite["mean_ec50"].iloc[j]
        null = boot_means[:, j]
        null = null[~np.isnan(null)]
        p = 2 * min((null <= obs).mean(), (null >= obs).mean())
        boot_p.append(p)
    composite["boot_p"]     = boot_p
    composite["boot_p025"]  = np.nanquantile(boot_means, 0.025, axis=0)
    composite["boot_p975"]  = np.nanquantile(boot_means, 0.975, axis=0)
    composite["significant"] = composite["boot_p"] < 0.05
    return composite


# ── B. Event-based mixed-effects model ──────────────────────────────────────

def _season_of(month: int) -> str:
    return ("Winter" if month in (12, 1, 2) else
            "Spring" if month in (3, 4, 5) else
            "Summer" if month in (6, 7, 8) else "Autumn")


def build_post_event_df(df_real: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    obs_dates, obs_ec50 = df_real["Datetime"], df_real["EC50"]
    t0 = df_real["Datetime"].min()

    rows = []
    for _, ev in events.iterrows():
        end_date = ev["end_date"]
        lags = np.arange(1, 13)
        dates = pd.DatetimeIndex([end_date + pd.DateOffset(months=int(l)) for l in lags])
        vals = _nearest_ec50(dates, obs_dates, obs_ec50)
        for lag, d, v in zip(lags, dates, vals):
            if np.isnan(v):
                continue
            rows.append(dict(
                event_id=ev["event_id"], lag_post_end=int(lag), EC50=v,
                intensity_max=ev["intensity_max"], duration_days=ev["duration_days"],
                season=_season_of(d.month),
                time_index=(d - t0).days / 365.25,
            ))
    return pd.DataFrame(rows)


def run_mixed_effects(df_real: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    post_event_df = build_post_event_df(df_real, events)
    print(f"  Post-event observations (real EC50 only): {len(post_event_df)}")

    model = smf.mixedlm(
        "EC50 ~ lag_post_end * intensity_max + duration_days + C(season) + time_index",
        data=post_event_df, groups=post_event_df["event_id"],
    )
    fit = model.fit(reml=True)
    print(fit.summary())

    summary = {
        "n_obs": int(len(post_event_df)),
        "n_events": int(post_event_df["event_id"].nunique()),
        "params": {k: float(v) for k, v in fit.params.items()},
        "pvalues": {k: float(v) for k, v in fit.pvalues.items()},
        "tvalues": {k: float(v) for k, v in fit.tvalues.items()},
    }

    intensity_grid = [0.5, 1.0, 1.5, 2.0]
    lag_grid = np.arange(1, 13)
    med_duration = float(events["duration_days"].median())
    pred_rows = []
    for inten in intensity_grid:
        for lag in lag_grid:
            x = dict(
                Intercept=1.0, lag_post_end=lag, intensity_max=inten,
                duration_days=med_duration, time_index=post_event_df["time_index"].median(),
            )
            base = fit.params.get("Intercept", 0.0)
            pred = (base
                    + fit.params.get("lag_post_end", 0.0) * lag
                    + fit.params.get("intensity_max", 0.0) * inten
                    + fit.params.get("lag_post_end:intensity_max", 0.0) * lag * inten
                    + fit.params.get("duration_days", 0.0) * med_duration
                    + fit.params.get("time_index", 0.0) * x["time_index"]
                    + fit.params.get("C(season)[T.Summer]", 0.0))  # reference summer for the predicted curve
            pred_rows.append(dict(lag_post_end=lag, intensity_max=inten, EC50_pred=pred))
    predictions = pd.DataFrame(pred_rows)
    return predictions, summary


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    df_full, df_real, events, monthly = load_data()

    print("── SEA (Python port, real EC50 only) ──────────────────────────────")
    sea = run_sea(df_real, events)
    sea.to_csv(RESULTS / "sea_results.csv", index=False)
    n_sig = int(sea["significant"].sum())
    print(f"✓ SEA: {n_sig}/{len(sea)} lags nominally significant (bootstrap p<0.05) — "
          f"{'diffuse across most lags (trend-confounding signature)' if n_sig > len(sea) * 0.7 else 'localized'}")

    print("\n── Mixed-effects model (Python port, corrected time-indexing) ─────")
    predictions, summary = run_mixed_effects(df_real, events)
    predictions.to_csv(RESULTS / "mixed_effects_predictions.csv", index=False)
    (RESULTS / "mixed_effects_summary.json").write_text(json.dumps(summary, indent=2))
    p_lag = summary["pvalues"].get("lag_post_end", float("nan"))
    print(f"✓ mixed-effects: lag_post_end p={p_lag:.4f} "
          f"({'nominally significant' if p_lag < 0.05 else 'not significant'} at alpha=0.05)")


if __name__ == "__main__":
    run()
