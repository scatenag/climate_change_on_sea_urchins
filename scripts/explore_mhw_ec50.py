"""
Exploratory analysis: Marine Heatwaves → EC50 temporal lag

Key concern: EC50 is measured only ~50% of months; the rest are rolling-mean
imputations from the original notebook. We NEVER use imputed values for
statistical tests. Real measurements only for CCF, scatter, regression.
The full (imputed) series is used only for visualization context.

Outputs:
    explore_mhw_ec50.pdf  — 4-panel figure
    explore_ccf.csv       — CCF values at each lag
    explore_lag_scatter.csv — scatter data at each lag (real EC50 only)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.tsa.stattools import ccf as sm_ccf
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Load data ─────────────────────────────────────────────────────────────────
data    = pd.read_csv(ROOT / "data_extended.csv",  parse_dates=["Datetime"])
monthly = pd.read_csv(ROOT / "mhw_monthly.csv",    parse_dates=["Datetime"])
ci_df   = pd.read_csv(ROOT / "data_ec50_ci.csv",   parse_dates=["Datetime"])
events  = pd.read_csv(ROOT / "mhw_events.csv",     parse_dates=["start_date","end_date","peak_date"])

# Merge MHW metrics into main dataframe
df = data.merge(monthly[["Datetime","mhw_days","mhw_peak_intensity","mhw_cum_intensity"]],
                on="Datetime", how="left")
df["mhw_days"]           = df["mhw_days"].fillna(0)
df["mhw_peak_intensity"] = df["mhw_peak_intensity"].fillna(0)

# Attach imputation flag from CI file
df = df.merge(ci_df[["Datetime","EC50_imputed"]], on="Datetime", how="left")
df["EC50_imputed"] = df["EC50_imputed"].fillna(True)

# Two views:
#   df_real  — only months with an actual bioassay measurement
#   df_full  — all months (including rolling-mean imputations, for viz only)
df_full = df.copy()
df_real = df[~df["EC50_imputed"]].copy().reset_index(drop=True)

print(f"Total months:          {len(df_full)}")
print(f"Real EC50 measurements:{len(df_real)}")
print(f"Imputed months:        {df_full['EC50_imputed'].sum()}")
print(f"MHW months (any day):  {(df_full['mhw_days'] > 0).sum()}")

# ── 1. Cross-Correlation Function (CCF) ───────────────────────────────────────
# Problem: CCF needs a regular time series. With sparse real EC50 we have two
# options:
#   A) Use the full series (incl. imputed) — maximises N but inflates signal
#   B) Use the real series only, interpolated onto monthly grid — honest
# We do BOTH and show both; if (A) and (B) agree, the signal is real.

MAX_LAG = 12

# Option A: full series
mhw_full = df_full["mhw_peak_intensity"].values
ec50_full = df_full["EC50"].values
# Remove NaN pairs
mask_a = ~(np.isnan(mhw_full) | np.isnan(ec50_full))
ccf_a  = sm_ccf(mhw_full[mask_a], ec50_full[mask_a], nlags=MAX_LAG, adjusted=False)

# Option B: real measurements only — Spearman at each explicit lag
# For lag k: pair MHW(t) with EC50(t+k) where EC50 at t+k is a real measurement
lag_results = []
for lag in range(0, MAX_LAG + 1):
    shifted = df_real.copy()
    # Find MHW values k months before each real EC50 measurement
    mhw_lagged = []
    ec50_vals  = []
    for _, row in df_real.iterrows():
        target_date = row["Datetime"] - pd.DateOffset(months=lag)
        # Match to nearest month in df_full
        diffs = (df_full["Datetime"] - target_date).abs()
        idx = diffs.idxmin()
        if diffs[idx] <= pd.Timedelta("20 days"):
            mhw_lagged.append(df_full.loc[idx, "mhw_peak_intensity"])
            ec50_vals.append(row["EC50"])

    mhw_arr  = np.array(mhw_lagged)
    ec50_arr = np.array(ec50_vals)
    n = len(mhw_arr)

    if n >= 10:
        r, p = stats.spearmanr(mhw_arr, ec50_arr)
    else:
        r, p = np.nan, np.nan

    # Also: limit to months AFTER a real MHW event (mhw_peak_intensity > 0)
    nonzero = mhw_arr > 0
    if nonzero.sum() >= 6:
        r_nz, p_nz = stats.spearmanr(mhw_arr[nonzero], ec50_arr[nonzero])
    else:
        r_nz, p_nz = np.nan, np.nan

    lag_results.append(dict(lag=lag, n=n, spearman_r=r, p_value=p,
                            spearman_r_nonzero=r_nz, p_nonzero=p_nz,
                            n_nonzero=int(nonzero.sum())))

lag_df = pd.DataFrame(lag_results)
lag_df.to_csv(ROOT / "explore_ccf.csv", index=False)

print("\nSpearman(MHW_intensity(t-k), EC50_real(t)):")
print(lag_df[["lag","n","spearman_r","p_value"]].to_string(index=False))

# ── 2. Figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1 (top, full width): EC50 time series + MHW shading
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(df_full["Datetime"], df_full["EC50"], color="steelblue", lw=1.2,
         label="EC50 (incl. imputed)", zorder=2)
ax0.scatter(df_real["Datetime"], df_real["EC50"], color="navy", s=18,
            zorder=3, label="EC50 (real measurement)")
for _, ev in events.iterrows():
    ax0.axvspan(ev["start_date"], ev["end_date"],
                alpha=0.15, color="tomato", lw=0)
ax0.set_ylabel("EC50 (mg/L)")
ax0.set_title("EC50 time series with Marine Heatwave events (red shading)", fontsize=11)
handles = [Patch(facecolor="steelblue", alpha=0.5, label="EC50 (incl. imputed)"),
           Patch(facecolor="navy", label="EC50 (real measurement)"),
           Patch(facecolor="tomato", alpha=0.4, label="MHW event")]
ax0.legend(handles=handles, fontsize=8, loc="upper right")
ax0.set_xlim(df_full["Datetime"].min(), df_full["Datetime"].max())

# Panel 2 (middle left): CCF full series vs real-only Spearman
ax1 = fig.add_subplot(gs[1, :2])
conf95 = 1.96 / np.sqrt(mask_a.sum())
lags_x = np.arange(0, MAX_LAG + 1)
lags_x = np.arange(0, MAX_LAG + 1)
colors_bar = ["tomato" if p < 0.05 else "steelblue" for p in lag_df["p_value"]]
ax1.bar(lags_x, lag_df["spearman_r"], color=colors_bar, alpha=0.85,
        label="Spearman r (real EC50 only)")
conf95 = 1.96 / np.sqrt(len(df_real))
ax1.axhline(0, color="k", lw=0.8)
ax1.axhline( conf95, color="grey", lw=1, ls="--", alpha=0.6, label="95% CI")
ax1.axhline(-conf95, color="grey", lw=1, ls="--", alpha=0.6)
for _, row in lag_df.iterrows():
    if row["p_value"] < 0.001:
        marker = "***"
    elif row["p_value"] < 0.01:
        marker = "**"
    elif row["p_value"] < 0.05:
        marker = "*"
    else:
        continue
    ypos = row["spearman_r"] - 0.04 if row["spearman_r"] < 0 else row["spearman_r"] + 0.01
    ax1.text(row["lag"], ypos, marker, ha="center", fontsize=8, color="darkred")
ax1.set_xlabel("Lag k (months): MHW(t−k) → EC50(t)")
ax1.set_ylabel("Spearman r")
ax1.set_title("MHW intensity → EC50 lag correlation\n(red = significant p<0.05; real measurements only)", fontsize=10)
ax1.legend(fontsize=8)
ax1.set_xticks(lags_x)

# Panel 3 (middle right): p-value profile
ax2 = fig.add_subplot(gs[1, 2])
ax2.semilogy(lag_df["lag"], lag_df["p_value"].clip(1e-4), "o-", color="navy",
             label="all pairs", ms=5)
ax2.semilogy(lag_df["lag"], lag_df["p_nonzero"].clip(1e-4), "s--", color="tomato",
             label="MHW months only", ms=5)
ax2.axhline(0.05, color="k", lw=1, ls="--", label="p=0.05")
ax2.set_xlabel("Lag (months)")
ax2.set_ylabel("p-value (log scale)")
ax2.set_title("Significance profile", fontsize=10)
ax2.legend(fontsize=8)
ax2.set_xticks(lags_x)

# Panels 4–6 (bottom): scatter plots at the 3 most interesting lags
best_lags = lag_df.dropna(subset=["spearman_r"]).nsmallest(3, "p_value")["lag"].tolist()
if len(best_lags) < 3:
    best_lags = [0, 3, 6]

scatter_rows = []
for col_idx, lag in enumerate(best_lags[:3]):
    ax = fig.add_subplot(gs[2, col_idx])
    mhw_lagged, ec50_vals, months_list = [], [], []
    for _, row in df_real.iterrows():
        target_date = row["Datetime"] - pd.DateOffset(months=lag)
        diffs = (df_full["Datetime"] - target_date).abs()
        idx   = diffs.idxmin()
        if diffs[idx] <= pd.Timedelta("20 days"):
            mhw_lagged.append(df_full.loc[idx, "mhw_peak_intensity"])
            ec50_vals.append(row["EC50"])
            months_list.append(row["Datetime"])
            scatter_rows.append(dict(lag=lag, Datetime=row["Datetime"],
                                     mhw_intensity=df_full.loc[idx, "mhw_peak_intensity"],
                                     EC50=row["EC50"]))

    x, y = np.array(mhw_lagged), np.array(ec50_vals)
    # colour by MHW active (>0) vs inactive
    colors = np.where(x > 0, "tomato", "steelblue")
    ax.scatter(x, y, c=colors, s=20, alpha=0.7, edgecolors="none")

    # Regression line
    if len(x) >= 5:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, intercept + slope*xr, "k--", lw=1.2)
        sig = "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.set_title(f"Lag {lag}m: r={r:.2f}, p={p:.3f} {sig}", fontsize=9)
    else:
        ax.set_title(f"Lag {lag}m (n={len(x)})", fontsize=9)

    ax.set_xlabel("MHW peak intensity (°C above threshold)")
    ax.set_ylabel("EC50 real (mg/L)")

scatter_df = pd.DataFrame(scatter_rows)
scatter_df.to_csv(ROOT / "explore_lag_scatter.csv", index=False)

fig.suptitle("MHW → EC50 Exploratory Analysis\n"
             f"(real EC50 measurements n={len(df_real)}, MHW events n={len(events)})",
             fontsize=13, fontweight="bold")

out_pdf = ROOT / "explore_mhw_ec50.pdf"
fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
plt.close()
print(f"\nFigure saved: {out_pdf}")

# ── Summary ────────────────────────────────────────────────────────────────────
sig_lags = lag_df[lag_df["p_value"] < 0.05]
print("\n── Summary ──────────────────────────────────────────────────────────")
if sig_lags.empty:
    print("No significant lag found at p<0.05 (real EC50 measurements only).")
    print("Consider: weaker signal, insufficient EC50 coverage, or DLNM needed.")
else:
    print(f"Significant lags (p<0.05): {sig_lags['lag'].tolist()}")
    best = lag_df.dropna(subset=["spearman_r"]).loc[lag_df["p_value"].idxmin()]
    print(f"Best lag: {int(best['lag'])} months  r={best['spearman_r']:.3f}  p={best['p_value']:.4f}")
print()
