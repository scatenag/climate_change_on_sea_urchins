"""
Figure: the one MHW→EC50 signal that survives detrending — exposure DURATION,
lagged one year. Reads results/mhw_lag_annual.csv + mhw_lag_annual_summary.json
(from climate_change_on_sea_urchins.mhw_lag_annual).

  (a) detrended-correlation grid (predictor × lag): only total MHW days at lag 1 yr
      stands out; event count shows nothing → it's duration, not number.
  (b) the surviving signal: detrended previous-year MHW days vs detrended EC50.

Run:  .venv/bin/python3 scripts/make_mhw_lag_annual_figure.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
grid = pd.read_csv(ROOT / "results" / "mhw_lag_annual.csv")
summ = json.load((ROOT / "results" / "mhw_lag_annual_summary.json").open())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.25, 1]})

# --- (a) detrended rho grid ---
preds = ["event_count", "total_mhw_days", "cum_intensity_sum", "max_intensity"]
lags = [0, 1, 2, 3]
M = grid.pivot(index="predictor", columns="lag_years", values="rho_detrended").loc[preds, lags]
P = grid.pivot(index="predictor", columns="lag_years", values="p_detrended").loc[preds, lags]
im = ax1.imshow(M.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
ax1.set_xticks(range(len(lags)), lags)
ax1.set_yticks(range(len(preds)), ["MHW event count", "MHW days (duration)",
                                    "cumulative intensity", "max intensity"])
ax1.set_xlabel("Lag (years): MHW at t−lag → EC50 at t")
for i in range(len(preds)):
    for j in range(len(lags)):
        star = " *" if P.values[i, j] < 0.05 else ""
        ax1.text(j, i, f"{M.values[i,j]:+.2f}{star}", ha="center", va="center",
                 fontsize=8, color="black")
ax1.set_title("(a) Detrended correlation — only duration, lag 1 yr, survives",
              fontsize=9.5, loc="left")
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Spearman ρ (detrended)")

# --- (b) the surviving signal scatter ---
df = pd.read_csv(ROOT / "data" / "data_extended.csv", parse_dates=["Datetime"])
ci = pd.read_csv(ROOT / "data" / "data_ec50_ci.csv", parse_dates=["Datetime"])
df = df.merge(ci[["Datetime", "EC50_imputed"]], on="Datetime")
real = df[df.EC50_imputed == False].dropna(subset=["EC50"])
ec = real.assign(y=real.Datetime.dt.year).groupby("y")["EC50"].mean()
ann = pd.read_csv(ROOT / "data" / "mhw_annual.csv").set_index("year")
j = pd.concat([ec.rename("ec"), ann["total_mhw_days"].shift(1).rename("m")], axis=1).dropna()
j = j[(j.index >= 2004) & (j.index <= 2025)]
det = lambda s: s - np.polyval(np.polyfit(s.index.values, s.values, 1), s.index.values)
xr, yr = det(j["m"]), det(j["ec"])
b = summ["best_signal"]
ax2.scatter(xr, yr, s=32, color="#b8500f", alpha=0.75, edgecolor="none")
m, q = np.polyfit(xr, yr, 1)
xs = np.array([xr.min(), xr.max()])
ax2.plot(xs, m * xs + q, color="#b8500f", lw=1.8)
ax2.axhline(0, color="grey", lw=0.5); ax2.axvline(0, color="grey", lw=0.5)
ax2.set_xlabel("Previous-year MHW days — residual")
ax2.set_ylabel("EC50 — residual")
ax2.set_title(f"(b) The surviving signal (ρ={b['rho_detrended_spearman']:.2f}, "
              f"p={b['p_detrended_spearman']:.3f})", fontsize=9.5, loc="left")
ax2.spines[["top", "right"]].set_visible(False)

fig.suptitle("Exposure duration (not event count) lagged one year — exploratory, "
             "does not survive multiple-testing correction", fontsize=10.5, y=1.02)
fig.tight_layout()
for out in [ROOT / "figures" / "fig_mhw_lag_annual.png",
            ROOT / "drafts" / "nuova pubblicazione" / "fig_mhw_lag_annual.png"]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ wrote {out.relative_to(ROOT)}")
