"""
Figure of the annual lagged MHW -> EC50 results (results/mhw_lag_annual.csv and
mhw_lag_annual_summary.json, from climate_change_on_sea_urchins.mhw_lag_annual):
  (a) detrended Spearman rho of annual mean EC50 on each annual MHW descriptor,
      lagged 0-3 years (* = detrended p < 0.05);
  (b) scatter of the detrended pair with the smallest detrended p among the
      negative associations (the module's best_signal), with rho, p and the
      BH-FDR-adjusted p in the subtitle.
Titles describe what is plotted; no conclusion is written into the figure.
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

from climate_change_on_sea_urchins.mhw_lag_annual import YEAR_MIN, YEAR_MAX

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
ax1.set_title("(a) Detrended Spearman ρ, annual EC50 vs MHW descriptor at each lag\n"
              "* = detrended p < 0.05 (uncorrected)", fontsize=9, loc="left")
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Spearman ρ (detrended)")

# --- (b) scatter of the best_signal pair ---
df = pd.read_csv(ROOT / "data" / "data_extended.csv", parse_dates=["Datetime"])
ci = pd.read_csv(ROOT / "data" / "data_ec50_ci.csv", parse_dates=["Datetime"])
df = df.merge(ci[["Datetime", "EC50_imputed"]], on="Datetime")
real = df[df.EC50_imputed == False].dropna(subset=["EC50"])
ec = real.assign(y=real.Datetime.dt.year).groupby("y")["EC50"].mean()
ann = pd.read_csv(ROOT / "data" / "mhw_annual.csv").set_index("year")
b = summ["best_signal"]
pred, lag = b["predictor"], int(b["lag_years"])
j = pd.concat([ec.rename("ec"), ann[pred].shift(lag).rename("m")], axis=1).dropna()
j = j[(j.index >= YEAR_MIN) & (j.index <= YEAR_MAX)]
det = lambda s: s - np.polyval(np.polyfit(s.index.values, s.values, 1), s.index.values)
xr, yr = det(j["m"]), det(j["ec"])
ax2.scatter(xr, yr, s=32, color="#b8500f", alpha=0.75, edgecolor="none")
m, q = np.polyfit(xr, yr, 1)
xs = np.array([xr.min(), xr.max()])
ax2.plot(xs, m * xs + q, color="#b8500f", lw=1.8)
ax2.axhline(0, color="grey", lw=0.5); ax2.axvline(0, color="grey", lw=0.5)
ax2.set_xlabel(f"{pred}, lagged {lag} yr — residual")
ax2.set_ylabel("EC50 — residual")
ax2.set_title(f"(b) {pred} lagged {lag} yr vs annual EC50, both detrended\n"
              f"ρ={b['rho_detrended_spearman']:.2f}, p={b['p_detrended_spearman']:.3f}, "
              f"BH-FDR p={b['p_detrended_fdr_bh']:.2f}, n={b['n']}", fontsize=9, loc="left")
ax2.spines[["top", "right"]].set_visible(False)

fig.suptitle("Annual MHW descriptors vs annual mean EC50, lagged 0–3 years (detrended)",
             fontsize=10.5, y=1.02)
fig.tight_layout()
for out in [ROOT / "figures" / "fig_mhw_lag_annual.png",
            ROOT / "drafts" / "nuova pubblicazione" / "fig_mhw_lag_annual.png"]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ wrote {out.relative_to(ROOT)}")
