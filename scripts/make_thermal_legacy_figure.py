"""
Figure: does chronic heat dose above 24 C predict EC50 beyond the shared trend?
Reads results/thermal_legacy.csv and results/thermal_legacy_summary.json (from
climate_change_on_sea_urchins.thermal_legacy) and renders two panels:

  (a) detrended Spearman rho for each of the 5 windows (12/24/36/48/60 months) —
      green bars survive Bonferroni correction across the 5 windows, grey bars
      do not.
  (b) EC50 vs cumulative thermal dose for the strongest surviving window, after
      removing each series' own linear time trend — the detrended relationship
      that panel (a) summarises.

Run:  .venv/bin/python3 scripts/make_thermal_legacy_figure.py
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
d = pd.read_csv(ROOT / "results" / "thermal_legacy.csv", parse_dates=["Datetime"])
summary = json.loads((ROOT / "results" / "thermal_legacy_summary.json").read_text())
thr = int(summary["threshold_C"])
per_window = pd.DataFrame(summary["per_window"]).sort_values("window_months")
survive_bonf = set(summary["windows_surviving_bonferroni"])

C_SURVIVE, C_FAIL, C_DET = "#2a7a3b", "#9a9a9a", "#37618e"
t = (d["Datetime"] - d["Datetime"].min()).dt.days.values.astype(float)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

# (a) detrended rho by window, colored by Bonferroni survival
colors = [C_SURVIVE if w in survive_bonf else C_FAIL for w in per_window["window_months"]]
ax1.bar([str(int(w)) for w in per_window["window_months"]], per_window["detrended_spearman_r"],
        color=colors)
for x, row in enumerate(per_window.itertuples()):
    label = f"p={row.p_bonferroni:.3f}" if row.p_bonferroni >= 0.001 else f"p={row.p_bonferroni:.1e}"
    ax1.text(x, row.detrended_spearman_r - (0.015 if row.detrended_spearman_r < 0 else -0.015),
              label, ha="center", va="top" if row.detrended_spearman_r < 0 else "bottom", fontsize=8)
ax1.axhline(0, color="black", lw=0.8)
ax1.set_xlabel("Cumulative window (months)")
ax1.set_ylabel("Detrended Spearman ρ (dose vs EC50)")
ax1.set_title(f"(a) Green = survives Bonferroni across {len(per_window)} windows",
              fontsize=10, loc="left")
ax1.spines[["top", "right"]].set_visible(False)

# (b) scatter for the strongest surviving window (fall back to the overall best
# detrended p if nothing survives Bonferroni)
if survive_bonf:
    best_win = int(per_window.loc[per_window["p_bonferroni"].idxmin(), "window_months"])
else:
    best_win = int(per_window.loc[per_window["detrended_p"].idxmin(), "window_months"])

dose_col = f"dose_{thr}C_{best_win}m"
x, y = d[dose_col].values, d["EC50"].values
rx = x - np.polyval(np.polyfit(t, x, 1), t)
ry = y - np.polyval(np.polyfit(t, y, 1), t)
r_det, p_det = stats.spearmanr(rx, ry)
ax2.axhline(0, color="grey", lw=0.6)
ax2.axvline(0, color="grey", lw=0.6)
ax2.scatter(rx, ry, s=20, color=C_DET, alpha=0.6, edgecolor="none")
m, b = np.polyfit(rx, ry, 1)
xs = np.array([rx.min(), rx.max()])
ax2.plot(xs, m * xs + b, color=C_DET, lw=2)
ax2.set_xlabel(f"Thermal dose ({thr}°C, {best_win}m) — residual\n(linear time trend removed)")
ax2.set_ylabel("EC50 — residual")
ax2.set_title(f"(b) {best_win}-month window, detrended (ρ = {r_det:.2f}, p = {p_det:.4f})",
              fontsize=10, loc="left")
ax2.spines[["top", "right"]].set_visible(False)

not_surviving = summary["windows_not_surviving"]
if not_surviving:
    span = f"{min(summary['windows_surviving_bonferroni'] + summary['windows_surviving_fdr_only'])}–{max(summary['windows_surviving_bonferroni'] + summary['windows_surviving_fdr_only'])}"
    long_span = f"{min(not_surviving)}–{max(not_surviving)}"
    suptitle = (f"Chronic thermal dose above {thr}°C predicts EC50 at {span}-month, "
                f"but not {long_span}-month, timescales")
else:
    suptitle = f"Chronic thermal dose above {thr}°C predicts EC50 across all tested windows"
fig.suptitle(suptitle, fontsize=11, y=1.02)
fig.tight_layout()
for out in [ROOT / "figures" / "fig_thermal_legacy.png",
            ROOT / "drafts" / "nuova pubblicazione" / "fig_thermal_legacy.png"]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ wrote {out.relative_to(ROOT)}")
