"""
Figure: does chronic heat dose above 24 C predict EC50 beyond the shared trend?
Reads results/thermal_legacy.csv and results/thermal_legacy_summary.json (from
climate_change_on_sea_urchins.thermal_legacy) and renders two panels:

  (a) detrended Spearman rho for each of the 5 windows (12/24/36/48/60 months),
      colored by cross-validation status: green = robust (survives Bonferroni
      on BOTH the rank-based and the parametric OLS partial test), amber =
      suggestive (rank test only, not corroborated), grey = neither.
  (b) EC50 vs cumulative thermal dose for the strongest robust window (falling
      back to the best suggestive, then the best raw, window if none is
      robust), after removing each series' own linear time trend.

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
robust = set(summary["windows_robust"])
suggestive = set(summary["windows_suggestive_rank_only"])

C_ROBUST, C_SUGGESTIVE, C_FAIL, C_DET = "#2a7a3b", "#e0a11e", "#9a9a9a", "#37618e"
t = (d["Datetime"] - d["Datetime"].min()).dt.days.values.astype(float)


def _color(w):
    if w in robust:
        return C_ROBUST
    if w in suggestive:
        return C_SUGGESTIVE
    return C_FAIL


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

# (a) detrended rho by window, colored by cross-validation status
colors = [_color(w) for w in per_window["window_months"]]
ax1.bar([str(int(w)) for w in per_window["window_months"]], per_window["detrended_spearman_r"],
        color=colors)
def _fmt_p(p):
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"

for x, row in enumerate(per_window.itertuples()):
    label = f"rank p={_fmt_p(row.p_bonferroni)}\nOLS p={_fmt_p(row.partial_p_bonferroni)}"
    ax1.text(x, row.detrended_spearman_r - (0.03 if row.detrended_spearman_r < 0 else -0.03),
              label, ha="center", va="top" if row.detrended_spearman_r < 0 else "bottom", fontsize=7.5)
ax1.axhline(0, color="black", lw=0.8)
ymin = per_window["detrended_spearman_r"].min()
ax1.set_ylim(top=max(0.02, per_window["detrended_spearman_r"].max() + 0.05),
             bottom=ymin - 0.09)
ax1.set_xlabel("Cumulative window (months)")
ax1.set_ylabel("Detrended Spearman ρ (dose vs EC50)")
ax1.set_title("(a) Green = robust (rank + OLS)  ·  Amber = rank-only  ·  Grey = neither",
              fontsize=9.5, loc="left")
ax1.spines[["top", "right"]].set_visible(False)

# (b) scatter for the strongest robust window (fall back to suggestive, then
# overall best detrended p, if nothing is robust)
if robust:
    pool = per_window[per_window["window_months"].isin(robust)]
elif suggestive:
    pool = per_window[per_window["window_months"].isin(suggestive)]
else:
    pool = per_window
best_win = int(pool.loc[pool["detrended_p"].idxmin(), "window_months"])
best_status = "robust" if best_win in robust else ("rank-only, not cross-validated" if best_win in suggestive else "not surviving")

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
ax2.set_title(f"(b) {best_win}-month window, detrended (ρ = {r_det:.2f}, p = {p_det:.4f}) — {best_status}",
              fontsize=9.5, loc="left")
ax2.spines[["top", "right"]].set_visible(False)

not_surviving = summary["windows_not_surviving"]
if robust:
    suptitle = (f"Chronic thermal dose above {thr}°C robustly predicts EC50 only at "
                f"{'/'.join(str(w) for w in sorted(robust))}-month timescale(s)")
elif suggestive:
    suptitle = (f"Chronic thermal dose above {thr}°C shows a rank-only, not "
                f"cross-validated signal at {'/'.join(str(w) for w in sorted(suggestive))} months")
else:
    suptitle = f"Chronic thermal dose above {thr}°C is not separable from the shared trend"
fig.suptitle(suptitle, fontsize=11, y=1.02)
fig.tight_layout()
for out in [ROOT / "figures" / "fig_thermal_legacy.png",
            ROOT / "drafts" / "nuova pubblicazione" / "fig_thermal_legacy.png"]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ wrote {out.relative_to(ROOT)}")
