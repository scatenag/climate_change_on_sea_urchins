"""
Figure: why the strong EC50–thermal-dose correlation does NOT prove chronic-heat
causation. Reads results/thermal_legacy.csv (from
climate_change_on_sea_urchins.thermal_legacy) and renders two panels:

  (a) EC50 vs cumulative thermal dose (18 C, 36-month window) — raw: a strong,
      tidy negative relationship.
  (b) the same two variables after removing each one's linear time trend — the
      relationship collapses, showing the raw signal is the shared secular ramp,
      not thermal dose tracking EC50's off-trend variation.

Run:  .venv/bin/python3 scripts/make_thermal_legacy_figure.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(ROOT / "results" / "thermal_legacy.csv", parse_dates=["Datetime"])

DOSE = "dose_18C_36m"
C_RAW, C_DET = "#b8500f", "#37618e"
t = (d["Datetime"] - d["Datetime"].min()).dt.days.values.astype(float)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

# (a) raw
x, y = d[DOSE].values, d["EC50"].values
r_raw = stats.spearmanr(x, y)[0]
ax1.scatter(x, y, s=20, color=C_RAW, alpha=0.6, edgecolor="none")
m, b = np.polyfit(x, y, 1)
xs = np.array([x.min(), x.max()])
ax1.plot(xs, m * xs + b, color=C_RAW, lw=2)
ax1.set_xlabel("Cumulative thermal dose\n(degree-days >18 °C, 36-month window)")
ax1.set_ylabel("Copper EC50 (µg L⁻¹)")
ax1.set_title(f"(a) Raw: strong correlation (ρ = {r_raw:.2f})", fontsize=10, loc="left")
ax1.spines[["top", "right"]].set_visible(False)

# (b) detrended
rx = x - np.polyval(np.polyfit(t, x, 1), t)
ry = y - np.polyval(np.polyfit(t, y, 1), t)
r_det, p_det = stats.spearmanr(rx, ry)
ax2.axhline(0, color="grey", lw=0.6)
ax2.axvline(0, color="grey", lw=0.6)
ax2.scatter(rx, ry, s=20, color=C_DET, alpha=0.6, edgecolor="none")
ax2.set_xlabel("Thermal dose — residual\n(linear time trend removed)")
ax2.set_ylabel("EC50 — residual")
ax2.set_title(f"(b) Detrended: signal collapses (ρ = {r_det:.2f}, p = {p_det:.2f})",
              fontsize=10, loc="left")
ax2.spines[["top", "right"]].set_visible(False)

fig.suptitle("The chronic-heat hypothesis is consistent with the data but not "
             "separable from the shared trend", fontsize=11, y=1.02)
fig.tight_layout()
for out in [ROOT / "figures" / "fig_thermal_legacy.png",
            ROOT / "drafts" / "nuova pubblicazione" / "fig_thermal_legacy.png"]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ wrote {out.relative_to(ROOT)}")
