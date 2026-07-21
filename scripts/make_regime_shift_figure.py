"""
Figure: the wild population changed state ~3 years after its environment did.

Reads results produced by climate_change_on_sea_urchins.regime_shift and renders:

  (a) MHW exposure (annual MHW days) with its ~2013 regime shift, overlaid on
      EC50 with its ~2016 regime shift — visualising the multi-year accumulation
      lag between environmental forcing and biological collapse.
  (b) the multifactorial environmental stress index (PC1 of deseasonalised
      T/S/CO2/O2/pH anomalies) rising through the same window.

Run:  .venv/bin/python3 scripts/make_regime_shift_figure.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

cp = pd.read_csv(RES / "regime_shift_changepoints.csv")
summ = json.load((RES / "regime_shift_summary.json").open())
ann = pd.read_csv(ROOT / "data" / "mhw_annual.csv")
stress = pd.read_csv(RES / "regime_shift_stress_index.csv", parse_dates=["Datetime"])

ec50 = pd.read_csv(ROOT / "data" / "data_extended.csv", parse_dates=["Datetime"])
ci = pd.read_csv(ROOT / "data" / "data_ec50_ci.csv", parse_dates=["Datetime"])
ec50 = ec50.merge(ci[["Datetime", "EC50_imputed"]], on="Datetime")
ec50 = ec50[ec50.EC50_imputed == False].dropna(subset=["EC50"])
ec50_year = ec50.assign(y=ec50.Datetime.dt.year).groupby("y")["EC50"].mean()

mhw_break = summ["mhw_exposure_break_year"]
ec50_break = int(summ["ec50_regime_shift"]["break"][:4])
lag = summ["exposure_precedes_response_years"]

C_MHW, C_EC, C_STR = "#c0392b", "#1f3b73", "#6b4c9a"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})

# --- (a) exposure vs response ---
ax1.bar(ann["year"], ann["total_mhw_days"], color=C_MHW, alpha=0.30,
        label="MHW exposure (days yr⁻¹)")
ax1.axvline(mhw_break, color=C_MHW, ls="--", lw=1.5)
ax1.set_ylabel("MHW days per year", color=C_MHW)
ax1.tick_params(axis="y", labelcolor=C_MHW)

axb = ax1.twinx()
axb.plot(ec50_year.index, ec50_year.values, "-o", color=C_EC, ms=4, lw=1.8,
         label="EC50 (copper tolerance)")
axb.axvline(ec50_break, color=C_EC, ls="--", lw=1.5)
axb.set_ylabel("Copper EC50 (µg L⁻¹)", color=C_EC)
axb.tick_params(axis="y", labelcolor=C_EC)

ymid = ec50_year.max() * 0.9
axb.annotate("", xy=(ec50_break, ymid), xytext=(mhw_break, ymid),
             arrowprops=dict(arrowstyle="->", color="black", lw=1.3))
axb.text((mhw_break + ec50_break) / 2, ymid * 1.02,
         f"~{lag}-yr accumulation", ha="center", va="bottom", fontsize=8.5)
ax1.text(mhw_break, 0.97, f" exposure shift {mhw_break}", transform=ax1.get_xaxis_transform(),
         color=C_MHW, fontsize=8, va="top", ha="right")
axb.text(ec50_break, 0.05, f" response shift {ec50_break}", transform=axb.get_xaxis_transform(),
         color=C_EC, fontsize=8, va="bottom", ha="left")
ax1.set_title("(a) Environmental exposure shifts ~3 years before the biological collapse",
              fontsize=10, loc="left")

# --- (b) multifactorial stress index ---
s = stress.dropna().sort_values("Datetime")
ann_stress = s.assign(y=s.Datetime.dt.year).groupby("y")["stress_pc1"].mean()
ax2.axhline(0, color="grey", lw=0.6)
ax2.fill_between(ann_stress.index, ann_stress.values, 0,
                 where=(ann_stress.values >= 0), color=C_STR, alpha=0.5)
ax2.fill_between(ann_stress.index, ann_stress.values, 0,
                 where=(ann_stress.values < 0), color=C_STR, alpha=0.2)
ax2.plot(ann_stress.index, ann_stress.values, color=C_STR, lw=1.8)
ve = summ["multifactorial_stress_index"]["pc1_variance_explained"] * 100
ax2.set_ylabel("Stress index\n(PC1, a.u.)")
ax2.set_xlabel("Year")
ax2.set_title(f"(b) Multifactorial climate-change stress index "
              f"(PC1 of T/S/CO₂/O₂/pH anomalies, {ve:.0f}% variance)",
              fontsize=10, loc="left")

fig.tight_layout()
for out in [ROOT / "figures" / "fig_regime_shift.png",
            ROOT / "drafts" / "nuova pubblicazione" / "fig_regime_shift.png"]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ wrote {out.relative_to(ROOT)}")
