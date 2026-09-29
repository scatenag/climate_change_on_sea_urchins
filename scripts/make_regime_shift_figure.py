"""
Figure of the regime-shift results (climate_change_on_sea_urchins.regime_shift):
  (a) annual MHW days and the annual mean of the real EC50 measurements, each
      with its Pettitt break (the last period before the change), with the
      breaks' p-values in the subtitle;
  (b) PC1 of the deseasonalised T/S/CO2/O2/pH anomalies, annual mean.
Titles describe what is plotted; no conclusion is written into the figure.
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
ec50_break_month = summ["ec50_regime_shift"]["break"][:7]
ec50_break = int(ec50_break_month[:4])
ec50_p = summ["ec50_regime_shift"]["p"]
mhw_p = float(cp.loc[cp["series"] == "MHW_total_mhw_days", "p_value"].iloc[0])

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
         label="EC50, annual mean of real measurements")
axb.axvline(ec50_break, color=C_EC, ls="--", lw=1.5)
axb.set_ylabel("Copper EC50 (µg L⁻¹)", color=C_EC)
axb.tick_params(axis="y", labelcolor=C_EC)

# Titles describe what is plotted, never a conclusion; break dates and p go
# in a subtitle computed from the precomputed results.
ax1.text(mhw_break, 0.97, f" Pettitt break {mhw_break}", transform=ax1.get_xaxis_transform(),
         color=C_MHW, fontsize=8, va="top", ha="right")
axb.text(ec50_break, 0.05, f" Pettitt break {ec50_break_month}", transform=axb.get_xaxis_transform(),
         color=C_EC, fontsize=8, va="bottom", ha="left")
ax1.set_title("(a) Annual MHW days and annual mean EC50, with their Pettitt breaks\n"
              f"break = last period before the change · MHW days {mhw_break} (p={mhw_p:.2g}) · "
              f"EC50 {ec50_break_month} (p={ec50_p:.1e})",
              fontsize=9, loc="left")

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
ax2.set_ylabel("PC1 (a.u.)")
ax2.set_xlabel("Year")
ax2.set_title(f"(b) PC1 of the deseasonalised T/S/CO₂/O₂/pH anomalies, annual mean "
              f"({ve:.0f}% of their variance)",
              fontsize=9, loc="left")

fig.tight_layout()
for out in [ROOT / "figures" / "fig_regime_shift.png",
            ROOT / "drafts" / "nuova pubblicazione" / "fig_regime_shift.png"]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ wrote {out.relative_to(ROOT)}")
