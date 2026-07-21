"""
Figure: is the 20-year decline in copper EC50 geochemistry or biology?

Reads results/cu_speciation_decomposition.csv + cu_speciation_summary.json
(produced by climate_change_on_sea_urchins.cu_speciation) and renders a
two-panel publication figure:

  (a) nominal EC50 vs speciation-corrected ("free-Cu2+-equivalent") EC50 over
      time — the two series are nearly indistinguishable, i.e. removing the
      ocean-acidification bioavailability effect barely changes the trend.
  (b) attribution of the pre/post-2016 decline into a small geochemical
      component and a large residual biological component.

Run:  .venv/bin/python3 scripts/make_speciation_figure.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

d = pd.read_csv(RESULTS / "cu_speciation_decomposition.csv", parse_dates=["Datetime"])
s = json.load((RESULTS / "cu_speciation_summary.json").open())

C_NOM = "#1f3b73"   # nominal
C_BIO = "#c0392b"   # speciation-corrected
C_GEO = "#5aa9e6"   # geochemical share

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.4, 1]}
)

# ---- (a) time series ----
ax1.scatter(d["Datetime"], d["EC50"], s=16, color=C_NOM, alpha=0.55,
            label="Nominal EC50 (total Cu)", zorder=3)
ax1.scatter(d["Datetime"], d["EC50_bio_lit"], s=16, color=C_BIO, alpha=0.55,
            marker="^", label="Speciation-corrected (free-Cu²⁺-equiv.)", zorder=3)

# linear fits for the eye
xt = (d["Datetime"] - d["Datetime"].min()).dt.days.values
for col, c in [("EC50", C_NOM), ("EC50_bio_lit", C_BIO)]:
    m, b = np.polyfit(xt, d[col].values, 1)
    ax1.plot(d["Datetime"], m * xt + b, color=c, lw=2, zorder=4)

ax1.axvline(pd.Timestamp("2016-01-01"), color="grey", ls="--", lw=1)
ax1.text(pd.Timestamp("2016-02-01"), ax1.get_ylim()[1] * 0.96, "2016",
         color="grey", fontsize=8, va="top")
ax1.set_ylabel("Copper EC50 (µg L⁻¹)")
ax1.set_xlabel("Year")
ax1.set_title("(a) Correcting for Cu speciation barely shifts the decline",
              fontsize=10, loc="left")
ax1.legend(frameon=False, fontsize=8, loc="upper right")
ax1.spines[["top", "right"]].set_visible(False)

# ---- (b) attribution ----
nominal = s["ec50_decline_nominal_pct"]
geo = s["geochemical_share_literature_pct"]          # % of the decline
geo_pts = nominal * geo / 100.0                       # in EC50-decline percentage points
bio_pts = nominal - geo_pts

ax2.bar(0, geo_pts, width=0.6, color=C_GEO, label="Geochemical (OA bioavailability)")
ax2.bar(0, bio_pts, bottom=geo_pts, width=0.6, color=C_BIO,
        label="Residual biological")
ax2.set_xlim(-0.8, 0.8)
ax2.set_xticks([])
ax2.set_ylabel("Decline in nominal EC50 (%)")
ax2.set_title("(b) Attribution of the\npre→post-2016 decline", fontsize=10, loc="left")
ax2.annotate(f"{geo_pts:.1f} pp\n(≈{geo:.0f}%)", (0, geo_pts / 2),
             ha="center", va="center", fontsize=8, color="#0b3d66")
ax2.annotate(f"{bio_pts:.1f} pp\n(≈{100 - geo:.0f}%)", (0, geo_pts + bio_pts / 2),
             ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax2.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.32))
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
for out in [ROOT / "figures" / "fig_cu_speciation_decomposition.png",
            ROOT / "drafts" / "nuova pubblicazione" / "fig_cu_speciation_decomposition.png"]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ wrote {out.relative_to(ROOT)}")
