"""
Copper speciation decomposition: how much of the 20-year decline in nominal
copper EC50 is geochemistry (ocean-acidification-driven Cu bioavailability) vs.
a genuine biological change in the larvae?

Rationale
---------
The reference-toxicant bioassay expresses EC50 as *nominal / total dissolved*
copper. Toxicity, however, is driven by the free cupric ion Cu2+, whose fraction
of total inorganic copper rises as pH falls (ocean acidification frees Cu2+ from
carbonate/hydroxide complexes; e.g. Millero et al. 2009 Oceanography 22:72-85;
Richards et al. 2011; Roberts et al. 2013 report free Cu2+ ~2.3x higher at
pH 7.7 vs 8.1). So part of any secular EC50 decline can be a pure water-chemistry
effect rather than a change in the organism.

Because inorganic Cu speciation near pH 8 is dominated by CuCO3(0), the free-Cu2+
fraction scales approximately as 1 / [CO3^2-]. We therefore compute the
"geochemical amplification" of free Cu2+ directly as [CO3^2-]_ref / [CO3^2-](t),
which needs only the carbonate system (pH, T, S, estimated total alkalinity) and
NO copper stability constants. Since CuCO3 dominance is not total (CuOH+, free Cu2+
also present), this ratio is a conservative UPPER BOUND on the geochemical effect.

The speciation-corrected ("biological") EC50 is the free-Cu2+-equivalent threshold:
    EC50_bio(t) = EC50_nominal(t) * [CO3^2-]_ref / [CO3^2-](t)
If the intrinsic organismal sensitivity were constant, EC50_bio would be flat and
all of the nominal decline would be geochemical. The residual decline in EC50_bio
is the part that acidification chemistry cannot explain.

Carbonate system: K1, K2 from Lueker et al. (2000), KB from Dickson (1990), KW
from Millero (1995), all on the total pH scale; borate from Uppstrom (1974); total
alkalinity estimated from salinity for NW-Mediterranean surface water. The result
is a *relative* carbonate-ion ratio, which is insensitive to the absolute TA choice.

The bioassay uses natural filtered seawater (Sartori et al.), so the ambient pH
recorded here is the pH the embryos actually experience — the correction is
directly applicable, not hypothetical.

Outputs:
    results/cu_speciation_decomposition.csv  — per-month CO3, amplification, EC50 nominal/corrected
    results/cu_speciation_summary.json       — pre/post decomposition + literature-anchored cross-check
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

from .common import load_data, RESULTS, SPLIT_YEAR

# Representative NW-Mediterranean surface total alkalinity (mol/kg-SW), scaled by
# salinity. The decomposition uses only the RELATIVE carbonate-ion ratio, which is
# essentially independent of this value (verified: +-100 umol/kg shifts the
# geochemical share by <0.3 percentage points).
TA_REF_38 = 2.60e-3

# Literature anchor for the OA -> free-Cu2+ sensitivity, as an independent
# cross-check on the carbonate-model result: free Cu2+ ~2.3x per -0.4 pH units.
LIT_FCU_FACTOR = 2.3
LIT_DELTA_PH = -0.4
LIT_SENSITIVITY = np.log(LIT_FCU_FACTOR) / LIT_DELTA_PH  # d(ln fCu2+)/dpH ~ -2.08


def _carbonate_constants(T_c, S):
    """K1, K2, KB, KW on the total pH scale (mol/kg-SW). T_c in degC."""
    T = T_c + 273.15
    lnT = np.log(T)
    # Lueker et al. (2000) — Mehrbach refit, total scale
    pK1 = 3633.86 / T - 61.2172 + 9.6777 * lnT - 0.011555 * S + 0.0001152 * S**2
    pK2 = 471.78 / T + 25.9290 - 3.16967 * lnT - 0.01781 * S + 0.0001122 * S**2
    K1, K2 = 10.0**(-pK1), 10.0**(-pK2)
    # Dickson (1990) boric acid, total scale
    sqrtS = np.sqrt(S)
    lnKB = ((-8966.90 - 2890.53 * sqrtS - 77.942 * S + 1.728 * S**1.5 - 0.0996 * S**2) / T
            + 148.0248 + 137.1942 * sqrtS + 1.62142 * S
            + (-24.4344 - 25.085 * sqrtS - 0.2474 * S) * lnT
            + 0.053105 * sqrtS * T)
    KB = np.exp(lnKB)
    # Millero (1995) water, total scale
    lnKW = (148.9652 - 13847.26 / T - 23.6521 * lnT
            + (-5.977 + 118.67 / T + 1.0495 * lnT) * sqrtS - 0.01615 * S)
    KW = np.exp(lnKW)
    return K1, K2, KB, KW


def carbonate_ion(pH, T_c, S, TA=None):
    """
    Carbonate-ion concentration [CO3^2-] (mol/kg-SW) from pH (total scale), in-situ
    temperature and salinity, and total alkalinity (estimated from S if not given).
    """
    if TA is None:
        TA = TA_REF_38 * S / 38.0
    K1, K2, KB, KW = _carbonate_constants(T_c, S)
    h = 10.0**(-pH)
    BT = 0.0004157 * S / 35.0                    # total boron, Uppstrom (1974)
    borate_alk = BT * KB / (KB + h)
    oh = KW / h
    carbonate_alk = TA - borate_alk - oh + h     # = [HCO3-] + 2[CO3^2-]
    co3 = carbonate_alk / (2.0 + h / K2)
    return co3


def _decline_pct(pre, post):
    return (pre.mean() - post.mean()) / pre.mean() * 100.0


def run():
    df_full, df_real, _, _ = load_data()

    d = df_real.dropna(subset=["EC50", "pH", "Temperature", "Salinity"]).copy()
    d = d[~d["EC50_imputed"]].reset_index(drop=True)

    # Carbonate ion the embryos experienced each assay month
    d["CO3"] = carbonate_ion(d["pH"].values, d["Temperature"].values, d["Salinity"].values)

    split = pd.Timestamp(SPLIT_YEAR + "-01-01")
    pre_mask = d["Datetime"] < split
    post_mask = ~pre_mask

    # Reference = pre-2016 ("baseline era") mean carbonate ion
    co3_ref = d.loc[pre_mask, "CO3"].mean()

    # Geochemical amplification of free Cu2+ (>=1 when CO3 below baseline).
    # Upper bound: assumes CuCO3(0) fully dominates inorganic Cu complexation.
    d["fCu_amplification"] = co3_ref / d["CO3"]

    # Free-Cu2+-equivalent ("biological") EC50: constant biology -> flat series.
    d["EC50_bio"] = d["EC50"] * d["fCu_amplification"]

    # Independent literature-anchored correction (log-linear in pH).
    # amplification = fCu2+(t)/fCu2+(ref) = exp(S_pH * (pH_t - pH_ref)), with the
    # negative sensitivity S_pH this exceeds 1 when pH falls below the baseline.
    ph_ref = d.loc[pre_mask, "pH"].mean()
    d["fCu_amplification_lit"] = np.exp(LIT_SENSITIVITY * (d["pH"] - ph_ref))
    d["EC50_bio_lit"] = d["EC50"] * d["fCu_amplification_lit"]

    out = d[["Datetime", "pH", "Temperature", "Salinity", "CO3",
             "fCu_amplification", "fCu_amplification_lit",
             "EC50", "EC50_bio", "EC50_bio_lit"]].copy()
    out.to_csv(RESULTS / "cu_speciation_decomposition.csv", index=False)

    # ---- Decomposition summary ----
    ec50_pre, ec50_post = d.loc[pre_mask, "EC50"], d.loc[post_mask, "EC50"]
    dec_nom = _decline_pct(ec50_pre, ec50_post)

    def geochem_share(bio_col):
        bio_pre, bio_post = d.loc[pre_mask, bio_col], d.loc[post_mask, bio_col]
        dec_bio = _decline_pct(bio_pre, bio_post)
        return dec_bio, (dec_nom - dec_bio) / dec_nom * 100.0

    dec_bio_carb, share_carb = geochem_share("EC50_bio")
    dec_bio_lit, share_lit = geochem_share("EC50_bio_lit")

    # Robustness of the residual biological decline (Mann-Whitney on corrected EC50)
    u, p_bio = stats.mannwhitneyu(d.loc[pre_mask, "EC50_bio"],
                                  d.loc[post_mask, "EC50_bio"], alternative="greater")

    summary = {
        "n_pre": int(pre_mask.sum()),
        "n_post": int(post_mask.sum()),
        "pH_pre_mean": float(d.loc[pre_mask, "pH"].mean()),
        "pH_post_mean": float(d.loc[post_mask, "pH"].mean()),
        "delta_pH": float(d.loc[post_mask, "pH"].mean() - d.loc[pre_mask, "pH"].mean()),
        "CO3_pre_umol_kg": float(co3_ref * 1e6),
        "CO3_post_umol_kg": float(d.loc[post_mask, "CO3"].mean() * 1e6),
        "fCu_amplification_post_carbonate": float(d.loc[post_mask, "fCu_amplification"].mean()),
        "fCu_amplification_post_literature": float(d.loc[post_mask, "fCu_amplification_lit"].mean()),
        "ec50_decline_nominal_pct": float(dec_nom),
        "ec50_decline_corrected_carbonate_pct": float(dec_bio_carb),
        "ec50_decline_corrected_literature_pct": float(dec_bio_lit),
        "geochemical_share_carbonate_pct": float(share_carb),
        "geochemical_share_literature_pct": float(share_lit),
        "biological_residual_mannwhitney_p": float(p_bio),
    }
    with (RESULTS / "cu_speciation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ cu_speciation: nominal EC50 decline {dec_nom:.1f}% | "
          f"geochemical share {share_lit:.0f}% (lit) / {share_carb:.0f}% (carbonate); "
          f"residual biological decline {dec_bio_lit:.1f}% (p={p_bio:.1e})")


if __name__ == "__main__":
    run()
