# Livorno, *Paracentrotus lividus*: case notes

Case documentation for `study.yaml` in this directory. Module outputs hold only method
descriptions, values computed in the run and identities taken from the study spec. Everything
specific to this case (comparisons with the manuscript, results of past investigations, the
state of the source data) lives here. The texts below were moved out of the module outputs on
2026-09-28 (V2.2): copied as they were, minus people's names and superseded decisions, each
with the whole-record data it refers to.

The manuscript is the Marine Pollution Bulletin submission of 2026; it cites the package's
v1.5.0 release (Zenodo concept DOI `10.5281/zenodo.19352308`), whose frozen data are in
`tests/fixtures/paper_mpb_2026/`.

---

## Trial 224 (2020-01-01): closed

The source sheet's trial ID 224 recorded its three negative-control replicates as `14, 1, 14`.
The `1` was a typo; it is corrected at the source to **`14, 11, 14`** (verified on the source
sheet on 2026-09-28). The two January 2020 rows, IDs 224 and 225, are two distinct assays, not
a duplicate.

- The frozen fixture `tests/fixtures/paper_mpb_2026/` still carries the `1`. The values
  `tests/test_paper_values.py` checks on it are the fixture's, not the corrected data's.
- `data/ec50_raw.csv` carried the `1` too until 2026-09-28, because the auto-update never
  committed that file (fixed then). The committed `results/` had already been computed on the
  corrected data.
- The replicate-outlier check in `negative_control.py` was added after this typo shifted the
  negative-control trend; on the corrected data it flags nothing.

Section 3.6 values, computed with the module, for comparison with the manuscript:

| quantity | frozen fixture (224: 14, 1, 14; 295 trials) | trial 224 corrected only (295 trials) | source sheet 2026-09-28 (296 trials) |
|---|---|---|---|
| trials / with negative control | 295 / 232 | 295 / 232 | 296 / 233 |
| trend, Spearman rho / p | 0.0706 / 0.2844 | 0.0734 / 0.2654 | 0.0834 / 0.2046 |
| pre/post at 2016-01-01: means | 13.72 / 14.12 | 13.72 / 14.15 | 13.72 / 14.17 |
| pre/post at 2016-01-01: Mann-Whitney p | 0.416 | 0.388 | 0.338 |
| pre/post at 2016-01-01: Levene p (center='median') | 0.374 | 0.286 | 0.303 |
| QLR/AR(1), trim 0.10: break, n pre/post, p | 2022-10-26, 203/29, 0.125 | 2022-10-26, 203/29, 0.128 | 2022-10-26, 203/30, 0.086 |

Manuscript values as submitted: trend rho 0.07, p 0.30; pre/post 13.7% / 14.1%, Mann-Whitney
p 0.41, Levene p 0.38. Whether the final manuscript recomputed section 3.6 on the corrected
trial 224 is still to be checked.

---

## Negative control (`negative_control.json`)

- **Published split.** The manuscript's section 3.6 pre/post numbers correspond to a split at
  2016-01-01, the split in effect before `split_date` moved to 2016-06-01: "Reproduces the
  manuscript's published section 3.6 numbers (13.7%/14.1%, Mann-Whitney p=0.41, Levene p=0.38
  with scipy's default center='median') almost exactly." (On the frozen fixture; see the table
  above.) This comparison is still computed inside the module's output
  (`pre_post_published_split`) and is due to move into `tests/test_paper_values.py`.
- **Changepoint trims.** "trim=0.10 locates the manuscript's reported break (Oct 2022,
  n=203/29); trim=0.15 (changepoint.py's module default) excludes that index from its search
  window (it sits at ~88% of the series) and finds a different, also non-significant break
  instead. Both trims are within the 10-15% range the manuscript's methods declare; both agree
  on the conclusion: no significant changepoint in the negative-control series."

---

## Changepoint of the response series (`changepoint_response.json`)

Past investigation, whole record: across the monthly series, the ordinal per-trial sequence
and 300 random within-date orderings of the latter (phi 0.246-0.315, winning break month split
65/35 between September and June), only the break **year**, 2016, was stable. The module
reports the break it computes; the stability across orderings is this investigation's result,
not something the module re-checks.

---

## Annual MHW-metric changepoint (`mhw_annual_changepoint.json`)

Written to reproduce the manuscript's section 3.5, second paragraph. The paragraph was removed
from the manuscript on 2026-09-14, after this module's own investigation surfaced the
discrepancy below; the module still computes every variant.

Manuscript reference values: phi 0.19, break year 2014, bootstrap p 0.076, CI90 2010-2020.

What was not settled is which annual series the manuscript meant and over which year range. The
text said "the annual composite MHW-exposure metric described above (total MHW days)", which
reads as `data/mhw_annual.csv`'s `total_mhw_days`; but `total_mhw_days` alone never reproduces
the reported phi (always negative here, the manuscript reports +0.19) under any year range
tried. "Composite" more likely pointed to section 2.3.3's four annual MHW descriptors: the mean
of their per-year z-scores, restricted to 2004-2025, gives phi +0.184, matching +0.19 almost
exactly. That fixes phi but not the rest: on the same series the Quandt-F search reports a
break at 2022, bootstrap p 0.003, CI90 [2020, 2022].

Structural cause, as recorded in the output until 2026-09-28: "2023 and 2025 have composite
z-scores of +1.82 and +1.94 against +0.64 for 2014 -- whenever both are included in the series,
they dominate qlr_ar1_changepoint's Quandt-F search over any two-mean split, so the reported
break lands at 2022 (splitting the extreme 2022-2025 tail from the rest) regardless of the
metric definition. The manuscript's 2014 break and phi only co-occur with year ranges/metrics
that exclude or dilute that tail, and none of those also reproduce the marginal bootstrap p
(0.076) or the wide CI90 ([2010,2020]) the manuscript reports alongside it."

"No (metric, year_range) variant above reproduces all four manuscript_reference values
simultaneously. [...] The composite_zscore_4_descriptors / 2004-2025 variant reproduces phi
(+0.184 vs +0.19) most closely, but its break_year/bootstrap_p/CI90 do not match; no variant is
otherwise closer on balance, and none is marked as primary or preferred."

Also checked against the data before the site-coordinate correction (the La Spezia grid cell):
phi +0.095, break 2022, so the manuscript's numbers do not come from that version either.

---

## Severe/Extreme MHW driver, ARIMA arm (`robustness_severe_ccf_note.json`)

Whole record, as recorded until 2026-09-28: "13 of 17 nonzero months of mhw_severe_intensity
fall after the 2016-06 EC50 regime shift (SPLIT_DATE). [...] With the shift removed from EC50
(pre/post demeaned), significant lags (p<0.05) at the best-converging order drop from 10/13 to
1/13." The 10/13 to 1/13 reconstruction was done once, during the ARIMA investigation, and is
documented in [issue #9](https://github.com/scatenag/climate_change_on_sea_urchins/issues/9).
The module now writes the method reason with the counts it computes.

---

## Thermal legacy (`thermal_legacy_summary.json`)

The case's hypothesis, as stated in the output until 2026-09-28: "chronic cumulative heat stress
(degree-days above 24C, the gametogenesis-blocking threshold per Amato et al. 2025) on the wild
adult population drives the EC50 decline (copper is the revealer, not the cause)". The 24 °C
threshold comes from Amato et al. (2025); the output now describes the method, with the
threshold taken from the value the dose is computed with.

---

## Annual lagged MHW exposure (`mhw_lag_annual_summary.json`)

Interpretation of the whole-record result, as fixed text until 2026-09-28: "Exposure DURATION
(MHW days) in the previous year predicts EC50 after detrending (Spearman rho=-0.49, p=0.021);
event COUNT does not. Robust to jackknife and direction-asymmetric, but Pearson-weak and does
not survive FDR — a suggestive, pre-registration-worthy carry-over hypothesis, not a
demonstrated effect. Consistent with a delayed (not acute) mechanism, unlike the 2025
acute-exposure experiment." The module now generates its interpretation from the values it
computes.

---

## Regime shift (`regime_shift_summary.json`)

Interpretation of the whole-record result, as fixed text until 2026-09-28 (with the computed
year, p, lag and variance filled in): "Regime shift in EC50 confirmed ~2016 (Pettitt p=1.6e-22).
MHW exposure shifts ~2013, ~3 yr BEFORE the biological collapse — consistent with multi-year
population-scale accumulation, not an acute lag. Environmental stress is multifactorial (PC1 =
71% of T/S/CO2/O2/pH variance). Canonical critical-slowing-down early-warning signals are NOT
present (absolute variance falls, AR(1) not rising) — this is a documented regime shift, not a
demonstrated dynamical tipping point." The stress index's note also read "one coordinated
climate-change axis". The module now generates the verdict from the values it computes,
including the early-warning flag.

---

## Trial-level pre/post contrast (`period_contrast_raw.json`)

The trial-level contrast is the secondary view of the pre/post contrast that the manuscript
reports alongside the monthly one (section 3.1).
