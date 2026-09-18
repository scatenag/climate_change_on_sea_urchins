"""
Regression test for the ENTIRE results/ directory -- not just the ~15
manuscript-cited numbers tests/test_paper_values.py covers. Reruns the full
Python pipeline (pipeline.main(), all 16 modules) plus the R DLNM script
(when Rscript/dlnm are available) against the same frozen fixture data
test_paper_values.py uses (tests/fixtures/paper_mpb_2026/, see docs/adr/0004
-- no second fixture), and compares every output file against the frozen
reference in tests/fixtures/results_v1_5_0/.

Marked @pytest.mark.golden: reruns the whole pipeline, ~2-3 minutes. Run
alone with `pytest -m golden`.

Tolerances are declared per file, not applied uniformly, grouped by what
actually produces the number:

  TIGHT (rtol=1e-6)       -- closed-form statistics: no RNG, no iterative
                             optimizer. Empirically EXACT (0 relative
                             difference on every one of these files) across
                             two independent full-pipeline runs on this
                             fixture, verified 2026-09-18 -- the tolerance
                             is headroom for a different machine/BLAS
                             backend someday, not evidence of drift seen
                             here.
  MODERATE (rtol=1e-4)    -- seeded bootstrap/RNG (fixed seed; same
                             empirical result: exact here, tolerance is
                             cross-platform headroom).
  LOOSE (rtol=1e-3)       -- iterative MLE optimizers (SARIMAX, MixedLM/
                             REML): convergence-tolerance-based stopping
                             can in principle amplify tiny floating-point
                             differences more than a closed-form statistic
                             would, even though this run measured 0 here.
  KNOWN_ISSUE (rtol=1e-2) -- ccf_results_prewhitened.csv /
                             prewhitening_diagnostics.json specifically: the
                             ARIMA prewhitening grid-search is NOT invariant
                             to a linear rescaling of the input (found
                             during the v1.5.0 CO2-unit fix: ~0.6% shift in
                             spearman_r at some lags between the pre-fix
                             and post-fix data). Not triggered by re-running
                             on this fixed fixture (also measured at 0 here)
                             -- the wider tolerance is deliberate headroom
                             for that known, separate sensitivity, not a
                             fix. Ask to have a GitHub issue opened for it
                             (no `gh` CLI in this environment to do it here):
                             standardizing the series before the ARIMA fit
                             would make the result scale-invariant by
                             construction -- not done in this session, per
                             the constraint that this session changes no
                             analysis logic.

R/DLNM outputs (dlnm_results.csv, dlnm_lag_profile.csv, dlnm_slice_lag.csv)
are SKIPPED, not failed, when Rscript/dlnm aren't available in the test
environment -- R stays an optional dependency (see CLAUDE.md).
"""
import json
from pathlib import Path

import pandas as pd
import pytest

REFERENCE_DIR = Path(__file__).parent / "fixtures" / "results_v1_5_0"

TIGHT = 1e-6
MODERATE = 1e-4
LOOSE = 1e-3
KNOWN_ISSUE = 1e-2

TOLERANCE_BY_FILE = {
    # -- deterministic / closed-form -----------------------------------------
    "ardl_response.csv": TIGHT,
    "ccf_results.csv": TIGHT,
    "ccf_results_diff.csv": TIGHT,
    "corr_all.csv": TIGHT, "corr_post.csv": TIGHT, "corr_pre.csv": TIGHT,
    "corr_pval_all.csv": TIGHT, "corr_pval_post.csv": TIGHT, "corr_pval_pre.csv": TIGHT,
    "cu_speciation_decomposition.csv": TIGHT, "cu_speciation_summary.json": TIGHT,
    "dist_CO2.csv": TIGHT, "dist_EC50.csv": TIGHT, "dist_O2.csv": TIGHT,
    "dist_Salinity.csv": TIGHT, "dist_Temperature.csv": TIGHT, "dist_pH.csv": TIGHT,
    "granger_results.json": TIGHT,
    "kruskal_stats.json": TIGHT,
    "mhw_lag_annual.csv": TIGHT, "mhw_lag_annual_summary.json": TIGHT,
    "pca_anomaly.csv": TIGHT,
    "period_contrast_raw.json": TIGHT, "period_means.csv": TIGHT,
    "regime_shift_changepoints.csv": TIGHT, "regime_shift_stress_index.csv": TIGHT,
    "regime_shift_summary.json": TIGHT,
    "robustness_ccm.csv": TIGHT,  # deterministic: skccm's train_test_split is a
                                  # positional slice, not a shuffle -- no RNG at all
    "robustness_severe_ccf.csv": TIGHT, "robustness_summer_temp.csv": TIGHT,
    "stationarity_results.json": TIGHT,
    "thermal_legacy.csv": TIGHT, "thermal_legacy_summary.json": TIGHT,
    "thermal_threshold_sensitivity.csv": TIGHT,
    "trend_CO2.csv": TIGHT, "trend_EC50.csv": TIGHT, "trend_O2.csv": TIGHT,
    "trend_Salinity.csv": TIGHT, "trend_Temperature.csv": TIGHT, "trend_pH.csv": TIGHT,
    "trends_all.csv": TIGHT, "trends_post.csv": TIGHT, "trends_pre.csv": TIGHT,

    # -- seeded bootstrap / RNG (all seeds fixed and exposed, per CLAUDE.md) --
    "changepoint_ec50.json": MODERATE,
    "negative_control.json": MODERATE,
    "mhw_annual_changepoint.json": MODERATE,
    "robustness_ml_importance.csv": MODERATE, "robustness_ml_cv_r2.json": MODERATE,
    "robustness_wavelet.json": MODERATE,
    "sea_results.csv": MODERATE,

    # -- iterative MLE optimizer (SARIMAX, MixedLM/REML) -----------------------
    "forecast_bad.csv": LOOSE, "forecast_good.csv": LOOSE, "forecast_mean.csv": LOOSE,
    "forecast_env_bad.csv": LOOSE, "forecast_env_good.csv": LOOSE, "forecast_env_mean.csv": LOOSE,
    "forecast_meta.json": LOOSE,
    "mixed_effects_predictions.csv": LOOSE, "mixed_effects_summary.json": LOOSE,

    # -- known issue: ARIMA prewhitening, not scale-invariant (see module docstring) --
    "ccf_results_prewhitened.csv": KNOWN_ISSUE,
    "prewhitening_diagnostics.json": KNOWN_ISSUE,

    # -- R / DLNM: deterministic given fixed data; skipped if R unavailable ----
    "dlnm_results.csv": TIGHT,
    "dlnm_lag_profile.csv": TIGHT,
    "dlnm_slice_lag.csv": TIGHT,
}

R_ONLY_FILES = {"dlnm_results.csv", "dlnm_lag_profile.csv", "dlnm_slice_lag.csv"}


def _reference_files():
    return sorted(p.name for p in REFERENCE_DIR.iterdir())


def _assert_csv_matches(name, actual_dir, tol):
    ref = pd.read_csv(REFERENCE_DIR / name)
    act = pd.read_csv(actual_dir / name)
    pd.testing.assert_frame_equal(ref, act, check_exact=False, rtol=tol, atol=1e-12)


def _assert_json_value_matches(ref, act, tol, path):
    if isinstance(ref, dict):
        assert isinstance(act, dict), f"{path}: expected dict, got {type(act).__name__}"
        assert ref.keys() == act.keys(), f"{path}: key mismatch {set(ref) ^ set(act)}"
        for k in ref:
            _assert_json_value_matches(ref[k], act[k], tol, f"{path}.{k}")
    elif isinstance(ref, list):
        assert isinstance(act, list), f"{path}: expected list, got {type(act).__name__}"
        assert len(ref) == len(act), f"{path}: length {len(ref)} (reference) vs {len(act)} (actual)"
        for i, (rv, av) in enumerate(zip(ref, act)):
            _assert_json_value_matches(rv, av, tol, f"{path}[{i}]")
    elif isinstance(ref, bool):
        assert ref == act, f"{path}: {act!r} != {ref!r}"
    elif isinstance(ref, (int, float)):
        assert act == pytest.approx(ref, rel=tol, abs=1e-12), f"{path}: {act} != {ref} (rtol={tol})"
    else:
        assert ref == act, f"{path}: {act!r} != {ref!r}"


def _assert_json_matches(name, actual_dir, tol):
    ref = json.loads((REFERENCE_DIR / name).read_text())
    act = json.loads((actual_dir / name).read_text())
    _assert_json_value_matches(ref, act, tol, path=name)


@pytest.mark.golden
def test_golden_master_reference_matches_declared_tolerances():
    """Guards the test itself, not the pipeline: every file in the frozen
    reference must have a declared tolerance and vice versa -- catches a
    forgotten file (or a stale tolerance entry) the moment results/'s shape
    changes, instead of silently not checking a new file."""
    ref_files = set(_reference_files())
    declared = set(TOLERANCE_BY_FILE)
    assert ref_files == declared, (
        f"in reference but no declared tolerance: {ref_files - declared}; "
        f"declared but not in reference: {declared - ref_files}"
    )


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(TOLERANCE_BY_FILE))
def test_golden_master_file(golden_pipeline_results, name):
    if name in R_ONLY_FILES and not (golden_pipeline_results / name).exists():
        pytest.skip("Rscript/dlnm not available in this environment")
    tol = TOLERANCE_BY_FILE[name]
    if name.endswith(".csv"):
        _assert_csv_matches(name, golden_pipeline_results, tol)
    else:
        _assert_json_matches(name, golden_pipeline_results, tol)
