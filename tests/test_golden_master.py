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
                             Also covers ccf_results_prewhitened.csv and
                             robustness_severe_ccf_note.json's numeric
                             diagnostics (aic, ljung_box_p) as of the
                             fix/arima-convergence branch -- see below.
  KNOWN_ISSUE (rtol=1e-2)    -- prewhitening_diagnostics.json only (the
                             ARIMA order/AIC/Ljung-Box diagnostics): held up
                             on the one real cross-machine run measured so
                             far (GitHub Actions, 2026-09-21) -- order
                             selection itself was stable there.

ARIMA_FIT (rtol=0.5, atol=0.02), the very wide gross-error-only tolerance
introduced 2026-09-21 for ccf_results_prewhitened.csv and
robustness_severe_ccf.csv's r_arima/p_arima columns, is RETIRED as of the
fix/arima-convergence branch (issue #4) -- see CHANGELOG.md for what changed
and why the wide margin is no longer needed:
  - _best_arima_order() (mhw_analysis.py) now discards candidates that don't
    converge instead of picking whichever has the lowest AIC regardless.
    Two of the four MHW driver series had a non-converging candidate at the
    AIC optimum (mhw_severe_intensity: ARIMA(3,0,3); mhw_days: ARIMA(3,0,1))
    -- exactly the kind of near-tied, numerically unstable fit that could
    plausibly flip to a different order across machines. Restricting to
    convergent candidates is expected to remove most of that instability,
    which is why ccf_results_prewhitened.csv moves down to LOOSE.
  - robustness_severe_ccf.csv's r_arima/p_arima columns are now always NaN
    by design, not a fitted value at all: the ARIMA-prewhitened arm for
    mhw_severe_intensity is marked not_applicable (see
    robustness_severe_ccf_note.json and mhw_robustness.py) because the
    filter, estimated on the driver alone, does not remove EC50's own 2016
    regime shift from the residuals -- 13 of the driver's 17 nonzero months
    fall after that shift. The column-level override is gone; the file now
    takes its default TIGHT tolerance like the rest of its columns (NaN
    trivially matches NaN across platforms).

R/DLNM outputs (dlnm_results.csv, dlnm_lag_profile.csv, dlnm_slice_lag.csv)
are SKIPPED, not failed, when Rscript/dlnm aren't available in the test
environment -- R stays an optional dependency (see CLAUDE.md).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REFERENCE_DIR = Path(__file__).parent / "fixtures" / "results_v1_5_0"

# Each tier is (rtol, atol) -- numpy/pandas-style combined tolerance:
# |actual - reference| <= atol + rtol * |reference|. atol matters for
# values near zero, where a relative tolerance alone is nearly meaningless.
TIGHT = (1e-6, 1e-12)
MODERATE = (1e-4, 1e-12)
LOOSE = (1e-3, 1e-12)
KNOWN_ISSUE = (1e-2, 1e-12)

# (file, column) -> tolerance tier, for files that mix a stable and an
# ARIMA-derived column and so can't take one tolerance for the whole file.
# Empty as of fix/arima-convergence (see module docstring): the mechanism
# stays for the next file that needs a per-column override.
COLUMN_TOLERANCE_OVERRIDES = {}

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
    "robustness_severe_ccf.csv": TIGHT,  # r_raw/p_raw/r_diff/p_diff are plain
                                          # Spearman; r_arima/p_arima are always
                                          # NaN (arm marked not_applicable, see
                                          # robustness_severe_ccf_note.json)
    "robustness_summer_temp.csv": TIGHT,
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

    # -- ARIMA prewhitening, convergence-filtered as of fix/arima-convergence --
    "ccf_results_prewhitened.csv": LOOSE,          # the fitted correlations themselves
    "prewhitening_diagnostics.json": KNOWN_ISSUE,  # order/AIC/Ljung-Box -- held up on
                                                    # the one real cross-machine run so far
    "robustness_severe_ccf_note.json": LOOSE,      # arm marked not_applicable; numeric
                                                    # diagnostics still MLE-derived

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
    overrides = {col: t for (fname, col), t in COLUMN_TOLERANCE_OVERRIDES.items() if fname == name}

    if not overrides:
        rtol, atol = tol
        pd.testing.assert_frame_equal(ref, act, check_exact=False, rtol=rtol, atol=atol)
        return

    # Mixed file: some columns need a different tolerance than the file's
    # default -- compare column by column instead of the whole-frame fast path.
    assert list(ref.columns) == list(act.columns), f"{name}: column mismatch"
    assert len(ref) == len(act), f"{name}: row count {len(ref)} (reference) vs {len(act)} (actual)"
    for col in ref.columns:
        col_rtol, col_atol = overrides.get(col, tol)
        if pd.api.types.is_numeric_dtype(ref[col]):
            r = ref[col].to_numpy(dtype=float)
            a = act[col].to_numpy(dtype=float)
            ok = np.isclose(r, a, rtol=col_rtol, atol=col_atol, equal_nan=True)
            assert ok.all(), (
                f"{name}.{col}: mismatch at row(s) {list(np.flatnonzero(~ok))} "
                f"(rtol={col_rtol}, atol={col_atol}): reference={r[~ok]} actual={a[~ok]}"
            )
        else:
            assert (ref[col].astype(str) == act[col].astype(str)).all(), f"{name}.{col}: non-numeric mismatch"


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
        rtol, atol = tol
        assert act == pytest.approx(ref, rel=rtol, abs=atol), f"{path}: {act} != {ref} (rtol={rtol}, atol={atol})"
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


@pytest.mark.golden
def test_zzz_debug_dump_arima_files(golden_pipeline_results):
    """TEMPORARY, for cross-platform verification only -- not part of the
    suite's real coverage, will be removed before this PR is finalized."""
    dump = {}
    for fname in ["prewhitening_diagnostics.json", "robustness_severe_ccf_note.json"]:
        dump[fname] = json.loads((golden_pipeline_results / fname).read_text())
    pytest.fail(json.dumps(dump, indent=2))
