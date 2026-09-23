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

ARIMA_FIT (rtol=0.5, atol=0.02), the very wide gross-error-only tolerance
introduced 2026-09-21 for ccf_results_prewhitened.csv and
robustness_severe_ccf.csv's r_arima/p_arima columns, is RETIRED as of the
fix/arima-convergence branch (issue #9) -- see CHANGELOG.md:
  - robustness_severe_ccf.csv's r_arima/p_arima columns are now always NaN
    by design, not a fitted value at all: the ARIMA-prewhitened arm for
    mhw_severe_intensity is marked not_applicable (see
    robustness_severe_ccf_note.json and mhw_robustness.py) because the
    filter, estimated on the driver alone, does not remove EC50's own 2016
    regime shift from the residuals -- 13 of the driver's 17 nonzero months
    fall after that shift. The column-level override is gone; the file now
    takes its default TIGHT tolerance like the rest of its columns (NaN
    trivially matches NaN across platforms).
  - ccf_results_prewhitened.csv, prewhitening_diagnostics.json, and
    robustness_severe_ccf_note.json's `diagnostics` block move to
    STRUCTURAL_ONLY_FILES instead of any numeric tolerance tier -- see below.
    A wide numeric tolerance was the wrong fix for these three: no rtol/atol
    is safe to declare when the *order itself* isn't reproducible.

STRUCTURAL_ONLY_FILES -- no numeric comparison at all, only structural
checks (columns, row count, which driver/lag combinations exist, where NaN
falls). _best_arima_order() (mhw_analysis.py) now discards ARIMA candidates
that don't converge (see its docstring), which removes one real bug -- but
does not make order selection itself reproducible across machines.
Confirmed directly on two consecutive CI runs of this exact branch, same
code, same data, both against GitHub's ubuntu-latest hosted runners:
  - run 35732697098 (2026-09-22): mhw_days -> ARIMA(1,0,1), matching this
    machine exactly.
  - run 35732781059 (2026-09-22), immediately after, no code change (only a
    CHANGELOG.md edit): mhw_days -> an order starting with p=3, matching
    the OLD pre-fix (non-convergent-filtered) reference instead.
For a near-degenerate driver (mhw_days: 45% zero months; mhw_severe_intensity:
94%), *which* candidates converge at all depends on the runner's hardware/
BLAS path, which can change which order wins by AIC even after discarding
non-convergent ones -- a golden master cannot pin numeric values that aren't
a property of the code and data alone. See issue #9 for the V3.1 fix (a
deterministic order-selection procedure); tests/test_mhw_analysis.py covers
the fit/filter/correlate computation itself with a forced, fixed order,
independent of this problem.

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

# Files with no numeric tolerance at all -- see module docstring
# ("STRUCTURAL_ONLY_FILES"). Checked with dedicated structural assertions
# instead of TOLERANCE_BY_FILE + _assert_csv_matches/_assert_json_matches.
STRUCTURAL_ONLY_FILES = {
    "ccf_results_prewhitened.csv",
    "prewhitening_diagnostics.json",
    "robustness_severe_ccf_note.json",
}

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

    # ccf_results_prewhitened.csv, prewhitening_diagnostics.json and
    # robustness_severe_ccf_note.json are in STRUCTURAL_ONLY_FILES instead
    # of here -- see module docstring.

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


_DIAGNOSTICS_KEYS = {"driver", "order", "aic", "converged", "ljung_box_p", "white_noise"}


def _assert_arima_diagnostics_shape(diag, path):
    """Structural-only check for one driver's ARIMA diagnostics block: right
    keys, right types, internally consistent -- never compares order/aic/
    ljung_box_p/white_noise *values* against a reference (see module
    docstring on why those aren't reproducible across machines)."""
    assert set(diag.keys()) == _DIAGNOSTICS_KEYS, f"{path}: unexpected keys {set(diag.keys()) ^ _DIAGNOSTICS_KEYS}"
    assert isinstance(diag["order"], list) and len(diag["order"]) == 3, f"{path}.order: expected a 3-element list"
    assert all(isinstance(x, int) for x in diag["order"]), f"{path}.order: expected 3 ints"
    assert isinstance(diag["aic"], (int, float)), f"{path}.aic: expected a number"
    assert diag["converged"] is True, f"{path}.converged: expected True (non-convergent candidates are discarded)"
    assert isinstance(diag["ljung_box_p"], dict) and set(diag["ljung_box_p"]) == {"6", "12", "24"}, \
        f"{path}.ljung_box_p: expected keys '6'/'12'/'24'"
    assert all(isinstance(v, (int, float)) for v in diag["ljung_box_p"].values()), \
        f"{path}.ljung_box_p: expected numeric p-values"
    assert isinstance(diag["white_noise"], bool), f"{path}.white_noise: expected a bool"


def _assert_prewhitened_ccf_structure(name, actual_dir):
    """ccf_results_prewhitened.csv: same columns, same set of
    (driver, variable, lag) rows, same `n` per row (data-availability-driven,
    not fit-derived, so still reproducible), and the same NaN pattern in
    spearman_r/p_value (insufficient-data rows are deterministic; the FITTED
    values in non-NaN rows are not compared)."""
    ref = pd.read_csv(REFERENCE_DIR / name)
    act = pd.read_csv(actual_dir / name)
    assert list(ref.columns) == list(act.columns), f"{name}: column mismatch"

    key_cols = ["driver", "variable", "lag"]
    ref_keys = set(ref[key_cols].apply(tuple, axis=1))
    act_keys = set(act[key_cols].apply(tuple, axis=1))
    assert ref_keys == act_keys, f"{name}: driver/variable/lag combinations differ: {ref_keys ^ act_keys}"
    assert len(ref) == len(act), f"{name}: row count {len(ref)} (reference) vs {len(act)} (actual)"

    merged = ref.merge(act, on=key_cols, suffixes=("_ref", "_act"))
    assert (merged["n_ref"] == merged["n_act"]).all(), f"{name}: n (valid-pair count) differs where it shouldn't"
    for col in ("spearman_r", "p_value"):
        ref_nan = merged[f"{col}_ref"].isna()
        act_nan = merged[f"{col}_act"].isna()
        assert (ref_nan == act_nan).all(), f"{name}.{col}: NaN pattern differs from reference"


def _assert_prewhitening_diagnostics_structure(name, actual_dir):
    """prewhitening_diagnostics.json: same set of drivers, each with a
    well-formed diagnostics block -- order/aic/ljung_box_p values not
    compared against the reference."""
    ref = json.loads((REFERENCE_DIR / name).read_text())
    act = json.loads((actual_dir / name).read_text())
    assert set(ref.keys()) == set(act.keys()), f"{name}: driver set differs: {set(ref) ^ set(act)}"
    for driver, diag in act.items():
        assert diag["driver"] == driver, f"{name}.{driver}: diagnostics 'driver' field doesn't match its own key"
        _assert_arima_diagnostics_shape(diag, f"{name}.{driver}")


def _assert_severe_note_structure(name, actual_dir):
    """robustness_severe_ccf_note.json: driver/arm/status/reason are fixed
    strings (not fit-derived) and compared exactly; only the nested
    `diagnostics` block is structural-only."""
    ref = json.loads((REFERENCE_DIR / name).read_text())
    act = json.loads((actual_dir / name).read_text())
    for key in ("driver", "arm", "status", "reason"):
        assert ref[key] == act[key], f"{name}.{key}: {act[key]!r} != {ref[key]!r}"
    _assert_arima_diagnostics_shape(act["diagnostics"], f"{name}.diagnostics")


@pytest.mark.golden
def test_golden_master_reference_matches_declared_tolerances():
    """Guards the test itself, not the pipeline: every file in the frozen
    reference must have a declared tolerance (or be structural-only) and
    vice versa -- catches a forgotten file (or a stale entry) the moment
    results/'s shape changes, instead of silently not checking a new file."""
    ref_files = set(_reference_files())
    declared = set(TOLERANCE_BY_FILE) | STRUCTURAL_ONLY_FILES
    assert ref_files == declared, (
        f"in reference but not declared: {ref_files - declared}; "
        f"declared but not in reference: {declared - ref_files}"
    )


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(set(TOLERANCE_BY_FILE) | STRUCTURAL_ONLY_FILES))
def test_golden_master_file(golden_pipeline_results, name):
    if name in R_ONLY_FILES and not (golden_pipeline_results / name).exists():
        pytest.skip("Rscript/dlnm not available in this environment")
    if name == "ccf_results_prewhitened.csv":
        _assert_prewhitened_ccf_structure(name, golden_pipeline_results)
        return
    if name == "prewhitening_diagnostics.json":
        _assert_prewhitening_diagnostics_structure(name, golden_pipeline_results)
        return
    if name == "robustness_severe_ccf_note.json":
        _assert_severe_note_structure(name, golden_pipeline_results)
        return
    tol = TOLERANCE_BY_FILE[name]
    if name.endswith(".csv"):
        _assert_csv_matches(name, golden_pipeline_results, tol)
    else:
        _assert_json_matches(name, golden_pipeline_results, tol)
