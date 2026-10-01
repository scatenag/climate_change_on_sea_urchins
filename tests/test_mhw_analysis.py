"""
Covers two independent things in mhw_analysis.py:
  1. The ARIMA pre-whitening *computation* (fit driver, filter target,
     correlate residuals), independently of order *selection*.
  2. _mask_imputed()'s structural property: it masks a target's imputed
     months because a "<target>_imputed" column exists, never because the
     target happens to be named "EC50" -- reused at all three masking
     points in the module (prewhitened residuals, first-differenced arm,
     raw-levels arm).

Order selection (_best_arima_order) is not reproducible across machines for
near-degenerate drivers -- confirmed by two consecutive CI runs of identical
code and data picking different orders for mhw_days (see issue #9 and
test_golden_master.py's STRUCTURAL_ONLY_FILES). Even a fixed order is not
enough on the fixture's MHW driver: its runs of zeros give the MA part of
the filter near-tied residuals that Spearman ranks according to each
machine's arithmetic (issue #9; found when this test, then comparing r with
hand-written values at a fixed order, failed on CI). So the computation is
checked with fixed order AND parameters (`params`) on a synthetic continuous
driver, without near-ties, at tight tolerances; on the fixture only the
structure is checked.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from climate_change_on_sea_urchins import common
from climate_change_on_sea_urchins.common import IMPUTED_COL, RESPONSE_COL
from climate_change_on_sea_urchins.mhw_analysis import _mask_imputed, compute_ccf_prewhitened

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "paper_mpb_2026"

# ── compute_ccf_prewhitened ───────────────────────────────────────────────────
# The ARIMA residuals of a driver with long runs of zeros (mhw_peak_intensity:
# 41% of months) contain near-ties, down to 1e-17 apart, that Spearman ranks
# according to each machine's arithmetic: on the fixture the r values are not
# reproducible across machines even at a fixed order (issue #9; the MA part of
# the filter converges geometrically within a run of zeros). So the whole
# computation is checked with FIXED parameters on a synthetic continuous
# driver, without zeros and hence without near-ties: filters, residuals,
# masking of imputed months, pairs per lag, r and n at tight tolerances, and
# p consistent with r and n. On the fixture only the structure is checked.

TIGHT = (1e-9, 1e-12)
SYNTHETIC_PARAMS = np.array([0.0, 0.5, 0.2, 1.0])  # const, ar.L1, ma.L1, sigma2
# (lag, spearman_r, n), computed once locally with the parameters above; the
# response depends on the driver two months earlier, hence lag 2.
SYNTHETIC_EXPECTED = [
    (0, 0.006896312243905825, 120),
    (1, 0.02481422320994513, 120),
    (2, -0.3707377866400798, 119),
    (3, 0.06810703420872913, 118),
    (4, 0.004728079245905078, 117),
    (5, -0.007250221043324491, 116),
    (6, 0.06390650828431937, 116),
    (7, 0.08746153239169889, 115),
    (8, 0.17947388671756342, 114),
    (9, 0.1115925876638499, 113),
    (10, 0.033604400861038025, 112),
    (11, 0.1185721119349438, 112),
    (12, 0.09944717444717444, 111),
]
# Pairs per lag on the frozen fixture: deterministic (they depend on which
# months are observed and real, not on the fit).
FIXTURE_N = [163, 163, 162, 161, 160, 159, 159, 159, 159, 159, 159, 159, 159]


def _synthetic_prewhitening_frame():
    rng = np.random.default_rng(42)
    n = 150
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.6 * x[i - 1] + rng.normal()
    df = pd.DataFrame({
        "Datetime": pd.date_range("2010-01-01", periods=n, freq="MS"),
        "driver": x + 3,
        RESPONSE_COL: (40 - 0.05 * np.arange(n) - 0.8 * np.r_[np.zeros(2), x[:-2]]
                       + rng.normal(size=n)),
    })
    df[IMPUTED_COL] = np.arange(n) % 5 == 0  # one month in five imputed
    return df


def _assert_p_consistent(row):
    # Spearman's p is a function of r and n: checked for consistency with
    # this run's r and n, with scipy's own formula.
    r, n = row["spearman_r"], row["n"]
    dof = n - 2
    t = r * np.sqrt(dof / ((r + 1.0) * (1.0 - r)))
    assert row["p_value"] == pytest.approx(2 * stats.t.sf(abs(t), dof), rel=1e-12), \
        f"lag {row['lag']}: p_value inconsistent with this run's r and n"


def test_compute_ccf_prewhitened_with_fixed_params_is_exact():
    df = _synthetic_prewhitening_frame()
    pw, diag = compute_ccf_prewhitened(df, "driver", [RESPONSE_COL],
                                       order=(1, 0, 1), params=SYNTHETIC_PARAMS)
    assert diag["order"] == [1, 0, 1]
    assert diag["converged"] is None  # nothing was fitted
    real = ~df[IMPUTED_COL].to_numpy()
    rtol, atol = TIGHT
    for lag, r_expected, n_expected in SYNTHETIC_EXPECTED:
        row = pw[pw["lag"] == lag].iloc[0]
        # imputed months never form a pair: n counts real response months t
        # with a driver month t - lag
        assert row["n"] == n_expected == real[lag:].sum(), f"lag {lag}: n differs"
        assert row["spearman_r"] == pytest.approx(r_expected, rel=rtol, abs=atol), \
            f"lag {lag}: spearman_r differs beyond TIGHT tolerance"
        _assert_p_consistent(row)


def test_compute_ccf_prewhitened_forced_order_structure(fixture_df):
    pw, diag = compute_ccf_prewhitened(fixture_df, "mhw_peak_intensity", [RESPONSE_COL],
                                       order=(1, 0, 1))
    assert diag["order"] == [1, 0, 1]
    assert diag["converged"] is True
    assert list(pw["lag"]) == list(range(13))
    assert list(pw["n"]) == FIXTURE_N
    assert pw["spearman_r"].between(-1, 1).all()
    for _, row in pw.iterrows():
        _assert_p_consistent(row)


@pytest.fixture(scope="module")
def fixture_df():
    mp = pytest.MonkeyPatch()
    mp.setattr(common, "ROOT", FIXTURE_ROOT)
    mp.setattr(common, "DATA", FIXTURE_ROOT / "data")
    try:
        df_full, _df_real, _events, _monthly = common.load_data()
    finally:
        mp.undo()
    return df_full


# ── _mask_imputed: structural masking, independent of target's literal name ──

def test_mask_imputed_nans_out_imputed_positions_regardless_of_column_name():
    # Deliberately NOT "EC50"/"EC50_imputed": proves the property is
    # structural (does a "<target>_imputed" column exist?), not a check on
    # the response's specific name -- the same function is reused at every
    # masking point in mhw_analysis.py (prewhitened residuals, first-diff
    # arm, raw-levels arm), each with a different target/values shape.
    df = pd.DataFrame({
        "righting_response": [1.0, 2.0, 3.0, 4.0],
        "righting_response_imputed": [False, True, False, True],
    })
    values = np.array([10.0, 20.0, 30.0, 40.0])

    out = _mask_imputed(df, "righting_response", values)

    assert list(np.isnan(out)) == [False, True, False, True]
    assert out[0] == 10.0 and out[2] == 30.0


def test_mask_imputed_is_a_noop_when_target_has_no_imputed_column():
    # O2, CO2, Temperature, Salinity, pH are never flagged imputed the way
    # the response series is -- compute_ccf_prewhitened calls this for every
    # target, so the no-op path is exercised on every run, not a corner case.
    df = pd.DataFrame({"pH": [8.0, 8.1, 8.2]})
    values = np.array([1.0, 2.0, 3.0])

    out = _mask_imputed(df, "pH", values)

    assert np.array_equal(out, values)


def test_response_imputed_months_are_nan_in_prewhitened_residuals(fixture_df):
    """The prewhitening arm's masking point, exercised end-to-end (not just
    _mask_imputed in isolation): at lag 0, residuals line up 1:1 with
    fixture_df's rows, so every imputed month must be NaN and every real
    month must not be."""
    pw, _ = compute_ccf_prewhitened(fixture_df, "mhw_peak_intensity", [RESPONSE_COL], order=(1, 0, 1))
    lag0_n = pw.loc[pw["lag"] == 0, "n"].iloc[0]
    real_months = int((~fixture_df[f"{RESPONSE_COL}_imputed"]).sum())
    # n counts valid (non-NaN) driver/target pairs at lag 0; the driver has
    # no NaN of its own (ffill/bfill'd before fitting), so n == real_months
    # exactly iff every imputed month -- and only imputed months -- became NaN.
    assert lag0_n == real_months


def test_ccf_results_diff_and_raw_mask_imputed_response_months():
    """run()'s two non-prewhitened arms (raw levels, first differences) use
    the same structural masking as the prewhitened arm -- checked here on
    the real project data (not the frozen fixture) via a direct call to
    difference_series + _mask_imputed, matching run()'s own construction."""
    from climate_change_on_sea_urchins.common import load_data
    from climate_change_on_sea_urchins.mhw_analysis import difference_series

    df, _df_real, _events, _monthly = load_data()
    imputed = df[f"{RESPONSE_COL}_imputed"].values

    df_ccf = df.copy()
    df_ccf[RESPONSE_COL] = _mask_imputed(df_ccf, RESPONSE_COL, df_ccf[RESPONSE_COL].values)
    assert np.array_equal(df_ccf[RESPONSE_COL].isna().values, imputed)

    df_diff = difference_series(df, [RESPONSE_COL])
    df_diff[RESPONSE_COL] = _mask_imputed(df, RESPONSE_COL, df_diff[RESPONSE_COL].values)
    assert (df_diff.loc[imputed, RESPONSE_COL].isna()).all()
