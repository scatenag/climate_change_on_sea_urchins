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
test_golden_master.py's STRUCTURAL_ONLY_FILES). That instability lives in
*which* candidates converge on a given machine, not in the fit/filter/
correlate arithmetic itself. This test isolates the latter: it forces a
fixed, low order on a main-grid (non-degenerate) driver via
compute_ccf_prewhitened's `order` parameter, skipping the grid search
entirely, and checks the resulting correlations under the same LOOSE
tolerance used elsewhere in this project for iterative-MLE-optimizer
outputs. If this test doesn't hold up on CI, the instability is in the
computation path itself, not just order selection -- a materially different
(and more serious) finding.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_change_on_sea_urchins import common
from climate_change_on_sea_urchins.common import RESPONSE_COL
from climate_change_on_sea_urchins.mhw_analysis import _mask_imputed, compute_ccf_prewhitened

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "paper_mpb_2026"

LOOSE = (1e-3, 1e-12)

# Computed once locally: compute_ccf_prewhitened(df, "mhw_peak_intensity",
# ["EC50"], order=(1, 0, 1)) on the frozen fixture. mhw_peak_intensity is a
# main-grid driver, not one of the near-degenerate event drivers (41% zeros,
# vs. mhw_severe_intensity's 94%) -- a fixed low order on it is expected to
# be well-conditioned.
EXPECTED = [
    {"lag": 0, "spearman_r": -0.06420658276112413, "p_value": 0.4154921276091714, "n": 163},
    {"lag": 1, "spearman_r": -0.18608575656308712, "p_value": 0.017392118051698494, "n": 163},
    {"lag": 2, "spearman_r": -0.2758562777234423, "p_value": 0.00038074299075663634, "n": 162},
    {"lag": 3, "spearman_r": -0.174358753163101, "p_value": 0.02695982341335332, "n": 161},
    {"lag": 4, "spearman_r": -0.0925934606820579, "p_value": 0.24420309552113717, "n": 160},
    {"lag": 5, "spearman_r": -0.0344190351086697, "p_value": 0.6666786904682067, "n": 159},
    {"lag": 6, "spearman_r": -0.0943605206591832, "p_value": 0.23677167071489152, "n": 159},
    {"lag": 7, "spearman_r": -0.07531347026510628, "p_value": 0.3454178116906895, "n": 159},
    {"lag": 8, "spearman_r": -0.1911720802483879, "p_value": 0.015783895778498865, "n": 159},
    {"lag": 9, "spearman_r": -0.16574516360162408, "p_value": 0.03680319082903203, "n": 159},
    {"lag": 10, "spearman_r": -0.117222951994268, "p_value": 0.14114494740072972, "n": 159},
    {"lag": 11, "spearman_r": -0.06683783138285168, "p_value": 0.4025477549269739, "n": 159},
    {"lag": 12, "spearman_r": -0.09717279675185098, "p_value": 0.22302628331201932, "n": 159},
]


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


def test_compute_ccf_prewhitened_forced_order_is_stable(fixture_df):
    pw, diag = compute_ccf_prewhitened(fixture_df, "mhw_peak_intensity", [RESPONSE_COL], order=(1, 0, 1))
    assert diag["order"] == [1, 0, 1]
    assert diag["converged"] is True

    rtol, atol = LOOSE
    for expected in EXPECTED:
        row = pw[pw["lag"] == expected["lag"]].iloc[0]
        assert row["n"] == expected["n"], f"lag {expected['lag']}: n differs"
        assert row["spearman_r"] == pytest.approx(expected["spearman_r"], rel=rtol, abs=atol), \
            f"lag {expected['lag']}: spearman_r differs beyond LOOSE tolerance"
        assert row["p_value"] == pytest.approx(expected["p_value"], rel=rtol, abs=atol), \
            f"lag {expected['lag']}: p_value differs beyond LOOSE tolerance"


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
