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
from scipy import stats

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
#
# Recomputed 2026-09-30 (fix/mhw-nan-without-sst). The fixture's July 2026 is
# beyond its SST coverage: load_data() used to set its MHW metrics to 0 ("no
# heatwave"), and the ARIMA filter was estimated on that invented 0. Now the
# month is missing and the filter is estimated on the driver's observed span
# only (tests/test_mhw_missing_sst.py). n is unchanged at every lag (July
# 2026's response is imputed, so that month never entered a pair); r and p
# move in the third or fourth decimal, no lag crosses 0.05. p is no longer
# compared with a hand-written value (see the test).
EXPECTED = [
    {"lag": 0, "spearman_r": -0.062172676941493345, "n": 163},
    {"lag": 1, "spearman_r": -0.1805411329334765, "n": 163},
    {"lag": 2, "spearman_r": -0.27447037022923926, "n": 162},
    {"lag": 3, "spearman_r": -0.17375213779057413, "n": 161},
    {"lag": 4, "spearman_r": -0.09044298605414273, "n": 160},
    {"lag": 5, "spearman_r": -0.0319381418676857, "n": 159},
    {"lag": 6, "spearman_r": -0.09510993485040777, "n": 159},
    {"lag": 7, "spearman_r": -0.07220569038520819, "n": 159},
    {"lag": 8, "spearman_r": -0.1858311440171961, "n": 159},
    {"lag": 9, "spearman_r": -0.1626164318127538, "n": 159},
    {"lag": 10, "spearman_r": -0.11448531167900645, "n": 159},
    {"lag": 11, "spearman_r": -0.06268211129687128, "n": 159},
    {"lag": 12, "spearman_r": -0.09417840936231192, "n": 159},
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
        # Spearman's p is a function of r and n: compared with a hand-written
        # value it would only amplify the optimizer's cross-machine noise on r
        # (issue #9; on the CI runner p differed by 3e-3 relative while r stayed
        # within LOOSE). Checked instead for consistency with the r and n of
        # this same run, with scipy's own formula -- no noise on one machine.
        r, n = row["spearman_r"], row["n"]
        dof = n - 2
        t = r * np.sqrt(dof / ((r + 1.0) * (1.0 - r)))
        assert row["p_value"] == pytest.approx(2 * stats.t.sf(abs(t), dof), rel=1e-12), \
            f"lag {expected['lag']}: p_value inconsistent with this run's r and n"


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
