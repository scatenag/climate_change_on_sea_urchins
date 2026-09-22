"""
Covers mhw_analysis.py's ARIMA pre-whitening *computation* (fit driver,
filter target, correlate residuals) independently of order *selection*.

Order selection (_best_arima_order) is not reproducible across machines for
near-degenerate drivers -- confirmed by two consecutive CI runs of identical
code and data picking different orders for mhw_days (see issue #4 and
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

import pytest

from climate_change_on_sea_urchins import common
from climate_change_on_sea_urchins.mhw_analysis import compute_ccf_prewhitened

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
    try:
        df_full, _df_real, _events, _monthly = common.load_data()
    finally:
        mp.undo()
    return df_full


def test_compute_ccf_prewhitened_forced_order_is_stable(fixture_df):
    pw, diag = compute_ccf_prewhitened(fixture_df, "mhw_peak_intensity", ["EC50"], order=(1, 0, 1))
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
