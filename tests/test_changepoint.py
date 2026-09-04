"""
Tests for changepoint.py's QLR/AR(1) procedure: does it locate a real break
tightly with a low p-value, and does it stay quiet (high p-value) on a
break-free AR(1) series across several seeds? Plus a determinism check.
"""
import numpy as np
import pytest

from climate_change_on_sea_urchins.changepoint import qlr_ar1_changepoint


def _make_ar1(n, phi, mean, std, seed):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, std, size=n)
    y = np.empty(n)
    y[0] = mean + eps[0]
    for t in range(1, n):
        y[t] = mean + phi * (y[t - 1] - mean) + eps[t]
    return y


def test_locates_large_known_break():
    n, true_break = 200, 100
    pre = _make_ar1(true_break, 0.3, mean=0.0, std=1.0, seed=1)
    post = _make_ar1(n - true_break, 0.3, mean=20.0, std=1.0, seed=2)
    y = np.concatenate([pre, post])

    res = qlr_ar1_changepoint(y, B=500, seed=0)

    assert abs(res["break_index"] - true_break) <= 5
    assert res["p_value"] < 0.01


@pytest.mark.parametrize("gen_seed", [1, 2, 3, 4, 5])
def test_no_break_gives_high_p(gen_seed):
    # A well-calibrated test occasionally dips below the nominal 5% level by
    # chance; the check that matters is that it isn't systematically
    # fabricating breaks, so the bar here (1%) is looser than the nominal
    # significance level used elsewhere in this module.
    y = _make_ar1(150, 0.3, mean=0.0, std=1.0, seed=gen_seed)

    res = qlr_ar1_changepoint(y, B=500, seed=0)

    assert res["p_value"] > 0.01


def test_reproducible_with_same_seed():
    y = _make_ar1(150, 0.3, mean=0.0, std=1.0, seed=99)
    res1 = qlr_ar1_changepoint(y, B=300, seed=42)
    res2 = qlr_ar1_changepoint(y, B=300, seed=42)
    assert res1 == res2
