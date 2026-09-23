"""
QLR/AR(1) changepoint detection: a second, more conservative test for a
mean-shift in a univariate series, complementing (not replacing) the
Pettitt test already reported by regime_shift.py.

Where regime_shift.pettitt() answers "when did the series' mean level
change" with a single non-parametric rank test applied directly to the raw
series, this module asks the same question through a parametric lens that
accounts for serial correlation: a Quandt (1960) Likelihood Ratio test for
an unknown breakpoint, after Cochrane-Orcutt AR(1) prewhitening. It also
reports significance (parametric bootstrap p-value) and a location
confidence interval (residual bootstrap) for the break itself, neither of
which the plain Pettitt test provides.

Algorithm, in this exact order:
  (i)   auxiliary Pettitt split of the raw series -- internal only, used to
        seed step (ii); never reported as a result on its own.
  (ii)  residuals of the two-mean model centered on that split; AR(1)
        estimate phi = Pearson correlation of the residuals against
        themselves lagged by one step.
  (iii) Cochrane-Orcutt whitening of the RAW series: y*_t = y_t - phi*y_{t-1}
        for t = 2..n -- the first observation is lost, so positions on the
        whitened series are remapped (+1) when reported against the
        original series.
  (iv)  Quandt F statistic over trimmed candidate breakpoints on the
        whitened series; take the maximum and its position.
  (v)   significance via a parametric bootstrap under the no-break AR(1)
        null (same phi, no mean shift).
  (vi)  90% CI on the break position via a residual bootstrap of the
        two-mean model on the whitened series.

The public function, qlr_ar1_changepoint(), operates on any univariate
sequence (uniform or non-uniform spacing, treated purely ordinally) --
besides EC50 here, it is meant for reuse on the negative-control series and
on the annual MHW metric.

Applied here to the two EC50 representations cited in the manuscript: the
monthly regularised series (data/ec50_sheets.csv, 163 months with >=1 real
determination) and the full-resolution ordinal sequence of individual
bioassay determinations (data/ec50_raw.csv). The manuscript treats the
monthly series as the primary changepoint analysis (fully reproducible: its
163 dates are unique, so its order is unambiguous) and the ordinal sequence
as a secondary check that does not resolve the month.

The ordinal sequence is NOT a well-defined object on its own: many
determinations record only the month, so ~110 of the 295 rows share a
Datetime with at least one other row, and their relative order is not
implied by the data. Sorting by Datetime alone leaves that order to
whatever the sort implementation does with ties -- which is not guaranteed
stable across pandas versions, and differs from one arbitrary choice to
another rather than converging on a single "true" order (verified: phi
ranged 0.246-0.315 and the winning break-month split 65/35 between
September and June across 300 random within-date permutations). This
module therefore requires data/ec50_raw.csv to be sorted by (Datetime, ID)
-- ID being the source sheet's own row order, the only tiebreaker available
that isn't itself arbitrary -- and fails loudly if it isn't (see run()).
This fixes reproducibility (same input -> same output, always) but does
NOT make the resulting break-month meaningful on its own; only the YEAR is
treated as a finding here, consistently with the manuscript.

regime_shift.py is untouched: its own Pettitt break stays where it is, and
uses its own (0-indexed) convention, unrelated to the internal auxiliary
split used here.

Output: results/changepoint_ec50.json
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from .common import RESULTS, RESPONSE_COL, load_ec50_raw, load_ec50_monthly

DEFAULT_B = 3000
DEFAULT_SEED = 0
TRIM = 0.15


def _pettitt_split(y):
    """Pettitt (1979) rank-based single-changepoint estimate.

    Returns a 1-indexed split point t* (1 <= t* <= n-1): the first t*
    observations are 'pre', the rest 'post'.

    Internal only -- this seeds the AR(1) residuals in step (ii) and is
    never reported on its own. Note this uses a different index convention
    from regime_shift.pettitt() (which returns the 0-indexed argmax
    directly, fine for that module's own use); the two are independent,
    self-contained implementations of the same 1979 test.
    """
    y = np.asarray(y, float)
    n = len(y)
    r = stats.rankdata(y)
    U = np.array([2 * r[:t].sum() - t * (n + 1) for t in range(1, n + 1)])
    # argmax(|U|) is a 0-indexed position into an array whose i-th entry is
    # U evaluated at t = i+1, so the split point is that index plus one.
    t_star = int(np.argmax(np.abs(U))) + 1
    return min(max(t_star, 1), n - 1)


def _two_mean_ssr(y, k):
    """Residual sum of squares of a two-mean (step) model split at k."""
    pre, post = y[:k], y[k:]
    resid = np.concatenate([pre - pre.mean(), post - post.mean()])
    return float((resid ** 2).sum())


def _quandt_f(y, candidates):
    """Max Quandt F statistic over candidate two-mean breakpoints in y."""
    n = len(y)
    grand_resid = y - y.mean()
    ssr_restricted = float((grand_resid ** 2).sum())
    best_f, best_k = -np.inf, candidates[0]
    for k in candidates:
        ssr_free = _two_mean_ssr(y, k)
        f = ((ssr_restricted - ssr_free) / 1.0) / (ssr_free / (n - 2))
        if f > best_f:
            best_f, best_k = f, k
    return best_f, best_k


def qlr_ar1_changepoint(y, trim=TRIM, B=DEFAULT_B, seed=DEFAULT_SEED):
    """Quandt Likelihood Ratio changepoint test on an AR(1)-corrupted
    mean-shift series. See module docstring for the algorithm.

    Parameters
    ----------
    y : array-like of float
        Univariate series in order.
    trim : float
        Fraction of the whitened series excluded from each side when
        searching for the break: candidates run from floor(trim*n) up to
        (but excluding) ceil((1-trim)*n), n = length of the whitened series.
    B : int
        Bootstrap replicates, used for both the parametric-null p-value and
        the residual-bootstrap CI.
    seed : int
        RNG seed. Required for reproducibility across runs -- without a
        fixed seed, results are not comparable run to run.

    Returns a dict: n, phi, F_max, break_index (0-indexed into the
    ORIGINAL series y), p_value, ci90_lo_index, ci90_hi_index, B, seed.
    """
    y = np.asarray(y, float)
    n0 = len(y)
    rng = np.random.default_rng(seed)

    # (i) auxiliary Pettitt split -> internal only
    k0 = _pettitt_split(y)

    # (ii) residuals of the two-mean model at k0; phi = Pearson correlation
    # of the residuals against themselves lagged by one step. This is
    # deliberately NOT sum(r_t*r_{t-1}) / sum(r_t**2) on globally-centered
    # residuals (the "textbook" no-intercept OLS estimator): the two differ
    # by a few thousandths on real data, because these residuals are
    # two-piece (pre/post) demeaned, not globally demeaned, so their overall
    # mean isn't exactly zero and the two normalisations diverge slightly.
    pre, post = y[:k0], y[k0:]
    resid0 = np.concatenate([pre - pre.mean(), post - post.mean()])
    phi = float(np.corrcoef(resid0[:-1], resid0[1:])[0, 1])

    # (iii) Cochrane-Orcutt whitening of the RAW series (not the residuals).
    # y*_t = y_t - phi*y_{t-1}, t = 2..n -> loses the first observation, so
    # index j in the whitened series corresponds to index j+1 in y.
    ystar = y[1:] - phi * y[:-1]
    n = len(ystar)

    # (iv) Quandt F over the trimmed interior of the whitened series.
    lo = int(np.floor(trim * n))
    hi = int(np.ceil((1 - trim) * n))
    candidates = list(range(max(1, lo), min(n - 1, hi - 1) + 1))
    f_obs, k_star = _quandt_f(ystar, candidates)
    break_index = k_star + 1  # remap to the original series' indexing

    # (v) parametric bootstrap under the no-break AR(1) null. F is a ratio
    # of two SSRs, so it is invariant to the scale of the simulated series
    # -- the innovation standard deviation used here (fixed at 1) has no
    # bearing on the resulting distribution of F, only phi does.
    f_sims = np.empty(B)
    for b in range(B):
        eps = rng.normal(0.0, 1.0, size=n0)
        sim = np.empty(n0)
        sim[0] = eps[0]
        for t in range(1, n0):
            sim[t] = phi * sim[t - 1] + eps[t]
        sim_star = sim[1:] - phi * sim[:-1]
        f_b, _ = _quandt_f(sim_star, candidates)
        f_sims[b] = f_b
    p_value = float((np.sum(f_sims >= f_obs) + 1) / (B + 1))

    # (vi) 90% CI via residual bootstrap of the two-mean model fitted at
    # k_star on the whitened series, refitting the Quandt F each time.
    fitted = np.concatenate([
        np.full(k_star, ystar[:k_star].mean()),
        np.full(n - k_star, ystar[k_star:].mean()),
    ])
    resid_star = ystar - fitted
    boot_ks = np.empty(B, dtype=int)
    for b in range(B):
        resampled = fitted + rng.choice(resid_star, size=n, replace=True)
        _, kb = _quandt_f(resampled, candidates)
        boot_ks[b] = kb
    ci_lo, ci_hi = np.percentile(boot_ks, [5, 95])

    return {
        "n": n0,
        "phi": phi,
        "F_max": float(f_obs),
        "break_index": int(break_index),
        "p_value": p_value,
        "ci90_lo_index": int(round(ci_lo)) + 1,
        "ci90_hi_index": int(round(ci_hi)) + 1,
        "B": B,
        "seed": seed,
    }


def _apply_to_dated_series(dates, values, B, seed, ordering=None):
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    res = qlr_ar1_changepoint(values, B=B, seed=seed)

    def _date_at(idx):
        idx = min(max(idx, 0), len(dates) - 1)
        return dates.iloc[idx].date().isoformat()

    out = {
        "n": res["n"],
        "phi": res["phi"],
        "F_max": res["F_max"],
        "break_index": res["break_index"],
        "break_date": _date_at(res["break_index"]),
        "p_value": res["p_value"],
        "ci90_lo_date": _date_at(res["ci90_lo_index"]),
        "ci90_hi_date": _date_at(res["ci90_hi_index"]),
        "B": res["B"],
        "seed": res["seed"],
    }
    if ordering is not None:
        out["ordering"] = ordering
    return out


def run(B=DEFAULT_B, seed=DEFAULT_SEED):
    # (Datetime, ID), not Datetime alone -- see module docstring: ~110 of
    # 295 rows tie on Datetime, and that tie order changes phi/F/the winning
    # break by enough to matter (measured: phi 0.246-0.315, break split
    # 65/35 between September/June across 300 random within-date orderings).
    # load_ec50_raw() already sorts this way and renames the response column
    # to RESPONSE_COL.
    raw = load_ec50_raw()
    monthly = load_ec50_monthly()  # months are unique, no tie issue

    summary = {
        "ordinal_sequence": _apply_to_dated_series(
            raw["Datetime"], raw[RESPONSE_COL].values, B, seed, ordering="Datetime, then ID"),
        "monthly_series": _apply_to_dated_series(monthly["Datetime"], monthly[RESPONSE_COL].values, B, seed),
        "note": (
            "The monthly series is the primary changepoint analysis: its 163 "
            "dates are unique, so its order (and therefore phi/F/break) is "
            "unambiguous and fully reproducible. The ordinal (full-resolution) "
            "sequence is a secondary check only: ~110 of its 295 rows share a "
            "Datetime (many determinations record only the month), so its "
            "order is not implied by the data and requires the explicit "
            "(Datetime, ID) tiebreak above to even be reproducible -- with "
            "that fixed, it still does not resolve the break-MONTH (it is "
            "sensitive to which of many equally-valid tie orders is chosen; "
            "see module docstring), only the YEAR (2016) is a stable finding "
            "across representations and orderings tested."
        ),
    }

    with (RESULTS / "changepoint_ec50.json").open("w") as f:
        json.dump(summary, f, indent=2)

    o, m = summary["ordinal_sequence"], summary["monthly_series"]
    print(f"✓ changepoint (QLR/AR(1)): ordinal n={o['n']} break={o['break_date']} "
          f"phi={o['phi']:+.4f} F={o['F_max']:.1f} p={o['p_value']:.4f} | "
          f"monthly n={m['n']} break={m['break_date']} phi={m['phi']:+.4f} "
          f"F={m['F_max']:.1f} p={m['p_value']:.4f}")


if __name__ == "__main__":
    run()
