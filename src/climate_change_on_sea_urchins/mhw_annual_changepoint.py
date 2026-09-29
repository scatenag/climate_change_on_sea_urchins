"""
Changepoint (QLR/AR(1), qlr_ar1_changepoint() from changepoint.py, unchanged)
on an annual MHW-exposure series, computed for every combination of two
series definitions and several year ranges:
  - composite_zscore_4_descriptors: the mean of the per-year z-scores of the
    four annual MHW descriptors (event_count, total_mhw_days, max_intensity,
    cum_intensity_sum), z-scored within the year range;
  - total_mhw_days_only: total MHW days per year.
Every variant is reported side by side; none is marked as primary.

Why this module exists, what it was compared against and what that
comparison found are case documentation, not module output: see
examples/livorno_paracentrotus/NOTES.md.

Output: results/<study_id>/mhw_annual_changepoint.json
"""
import json

import numpy as np
import pandas as pd

from .changepoint import DEFAULT_B, DEFAULT_SEED, TRIM, qlr_ar1_changepoint
from .common import default_results_dir, load_mhw_annual

# Not run for temporal windows (V2.2): this module iterates its own fixed
# year ranges (YEAR_RANGES below), which a window would contradict.
SUPPORTS_WINDOW = False
WINDOW_UNSUPPORTED_REASON = (
    "mhw_annual_changepoint: iterates its own fixed year ranges, which a temporal window "
    "would contradict"
)

# The four annual MHW descriptors section 2.3.3 defines (same columns
# mhw_lag_annual.py's PREDICTORS sweeps individually over lags).
DESCRIPTOR_COLS = ["event_count", "total_mhw_days", "max_intensity", "cum_intensity_sum"]

# Every year range tried, in a fixed alphabetical-by-label order -- not an
# order of preference. "2004-2025" happens to be the range
# mhw_lag_annual.py already uses elsewhere in this pipeline (excludes 2003
# as incomplete, 2026 as still in progress); it is listed here on equal
# footing with the rest, not singled out.
YEAR_RANGES = {
    "2003-2025": (2003, 2025),
    "2003-2026": (2003, 2026),
    "2004-2025": (2004, 2025),
    "2004-2026": (2004, 2026),
}



def _composite_zscore(ann, lo, hi):
    sub = ann.loc[lo:hi, DESCRIPTOR_COLS]
    z = (sub - sub.mean()) / sub.std(ddof=1)
    return z.mean(axis=1)


def _variant(series, metric_label, range_label, B, seed):
    res = qlr_ar1_changepoint(series.values, trim=TRIM, B=B, seed=seed)
    years = series.index
    return {
        "metric": metric_label,
        "year_range": range_label,
        "n": res["n"],
        "phi": res["phi"],
        "F_max": res["F_max"],
        "break_year": int(years[res["break_index"]]),
        "bootstrap_p": res["p_value"],
        "ci90_lo_year": int(years[res["ci90_lo_index"]]),
        "ci90_hi_year": int(years[res["ci90_hi_index"]]),
        "B": res["B"],
        "seed": res["seed"],
        "trim": TRIM,
    }


def run(B=DEFAULT_B, seed=DEFAULT_SEED, results=None):
    results = results if results is not None else default_results_dir()
    ann = load_mhw_annual().set_index("year")

    variants = []
    # Metrics and ranges are each iterated in a fixed, non-suggestive order
    # (alphabetical); nothing here ranks or recommends a variant.
    for metric_label in ("composite_zscore_4_descriptors", "total_mhw_days_only"):
        for range_label in sorted(YEAR_RANGES):
            lo, hi = YEAR_RANGES[range_label]
            if metric_label == "composite_zscore_4_descriptors":
                series = _composite_zscore(ann, lo, hi)
            else:
                series = ann.loc[lo:hi, "total_mhw_days"]
            variants.append(_variant(series, metric_label, range_label, B, seed))

    # Output: values computed in this run and a method description only.
    # What this module was written to reproduce, and what the comparison
    # found, is case documentation (examples/livorno_paracentrotus/NOTES.md).
    summary = {
        "variants": variants,
        "note": (
            "Every (metric, year range) variant is reported side by side, in a "
            "fixed alphabetical order; none is marked as primary or preferred."
        ),
    }

    with (results / "mhw_annual_changepoint.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ mhw_annual_changepoint: {len(variants)} variants computed")


if __name__ == "__main__":
    run()
