"""
Changepoint (QLR/AR(1)) on the annual MHW-exposure metric (manuscript
section 3.5, second paragraph) -- STATUS: UNRESOLVED, not a finished result.
No longer cited in the manuscript as of 2026-09-14 (D. Sartori removed the
paragraph this was meant to reproduce, once this module's own investigation
surfaced the discrepancy below). Left in place as a record of what was
tried; see cited_in_manuscript / manuscript_citation_note in the output.

reuses qlr_ar1_changepoint() from changepoint.py unchanged, exactly as
already applied to the two EC50 representations there. What is NOT settled
is which annual series the manuscript means and over which year range: the
text says "the annual composite MHW-exposure metric described above (total
MHW days)", and that parenthetical reads as if it just meant
data/mhw_annual.csv's total_mhw_days column -- but total_mhw_days alone
never reproduces the reported phi (always negative here; the manuscript
reports +0.19), under any year range tried. "composite" more likely points
back to section 2.3.3's four annual MHW descriptors (event_count,
total_mhw_days, max_intensity, cum_intensity_sum): the mean of their
per-year z-scores, restricted to 2004-2025 (mhw_lag_annual.py's own
YEAR_MIN/YEAR_MAX -- excludes 2003 as incomplete and 2026 as still in
progress), gives phi=+0.184, matching the manuscript's +0.19 almost exactly.

That fixes phi, but not the rest: qlr_ar1_changepoint's own Quandt-F search
on that same composite/2004-2025 series reports a break at 2022, bootstrap
p=0.003, CI90 [2020,2022] -- not the manuscript's 2014 / p=0.076 / CI90
[2010,2020]. No metric x year-range combination tried reproduces all four
reference values together (see run()'s docstring for the structural reason
and the full variant table in the output).

This was checked against the pre-coordinate-fix data too (the La Spezia
grid cell, before the site-coordinate correction) as a candidate source of
the discrepancy: phi=+0.095, break=2022 there too -- ruled out, the
manuscript's numbers don't come from that version of the data either.

Verified 2026-09-11 (G. Scatena / D.): resolving this is Davide's call, not
made here. This module records every variant tried, side by side with the
manuscript's four reference values, and states plainly that none matches --
deliberately NOT picking one as "the" answer, including in field order or
naming (see run()).

Output: results/mhw_annual_changepoint.json
"""
import json

import numpy as np
import pandas as pd

from .changepoint import DEFAULT_B, DEFAULT_SEED, TRIM, qlr_ar1_changepoint
from .common import RESULTS, load_mhw_annual

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

MANUSCRIPT_REFERENCE = {
    "phi": 0.19,
    "break_year": 2014,
    "bootstrap_p": 0.076,
    "ci90_lo_year": 2010,
    "ci90_hi_year": 2020,
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


def run(B=DEFAULT_B, seed=DEFAULT_SEED):
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

    summary = {
        "status": "unresolved",
        "cited_in_manuscript": False,
        "manuscript_citation_note": (
            "The section 3.5 2nd-paragraph this module was written to "
            "reproduce was removed from the manuscript (D. Sartori, "
            "2026-09-14) -- this module's own investigation is what "
            "surfaced the discrepancy that led to the removal. Left in "
            "place, unresolved, as a record of what was tried; nothing "
            "below is reproducing a manuscript number any more."
        ),
        "manuscript_reference": MANUSCRIPT_REFERENCE,
        "variants": variants,
        "structural_cause": (
            "2023 and 2025 have composite z-scores of +1.82 and +1.94 "
            "against +0.64 for 2014 -- whenever both are included in the "
            "series, they dominate qlr_ar1_changepoint's Quandt-F search "
            "over any two-mean split, so the reported break lands at 2022 "
            "(splitting the extreme 2022-2025 tail from the rest) "
            "regardless of the metric definition. The manuscript's 2014 "
            "break and phi only co-occur with year ranges/metrics that "
            "exclude or dilute that tail, and none of those also reproduce "
            "the marginal bootstrap p (0.076) or the wide CI90 "
            "([2010,2020]) the manuscript reports alongside it."
        ),
        "note": (
            "No (metric, year_range) variant above reproduces all four "
            "manuscript_reference values simultaneously. This is recorded "
            "as an open, unresolved discrepancy, not corrected or resolved "
            "here -- see module docstring. The composite_zscore_4_"
            "descriptors / 2004-2025 variant reproduces phi (+0.184 vs "
            "+0.19) most closely, but its break_year/bootstrap_p/CI90 do "
            "not match; no variant is otherwise closer on balance, and none "
            "is marked as primary or preferred."
        ),
    }

    with (RESULTS / "mhw_annual_changepoint.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"✓ mhw_annual_changepoint: {len(variants)} variants tried, "
        f"UNRESOLVED against manuscript reference (phi={MANUSCRIPT_REFERENCE['phi']}, "
        f"break={MANUSCRIPT_REFERENCE['break_year']}, "
        f"p={MANUSCRIPT_REFERENCE['bootstrap_p']}) -- see results/mhw_annual_changepoint.json"
    )


if __name__ == "__main__":
    run()
