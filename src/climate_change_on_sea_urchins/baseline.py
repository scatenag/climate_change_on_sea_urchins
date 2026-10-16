"""The climatology baseline proposed to a new study, the checks on a baseline the user chooses, and the day the
download starts (ADR-0014).

The proposal is 1991-2020 (`DEFAULT_BASELINE`, a named choice with its reason in the ADR). What the product covers is
never written here: the coverage comes from the Copernicus catalogue (`download.Coverage`), as the end of the series
already does. When the product does not cover the default, the first thirty covered years are proposed; when it covers
less than thirty, all of it, with a warning."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .download import Coverage

DEFAULT_BASELINE = (1991, 2020)
STANDARD_YEARS = 30           # a period of 30 years is recommended (Hobday et al. 2016, ADR-0014)
MIN_YEARS = 10                # below this the 90th percentile of a day is the handful of values themselves: an error
SHORT_YEARS = 15              # below this the threshold is noisy: a warning


@dataclass(frozen=True)
class Suggestion:
    start: int
    end: int
    note: str | None = None


def covered_years(coverage: Coverage) -> tuple[int, int]:
    """First and last calendar year the product covers in full."""
    first = coverage.start.year if (coverage.start.month, coverage.start.day) == (1, 1) else coverage.start.year + 1
    last = coverage.end.year if (coverage.end.month, coverage.end.day) == (12, 31) else coverage.end.year - 1
    return first, last


def suggest_baseline(coverage: Coverage) -> Suggestion:
    first, last = covered_years(coverage)
    d0, d1 = DEFAULT_BASELINE
    if first <= d0 and last >= d1:
        return Suggestion(d0, d1)
    if last - first + 1 >= STANDARD_YEARS:
        return Suggestion(first, first + STANDARD_YEARS - 1,
                          f"The product does not cover {d0}-{d1} in full (it covers {first}-{last}): the first "
                          f"{STANDARD_YEARS} covered years are proposed.")
    return Suggestion(first, last,
                      f"The product covers {first}-{last}, shorter than {STANDARD_YEARS} years: all of it is "
                      f"proposed, and the climatology will be less reliable.")


def check_baseline(start: int, end: int, coverage: Coverage) -> tuple[list[str], list[str]]:
    """(errors, warnings) for the baseline years the user chose."""
    first, last = covered_years(coverage)
    errors, warnings = [], []
    if start > end:
        errors.append(f"The baseline start year ({start}) is after its end year ({end}).")
        return errors, warnings
    if start < first:
        errors.append(f"The baseline starts in {start}, before the first year the product covers in full ({first}).")
    if end > last:
        errors.append(f"The baseline ends in {end}, after the last year the product covers in full ({last}).")
    if not errors and end - start + 1 < MIN_YEARS:
        errors.append(f"A baseline of {end - start + 1} years is too short: the heatwave threshold (90th percentile of each "
                      f"day of the year) would be made of the few values themselves. Use at least {MIN_YEARS} years "
                      f"({STANDARD_YEARS} are recommended).")
    if errors:
        return errors, warnings
    if end - start + 1 < SHORT_YEARS:
        warnings.append(f"A baseline of {end - start + 1} years is shorter than {SHORT_YEARS}: the heatwave "
                        f"threshold (90th percentile) will be noisy; {STANDARD_YEARS} years are recommended.")
    if end >= last:
        warnings.append(f"The baseline reaches the last covered year ({last}): no year is left after it to compare with it.")
    return errors, warnings


def download_start(baseline_start: int, coverage: Coverage, first_measurement_year: int | None = None,
                   max_lag_months: int = 12) -> tuple[dt.date, str | None]:
    """The day the download starts: the baseline's first day, or earlier if the measurements start before it (the lags
    use the temperature of the months before), and never before the product's first day (said in the note)."""
    start = dt.date(baseline_start, 1, 1)
    if first_measurement_year is not None:
        years_back = -(-max_lag_months // 12)
        start = min(start, dt.date(first_measurement_year - years_back, 1, 1))
    if start < coverage.start:
        return coverage.start, (f"The temperature is needed from {start}, before the product's first day: the download "
                                f"starts on the first day ({coverage.start}), and the earliest lags of the earliest "
                                "measurements cannot be paired.")
    return start, None
