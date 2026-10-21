"""
Marine heatwave climatology and events, faithful to Hobday et al. (2016) (ADR-0019).

The generic engine (dataset_builder) uses this module; the pipeline of the Livorno case study keeps `mhw_detection`, whose
published numbers it produced. The two differ on purpose and the difference is measured, not hidden: see ADR-0019.

The rules, with Hobday's standard parameters, all exposed as arguments:
  - a day of the year is the (month, day) of a 366-day calendar, so that 1 March is day 61 in every year; 29 February is not
    used for the percentile and the mean and takes the mean of 28 February and 1 March;
  - the 90th percentile and the mean of a day of the year are taken over all the values within 5 days before and after that day,
    in every year of the baseline (110 values in a 10-year baseline, 330 in a 30-year one), the neighbours being the neighbouring
    days of the series; a neighbour outside the series does not exist;
  - both are then smoothed with a circular 31-day moving average;
  - a day exceeds the threshold when the temperature is strictly above it; an event is a run of at least 5 exceeding days;
    two events separated by at most 2 days are one event (the gap days belong to it).
The category is the floor of (T - climatology) / (threshold - climatology) at the event's peak, 4 at most: Moderate to Extreme.

Not done: the padding of missing days (maxPadLength) and cold spells. The series must have one value per day and no gap; the
builder has already refused one that has.

Nothing here reads the study loaded by `common`: the series and the baseline years are arguments.
"""
import numpy as np
import pandas as pd

from . import mhw_detection

PCTILE = 90
HALF_WIDTH = 5
SMOOTH_WIDTH = 31
MIN_DAYS = 5
MAX_GAP = 2
METHOD = ("Hobday et al. 2016: 90th percentile of the values within ±5 days in the baseline years, smoothed over 31 days; "
          "events of at least 5 days, joined across gaps of up to 2 days")

_LEAP_MONTH_START = np.cumsum([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30])      # days before each month in a leap year
_FEB29 = 60


def calendar_doy(datetimes) -> np.ndarray:
    """The day of the year of a 366-day calendar: 1 March is 61 in every year."""
    dt = pd.DatetimeIndex(datetimes)
    return (_LEAP_MONTH_START[dt.month.to_numpy() - 1] + dt.day.to_numpy()).astype(int)


def _check_daily(df: pd.DataFrame) -> None:
    if df["Temperature"].isna().any():
        raise ValueError("the temperature series has missing days: Hobday's climatology needs one value per day")
    steps = df["Datetime"].diff().dropna()
    if (steps != pd.Timedelta(days=1)).any():
        raise ValueError("the temperature series is not one value per day without gaps")


def _runavg(values: np.ndarray, width: int) -> np.ndarray:
    """Circular moving average of an odd width (the series is repeated three times and the middle copy kept)."""
    n = len(values)
    tripled = np.concatenate([values, values, values])
    return np.convolve(tripled, np.ones(width) / width, mode="same")[n:2 * n]


def compute_climatology(df: pd.DataFrame, clim_start: int, clim_end: int, pctile: float = PCTILE,
                        half_width: int = HALF_WIDTH, smooth_width: int = SMOOTH_WIDTH) -> pd.DataFrame:
    """Threshold and seasonal climatology for each of the 366 days of the calendar, from the years clim_start..clim_end of
    the daily series `df` (Datetime, Temperature). Same columns as `mhw_detection.compute_climatology`; `threshold` and
    `climatology` before the smoothing."""
    df = df.reset_index(drop=True)
    _check_daily(df)
    years = df["Datetime"].dt.year.to_numpy()
    if clim_start < years.min() or clim_end > years.max() or clim_start > clim_end:
        raise ValueError(f"the baseline {clim_start}-{clim_end} is not inside the years of the series {years.min()}-{years.max()}")
    in_baseline = np.where((years >= clim_start) & (years <= clim_end))[0]
    if len(in_baseline) == 0:
        raise ValueError(f"the series has no day in the baseline {clim_start}-{clim_end}")
    first, last = in_baseline[0], in_baseline[-1]
    temp = df["Temperature"].to_numpy(float)
    doy = calendar_doy(df["Datetime"])
    offsets = np.arange(-half_width, half_width + 1)

    threshold = np.full(366, np.nan)
    seasonal = np.full(366, np.nan)
    for d in range(1, 367):
        if d == _FEB29:
            continue
        days = np.where(doy[first:last + 1] == d)[0]
        if len(days) == 0:
            continue
        idx = (first + days[:, None] + offsets[None, :]).ravel()
        idx = idx[(idx >= 0) & (idx < len(temp))]
        values = temp[idx]
        threshold[d - 1] = np.percentile(values, pctile)
        seasonal[d - 1] = values.mean()
    for arr in (threshold, seasonal):
        arr[_FEB29 - 1] = 0.5 * arr[_FEB29 - 2] + 0.5 * arr[_FEB29]
    if np.isnan(threshold).any():
        raise ValueError(f"the baseline {clim_start}-{clim_end} does not cover every day of the year")

    clim = pd.DataFrame({"doy": np.arange(1, 367), "climatology": seasonal, "threshold": threshold})
    clim["climatology_smooth"] = _runavg(seasonal, smooth_width)
    clim["threshold_smooth"] = _runavg(threshold, smooth_width)
    clim["baseline_diff"] = (clim["threshold_smooth"] - clim["climatology_smooth"]).clip(lower=0.01)
    return clim


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of every run of True."""
    padded = np.concatenate([[False], flag, [False]]).astype(int)
    edges = np.diff(padded)
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1))


def detect_events(df: pd.DataFrame, clim: pd.DataFrame, min_days: int = MIN_DAYS, max_gap: int = MAX_GAP) -> tuple[list, pd.DataFrame]:
    """The events of the daily series and the daily table with the same columns as `mhw_detection.detect_events`
    (the metrics are the pipeline's: intensity above the threshold, anomaly from the climatology)."""
    df = df.reset_index(drop=True).copy()
    _check_daily(df)
    df["doy"] = calendar_doy(df["Datetime"])
    by_doy = clim.set_index("doy")
    for col in ["climatology_smooth", "threshold_smooth", "baseline_diff"]:
        df[col] = df["doy"].map(by_doy[col]).to_numpy()
    df["anomaly"] = df["Temperature"] - df["climatology_smooth"]
    df["intensity"] = (df["Temperature"] - df["threshold_smooth"]).clip(lower=0)
    df["intensity_norm"] = (df["anomaly"] / df["baseline_diff"]).clip(lower=0)
    df["exceeds_threshold"] = df["Temperature"] > df["threshold_smooth"]
    df["in_event"] = False
    df["event_id"] = pd.NA

    spans: list[list[int]] = []
    for a, b in _runs(df["exceeds_threshold"].to_numpy()):
        if b - a + 1 < min_days:
            continue
        if spans and a - spans[-1][1] - 1 <= max_gap:
            spans[-1][1] = b
        else:
            spans.append([a, b])

    events = []
    for a, b in spans:
        events.append(mhw_detection._build_event(df.iloc[a:b + 1], len(events) + 1))
        df.loc[a:b, "in_event"] = True
        df.loc[a:b, "event_id"] = len(events)
    return events, df
