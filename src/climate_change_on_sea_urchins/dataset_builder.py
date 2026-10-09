"""The dataset builder (M1.4): from the daily SST and a monthly response series to the monthly table and
the marine-heatwave catalogue a study is analysed on.

Every number comes from the functions the Livorno pipeline already uses (heatwave detection in
mhw_detection.py, the fill in common.impute_series), driven by what the study declares:

- Monthly temperature: the mean of the daily SST. **A month with any missing day is missing** (decision of
  2026-10-03): a partial mean is distorted by the seasonal cycle, and it would fall on the months the lags
  use most. It fires on an incomplete last month and on an incomplete first month (a period starting
  mid-month) alike. The months it leaves out are listed in the coverage, and are left out of the monthly
  heatwave metrics too (an event spanning them stays in the catalogue: it is an event, whatever the month).
- Response: imputed only as the study declares (`imputation`: absent unless declared), `passes` times, each
  on the series the previous one produced. The declared method is the centered rolling mean, which fills
  within half a window of the observations, edges included, exactly as Livorno's does (an open choice,
  docs/adr/0000 item 8). `response_imputed` is True for a month with no real measurement, filled or not.
- Heatwaves: Hobday et al. (2016) on the daily SST, with the climatology baseline the study declares, which
  the SST must cover in full. The detection needs a continuous daily series: a gap inside it is refused
  with the days named (what to do about gaps is a method choice, issue #41); missing days at the very ends
  are just where the series starts and stops.

Nothing here reads or writes data/: the inputs arrive as frames and the outputs are written where the
caller says (`write_dataset`).
"""
from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import common, mhw_detection

TEMPERATURE_RULE = "a month with any missing SST day is missing (decision of 2026-10-03)"
MONTHLY_COLUMNS = ["Datetime", "Temperature", "response", "response_imputed",
                   "response_ci_upper", "response_ci_lower", "response_n"]


class DatasetError(Exception):
    """The inputs cannot be built into a dataset; the message says why."""


@dataclass
class Dataset:
    monthly: pd.DataFrame
    mhw_events: pd.DataFrame
    mhw_monthly: pd.DataFrame
    mhw_annual: pd.DataFrame
    coverage: dict
    imputation: dict | None
    baseline: tuple[int, int]
    inputs: dict = field(default_factory=dict)


def monthly_temperature(sst: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Monthly mean of the daily SST; NaN for a month with any missing day. Returns the table (every
    month from the first to the last day of the SST) and the report of the incomplete months."""
    s = sst.dropna(subset=["Datetime"]).copy()
    s["month"] = s["Datetime"].dt.to_period("M").dt.to_timestamp()
    rows, incomplete = [], []
    for month in pd.date_range(s["month"].min(), s["month"].max(), freq="MS"):
        expected = calendar.monthrange(month.year, month.month)[1]
        day = s[s["month"] == month]["Temperature"]
        present = int(day.notna().sum())
        if present == expected:
            rows.append((month, float(day.mean())))
        else:
            rows.append((month, np.nan))
            incomplete.append({"month": month.strftime("%Y-%m"), "days_present": present, "days_expected": expected})
    out = pd.DataFrame(rows, columns=["Datetime", "Temperature"])
    return out, {"rule": TEMPERATURE_RULE, "complete_months": int(len(rows) - len(incomplete)),
                 "incomplete_months": incomplete}


def _continuous_daily(sst: pd.DataFrame) -> pd.DataFrame:
    """The SST as one value per day from its first to its last observed day; a gap inside is refused."""
    s = sst[["Datetime", "Temperature"]].dropna(subset=["Datetime"]).sort_values("Datetime")
    if s["Datetime"].duplicated().any():
        raise DatasetError(f"the SST has more than one value for {s.loc[s['Datetime'].duplicated(), 'Datetime'].iloc[0].date()}")
    valid = s.dropna(subset=["Temperature"])
    if valid.empty:
        raise DatasetError("the SST has no values")
    full = pd.DataFrame({"Datetime": pd.date_range(valid["Datetime"].min(), valid["Datetime"].max(), freq="D")})
    full = full.merge(valid, on="Datetime", how="left")
    gaps = full[full["Temperature"].isna()]
    if len(gaps):
        first = gaps["Datetime"].iloc[0].date()
        raise DatasetError(
            f"the SST series has a gap from {first} ({len(gaps)} missing days in total): the heatwave detection "
            "needs one value per day, and what to do about gaps is not decided (issue #41). Re-download the "
            "period or cut the series before the gap.")
    return full


def build_dataset(*, sst: pd.DataFrame, response: pd.DataFrame, imputation, baseline: tuple[int, int]) -> Dataset:
    """`sst`: Datetime, Temperature (daily). `response`: one row per month with Datetime (first of the
    month), value, ci_upper, ci_lower, n (response_csv's `series`). `imputation`: an ImputationSpec or None.
    `baseline`: the climatology years (start, end)."""
    if response is None or response.empty:
        raise DatasetError("the response series has no months")
    resp = response.copy()
    resp["Datetime"] = pd.to_datetime(resp["Datetime"])
    if resp["Datetime"].duplicated().any():
        raise DatasetError(f"the response has more than one row for {resp.loc[resp['Datetime'].duplicated(), 'Datetime'].iloc[0]:%Y-%m}")

    daily = _continuous_daily(sst)
    y0, y1 = baseline
    if daily["Datetime"].min() > pd.Timestamp(f"{y0}-01-01") or daily["Datetime"].max() < pd.Timestamp(f"{y1}-12-31"):
        raise DatasetError(
            f"the SST covers {daily['Datetime'].min().date()} to {daily['Datetime'].max().date()}, not the whole "
            f"climatology baseline {y0}-{y1} the study declares")

    temperature, temp_report = monthly_temperature(daily)

    clim = mhw_detection.compute_climatology(daily, y0, y1)
    events, flagged = mhw_detection.detect_events(daily, clim)
    mhw_events = pd.DataFrame(events)
    excluded = {m["month"] for m in temp_report["incomplete_months"]}
    mhw_monthly = mhw_detection.to_monthly(flagged)
    dropped = mhw_monthly["Datetime"].dt.strftime("%Y-%m").isin(excluded)
    excluded_info = [m for m in temp_report["incomplete_months"] if m["month"] in set(mhw_monthly.loc[dropped, "Datetime"].dt.strftime("%Y-%m"))]
    mhw_monthly = mhw_monthly[~dropped].reset_index(drop=True)
    mhw_annual = mhw_detection.to_annual(events)

    first = min(temperature["Datetime"].min(), resp["Datetime"].min())
    last = max(temperature["Datetime"].max(), resp["Datetime"].max())
    grid = pd.DataFrame({"Datetime": pd.date_range(first, last, freq="MS")})
    r = resp.set_index("Datetime")
    observed = grid["Datetime"].map(r["value"]).astype(float)
    imputed_flag = observed.isna()
    values = observed.copy()
    if imputation is not None:
        for _ in range(imputation.passes):
            values = common.impute_series(values, imputation.window_months, imputation.min_periods)
    monthly = grid.merge(temperature, on="Datetime", how="left")
    monthly["response"] = values.to_numpy()
    monthly["response_imputed"] = imputed_flag.to_numpy()
    for src, dst in [("ci_upper", "response_ci_upper"), ("ci_lower", "response_ci_lower"), ("n", "response_n")]:
        monthly[dst] = grid["Datetime"].map(r[src]).astype(float).where(~imputed_flag).to_numpy()
    monthly = monthly[MONTHLY_COLUMNS]

    with_value = ~imputed_flag
    coverage = {
        "sst": {"first_day": daily["Datetime"].min().date().isoformat(), "last_day": daily["Datetime"].max().date().isoformat(),
                "days": int(len(daily))},
        "temperature": temp_report,
        "response": {"months_with_value": int(with_value.sum()),
                     "months_without_value": int((~with_value).sum()),
                     "months_imputed": int((imputed_flag & monthly["response"].notna()).sum()),
                     "first": grid.loc[with_value, "Datetime"].min().strftime("%Y-%m") if with_value.any() else None,
                     "last": grid.loc[with_value, "Datetime"].max().strftime("%Y-%m") if with_value.any() else None},
        "mhw": {"baseline": [y0, y1], "events": int(len(mhw_events)),
                "months_excluded_incomplete_sst": excluded_info},
    }
    return Dataset(monthly=monthly, mhw_events=mhw_events, mhw_monthly=mhw_monthly, mhw_annual=mhw_annual,
                   coverage=coverage, baseline=(y0, y1),
                   imputation=None if imputation is None else imputation.model_dump())


def write_dataset(ds: Dataset, out_dir: str | Path, *, now: Callable[[], dt.datetime] | None = None) -> dict[str, Path]:
    """Write the monthly table, the heatwave catalogue and a manifest (coverage, declared choices, file
    hashes) into `out_dir`: nothing else."""
    now = now or (lambda: dt.datetime.now(dt.timezone.utc))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {"monthly.csv": ds.monthly, "mhw_events.csv": ds.mhw_events,
              "mhw_monthly.csv": ds.mhw_monthly, "mhw_annual.csv": ds.mhw_annual}
    paths, files = {}, {}
    for name, frame in tables.items():
        path = out / name
        frame.to_csv(path, index=False)
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": int(len(frame))}
        paths[name.removesuffix(".csv")] = path
    try:
        version = metadata.version("climate_change_on_sea_urchins")
    except metadata.PackageNotFoundError:
        version = "unknown"
    manifest = {"format": 1, "kind": "dataset", "built_at": now().isoformat(),
                "baseline": list(ds.baseline), "imputation": ds.imputation, "coverage": ds.coverage,
                "files": files, "versions": {"climate_change_on_sea_urchins": version}}
    mpath = out / "dataset.manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    paths["manifest"] = mpath
    return paths
