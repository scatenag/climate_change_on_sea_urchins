"""Shared data loading for all analysis modules -- the single boundary for
reading data/ and writing results/ (CLAUDE.md invariant #5). No other
module in src/ should build a data/ or results/ path itself; if one needs
data this file doesn't already expose, add a function here instead."""
import numpy as np
import pandas as pd
from pathlib import Path

from .study_spec import StudySpecError, load_selected_study

ROOT  = Path(__file__).resolve().parent.parent.parent
_study = load_selected_study()

# Where this study's data/ lives -- resolved to an absolute, existing
# directory by load_study() from the spec's data_dir (relative to the
# study.yaml itself). For Livorno this is today's data/. Read through this
# name, never rebuilt as ROOT / "data": a module that binds a path derived
# from DATA to a SEPARATE constant at import time (rather than computing it
# inside a function, from this name, at call time) breaks test fixtures
# that redirect DATA -- this is exactly the bug fixed in mhw_detection.py,
# see its module docstring and tests/test_data_boundary.py.
DATA = Path(_study.data_dir)


def results_dir(study_id: str, window_id: str | None = None) -> Path:
    """Where a study's results live -- the only place this path is built
    (dashboard, tests and the R script all resolve through it). One
    directory per study under results/ (results/<study_id>/), so a second
    study's pipeline run can never overwrite Livorno's -- see docs/adr/0008.
    A declared window's results go one level down, in a sibling of the
    other windows: results/<study_id>/<window_id>/."""
    base = ROOT / "results" / study_id
    return base if window_id is None else base / window_id


RESULTS = results_dir(_study.id)
RESULTS.mkdir(parents=True, exist_ok=True)

TAU_MAX = 12

# The daily-SST baseline period Marine Heatwave detection computes its
# threshold from (Hobday et al. 2016) -- a scientific choice declared in
# the spec (StudySpec.mhw_climatology), not a code default (CLAUDE.md
# invariant #6). Used by mhw_detection.py.
MHW_CLIM_START = _study.mhw_climatology.baseline_start_year
MHW_CLIM_END   = _study.mhw_climatology.baseline_end_year

# Canonical response-series column names. Every module downstream of
# load_data() reads the response column through these two constants, never
# through the literal "EC50"/"EC50_imputed" -- an indirection, not an alias:
# there is exactly one column in the DataFrame either way, so a module that
# hasn't migrated yet cannot silently diverge from one that has by writing
# to the "other" copy.
#
# Flipped from "EC50"/"EC50_imputed" to these generic values now that every
# module has migrated off the literal (V2.1 response-abstraction, note-
# tecniche.md sec 4) -- the .rename() calls below decouple the on-disk CSV
# column names (which never change) from what the rest of the code calls
# the column, so this was the only change that step needed.
RESPONSE_COL = "response"
IMPUTED_COL  = "response_imputed"


# The response imputation: months without a real measurement filled with a
# centered rolling mean. The one implementation, shared with
# scripts/build_dataset.py (which imports it). Its parameters are a
# scientific choice still written in code, not yet in the spec -- recorded
# in docs/adr/0000 (with the fact that, in effect, it is applied twice:
# once by build_dataset.py, once more by load_data() below).
IMPUTE_WINDOW_MONTHS = 12
IMPUTE_MIN_PERIODS = 3


def impute_response(values: pd.Series) -> pd.Series:
    """Fill NaNs with the centered rolling mean of `values` itself --
    whatever `values` covers is all the imputation ever sees."""
    return values.fillna(
        values.rolling(window=IMPUTE_WINDOW_MONTHS, min_periods=IMPUTE_MIN_PERIODS, center=True).mean()
    )


def aggregate_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple within-month measurements to a single monthly value.
    The one implementation: scripts/fetch_ec50.py (the update job) and the
    dashboard's live read of the sheet both call it.

    Strategy:
    - EC50: arithmetic mean of all measurements in the month
    - CI: use the mean of individual half-widths (UL-EC50, EC50-LL),
          then compute standard error across replicates and take
          the wider of the two as the final CI bound.
    """
    raw = raw.copy()
    raw["DATE"] = pd.to_datetime(raw["DATE"], dayfirst=False)
    # Normalize to first-of-month
    raw["Datetime"] = raw["DATE"].dt.to_period("M").dt.to_timestamp()

    # Half-widths from individual bioassay CI
    raw["hw_upper"] = raw["UL"] - raw["EC50"]
    raw["hw_lower"] = raw["EC50"] - raw["LL"]

    agg = raw.groupby("Datetime").agg(
        EC50=("EC50", "mean"),
        EC50_std=("EC50", "std"),
        EC50_n=("EC50", "count"),
        mean_hw_upper=("hw_upper", "mean"),
        mean_hw_lower=("hw_lower", "mean"),
    ).reset_index()

    # Standard error across replicates
    agg["se"] = agg["EC50_std"] / np.sqrt(agg["EC50_n"])

    # Final CI: use the larger of (propagated bioassay CI) vs (replicate SE * 1.96)
    agg["EC50_ci_upper"] = agg["EC50"] + np.maximum(
        agg["mean_hw_upper"], 1.96 * agg["se"].fillna(0)
    )
    agg["EC50_ci_lower"] = agg["EC50"] - np.maximum(
        agg["mean_hw_lower"], 1.96 * agg["se"].fillna(0)
    )

    result = agg[["Datetime", "EC50", "EC50_ci_upper", "EC50_ci_lower", "EC50_n"]].copy()
    result = result.sort_values("Datetime").reset_index(drop=True)
    return result


def impute_ec50(monthly_full: pd.DataFrame, ec50: pd.DataFrame) -> pd.DataFrame:
    """
    Merge EC50 into the full monthly grid.
    Months without bioassay data are filled with a 12-month centered rolling mean.
    CI bounds are NaN for imputed months (flag: EC50_imputed=True).
    The one implementation: scripts/build_dataset.py (the update job) and
    load_data(ec50_monthly=...) (the dashboard's live read) both call it.
    """
    df = pd.merge(monthly_full, ec50, on="Datetime", how="left")

    df["EC50_imputed"] = df["EC50"].isna()

    # Fill missing EC50 with the package's imputation (12-month centered
    # rolling mean, same as the original notebook) -- one implementation,
    # shared with the per-window re-imputation in common.load_data().
    df["EC50"] = impute_response(df["EC50"])

    # CI bounds remain NaN for imputed months
    return df


def _build_from_monthly(data: pd.DataFrame, ec50_monthly: pd.DataFrame
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """data_extended.csv and data_ec50_ci.csv as scripts/build_dataset.py
    builds them, from a monthly response aggregate (aggregate_monthly) read
    elsewhere than data/ -- the environmental columns are data_extended's."""
    env = data.drop(columns=["EC50"])
    start = min(env["Datetime"].min(), ec50_monthly["Datetime"].min())
    end = max(env["Datetime"].max(), ec50_monthly["Datetime"].max())
    grid = pd.DataFrame({"Datetime": pd.date_range(start=start, end=end, freq="MS")})
    built = impute_ec50(pd.merge(grid, env, on="Datetime", how="left"), ec50_monthly)
    data = built[list(env.columns) + ["EC50"]]
    ci_df = built[["Datetime", "EC50", "EC50_ci_upper", "EC50_ci_lower", "EC50_n", "EC50_imputed"]]
    return data, ci_df


def load_data(window=None, ec50_monthly: pd.DataFrame | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_full   — all months (response includes rolling-mean imputations)
        df_real   — only months with real response bioassay measurements
        mhw_events
        mhw_monthly

    With a `window` (a WindowSpec): df_full/df_real hold only the window's
    months, and the response is RE-IMPUTED from the window's real values
    alone -- the imputations already stored in data/ are never read, since
    at the window's edges they were estimated with values from outside it
    (rule (b), tests/test_windows.py). Both imputation passes of the
    whole-record path are reproduced, both inside the window, so a window
    covering the whole record gives the same series. mhw_events and
    mhw_monthly are returned whole: the MHW catalogue is computed once on
    the whole record by design, and events before the window may enter a
    window statistic as lagged predictors (rule (c)).

    With `ec50_monthly` (aggregate_monthly of the source sheet, read live by
    the dashboard): the response is rebuilt from it exactly as the update
    job builds data/ (scripts/build_dataset.py), then prepared as below --
    the same values as the job's when the sheet has not changed
    (tests/test_live_response.py).
    """
    data    = pd.read_csv(DATA / "data_extended.csv",  parse_dates=["Datetime"])
    ci_df   = pd.read_csv(DATA / "data_ec50_ci.csv",   parse_dates=["Datetime"])
    if ec50_monthly is not None:
        data, ci_df = _build_from_monthly(data, ec50_monthly)
    data    = data.rename(columns={"EC50": RESPONSE_COL})
    ci_df   = ci_df.rename(columns={"EC50_imputed": IMPUTED_COL})
    monthly = pd.read_csv(DATA / "mhw_monthly.csv",    parse_dates=["Datetime"])
    events  = pd.read_csv(DATA / "mhw_events.csv",
                          parse_dates=["start_date","end_date","peak_date"])

    # Merge MHW monthly metrics. Months beyond the SST coverage (the daily SST
    # arrives months after the response) have no MHW catalogue: their metrics
    # stay missing, never 0, which would mean "no heatwave"
    # (tests/test_mhw_missing_sst.py).
    df = data.merge(
        monthly[["Datetime","mhw_days","mhw_peak_intensity","mhw_cum_intensity"]],
        on="Datetime", how="left"
    )

    # Imputation flag
    df = df.merge(ci_df[["Datetime",IMPUTED_COL,"EC50_ci_upper","EC50_ci_lower"]],
                  on="Datetime", how="left")
    df[IMPUTED_COL] = df[IMPUTED_COL].fillna(True)

    if window is not None:
        df = df[in_window(df["Datetime"], window)].reset_index(drop=True)
        observed = df[RESPONSE_COL].where(~df[IMPUTED_COL].astype(bool))
        df[RESPONSE_COL] = impute_response(observed)  # build_dataset.py's pass, inside the window

    # Rolling-mean impute the response for df_full (mirrors original notebook approach)
    df[RESPONSE_COL] = impute_response(df[RESPONSE_COL])

    # Fill Temperature gaps (after Copernicus monthly ends) from daily SST monthly averages.
    # Without this, 2024 rows show only Jan–Apr (winter avg ~14°C), breaking trend analysis.
    sst_path = DATA / "sst_daily.csv"
    if sst_path.exists():
        sst = pd.read_csv(sst_path, parse_dates=["Datetime"])
        sst["month"] = sst["Datetime"].dt.to_period("M").dt.to_timestamp()
        sst_monthly = sst.groupby("month")["Temperature"].mean().reset_index()
        sst_monthly.columns = ["Datetime", "Temperature_sst"]
        df = df.merge(sst_monthly, on="Datetime", how="left")
        mask = df["Temperature"].isna() & df["Temperature_sst"].notna()
        df.loc[mask, "Temperature"] = df.loc[mask, "Temperature_sst"]
        df.drop(columns=["Temperature_sst"], inplace=True)

    df_full = df.copy()
    df_real = df[~df[IMPUTED_COL]].copy().reset_index(drop=True)

    return df_full, df_real, events, monthly


ENV_COLS  = ["O2", "CO2", "Temperature", "Salinity", "pH"]
ALL_COLS  = ENV_COLS + [RESPONSE_COL]
MHW_COLS  = ["mhw_peak_intensity", "mhw_days"]


def load_ec50_raw() -> pd.DataFrame:
    """The per-trial (single-bioassay) raw sequence, data/ec50_raw.csv --
    one row per determination, not yet aggregated to any period (distinct
    from load_data()'s monthly df_full/df_real). Every column from the CSV
    is preserved (ID, the negative-control replicate columns, etc.); only
    the response value column is renamed to RESPONSE_COL, so callers never
    reference the literal "EC50".

    Sorted by (Datetime, ID), not Datetime alone: ~110 of 295 rows share a
    Datetime (many determinations record only the month), and that tie
    order changes downstream statistics enough to matter (see
    changepoint.py's module docstring) -- ID (the source sheet's row order)
    is what makes the sequence reproducible.
    """
    raw = pd.read_csv(DATA / "ec50_raw.csv", parse_dates=["Datetime"])
    raw = raw.rename(columns={"EC50": RESPONSE_COL})
    if "ID" not in raw.columns:
        raise ValueError(
            "data/ec50_raw.csv is missing the ID column -- re-run "
            "scripts/fetch_ec50.py. Many determinations share a Datetime "
            "(month-only dates), so ID (the source sheet's row order) is "
            "required to give the ordinal sequence a reproducible order; "
            "sorting by Datetime alone is not enough."
        )
    return raw.sort_values(["Datetime", "ID"]).reset_index(drop=True)


def load_ec50_monthly() -> pd.DataFrame:
    """The monthly-aggregated response series as fetched directly from the
    source, data/ec50_sheets.csv -- distinct from load_data()'s df_full/
    df_real, which merge in environmental variables and MHW metrics on top
    of it. Its dates are unique (one row per month), unlike ec50_raw.csv's.
    Response value column renamed to RESPONSE_COL, same as load_ec50_raw().
    """
    monthly = pd.read_csv(DATA / "ec50_sheets.csv", parse_dates=["Datetime"])
    monthly = monthly.rename(columns={"EC50": RESPONSE_COL})
    return monthly.sort_values("Datetime").reset_index(drop=True)


def load_mhw_annual() -> pd.DataFrame:
    """Annual MHW metrics, data/mhw_annual.csv (written by mhw_detection)."""
    return pd.read_csv(DATA / "mhw_annual.csv")


def load_sst_daily() -> pd.DataFrame:
    """Daily SST, data/sst_daily.csv -- mhw_detection's own input, and
    thermal_legacy's cumulative-dose predictor."""
    return pd.read_csv(DATA / "sst_daily.csv", parse_dates=["Datetime"])


def _response_split_date(response) -> pd.Timestamp:
    """response.split_date (ISO string, see ResponseSpec and ADR-0007),
    validated against the response series' own date range -- rejects a
    split_date that would leave period_split.py (or anything else slicing
    on it) with an empty or single-point pre/post side, at load time, with
    an explicit message, instead of downstream in the pipeline."""
    split_date = pd.Timestamp(response.split_date)
    monthly = load_ec50_monthly()
    series_min, series_max = monthly["Datetime"].min(), monthly["Datetime"].max()
    if not (series_min < split_date < series_max):
        raise StudySpecError(
            f"response {response.id!r}: split_date {response.split_date!r} does not fall "
            f"strictly within the response series' date range "
            f"({series_min.date()}..{series_max.date()}) -- pre/post analyses would run "
            "with an empty or single-point side. Fix split_date in the study spec."
        )
    return split_date


SPLIT_DATE = _response_split_date(_study.responses[0])
SPLIT_YEAR = str(SPLIT_DATE.year)  # kept for callers that only need the year (e.g. axis labels)


def in_window(dates: pd.Series, window) -> pd.Series:
    """Boolean mask of `dates` inside `window` (both ends inclusive); all
    True when window is None."""
    if window is None:
        return pd.Series(True, index=dates.index)
    return (dates >= pd.Timestamp(window.start)) & (dates <= pd.Timestamp(window.end))


def supports_window(module) -> bool:
    """Window support is declared explicitly by a module-level
    `SUPPORTS_WINDOW = True`; anything else -- False, a missing
    declaration, a truthy non-True value -- counts as unsupported, so a
    module not yet migrated is excluded from windows instead of receiving
    one and ignoring it."""
    return getattr(module, "SUPPORTS_WINDOW", False) is True


def response_coverage(window=None) -> dict:
    """First and last month with a real response measurement (and their
    count), inside `window` if given -- the window's effective coverage,
    which can be narrower than its declared start/end."""
    _, real, _, _ = load_data()
    months = real.loc[in_window(real["Datetime"], window), "Datetime"]
    return {
        "first_real_response_month": months.min().date().isoformat() if len(months) else None,
        "last_real_response_month": months.max().date().isoformat() if len(months) else None,
        "n_real_response_months": int(len(months)),
    }


def check_window_overlaps_data(window) -> None:
    """Rejects only a window with no real response month at all; a partial
    overlap is accepted, and window.json records the effective coverage."""
    if response_coverage(window)["n_real_response_months"] == 0:
        raise StudySpecError(
            f"window {window.id!r} ({window.start}..{window.end}) has no overlap with the "
            "response series: no real measurement falls inside it."
        )


def split_date_in_window(window) -> bool:
    """Whether SPLIT_DATE leaves a non-empty pre side AND post side of real
    response months inside the window -- judged on the data, not on the
    declared bounds alone, so a window starting a week before SPLIT_DATE
    does not run a pre/post test on an empty side. When False, pre/post
    analyses are skipped for that window and window.json says so."""
    _, real, _, _ = load_data()
    months = real.loc[in_window(real["Datetime"], window), "Datetime"]
    return bool((months < SPLIT_DATE).any() and (months >= SPLIT_DATE).any())


def provenance() -> dict:
    """Code commit and study-spec hash (CLAUDE.md invariant #8), for
    window.json. code_commit is None outside a git checkout."""
    import hashlib
    import subprocess
    from .study_spec import selected_study_path

    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = _git("rev-parse", "HEAD")
    dirty = None if commit is None else bool(_git("status", "--porcelain", "--untracked-files=no"))
    spec_path = selected_study_path()
    return {
        "code_commit": commit,
        "code_dirty": dirty,
        "study_spec": str(spec_path),
        "study_spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    }


def _checked_windows(windows) -> tuple:
    for w in windows:
        check_window_overlaps_data(w)
    return tuple(windows)


# The selected study's declared windows, each checked against the data at
# load time (as SPLIT_DATE is): a window with no overlap fails here, with
# an explicit message, not halfway through the pipeline.
WINDOWS = _checked_windows(_study.windows)


def default_results_dir() -> Path:
    """Fallback for a module's `results` parameter when a caller doesn't
    pass one (`python -m module`, ad-hoc use); pipeline.py always passes
    one. Same shape and same reason as default_response_spec() below: a
    module reaching this can only write the selected study's whole-record
    results, never a window's. Read at call time, never bound at import."""
    return RESULTS


def default_response_spec():
    """Loads config.RESPONSE_SPEC -- the one place this happens outside
    pipeline.py's own explicit load. Fallback for a module's `response`
    parameter when a caller doesn't pass one explicitly (tests, `python -m
    module`); pipeline.py always passes one.

    Transitional (V2.1 response-abstraction, note-tecniche.md sec 4): remove
    once every caller passes the object explicitly. A module that reaches
    this default can only ever process the one case config.py currently
    points at -- that's why it's a fallback, not the primary path, and why
    this import is local to the function rather than a module-level import
    of config (which would tie every module that imports common.py, not
    just the ones that actually hit this fallback, to a single case).
    """
    import config
    return config.RESPONSE_SPEC
