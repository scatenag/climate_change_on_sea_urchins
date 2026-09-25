"""Shared data loading for all analysis modules -- the single boundary for
reading data/ and writing results/ (CLAUDE.md invariant #5). No other
module in src/ should build a data/ or results/ path itself; if one needs
data this file doesn't already expose, add a function here instead."""
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


def results_dir(study_id: str) -> Path:
    """Where a study's results live -- the only place this path is built
    (dashboard, tests and the R script all resolve through it). One
    directory per study under results/ (results/<study_id>/), so a second
    study's pipeline run can never overwrite Livorno's -- see docs/adr/0008."""
    return ROOT / "results" / study_id


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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_full   — all months (response includes rolling-mean imputations)
        df_real   — only months with real response bioassay measurements
        mhw_events
        mhw_monthly
    """
    data    = pd.read_csv(DATA / "data_extended.csv",  parse_dates=["Datetime"])
    data    = data.rename(columns={"EC50": RESPONSE_COL})
    ci_df   = pd.read_csv(DATA / "data_ec50_ci.csv",   parse_dates=["Datetime"])
    ci_df   = ci_df.rename(columns={"EC50_imputed": IMPUTED_COL})
    monthly = pd.read_csv(DATA / "mhw_monthly.csv",    parse_dates=["Datetime"])
    events  = pd.read_csv(DATA / "mhw_events.csv",
                          parse_dates=["start_date","end_date","peak_date"])

    # Merge MHW monthly metrics
    df = data.merge(
        monthly[["Datetime","mhw_days","mhw_peak_intensity","mhw_cum_intensity"]],
        on="Datetime", how="left"
    )
    df["mhw_days"]            = df["mhw_days"].fillna(0)
    df["mhw_peak_intensity"]  = df["mhw_peak_intensity"].fillna(0)
    df["mhw_cum_intensity"]   = df["mhw_cum_intensity"].fillna(0)

    # Imputation flag
    df = df.merge(ci_df[["Datetime",IMPUTED_COL,"EC50_ci_upper","EC50_ci_lower"]],
                  on="Datetime", how="left")
    df[IMPUTED_COL] = df[IMPUTED_COL].fillna(True)

    # Rolling-mean impute the response for df_full (mirrors original notebook approach)
    df[RESPONSE_COL] = df[RESPONSE_COL].fillna(
        df[RESPONSE_COL].rolling(window=12, min_periods=3, center=True).mean()
    )

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
