"""Step 1 of the user's own data: site, period and credentials, and the download of the daily sea-surface temperature
(milestone MA.2, docs/roadmap/MILESTONE-M1.md; baseline: ADR-0014; credentials and download: ADR-0016).

Only functions: the interface runs in `render(ctx)` at every run, and what the step produces lives in the session's
context (`ctx.data`), never in the module.

Credentials. They are the user's, typed in a form that clears on submit. The submit button's callback reads them, empties
the two fields in the session state at once, and passes them as arguments to `download.fetch_daily_sst` and to nothing
else: not to a file, not to the context, not to a cache (no st.cache_* here), not to a log or an environment variable, not
into an error message (every message goes through the masking of `download`). The toolbox's own sign-in, which stores
them in a file, is never used. The local variables that hold them are deleted as soon as the call returns. The download
runs in the callback, in this process and in the script thread of the session, and does not hold the global lock (it
lasts tens of seconds); the short numeric work of the view (heatwave detection) takes the lock for its own duration only,
once per download."""
from __future__ import annotations

import importlib.util
import math
import re
import shutil
import time
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from . import RERUN_LOCK, baseline, catalog, dataset_builder, download, step_run
from .study_spec import AnalysisSpec, SiteSpec

CATALOG_ID = "sst_daily"
EXTRA_MODULES = ("copernicusmarine", "xarray", "h5netcdf", "h5py")
BBOX_DELTA = 0.1                    # degrees around the site: the temperature is the mean over the sea cells of this box
USER_KEY, PW_KEY = "s1_user", "s1_pw"
_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", re.ASCII)
_YEAR = re.compile(r"\d{4}", re.ASCII)
_NAME = re.compile(r"[\w .,'()\-]{1,60}")


class InputError(Exception):
    """An input of the form that is missing or wrong; the message says which and why."""


@dataclass(frozen=True)
class SiteInputs:
    site: SiteSpec
    baseline: tuple[int, int]
    first_year: int | None

    @property
    def key(self) -> tuple:
        return (self.site.lat, self.site.lon, self.site.name, self.baseline, self.first_year)


def acquisition_available() -> bool:
    """Whether the packages of the `acquisition` extra are installed on this server."""
    return all(importlib.util.find_spec(m) is not None for m in EXTRA_MODULES)


def make_client():
    """The real Copernicus client (replaced by tests)."""
    return download.CopernicusClient()


def max_lag_months() -> int:
    """The longest lag a study tests by default (the specification's own default): the download starts early enough."""
    return AnalysisSpec().max_lag_months


# ── the inputs ───────────────────────────────────────────────────────────────────────────────────

def parse_coordinate(text: str | None, what: str, low: float, high: float) -> float:
    t = (text or "").strip()
    if not t:
        raise InputError(f"Enter the {what}.")
    if "," in t:
        raise InputError(f"The {what} {t!r} uses a comma: write it with a dot as the decimal separator (for example 43.43).")
    if not _DECIMAL.fullmatch(t):
        raise InputError(f"The {what} {t!r} is not a plain decimal number (digits, an optional sign and an optional dot, "
                         "for example 43.43).")
    value = float(t)
    if not math.isfinite(value) or not low <= value <= high:
        raise InputError(f"The {what} must be between {low:g} and {high:g}.")
    return value


def parse_inputs(lat, lon, name, base_start, base_end, first_year, coverage) -> tuple[SiteInputs | None, list[str], list[str]]:
    """(inputs, errors, warnings): everything wrong is reported at once, nothing is filled in for the user."""
    errors, warnings = [], []
    values = {}
    for what, text, lo, hi in (("latitude", lat, -90, 90), ("longitude", lon, -180, 180)):
        try:
            values[what] = parse_coordinate(text, what, lo, hi)
        except InputError as e:
            errors.append(str(e))
    name = (name or "").strip()
    if not name:
        errors.append("Enter a name for the site.")
    elif not _NAME.fullmatch(name):
        errors.append("The name of the site may use letters, digits, spaces and . , ' ( ) - only, up to 60 characters.")
    year = None
    text = (first_year or "").strip()
    if text:
        last = baseline.covered_years(coverage)[1]
        if not _YEAR.fullmatch(text) or not 1900 <= int(text) <= last:
            errors.append(f"The first year of your measurements {text!r} is not a four-digit year up to {last}: write it with "
                          "four digits, or leave it empty.")
        else:
            year = int(text)
    if base_start is None or base_end is None:
        errors.append("Enter both years of the baseline.")
    if "latitude" in values and "longitude" in values:
        variable = catalog.get(CATALOG_ID)
        if not catalog.covers(variable, values["latitude"], values["longitude"]):
            d = variable.domain
            errors.append(f"The site (latitude {values['latitude']}, longitude {values['longitude']}) is outside the domain of "
                          f"the sea-surface temperature product (latitude {d.lat_min:.2f} to {d.lat_max:.2f}, longitude "
                          f"{d.lon_min:.2f} to {d.lon_max:.2f}): this version covers the Mediterranean only.")
    if base_start is not None and base_end is not None:
        base_errors, base_warnings = baseline.check_baseline(int(base_start), int(base_end), coverage)
        errors += base_errors
        warnings += base_warnings
    if errors:
        return None, errors, warnings
    site = SiteSpec(id="site", lat=values["latitude"], lon=values["longitude"], name=name, bbox_delta=BBOX_DELTA)
    return SiteInputs(site, (int(base_start), int(base_end)), year), errors, warnings


# ── the download ─────────────────────────────────────────────────────────────────────────────────

def download_sst(ctx, inputs: SiteInputs, credentials: download.Credentials, client, coverage: download.Coverage) -> dict:
    """Download the series for `inputs` into the session folder and return what the view needs. Whatever an earlier
    download left in the folder is removed first: a failed download leaves nothing that could be mistaken for the new one."""
    shutil.rmtree(ctx.workdir / "inputs", ignore_errors=True)
    start, note = baseline.download_start(inputs.baseline[0], coverage, inputs.first_year, max_lag_months())
    t0 = time.perf_counter()
    result = download.fetch_daily_sst(inputs.site, start=start, credentials=credentials, client=client, catalog_id=CATALOG_ID)
    seconds = time.perf_counter() - t0
    result.manifest["timing"] = {"download_seconds": round(seconds, 1)}
    paths = download.write_result(result, ctx.workdir / "inputs")
    m = result.manifest
    return {"inputs_key": inputs.key, "site": m["site"], "baseline": inputs.baseline, "start_note": note,
            "n_days": m["n_days"], "missing_days": m["missing_days"]["count"],
            "start": m["period"]["obtained"]["start"], "end": m["period"]["obtained"]["end"],
            "seconds": round(seconds, 1), "csv": str(paths["csv"])}


def build_view(csv_path, baseline_years: tuple[int, int]) -> dict:
    """The heatwaves on the downloaded series, computed once per download (the short numeric work takes the lock)."""
    sst = pd.read_csv(csv_path, parse_dates=["Datetime"])
    try:
        with RERUN_LOCK:
            ds = dataset_builder.environment_dataset(sst, baseline_years)
    except dataset_builder.DatasetError as e:
        return {"error": str(e), "sst": sst}
    events = ds.mhw_events
    years = pd.to_datetime(events["start_date"]).dt.year if len(events) else pd.Series(dtype=int)
    in_base = int(years.between(*baseline_years).sum())
    return {"error": None, "sst": sst, "n_events": int(len(events)), "in_baseline": in_base, "after": int((years > baseline_years[1]).sum()),
            "top": events.sort_values("intensity_max", ascending=False).head(10)[
                ["start_date", "end_date", "duration_days", "intensity_max", "category"]] if len(events) else None,
            "excluded": ds.coverage["temperature"]["incomplete_months"]}


def _safe_message(error: Exception, credentials: download.Credentials) -> str:
    text = str(error) if isinstance(error, download.DownloadError) else f"Unexpected error ({type(error).__name__}): {error}"
    return download._mask(text, credentials)


def _on_submit(ctx) -> None:
    """The submit button's callback: take the credentials out of the session state at once, then validate, download and
    build the view; the outcome goes to `ctx.data` for the page to show."""
    user, password = st.session_state.get(USER_KEY, ""), st.session_state.get(PW_KEY, "")
    st.session_state[USER_KEY] = ""
    st.session_state[PW_KEY] = ""
    credentials = download.Credentials((user or "").strip(), (password or "").strip())
    del user, password
    step_run.reset(ctx)                                                         # and the analysis of it
    for key in ("step1", "step1_view", "step1_error", "step1_errors", "step2"):      # a new series: the study built on the old one goes
        ctx.data.pop(key, None)
    shutil.rmtree(ctx.workdir / "study", ignore_errors=True)
    coverage = ctx.data["coverage"]
    inputs, errors, _ = parse_inputs(st.session_state.get("s1_lat"), st.session_state.get("s1_lon"), st.session_state.get("s1_name"),
                                     st.session_state.get("s1_bl_start"), st.session_state.get("s1_bl_end"),
                                     st.session_state.get("s1_first_year"), coverage)
    if inputs is None:
        ctx.data["step1_errors"] = errors
        return
    message = None
    try:
        done = download_sst(ctx, inputs, credentials, make_client(), coverage)
        ctx.data["step1"] = done
        ctx.data["step1_view"] = build_view(done["csv"], done["baseline"])
    except Exception as e:
        shutil.rmtree(ctx.workdir / "inputs", ignore_errors=True)
        message = _safe_message(e, credentials)
    del credentials
    if message:
        ctx.data["step1_error"] = message


# ── the interface ────────────────────────────────────────────────────────────────────────────────

def _coverage(ctx):
    """The product's coverage, read from the Copernicus catalogue once per session (no credentials needed)."""
    if "coverage" in ctx.data:
        return ctx.data["coverage"]
    with st.spinner("Reading the coverage of the sea-surface temperature product from the Copernicus catalogue…"):
        message = None
        try:
            ctx.data["coverage"] = make_client().coverage_of(catalog.get(CATALOG_ID))
        except Exception as e:      # the catalogue is public: nothing secret in the message
            message = f"{type(e).__name__}: {e}"
    if message:
        st.error(f"The Copernicus catalogue could not be read ({message}).")
        st.button("Try again")
        return None
    return ctx.data["coverage"]


def _show_result(ctx, current_key) -> None:
    for e in ctx.data.get("step1_errors", []):
        st.error(e)
    if ctx.data.get("step1_error"):
        st.error(ctx.data["step1_error"])
    done, view = ctx.data.get("step1"), ctx.data.get("step1_view")
    if not done or not view:
        return
    if not (ctx.workdir / "inputs" / "sst_daily.csv").is_file():
        ctx.data.pop("step1", None); ctx.data.pop("step1_view", None)
        st.warning("The downloaded series is no longer there (the session folder is deleted after 6 hours without use): "
                   "download it again.")
        return
    if current_key != done["inputs_key"]:
        st.warning("The inputs changed since this download (or are not valid now): what is shown below is for the earlier inputs.")
    st.success(f"{done['n_days']} days of daily sea-surface temperature, {done['start']} to {done['end']}, downloaded in "
               f"{done['seconds']} seconds ({done['missing_days']} days without a value).")
    if done["start_note"]:
        st.info(done["start_note"])
    st.line_chart(view["sst"].set_index("Datetime")["Temperature"], height=220)
    if view["error"]:
        st.error(f"The heatwaves cannot be detected on this series: {view['error']}")
        return
    b0, b1 = done["baseline"]
    n = view["n_events"]
    st.write(f"{n} marine heatwave{'s' if n != 1 else ''} detected against the climatology of {b0}–{b1} "
             f"(the method of Hobday et al. 2016: the 90th percentile of the values within 5 days of each day of those years, "
             f"smoothed over 31 days; ADR-0019): {view['in_baseline']} in the baseline years and "
             f"{view['after']} after. The baseline years are, by construction, about one day in ten above the threshold; "
             "with a fixed baseline and a warming sea the later years show more heatwave days partly because of the trend "
             "(ADR-0014).")
    if view["top"] is not None:
        st.caption("The ten most intense events.")
        st.dataframe(view["top"], hide_index=True)
    excluded = view["excluded"]
    if excluded:
        st.caption(f"{len(excluded)} month(s) with a missing day have no monthly temperature: "
                   + ", ".join(m["month"] for m in excluded[:6]) + ("…" if len(excluded) > 6 else "") + ".")


def render(ctx) -> None:
    st.markdown("**Step 1 — Site, period and credentials**")
    if not acquisition_available():
        st.error("The download is not available on this server: the packages of the `acquisition` extra "
                 "(copernicusmarine, xarray, h5netcdf, h5py) are not installed.")
        return
    st.write("Your Copernicus Marine credentials are used for this download only: they are passed to the Copernicus "
             "toolbox for the calls of this session, never saved to a file, a cache or a log, and emptied from the page as "
             "soon as you press the button. Free accounts: marine.copernicus.eu.")
    coverage = _coverage(ctx)
    if coverage is None:
        return
    st.write(f"The sea-surface temperature product covers {coverage.start} to {coverage.end} (read from the Copernicus catalogue).")
    suggestion = baseline.suggest_baseline(coverage)
    if suggestion.note:
        st.info(suggestion.note)
    variable = catalog.get(CATALOG_ID)

    lat = st.text_input("Latitude (degrees north)", key="s1_lat", placeholder="e.g. 43.43")
    lon = st.text_input("Longitude (degrees east)", key="s1_lon", placeholder="e.g. 10.40")
    name = st.text_input("Name of the site", key="s1_name")
    c1, c2 = st.columns(2)
    base_start = c1.number_input("Baseline: first year", value=suggestion.start, step=1, key="s1_bl_start")
    base_end = c2.number_input("Baseline: last year", value=suggestion.end, step=1, key="s1_bl_end")
    st.caption("The baseline is the period whose temperatures define 'normal': the anomaly and the heatwaves are measured "
               "against it (ADR-0014).")
    first_year = st.text_input("First year of your measurements (optional)", key="s1_first_year",
                               help="If your measurements start before the baseline, the download starts earlier, so the lagged "
                                    "temperature of the first months is available.")
    st.caption(f"The temperature of the site is the mean over the sea cells within {BBOX_DELTA} degrees of it and over depth "
               f"{variable.depth_min_m} to {variable.depth_max_m} m (from the catalogue). The download starts early enough "
               f"for lags up to {max_lag_months()} months.")
    inputs, errors, warnings = parse_inputs(lat, lon, name, base_start, base_end, first_year, coverage)
    for w in warnings:
        st.warning(w)
    if inputs is not None:
        st.map(pd.DataFrame({"lat": [inputs.site.lat], "lon": [inputs.site.lon]}), zoom=6)

    with st.form("s1_credentials", clear_on_submit=True):
        st.text_input("Copernicus username", key=USER_KEY)
        st.text_input("Copernicus password", type="password", key=PW_KEY)
        st.form_submit_button("Download the sea-surface temperature", on_click=_on_submit, args=(ctx,))
    _show_result(ctx, inputs.key if inputs is not None else None)
