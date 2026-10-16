"""Step 1 of the user's own data: site, period and credentials, and the download of the daily sea-surface temperature
(milestone MA.2, docs/roadmap/MILESTONE-M1.md; baseline: ADR-0014).

Only functions: the interface runs in `render(ctx)` at every run, and what the step produces lives in the session's
context (`ctx.data`), never in the module.

Credentials. They are the user's, typed in a form that clears on submit, in a password field that has no state key, and
they go to `download.fetch_daily_sst` as arguments and nowhere else: not to a file, not to the context, not to a cache
(no st.cache_* here), not to a log or an environment variable, not into an error message (every message goes through
the masking of `download`). The toolbox's own sign-in, which stores them in a file, is never used. The local variables
that hold them are deleted as soon as the call returns. Streamlit keeps the last value of a widget in the session's
memory until the next interaction clears it: that is the one place they sit, in memory, for the length of one run.

The download runs in this process, in the script thread of the session, and does not hold the global lock (it lasts
tens of seconds). The short numeric work of the view (heatwave detection) takes the lock for its own duration only."""
from __future__ import annotations

import importlib.util
import math
import time
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from . import RERUN_LOCK, baseline, catalog, dataset_builder, download
from .study_spec import SiteSpec

CATALOG_ID = "sst_daily"
EXTRA_MODULES = ("copernicusmarine", "xarray", "h5netcdf", "h5py")


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


# ── the inputs ───────────────────────────────────────────────────────────────────────────────────

def parse_coordinate(text: str | None, what: str, low: float, high: float) -> float:
    t = (text or "").strip()
    if not t:
        raise InputError(f"Enter the {what}.")
    if "," in t:
        raise InputError(f"The {what} {t!r} uses a comma: write it with a dot as the decimal separator (for example 43.43).")
    try:
        value = float(t)
    except ValueError:
        raise InputError(f"The {what} {t!r} is not a number.") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise InputError(f"The {what} must be between {low:g} and {high:g}.")
    return value


def parse_inputs(lat: str, lon: str, name: str, base_start: int, base_end: int, first_year: str,
                 coverage: download.Coverage) -> tuple[SiteInputs | None, list[str], list[str]]:
    """(inputs, errors, warnings): everything wrong is reported at once, nothing is filled in for the user."""
    errors, warnings = [], []
    values = {}
    for what, text, lo, hi in (("latitude", lat, -90, 90), ("longitude", lon, -180, 180)):
        try:
            values[what] = parse_coordinate(text, what, lo, hi)
        except InputError as e:
            errors.append(str(e))
    if not (name or "").strip():
        errors.append("Enter a name for the site.")
    year = None
    if (first_year or "").strip():
        try:
            year = int(first_year.strip())
            if year <= 0:
                raise ValueError
        except ValueError:
            errors.append(f"The first year of your measurements {first_year.strip()!r} is not a year: write it with four digits, or leave it empty.")
    if "latitude" in values and "longitude" in values:
        variable = catalog.get(CATALOG_ID)
        if not catalog.covers(variable, values["latitude"], values["longitude"]):
            d = variable.domain
            errors.append(f"The site (latitude {values['latitude']}, longitude {values['longitude']}) is outside the domain of "
                          f"the sea-surface temperature product (latitude {d.lat_min:.2f} to {d.lat_max:.2f}, longitude "
                          f"{d.lon_min:.2f} to {d.lon_max:.2f}): this version covers the Mediterranean only.")
    base_errors, base_warnings = baseline.check_baseline(int(base_start), int(base_end), coverage)
    errors += base_errors
    warnings += base_warnings
    if errors:
        return None, errors, warnings
    site = SiteSpec(id="site", lat=values["latitude"], lon=values["longitude"], name=name.strip(), bbox_delta=0.1)
    return SiteInputs(site, (int(base_start), int(base_end)), year), errors, warnings


# ── the download ─────────────────────────────────────────────────────────────────────────────────

def download_sst(ctx, inputs: SiteInputs, credentials: download.Credentials, client, coverage: download.Coverage) -> dict:
    """Download the series for `inputs` into the session folder and return what the view needs."""
    start, note = baseline.download_start(inputs.baseline[0], coverage, inputs.first_year)
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


def _safe_message(error: Exception, credentials: download.Credentials) -> str:
    text = str(error) if isinstance(error, download.DownloadError) else f"Unexpected error ({type(error).__name__}): {error}"
    return download._mask(text, credentials)


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
    done = ctx.data.get("step1")
    if not done:
        return
    if current_key is not None and current_key != done["inputs_key"]:
        st.warning("The inputs changed since this download: what is shown below is for the earlier inputs.")
    st.success(f"{done['n_days']} days of daily sea-surface temperature, {done['start']} to {done['end']}, downloaded in "
               f"{done['seconds']} seconds ({done['missing_days']} days without a value).")
    if done["start_note"]:
        st.info(done["start_note"])
    sst = pd.read_csv(done["csv"], parse_dates=["Datetime"])
    st.line_chart(sst.set_index("Datetime")["Temperature"], height=220)
    try:
        with RERUN_LOCK:
            ds = dataset_builder.environment_dataset(sst, tuple(done["baseline"]))
    except dataset_builder.DatasetError as e:
        st.error(f"The heatwaves cannot be detected on this series: {e}")
        return
    events = ds.mhw_events
    b0, b1 = done["baseline"]
    st.write(f"{len(events)} marine heatwave{'s' if len(events) != 1 else ''} detected against the climatology of {b0}–{b1} "
             f"(daily mean and 90th percentile of those years).")
    if len(events):
        top = events.sort_values("intensity_max", ascending=False).head(10)
        st.caption("The ten most intense events.")
        st.dataframe(top[["start_date", "end_date", "duration_days", "intensity_max", "category"]], hide_index=True)
    excluded = ds.coverage["temperature"]["incomplete_months"]
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
             "toolbox for the calls of this session, never saved to a file, a cache or a log, and not kept after the call. "
             "Free accounts: marine.copernicus.eu.")
    coverage = _coverage(ctx)
    if coverage is None:
        return
    st.write(f"The sea-surface temperature product covers {coverage.start} to {coverage.end} (read from the Copernicus catalogue).")
    suggestion = baseline.suggest_baseline(coverage)
    if suggestion.note:
        st.info(suggestion.note)

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
    inputs, errors, warnings = parse_inputs(lat, lon, name, base_start, base_end, first_year, coverage)
    for w in warnings:
        st.warning(w)
    if inputs is not None:
        st.map(pd.DataFrame({"lat": [inputs.site.lat], "lon": [inputs.site.lon]}), zoom=6)

    with st.form("s1_credentials", clear_on_submit=True):
        user = st.text_input("Copernicus username")
        password = st.text_input("Copernicus password", type="password")
        submitted = st.form_submit_button("Download the sea-surface temperature")
    if submitted:
        if inputs is None:
            for e in errors:
                st.error(e)
        else:
            credentials = download.Credentials(user, password)
            del user, password
            with st.status("Downloading the sea-surface temperature from Copernicus Marine…", expanded=True) as status:
                try:
                    ctx.data["step1"] = download_sst(ctx, inputs, credentials, make_client(), coverage)
                    status.update(label="Download finished", state="complete")
                except Exception as e:
                    ctx.data.pop("step1", None)
                    status.update(label="The download did not complete", state="error")
                    message = _safe_message(e, credentials)
                else:
                    message = None
            del credentials
            if message:
                st.error(message)
    _show_result(ctx, inputs.key if inputs is not None else None)
