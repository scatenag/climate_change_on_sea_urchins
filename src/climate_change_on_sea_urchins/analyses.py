"""What each analysis needs, and which analyses a study allows (M1.7, ADR-0020).

A declarative registry: for every analysis of the pipeline (and the generic correlation of `ccsu-run-study`) the variables it
reads, whether it needs the daily sea-surface temperature, the heatwave catalogue, the individual determinations of the response,
the negative controls, a split date, a named contaminant, or R; the files it writes; and, where the code fixes a choice that a
study should declare, a note. `evaluate` turns the registry and what a study provides into a state per analysis:

  runs            the command runs it in this version (only the generic correlation);
  not_applicable  something it needs is missing from the study, with the reason(s);
  pending         the study has what it needs but the command does not run it yet (milestone M1.9).

The needs are those of the implementations of today, found by taking one input at a time away from the frozen fixture and running
each module (150 runs, ADR-0020) and kept in step with the sources by tests/test_analyses.py. They are revised per analysis when
M1.9 moves it into the command. Nothing here imports an analysis, reads a study loaded by `common`, or writes a file: the module
is names and rules, so the app can import it.

The monthly temperature is the monthly mean of the daily SST that the dataset builder makes: `sst_daily` provides it. The other
variables of Livorno's environment (salinity, oxygen, pH, CO2) are monthly variables the catalogue does not have yet (M1.8): their
ids are the planned ones, and a reason says so while they are not in the catalogue.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, field_validator

from . import catalog

VARIABLES = ("temperature", "salinity", "o2", "ph", "co2")
LABELS = {"temperature": "temperature", "salinity": "salinity", "o2": "dissolved oxygen", "ph": "pH", "co2": "CO2"}
# the catalogue ids that provide each variable (several would be alternatives)
VARIABLE_SOURCES = {"temperature": ("sst_daily",), "salinity": ("salinity_monthly",), "o2": ("o2_monthly",),
                    "ph": ("ph_monthly",), "co2": ("co2_monthly",)}
PLANNED_CATALOG_IDS = frozenset({"salinity_monthly", "o2_monthly", "ph_monthly", "co2_monthly"})     # M1.8 adds them
PENDING_REASON = "this version of the tool does not run it yet"
SST_REASON_DAILY = "it needs the daily sea-surface temperature (sst_daily), and the study does not list it"
SST_REASON_HEAT = ("it needs the heatwave catalogue, which is built from the daily sea-surface temperature (sst_daily), "
                   "and the study does not list it")


class Needs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    variables: tuple[str, ...] = ()
    daily_sst: bool = False
    heatwaves: bool = False
    per_trial: bool = False         # the individual determinations, not only a monthly value
    controls: bool = False          # the negative-control columns
    min_months: int | None = None   # months with a response and a temperature below which the analysis stops
    split_date: bool = False
    contaminant: str | None = None
    r: bool = False

    @field_validator("variables")
    @classmethod
    def _known(cls, v):
        bad = [x for x in v if x not in VARIABLES]
        if bad:
            raise ValueError(f"unknown variable(s) {bad}; the names are {list(VARIABLES)}")
        return v


@dataclass(frozen=True)
class Analysis:
    id: str
    label: str
    needs: Needs
    outputs: tuple[str, ...]
    module: str | None = None       # the module of the pipeline; None for the generic correlation and for the R script
    in_command: bool = False
    note: str | None = None


NOTE_ID = "it orders the individual determinations on their date and an ID column, which the study format has no field for yet"
NOTE_LAGS = "its lags go to 12, fixed in the code, not the study's maximum lag"

_ALL = VARIABLES
ANALYSES: tuple[Analysis, ...] = (
    Analysis("ccf_sst_anomaly", "Lagged correlation of the SST anomaly with the response (first differences)",
             Needs(daily_sst=True), ("ccf.csv",), in_command=True),
    Analysis("timeseries", "Trend, seasonal part and PCA anomaly of the series", Needs(variables=_ALL, min_months=221),
             ("pca_anomaly.csv", "trend_CO2.csv", "trend_EC50.csv", "trend_O2.csv", "trend_Salinity.csv", "trend_Temperature.csv",
              "trend_pH.csv", "trends_all.csv", "trends_post.csv", "trends_pre.csv"), module="timeseries",
             note="its cross-validated PCA uses 20 splits of 11 months, so 221 months; its before/after outputs need the split date"),
    Analysis("period_split", "Before/after contrast of each variable and of the individual determinations",
             Needs(variables=_ALL, heatwaves=True, per_trial=True, split_date=True),
             ("dist_CO2.csv", "dist_EC50.csv", "dist_O2.csv", "dist_Salinity.csv", "dist_Temperature.csv", "dist_pH.csv",
              "kruskal_stats.json", "period_contrast_raw.json", "period_means.csv"), module="period_split", note=NOTE_ID),
    Analysis("correlations", "Spearman correlation matrices of the trends, all / before / after", Needs(variables=_ALL, heatwaves=True),
             ("corr_all.csv", "corr_post.csv", "corr_pre.csv", "corr_pval_all.csv", "corr_pval_post.csv", "corr_pval_pre.csv"),
             module="correlations", note="its before/after matrices need the split date"),
    Analysis("stationarity", "ADF and KPSS stationarity tests", Needs(heatwaves=True),
             ("stationarity_results.json",), module="stationarity", note="it is applied to the variables and heatwave metrics the study provides"),
    Analysis("mhw_analysis", "Heatwave metric against each variable: lagged correlation, Granger causality, ARDL",
             Needs(heatwaves=True),
             ("ardl_response.csv", "ccf_results.csv", "ccf_results_diff.csv", "ccf_results_prewhitened.csv", "granger_results.json",
              "prewhitening_diagnostics.json"), module="mhw_analysis",
             note="it is applied to the variables the study provides; " + NOTE_LAGS),
    Analysis("mhw_lag_extra", "Superposed epoch analysis and mixed-effects model on the heatwave events", Needs(heatwaves=True),
             ("mixed_effects_predictions.csv", "mixed_effects_summary.json", "sea_results.csv"), module="mhw_lag_extra"),
    Analysis("mhw_robustness", "Robustness battery for the heatwave to response lag", Needs(variables=("temperature",), heatwaves=True),
             ("robustness_ccm.csv", "robustness_ml_cv_r2.json", "robustness_ml_importance.csv", "robustness_severe_ccf.csv",
              "robustness_severe_ccf_note.json", "robustness_summer_temp.csv", "robustness_wavelet.json"), module="mhw_robustness",
             note=NOTE_LAGS + "; its summer is June to August (northern hemisphere); its severe-event arm reports how many events fall after the split date"),
    Analysis("cu_speciation", "Copper speciation: share of the decline that a change of pH can explain",
             Needs(variables=("ph", "salinity", "temperature"), split_date=True, contaminant="copper"),
             ("cu_speciation_decomposition.csv", "cu_speciation_summary.json"), module="cu_speciation",
             note="it uses a fixed alkalinity of the Mediterranean, and does not read the response's nature, adverse direction or unit"),
    Analysis("thermal_legacy", "Cumulative thermal dose above a stress threshold", Needs(daily_sst=True),
             ("thermal_legacy.csv", "thermal_legacy_summary.json", "thermal_threshold_sensitivity.csv"), module="thermal_legacy",
             note="its stress threshold (24 °C) is fixed in the code, chosen for one species; a study would have to declare its own"),
    Analysis("regime_shift", "Environmental stress index and its change points", Needs(variables=_ALL, heatwaves=True),
             ("regime_shift_changepoints.csv", "regime_shift_stress_index.csv", "regime_shift_summary.json"), module="regime_shift",
             note="it reads the split date (a year) but does not use it"),
    Analysis("changepoint", "Change point of the response (autocorrelation-aware QLR)", Needs(per_trial=True),
             ("changepoint_response.json",), module="changepoint", note=NOTE_ID),
    Analysis("negative_control", "Stability of the negative controls", Needs(per_trial=True, controls=True, split_date=True),
             ("negative_control.json",), module="negative_control",
             note="it reads three replicate columns with fixed names (ctrl_neg_rep1 to 3) and also reports the contrast at 2016-01-01, a date of the case; " + NOTE_ID),
    Analysis("mhw_lag_annual", "Annual heatwave metrics against the response, by lag in years", Needs(heatwaves=True),
             ("mhw_lag_annual.csv", "mhw_lag_annual_summary.json"), module="mhw_lag_annual"),
    Analysis("mhw_annual_changepoint", "Change point of the annual heatwave series", Needs(heatwaves=True),
             ("mhw_annual_changepoint.json",), module="mhw_annual_changepoint"),
    Analysis("forecast", "Forecast of the response under environmental scenarios", Needs(variables=("ph", "temperature"), heatwaves=True),
             ("forecast_bad.csv", "forecast_env_bad.csv", "forecast_env_good.csv", "forecast_env_mean.csv", "forecast_good.csv",
              "forecast_mean.csv", "forecast_meta.json"), module="forecast",
             note="its scenarios and the years it is trained on are fixed in the code; " + NOTE_LAGS),
    Analysis("dlnm", "Distributed-lag non-linear model of the heatwave exposure (R)", Needs(heatwaves=True, r=True),
             ("dlnm_lag_profile.csv", "dlnm_results.csv", "dlnm_slice_lag.csv")),
)


def _unmet(needs: Needs, *, listed, months, granularity, controls, split_date, contaminant, r_available) -> list[str]:
    """The reasons, in a fixed order: variables, daily SST, heatwaves, months, individual determinations, controls, split date,
    contaminant, R. Never a word the study wrote."""
    reasons = []
    for var in needs.variables:
        sources = VARIABLE_SOURCES[var]
        if not any(s in listed for s in sources):
            text = f"it needs {LABELS[var]}, and the study does not list it"
            if not any(s in catalog.CATALOG for s in sources):
                text += " (it is not in the variable catalogue yet)"
            reasons.append(text)
    if "sst_daily" not in listed:
        if needs.daily_sst:
            reasons.append(SST_REASON_DAILY)
        elif needs.heatwaves:
            reasons.append(SST_REASON_HEAT)
    if needs.min_months and months is not None and months < needs.min_months:
        reasons.append(f"it needs at least {needs.min_months} months with a response and a temperature, and the study has {months}")
    if needs.per_trial and granularity != "per_trial":
        reasons.append("it needs the individual determinations of the response, and the study gives one value per period")
    if needs.controls and not controls:
        reasons.append("it needs the negative-control columns of the response, and the study declares none")
    if needs.split_date and not split_date:
        reasons.append("it needs a split date, and the study declares none")
    if needs.contaminant and (contaminant or "").strip().lower() != needs.contaminant:
        reasons.append(f"it needs the contaminant to be {needs.contaminant}, and the study does not declare {needs.contaminant} as its contaminant")
    if needs.r and not r_available:
        reasons.append("it needs R (Rscript), which is not available here")
    return reasons


def evaluate(*, listed, granularity, controls, split_date, contaminant, r_available, months=None) -> list[dict]:
    """One state per analysis, in the order of the registry. `listed`: the catalogue ids the study lists; `granularity`: what a
    row of the response holds ("per_trial", "aggregated" or None); `controls`, `split_date`: whether the study declares them;
    `contaminant`: its name or None; `r_available`: whether Rscript can run; `months`: months with a response and a temperature
    (None when not known: the minimum is then not checked)."""
    out = []
    for a in ANALYSES:
        reasons = _unmet(a.needs, listed=set(listed), months=months, granularity=granularity, controls=controls, split_date=split_date,
                         contaminant=contaminant, r_available=r_available)
        if reasons:
            status = "not_applicable"
        elif a.in_command:
            status, reasons = "runs", []
        else:
            status, reasons = "pending", [PENDING_REASON]
        out.append({"id": a.id, "label": a.label, "status": status, "reasons": reasons, "note": a.note, "outputs": list(a.outputs)})
    return out


COPPER_CAS = "7440-50-8"


def evaluate_study(spec, *, r_available: bool, months: int | None = None) -> list[dict]:
    """`evaluate` from a loaded format 2 study (one response); other formats are refused, not guessed."""
    if spec.format_version != 2:
        raise ValueError("the analyses are evaluated for format 2 studies only")
    response = spec.responses[0]
    source = response.source
    contaminant = None
    if response.contaminant:
        contaminant = "copper" if response.contaminant.cas == COPPER_CAS else response.contaminant.name
    return evaluate(
        listed={e.catalog_id for e in spec.environment}, months=months,
        granularity=getattr(source, "granularity", None),
        controls=bool(getattr(source, "control_columns", None)),
        split_date=response.split_date is not None,
        contaminant=contaminant, r_available=r_available)


def document(states: list[dict], *, r_available: bool) -> dict:
    """What `ccsu-run-study` writes as analyses.json. `r_available` is a fact about the machine that ran it, kept apart."""
    return {"format": 1, "machine": {"r_available": r_available}, "analyses": states}
