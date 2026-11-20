"""M1.7: every analysis declares what it needs, and a study is told which analyses its data allow, which are off and why.

The registry (analyses.py) is declarative; what keeps it honest is checked here against the code and the frozen results: every
module of the pipeline has an entry, every file of the golden master belongs to exactly one analysis, and what an entry says its
module reads (variables, heatwaves, individual determinations, controls, the daily SST) is what the module's source reads."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from climate_change_on_sea_urchins import analyses, catalog, pipeline
from climate_change_on_sea_urchins.analyses import ANALYSES, Needs, evaluate

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "climate_change_on_sea_urchins"
REFERENCE = ROOT / "tests" / "fixtures" / "results_v1_5_0"
BY_ID = {a.id: a for a in ANALYSES}
ALL_VARS = ("temperature", "salinity", "o2", "ph", "co2")
EVERYTHING = dict(listed={"sst_daily", "salinity_monthly", "o2_monthly", "ph_monthly", "co2_monthly"}, granularity="per_trial",
                  controls=True, split_date=True, contaminant="copper", r_available=True)


def _state(states, analysis_id):
    return next(s for s in states if s["id"] == analysis_id)


# --- the registry is complete and consistent --------------------------------------------------------------

def test_every_analysis_module_of_the_pipeline_has_an_entry_and_nothing_else_has_a_module():
    modules = {label for label, _ in pipeline._MODULES if label != "mhw_detection"}      # the heatwave catalogue is built by the dataset builder
    declared = {a.id for a in ANALYSES if a.module}
    assert declared == modules, (sorted(declared ^ modules))


def test_ids_are_unique_and_there_is_one_generic_analysis_the_command_runs_and_one_that_needs_r():
    ids = [a.id for a in ANALYSES]
    assert len(ids) == len(set(ids))
    assert [a.id for a in ANALYSES if a.in_command] == ["ccf_sst_anomaly"]
    assert [a.id for a in ANALYSES if a.needs.r] == ["dlnm"]


def test_every_file_of_the_golden_master_belongs_to_exactly_one_analysis():
    claimed = [f for a in ANALYSES for f in a.outputs]
    assert len(claimed) == len(set(claimed)), "a file claimed by two analyses"
    reference = {p.name for p in REFERENCE.iterdir()}
    own = {f for a in ANALYSES if a.module or a.id == "dlnm" for f in a.outputs}
    assert own == reference, (sorted(own ^ reference))


def test_a_generic_output_is_not_a_file_of_livornos_results():
    own = set(BY_ID["ccf_sst_anomaly"].outputs)
    assert own == {"ccf.csv"} and not own & {p.name for p in REFERENCE.iterdir()}


def _source(analysis):
    if analysis.id == "dlnm":
        return (ROOT / "scripts" / "mhw_lag_analysis.R").read_text()
    return (PKG / f"{analysis.module}.py").read_text()


def _string_constants(source):
    return {n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


@pytest.mark.parametrize("analysis", [a for a in ANALYSES if a.module], ids=lambda a: a.id)
def test_the_declared_variables_cover_the_environment_columns_the_module_names(analysis):
    named = _string_constants(_source(analysis)) & {"pH", "Salinity", "O2", "CO2"}
    wanted = {"pH": "ph", "Salinity": "salinity", "O2": "o2", "CO2": "co2"}
    assert {wanted[c] for c in named} <= set(analysis.needs.variables), (analysis.id, named)


@pytest.mark.parametrize("analysis", [a for a in ANALYSES if a.module], ids=lambda a: a.id)
def test_what_the_module_reads_is_declared_and_what_is_declared_is_read(analysis):
    src = _source(analysis)
    heat = "mhw_" in src or "MHW_COLS" in src or "load_mhw_annual" in src
    assert analysis.needs.heatwaves == heat, f"{analysis.id}: heatwaves declared {analysis.needs.heatwaves}, source reads them {heat}"
    assert analysis.needs.per_trial == ("load_ec50_raw" in src), analysis.id
    assert analysis.needs.daily_sst == ("load_sst_daily" in src), analysis.id
    assert analysis.needs.controls == ("REPLICA_COLS" in src), analysis.id


def test_the_variables_are_the_five_of_livornos_environment_and_come_from_the_catalogue_or_are_planned():
    assert set(analyses.VARIABLE_SOURCES) == set(ALL_VARS)
    for var, ids in analyses.VARIABLE_SOURCES.items():
        for i in ids:
            assert i in catalog.CATALOG or i in analyses.PLANNED_CATALOG_IDS, (var, i)
    assert analyses.VARIABLE_SOURCES == {"temperature": ("sst_daily",), "salinity": ("salinity_monthly",), "o2": ("o2_monthly",),
                                         "ph": ("ph_monthly",), "co2": ("co2_monthly",)}     # the monthly temperature is the monthly mean of the daily SST


# --- what a study is told -----------------------------------------------------------------------------------

def test_with_everything_available_only_the_generic_analysis_runs_and_the_others_are_pending_not_on():
    states = evaluate(**EVERYTHING)
    assert [s["id"] for s in states if s["status"] == "runs"] == ["ccf_sst_anomaly"]
    assert all(s["status"] == "pending" and s["reasons"] == [analyses.PENDING_REASON] for s in states if s["id"] != "ccf_sst_anomaly")
    assert [s["id"] for s in states] == [a.id for a in ANALYSES]


def test_a_study_without_ph_has_the_speciation_and_the_forecast_off_with_the_reason():
    listed = EVERYTHING["listed"] - {"ph_monthly"}
    states = evaluate(**{**EVERYTHING, "listed": listed})
    for analysis_id in ("cu_speciation", "forecast"):
        s = _state(states, analysis_id)
        assert s["status"] == "not_applicable" and any("pH" in r for r in s["reasons"]), s
    # the analyses that use the five variables need pH as well
    assert {s["id"] for s in states if s["status"] == "not_applicable"} >= {"cu_speciation", "forecast", "timeseries", "period_split", "correlations", "regime_shift"}


def test_the_reason_says_when_the_variable_is_not_in_the_catalogue_yet_and_says_so_no_more_when_it_is(monkeypatch):
    def ph_reason():
        reasons = _state(evaluate(**{**EVERYTHING, "listed": {"sst_daily"}}), "cu_speciation")["reasons"]
        return next(r for r in reasons if "pH" in r)
    why = ph_reason()
    assert "does not list" in why and "not in the variable catalogue yet" in why
    only_sst = " ".join(_state(evaluate(**{**EVERYTHING, "listed": {"sst_daily"}}), "cu_speciation")["reasons"])
    assert "salinity" in only_sst and "pH" in only_sst and "temperature" not in only_sst      # the SST provides the temperature and nothing else
    monkeypatch.setitem(catalog.CATALOG, "ph_monthly", catalog.CATALOG["sst_daily"])        # as if M1.8 had added it
    why = ph_reason()
    assert "does not list" in why and "not in the variable catalogue yet" not in why


def test_each_requirement_turns_its_analyses_off_with_its_own_reason():
    cases = [
        (dict(listed={"salinity_monthly", "o2_monthly", "ph_monthly", "co2_monthly"}), "thermal_legacy", "daily sea-surface temperature"),
        (dict(listed=set()), "mhw_analysis", "heatwave"),
        (dict(granularity="aggregated"), "changepoint", "individual determinations"),
        (dict(granularity=None), "negative_control", "individual determinations"),
        (dict(controls=False), "negative_control", "negative-control"),
        (dict(split_date=False), "period_split", "split date"),
        (dict(contaminant=None), "cu_speciation", "copper"),
        (dict(contaminant="cadmium"), "cu_speciation", "copper"),
        (dict(r_available=False), "dlnm", "Rscript"),
    ]
    for change, analysis_id, phrase in cases:
        s = _state(evaluate(**{**EVERYTHING, **change}), analysis_id)
        assert s["status"] == "not_applicable" and any(phrase in r for r in s["reasons"]), (change, s)


def test_every_unmet_need_is_listed_in_a_fixed_order_and_the_others_do_not_appear():
    s = _state(evaluate(listed={"sst_daily"}, granularity="aggregated", controls=False, split_date=False, contaminant=None, r_available=True), "negative_control")
    assert s["status"] == "not_applicable" and len(s["reasons"]) == 3
    assert ["individual determinations" in s["reasons"][0], "negative-control" in s["reasons"][1], "split date" in s["reasons"][2]] == [True] * 3


def test_a_reason_never_repeats_what_the_study_declares():
    s = _state(evaluate(**{**EVERYTHING, "contaminant": "<b>x</b>[l](http://e)"}), "cu_speciation")
    assert s["status"] == "not_applicable" and "<b>" not in json.dumps(s) and "http" not in json.dumps(s)


def test_the_generic_analysis_itself_is_off_when_the_study_has_no_daily_sst():
    s = _state(evaluate(**{**EVERYTHING, "listed": set()}), "ccf_sst_anomaly")
    assert s["status"] == "not_applicable" and any("daily sea-surface temperature" in r for r in s["reasons"])


def test_a_note_about_a_fixed_choice_travels_with_the_state():
    assert "24" in (_state(evaluate(**EVERYTHING), "thermal_legacy")["note"] or "")
    assert "fixed" in (_state(evaluate(**EVERYTHING), "forecast")["note"] or "")
    assert _state(evaluate(**EVERYTHING), "mhw_lag_annual")["note"] is None


def test_the_document_is_plain_data_with_the_outputs_and_no_path():
    doc = analyses.document(evaluate(**EVERYTHING), r_available=True)
    assert doc["format"] == 1 and doc["machine"] == {"r_available": True} and len(doc["analyses"]) == len(ANALYSES)
    assert all("/" not in f and ".." not in f for a in doc["analyses"] for f in a["outputs"])      # names, never paths
    row = next(a for a in doc["analyses"] if a["id"] == "cu_speciation")
    assert row["outputs"] == ["cu_speciation_decomposition.csv", "cu_speciation_summary.json"] and row["status"] == "pending"


def test_needs_are_data_that_cannot_be_extended_by_a_typo():
    with pytest.raises(Exception):
        Needs(varibles=("ph",))
    with pytest.raises(Exception):
        Needs(variables=("pH",))                              # the analysis names are lower case: a wrong name is an error, not a silent miss


# --- from a loaded study ---------------------------------------------------------------------------------------------

def _spec(*, granularity="per_trial", control_columns=None, split_date=None, contaminant=None, environment=("sst_daily",)):
    from climate_change_on_sea_urchins.study_spec import StudySpec
    source = {"type": "csv", "file": "response.csv", "temporal_resolution": "month", "granularity": granularity,
              "column_map": {"date": "date", "value": "value"}}
    if control_columns:
        source["control_columns"] = control_columns
    response = {"id": "r", "label": "Response", "nature": "index", "adverse_direction": "decrease", "unit": "1", "source": source,
                "aggregation": {"period": "month", "method": "mean", "count_field": "n"}}
    if split_date:
        response["split_date"] = split_date
    if contaminant:
        response["contaminant"] = {"name": contaminant}
    return StudySpec.model_validate({"format_version": 2, "id": "s", "description": "d",
                                     "mhw_climatology": {"baseline_start_year": 2003, "baseline_end_year": 2012},
                                     "sites": [{"id": "a", "lat": 43.4, "lon": 10.4, "name": "A", "bbox_delta": 0.1}],
                                     "environment": [{"catalog_id": c} for c in environment], "responses": [response]})


def _off(states):
    return {s["id"] for s in states if s["status"] == "not_applicable"}


def test_the_contaminant_is_compared_without_regard_to_case_or_spaces():
    for name in ("Copper", " COPPER ", "copper"):
        assert _state(evaluate(**{**EVERYTHING, "contaminant": name}), "cu_speciation")["status"] == "pending", name


def test_a_loaded_study_gives_the_same_states_as_the_values_it_holds():
    base = analyses.evaluate_study(_spec(), r_available=True)
    assert {"cu_speciation", "negative_control", "period_split", "forecast"} <= _off(base)
    assert "changepoint" not in _off(base)                                           # per-trial, the daily SST: those are there
    full = analyses.evaluate_study(_spec(control_columns=["c1", "c2", "c3"], split_date="2016-06-01", contaminant="copper"), r_available=True)
    assert "negative_control" not in _off(full)                                      # per-trial, controls and a split date: all declared
    for analysis_id in ("cu_speciation", "period_split"):
        why = " ".join(_state(full, analysis_id)["reasons"])
        assert "pH" in why and "split date" not in why and "individual determinations" not in why and "copper" not in why     # only the variables are missing


def test_each_declaration_of_a_loaded_study_matters():
    assert "changepoint" in _off(analyses.evaluate_study(_spec(granularity="aggregated"), r_available=True))
    assert "negative_control" in _off(analyses.evaluate_study(_spec(split_date="2016-06-01"), r_available=True))          # no controls
    assert "negative_control" in _off(analyses.evaluate_study(_spec(control_columns=["c1"]), r_available=True))           # no split date
    assert "period_split" in _off(analyses.evaluate_study(_spec(), r_available=True))                                     # no split date
    assert "cu_speciation" in _off(analyses.evaluate_study(_spec(contaminant="cadmium"), r_available=True))
    assert "dlnm" in _off(analyses.evaluate_study(_spec(), r_available=False)) and "dlnm" not in _off(analyses.evaluate_study(_spec(), r_available=True))


# --- the declarations against the evidence: the runs that took one input away, and the sources -------------------------

# What each module cannot do without, found on 2026-11-20 by taking one input at a time from the frozen paper fixture (a column of the
# monthly table, the individual determinations, the control columns, the daily SST) and running the module; the temperature is the
# module's own use of the column (code reading), because every module that goes through load_data fails without it, used or not.
# (variables, per_trial, controls, daily_sst); every other analysis needs none of the four.
EVIDENCE = {
    "timeseries": (ALL_VARS, False, False, False),
    "period_split": (ALL_VARS, True, False, False),
    "correlations": (ALL_VARS, False, False, False),
    "cu_speciation": (("temperature", "salinity", "ph"), False, False, False),
    "thermal_legacy": ((), False, False, True),
    "regime_shift": (ALL_VARS, False, False, False),
    "changepoint": ((), True, False, False),
    "negative_control": ((), True, True, False),
    "forecast": (("temperature", "ph"), False, False, False),
    "mhw_robustness": (("temperature",), False, False, False),
}


@pytest.mark.parametrize("analysis", [a for a in ANALYSES if a.module], ids=lambda a: a.id)
def test_the_registry_says_what_the_runs_found_the_module_cannot_do_without(analysis):
    variables, per_trial, controls, daily = EVIDENCE.get(analysis.id, ((), False, False, False))
    n = analysis.needs
    assert (set(n.variables), n.per_trial, n.controls, n.daily_sst) == (set(variables), per_trial, controls, daily), analysis.id


def _split_uses(source):
    tree = ast.parse(source)
    return [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in {"SPLIT_DATE", "SPLIT_YEAR"} and isinstance(n.ctx, ast.Load)]


@pytest.mark.parametrize("analysis", [a for a in ANALYSES if a.module], ids=lambda a: a.id)
def test_a_module_that_uses_the_split_date_declares_it_or_says_in_a_note_that_part_of_it_does(analysis):
    uses = bool(_split_uses(_source(analysis)))
    declared = analysis.needs.split_date or "split date" in (analysis.note or "")
    assert uses == declared, f"{analysis.id}: uses the split date {uses}, declares it {declared}"


def _fixed_splits(source):
    out = []
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "TimeSeriesSplit":
            kw = {k.arg: k.value.value for k in n.keywords if isinstance(k.value, ast.Constant)}
            if isinstance(kw.get("n_splits"), int) and isinstance(kw.get("test_size"), int):
                out.append(kw["n_splits"] * kw["test_size"] + 1)
    return out


@pytest.mark.parametrize("analysis", [a for a in ANALYSES if a.module], ids=lambda a: a.id)
def test_the_months_below_which_a_module_stops_are_declared(analysis):
    fixed = _fixed_splits(_source(analysis))                  # a cross-validation with a fixed number of splits of a fixed size needs this many
    assert analysis.needs.min_months == (max(fixed) if fixed else None), analysis.id


def test_a_study_with_too_few_months_has_the_analysis_off_with_the_numbers_and_one_with_enough_does_not():
    short = _state(evaluate(**{**EVERYTHING, "months": 220}), "timeseries")
    assert short["status"] == "not_applicable" and any("221 months" in r and "study has 220" in r for r in short["reasons"])
    assert _state(evaluate(**{**EVERYTHING, "months": 221}), "timeseries")["status"] == "pending"
    assert _state(evaluate(**EVERYTHING), "timeseries")["status"] == "pending"            # unknown: not checked
    assert _state(evaluate(**{**EVERYTHING, "months": 10}), "correlations")["status"] == "pending"


def test_the_r_script_is_declared_as_it_reads_the_heatwaves_and_nothing_else():
    src = (ROOT / "scripts" / "mhw_lag_analysis.R").read_text()
    n = BY_ID["dlnm"].needs
    assert n.heatwaves == ("mhw_" in src) and n.per_trial == ("ec50_raw" in src) and n.daily_sst == ("sst_daily" in src)
    assert n.controls == ("ctrl" in src) and not n.variables and n.r


def test_every_variable_the_catalogue_has_is_named_in_the_registry():
    assert set(catalog.CATALOG) <= {i for ids in analyses.VARIABLE_SOURCES.values() for i in ids}, "a catalogue entry the registry does not know"


def test_the_notes_say_the_fixed_choices_the_review_found():
    notes = {a.id: a.note or "" for a in ANALYSES}
    for analysis_id in ("mhw_analysis", "mhw_robustness", "forecast"):
        assert "lags go to 12" in notes[analysis_id], analysis_id
    assert "June to August" in notes["mhw_robustness"] and "fixed alkalinity" in notes["cu_speciation"]
    assert "ctrl_neg_rep1" in notes["negative_control"] and "ID column" in notes["changepoint"] and "ID column" in notes["period_split"]


def test_a_study_of_another_format_is_refused_not_guessed():
    from climate_change_on_sea_urchins.study_spec import load_study
    with pytest.raises(ValueError):
        analyses.evaluate_study(load_study(ROOT / "examples" / "livorno_paracentrotus" / "study.yaml"), r_available=True)


def test_every_listed_variable_counts_and_the_cas_of_copper_is_enough(monkeypatch):
    monkeypatch.setitem(catalog.CATALOG, "ph_monthly", catalog.CATALOG["sst_daily"])        # as if M1.8 had added it
    assert "cu_speciation" in _off(analyses.evaluate_study(_spec(environment=("sst_daily",)), r_available=True))
    assert "thermal_legacy" not in _off(analyses.evaluate_study(_spec(environment=("sst_daily",)), r_available=True))
    two = analyses.evaluate_study(_spec(environment=("ph_monthly", "sst_daily")), r_available=True)        # the SST is not the first entry
    assert "thermal_legacy" not in _off(two) and not any("pH" in r for r in _state(two, "forecast")["reasons"])
    cas = analyses.evaluate_study(_spec(contaminant="cobre"), r_available=True)
    from climate_change_on_sea_urchins.study_spec import StudySpec
    spec = _spec()
    data = spec.model_dump(); data["responses"][0]["contaminant"] = {"name": "cobre", "cas": "7440-50-8"}
    with_cas = analyses.evaluate_study(StudySpec.model_validate(data), r_available=True)
    assert "cu_speciation" in _off(cas) and not any("copper" in r for r in _state(with_cas, "cu_speciation")["reasons"])


def test_the_analyses_that_need_a_split_date_are_these_three():
    assert {a.id for a in ANALYSES if a.needs.split_date} == {"period_split", "cu_speciation", "negative_control"}
