"""Study specification, format 2 (M1.1): format version, environment by catalogue id, a CSV
response source, optional split_date, declared imputation (absent by default), declared contaminant.

Format 1 (the Livorno study.yaml, no format_version key) keeps loading unchanged. A format-2 study
loads and validates but is not run by the pipeline yet (that comes with ccsu-run-study, M1.5): the
guard in common.py says so explicitly instead of failing somewhere downstream.
"""
import textwrap
from pathlib import Path

import pytest
import yaml

from climate_change_on_sea_urchins.study_spec import StudySpecError, load_study

EXAMPLE = Path(__file__).parent.parent / "examples" / "livorno_paracentrotus" / "study.yaml"

FORMAT2 = textwrap.dedent("""\
    format_version: 2
    id: synthetic-study
    description: A study made of a CSV and the daily SST.
    mhw_climatology: {baseline_start_year: 2003, baseline_end_year: 2012}
    sites:
      - {id: s1, lat: 43.43, lon: 10.40, name: Somewhere, bbox_delta: 0.1}
    environment:
      - {catalog_id: sst_daily}
    responses:
      - id: r1
        label: Response
        nature: index
        adverse_direction: decrease
        unit: "1"
        source:
          type: csv
          file: response.csv
          temporal_resolution: month
          granularity: per_trial
          column_map: {date: date, value: value}
        aggregation: {period: month, method: mean, count_field: n_assays}
    """)


def _write(tmp_path, text):
    p = tmp_path / "study.yaml"
    p.write_text(text)
    return p


def _livorno_variant(tmp_path, mutate):
    """The Livorno study with `mutate` applied, written outside examples/ (data_dir made absolute)."""
    d = yaml.safe_load(EXAMPLE.read_text())
    d["data_dir"] = str(EXAMPLE.parent.parent.parent / "data")
    mutate(d)
    p = tmp_path / "study.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


def _modified(mutate):
    d = yaml.safe_load(FORMAT2)
    mutate(d)
    return yaml.safe_dump(d, sort_keys=False)


# --- format version -------------------------------------------------------------------------

def test_livorno_without_a_version_is_format_1():
    assert load_study(EXAMPLE).format_version == 1


def test_format_2_loads(tmp_path):
    spec = load_study(_write(tmp_path, FORMAT2))
    assert spec.format_version == 2
    assert spec.data_dir is None
    assert spec.responses[0].split_date is None


@pytest.mark.parametrize("version", [0, 3, 99])
def test_unsupported_version_is_refused_naming_it(tmp_path, version):
    text = FORMAT2.replace("format_version: 2", f"format_version: {version}")
    with pytest.raises(StudySpecError, match=rf"format_version {version}.*not supported"):
        load_study(_write(tmp_path, text))


def test_unsupported_version_message_lists_the_supported_ones(tmp_path):
    text = FORMAT2.replace("format_version: 2", "format_version: 7")
    with pytest.raises(StudySpecError, match=r"1.*2"):
        load_study(_write(tmp_path, text))


# Unknown fields are refused at EVERY level of the specification, not only at the root: a typo
# where a user would write one ("imputaton", "split_dat") must not silently switch the choice off.
# Each case: (where, how to introduce the typo into the response / site / aggregation).

def _typo_in_response(name, value="x"):
    return lambda d: d["responses"][0].__setitem__(name, value)

def _typo_in_site(d):
    d["sites"][0]["lattitude"] = 43.0

def _typo_in_aggregation(d):
    d["responses"][0]["aggregation"]["metod"] = "mean"

TYPOS = [
    pytest.param(_typo_in_response("split_dat", "2016-06-01"), "split_dat", id="split_dat-in-response"),
    pytest.param(_typo_in_response("imputaton", {"method": "centered_rolling_mean", "window_months": 12,
                                                  "min_periods": 3, "passes": 1}), "imputaton", id="imputaton-in-response"),
    pytest.param(_typo_in_site, "lattitude", id="field-in-site"),
    pytest.param(_typo_in_aggregation, "metod", id="field-in-aggregation"),
]


def test_unknown_top_level_field_is_rejected(tmp_path):
    text = FORMAT2 + "split_dat: 2016-06-01\n"
    with pytest.raises(StudySpecError, match="split_dat"):
        load_study(_write(tmp_path, text))


@pytest.mark.parametrize("mutate,field", TYPOS)
def test_nested_typo_is_rejected_in_format_2(tmp_path, mutate, field):
    with pytest.raises(StudySpecError, match=field):
        load_study(_write(tmp_path, _modified(mutate)))


@pytest.mark.parametrize("mutate,field", TYPOS)
def test_nested_typo_is_rejected_in_format_1(tmp_path, mutate, field):
    with pytest.raises(StudySpecError, match=field):
        load_study(_livorno_variant(tmp_path, mutate))


@pytest.mark.parametrize("path,field", [
    (("responses", 0, "source"), "sheet_idd"),
    (("responses", 0, "source", "column_map"), "ci_lo"),
    (("mhw_climatology",), "baseline_start"),
    (("environment", 0), "datasett"),
])
def test_typos_in_other_format_1_blocks_are_rejected(tmp_path, path, field):
    def add(d):
        node = d
        for key in path:
            node = node[key]
        node[field] = "x"
    with pytest.raises(StudySpecError, match=field):
        load_study(_livorno_variant(tmp_path, add))


def test_typo_in_csv_source_and_column_map_is_rejected_in_format_2(tmp_path):
    for path, field in [(("responses", 0, "source"), "granularty"), (("responses", 0, "source", "column_map"), "vale")]:
        def add(d, path=path, field=field):
            node = d
            for key in path:
                node = node[key]
            node[field] = "x"
        with pytest.raises(StudySpecError, match=field):
            load_study(_write(tmp_path, _modified(add)))


# --- environment by catalogue id ------------------------------------------------------------

def test_environment_entry_refers_to_the_catalogue(tmp_path):
    (ref,) = load_study(_write(tmp_path, FORMAT2)).environment
    assert ref.catalog_id == "sst_daily"


def test_unknown_catalogue_id_is_refused_naming_the_known_ones(tmp_path):
    text = _modified(lambda d: d["environment"].__setitem__(0, {"catalog_id": "nope"}))
    with pytest.raises(StudySpecError, match=r"nope.*sst_daily"):
        load_study(_write(tmp_path, text))


def test_environment_must_not_repeat_a_catalogue_id(tmp_path):
    text = _modified(lambda d: d["environment"].append({"catalog_id": "sst_daily"}))
    with pytest.raises(StudySpecError, match="sst_daily"):
        load_study(_write(tmp_path, text))


def test_a_site_outside_the_catalogue_domain_is_refused_with_the_reason(tmp_path):
    def move(d):
        d["sites"][0].update(lat=60.0, lon=5.0)
    with pytest.raises(StudySpecError, match=r"Somewhere.*sst_daily.*outside"):
        load_study(_write(tmp_path, _modified(move)))


def test_format_2_refuses_the_format_1_style_environment_entry(tmp_path):
    entry = {"id": "x", "provider": "copernicus_marine", "dataset": "d", "variable": "v", "unit": "u"}
    text = _modified(lambda d: d["environment"].__setitem__(0, entry))
    with pytest.raises(StudySpecError, match="environment"):
        load_study(_write(tmp_path, text))


def test_format_1_refuses_a_catalogue_entry(tmp_path):
    p = _livorno_variant(tmp_path, lambda d: d.__setitem__("environment", [{"catalog_id": "sst_daily"}]))
    with pytest.raises(StudySpecError, match="format_version 2"):
        load_study(p)


# --- CSV response source --------------------------------------------------------------------

def test_csv_source_fields(tmp_path):
    source = load_study(_write(tmp_path, FORMAT2)).responses[0].source
    assert source.type == "csv"
    assert source.file == "response.csv"
    assert source.granularity == "per_trial"
    assert source.column_map.ci_low is None and source.column_map.ci_high is None


@pytest.mark.parametrize("name", ["../x.csv", "/etc/passwd.csv", "sub/x.csv", "x.txt", ".hidden.csv", "x\\y.csv", ""])
def test_csv_file_must_be_a_bare_csv_filename(tmp_path, name):
    text = _modified(lambda d: d["responses"][0]["source"].__setitem__("file", name))
    with pytest.raises(StudySpecError, match="file"):
        load_study(_write(tmp_path, text))


def test_ci_columns_come_in_pairs(tmp_path):
    def only_low(d):
        d["responses"][0]["source"]["column_map"]["ci_low"] = "lo"
    with pytest.raises(StudySpecError, match="ci_low.*ci_high"):
        load_study(_write(tmp_path, _modified(only_low)))


def test_ci_columns_as_pair_load(tmp_path):
    def both(d):
        d["responses"][0]["source"]["column_map"].update(ci_low="lo", ci_high="hi")
    cm = load_study(_write(tmp_path, _modified(both))).responses[0].source.column_map
    assert (cm.ci_low, cm.ci_high) == ("lo", "hi")


def test_per_trial_values_need_an_aggregation_rule(tmp_path):
    text = _modified(lambda d: d["responses"][0].pop("aggregation"))
    with pytest.raises(StudySpecError, match="aggregation"):
        load_study(_write(tmp_path, text))


def test_already_aggregated_values_need_no_aggregation(tmp_path):
    def aggregated(d):
        d["responses"][0]["source"]["granularity"] = "aggregated"
        d["responses"][0].pop("aggregation")
    assert load_study(_write(tmp_path, _modified(aggregated))).responses[0].aggregation is None


def test_csv_source_is_not_allowed_in_format_1(tmp_path):
    csv = {"type": "csv", "file": "r.csv", "temporal_resolution": "month", "granularity": "aggregated",
           "column_map": {"date": "d", "value": "v"}}
    p = _livorno_variant(tmp_path, lambda d: d["responses"][0].__setitem__("source", csv))
    with pytest.raises(StudySpecError, match="format_version 2"):
        load_study(p)


# --- split_date -----------------------------------------------------------------------------

def test_split_date_is_optional_in_format_2(tmp_path):
    assert load_study(_write(tmp_path, FORMAT2)).responses[0].split_date is None


def test_split_date_in_format_2_must_be_an_iso_date(tmp_path):
    text = _modified(lambda d: d["responses"][0].__setitem__("split_date", "June 2016"))
    with pytest.raises(StudySpecError, match="split_date"):
        load_study(_write(tmp_path, text))


def test_split_date_stays_required_in_format_1(tmp_path):
    p = _livorno_variant(tmp_path, lambda d: d["responses"][0].pop("split_date"))
    with pytest.raises(StudySpecError, match="split_date"):
        load_study(p)


# --- imputation (declared, absent by default) -----------------------------------------------

def test_imputation_is_absent_by_default(tmp_path):
    assert load_study(_write(tmp_path, FORMAT2)).responses[0].imputation is None


def test_imputation_declared(tmp_path):
    def declare(d):
        d["responses"][0]["imputation"] = {
            "method": "centered_rolling_mean", "window_months": 12, "min_periods": 3, "passes": 2}
    imp = load_study(_write(tmp_path, _modified(declare))).responses[0].imputation
    assert (imp.method, imp.window_months, imp.min_periods, imp.passes) == ("centered_rolling_mean", 12, 3, 2)


@pytest.mark.parametrize("bad", [
    {"method": "magic", "window_months": 12, "min_periods": 3, "passes": 1},
    {"method": "centered_rolling_mean", "window_months": 12, "min_periods": 13, "passes": 1},
    {"method": "centered_rolling_mean", "window_months": 12, "min_periods": 0, "passes": 1},
    {"method": "centered_rolling_mean", "window_months": 12, "min_periods": 3, "passes": 0},
    {"method": "centered_rolling_mean", "window_months": 12, "min_periods": 3},
])
def test_invalid_imputation_is_refused(tmp_path, bad):
    text = _modified(lambda d: d["responses"][0].__setitem__("imputation", bad))
    with pytest.raises(StudySpecError, match="imputation"):
        load_study(_write(tmp_path, text))


def test_livorno_declares_the_imputation_the_code_applies():
    # Until M1.4 makes the dataset builder read the declaration (invariant 6: a scientific choice
    # must be readable in the study file), the spec and the code must say the same thing.
    from climate_change_on_sea_urchins import common
    imp = load_study(EXAMPLE).responses[0].imputation
    assert imp is not None
    assert imp.method == "centered_rolling_mean"
    assert imp.window_months == common.IMPUTE_WINDOW_MONTHS
    assert imp.min_periods == common.IMPUTE_MIN_PERIODS
    # build_dataset.py applies it once and load_data() once more (docs/adr/0000, item 8).
    assert imp.passes == 2


# --- contaminant ----------------------------------------------------------------------------

def test_livorno_declares_its_contaminant():
    contaminant = load_study(EXAMPLE).responses[0].contaminant
    assert contaminant.name == "copper"


def test_contaminant_is_optional(tmp_path):
    assert load_study(_write(tmp_path, FORMAT2)).responses[0].contaminant is None


# --- the pipeline does not run format 2 yet -------------------------------------------------

def test_the_pipeline_refuses_a_format_2_study_explicitly(tmp_path):
    import subprocess, sys, os
    p = _write(tmp_path, FORMAT2)
    root = Path(__file__).parent.parent
    env = {**os.environ, "CCSU_STUDY": str(p), "PYTHONPATH": f"{root / 'src'}:{root}"}
    out = subprocess.run([sys.executable, "-c", "import climate_change_on_sea_urchins.common"],
                         capture_output=True, text=True, env=env)
    assert out.returncode != 0
    assert "format_version 2" in out.stderr and "not run" in out.stderr
