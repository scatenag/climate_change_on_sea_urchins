"""
Tests for the declarative study specification (study_spec.py) -- the
contract V2.1 introduces. Written before config.py's inversion, per
CLAUDE.md's "test prima dell'implementazione per ogni contratto o
interfaccia nuova".
"""
import textwrap
from pathlib import Path

import pytest

from climate_change_on_sea_urchins.study_spec import (
    StudySpec, StudySpecError, export_schema, load_study,
)

EXAMPLE = Path(__file__).parent.parent / "examples" / "livorno_paracentrotus" / "study.yaml"


def test_loads_the_livorno_example():
    spec = load_study(EXAMPLE)
    assert isinstance(spec, StudySpec)
    assert len(spec.sites) == 1
    assert len(spec.responses) == 1


def test_site_matches_current_config_values():
    import config
    site = load_study(EXAMPLE).sites[0]
    assert site.lat == config.SITE_LAT
    assert site.lon == config.SITE_LON
    assert site.name == config.SITE_NAME
    assert site.bbox_delta == config.BBOX_DELTA


def test_response_sheet_id_matches_current_config():
    import config
    response = load_study(EXAMPLE).responses[0]
    assert response.source.sheet_id == config.EC50_SHEET_ID
    assert config.EC50_EXPORT_URL == (
        f"https://docs.google.com/spreadsheets/d/{config.EC50_SHEET_ID}/export?format=csv"
    )


def test_response_column_map_is_explicit_not_deduced():
    # note-dati-sorgente.md: pos/neg are NOT controls -- the mapping must be
    # explicit (date/value/ci_low/ci_high), never inferred from column names.
    cm = load_study(EXAMPLE).responses[0].source.column_map
    assert cm.date == "DATE"
    assert cm.value == "EC50"
    assert cm.ci_low == "LL"
    assert cm.ci_high == "UL"


def test_response_source_temporal_resolution_declared():
    source = load_study(EXAMPLE).responses[0].source
    assert source.temporal_resolution == "month"


def test_response_control_columns_are_per_trial_not_a_separate_series():
    controls = load_study(EXAMPLE).responses[0].source.control_columns
    assert controls is not None
    assert len(controls) == 3


def test_response_aggregation_is_distinct_from_source():
    # The per-trial source and the period-level aggregation are two
    # different representations: n_assays exists only in the aggregation,
    # control_columns only in the source.
    response = load_study(EXAMPLE).responses[0]
    assert response.aggregation.period == "month"
    assert response.aggregation.method == "mean"
    assert response.aggregation.count_field == "n_assays"
    assert not hasattr(response.source, "count_field")
    assert not hasattr(response.aggregation, "control_columns")


def test_response_adverse_direction_is_decrease_for_ec50():
    # Lower EC50 = more sensitive = worse -- adverse_direction says which
    # way the value moves when things get worse, not "higher/lower is
    # better" (that flips meaning depending on who's asking).
    assert load_study(EXAMPLE).responses[0].adverse_direction == "decrease"


def test_response_unit_is_microgram_per_liter():
    # The manuscript reports 46.54 ug/L. dashboard.py labeled it mg/L until
    # issue #3 -- a factor-1000 labeling bug, never the source of truth.
    assert load_study(EXAMPLE).responses[0].unit == "ug/L"


def test_response_nature_is_not_biomarker():
    assert load_study(EXAMPLE).responses[0].nature == "population_proxy"


def test_response_label_matches_ec50_for_output_identity():
    # Invariant #4 (CLAUDE.md): artifact identity (results/ CSV headers,
    # dashboard labels) is derived from the spec, never a hardcoded literal.
    # For Livorno this equals "EC50", so existing outputs stay byte-identical
    # once modules read it from here instead of the string "EC50".
    assert load_study(EXAMPLE).responses[0].label == "EC50"


def test_missing_file_raises_study_spec_error_not_a_bare_traceback(tmp_path):
    with pytest.raises(StudySpecError, match="no such file"):
        load_study(tmp_path / "does_not_exist.yaml")


def test_rejects_malformed_yaml_with_line_number(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: x\n  bad_indent: [1, 2\n")
    with pytest.raises(StudySpecError, match=r"line \d+"):
        load_study(bad)


def test_rejects_missing_required_field_with_field_path(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(textwrap.dedent("""\
        id: x
        description: y
        sites:
          - id: s1
            lat: 1.0
            lon: 2.0
            name: n
            # bbox_delta missing on purpose
        responses: []
    """))
    with pytest.raises(StudySpecError, match=r"bbox_delta"):
        load_study(bad)


def test_json_schema_exports_without_error():
    schema = StudySpec.model_json_schema()
    assert schema["title"] == "StudySpec"


def test_export_schema_writes_file(tmp_path):
    out = tmp_path / "study.schema.json"
    export_schema(out)
    assert out.exists()
    assert '"title": "StudySpec"' in out.read_text()


def test_committed_schema_is_up_to_date():
    """docs/schema/study.schema.json is a committed, static export -- guard
    against it silently going stale the next time a field changes."""
    import json
    committed = Path(__file__).parent.parent / "docs" / "schema" / "study.schema.json"
    assert json.loads(committed.read_text()) == StudySpec.model_json_schema()
