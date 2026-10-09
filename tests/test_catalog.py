"""The variable catalogue (M1.1): what the tool can download, declared as data.

First entry: the daily SST. Its dataset, variable and depth must be exactly what
scripts/fetch_copernicus_daily.py downloads today for Livorno (read from the script's source
without importing it: it needs the download stack), so the catalogue cannot silently drift from the
job that produces the data. The catalogue holds no executable field: only names, units and numbers.
"""
import ast
from pathlib import Path

import pytest

from climate_change_on_sea_urchins import catalog
from climate_change_on_sea_urchins.catalog import CatalogError, CatalogVariable

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_copernicus_daily.py"


def _script_constants() -> dict:
    values = {}
    for node in ast.parse(SCRIPT.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value.value
    return values


def test_sst_entry_reproduces_what_the_daily_script_downloads():
    script = _script_constants()
    sst = catalog.get("sst_daily")
    assert sst.dataset_multiyear == script["DATASET_ID"]
    assert sst.depth_min_m == script["DEPTH_MIN"]
    assert sst.depth_max_m == script["DEPTH_MAX"]
    assert sst.variable == "thetao"
    assert sst.cadence == "P1D"
    assert sst.provider == "copernicus_marine"


def test_sst_units_and_no_conversion():
    sst = catalog.get("sst_daily")
    assert sst.unit_native == sst.unit_analysis == "degC"
    assert sst.conversion is None


def test_analysis_forecast_fallback_is_the_id_the_catalogue_really_has():
    # The script's own fallback id (DATASET_ID_FALLBACK) does not exist in the Copernicus
    # catalogue (DatasetNotFound, checked 2026-10-02); the daily analysis-forecast temperature
    # dataset is ...phy-tem_anfc... (no "p"). The catalogue records the one that exists.
    sst = catalog.get("sst_daily")
    assert sst.dataset_analysis_forecast == "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m"
    assert sst.dataset_analysis_forecast != _script_constants()["DATASET_ID_FALLBACK"]


def test_domain_is_the_mediterranean_product_and_covers_livorno():
    sst = catalog.get("sst_daily")
    assert catalog.covers(sst, lat=43.4278, lon=10.3956)


@pytest.mark.parametrize("lat,lon", [
    (60.0, 5.0),      # North Sea
    (-33.9, 18.4),    # Cape Town
    (43.0, -40.0),    # open Atlantic
    (43.0, 60.0),     # Central Asia
])
def test_coordinates_outside_the_domain_are_not_covered(lat, lon):
    assert not catalog.covers(catalog.get("sst_daily"), lat=lat, lon=lon)


def test_unknown_id_is_rejected_naming_the_known_ones():
    with pytest.raises(CatalogError, match="sst_daily") as e:
        catalog.get("sea_surface_banana")
    assert "sea_surface_banana" in str(e.value)


def test_entries_are_validated_models_without_extra_fields():
    # A field the model does not declare is rejected, never silently kept: the catalogue is
    # data, and nothing in it can carry code or a path.
    sst = catalog.get("sst_daily")
    data = sst.model_dump()
    data["command"] = "rm -rf /"
    with pytest.raises(Exception, match="command"):
        CatalogVariable.model_validate(data)


def test_catalogue_ids_match_their_keys():
    for key, entry in catalog.CATALOG.items():
        assert entry.id == key
