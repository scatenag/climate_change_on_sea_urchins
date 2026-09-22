"""Site and data-source configuration for this monitoring framework instance.

Reads examples/livorno_paracentrotus/study.yaml through
climate_change_on_sea_urchins.study_spec.load_study() and re-exports the
same names every other file already imports -- SITE_LAT, SITE_LON,
SITE_NAME, BBOX_DELTA, EC50_SHEET_ID, EC50_EXPORT_URL -- so nothing else
changes (see docs/roadmap/note-tecniche.md sec 4 and
docs/adr/0000-decisioni-rimandate.md for what is deliberately not moved
here yet).

To adapt the framework to a different site or response series, edit that
YAML file, not this one -- see docs/ADAPTING.md.
"""
from pathlib import Path

from climate_change_on_sea_urchins.study_spec import load_study

_STUDY_PATH = Path(__file__).resolve().parent / "examples" / "livorno_paracentrotus" / "study.yaml"
_study = load_study(_STUDY_PATH)
_site = _study.sites[0]
_response = _study.responses[0]

# --- Oceanographic monitoring site (Copernicus Marine grid cell) -----------
SITE_LAT   = _site.lat
SITE_LON   = _site.lon
SITE_NAME  = _site.name
BBOX_DELTA = _site.bbox_delta

# --- Biological bioassay data source (ISPRA Google Sheets export) ----------
EC50_SHEET_ID   = _response.source.sheet_id
EC50_EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{EC50_SHEET_ID}/export?format=csv"

# --- Unit conversions --------------------------------------------------------
# Copernicus's `spco2` variable (Mediterranean BGC reanalysis) is delivered in
# Pascal -- and correctly documented as such: the CF standard_name it carries,
# surface_partial_pressure_of_carbon_dioxide_in_sea_water, is associated with
# Pascal in the CF conventions vocabulary, not microatmospheres (uatm). The
# bug was this pipeline misreading the value as already being in uatm, not
# any error on Copernicus's side. 1 atm = 101325 Pa, so Pa -> uatm is
# *(1e6 / 101325). Applied once at ingestion in scripts/fetch_copernicus.py
# and scripts/fetch_copernicus_update.py.
# Deliberately NOT in study.yaml: this is a physical constant, not a
# scientific choice a study file should be able to change (V2.1 constraint,
# see docs/roadmap/ripresa-tool-tre-sessioni.md "Sessione 3").
CO2_PA_TO_UATM = 1e6 / 101325  # == 9.8692...
