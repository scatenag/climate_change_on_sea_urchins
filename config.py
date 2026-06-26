"""Site and data-source configuration for this monitoring framework instance.

This is the single place to edit when adapting the framework to a different
oceanographic monitoring site, species, or biological indicator. See
docs/ADAPTING.md for a full walkthrough of what else needs to change.
"""

# --- Oceanographic monitoring site (Copernicus Marine grid cell) -----------
SITE_LAT   = 43.4278
SITE_LON   = 10.3956
SITE_NAME  = "Livorno Sud"
BBOX_DELTA = 0.1   # degrees, bounding box half-width around the site

# --- Biological bioassay data source (ISPRA Google Sheets export) ----------
EC50_SHEET_ID   = "1e0-16D84ehRyotSC2BH9e9YqAnZksbv4gZDFgyPki8g"
EC50_EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{EC50_SHEET_ID}/export?format=csv"
