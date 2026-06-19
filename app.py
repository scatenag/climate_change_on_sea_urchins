"""
Climate Change on Sea Urchins — Streamlit Dashboard entry point.

The dashboard itself lives in the installable package, at
src/climate_change_on_sea_urchins/dashboard.py. This file exists so that
`streamlit run app.py` keeps working from a repo checkout (and so the
existing Streamlit Community Cloud deployment, which points at this exact
path, requires no reconfiguration).

Equivalent, once the package is installed (`pip install -e .`):
    ccsu-dashboard
"""
import climate_change_on_sea_urchins.dashboard  # noqa: F401
