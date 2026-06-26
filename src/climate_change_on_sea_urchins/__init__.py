"""
climate_change_on_sea_urchins
==============================

Open-source, FAIR pipeline integrating Copernicus Marine oceanographic data
with biological sentinel bioassay data, using Paracentrotus lividus gamete
EC50 toxicity as a proof of concept. See docs/ADAPTING.md in the repository
for how to point this package at a different site, species, or indicator.

Quick start::

    from climate_change_on_sea_urchins import load_data
    df_full, df_real, events, monthly = load_data()

Or run the full statistical pipeline (writes results/*.csv, *.json)::

    from climate_change_on_sea_urchins.pipeline import main
    main()

equivalently, after installing the package, from the command line::

    ccsu-run-pipeline
"""
from .common import load_data, SPLIT_YEAR, TAU_MAX, ENV_COLS, ALL_COLS, MHW_COLS

__version__ = "1.2.0"

__all__ = [
    "load_data",
    "SPLIT_YEAR",
    "TAU_MAX",
    "ENV_COLS",
    "ALL_COLS",
    "MHW_COLS",
]
