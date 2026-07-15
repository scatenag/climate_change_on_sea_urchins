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
import threading

from .common import load_data, SPLIT_YEAR, TAU_MAX, ENV_COLS, ALL_COLS, MHW_COLS

__version__ = "1.3.0"

__all__ = [
    "load_data",
    "SPLIT_YEAR",
    "TAU_MAX",
    "ENV_COLS",
    "ALL_COLS",
    "MHW_COLS",
    "RERUN_LOCK",
]

# Serializes full dashboard script reruns across concurrent Streamlit
# sessions sharing one process. app.py holds this for the duration of its
# del-sys.modules + reimport of the dashboard submodule (see app.py's own
# comment for why that pattern exists). This package's own __init__ module
# is never touched by that deletion loop (it only targets names starting
# with "climate_change_on_sea_urchins.dashboard"), so this lock instance
# survives across reruns instead of being recreated fresh each time --
# a plain module-level Lock() defined in app.py itself would NOT work,
# since app.py's whole body is re-executed by Streamlit on every rerun.
#
# Why this exists: two sessions' reruns can end up in flight at once (a
# second visitor, or a browser reconnecting after a dropped WebSocket
# without the old session having fully finished -- confirmed in a
# production log via Streamlit's own "Session ... is already connected!"
# message shortly before a segfault). Without this lock, two threads could
# race on `del sys.modules[...]` for the same module name (that `del`
# bypasses Python's own per-module import lock entirely) or run heavy
# numpy/scipy/pandas C calls concurrently -- either is a plausible native
# crash. See project memory / commit history for the fuller investigation.
RERUN_LOCK = threading.Lock()
