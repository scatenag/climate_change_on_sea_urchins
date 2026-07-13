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
import sys

# Streamlit re-executes THIS file's bytecode on every rerun, but a plain
# `import climate_change_on_sea_urchins.dashboard` is only a fresh import
# (running dashboard.py's module body -- every widget, every tab) on the
# FIRST run. From the second rerun on, the module is already in
# sys.modules, so the import is a no-op: dashboard.py's code never
# executes again, no error is raised, and Streamlit reports the script run
# as having completed successfully -- it just produced no new UI content.
# Confirmed with a minimal reproduction (a single st.radio, no other code)
# reduced from this exact shim-into-a-package pattern: switching the
# radio's value blanks the page every time; the identical script run
# directly as the main file (no package import at all) works every time.
# Forcing dashboard.py out of sys.modules before each import makes every
# rerun a genuine fresh import, matching what a single-file app gets for
# free.
for _mod_name in [m for m in sys.modules if m.startswith("climate_change_on_sea_urchins.dashboard")]:
    del sys.modules[_mod_name]

import climate_change_on_sea_urchins.dashboard  # noqa: F401
