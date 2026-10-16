"""The 'your own data' way into the generic app: three steps. Step 1 is filled (MA.2, step_site.py); MA.3 and MA.4 fill
the others. Only functions: the interface runs when `render` is called, at every run, and nothing of a study is kept
in the module -- the study's context is passed in (docs/roadmap/MILESTONE-M1.md, 5.1)."""
from __future__ import annotations

import streamlit as st

from . import step_site

STEPS = [
    ("Step 1", "Site, period and credentials", "Choose a point in the Mediterranean and the years to cover; the sea-surface "
                                               "temperature is downloaded from Copernicus Marine with your credentials."),
    ("Step 2", "Your response series", "Upload the CSV of your measurements (or a test series with a known lag) and answer "
                                       "a few questions about it."),
    ("Step 3", "Run and read the result", "The lagged correlation between the temperature anomaly and your series, with "
                                          "its limits stated."),
]


def render(ctx) -> None:
    st.header("Your own data")
    st.text(f"Study: {ctx.label or 'none yet'}")
    st.markdown("Your work is not saved: closing the tab loses it. Your files stay in a folder of this session, "
                "deleted after 6 hours without use or when you press *Delete my data*.")
    step_site.render(ctx)                      # step 1: site, period, credentials, download
    for number, title, what in STEPS[1:]:
        st.subheader(f"{number} — {title}")
        st.write(what)
        st.caption("Not available yet.")
