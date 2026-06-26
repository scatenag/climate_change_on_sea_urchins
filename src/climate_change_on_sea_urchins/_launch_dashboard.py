"""Console-script launcher for the Streamlit dashboard (`ccsu-dashboard`).

Kept separate from dashboard.py: importing *this* module must not execute
the dashboard's Streamlit script — it only hands a path to Streamlit's own
CLI, which runs dashboard.py as its own script invocation.
"""
import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as stcli

    dashboard_path = str(Path(__file__).parent / "dashboard.py")
    sys.argv = ["streamlit", "run", dashboard_path] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
