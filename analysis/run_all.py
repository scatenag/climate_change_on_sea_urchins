"""Run all analysis modules in sequence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

modules = [
    ("01_timeseries",  "01_timeseries"),
    ("02_period_split","02_period_split"),
    ("03_correlations","03_correlations"),
    ("04_stationarity","04_stationarity"),
    ("05_mhw_analysis","05_mhw_analysis"),
    ("06_forecast",    "06_forecast"),
]

for label, mod_name in modules:
    print(f"\n{'='*60}")
    print(f"  Running {label}")
    print(f"{'='*60}")
    import importlib
    mod = importlib.import_module(mod_name)
    mod.run()

print("\n✓ All analysis modules complete — results/ populated")
