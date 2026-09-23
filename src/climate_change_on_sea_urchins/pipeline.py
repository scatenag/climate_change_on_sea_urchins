"""Run all analysis modules in sequence, populating results/."""
import config
from . import (
    mhw_detection, timeseries, period_split, correlations, stationarity,
    mhw_analysis, mhw_lag_extra, mhw_robustness, cu_speciation, thermal_legacy,
    regime_shift, changepoint, negative_control, mhw_lag_annual,
    mhw_annual_changepoint, forecast,
)

# Modules whose output carries the response series' identity (row/column
# labels, "variable"/"series" fields, dict keys, output filenames) --
# main() passes them config.RESPONSE_SPEC explicitly instead of letting
# them fall back to common.default_response_spec() (see its docstring on
# why that fallback exists and why it isn't the primary path).
_NEEDS_RESPONSE = {
    "correlations", "stationarity", "regime_shift", "period_split",
    "cu_speciation", "thermal_legacy", "forecast",
}

_MODULES = [
    # mhw_detection runs first: it regenerates data/mhw_events.csv,
    # mhw_monthly.csv and mhw_annual.csv from data/sst_daily.csv, which every
    # module below depends on via common.load_data(). Keeping this inside the
    # automated pipeline (rather than a separate manual script) is what
    # prevents the MHW catalogue from silently drifting out of sync with
    # sst_daily.csv, as happened for months in this project's history.
    ("mhw_detection", mhw_detection),
    ("timeseries",    timeseries),
    ("period_split",  period_split),
    ("correlations",  correlations),
    ("stationarity",  stationarity),
    ("mhw_analysis",  mhw_analysis),
    ("mhw_lag_extra", mhw_lag_extra),
    ("mhw_robustness", mhw_robustness),
    ("cu_speciation", cu_speciation),
    ("thermal_legacy", thermal_legacy),
    ("regime_shift",  regime_shift),
    ("changepoint",   changepoint),
    ("negative_control", negative_control),
    ("mhw_lag_annual", mhw_lag_annual),
    ("mhw_annual_changepoint", mhw_annual_changepoint),
    ("forecast",      forecast),
]


def main() -> None:
    response = config.RESPONSE_SPEC
    for label, module in _MODULES:
        print(f"\n{'=' * 60}")
        print(f"  Running {label}")
        print(f"{'=' * 60}")
        if label in _NEEDS_RESPONSE:
            module.run(response=response)
        else:
            module.run()

    print("\n✓ All analysis modules complete — results/ populated")


if __name__ == "__main__":
    main()
