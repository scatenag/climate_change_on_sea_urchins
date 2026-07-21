"""Run all analysis modules in sequence, populating results/."""
from . import (
    mhw_detection, timeseries, period_split, correlations, stationarity,
    mhw_analysis, mhw_lag_extra, mhw_robustness, cu_speciation, thermal_legacy,
    regime_shift, mhw_lag_annual, forecast,
)

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
    ("mhw_lag_annual", mhw_lag_annual),
    ("forecast",      forecast),
]


def main() -> None:
    for label, module in _MODULES:
        print(f"\n{'=' * 60}")
        print(f"  Running {label}")
        print(f"{'=' * 60}")
        module.run()

    print("\n✓ All analysis modules complete — results/ populated")


if __name__ == "__main__":
    main()
