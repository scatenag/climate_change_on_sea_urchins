"""Run all analysis modules in sequence, populating results/."""
from . import timeseries, period_split, correlations, stationarity, mhw_analysis, forecast

_MODULES = [
    ("timeseries",   timeseries),
    ("period_split", period_split),
    ("correlations", correlations),
    ("stationarity", stationarity),
    ("mhw_analysis", mhw_analysis),
    ("forecast",     forecast),
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
