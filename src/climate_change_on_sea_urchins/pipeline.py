"""Run all analysis modules in sequence, populating results/."""
import json
from pathlib import Path

import config
from . import common
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
    "timeseries", "correlations", "stationarity", "regime_shift", "period_split",
    "cu_speciation", "thermal_legacy", "forecast",
    "mhw_analysis", "mhw_lag_extra", "mhw_robustness", "mhw_lag_annual",
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


def main(results: Path | None = None) -> None:
    """`results`: where every analysis module writes. Defaults to the
    selected study's directory (common.results_dir); passed to each module
    explicitly rather than read by it from a shared constant, so one run
    can later target several directories (one per window). mhw_detection
    is the exception: it writes data_dir, never results."""
    response = config.RESPONSE_SPEC
    if results is None:
        results = common.results_dir(config.STUDY_ID)
    results.mkdir(parents=True, exist_ok=True)
    for label, module in _MODULES:
        print(f"\n{'=' * 60}")
        print(f"  Running {label}")
        print(f"{'=' * 60}")
        if label == "mhw_detection":
            module.run()
        elif label in _NEEDS_RESPONSE:
            module.run(response=response, results=results)
        else:
            module.run(results=results)

    print("\n✓ All analysis modules complete — results/ populated")

    if common.WINDOWS:
        prov = common.provenance()
        for window in common.WINDOWS:
            run_window(window, results / window.id, response=response, provenance=prov)


def run_window(window, results: Path, response=None, provenance: dict | None = None) -> dict:
    """Runs one declared window into `results` (results/<study_id>/<window_id>/)
    and writes window.json there. Only modules declaring SUPPORTS_WINDOW =
    True run, and they receive the window; every other module does not run
    AT ALL for the window -- neither with the window nor on the whole
    record, whose results would otherwise sit under the window's label --
    and window.json says so. mhw_detection is not re-run: the MHW catalogue
    is computed once, on the whole record, by design (its climatology is a
    parameter declared in the spec, not a per-window estimate).

    A module may return {"skipped": "<reason>"} instead of producing output
    (e.g. pre/post analyses when SPLIT_DATE is outside the window)."""
    results.mkdir(parents=True, exist_ok=True)
    report = {}
    for label, module in _MODULES:
        if label == "mhw_detection":
            continue
        if not common.supports_window(module):
            report[label] = {
                "status": "not_run",
                "reason": getattr(module, "WINDOW_UNSUPPORTED_REASON",
                                  "no window support declared (module not migrated yet)"),
            }
            continue
        print(f"\n  [{window.id}] Running {label}")
        before = {p.name for p in results.iterdir()}
        kwargs = {"results": results, "window": window}
        if label in _NEEDS_RESPONSE:
            kwargs["response"] = response
        outcome = module.run(**kwargs) or {}
        written = sorted({p.name for p in results.iterdir()} - before)
        if "skipped" in outcome:
            report[label] = {"status": "skipped", "reason": outcome["skipped"], "outputs": written}
        else:
            report[label] = {"status": "run", "outputs": written}

    manifest = {
        "window": {"id": window.id, "start": window.start.isoformat(), "end": window.end.isoformat()},
        "provenance": provenance if provenance is not None else common.provenance(),
        "coverage": common.response_coverage(window),
        "split_date": common.SPLIT_DATE.date().isoformat(),
        "split_date_in_window": common.split_date_in_window(window),
        "modules": report,
    }
    (results / "window.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"✓ window {window.id}: {sum(r['status'] == 'run' for r in report.values())} modules run, "
          f"{sum(r['status'] != 'run' for r in report.values())} not run or skipped — see window.json")
    return manifest


if __name__ == "__main__":
    main()
