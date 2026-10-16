"""`ccsu-run-study` (M1.5): one study, one process. From a study file (format 2) and a folder holding the
response CSV and the downloaded SST to the lagged correlation between the temperature and the response,
with the number of tests declared and the provenance written next to the results.

Steps (each one written to `progress.json`, which another process can read at any time): load the study;
read the response CSV (response_csv); read the SST; build the dataset (dataset_builder); correlate; write.

The correlation: Spearman between the driver's first difference at t-k and the response's first difference
at t, lags 0..`analysis.max_lag_months` of the study, on pairs of **consecutive months that are both real**
(a real measurement, not an imputed one) whatever imputation the study declares: a difference with an
imputed end is not an observation. Benjamini-Hochberg over the lags performed. The number of tests is
declared in `run.json`; a lag with fewer than MIN_PAIRS pairs is listed there with its reason, not dropped.

The driver is the **monthly SST anomaly** (`anomaly_mean` of the dataset builder: the monthly mean of the daily
anomalies against the climatology of the baseline years the study declares), not the temperature: with the
temperature the seasonal cycle alone makes several lags significant (ADR-0012). So this correlation looks for
the non-seasonal association only: a dependence that goes through the seasonal part of the temperature alone
is not seen by it. The heatwave metrics join later (M1.9).

Limit accepted for milestone M1 (docs/adr/0000, item 19): `common` loads the Livorno study when it is
imported, so this command runs in a process where that state exists. Nothing on the path of this command may
read it -- the data and results folders, the split date, a module-level heatwave baseline, `config` --
or the new study would be computed with Livorno's values and no error. tests/test_run_study.py forbids those
references in everything this command calls, and runs one study in processes with different studies loaded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from . import dataset_builder, mhw_analysis, response_csv
from .study_spec import load_study

STEPS = ["load_study", "read_response", "read_sst", "build_dataset", "correlate", "write_results"]
MIN_PAIRS = 10        # a lag with fewer pairs is not tested (the rule of mhw_analysis.ccf_at_lags)
DRIVER = "SST anomaly"
SST_FILE = "sst_daily.csv"
SST_MANIFEST = "sst_daily.manifest.json"


class RunError(Exception):
    """A study this command cannot run, or inputs that cannot be used; the message says why."""


def consecutive_real_difference(monthly: pd.DataFrame, value_col: str, imputed_col: str) -> np.ndarray:
    """First difference of `value_col` kept only where month t and month t-1 both have a real measurement
    (not flagged imputed, value present); NaN elsewhere."""
    real = (~monthly[imputed_col].astype(bool)).to_numpy() & monthly[value_col].notna().to_numpy()
    both = real & np.r_[False, real[:-1]]
    diff = np.r_[np.nan, np.diff(monthly[value_col].to_numpy(float))]
    return np.where(both, diff, np.nan)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _code_info() -> dict:
    here = Path(__file__).resolve().parent

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    commit = git("rev-parse", "HEAD")
    return {"commit": commit, "dirty": None if commit is None else bool(git("status", "--porcelain", "--untracked-files=no"))}


def _versions() -> dict:
    def v(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "unknown"
    return {"climate_change_on_sea_urchins": v("climate_change_on_sea_urchins"), "pandas": pd.__version__,
            "numpy": np.__version__, "python": platform.python_version()}


def _check_servable(spec):
    if spec.format_version != 2:
        raise RunError("this study is format_version 1: format 1 studies are run by ccsu-run-pipeline, not ccsu-run-study")
    if len(spec.sites) != 1:
        raise RunError("this version of ccsu-run-study runs one site per study")
    if len(spec.responses) != 1:
        raise RunError("this version of ccsu-run-study runs one response per study")
    if "sst_daily" not in [e.catalog_id for e in spec.environment]:
        raise RunError("the study does not list the environment variable 'sst_daily', the only driver of this version")
    if spec.responses[0].source.type != "csv":
        raise RunError("the response source must be a csv file: remote sources are not read by ccsu-run-study")


def run_study(study_path, inputs_dir, out_dir, *, now: Callable[[], dt.datetime] | None = None,
              on_step: Callable[[str], None] | None = None) -> int:
    """Run the study; 0 on success, 1 on a failure (written to progress.json with the step and the reason)."""
    now = now or (lambda: dt.datetime.now(dt.timezone.utc))
    study_path, inputs, out = Path(study_path), Path(inputs_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("ccf.csv", "run.json"):
        (out / stale).unlink(missing_ok=True)
    started = now().isoformat()
    progress = {"status": "running", "step": None, "steps": [{"name": s, "status": "pending", "at": None} for s in STEPS],
                "started_at": started, "error": None}

    def write_progress():
        _write_json_atomic(out / "progress.json", progress)

    def begin(name):
        progress["step"] = name
        for s in progress["steps"]:
            if s["name"] == name:
                s["status"], s["at"] = "running", now().isoformat()
        write_progress()
        if on_step:
            on_step(name)

    def finish(name):
        for s in progress["steps"]:
            if s["name"] == name:
                s["status"] = "done"

    error = None
    try:
        begin("load_study")
        spec = load_study(study_path)
        _check_servable(spec)
        response, site = spec.responses[0], spec.sites[0]
        max_lag = spec.analysis.max_lag_months
        finish("load_study")

        begin("read_response")
        response_file = inputs / response.source.file
        if not response_file.is_file():
            raise RunError(f"the response file {response.source.file!r} is not in the inputs folder")
        series = response_csv.read_response(response_file, response)
        finish("read_response")

        begin("read_sst")
        sst_file = inputs / SST_FILE
        if not sst_file.is_file():
            raise RunError(f"{SST_FILE!r} is not in the inputs folder")
        sst = pd.read_csv(sst_file, parse_dates=["Datetime"])
        finish("read_sst")

        begin("build_dataset")
        ds = dataset_builder.build_dataset(
            sst=sst, response=series.series, imputation=response.imputation,
            baseline=(spec.mhw_climatology.baseline_start_year, spec.mhw_climatology.baseline_end_year))
        finish("build_dataset")

        begin("correlate")
        monthly = ds.monthly
        anomaly = monthly[["Datetime"]].merge(ds.mhw_monthly[["Datetime", "anomaly_mean"]], on="Datetime", how="left")["anomaly_mean"]
        frame = pd.DataFrame({DRIVER: anomaly.diff(),
                              "response": consecutive_real_difference(monthly, "response", "response_imputed")})
        raw = mhw_analysis.ccf_at_lags(frame, DRIVER, ["response"], max_lag)
        ok = raw["p_value"].notna()
        raw["p_bh"] = np.nan
        if ok.any():
            raw.loc[ok, "p_bh"] = multipletests(raw.loc[ok, "p_value"], method="fdr_bh")[1]
        sign = -1.0 if response.adverse_direction == "decrease" else 1.0
        ccf = pd.DataFrame({"driver": DRIVER, "response": response.label, "lag": raw["lag"], "n": raw["n"],
                            "spearman_r": raw["spearman_r"], "p_value": raw["p_value"], "p_bh": raw["p_bh"],
                            "r_toward_adverse": sign * raw["spearman_r"]})
        not_performed = [{"lag": int(r.lag), "n": int(r.n), "reason": f"fewer than {MIN_PAIRS} pairs ({int(r.n)})"}
                         for r in raw[~ok].itertuples()]
        finish("correlate")

        begin("write_results")
        ccf.to_csv(out / "ccf.csv", index=False)
        dataset_builder.write_dataset(ds, out / "dataset", now=now)
        manifest_path = inputs / SST_MANIFEST
        # a declaration next to the response file (the known-truth generator writes one), copied as it is
        response_manifest_path = response_file.with_name(response_file.stem + ".manifest.json")
        run_record = {
            "format": 1, "status": "done", "started_at": started, "finished_at": now().isoformat(),
            "study": {"id": spec.id, "sha256": _sha256(study_path)},
            "inputs": {response.source.file: {"sha256": _sha256(response_file)}, SST_FILE: {"sha256": _sha256(sst_file)}},
            "download_manifest": json.loads(manifest_path.read_text()) if manifest_path.is_file() else None,
            "response_manifest": json.loads(response_manifest_path.read_text()) if response_manifest_path.is_file() else None,
            "site": {"id": site.id, "name": site.name, "lat": site.lat, "lon": site.lon},
            "parameters": {"driver": DRIVER, "transform": "first differences", "pairs": "consecutive months, both real",
                           "max_lag_months": max_lag, "correction": "Benjamini-Hochberg", "min_pairs": MIN_PAIRS,
                           "response_label": response.label, "response_unit": response.unit,
                           "adverse_direction": response.adverse_direction,
                           "imputation_declared": None if response.imputation is None else response.imputation.model_dump(),
                           "climatology_baseline": [spec.mhw_climatology.baseline_start_year, spec.mhw_climatology.baseline_end_year]},
            "tests": {"declared": max_lag + 1, "performed": int(ok.sum()), "not_performed": not_performed},
            "response_csv": series.notes,
            "coverage": ds.coverage,
            "code": _code_info(), "versions": _versions(),
        }
        _write_json_atomic(out / "run.json", run_record)
        finish("write_results")
    except Exception as e:      # any failure ends the run with its reason on file; the credentials are never in play here
        error = f"{type(e).__name__}: {e}" if not isinstance(e, (RunError, response_csv.ResponseCsvError, dataset_builder.DatasetError)) else str(e)
    if error is not None:
        progress["status"], progress["error"] = "failed", error
        write_progress()
        return 1
    progress["status"], progress["step"] = "done", None
    write_progress()
    return 0


def main(argv: list[str] | None = None) -> int:
    """`ccsu-run-study STUDY.yaml --inputs FOLDER [--out FOLDER]`."""
    ap = argparse.ArgumentParser(prog="ccsu-run-study", description=main.__doc__, epilog=(
        "The correlation is between the first differences of the monthly SST anomaly (against the climatology the "
        "study declares) and of the response, on consecutive real months. It looks for the non-seasonal association "
        "only: a dependence that goes through the seasonal part of the temperature alone is not seen."))
    ap.add_argument("study", type=Path)
    ap.add_argument("--inputs", type=Path, required=True, help="folder with the response CSV and sst_daily.csv")
    ap.add_argument("--out", type=Path, default=None, help="default: results/<study id>")
    args = ap.parse_args(argv)
    out = args.out
    if out is None:
        try:
            out = Path("results") / load_study(args.study).id
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    code = run_study(args.study, args.inputs, out)
    progress = json.loads((out / "progress.json").read_text())
    if code:
        print(f"failed at {progress['step']}: {progress['error']}", file=sys.stderr)
    else:
        run = json.loads((out / "run.json").read_text())
        print(f"done: {run['tests']['performed']} of {run['tests']['declared']} tests, results in {out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
