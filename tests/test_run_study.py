"""`ccsu-run-study` (M1.5): one command, one study, one process. From a study file and a folder with the
response CSV and the downloaded SST to the lagged correlation between the temperature and the response, with
the number of tests declared and the provenance written next to the results.

The correlation is on first differences, on pairs of consecutive months that are both real (not imputed),
whatever imputation the study declares. Livorno's pipeline is untouched.

Guard: `ccsu-run-study` runs in a process where `common` has loaded Livorno at import. A function on its path
that read `common.DATA`, `common.RESULTS`, `common.SPLIT_DATE`, a module-level heatwave baseline or `config`
would compute the new study with Livorno's values, without any error. A static test forbids those references
in everything the command calls, and a test runs the same study twice, in a process where Livorno is
loaded and in one where another format-1 study is, and requires identical results.
"""
import ast
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from statsmodels.stats.multitest import multipletests

from climate_change_on_sea_urchins import run_study
from climate_change_on_sea_urchins.run_study import consecutive_real_difference

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "climate_change_on_sea_urchins"
FIXTURE = ROOT / "tests" / "fixtures" / "paper_mpb_2026" / "data"
CLOCK = lambda: dt.datetime(2026, 10, 16, 9, 0, tzinfo=dt.timezone.utc)


# --- a synthetic study on disk ----------------------------------------------------------------------

def _sst(start="2003-01-01", end="2008-12-31", seed=0):
    days = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    t = 15 + 8 * np.sin(2 * np.pi * (days.dayofyear - 110) / 365.25) + rng.normal(0, 0.3, len(days))
    return pd.DataFrame({"Datetime": days.strftime("%Y-%m-%d"), "Temperature": t})


def _write_study(folder: Path, *, max_lag=None, imputation=True, baseline=(2003, 2004), source_type="csv",
                 sites=1, environment=("sst_daily",), response_file="response.csv", study_id="synthetic-study"):
    folder.mkdir(parents=True, exist_ok=True)
    d = {"format_version": 2, "id": study_id, "description": "synthetic",
         "mhw_climatology": {"baseline_start_year": baseline[0], "baseline_end_year": baseline[1]},
         "sites": [{"id": f"s{i}", "lat": 43.43, "lon": 10.40, "name": f"Site {i}", "bbox_delta": 0.1} for i in range(sites)],
         "environment": [{"catalog_id": e} for e in environment],
         "responses": [{"id": "r", "label": "Fertility", "nature": "index", "adverse_direction": "decrease", "unit": "1",
                        "source": ({"type": "csv", "file": response_file, "temporal_resolution": "month", "granularity": "per_trial",
                                    "column_map": {"date": "date", "value": "value"}} if source_type == "csv" else
                                   {"type": "google_sheet", "sheet_id": "x", "temporal_resolution": "month",
                                    "column_map": {"date": "d", "value": "v", "ci_low": "l", "ci_high": "h"}}),
                        "aggregation": {"period": "month", "method": "mean", "count_field": "n_assays"}}]}
    if imputation:
        d["responses"][0]["imputation"] = {"method": "centered_rolling_mean", "window_months": 12, "min_periods": 3, "passes": 1}
    if max_lag is not None:
        d["analysis"] = {"max_lag_months": max_lag}
    path = folder / "study.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False))
    return path


def _write_inputs(folder: Path, *, seed=1, missing=0.5, sst=None):
    folder.mkdir(parents=True, exist_ok=True)
    sst = _sst() if sst is None else sst
    sst.to_csv(folder / "sst_daily.csv", index=False)
    (folder / "sst_daily.manifest.json").write_text(json.dumps({"format": 1, "kind": "daily_sst", "dataset": {"id": "x", "version": "202511"}}))
    rng = np.random.default_rng(seed)
    months = pd.date_range("2003-06-01", "2008-06-01", freq="MS")
    rows = []
    for m in months:
        if rng.random() < missing:
            continue
        for k in range(int(rng.integers(1, 4))):
            rows.append((m + pd.Timedelta(days=int(rng.integers(0, 25))), float(rng.normal(30, 5))))
    pd.DataFrame(rows, columns=["date", "value"]).assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_csv(folder / "response.csv", index=False)


@pytest.fixture
def study(tmp_path):
    spec = _write_study(tmp_path / "study")
    _write_inputs(tmp_path / "inputs")
    return spec, tmp_path / "inputs", tmp_path / "out"


def _run(study, **kw):
    spec, inputs, out = study
    return run_study.run_study(spec, inputs, out, now=CLOCK, **kw), out


# --- the end-to-end run -----------------------------------------------------------------------------

def test_runs_from_the_study_and_the_inputs_to_results_and_provenance(study):
    code, out = _run(study)
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == ["ccf.csv", "dataset", "progress.json", "run.json"]
    ccf = pd.read_csv(out / "ccf.csv")
    assert list(ccf.columns) == ["driver", "response", "lag", "n", "spearman_r", "p_value", "p_bh", "r_toward_adverse"]
    assert ccf["lag"].tolist() == list(range(13))
    assert (ccf["driver"] == "SST anomaly").all() and (ccf["response"] == "Fertility").all()


def test_the_number_of_tests_is_declared_and_comes_from_max_lag(tmp_path):
    spec = _write_study(tmp_path / "study", max_lag=6)
    _write_inputs(tmp_path / "inputs")
    assert run_study.run_study(spec, tmp_path / "inputs", tmp_path / "out", now=CLOCK) == 0
    run = json.loads((tmp_path / "out" / "run.json").read_text())
    assert run["parameters"]["max_lag_months"] == 6
    assert run["tests"]["declared"] == 7 and run["tests"]["performed"] == 7
    assert len(pd.read_csv(tmp_path / "out" / "ccf.csv")) == 7


def test_default_max_lag_is_twelve_and_is_in_the_provenance(study):
    _, out = _run(study)
    run = json.loads((out / "run.json").read_text())
    assert run["parameters"]["max_lag_months"] == 12 and run["tests"]["declared"] == 13


def test_a_lag_with_too_few_pairs_is_declared_not_dropped_silently(tmp_path):
    spec = _write_study(tmp_path / "study", max_lag=12, imputation=False)
    _write_inputs(tmp_path / "inputs", missing=0.93)
    assert run_study.run_study(spec, tmp_path / "inputs", tmp_path / "out", now=CLOCK) == 0
    run = json.loads((tmp_path / "out" / "run.json").read_text())
    assert run["tests"]["declared"] == 13
    assert run["tests"]["performed"] + len(run["tests"]["not_performed"]) == 13
    assert run["tests"]["not_performed"] and "fewer than" in run["tests"]["not_performed"][0]["reason"]


def test_bh_is_over_the_tests_performed(study):
    _, out = _run(study)
    ccf = pd.read_csv(out / "ccf.csv")
    ok = ccf["p_value"].notna()
    np.testing.assert_allclose(ccf.loc[ok, "p_bh"], multipletests(ccf.loc[ok, "p_value"], method="fdr_bh")[1])


def test_the_adverse_direction_of_the_study_orients_r(study):
    _, out = _run(study)
    ccf = pd.read_csv(out / "ccf.csv")
    # adverse_direction is "decrease": the response getting worse means going down, so a negative r is toward adverse
    np.testing.assert_allclose(ccf["r_toward_adverse"], -ccf["spearman_r"])


def test_provenance_records_study_inputs_code_and_parameters(study):
    spec, inputs, _ = study
    _, out = _run(study)
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "done" and run["finished_at"] == "2026-10-16T09:00:00+00:00"
    assert run["study"]["id"] == "synthetic-study"
    assert run["study"]["sha256"] == hashlib.sha256(spec.read_bytes()).hexdigest()
    assert run["inputs"]["response.csv"]["sha256"] == hashlib.sha256((inputs / "response.csv").read_bytes()).hexdigest()
    assert run["inputs"]["sst_daily.csv"]["sha256"] == hashlib.sha256((inputs / "sst_daily.csv").read_bytes()).hexdigest()
    assert run["download_manifest"]["dataset"]["version"] == "202511"
    p = run["parameters"]
    assert p["driver"] == "SST anomaly" and p["transform"] == "first differences" and p["pairs"] == "consecutive months, both real"
    assert p["correction"] == "Benjamini-Hochberg" and p["response_label"] == "Fertility"
    assert p["climatology_baseline"] == [2003, 2004]      # the anomaly is against the climatology the study declares
    assert run["coverage"]["temperature"]["rule"].startswith("a month with any missing")
    assert "climate_change_on_sea_urchins" in run["versions"]
    text = json.dumps(run)
    assert str(Path.home()) not in text


def test_the_dataset_is_written_next_to_the_results(study):
    _, out = _run(study)
    assert (out / "dataset" / "monthly.csv").exists() and (out / "dataset" / "dataset.manifest.json").exists()


# --- the driver is the SST anomaly -------------------------------------------------------------------

def test_the_driver_is_the_anomaly_the_dataset_builder_writes(study):
    from scipy import stats
    _, out = _run(study)
    monthly = pd.read_csv(out / "dataset" / "monthly.csv", parse_dates=["Datetime"])
    mhw = pd.read_csv(out / "dataset" / "mhw_monthly.csv", parse_dates=["Datetime"])
    anomaly = monthly[["Datetime"]].merge(mhw, on="Datetime", how="left")["anomaly_mean"].to_numpy()
    dx = np.r_[np.nan, np.diff(anomaly)]
    dy = consecutive_real_difference(monthly, "response", "response_imputed")
    ccf = pd.read_csv(out / "ccf.csv")
    for lag in (0, 3):
        x, y = (dx, dy) if lag == 0 else (dx[:-lag], dy[lag:])
        ok = np.isfinite(x) & np.isfinite(y)
        r, p = stats.spearmanr(x[ok], y[ok])
        row = ccf[ccf["lag"] == lag].iloc[0]
        assert row["n"] == ok.sum() and row["spearman_r"] == pytest.approx(r) and row["p_value"] == pytest.approx(p)


def test_a_purely_seasonal_response_gives_no_significant_lag(tmp_path):
    """The seasonal cycle of the temperature is not the driver: a response that only has its own seasonal cycle,
    independent of the anomaly, must not come out associated at several lags (it does with the raw temperature)."""
    spec = _write_study(tmp_path / "study", imputation=False)
    _write_inputs(tmp_path / "inputs")
    months = pd.date_range("2003-06-01", "2008-06-01", freq="MS")
    rng = np.random.default_rng(5)
    value = 30 + 8 * np.sin(2 * np.pi * months.month / 12 + 0.7) + rng.normal(0, 1.0, len(months))
    pd.DataFrame({"date": months.strftime("%Y-%m-%d"), "value": value}).to_csv(tmp_path / "inputs" / "response.csv", index=False)
    assert run_study.run_study(spec, tmp_path / "inputs", tmp_path / "out", now=CLOCK) == 0
    ccf = pd.read_csv(tmp_path / "out" / "ccf.csv")
    assert ccf["n"].min() >= 40 and (ccf["p_bh"] >= 0.05).all()


# --- consecutive real pairs -------------------------------------------------------------------------

def _monthly(values, imputed):
    n = len(values)
    return pd.DataFrame({"Datetime": pd.date_range("2004-01-01", periods=n, freq="MS"),
                         "Temperature": np.arange(n, dtype=float) ** 1.5,
                         "response": values, "response_imputed": imputed})


def test_a_difference_needs_both_months_real():
    m = _monthly([1.0, 2.0, 3.5, 4.0, 4.5], [False, False, True, False, False])
    d = consecutive_real_difference(m, "response", "response_imputed")
    assert np.isnan(d[0])                          # no previous month
    assert d[1] == 1.0                             # months 0 and 1, both real
    assert np.isnan(d[2]) and np.isnan(d[3])       # month 2 imputed: the difference into it and out of it
    assert d[4] == 0.5                             # months 3 and 4, both real


def test_pairs_are_counted_in_the_results(tmp_path):
    spec = _write_study(tmp_path / "study", max_lag=1)
    _write_inputs(tmp_path / "inputs")
    run_study.run_study(spec, tmp_path / "inputs", tmp_path / "out", now=CLOCK)
    monthly = pd.read_csv(tmp_path / "out" / "dataset" / "monthly.csv", parse_dates=["Datetime"])
    real = ~monthly["response_imputed"].astype(bool).to_numpy() & monthly["response"].notna().to_numpy()
    anomaly = monthly[["Datetime"]].merge(pd.read_csv(tmp_path / "out" / "dataset" / "mhw_monthly.csv", parse_dates=["Datetime"]),
                                          on="Datetime", how="left")["anomaly_mean"].notna().to_numpy()
    pairs = real & np.r_[False, real[:-1]] & anomaly & np.r_[False, anomaly[:-1]]
    assert pd.read_csv(tmp_path / "out" / "ccf.csv").query("lag == 0")["n"].iloc[0] == int(pairs.sum())


# --- progress and failure ---------------------------------------------------------------------------

def test_progress_is_readable_at_every_step(study):
    seen = []
    spec, inputs, out = study
    run_study.run_study(spec, inputs, out, now=CLOCK, on_step=lambda name: seen.append((name, json.loads((out / "progress.json").read_text()))))
    names = [n for n, _ in seen]
    assert names == ["load_study", "read_response", "read_sst", "build_dataset", "correlate", "write_results"]
    assert all(p["status"] == "running" and p["step"] == n for n, p in seen)
    final = json.loads((out / "progress.json").read_text())
    assert final["status"] == "done" and [s["status"] for s in final["steps"]] == ["done"] * 6


def test_a_failure_is_written_with_its_reason_and_no_results(tmp_path):
    spec = _write_study(tmp_path / "study", response_file="missing.csv")
    _write_inputs(tmp_path / "inputs")
    assert run_study.run_study(spec, tmp_path / "inputs", tmp_path / "out", now=CLOCK) == 1
    p = json.loads((tmp_path / "out" / "progress.json").read_text())
    assert p["status"] == "failed" and p["step"] == "read_response" and "missing.csv" in p["error"]
    assert not (tmp_path / "out" / "ccf.csv").exists()


def test_a_bad_csv_fails_at_reading_with_the_csv_message(tmp_path):
    spec = _write_study(tmp_path / "study")
    _write_inputs(tmp_path / "inputs")
    (tmp_path / "inputs" / "response.csv").write_text("date,value\n31/01/2020,12\n")
    assert run_study.run_study(spec, tmp_path / "inputs", tmp_path / "out", now=CLOCK) == 1
    p = json.loads((tmp_path / "out" / "progress.json").read_text())
    assert p["step"] == "read_response" and "date format" in p["error"]


@pytest.mark.parametrize("kw,reason", [
    (dict(source_type="google_sheet"), "csv"),
    (dict(sites=2), "one site"),
    (dict(environment=()), "sst_daily"),
])
def test_what_the_slice_does_not_serve_is_refused_with_the_reason(tmp_path, kw, reason):
    spec = _write_study(tmp_path / "study", **kw)
    _write_inputs(tmp_path / "inputs")
    assert run_study.run_study(spec, tmp_path / "inputs", tmp_path / "out", now=CLOCK) == 1
    assert reason in json.loads((tmp_path / "out" / "progress.json").read_text())["error"]


def test_a_format_1_study_is_sent_to_the_pipeline(tmp_path):
    out = tmp_path / "out"
    assert run_study.run_study(ROOT / "examples" / "livorno_paracentrotus" / "study.yaml", tmp_path, out, now=CLOCK) == 1
    assert "ccsu-run-pipeline" in json.loads((out / "progress.json").read_text())["error"]


def test_the_command_line_runs_and_reports(study, capsys):
    spec, inputs, out = study
    assert run_study.main([str(spec), "--inputs", str(inputs), "--out", str(out)]) == 0
    assert "done" in capsys.readouterr().out and (out / "ccf.csv").exists()


# --- the analysis block of the study --------------------------------------------------------------

def test_max_lag_in_the_study_is_validated(tmp_path):
    from climate_change_on_sea_urchins.study_spec import StudySpecError, load_study
    for bad in (0, -1, 61, "x"):
        spec = _write_study(tmp_path / f"s{bad}", max_lag=bad)
        with pytest.raises(StudySpecError, match="max_lag_months"):
            load_study(spec)
    assert load_study(_write_study(tmp_path / "ok")).analysis.max_lag_months == 12


def test_a_typo_inside_analysis_is_refused(tmp_path):
    from climate_change_on_sea_urchins.study_spec import StudySpecError, load_study
    spec = _write_study(tmp_path / "s", max_lag=6)
    d = yaml.safe_load(spec.read_text()); d["analysis"]["max_lag_month"] = 6
    spec.write_text(yaml.safe_dump(d))
    with pytest.raises(StudySpecError, match="max_lag_month"):
        load_study(spec)


def test_livorno_declares_the_lag_the_code_uses():
    from climate_change_on_sea_urchins import common
    from climate_change_on_sea_urchins.study_spec import load_study
    assert load_study(ROOT / "examples" / "livorno_paracentrotus" / "study.yaml").analysis.max_lag_months == common.TAU_MAX


# --- the guard: nothing on the path reads the module state of the loaded study -----------------------

FORBIDDEN_ATTRS = {"DATA", "RESULTS", "SPLIT_DATE", "SPLIT_YEAR", "WINDOWS", "ROOT", "MHW_CLIM_START", "MHW_CLIM_END", "TAU_MAX",
                   "RESPONSE_SPEC", "STUDY_ID", "SITE_LAT", "SITE_LON", "SITE_NAME", "BBOX_DELTA"}
FORBIDDEN_NAMES = FORBIDDEN_ATTRS | {"CLIM_START", "CLIM_END", "config", "load_data", "load_sst_daily", "load_ec50_raw", "load_ec50_monthly",
                                     "load_mhw_annual", "default_results_dir", "default_response_spec", "results_dir", "provenance",
                                     "selected_study_path", "load_selected_study"}


def _modules():
    return {p.stem: ast.parse(p.read_text()) for p in PKG.glob("*.py")}


def _functions(tree):
    return {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _reachable(entry=("run_study", "run_study")):
    """Functions of the package reachable by name from the entry point (calls to a function of the same module,
    to `from .x import f` names and to `x.f` with x a sibling module)."""
    mods = _modules()
    funcs = {m: _functions(t) for m, t in mods.items()}
    imports = {}
    for m, t in mods.items():
        names = {}
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and n.level == 1:
                for a in n.names:
                    if n.module is None and a.name in mods:
                        names[a.asname or a.name] = ("module", a.name)
                    elif n.module in mods:
                        names[a.asname or a.name] = ("func", n.module, a.name)
        imports[m] = names
    seen, stack = set(), [entry]
    while stack:
        m, f = stack.pop()
        if (m, f) in seen or f not in funcs.get(m, {}):
            continue
        seen.add((m, f))
        for n in ast.walk(funcs[m][f]):
            if isinstance(n, ast.Call):
                fn = n.func
                if isinstance(fn, ast.Name):
                    imp = imports[m].get(fn.id)
                    stack.append((imp[1], imp[2]) if imp and imp[0] == "func" else (m, fn.id))
                elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                    imp = imports[m].get(fn.value.id)
                    if imp and imp[0] == "module":
                        stack.append((imp[1], fn.attr))
    return seen, funcs


def test_the_chain_is_found():
    seen, _ = _reachable()
    for needed in [("run_study", "run_study"), ("dataset_builder", "build_dataset"), ("response_csv", "read_response"),
                   ("mhw_detection", "detect_events"), ("mhw_analysis", "ccf_at_lags"), ("common", "aggregate_period"),
                   ("common", "impute_series")]:
        assert needed in seen, f"{needed} is not reached from run_study.run_study: the guard would look at nothing"


def test_nothing_on_the_path_reads_the_loaded_studys_state():
    seen, funcs = _reachable()
    offenders = []
    for m, f in sorted(seen):
        node = funcs[m][f]
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in {"common", "config", "study_spec"} and n.attr in FORBIDDEN_ATTRS | FORBIDDEN_NAMES:
                offenders.append(f"{m}.{f}: {n.value.id}.{n.attr}")
            elif isinstance(n, ast.Name) and n.id in FORBIDDEN_NAMES and not isinstance(n.ctx, ast.Store):
                offenders.append(f"{m}.{f}: {n.id}")
    assert not offenders, "reads the module state of the study loaded at import: " + ", ".join(offenders)


# --- same results whichever study common has loaded -------------------------------------------------

def _other_format_1_study(tmp_path) -> Path:
    d = yaml.safe_load((ROOT / "examples" / "livorno_paracentrotus" / "study.yaml").read_text())
    d["id"] = "another-format-1"
    d["data_dir"] = str(FIXTURE)
    d["mhw_climatology"] = {"baseline_start_year": 2005, "baseline_end_year": 2007}
    d["responses"][0]["split_date"] = "2012-01-01"
    d["analysis"] = {"max_lag_months": 9}
    p = tmp_path / "other-study.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


def _in_subprocess(spec, inputs, out, env_study):
    env = {k: v for k, v in os.environ.items() if k != "CCSU_STUDY"}
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    if env_study:
        env["CCSU_STUDY"] = str(env_study)
    code = ("import sys, datetime as dt; from climate_change_on_sea_urchins import common, run_study; "
            f"print(common._study.id); sys.exit(run_study.run_study({str(spec)!r}, {str(inputs)!r}, {str(out)!r}, "
            "now=lambda: dt.datetime(2026, 10, 16, tzinfo=dt.timezone.utc)))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return r.stdout.split()[0]


def test_the_same_results_whichever_study_the_process_has_loaded(tmp_path):
    spec = _write_study(tmp_path / "study")
    _write_inputs(tmp_path / "inputs")
    loaded_a = _in_subprocess(spec, tmp_path / "inputs", tmp_path / "out_a", None)
    loaded_b = _in_subprocess(spec, tmp_path / "inputs", tmp_path / "out_b", _other_format_1_study(tmp_path))
    assert (loaded_a, loaded_b) == ("livorno-paracentrotus", "another-format-1")      # the premise: two different loaded studies
    for name in ["ccf.csv", "dataset/monthly.csv", "dataset/mhw_events.csv", "dataset/mhw_monthly.csv", "dataset/mhw_annual.csv"]:
        assert (tmp_path / "out_a" / name).read_bytes() == (tmp_path / "out_b" / name).read_bytes(), name
    a, b = (json.loads((tmp_path / f"out_{x}" / "run.json").read_text()) for x in "ab")
    for r in (a, b):
        r.pop("code", None)
    assert a == b
