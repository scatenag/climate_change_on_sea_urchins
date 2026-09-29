"""
Temporal windows (V2.2 PR 5b): one pipeline run computes each declared
window's statistics into results/<study_id>/<window_id>/.

The rule every window-supporting module follows (docs/roadmap/STATO.md):
  (a) no value after the window's end enters a statistic of the window;
  (b) everything estimated from the data -- fits, detrending, smoothing,
      imputation, climatologies, references -- is estimated only on data
      inside the window;
  (c) lagged values, of a predictor or of the response, may come from
      before the window's start (conditioning on the past).
Parameters declared in the spec (the MHW climatology) do not depend on the
window.

Fast tests pin the contract; the two `golden` tests run real modules on the
frozen fixture: rule (a) end to end, and equivalence of a whole-record
window with the run without a window.
"""
import datetime as dt
import filecmp
import importlib.util
import json
import shutil
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_change_on_sea_urchins import common, pipeline
from climate_change_on_sea_urchins.study_spec import StudySpecError, WindowSpec

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_DATA = REPO_ROOT / "tests" / "fixtures" / "paper_mpb_2026" / "data"

# The modules migrated in 5b-1; 5b-2..5b-4 extend this set.
MIGRATED = {"changepoint", "period_split", "stationarity", "thermal_legacy"}
DECLARED_UNSUPPORTED = {"forecast", "mhw_annual_changepoint"}

SPLIT_WINDOW = WindowSpec(id="w2010-2020", start=dt.date(2010, 1, 1), end=dt.date(2020, 12, 31))
NO_SPLIT_WINDOW = WindowSpec(id="w2004-2015", start=dt.date(2004, 1, 1), end=dt.date(2015, 12, 31))


# --- where results go -------------------------------------------------------

def test_results_dir_for_a_window_is_a_subdirectory_of_the_study():
    assert common.results_dir("s") == common.ROOT / "results" / "s"
    assert common.results_dir("s", "w") == common.ROOT / "results" / "s" / "w"


# --- one imputation, shared with scripts/build_dataset.py ---------------------

def test_impute_response_fills_gaps_with_centered_rolling_mean():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0])
    expected = s.fillna(s.rolling(12, min_periods=3, center=True).mean())
    pd.testing.assert_series_equal(common.impute_response(s), expected)
    assert s.isna().sum() == 3, "input must not be modified in place"


def test_build_dataset_uses_the_package_imputation_not_its_own():
    src = (REPO_ROOT / "scripts" / "build_dataset.py").read_text()
    assert "impute_response" in src
    assert ".rolling(" not in src, "build_dataset.py must not carry its own copy of the imputation"


# --- explicit declaration of window support -----------------------------------

def test_window_support_is_explicit_and_absence_means_unsupported():
    assert common.supports_window(types.SimpleNamespace(SUPPORTS_WINDOW=True))
    assert not common.supports_window(types.SimpleNamespace(SUPPORTS_WINDOW=False))
    assert not common.supports_window(types.SimpleNamespace())
    assert not common.supports_window(types.SimpleNamespace(SUPPORTS_WINDOW="yes")), "only literal True counts"


def test_declared_window_support_matches_the_migration_so_far():
    supported = {label for label, mod in pipeline._MODULES if common.supports_window(mod)}
    assert supported == MIGRATED
    for label, mod in pipeline._MODULES:
        if label in DECLARED_UNSUPPORTED:
            assert getattr(mod, "SUPPORTS_WINDOW", None) is False
            assert getattr(mod, "WINDOW_UNSUPPORTED_REASON", "").strip()


def test_every_supporting_module_accepts_a_window_parameter():
    import inspect
    for label, mod in pipeline._MODULES:
        if common.supports_window(mod):
            assert "window" in inspect.signature(mod.run).parameters, label


# --- windows against the data -------------------------------------------------

def test_a_window_without_any_overlap_with_the_data_is_rejected():
    far = WindowSpec(id="future", start=dt.date(2090, 1, 1), end=dt.date(2095, 12, 31))
    with pytest.raises(StudySpecError, match="no overlap"):
        common.check_window_overlaps_data(far)


def test_a_partially_overlapping_window_is_accepted_and_its_coverage_reported():
    last_real = common.response_coverage()["last_real_response_month"]
    partial = WindowSpec(id="tail", start=dt.date(2020, 1, 1), end=dt.date(2090, 12, 31))
    common.check_window_overlaps_data(partial)
    cov = common.response_coverage(partial)
    assert cov["first_real_response_month"] >= "2020-01-01"
    assert cov["last_real_response_month"] == last_real
    assert cov["n_real_response_months"] > 0


def test_split_date_in_window():
    assert common.split_date_in_window(SPLIT_WINDOW)
    assert not common.split_date_in_window(NO_SPLIT_WINDOW)


def test_window_data_reimputes_the_response_inside_the_window_only():
    """Rule (b): imputation is estimated inside the window. Rule (a), at
    the loading boundary: altering real response values after the window's
    end changes nothing in what load_data(window) returns."""
    w = NO_SPLIT_WINDOW
    full, real, _, _ = common.load_data(window=w)
    assert full["Datetime"].min() >= pd.Timestamp(w.start)
    assert full["Datetime"].max() <= pd.Timestamp(w.end)
    assert real["Datetime"].between(pd.Timestamp(w.start), pd.Timestamp(w.end)).all()

    whole, whole_real, _, _ = common.load_data()
    grid = whole["Datetime"].between(pd.Timestamp(w.start), pd.Timestamp(w.end))
    observed = whole.loc[grid, common.RESPONSE_COL].where(~whole.loc[grid, common.IMPUTED_COL])
    # Two passes, both inside the window: the whole-record path imputes
    # twice (scripts/build_dataset.py, then load_data()), and the second
    # pass fills gaps the first cannot (8 months in the current data).
    expected = common.impute_response(common.impute_response(observed.reset_index(drop=True)))
    pd.testing.assert_series_equal(
        full[common.RESPONSE_COL].reset_index(drop=True), expected, check_names=False)


# --- the pipeline over windows (module stubs) ---------------------------------

def _stub(label, supports=None, reason=None, calls=None):
    mod = types.SimpleNamespace()
    if supports is not None:
        mod.SUPPORTS_WINDOW = supports
    if reason is not None:
        mod.WINDOW_UNSUPPORTED_REASON = reason

    def run(results=None, window=None, response=None):
        if calls is None:
            raise AssertionError(f"{label} must not run for a window")
        calls.append({"window": window, "results": results})
        (results / f"{label}.csv").write_text("x\n1\n")
    mod.run = run
    return mod


def test_run_window_runs_only_declared_modules_and_records_the_rest(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "_MODULES", [
        ("mhw_detection", _stub("mhw_detection")),
        ("ok", _stub("ok", supports=True, calls=calls)),
        ("undeclared", _stub("undeclared")),
        ("declared_no", _stub("declared_no", supports=False, reason="makes no sense in a window")),
    ])
    out = tmp_path / NO_SPLIT_WINDOW.id
    pipeline.run_window(NO_SPLIT_WINDOW, out, response=None, provenance={"code_commit": "abc"})

    assert [c["window"] for c in calls] == [NO_SPLIT_WINDOW]
    assert calls[0]["results"] == out
    assert sorted(p.name for p in out.iterdir()) == ["ok.csv", "window.json"], \
        "no output of a module that did not run may appear in the window's directory"

    wj = json.loads((out / "window.json").read_text())
    assert wj["window"] == {"id": "w2004-2015", "start": "2004-01-01", "end": "2015-12-31"}
    assert wj["provenance"] == {"code_commit": "abc"}
    assert wj["split_date_in_window"] is False
    assert set(wj["coverage"]) == {"first_real_response_month", "last_real_response_month",
                                   "n_real_response_months"}
    mods = wj["modules"]
    assert mods["ok"] == {"status": "run", "outputs": ["ok.csv"]}
    assert mods["undeclared"]["status"] == "not_run"
    assert mods["declared_no"] == {"status": "not_run", "reason": "makes no sense in a window"}
    assert "mhw_detection" not in mods, "MHW detection runs once on the whole record, not per window"


def test_provenance_carries_commit_and_spec_hash():
    prov = common.provenance()
    assert set(prov) == {"code_commit", "code_dirty", "study_spec", "study_spec_sha256"}
    assert len(prov["study_spec_sha256"]) == 64


# --- golden: real modules on the frozen fixture -------------------------------

def _load_build_dataset():
    spec = importlib.util.spec_from_file_location("build_dataset", REPO_ROOT / "scripts" / "build_dataset.py")
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    return bd


def _data_copy(dest: Path, perturb_after: dt.date | None) -> Path:
    """Copy the frozen fixture's data and, optionally, multiply every
    response value after `perturb_after` by 1000 -- at the source: the
    per-trial file and the per-month aggregate (the aggregation is strictly
    per month, so scaling the trials scales exactly each month's mean and CI
    bounds). The derived datasets (data_extended.csv, data_ec50_ci.csv:
    merge + imputation) are then REGENERATED by scripts/build_dataset.py
    from the altered source, so a module that read the imputations already
    computed in data/ could not hide a leak. The MHW catalogue is left as
    is: computed on the whole record by design."""
    shutil.copytree(FIXTURE_DATA, dest)
    if perturb_after is not None:
        cut = pd.Timestamp(perturb_after)
        raw = pd.read_csv(dest / "ec50_raw.csv", parse_dates=["Datetime"])
        raw.loc[raw["Datetime"] > cut, "EC50"] *= 1000
        raw.to_csv(dest / "ec50_raw.csv", index=False)
        sheets = pd.read_csv(dest / "ec50_sheets.csv", parse_dates=["Datetime"])
        late = sheets["Datetime"] > cut
        for col in ("EC50", "EC50_ci_upper", "EC50_ci_lower"):
            sheets.loc[late, col] *= 1000
        sheets.to_csv(dest / "ec50_sheets.csv", index=False)
    bd = _load_build_dataset()
    bd.ENV_PATH, bd.EC50_PATH, bd.ORIG_PATH = dest / "env_copernicus.csv", dest / "ec50_sheets.csv", dest / "data.csv"
    bd.OUT_DATA, bd.OUT_CI = dest / "data_extended.csv", dest / "data_ec50_ci.csv"
    bd.main()
    return dest


def _run_windows(data_dir: Path, results_root: Path, windows, monkeypatch) -> None:
    monkeypatch.setattr(common, "DATA", data_dir)
    for w in windows:
        pipeline.run_window(w, results_root / w.id, response=None, provenance={"fixed": "for comparison"})


def _run_whole_record(module, data_dir: Path, out: Path, monkeypatch) -> Path:
    monkeypatch.setattr(common, "DATA", data_dir)
    out.mkdir(parents=True)
    module.run(results=out)
    return out


def _assert_declared_counts_match_the_window(out: Path, w: WindowSpec, data_dir: Path) -> None:
    """Counts, dates and years written in output TEXT must come from the
    window's data. On the whole-record fixture a literal and a data-derived
    number coincide by construction (163 months, 295 trials), so only a
    window shorter than the record can tell them apart."""
    ci = pd.read_csv(data_dir / "data_ec50_ci.csv", parse_dates=["Datetime"])
    real = ci.loc[~ci["EC50_imputed"], "Datetime"]
    real = real[real.between(pd.Timestamp(w.start), pd.Timestamp(w.end))]
    raw = pd.read_csv(data_dir / "ec50_raw.csv", parse_dates=["Datetime"])
    raw = raw[raw["Datetime"].between(pd.Timestamp(w.start), pd.Timestamp(w.end))]
    n_months, n_trials = len(real), len(raw)
    n_tied = int(raw["Datetime"].duplicated(keep=False).sum())
    assert n_months < 163 and n_trials < 295, "the check is only meaningful on a window shorter than the record"

    cp = json.loads((out / "changepoint_response.json").read_text())
    assert cp["monthly_series"]["n"] == n_months
    assert cp["ordinal_sequence"]["n"] == n_trials
    assert f"its {n_months} dates" in cp["note"]
    assert f"~{n_tied} of its {n_trials} rows" in cp["note"]

    st = {r["variable"]: r for r in json.loads((out / "stationarity_results.json").read_text())}
    assert st["EC50"]["n"] == n_months

    assert len(pd.read_csv(out / "thermal_legacy.csv")) == n_months

    if (out / "period_contrast_raw.json").exists():
        pc = json.loads((out / "period_contrast_raw.json").read_text())
        assert pc["n_pre"] + pc["n_post"] == n_trials
        assert f"the {n_trials} trials" in pc["note"]
        split = pd.Timestamp(pc["split_date"])
        dist_files = list(out.glob("dist_*.csv"))
        assert dist_files
        for f in dist_files:
            d = pd.read_csv(f, parse_dates=["Datetime"])
            assert d["Datetime"].between(pd.Timestamp(w.start), pd.Timestamp(w.end)).all(), f.name
            # Every row's label must be true of that row: the span of the
            # months present in THIS file, on its side of the split.
            for side in (d["Datetime"] < split, d["Datetime"] >= split):
                months = d.loc[side, "Datetime"]
                assert set(d.loc[side, "period"]) == {f"{months.min():%Y-%m}–{months.max():%Y-%m}"}, f.name
        ec = pd.read_csv(out / "dist_EC50.csv", parse_dates=["Datetime"])
        assert sorted(ec["Datetime"]) == sorted(real), "dist_EC50.csv must hold exactly the window's real months"


@pytest.mark.golden
def test_rule_a_values_after_the_window_end_never_reach_a_window_result(tmp_path, monkeypatch):
    windows = [SPLIT_WINDOW, NO_SPLIT_WINDOW]
    base_data = _data_copy(tmp_path / "base_data", perturb_after=None)
    _run_windows(base_data, tmp_path / "base", windows, monkeypatch)

    # Both roads exercised, and neither passes vacuously.
    for w in windows:
        wj = json.loads((tmp_path / "base" / w.id / "window.json").read_text())
        for label in MIGRATED:
            entry = wj["modules"][label]
            if label == "period_split" and w is NO_SPLIT_WINDOW:
                assert entry["status"] == "skipped", "split_date outside the window: pre/post skipped"
                assert "split_date" in entry["reason"]
                assert not entry.get("outputs")
            else:
                assert entry["status"] == "run" and entry["outputs"], f"{w.id}/{label} produced nothing"
        for label in DECLARED_UNSUPPORTED:
            assert wj["modules"][label]["status"] == "not_run"
    assert json.loads((tmp_path / "base" / SPLIT_WINDOW.id / "window.json").read_text())["split_date_in_window"]
    for w in windows:
        _assert_declared_counts_match_the_window(tmp_path / "base" / w.id, w, base_data)

    for w in windows:
        pert_data = _data_copy(tmp_path / f"pert_data_{w.id}", perturb_after=w.end)

        # The perturbation is strong enough to be seen when it is NOT fenced
        # off: the same module on the whole record must change.
        from climate_change_on_sea_urchins import thermal_legacy
        a = _run_whole_record(thermal_legacy, base_data, tmp_path / f"whole_base_{w.id}", monkeypatch)
        b = _run_whole_record(thermal_legacy, pert_data, tmp_path / f"whole_pert_{w.id}", monkeypatch)
        assert not filecmp.cmp(a / "thermal_legacy.csv", b / "thermal_legacy.csv", shallow=False)

        _run_windows(pert_data, tmp_path / f"pert_{w.id}", [w], monkeypatch)
        base_dir, pert_dir = tmp_path / "base" / w.id, tmp_path / f"pert_{w.id}" / w.id
        base_files = sorted(p.name for p in base_dir.iterdir())
        assert base_files == sorted(p.name for p in pert_dir.iterdir())
        changed = [n for n in base_files if not filecmp.cmp(base_dir / n, pert_dir / n, shallow=False)]
        assert not changed, f"{w.id}: response values after {w.end} leaked into {changed}"


@pytest.mark.golden
def test_a_whole_record_window_reproduces_the_run_without_a_window(golden_pipeline_results, tmp_path, monkeypatch):
    """Point 3: a window covering the entire record must give, for every
    supported module, the same results as the run without a window, within
    the golden master's tolerances. From 5b-2 on this also checks that
    re-imputing the response inside the window matches data/'s imputation."""
    from test_golden_master import TOLERANCE_BY_FILE, _assert_csv_matches, _assert_json_matches

    whole = WindowSpec(id="whole", start=dt.date(1990, 1, 1), end=dt.date(2100, 12, 31))
    _run_windows(FIXTURE_DATA, tmp_path, [whole], monkeypatch)
    out = tmp_path / whole.id
    produced = sorted(p.name for p in out.iterdir() if p.name != "window.json")
    assert produced, "the whole-record window produced nothing"
    for name in produced:
        tol = TOLERANCE_BY_FILE[name]
        if name.endswith(".csv"):
            _assert_csv_matches(name, out, tol, ref_dir=golden_pipeline_results)
        else:
            _assert_json_matches(name, out, tol, ref_dir=golden_pipeline_results)
