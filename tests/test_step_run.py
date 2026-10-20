"""MA.4: step 3 of the user's own data -- run the study in a separate process and read the result. The process receives files
only (an environment built from an allowlist, a fresh empty HOME, paths as arguments); the page follows its progress file; the
result is described from the values, with the limits of the method stated and, for a test series, the verdict of the known lag.
Through streamlit.testing from app.py with real ccsu-run-study subprocesses on the frozen fixture."""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from climate_change_on_sea_urchins import app_entry, known_truth, step_run, step_site
from test_step_response import (after_step1, _build, _csv, _ctx, _errors, _full_flow, _generate, _pick, _shown, _trials,  # noqa: F401
                                _start_upload, _fill_answers)
from test_step_site import _open  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "climate_change_on_sea_urchins"
SECRET = "S3cr3t-pa55-w0rd-for-the-env"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for var in [v for v in os.environ if v.startswith("COPERNICUSMARINE")]:
        monkeypatch.delenv(var)
    monkeypatch.setattr(app_entry, "SESSIONS_ROOT", tmp_path / "sessions")
    monkeypatch.setattr(step_site, "acquisition_available", lambda: True)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(ROOT / "src"), str(ROOT)]))


def _wait(ctx, timeout=120):
    """Until the analysis process of the session has ended."""
    proc = ctx.data["run"]["proc"]
    proc.wait(timeout=timeout)


def _run_button(at):
    return next(b for b in at.button if b.label in ("Run the analysis", "Run the analysis again"))


def _run_to_the_end(at):
    _run_button(at).click().run()
    _wait(_ctx(at))
    at.run()
    return at


# --- the order of the steps --------------------------------------------------------------------------------

def test_step_3_asks_for_a_study_first():
    at = _open()
    assert "Build the study in step 2 first" in _shown(at) and not [b for b in at.button if "Run the analysis" in b.label]


# --- a real run, end to end ---------------------------------------------------------------------------------

def test_the_test_series_is_run_and_its_known_lag_is_found_and_said(after_step1):
    at = _generate(after_step1, seed=1, lag=3, share=0.5)
    at = _run_to_the_end(at)
    assert not at.exception, [e.value for e in at.exception]
    out = _ctx(at).workdir / "run"
    ccf = pd.read_csv(out / "ccf.csv")
    shown = _shown(at)
    assert "The analysis finished" in shown and "The known lag was FOUND" in shown and "lag 3" in shown
    assert len(at.dataframe) >= 1
    table = at.dataframe[-1].value            # the last one: step 1 shows its own table of events above
    assert table["lag"].tolist() == ccf["lag"].tolist() and table["pairs"].tolist() == ccf["n"].tolist()
    assert known_truth.verify(out)["recovered"]


def test_the_limits_of_the_method_are_stated_every_time(after_step1):
    at = _run_to_the_end(_generate(after_step1))
    shown = _shown(at)
    for phrase in ("non-seasonal association only", "first differences of consecutive real months", "Benjamini-Hochberg",
                   "autocorrelated", "not a cause"):
        assert phrase in shown, phrase


def test_for_data_without_a_known_lag_the_sentence_comes_from_the_values_and_no_verdict_is_given(after_step1, monkeypatch):
    at = _full_flow(after_step1, monkeypatch, _csv(_trials(n_months=120, seed=4)))
    at = _run_to_the_end(at)
    assert not at.exception
    ccf = pd.read_csv(_ctx(at).workdir / "run" / "ccf.csv")
    sig = step_run.significant_lags(ccf)
    shown = _shown(at)
    assert "were tested" in shown and "FOUND" not in shown and "known lag" not in shown
    if sig.empty:
        assert "No lag passes the Benjamini-Hochberg correction" in shown
    else:
        for lag in sig["lag"]:
            assert f"lag {int(lag)} (r = " in shown


def test_the_sentence_names_the_direction_against_the_declared_adverse_one():
    ccf = pd.DataFrame({"lag": [0, 1, 2], "n": [50, 50, 50], "spearman_r": [0.1, -0.4, 0.5], "p_value": [0.5, 0.001, 0.0005],
                        "p_bh": [0.5, 0.01, 0.01], "r_toward_adverse": [-0.1, 0.4, -0.5]})
    run = {"tests": {"declared": 3, "performed": 3, "not_performed": []}}
    text = step_run.describe_result(ccf, run)
    assert "lag 1 (r = -0.40, toward the adverse direction" in text and "lag 2 (r = +0.50, away from the adverse direction" in text
    assert "lag 0" not in text
    skipped = {"tests": {"declared": 13, "performed": 11, "not_performed": [{"lag": 12}, {"lag": 11}]}}
    assert "11 of 13 lags were tested (2 had too few pairs)" in step_run.describe_result(ccf, skipped)
    none = ccf.assign(p_bh=0.5)
    assert "No lag passes" in step_run.describe_result(none, run)
    assert "0 of 3" not in text and "3 of 3 lags were tested" in text


def test_a_verdict_that_did_not_find_the_lag_says_so_with_the_numbers():
    v = {"recovered": False, "known_lag": 3, "top_lag": 5, "p_bh_at_lag": 0.4, "lags_significant_after_bh": [5], "r_toward_adverse_at_lag": 0.1}
    text = step_run.describe_verdict(v)
    assert "NOT found" in text and "lag 5" in text and "0.4" in text and "[5]" in text
    found = {**v, "recovered": True, "top_lag": 3, "p_bh_at_lag": 0.001, "lags_significant_after_bh": [3], "r_toward_adverse_at_lag": 0.5}
    assert "was FOUND" in step_run.describe_verdict(found)


# --- the process receives files only ---------------------------------------------------------------------------

def test_the_environment_of_the_process_is_built_from_an_allowlist(tmp_path):
    parent = {"PATH": "/usr/bin", "PYTHONPATH": "/x", "HOME": "/home/me", "COPERNICUSMARINE_SERVICE_PASSWORD": SECRET,
              "COPERNICUS_USER": "u", "SECRET_TOKEN": SECRET, "AWS_SECRET_ACCESS_KEY": SECRET, "CCSU_STUDY": "x", "LANG": "C"}
    env = step_run.child_env(parent, tmp_path / "home")
    assert set(env) == {"PATH", "PYTHONPATH", "LANG", "HOME", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS"}
    assert env["HOME"] == str(tmp_path / "home") and SECRET not in repr(env)


def test_a_variable_of_the_app_process_does_not_reach_the_analysis_and_nothing_secret_is_an_argument(after_step1, monkeypatch):
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", SECRET)
    monkeypatch.setenv("MY_APP_SECRET", SECRET)
    seen = []
    real = subprocess.Popen
    def spy(cmd, *a, **k):
        seen.append((list(cmd), dict(k.get("env") or {}), k.get("cwd")))
        return real(cmd, *a, **k)
    monkeypatch.setattr(step_run.subprocess, "Popen", spy)
    at = _run_to_the_end(_generate(after_step1))
    assert len(seen) == 1
    cmd, env, cwd = seen[0]
    assert SECRET not in " ".join(cmd) + repr(env) and "MY_APP_SECRET" not in env
    assert cmd[1:3] == ["-m", "climate_change_on_sea_urchins.run_study"] and all(Path(c).is_absolute() for c in (cmd[3], cmd[5], cmd[7]))
    assert env["HOME"] != os.environ["HOME"] and Path(env["HOME"]).is_dir() and Path(cwd) == _ctx(at).workdir


def test_the_environment_is_read_only_in_the_function_that_builds_the_allowlist():
    tree = ast.parse((PKG / "step_run.py").read_text())
    users = [f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
             for n in ast.walk(f) if isinstance(n, ast.Attribute) and n.attr == "environ"]
    assert set(users) <= {"start"}                  # os.environ is handed to child_env, which keeps the allowlisted names only


# --- progress, interruption, failure, the limit -------------------------------------------------------------------

SLOW = [sys.executable, "-c", "import time, json, pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'progress.json').write_text(json.dumps({'status': 'running', 'step': 'correlate', 'steps': [{'name': 'load_study', 'status': 'done'}, "
        "{'name': 'correlate', 'status': 'running'}]}))\ntime.sleep(60)"]


def _slow(monkeypatch):
    monkeypatch.setattr(step_run, "build_command", lambda study, inputs, out: SLOW + ["--out", str(out)])


def test_the_page_follows_the_progress_and_the_global_lock_is_free_meanwhile(after_step1, monkeypatch):
    import climate_change_on_sea_urchins as ccsu
    _slow(monkeypatch)
    at = _generate(after_step1)
    _run_button(at).click().run()
    deadline = time.time() + 20
    while time.time() < deadline and not (_ctx(at).workdir / "run" / "progress.json").exists():
        time.sleep(0.2)
    at.run()
    try:
        assert not at.exception and "Running: correlating" in _shown(at)
        assert not ccsu.RERUN_LOCK.locked()
        assert any("Interrupt" in b.label for b in at.button)
    finally:
        step_run.stop(_ctx(at))


def test_an_analysis_can_be_interrupted(after_step1, monkeypatch):
    _slow(monkeypatch)
    at = _generate(after_step1)
    _run_button(at).click().run()
    proc = _ctx(at).data["run"]["proc"]
    t0 = time.time()
    next(b for b in at.button if "Interrupt" in b.label).click().run()
    assert proc.poll() is not None and time.time() - t0 < 4          # SIGTERM does it; the SIGKILL fallback waits 5 s
    at.run()
    assert "was interrupted" in _shown(at)
    assert not (_ctx(at).workdir / "run.pid").exists()


def test_changing_the_choice_or_deleting_the_data_stops_the_analysis(after_step1, monkeypatch):
    _slow(monkeypatch)
    at = _generate(after_step1)
    _run_button(at).click().run()
    proc = _ctx(at).data["run"]["proc"]
    next(b for b in at.button if "Change" in b.label).click().run()
    assert proc.poll() is not None
    at2 = _generate(_after_again(monkeypatch))
    _run_button(at2).click().run()
    proc2 = _ctx(at2).data["run"]["proc"]
    next(b for b in at2.button if "Delete my data" in b.label).click().run()
    assert proc2.poll() is not None


def _after_again(monkeypatch):
    import datetime as dt
    from test_step_site import FIXTURE_SST, RecordingClient, COVERAGE_2003, _fill, _submit
    sst = pd.read_csv(FIXTURE_SST, parse_dates=["Datetime"]).set_index("Datetime")["Temperature"]
    c = RecordingClient(coverage=COVERAGE_2003.__class__(start=dt.date(2003, 1, 1), end=sst.index.max().date(), version="202511"), series=sst)
    monkeypatch.setattr(step_site, "make_client", lambda: c)
    return _submit(_fill(_open()))


def test_a_new_study_or_a_new_download_stops_and_drops_the_analysis(after_step1, monkeypatch):
    _slow(monkeypatch)
    at = _generate(after_step1)
    _run_button(at).click().run()
    proc = _ctx(at).data["run"]["proc"]
    at = _generate(at, seed=2)                      # another study: the analysis of the old one goes
    assert proc.poll() is not None and "run" not in _ctx(at).data and not (_ctx(at).workdir / "run").exists()


def test_a_new_download_in_step_1_stops_and_drops_the_analysis(after_step1, monkeypatch):
    from test_step_site import _fill, _submit
    _slow(monkeypatch)
    at = _generate(after_step1)
    _run_button(at).click().run()
    proc = _ctx(at).data["run"]["proc"]
    _submit(_fill(at, name="Another place"))
    assert proc.poll() is not None and "run" not in _ctx(at).data and not (_ctx(at).workdir / "run").exists()


def test_a_failing_analysis_says_where_and_why(after_step1, monkeypatch):
    fail = [sys.executable, "-c", "import sys, json, pathlib\nout = pathlib.Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True)\n"
            "(out / 'progress.json').write_text(json.dumps({'status': 'failed', 'step': 'read_response', 'error': 'the response file is broken'}))\nsys.exit(1)"]
    monkeypatch.setattr(step_run, "build_command", lambda study, inputs, out: fail + ["--out", str(out)])
    at = _run_to_the_end(_generate(after_step1))
    assert not at.exception and "did not complete at the step" in _shown(at) + " ".join(e.value for e in at.error)
    assert "the response file is broken" in "\n".join(c.value for c in at.code)
    assert any("again" in b.label for b in at.button)


def test_a_process_that_dies_without_a_progress_file_still_gives_a_reason(after_step1, monkeypatch):
    die = [sys.executable, "-c", "import sys; print('boom from the process'); sys.exit(3)"]
    monkeypatch.setattr(step_run, "build_command", lambda study, inputs, out: die)
    at = _run_to_the_end(_generate(after_step1))
    assert "boom from the process" in "\n".join(c.value for c in at.code)


def _fake_running_sessions(root: Path, n: int):
    procs = []
    for i in range(n):
        d = root / (f"{i:032x}")
        d.mkdir(parents=True)
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        (d / "run.pid").write_text(str(p.pid))
        procs.append(p)
    return procs


def test_the_third_analysis_on_a_server_waits(after_step1, monkeypatch):
    at = _generate(after_step1)
    others = _fake_running_sessions(_ctx(at).workdir.parent, step_run.MAX_CONCURRENT)
    try:
        _run_button(at).click().run()
        assert "already" in _shown(at) and "wait a minute" in _shown(at) and "run" not in _ctx(at).data
    finally:
        for p in others:
            p.kill(); p.wait()
    _run_button(at).click().run()                    # one slot is free again
    assert "run" in _ctx(at).data
    step_run.stop(_ctx(at))


def test_finished_processes_do_not_count_against_the_limit(tmp_path):
    d = tmp_path / ("a" * 32); d.mkdir()
    p = subprocess.Popen([sys.executable, "-c", "pass"]); p.wait()
    (d / "run.pid").write_text(str(p.pid))
    assert step_run.running_elsewhere(tmp_path) == 0
    (tmp_path / ("b" * 32)).mkdir(); (tmp_path / ("b" * 32) / "run.pid").write_text("not a number")
    assert step_run.running_elsewhere(tmp_path) == 0


def test_two_sessions_have_their_own_analysis_and_results(after_step1, monkeypatch):
    a = _run_to_the_end(_generate(after_step1))
    b = _open()
    assert "The analysis finished" not in _shown(b) and not (_ctx(b).workdir / "run").exists()


def test_the_streamlit_lower_bound_has_the_fragments_the_page_uses():
    import tomllib
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert any(d.replace(" ", "").startswith("streamlit>=1.37") for d in deps)
    assert re.search(r"(?m)^streamlit>=1\.37,", (ROOT / "requirements.txt").read_text())


@pytest.mark.parametrize("forbidden", [r"\bprint\(", r"logging", r"st\.cache", r"\bshell\s*=\s*True", r"\bos\.system\(", r"COPERNICUS", r"password"])
def test_the_step_has_no_shell_cache_log_or_credential_in_its_code(forbidden):
    tree = ast.parse((PKG / "step_run.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    assert not re.search(forbidden, ast.unparse(tree), re.IGNORECASE), forbidden
