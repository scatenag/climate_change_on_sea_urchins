"""M1.6: the known-truth series. A response generated from the SST anomaly with a lag the tester knows, with the
characters of the real series (about two fifths of the months missing, in runs; one to six trials per month),
run through ccsu-run-study; the lag must be found with the expected sign, and not be found when the response
does not depend on the SST. The generator declares what it did next to the response (response.manifest.json),
and ccsu-run-study copies that declaration into run.json."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from climate_change_on_sea_urchins import known_truth, run_study
from climate_change_on_sea_urchins.study_spec import load_study

ROOT = Path(__file__).resolve().parent.parent
SST = ROOT / "tests" / "fixtures" / "paper_mpb_2026" / "data" / "sst_daily.csv"
CLOCK = lambda: dt.datetime(2026, 10, 16, 9, 0, tzinfo=dt.timezone.utc)
BASELINE = (2003, 2012)


def _make(tmp_path, name="study", seed=1, **kw):
    kw.setdefault("lat", 43.4278); kw.setdefault("lon", 10.3956); kw.setdefault("baseline", BASELINE)
    return known_truth.make_study(SST, tmp_path / name, seed=seed, **kw)


def _run(tmp_path, folder_name="study", out_name="out"):
    folder = tmp_path / folder_name
    code = run_study.run_study(folder / "study.yaml", folder, tmp_path / out_name, now=CLOCK)
    assert code == 0, (tmp_path / out_name / "progress.json").read_text()
    return tmp_path / out_name


# --- what the generator writes ------------------------------------------------------------------------

def test_the_study_is_valid_and_declares_what_the_run_needs(tmp_path):
    _make(tmp_path)
    folder = tmp_path / "study"
    assert sorted(p.name for p in folder.iterdir()) == ["response.csv", "response.manifest.json", "sst_daily.csv", "study.yaml"]
    spec = load_study(folder / "study.yaml")
    assert spec.format_version == 2 and spec.responses[0].source.type == "csv"
    assert spec.responses[0].adverse_direction == "decrease"
    assert (spec.mhw_climatology.baseline_start_year, spec.mhw_climatology.baseline_end_year) == BASELINE
    assert spec.responses[0].imputation is not None            # declared, as in Livorno's study
    assert (spec.sites[0].lat, spec.sites[0].lon) == (43.4278, 10.3956)


def test_the_same_seed_gives_the_same_bytes_and_another_seed_another_series(tmp_path):
    _make(tmp_path, "a", seed=7); _make(tmp_path, "b", seed=7); _make(tmp_path, "c", seed=8)
    for name in ("response.csv", "response.manifest.json", "study.yaml"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert (tmp_path / "a" / "response.csv").read_bytes() != (tmp_path / "c" / "response.csv").read_bytes()


def test_the_declaration_says_everything_that_was_chosen(tmp_path):
    truth = _make(tmp_path, seed=3, lag=5, signal_share=0.5)
    on_disk = json.loads((tmp_path / "study" / "response.manifest.json").read_text())
    assert on_disk == truth
    assert truth["kind"] == "synthetic_response" and truth["seed"] == 3 and truth["lag_months"] == 5
    assert truth["signal_share_of_variance"] == 0.5
    assert truth["driver"] == "SST anomaly" and truth["driver_baseline"] == list(BASELINE)
    assert truth["expected_sign_of_r"] == "negative" and truth["noise_ar1"] == 0.3
    assert truth["missing"]["mean_missing_run_months"] > 0 and truth["trials_per_month"]
    assert truth["months"]["with_value"] > 0


@pytest.mark.parametrize("bad", [0, -1, 12, 24, 13])
def test_a_lag_that_is_zero_a_multiple_of_twelve_or_beyond_the_tested_range_is_refused(tmp_path, bad):
    with pytest.raises(known_truth.KnownTruthError, match="lag"):
        _make(tmp_path, lag=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_a_signal_share_outside_zero_one_is_refused(tmp_path, bad):
    with pytest.raises(known_truth.KnownTruthError, match="signal"):
        _make(tmp_path, signal_share=bad)


def test_the_series_has_the_characters_of_the_real_one(tmp_path):
    fractions, per_month = [], []
    for seed in range(30):
        t = _make(tmp_path, f"s{seed}", seed=seed)
        rows = pd.read_csv(tmp_path / f"s{seed}" / "response.csv", parse_dates=["date"])
        months = rows["date"].dt.to_period("M")
        counts = months.value_counts()
        per_month.extend(counts.tolist())
        fractions.append(1 - t["months"]["with_value"] / t["months"]["defined"])
        assert counts.min() >= 1 and counts.max() <= 6
        assert (rows["date"].dt.day.between(1, 28)).all()
    assert abs(np.mean(fractions) - 0.42) < 0.05            # the sheet has 42% of its months without a measurement
    assert 1.5 < np.mean(per_month) < 2.1                    # and 1.8 trials per month on average


def test_the_missing_months_come_in_runs_not_one_by_one(tmp_path):
    runs = []
    for seed in range(20):
        t = _make(tmp_path, f"r{seed}", seed=seed)
        months = pd.read_csv(tmp_path / f"r{seed}" / "response.csv", parse_dates=["date"])["date"].dt.to_period("M").unique()
        grid = pd.period_range(months.min(), months.max(), freq="M")
        flag = ~grid.isin(months)
        run = 0
        for f in flag:
            if f: run += 1
            elif run: runs.append(run); run = 0
    assert 2.0 < np.mean(runs) < 3.2 and max(runs) >= 6      # mean 2.5 months in the sheet, the longest 9


# --- the run carries the declaration -----------------------------------------------------------------

def test_the_run_copies_the_declaration_into_run_json(tmp_path):
    truth = _make(tmp_path, seed=2)
    out = _run(tmp_path)
    run = json.loads((out / "run.json").read_text())
    assert run["response_manifest"] == truth


def test_a_response_without_a_declaration_gives_null(tmp_path):
    _make(tmp_path)
    (tmp_path / "study" / "response.manifest.json").unlink()
    out = _run(tmp_path)
    assert json.loads((out / "run.json").read_text())["response_manifest"] is None


# --- the exit criterion of the slice --------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_the_known_lag_is_found_with_the_expected_sign(tmp_path, seed):
    _make(tmp_path, seed=seed)            # lag 3, 50% of the variance, 42% of the months missing in runs
    verdict = known_truth.verify(_run(tmp_path))
    assert verdict["recovered"], verdict
    assert verdict["top_lag"] == 3 and verdict["r_toward_adverse_at_lag"] > 0 and verdict["p_bh_at_lag"] < 0.05


@pytest.mark.parametrize("seed", range(8))
def test_a_response_independent_of_the_sst_is_not_found(tmp_path, seed):
    _make(tmp_path, seed=seed, signal_share=0.0)
    verdict = known_truth.verify(_run(tmp_path))
    assert not verdict["recovered"], verdict


def test_the_other_lags_that_pass_fdr_are_reported(tmp_path):
    _make(tmp_path, seed=0)
    verdict = known_truth.verify(_run(tmp_path))
    assert 3 in verdict["lags_significant_after_bh"] and isinstance(verdict["lags_significant_after_bh"], list)
    assert verdict["tests_performed"] >= 1 and verdict["tests_declared"] == 13


# --- the verdict itself, on runs written by hand ---------------------------------------------------------

def _fabricated(tmp_path, *, top, r_k, p_k, k=3):
    """A finished run on paper: |r| is largest at `top`; at the known lag k it is r_k (toward adverse) with BH p p_k."""
    run = tmp_path / "fake"; run.mkdir()
    r = np.full(13, 0.05)
    r[k] = r_k
    if top != k:
        r[top] = 0.4
    p_bh = np.full(13, 0.9); p_bh[k] = p_k
    ccf = pd.DataFrame({"driver": "SST anomaly", "response": "x", "lag": range(13), "n": 100, "spearman_r": -r,
                        "p_value": 0.5, "p_bh": p_bh, "r_toward_adverse": r})
    ccf.to_csv(run / "ccf.csv", index=False)
    (run / "run.json").write_text(json.dumps({"response_manifest": {"kind": "synthetic_response", "lag_months": k,
                                                                      "signal_share_of_variance": 0.5, "seed": 1},
                                              "tests": {"declared": 13, "performed": 13, "not_performed": []}}))
    return run


def test_the_verdict_needs_the_top_lag_the_sign_and_bh(tmp_path):
    cases = {"recovered": (True, dict(top=3, r_k=0.4, p_k=0.01)),
             "top lag elsewhere": (False, dict(top=5, r_k=0.2, p_k=0.01)),
             "wrong sign": (False, dict(top=3, r_k=-0.4, p_k=0.01)),
             "not significant after BH": (False, dict(top=3, r_k=0.4, p_k=0.2))}
    for name, (expected, kw) in cases.items():
        sub = tmp_path / name.replace(" ", "_"); sub.mkdir()
        assert known_truth.verify(_fabricated(sub, **kw))["recovered"] is expected, name


def test_a_run_without_a_declaration_cannot_be_verified(tmp_path):
    run = _fabricated(tmp_path, top=3, r_k=0.4, p_k=0.01)
    d = json.loads((run / "run.json").read_text()); d["response_manifest"] = None
    (run / "run.json").write_text(json.dumps(d))
    with pytest.raises(known_truth.KnownTruthError, match="declaration"):
        known_truth.verify(run)


# --- the command line -------------------------------------------------------------------------------------

def test_the_command_line_makes_and_verifies(tmp_path, capsys):
    assert known_truth.main(["make", "--sst", str(SST), "--out", str(tmp_path / "study"), "--seed", "1",
                             "--lat", "43.4278", "--lon", "10.3956", "--baseline", "2003", "2012"]) == 0
    out = _run(tmp_path)
    assert known_truth.main(["verify", str(out)]) == 0
    assert json.loads((out / "verification.json").read_text())["recovered"] is True
    assert "recovered" in capsys.readouterr().out
