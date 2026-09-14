"""
Tests that the package reproduces the manuscript's published numbers,
section by section, with a declared tolerance -- the invariant CLAUDE.md
calls out: "Nessun risultato scientifico cambia senza approvazione
esplicita", verified here rather than just asserted.

Runs the four manuscript-value modules (negative_control, period_split,
thermal_legacy, mhw_annual_changepoint) against the frozen data snapshot in
tests/fixtures/paper_mpb_2026/data/ -- not against the live, ever-growing
data/ (which would make CI red the moment the auto-update workflow adds a
month, per STATO.md's own "Da non dimenticare"), and not against a stale
results/ (which could silently drift from what the current code actually
produces). See conftest.py's paper_results fixture for how the re-run is
wired without touching any analysis module.
"""
import json

import pandas as pd
import pytest


def _results(results_dir, name):
    return json.loads((results_dir / name).read_text())


# ---------------------------------------------------------------------------
# Section 3.6 -- negative control (Compito A)
# ---------------------------------------------------------------------------

def test_negative_control_trial_counts(paper_results):
    r = _results(paper_results, "negative_control.json")
    assert r["n_trials_total"] == 295
    assert r["n_with_negative_control"] == 232
    assert r["n_without_negative_control"] == 63


def test_negative_control_trend_is_open_not_accepted(paper_results):
    # NOT a rounding difference: trial ID 224 (2020-01-01, replicates
    # [14, 1, 14]) is an uncorrected source-data outlier -- see
    # negative_control.py's module docstring and this result's own "note"
    # field. This module computes on the data exactly as fetched (outlier
    # included, unmodified); the manuscript's own p=0.30 is recorded
    # alongside it for comparison, not asserted as reproduced. Both values
    # are well above any conventional significance threshold, so the
    # qualitative conclusion (no trend) is unaffected either way -- but the
    # numeric mismatch stays open pending Davide's decision on the source
    # row, not silently accepted here.
    r = _results(paper_results, "negative_control.json")["trend"]
    assert r["status"] == "open"
    assert r["spearman_rho"] == pytest.approx(0.0706, abs=0.001)
    assert r["spearman_p"] == pytest.approx(0.2844, abs=0.001)
    assert r["manuscript_spearman_rho"] == pytest.approx(0.07, abs=0.001)
    assert r["manuscript_spearman_p"] == pytest.approx(0.30, abs=0.001)


def test_negative_control_replicate_outlier_flag_catches_trial_224(paper_results):
    # Data-quality check added alongside the trend investigation above: a
    # single replicate deviating from the other two of its trial by more
    # than 3 pooled-SD should be flagged -- verified to catch exactly the
    # known case (trial 224) and nothing else on the current data.
    r = _results(paper_results, "negative_control.json")["replicate_outlier_flags"]
    assert r["n_flagged"] == 1
    assert r["flags"][0]["trial_id"] == 224
    assert r["flags"][0]["date"] == "2020-01-01"
    assert r["flags"][0]["replicate_values"] == [14.0, 1.0, 14.0]


def test_negative_control_pre_post_at_published_split(paper_results):
    # The manuscript's published section 3.6 pre/post numbers correspond to
    # the 2016-01-01 split, not the current SPLIT_DATE (2016-06-01) -- see
    # negative_control.py's module docstring. Both splits agree
    # qualitatively (no significant level or dispersion change); this test
    # checks the split that reproduces the manuscript's own numbers.
    r = _results(paper_results, "negative_control.json")["pre_post_published_split"]
    assert r["mean_pre"] == pytest.approx(13.7, abs=0.1)
    assert r["mean_post"] == pytest.approx(14.1, abs=0.1)
    assert r["mannwhitney_p"] == pytest.approx(0.41, abs=0.01)
    # Manuscript's Levene p=0.38 is scipy's plain default call
    # (center='median'), not a deliberately chosen robust variant.
    assert r["levene_p"]["median"] == pytest.approx(0.38, abs=0.01)


def test_negative_control_changepoint(paper_results):
    # trim=0.10 is the primary result -- the module default trim=0.15
    # (used for the EC50 series) excludes this break from its search
    # window; see negative_control.py's module docstring.
    r = _results(paper_results, "negative_control.json")["changepoint_qlr_ar1"]["trim_0.10_primary"]
    assert r["break_date"].startswith("2022-10")
    assert r["n_pre"] == 203
    assert r["n_post"] == 29
    assert r["mean_pre"] == pytest.approx(13.8, abs=0.1)
    assert r["mean_post"] == pytest.approx(14.9, abs=0.1)
    assert r["p_value"] == pytest.approx(0.12, abs=0.02)


# ---------------------------------------------------------------------------
# Section 3.1 -- pre/post contrast on individual bioassay trials (Compito B)
# ---------------------------------------------------------------------------

def test_period_contrast_raw_trial_level(paper_results):
    r = _results(paper_results, "period_contrast_raw.json")
    assert r["n_pre"] == 176
    assert r["n_post"] == 119
    assert r["mean_pre"] == pytest.approx(46.54, abs=0.01)
    assert r["mean_post"] == pytest.approx(26.53, abs=0.01)
    assert r["sd_pre"] == pytest.approx(7.64, abs=0.01)
    assert r["sd_post"] == pytest.approx(5.54, abs=0.01)
    assert r["median_pre"] == pytest.approx(46.32, abs=0.01)
    assert r["median_post"] == pytest.approx(26.49, abs=0.01)
    assert r["diff_abs"] == pytest.approx(20.01, abs=0.01)
    assert r["decline_pct"] == pytest.approx(43.0, abs=0.1)
    assert r["mannwhitney_p"] == pytest.approx(1.5e-43, rel=0.1)


# ---------------------------------------------------------------------------
# Table S2 -- thermal threshold sensitivity (Compito C)
# ---------------------------------------------------------------------------

def _sensitivity_row(results_dir, threshold_c):
    df = pd.read_csv(results_dir / "thermal_threshold_sensitivity.csv")
    row = df[df["threshold_C"] == threshold_c]
    assert len(row) == 1, f"expected exactly one row for threshold_C={threshold_c}"
    return row.iloc[0]


@pytest.mark.parametrize("threshold_c,rho,p,partial_p,survives", [
    (22.0, -0.16, 0.038, 0.75, False),
    (23.0, -0.19, 0.015, 0.11, False),
    (24.0, -0.26, 0.0009, 0.0042, True),
    (25.0, -0.33, 0.0001, 0.0007, True),
    (26.0, -0.33, 0.0001, 0.0039, True),
])
def test_thermal_threshold_sensitivity_pattern(paper_results, threshold_c, rho, p, partial_p, survives):
    # The pattern that must hold unambiguously (per task instructions): fails
    # at 22/23C, survives Bonferroni (both the rank and the parametric test)
    # from 24C up, with the association strengthening. Exact third-decimal
    # values are explicitly expected to drift as the underlying data is
    # realigned (see task context) -- tolerances here are loose on purpose,
    # tight on the pattern (survives/fails) instead.
    row = _sensitivity_row(paper_results, threshold_c)
    assert row["detrended_spearman_r"] == pytest.approx(rho, abs=0.03)
    assert row["detrended_p"] == pytest.approx(p, abs=p if p < 0.01 else 0.01)
    assert row["partial_p_dose_given_time"] == pytest.approx(partial_p, abs=max(partial_p * 0.3, 0.005))
    assert bool(row["bonferroni_survives"]) == survives


def test_thermal_threshold_sensitivity_24C_is_primary_and_matches_main_analysis(paper_results):
    # threshold_C=24 is the a-priori primary value (Amato et al. 2025), not
    # one of five equally-weighted alternatives -- and its numbers must be
    # identical to the 24-month-window row thermal_legacy_summary.json
    # already reports (same computation, independent code path).
    row = _sensitivity_row(paper_results, 24.0)
    assert bool(row["is_primary_threshold"]) is True

    summary = _results(paper_results, "thermal_legacy_summary.json")
    w24 = next(r for r in summary["per_window"] if r["window_months"] == 24)
    assert row["detrended_spearman_r"] == pytest.approx(w24["detrended_spearman_r"], abs=1e-6)
    assert row["detrended_p"] == pytest.approx(w24["detrended_p"], abs=1e-6)
    assert row["partial_p_dose_given_time"] == pytest.approx(w24["partial_p_dose_given_time"], abs=1e-6)


# ---------------------------------------------------------------------------
# Section 3.5, 2nd paragraph -- annual MHW-metric changepoint (Compito D)
#
# CLOSED 2026-09-14: D. Sartori removed the paragraph this was written to
# reproduce from the manuscript (this module's own investigation is what
# surfaced the discrepancy that led to the removal). There is no longer a
# manuscript number to reproduce, so this only checks the module still
# records that plainly and keeps its variant table intact -- not a
# reproduction test any more.
# ---------------------------------------------------------------------------

def test_mhw_annual_changepoint_no_longer_cited(paper_results):
    r = _results(paper_results, "mhw_annual_changepoint.json")
    assert r["cited_in_manuscript"] is False
    assert len(r["variants"]) == 8, "expected 2 metrics x 4 year ranges, kept as a record"
