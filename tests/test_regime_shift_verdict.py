"""
regime_shift_summary.json's verdict is generated from the values computed
in the run, never a fixed conclusion. Until 2026-09 it always said the
early-warning signals were absent with "AR(1) not rising", while the value
written next to it showed AR(1) rising (tau +0.14, p=0.017); and it
presented the distance between two breaks as a lag without reporting
whether the MHW break was significant at all.
"""
import contextlib
import io
import json
from pathlib import Path

import pytest

from climate_change_on_sea_urchins import common, regime_shift

FIXTURE_DATA = Path(__file__).parent / "fixtures" / "paper_mpb_2026" / "data"


def _summary(tmp_path, monkeypatch, alpha=None):
    monkeypatch.setattr(common, "DATA", FIXTURE_DATA)
    if alpha is not None:
        monkeypatch.setattr(regime_shift, "ALPHA", alpha)
    with contextlib.redirect_stdout(io.StringIO()):
        regime_shift.run(results=tmp_path)
    return json.loads((tmp_path / "regime_shift_summary.json").read_text())


def test_verdict_reports_both_breaks_with_p_and_the_date_convention(tmp_path, monkeypatch):
    s = _summary(tmp_path, monkeypatch)
    v = s["verdict"]
    assert "last month before the change 2016-05, first month after 2016-06" in v
    assert "last year before the change 2013, first year after 2014 (p=0.0055, significant" in v
    assert f"{s['exposure_precedes_response_years']} yr earlier" in v


def test_no_distance_between_breaks_when_one_is_not_significant(tmp_path, monkeypatch):
    v = _summary(tmp_path, monkeypatch, alpha=0.001)["verdict"]  # MHW break p=0.0055
    assert "(p=0.0055, not significant at 0.001)" in v
    assert "yr earlier" not in v and "yr later" not in v
    assert "No distance between the two breaks is reported" in v


def test_early_warning_sentence_agrees_with_the_flag_and_the_trends(tmp_path, monkeypatch):
    s = _summary(tmp_path, monkeypatch)
    ews, v = s["early_warning_signals"], s["verdict"]
    assert ("not detected" in v) == (not s["critical_slowing_down_detected"])
    assert f"AR(1) tau={ews['ar1_kendall_tau']:+.2f}, p={ews['ar1_p']:.2g}" in v
    assert "not rising" not in v
