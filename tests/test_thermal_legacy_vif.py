"""
VIF added to thermal_legacy.py's per-window output: exact for two
predictors, VIF = 1/(1 - r**2) where r is the dose_time_collinearity
already computed there.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _per_window():
    path = ROOT / "results" / "thermal_legacy_summary.json"
    if not path.exists():
        pytest.skip("results/thermal_legacy_summary.json not generated yet")
    return json.loads(path.read_text())["per_window"]


def test_vif_matches_formula():
    for row in _per_window():
        r = row["dose_time_collinearity"]
        expected = 1.0 / (1.0 - r ** 2)
        assert row["vif"] == pytest.approx(expected, rel=1e-9)


def test_vif_present_for_every_window():
    rows = _per_window()
    assert len(rows) == 5
    for row in rows:
        assert "vif" in row
        assert row["vif"] > 1.0  # VIF is always >= 1 for a correlated predictor
