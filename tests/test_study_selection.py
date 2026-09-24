"""
config.py selects the study to load from the CCSU_STUDY environment variable,
defaulting to the Livorno study. Each case runs in a subprocess: config is
loaded at import time, and reloading it in-process would leave every other
module holding the previous study's values.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LIVORNO = REPO_ROOT / "examples" / "livorno_paracentrotus" / "study.yaml"


def _config_values(env_study=None):
    env = {k: v for k, v in os.environ.items() if k != "CCSU_STUDY"}
    if env_study is not None:
        env["CCSU_STUDY"] = str(env_study)
    out = subprocess.run(
        [sys.executable, "-c",
         "import config; print(config.STUDY_ID, config.SITE_LAT, config.SITE_LON)"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    return out


def test_default_is_the_livorno_study():
    out = _config_values()
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["livorno-paracentrotus", "43.4278", "10.3956"]


def test_ccsu_study_selects_another_study(tmp_path):
    other = tmp_path / "study.yaml"
    text = LIVORNO.read_text().replace("id: livorno-paracentrotus", "id: other-study") \
                              .replace("lat: 43.4278", "lat: 42.0") \
                              .replace("lon: 10.3956", "lon: 9.5")
    other.write_text(text)

    out = _config_values(other)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["other-study", "42.0", "9.5"]


def test_ccsu_study_pointing_nowhere_fails_loudly(tmp_path):
    out = _config_values(tmp_path / "missing.yaml")
    assert out.returncode != 0
    assert "CCSU_STUDY" in out.stderr
    assert "no such file" in out.stderr
