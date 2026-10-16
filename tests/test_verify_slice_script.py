"""scripts/verify_slice.py (the manual check of M1.6) end to end without the network: a stand-in for
ccsu-download-sst puts the frozen fixture SST where the real download would, and the script must generate, run
and verify, and report. The real service is not touched here (that is the workflow's job)."""
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("verify_slice", ROOT / "scripts" / "verify_slice.py")
vs = importlib.util.module_from_spec(spec); spec.loader.exec_module(vs)
SST = ROOT / "tests" / "fixtures" / "paper_mpb_2026" / "data" / "sst_daily.csv"


def _fake_downloader(folder: Path, *, ok=True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / "ccsu-download-sst"
    exe.write_text(f"""#!{sys.executable}
import shutil, sys
from pathlib import Path
a = sys.argv[1:]
if not {ok!r}:
    print("error: no data", file=sys.stderr); raise SystemExit(1)
out = Path(a[a.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
shutil.copyfile({str(SST)!r}, out / "sst_daily.csv")
print("8644 days written")
""")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return folder


def test_the_script_downloads_generates_runs_and_verifies(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(_fake_downloader(tmp_path / "bin")) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    assert vs.main([str(tmp_path / "out"), "1"]) == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["failures"] == [] and [r["point"]["name"] for r in summary["results"]] == ["adriatico", "ionio"]
    assert all(s["verdict"]["recovered"] for r in summary["results"] for s in r["seeds"])
    assert all((s["verdict"]["known_lag"], s["verdict"]["signal_share_of_variance"]) == (3, 0.5) for r in summary["results"] for s in r["seeds"])
    assert "recovered in every case" in (tmp_path / "out" / "summary.md").read_text()


def test_a_failed_download_is_reported_not_hidden(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(_fake_downloader(tmp_path / "bin", ok=False)) + os.pathsep + os.environ["PATH"])
    assert vs.main([str(tmp_path / "out"), "1"]) == 1
    assert "download failed" in (tmp_path / "out" / "summary.md").read_text()


def test_the_credentials_in_an_output_are_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(_fake_downloader(tmp_path / "bin", ok=False)) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "no data")        # a value that the outputs do contain
    assert vs.main([str(tmp_path / "out"), "1"]) == 2
