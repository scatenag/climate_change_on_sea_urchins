"""Manual verification of the vertical slice (milestone M1.6): on coordinates never used by the project, download
the real SST, generate a response of known lag from it, run ccsu-run-study, and check the lag is found.

Run by .github/workflows/verify_slice.yml with the repository's Copernicus secrets in the environment of this
process; the credentials are given only to the download (as ccsu-download-sst reads them) and the outputs are
checked not to contain them. Writes nothing in the repository: everything goes under the output directory.
Per point: download from 2003-01-01 -> known_truth make (lag 3, 50% of the variance, baseline 2003-2012) for each
seed -> ccsu-run-study -> known_truth verify. The verdict is the one of known_truth.verify.
usage: verify_slice.py <output dir> [seeds, default 1 2 3]
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

POINTS = [  # open sea, inside the Mediterranean product, never used by the project (Livorno, Pisa, Black Sea are)
    dict(name="adriatico", label="Central Adriatic", lat=42.5, lon=15.0),
    dict(name="ionio", label="Ionian Sea", lat=38.5, lon=18.5),
]
LAG, SHARE, BASELINE = 3, 0.5, (2003, 2012)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def run_point(point: dict, seeds: list[int], out: Path) -> dict:
    folder = out / point["name"]
    sst = folder / "sst"
    sst.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("ccsu-download-sst") or "ccsu-download-sst"
    t0 = time.perf_counter()
    d = run([exe, "--lat", str(point["lat"]), "--lon", str(point["lon"]), "--name", point["label"],
             "--start", "2003-01-01", "--out", str(sst)], env=os.environ.copy())
    result = {"point": point, "download": {"returncode": d.returncode, "seconds": round(time.perf_counter() - t0, 1),
                                           "stdout": d.stdout.strip(), "stderr": d.stderr.strip()}, "seeds": []}
    (folder / "download.txt").write_text(d.stdout + d.stderr)
    if d.returncode:
        return result
    for seed in seeds:
        study = folder / f"seed{seed}" / "study"
        res = folder / f"seed{seed}" / "run"
        m = run([sys.executable, "-m", "climate_change_on_sea_urchins.known_truth", "make", "--sst", str(sst / "sst_daily.csv"),
                 "--out", str(study), "--seed", str(seed), "--lat", str(point["lat"]), "--lon", str(point["lon"]),
                 "--baseline", str(BASELINE[0]), str(BASELINE[1]), "--lag", str(LAG), "--signal-share", str(SHARE)])
        r = run([sys.executable, "-m", "climate_change_on_sea_urchins.run_study", str(study / "study.yaml"),
                 "--inputs", str(study), "--out", str(res)]) if m.returncode == 0 else None
        v = run([sys.executable, "-m", "climate_change_on_sea_urchins.known_truth", "verify", str(res)]) if r is not None and r.returncode == 0 else None
        verdict = json.loads((res / "verification.json").read_text()) if v is not None and (res / "verification.json").exists() else None
        result["seeds"].append({"seed": seed, "make": m.returncode, "run": None if r is None else r.returncode,
                                "verify": None if v is None else v.returncode, "verdict": verdict,
                                "messages": [x.stdout.strip() + x.stderr.strip() for x in (m, r, v) if x is not None]})
    return result


def summary(results: list[dict]) -> tuple[str, list[str]]:
    failures, lines = [], ["## Vertical slice check (M1.6)", "",
                           f"Known lag {LAG} months, {SHARE:.0%} of the variance, baseline {BASELINE[0]}-{BASELINE[1]}, "
                           "response generated from the downloaded SST anomaly.", ""]
    for r in results:
        p = r["point"]
        lines += [f"### {p['label']} ({p['lat']}, {p['lon']})", f"- download: exit {r['download']['returncode']}, {r['download']['seconds']} s",
                  f"- `{r['download']['stdout'] or r['download']['stderr']}`", "",
                  "| seed | recovered | largest \\|r\\| at | BH p at the known lag | other lags significant after BH |", "|---|---|---|---|---|"]
        if r["download"]["returncode"]:
            failures.append(f"{p['label']}: download failed ({r['download']['stderr'][:200]})")
        for s in r["seeds"]:
            v = s["verdict"]
            if v is None:
                failures.append(f"{p['label']} seed {s['seed']}: no verdict ({' | '.join(s['messages'])[:300]})")
                lines.append(f"| {s['seed']} | no verdict | | | |")
                continue
            others = [x for x in v["lags_significant_after_bh"] if x != LAG]
            lines.append(f"| {s['seed']} | {'yes' if v['recovered'] else '**NO**'} | {v['top_lag']} | {v['p_bh_at_lag']:.2g} | {others} |")
            if not v["recovered"]:
                failures.append(f"{p['label']} seed {s['seed']}: lag {LAG} not recovered (largest |r| at {v['top_lag']})")
        lines.append("")
    lines += ["### Outcome", ""] + ([f"- FAIL: {f}" for f in failures] if failures else ["- the known lag was recovered in every case"])
    return "\n".join(lines) + "\n", failures


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    out = Path(argv[0])
    seeds = [int(x) for x in argv[1:]] or [1, 2, 3]
    out.mkdir(parents=True, exist_ok=True)
    results = [run_point(p, seeds, out) for p in POINTS]
    text, failures = summary(results)
    (out / "summary.md").write_text(text)
    (out / "summary.json").write_text(json.dumps({"results": results, "failures": failures}, indent=2))
    secrets = [v for v in (os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"), os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")) if v]
    for f in out.rglob("*"):
        if f.is_file() and f.suffix in (".txt", ".md", ".json", ".yaml", ".csv") and any(s in f.read_text(errors="ignore") for s in secrets):
            print(f"FAIL: credentials found in {f.name}", file=sys.stderr)
            return 2
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(text)
    print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
