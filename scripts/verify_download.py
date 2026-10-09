"""Manual verification of ccsu-download-sst against the real Copernicus service (milestone M1.2).

Run by .github/workflows/verify_download.yml with the repository's Copernicus secrets in the environment
of this process; the credentials are never printed and the outputs are checked not to contain them.
Writes nothing in the repository: everything goes under the output directory given as the argument.

Three cases, with the outcomes the milestone expects:
  - Livorno: a series that equals data/sst_daily.csv on the days both have a value (max |diff| < 1e-4 degC);
  - Pisa, a land cell near the coast: refused as a land cell, the nearest sea cell about ten km away;
  - the Black Sea, inside the domain's bounding box: refused with the message that does not choose between
    an inland point and a sea the product does not cover; if it returns data, that is a failure to report.
usage: verify_download.py <output dir> [path to data/sst_daily.csv]
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

CASES = [
    dict(name="livorno", label="Livorno", lat=43.4278, lon=10.3956, start="2003-01-01"),
    dict(name="pisa", label="Pisa", lat=43.72, lon=10.45, start="2020-01-01"),
    dict(name="mar_nero", label="Mar Nero", lat=43.0, lon=31.0, start="2020-01-01"),
]
MAX_DIFF = 1e-4
PISA_KM = (5, 25)


def compare_series(downloaded: Path, reference: Path) -> dict:
    """Days with a value in both series, and the largest absolute difference on them."""
    a = pd.read_csv(downloaded, parse_dates=["Datetime"]).set_index("Datetime")["Temperature"]
    b = pd.read_csv(reference, parse_dates=["Datetime"]).set_index("Datetime")["Temperature"]
    both = a.dropna().index.intersection(b.dropna().index)
    return {"common_days": int(len(both)), "max_abs_diff": float((a[both] - b[both]).abs().max()) if len(both) else None,
            "downloaded_days": int(a.notna().sum()), "reference_days": int(b.notna().sum()),
            "downloaded_end": a.dropna().index.max().date().isoformat()}


def nearest_sea_km(message: str):
    m = re.search(r"about (\d+) km away", message)
    return int(m.group(1)) if m else None


def run_case(case: dict, out: Path) -> dict:
    folder = out / case["name"]
    folder.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("ccsu-download-sst") or "ccsu-download-sst"
    cmd = [exe, "--lat", str(case["lat"]), "--lon", str(case["lon"]), "--name", case["label"],
           "--start", case["start"], "--out", str(folder)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    seconds = time.perf_counter() - t0
    (folder / "stdout.txt").write_text(r.stdout)
    (folder / "stderr.txt").write_text(r.stderr)
    return {"returncode": r.returncode, "seconds": round(seconds, 1), "stdout": r.stdout, "stderr": r.stderr}


def check(results: dict, livorno_cmp: dict | None) -> list[str]:
    """The expectations of the milestone; the list of those not met."""
    failures = []
    L = results["livorno"]
    if L["returncode"] != 0:
        failures.append(f"Livorno: download failed ({L['stderr'].strip()[:300]})")
    elif not livorno_cmp or not livorno_cmp["common_days"]:
        failures.append("Livorno: no common days with data/sst_daily.csv")
    elif livorno_cmp["max_abs_diff"] >= MAX_DIFF:
        failures.append(f"Livorno: max |diff| {livorno_cmp['max_abs_diff']:.3g} is not below {MAX_DIFF}")
    P = results["pisa"]
    km = nearest_sea_km(P["stderr"])
    if P["returncode"] == 0:
        failures.append("Pisa: returned data instead of being refused")
    elif "land in the product" not in P["stderr"]:
        failures.append(f"Pisa: refused, but not as a land cell ({P['stderr'].strip()[:300]})")
    elif km is None or not PISA_KM[0] <= km <= PISA_KM[1]:
        failures.append(f"Pisa: nearest sea cell at {km} km, expected about ten (between {PISA_KM[0]} and {PISA_KM[1]})")
    B = results["mar_nero"]
    if B["returncode"] == 0:
        failures.append("Mar Nero: RETURNED DATA instead of being refused")
    elif "inland, or in a sea the product does not cover" not in B["stderr"]:
        failures.append(f"Mar Nero: refused, but with another message ({B['stderr'].strip()[:300]})")
    return failures


def summary(results: dict, livorno_cmp, failures: list[str]) -> str:
    L, P, B = results["livorno"], results["pisa"], results["mar_nero"]
    lines = ["## Download check (M1.2)", "", "| Case | Exit | Seconds |", "|---|---|---|"]
    for k, label in (("livorno", "Livorno"), ("pisa", "Pisa"), ("mar_nero", "Mar Nero")):
        lines.append(f"| {label} | {results[k]['returncode']} | {results[k]['seconds']} |")
    lines += ["", "### Livorno against `data/sst_daily.csv`"]
    if livorno_cmp:
        lines += [f"- common days (value in both): **{livorno_cmp['common_days']}**",
                  f"- max |diff|: **{livorno_cmp['max_abs_diff']}** degC (limit {MAX_DIFF})",
                  f"- days with a value: downloaded {livorno_cmp['downloaded_days']}, reference {livorno_cmp['reference_days']}; "
                  f"downloaded series ends {livorno_cmp['downloaded_end']}",
                  f"- download time: **{L['seconds']} s**"]
    else:
        lines.append("- no comparison (download failed)")
    lines += ["", "### Pisa (43.72, 10.45)", "```", (P["stderr"] or P["stdout"]).strip(), "```",
              "", "### Mar Nero (43.0, 31.0)", "```", (B["stderr"] or B["stdout"]).strip(), "```", "",
              "### Outcome", ""]
    lines += [f"- FAIL: {f}" for f in failures] if failures else ["- all three as expected"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    out = Path(argv[0])
    reference = Path(argv[1]) if len(argv) > 1 else Path("data/sst_daily.csv")
    out.mkdir(parents=True, exist_ok=True)
    results = {c["name"]: run_case(c, out) for c in CASES}
    livorno_cmp = None
    csv = out / "livorno" / "sst_daily.csv"
    if results["livorno"]["returncode"] == 0 and csv.exists():
        livorno_cmp = compare_series(csv, reference)
    failures = check(results, livorno_cmp)
    text = summary(results, livorno_cmp, failures)
    (out / "summary.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(
        {"livorno": livorno_cmp, "seconds": {k: v["seconds"] for k, v in results.items()},
         "pisa_message": results["pisa"]["stderr"].strip(), "mar_nero_message": results["mar_nero"]["stderr"].strip(),
         "failures": failures}, indent=2))
    # The credentials must not be anywhere in the outputs.
    secrets = [v for v in (os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"), os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")) if v]
    for f in out.rglob("*"):
        if f.is_file() and any(s in f.read_text(errors="ignore") for s in secrets):
            print(f"FAIL: credentials found in {f.name}", file=sys.stderr)
            return 2
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(text)
    print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
