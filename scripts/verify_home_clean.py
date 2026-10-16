"""Manual verification (milestone MA.2): the real Copernicus toolbox, called as the generic app calls it -- `describe`
for the coverage and `subset` for the series, with the credentials passed as arguments -- writes nothing in the home
directory, the catalogue cache included.

Run by .github/workflows/verify_slice.yml with the repository's Copernicus secrets in the environment. The check:
  1. reads the credentials from the environment and REMOVES them from it, so the toolbox cannot fall back on it;
  2. points HOME at a new empty folder (and runs in another empty folder) and imports the toolbox after that, so any
     write to the home directory, a cache or a configuration file, lands in the folder it is looked for in;
  3. also snapshots the runner's real home (the usual places: ~/.copernicusmarine, ~/.cache, ~/.config, ~/.netrc,
     ~/motuclient, ~/.local, and the top-level names) before and after;
  4. calls `CopernicusClient.coverage_of` (describe) and `download.fetch_daily_sst` for ten days at one point (subset);
  5. lists every file created or changed: if any, it says what and where, and the exit code is 1.
Nothing here is written in the repository. usage: verify_home_clean.py <output dir>"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

WATCHED = [".copernicusmarine", ".cache", ".config", ".netrc", "motuclient", ".local"]


def snapshot(root: Path, names: list[str] | None = None) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) of every file under `root` (or under the listed names), by relative path; the top-level
    names of `root` themselves are recorded as well when `names` is given."""
    out: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return out
    targets = [root / n for n in names] if names is not None else [root]
    if names is not None:
        for child in root.iterdir():          # every top-level name; a file directly in the home also by size and time
            if child.is_file() and not child.is_symlink():
                st = child.stat(); out[child.name] = (st.st_size, st.st_mtime_ns)
            else:
                out.setdefault(f"<top-level> {child.name}", (0, 0))
    for t in targets:
        if t.is_file():
            st = t.stat(); out[str(t.relative_to(root))] = (st.st_size, st.st_mtime_ns)
        elif t.is_dir():
            for p in t.rglob("*"):
                if p.is_file() and not p.is_symlink():
                    st = p.stat(); out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def changes(before: dict, after: dict) -> list[dict]:
    rows = []
    for path, meta in sorted(after.items()):
        if path not in before:
            rows.append({"path": path, "what": "created", "bytes": meta[0]})
        elif before[path] != meta:
            rows.append({"path": path, "what": "changed", "bytes": meta[0]})
    return rows


def run_check(credentials, client_factory, *, fresh_home: Path, fresh_cwd: Path, real_home: Path, days: int = 10) -> dict:
    """Call describe and subset through `client_factory()` and report what was written where. `credentials` is a
    (username, password) pair; the environment must not hold it any more when the client is built."""
    import datetime as dt
    from climate_change_on_sea_urchins import catalog, download
    from climate_change_on_sea_urchins.study_spec import SiteSpec

    before = {"fresh_home": snapshot(fresh_home), "fresh_cwd": snapshot(fresh_cwd), "real_home": snapshot(real_home, WATCHED)}
    client = client_factory()
    variable = catalog.get("sst_daily")
    coverage = client.coverage_of(variable)                                     # describe
    site = SiteSpec(id="s", lat=43.4278, lon=10.3956, name="Livorno Sud", bbox_delta=0.1)
    result = download.fetch_daily_sst(site, start=coverage.end - dt.timedelta(days=days - 1), end=coverage.end,
                                      credentials=download.Credentials(*credentials), client=client)   # subset
    after = {"fresh_home": snapshot(fresh_home), "fresh_cwd": snapshot(fresh_cwd), "real_home": snapshot(real_home, WATCHED)}
    written = {k: changes(before[k], after[k]) for k in before}
    return {"coverage": {"start": coverage.start.isoformat(), "end": coverage.end.isoformat(), "version": coverage.version},
            "days_downloaded": int(len(result.series)), "written": written,
            "clean": not any(written.values())}


def summary(report: dict) -> str:
    lines = ["## Real toolbox: nothing written in the home directory (MA.2)", "",
             f"describe and subset: {report['days_downloaded']} days of the daily SST downloaded "
             f"(product {report['coverage']['start']} to {report['coverage']['end']}, version {report['coverage']['version']}).", ""]
    names = {"fresh_home": "the new empty home directory", "fresh_cwd": "the working directory", "real_home": "the runner's real home (usual places)"}
    for key, rows in report["written"].items():
        lines.append(f"- {names[key]}: " + ("nothing created or changed" if not rows else f"**{len(rows)} file(s)**"))
        for r in rows[:50]:
            lines.append(f"    - {r['what']}: `{r['path']}` ({r['bytes']} bytes)")
    lines += ["", "### Outcome", "", "- clean" if report["clean"] else "- **FAIL: the toolbox wrote files (above)**"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    user, password = os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None), os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
    if not user or not password:
        print("error: the credentials are not in the environment", file=sys.stderr)
        return 2
    real_home = Path(os.environ.get("HOME", str(Path.home())))
    fresh_home = Path(tempfile.mkdtemp(prefix="ccsu-home-"))
    fresh_cwd = Path(tempfile.mkdtemp(prefix="ccsu-cwd-"))
    os.environ["HOME"] = str(fresh_home)
    os.chdir(fresh_cwd)

    def factory():
        from climate_change_on_sea_urchins import download
        return download.CopernicusClient()           # imports the toolbox now, with HOME already moved

    report = run_check((user, password), factory, fresh_home=fresh_home, fresh_cwd=fresh_cwd, real_home=real_home)
    text = summary(report)
    if any(secret in text or secret in json.dumps(report) for secret in (user, password)):
        print("FAIL: credentials found in the report: nothing is written", file=sys.stderr)    # before any file or summary
        return 2
    (out / "home-check.json").write_text(json.dumps(report, indent=2))
    (out / "home-check.md").write_text(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)
    print(text)
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
