"""The daily SST download ends where the Copernicus catalogue says the multiyear product
ends, never at a date written in the code.

A literal end date froze data/sst_daily.csv twice: two years behind in 2026-07 (commit
09b928a), then at 2026-06-30 while the product reached 2026-08-31. Both times every MHW
analysis silently stopped advancing. The first test reads the script's source and fails if a
date literal other than the series start comes back; the second checks how the end is read
from a catalogue description, without network and without copernicusmarine installed (CI
does not install it).
"""
import ast
import datetime as dt
import importlib.util
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "fetch_copernicus_daily.py"
DATE_LITERAL = re.compile(r"^\d{4}-\d{2}-\d{2}")


def test_no_end_date_literal_in_the_daily_sst_script():
    tree = ast.parse(SCRIPT.read_text())
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and DATE_LITERAL.match(node.value)
    }
    # The start of the series is a declared choice; any other date is an end written by hand.
    assert literals <= {"2003-01-01"}, f"date literals in {SCRIPT.name}: {sorted(literals)}"

    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "end_datetime":
            assert not isinstance(node.value, ast.Constant), "end_datetime is a literal"
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            assert "END" not in names, "a module-level END constant is back"


def _load_script(monkeypatch):
    # The script imports the download stack at module level; stub it so the pure
    # function under test can be loaded where it is not installed.
    for name in ("copernicusmarine", "xarray"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    spec = importlib.util.spec_from_file_location("fetch_copernicus_daily", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ms(day: str) -> float:
    return dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() * 1000


def _description(service_ends: dict[str, float], variable: str = "thetao"):
    ns = types.SimpleNamespace
    services = [
        ns(service_name=name, variables=[ns(short_name=variable, coordinates=[
            ns(coordinate_id="depth", maximum_value=5754.0),
            ns(coordinate_id="time", maximum_value=end),
        ])])
        for name, end in service_ends.items()
    ]
    part = ns(name="default", services=services)
    return ns(products=[ns(datasets=[ns(versions=[ns(label="202511", parts=[part])])])])


def test_end_is_read_from_the_catalogue(monkeypatch):
    module = _load_script(monkeypatch)
    description = _description({"arco-time-series": _ms("2026-08-31")})
    assert module.coverage_end(description, "thetao") == "2026-08-31"


def test_services_disagreeing_take_the_earliest_end(monkeypatch):
    module = _load_script(monkeypatch)
    description = _description({
        "arco-time-series": _ms("2026-08-31"),
        "arco-geo-series": _ms("2026-07-31"),
    })
    assert module.coverage_end(description, "thetao") == "2026-07-31"


def test_no_coverage_for_the_variable_fails_loudly(monkeypatch):
    module = _load_script(monkeypatch)
    description = _description({"arco-time-series": _ms("2026-08-31")}, variable="so")
    try:
        module.coverage_end(description, "thetao")
    except RuntimeError as e:
        assert "thetao" in str(e)
    else:
        raise AssertionError("no error for a variable the catalogue does not cover")
