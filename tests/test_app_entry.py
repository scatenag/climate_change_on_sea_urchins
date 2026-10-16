"""MA.1: the entry of the generic app. `app.py` is the real entry point (Streamlit Cloud points at it): it routes
the choice between the sea-urchin case (today's dashboard, unchanged) and the user's own data (three steps, empty
in this piece). The study context of a session lives in st.session_state and is passed explicitly; the session's
working folder is swept after 6 hours without use. Run through streamlit.testing from app.py, with no credentials
and none of their variables in the environment."""
from __future__ import annotations

import ast
import os
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from climate_change_on_sea_urchins import app_entry, own_data, step_site

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "climate_change_on_sea_urchins"
TIMEOUT = 120


@pytest.fixture(autouse=True)
def _no_credentials_and_private_sessions_root(monkeypatch, tmp_path):
    for var in [v for v in os.environ if v.startswith("COPERNICUSMARINE")]:
        monkeypatch.delenv(var)
    monkeypatch.setattr(app_entry, "SESSIONS_ROOT", tmp_path / "sessions")
    monkeypatch.setattr(step_site, "acquisition_available", lambda: False)      # no toolbox, no network in these tests


def _app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)


def _texts(at):
    return " ".join(e.value for kind in (at.markdown, at.text, at.caption, at.header, at.subheader) for e in kind)


def _button(at, label_part):
    return next(b for b in at.button if label_part in b.label)


# --- the choice ---------------------------------------------------------------------------------------

def test_the_entry_offers_the_two_ways_in_and_nothing_else_yet():
    at = _app().run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert any("sea-urchin" in l for l in labels) and any("own data" in l for l in labels)
    assert not [r for r in at.radio if r.label == "Section"]            # today's dashboard is not rendered


def test_the_sea_urchin_case_renders_todays_dashboard():
    at = _app().run()
    _button(at, "sea-urchin").click().run()
    assert not at.exception
    assert [t.value for t in at.title] == ["Climate Response Explorer"]
    assert [r for r in at.radio if r.label == "Section"]              # the dashboard's own navigation, as today
    at.run()                                                          # a second run is a real fresh run, not a blank page
    assert not at.exception and [r for r in at.radio if r.label == "Section"]


def test_the_own_data_way_shows_three_steps_that_are_not_available_yet():
    at = _app().run()
    _button(at, "own data").click().run()
    assert not at.exception
    text = _texts(at)
    for step in ("Step 1", "Step 2", "Step 3"):
        assert step in text
    assert "not available yet" in text.lower()
    assert not [r for r in at.radio if r.label == "Section"]


def test_changing_the_choice_leaves_nothing_of_the_other():
    at = _app().run()
    _button(at, "sea-urchin").click().run()
    assert "active_main_tab" in at.session_state                        # the dashboard's own widget state
    at.session_state["leftover_of_the_dashboard"] = 1                   # and state that is no widget's
    _button(at, "Change").click().run()
    assert not at.exception
    assert "active_main_tab" not in at.session_state and "leftover_of_the_dashboard" not in at.session_state
    assert app_entry.CONTEXT_KEY in at.session_state                    # the session's own context stays
    _button(at, "own data").click().run()
    assert not at.exception and not [r for r in at.radio if r.label == "Section"]


# --- the context of a session -------------------------------------------------------------------------

def test_two_sessions_in_one_process_each_show_their_own_study(monkeypatch):
    seen = []
    real = own_data.render
    monkeypatch.setattr(own_data, "render", lambda ctx: (seen.append(ctx), real(ctx))[1])
    a, b = _app(), _app()
    a.session_state[app_entry.CONTEXT_KEY] = app_entry.SessionContext.new("EC50")
    b.session_state[app_entry.CONTEXT_KEY] = app_entry.SessionContext.new("RSPTEST")
    for at in (a, b):
        at.run(); _button(at, "own data").click().run()
        assert not at.exception
    assert "EC50" in _texts(a) and "RSPTEST" not in _texts(a)
    assert "RSPTEST" in _texts(b) and "EC50" not in _texts(b)
    ctxs = {c.session_id: c for c in seen}
    assert len(ctxs) == 2
    assert len({c.workdir for c in ctxs.values()}) == 2 and all(c.workdir.is_dir() for c in ctxs.values())


def test_the_context_is_created_once_per_session_and_keeps_its_folder():
    at = _app().run()
    ctx = at.session_state[app_entry.CONTEXT_KEY]
    at.run()
    assert at.session_state[app_entry.CONTEXT_KEY] is ctx and ctx.workdir.is_dir()
    assert ctx.workdir.parent == app_entry.SESSIONS_ROOT


def test_delete_my_data_removes_the_folder_and_starts_over(tmp_path):
    at = _app().run()
    _button(at, "own data").click().run()
    ctx = at.session_state[app_entry.CONTEXT_KEY]
    (ctx.workdir / "something.csv").write_text("a,b\n")
    _button(at, "Delete my data").click().run()
    assert not at.exception
    assert not (ctx.workdir / "something.csv").exists()
    assert at.session_state[app_entry.CONTEXT_KEY].workdir.is_dir()


# --- the sweep of old session folders -----------------------------------------------------------------------

def _folder(root, name, age_hours):
    d = root / name
    d.mkdir(parents=True)
    (d / "x.csv").write_text("x")
    alive = d / app_entry.ALIVE_FILE
    alive.write_text("")
    t = time.time() - age_hours * 3600
    os.utime(alive, (t, t))
    return d


def test_the_sweep_removes_what_has_been_idle_for_six_hours_and_nothing_else(tmp_path):
    root = tmp_path / "root"
    old = _folder(root, "a" * 32, 7)
    fresh = _folder(root, "b" * 32, 5)
    mine = _folder(root, "c" * 32, 9)
    not_a_session = root / "keep-me"; not_a_session.mkdir()
    (root / "file.txt").write_text("x")
    for p in (not_a_session, root / "file.txt"):
        os.utime(p, (time.time() - 30 * 3600,) * 2)             # old, but not session folders
    outside = tmp_path / "outside"; outside.mkdir(); (outside / "precious").write_text("x")
    os.utime(outside, (time.time() - 30 * 3600,) * 2)               # idle for days, behind a link named like a session
    (root / ("d" * 32)).symlink_to(outside, target_is_directory=True)
    removed = app_entry.sweep_stale_sessions(root, keep=mine.name)
    assert removed == [old.name]
    assert not old.exists() and fresh.exists() and mine.exists()
    assert not_a_session.exists() and (root / "file.txt").exists()
    assert (outside / "precious").exists()


def test_a_session_that_comes_back_is_alive_again(tmp_path):
    root = tmp_path / "root"
    d = _folder(root, "e" * 32, 8)
    app_entry.touch_alive(d)
    assert app_entry.sweep_stale_sessions(root, keep="z" * 32) == [] and d.exists()


def test_the_sweep_runs_when_a_session_starts(tmp_path):
    old = _folder(app_entry.SESSIONS_ROOT, "f" * 32, 8)
    _app().run()
    assert not old.exists()


# --- the guards ---------------------------------------------------------------------------------------------

FORBIDDEN = {"DATA", "RESULTS", "SPLIT_DATE", "WINDOWS", "config", "cache_data", "cache_resource", "cache"}


def _names(path):
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name): found.add(node.id)
        elif isinstance(node, ast.Attribute): found.add(node.attr)
        elif isinstance(node, ast.ImportFrom): found.add(node.module or ""); found.update(a.name for a in node.names)
        elif isinstance(node, ast.Import): found.update(a.name for a in node.names)
    return found


@pytest.mark.parametrize("module", ["app_entry.py", "own_data.py"])
def test_the_new_modules_do_not_read_the_loaded_study_or_cache(module):
    """§5.1: nothing of a study in module state; no st.cache_* in the entry (it will never see credentials)."""
    assert not (_names(PKG / module) & FORBIDDEN)


@pytest.mark.parametrize("module", ["app_entry.py", "own_data.py"])
def test_the_new_modules_have_no_interface_at_module_level(module):
    """They are imported once; the interface runs in functions, called at every run (the blank-page bug of app.py)."""
    tree = ast.parse((PKG / module).read_text())
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            assert not (isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name)
                        and node.value.func.value.id == "st"), f"{module}: st.{node.value.func.attr} at module level"


def test_the_lock_and_the_reimport_stay_in_the_sea_urchin_branch_only():
    tree = ast.parse((ROOT / "app.py").read_text())
    def mentions(node, what):
        return any((isinstance(n, ast.Attribute) and n.attr == what) or (isinstance(n, ast.Name) and n.id == what)
                   or (isinstance(n, ast.Import) and any(a.name.endswith(".dashboard") for a in n.names)) for n in ast.walk(node))
    top = [n for n in tree.body if not isinstance(n, ast.If)]
    assert not any(isinstance(n, ast.With) or (isinstance(n, ast.Import) and mentions(n, "dashboard")) for n in top)
    branches = [n for n in tree.body if isinstance(n, ast.If)]
    assert len(branches) == 1 and mentions(branches[0], "RERUN_LOCK") and mentions(branches[0], "dashboard")
