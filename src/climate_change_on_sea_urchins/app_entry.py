"""Entry of the generic app (MA.1). `app.py` calls `render()` at every run: it shows the choice between the sea-urchin
case and the user's own data, keeps the study context of the session, and returns the choice so that `app.py` can run
today's dashboard for the sea-urchin case (with its lock and forced reimport, which belong to that branch only).

Only functions here, no interface at module level: this module is imported once, like any library, and the interface
runs when `render()` is called. The study context lives in `st.session_state` and is passed to the pieces explicitly
(MILESTONE-M1.md, 5.1); nothing of a study is read from `common` and nothing is cached.

The working folder of a session: one per session under the system temporary folder, never shared. Streamlit has no
reliable end-of-session event, so a folder is removed when its session has not been seen for 6 hours (checked when any
session starts) or when its user presses *Delete my data*."""
from __future__ import annotations

import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from . import own_data

CONTEXT_KEY = "ccsu_ctx"
CHOICE_KEY = "ccsu_choice"
KEEP_KEYS = {CONTEXT_KEY, CHOICE_KEY}
RICCI, OWN = "sea-urchin", "own-data"

SESSIONS_ROOT = Path(tempfile.gettempdir()) / "ccsu-sessions"
ALIVE_FILE = ".alive"
MAX_IDLE_SECONDS = 6 * 3600
_SESSION_FOLDER = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class SessionContext:
    """What a session knows about its study; passed to every piece of the interface."""
    session_id: str
    workdir: Path
    label: str | None = None
    data: dict = field(default_factory=dict)      # what the steps have produced in this session (never credentials)

    @classmethod
    def new(cls, label: str | None = None) -> "SessionContext":
        session_id = uuid.uuid4().hex
        workdir = SESSIONS_ROOT / session_id
        workdir.mkdir(parents=True, exist_ok=True)
        touch_alive(workdir)
        return cls(session_id, workdir, label)


def touch_alive(folder: Path) -> None:
    (folder / ALIVE_FILE).touch()


def sweep_stale_sessions(root: Path | None = None, *, keep: str | None = None, now: float | None = None,
                         max_idle: float = MAX_IDLE_SECONDS) -> list[str]:
    """Remove the session folders under `root` not seen for `max_idle` seconds; return their names. Only direct
    children named like a session id are touched: symbolic links, files and anything else are left alone, and the
    folder named `keep` (the session starting) never goes."""
    root = SESSIONS_ROOT if root is None else Path(root)
    if not root.is_dir():
        return []
    now = time.time() if now is None else now
    removed = []
    for entry in sorted(root.iterdir()):
        if entry.is_symlink() or not entry.is_dir() or not _SESSION_FOLDER.match(entry.name) or entry.name == keep:
            continue
        alive = entry / ALIVE_FILE
        last_seen = alive.stat().st_mtime if alive.is_file() else entry.stat().st_mtime
        if now - last_seen > max_idle:
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name)
    return removed


def session_context() -> SessionContext:
    """The context of this session: created on the first run (sweeping the old folders first), marked alive at
    every run."""
    ctx = st.session_state.get(CONTEXT_KEY)
    if ctx is None:
        sweep_stale_sessions()
        ctx = SessionContext.new()
        st.session_state[CONTEXT_KEY] = ctx
    ctx.workdir.mkdir(parents=True, exist_ok=True)
    touch_alive(ctx.workdir)
    return ctx


def _choose(choice: str) -> None:
    st.session_state[CHOICE_KEY] = choice


def _change_choice() -> None:
    """Back to the entry, leaving nothing of the other way: every key but the session's own goes."""
    for key in [k for k in st.session_state if k not in KEEP_KEYS]:
        del st.session_state[key]
    st.session_state[CHOICE_KEY] = None


def _delete_my_data() -> None:
    ctx = st.session_state[CONTEXT_KEY]
    shutil.rmtree(ctx.workdir, ignore_errors=True)
    st.session_state[CONTEXT_KEY] = SessionContext.new()


def render() -> str | None:
    """Show the entry or the way chosen; return the choice (None at the entry). For the sea-urchin case only the
    way back is shown here: the dashboard itself is run by `app.py`."""
    ctx = session_context()
    choice = st.session_state.get(CHOICE_KEY)
    if choice is None:
        st.title("Climate Response Explorer")
        st.write("How does an environmental series relate, with a delay, to a biological response? Choose where to start.")
        left, right = st.columns(2)
        left.button("The sea-urchin case", on_click=_choose, args=(RICCI,))
        right.button("Your own data", on_click=_choose, args=(OWN,))
        return None
    st.button("← Change choice", on_click=_change_choice)
    if choice == OWN:
        own_data.render(ctx)
        st.button("Delete my data", on_click=_delete_my_data)
    return choice
