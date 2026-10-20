"""Step 3 of the user's own data: run the study and read the result (milestone MA.4, docs/roadmap/MILESTONE-M1.md; ADR-0018).

The analysis runs in a separate process, `ccsu-run-study` (`python -m climate_change_on_sea_urchins.run_study`), started
from the session's study folder and writing into the session's `run` folder. That process receives FILES ONLY: its
environment is built from an allowlist (no Copernicus variable, no credential, no variable of the app's process that is not
named here, a fresh empty HOME) and its arguments are paths. It never sees the credentials of step 1, which the app process
emptied from the page the moment the download button was pressed.

The page follows the run by reading `progress.json` (written atomically by the command at every step), with a fragment that
reruns itself every two seconds while the process runs, outside the global lock; at most MAX_CONCURRENT runs go on at once on a
server process (counted under a lock, and by the pid files of the session folders, each checked against the start time of its
process; the others are told to wait); a run is stopped after MAX_SECONDS, can be interrupted, and is stopped when the user
changes choice, deletes the data, downloads again, builds another study, or the session folder is swept.

What the page says about the result comes from the values (`ccf.csv`, `run.json`): the tests declared, performed and not
performed; the lags that pass the Benjamini-Hochberg correction and the sign of the association relative to the direction
declared as adverse; for a test series with a known lag, the verdict of `known_truth.verify`. The limits of the method are
stated every time. Only functions here; what the step produces lives in `ctx.data` and in the session folder."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from . import known_truth

MAX_CONCURRENT = 2
MAX_SECONDS = 600                     # an analysis that takes longer is stopped (the real ones take seconds)
_START_LOCK = threading.Lock()       # the count of running analyses and the start of one are one step
RUN_DIR = "run"
PID_FILE = "run.pid"
LOG_FILE = "run.log"
POLL_SECONDS = 2
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "PYTHONPATH", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "SYSTEMROOT")
STEP_NAMES = {"load_study": "reading the study", "read_response": "reading the response", "read_sst": "reading the temperature",
              "build_dataset": "building the dataset (monthly table and heatwaves)", "correlate": "correlating",
              "write_results": "writing the results"}
LIMITS = (
    "It looks for the **non-seasonal association only**: the driver is the monthly temperature anomaly, so a dependence that goes "
    "through the seasonal part of the temperature alone is not seen.",
    "It correlates **first differences of consecutive real months** (a month without a measurement, or one that follows one "
    "without, gives no pair): the number of pairs at every lag is in the table.",
    "The p values are **Spearman's, corrected by Benjamini-Hochberg over the lags that were actually tested** (not the 13 declared "
    "when some had too few pairs). The first differences are autocorrelated, which makes each p a little too small: in a simulation on "
    "a series like the Livorno one (a response independent of the temperature, about half the months missing, 13 lags) 6-8% of "
    "analyses showed a lag that passes the correction; with a shorter series or another autocorrelation it can differ.",
    "The temperature is the mean over the grid cells around the site (about 4 km each): near an island, a lagoon or a strait it may "
    "not represent the water your organisms live in.",
    "An association is not a cause, and a lag that does not pass the correction is not evidence of no effect.",
)


# ── the process ──────────────────────────────────────────────────────────────────────────────────

def child_env(parent: dict, home: Path) -> dict:
    """The environment of the analysis process: the allowlisted variables of `parent`, a fresh empty HOME, and nothing else."""
    env = {k: v for k, v in parent.items() if k in ENV_ALLOWLIST}
    env["HOME"] = str(home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    return env


def build_command(study: Path, inputs: Path, out: Path) -> list[str]:
    return [sys.executable, "-m", "climate_change_on_sea_urchins.run_study", str(study), "--inputs", str(inputs), "--out", str(out)]


def _starttime(pid: int) -> str | None:
    """The start time of a process in clock ticks since boot (Linux), or None when it cannot be read or the process is gone or a
    zombie: a pid file is trusted only together with it, because a pid alone can be reused by an unrelated process."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[-1].split()
    except OSError:
        return None
    return None if fields[0] == "Z" else fields[19]


def write_pid_file(path: Path, pid: int) -> None:
    path.write_text(f"{pid} {_starttime(pid) or ''}".strip())


def _recorded(pid_file: Path) -> int | None:
    """The pid of the analysis the file records if that very process is still running, else None."""
    try:
        parts = pid_file.read_text().split()
        pid = int(parts[0])
    except (OSError, ValueError, IndexError):
        return None
    if len(parts) < 2:
        return None                        # without the start time we cannot tell it is ours
    return pid if _starttime(pid) == parts[1] else None


def running_elsewhere(root: Path) -> int:
    """How many analyses are running now under the session folders of `root`."""
    return sum(1 for f in root.glob(f"*/{PID_FILE}") if _recorded(f) is not None)


def kill_recorded(folder: Path) -> None:
    """Stop the analysis a session folder records, if it still runs (used when the folder is swept)."""
    pid = _recorded(folder / PID_FILE)
    if pid is not None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)
            except OSError:
                break
            time.sleep(0.2)
            if _starttime(pid) is None:
                break


def start(ctx, sessions_root: Path) -> str | None:
    """Start the analysis of the session's study. Returns None if it started, else why it did not."""
    study = ctx.workdir / "study"
    if not (study / "study.yaml").is_file():
        return "Build the study in step 2 first."
    if running(ctx):
        return "An analysis is already running in this session."
    reset(ctx)
    out = ctx.workdir / RUN_DIR
    home = ctx.workdir / "home"
    home.mkdir(exist_ok=True)
    with _START_LOCK:                       # the count and the start are one step: two sessions cannot both take the last slot
        if running_elsewhere(sessions_root) >= MAX_CONCURRENT:
            return (f"The server is running {MAX_CONCURRENT} analyses already: wait a moment and press the button again. "
                    "Each takes from a few seconds to a minute.")
        try:
            with open(ctx.workdir / LOG_FILE, "wb") as log:
                proc = subprocess.Popen(build_command(study / "study.yaml", study, out), env=child_env(dict(os.environ), home),
                                        cwd=str(home), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            write_pid_file(ctx.workdir / PID_FILE, proc.pid)
        except OSError as e:
            (ctx.workdir / PID_FILE).unlink(missing_ok=True)
            return f"The analysis could not be started ({type(e).__name__}: {str(e).replace(str(ctx.workdir), '<session folder>')})."
    ctx.data["run"] = {"proc": proc, "started": time.time(), "interrupted": False, "timed_out": False}
    return None


def running(ctx) -> bool:
    run = ctx.data.get("run")
    return bool(run) and run["proc"].poll() is None


def stop(ctx) -> None:
    """Stop the analysis of the session if it is running (its whole process group)."""
    run = ctx.data.get("run")
    if run and run["proc"].poll() is None:
        try:
            os.killpg(run["proc"].pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            run["proc"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(run["proc"].pid, signal.SIGKILL)
            except OSError:
                pass
        run["interrupted"] = True
    (ctx.workdir / PID_FILE).unlink(missing_ok=True)


def reset(ctx) -> None:
    """The analysis, its results and its process go: what they were made from has changed."""
    stop(ctx)
    ctx.data.pop("run", None)
    shutil.rmtree(ctx.workdir / RUN_DIR, ignore_errors=True)
    (ctx.workdir / LOG_FILE).unlink(missing_ok=True)


def read_state(ctx) -> dict:
    """What the page needs: status (starting, running, done, failed, interrupted), the step, the error, the elapsed seconds."""
    run = ctx.data.get("run")
    if not run:
        return {"status": "none"}
    progress_file = ctx.workdir / RUN_DIR / "progress.json"
    progress = {}
    if progress_file.is_file():
        try:
            progress = json.loads(progress_file.read_text())
        except (OSError, ValueError):
            progress = {}
    returncode = run["proc"].poll()
    elapsed = round(time.time() - run["started"])
    if returncode is None and elapsed > MAX_SECONDS:
        stop(ctx)
        run["timed_out"] = True
        returncode = run["proc"].poll()
    if returncode is None:
        return {"status": "running" if progress else "starting", "step": progress.get("step"), "steps": progress.get("steps", []),
                "elapsed": elapsed}
    (ctx.workdir / PID_FILE).unlink(missing_ok=True)
    if run.get("timed_out"):
        return {"status": "failed", "step": progress.get("step"), "elapsed": elapsed,
                "error": f"The analysis took longer than {MAX_SECONDS // 60 or 1} minute(s) and was stopped."}
    if run.get("interrupted"):
        return {"status": "interrupted", "elapsed": elapsed}
    if progress.get("status") == "done" and returncode == 0:
        return {"status": "done", "elapsed": elapsed}
    if progress.get("error"):
        error = str(progress["error"])[:1500]                         # the head of the command's own message
    else:
        try:
            error = (ctx.workdir / LOG_FILE).read_text(errors="replace")[-1500:]      # the tail of what it printed
        except OSError:
            error = ""
        error = error or f"the process exited with code {returncode}"
    return {"status": "failed", "step": progress.get("step"), "error": error.replace(str(ctx.workdir), "<session folder>"),
            "elapsed": elapsed}


# ── what the result says ─────────────────────────────────────────────────────────────────────────

def significant_lags(ccf: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    tested = ccf[ccf["p_value"].notna()]
    return tested[tested["p_bh"] < alpha].sort_values("lag")


def describe_result(ccf: pd.DataFrame, run: dict) -> str:
    """A sentence from the values, never a fixed conclusion."""
    tests = run["tests"]
    head = (f"{tests['performed']} of {tests['declared']} lags were tested "
            f"({len(tests['not_performed'])} had too few pairs).")
    if tests["performed"] == 0:
        return head + " No lag could be tested: nothing can be said about the association."
    if tests["performed"] < tests["declared"]:
        head += f" The correction is over the {tests['performed']} tested."
    sig = significant_lags(ccf)
    if sig.empty:
        return head + " No lag passes the Benjamini-Hochberg correction (p corrected < 0.05)."
    parts = []
    for r in sig.itertuples():
        way = "toward the adverse direction" if r.r_toward_adverse > 0 else "away from the adverse direction"
        parts.append(f"lag {int(r.lag)} (r = {r.spearman_r:+.2f}, after a rise of the anomaly the response moves {way}, "
                     f"p corrected = {r.p_bh:.3g}, {int(r.n)} pairs)")
    return head + " Passing the Benjamini-Hochberg correction: " + "; ".join(parts) + "."


def describe_verdict(verdict: dict) -> str:
    k = verdict["known_lag"]
    others = [lag for lag in verdict["lags_significant_after_bh"] if lag != k]
    if verdict["recovered"]:
        text = (f"The known lag was FOUND: the largest |r| is at lag {verdict['top_lag']} = {k}, with the expected sign "
                f"(r toward adverse = {verdict['r_toward_adverse_at_lag']:+.2f}) and corrected p = {verdict['p_bh_at_lag']:.2g}.")
    else:
        top, r_adv, p = verdict["top_lag"], verdict["r_toward_adverse_at_lag"], verdict["p_bh_at_lag"]
        if top is None or p is None:
            why = "the known lag was not tested (too few pairs)"
        elif top != k:
            why = f"the largest |r| is at lag {top}, not at {k}"
        elif r_adv is not None and r_adv <= 0:
            why = "the sign is the opposite of the expected one"
        else:
            why = f"corrected p is {p:.2f}, not below 0.05"
        text = f"The known lag ({k}) was NOT found: {why}."
    if others:
        text += " Other lags also pass the correction: " + ", ".join(str(o) for o in others) + "."
    return text


# ── the interface ────────────────────────────────────────────────────────────────────────────────

def _show_result(ctx) -> None:
    out = ctx.workdir / RUN_DIR
    ccf = pd.read_csv(out / "ccf.csv")
    run = json.loads((out / "run.json").read_text())
    if run["tests"]["performed"] == 0:
        st.warning("The analysis ran, but it could not test any lag.")
    else:
        st.success("The analysis finished.")
    st.write(describe_result(ccf, run))
    if run.get("response_manifest"):
        st.info(describe_verdict(known_truth.verify(out)) + " (The response of this study was generated with that lag.)")
    st.caption("Spearman r between the first difference of the temperature anomaly k months earlier and of the response, by lag.")
    st.bar_chart(ccf.set_index("lag")["spearman_r"], height=220)
    table = ccf.rename(columns={"n": "pairs", "spearman_r": "r", "r_toward_adverse": "r toward adverse", "p_value": "p", "p_bh": "p corrected"})
    st.dataframe(table[["lag", "pairs", "r", "r toward adverse", "p", "p corrected"]], hide_index=True)
    p = run["parameters"]
    st.caption(f"Driver: {p['driver']} against the climatology {p['climatology_baseline'][0]}–{p['climatology_baseline'][1]}; "
               f"response: `{p['response_label']}` (adverse direction: {p['adverse_direction']}); lags 0 to {p['max_lag_months']}; "
               f"transform: {p['transform']}; pairs: {p['pairs']}; correction: {p['correction']}.")
    excluded = run["coverage"]["temperature"]["incomplete_months"]
    if excluded:
        st.caption(f"{len(excluded)} month(s) with a missing temperature day have no monthly temperature.")
    st.markdown("**What this result is, and is not**")
    for line in LIMITS:
        st.markdown(f"- {line}")


@st.fragment(run_every=POLL_SECONDS)
def _follow(ctx) -> None:
    state = read_state(ctx)
    if state["status"] in ("starting", "running"):
        label = STEP_NAMES.get(state.get("step"), "starting") if state["status"] == "running" else "starting"
        st.info(f"Running: {label} ({state['elapsed']} s).")
        done = sum(1 for s in state.get("steps", []) if s["status"] == "done")
        if state.get("steps"):
            st.progress(done / len(state["steps"]))
        if st.button("Interrupt the analysis", key="s3_interrupt"):
            stop(ctx)
            st.rerun()
    else:
        st.rerun()


def render(ctx) -> None:
    st.markdown("**Step 3 — Run and read the result**")
    if not (ctx.workdir / "study" / "study.yaml").is_file():
        st.info("Build the study in step 2 first.")
        return
    state = read_state(ctx)
    if state["status"] in ("none", "interrupted", "failed", "done"):
        label = "Run the analysis again" if state["status"] != "none" else "Run the analysis"
        if st.button(label, key="s3_run"):
            problem = start(ctx, ctx.workdir.parent)           # the session folders are siblings under the sessions root
            if problem:
                st.warning(problem)
            state = read_state(ctx)
    if state["status"] in ("starting", "running"):
        _follow(ctx)
    elif state["status"] == "interrupted":
        st.warning("The analysis was interrupted.")
    elif state["status"] == "failed":
        st.error(f"The analysis did not complete at the step {STEP_NAMES.get(state.get('step'), state.get('step') or 'start')}:")
        st.code(str(state["error"])[-1500:], language=None)
    elif state["status"] == "done":
        _show_result(ctx)
