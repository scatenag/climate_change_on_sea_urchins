"""Step 3 of the user's own data: run the study and read the result (milestone MA.4, docs/roadmap/MILESTONE-M1.md; ADR-0018).

The analysis runs in a separate process, `ccsu-run-study` (`python -m climate_change_on_sea_urchins.run_study`), started
from the session's study folder and writing into the session's `run` folder. That process receives FILES ONLY: its
environment is built from an allowlist (no Copernicus variable, no credential, no variable of the app's process that is not
named here, a fresh empty HOME) and its arguments are paths. It never sees the credentials of step 1, which the app process
emptied from the page the moment the download button was pressed.

The page follows the run by reading `progress.json` (written atomically by the command at every step), with a fragment that
reruns itself every two seconds while the process runs, outside the global lock; at most MAX_CONCURRENT runs go on at once on a
server (the others are told to wait); a run can be interrupted, and is stopped when the user changes choice, deletes the data,
downloads again or builds another study.

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
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from . import known_truth

MAX_CONCURRENT = 2
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
    "The p values are **Spearman's, corrected by Benjamini-Hochberg over the lags that were tested**; the first differences are "
    "autocorrelated, which makes each p a little too small: with a response that does not depend on the temperature, about 6-8% "
    "of analyses show a lag that passes the correction (ADR-0000, item 20).",
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


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:        # a zombie is dead for our purpose
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[-1].split()[0] != "Z"
    except OSError:
        return True


def running_elsewhere(root: Path) -> int:
    """How many analyses are running now under the session folders of `root`."""
    n = 0
    for pid_file in root.glob(f"*/{PID_FILE}"):
        try:
            pid = int(pid_file.read_text().strip())
        except (OSError, ValueError):
            continue
        if _alive(pid):
            n += 1
    return n


def start(ctx, sessions_root: Path) -> str | None:
    """Start the analysis of the session's study. Returns None if it started, else why it did not."""
    study = ctx.workdir / "study"
    if not (study / "study.yaml").is_file():
        return "Build the study in step 2 first."
    if running(ctx):
        return "An analysis is already running in this session."
    if running_elsewhere(sessions_root) >= MAX_CONCURRENT:
        return (f"The server is running {MAX_CONCURRENT} analyses already: wait a minute and press the button again. "
                "Each takes about a minute.")
    reset(ctx)
    out = ctx.workdir / RUN_DIR
    home = ctx.workdir / "home"
    home.mkdir(exist_ok=True)
    log = open(ctx.workdir / LOG_FILE, "wb")
    try:
        proc = subprocess.Popen(build_command(study / "study.yaml", study, out), env=child_env(dict(os.environ), home), cwd=str(ctx.workdir),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        log.close()
    (ctx.workdir / PID_FILE).write_text(str(proc.pid))
    ctx.data["run"] = {"proc": proc, "started": time.time(), "interrupted": False}
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
    if returncode is None:
        return {"status": "running" if progress else "starting", "step": progress.get("step"), "steps": progress.get("steps", []),
                "elapsed": elapsed}
    (ctx.workdir / PID_FILE).unlink(missing_ok=True)
    if run.get("interrupted"):
        return {"status": "interrupted", "elapsed": elapsed}
    if progress.get("status") == "done" and returncode == 0:
        return {"status": "done", "elapsed": elapsed}
    log = ""
    try:
        log = (ctx.workdir / LOG_FILE).read_text(errors="replace")[-1500:]
    except OSError:
        pass
    return {"status": "failed", "step": progress.get("step"), "error": progress.get("error") or log or f"the process exited with code {returncode}",
            "elapsed": elapsed}


# ── what the result says ─────────────────────────────────────────────────────────────────────────

def significant_lags(ccf: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    tested = ccf[ccf["p_value"].notna()]
    return tested[tested["p_bh"] < alpha].sort_values("lag")


def describe_result(ccf: pd.DataFrame, run: dict) -> str:
    """A sentence from the values, never a fixed conclusion."""
    tests = run["tests"]
    sig = significant_lags(ccf)
    head = (f"{tests['performed']} of {tests['declared']} lags were tested "
            f"({len(tests['not_performed'])} had too few pairs).")
    if sig.empty:
        return head + " No lag passes the Benjamini-Hochberg correction (p corrected < 0.05)."
    parts = []
    for r in sig.itertuples():
        way = "toward the adverse direction" if r.r_toward_adverse > 0 else "away from the adverse direction"
        parts.append(f"lag {int(r.lag)} (r = {r.spearman_r:+.2f}, {way}, p corrected = {r.p_bh:.3g}, {int(r.n)} pairs)")
    return head + " Passing the Benjamini-Hochberg correction: " + "; ".join(parts) + "."


def describe_verdict(verdict: dict) -> str:
    k = verdict["known_lag"]
    others = [lag for lag in verdict["lags_significant_after_bh"] if lag != k]
    if verdict["recovered"]:
        text = (f"The known lag was FOUND: the largest |r| is at lag {verdict['top_lag']} = {k}, with the expected sign "
                f"(r toward adverse = {verdict['r_toward_adverse_at_lag']:+.2f}) and corrected p = {verdict['p_bh_at_lag']:.2g}.")
    else:
        text = (f"The known lag ({k}) was NOT found: the largest |r| is at lag {verdict['top_lag']}; at the known lag the "
                f"corrected p is {verdict['p_bh_at_lag']}.")
    if others:
        text += f" Other lags also pass the correction: {others}."
    return text


# ── the interface ────────────────────────────────────────────────────────────────────────────────

def _show_result(ctx) -> None:
    out = ctx.workdir / RUN_DIR
    ccf = pd.read_csv(out / "ccf.csv")
    run = json.loads((out / "run.json").read_text())
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
               f"response: {p['response_label']} (adverse direction: {p['adverse_direction']}); lags 0 to {p['max_lag_months']}; "
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
