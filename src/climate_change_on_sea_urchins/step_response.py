"""Step 2 of the user's own data: the response series (milestone MA.3, docs/roadmap/MILESTONE-M1.md; ADR-0011 for the
format, ADR-0013 for the test series).

Two ways in. A CSV file the user uploads: the app says what the file looks like (`response_csv.suggest_format`) and then
ASKS for the separator, the columns, the decimal separator and the date format, every one with no preset answer -- a
dd/mm file read as mm/dd, or a decimal comma read as a point, would be misread without a sign (ADR-0011) -- and for what the
specification asks (what one row holds, name, unit, what kind of quantity, which way the value moves when the organisms do
worse). Or a test series with a known lag, generated from the temperature of step 1 (`known_truth`).

The outcome is a study folder in the session: `study.yaml` (format 2, written with `yaml.safe_dump`, loaded back with the
study loader), `response.csv` (the uploaded bytes, under a fixed name: the name of the uploaded file decides nothing), and
the SST with its manifest. `ccsu-run-study` runs it (step 3).

What comes from the user's file or typing is untrusted: the problems of the reader, which quote cells of the file, are shown
as code, never as Markdown; names and units are limited to plain characters; nothing is cached; the only YAML is the one the
app writes. Only functions here; what the step produces lives in `ctx.data`."""
from __future__ import annotations

import csv
import hashlib
import io
import re
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from . import RERUN_LOCK, dataset_builder, known_truth, response_csv, step_run, step_site
from .study_spec import StudySpec, StudySpecError, load_study

SOURCE_UPLOAD = "Upload a CSV file"
SOURCE_TEST = "A test series with a known lag"
DELIMITERS = {"comma (,)": ",", "semicolon (;)": ";", "tab": "\t", "vertical bar (|)": "|"}
DECIMALS = {"point (12.5)": ".", "comma (12,5)": ","}
DATE_FORMATS = {
    "%Y-%m-%d": "2020-01-31 (year-month-day)", "%d/%m/%Y": "31/01/2020 (day/month/year)",
    "%m/%d/%Y": "01/31/2020 (month/day/year)", "%d-%m-%Y": "31-01-2020 (day-month-year)",
    "%d.%m.%Y": "31.01.2020 (day.month.year)", "%Y/%m/%d": "2020/01/31 (year/month/day)",
    "%Y-%m": "2020-01 (year-month)", "%d-%b-%Y": "31-Jan-2020 (day-month name-year)",
}
ROWS = {"One measurement (several in a month are averaged)": "per_trial", "One value per month, already combined": "aggregated"}
NATURES = {"A proxy of the state of a population": "population_proxy", "An index": "index",
           "A toxicological endpoint (for example an EC50)": "toxicological_endpoint",
           "A biomarker measured on individuals": "biomarker"}
ADVERSE = {"It goes down (decrease)": "decrease", "It goes up (increase)": "increase"}
NO_COLUMN = "(none)"
MAX_COLUMNS = 200
MIN_PAIRS = 10                       # the command's own minimum (run_study.MIN_PAIRS: a test keeps the two equal)
STUDY_DIR = "study"
RESPONSE_FILE = "response.csv"
_LABEL = re.compile(r"[\w .,'()\-]{1,60}")
_UNIT = re.compile(r"[\w /.^%()*\-µ]{1,30}")
assert set(DATE_FORMATS) == set(response_csv._DATE_CANDIDATES)      # the formats the reader can suggest are all offered


# ── what a file is ───────────────────────────────────────────────────────────────────────────────

def header_columns(text: str, delimiter: str) -> list[str]:
    """The names in the header (the first line with something in it), stripped."""
    for row in csv.reader(io.StringIO(text), delimiter=delimiter):
        if any(c.strip() for c in row):
            return [c.strip() for c in row]
    return []


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:10]


def validate_answers(a: dict) -> list[str]:
    """What is missing or wrong in the answers, all at once, in words; nothing is filled in for the user."""
    e = []
    for key, text in (("delimiter", "Choose the field separator of the file."), ("date_col", "Choose the column with the date."),
                      ("value_col", "Choose the column with the measurement value."), ("decimal", "Choose the decimal separator of the numbers."),
                      ("date_format", "Choose the format of the dates."), ("rows", "Say what one row of the file holds."),
                      ("nature", "Say what kind of quantity the response is."),
                      ("adverse", "Say which way the value moves when the organisms do worse.")):
        if not a.get(key):
            e.append(text)
    label, unit = (a.get("label") or "").strip(), (a.get("unit") or "").strip()
    if not label:
        e.append("Enter a name for the response.")
    elif not _LABEL.fullmatch(label):
        e.append("The name of the response may use letters, digits, spaces and . , ' ( ) - only, up to 60 characters.")
    if not unit:
        e.append("Enter the unit of the response.")
    elif not _UNIT.fullmatch(unit):
        e.append("The unit may use letters, digits, spaces and / . ^ % ( ) * - µ only, up to 30 characters.")
    if a.get("delimiter") and a.get("decimal") and DELIMITERS[a["delimiter"]] == DECIMALS[a["decimal"]]:
        e.append("The field separator and the decimal separator cannot be the same character.")
    if a.get("date_col") and a.get("date_col") == a.get("value_col"):
        e.append("Choose different columns for the date and for the value.")
    lo, hi = a.get("ci_low"), a.get("ci_high")
    if (lo is None) != (hi is None):
        e.append("Give both ends of the confidence interval, or neither.")
    elif lo is not None and len({lo, hi, a.get("date_col"), a.get("value_col")}) < 4:
        e.append("The columns of the interval must be different from each other and from the date and the value.")
    return e


def response_entry(a: dict) -> dict:
    """The response of the specification from the answers (already validated)."""
    cmap = {"date": a["date_col"], "value": a["value_col"]}
    if a.get("ci_low"):
        cmap.update(ci_low=a["ci_low"], ci_high=a["ci_high"])
    per_trial = ROWS[a["rows"]] == "per_trial"
    entry = {"id": "response", "label": a["label"].strip(), "nature": NATURES[a["nature"]], "adverse_direction": ADVERSE[a["adverse"]],
             "unit": a["unit"].strip(),
             "source": {"type": "csv", "file": RESPONSE_FILE, "temporal_resolution": "month",
                        "granularity": "per_trial" if per_trial else "aggregated", "column_map": cmap,
                        "delimiter": DELIMITERS[a["delimiter"]], "decimal": DECIMALS[a["decimal"]],
                        "date_format": next(f for f, label in DATE_FORMATS.items() if label == a["date_format"])}}
    if per_trial:
        entry["aggregation"] = {"period": "month", "method": "mean", "count_field": "n_assays"}
    return entry


def study_dict(step1: dict, response: dict) -> dict:
    b0, b1 = step1["baseline"]
    site = step1["site"]
    return {"format_version": 2, "id": "own-study", "description": "A study built in the app from the user's own data.",
            "mhw_climatology": {"baseline_start_year": b0, "baseline_end_year": b1},
            "sites": [{k: site[k] for k in ("id", "name", "lat", "lon", "bbox_delta")}],
            "environment": [{"catalog_id": step_site.CATALOG_ID}], "responses": [response]}


def write_study(ctx, step1: dict, response: dict, csv_bytes: bytes) -> Path:
    """The study folder of the session. The study is loaded back through the loader (safe YAML, every validator)."""
    folder = ctx.workdir / STUDY_DIR
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    inputs = ctx.workdir / "inputs"
    shutil.copyfile(inputs / "sst_daily.csv", folder / "sst_daily.csv")
    shutil.copyfile(inputs / "sst_daily.manifest.json", folder / "sst_daily.manifest.json")
    (folder / RESPONSE_FILE).write_bytes(csv_bytes)
    (folder / "study.yaml").write_text(yaml.safe_dump(study_dict(step1, response), sort_keys=False, allow_unicode=True))
    load_study(folder / "study.yaml")
    return folder


def usability(series: pd.DataFrame, start: str, end: str) -> tuple[int, int]:
    """(months with a value inside the temperature period, pairs of consecutive such months): what the command can use."""
    months = pd.PeriodIndex(series.loc[series["value"].notna(), "Datetime"], freq="M")
    lo, hi = pd.Period(start, "M"), pd.Period(end, "M")
    inside = sorted(p for p in set(months) if lo <= p <= hi)
    return len(inside), sum(1 for a, b in zip(inside, inside[1:]) if (b - a).n == 1)


def _problem(message: str, headline: str = "The file cannot be read as declared:") -> None:
    st.error(headline)
    st.code(message, language=None)         # cells of the file are quoted in it: shown as code, never as Markdown


# ── the interface ────────────────────────────────────────────────────────────────────────────────

def _render_test(ctx, step1: dict):
    st.write("A response generated from the temperature anomaly you downloaded, with a lag you know, so that you can see the "
             "tool find it. It is a check of the tool, not a measurement.")
    seed = st.number_input("Seed", value=1, step=1, min_value=0, key="s2_seed", help="The same seed gives the same series.")
    lag = st.number_input("Lag (months)", value=3, step=1, key="s2_lag",
                          help="How many months the response follows the temperature anomaly: not 0, not a multiple of 12.")
    share = st.slider("Share of the variance that comes from the temperature", 0.0, 1.0, 0.5, 0.05, key="s2_share")
    key = ("test", int(seed), int(lag), float(share))
    if st.button("Generate the test series"):
        ctx.data.pop("step2", None)
        step_run.reset(ctx)
        if int(seed) < 0:
            st.error("The seed must be 0 or more.")
            return key
        folder = ctx.workdir / STUDY_DIR
        shutil.rmtree(folder, ignore_errors=True)
        try:
            with RERUN_LOCK:                     # the heatwave detection inside is numeric work: as in step 1
                truth = known_truth.make_study(ctx.workdir / "inputs" / "sst_daily.csv", folder, seed=int(seed),
                                               lat=step1["site"]["lat"], lon=step1["site"]["lon"], baseline=tuple(step1["baseline"]),
                                               lag=int(lag), signal_share=float(share), name=step1["site"]["name"])
            load_study(folder / "study.yaml")
        except (known_truth.KnownTruthError, StudySpecError, dataset_builder.DatasetError, ValueError) as e:
            shutil.rmtree(folder, ignore_errors=True)
            st.error(str(e).replace(str(ctx.workdir), "<session folder>"))
            return key
        ctx.data["step2"] = {"kind": "test", "answers_key": key, "label": "Test series", "truth": truth,
                             "text": f"A test series was generated (seed {truth['seed']}): the response goes down when the temperature "
                                     f"anomaly goes up, with a known lag of {truth['lag_months']} months and "
                                     f"{truth['signal_share_of_variance']:.0%} of its variance from the temperature. It is a check of "
                                     "the tool, not a measurement.", "series": None}
        ctx.label = "Test series"
    return key


def _choose(label, options, key, fp, kind="select"):
    widget = st.selectbox if kind == "select" else st.radio
    return widget(label, options, index=None, key=f"{key}_{fp}")


def suspect_messages(findings: list[dict]) -> list[str]:
    """The warnings for the findings of `response_csv.missing_code_suspects`: numbers and line numbers only, nothing typed in the
    file, so nothing that could be markup."""
    out = []
    for f in findings:
        where = ", ".join(str(n) for n in f["lines"]) + (", ..." if f["count"] > len(f["lines"]) else "")
        if f["kind"] == "code":
            what = f"{f['count']} of {f['of']} values are exactly {f['value']:g}, a number often used as a code for \"missing\""
        elif f["kind"] == "zeros":
            what = f"{f['count']} of {f['of']} values are exactly 0, which is sometimes written for \"missing\""
        else:
            what = f"{f['count']} of {f['of']} values are exactly {f['value']:g}, a repeat that a measurement rarely makes"
        out.append(f"{what} (lines {where}). If they stand for no measurement they would be read as real measurements: leave "
                   "those cells empty or remove the rows, and upload the file again. If they are real, ignore this.")
    return out


def _first_line(text: str) -> str:
    return next((line for line in text.splitlines() if line.strip()), "")


def _render_upload(ctx, step1: dict):
    """The upload branch. Returns the key of the answers on the page (None while there is nothing to build from)."""
    uploaded = st.file_uploader("CSV file with your measurements", type=["csv"], key="s2_file")
    if uploaded is None:
        st.caption(f"One row per measurement or per month, a header line, up to {response_csv.MAX_BYTES // 1_000_000} MB. "
                   "From Excel: Save as \"CSV UTF-8\".")
        return None
    if uploaded.size > response_csv.MAX_BYTES:
        st.error(f"The file is {uploaded.size} bytes, over the limit of {response_csv.MAX_BYTES}.")
        return None
    data = uploaded.getvalue()
    try:
        text = response_csv.decode_text(data)
    except response_csv.ResponseCsvError as e:
        _problem(str(e), "The file cannot be used:")
        return None
    fp = fingerprint(data)
    guess = response_csv.suggest_format(_first_line(text), date_column="", value_column="")["delimiter"]
    if guess:
        st.write(f"The file looks like it is separated by {guess!r} (from its first line). That is only what it looks like: "
                 "choose the separator yourself.")
    delim_label = _choose("Field separator in the file", list(DELIMITERS), "s2_delim", fp)
    a = {"delimiter": delim_label}
    if not delim_label:
        return (fp, ())
    try:
        columns = header_columns(text, DELIMITERS[delim_label])
    except csv.Error as e:
        _problem(f"the header cannot be parsed as CSV ({e}); look for a quote that is never closed")
        return (fp, ())
    if len(columns) > MAX_COLUMNS or any(len(c) > 100 for c in columns):
        st.error(f"The header has {len(columns)} columns (the longest name has {max((len(c) for c in columns), default=0)} characters): "
                 f"this is not a table of measurements, or the separator is wrong. Up to {MAX_COLUMNS} columns and 100 characters "
                 "per name are accepted.")
        return (fp, ())
    if len(columns) < 2:
        st.warning("With this separator the header has a single column: the separator is probably another one.")
    c1, c2 = st.columns(2)
    a["date_col"] = c1.selectbox("Column with the date", columns, index=None, key=f"s2_date_{fp}")
    a["value_col"] = c2.selectbox("Column with the measurement value", columns, index=None, key=f"s2_value_{fp}")
    if a["date_col"] and a["value_col"] and a["date_col"] != a["value_col"]:
        try:
            suggestion = response_csv.suggest_format(text, date_column=a["date_col"], value_column=a["value_col"])
        except response_csv.ResponseCsvError as e:
            _problem(str(e))
            return (fp, ())
        if suggestion["decimal"]:
            st.write(f"The numbers of that column look like they use {suggestion['decimal']!r} as the decimal separator.")
        if suggestion["date_ambiguous"]:
            names = ", ".join(DATE_FORMATS[f] for f in suggestion["date_formats"])
            st.warning(f"The dates in this file fit more than one format ({names}): only you know which one it is.")
        elif suggestion["date_formats"]:
            st.write(f"The dates fit {DATE_FORMATS[suggestion['date_formats'][0]]}.")
        elif suggestion["two_digit_year"]:
            st.warning("The dates have a two-digit year, which is not accepted: export the file with four digits.")
        else:
            st.warning("The dates do not fit any of the formats this version reads.")
    a["decimal"] = _choose("Decimal separator of the numbers", list(DECIMALS), "s2_decimal", fp, "radio")
    a["date_format"] = _choose("Format of the dates", list(DATE_FORMATS.values()), "s2_dfmt", fp)
    a["rows"] = _choose("What does one row of the file hold?", list(ROWS), "s2_rows", fp, "radio")
    if a["decimal"] and a["value_col"]:                       # only with the decimal separator the user chose: nothing is assumed
        memo = ctx.data.setdefault("s2_suspects", {})                  # computed once per file and choices, not at every run
        ck = (fp, DELIMITERS[delim_label], DECIMALS[a["decimal"]], a["value_col"])
        if ck not in memo:
            memo.clear()
            memo[ck] = response_csv.missing_code_suspects(text, ck[1], ck[2], ck[3])
        for message in suspect_messages(memo[ck]):
            st.warning(message)
    others = [NO_COLUMN] + list(columns)
    lo = st.selectbox("Column with the lower end of the interval (optional)", others, index=0, key=f"s2_lo_{fp}")
    hi = st.selectbox("Column with the upper end of the interval (optional)", others, index=0, key=f"s2_hi_{fp}")
    a["ci_low"], a["ci_high"] = (None if lo == NO_COLUMN else lo), (None if hi == NO_COLUMN else hi)
    a["label"] = st.text_input("Name of the response (for the results)", key=f"s2_label_{fp}")
    a["unit"] = st.text_input("Unit of the response (for example ug/L)", key=f"s2_unit_{fp}")
    a["nature"] = _choose("What kind of quantity is it?", list(NATURES), "s2_nature", fp)
    a["adverse"] = _choose("Which way does the value move when the organisms do worse?", list(ADVERSE), "s2_adverse", fp, "radio")
    st.caption("No imputation is applied: a month without a measurement stays without one. Dates are read as monthly periods.")
    key = (fp, tuple(sorted((k, v) for k, v in a.items())))
    if not st.button("Check the file and build the study"):
        return key
    ctx.data.pop("step2", None)
    step_run.reset(ctx)
    shutil.rmtree(ctx.workdir / STUDY_DIR, ignore_errors=True)
    errors = validate_answers(a)
    if errors:
        for e in errors:
            st.error(e)
        return key
    try:
        entry = response_entry(a)
        spec = StudySpec.model_validate(study_dict(step1, entry))
        result = response_csv.read_response_text(data, spec.responses[0])
        write_study(ctx, step1, entry, data)
    except response_csv.ResponseCsvError as e:
        _problem(str(e))
        return key
    except (StudySpecError, ValueError) as e:
        _problem(str(e), "The answers do not make a valid study:")
        return key
    ctx.data["step2"] = {"kind": "csv", "answers_key": key, "label": a["label"].strip(), "truth": None,
                         "series": result.series, "text": None, "notes": result.notes}
    ctx.label = a["label"].strip()
    return key


def _show_study(ctx, step1: dict, current_key) -> None:
    done = ctx.data.get("step2")
    if not done:
        return
    folder = ctx.workdir / STUDY_DIR
    if not (folder / "study.yaml").is_file():
        ctx.data.pop("step2", None)
        st.warning("The study is no longer there (the session folder is deleted after 6 hours without use): build it again.")
        return
    if done["answers_key"] != current_key:
        st.warning("The answers changed since this study was built (or the file was removed): build it again; the study below, "
                   "and what step 3 would run, is the one built before.")
    if done["kind"] == "test":
        st.success(done["text"])
    else:
        s = done["series"]
        valid = s[s["value"].notna()]
        first, last = valid["Datetime"].min(), valid["Datetime"].max()
        span = (last.year - first.year) * 12 + last.month - first.month + 1
        inside, pairs = usability(s, step1["start"], step1["end"])
        head = (f"{len(valid)} months with a value, {first:%Y-%m} to {last:%Y-%m}; {span - len(valid)} months in between have none "
                "and stay without one.")
        if inside == 0:
            st.warning(f"The study was built, but the series has no month inside the temperature period ({step1['start']} to "
                       f"{step1['end']}): no lag can be tested. {head}")
        elif pairs < MIN_PAIRS:
            st.warning(f"The study was built, but the series has only {pairs} pairs of consecutive months inside the temperature "
                       f"period and the command needs at least {MIN_PAIRS}: no lag can be tested. {head}")
        else:
            st.success(f"The study is ready: {head} {pairs} pairs of consecutive months inside the temperature period. "
                       "Next: step 3 (run the study).")
        st.line_chart(s.set_index("Datetime")["value"], height=200)
    st.download_button("Download study.yaml", (folder / "study.yaml").read_bytes(), file_name="study.yaml", mime="text/yaml")


def render(ctx) -> None:
    st.markdown("**Step 2 — Your response series**")
    step1 = ctx.data.get("step1")
    if not step1 or not (ctx.workdir / "inputs" / "sst_daily.csv").is_file():
        st.info("Complete step 1 first: the response series is read against the site, the baseline and the temperature of step 1.")
        return
    view = ctx.data.get("step1_view") or {}
    if view.get("error"):
        st.error("Step 2 cannot go on: the temperature downloaded in step 1 cannot be used as it is. " + view["error"])
        return
    source = st.radio("Where does the response series come from?", [SOURCE_UPLOAD, SOURCE_TEST], index=None, key="s2_source")
    key = None
    if source == SOURCE_UPLOAD:
        key = _render_upload(ctx, step1)
    elif source == SOURCE_TEST:
        key = _render_test(ctx, step1)
    _show_study(ctx, step1, key)
