"""The CSV response source (M1.3): a response series in a file the user provides, read strictly.

The format of the file is declared in the study (`delimiter`, `decimal`, `date_format` of
`ResponseCsvSourceSpec`), never guessed: a European CSV (';' and decimal commas, 31/01/2020) read with
the English defaults would be misread without a sign, and dd/mm against mm/dd cannot be told apart by
looking at the data. So a file that does not match its declaration is refused, naming the line, the column
and the value, and saying what the file looks like (`suggest_format`, which the guided input of milestone
M1 uses to ask the user instead of deciding for them).

What is refused: an empty file or one without data rows; a declared column that is missing (with the
columns the file does have); a repeated header name; a delimiter other than the declared one; rows with the
wrong number of fields; dates that do not match the format (a date with a time of day is not truncated);
values that are not finite numbers (nan, inf and 1e999 included); a value outside its own confidence
interval; two rows in the same period when the rows are already aggregated; a file that is not UTF-8 or
is too large. Problems are collected (up to MAX_PROBLEMS, then counted) so a user fixes a file once.

What is not refused and not filled: an empty value is a missing measurement, kept as missing and counted;
a month with no row at all does not appear. Without an interval in the file the series has none: no bound
is made up. The aggregation of per-trial rows is `common.aggregate_period`, the one implementation also
behind the sheet of the Livorno study.

The file arrives as a bare name resolved by whoever loads the study (the package, the upload), never as a
path chosen by the study (study_spec.py).
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import common

MAX_BYTES = 10_000_000
MAX_ROWS = 200_000
MAX_PROBLEMS = 20
_DELIMITERS = [",", ";", "\t", "|"]
_DATE_CANDIDATES = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d", "%Y-%m", "%d-%b-%y", "%d-%b-%Y"]
_PLAIN_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


class ResponseCsvError(Exception):
    """The file cannot be read as declared; the message lists the problems."""

    def __init__(self, problems: list[str], total: int | None = None):
        self.problems = problems
        total = total if total is not None else len(problems)
        head = f"{total} problems in the response CSV" if total != 1 else "1 problem in the response CSV"
        more = f" (first {len(problems)} shown)" if total > len(problems) else ""
        super().__init__(head + more + ":\n- " + "\n- ".join(problems))


@dataclass
class ResponseData:
    observations: pd.DataFrame      # one row per line of the file: Datetime, value, ci_low, ci_high, controls, line
    series: pd.DataFrame            # one row per month: Datetime, value, ci_upper, ci_lower, n
    notes: dict = field(default_factory=dict)


# ── what a file looks like ───────────────────────────────────────────────────────────────────────

def _rows(text: str, delimiter: str) -> list[tuple[int, list[str]]]:
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    return [(reader.line_num, [c.strip() for c in row]) for row in reader if any(c.strip() for c in row)]


def suggest_format(text: str, *, date_column: str, value_column: str) -> dict:
    """What a file looks like, for asking the user -- never a decision. delimiter: the separator the
    header suggests; decimal: '.' or ',' from the numbers of the value column (None if undecided);
    date_formats: every candidate format that parses all the dates; date_ambiguous: more than one does."""
    header_line = next((l for l in text.splitlines() if l.strip()), "")
    counts = {d: header_line.count(d) for d in _DELIMITERS}
    delimiter = max(counts, key=counts.get) if any(counts.values()) else None
    out = {"delimiter": delimiter, "decimal": None, "date_formats": [], "date_ambiguous": False}
    if delimiter is None:
        return out
    rows = _rows(text, delimiter)
    if not rows or date_column not in rows[0][1] or value_column not in rows[0][1]:
        return out
    di, vi = rows[0][1].index(date_column), rows[0][1].index(value_column)
    body = [r for _, r in rows[1:] if len(r) > max(di, vi)]
    values = [r[vi] for r in body if r[vi]]
    if any(re.fullmatch(r"[+-]?\d+,\d+", v) for v in values):
        out["decimal"] = ","
    elif any(re.fullmatch(r"[+-]?\d+\.\d+", v) for v in values):
        out["decimal"] = "."
    dates = [r[di] for r in body if r[di]]
    for fmt in _DATE_CANDIDATES:
        try:
            for d in dates:
                dt.datetime.strptime(d, fmt)
        except ValueError:
            continue
        if dates:
            out["date_formats"].append(fmt)
    out["date_ambiguous"] = len(out["date_formats"]) > 1
    return out


# ── reading ──────────────────────────────────────────────────────────────────────────────────────

def _number(token: str, decimal: str) -> float:
    """float of `token` with the declared decimal separator; ValueError with the reason otherwise."""
    other = "," if decimal == "." else "."
    if other in token:
        raise ValueError(f"has a '{other}' but the declared decimal separator is '{decimal}'")
    t = token.replace(",", ".") if decimal == "," else token
    if re.fullmatch(r"[+-]?(nan|inf|infinity)", t, flags=re.IGNORECASE):
        raise ValueError("is not a finite number")
    if not _PLAIN_NUMBER.match(t):
        raise ValueError("is not a number")
    x = float(t)
    if not math.isfinite(x):
        raise ValueError("is not a finite number")
    return x


def read_response(path: str | Path, response) -> ResponseData:
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ResponseCsvError([f"the file is {size} bytes, over the limit of {MAX_BYTES}"])
    return read_response_text(path.read_bytes(), response)


def read_response_text(text: str | bytes, response) -> ResponseData:
    src = response.source
    if src.type != "csv":
        raise ResponseCsvError([f"the response {response.id!r} has a {src.type!r} source, not a csv one"])
    if isinstance(text, bytes):
        if len(text) > MAX_BYTES:
            raise ResponseCsvError([f"the file is {len(text)} bytes, over the limit of {MAX_BYTES}"])
        try:
            text = text.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise ResponseCsvError([f"the file is not UTF-8 (byte {e.start}); save it as UTF-8"]) from None
    if not text.strip():
        raise ResponseCsvError(["the file is empty"])

    cm = src.column_map
    wanted = {"date": cm.date, "value": cm.value}
    if cm.ci_low:
        wanted.update(ci_low=cm.ci_low, ci_high=cm.ci_high)
    controls = list(src.control_columns or [])

    rows = _rows(text, src.delimiter)
    if not rows:
        raise ResponseCsvError(["the file is empty"])
    if len(rows) - 1 > MAX_ROWS:
        raise ResponseCsvError([f"the file has {len(rows) - 1} rows, over the limit of {MAX_ROWS}"])
    _, header = rows[0]
    problems: list[str] = []

    dup = sorted({h for h in header if header.count(h) > 1})
    if dup:
        problems.append(f"line 1: the column name(s) {', '.join(map(repr, dup))} appear twice in the header")
    missing = [c for c in list(wanted.values()) + controls if c not in header]
    if missing:
        hint = ""
        if len(header) == 1:
            other = [d for d in _DELIMITERS if d != src.delimiter and d in header[0]]
            if other:
                hint = (f" The header has a single column and contains {', '.join(repr(d) for d in other)}: the file looks "
                        f"{other[0]!r}-separated but the study declares delimiter {src.delimiter!r}.")
        problems.append(f"line 1: the declared column(s) {', '.join(map(repr, missing))} are not in the header; the file has "
                        f"{', '.join(map(repr, header))}.{hint}")
        raise ResponseCsvError(problems)
    if problems:
        raise ResponseCsvError(problems)
    if len(rows) == 1:
        raise ResponseCsvError(["the file has a header and no data rows"])

    idx = {name: header.index(col) for name, col in wanted.items()}
    cidx = {c: header.index(c) for c in controls}
    suggestion = None

    def suggest():
        nonlocal suggestion
        if suggestion is None:
            suggestion = suggest_format(text, date_column=cm.date, value_column=cm.value)
        return suggestion

    recs, total = [], 0

    def problem(msg):
        nonlocal total
        total += 1
        if len(problems) < MAX_PROBLEMS:
            problems.append(msg)

    for line, row in rows[1:]:
        if len(row) != len(header):
            problem(f"line {line}: {len(row)} fields, the header has {len(header)}")
            continue
        rec = {"line": line}
        tok = row[idx["date"]]
        try:
            d = dt.datetime.strptime(tok, src.date_format)
            rec["date"] = d
        except ValueError:
            s = suggest()
            look = (f"; the dates of this file look like {', '.join(s['date_formats'])}" +
                    (" (more than one fits: say which)" if s["date_ambiguous"] else "")) if s["date_formats"] else ""
            problem(f"line {line}, column {cm.date!r}: {tok!r} does not match the date format {src.date_format!r}{look}")
            continue
        for name in ["value"] + (["ci_low", "ci_high"] if cm.ci_low else []) + controls:
            col = cm.value if name == "value" else wanted.get(name, name)
            raw = row[idx[name]] if name in idx else row[cidx[name]]
            if raw == "":
                rec[name] = np.nan
                continue
            try:
                rec[name] = _number(raw, src.decimal)
            except ValueError as e:
                msg = str(e)
                if "declared decimal" in msg:
                    s = suggest()
                    msg += f" (the numbers of this file look like decimal {s['decimal']!r})" if s["decimal"] else ""
                problem(f"line {line}, column {col!r}: {raw!r} {msg}")
        recs.append(rec)

    obs = pd.DataFrame(recs)
    if total == 0 and not obs.empty:
        if cm.ci_low:
            bad = obs[obs["value"].notna() & obs["ci_low"].notna() & obs["ci_high"].notna()
                      & ((obs["ci_low"] > obs["value"]) | (obs["ci_high"] < obs["value"]))]
            for _, r in bad.iterrows():
                problem(f"line {int(r['line'])}: the value {r['value']:g} is outside its interval [{r['ci_low']:g}, {r['ci_high']:g}]")
        if src.granularity == "aggregated":
            period = obs["date"].dt.to_period("M")
            for p, grp in obs.groupby(period):
                if len(grp) > 1:
                    problem(", ".join(f"line {int(l)}" for l in grp["line"][:5]) +
                            f": more than one row in the same period ({p}), but the rows are declared already aggregated")
    if total:
        raise ResponseCsvError(problems, total)

    obs = obs.rename(columns={"date": "Datetime"}).sort_values(["Datetime", "line"]).reset_index(drop=True)
    notes = {"rows": int(len(obs)), "missing_values": int(obs["value"].isna().sum()),
             "first": obs["Datetime"].min().date().isoformat(), "last": obs["Datetime"].max().date().isoformat()}

    if src.granularity == "per_trial":
        series = common.aggregate_period(obs, date="Datetime", value="value",
                                         ci_low="ci_low" if cm.ci_low else None,
                                         ci_high="ci_high" if cm.ci_low else None)
    else:
        series = pd.DataFrame({"Datetime": obs["Datetime"].dt.to_period("M").dt.to_timestamp(), "value": obs["value"],
                               "ci_upper": obs["ci_high"] if cm.ci_low else np.nan,
                               "ci_lower": obs["ci_low"] if cm.ci_low else np.nan,
                               "n": np.where(obs["value"].notna(), 1, 0)})
        series = series.sort_values("Datetime").reset_index(drop=True)
    notes["months"] = int(len(series))
    return ResponseData(observations=obs, series=series, notes=notes)
