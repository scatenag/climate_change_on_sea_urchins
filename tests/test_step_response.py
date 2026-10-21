"""MA.3: step 2 of the user's own data -- the response series. A CSV file the user uploads, read with the format the user
confirms (separator, decimal, date format: never guessed, always chosen), with the questions of the study specification and
the errors of the CSV reader shown in plain text with their lines; or a test series with a known lag generated from the
temperature just downloaded. The outcome is a study folder (study.yaml of format 2, response.csv, the SST) that
ccsu-run-study can run. Through streamlit.testing from app.py, with the Copernicus client replaced by the test client."""
from __future__ import annotations

import ast
import datetime as dt
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_change_on_sea_urchins import app_entry, known_truth, response_csv, run_study, step_site, step_response
from climate_change_on_sea_urchins.study_spec import load_study
from test_step_site import (COVERAGE_2003, FIXTURE_SST, RecordingClient, _by_label, _ctx, _errors, _fill, _open,
                            _submit, _text)

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "climate_change_on_sea_urchins"
FIXTURE = ROOT / "tests" / "fixtures" / "paper_mpb_2026" / "data"
CLOCK = lambda: dt.datetime(2026, 10, 17, 9, 0, tzinfo=dt.timezone.utc)

SOURCE = "Upload a CSV file"
TEST_SERIES = "A test series with a known lag"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for var in [v for v in os.environ if v.startswith("COPERNICUSMARINE")]:
        monkeypatch.delenv(var)
    monkeypatch.setattr(app_entry, "SESSIONS_ROOT", tmp_path / "sessions")
    monkeypatch.setattr(step_site, "acquisition_available", lambda: True)


@pytest.fixture
def after_step1(monkeypatch):
    """The app at step 2 with step 1 done on the frozen fixture SST (baseline 2003-2012)."""
    sst = pd.read_csv(FIXTURE_SST, parse_dates=["Datetime"]).set_index("Datetime")["Temperature"]
    c = RecordingClient(coverage=COVERAGE_2003.__class__(start=dt.date(2003, 1, 1), end=sst.index.max().date(), version="202511"), series=sst)
    monkeypatch.setattr(step_site, "make_client", lambda: c)
    return _submit(_fill(_open()))


class Upload:
    """What st.file_uploader returns."""
    def __init__(self, data: bytes, name="whatever.csv"):
        self._data, self.name, self.size = data, name, len(data)

    def getvalue(self):
        return self._data

    def read(self):
        return self._data

    type = "text/csv"


def _upload(monkeypatch, data, name="my measurements.csv"):
    up = Upload(data if isinstance(data, bytes) else data.encode(), name)
    monkeypatch.setattr(step_response.st, "file_uploader", lambda *a, **k: up)


def _trials(n_months=60, seed=0, start="2005-01-01"):
    rng = np.random.default_rng(seed)
    rows = []
    for m in pd.date_range(start, periods=n_months, freq="MS"):
        if rng.random() < 0.4:
            continue
        for _ in range(int(rng.integers(1, 4))):
            rows.append((m + pd.Timedelta(days=int(rng.integers(0, 25))), round(float(rng.normal(30, 5)), 2)))
    return pd.DataFrame(rows, columns=["when", "ec"])


def _csv(df, delimiter=",", decimal=".", fmt="%Y-%m-%d"):
    out = df.copy()
    out["when"] = out["when"].dt.strftime(fmt)
    return out.to_csv(index=False, sep=delimiter, decimal=decimal)


def _pick(at, label, option):
    for kind in (at.selectbox, at.radio):
        for w in kind:
            if w.label == label:
                w.set_value(option)
                return
    raise AssertionError(f"no widget {label!r}")


def _fill_answers(at, **kw):
    cfg = dict(delimiter="comma (,)", decimal="point (12.5)", date_format="2020-01-31 (year-month-day)",
               rows="One measurement (several in a month are averaged)", date_col="when", value_col="ec", label="Fertility",
               unit="ug/L", nature="An index", adverse="It goes down (decrease)")
    cfg.update(kw)
    _pick(at, "Field separator in the file", cfg["delimiter"]); at.run()
    _pick(at, "Column with the date", cfg["date_col"]); _pick(at, "Column with the measurement value", cfg["value_col"]); at.run()
    for label, key in (("Decimal separator of the numbers", "decimal"), ("Format of the dates", "date_format"),
                       ("What does one row of the file hold?", "rows")):
        if cfg[key]:                                  # None leaves the question unanswered, as a user who skips it
            _pick(at, label, cfg[key])
    _by_label(at, "Name of the response (for the results)").set_value(cfg["label"])
    _by_label(at, "Unit of the response (for example ug/L)").set_value(cfg["unit"])
    if cfg["nature"]:
        _pick(at, "What kind of quantity is it?", cfg["nature"])
    if cfg["adverse"]:
        _pick(at, "Which way does the value move when the organisms do worse?", cfg["adverse"])
    return at


def _build(at):
    next(b for b in at.button if "Check the file and build the study" in b.label).click().run()
    return at


def _start_upload(at, monkeypatch, data):
    _upload(monkeypatch, data)
    _pick(at, "Where does the response series come from?", SOURCE)
    at.run()
    return at


def _study_dir(at):
    return _ctx(at).workdir / "study"


def _full_flow(at, monkeypatch, text, **kw):
    _start_upload(at, monkeypatch, text)
    _fill_answers(at, **kw)
    return _build(at)


# --- the order of the steps ---------------------------------------------------------------------------------

def test_step_2_asks_for_step_1_first():
    at = _open()
    assert "Complete step 1" in _text(at) and not [r for r in at.radio if r.label.startswith("Where does the response")]


def test_after_step_1_the_two_sources_are_offered_and_none_is_preselected(after_step1):
    at = after_step1
    radio = next(r for r in at.radio if r.label == "Where does the response series come from?")
    assert radio.value is None and SOURCE in radio.options and TEST_SERIES in radio.options


# --- the test series ------------------------------------------------------------------------------------------

def _generate(at, seed=1, lag=3, share=0.5):
    _pick(at, "Where does the response series come from?", TEST_SERIES); at.run()
    next(n for n in at.number_input if n.label == "Seed").set_value(seed)
    next(n for n in at.number_input if n.label == "Lag (months)").set_value(lag)
    next(s for s in at.slider if s.label.startswith("Share of the variance")).set_value(share)
    next(b for b in at.button if "Generate the test series" in b.label).click().run()
    return at


def test_the_test_series_is_the_one_the_generator_makes_with_the_same_seed(after_step1, tmp_path):
    at = _generate(after_step1, seed=5, lag=4, share=0.6)
    assert not at.exception, [e.value for e in at.exception]
    folder = _study_dir(at)
    spec = load_study(folder / "study.yaml")
    assert spec.format_version == 2 and spec.responses[0].source.type == "csv"
    step1 = _ctx(at).data["step1"]
    known_truth.make_study(_ctx(at).workdir / "inputs" / "sst_daily.csv", tmp_path / "ref", seed=5, lat=step1["site"]["lat"],
                           lon=step1["site"]["lon"], baseline=tuple(step1["baseline"]), lag=4, signal_share=0.6, name=step1["site"]["name"])
    for name in ("response.csv", "response.manifest.json"):
        assert (folder / name).read_bytes() == (tmp_path / "ref" / name).read_bytes()
    assert "known lag" in _text(at) and "4" in _text(at) and "seed 5" in _text(at)


@pytest.mark.parametrize("lag", [0, 12, 13])
def test_a_lag_the_generator_refuses_is_shown_with_its_reason(after_step1, lag):
    at = _generate(after_step1, lag=lag)
    assert not at.exception and "lag" in _errors(at) and not (_study_dir(at) / "study.yaml").exists()


def test_the_study_of_a_test_series_runs_in_the_command(after_step1, tmp_path):
    at = _generate(after_step1)
    folder = _study_dir(at)
    assert run_study.run_study(folder / "study.yaml", folder, tmp_path / "out", now=CLOCK) == 0
    assert known_truth.verify(tmp_path / "out")["tests_performed"] >= 1


# --- the CSV: the format is asked, never guessed -------------------------------------------------------------

def test_the_suggestions_say_what_the_file_looks_like_and_the_choice_is_still_the_users(after_step1, monkeypatch):
    text = _csv(_trials(), ";", ",", "%d/%m/%Y")
    at = _start_upload(after_step1, monkeypatch, text)
    assert "looks like" in _shown(at) and "';'" in _shown(at)
    sep = next(s for s in at.selectbox if s.label == "Field separator in the file")
    assert sep.value is None
    _pick(at, "Field separator in the file", "semicolon (;)"); at.run()
    _pick(at, "Column with the date", "when"); _pick(at, "Column with the measurement value", "ec"); at.run()
    assert "decimal separator" in _shown(at) and "','" in _shown(at)
    assert next(r for r in at.radio if r.label == "Decimal separator of the numbers").value is None
    assert next(s for s in at.selectbox if s.label == "Format of the dates").value is None


def test_a_file_that_is_read_with_the_declared_format_builds_the_study(after_step1, monkeypatch):
    df = _trials()
    at = _full_flow(after_step1, monkeypatch, _csv(df))
    assert not at.exception and not at.error, _errors(at)
    folder = _study_dir(at)
    spec = load_study(folder / "study.yaml")
    r = spec.responses[0]
    assert (r.label, r.unit, r.adverse_direction, r.nature) == ("Fertility", "ug/L", "decrease", "index")
    assert r.source.file == "response.csv" and r.source.granularity == "per_trial" and r.imputation is None
    step1 = _ctx(at).data["step1"]
    assert (spec.mhw_climatology.baseline_start_year, spec.mhw_climatology.baseline_end_year) == tuple(step1["baseline"])
    assert (spec.sites[0].lat, spec.sites[0].lon, spec.sites[0].name) == (step1["site"]["lat"], step1["site"]["lon"], "Test site")
    assert [e.catalog_id for e in spec.environment] == ["sst_daily"]
    assert (folder / "response.csv").read_bytes() == _csv(df).encode()
    got = response_csv.read_response(folder / "response.csv", r).series
    assert len(got) == df["when"].dt.to_period("M").nunique()
    assert "months with a value" in _text(at)
    assert _ctx(at).label == "Fertility"


def test_a_european_file_gives_the_same_series_as_the_english_one(after_step1, monkeypatch, tmp_path):
    df = _trials(seed=3)
    en = _full_flow(after_step1, monkeypatch, _csv(df))
    a = response_csv.read_response(_study_dir(en) / "response.csv", load_study(_study_dir(en) / "study.yaml").responses[0]).series
    up = Upload(_csv(df, ";", ",", "%d/%m/%Y").encode())
    monkeypatch.setattr(step_response.st, "file_uploader", lambda *a, **k: up)
    eu = _open_again_with_step1(monkeypatch)
    eu = _full_flow_loaded(eu, delimiter="semicolon (;)", decimal="comma (12,5)", date_format="31/01/2020 (day/month/year)")
    assert not eu.error, _errors(eu)
    b = response_csv.read_response(_study_dir(eu) / "response.csv", load_study(_study_dir(eu) / "study.yaml").responses[0]).series
    pd.testing.assert_frame_equal(a, b)


def _open_again_with_step1(monkeypatch):
    sst = pd.read_csv(FIXTURE_SST, parse_dates=["Datetime"]).set_index("Datetime")["Temperature"]
    c = RecordingClient(coverage=COVERAGE_2003.__class__(start=dt.date(2003, 1, 1), end=sst.index.max().date(), version="202511"), series=sst)
    monkeypatch.setattr(step_site, "make_client", lambda: c)
    return _submit(_fill(_open()))


def _full_flow_loaded(at, **kw):
    _pick(at, "Where does the response series come from?", SOURCE); at.run()
    _fill_answers(at, **kw)
    return _build(at)


def test_livorno_trials_exported_and_uploaded_give_the_same_monthly_series_as_the_sheet(after_step1, monkeypatch):
    raw = pd.read_csv(FIXTURE / "ec50_raw.csv", parse_dates=["Datetime"]).sort_values(["Datetime", "ID"])
    expected = pd.read_csv(FIXTURE / "ec50_sheets.csv", parse_dates=["Datetime"])
    df = pd.DataFrame({"when": raw["Datetime"], "ec": raw["EC50"]})
    at = _full_flow(after_step1, monkeypatch, _csv(df, ";", ",", "%d/%m/%Y"), delimiter="semicolon (;)", decimal="comma (12,5)",
                    date_format="31/01/2020 (day/month/year)", label="EC50", unit="ug/L")
    assert not at.error, _errors(at)
    got = response_csv.read_response(_study_dir(at) / "response.csv", load_study(_study_dir(at) / "study.yaml").responses[0]).series
    assert got["Datetime"].tolist() == expected["Datetime"].tolist()
    np.testing.assert_allclose(got["value"], expected["EC50"], rtol=0, atol=1e-12)
    assert got["n"].tolist() == expected["EC50_n"].tolist()


def test_the_wrong_separator_is_noticed_at_once_with_a_single_column_header(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, _csv(_trials(), ";", ",", "%d/%m/%Y"))
    _pick(at, "Field separator in the file", "comma (,)"); at.run()
    assert not at.exception and "single column" in _shown(at)
    assert next(s for s in at.selectbox if s.label == "Column with the date").options == ["when;ec"]


def test_a_decimal_comma_declared_as_a_point_is_named_with_its_line(after_step1, monkeypatch):
    text = _csv(_trials(), ";", ",", "%Y-%m-%d")
    at = _full_flow(after_step1, monkeypatch, text, delimiter="semicolon (;)", decimal="point (12.5)")
    assert re.search(r"line \d+", _code(at)) and "decimal" in _code(at)
    assert not (_study_dir(at) / "study.yaml").exists()


def _shown(at):
    """The text of the page as the user reads it (the values of the text elements, quotes as typed)."""
    return " ".join(e.value for kind in (at.markdown, at.text, at.caption, at.warning, at.info, at.success, at.error, at.subheader, at.header)
                    for e in kind)


def _code(at):
    return "\n".join(c.value for c in at.code)


AMBIGUOUS = "when,ec\n" + "\n".join(f"{d:02d}/{m:02d}/2010,{20 + d + m}" for m in range(1, 5) for d in (3, 7, 11)) + "\n"


def test_a_date_the_file_cannot_settle_blocks_until_the_user_chooses(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, AMBIGUOUS)
    _pick(at, "Field separator in the file", "comma (,)"); at.run()
    _pick(at, "Column with the date", "when"); _pick(at, "Column with the measurement value", "ec"); at.run()
    assert "more than one" in _shown(at) and "only you know" in _shown(at)
    _fill_answers(at, date_format=None)             # everything answered but the format
    _build(at)
    assert "format of the dates" in _errors(at) and not (_study_dir(at) / "study.yaml").exists()
    _fill_answers(at, date_format="31/01/2020 (day/month/year)")
    _build(at)
    assert not at.error and (_study_dir(at) / "study.yaml").exists()


def test_the_answers_without_a_date_format_are_refused_by_the_validation():
    answers = dict(delimiter="comma (,)", date_col="when", value_col="ec", decimal="point (12.5)", date_format=None,
                   rows="One measurement (several in a month are averaged)", label="F", unit="ug/L", nature="An index",
                   adverse="It goes down (decrease)")
    assert "Choose the format of the dates." in step_response.validate_answers(answers)
    assert step_response.validate_answers({**answers, "date_format": "31/01/2020 (day/month/year)"}) == []


@pytest.mark.parametrize("body, expect", [
    ("", "empty"),
    ("when,ec\n", "no data"),
    ("when,ec\n2010-01-05,abc\n", "line 2"),
    ("when,ec\n2010-01-05,10\nnot-a-date,11\n", "line 3"),
    ("when,ec\n2010-01-05,nan\n", "finite"),
    ("when,ec\n2010-01-05,10,9\n", "line 2"),
])
def test_each_kind_of_problem_of_the_file_is_shown_with_its_line(after_step1, monkeypatch, body, expect):
    at = _start_upload(after_step1, monkeypatch, body)
    if body.strip() and "when" in body:
        _fill_answers(at)
        _build(at)
        assert not at.exception and expect in _code(at)
    else:
        assert not at.exception and expect in _code(at)          # a problem of the file, in the code block
    assert not (_study_dir(at) / "study.yaml").exists()


def test_a_file_that_is_not_utf8_gets_the_excel_hint(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, "when,ec\n2010-01-05,caffè\n".encode("latin-1"))
    assert "UTF-8" in _code(at) and "Excel" in _code(at)          # a message of the reader goes in the code block


def test_a_file_over_the_limit_is_refused_before_it_is_read(after_step1, monkeypatch):
    up = Upload(b"x")
    up.size = response_csv.MAX_BYTES + 1
    up.getvalue = lambda: (_ for _ in ()).throw(AssertionError("read a file over the limit"))
    monkeypatch.setattr(step_response.st, "file_uploader", lambda *a, **k: up)
    _pick(after_step1, "Where does the response series come from?", SOURCE)
    after_step1.run()
    assert not after_step1.exception and "over the limit" in _errors(after_step1)


# --- the questions of the specification: nothing is answered for the user ---------------------------------------

def test_the_questions_have_no_preset_answer_and_the_build_waits_for_them(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, _csv(_trials()))
    _fill_answers(at, nature=None, adverse=None, label="", unit="")
    _build(at)
    e = _errors(at)
    for what in ("name", "unit", "kind of quantity", "which way"):
        assert what in e, what
    assert not (_study_dir(at) / "study.yaml").exists()


@pytest.mark.parametrize("label, unit", [("[x](http://evil.example)", "ug/L"), ("Fertility", "<b>x</b>"), ("x" * 80, "ug/L")])
def test_a_name_or_unit_that_is_markup_or_too_long_is_refused(after_step1, monkeypatch, label, unit):
    at = _start_upload(after_step1, monkeypatch, _csv(_trials()))
    _fill_answers(at, label=label, unit=unit)
    _build(at)
    assert ("name" in _errors(at) or "unit" in _errors(at)) and not (_study_dir(at) / "study.yaml").exists()


def test_text_from_the_file_is_not_rendered_as_markdown(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, "when,ec\n2010-01-05,[click](http://evil.example)\n")
    _fill_answers(at)
    _build(at)
    assert "[click](http://evil.example)" in _code(at)
    assert "[click](http://evil.example)" not in " ".join(m.value for m in at.markdown) + " ".join(e.value for e in at.error)


def test_two_columns_cannot_be_the_same(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, _csv(_trials()))
    _fill_answers(at, value_col="when")
    _build(at)
    assert "different columns" in _errors(at)


def test_the_name_of_the_uploaded_file_decides_nothing(after_step1, monkeypatch, tmp_path):
    _upload(monkeypatch, _csv(_trials()), name="../../../../tmp/evil.csv")
    _pick(after_step1, "Where does the response series come from?", SOURCE); after_step1.run()
    _fill_answers(after_step1)
    _build(after_step1)
    folder = _study_dir(after_step1)
    assert sorted(p.name for p in folder.iterdir()) == ["response.csv", "sst_daily.csv", "sst_daily.manifest.json", "study.yaml"]
    assert not Path("/tmp/evil.csv").exists()


def test_the_study_file_has_no_field_that_points_outside_the_folder(after_step1, monkeypatch):
    at = _full_flow(after_step1, monkeypatch, _csv(_trials()))
    import yaml
    raw = yaml.safe_load((_study_dir(at) / "study.yaml").read_text())
    assert "data_dir" not in raw and raw["responses"][0]["source"]["file"] == "response.csv"
    assert raw["responses"][0]["source"]["type"] == "csv"
    assert not [k for k in raw if k in ("data_dir",)]


def test_the_study_of_an_uploaded_file_runs_in_the_command(after_step1, monkeypatch, tmp_path):
    at = _full_flow(after_step1, monkeypatch, _csv(_trials(n_months=120)))
    folder = _study_dir(at)
    assert run_study.run_study(folder / "study.yaml", folder, tmp_path / "out", now=CLOCK) == 0


def test_a_study_file_that_does_not_load_back_is_refused_not_kept(after_step1, monkeypatch):
    monkeypatch.setattr(step_response.yaml, "safe_dump", lambda *a, **k: "format_version: 2\nunknown_field: 1\n")
    at = _full_flow(after_step1, monkeypatch, _csv(_trials()))
    assert not at.exception and "do not make a valid study" in _errors(at) + _code(at).lower() + " ".join(e.value for e in at.error).lower()
    assert "step2" not in _ctx(at).data


# --- values that look like codes for "missing": a warning, not a refusal -----------------------------------------

CODED = "when,ec\n" + "\n".join(f"{2005 + i // 12}-{1 + i % 12:02d}-10,{-999 if i % 7 == 3 else round(20 + (i * 7) % 13 + 0.5, 1)}" for i in range(60)) + "\n"


def test_a_missing_code_in_the_value_column_is_warned_before_the_build_and_the_study_still_builds(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, CODED)
    _fill_answers(at)
    at.run()
    warnings_before = " ".join(w.value for w in at.warning)
    assert "-999" in warnings_before and "read as real measurements" in warnings_before and "line" in warnings_before
    _build(at)
    assert not at.error, _errors(at)                      # a warning, never a refusal
    assert (_study_dir(at) / "study.yaml").exists()
    assert "-999" in " ".join(w.value for w in at.warning)  # and it stays on the page next to the result


def test_ordinary_data_give_no_missing_code_warning(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, _csv(_trials(n_months=120)))
    _fill_answers(at)
    at.run()
    assert not [w for w in at.warning if "read as real measurements" in w.value]


def test_the_warning_waits_for_the_decimal_separator_the_user_chose(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, CODED)
    _pick(at, "Field separator in the file", "comma (,)"); at.run()
    _pick(at, "Column with the date", "when"); _pick(at, "Column with the measurement value", "ec"); at.run()
    assert not [w for w in at.warning if "read as real measurements" in w.value]       # no decimal chosen yet: nothing is assumed


def test_the_suspect_values_are_looked_for_once_per_file_and_choices_not_at_every_run(after_step1, monkeypatch):
    calls, real = [], response_csv.missing_code_suspects
    monkeypatch.setattr(response_csv, "missing_code_suspects", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    at = _start_upload(after_step1, monkeypatch, CODED)
    _fill_answers(at)
    at.run(); at.run(); _by_label(at, "Name of the response (for the results)").set_value("Other").run()
    assert len(calls) == 1


def test_the_warning_cannot_be_used_to_inject_markup(after_step1, monkeypatch):
    text = "when,ec\n" + "\n".join(f"2008-{1 + i % 12:02d}-10,{-999 if i % 3 == 0 else 20 + i}" for i in range(24)) + "\n"
    at = _start_upload(after_step1, monkeypatch, text)
    _fill_answers(at)
    at.run()
    shown = " ".join(w.value for w in at.warning)
    assert "-999" in shown and "[" not in shown and "<" not in shown


# --- what the independent review found ----------------------------------------------------------------------

def test_a_file_with_a_quote_that_never_closes_is_a_problem_of_the_file_not_a_crash(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, '"when,ec\n' + "\n".join("x" * 200 for _ in range(1000)) + "\n")
    assert not at.exception                                         # at the upload, before any question
    _pick(at, "Field separator in the file", "comma (,)"); at.run()
    assert not at.exception and "cannot be parsed as CSV" in _code(at)


def test_a_negative_seed_is_refused_not_a_crash(after_step1):
    at = _generate(after_step1, seed=-1)                  # the widget has a minimum of 0; the step refuses a negative one too
    assert not at.exception
    made = _study_dir(at) / "response.manifest.json"
    assert "seed" in _errors(at) or (made.exists() and json.loads(made.read_text())["seed"] >= 0)


def test_a_gap_in_the_downloaded_series_stops_step_2_with_the_reason(monkeypatch):
    sst = pd.read_csv(FIXTURE_SST, parse_dates=["Datetime"]).set_index("Datetime")["Temperature"]
    sst[pd.Timestamp("2010-06-15")] = float("nan")
    c = RecordingClient(coverage=COVERAGE_2003.__class__(start=dt.date(2003, 1, 1), end=sst.index.max().date(), version="202511"), series=sst)
    monkeypatch.setattr(step_site, "make_client", lambda: c)
    at = _submit(_fill(_open()))
    assert not at.exception and "gap" in _shown(at) and "cannot go on" in _shown(at)
    assert not [r for r in at.radio if r.label.startswith("Where does the response")]


def test_the_message_counts_only_the_months_that_have_a_value(after_step1, monkeypatch):
    text = "when,ec\n" + "\n".join(f"2010-{m:02d}-10,{'' if m in (3, 4) else 20 + m}" for m in range(1, 10)) + "\n"
    at = _full_flow(after_step1, monkeypatch, text)
    assert not at.error, _errors(at)
    assert "7 months with a value" in _shown(at) and "2 months in between have none" in _shown(at)


def test_a_series_that_does_not_overlap_the_temperature_is_not_called_ready(after_step1, monkeypatch):
    rows = pd.DataFrame({"when": pd.date_range("1990-01-01", periods=38, freq="MS"), "ec": range(38)})
    at = _full_flow(after_step1, monkeypatch, _csv(rows))
    assert "The study is ready" not in _shown(at) and "no month" in _shown(at) and "no lag can be tested" in _shown(at)


def test_a_series_with_too_few_consecutive_months_says_how_many_it_has(after_step1, monkeypatch):
    rows = pd.DataFrame({"when": pd.date_range("2008-01-01", periods=4, freq="MS"), "ec": [1, 2, 3, 4]})
    at = _full_flow(after_step1, monkeypatch, _csv(rows))
    assert "3 pairs of consecutive months" in _shown(at) and "no lag can be tested" in _shown(at)


def test_only_months_that_follow_each_other_make_a_pair(after_step1, monkeypatch):
    months = pd.to_datetime(["2008-01-01", "2008-02-01", "2008-04-01", "2008-05-01", "2008-07-01", "2008-08-01"])
    at = _full_flow(after_step1, monkeypatch, _csv(pd.DataFrame({"when": months, "ec": [1, 2, 3, 4, 5, 6]})))
    assert "3 pairs of consecutive months" in _shown(at) and "6 months with a value" in _shown(at)


def test_a_good_series_says_how_many_pairs_the_command_will_have(after_step1, monkeypatch):
    at = _full_flow(after_step1, monkeypatch, _csv(_trials(n_months=120)))
    assert re.search(r"\d+ pairs of consecutive months", _shown(at)) and "no lag can be tested" not in _shown(at)


def test_changing_an_answer_after_the_build_flags_the_study_as_not_matching(after_step1, monkeypatch):
    at = _full_flow(after_step1, monkeypatch, _csv(_trials()))
    assert "answers changed since" not in _shown(at)
    _by_label(at, "Unit of the response (for example ug/L)").set_value("mg/L").run()
    assert "answers changed since" in _shown(at)


def test_a_second_file_does_not_leave_the_first_ones_study_standing(after_step1, monkeypatch):
    at = _full_flow(after_step1, monkeypatch, _csv(_trials()))
    up = Upload(_csv(_trials(seed=9)).encode())
    monkeypatch.setattr(step_response.st, "file_uploader", lambda *a, **k: up)
    at.run()
    assert "answers changed since" in _shown(at)
    assert next(s for s in at.selectbox if s.label == "Field separator in the file").value is None      # the new file starts unanswered


def test_a_test_series_whose_settings_were_moved_after_generating_is_flagged(after_step1):
    at = _generate(after_step1, seed=2)
    next(n for n in at.number_input if n.label == "Seed").set_value(3).run()
    assert "answers changed since" in _shown(at)


def test_the_test_series_is_made_under_the_global_lock(after_step1, monkeypatch):
    import climate_change_on_sea_urchins as ccsu
    held, real = [], known_truth.make_study
    monkeypatch.setattr(known_truth, "make_study", lambda *a, **k: (held.append(ccsu.RERUN_LOCK.locked()), real(*a, **k))[1])
    _generate(after_step1)
    assert held == [True]


def test_a_header_with_hundreds_of_columns_is_refused(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, ",".join(f"c{i}" for i in range(300)) + "\n" + ",".join("1" for _ in range(300)) + "\n")
    _pick(at, "Field separator in the file", "comma (,)"); at.run()
    assert not at.exception and "300 columns" in _errors(at)


def test_the_upload_limit_is_set_in_the_server_configuration_too():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text()
    assert re.search(r"(?m)^maxUploadSize\s*=\s*10\s*$", cfg)


@pytest.mark.parametrize("missing", ["decimal", "rows", "date_format", "nature", "adverse"])
def test_every_question_is_required_and_none_is_preset(after_step1, monkeypatch, missing):
    at = _start_upload(after_step1, monkeypatch, _csv(_trials()))
    _fill_answers(at, **{missing: None})
    _build(at)
    assert not at.exception and at.error and not (_study_dir(at) / "study.yaml").exists(), missing


def test_the_what_does_a_row_hold_question_starts_unanswered(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, _csv(_trials()))
    _pick(at, "Field separator in the file", "comma (,)"); at.run()
    assert next(r for r in at.radio if r.label == "What does one row of the file hold?").value is None


def test_the_validation_requires_each_of_these_answers():
    base = dict(delimiter="comma (,)", date_col="when", value_col="ec", decimal="point (12.5)", date_format="31/01/2020 (day/month/year)",
                rows="One measurement (several in a month are averaged)", label="F", unit="ug/L", nature="An index",
                adverse="It goes down (decrease)")
    assert step_response.validate_answers(base) == []
    for key in ("delimiter", "date_col", "value_col", "decimal", "date_format", "rows", "nature", "adverse", "label", "unit"):
        assert step_response.validate_answers({**base, key: None}), key
    comma = {**base, "decimal": "comma (12,5)"}
    assert any("same character" in e for e in step_response.validate_answers(comma))


def test_the_options_written_in_the_study(after_step1, monkeypatch):
    df = pd.DataFrame({"when": pd.date_range("2005-01-01", periods=40, freq="MS")})
    df["ec"] = np.linspace(20, 30, 40)
    df["lo"], df["hi"] = df["ec"] - 1, df["ec"] + 1
    text = df.assign(when=df["when"].dt.strftime("%Y-%m-%d")).to_csv(index=False, sep="\t")
    at = _start_upload(after_step1, monkeypatch, text)
    _fill_answers(at, delimiter="tab", rows="One value per month, already combined", adverse="It goes up (increase)",
                  nature="A toxicological endpoint (for example an EC50)")
    _pick(at, "Column with the lower end of the interval (optional)", "lo"); _pick(at, "Column with the upper end of the interval (optional)", "hi")
    _build(at)
    assert not at.error, _errors(at) + _code(at)
    r = load_study(_study_dir(at) / "study.yaml").responses[0]
    assert (r.adverse_direction, r.nature, r.source.delimiter, r.source.granularity) == ("increase", "toxicological_endpoint", "\t", "aggregated")
    assert (r.source.column_map.ci_low, r.source.column_map.ci_high) == ("lo", "hi") and r.aggregation is None


def test_a_failing_rebuild_removes_the_previous_good_study(after_step1, monkeypatch):
    at = _full_flow(after_step1, monkeypatch, _csv(_trials()))
    assert (_study_dir(at) / "study.yaml").exists()
    _by_label(at, "Unit of the response (for example ug/L)").set_value("<b>")
    _build(at)
    assert not (_study_dir(at) / "study.yaml").exists() and "step2" not in _ctx(at).data


def test_the_test_series_study_after_a_gap_in_the_sst_is_refused_with_the_reason(after_step1):
    sst_file = _ctx(after_step1).workdir / "inputs" / "sst_daily.csv"
    df = pd.read_csv(sst_file)
    df.loc[400:410, "Temperature"] = float("nan")
    df.to_csv(sst_file, index=False)
    at = _generate(after_step1)
    assert not at.exception and "gap" in _errors(at)


def test_the_study_can_be_downloaded(after_step1, monkeypatch):
    at = _full_flow(after_step1, monkeypatch, _csv(_trials()))
    assert any("study.yaml" in (b.proto.label if hasattr(b, "proto") else "") for b in at.get("download_button"))


# --- stale studies, sessions ----------------------------------------------------------------------------------

def test_a_new_download_in_step_1_drops_the_study_built_on_the_old_series(after_step1, monkeypatch):
    at = _full_flow(after_step1, monkeypatch, _csv(_trials()))
    assert (_study_dir(at) / "study.yaml").exists()
    _submit(_fill(at, name="Another place"))                      # step 1 downloaded again
    assert "step2" not in _ctx(at).data and not (_study_dir(at) / "study.yaml").exists()
    assert "The study is ready" not in _shown(at)


def test_another_session_does_not_see_the_study(after_step1, monkeypatch):
    a = _full_flow(after_step1, monkeypatch, _csv(_trials()))
    b = _open()
    assert not (_ctx(b).workdir / "study").exists() and "Fertility" not in _text(b)


# --- guards ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("forbidden", [r"\bos\.environ\b", r"\bgetenv\b", r"st\.cache", r"\bprint\(", r"logging", r"\bunsafe_allow_html\b",
                                       r"yaml\.load\(", r"\beval\(", r"\bexec\("])
def test_the_step_reads_no_environment_caches_nothing_and_loads_no_unsafe_yaml(forbidden):
    tree = ast.parse((PKG / "step_response.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    assert not re.search(forbidden, ast.unparse(tree)), forbidden


def test_the_neutral_modules_still_do_not_read_the_loaded_study():
    from test_app_entry import FORBIDDEN, _names
    assert not (_names(PKG / "step_response.py") & FORBIDDEN)


def test_the_commands_minimum_of_pairs_is_the_one_the_step_warns_with():
    assert step_response.MIN_PAIRS == run_study.MIN_PAIRS


# --- the wording of the warnings -------------------------------------------------------------------------------------

def test_the_warning_gives_the_value_as_it_is_in_the_file_and_agrees_in_number():
    one = step_response.suspect_messages([{"kind": "repeated", "value": 12.3456789, "count": 1, "of": 22, "lines": [2]}])[0]
    assert "1 of 22 values is exactly 12.3456789" in one and "(line 2)" in one
    many = step_response.suspect_messages([{"kind": "code", "value": -999.0, "count": 8, "of": 30, "lines": [2, 3, 4, 5, 6]}])[0]
    assert "8 of 30 values are exactly -999" in many and "(lines 2, 3, 4, 5, 6, ...)" in many
