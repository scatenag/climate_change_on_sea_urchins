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
    _pick(at, "Decimal separator of the numbers", cfg["decimal"]); _pick(at, "Format of the dates", cfg["date_format"])
    _pick(at, "What does one row of the file hold?", cfg["rows"])
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
        assert not at.exception and expect in _text(at) + _code(at)
    assert not (_study_dir(at) / "study.yaml").exists()


def test_a_file_that_is_not_utf8_gets_the_excel_hint(after_step1, monkeypatch):
    at = _start_upload(after_step1, monkeypatch, "when,ec\n2010-01-05,caffè\n".encode("latin-1"))
    assert "UTF-8" in _text(at) + _code(at) and "Excel" in _text(at) + _code(at)


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


def test_the_study_can_be_downloaded(after_step1, monkeypatch):
    at = _full_flow(after_step1, monkeypatch, _csv(_trials()))
    assert any("study.yaml" in (b.proto.label if hasattr(b, "proto") else "") for b in at.get("download_button"))


# --- stale studies, sessions ----------------------------------------------------------------------------------

def test_a_study_built_for_earlier_step_1_inputs_is_flagged_when_they_change(after_step1, monkeypatch):
    at = _full_flow(after_step1, monkeypatch, _csv(_trials()))
    _by_label(at, "Name of the site").set_value("Another place").run()
    assert "earlier inputs" in _text(at) and "step 1" in _text(at).lower()


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
