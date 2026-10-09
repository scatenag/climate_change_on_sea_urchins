"""
Declarative study specification (V2.1): describes a case -- site, response-
series source and aggregation, environment, temporal window -- as data, not
as module-level constants in config.py. config.py loads a study.yaml
through this module instead of hardcoding those values (see docs/adr/0000-
decisioni-rimandate.md for what is deliberately NOT moved here yet).

Fields are driven by what the real EC50 source sheet contains, not an
idealized case -- see docs/roadmap/note-dati-sorgente.md, which documents
its actual traps: `pos`/`neg` are NOT controls (they are the confidence
interval's asymmetric half-widths), most dates record only the month, and
the negative-control replicates are fields of the trial, not a separate
series.

Two representations, kept distinct on purpose (a single ResponseSpec that
mixed them would blur what only exists at the per-trial level -- the
control replicates -- with what only exists at the aggregated level -- the
assay count):
  - ResponseSourceSpec: the per-trial (single-bioassay) source, one row per
    determination, not yet aggregated to any period.
  - ResponseAggregationSpec: how that per-trial source rolls up into the
    period-level series most analyses actually run on.

Format versions (M1.1). `format_version: 1` is the Livorno study as it has always been written
(the key may be absent); `format_version: 2` adds what a study of anyone's own data needs: the
environment named by id in the variable catalogue (catalog.py) instead of dataset names, a CSV
response source, `split_date` optional, the response imputation declared (absent unless declared)
and the contaminant of the endpoint declared. A format this version does not know is refused,
naming it. A format-2 study loads and validates here but the pipeline does not run it yet
(ccsu-run-study, M1.5): common.py says so explicitly.

VariableSpec and WindowSpec exist as models (and appear in the example
study.yaml) for completeness but aren't wired to any fetch script or
consumed anywhere yet -- that is v2.1/provider-adapters (see docs/adr/
0000-decisioni-rimandate.md for what else is still deliberately deferred).
data_dir, split_date (on ResponseSpec, ADR-0007) and mhw_climatology are
wired: common.py resolves and validates them at its own load time.
"""
from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from . import catalog

SUPPORTED_FORMATS = (1, 2)


class SiteSpec(BaseModel):
    """An oceanographic monitoring site -- in practice, a Copernicus Marine
    grid cell (a nearest-cell approximation of the real site; see CLAUDE.md's
    "Contesto scientifico utile" on when that approximation breaks down)."""
    model_config = ConfigDict(extra="forbid")
    id: str
    lat: float
    lon: float
    name: str
    bbox_delta: float = Field(..., description="Bounding-box half-width, degrees")


class ResponseColumnMap(BaseModel):
    """Explicit source-column -> canonical-field mapping for the per-trial
    response source. Never deduced from column names: in the current source
    sheet, the columns named `pos`/`neg` are NOT controls -- they are
    `UL-EC50` and `EC50-LL`, the asymmetric confidence-interval half-widths
    (see note-dati-sorgente.md). Declaring the mapping explicitly is what
    keeps that kind of misreading from happening again, to a human or to
    code working from the sheet."""
    model_config = ConfigDict(extra="forbid")
    date: str
    value: str
    ci_low: str
    ci_high: str


class ResponseSourceSpec(BaseModel):
    """The per-trial (single-bioassay) source: one row per determination."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["google_sheet"]
    sheet_id: str
    temporal_resolution: str = Field(
        ..., description="Declared, not implied. In the current source, most "
        "rows record only the month, not a real day -- treating the dates as "
        "day-resolution would be a silent overstatement of precision."
    )
    column_map: ResponseColumnMap
    control_columns: list[str] | None = Field(
        default=None,
        description="Per-trial validity-control replicate columns (e.g. "
        "negative-control malformation-rate replicates), present only on a "
        "subset of trials. These are fields of the observation, not an "
        "independent series -- they share the trial's own date.",
    )


class ResponseCsvColumnMap(BaseModel):
    """Column mapping for a CSV response source: explicit, never deduced from names. The
    confidence interval is optional, but its two ends come together."""
    model_config = ConfigDict(extra="forbid")
    date: str
    value: str
    ci_low: str | None = None
    ci_high: str | None = None

    @model_validator(mode="after")
    def _ci_in_pairs(self):
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("ci_low and ci_high must be given together, or both left out")
        return self


_BARE_CSV_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.csv$")


class ResponseCsvSourceSpec(BaseModel):
    """A response series in a CSV file the user provides (format 2). The file is named, never
    located: a bare file name, resolved by whoever loads the study (the package, the upload),
    so a study can never point the tool at a path of its own choosing."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["csv"]
    file: str
    temporal_resolution: Literal["day", "month"] = Field(
        ..., description="Declared, not implied, as for the sheet source.")
    granularity: Literal["per_trial", "aggregated"] = Field(
        ..., description="per_trial: one row per determination, rolled up by `aggregation`; "
        "aggregated: one row per period already.")
    column_map: ResponseCsvColumnMap
    control_columns: list[str] | None = None

    @field_validator("file")
    @classmethod
    def _file_is_a_bare_csv_name(cls, v: str) -> str:
        if not _BARE_CSV_NAME.match(v):
            raise ValueError(
                f"file {v!r} must be a bare file name ending in .csv (letters, digits, '.', '_', '-'; "
                "no directories, not starting with a dot)")
        return v


class ResponseAggregationSpec(BaseModel):
    """How the per-trial source above rolls up into a period-level series.
    A distinct representation from the source itself: the count this
    produces exists only here, never per trial -- it counts trials
    aggregated into a period, not a biological sample size."""
    model_config = ConfigDict(extra="forbid")
    period: str = Field(..., description="e.g. 'month'")
    method: Literal["mean"]
    count_field: str = Field(
        ..., description="Canonical name for the resulting per-period trial "
        "count (e.g. 'n_assays'). Deliberately not 'n': that reads as "
        "biological sample size, which this is not."
    )


class ImputationSpec(BaseModel):
    """How months without a real measurement are filled. A scientific choice, written in the
    study (CLAUDE.md invariant 6); a response that declares none is never imputed."""
    model_config = ConfigDict(extra="forbid")
    method: Literal["centered_rolling_mean"]
    window_months: int = Field(..., ge=2)
    min_periods: int = Field(..., ge=1)
    passes: int = Field(..., ge=1, le=3, description="How many times the fill is applied, each "
                        "pass on the series the previous one produced.")

    @model_validator(mode="after")
    def _min_periods_fit_the_window(self):
        if self.min_periods > self.window_months:
            raise ValueError("min_periods cannot exceed window_months")
        return self


class ContaminantSpec(BaseModel):
    """The substance an ecotoxicological endpoint is about; analyses specific to a substance
    (copper speciation) declare they need it."""
    model_config = ConfigDict(extra="forbid")
    name: str
    cas: str | None = None


class ResponseSpec(BaseModel):
    """A response series: what is measured, on what population/individual,
    and how the raw per-trial source becomes the aggregated series most
    analyses run on. See docs/roadmap/note-dati-sorgente.md for why each
    field here is shaped the way it is."""
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str = Field(
        ..., description="Human-readable name for output artifacts (results/ "
        "CSV/JSON headers, dashboard labels) -- distinct from `id`, which is "
        "a slug. Writing a literal like 'EC50' at an output boundary instead "
        "of reading it from here is the artifact-identity invariant "
        "(CLAUDE.md #4) being violated, not honored."
    )
    nature: Literal["population_proxy", "index", "toxicological_endpoint", "biomarker"] = Field(
        ..., description="What kind of quantity this is. Never default to "
        "'biomarker' as a generic label for 'response series' -- most "
        "response series, including the current EC50 case, are not "
        "biomarkers, and calling one that is scientifically wrong."
    )
    adverse_direction: Literal["decrease", "increase"] = Field(
        ..., description="Which way the value moves when the organism's "
        "condition gets WORSE -- not 'higher/lower is better', since "
        "'better' flips meaning depending on who's asking and this field "
        "must not."
    )
    unit: str = Field(..., description="UCUM unit string, e.g. 'ug/L'")
    split_date: str | None = Field(
        default=None, description="ISO date (YYYY-MM-DD): the pre/post regime-shift "
        "boundary estimated for THIS response series (docs/adr/0001 -- a "
        "rank-based estimate, deliberately not reconciled with "
        "changepoint.py's own QLR/AR(1) estimate on the same series). Per-"
        "response, not a shared constant: a second response series has its "
        "own regime shift, possibly at a different date. Required in format 1; "
        "optional in format 2, where without it the pre/post analyses are switched "
        "off with the reason. Validated at common.py's load time against the actual "
        "response series' date range, not here (this module never reads data/)."
    )
    source: Annotated[ResponseSourceSpec | ResponseCsvSourceSpec, Field(discriminator="type")]
    aggregation: ResponseAggregationSpec | None = None
    imputation: ImputationSpec | None = None
    contaminant: ContaminantSpec | None = None

    @field_validator("split_date")
    @classmethod
    def _split_date_is_iso(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                dt.date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"split_date {v!r} is not an ISO date (YYYY-MM-DD)") from None
        return v

    @model_validator(mode="after")
    def _aggregation_when_rows_are_trials(self):
        needs = isinstance(self.source, ResponseSourceSpec) or self.source.granularity == "per_trial"
        if needs and self.aggregation is None:
            raise ValueError(f"response {self.id!r}: per-trial rows need an `aggregation` rule")
        return self


class VariableSpec(BaseModel):
    """An environmental variable fetched from a provider. Declared for this
    step's completeness; not yet consumed by any fetch script -- see module
    docstring."""
    model_config = ConfigDict(extra="forbid")
    id: str
    provider: str
    dataset: str
    variable: str
    unit: str


class EnvironmentRef(BaseModel):
    """An environmental variable named by its id in the catalogue (format 2)."""
    model_config = ConfigDict(extra="forbid")
    catalog_id: str


class WindowSpec(BaseModel):
    """A named temporal window of the study: one pipeline run computes each
    declared window's statistics into results/<study_id>/<window_id>/.
    Checked here against itself only; overlap with the actual data is
    checked in common.py (this module never reads data/)."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(
        ..., pattern=r"^[a-z0-9][a-z0-9_-]*$",
        description="Becomes a directory name under results/<study_id>/, "
        "hence a lowercase slug."
    )
    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def _start_before_end(self):
        if not self.start < self.end:
            raise ValueError(f"window {self.id!r}: start {self.start} must precede end {self.end}")
        return self


class MhwClimatologySpec(BaseModel):
    """The daily-SST baseline period Marine Heatwave detection (Hobday et
    al. 2016) computes its per-day-of-year threshold from. A scientific
    choice, not a code default (CLAUDE.md invariant #6) -- previously
    mhw_detection.py's own CLIM_START/CLIM_END module constants."""
    model_config = ConfigDict(extra="forbid")
    baseline_start_year: int
    baseline_end_year: int


class StudySpec(BaseModel):
    """Top-level study specification: everything a case declares about
    itself. See examples/livorno_paracentrotus/study.yaml for the current
    case, described with exactly today's config.py values."""
    model_config = ConfigDict(extra="forbid")
    format_version: int = Field(default=1, description="Format of this file; absent means 1.")
    id: str
    description: str
    data_dir: str | None = Field(
        default=None, description="Where this study's data/ lives, relative to this "
        "study.yaml's own location. Required in format 1. Resolved to an absolute path by "
        "load_study(), which also checks it exists -- callers (common.py) "
        "always see an absolute, existing directory."
    )
    mhw_climatology: MhwClimatologySpec
    sites: list[SiteSpec]
    responses: list[ResponseSpec]
    environment: list[VariableSpec | EnvironmentRef] = Field(default_factory=list)
    windows: list[WindowSpec] = Field(default_factory=list)

    @field_validator("format_version")
    @classmethod
    def _format_is_known(cls, v: int) -> int:
        if v not in SUPPORTED_FORMATS:
            raise ValueError(
                f"format_version {v} is not supported (this version reads formats "
                f"{' and '.join(map(str, SUPPORTED_FORMATS))})")
        return v

    @model_validator(mode="after")
    def _rules_of_the_declared_format(self):
        if self.format_version == 1:
            if self.data_dir is None:
                raise ValueError("data_dir is required in format 1")
            for ref in self.environment:
                if isinstance(ref, EnvironmentRef):
                    raise ValueError(
                        f"environment entry {{catalog_id: {ref.catalog_id}}} needs format_version 2")
            for r in self.responses:
                if r.split_date is None:
                    raise ValueError(f"response {r.id!r}: split_date is required in format 1")
                if isinstance(r.source, ResponseCsvSourceSpec):
                    raise ValueError(f"response {r.id!r}: a csv source needs format_version 2")
            return self
        # format 2
        for ref in self.environment:
            if not isinstance(ref, EnvironmentRef):
                raise ValueError(
                    "format_version 2 environment entries refer to the variable catalogue "
                    "({catalog_id: ...}), not to datasets")
        ids = [ref.catalog_id for ref in self.environment]
        for dup in sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"environment lists {dup!r} more than once")
        for catalog_id in ids:
            try:
                variable = catalog.get(catalog_id)
            except catalog.CatalogError as e:
                raise ValueError(str(e)) from None
            for site in self.sites:
                if not catalog.covers(variable, site.lat, site.lon):
                    d = variable.domain
                    raise ValueError(
                        f"site {site.name!r} (lat {site.lat}, lon {site.lon}): catalogue variable "
                        f"{catalog_id!r} has no data there, outside its domain "
                        f"(lat {d.lat_min:.2f} to {d.lat_max:.2f}, lon {d.lon_min:.2f} to {d.lon_max:.2f})")
        return self

    @model_validator(mode="after")
    def _window_ids_unique(self):
        ids = [w.id for w in self.windows]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"window ids must be unique, repeated: {dup}")
        return self


class StudySpecError(Exception):
    """Raised for both YAML syntax errors and schema validation errors.

    YAML syntax errors carry a line/column (from PyYAML's own parser).
    Validation errors carry a field path (from pydantic) instead of a line
    number: pydantic validates the already-parsed structure, which has no
    memory of where in the file each value came from. Getting real line
    numbers there too would need a line-preserving YAML loader (e.g.
    ruamel.yaml) -- not added for this step; field paths already say
    which value is wrong precisely enough for the cases this exists to
    catch."""


def load_study(path: str | Path) -> StudySpec:
    """Load and validate a study.yaml. Raises StudySpecError (never a bare
    OSError, yaml.YAMLError or pydantic.ValidationError) on any problem."""
    path = Path(path)
    if not path.is_file():
        raise StudySpecError(f"{path}: no such file")

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        raise StudySpecError(f"{path}: invalid YAML{where}: {e}") from e

    try:
        spec = StudySpec.model_validate(raw)
    except ValidationError as e:
        raise StudySpecError(f"{path}: invalid study spec:\n{e}") from e

    if spec.data_dir is not None:
        resolved_data_dir = (path.parent / spec.data_dir).resolve()
        if not resolved_data_dir.is_dir():
            raise StudySpecError(
                f"{path}: data_dir {spec.data_dir!r} does not exist ({resolved_data_dir})"
            )
        spec.data_dir = str(resolved_data_dir)
    return spec


DEFAULT_STUDY_PATH = (Path(__file__).resolve().parent.parent.parent
                      / "examples" / "livorno_paracentrotus" / "study.yaml")


def selected_study_path() -> Path:
    """The study.yaml this process runs: named by the CCSU_STUDY environment
    variable, or DEFAULT_STUDY_PATH (Livorno) if unset."""
    return Path(os.environ.get("CCSU_STUDY") or DEFAULT_STUDY_PATH)


def load_selected_study() -> StudySpec:
    """Loads selected_study_path() -- the one place the selection happens;
    config.py and common.py both call this (common.py can't import
    config.py: config imports this package, whose __init__ imports
    common)."""
    env = os.environ.get("CCSU_STUDY")
    try:
        return load_study(selected_study_path())
    except StudySpecError as e:
        if env:
            raise StudySpecError(f"CCSU_STUDY={env!r}: {e}") from e
        raise


def export_schema(out_path: str | Path) -> None:
    """Write StudySpec's JSON Schema to out_path (editor autocompletion,
    third-party validation)."""
    import json
    Path(out_path).write_text(json.dumps(StudySpec.model_json_schema(), indent=2) + "\n")


def validate_study_cli(argv: list[str] | None = None) -> None:
    """Console entry point: `ccsu-validate-study path/to/study.yaml`."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: ccsu-validate-study <path/to/study.yaml>", file=sys.stderr)
        raise SystemExit(2)
    try:
        spec = load_study(argv[0])
    except StudySpecError as e:
        print(f"✗ {e}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"✓ {argv[0]}: valid study spec "
        f"(id={spec.id!r}, {len(spec.sites)} site(s), {len(spec.responses)} response(s))"
    )
