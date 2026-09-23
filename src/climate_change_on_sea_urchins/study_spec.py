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

Scope for this step (branch v2.1/study-spec): only SiteSpec and the
ResponseSpec fields config.py's current constants + note-dati-sorgente.md
require. VariableSpec and WindowSpec exist as models (and appear in the
example study.yaml) because this step's plan calls for all five, but
neither is wired to any fetch script or to SPLIT_DATE yet -- that is
v2.1/provider-adapters and a later decision, respectively (see docs/adr/
0000-decisioni-rimandate.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


class SiteSpec(BaseModel):
    """An oceanographic monitoring site -- in practice, a Copernicus Marine
    grid cell (a nearest-cell approximation of the real site; see CLAUDE.md's
    "Contesto scientifico utile" on when that approximation breaks down)."""
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
    date: str
    value: str
    ci_low: str
    ci_high: str


class ResponseSourceSpec(BaseModel):
    """The per-trial (single-bioassay) source: one row per determination."""
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


class ResponseAggregationSpec(BaseModel):
    """How the per-trial source above rolls up into a period-level series.
    A distinct representation from the source itself: the count this
    produces exists only here, never per trial -- it counts trials
    aggregated into a period, not a biological sample size."""
    period: str = Field(..., description="e.g. 'month'")
    method: Literal["mean"]
    count_field: str = Field(
        ..., description="Canonical name for the resulting per-period trial "
        "count (e.g. 'n_assays'). Deliberately not 'n': that reads as "
        "biological sample size, which this is not."
    )


class ResponseSpec(BaseModel):
    """A response series: what is measured, on what population/individual,
    and how the raw per-trial source becomes the aggregated series most
    analyses run on. See docs/roadmap/note-dati-sorgente.md for why each
    field here is shaped the way it is."""
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
    source: ResponseSourceSpec
    aggregation: ResponseAggregationSpec


class VariableSpec(BaseModel):
    """An environmental variable fetched from a provider. Declared for this
    step's completeness; not yet consumed by any fetch script -- see module
    docstring."""
    id: str
    provider: str
    dataset: str
    variable: str
    unit: str


class WindowSpec(BaseModel):
    """A named temporal window of the study. Declared for completeness; not
    yet consumed anywhere -- see module docstring and docs/adr/0000."""
    id: str
    start: str
    end: str


class StudySpec(BaseModel):
    """Top-level study specification: everything a case declares about
    itself. See examples/livorno_paracentrotus/study.yaml for the current
    case, described with exactly today's config.py values."""
    id: str
    description: str
    sites: list[SiteSpec]
    responses: list[ResponseSpec]
    environment: list[VariableSpec] = Field(default_factory=list)
    windows: list[WindowSpec] = Field(default_factory=list)


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
        return StudySpec.model_validate(raw)
    except ValidationError as e:
        raise StudySpecError(f"{path}: invalid study spec:\n{e}") from e


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
