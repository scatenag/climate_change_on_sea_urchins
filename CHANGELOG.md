# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **`results/` is per-study now** (ADR-0008, V2.2 prerequisite): `common.results_dir(study_id)`
  (introduced already-generic in "one resolver" above) returns `results/<study_id>/` instead of
  always `results/` — a second study's pipeline run can no longer overwrite Livorno's. The 64
  existing files moved with `git mv` to `results/livorno-paracentrotus/`, content unchanged; the
  golden master is unaffected (its fixtures assign `RESULTS` directly per module, independent of
  what `results_dir()` would compute for a real run — the same property already relied on for the
  `results_dir()` PR). `README.md`'s four links to specific manuscript-reproduction output files
  updated to the new path; `tests/test_thermal_legacy_vif.py` built its path as `ROOT / "results"`
  instead of reading `common.RESULTS` — silently switched to `pytest.skip()` under the old path,
  now fixed to use the resolver. `scripts/make_*_figure.py` and `build_narrative_notebook.py`
  still build `ROOT / "results"` themselves and will not find Livorno's files after this move —
  left as-is (out of the installable package, no test exercises them, same treatment as the
  twin `data/` pattern found in `scripts/*.py` during the previous PR); noted in
  `docs/roadmap/STATO.md`.

- `tests/test_golden_master.py`: regression test for the entire `results/` directory (64
  files), not just the ~15 manuscript-cited numbers `test_paper_values.py` covers. Reruns the
  full pipeline (all 16 Python modules, plus the R DLNM script when available) against the
  same frozen fixture `test_paper_values.py` uses, and compares every output file against a
  frozen reference (`tests/fixtures/results_v1_5_0/`) with per-file tolerances declared and
  justified by category (closed-form, seeded bootstrap/RNG, iterative MLE optimizer, and the
  known ARIMA-prewhitening scale-sensitivity case) rather than one uniform tolerance. Marked
  `@pytest.mark.golden` (reruns the whole pipeline, ~2-3 min) so it can be run standalone with
  `pytest -m golden`, separate from the fast default suite. Verified with a sabotage test:
  corrupting one reference value makes the test fail naming that exact file/field; restoring
  it returns to green.
- `tests/conftest.py::golden_pipeline_results`: generalizes the existing `paper_results`
  fixture from 4 modules to the full pipeline (`pipeline._MODULES`).
- **V2.1, "specifica e astrazione della risposta" (branch `v2.1/study-spec`)**: declarative
  `examples/livorno_paracentrotus/study.yaml`, validated by new pydantic models in
  `src/climate_change_on_sea_urchins/study_spec.py` (`SiteSpec`, `ResponseSpec` — split into
  a per-trial `ResponseSourceSpec` and a separate `ResponseAggregationSpec`, since the raw
  source and the monthly-aggregated series are two different representations, not one —
  `VariableSpec`, `WindowSpec`, `StudySpec`). `config.py` inverted to read this file instead
  of hardcoding site/source values as module constants, re-exporting the exact same names
  (`SITE_LAT`, `SITE_LON`, `SITE_NAME`, `BBOX_DELTA`, `EC50_SHEET_ID`, `EC50_EXPORT_URL`) so
  none of the six files that import them changed. `CO2_PA_TO_UATM` deliberately stays a
  hardcoded physical constant, not part of the spec. New `ccsu-validate-study` console script
  and `docs/schema/study.schema.json` (JSON Schema export). New direct dependencies:
  `pydantic>=2,<3`, `PyYAML>=6,<7` (both were already present transitively via Streamlit, now
  declared). Verified with the golden master: zero drift, no new tolerances needed.
- **V2.1, response-series abstraction -- complete** (four PRs: #8, #10, #11, #12). `EC50` is no
  longer a hardcoded identity anywhere in `src/` outside `common.py` (the one place allowed to
  know the on-disk literal) and `dashboard.py` (separate branch, not started).
  - `ResponseSpec` gains a `label` field (display identity for outputs -- for Livorno,
    `"EC50"`).
  - `common.py` gains `RESPONSE_COL`/`IMPUTED_COL` (indirection constants -- not an alias:
    `load_data()` renames the CSVs' fixed on-disk columns into these at the load boundary, so
    every downstream module reads the response column through the constant, never the
    literal) and `load_ec50_raw()`/`load_ec50_monthly()`, extending the single data-reading
    boundary to `data/ec50_raw.csv`/`ec50_sheets.csv` -- previously read directly by
    `changepoint.py`, `period_split.py`, and (#12, closing this out) `negative_control.py`
    (which doesn't touch the response column, only the negative-control replicate columns,
    but was the same invariant-#5 boundary violation).
  - `config.py` exports `RESPONSE_SPEC`; `pipeline.py` loads it once and passes it as
    `response=` to the 9 modules whose output carries the response's identity
    (`timeseries`, `correlations`, `stationarity`, `regime_shift`, `period_split`,
    `cu_speciation`, `thermal_legacy`, `forecast`, `mhw_analysis`, `mhw_lag_extra` -- 10 names,
    `mhw_analysis` and `mhw_lag_extra` added when the MHW family was migrated in #10) -- every
    row/column label, `"variable"`/`"series"` field, per-variable dict key, output filename
    and derived column name in those modules' output comes from `response.label`, never from
    the internal `RESPONSE_COL`.
  - `mhw_analysis.py`'s imputed-months masking (`target == "EC50" and "EC50_imputed" in
    df.columns`, duplicated at three call sites) became `_mask_imputed(df, target, values)`:
    masks wherever `f"{target}_imputed"` exists as a column, independent of `target`'s literal
    name. `tests/test_mhw_analysis.py` covers it directly.
  - `docs/adr/0000-decisioni-rimandate.md` #6: `label` isn't sanitized for filename use yet,
    deferred to V2.2's second case.
  - `RESPONSE_COL`/`IMPUTED_COL` flipped from `"EC50"`/`"EC50_imputed"` to `"response"`/
    `"response_imputed"` (#11) once every module had migrated off the literal. This is the
    verification step the whole indirection was for: it surfaced two real leaks the earlier
    module-by-module review missed -- `timeseries.py` was never migrated at all (its
    `trend_EC50.csv`/`trends_*.csv` output silently switched to `trend_response.csv` and a
    `"variable": "response"` field), and `period_split.py`'s `dist_EC50.csv` had already been
    given the right *filename* but not the right *column header* inside the file. Both fixed;
    golden master returned to 66/66 with zero new tolerances.
  - Golden master held at zero drift through every step (Livorno's `label == "EC50"` even
    after the flip, since `RESPONSE_COL` and `label` are independent by design).
- **Study selection (V2.2 prerequisite)**: `config.py` loads the study named by the
  `CCSU_STUDY` environment variable (path to a `study.yaml`), defaulting to
  `examples/livorno_paracentrotus/study.yaml` -- existing runs, CI and the auto-update workflow
  are unchanged. A bad path fails naming `CCSU_STUDY`. `config.STUDY_ID` exported. Not yet
  safe to run the pipeline with another study: data and results are still read from and
  written to the shared `data/`/`results/` until per-study namespacing lands.
- **One resolver for where results live** (V2.2 prerequisite): `common.results_dir(study_id)`,
  still returning `results/` for every study, so moving to a per-study directory becomes a
  change to that function alone. `common.RESULTS`, the dashboard, `tests/test_pipeline.py`,
  `tests/test_data_quality.py` and the R DLNM script (new optional output-directory argument,
  passed by the auto-update workflow from Python's resolver) all go through it;
  `tests/test_results_dir.py` fails if any module builds the path itself. Study selection moved
  from `config.py` to `study_spec.load_selected_study()` so `common.py` can use it too
  (importing `config` from `common` would be circular). No output changes.
- **`data_dir`, `split_date`, `mhw_climatology` join the study spec** (V2.2 prerequisite, ADR-0007
  for `split_date`): the same "one resolver, value unchanged" treatment as `RESULTS` above,
  applied to `data/`.
  - `StudySpec.data_dir` (relative to the study.yaml itself, resolved to an absolute, validated-
    to-exist path by `load_study()`) and the new `common.DATA` constant. Every `data/` read in
    `src/` now goes through `common.DATA` or one of `common.py`'s `load_*()` functions
    (`load_data`, `load_ec50_raw`, `load_ec50_monthly`, `load_mhw_annual`, new `load_sst_daily`)
    instead of `ROOT / "data"` — `regime_shift.py`, `mhw_lag_annual.py`,
    `mhw_annual_changepoint.py`, `thermal_legacy.py`, `dashboard.py` migrated.
    `tests/test_data_boundary.py::test_no_module_builds_the_data_path_itself` fails if any
    module in `src/` builds the path itself again.
  - **This search found (and this PR fixes) a real, live bug**: `mhw_detection.py` computed
    `SST_PATH`/`OUT_EVENTS`/`OUT_MONTHLY`/`OUT_ANNUAL` once, at import time, from `common.ROOT`.
    `tests/conftest.py`'s fixtures redirect `ROOT`/`DATA` for the frozen golden-master fixture
    *after* import, which had no effect on those already-bound paths: every golden-master run
    silently read the **real** `data/sst_daily.csv` and overwrote the **real**
    `data/mhw_events.csv`/`mhw_monthly.csv`/`mhw_annual.csv`, while the rest of the pipeline
    (correctly redirected) used the frozen fixture's own static copies instead. The golden
    master stayed green throughout — not because the redirection worked, but because MHW
    detection is deterministic and the real `data/sst_daily.csv` had not changed since the
    fixture was frozen (verified byte-identical, 2026-09-25); a single auto-update in that
    window would have gone undetected. Fixed by computing these paths from `common.DATA`
    *inside* `run()`, at call time, never bound to a separate module-level name — the same
    bug class as three earlier incidents this project has had (`.gitignore`, golden-master
    tolerances, `#4`→`#9`), now with two dedicated regression tests
    (`tests/test_data_boundary.py`) instead of only a fixed instance: one exercises
    `mhw_detection.run()` against a deliberately-wrong directory with synthetic content and
    checks both that the output came from it and that the real `data/` was untouched; the
    other is a minimal, self-contained demonstration of why the bug shape is dangerous, for
    whoever adds the next `data/`-touching module. `tests/conftest.py::golden_pipeline_results`
    also gained its own general-purpose guard: it snapshots the real `data/` and `results/`
    directories before running and asserts they're byte-for-byte, mtime-for-mtime unchanged
    after — independent of which module or mechanism would cause a write.
  - `StudySpec.mhw_climatology` (`baseline_start_year`/`baseline_end_year`): the MHW-detection
    climatology baseline period (Hobday et al. 2016), previously `mhw_detection.py`'s own
    `CLIM_START`/`CLIM_END` — a scientific choice, not a code default (CLAUDE.md invariant #6).
  - `ResponseSpec.split_date` (ADR-0007): `common.SPLIT_DATE` is no longer a bare module
    constant — it's per-response (a second response series has its own regime shift) and
    **validated at `common.py`'s load time against the actual response series' date range**,
    rejecting the spec with an explicit message if `split_date` would leave `period_split.py`
    (or anything else slicing on it) with an empty or single-point pre/post side, instead of
    surfacing that downstream in the pipeline. `docs/adr/0000` entry 5 (this exact deferred
    decision) is closed and folded into ADR-0007.
  - `tests/test_pipeline.py`'s synthetic-different-site test and `tests/test_study_selection.py`
    updated for `data_dir` (both were written before this field existed).

### Changed

- **ARIMA order selection discards candidates that don't converge, and the golden master
  stops pinning numeric values it can't reproduce (issue #9, branch `fix/arima-convergence`)**.
  - `_best_arima_order()` (`mhw_analysis.py`) used to pick the lowest-AIC candidate
    regardless of whether the optimizer actually converged. `warnings.simplefilter("ignore")`
    was silencing statsmodels' own `ConvergenceWarning` along with everything else. It now
    captures warnings instead of blanket-ignoring them, and discards any candidate that
    raises `ConvergenceWarning` or whose own `mle_retvals` says it didn't converge, before
    comparing AIC (`_fit_arima_if_converged()`, shared with a new `order` parameter on
    `compute_ccf_prewhitened()` that skips the search entirely when a fixed order is passed).
  - This fixes a real bug — a discarded fit's residuals are unreliable in a way AIC alone
    doesn't reveal — but does **not** make order selection itself reproducible across
    machines. Confirmed directly: two consecutive CI runs of the identical branch (no code
    change between them) picked different orders for `mhw_days` on GitHub's own hosted
    runners. For a near-degenerate driver (`mhw_days`: 45% zero months; `mhw_severe_intensity`:
    94%), *which* candidates converge at all depends on the runner's hardware/BLAS path, which
    can change which order wins by AIC even after discarding non-convergent ones. No
    Bonferroni-survival check or manuscript comparison was performed on the resulting numbers
    for this reason: they are not a property of the code and data alone, so they are not
    reported here or anywhere else. A deterministic order-selection procedure is left to
    V3.1 (see issue #9).
  - Consequently, `tests/test_golden_master.py` no longer compares the fitted values in
    `results/ccf_results_prewhitened.csv`, `results/prewhitening_diagnostics.json`, or
    `results/robustness_severe_ccf_note.json`'s `diagnostics` block against the frozen
    reference — the previous `ARIMA_FIT` tolerance tier (rtol=0.5, atol=0.02, introduced
    2026-09-21) is retired, and no numeric tolerance replaces it for these three: a new
    `STRUCTURAL_ONLY_FILES` set checks columns, row count, which driver/lag combinations
    exist, and where NaN falls, never the fitted numbers themselves (see the module
    docstring). `tests/test_mhw_analysis.py` (new) separately covers the fit/filter/
    correlate computation on its own, with a forced fixed order, under `LOOSE` tolerance.
  - `mhw_robustness.py::run_severe_ccf`'s ARIMA-prewhitened arm for `mhw_severe_intensity` is
    marked `not_applicable`: `results/robustness_severe_ccf.csv`'s `r_arima`/`p_arima` columns
    are always `NaN` (unconditionally, so still `TIGHT`-tolerance-safe), with the reason (and
    diagnostics) written to the new `results/robustness_severe_ccf_note.json`. Reason: 13 of
    the driver's 17 nonzero months fall after the 2016-06 EC50 regime shift (`SPLIT_DATE`);
    the filter is estimated on the driver alone and doesn't remove that shift from the target,
    so the residual correlation is confounded by it rather than reflecting a lagged response —
    confirmed by re-running with the shift removed from EC50 (pre/post demeaned): significant
    lags collapse from 10/13 to 1/13 at the order convergent on this machine. The
    first-differenced arm (`r_diff`/`p_diff`) is unaffected and is the retained result for
    this driver.

### Fixed

- **`tests/test_mhw_analysis.py` read the real `data/`, not the frozen fixture, since #17.** Its
  fixture redirected `common.ROOT` only; since #17, `load_data()` reads `common.DATA`, no longer
  derived from `ROOT`. Green while the real `data/` equalled the fixture; broke when the
  2026-09-25 auto-update added one EC50 month (n 163 -> 164 at lag 0; fixture: 163 real months,
  real data: 164). Main was red on this test from that commit on, unseen: auto-update commits skip
  CI. Fixed by redirecting `common.DATA` too. `tests/test_data_boundary.py` gains a static guard
  over `tests/`: any test redirecting `common.ROOT` must also redirect `common.DATA` (the existing
  search only covered `src/`). #17 had fixed the same omission in `test_pipeline.py` but missed
  this file.

- `forecast.py` read `data/mhw_annual.csv` as `RESULTS.parent / "data"`: a direct data read
  the single-boundary search had missed (it doesn't spell `ROOT / "data"`), and one that would
  have broken as soon as results moved under a subdirectory. Now `common.load_mhw_annual()`.
- EC50 unit labeled `mg/L` instead of `µg/L` throughout the dashboard (23 occurrences: axis
  titles, hover templates, metrics, captions) — a factor-1000 error visible on the public app
  (issue #3). The values were always µg/L (the manuscript reports 46.54 µg/L); only the label
  was wrong. The same wrong unit also fixed in a `forecast.py` comment, in
  `scripts/explore_mhw_ec50.py`'s axis labels, and in a `tests/test_data_quality.py` comment.
- The CO₂ unit note in `README.md` and in the dashboard said Copernicus's CF metadata implied
  µatm. It doesn't: the `standard_name` of `spco2` is associated with Pascal, so Copernicus
  declares the right unit and the misreading was this pipeline's. Same correction as the one
  already applied to `config.py`/`study.yaml`.

- `.gitignore`'s `trend_*.csv` rule (unanchored, matches any directory depth) was silently
  dropping 6 files from `git add tests/fixtures/results_v1_5_0/` — the committed golden-master
  reference shipped incomplete; caught by the coverage-guard test in `test_golden_master.py` on
  the first real CI run. Fixed with a scoped negation, not by removing the original rule.
- Two `test_golden_master.py` tolerances were set from single-machine measurements (both showed
  exactly 0 relative difference locally) and turned out far too tight on a different machine:
  `ccf_results_prewhitened.csv` and `robustness_severe_ccf.csv`'s `r_arima`/`p_arima` columns
  (the latter mis-categorized as fully deterministic — missed that it shares the same ARIMA
  prewhitening call). New `ARIMA_FIT` tolerance tier (rtol=0.5, atol=0.02, explicitly a
  gross-error check only) plus per-column tolerance overrides for files that mix a stable and
  an ARIMA-derived column.

## [1.5.0] - 2026-09-14

The version cited in Sartori, Scatena, Gaion et al. (submitted, *Marine Pollution Bulletin*) as
an independent means of verifying its published numbers.

### Fixed

- CO2 units corrected: Copernicus's `spco2` is Pascal, not microatmospheres (commit `7d4b75d`,
  2026-09-10) — conversion applied at ingestion via `config.CO2_PA_TO_UATM`. `results/` had not
  been regenerated since that fix landed (it was regenerated in three separate, inconsistent
  runs predating the seed work below); this release's single official run (see "vendemmia" below)
  is the first to reflect it consistently. Verified via a controlled before/after comparison
  (current data vs. data restored to just before `7d4b75d`, both with current code/seeds):
  differs only where expected (CO2 means/SD/trend/distributions), identical elsewhere
  (correlations, both changepoint analyses, the regime-shift stress index, copper speciation,
  forecast) to floating-point noise.
- Deterministic ordering of the raw EC50 sequence by (Datetime, ID) before the QLR/AR(1)
  changepoint — commit `cb333cd`.

### Added

- `negative_control.py`: new module reproducing manuscript section 3.6 (assay negative-control
  series) — trend, pre/post level and dispersion, and a QLR/AR(1) changepoint search (reuses
  `changepoint.qlr_ar1_changepoint`, not reimplemented). `data/ec50_raw.csv` gained three new
  columns (`ctrl_neg_rep1/2/3`, via `scripts/fetch_ec50.py`) to carry the source data. Also adds
  a data-quality check flagging single-replicate outliers (>3 pooled-SD from the other two
  replicates of the same trial) — flags trial 224 (2020-01-01); confirmed by D. Sartori (2026-09-14)
  to be two distinct trials, not a data-entry error, so the value stands uncorrected.
- `period_split.py`: adds `results/period_contrast_raw.json`, the manuscript's section 3.1
  pre/post contrast computed on the 295 individual EC50 trials (`data/ec50_raw.csv`), alongside
  the existing monthly-series contrast.
- `thermal_legacy.py`: adds `run_threshold_sensitivity()` / `results/thermal_threshold_sensitivity.csv`
  (manuscript Table S2) — the same detrended/partial tests swept over 22–26°C at the fixed
  24-month window, with 24°C (the a-priori primary threshold) marked as such, not presented on
  equal footing with the robustness-check values. `run()`'s existing computation/output is
  unchanged.
- `mhw_annual_changepoint.py`: new module applying `qlr_ar1_changepoint` to the annual MHW
  exposure metric. Investigated for the manuscript's section 3.5 2nd paragraph, but no (metric,
  year-range) variant tried reproduced the reference values (phi, break year, bootstrap p, CI90)
  together; that paragraph was removed from the manuscript as a result (D. Sartori, 2026-09-14).
  `results/mhw_annual_changepoint.json` keeps every variant tried, flagged
  `cited_in_manuscript: false`, as a record of what was investigated.
- The single official pipeline run ("vendemmia") this release's `results/` reflects — regenerated
  once from current code, current seeds, and the data vintage cut on the dates below, superseding
  three earlier, inconsistent runs.
- `tests/fixtures/paper_mpb_2026/`: frozen snapshot of `data/*.csv` this vintage was computed
  from. `tests/test_paper_values.py` (via the new `tests/conftest.py`) now re-runs the four
  manuscript-value modules above against this fixture instead of reading precomputed `results/`,
  so it no longer depends on the live, auto-updating `data/` and stays green as the real series
  grows past this vintage.
- `requirements-lock.txt`: `pip freeze` of the exact environment this vintage was verified
  reproducible in (re-running the pipeline against it changes nothing in `results/`).

### Changed

- README badge and `CITATION.cff` now point to the Zenodo **concept DOI** (10.5281/zenodo.19352308)
  instead of a version-specific one (10.5281/zenodo.22304864, the v1.4.0 record) — the concept
  DOI always resolves to the latest release, so it will not need updating again.
- Repo root decluttered ahead of the release: `analysis.ipynb` and `narrative_sentinel_regime_shift.ipynb`
  moved to `notebooks/`; `marineHeatWaves.py` (vendored, unused reference implementation) moved
  to `legacy/`. `paper.md`/`paper.bib` (the abandoned JOSS submission's source) removed from the
  repository entirely — kept locally under `drafts/` (already git-ignored), not tracked going
  forward. Three paper-only illustrations (`figures/fig0_pipeline.png`, `fig1_timeseries.png`,
  `fig2_app_screenshot.png`) removed with them; the four analysis figures `dashboard.py` loads at
  runtime (`fig_cu_speciation_decomposition.png`, `fig_thermal_legacy.png`, `fig_regime_shift.png`,
  `fig_mhw_lag_annual.png`) stay tracked in `figures/` — an initial pass removed those too, caught
  and reverted before merge.
- Removed obsolete/superseded tracked files (recoverable from git history/old releases, not kept
  locally): `docs/MHW_ANALYSIS_RESEARCH.md` (unreferenced research notes predating the current
  detrended/robustness analyses, moved to `drafts/`); `results/forecast_bio_{bad,good,mean}.csv`
  and `results/mhw_lag_correlations.csv` (outputs of the frozen `legacy/analysis_2023_exploratory.ipynb`,
  not produced by any current module); `assets/sea_urchin.png` (superseded by
  `sea_urchin_transparent.png`, the only one `dashboard.py` uses).

## [1.4.0] - 2026-09-04

### Added

- Explicit `SPLIT_DATE` set to 2016-06-01, propagated to all modules.
- New `changepoint.py` module with a QLR (Quandt-Andrews) breakpoint estimation procedure on
  AR(1) residuals.
- VIF (Variance Inflation Factor) per window in the thermal-legacy analysis.

### Changed

- Updated the Zenodo DOI to the v1.4.0 record (10.5281/zenodo.22304864).
