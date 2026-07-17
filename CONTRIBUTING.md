# Contributing

Thank you for your interest in this project.

## Reporting issues

Please open a [GitHub issue](https://github.com/scatenag/climate_change_on_sea_urchins/issues)
to report bugs, request features, or ask questions.

## Running the tests

```bash
pip install -e ".[test]"
pytest tests/
```

Some tests require pre-computed results. Run the full analysis pipeline first:

```bash
ccsu-run-pipeline
```

## Updating the data

EC50 data are fetched automatically from ISPRA's Google Sheets export on each dashboard session.
To refresh the Copernicus Marine environmental data locally:

```bash
# Requires Copernicus Marine credentials:
# export COPERNICUSMARINE_SERVICE_USERNAME=your_username
# export COPERNICUSMARINE_SERVICE_PASSWORD=your_password

pip install -e ".[acquisition]"
python scripts/fetch_copernicus_update.py
python scripts/build_dataset.py
ccsu-run-pipeline
```

`ccsu-run-pipeline` regenerates the MHW event catalogue (`data/mhw_events.csv` and friends) from
`data/sst_daily.csv` as its first step — there is no separate MHW-detection script to remember
to run.

## Code style

- Python 3.10+. Core dependencies are declared in `pyproject.toml`; install with `pip install -e .`
- The analysis modules in `src/climate_change_on_sea_urchins/` must write their outputs to
  `results/` and be runnable independently (each exposes a `run()` function;
  `pipeline.py`/`ccsu-run-pipeline` orchestrates all of them in order)
- The dashboard (`src/climate_change_on_sea_urchins/dashboard.py`, exposed at the repo root as
  `app.py` and via `ccsu-dashboard`) must read only pre-computed CSVs — no live analysis in
  the dashboard, with one exception: the Pre/Post split tab recomputes Kruskal-Wallis/
  Mann-Whitney live, since non-parametric tests on a few hundred points are cheap enough to
  run at request time
- `scripts/` (data acquisition) stay outside the installable package: they require Copernicus
  credentials and are meant to be run occasionally/manually, not imported

## Licence

By contributing you agree that your contributions will be licensed under the
[MIT Licence](LICENSE).
