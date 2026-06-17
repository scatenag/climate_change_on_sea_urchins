# Contributing

Thank you for your interest in this project.

## Reporting issues

Please open a [GitHub issue](https://github.com/scatenag/climate_change_on_sea_urchins/issues)
to report bugs, request features, or ask questions.

## Running the tests

```bash
pip install -r requirements.txt pytest
pytest tests/
```

Some tests require pre-computed results. Run the full analysis pipeline first:

```bash
python analysis/run_all.py
```

## Updating the data

EC50 data are fetched automatically from ISPRA's Google Sheets export on each dashboard session.
To refresh the Copernicus Marine environmental data locally:

```bash
# Requires Copernicus Marine credentials:
# export COPERNICUSMARINE_SERVICE_USERNAME=your_username
# export COPERNICUSMARINE_SERVICE_PASSWORD=your_password

python scripts/fetch_copernicus_update.py
python scripts/build_dataset.py
python scripts/detect_mhw.py
python analysis/run_all.py
```

## Code style

- Python 3.10+, standard library + packages in `requirements.txt`
- Analysis modules in `analysis/` must write their outputs to `results/` and be runnable independently
- `app.py` must read only pre-computed CSVs — no live analysis in the dashboard, with one
  exception: the Pre/Post split tab recomputes Kruskal-Wallis/Mann-Whitney live, since
  non-parametric tests on a few hundred points are cheap enough to run at request time

## Licence

By contributing you agree that your contributions will be licensed under the
[MIT Licence](LICENSE).
