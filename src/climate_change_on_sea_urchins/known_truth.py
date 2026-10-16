"""M1.6: series with a known truth, to check the tool end to end.

`make_study` writes a study folder (study.yaml, response.csv, the SST given, and `response.manifest.json`) in
which the response depends on the monthly SST anomaly -- the driver of `ccsu-run-study` -- at a lag the tester
chooses, with the characters of the real series (the sheet of the sea-urchin case, measured on 2026-10-16 from the
frozen fixture): about two fifths of the months without a measurement, in runs; one to six trials per month, to
be aggregated. `verify` reads a finished run and says whether the lag was found. The generator declares every
choice in the manifest, and ccsu-run-study copies it into run.json, so the answer travels with the result.

    response(t) = level + scale * ( sqrt(share) * z(-anomaly(t - lag)) + sqrt(1 - share) * z(AR(1) noise) )

z() standardises over the months where the driver is known. `share` is the share of the variance of the
monthly value (the trial scatter comes on top). The sign is negative: the response goes down when the
temperature goes up, as the study declares (adverse_direction: decrease). The only randomness is one generator
seeded with `seed`, drawn in a fixed order.

    python -m climate_change_on_sea_urchins.known_truth make --sst sst_daily.csv --out DIR --seed 1 \\
        --lat 43.43 --lon 10.40 --baseline 2003 2012 [--lag 3] [--signal-share 0.5]
    python -m climate_change_on_sea_urchins.known_truth verify RUN_DIR      # exit 0 if the lag was recovered
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import dataset_builder

MAX_LAG = 12                      # the lags ccsu-run-study tests by default (analysis.max_lag_months)
NOISE_AR1 = 0.3
LEVEL, SCALE = 30.0, 5.0
TRIAL_SD_SHARE = 0.3              # the scatter between the trials of a month, as a share of SCALE
# Measured on 2026-10-16 on the sea-urchin sheet (tests/fixtures/paper_mpb_2026/data/ec50_sheets.csv, 280 months
# from 2003-02): 117 months without a measurement (42%) in 47 runs of mean length 2.49, the real months in 48 runs
# of mean length 3.4; trials per month 1:66 2:77 3:12 4:2 5:5 6:1 (163 months).
MEAN_MISSING_RUN = 2.49
MEAN_REAL_RUN = 3.4
TRIALS_PER_MONTH = {1: 66, 2: 77, 3: 12, 4: 2, 5: 5, 6: 1}
IMPUTATION = {"method": "centered_rolling_mean", "window_months": 12, "min_periods": 3, "passes": 2}   # as Livorno's study
SIGNIFICANCE = 0.05


class KnownTruthError(Exception):
    """A request the generator or the verdict cannot honour; the message says why."""


def _monthly_anomaly(sst: pd.DataFrame, baseline: tuple[int, int]) -> pd.Series:
    """The SST anomaly exactly as ccsu-run-study takes it: `anomaly_mean` of the dataset builder."""
    days = sst["Datetime"]
    months = pd.date_range(days.min().replace(day=1), days.max(), freq="MS")
    dummy = pd.DataFrame({"Datetime": months, "value": 1.0, "ci_upper": np.nan, "ci_lower": np.nan, "n": 1})
    ds = dataset_builder.build_dataset(sst=sst, response=dummy, imputation=None, baseline=baseline)
    return ds.mhw_monthly.set_index("Datetime")["anomaly_mean"].reindex(months)


def _standardise(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def _missing_months(rng: np.random.Generator, n: int) -> np.ndarray:
    """Two-state chain with geometric runs: the months without a measurement come in runs of mean MEAN_MISSING_RUN,
    the real ones in runs of mean MEAN_REAL_RUN (so 42% are missing)."""
    leave_missing, leave_real = 1 / MEAN_MISSING_RUN, 1 / MEAN_REAL_RUN
    missing = np.zeros(n, bool)
    missing[0] = rng.random() < MEAN_MISSING_RUN / (MEAN_MISSING_RUN + MEAN_REAL_RUN)
    for i in range(1, n):
        missing[i] = (rng.random() >= leave_missing) if missing[i - 1] else (rng.random() < leave_real)
    return missing


def make_study(sst_path, out_dir, *, seed: int, lat: float, lon: float, baseline: tuple[int, int], lag: int = 3,
               signal_share: float = 0.5, name: str = "Known-truth site", study_id: str = "known-truth") -> dict:
    """Write the study folder and return the declaration (also written as response.manifest.json)."""
    if not 1 <= lag <= MAX_LAG or lag % 12 == 0:
        raise KnownTruthError(f"lag {lag}: it must be between 1 and {MAX_LAG} months and not a multiple of 12, "
                              "or the seasonal cycle could be mistaken for the signal")
    if not 0.0 <= signal_share <= 1.0:
        raise KnownTruthError(f"signal share {signal_share}: it is a share of the variance, between 0 and 1")
    sst_path, out = Path(sst_path), Path(out_dir)
    sst = pd.read_csv(sst_path, parse_dates=["Datetime"])
    anomaly = _monthly_anomaly(sst, tuple(baseline))
    months = anomaly.index
    signal = (-anomaly.shift(lag)).to_numpy()
    defined = np.isfinite(signal)
    if defined.sum() < 60:
        raise KnownTruthError("fewer than 60 months with the driver known: too short a series to test a lag")

    rng = np.random.default_rng(seed)
    noise = np.zeros(len(months))
    for i in range(1, len(months)):
        noise[i] = NOISE_AR1 * noise[i - 1] + rng.normal(0, np.sqrt(1 - NOISE_AR1 ** 2))
    z = np.full(len(months), np.nan)
    z[defined] = (np.sqrt(signal_share) * _standardise(signal[defined])
                  + np.sqrt(1 - signal_share) * _standardise(noise[defined]))
    value = LEVEL + SCALE * z

    idx = np.flatnonzero(defined)
    missing = _missing_months(rng, len(idx))
    trial_n = np.array(list(TRIALS_PER_MONTH)), np.array(list(TRIALS_PER_MONTH.values()), float)
    rows = []
    for j, i in enumerate(idx):
        count = int(rng.choice(trial_n[0], p=trial_n[1] / trial_n[1].sum()))
        days = rng.integers(1, 29, size=count)
        scatter = rng.normal(0, TRIAL_SD_SHARE * SCALE, size=count)
        if missing[j]:
            continue
        for d, e in zip(days, scatter):
            rows.append(((months[i] + pd.Timedelta(days=int(d) - 1)).strftime("%Y-%m-%d"), round(float(value[i] + e), 4)))
    response = pd.DataFrame(rows, columns=["date", "value"])

    out.mkdir(parents=True, exist_ok=True)
    response.to_csv(out / "response.csv", index=False)
    shutil.copyfile(sst_path, out / "sst_daily.csv")
    sst_manifest = sst_path.with_name(sst_path.stem + ".manifest.json")
    if sst_manifest.is_file():
        shutil.copyfile(sst_manifest, out / "sst_daily.manifest.json")
    observed = int((~missing).sum())
    truth = {
        "format": 1, "kind": "synthetic_response", "seed": seed, "lag_months": lag,
        "signal_share_of_variance": signal_share, "expected_sign_of_r": "negative",
        "driver": "SST anomaly", "driver_baseline": list(baseline),
        "noise_ar1": NOISE_AR1, "level": LEVEL, "scale": SCALE, "trial_scatter_sd": TRIAL_SD_SHARE * SCALE,
        "missing": {"mean_missing_run_months": MEAN_MISSING_RUN, "mean_real_run_months": MEAN_REAL_RUN,
                    "measured_on": "the sea-urchin sheet, 2026-10-16 (ec50_sheets.csv, 280 months)"},
        "trials_per_month": {str(k): v for k, v in TRIALS_PER_MONTH.items()},
        "months": {"first": months[idx[0]].strftime("%Y-%m"), "last": months[idx[-1]].strftime("%Y-%m"),
                   "defined": int(len(idx)), "with_value": observed},
        "formula": "level + scale * (sqrt(share) * z(-anomaly(t - lag)) + sqrt(1 - share) * z(AR(1) noise)); "
                   "z standardises over the months where the driver is known",
    }
    (out / "response.manifest.json").write_text(json.dumps(truth, indent=2) + "\n")

    spec = {
        "format_version": 2, "id": study_id,
        "description": "A response generated from the SST anomaly with a known lag (see response.manifest.json): "
                       "it checks the tool, it is not a measurement.",
        "mhw_climatology": {"baseline_start_year": baseline[0], "baseline_end_year": baseline[1]},
        "sites": [{"id": "site", "lat": lat, "lon": lon, "name": name, "bbox_delta": 0.1}],
        "environment": [{"catalog_id": "sst_daily"}],
        "responses": [{
            "id": "synthetic-response", "label": "Synthetic response", "nature": "index", "adverse_direction": "decrease",
            "unit": "1",
            "source": {"type": "csv", "file": "response.csv", "temporal_resolution": "month", "granularity": "per_trial",
                       "column_map": {"date": "date", "value": "value"}},
            "aggregation": {"period": "month", "method": "mean", "count_field": "n_assays"},
            "imputation": IMPUTATION}],
    }
    (out / "study.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return truth


def verify(run_dir) -> dict:
    """Did ccsu-run-study find the lag the response was generated with? Recovered means: the largest |r| over the
    tested lags falls on that lag, with the expected sign (the response going down as the anomaly goes up, that is
    `r_toward_adverse` > 0 for a decreasing response), and its Benjamini-Hochberg p is below 0.05. The other lags
    that pass the correction are reported, not hidden."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    truth = run.get("response_manifest")
    if not truth or truth.get("kind") != "synthetic_response":
        raise KnownTruthError("the run's inputs carry no known-truth declaration (response.manifest.json): "
                              "there is nothing to check the result against")
    ccf = pd.read_csv(run_dir / "ccf.csv")
    tested = ccf[ccf["p_value"].notna()]
    k = int(truth["lag_months"])
    top = int(tested.loc[tested["spearman_r"].abs().idxmax(), "lag"]) if len(tested) else None
    at_k = tested[tested["lag"] == k]
    r_adverse = float(at_k["r_toward_adverse"].iloc[0]) if len(at_k) else None
    p_bh = float(at_k["p_bh"].iloc[0]) if len(at_k) else None
    verdict = {
        "recovered": bool(top == k and r_adverse is not None and r_adverse > 0 and p_bh < SIGNIFICANCE),
        "known_lag": k, "signal_share_of_variance": truth.get("signal_share_of_variance"), "seed": truth.get("seed"),
        "top_lag": top, "r_toward_adverse_at_lag": r_adverse, "p_bh_at_lag": p_bh,
        "lags_significant_after_bh": [int(x) for x in tested.loc[tested["p_bh"] < SIGNIFICANCE, "lag"]],
        "tests_declared": run["tests"]["declared"], "tests_performed": run["tests"]["performed"],
    }
    return verdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="known_truth", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    mk = sub.add_parser("make", help="write a study folder with a response of known lag")
    mk.add_argument("--sst", type=Path, required=True, help="daily SST (Datetime,Temperature), e.g. from ccsu-download-sst")
    mk.add_argument("--out", type=Path, required=True)
    mk.add_argument("--seed", type=int, required=True)
    mk.add_argument("--lat", type=float, required=True)
    mk.add_argument("--lon", type=float, required=True)
    mk.add_argument("--baseline", type=int, nargs=2, required=True, metavar=("START_YEAR", "END_YEAR"))
    mk.add_argument("--lag", type=int, default=3)
    mk.add_argument("--signal-share", type=float, default=0.5)
    vf = sub.add_parser("verify", help="say whether a finished run found the known lag (exit 0 if it did)")
    vf.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.command == "make":
            t = make_study(args.sst, args.out, seed=args.seed, lat=args.lat, lon=args.lon, baseline=tuple(args.baseline),
                           lag=args.lag, signal_share=args.signal_share)
            print(f"written to {args.out}: lag {t['lag_months']} months, {t['signal_share_of_variance']:.0%} of the variance, "
                  f"{t['months']['with_value']} of {t['months']['defined']} months with a measurement, seed {t['seed']}")
            return 0
        verdict = verify(args.run_dir)
    except (KnownTruthError, dataset_builder.DatasetError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    (args.run_dir / "verification.json").write_text(json.dumps(verdict, indent=2) + "\n")
    word = "recovered" if verdict["recovered"] else "NOT recovered"
    print(f"{word}: known lag {verdict['known_lag']}, largest |r| at lag {verdict['top_lag']}, "
          f"BH p at the known lag {verdict['p_bh_at_lag']}; lags significant after BH: {verdict['lags_significant_after_bh']}")
    return 0 if verdict["recovered"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
