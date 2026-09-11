"""
Negative-control analysis (manuscript section 3.6).

The bioassay's negative control is a per-trial validity field, not an
independent series: three replicate readings of percent malformed larvae
(out of 100 examined) recorded alongside each EC50 determination, present
only when the trial's validity criterion was supported by data (232 of 295
trials in data/ec50_raw.csv; see scripts/fetch_ec50.py). Its role here is as
a check on the EC50 decline: if the assay's own quality-control metric also
drifted or shifted, part of the EC50 decline could be a method artifact
rather than a biological signal. It does not.

Four checks, all on the per-trial mean of the three replicates, ordered by
(Datetime, ID) -- the same tie-break convention changepoint.py uses for the
ordinal EC50 sequence, for the same reason (~110 of 295 trials share a
Datetime):
  1. Spearman rank correlation against time (monotonic trend, full record).
  2. Mann-Whitney U for a level shift, pre/post SPLIT_DATE.
  3. Levene's test for a dispersion change, same split.
  4. qlr_ar1_changepoint() (reused from changepoint.py, not reimplemented)
     for an unconstrained changepoint search on the series itself.

Two things surfaced while reproducing the manuscript's published numbers for
this section (flagged here, not silently corrected -- see G. Scatena,
2026-09-11):

  - The published pre/post contrast (13.7%/14.1%, Mann-Whitney p=0.41,
    Levene p=0.38) is NOT reproduced at the current SPLIT_DATE
    (2016-06-01): that split gives the same levels but Mann-Whitney
    p=0.378 and Levene p=0.627. It IS reproduced almost exactly at
    2016-01-01, the split that was in effect before SPLIT_DATE moved to
    2016-06-01 (see common.py) -- the manuscript's section 3.1 pre/post
    split moved but this one, in section 3.6, was not carried forward.
    Both cuts agree qualitatively (no significant level or dispersion
    change); this module reports the current SPLIT_DATE as primary (single
    source of truth) and records the published-split numbers separately,
    labeled as such, for provenance.
  - The manuscript's Levene p=0.38 corresponds to scipy's plain
    stats.levene(pre, post) call (center='median' by default) at the
    2016-01-01 split -- not a deliberately chosen robust variant. All three
    center= variants are recorded for both splits regardless.
  - The manuscript's Spearman p=0.30 is OPEN, not a rounding difference:
    trial ID 224 (2020-01-01) has replicates [14, 1, 14] -- the "1" is a
    source-data outlier, still uncorrected in the sheet as of this run.
    This module computes rho=+0.0706/p=0.2843 directly from
    data/ec50_raw.csv as fetched (outlier included, unmodified). D. (2026-
    09-11) independently reports that computing "with the outlier still
    present" gives rho=+0.0666/p=0.3124 -- matching the manuscript's
    0.07/0.30 -- and "without it" gives rho=+0.0706/p=0.2843, matching this
    module's as-fetched value; i.e. that account attributes this module's
    own as-fetched output to the corrected scenario, not the as-fetched one
    it actually is. Not reconciled here (dropping trial 224, or replacing
    its outlier replicate with the mean of the other two, both tried and
    neither reproduces 0.0666/0.3124). Both value pairs are recorded in
    results/negative_control.json, labeled OPEN, pending Davide's decision
    on whether/how to correct trial 224's source data. See
    replicate_outlier_flags below for the general check this case
    motivated.
  - qlr_ar1_changepoint's module default trim=0.15 (used for the EC50
    series) excludes the manuscript's reported break (index 203 of 232,
    ~88% through the series) from its search window and finds a different,
    also non-significant break instead. trim=0.10 -- within the 10-15%
    range the manuscript's methods declare -- locates the reported break.
    Both are recorded; the conclusion (no significant changepoint) holds
    under either.

Output: results/negative_control.json
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from .changepoint import DEFAULT_B, DEFAULT_SEED, qlr_ar1_changepoint
from .common import RESULTS, ROOT, SPLIT_DATE

REPLICA_COLS = ["ctrl_neg_rep1", "ctrl_neg_rep2", "ctrl_neg_rep3"]

# The split the manuscript's published section 3.6 numbers actually
# correspond to -- see module docstring. Not the current SPLIT_DATE.
PUBLISHED_SPLIT_DATE = pd.Timestamp("2016-01-01")

LEVENE_CENTERS = ("mean", "median", "trimmed")  # scipy default is 'median'

# Quality check added 2026-09 after trial ID 224 (2020-01-01, replicates
# [14, 1, 14]) turned out to be an uncorrected source-data outlier that
# shifted the Spearman trend result (see module docstring). Flags any
# single replicate that deviates from the mean of the OTHER TWO replicates
# of its own trial by more than OUTLIER_SD_THRESHOLD standard deviations of
# the pooled replicate distribution (all 3*n readings across every trial
# with a control) -- a fixed, global dispersion reference, not a per-trial
# one: with only two "other" values, a per-trial standard deviation is
# degenerate (zero whenever the other two happen to tie, which is common at
# this measurement's resolution) and would falsely flag any disagreement at
# all. Verified against trial 224: flags exactly that one record and no
# other, on the current data.
OUTLIER_SD_THRESHOLD = 3.0


def _flag_replicate_outliers(ctrl, threshold=OUTLIER_SD_THRESHOLD):
    values = ctrl[REPLICA_COLS].to_numpy(dtype=float)
    global_sd = values.ravel().std(ddof=1)

    flags = []
    for i, col in enumerate(REPLICA_COLS):
        other_mean = np.delete(values, i, axis=1).mean(axis=1)
        deviation = np.abs(values[:, i] - other_mean)
        for j in np.flatnonzero(deviation > threshold * global_sd):
            flags.append({
                "trial_id": int(ctrl["ID"].iloc[j]),
                "date": ctrl["Datetime"].iloc[j].date().isoformat(),
                "flagged_replicate": col,
                "replicate_values": [float(v) for v in values[j]],
                "deviation_from_other_two": float(deviation[j]),
                "threshold": float(threshold * global_sd),
            })
    return global_sd, flags


def _load_negative_control():
    raw = pd.read_csv(ROOT / "data" / "ec50_raw.csv", parse_dates=["Datetime"])
    if not set(REPLICA_COLS).issubset(raw.columns):
        raise ValueError(
            "data/ec50_raw.csv is missing the negative-control replicate "
            "columns -- re-run scripts/fetch_ec50.py."
        )
    raw = raw.sort_values(["Datetime", "ID"]).reset_index(drop=True)
    n_total = len(raw)

    ctrl = raw.dropna(subset=REPLICA_COLS).copy().reset_index(drop=True)
    ctrl["ctrl_mean"] = ctrl[REPLICA_COLS].mean(axis=1)
    return ctrl, n_total


def _pre_post_contrast(ctrl, split_date):
    pre = ctrl.loc[ctrl["Datetime"] < split_date, "ctrl_mean"]
    post = ctrl.loc[ctrl["Datetime"] >= split_date, "ctrl_mean"]

    _, p_mwu = stats.mannwhitneyu(pre, post, alternative="two-sided")
    levene_p = {c: float(stats.levene(pre, post, center=c).pvalue) for c in LEVENE_CENTERS}

    return {
        "split_date": split_date.date().isoformat(),
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
        "mean_pre": float(pre.mean()),
        "mean_post": float(post.mean()),
        "mannwhitney_p": float(p_mwu),
        "levene_p": levene_p,
    }


def _changepoint_at(ctrl, trim, B, seed):
    res = qlr_ar1_changepoint(ctrl["ctrl_mean"].values, trim=trim, B=B, seed=seed)
    k = min(max(res["break_index"], 0), len(ctrl) - 1)
    return {
        **res,
        "trim": trim,
        "break_date": ctrl["Datetime"].iloc[k].date().isoformat(),
        "n_pre": res["break_index"],
        "n_post": res["n"] - res["break_index"],
        "mean_pre": float(ctrl["ctrl_mean"].iloc[:res["break_index"]].mean()),
        "mean_post": float(ctrl["ctrl_mean"].iloc[res["break_index"]:].mean()),
    }


def run(B=DEFAULT_B, seed=DEFAULT_SEED):
    ctrl, n_total = _load_negative_control()
    n_with = len(ctrl)

    # 1) monotonic trend, full record -- invariant to whether time is coded
    # as row order or calendar date once the series is ordered (see docstring).
    rho, p_trend = stats.spearmanr(np.arange(n_with), ctrl["ctrl_mean"])

    current_split = _pre_post_contrast(ctrl, SPLIT_DATE)
    published_split = _pre_post_contrast(ctrl, PUBLISHED_SPLIT_DATE)

    cp_010 = _changepoint_at(ctrl, 0.10, B, seed)
    cp_015 = _changepoint_at(ctrl, 0.15, B, seed)

    global_sd, outlier_flags = _flag_replicate_outliers(ctrl)

    summary = {
        "n_trials_total": n_total,
        "n_with_negative_control": n_with,
        "n_without_negative_control": n_total - n_with,
        "note_validity_criterion": (
            "The negative control is a per-trial validity field, present "
            f"only when the trial's validity criterion was supported by "
            f"data: {n_with} of {n_total} trials."
        ),
        "trend": {
            "status": "open",
            "spearman_rho": float(rho),
            "spearman_p": float(p_trend),
            "manuscript_spearman_rho": 0.07,
            "manuscript_spearman_p": 0.30,
            "note": (
                "OPEN, not an accepted rounding difference: trial ID 224 "
                "(2020-01-01, replicates [14, 1, 14]) is a source-data "
                "outlier (see replicate_outlier_flags), still uncorrected "
                "in the sheet as of this run. spearman_rho/spearman_p above "
                "are this module's own computation on data/ec50_raw.csv "
                "exactly as fetched -- outlier included, unmodified. "
                "Independently, G. Scatena/Davide (2026-09-11) report that "
                "computing 'with the outlier still present' gives "
                "rho=+0.0666/p=0.3124 (matching the manuscript's 0.07/0.30) "
                "and 'without it' gives rho=+0.0706/p=0.2843 (matching this "
                "module's own value above) -- i.e. they attribute this "
                "module's as-fetched output to the CORRECTED scenario, not "
                "the as-fetched one this module actually ran. That "
                "attribution is not reconciled here (several ways of "
                "excluding/correcting trial 224 were tried and none "
                "reproduced 0.0666/0.3124); recorded as reported, pending "
                "Davide's decision on the source row."
            ),
        },
        "pre_post_current_split": current_split,
        "pre_post_published_split": {
            **published_split,
            "note": (
                "Reproduces the manuscript's published section 3.6 numbers "
                "(13.7%/14.1%, Mann-Whitney p=0.41, Levene p=0.38 with "
                "scipy's default center='median') almost exactly. This is "
                "the split in effect before SPLIT_DATE moved to 2016-06-01 "
                "(see common.py) -- not carried forward to this section."
            ),
        },
        "changepoint_qlr_ar1": {
            "trim_0.10_primary": cp_010,
            "trim_0.15_module_default": cp_015,
            "note": (
                "trim=0.10 locates the manuscript's reported break (Oct "
                "2022, n=203/29); trim=0.15 (changepoint.py's module "
                "default, used for the EC50 series) excludes that index "
                "from its search window (it sits at ~88% of the series) "
                "and finds a different, also non-significant break instead. "
                "Both trims are within the 10-15% range the manuscript's "
                "methods declare; both agree on the conclusion: no "
                "significant changepoint in the negative-control series."
            ),
        },
        "replicate_outlier_flags": {
            "threshold_sd": OUTLIER_SD_THRESHOLD,
            "pooled_replicate_sd": float(global_sd),
            "n_flagged": len(outlier_flags),
            "flags": outlier_flags,
            "note": (
                "Flags any single replicate deviating from the mean of the "
                "OTHER TWO replicates of its own trial by more than "
                f"{OUTLIER_SD_THRESHOLD:g} standard deviations of the "
                "pooled replicate distribution (all readings across every "
                "trial with a control). Added after trial 224 (2020-01-01) "
                "turned out to be an uncorrected source-data outlier (see "
                "the 'trend' section above) -- to catch the next one."
            ),
        },
    }

    with (RESULTS / "negative_control.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"✓ negative control: n={n_with}/{n_total} with control | "
        f"trend rho={rho:+.3f} p={p_trend:.3f} [OPEN, see JSON] | "
        f"pre/post(SPLIT_DATE) {current_split['mean_pre']:.1f}%/"
        f"{current_split['mean_post']:.1f}% MWU p={current_split['mannwhitney_p']:.3f} | "
        f"QLR(trim=0.10) break={cp_010['break_date']} p={cp_010['p_value']:.3f} | "
        f"replicate outliers flagged: {len(outlier_flags)}"
    )


if __name__ == "__main__":
    run()
