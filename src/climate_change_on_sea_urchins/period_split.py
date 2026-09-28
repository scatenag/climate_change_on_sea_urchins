"""
Pre/Post 2016 statistical comparison.
Outputs: results/kruskal_stats.json, results/period_means.csv,
results/period_contrast_raw.json
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from .common import (
    load_data, load_ec50_raw, default_results_dir, ALL_COLS, MHW_COLS, RESPONSE_COL,
    SPLIT_DATE, default_response_spec, in_window, split_date_in_window,
)

# Windows (V2.2): the whole module is a pre/post contrast -- statistic only,
# on the window's points alone (monthly series, per-trial determinations,
# and the response re-imputed inside the window for the env-means rows).
# When SPLIT_DATE does not leave both a pre and a post side inside the
# window, the module is skipped for that window and writes nothing.
SUPPORTS_WINDOW = True


def _raw_trial_contrast(window=None):
    """Pre/post SPLIT_DATE contrast on the 295 individual EC50 bioassay
    determinations (data/ec50_raw.csv), as opposed to the monthly series
    used elsewhere in this module. Manuscript section 3.1 reports this
    trial-level contrast alongside the monthly one.
    """
    raw = load_ec50_raw()
    raw = raw[in_window(raw["Datetime"], window)]

    pre = raw.loc[raw["Datetime"] < SPLIT_DATE, RESPONSE_COL]
    post = raw.loc[raw["Datetime"] >= SPLIT_DATE, RESPONSE_COL]
    _, p_mwu = stats.mannwhitneyu(pre, post, alternative="two-sided")

    diff_abs = float(pre.mean() - post.mean())
    return {
        "split_date": SPLIT_DATE.date().isoformat(),
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
        "mean_pre": float(pre.mean()),
        "mean_post": float(post.mean()),
        "sd_pre": float(pre.std()),
        "sd_post": float(post.std()),
        "median_pre": float(pre.median()),
        "median_post": float(post.median()),
        "diff_abs": diff_abs,
        "decline_pct": 100.0 * diff_abs / pre.mean(),
        "mannwhitney_p": float(p_mwu),
        "note": (
            "The split point (SPLIT_DATE) is estimated from this same "
            f"data, and the {len(pre) + len(post)} trials are serially correlated (not "
            "independent draws) -- this Mann-Whitney p-value is "
            "descriptive, not inferential. The monthly-series contrast in "
            "period_means.csv/kruskal_stats.json is this module's primary "
            "pre/post test; this is a secondary, trial-level view of the "
            "same contrast, reported in the manuscript alongside it."
        ),
    }


def run(response=None, results=None, window=None):
    results = results if results is not None else default_results_dir()
    if window is not None and not split_date_in_window(window):
        return {"skipped": f"split_date {SPLIT_DATE.date()} does not leave both a pre and a post "
                           f"side of real response months inside window {window.id!r}"}
    if response is None:
        response = default_response_spec()
    label = response.label  # display identity for output filenames/keys below

    df, df_real, events, _ = load_data(window=window)
    df = df.dropna(subset=ALL_COLS)

    pre  = df[df["Datetime"] <  SPLIT_DATE]
    post = df[df["Datetime"] >= SPLIT_DATE]

    # Use real response measurements only for response tests; full series for env vars
    pre_ec50  = df_real[df_real["Datetime"] <  SPLIT_DATE][RESPONSE_COL]
    post_ec50 = df_real[df_real["Datetime"] >= SPLIT_DATE][RESPONSE_COL]

    kruskal = {}
    for col in ALL_COLS:
        if col == RESPONSE_COL:
            a, b = pre_ec50.dropna(), post_ec50.dropna()
        else:
            a, b = pre[col].dropna(), post[col].dropna()

        kw  = stats.kruskal(a, b)
        mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
        # Output identity: a per-variable dict keyed by display name, never
        # the internal RESPONSE_COL (see common.py).
        key = label if col == RESPONSE_COL else col
        kruskal[key] = {
            "kruskal_stat": float(kw.statistic),
            "kruskal_p":    float(kw.pvalue),
            "mannwhitney_stat": float(mwu.statistic),
            "mannwhitney_p":    float(mwu.pvalue),
            "n_pre":  int(len(a)),
            "n_post": int(len(b)),
            "mean_pre":  float(a.mean()),
            "mean_post": float(b.mean()),
            "median_pre":  float(a.median()),
            "median_post": float(b.median()),
        }

    (results / "kruskal_stats.json").write_text(json.dumps(kruskal, indent=2))

    # Period means table
    rows = []
    for col in ALL_COLS + MHW_COLS:
        if col not in df.columns:
            continue
        all_val = df[col].mean() if col != RESPONSE_COL else df_real[RESPONSE_COL].mean()
        pre_val = pre[col].mean() if col != RESPONSE_COL else pre_ec50.mean()
        post_val = post[col].mean() if col != RESPONSE_COL else post_ec50.mean()
        var_name = label if col == RESPONSE_COL else col
        rows.append({"variable": var_name, "mean_all": all_val,
                     "mean_pre": pre_val, "mean_post": post_val,
                     "change_pct": 100*(post_val - pre_val)/abs(pre_val) if pre_val else np.nan})
    pd.DataFrame(rows).to_csv(results / "period_means.csv", index=False)

    # Trial-level (295 individual determinations) pre/post contrast --
    # manuscript section 3.1, secondary to the monthly-series test above.
    raw_contrast = _raw_trial_contrast(window)
    (results / "period_contrast_raw.json").write_text(json.dumps(raw_contrast, indent=2))

    # Distribution data for boxplots (Streamlit). Filename AND column header
    # built from the response's display label, never RESPONSE_COL -- for
    # Livorno label == "EC50", so dist_EC50.csv is unchanged.
    # Period labels from the data, never literals, FILE BY FILE: first and
    # last month actually present in this file's rows on each side of the
    # split, year AND month at both ends. A label must be true of every row
    # it is attached to: the response (real months only) and the
    # environmental variables cover different months, so their labels may
    # differ; the window-level response coverage is in window.json.
    def _span(months):
        return f"{months.min():%Y-%m}–{months.max():%Y-%m}"
    for col in ALL_COLS:
        src = df_real if col == RESPONSE_COL else df
        out = src[["Datetime", col]].copy()
        if col == RESPONSE_COL:
            out = out.rename(columns={RESPONSE_COL: label})
        months = out["Datetime"]
        pre_label = _span(months[months < SPLIT_DATE])
        post_label = _span(months[months >= SPLIT_DATE])
        out["period"] = np.where(out["Datetime"] < SPLIT_DATE, pre_label, post_label)
        file_id = label if col == RESPONSE_COL else col
        out.to_csv(results / f"dist_{file_id}.csv", index=False)

    print(f"✓ period_split: stats saved for {len(ALL_COLS)} variables")
    ec50_res = kruskal[label]
    print(f"  {label} KW p={ec50_res['kruskal_p']:.2e}  "
          f"pre_mean={ec50_res['mean_pre']:.2f}  post_mean={ec50_res['mean_post']:.2f}")
    print(f"  {label} raw trials: n={raw_contrast['n_pre']}/{raw_contrast['n_post']}  "
          f"decline={raw_contrast['decline_pct']:.1f}%  MWU p={raw_contrast['mannwhitney_p']:.2e}")

if __name__ == "__main__":
    run()
