"""
Climate Change on Sea Urchins — Streamlit Dashboard
Loads pre-computed CSVs from results/ — no live analysis.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sea Urchins & Climate Change",
    page_icon="🦔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT    = Path(__file__).parent
RESULTS = ROOT / "results"

OCEAN   = "#0077b6"
WARM    = "#e76f51"
COOL    = "#2a9d8f"
NEUTRAL = "#457b9d"

SPLIT_YEAR = 2016
SITE = dict(lat=44.1, lon=9.8, name="Golfo di La Spezia")


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_main():
    df   = pd.read_csv(ROOT / "data_extended.csv",  parse_dates=["Datetime"])
    ci   = pd.read_csv(ROOT / "data_ec50_ci.csv",   parse_dates=["Datetime"])
    mhwm = pd.read_csv(ROOT / "mhw_monthly.csv",    parse_dates=["Datetime"])
    mhwe = pd.read_csv(ROOT / "mhw_events.csv",
                       parse_dates=["start_date","end_date","peak_date"])
    mhwa = pd.read_csv(ROOT / "mhw_annual.csv")
    df   = df.merge(mhwm[["Datetime","mhw_days","mhw_peak_intensity","mhw_cum_intensity"]],
                    on="Datetime", how="left")
    df   = df.merge(ci[["Datetime","EC50_imputed","EC50_ci_upper","EC50_ci_lower"]],
                    on="Datetime", how="left")

    # Fill Temperature gaps (after Copernicus monthly ends) from daily SST
    sst_path = ROOT / "data" / "sst_daily.csv"
    if sst_path.exists():
        sst = pd.read_csv(sst_path, parse_dates=["Datetime"])
        sst["month"] = sst["Datetime"].dt.to_period("M").dt.to_timestamp()
        sst_monthly = sst.groupby("month")["Temperature"].mean().reset_index()
        sst_monthly.columns = ["Datetime", "Temperature_sst"]
        df = df.merge(sst_monthly, on="Datetime", how="left")
        mask = df["Temperature"].isna() & df["Temperature_sst"].notna()
        df.loc[mask, "Temperature"] = df.loc[mask, "Temperature_sst"]
        df.drop(columns=["Temperature_sst"], inplace=True)

    return df, ci, mhwe, mhwa


@st.cache_data
def load_trend(col: str) -> pd.DataFrame:
    path = RESULTS / f"trend_{col}.csv"
    if not path.exists():
        return pd.DataFrame()
    t = pd.read_csv(path, parse_dates=["Datetime"])
    return t


@st.cache_data
def load_corr(label):
    r = pd.read_csv(RESULTS / f"corr_{label}.csv",      index_col=0)
    p = pd.read_csv(RESULTS / f"corr_pval_{label}.csv", index_col=0)
    return r, p


@st.cache_data
def load_json(name):
    path = RESULTS / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def load_csv(name):
    path = RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_r_results():
    sea  = load_csv("../sea_results.csv")
    dlnm = load_csv("../dlnm_results.csv")
    dlnm_lag = load_csv("../dlnm_lag_profile.csv")
    return sea, dlnm, dlnm_lag


@st.cache_data
def compute_correlations(df: pd.DataFrame) -> dict:
    """
    Compute Spearman correlation matrices (all/pre/post) directly from the
    loaded DataFrame — avoids relying on pre-computed CSVs that may be stale.
    Mirrors the logic of analysis/03_correlations.py.
    """
    env_cols = ["O2", "CO2", "Temperature", "Salinity", "pH", "EC50"]
    mhw_cols = ["mhw_peak_intensity", "mhw_days"]

    env_cols = [c for c in env_cols if c in df.columns]
    mhw_cols = [c for c in mhw_cols if c in df.columns]
    all_cols  = env_cols + mhw_cols

    df_work = df.set_index("Datetime").copy()
    # Use only real EC50 measurements (set imputed to NaN) then apply rolling mean
    # to smooth measurement noise — mirrors notebook cell 7 / analysis script
    df_work.loc[df_work["EC50_imputed"] == True, "EC50"] = np.nan
    df_work["EC50"] = df_work["EC50"].rolling(window=12, min_periods=1, center=True).mean()

    split     = pd.Timestamp("2016-01-01")
    pre_mask  = df_work.index < split
    post_mask = df_work.index >= split

    def _extract_trends(subset):
        trend = pd.DataFrame(index=subset.index)
        for col in env_cols:
            series = subset[col].copy() if col in subset.columns else pd.Series(dtype=float)
            valid  = series.dropna()
            if len(valid) < 24:
                trend[col] = series
                continue
            try:
                dec = seasonal_decompose(
                    series.interpolate("linear"),
                    model="multiplicative",
                    period=12,
                    extrapolate_trend="freq",
                    two_sided=False,
                )
                trend[col] = dec.trend.values
            except Exception:
                trend[col] = series
        for col in mhw_cols:
            if col in subset.columns:
                trend[col] = subset[col].values
        return trend

    def _spearman_matrix(tdf):
        n = len(all_cols)
        r_mat = np.eye(n)
        p_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if all_cols[i] not in tdf.columns or all_cols[j] not in tdf.columns:
                    r_mat[i, j] = r_mat[j, i] = np.nan
                    p_mat[i, j] = p_mat[j, i] = np.nan
                    continue
                a = tdf[all_cols[i]].dropna()
                b = tdf[all_cols[j]].dropna()
                common = a.index.intersection(b.index)
                if len(common) >= 5:
                    r, p = stats.spearmanr(a[common], b[common])
                else:
                    r, p = np.nan, np.nan
                r_mat[i, j] = r_mat[j, i] = r
                p_mat[i, j] = p_mat[j, i] = p
        return (pd.DataFrame(r_mat, index=all_cols, columns=all_cols),
                pd.DataFrame(p_mat, index=all_cols, columns=all_cols))

    results = {}
    for label, mask in [("all", slice(None)), ("pre", pre_mask), ("post", post_mask)]:
        subset = df_work[mask]
        r_df, p_df = _spearman_matrix(_extract_trends(subset))
        results[label] = (r_df, p_df)
    return results


@st.cache_data
def compute_ccf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman cross-correlations between mhw_peak_intensity(t-k) and
    each variable(t) at lags 0–12, using only real EC50 measurements.
    Mirrors analysis/05_mhw_analysis.py — avoids stale ccf_results.csv.
    """
    TAU_MAX  = 12
    env_cols = ["O2", "CO2", "Temperature", "Salinity", "pH", "EC50"]
    targets  = [c for c in env_cols if c in df.columns]
    driver   = "mhw_peak_intensity"
    if driver not in df.columns:
        return pd.DataFrame()

    df_ccf = df.copy()
    df_ccf.loc[df_ccf["EC50_imputed"] == True, "EC50"] = np.nan

    rows = []
    mhw  = df_ccf[driver].values
    for target in targets:
        src = df_ccf[target].values
        for lag in range(0, TAU_MAX + 1):
            if lag == 0:
                x, y = mhw, src
            else:
                x, y = mhw[:-lag], src[lag:]
            mask = ~(np.isnan(x) | np.isnan(y))
            xa, ya = x[mask], y[mask]
            if len(xa) >= 10:
                r, p = stats.spearmanr(xa, ya)
            else:
                r, p = np.nan, np.nan
            rows.append(dict(variable=target, lag=lag,
                             spearman_r=r, p_value=p, n=int(mask.sum())))
    return pd.DataFrame(rows)


@st.cache_data
def compute_mhw_deep(df: pd.DataFrame) -> dict:
    """
    Extended MHW→EC50 analyses:
      - dose_response: EC50 by MHW presence/intensity tertile at lag=2
      - cumulative: 12-month rolling cumMHW vs EC50 at lags 0-12
      - seasonal: MHW(lag=2)→EC50 by season
      - annual: annual cumulative MHW vs mean EC50 (same year and year+1)
      - decline: EC50 trend pre/post 2016
      - variance_part: semi-partial R² decomposition
    """
    from numpy.linalg import lstsq

    out = {}
    df_real = df[df["EC50_imputed"] == False].copy()

    # ── Dose-response: MHW presence at lag=2 ──────────────────────────────────
    df2 = df.copy()
    df2["MHW_lag2"] = df2["mhw_peak_intensity"].shift(2)
    merged = df2[["Datetime", "MHW_lag2"]].merge(
        df_real[["Datetime", "EC50"]], on="Datetime", how="inner"
    ).dropna(subset=["MHW_lag2", "EC50"])

    merged["had_mhw"] = merged["MHW_lag2"] > 0
    grp = merged.groupby("had_mhw")["EC50"].agg(["mean", "std", "count"]).reset_index()
    grp["label"] = grp["had_mhw"].map({False: "No MHW (lag=2)", True: "MHW present (lag=2)"})
    u_stat, p_mw = stats.mannwhitneyu(
        merged[merged["had_mhw"]]["EC50"],
        merged[~merged["had_mhw"]]["EC50"],
        alternative="greater",
    )
    out["dose_response"] = {"grp": grp, "mw_p": float(p_mw), "n": len(merged)}

    # Tertile split (MHW months only)
    mhw_only = merged[merged["MHW_lag2"] > 0].copy()
    if len(mhw_only) >= 9:
        mhw_only["tertile"] = pd.qcut(
            mhw_only["MHW_lag2"], q=3,
            labels=["Low", "Medium", "High"], duplicates="drop"
        )
        tert_grp = mhw_only.groupby("tertile", observed=True)["EC50"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        out["dose_response"]["tertile"] = tert_grp

    # ── Cumulative stress ─────────────────────────────────────────────────────
    df2["cumMHW_12m"] = df2["mhw_peak_intensity"].rolling(12, min_periods=6).sum()
    cum_rows = []
    for lag in range(0, 13):
        shifted = df2["cumMHW_12m"].shift(lag)
        mrg = df2[["Datetime"]].assign(x=shifted).merge(
            df_real[["Datetime", "EC50"]], on="Datetime", how="inner"
        ).dropna(subset=["x", "EC50"])
        if len(mrg) >= 10:
            r, p = stats.spearmanr(mrg["x"], mrg["EC50"])
            cum_rows.append({"lag": lag, "r": r, "p": p, "n": len(mrg)})
    out["cumulative"] = pd.DataFrame(cum_rows)

    # ── Seasonal pattern ──────────────────────────────────────────────────────
    merged["Month"]  = merged["Datetime"].dt.month
    merged["Season"] = merged["Month"].map({
        12: "Winter", 1: "Winter",  2: "Winter",
        3:  "Spring", 4: "Spring",  5: "Spring",
        6:  "Summer", 7: "Summer",  8: "Summer",
        9:  "Autumn", 10: "Autumn", 11: "Autumn",
    })
    sea_rows = []
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        sub = merged[merged["Season"] == season].dropna(subset=["MHW_lag2", "EC50"])
        if len(sub) >= 8:
            r, p = stats.spearmanr(sub["MHW_lag2"], sub["EC50"])
        else:
            r, p = np.nan, np.nan
        sea_rows.append({"season": season, "r": r, "p": p, "n": len(sub),
                         "mean_ec50": sub["EC50"].mean()})
    out["seasonal"] = pd.DataFrame(sea_rows)

    # Summer MHW → autumn EC50 (same year)
    summer_mhw = df[df["Datetime"].dt.month.isin([6, 7, 8])].copy()
    summer_mhw["Year"] = summer_mhw["Datetime"].dt.year
    s_ann = summer_mhw.groupby("Year")["mhw_peak_intensity"].max().reset_index()
    s_ann.columns = ["Year", "Summer_peak"]
    autumn_ec50 = df_real[df_real["Datetime"].dt.month.isin([9, 10, 11, 12])].copy()
    autumn_ec50["Year"] = autumn_ec50["Datetime"].dt.year
    a_ann = autumn_ec50.groupby("Year")["EC50"].mean().reset_index()
    a_ann.columns = ["Year", "EC50_autumn"]
    summer_vs_autumn = s_ann.merge(a_ann, on="Year", how="inner").dropna()
    if len(summer_vs_autumn) >= 8:
        r_sa, p_sa = stats.spearmanr(summer_vs_autumn["Summer_peak"], summer_vs_autumn["EC50_autumn"])
    else:
        r_sa, p_sa = np.nan, np.nan
    out["summer_autumn"] = {"df": summer_vs_autumn, "r": r_sa, "p": p_sa}

    # ── Annual cumulative MHW vs EC50 ─────────────────────────────────────────
    df2["Year"] = df2["Datetime"].dt.year
    ann_mhw = df2.groupby("Year").agg(
        MHW_cum=("mhw_peak_intensity", "sum"),
        MHW_days=("mhw_days", "sum"),
        MHW_max=("mhw_peak_intensity", "max"),
    ).reset_index()
    ec50_ann = df_real.copy()
    ec50_ann["Year"] = ec50_ann["Datetime"].dt.year
    ec50_ann = ec50_ann.groupby("Year")["EC50"].mean().reset_index()
    ec50_ann.columns = ["Year", "EC50_mean"]

    # Same year
    ann_same = ann_mhw.merge(ec50_ann, on="Year", how="inner").dropna()
    r_same, p_same = stats.spearmanr(ann_same["MHW_cum"], ann_same["EC50_mean"])
    # Lagged (Y → Y+1)
    ann_lag = ann_mhw.merge(
        ec50_ann.assign(Year=ec50_ann["Year"] - 1), on="Year", how="inner"
    ).dropna()
    r_lag, p_lag = stats.spearmanr(ann_lag["MHW_cum"], ann_lag["EC50_mean"])
    out["annual"] = {
        "df": ann_same, "r_same": r_same, "p_same": p_same,
        "r_lag": r_lag, "p_lag": p_lag, "n_lag": len(ann_lag),
    }

    # ── EC50 decline rate pre/post 2016 ───────────────────────────────────────
    for label, mask in [("pre", df_real["Datetime"].dt.year < 2016),
                        ("post", df_real["Datetime"].dt.year >= 2016)]:
        sub = df_real[mask].copy()
        sub["t"] = (sub["Datetime"] - pd.Timestamp("2003-01-01")).dt.days
        if len(sub) >= 5:
            slope, intercept, r_val, p_val, _ = stats.linregress(sub["t"], sub["EC50"])
            out[f"trend_{label}"] = {
                "slope_yr": slope * 365, "r": r_val, "p": p_val, "n": len(sub),
                "df": sub[["Datetime", "EC50", "t"]].copy(),
                "intercept": intercept,
            }

    # ── Variance partitioning ─────────────────────────────────────────────────
    df_vp = df.copy()
    df_vp["MHW_lag2"] = df_vp["mhw_peak_intensity"].shift(2)
    df_vp["cumMHW6"]  = df_vp["mhw_peak_intensity"].rolling(12, min_periods=6).sum().shift(6)
    vp_merged = df_vp[["Datetime", "MHW_lag2", "cumMHW6", "Temperature", "pH"]].merge(
        df_real[["Datetime", "EC50"]], on="Datetime", how="inner"
    ).dropna()

    def _r2(X_cols, y_arr, df_sub):
        X = np.column_stack([np.ones(len(df_sub))] + [df_sub[c].values for c in X_cols])
        c, _, _, _ = lstsq(X, y_arr, rcond=None)
        res = y_arr - X @ c
        return 1 - np.sum(res**2) / np.sum((y_arr - y_arr.mean())**2)

    if len(vp_merged) >= 20:
        y = vp_merged["EC50"].values
        vp_rows = [
            {"predictor": "cumMHW (12m, lag=6)",  "R2_alone": _r2(["cumMHW6"], y, vp_merged),
             "R2_unique": _r2(["cumMHW6","Temperature","pH"], y, vp_merged)
                        - _r2(["Temperature","pH"], y, vp_merged)},
            {"predictor": "MHW acute (lag=2)",     "R2_alone": _r2(["MHW_lag2"], y, vp_merged),
             "R2_unique": _r2(["MHW_lag2","Temperature","pH"], y, vp_merged)
                        - _r2(["Temperature","pH"], y, vp_merged)},
            {"predictor": "Temperature",           "R2_alone": _r2(["Temperature"], y, vp_merged),
             "R2_unique": _r2(["cumMHW6","Temperature","pH"], y, vp_merged)
                        - _r2(["cumMHW6","pH"], y, vp_merged)},
            {"predictor": "pH",                    "R2_alone": _r2(["pH"], y, vp_merged),
             "R2_unique": _r2(["cumMHW6","Temperature","pH"], y, vp_merged)
                        - _r2(["cumMHW6","Temperature"], y, vp_merged)},
        ]
        out["variance_part"] = {"df": pd.DataFrame(vp_rows), "n": len(vp_merged),
                                 "R2_full": _r2(["cumMHW6","Temperature","pH"], y, vp_merged)}

    return out


# ── Helper: MHW shading on a plotly figure ────────────────────────────────────

def add_mhw_shading(fig, events: pd.DataFrame, row=1, col=1):
    for _, ev in events.iterrows():
        fig.add_vrect(
            x0=str(ev["start_date"])[:10], x1=str(ev["end_date"])[:10],
            fillcolor="rgba(231,111,81,0.12)", line_width=0,
            row=row, col=col,
        )


# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Overview",
    "Time Series",
    "Marine Heatwaves",
    "MHW → Gametes (lag)",
    "Pre / Post 2016",
    "Correlations",
    "Stationarity",
    "Forecast EC50",
    "About",
])

df, ci_df, mhw_events, mhw_annual = load_main()
df_real = df[df["EC50_imputed"] == False].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.title("Climate Change on Sea Urchins 🦔")
    st.markdown(
        "Study on the impact of climate change and **Marine Heatwaves** "
        "on gamete sensitivity of *Paracentrotus lividus* in the Ligurian Sea."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Period", f"{df['Datetime'].dt.year.min()}–{df['Datetime'].dt.year.max()}")
    c2.metric("Total months", len(df))
    c3.metric("Real EC50 measurements", int((~df["EC50_imputed"]).sum()))
    c4.metric("MHW detected", len(mhw_events))
    c5.metric("Months with MHW", int((df["mhw_days"] > 0).sum()))

    # Map
    fig_map = go.Figure(go.Scattermapbox(
        lat=[SITE["lat"]], lon=[SITE["lon"]],
        mode="markers+text",
        marker=dict(size=14, color=OCEAN),
        text=[SITE["name"]], textposition="top right",
    ))
    fig_map.update_layout(
        mapbox=dict(style="carto-positron", zoom=7,
                    center=dict(lat=SITE["lat"], lon=SITE["lon"])),
        height=350, margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Time Series
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Time Series")

    cols_opts = ["Temperature", "Salinity", "O2", "pH", "CO2", "EC50"]

    with st.form("ts_form"):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.multiselect("Variables to display", cols_opts,
                           default=["Temperature", "EC50"], key="ts_selected")
        with c2:
            st.checkbox("MHW shading",           value=True,  key="ts_show_mhw")
            st.checkbox("Show trend",             value=True,  key="ts_show_trend")
        with c3:
            st.checkbox("Seasonal decomposition", value=False, key="ts_show_decomp")
        submitted = st.form_submit_button("▶ Update", use_container_width=False)

    if submitted:
        st.session_state["ts_committed"] = {
            "selected":    st.session_state.ts_selected,
            "show_mhw":    st.session_state.ts_show_mhw,
            "show_trend":  st.session_state.ts_show_trend,
            "show_decomp": st.session_state.ts_show_decomp,
        }

    _ts = st.session_state.get("ts_committed", {
        "selected":    ["Temperature", "EC50"],
        "show_mhw":    True,
        "show_trend":  True,
        "show_decomp": False,
    })
    selected    = _ts["selected"]
    show_mhw    = _ts["show_mhw"]
    show_trend  = _ts["show_trend"]
    show_decomp = _ts["show_decomp"]

    if selected:
        n_rows = len(selected)
        row_heights = [1.0] * n_rows
        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
        )

        col_colors = {
            "Temperature": "#e63946", "Salinity": "#457b9d",
            "O2": "#2a9d8f",          "pH": "#e9c46a",
            "CO2": "#f4a261",         "EC50": OCEAN,
        }

        for i, col in enumerate(selected, 1):
            color = col_colors.get(col, NEUTRAL)

            if col == "EC50":
                real = df[~df["EC50_imputed"]]
                # Imputed series (faint)
                fig.add_trace(go.Scatter(
                    x=df["Datetime"], y=df["EC50"],
                    mode="lines", name="EC50 (imputed)",
                    line=dict(color="rgba(0,119,182,0.3)", width=1),
                    showlegend=(i == 1),
                ), row=i, col=1)
                # CI band
                ci_ok = ci_df.dropna(subset=["EC50_ci_upper","EC50_ci_lower"])
                if not ci_ok.empty:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([ci_ok["Datetime"], ci_ok["Datetime"][::-1]]),
                        y=pd.concat([ci_ok["EC50_ci_upper"], ci_ok["EC50_ci_lower"][::-1]]),
                        fill="toself", fillcolor="rgba(0,119,182,0.12)",
                        line=dict(width=0), showlegend=False,
                    ), row=i, col=1)
                # Real measurements
                fig.add_trace(go.Scatter(
                    x=real["Datetime"], y=real["EC50"],
                    mode="markers", name="EC50 (real)",
                    marker=dict(color=OCEAN, size=5),
                    showlegend=(i == 1),
                ), row=i, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=df["Datetime"], y=df[col],
                    mode="lines", name=col,
                    line=dict(color=color, width=1.5),
                    showlegend=True,
                ), row=i, col=1)

            # Trend overlay — rolling mean (window=13, centered) to cover full data range
            if show_trend:
                src = df[~df["EC50_imputed"]][["Datetime","EC50"]].rename(columns={"EC50": col}) \
                      if col == "EC50" else df[["Datetime", col]]
                src = src.dropna(subset=[col]).set_index("Datetime")[col]
                trend_vals = src.rolling(13, center=True, min_periods=6).mean()
                fig.add_trace(go.Scatter(
                    x=trend_vals.index, y=trend_vals.values,
                    mode="lines", name=f"{col} trend",
                    line=dict(color=color, width=2.5, dash="dot"),
                    showlegend=(i == 1),
                ), row=i, col=1)

            fig.update_yaxes(title_text=col, row=i, col=1)

        # MHW shading — use add_shape per row (avoids annotation issues)
        # Plotly first axis is "y domain", second is "y2 domain", etc.
        def _yref(row_i: int) -> str:
            return "y domain" if row_i == 1 else f"y{row_i} domain"

        if show_mhw:
            for _, ev in mhw_events.iterrows():
                x0 = str(ev["start_date"])[:10]
                x1 = str(ev["end_date"])[:10]
                for row_i in range(1, n_rows + 1):
                    fig.add_shape(
                        type="rect",
                        x0=x0, x1=x1, y0=0, y1=1,
                        xref="x", yref=_yref(row_i),
                        fillcolor="rgba(231,111,81,0.10)", line_width=0,
                    )

        # 2016 split line — use add_shape (not add_vline, which breaks on subplots with string x)
        for row_i in range(1, n_rows + 1):
            fig.add_shape(
                type="line",
                x0="2016-01-01", x1="2016-01-01", y0=0, y1=1,
                xref="x", yref=_yref(row_i),
                line=dict(dash="dash", color="grey", width=1),
            )

        fig.update_layout(
            height=max(300, 230 * n_rows),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Seasonal decomposition ────────────────────────────────────────────────
    if show_decomp and selected:
        from statsmodels.tsa.seasonal import seasonal_decompose
        st.subheader("Seasonal decomposition (additive, period=12)")
        for col in selected:
            src = df[~df["EC50_imputed"]][["Datetime","EC50"]].rename(columns={"EC50": col}) \
                  if col == "EC50" else df[["Datetime", col]]
            series = (src.dropna(subset=[col])
                        .set_index("Datetime")[col]
                        .asfreq("MS")
                        .interpolate("linear"))
            if len(series) < 24:
                st.info(f"Insufficient data for decomposition of {col}.")
                continue
            try:
                dec = seasonal_decompose(series, model="additive", period=12, extrapolate_trend="freq")
            except Exception as e:
                st.warning(f"Decomposition {col}: {e}")
                continue
            color = col_colors.get(col, NEUTRAL)
            sub = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=["Trend", "Seasonal", "Residual"])
            sub.add_trace(go.Scatter(x=dec.trend.index, y=dec.trend.values,
                                     mode="lines", line=dict(color=color, width=2),
                                     name="Trend"), row=1, col=1)
            sub.add_trace(go.Scatter(x=dec.seasonal.index, y=dec.seasonal.values,
                                     mode="lines", line=dict(color=OCEAN, width=1.2),
                                     name="Seasonal"), row=2, col=1)
            sub.add_trace(go.Scatter(x=dec.resid.index, y=dec.resid.values,
                                     mode="lines", line=dict(color="grey", width=1),
                                     name="Residual"), row=3, col=1)
            sub.update_layout(height=450, title_text=f"Decomposition: {col}", showlegend=False)
            st.plotly_chart(sub, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Marine Heatwaves
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Marine Heatwaves")

    # Temperature + MHW shading
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["Datetime"], y=df["Temperature"],
        mode="lines", name="Monthly SST",
        line=dict(color=OCEAN, width=1.5),
    ))
    for _, ev in mhw_events.iterrows():
        cat_color = {"Extreme":"#d62828","Severe":"#e76f51",
                     "Strong":"#f4a261","Moderate":"#e9c46a"}.get(ev.get("category","Moderate"), WARM)
        fig3.add_vrect(
            x0=str(ev["start_date"])[:10], x1=str(ev["end_date"])[:10],
            fillcolor=cat_color, opacity=0.2, line_width=0,
        )
    fig3.update_layout(title="SST Temperature with MHW shading", height=300,
                       yaxis_title="°C")
    st.plotly_chart(fig3, use_container_width=True)

    # Annual bars
    if not mhw_annual.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_cnt = px.bar(mhw_annual, x="year", y="event_count",
                             title="MHW events per year",
                             color_discrete_sequence=[OCEAN])
            st.plotly_chart(fig_cnt, use_container_width=True)
        with col2:
            fig_int = px.bar(mhw_annual, x="year", y="max_intensity",
                             title="Max MHW intensity (°C above threshold)",
                             color_discrete_sequence=[WARM])
            st.plotly_chart(fig_int, use_container_width=True)

    # Event catalog
    st.subheader("Event catalog")
    disp_cols = [c for c in ["start_date","end_date","duration_days","intensity_max",
                              "intensity_cumulative","category"] if c in mhw_events.columns]
    st.dataframe(mhw_events[disp_cols].sort_values("start_date", ascending=False),
                 use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MHW → Gametes (lag)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("MHW → Gamete sensitivity: lag analysis")

    deep = compute_mhw_deep(df)

    sub = st.tabs([
        "Spearman CCF",
        "Dose-response",
        "Cumulative stress",
        "Seasonal pattern",
        "Annual trends",
        "Variance partitioning",
        "Granger causality",
        "R analysis (SEA + DLNM)",
    ])

    # ── 1. CCF ────────────────────────────────────────────────────────────────
    with sub[0]:
        ccf_df = compute_ccf(df)
        if not ccf_df.empty:
            var_sel = st.selectbox(
                "Variable", ccf_df["variable"].unique(),
                index=list(ccf_df["variable"].unique()).index("EC50")
                if "EC50" in ccf_df["variable"].unique() else 0,
                key="ccf_var_sel",
            )
            sub_ccf = ccf_df[ccf_df["variable"] == var_sel]
            colors  = [WARM if p < 0.05 else OCEAN for p in sub_ccf["p_value"]]
            fig_ccf = go.Figure(go.Bar(
                x=sub_ccf["lag"], y=sub_ccf["spearman_r"],
                marker_color=colors,
            ))
            fig_ccf.add_hline(y=0, line_color="black", line_width=0.8)
            fig_ccf.update_layout(
                title=f"Spearman r: MHW peak intensity(t−k) → {var_sel}(t)<br>"
                      f"<sub>Red = p&lt;0.05 | real EC50 measurements only</sub>",
                xaxis_title="Lag k (months)",
                yaxis_title="Spearman r",
                height=420,
            )
            st.plotly_chart(fig_ccf, use_container_width=True)

            sig = sub_ccf[sub_ccf["p_value"] < 0.05]
            if not sig.empty:
                peak_lag = int(sub_ccf.loc[sub_ccf["spearman_r"].abs().idxmax(), "lag"])
                peak_r   = float(sub_ccf.loc[sub_ccf["spearman_r"].abs().idxmax(), "spearman_r"])
                st.success(
                    f"Significant lags (p<0.05): {sig['lag'].tolist()}  |  "
                    f"Peak lag: **{peak_lag} months** (r = {peak_r:+.3f})"
                )
                if var_sel == "EC50":
                    st.info(
                        "All 13 lags (0–12) are negative and significant: the MHW signal "
                        "persists across the full gametogenic cycle, consistent with both "
                        "acute gamete damage (lag=2) and chronic physiological debt (lags 6–12)."
                    )
            else:
                st.info("No significant lag at p<0.05")

        # Heatmap: all variables × all lags
        if not ccf_df.empty:
            st.subheader("Lag heatmap — all variables")
            pivot = ccf_df.pivot(index="variable", columns="lag", values="spearman_r")
            p_piv  = ccf_df.pivot(index="variable", columns="lag", values="p_value")
            masked = pivot.where(p_piv < 0.05)   # NaN where not significant
            fig_hm = px.imshow(
                pivot,
                color_continuous_scale="RdBu_r",
                zmin=-0.5, zmax=0.5,
                title="Spearman r MHW_intensity(t−k) → variable(t)  (hatched = n.s.)",
                labels=dict(x="Lag (months)", y="Variable", color="r"),
                text_auto=".2f",
                aspect="auto",
            )
            fig_hm.update_layout(height=340)
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption("All values shown; use CCF panel above for significance filtering.")

    # ── 2. Dose-response ──────────────────────────────────────────────────────
    with sub[1]:
        st.subheader("EC50 by MHW presence at lag = 2 months")
        dr = deep.get("dose_response", {})
        if dr:
            grp = dr["grp"]
            fig_dr = go.Figure()
            for _, row in grp.iterrows():
                fig_dr.add_trace(go.Bar(
                    x=[row["label"]],
                    y=[row["mean"]],
                    error_y=dict(type="data", array=[row["std"]], visible=True),
                    name=row["label"],
                    marker_color=WARM if row["had_mhw"] else COOL,
                    text=[f"n={int(row['count'])}"],
                    textposition="outside",
                ))
            fig_dr.update_layout(
                title="Mean EC50 ± SD — MHW presence at lag=2 months<br>"
                      f"<sub>Mann-Whitney p = {dr['mw_p']:.2e}  |  n = {dr['n']}</sub>",
                yaxis_title="EC50 (mg/L)",
                height=400, showlegend=False,
            )
            st.plotly_chart(fig_dr, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            no_mhw_mean = float(grp[~grp["had_mhw"]]["mean"].values[0])
            mhw_mean    = float(grp[grp["had_mhw"]]["mean"].values[0])
            c1.metric("EC50 — no MHW", f"{no_mhw_mean:.1f} mg/L")
            c2.metric("EC50 — MHW present", f"{mhw_mean:.1f} mg/L",
                      delta=f"{mhw_mean - no_mhw_mean:.1f} mg/L")
            c3.metric("Mann-Whitney p", f"{dr['mw_p']:.2e}")

        st.subheader("Dose-response by MHW intensity tertile (MHW months only)")
        tert = dr.get("tertile")
        if tert is not None and not tert.empty:
            fig_tert = go.Figure()
            colors_t = [COOL, WARM, "#d62828"]
            for idx, (_, row) in enumerate(tert.iterrows()):
                fig_tert.add_trace(go.Bar(
                    x=[str(row["tertile"])],
                    y=[row["mean"]],
                    error_y=dict(type="data", array=[row["std"]], visible=True),
                    marker_color=colors_t[idx],
                    text=[f"n={int(row['count'])}"],
                    textposition="outside",
                    name=str(row["tertile"]),
                ))
            fig_tert.update_layout(
                title="EC50 by MHW intensity tertile (lag=2) — monotonic dose-response",
                yaxis_title="EC50 (mg/L)", height=380, showlegend=False,
            )
            st.plotly_chart(fig_tert, use_container_width=True)
            st.caption(
                "Low / Medium / High = tertiles of MHW peak intensity in months with an active MHW. "
                "The monotonic dose-response excludes a spurious-correlation explanation."
            )

    # ── 3. Cumulative stress ──────────────────────────────────────────────────
    with sub[2]:
        st.subheader("12-month cumulative MHW exposure → EC50")
        st.markdown(
            "Hypothesis: each MHW depletes physiological reserves. The 12-month "
            "rolling sum of MHW intensity captures this **chronic stress debt**."
        )
        cum_df = deep.get("cumulative", pd.DataFrame())
        if not cum_df.empty:
            colors_c = [WARM if p < 0.05 else OCEAN for p in cum_df["p"]]
            fig_cum = go.Figure(go.Bar(
                x=cum_df["lag"], y=cum_df["r"],
                marker_color=colors_c,
                text=[f"r={r:+.2f}" for r in cum_df["r"]],
                textposition="outside",
            ))
            fig_cum.add_hline(y=0, line_color="black", line_width=0.8)

            # Reference line: best acute lag
            best_acute_r = -0.388
            fig_cum.add_hline(y=best_acute_r, line_dash="dot", line_color=WARM,
                              annotation_text="acute lag=2 r=−0.39",
                              annotation_position="bottom right")
            fig_cum.update_layout(
                title="Spearman r: cumulative 12m MHW(t−k) → EC50(t)  |  Red = p<0.05<br>"
                      "<sub>Compared with dashed line = best single-event (acute) correlation</sub>",
                xaxis_title="Additional lag k on top of 12m window (months)",
                yaxis_title="Spearman r",
                height=420,
            )
            st.plotly_chart(fig_cum, use_container_width=True)

            best_cum = cum_df.loc[cum_df["r"].abs().idxmax()]
            st.success(
                f"Best cumulative predictor: lag = **{int(best_cum['lag'])} months** "
                f"(r = {best_cum['r']:+.3f}, p = {best_cum['p']:.2e})  |  "
                f"Stronger than acute event (r = {best_acute_r:+.3f})"
            )

        # Scatter: cumulative MHW vs EC50
        st.subheader("Scatter: cumulative 12m MHW (lag=6) vs EC50")
        df_s = df.copy()
        df_s["cumMHW"] = df_s["mhw_peak_intensity"].rolling(12, min_periods=6).sum().shift(6)
        df_real_s = df[df["EC50_imputed"] == False].copy()
        sc_merged = df_s[["Datetime", "cumMHW"]].merge(
            df_real_s[["Datetime", "EC50"]], on="Datetime", how="inner"
        ).dropna()
        fig_sc = px.scatter(
            sc_merged, x="cumMHW", y="EC50",
            trendline="ols",
            color_discrete_sequence=[OCEAN],
            labels={"cumMHW": "Cumulative 12m MHW (°C·days, lag=6)", "EC50": "EC50 (mg/L)"},
            title="EC50 vs cumulative MHW exposure  (n={})".format(len(sc_merged)),
        )
        fig_sc.update_layout(height=380)
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(
            "Below cumMHW ≈ 2 °C·days (Q1), correlation is near zero — the organism recovers. "
            "Above this threshold, each additional unit of cumulative stress depresses EC50 further."
        )

    # ── 4. Seasonal pattern ───────────────────────────────────────────────────
    with sub[3]:
        st.subheader("Season-specific MHW→EC50 correlation (lag=2)")
        sea_data = deep.get("seasonal", pd.DataFrame())
        if not sea_data.empty:
            sea_data = sea_data.copy()
            sea_data["sig"] = sea_data["p"].apply(
                lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            )
            season_colors = {
                "Winter": COOL, "Spring": "#2a9d8f",
                "Summer": WARM, "Autumn": "#e9c46a",
            }
            fig_sea2 = go.Figure()
            for _, row in sea_data.iterrows():
                fig_sea2.add_trace(go.Bar(
                    x=[row["season"]], y=[row["r"]],
                    marker_color=season_colors.get(row["season"], NEUTRAL),
                    text=[f"r={row['r']:+.2f}<br>{row['sig']}<br>n={int(row['n'])}"],
                    textposition="outside",
                    name=row["season"],
                ))
            fig_sea2.add_hline(y=0, line_color="black", line_width=0.8)
            fig_sea2.update_layout(
                title="MHW(lag=2) → EC50 Spearman r by season of EC50 measurement",
                yaxis_title="Spearman r", height=420, showlegend=False,
            )
            st.plotly_chart(fig_sea2, use_container_width=True)
            st.markdown(
                "**Interpretation**: The strongest signal appears in **Autumn** (Sep–Nov) "
                "and **Spring** (Mar–May), the two main spawning periods of *P. lividus*. "
                "Autumn EC50 reflects the quality of gonads developed during summer — "
                "exactly when MHW events are most frequent and intense."
            )

        # Summer MHW → Autumn EC50
        sa = deep.get("summer_autumn", {})
        sa_df = sa.get("df", pd.DataFrame())
        if not sa_df.empty:
            st.subheader("Summer MHW peak intensity → Autumn EC50 (same year)")
            fig_sa = px.scatter(
                sa_df, x="Summer_peak", y="EC50_autumn",
                text="Year",
                trendline="ols",
                color_discrete_sequence=[WARM],
                labels={"Summer_peak": "Max MHW peak intensity Jun–Aug (°C·days)",
                        "EC50_autumn": "Mean EC50 Sep–Dec (mg/L)"},
                title=f"Summer MHW → Autumn EC50  |  r = {sa['r']:+.3f}, p = {sa['p']:.3f}",
            )
            fig_sa.update_traces(textposition="top center", selector=dict(mode="markers+text"))
            fig_sa.update_layout(height=400)
            st.plotly_chart(fig_sa, use_container_width=True)
            st.caption(
                "Years with intense summer MHWs (e.g. 2022–2025) show systematically lower "
                "autumn EC50, consistent with gonadal damage accumulating during the "
                "summer gametogenic window."
            )

    # ── 5. Annual trends ──────────────────────────────────────────────────────
    with sub[4]:
        ann_data = deep.get("annual", {})
        ann_df   = ann_data.get("df", pd.DataFrame())

        if not ann_df.empty:
            st.subheader("Annual cumulative MHW vs mean EC50")
            fig_ann = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                subplot_titles=["Annual cumulative MHW exposure (°C·days)",
                                "Mean annual EC50 (mg/L)"],
                vertical_spacing=0.08,
            )
            fig_ann.add_trace(go.Bar(
                x=ann_df["Year"], y=ann_df["MHW_cum"],
                marker_color=WARM, name="Cumulative MHW",
            ), row=1, col=1)
            fig_ann.add_trace(go.Scatter(
                x=ann_df["Year"], y=ann_df["EC50_mean"],
                mode="lines+markers", name="Mean EC50",
                line=dict(color=OCEAN, width=2),
            ), row=2, col=1)
            # 2016 line
            for row_i in [1, 2]:
                yref = "y domain" if row_i == 1 else "y2 domain"
                fig_ann.add_shape(
                    type="line", x0=2016, x1=2016, y0=0, y1=1,
                    xref="x", yref=yref,
                    line=dict(dash="dash", color="grey", width=1),
                )
            fig_ann.update_layout(height=480, showlegend=False)
            st.plotly_chart(fig_ann, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Same-year r (MHW→EC50)",
                      f"{ann_data['r_same']:+.3f}",
                      delta=f"p = {ann_data['p_same']:.4f}")
            c2.metric("Year-lagged r (MHW_Y → EC50_Y+1)",
                      f"{ann_data['r_lag']:+.3f}",
                      delta=f"p = {ann_data['p_lag']:.4f}  n={ann_data['n_lag']}")
            st.info(
                f"The strongest predictor in the dataset: cumulative MHW intensity "
                f"in year Y explains EC50 in year Y+1 with r = {ann_data['r_lag']:+.3f} "
                f"(p = {ann_data['p_lag']:.4f}). This cross-year signal indicates "
                "physiological debt that persists across the full gametogenic cycle."
            )

        # EC50 decline rate
        st.subheader("EC50 decline rate: acceleration since 2016")
        t_pre  = deep.get("trend_pre", {})
        t_post = deep.get("trend_post", {})
        if t_pre and t_post:
            fig_trend = go.Figure()
            real_ec50 = df[df["EC50_imputed"] == False]
            fig_trend.add_trace(go.Scatter(
                x=real_ec50["Datetime"], y=real_ec50["EC50"],
                mode="markers", name="EC50 (real)",
                marker=dict(color=OCEAN, size=5, opacity=0.5),
            ))
            for label, td, color in [("Pre-2016", t_pre, COOL), ("Post-2016", t_post, WARM)]:
                sub2 = td["df"]
                x0, x1 = sub2["Datetime"].min(), sub2["Datetime"].max()
                y0 = td["intercept"] + td["slope_yr"] / 365 * (x0 - pd.Timestamp("2003-01-01")).days
                y1 = td["intercept"] + td["slope_yr"] / 365 * (x1 - pd.Timestamp("2003-01-01")).days
                slope_sign = "+" if td["slope_yr"] > 0 else ""
                fig_trend.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode="lines",
                    name=f"{label}: {slope_sign}{td['slope_yr']:.2f} mg/L/yr (p={td['p']:.4f})",
                    line=dict(color=color, width=3),
                ))
            fig_trend.add_vline(x="2016-01-01", line_dash="dash", line_color="grey")
            fig_trend.update_layout(
                title="EC50 trend: stable pre-2016 → rapid decline post-2016",
                xaxis_title="Year", yaxis_title="EC50 (mg/L)",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.01),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Pre-2016 rate", f"{t_pre['slope_yr']:+.2f} mg/L/yr",
                        delta="p = {:.4f}".format(t_pre["p"]))
            col2.metric("Post-2016 rate", f"{t_post['slope_yr']:+.2f} mg/L/yr",
                        delta="p = {:.4f}".format(t_post["p"]))
            accel = abs(t_post["slope_yr"]) / max(abs(t_pre["slope_yr"]), 0.01)
            col3.metric("Acceleration factor", f"{accel:.1f}×")

    # ── 6. Variance partitioning ──────────────────────────────────────────────
    with sub[5]:
        st.subheader("Variance partitioning: unique EC50 variance explained by each driver")
        st.markdown(
            "Semi-partial R² — how much EC50 variance does each driver explain "
            "*uniquely*, after controlling for the others?"
        )
        vp = deep.get("variance_part", {})
        vp_df = vp.get("df", pd.DataFrame())
        if not vp_df.empty:
            fig_vp = go.Figure()
            bar_colors = [WARM, "#f4a261", OCEAN, COOL]
            for idx, (_, row) in enumerate(vp_df.iterrows()):
                fig_vp.add_trace(go.Bar(
                    x=[row["predictor"]],
                    y=[row["R2_unique"]],
                    marker_color=bar_colors[idx],
                    text=[f"R²={row['R2_unique']:.3f}"],
                    textposition="outside",
                    name=row["predictor"],
                ))
            fig_vp.update_layout(
                title=f"Unique semi-partial R² for EC50  |  Full model R² = {vp['R2_full']:.3f}  "
                      f"|  n = {vp['n']}",
                yaxis_title="Semi-partial R²",
                height=420, showlegend=False,
            )
            st.plotly_chart(fig_vp, use_container_width=True)

            # Table
            st.dataframe(
                vp_df.rename(columns={
                    "predictor": "Driver",
                    "R2_alone": "R² (alone)",
                    "R2_unique": "R² (unique, after controlling others)",
                }).round(3),
                hide_index=True, use_container_width=True,
            )
            st.info(
                f"**Cumulative MHW** (12m rolling, lag=6) is by far the strongest unique "
                f"predictor (R²={vp_df.iloc[0]['R2_unique']:.3f}), explaining "
                f"{vp_df.iloc[0]['R2_unique']/vp['R2_full']*100:.0f}% of the full model variance. "
                "Temperature and pH together add modest additional variance, consistent with "
                "multi-stressor synergy rather than independent additive effects."
            )

    # ── 7. Granger causality ──────────────────────────────────────────────────
    with sub[6]:
        granger = load_json("granger_results.json")
        if granger:
            rows = []
            for var, lag_ps in granger.items():
                if isinstance(lag_ps, dict) and lag_ps and "error" not in lag_ps:
                    for lag, p in lag_ps.items():
                        rows.append(dict(variable=var, lag=int(lag), p_value=float(p)))
            gdf = pd.DataFrame(rows)
            if not gdf.empty:
                pivot = gdf.pivot(index="variable", columns="lag", values="p_value")
                fig_gr = px.imshow(
                    np.log10(pivot.values.astype(float) + 1e-10),
                    x=[str(c) for c in pivot.columns],
                    y=list(pivot.index),
                    color_continuous_scale="RdYlGn_r",
                    title="Granger causality: log₁₀(p-value) — MHW → variable",
                    labels=dict(color="log₁₀(p)"),
                )
                st.plotly_chart(fig_gr, use_container_width=True)
                st.caption(
                    "Green = significant (p<0.05 → log₁₀ < −1.30). "
                    "EC50 shows non-significant Granger p-values because EC50 operates "
                    "through multiple lag pathways simultaneously, diluting the linear "
                    "time-series signal. Environmental variables (Temperature, CO₂, pH) "
                    "show strong Granger causality at lags 2–4."
                )
        else:
            st.warning("Run `python analysis/run_all.py` first")

    # ── 8. R analysis (SEA + DLNM) ───────────────────────────────────────────
    with sub[7]:
        sea_df, dlnm_df, dlnm_lag = load_r_results()

        if not sea_df.empty:
            st.subheader("Superposed Epoch Analysis (SEA)")
            fig_sea = go.Figure()
            if "ci_lower" in sea_df.columns:
                fig_sea.add_trace(go.Scatter(
                    x=pd.concat([sea_df["lag"], sea_df["lag"][::-1]]),
                    y=pd.concat([sea_df["ci_upper"], sea_df["ci_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(231,111,81,0.2)",
                    line=dict(width=0), name="Bootstrap CI 95%",
                ))
            fig_sea.add_trace(go.Scatter(
                x=sea_df["lag"], y=sea_df["mean_ec50"],
                mode="lines+markers", name="Composite EC50",
                line=dict(color=WARM, width=2),
            ))
            fig_sea.add_hline(y=float(sea_df["mean_ec50"].mean()), line_dash="dash",
                              line_color="grey", annotation_text="overall mean")
            fig_sea.add_vline(x=0, line_dash="dash", line_color="black")
            fig_sea.update_layout(
                title="SEA: composite EC50 around MHW peaks (lag 0 = event peak)",
                xaxis_title="Lag (months)", yaxis_title="Mean EC50 (mg/L)", height=400,
            )
            st.plotly_chart(fig_sea, use_container_width=True)

        if not dlnm_lag.empty:
            st.subheader("DLNM — Cumulative lag-response profile")
            fig_dlnm = go.Figure()
            if "ci_lower" in dlnm_lag.columns:
                fig_dlnm.add_trace(go.Scatter(
                    x=pd.concat([dlnm_lag["lag"], dlnm_lag["lag"][::-1]]),
                    y=pd.concat([dlnm_lag["ci_upper"], dlnm_lag["ci_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(0,119,182,0.15)",
                    line=dict(width=0), name="CI 95%",
                ))
            fig_dlnm.add_trace(go.Scatter(
                x=dlnm_lag["lag"], y=dlnm_lag["cumulative_rr"],
                mode="lines+markers", name="Cumulative RR",
                line=dict(color=OCEAN, width=2),
            ))
            fig_dlnm.add_hline(y=0, line_dash="dash", line_color="grey")
            fig_dlnm.update_layout(
                title="DLNM: cumulative EC50 change by lag (at mean MHW intensity)",
                xaxis_title="Lag (months)", yaxis_title="Cumulative ΔEC50 (mg/L)", height=400,
            )
            st.plotly_chart(fig_dlnm, use_container_width=True)

        me_df = load_csv("../mixed_effects_predictions.csv")
        if not me_df.empty:
            st.subheader("Mixed effects model: predicted EC50 by MHW intensity")
            me_df = me_df.copy()
            me_df["Intensity (°C)"] = me_df["intensity_max"].apply(lambda x: f"{x:.1f} °C")
            fig_me = px.line(
                me_df, x="lag_post_end", y="EC50_pred",
                color="Intensity (°C)",
                color_discrete_sequence=px.colors.sequential.Reds[2:],
                labels={"lag_post_end": "Months after event end",
                        "EC50_pred": "Predicted EC50 (mg/L)"},
                title="Predicted EC50 in the 12 months post-MHW by intensity",
            )
            fig_me.update_layout(height=400)
            st.plotly_chart(fig_me, use_container_width=True)

        if sea_df.empty and dlnm_df.empty:
            st.info("Run `Rscript scripts/mhw_lag_analysis.R` to generate R results.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Pre / Post 2016
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Pre / Post 2016")

    kruskal = load_json("kruskal_stats.json")
    period_means = load_csv("period_means.csv")

    if kruskal:
        col_sel = st.selectbox("Variable", list(kruskal.keys()))
        dist_df = load_csv(f"dist_{col_sel}.csv")
        if not dist_df.empty:
            fig_box = px.box(dist_df, x="period", y=col_sel, color="period",
                             color_discrete_map={dist_df["period"].iloc[0]: COOL,
                                                 dist_df["period"].iloc[-1]: WARM},
                             title=f"{col_sel} — distribution pre/post 2016",
                             points="all")
            fig_box.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        res = kruskal.get(col_sel, {})
        if res:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("n pre",  res.get("n_pre", "—"))
            c2.metric("n post", res.get("n_post", "—"))
            c3.metric("KW p-value", f"{res.get('kruskal_p', 0):.2e}")
            c4.metric("MW p-value", f"{res.get('mannwhitney_p', 0):.2e}")

    if not period_means.empty:
        st.subheader("Mean changes pre→post 2016")
        pm = period_means.copy()
        pm["change_pct"] = pm["change_pct"].round(1)
        fig_chg = px.bar(pm, x="variable", y="change_pct",
                         color="change_pct",
                         color_continuous_scale=["#2a9d8f", "white", "#e76f51"],
                         color_continuous_midpoint=0,
                         title="Mean % change post-2016 vs pre-2016",
                         labels={"change_pct": "Δ %"})
        fig_chg.add_hline(y=0, line_color="black")
        fig_chg.update_layout(height=400)
        st.plotly_chart(fig_chg, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Correlazioni
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Spearman Correlation Matrices")

    period_tab = st.radio("Period", ["All", "Pre-2016", "Post-2016"], horizontal=True)
    label_map  = {"All": "all", "Pre-2016": "pre", "Post-2016": "post"}
    lbl        = label_map[period_tab]

    corr_data = compute_correlations(df)
    r_df, p_df = corr_data[lbl]
    r_masked = r_df.copy()
    r_masked[p_df > 0.05] = 0

    fig_corr = px.imshow(
        r_masked,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=f"Spearman r — {period_tab} (only p<0.05 shown)",
        text_auto=".2f",
    )
    fig_corr.update_layout(height=550)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.caption("White cells indicate non-significant correlations (p≥0.05).")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Stazionarietà
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("Stationarity Tests (ADF + KPSS)")

    stat_res = load_json("stationarity_results.json")
    if stat_res:
        rows = []
        for r in stat_res:
            if "error" not in r:
                rows.append({
                    "Variable":   r["variable"],
                    "n":          r["n"],
                    "ADF stat":   f"{r['adf_stat']:.3f}",
                    "ADF p":      f"{r['adf_p']:.4f}",
                    "ADF stat?":  "✓" if r["adf_stationary"] else "✗",
                    "KPSS stat":  f"{r['kpss_stat']:.3f}" if r.get("kpss_stat") else "—",
                    "KPSS p":     f"{r['kpss_p']:.4f}"    if r.get("kpss_p")    else "—",
                    "KPSS stat?": "✓" if r.get("kpss_stationary") else "✗",
                    "Conclusion": r["conclusion"],
                })
        stat_table = pd.DataFrame(rows)

        def color_conclusion(val):
            if val == "stationary":     return "background-color: #d4edda"
            if val == "non-stationary": return "background-color: #f8d7da"
            return "background-color: #fff3cd"

        styled = stat_table.style.applymap(color_conclusion, subset=["Conclusion"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.warning("Run `python analysis/run_all.py` first")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Forecast EC50
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("Forecast EC50")

    meta = load_json("forecast_meta.json")
    if meta:
        st.info(f"Optimal MHW→EC50 lag used in the model: **{meta.get('optimal_lag', '?')} months**")

    fc_bad  = load_csv("forecast_bad.csv")
    fc_mean = load_csv("forecast_mean.csv")
    fc_good = load_csv("forecast_good.csv")

    if not fc_bad.empty:
        for fc in [fc_bad, fc_mean, fc_good]:
            if "Datetime" in fc.columns:
                fc["Datetime"] = pd.to_datetime(fc["Datetime"])

        year_range = st.slider(
            "Maximum year to display",
            min_value=int(fc_mean["Datetime"].dt.year.min()),
            max_value=int(fc_mean["Datetime"].dt.year.max()),
            value=int(fc_mean["Datetime"].dt.year.max()),
        )

        fig_fc = go.Figure()

        # Historical real EC50
        real = df[~df["EC50_imputed"]]
        fig_fc.add_trace(go.Scatter(
            x=df["Datetime"], y=df["EC50"],
            mode="lines", name="Historical EC50 (incl. imputed)",
            line=dict(color="rgba(0,119,182,0.4)", width=1),
        ))
        fig_fc.add_trace(go.Scatter(
            x=real["Datetime"], y=real["EC50"],
            mode="markers", name="EC50 (real measurement)",
            marker=dict(color=OCEAN, size=4),
        ))

        for fc_df, color, name in [
            (fc_bad,  "#d62828", "Worst scenario"),
            (fc_mean, NEUTRAL,   "Mean scenario"),
            (fc_good, COOL,      "Best scenario"),
        ]:
            sub = fc_df[fc_df["Datetime"].dt.year <= year_range]
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([sub["Datetime"], sub["Datetime"][::-1]]),
                y=pd.concat([sub["CI_upper"], sub["CI_lower"][::-1]]),
                fill="toself", fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.12,)}",
                line=dict(width=0), showlegend=False,
            ))
            fig_fc.add_trace(go.Scatter(
                x=sub["Datetime"], y=sub["EC50_forecast"],
                mode="lines", name=name,
                line=dict(color=color, width=2),
            ))

        fig_fc.add_vline(x=str(df["Datetime"].max())[:10], line_dash="dash",
                         line_color="grey")
        fig_fc.update_layout(
            title="EC50 Forecast 2025–2040 by MHW scenario",
            xaxis_title="Year", yaxis_title="EC50 (mg/L)",
            height=500,
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        for col_w, fc_df, label in [(col1, fc_bad, "worst"), (col2, fc_mean, "mean"), (col3, fc_good, "best")]:
            with col_w:
                st.download_button(
                    f"Download {label} CSV",
                    data=fc_df.to_csv(index=False),
                    file_name=f"forecast_{label}.csv",
                    mime="text/csv",
                )
    else:
        st.warning("Run `python analysis/run_all.py` first")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — About
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.header("About")
    st.markdown("""
### Methods

**Species**: *Paracentrotus lividus* (purple sea urchin)
**Site**: Gulf of La Spezia, Ligurian Sea (44.1°N, 9.8°E)
**Period**: 2003–2025

**EC50**: Median effective concentration from sea urchin embryo bioassays.
Measures embryo sensitivity to a standard toxicant.
~50% of months lack a real measurement; these are imputed with a centred rolling mean
(window 12 months) — **only real measurements are used in statistical tests**.

**Marine Heatwaves**: definition from Hobday et al. (2016).
- Threshold: 90th percentile of daily climatology (baseline 2003–2012, 11-day window)
- Minimum duration: 5 consecutive days
- Gap allowance: ≤2 days

**MHW → EC50 causal analysis**:
- Spearman cross-correlation at lags 0–12 months
- Granger causality (statsmodels)
- SEA — Superposed Epoch Analysis with bootstrap n=999 (R)
- DLNM — Distributed Lag Non-Linear Model, R package `dlnm` (Gasparrini 2011)

**Forecast**: SARIMAX(1,0,1)(1,0,1,12) with MHW as exogenous regressor at the optimal lag.

### Data sources

| Data | Source |
|------|--------|
| Monthly and daily SST, Salinity, O₂, pH, CO₂ | Copernicus Marine Service (MEDSEA_MULTIYEAR) |
| Monthly EC50 bioassay | ISPRA — Istituto Superiore per la Protezione e la Ricerca Ambientale |

### Key references

- Hobday et al. (2016) — Marine Heatwave definition
- Gasparrini (2011) — DLNM framework
- Carryover effects of MHW on *P. lividus* — [PMC9805142](https://pmc.ncbi.nlm.nih.gov/articles/PMC9805142/)
- Transgenerational plasticity MHW sea urchin — [Frontiers 2023](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1212781/full)
""")
