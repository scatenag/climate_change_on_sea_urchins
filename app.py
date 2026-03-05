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
    "MHW → Gameti (lag)",
    "Pre / Post 2016",
    "Correlazioni",
    "Stazionarietà",
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
        "Studio sull'impatto dei cambiamenti climatici e delle **Marine Heatwaves** "
        "sulla sensibilità dei gameti di *Paracentrotus lividus* nel Mar Ligure."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Periodo", f"{df['Datetime'].dt.year.min()}–{df['Datetime'].dt.year.max()}")
    c2.metric("Mesi totali", len(df))
    c3.metric("Misure EC50 reali", int((~df["EC50_imputed"]).sum()))
    c4.metric("MHW rilevate", len(mhw_events))
    c5.metric("Mesi con MHW", int((df["mhw_days"] > 0).sum()))

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
    st.header("Serie temporali")

    cols_opts = ["Temperature", "Salinity", "O2", "pH", "CO2", "EC50"]
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        selected = st.multiselect("Variabili da visualizzare", cols_opts,
                                  default=["Temperature", "EC50"])
    with c2:
        show_mhw   = st.checkbox("Shading MHW", value=True)
        show_trend = st.checkbox("Mostra trend", value=True)
    with c3:
        show_decomp = st.checkbox("Decomposizione stagionale", value=False)

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
                    mode="lines", name="EC50 (imputato)",
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
                    mode="markers", name="EC50 (reale)",
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

            # Trend overlay
            if show_trend:
                tr = load_trend(col)
                if not tr.empty:
                    fig.add_trace(go.Scatter(
                        x=tr["Datetime"], y=tr["trend"],
                        mode="lines", name=f"{col} trend",
                        line=dict(color=color, width=2.5, dash="dot"),
                        showlegend=(i == 1),
                    ), row=i, col=1)

            fig.update_yaxes(title_text=col, row=i, col=1)

        # MHW shading — use add_shape per row (avoids annotation issues)
        if show_mhw:
            for _, ev in mhw_events.iterrows():
                x0 = str(ev["start_date"])[:10]
                x1 = str(ev["end_date"])[:10]
                for row_i in range(1, n_rows + 1):
                    fig.add_shape(
                        type="rect",
                        x0=x0, x1=x1, y0=0, y1=1,
                        xref="x", yref=f"y{row_i} domain",
                        fillcolor="rgba(231,111,81,0.10)", line_width=0,
                    )

        # 2016 split line — use add_shape (not add_vline, which breaks on subplots with string x)
        for row_i in range(1, n_rows + 1):
            fig.add_shape(
                type="line",
                x0="2016-01-01", x1="2016-01-01", y0=0, y1=1,
                xref="x", yref=f"y{row_i} domain",
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
        st.subheader("Decomposizione stagionale (moltiplicativa, periodo=12)")
        for col in selected:
            tr = load_trend(col)
            if tr.empty:
                st.info(f"Nessun dato di decomposizione per {col}. Rieseguire `python analysis/run_all.py`.")
                continue
            sub = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=["Trend", "Stagionalità", "Residuo"])
            color = col_colors.get(col, NEUTRAL)
            sub.add_trace(go.Scatter(x=tr["Datetime"], y=tr["trend"],
                                     mode="lines", line=dict(color=color, width=2),
                                     name="Trend"), row=1, col=1)
            sub.add_trace(go.Scatter(x=tr["Datetime"], y=tr["seasonal"],
                                     mode="lines", line=dict(color=OCEAN, width=1.2),
                                     name="Stagionale"), row=2, col=1)
            sub.add_trace(go.Scatter(x=tr["Datetime"], y=tr["residual"],
                                     mode="lines", line=dict(color="grey", width=1),
                                     name="Residuo"), row=3, col=1)
            sub.update_layout(height=450, title_text=f"Decomposizione: {col}",
                              showlegend=False)
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
        mode="lines", name="SST mensile",
        line=dict(color=OCEAN, width=1.5),
    ))
    for _, ev in mhw_events.iterrows():
        cat_color = {"Extreme":"#d62828","Severe":"#e76f51",
                     "Strong":"#f4a261","Moderate":"#e9c46a"}.get(ev.get("category","Moderate"), WARM)
        fig3.add_vrect(
            x0=str(ev["start_date"])[:10], x1=str(ev["end_date"])[:10],
            fillcolor=cat_color, opacity=0.2, line_width=0,
        )
    fig3.update_layout(title="Temperatura SST con shading MHW", height=300,
                       yaxis_title="°C")
    st.plotly_chart(fig3, use_container_width=True)

    # Annual bars
    if not mhw_annual.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_cnt = px.bar(mhw_annual, x="year", y="event_count",
                             title="Numero eventi MHW per anno",
                             color_discrete_sequence=[OCEAN])
            st.plotly_chart(fig_cnt, use_container_width=True)
        with col2:
            fig_int = px.bar(mhw_annual, x="year", y="max_intensity",
                             title="Intensità massima MHW (°C sopra soglia)",
                             color_discrete_sequence=[WARM])
            st.plotly_chart(fig_int, use_container_width=True)

    # Event catalog
    st.subheader("Catalogo eventi")
    disp_cols = [c for c in ["start_date","end_date","duration_days","intensity_max",
                              "intensity_cumulative","category"] if c in mhw_events.columns]
    st.dataframe(mhw_events[disp_cols].sort_values("start_date", ascending=False),
                 use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MHW → Gameti (lag)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("MHW → Sensibilità dei gameti: analisi del ritardo")

    sub = st.tabs(["CCF / Spearman lag", "Granger causality", "Analisi R (SEA + DLNM)"])

    # ── CCF
    with sub[0]:
        ccf_df = load_csv("ccf_results.csv")
        if not ccf_df.empty:
            var_sel = st.selectbox("Variabile", ccf_df["variable"].unique(),
                                   index=list(ccf_df["variable"].unique()).index("EC50")
                                   if "EC50" in ccf_df["variable"].unique() else 0)
            sub_ccf = ccf_df[ccf_df["variable"] == var_sel]
            colors  = [WARM if p < 0.05 else OCEAN for p in sub_ccf["p_value"]]
            fig_ccf = go.Figure(go.Bar(
                x=sub_ccf["lag"], y=sub_ccf["spearman_r"],
                marker_color=colors,
                error_y=None,
            ))
            fig_ccf.add_hline(y=0, line_color="black", line_width=0.8)
            fig_ccf.update_layout(
                title=f"Spearman r: MHW_intensity(t−k) → {var_sel}(t)<br>"
                      f"<sub>Rosso = p&lt;0.05; solo misure reali EC50</sub>",
                xaxis_title="Lag k (mesi)",
                yaxis_title="Spearman r",
                height=400,
            )
            st.plotly_chart(fig_ccf, use_container_width=True)

            sig = sub_ccf[sub_ccf["p_value"] < 0.05]
            if not sig.empty:
                st.success(f"Lag significativi (p<0.05): {sig['lag'].tolist()} — "
                           f"peak lag: {sub_ccf.loc[sub_ccf['spearman_r'].abs().idxmax(), 'lag']}")
            else:
                st.info("Nessun lag significativo a p<0.05")
        else:
            st.warning("Eseguire prima `python analysis/run_all.py`")

    # ── Granger
    with sub[1]:
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
                    title="Granger causality: log10(p-value) — MHW → variabile",
                    labels=dict(color="log10(p)"),
                )
                fig_gr.add_shape(type="line", x0=-0.5, x1=len(pivot.columns)-0.5,
                                 y0=-0.5, y1=len(pivot.index)-0.5,
                                 line=dict(color="rgba(0,0,0,0)"))
                st.plotly_chart(fig_gr, use_container_width=True)
                st.caption("Verde = significativo. Soglia p<0.05 → log10 < −1.30")
        else:
            st.warning("Eseguire prima `python analysis/run_all.py`")

    # ── R results
    with sub[2]:
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
                mode="lines+markers", name="EC50 composita",
                line=dict(color=WARM, width=2),
            ))
            fig_sea.add_hline(y=float(sea_df["mean_ec50"].mean()), line_dash="dash", line_color="grey",
                              annotation_text="media complessiva")
            fig_sea.add_vline(x=0, line_dash="dash", line_color="black")
            fig_sea.update_layout(title="SEA: EC50 composita attorno ai picchi MHW (lag 0 = picco evento)",
                                  xaxis_title="Lag (mesi)", yaxis_title="EC50 medio (mg/L)",
                                  height=400)
            st.plotly_chart(fig_sea, use_container_width=True)

        if not dlnm_lag.empty:
            st.subheader("DLNM — Profilo di risposta cumulativa per lag")
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
                mode="lines+markers", name="RR cumulativo",
                line=dict(color=OCEAN, width=2),
            ))
            fig_dlnm.add_hline(y=0, line_dash="dash", line_color="grey",
                              annotation_text="nessun effetto")
            fig_dlnm.update_layout(title="DLNM: variazione cumulativa EC50 per lag (a intensità MHW media)",
                                   xaxis_title="Lag (mesi)", yaxis_title="ΔΔEC50 cumulativo (mg/L)",
                                   height=400)
            st.plotly_chart(fig_dlnm, use_container_width=True)

        # Mixed effects model predictions
        me_df = load_csv("../mixed_effects_predictions.csv")
        if not me_df.empty:
            st.subheader("Modello misto: EC50 post-evento per intensità MHW")
            # Convert intensity to string for discrete line coloring
            me_df = me_df.copy()
            me_df["Intensità (°C)"] = me_df["intensity_max"].apply(lambda x: f"{x:.1f} °C")
            fig_me = px.line(
                me_df, x="lag_post_end", y="EC50_pred",
                color="Intensità (°C)",
                color_discrete_sequence=px.colors.sequential.Reds[2:],
                labels={"lag_post_end": "Mesi dopo fine evento",
                        "EC50_pred": "EC50 predetto (mg/L)"},
                title="EC50 previsto nei 12 mesi post-MHW per diversa intensità",
            )
            fig_me.update_layout(height=400)
            st.plotly_chart(fig_me, use_container_width=True)

        if sea_df.empty and dlnm_df.empty:
            st.info("Eseguire `Rscript scripts/mhw_lag_analysis.R` per generare i risultati R.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Pre / Post 2016
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Pre / Post 2016")

    kruskal = load_json("kruskal_stats.json")
    period_means = load_csv("period_means.csv")

    if kruskal:
        # Boxplots from distribution CSVs
        col_sel = st.selectbox("Variabile", list(kruskal.keys()))
        dist_df = load_csv(f"dist_{col_sel}.csv")
        if not dist_df.empty:
            fig_box = px.box(dist_df, x="period", y=col_sel, color="period",
                             color_discrete_map={dist_df["period"].iloc[0]: COOL,
                                                 dist_df["period"].iloc[-1]: WARM},
                             title=f"{col_sel} — distribuzione pre/post 2016",
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
        st.subheader("Variazioni medie pre→post 2016")
        pm = period_means.copy()
        pm["change_pct"] = pm["change_pct"].round(1)
        fig_chg = px.bar(pm, x="variable", y="change_pct",
                         color="change_pct",
                         color_continuous_scale=["#2a9d8f", "white", "#e76f51"],
                         color_continuous_midpoint=0,
                         title="Variazione % media post-2016 vs pre-2016",
                         labels={"change_pct": "Δ %"})
        fig_chg.add_hline(y=0, line_color="black")
        fig_chg.update_layout(height=400)
        st.plotly_chart(fig_chg, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Correlazioni
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Matrici di correlazione Spearman")

    period_tab = st.radio("Periodo", ["Tutto", "Pre-2016", "Post-2016"], horizontal=True)
    label_map  = {"Tutto": "all", "Pre-2016": "pre", "Post-2016": "post"}
    lbl        = label_map[period_tab]

    try:
        r_df, p_df = load_corr(lbl)
        # Mask non-significant
        r_masked = r_df.copy()
        r_masked[p_df > 0.05] = 0

        fig_corr = px.imshow(
            r_masked,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=f"Spearman r — {period_tab} (solo p<0.05 colorato)",
            text_auto=".2f",
        )
        fig_corr.update_layout(height=550)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.caption("Le celle bianche indicano correlazioni non significative (p≥0.05).")
    except FileNotFoundError:
        st.warning("Eseguire prima `python analysis/run_all.py`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Stazionarietà
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("Test di stazionarietà (ADF + KPSS)")

    stat_res = load_json("stationarity_results.json")
    if stat_res:
        rows = []
        for r in stat_res:
            if "error" not in r:
                rows.append({
                    "Variabile":      r["variable"],
                    "n":              r["n"],
                    "ADF stat":       f"{r['adf_stat']:.3f}",
                    "ADF p":          f"{r['adf_p']:.4f}",
                    "ADF stat?":      "✓" if r["adf_stationary"] else "✗",
                    "KPSS stat":      f"{r['kpss_stat']:.3f}" if r.get("kpss_stat") else "—",
                    "KPSS p":         f"{r['kpss_p']:.4f}"    if r.get("kpss_p")    else "—",
                    "KPSS stat?":     "✓" if r.get("kpss_stationary") else "✗",
                    "Conclusione":    r["conclusion"],
                })
        stat_table = pd.DataFrame(rows)

        def color_conclusion(val):
            if val == "stationary":     return "background-color: #d4edda"
            if val == "non-stationary": return "background-color: #f8d7da"
            return "background-color: #fff3cd"

        styled = stat_table.style.applymap(color_conclusion, subset=["Conclusione"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.warning("Eseguire prima `python analysis/run_all.py`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Forecast EC50
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("Forecast EC50")

    meta = load_json("forecast_meta.json")
    if meta:
        st.info(f"Lag ottimale MHW→EC50 usato nel modello: **{meta.get('optimal_lag', '?')} mesi**")

    fc_bad  = load_csv("forecast_bad.csv")
    fc_mean = load_csv("forecast_mean.csv")
    fc_good = load_csv("forecast_good.csv")

    if not fc_bad.empty:
        for fc in [fc_bad, fc_mean, fc_good]:
            if "Datetime" in fc.columns:
                fc["Datetime"] = pd.to_datetime(fc["Datetime"])

        year_range = st.slider(
            "Anno massimo visualizzato",
            min_value=int(fc_mean["Datetime"].dt.year.min()),
            max_value=int(fc_mean["Datetime"].dt.year.max()),
            value=int(fc_mean["Datetime"].dt.year.max()),
        )

        fig_fc = go.Figure()

        # Historical real EC50
        real = df[~df["EC50_imputed"]]
        fig_fc.add_trace(go.Scatter(
            x=df["Datetime"], y=df["EC50"],
            mode="lines", name="EC50 storico (incl. imputati)",
            line=dict(color="rgba(0,119,182,0.4)", width=1),
        ))
        fig_fc.add_trace(go.Scatter(
            x=real["Datetime"], y=real["EC50"],
            mode="markers", name="EC50 (misura reale)",
            marker=dict(color=OCEAN, size=4),
        ))

        for fc_df, color, name in [
            (fc_bad,  "#d62828", "Scenario peggiore"),
            (fc_mean, NEUTRAL,   "Scenario medio"),
            (fc_good, COOL,      "Scenario migliore"),
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
            title="Previsione EC50 2025–2040 per scenario MHW",
            xaxis_title="Anno", yaxis_title="EC50 (mg/L)",
            height=500,
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        for col_w, fc_df, label in [(col1, fc_bad, "Peggiore"), (col2, fc_mean, "Medio"), (col3, fc_good, "Migliore")]:
            with col_w:
                st.download_button(
                    f"Scarica {label} CSV",
                    data=fc_df.to_csv(index=False),
                    file_name=f"forecast_{label.lower()}.csv",
                    mime="text/csv",
                )
    else:
        st.warning("Eseguire prima `python analysis/run_all.py`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — About
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.header("About")
    st.markdown("""
### Metodi

**Specie**: *Paracentrotus lividus* (riccio di mare viola)
**Sito**: Golfo di La Spezia, Mar Ligure (44.1°N, 9.8°E)
**Periodo**: 2003–2025

**EC50**: Concentrazione efficace mediana in bioassay su embrioni di riccio.
Indica la sensibilità degli embrioni a un tossico standard.
~50% dei mesi manca la misura reale; questi mesi vengono imputati con una media mobile
centrata (finestra 12 mesi) — **solo le misure reali vengono usate nei test statistici**.

**Marine Heatwaves**: definizione Hobday et al. (2016).
- Soglia: 90° percentile della climatologia giornaliera (baseline 2003–2012, finestra 11 giorni)
- Durata minima: 5 giorni consecutivi
- Gap allowance: ≤2 giorni

**Analisi causale MHW → EC50**:
- Cross-correlazione di Spearman a lag 0–12 mesi
- Granger causality (statsmodels)
- SEA — Superposed Epoch Analysis con bootstrap n=999 (R)
- DLNM — Distributed Lag Non-Linear Model, pacchetto `dlnm` R (Gasparrini 2011)

**Forecast**: SARIMAX(1,0,1)(1,0,1,12) con MHW come regressore esogeno al lag ottimale.

### Fonti dati

| Dato | Fonte |
|------|-------|
| SST mensile e giornaliera, Salinità, O₂, pH, CO₂ | Copernicus Marine Service (MEDSEA_MULTIYEAR) |
| EC50 bioassay mensile | Google Sheets — laboratorio DISTAV, Università di Genova |

### Letteratura chiave

- Hobday et al. (2016) — definizione Marine Heatwave
- Gasparrini (2011) — DLNM framework
- Carryover effects MHW su *P. lividus* — [PMC9805142](https://pmc.ncbi.nlm.nih.gov/articles/PMC9805142/)
- Transgenerational plasticity MHW sea urchin — [Frontiers 2023](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1212781/full)
""")
