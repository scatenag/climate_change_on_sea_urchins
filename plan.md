# Piano di lavoro — Climate Change on Sea Urchins

## Obiettivi
1. **Dati aggiornati** — dataset 2003–2025 da sorgenti online (Copernicus Marine + Google Sheets EC50)
2. **Marine Heatwaves** — rilevamento + analisi causale ritardata (effetto su sensibilità gametica)
3. **Dashboard pubblica** — app Streamlit deployata su Streamlit Community Cloud

---

## Stato avanzamento

### ✅ FASE 1 — Pipeline dati (COMPLETATA)

| Script | Output | Risultato |
|--------|--------|-----------|
| `scripts/fetch_ec50.py` | `data/ec50_sheets.csv` | 158 mesi, 2003-02 → 2025-12 |
| `scripts/fetch_copernicus.py` | `data/env_copernicus.csv` | 256 mesi, 2003-01 → 2024-04 |
| `scripts/fetch_copernicus_daily.py` | `data/sst_daily.csv` | 8401 giorni, 2003 → 2025 |
| `scripts/build_dataset.py` | `data_extended.csv`, `data_ec50_ci.csv` | 276 mesi totali |
| `scripts/detect_mhw.py` | `mhw_events.csv`, `mhw_monthly.csv`, `mhw_annual.csv` | **129 eventi MHW** |

**MHW rilevati**: Moderate 90 / Strong 25 / Severe 13 / Extreme 1 — 151/276 mesi con almeno 1 giorno di MHW.
**CO2**: cross-check ratio ≈ 0.98 → unità compatibili (spco2 Copernicus = originale data.csv) ✓

**Dataset Copernicus (ID corretti verificati):**
| Variabile | Dataset ID |
|-----------|-----------|
| Temperature (monthly) | `cmems_mod_med_phy-temp_my_4.2km_P1M-m` |
| Salinity (monthly) | `cmems_mod_med_phy-sal_my_4.2km_P1M-m` |
| O2 (monthly) | `cmems_mod_med_bgc-bio_my_4.2km_P1M-m` |
| pH (monthly) | `cmems_mod_med_bgc-car_my_4.2km_P1M-m` |
| CO2/spco2 (monthly) | `cmems_mod_med_bgc-co2_my_4.2km_P1M-m` |
| Temperature (daily) | `cmems_mod_med_phy-temp_my_4.2km_P1D-m` |

---

### FASE 2 — Analisi (Python + R in parallelo)

#### 2.1 Aggiornamento `analysis.ipynb` (Python)
- [ ] Cambiare path dati: `data.csv` → `data_extended.csv`, caricare anche `data_ec50_ci.csv`
- [ ] Nuova sezione MHW (dopo cell-12): carica `mhw_events.csv`, `mhw_monthly.csv`, merge sul df principale
- [ ] Aggiornare `plotseasonal()`: CI bands EC50 (`fill_between`), shading MHW su Temperature (`axvspan`)
- [ ] Grafici MHW annuali: barre (count / durata media / intensità max)
- [ ] Espandere heatmap Spearman con `mhw_days`, `mhw_peak_intensity` come variabili aggiuntive
- [ ] CCF (Cross-Correlation Function): `mhw_peak_intensity` → ogni variabile, lag 0–12 (`statsmodels.tsa.stattools.ccf`)
- [ ] Granger causality: `grangercausalitytests`, matrice [variabile × lag] → `granger_results.json`
- [ ] ARDL (Autoregressive Distributed Lag, `statsmodels.tsa.ardl.ARDL`): `EC50 ~ EC50_lag + MHW(t-k)` — funzione di risposta cumulativa
- [ ] MHW come predittore AutoARIMA al lag ottimale trovato → aggiornare sezione forecast
- [ ] Export CSV per ogni sezione (vedi elenco sotto)

#### 2.2 Analisi ritardo MHW → EC50 in R (`scripts/mhw_lag_analysis.R`)

**Motivazione scientifica**: il ciclo di gametogenesi in *Paracentrotus lividus* (mediterraneo) dura 3–6 mesi. Un'esposizione a MHW durante la gametogenesi produce gameti di qualità ridotta (EC50 ↓) mesi dopo l'evento. Lag biologicamente atteso: **2–6 mesi**.

##### 2.2a — Superposed Epoch Analysis (SEA)
- Per ciascuno dei 129 eventi MHW: estrai EC50 da t-6 a t+12 mesi rispetto al picco
- Calcola composita media + CI bootstrap (n=999 permutazioni random degli eventi come controllo)
- Plot: linea media ± envelope permutazioni, asterischi nei lag significativi
- Identifica il lag di risposta EC50 e la persistenza dell'effetto

##### 2.2b — DLNM (Distributed Lag Non-Linear Model) — pacchetto R `dlnm`
```r
library(dlnm); library(mgcv)
# Cross-basis: B-spline sull'intensità MHW (df=4) × B-spline sul lag 0–12 (df=4)
cb <- crossbasis(mhw_intensity, lag=12,
                 argvar=list(fun="bs", df=4),
                 arglag=list(fun="bs", df=4))
# GAM con stagionalità
fit <- gam(EC50 ~ cb + s(month, bs="cc") + year, data=df)
pred <- crosspred(cb, fit, at=seq(0,3,0.1), cumul=TRUE)
```
- Output: superficie 3D [intensità × lag → EC50 response]
- Slice al lag ottimale: curva dose-risposta
- Slice ad intensità fissa: profilo temporale del ritardo
- Export: `dlnm_results.csv` (predizioni + CI al 95%)

##### 2.2c — Event-based mixed effects model
```r
library(lme4)
# Per ogni evento MHW, EC50 nei 12 mesi successivi
lmer(EC50_post ~ lag + intensity_max + duration_days + season + (1|event_id), data=event_df)
```
- Controlla per stagionalità e variabilità inter-evento

**Output R → CSV** (letti da notebook e app Streamlit):
- `sea_results.csv` — composita SEA per lag -6:+12
- `dlnm_results.csv` — superficie predetta + CI
- `dlnm_lag_profile.csv` — risposta cumulativa per lag
- `granger_results.json` — già prodotto da Python

---

#### 2.3 Export CSV da `analysis.ipynb`
Per ogni sezione, aggiungere cella di export:
- `pca_anomaly.csv` — reconstruction errors PCA
- `corr_all.csv`, `corr_pre.csv`, `corr_post.csv` — matrici Spearman
- `ccf_results.csv` — CCF peak lag per variabile
- `forecast_bad/mean/good.csv` + `forecast_intervals_*.csv` — ARIMA
- `kruskal_stats.json`, `stationarity_results.json`

---

### FASE 3 — App Streamlit (`app.py`)

Carica solo CSV precomputati (nessuna analisi live).

**9 Tab:**
| Tab | Contenuto | Dati caricati |
|-----|-----------|---------------|
| 1. Overview | Mappa sito, KPI (`st.metric`), copertura dati | `data_extended.csv` |
| 2. Time Series | Plotly multi-panel, CI EC50, toggle variabili, linea 2016 | `data_extended.csv`, `data_ec50_ci.csv`, trend CSVs |
| 3. Marine Heatwaves | Temp + shading eventi, catalogo tabella, barre annuali | `mhw_events.csv`, `mhw_monthly.csv`, `mhw_annual.csv` |
| 4. MHW → Gameti (lag) | SEA composita, superficie DLNM, ARDL response, Granger heatmap | `sea_results.csv`, `dlnm_results.csv`, `granger_results.json` |
| 5. Pre/Post 2016 | Boxplot/violin, Kruskal-Wallis, Mann-Whitney | `data_extended.csv`, `kruskal_stats.json` |
| 6. Correlazioni | 3 heatmap Plotly (All/Pre/Post), hover r+p, MHW incluso | `corr_all/pre/post.csv` |
| 7. Stazionarietà | Tabella ADF/KPSS colorata | `stationarity_results.json` |
| 8. Forecast EC50 | 3 scenari + CI 95%, slider anno, download CSV | `forecast_*.csv` |
| 9. About | Metodi, fonti, citazioni chiave | — |

**Deploy**: Streamlit Community Cloud (gratuito, GitHub integration)

---

### FASE 4 — Deploy
- [ ] Push GitHub (verificare repo pubblico)
- [ ] Deploy Streamlit Community Cloud
- [ ] URL pubblico funzionante

---

## Note tecniche

**Ambiente**:
- Python: `.venv/` (usa `source .venv/bin/activate`)
- R: analisi lag in `scripts/mhw_lag_analysis.R` — richiede `dlnm`, `lme4`, `mgcv`
- Streamlit app legge solo CSV → non richiede né R né copernicusmarine in cloud

**Lag biologicamente atteso**: 2–6 mesi (gametogenesi *P. lividus* Mediterraneo)

**Letteratura chiave**:
- Hobday et al. (2016) — definizione MHW
- Gasparrini (2011) — DLNM framework ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2998707/))
- "Live-fast-die-young" ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9805142/)) — carryover effects MHW su ricci
- Berkeley News (2025) — heat waves deprimono riproduzione ricci ([link](http://news.berkeley.edu/2025/11/14/even-moderate-heat-waves-depress-sea-urchin-reproduction-along-the-pacific-coast/))
- Transgenerational plasticity MHW sea urchin ([Frontiers](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1212781/full))

**Coordinate sito**: 44.1°N, 9.8°E (Golfo di La Spezia), depth 0–10m
