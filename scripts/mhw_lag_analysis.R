#!/usr/bin/env Rscript
# =============================================================================
# MHW → EC50 Lag Analysis — DLNM
# =============================================================================
# Scientific question: Do Marine Heatwave events cause delayed reductions
# in sea urchin gamete sensitivity (EC50)?
# Biological basis: Paracentrotus lividus gametogenesis takes 3-6 months
# → MHW thermal stress during gametogenesis → compromised gametes → lower EC50
# Expected lag: 2-6 months
#
# Method: DLNM — Distributed Lag Non-Linear Model (Gasparrini 2011)
#
# Superposed Epoch Analysis and the event-based mixed-effects model, formerly
# also computed here, are now a Python port in mhw_lag_extra.py (part of
# ccsu-run-pipeline) — kept in sync automatically with every data update.
# DLNM stays in R: it relies on the `dlnm` package's cross-basis splines,
# which have no comparably mature Python equivalent. This script is triggered
# automatically by the same GitHub Action that refreshes the rest of the
# data (see .github/workflows/update_ec50.yml) rather than being a manual
# step to remember to run.
#
# Inputs:
#   data/data_extended.csv    — monthly environmental + EC50 data
#   data/mhw_monthly.csv      — monthly MHW metrics
#
# Outputs (CSV, read by the dashboard):
#   results/dlnm_results.csv      — DLNM predicted surface (intensity × lag grid)
#   results/dlnm_lag_profile.csv  — Cumulative lag response at mean MHW intensity
#   results/dlnm_slice_lag.csv    — EC50 response over lags at fixed intensities
#
# Install packages (once):
#   install.packages(c("dlnm", "mgcv", "dplyr", "readr"))
# =============================================================================

suppressPackageStartupMessages({
  library(dlnm)
  library(mgcv)
  library(dplyr)
  library(readr)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly=FALSE)
script_flag <- grep("^--file=", args, value=TRUE)
if (length(script_flag) > 0) {
  script_path <- sub("^--file=", "", script_flag[1])
  ROOT <- normalizePath(file.path(dirname(script_path), ".."))
} else {
  ROOT <- normalizePath(".")
}
cat("ROOT:", ROOT, "\n")

DATA     <- read_csv(file.path(ROOT, "data", "data_extended.csv"),    show_col_types=FALSE)
MONTHLY  <- read_csv(file.path(ROOT, "data", "mhw_monthly.csv"),      show_col_types=FALSE)

cat("Loaded:", nrow(DATA), "monthly rows\n")

# ── Prepare monthly dataframe ─────────────────────────────────────────────────
DATA$Datetime <- as.Date(DATA$Datetime)
DATA$year     <- as.integer(format(DATA$Datetime, "%Y"))
DATA$month    <- as.integer(format(DATA$Datetime, "%m"))

MONTHLY$Datetime <- as.Date(MONTHLY$Datetime)
df <- DATA %>%
  left_join(MONTHLY %>% select(Datetime, mhw_days, mhw_peak_intensity, mhw_cum_intensity),
            by="Datetime") %>%
  mutate(
    mhw_days          = ifelse(is.na(mhw_days), 0, mhw_days),
    mhw_peak_intensity = ifelse(is.na(mhw_peak_intensity), 0, mhw_peak_intensity)
  ) %>%
  filter(!is.na(EC50))

cat("Analysis rows (non-NA EC50):", nrow(df), "\n")

# =============================================================================
# DLNM — Distributed Lag Non-Linear Model
# =============================================================================
cat("\n── DLNM ─────────────────────────────────────────────────────────────\n")

MAX_LAG <- 12

# Cross-basis: natural cubic spline on intensity (df=4), BS on lag 0-12 (df=4)
# mhw_peak_intensity is the exposure variable
cb <- crossbasis(
  df$mhw_peak_intensity,
  lag    = MAX_LAG,
  argvar = list(fun="ns", df=4),        # non-linear dose-response
  arglag = list(fun="bs", df=4, degree=3)  # flexible lag shape
)

# GAM: EC50 ~ cross-basis + cyclic spline on month + linear year trend
fit <- gam(
  EC50 ~ cb + s(month, bs="cc", k=6) + year,
  data   = df,
  family = gaussian(),
  method = "REML"
)

cat("DLNM GAM summary:\n")
print(summary(fit))

# Prediction grid: intensity × lag
intensity_vals <- seq(0, max(df$mhw_peak_intensity, na.rm=TRUE), length.out=30)
pred <- crosspred(cb, fit,
                  at      = intensity_vals,
                  lag     = c(0, MAX_LAG),   # range [min, max]
                  cumul   = TRUE,
                  cen     = 0)          # centre at zero intensity (no MHW)

# Save full surface (intensity × lag grid)
surface_df <- expand.grid(intensity=intensity_vals, lag=0:MAX_LAG)
surface_df$fit  <- as.vector(pred$matfit)
surface_df$low  <- as.vector(pred$matlow)
surface_df$high <- as.vector(pred$mathigh)
write_csv(surface_df, file.path(ROOT, "results", "dlnm_results.csv"))
cat("Saved dlnm_results.csv\n")

# Cumulative lag response at mean non-zero MHW intensity
mean_int <- mean(df$mhw_peak_intensity[df$mhw_peak_intensity > 0], na.rm=TRUE)
# Find nearest row in prediction grid to mean_int
nearest_idx <- which.min(abs(intensity_vals - mean_int))
lag_profile <- data.frame(
  lag     = 0:MAX_LAG,
  cumulative_rr = as.numeric(pred$cumfit[nearest_idx, ]),
  ci_lower      = as.numeric(pred$cumlow[nearest_idx, ]),
  ci_upper      = as.numeric(pred$cumhigh[nearest_idx, ]),
  intensity_ref = mean_int
)
write_csv(lag_profile, file.path(ROOT, "results", "dlnm_lag_profile.csv"))
cat("Saved dlnm_lag_profile.csv — mean MHW intensity:", round(mean_int, 3), "°C\n")

# Slice at fixed lags: dose-response at lag 0, 3, 6, 9, 12
slice_lags <- c(0, 3, 6, 9, 12)
all_lag_names <- colnames(pred$matfit)
slice_df <- do.call(rbind, lapply(slice_lags, function(l) {
  # Find nearest available lag column
  lag_idx <- which.min(abs(as.numeric(all_lag_names) - l))
  data.frame(
    lag       = l,
    intensity = intensity_vals,
    fit       = pred$matfit[, lag_idx],
    low       = pred$matlow[, lag_idx],
    high      = pred$mathigh[, lag_idx]
  )
}))
write_csv(slice_df, file.path(ROOT, "results", "dlnm_slice_lag.csv"))
cat("Saved dlnm_slice_lag.csv\n")

cat("\n✅ DLNM analysis complete.\n")
cat("Outputs in results/: dlnm_results.csv, dlnm_lag_profile.csv, dlnm_slice_lag.csv\n")
