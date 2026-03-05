#!/usr/bin/env Rscript
# =============================================================================
# MHW → EC50 Lag Analysis
# =============================================================================
# Scientific question: Do Marine Heatwave events cause delayed reductions
# in sea urchin gamete sensitivity (EC50)?
# Biological basis: Paracentrotus lividus gametogenesis takes 3-6 months
# → MHW thermal stress during gametogenesis → compromised gametes → lower EC50
# Expected lag: 2-6 months
#
# Methods:
#   A. Superposed Epoch Analysis (SEA) — event-based composites
#   B. DLNM — Distributed Lag Non-Linear Model (Gasparrini 2011)
#   C. Mixed effects model — EC50 post-event ~ lag + intensity + season
#
# Inputs:
#   data_extended.csv     — monthly environmental + EC50 data
#   mhw_events.csv        — MHW event catalog
#   mhw_monthly.csv       — monthly MHW metrics
#
# Outputs (CSV, read by analysis.ipynb and app.py):
#   sea_results.csv       — SEA composite with bootstrap CI
#   dlnm_results.csv      — DLNM predicted surface (intensity × lag grid)
#   dlnm_lag_profile.csv  — Cumulative lag response at mean MHW intensity
#   dlnm_slice_lag.csv    — EC50 response over lags at fixed intensities
#
# Install packages (once):
#   install.packages(c("dlnm", "mgcv", "lme4", "dplyr", "readr", "ggplot2"))
# =============================================================================

suppressPackageStartupMessages({
  library(dlnm)
  library(mgcv)
  library(lme4)
  library(lubridate)
  library(dplyr)
  library(readr)
  library(ggplot2)
})

# Nullish coalescing operator
`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0 && !all(is.na(a))) a else b

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

DATA     <- read_csv(file.path(ROOT, "data_extended.csv"),    show_col_types=FALSE)
EVENTS   <- read_csv(file.path(ROOT, "mhw_events.csv"),       show_col_types=FALSE)
MONTHLY  <- read_csv(file.path(ROOT, "mhw_monthly.csv"),      show_col_types=FALSE)

cat("Loaded:", nrow(DATA), "monthly rows,", nrow(EVENTS), "MHW events\n")

# ── Prepare monthly dataframe ─────────────────────────────────────────────────
DATA$Datetime <- as.Date(DATA$Datetime)
DATA$year     <- as.integer(format(DATA$Datetime, "%Y"))
DATA$month    <- as.integer(format(DATA$Datetime, "%m"))
DATA$t        <- as.integer(factor(DATA$Datetime))   # integer time index

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
# A. SUPERPOSED EPOCH ANALYSIS (SEA)
# =============================================================================
cat("\n── A. Superposed Epoch Analysis ──────────────────────────────────────\n")

LAG_MIN <- -6   # months before peak
LAG_MAX <- 12   # months after peak

# For each event, extract EC50 time series around peak
EVENTS$peak_date <- as.Date(EVENTS$peak_date)

event_windows <- lapply(seq_len(nrow(EVENTS)), function(i) {
  peak <- EVENTS$peak_date[i]
  lags <- LAG_MIN:LAG_MAX
  dates <- peak + months(lags)
  ec50_vals <- sapply(dates, function(d) {
    idx <- which(abs(as.numeric(df$Datetime - d)) < 20)  # nearest month
    if (length(idx) == 0) NA else df$EC50[idx[1]]
  })
  data.frame(event_id=i, lag=lags, EC50=ec50_vals,
             intensity_max=EVENTS$intensity_max[i],
             category=EVENTS$category[i])
})
epoch_df <- do.call(rbind, event_windows)

# Composite: mean ± bootstrap CI
sea_composite <- epoch_df %>%
  group_by(lag) %>%
  summarise(
    n         = sum(!is.na(EC50)),
    mean_ec50 = mean(EC50, na.rm=TRUE),
    sd_EC50   = sd(EC50, na.rm=TRUE),
    se_EC50   = sd(EC50, na.rm=TRUE) / sqrt(sum(!is.na(EC50))),
    .groups="drop"
  ) %>%
  mutate(
    ci_lower = mean_ec50 - 1.96 * se_EC50,
    ci_upper = mean_ec50 + 1.96 * se_EC50
  )

# Bootstrap significance vs null (random event dates)
set.seed(42)
n_boot <- 999
n_events <- nrow(EVENTS)
boot_means <- matrix(NA, nrow=n_boot, ncol=length(LAG_MIN:LAG_MAX))

for (b in seq_len(n_boot)) {
  rand_dates <- sample(df$Datetime, n_events, replace=FALSE)
  rand_windows <- lapply(rand_dates, function(peak) {
    lags <- LAG_MIN:LAG_MAX
    dates <- peak + months(lags)
    sapply(dates, function(d) {
      idx <- which(abs(as.numeric(df$Datetime - d)) < 20)
      if (length(idx) == 0) NA else df$EC50[idx[1]]
    })
  })
  rand_mat <- do.call(rbind, rand_windows)
  boot_means[b, ] <- colMeans(rand_mat, na.rm=TRUE)
}

# p-value: fraction of bootstrap means more extreme than observed
boot_p <- sapply(seq_along(LAG_MIN:LAG_MAX), function(j) {
  obs  <- sea_composite$mean_ec50[j]
  null <- boot_means[, j]
  2 * min(mean(null <= obs, na.rm=TRUE), mean(null >= obs, na.rm=TRUE))
})
sea_composite$boot_p     <- boot_p
sea_composite$boot_p025  <- apply(boot_means, 2, quantile, 0.025, na.rm=TRUE)
sea_composite$boot_p975  <- apply(boot_means, 2, quantile, 0.975, na.rm=TRUE)
sea_composite$significant <- sea_composite$boot_p < 0.05

write_csv(sea_composite, file.path(ROOT, "sea_results.csv"))
cat("Saved sea_results.csv\n")
cat("Significant lags:", paste(sea_composite$lag[sea_composite$significant], collapse=", "), "\n")

# =============================================================================
# B. DLNM — Distributed Lag Non-Linear Model
# =============================================================================
cat("\n── B. DLNM ───────────────────────────────────────────────────────────\n")

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
write_csv(surface_df, file.path(ROOT, "dlnm_results.csv"))
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
write_csv(lag_profile, file.path(ROOT, "dlnm_lag_profile.csv"))
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
write_csv(slice_df, file.path(ROOT, "dlnm_slice_lag.csv"))
cat("Saved dlnm_slice_lag.csv\n")

# =============================================================================
# C. Mixed effects model: EC50 post-event ~ lag + intensity + season
# =============================================================================
cat("\n── C. Event-based mixed effects model ───────────────────────────────\n")

# Build event-level panel: for each event, EC50 in months 1-12 after end
EVENTS$end_date <- as.Date(EVENTS$end_date)

post_event_df <- do.call(rbind, lapply(seq_len(nrow(EVENTS)), function(i) {
  end_date  <- EVENTS$end_date[i]
  end_month <- as.integer(format(end_date, "%m"))
  season_val <- ifelse(end_month %in% c(12,1,2), "Winter",
                ifelse(end_month %in% c(3,4,5),  "Spring",
                ifelse(end_month %in% c(6,7,8),  "Summer", "Autumn")))
  lags <- 1:12
  future_dates <- end_date + months(lags)
  ec50_vals <- sapply(future_dates, function(d) {
    idx <- which(abs(as.numeric(df$Datetime - d)) < 20)
    if (length(idx) == 0) NA else df$EC50[idx[1]]
  })
  data.frame(
    event_id      = i,
    lag_post_end  = lags,
    EC50          = ec50_vals,
    intensity_max = EVENTS$intensity_max[i],
    duration_days = EVENTS$duration_days[i],
    category      = EVENTS$category[i],
    year          = as.integer(format(end_date, "%Y")),
    season        = rep(season_val, length(lags))
  )
})) %>% filter(!is.na(EC50))

cat("Post-event observations:", nrow(post_event_df), "\n")

fit_me <- lmer(
  EC50 ~ lag_post_end * intensity_max + duration_days + season + year + (1|event_id),
  data = post_event_df,
  REML = TRUE
)
cat("Mixed effects model summary:\n")
print(summary(fit_me))

# Export fixed effect predictions
newdata <- expand.grid(
  lag_post_end  = 1:12,
  intensity_max = c(0.5, 1.0, 1.5, 2.0),  # typical intensities
  duration_days = median(EVENTS$duration_days),
  season        = "Summer",
  year          = 2015
)
newdata$EC50_pred <- predict(fit_me, newdata=newdata, re.form=NA)
write_csv(newdata, file.path(ROOT, "mixed_effects_predictions.csv"))
cat("Saved mixed_effects_predictions.csv\n")

cat("\n✅ All lag analyses complete.\n")
cat("Outputs: sea_results.csv, dlnm_results.csv, dlnm_lag_profile.csv,\n")
cat("         dlnm_slice_lag.csv, mixed_effects_predictions.csv\n")
