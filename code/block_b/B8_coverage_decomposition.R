## =============================================================================
## B8_coverage_decomposition.R -- Coverage Gap Decomposition
## =============================================================================
## Purpose : Decompose the E-WS coverage shortfall (82-88.5% vs. 90% nominal)
##           into bias vs. width components, and characterize the directional
##           pattern of non-coverage.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##
## Output:
##   output/tables/ST_B8_decomposition.tex
##   output/tables/ST_B8_decomposition.csv
##   data/precomputed/B8_coverage_decomposition.rds
## =============================================================================

library(dplyr)
library(tidyr)
library(xtable)

cat("=== Coverage Gap Decomposition ===\n\n")

## ---- Paths ----
PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
OUTPUT_DIR   <- file.path(PROJECT_ROOT, "data/precomputed")
SIM_DIR      <- file.path(OUTPUT_DIR, "simulation/results")
FIG_DIR      <- file.path(PROJECT_ROOT, "output/tables")

## ---- Load per-rep data ----
raw <- readRDS(file.path(SIM_DIR, "sim_raw_all.rds"))
cat("Per-rep data loaded: ", nrow(raw), " rows\n")
cat("Scenarios:", paste(unique(raw$scenario_id), collapse=", "), "\n")
cat("Parameters:", paste(unique(raw$param), collapse=", "), "\n")
cat("Estimators:", paste(unique(raw$estimator), collapse=", "), "\n")
R <- length(unique(raw$rep_id))
cat("R =", R, "\n\n")

## ==========================================================================
## 1. Directional non-coverage decomposition
## ==========================================================================
cat("--- 1. Directional Non-Coverage Decomposition ---\n\n")

decomp <- raw %>%
  mutate(
    ## Classify each non-covering rep
    miss_lo = as.integer(true_value < ci_lo),   # truth below CI (estimate too high)
    miss_hi = as.integer(true_value > ci_hi),   # truth above CI (estimate too low)
    covers  = as.integer(covers)
  ) %>%
  group_by(scenario_id, param, estimator) %>%
  summarise(
    R         = n(),
    coverage  = mean(covers) * 100,
    n_miss    = sum(1 - covers),
    n_miss_lo = sum(miss_lo),
    n_miss_hi = sum(miss_hi),
    ## Bias metrics
    mean_bias     = mean(bias),
    median_bias   = median(bias),
    mean_abs_bias = mean(abs(bias)),
    ## Width metrics
    mean_width    = mean(ci_width),
    median_width  = median(ci_width),
    ## Relative bias (%)
    rel_bias      = mean(bias) / abs(first(true_value)) * 100,
    .groups = "drop"
  ) %>%
  mutate(
    ## Directional non-coverage rates (%)
    miss_lo_pct  = n_miss_lo / R * 100,
    miss_hi_pct  = n_miss_hi / R * 100,
    ## Non-coverage decomposition
    noncov_pct   = 100 - coverage,
    ## Asymmetry: 0 = symmetric (width-driven), 1 = fully one-sided (bias-driven)
    asymmetry    = ifelse(n_miss > 0,
                          abs(n_miss_lo - n_miss_hi) / n_miss,
                          NA_real_),
    ## Dominant direction
    miss_direction = case_when(
      n_miss_lo > n_miss_hi ~ "below",
      n_miss_hi > n_miss_lo ~ "above",
      TRUE ~ "symmetric"
    )
  )

## ==========================================================================
## 2. Focus on E-WS fixed effects (the coverage gap of interest)
## ==========================================================================
cat("--- 2. E-WS Coverage Gap Analysis ---\n\n")

ews_fe <- decomp %>%
  filter(estimator == "E_WS",
         param %in% c("alpha_poverty", "beta_poverty", "log_kappa"))

cat("  E-WS Fixed-Effect Coverage Gap:\n\n")
print(ews_fe %>%
        select(scenario_id, param, coverage, noncov_pct,
               miss_lo_pct, miss_hi_pct, asymmetry, mean_bias, mean_width) %>%
        mutate(across(where(is.numeric), ~ round(., 2))),
      n = 20)

## ==========================================================================
## 3. Bias-width decomposition via "oracle" intervals
## ==========================================================================
cat("\n--- 3. Bias-Width Decomposition ---\n\n")

## For each param-estimator-scenario, compute:
## (a) Actual coverage
## (b) "Recentered" coverage: shift each CI to be centered on the true value,
##     keeping the same width. This isolates the width component.
## (c) Width gap = 90% - recentered_coverage (insufficient width)
## (d) Bias gap  = recentered_coverage - actual_coverage (centering error)
## So: total gap = bias_gap + width_gap = 90% - actual_coverage

bias_width <- raw %>%
  mutate(
    ## Recentered CI: center on true_value with same width
    half_width    = ci_width / 2,
    rc_lo         = true_value - half_width,
    rc_hi         = true_value + half_width,
    ## Recentered coverage (does estimate fall within recentered CI?)
    ## Equivalently: is |bias| < half_width?
    covers_recentered = as.integer(abs(bias) < half_width)
  ) %>%
  group_by(scenario_id, param, estimator) %>%
  summarise(
    coverage     = mean(covers) * 100,
    cov_recentered = mean(covers_recentered) * 100,
    .groups = "drop"
  ) %>%
  mutate(
    total_gap  = 90 - coverage,
    width_gap  = 90 - cov_recentered,
    bias_gap   = cov_recentered - coverage,
    ## Proportions
    pct_bias  = ifelse(total_gap > 0, bias_gap / total_gap * 100, NA_real_),
    pct_width = ifelse(total_gap > 0, width_gap / total_gap * 100, NA_real_)
  )

cat("  Bias vs. Width Decomposition (E-WS, fixed effects):\n\n")
print(bias_width %>%
        filter(estimator == "E_WS",
               param %in% c("alpha_poverty", "beta_poverty", "log_kappa")) %>%
        select(scenario_id, param, coverage, cov_recentered,
               total_gap, bias_gap, width_gap, pct_bias, pct_width) %>%
        mutate(across(where(is.numeric), ~ round(., 1))),
      n = 20)

## ==========================================================================
## 4. Full decomposition table for all estimators and parameters
## ==========================================================================
cat("\n--- 4. Full Decomposition Table ---\n\n")

full_decomp <- decomp %>%
  left_join(
    bias_width %>% select(scenario_id, param, estimator,
                          cov_recentered, total_gap, bias_gap, width_gap,
                          pct_bias, pct_width),
    by = c("scenario_id", "param", "estimator")
  )

## Focus on the S3 scenario (NSECE-calibrated) for the manuscript table
s3_decomp <- full_decomp %>%
  filter(scenario_id == "S3") %>%
  arrange(factor(param, levels = c("alpha_poverty", "beta_poverty", "log_kappa",
                                    "tau_ext", "tau_int")),
          factor(estimator, levels = c("E_UW", "E_WT", "E_WS")))

cat("  S3 Decomposition:\n\n")
print(s3_decomp %>%
        select(param, estimator, coverage, miss_lo_pct, miss_hi_pct,
               cov_recentered, total_gap, bias_gap, width_gap) %>%
        mutate(across(where(is.numeric), ~ round(., 1))),
      n = 20)

## ==========================================================================
## 5. Generate LaTeX table (S3 scenario, all estimators)
## ==========================================================================
cat("\n--- 5. Generating output files ---\n\n")

## Build manuscript table: focus on S3 (NSECE-calibrated)
param_labels <- c(
  alpha_poverty = "$\\alpha_{\\text{pov}}$",
  beta_poverty  = "$\\beta_{\\text{pov}}$",
  log_kappa     = "$\\log\\kappa$",
  tau_ext       = "$\\tau_{\\text{ext}}$",
  tau_int       = "$\\tau_{\\text{int}}$"
)

tex_lines <- c(
  "% Generated by 87_B8_coverage_decomposition.R",
  "\\small",
  "\\begin{adjustbox}{max width=\\textwidth}",
  "\\begin{tabular}{@{}l l r rr rr rr@{}}",
  "\\toprule",
  " & & & \\multicolumn{2}{c}{Non-coverage (\\%)} &",
  "   \\multicolumn{2}{c}{Gap from 90\\% (pp)} &",
  "   \\multicolumn{2}{c}{Attribution (\\%)} \\\\",
  "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
  "Parameter & Est. & Cov.\\ (\\%) & Below & Above & Bias & Width & \\% Bias & \\% Width \\\\",
  "\\midrule"
)

param_order <- c("alpha_poverty", "beta_poverty", "log_kappa", "tau_ext", "tau_int")
est_order   <- c("E_UW", "E_WT", "E_WS")

for (i in seq_along(param_order)) {
  p <- param_order[i]
  for (j in seq_along(est_order)) {
    e <- est_order[j]
    row <- s3_decomp %>% filter(param == p, estimator == e)
    if (nrow(row) == 0) next

    plabel <- ifelse(j == 1, param_labels[p], "")
    elabel <- gsub("_", "-", e)

    ## Handle cases where total_gap <= 0 (overcoverage)
    if (row$total_gap <= 0) {
      tex_lines <- c(tex_lines, sprintf(
        "%s & %s & %.1f & %.1f & %.1f & --- & --- & --- & --- \\\\",
        plabel, elabel, row$coverage, row$miss_lo_pct, row$miss_hi_pct
      ))
    } else {
      tex_lines <- c(tex_lines, sprintf(
        "%s & %s & %.1f & %.1f & %.1f & %.1f & %.1f & %.0f & %.0f \\\\",
        plabel, elabel, row$coverage, row$miss_lo_pct, row$miss_hi_pct,
        row$bias_gap, row$width_gap,
        row$pct_bias, row$pct_width
      ))
    }
  }
  if (i < length(param_order)) tex_lines <- c(tex_lines, "[3pt]")
}

tex_lines <- c(tex_lines,
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{adjustbox}"
)

tex_file <- file.path(FIG_DIR, "ST_B8_decomposition.tex")
writeLines(tex_lines, tex_file)
cat("  LaTeX table written:", tex_file, "\n")

csv_file <- file.path(FIG_DIR, "ST_B8_decomposition.csv")
write.csv(full_decomp, csv_file, row.names = FALSE)
cat("  CSV written:", csv_file, "\n")

## ==========================================================================
## 6. Save full results
## ==========================================================================
results <- list(
  decomp       = full_decomp,
  bias_width   = bias_width,
  s3_decomp    = s3_decomp,
  raw_summary  = decomp,
  R            = R,
  timestamp    = Sys.time()
)

rds_file <- file.path(OUTPUT_DIR, "B8_coverage_decomposition.rds")
saveRDS(results, rds_file)
cat("  Full results saved:", rds_file, "\n")

## ==========================================================================
## 7. Key findings
## ==========================================================================
cat("\n=== Key Findings (S3 Scenario) ===\n\n")

ews_s3 <- s3_decomp %>% filter(estimator == "E_WS")

cat("E-WS Coverage Gap Decomposition (S3):\n\n")
for (i in seq_len(nrow(ews_s3))) {
  r <- ews_s3[i, ]
  cat(sprintf("  %s: coverage=%.1f%%, gap=%.1f pp (bias=%.1f pp [%.0f%%], width=%.1f pp [%.0f%%])\n",
              r$param, r$coverage, r$total_gap,
              r$bias_gap, r$pct_bias, r$width_gap, r$pct_width))
  cat(sprintf("    Directional: miss_lo=%.1f%%, miss_hi=%.1f%% → %s\n",
              r$miss_lo_pct, r$miss_hi_pct, r$miss_direction))
}

cat("\n  Summary: The E-WS coverage gap is primarily driven by\n")
mean_pct_width <- mean(ews_s3$pct_width[!is.na(ews_s3$pct_width)])
mean_pct_bias  <- mean(ews_s3$pct_bias[!is.na(ews_s3$pct_bias)])
cat(sprintf("  width (%.0f%%) rather than bias (%.0f%%) for fixed effects.\n",
            mean_pct_width, mean_pct_bias))

cat("\nDONE.\n")
