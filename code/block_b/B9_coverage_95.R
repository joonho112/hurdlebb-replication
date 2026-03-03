## =============================================================================
## B9_coverage_95.R -- 95% Coverage Results
## =============================================================================
## Purpose : Compute 95% CI coverage alongside the primary 90% coverage.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##
## Approach:
##   For all estimators, 95% CI = estimate +/- 1.96 * se.
##
## Inputs:
##   data/precomputed/simulation/results/sim_raw_all.rds
##
## Outputs:
##   output/tables/ST_B9_coverage95.tex
##   output/tables/ST_B9_coverage95.csv
##   data/precomputed/B9_coverage95.rds
## =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()

## ── 1. Load data ──────────────────────────────────────────────────────────────
raw <- readRDS(file.path(PROJECT_ROOT,
  "data/precomputed/simulation/results/sim_raw_all.rds"))
cat("Loaded sim_raw_all.rds:", nrow(raw), "rows\n")

## ── 2. Compute 95% CIs and coverage ──────────────────────────────────────────

z90 <- qnorm(0.95)   # 1.6449
z95 <- qnorm(0.975)  # 1.96

raw <- raw %>%
  mutate(
    ## Reconstruct 90% Wald CI (verify matches existing)
    ci_lo_90 = estimate - z90 * se,
    ci_hi_90 = estimate + z90 * se,
    covers_90_wald = as.integer(true_value >= ci_lo_90 & true_value <= ci_hi_90),
    ## New 95% CI
    ci_lo_95 = estimate - z95 * se,
    ci_hi_95 = estimate + z95 * se,
    covers_95 = as.integer(true_value >= ci_lo_95 & true_value <= ci_hi_95),
    ci_width_95 = ci_hi_95 - ci_lo_95
  )

## Verify that 90% Wald matches the original covers column
check <- raw %>%
  group_by(param, estimator, scenario_id) %>%
  summarize(
    orig_cov = mean(covers),
    wald_cov = mean(covers_90_wald),
    diff = abs(mean(covers) - mean(covers_90_wald)),
    .groups = "drop"
  )
cat("\n── Verification: 90% Wald vs original quantile-based ──\n")
cat("Max absolute difference:", max(check$diff), "\n")
cat("Mean absolute difference:", mean(check$diff), "\n\n")

## ── 3. Aggregate to coverage rates ───────────────────────────────────────────

summary_df <- raw %>%
  group_by(param, estimator, scenario_id) %>%
  summarize(
    R       = n(),
    cov_90  = mean(covers) * 100,
    cov_95  = mean(covers_95) * 100,
    mcse_90 = sqrt(mean(covers) * (1 - mean(covers)) / n()) * 100,
    mcse_95 = sqrt(mean(covers_95) * (1 - mean(covers_95)) / n()) * 100,
    mean_width_90 = mean(ci_width),
    mean_width_95 = mean(ci_width_95),
    width_ratio_95_90 = mean(ci_width_95) / mean(ci_width),
    .groups = "drop"
  )

## ── 4. Print summary ────────────────────────────────────────────────────────

cat("── Coverage Summary (S3 scenario, key cells) ──\n\n")
s3 <- summary_df %>% filter(scenario_id == "S3")
for (i in seq_len(nrow(s3))) {
  r <- s3[i, ]
  cat(sprintf("  %-16s %-6s  90%%: %5.1f (±%.1f)  95%%: %5.1f (±%.1f)  Δ: %+.1f pp\n",
              r$param, r$estimator,
              r$cov_90, r$mcse_90,
              r$cov_95, r$mcse_95,
              r$cov_95 - r$cov_90))
}

cat("\n── Coverage Summary (S0 scenario) ──\n\n")
s0 <- summary_df %>% filter(scenario_id == "S0")
for (i in seq_len(nrow(s0))) {
  r <- s0[i, ]
  cat(sprintf("  %-16s %-6s  90%%: %5.1f (±%.1f)  95%%: %5.1f (±%.1f)  Δ: %+.1f pp\n",
              r$param, r$estimator,
              r$cov_90, r$mcse_90,
              r$cov_95, r$mcse_95,
              r$cov_95 - r$cov_90))
}

cat("\n── Coverage Summary (S4 scenario) ──\n\n")
s4 <- summary_df %>% filter(scenario_id == "S4")
for (i in seq_len(nrow(s4))) {
  r <- s4[i, ]
  cat(sprintf("  %-16s %-6s  90%%: %5.1f (±%.1f)  95%%: %5.1f (±%.1f)  Δ: %+.1f pp\n",
              r$param, r$estimator,
              r$cov_90, r$mcse_90,
              r$cov_95, r$mcse_95,
              r$cov_95 - r$cov_90))
}

## ── 5. Build LaTeX table (side-by-side 90% and 95% for S0/S3/S4) ────────

## Focus on the same structure as the main sim-results table
## but with both 90% and 95% columns

param_labels <- c(
  alpha_poverty = "$\\alpha_{\\text{pov}}$",
  beta_poverty  = "$\\beta_{\\text{pov}}$",
  log_kappa     = "$\\log\\kappa$",
  tau_ext       = "$\\tau_{\\text{ext}}$",
  tau_int       = "$\\tau_{\\text{int}}$"
)

param_order <- c("alpha_poverty", "beta_poverty", "log_kappa", "tau_ext", "tau_int")
est_order   <- c("E_UW", "E_WT", "E_WS")
scen_order  <- c("S0", "S3", "S4")

## Pivot to wide format: one row per param × estimator
wide <- summary_df %>%
  select(param, estimator, scenario_id, cov_90, cov_95, mcse_90, mcse_95) %>%
  pivot_wider(
    names_from  = scenario_id,
    values_from = c(cov_90, cov_95, mcse_90, mcse_95),
    names_glue  = "{.value}_{scenario_id}"
  )

## Format a coverage cell with MCSE
fmt_cell <- function(cov, mcse) {
  sprintf("%.1f\\,(%.1f)", cov, mcse)
}

## Build table lines
lines <- character()
lines <- c(lines, "% Generated by 88_B9_coverage_95.R — 90% and 95% coverage comparison")
lines <- c(lines, "\\small")
lines <- c(lines, "\\begin{adjustbox}{max width=\\textwidth}")
lines <- c(lines, "\\begin{tabular}{@{}l l  cc  cc  cc @{}}")
lines <- c(lines, "\\toprule")
lines <- c(lines, " & & \\multicolumn{2}{c}{S0 (Non-inform.)} & \\multicolumn{2}{c}{S3 (NSECE)} & \\multicolumn{2}{c}{S4 (Stress)} \\\\")
lines <- c(lines, "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}")
lines <- c(lines, "Parameter & Est. & 90\\% & 95\\% & 90\\% & 95\\% & 90\\% & 95\\% \\\\")
lines <- c(lines, "\\midrule")

for (p in param_order) {
  first_est <- TRUE
  for (e in est_order) {
    r <- wide %>% filter(param == p, estimator == e)
    if (nrow(r) == 0) next

    param_col <- if (first_est) param_labels[p] else ""
    est_label <- sub("E_", "E-", e)

    s0_90 <- fmt_cell(r$cov_90_S0, r$mcse_90_S0)
    s0_95 <- fmt_cell(r$cov_95_S0, r$mcse_95_S0)
    s3_90 <- fmt_cell(r$cov_90_S3, r$mcse_90_S3)
    s3_95 <- fmt_cell(r$cov_95_S3, r$mcse_95_S3)
    s4_90 <- fmt_cell(r$cov_90_S4, r$mcse_90_S4)
    s4_95 <- fmt_cell(r$cov_95_S4, r$mcse_95_S4)

    line <- sprintf("%s & %s & %s & %s & %s & %s & %s & %s \\\\",
                    param_col, est_label,
                    s0_90, s0_95, s3_90, s3_95, s4_90, s4_95)
    lines <- c(lines, line)
    first_est <- FALSE
  }
  ## Add vertical space between parameter groups
  if (p != tail(param_order, 1)) {
    lines <- c(lines, "[3pt]")
  }
}

lines <- c(lines, "\\bottomrule")
lines <- c(lines, "\\end{tabular}")
lines <- c(lines, "\\end{adjustbox}")

table_tex <- paste(lines, collapse = "\n")

## ── 6. Save outputs ─────────────────────────────────────────────────────────

out_dir <- file.path(PROJECT_ROOT, "output/tables")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
writeLines(table_tex, file.path(out_dir, "ST_B9_coverage95.tex"))
cat("\nWrote:", file.path(out_dir, "ST_B9_coverage95.tex"), "\n")

## CSV
csv_out <- summary_df %>% arrange(match(param, param_order), match(estimator, est_order), match(scenario_id, scen_order))
write.csv(csv_out, file.path(out_dir, "ST_B9_coverage95.csv"), row.names = FALSE)
cat("Wrote:", file.path(out_dir, "ST_B9_coverage95.csv"), "\n")

## RDS
saveRDS(list(
  summary     = summary_df,
  raw_augmented = raw,
  verification = check
), file.path(PROJECT_ROOT, "data/precomputed/B9_coverage95.rds"))
cat("Wrote:", file.path(PROJECT_ROOT, "data/precomputed/B9_coverage95.rds"), "\n")

cat("\nB9 analysis complete.\n")
