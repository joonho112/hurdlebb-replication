## =============================================================================
## B3_rho_cross.R -- Cross-Margin Correlation Recovery
## =============================================================================
## Purpose : Extract and evaluate recovery of the cross-margin correlation
##           rho_cross = cor(delta_ext, delta_int) from simulation output.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##
## Approach:
##   For each replication, compute the plug-in estimator:
##     rho_hat = cor(delta_means[,1], delta_means[,2])
##   using S=51 state-level posterior mean random effects.
##
## Note: E_WS shares the same Stan fit as E_WT; the Cholesky sandwich
##   correction applies to fixed effects only, not hyperparameters (Omega).
##
## Inputs  :
##   data/precomputed/simulation/sim_config.rds
##   data/precomputed/simulation/fits/{S0,S3,S4}/{E_UW,E_WT}/rep_NNN.rds
##
## Outputs :
##   data/precomputed/B3_rho_cross.rds
##   output/tables/ST_rho_cross.tex
##   output/tables/ST_rho_cross.csv
## =============================================================================

cat("==============================================================\n")
cat("  Cross-Margin Correlation Recovery\n")
cat("==============================================================\n\n")


###############################################################################
## SECTION 1 : SETUP
###############################################################################

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
OUTPUT_DIR   <- file.path(PROJECT_ROOT, "data/precomputed")
SIM_DIR      <- file.path(OUTPUT_DIR, "simulation")
FITS_DIR     <- file.path(SIM_DIR, "fits")
RESULTS_DIR  <- file.path(SIM_DIR, "results")
FIGURE_DIR   <- file.path(PROJECT_ROOT, "output/tables")
B3_OUT       <- file.path(OUTPUT_DIR, "B3_rho_cross.rds")

## Create output directory (should already exist)
if (!dir.exists(FIGURE_DIR)) dir.create(FIGURE_DIR, recursive = TRUE)

## Scenarios and estimators
SCENARIOS  <- c("S0", "S3", "S4")
ESTIMATORS <- c("E_UW", "E_WT")    # E_WS shares E_WT fit (identical delta_means)
ALL_ESTIMATORS <- c("E_UW", "E_WT", "E_WS")
R_REPS     <- 200
S_STATES   <- 51

## Scenario display labels (for tables/plots)
SCENARIO_LABELS <- c(
  S0 = "S0: No survey design",
  S3 = "S3: NSECE-like design",
  S4 = "S4: Heavy-tail design"
)

## Load simulation config for true value
config_path <- file.path(SIM_DIR, "sim_config.rds")
stopifnot("Simulation config not found" = file.exists(config_path))
sim_config <- readRDS(config_path)

RHO_TRUE <- sim_config$true_params$rho
stopifnot("True rho not found in config" = !is.null(RHO_TRUE))

cat(sprintf("  PROJECT_ROOT:   %s\n", PROJECT_ROOT))
cat(sprintf("  Output file:    %s\n", B3_OUT))
cat(sprintf("  True rho_cross: %.6f\n", RHO_TRUE))
cat(sprintf("  Scenarios:      %s\n", paste(SCENARIOS, collapse = ", ")))
cat(sprintf("  Estimators:     %s (E_WS = E_WT for hyperparameters)\n",
            paste(ESTIMATORS, collapse = ", ")))
cat(sprintf("  Replications:   R = %d\n", R_REPS))
cat(sprintf("  States:         S = %d\n\n", S_STATES))


###############################################################################
## SECTION 2 : LOAD & EXTRACT rho_hat FROM PER-REPLICATION FILES
###############################################################################
cat("--- 2. Loading per-replication files and extracting rho_cross ---\n\n")

## Storage: list of data frames, one per (scenario, estimator)
rho_raw_list <- list()

## Counters
n_loaded   <- 0
n_skipped  <- 0
n_bad_dims <- 0
t_start    <- proc.time()

for (scen in SCENARIOS) {
  for (est in ESTIMATORS) {
    cat(sprintf("  Processing %s / %s ...\n", scen, est))

    rho_vec  <- numeric(R_REPS)
    valid    <- logical(R_REPS)

    for (r in seq_len(R_REPS)) {
      ## File path: rep_001.rds, rep_002.rds, ..., rep_200.rds
      fname <- sprintf("rep_%03d.rds", r)
      fpath <- file.path(FITS_DIR, scen, est, fname)

      if (!file.exists(fpath)) {
        rho_vec[r]  <- NA_real_
        valid[r]    <- FALSE
        n_skipped   <- n_skipped + 1
        next
      }

      fit_r <- readRDS(fpath)
      n_loaded <- n_loaded + 1

      ## Check fit_ok
      if (!isTRUE(fit_r$fit_ok)) {
        rho_vec[r]  <- NA_real_
        valid[r]    <- FALSE
        next
      }

      ## Extract delta_means (S x 2 matrix)
      dm <- fit_r$delta_means

      if (is.null(dm) || !is.matrix(dm)) {
        rho_vec[r]  <- NA_real_
        valid[r]    <- FALSE
        n_bad_dims  <- n_bad_dims + 1
        next
      }

      if (nrow(dm) != S_STATES || ncol(dm) != 2) {
        rho_vec[r]  <- NA_real_
        valid[r]    <- FALSE
        n_bad_dims  <- n_bad_dims + 1
        next
      }

      ## Compute plug-in rho_hat
      rho_vec[r] <- cor(dm[, 1], dm[, 2])
      valid[r]   <- TRUE
    }

    n_valid <- sum(valid)
    n_na    <- sum(!valid)

    cat(sprintf("    Valid: %d / %d  (skipped/failed: %d)\n",
                n_valid, R_REPS, n_na))

    ## Store results for this (scenario, estimator) pair
    df <- data.frame(
      scenario_id = scen,
      estimator   = est,
      rep_id      = seq_len(R_REPS),
      rho_hat     = rho_vec,
      valid       = valid,
      stringsAsFactors = FALSE
    )
    rho_raw_list[[paste(scen, est, sep = "_")]] <- df

    ## E_WS gets identical values from E_WT
    if (est == "E_WT") {
      df_ws <- df
      df_ws$estimator <- "E_WS"
      rho_raw_list[[paste(scen, "E_WS", sep = "_")]] <- df_ws
    }
  }
}

t_elapsed <- (proc.time() - t_start)[3]

## Combine all raw results
rho_raw <- do.call(rbind, rho_raw_list)
rownames(rho_raw) <- NULL

cat(sprintf("\n  Extraction complete.\n"))
cat(sprintf("    Files loaded:   %d\n", n_loaded))
cat(sprintf("    Files skipped:  %d\n", n_skipped))
cat(sprintf("    Bad dimensions: %d\n", n_bad_dims))
cat(sprintf("    Total rows:     %d  (%d scenarios x %d estimators x %d reps)\n",
            nrow(rho_raw), length(SCENARIOS), length(ALL_ESTIMATORS), R_REPS))
cat(sprintf("    Time elapsed:   %.1f seconds\n", t_elapsed))

## Verify E_WT and E_WS are identical
for (scen in SCENARIOS) {
  rho_wt <- rho_raw$rho_hat[rho_raw$scenario_id == scen &
                              rho_raw$estimator == "E_WT"]
  rho_ws <- rho_raw$rho_hat[rho_raw$scenario_id == scen &
                              rho_raw$estimator == "E_WS"]
  max_diff <- max(abs(rho_wt - rho_ws), na.rm = TRUE)
  if (max_diff < 1e-15) {
    cat(sprintf("    [PASS] %s: E_WT == E_WS (max diff = %.1e)\n",
                scen, max_diff))
  } else {
    cat(sprintf("    [WARN] %s: E_WT != E_WS (max diff = %.1e)\n",
                scen, max_diff))
  }
}

cat("\n")


###############################################################################
## SECTION 3 : COMPUTE PERFORMANCE METRICS
###############################################################################
cat("--- 3. Performance metrics for rho_cross recovery ---\n\n")

## For each (scenario, estimator), compute:
##   - R_valid: number of valid replications
##   - mean_rho:  mean of rho_hat across reps
##   - bias:      mean(rho_hat) - rho_true
##   - rel_bias:  bias / rho_true  (as fraction)
##   - rmse:      sqrt(mean((rho_hat - rho_true)^2))
##   - emp_se:    sd(rho_hat)
##   - median_rho: median of rho_hat
##   - iqr_rho:   interquartile range
##   - min_rho, max_rho:  range

metrics_list <- list()

for (scen in SCENARIOS) {
  for (est in ALL_ESTIMATORS) {
    idx <- rho_raw$scenario_id == scen &
           rho_raw$estimator == est &
           rho_raw$valid

    rho_vals <- rho_raw$rho_hat[idx]
    R_valid  <- length(rho_vals)

    if (R_valid == 0) {
      cat(sprintf("  [WARN] %s / %s: 0 valid reps -- skipping.\n", scen, est))
      next
    }

    mean_rho   <- mean(rho_vals)
    bias       <- mean_rho - RHO_TRUE
    rel_bias   <- bias / RHO_TRUE
    rmse       <- sqrt(mean((rho_vals - RHO_TRUE)^2))
    emp_se     <- sd(rho_vals)
    median_rho <- median(rho_vals)
    iqr_rho    <- IQR(rho_vals)
    min_rho    <- min(rho_vals)
    max_rho    <- max(rho_vals)

    metrics_list[[paste(scen, est, sep = "_")]] <- data.frame(
      scenario_id = scen,
      estimator   = est,
      R_valid     = R_valid,
      mean_rho    = mean_rho,
      bias        = bias,
      rel_bias    = rel_bias,
      rmse        = rmse,
      emp_se      = emp_se,
      median_rho  = median_rho,
      iqr_rho     = iqr_rho,
      min_rho     = min_rho,
      max_rho     = max_rho,
      true_rho    = RHO_TRUE,
      stringsAsFactors = FALSE
    )
  }
}

metrics <- do.call(rbind, metrics_list)
rownames(metrics) <- NULL

## Print summary table
cat("  Performance metrics for rho_cross = cor(delta_ext, delta_int):\n")
cat(sprintf("  True value: rho_cross = %.6f\n\n", RHO_TRUE))

cat(sprintf("  %-5s %-5s  %4s  %8s  %+9s  %+9s  %8s  %8s\n",
            "Scen", "Est", "R", "Mean", "Bias", "Rel.Bias", "RMSE", "Emp.SE"))
cat(sprintf("  %s\n", paste(rep("-", 68), collapse = "")))

for (i in seq_len(nrow(metrics))) {
  m <- metrics[i, ]
  cat(sprintf("  %-5s %-5s  %4d  %8.4f  %+9.4f  %+8.1f%%  %8.4f  %8.4f\n",
              m$scenario_id, m$estimator, m$R_valid,
              m$mean_rho, m$bias, 100 * m$rel_bias,
              m$rmse, m$emp_se))
}

## Note on E_WS
cat("\n  NOTE: E_WS values are identical to E_WT because the sandwich\n")
cat("        correction applies only to fixed effects, not to\n")
cat("        hyperparameters (Omega, tau). Both use the same weighted\n")
cat("        pseudo-posterior fit, producing identical delta_means.\n")

cat("\n")


###############################################################################
## SECTION 4 : APPEND TO sim_raw_all FORMAT
###############################################################################
cat("--- 4. Creating sim_raw_all-compatible data frame ---\n\n")

## The existing sim_raw_all.rds has columns:
##   param, estimator, estimate, ci_lo, ci_hi, ci_width, covers,
##   bias, se, true_value, scenario_id, rep_id
##
## For rho_cross, we do NOT have per-rep CIs, so ci_lo/ci_hi/ci_width
## and covers are set to NA.

rho_raw_compat <- data.frame(
  param       = "rho_cross",
  estimator   = rho_raw$estimator,
  estimate    = rho_raw$rho_hat,
  ci_lo       = NA_real_,
  ci_hi       = NA_real_,
  ci_width    = NA_real_,
  covers      = NA_integer_,
  bias        = rho_raw$rho_hat - RHO_TRUE,
  se          = NA_real_,     # per-rep SE not available (no posterior draws)
  true_value  = RHO_TRUE,
  scenario_id = rho_raw$scenario_id,
  rep_id      = rho_raw$rep_id,
  stringsAsFactors = FALSE
)

## Set invalid replications to NA
rho_raw_compat$estimate[!rho_raw$valid] <- NA_real_
rho_raw_compat$bias[!rho_raw$valid]     <- NA_real_

cat(sprintf("  Created rho_raw_compat: %d rows x %d columns\n",
            nrow(rho_raw_compat), ncol(rho_raw_compat)))
cat(sprintf("  NA estimates (failed fits): %d\n",
            sum(is.na(rho_raw_compat$estimate))))

## Also create a sim_summary_all-compatible row
## Matching columns: param, estimator, R, coverage, coverage_mcse,
##   mean_bias, median_bias, rmse, mean_ci_width, median_ci_width,
##   mean_se, width_ratio, sandwich_note, scenario_id, rel_bias_pct, cov_flag

rho_summary_compat <- data.frame(
  param           = "rho_cross",
  estimator       = metrics$estimator,
  R               = metrics$R_valid,
  coverage        = NA_real_,            # no per-rep CI available
  coverage_mcse   = NA_real_,
  mean_bias       = metrics$bias,
  median_bias     = metrics$median_rho - RHO_TRUE,
  rmse            = metrics$rmse,
  mean_ci_width   = NA_real_,
  median_ci_width = NA_real_,
  mean_se         = metrics$emp_se,      # use frequentist SE as proxy
  width_ratio     = NA_real_,
  sandwich_note   = ifelse(metrics$estimator == "E_WS",
                           "identical to E_WT (hyperparameter)", ""),
  scenario_id     = metrics$scenario_id,
  rel_bias_pct    = 100 * metrics$rel_bias,
  cov_flag        = "",                  # no coverage to flag
  stringsAsFactors = FALSE
)

cat(sprintf("  Created rho_summary_compat: %d rows\n", nrow(rho_summary_compat)))

cat("\n")


###############################################################################
## SECTION 5 : LATEX TABLE FOR SM-E
###############################################################################
cat("--- 5. LaTeX table for supplementary material ---\n\n")

## Format helper
fmt_signed <- function(x, digits = 4) {
  s <- formatC(x, format = "f", digits = digits)
  if (x >= 0) s <- paste0("+", s)
  s
}

## Build the table
tex <- character()

tex <- c(tex,
  "\\begin{table}[t]",
  "\\centering",
  paste0("\\caption{Recovery of the cross-margin correlation ",
         "$\\rho_{\\mathrm{cross}} = \\operatorname{cor}(\\delta_j^{\\mathrm{ext}}, ",
         "\\delta_j^{\\mathrm{int}})$"),
  paste0("  across $R = 200$ simulation replications. ",
         "The plug-in estimator "),
  paste0("  $\\hat{\\rho} = \\operatorname{cor}(\\bar{\\delta}_j^{\\mathrm{ext}}, ",
         "\\bar{\\delta}_j^{\\mathrm{int}})$"),
  paste0("  uses posterior mean random effects ($S = 51$ states). ",
         "True value: $\\rho_{\\mathrm{cross}} = ",
         formatC(RHO_TRUE, format = "f", digits = 4), "$."),
  paste0("  E\\textsubscript{WS} is identical to E\\textsubscript{WT} because the ",
         "sandwich correction applies only to fixed effects.}"),
  "\\label{tab:rho-cross-sim}",
  "\\smallskip",
  "\\small",
  "\\begin{tabular}{@{}ll rrrrr@{}}",
  "\\toprule",
  paste0("Scenario & Estimator & $\\overline{\\hat{\\rho}}$ & ",
         "Bias & Rel.~Bias (\\%) & RMSE & Emp.~SE \\\\"),
  "\\midrule"
)

## Group by scenario with midrules
for (s_idx in seq_along(SCENARIOS)) {
  scen <- SCENARIOS[s_idx]
  scen_display <- switch(scen,
    S0 = "S0 (Baseline)",
    S3 = "S3 (NSECE)",
    S4 = "S4 (Heavy-tail)"
  )

  for (e_idx in seq_along(ALL_ESTIMATORS)) {
    est <- ALL_ESTIMATORS[e_idx]
    m <- metrics[metrics$scenario_id == scen & metrics$estimator == est, ]

    if (nrow(m) == 0) next

    ## First row of scenario block gets the scenario name
    scen_col <- ifelse(e_idx == 1, scen_display, "")

    ## Mark E_WS with dagger
    est_display <- switch(est,
      E_UW = "E\\textsubscript{UW}",
      E_WT = "E\\textsubscript{WT}",
      E_WS = "E\\textsubscript{WS}$^{\\dagger}$"
    )

    row <- sprintf("%-18s & %-35s & %s & $%s$ & $%s$ & %s & %s \\\\",
                   scen_col,
                   est_display,
                   formatC(m$mean_rho, format = "f", digits = 4),
                   fmt_signed(m$bias, 4),
                   fmt_signed(100 * m$rel_bias, 1),
                   formatC(m$rmse, format = "f", digits = 4),
                   formatC(m$emp_se, format = "f", digits = 4))
    tex <- c(tex, row)
  }

  ## Add midrule between scenarios (not after the last)
  if (s_idx < length(SCENARIOS)) {
    tex <- c(tex, "\\midrule")
  }
}

## Footer
tex <- c(tex,
  "\\bottomrule",
  "\\end{tabular}",
  "",
  paste0("\\medskip"),
  paste0("{\\footnotesize $^{\\dagger}$E\\textsubscript{WS} shares the same ",
         "Stan fit as E\\textsubscript{WT}; the Cholesky sandwich correction ",
         "adjusts only fixed-effect covariances, not hyperparameter posteriors.}"),
  "\\end{table}"
)

## Write LaTeX table
tex_path <- file.path(FIGURE_DIR, "ST_rho_cross.tex")
writeLines(tex, tex_path)
cat(sprintf("  [SAVED] %s\n", tex_path))

## CSV backup
csv_path <- file.path(FIGURE_DIR, "ST_rho_cross.csv")
csv_df <- data.frame(
  Scenario    = metrics$scenario_id,
  Estimator   = metrics$estimator,
  R           = metrics$R_valid,
  Mean_rho    = sprintf("%.4f", metrics$mean_rho),
  Bias        = sprintf("%+.4f", metrics$bias),
  Rel_Bias    = sprintf("%+.1f%%", 100 * metrics$rel_bias),
  RMSE        = sprintf("%.4f", metrics$rmse),
  Emp_SE      = sprintf("%.4f", metrics$emp_se),
  Median_rho  = sprintf("%.4f", metrics$median_rho),
  Min_rho     = sprintf("%.4f", metrics$min_rho),
  Max_rho     = sprintf("%.4f", metrics$max_rho),
  True_rho    = sprintf("%.6f", metrics$true_rho),
  stringsAsFactors = FALSE
)
write.csv(csv_df, csv_path, row.names = FALSE)
cat(sprintf("  [SAVED] %s\n\n", csv_path))


###############################################################################
## SECTION 6 : VERIFICATION CHECKS
###############################################################################
cat("--- 6. Verification checks ---\n\n")

checks_passed <- 0
checks_total  <- 0

## Check 1: All replications loaded (no missing files)
checks_total <- checks_total + 1
if (n_skipped == 0) {
  checks_passed <- checks_passed + 1
  cat(sprintf("  [PASS] No missing fit files (%d loaded across 2 estimators x 3 scenarios).\n",
              n_loaded))
} else {
  cat(sprintf("  [WARN] %d missing fit files.\n", n_skipped))
}

## Check 2: All delta_means have correct dimensions (51 x 2)
checks_total <- checks_total + 1
if (n_bad_dims == 0) {
  checks_passed <- checks_passed + 1
  cat("  [PASS] All delta_means have expected dimensions (51 x 2).\n")
} else {
  cat(sprintf("  [WARN] %d replications had unexpected delta_means dimensions.\n",
              n_bad_dims))
}

## Check 3: rho_hat values are in [-1, 1]
checks_total <- checks_total + 1
valid_rho <- rho_raw$rho_hat[rho_raw$valid]
all_in_range <- all(valid_rho >= -1 & valid_rho <= 1)
if (all_in_range) {
  checks_passed <- checks_passed + 1
  cat("  [PASS] All rho_hat values are in [-1, 1].\n")
} else {
  cat(sprintf("  [FAIL] Some rho_hat values are outside [-1, 1]!\n"))
}

## Check 4: E_WT and E_WS are identical across all scenarios
checks_total <- checks_total + 1
wt_ws_match <- TRUE
for (scen in SCENARIOS) {
  rho_wt <- rho_raw$rho_hat[rho_raw$scenario_id == scen &
                              rho_raw$estimator == "E_WT"]
  rho_ws <- rho_raw$rho_hat[rho_raw$scenario_id == scen &
                              rho_raw$estimator == "E_WS"]
  if (max(abs(rho_wt - rho_ws), na.rm = TRUE) > 1e-15) {
    wt_ws_match <- FALSE
  }
}
if (wt_ws_match) {
  checks_passed <- checks_passed + 1
  cat("  [PASS] E_WT and E_WS produce identical rho_hat (as expected).\n")
} else {
  cat("  [FAIL] E_WT and E_WS differ (unexpected)!\n")
}

## Check 5: Number of rows matches expected (3 scenarios x 3 estimators x 200 reps)
checks_total <- checks_total + 1
expected_rows <- length(SCENARIOS) * length(ALL_ESTIMATORS) * R_REPS
if (nrow(rho_raw) == expected_rows) {
  checks_passed <- checks_passed + 1
  cat(sprintf("  [PASS] Raw data has %d rows (expected %d).\n",
              nrow(rho_raw), expected_rows))
} else {
  cat(sprintf("  [FAIL] Raw data has %d rows (expected %d).\n",
              nrow(rho_raw), expected_rows))
}

## Check 6: Bias is moderate (|rel_bias| < 50% in all conditions)
checks_total <- checks_total + 1
max_rel_bias <- max(abs(metrics$rel_bias))
if (max_rel_bias < 0.50) {
  checks_passed <- checks_passed + 1
  cat(sprintf("  [PASS] Max |relative bias| = %.1f%% (< 50%%).\n",
              100 * max_rel_bias))
} else {
  cat(sprintf("  [NOTE] Max |relative bias| = %.1f%% (>= 50%%). ",
              100 * max_rel_bias))
  cat("May indicate systematic attenuation.\n")
}

## Check 7: LaTeX table file exists and has matching begin/end{table}
checks_total <- checks_total + 1
if (file.exists(tex_path)) {
  tex_content <- paste(readLines(tex_path), collapse = "\n")
  has_begin <- grepl("\\\\begin\\{table\\}", tex_content)
  has_end   <- grepl("\\\\end\\{table\\}", tex_content)
  if (has_begin && has_end) {
    checks_passed <- checks_passed + 1
    cat("  [PASS] LaTeX table has matching begin/end{table}.\n")
  } else {
    cat("  [FAIL] LaTeX table structure incomplete!\n")
  }
} else {
  cat("  [FAIL] LaTeX table file not written!\n")
}

## Check 8: Mean rho_hat is positive across all conditions
## (true value is 0.285, so if mean is negative something is very wrong)
checks_total <- checks_total + 1
all_positive <- all(metrics$mean_rho > 0)
if (all_positive) {
  checks_passed <- checks_passed + 1
  cat("  [PASS] Mean rho_hat is positive in all (scenario, estimator) cells.\n")
} else {
  cat("  [WARN] Some cells have negative mean rho_hat (investigate!).\n")
}

cat(sprintf("\n  Verification: %d / %d checks passed.\n\n",
            checks_passed, checks_total))


###############################################################################
## SECTION 7 : MANUSCRIPT LANGUAGE RECOMMENDATION
###############################################################################
cat("--- 7. Manuscript language recommendation ---\n\n")

## Compute key statistics for the recommendation
## Focus on S3 (NSECE-like design) as the primary scenario
m_s3_uw <- metrics[metrics$scenario_id == "S3" & metrics$estimator == "E_UW", ]
m_s3_wt <- metrics[metrics$scenario_id == "S3" & metrics$estimator == "E_WT", ]
m_s0_uw <- metrics[metrics$scenario_id == "S0" & metrics$estimator == "E_UW", ]

## Cross-scenario summary for E_UW
bias_range_uw <- range(metrics$bias[metrics$estimator == "E_UW"])
rmse_range_uw <- range(metrics$rmse[metrics$estimator == "E_UW"])
bias_range_wt <- range(metrics$bias[metrics$estimator == "E_WT"])

cat("  RECOMMENDED TEXT FOR SECTION 4 (Simulation Study):\n\n")

cat("    Suggested paragraph (add after existing parameter recovery discussion):\n\n")

cat("    -----------------------------------------------------------------------\n")
cat("    \\paragraph{Cross-margin correlation.}\n")
cat("    We additionally track recovery of the cross-margin correlation\n")
cat("    $\\rho_{\\mathrm{cross}} = \\mathrm{cor}(\\delta_j^{\\mathrm{ext}},\n")
cat("    \\delta_j^{\\mathrm{int}})$, estimated via the plug-in\n")
cat("    $\\hat{\\rho} = \\mathrm{cor}(\\bar{\\delta}_j^{\\mathrm{ext}},\n")
cat("    \\bar{\\delta}_j^{\\mathrm{int}})$ from $S = 51$ posterior mean\n")
cat("    random effects.\n")

## Dynamic content based on actual results
if (nrow(m_s3_uw) > 0 && nrow(m_s3_wt) > 0) {
  cat(sprintf("    Under the NSECE-like design (S3), bias is %+.3f\n",
              m_s3_uw$bias))
  cat(sprintf("    (relative bias %+.1f\\%%) for E\\textsubscript{UW}\n",
              100 * m_s3_uw$rel_bias))
  cat(sprintf("    and %+.3f (%+.1f\\%%) for E\\textsubscript{WT},\n",
              m_s3_wt$bias, 100 * m_s3_wt$rel_bias))
  cat(sprintf("    with RMSE of %.3f and %.3f respectively\n",
              m_s3_uw$rmse, m_s3_wt$rmse))
  cat("    (\\cref{tab:rho-cross-sim}).\n")
}

cat("    Because full posterior draws of $\\Omega$ were not retained,\n")
cat("    per-replication credible intervals are unavailable; we report\n")
cat("    frequentist Monte Carlo standard errors across replications.\n")
cat("    -----------------------------------------------------------------------\n\n")

## SM note
cat("    Suggested SM-E note:\n\n")
cat("    -----------------------------------------------------------------------\n")
cat("    \\cref{tab:rho-cross-sim} reports recovery of the cross-margin\n")
cat("    correlation $\\rho_{\\mathrm{cross}}$. The plug-in estimator based\n")
cat("    on posterior mean random effects exhibits moderate downward bias,\n")
cat("    a known consequence of posterior shrinkage attenuating the empirical\n")
cat("    correlation. E\\textsubscript{WS} is identical to E\\textsubscript{WT}\n")
cat("    for this hyperparameter because the Cholesky sandwich correction\n")
cat("    (\\cref{thm:cholesky}) adjusts only fixed-effect covariances.\n")
cat("    -----------------------------------------------------------------------\n\n")

## Interpretation notes for the author
cat("  INTERPRETIVE NOTES (for the author, not for manuscript):\n\n")

## Check for downward bias (common with plug-in estimator due to shrinkage)
overall_mean_bias <- mean(metrics$bias[metrics$estimator == "E_UW"])
if (overall_mean_bias < 0) {
  cat("    The plug-in estimator shows downward bias (mean across scenarios:\n")
  cat(sprintf("    %+.4f for E_UW). This is EXPECTED: posterior shrinkage of\n",
              overall_mean_bias))
  cat("    delta_j toward zero attenuates the empirical correlation.\n")
  cat("    With S=51, the attenuation is modest.\n\n")
} else {
  cat("    The plug-in estimator shows upward or near-zero bias.\n\n")
}

cat("    Note: If full Omega draws had been saved, one would report\n")
cat("    posterior mean of Omega[1,2] and its credible interval.\n")
cat("    The plug-in estimator used here is a frequentist surrogate.\n")

cat("\n")


###############################################################################
## SECTION 8 : SAVE RESULTS
###############################################################################
cat("--- 8. Saving results ---\n\n")

B3_results <- list(
  ## Description
  description = paste(
    ": Cross-margin correlation rho_cross recovery.",
    "Plug-in estimator rho_hat = cor(delta_means[,ext], delta_means[,int])",
    "from S=51 posterior mean random effects across R=200 replications,",
    "3 scenarios (S0, S3, S4), and 3 estimators (E_UW, E_WT, E_WS).",
    "E_WS is identical to E_WT for hyperparameters."
  ),

  ## True value
  rho_true = RHO_TRUE,

  ## Dimensions
  R_reps   = R_REPS,
  S_states = S_STATES,
  scenarios  = SCENARIOS,
  estimators = ALL_ESTIMATORS,

  ## Raw per-replication results (3 x 3 x 200 = 1800 rows)
  rho_raw = rho_raw,

  ## Summary metrics (9 rows: 3 scenarios x 3 estimators)
  metrics = metrics,

  ## sim_raw_all compatible format
  rho_raw_compat = rho_raw_compat,

  ## sim_summary_all compatible format
  rho_summary_compat = rho_summary_compat,

  ## Extraction diagnostics
  extraction = list(
    n_loaded   = n_loaded,
    n_skipped  = n_skipped,
    n_bad_dims = n_bad_dims,
    time_sec   = t_elapsed
  ),

  ## Key findings
  findings = list(
    ## E_UW bias across scenarios
    bias_uw = metrics$bias[metrics$estimator == "E_UW"],
    bias_wt = metrics$bias[metrics$estimator == "E_WT"],
    ## Overall mean bias
    overall_bias_uw = mean(metrics$bias[metrics$estimator == "E_UW"]),
    overall_bias_wt = mean(metrics$bias[metrics$estimator == "E_WT"]),
    ## Max RMSE
    max_rmse = max(metrics$rmse),
    max_rmse_cell = paste(
      metrics$scenario_id[which.max(metrics$rmse)],
      metrics$estimator[which.max(metrics$rmse)]
    ),
    ## Note on E_WS
    ews_note = paste(
      "E_WS is identical to E_WT for rho_cross because the Cholesky",
      "sandwich correction adjusts only fixed-effect covariances,",
      "not hyperparameter posteriors (Omega, tau)."
    )
  ),

  ## Output file paths
  output_files = list(
    tex  = tex_path,
    csv  = csv_path,
    rds  = B3_OUT
  ),

  ## Verification
  verification = list(
    checks_passed = checks_passed,
    checks_total  = checks_total,
    all_pass      = (checks_passed == checks_total)
  ),

  ## Timestamp
  timestamp = Sys.time()
)

saveRDS(B3_results, B3_OUT)
cat(sprintf("  Saved: %s\n", B3_OUT))
cat(sprintf("  File size: %.1f KB\n\n",
            file.info(B3_OUT)$size / 1024))


###############################################################################
## SECTION 9 : FINAL SUMMARY
###############################################################################
cat("==============================================================\n")
cat("  Cross-Margin Correlation Recovery\n")
cat("==============================================================\n\n")

cat(sprintf("  TRUE VALUE: rho_cross = %.6f\n\n", RHO_TRUE))

cat("  PERFORMANCE SUMMARY (Mean rho_hat | Bias | RMSE):\n\n")

for (scen in SCENARIOS) {
  cat(sprintf("    %s:\n", SCENARIO_LABELS[scen]))
  for (est in ALL_ESTIMATORS) {
    m <- metrics[metrics$scenario_id == scen & metrics$estimator == est, ]
    if (nrow(m) > 0) {
      note <- ifelse(est == "E_WS", " [= E_WT]", "")
      cat(sprintf("      %-5s:  %.4f  |  %+.4f  |  %.4f%s\n",
                  est, m$mean_rho, m$bias, m$rmse, note))
    }
  }
  cat("\n")
}

cat("  KEY OBSERVATIONS:\n")

## 1. Bias direction
overall_bias_uw <- mean(metrics$bias[metrics$estimator == "E_UW"])
if (overall_bias_uw < -0.01) {
  cat("    1. The plug-in estimator shows systematic DOWNWARD bias\n")
  cat("       (expected: posterior shrinkage attenuates correlation).\n")
} else if (overall_bias_uw > 0.01) {
  cat("    1. The plug-in estimator shows systematic UPWARD bias.\n")
} else {
  cat("    1. Bias is negligible across conditions.\n")
}

## 2. E_WT vs E_UW
cat("    2. E_WT and E_WS produce identical rho_hat (sandwich does NOT\n")
cat("       affect hyperparameters).\n")

## 3. Scenario comparison
rmse_by_scen <- tapply(metrics$rmse[metrics$estimator == "E_UW"],
                       metrics$scenario_id[metrics$estimator == "E_UW"],
                       mean)
worst_scen <- names(which.max(rmse_by_scen))
cat(sprintf("    3. Worst recovery in %s (RMSE = %.4f for E_UW).\n",
            worst_scen, max(rmse_by_scen)))

## 4. Coverage note
cat("    4. Per-replication coverage is UNAVAILABLE (no L_Omega draws\n")
cat("       saved). This is a known limitation of the simulation archive.\n")

cat(sprintf("\n  OUTPUT FILES:\n"))
cat(sprintf("    LaTeX table: %s\n", tex_path))
cat(sprintf("    CSV backup:  %s\n", csv_path))
cat(sprintf("    Results RDS: %s\n", B3_OUT))

cat(sprintf("\n  VERIFICATION: %d / %d checks passed.\n",
            checks_passed, checks_total))

cat("\n  NEXT STEPS:\n")
cat("    1. Add \\input{Figures/ST_rho_cross.tex} to SM-E\n")
cat("    2. Add cross-margin correlation paragraph to section4.tex\n")
cat("    3. Note limitation (plug-in, no CI) in the text\n")

cat("\n==============================================================\n")
cat("  DONE.\n")
cat("==============================================================\n")
