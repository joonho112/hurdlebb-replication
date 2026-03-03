## =============================================================================
## 07_block_b_analyses.R -- Block B Supplementary Analyses
## =============================================================================
## Purpose : Master orchestrator that sources all Block B analysis scripts
##           (B1--B9) in sequence. These scripts produce supplementary tables
##           and figures for the revision.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Prerequisites:
##   - 00_setup.R        : Package installation
##   - 01_data_preparation.R : Stan data object
##   - 02_model_fitting.R    : All model fits
##   - 03_survey_weighting.R : Sandwich variance + Cholesky correction
##   - 04_model_comparison.R : LOO-CV + PPC
##   - 05_marginal_effects.R : AME decomposition
##
## Outputs:
##   B1 : data/precomputed/B1_reversal_probability.rds
##   B2 : output/tables/T_m3b_comparison.tex, ST_tau_comparison.tex
##   B3 : output/tables/rho_cross_*.tex
##   B4 : output/tables/lkj_sensitivity_*.tex
##   B5 : output/tables/misspec_*.tex
##   B6 : output/tables/frequentist_comparison_*.tex
##   B8 : output/tables/coverage_decomposition_*.tex
##   B9 : output/tables/coverage_95_*.tex
##
## Usage:
##   source("code/07_block_b_analyses.R")
## =============================================================================

cat("\n")
cat("================================================================\n")
cat("  Block B Supplementary Analyses\n")
cat("================================================================\n")
cat(sprintf("  Start: %s\n\n", Sys.time()))

## --- Project root ---
PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
BLOCK_B_DIR  <- file.path(PROJECT_ROOT, "code", "block_b")

## --- Helper: source with timing ---
source_with_timing <- function(script_name) {
  script_path <- file.path(BLOCK_B_DIR, script_name)
  if (!file.exists(script_path)) {
    cat(sprintf("  [SKIP] %s not found.\n", script_name))
    return(invisible(NULL))
  }
  cat(sprintf("\n--- Sourcing %s ---\n", script_name))
  t0 <- proc.time()
  tryCatch(
    source(script_path, local = FALSE),
    error = function(e) {
      cat(sprintf("  [ERROR] %s: %s\n", script_name, conditionMessage(e)))
    }
  )
  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("--- %s complete (%.1f sec) ---\n\n", script_name, elapsed))
}

## --- Source Block B scripts in order ---
source_with_timing("B1_reversal_probability.R")
source_with_timing("B2_m3b_comparison.R")
source_with_timing("B3_rho_cross.R")
source_with_timing("B4_lkj_sensitivity.R")
source_with_timing("B5_misspec_analysis.R")
source_with_timing("B6_frequentist_comparison.R")
source_with_timing("B8_coverage_decomposition.R")
source_with_timing("B9_coverage_95.R")

cat("\n================================================================\n")
cat("  Block B Supplementary Analyses -- COMPLETE\n")
cat(sprintf("  End: %s\n", Sys.time()))
cat("================================================================\n")
