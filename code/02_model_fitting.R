## =============================================================================
## 02_model_fitting.R -- Master Model Fitting Orchestrator
## =============================================================================
## Purpose : Run all Stan model fitting scripts sequentially (M0 through M3b-W),
##           including sandwich variance estimation and Cholesky correction.
##           This script sources each model fitting step in order, with timing
##           and progress reporting.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Track   : A (requires NSECE restricted-use data; see data/DATA_ACCESS.md)
## Inputs  : data/precomputed/stan_data.rds  (from code/01_data_preparation.R)
##           stan/*.stan                      (Stan model files)
## Outputs : data/precomputed/fit_m*.rds     (CmdStanR fit objects)
##           data/precomputed/results_m*.rds (Summary tables, LOO, PPC)
##           data/precomputed/scores_m3b_weighted.rds
##           data/precomputed/sandwich_variance.rds
##           data/precomputed/cholesky_correction.rds
##
## IMPORTANT:
##   Total estimated runtime: ~24 hours on a modern 8-core machine.
##   Each model step can also be run individually.
##   Requires: cmdstanr, posterior, loo, dplyr, tidyr, Matrix
##   CmdStan must be installed (see cmdstanr::install_cmdstan()).
## =============================================================================

cat("##############################################################\n")
cat("##                                                          ##\n")
cat("##   HIERARCHICAL HURDLE BETA-BINOMIAL MODEL FITTING        ##\n")
cat("##   Full Replication Pipeline (Track A)                     ##\n")
cat("##                                                          ##\n")
cat("##############################################################\n\n")

## -- 0. Setup ----------------------------------------------------------------

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()

## Verify prerequisites
stan_data_path <- file.path(PROJECT_ROOT, "data/precomputed/stan_data.rds")
if (!file.exists(stan_data_path)) {
  stop("Stan data not found. Run code/01_data_preparation.R first.\n",
       "  Expected: ", stan_data_path,
       call. = FALSE)
}

## Check CmdStan installation
if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  stop("Package 'cmdstanr' is required. Install with:\n",
       "  install.packages('cmdstanr', repos = c('https://mc-stan.org/r-packages/', getOption('repos')))\n",
       "  cmdstanr::install_cmdstan()",
       call. = FALSE)
}

## Track overall timing
pipeline_start <- Sys.time()
step_times <- list()

## Helper: source a script with timing
run_step <- function(script_path, step_name, est_minutes) {
  cat(sprintf("\n================================================================\n"))
  cat(sprintf("  STEP: %s\n", step_name))
  cat(sprintf("  Script: %s\n", script_path))
  cat(sprintf("  Estimated runtime: ~%d minutes\n", est_minutes))
  cat(sprintf("================================================================\n\n"))

  if (!file.exists(script_path)) {
    stop("Script not found: ", script_path, call. = FALSE)
  }

  t0 <- Sys.time()
  source(script_path, local = FALSE)
  t1 <- Sys.time()

  elapsed <- as.numeric(difftime(t1, t0, units = "mins"))
  cat(sprintf("\n  [DONE] %s completed in %.1f minutes.\n\n", step_name, elapsed))
  return(elapsed)
}

## -- 1. M0: Pooled Hurdle Beta-Binomial --------------------------------------
step_times[["M0"]] <- run_step(
  file.path(PROJECT_ROOT, "code/models/10_fit_m0.R"),
  "M0 -- Pooled Hurdle Beta-Binomial (no state effects)",
  est_minutes = 5
)

## -- 2. M1: Random Intercepts ------------------------------------------------
step_times[["M1"]] <- run_step(
  file.path(PROJECT_ROOT, "code/models/20_fit_m1.R"),
  "M1 -- Random Intercepts Hurdle Beta-Binomial",
  est_minutes = 30
)

## -- 3. M2: Block-Diagonal SVC ----------------------------------------------
step_times[["M2"]] <- run_step(
  file.path(PROJECT_ROOT, "code/models/30_fit_m2.R"),
  "M2 -- Block-Diagonal SVC Hurdle Beta-Binomial",
  est_minutes = 120
)

## -- 4. M3a: Cross-Margin Covariance SVC ------------------------------------
step_times[["M3a"]] <- run_step(
  file.path(PROJECT_ROOT, "code/models/40_fit_m3a.R"),
  "M3a -- Cross-Margin Covariance SVC (10x10 joint covariance)",
  est_minutes = 240
)

## -- 5. M3b: Policy Moderator SVC -------------------------------------------
step_times[["M3b"]] <- run_step(
  file.path(PROJECT_ROOT, "code/models/50_fit_m3b.R"),
  "M3b -- Policy Moderator SVC (Gamma matrix, K x Q = 40 params)",
  est_minutes = 300
)

## -- 6. M3b-W: Survey-Weighted Pseudo-Posterior ------------------------------
step_times[["M3b-W"]] <- run_step(
  file.path(PROJECT_ROOT, "code/models/60_fit_m3b_weighted.R"),
  "M3b-W -- Survey-Weighted Pseudo-Posterior + Score Extraction",
  est_minutes = 360
)

## -- 7. Sandwich Variance Estimator ------------------------------------------
step_times[["Sandwich"]] <- run_step(
  file.path(PROJECT_ROOT, "code/models/61_sandwich_variance.R"),
  "Sandwich Variance -- Cluster-Robust V_sand = H^{-1} J H^{-1}",
  est_minutes = 5
)

## -- 8. Cholesky Affine Transformation ---------------------------------------
step_times[["Cholesky"]] <- run_step(
  file.path(PROJECT_ROOT, "code/models/62_cholesky_transform.R"),
  "Cholesky Transform -- Williams-Savitsky (2021) Correction",
  est_minutes = 5
)

## -- Final Summary -----------------------------------------------------------
pipeline_end <- Sys.time()
total_time <- as.numeric(difftime(pipeline_end, pipeline_start, units = "hours"))

cat("\n##############################################################\n")
cat("##                                                          ##\n")
cat("##   MODEL FITTING PIPELINE COMPLETE                        ##\n")
cat("##                                                          ##\n")
cat("##############################################################\n\n")

cat(sprintf("  Total elapsed time: %.1f hours\n\n", total_time))

cat("  Step-by-step timing:\n")
cat(sprintf("    %-40s %10s\n", "Step", "Minutes"))
cat(sprintf("    %s\n", paste(rep("-", 52), collapse = "")))
for (nm in names(step_times)) {
  cat(sprintf("    %-40s %10.1f\n", nm, step_times[[nm]]))
}
cat(sprintf("    %s\n", paste(rep("-", 52), collapse = "")))
cat(sprintf("    %-40s %10.1f\n", "TOTAL", sum(unlist(step_times))))

cat("\n  Output files in data/precomputed/:\n")
output_dir <- file.path(PROJECT_ROOT, "data/precomputed")
output_files <- list.files(output_dir, pattern = "\\.(rds|csv)$")
for (f in sort(output_files)) {
  fpath <- file.path(output_dir, f)
  fsize <- file.info(fpath)$size
  if (fsize > 1024^2) {
    cat(sprintf("    %-45s  %.1f MB\n", f, fsize / 1024^2))
  } else {
    cat(sprintf("    %-45s  %.1f KB\n", f, fsize / 1024))
  }
}

cat("\n  Next step:\n")
cat("    source('code/06_tables_figures.R')  # Generate manuscript outputs\n")

cat("\n##############################################################\n")
