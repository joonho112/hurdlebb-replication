## =============================================================================
## 60_fit_m3b_weighted.R -- Fit Survey-Weighted Policy Moderator SVC Model (M3b-W)
## =============================================================================
## Purpose : Compile and fit the survey-weighted pseudo-posterior M3b-W
##           model on NSECE 2019 data, compare weighted vs unweighted
##           fixed effects, and extract score matrices for sandwich variance.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Track   : A (requires NSECE restricted-use data; see data/DATA_ACCESS.md)
## Inputs  : data/precomputed/stan_data.rds
##           stan/hbb_m3b_weighted.stan
##           code/helpers/utils.R
##           data/precomputed/results_m3b.rds
## Outputs : data/precomputed/fit_m3b_weighted.rds
##           data/precomputed/results_m3b_weighted.rds
##           data/precomputed/scores_m3b_weighted.rds
## =============================================================================

cat("==============================================================\n")
cat("  HBB Replication: M3b-W Fitting  (Phase 6)\n")
cat("  Survey-Weighted Policy Moderator SVC Hurdle Beta-Binomial\n")
cat("==============================================================\n\n")

# == 0. Setup ================================================================

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()

## Source helper functions
source(file.path(PROJECT_ROOT, "code/helpers/utils.R"))

## Load required packages
library(cmdstanr)
library(posterior)
library(loo)
library(dplyr, warn.conflicts = FALSE)

## Paths
STAN_DATA_PATH   <- file.path(PROJECT_ROOT, "data/precomputed/stan_data.rds")
STAN_MODEL_PATH  <- file.path(PROJECT_ROOT, "stan/hbb_m3b_weighted.stan")
OUTPUT_DIR       <- file.path(PROJECT_ROOT, "data/precomputed")
FIT_OUT          <- file.path(OUTPUT_DIR, "fit_m3b_weighted.rds")
RESULTS_OUT      <- file.path(OUTPUT_DIR, "results_m3b_weighted.rds")
SCORES_OUT       <- file.path(OUTPUT_DIR, "scores_m3b_weighted.rds")
RESULTS_M3B_PATH <- file.path(OUTPUT_DIR, "results_m3b.rds")

## Ensure output directory exists
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)


###############################################################################
## SECTION 1 : LOAD DATA AND BUILD WEIGHTED DATA LIST
###############################################################################
cat("--- 1. Loading Stan data and constructing weighted data list ---\n")

stopifnot("Stan data file not found" = file.exists(STAN_DATA_PATH))
full_stan_data <- readRDS(STAN_DATA_PATH)
cat(sprintf("  Loaded: %s\n", STAN_DATA_PATH))
cat(sprintf("  N = %d, P = %d, S = %d, Q = %d\n",
            full_stan_data$N, full_stan_data$P, full_stan_data$S,
            full_stan_data$Q))

## Extract normalized weights: w_tilde is already in stan_data (from 01_data_prep.R)
N <- full_stan_data$N
w_tilde <- full_stan_data$w_tilde
stopifnot("Survey weights (w_tilde) not found in stan_data" = !is.null(w_tilde))
stopifnot("All weights must be positive" = all(w_tilde > 0))

cat(sprintf("  Normalized weights: N=%d, sum(w_tilde)=%.1f (should equal N=%d)\n",
            N, sum(w_tilde), N))

## Weight summary statistics
w_mean <- mean(w_tilde)
w_sd   <- sd(w_tilde)
w_min  <- min(w_tilde)
w_max  <- max(w_tilde)
w_cv   <- w_sd / w_mean
ess_kish  <- (sum(w_tilde))^2 / sum(w_tilde^2)
deff_kish <- N / ess_kish

cat(sprintf("\n  Weight summary (w_tilde):\n"))
cat(sprintf("    mean = %.4f\n", w_mean))
cat(sprintf("    sd   = %.4f\n", w_sd))
cat(sprintf("    min  = %.4f\n", w_min))
cat(sprintf("    max  = %.4f\n", w_max))
cat(sprintf("    CV   = %.4f\n", w_cv))
cat(sprintf("    Kish ESS   = %.0f\n", ess_kish))
cat(sprintf("    DEFF_kish  = %.2f\n", deff_kish))

## Build M3b-W data list: same as M3b plus w_tilde
P <- full_stan_data$P
S <- full_stan_data$S
Q <- full_stan_data$Q
K <- 2 * P  # Joint dimension: P extensive + P intensive = 10

stan_data_m3bw <- list(
  N       = N,
  P       = P,
  S       = S,
  Q       = Q,
  y       = full_stan_data$y,
  n_trial = full_stan_data$n_trial,
  z       = full_stan_data$z,
  X       = full_stan_data$X,
  state   = full_stan_data$state,
  v_state = full_stan_data$V,    # S x Q state-level policy design matrix
  w_tilde = w_tilde              # NEW: normalized survey weights
)

cat(sprintf("\n  M3b-W data list: N=%d, P=%d, S=%d, Q=%d, K=2P=%d\n",
            N, P, S, Q, K))
cat(sprintf("  Zero rate: %.1f%%\n", 100 * mean(stan_data_m3bw$z == 0)))
cat(sprintf("  w_tilde included: length=%d, sum=%.1f\n",
            length(w_tilde), sum(w_tilde)))
cat("  [PASS] Data loaded and M3b-W data list constructed.\n\n")


###############################################################################
## SECTION 2 : COMPILE STAN MODEL
###############################################################################
cat("--- 2. Compiling Stan model (M3b-W) ---\n")

stopifnot("Stan model file not found" = file.exists(STAN_MODEL_PATH))
cat(sprintf("  Model file: %s\n", STAN_MODEL_PATH))

## Compile (cmdstanr caches compiled models; recompiles only if .stan changed)
m3bw_model <- tryCatch(
  cmdstan_model(STAN_MODEL_PATH),
  error = function(e) {
    stop(
      "Stan model compilation FAILED.\n",
      "  Error: ", conditionMessage(e), "\n",
      "  Check the .stan file for syntax errors.",
      call. = FALSE
    )
  }
)
cat("  [PASS] Model compiled successfully.\n\n")


###############################################################################
## SECTION 3 : INITIALISATION FROM M3b UNWEIGHTED POSTERIORS (WARM START)
###############################################################################
cat("--- 3. Setting initial values (warm start from M3b unweighted) ---\n")

if (file.exists(RESULTS_M3B_PATH)) {
  results_m3b <- readRDS(RESULTS_M3B_PATH)
  cat("  [INFO] M3b unweighted results found. Using posteriors for warm start.\n")

  ## Extract M3b posterior means
  m3b_alpha     <- as.numeric(results_m3b$alpha_means)
  m3b_beta      <- as.numeric(results_m3b$beta_means)
  m3b_log_kappa <- as.numeric(results_m3b$log_kappa_mean)
  m3b_tau       <- as.numeric(results_m3b$tau_means)  # length K

  ## Gamma: use M3b posterior means if available, else zero
  if (!is.null(results_m3b$gamma_mean_mat)) {
    m3b_gamma <- results_m3b$gamma_mean_mat  # K x Q matrix
    cat("  [INFO] Using M3b Gamma posterior means for warm start.\n")
  } else {
    m3b_gamma <- matrix(0, K, Q)
    cat("  [INFO] M3b Gamma not found. Initializing Gamma to zero.\n")
  }

  init_fun <- function() {
    list(
      alpha     = m3b_alpha,
      beta      = m3b_beta,
      log_kappa = m3b_log_kappa,
      tau       = m3b_tau,
      L_Omega   = diag(K),
      Gamma     = m3b_gamma,
      z_eps     = rep(list(rep(0, K)), S)
    )
  }

  cat(sprintf("  alpha init    : [%s]\n",
              paste(sprintf("%+.3f", m3b_alpha), collapse = ", ")))
  cat(sprintf("  beta init     : [%s]\n",
              paste(sprintf("%+.3f", m3b_beta), collapse = ", ")))
  cat(sprintf("  log_kappa init: %.3f\n", m3b_log_kappa))
  cat(sprintf("  tau init      : [%s]\n",
              paste(sprintf("%.3f", m3b_tau), collapse = ", ")))
  cat("  L_Omega       : identity matrix (K x K)\n")
  cat(sprintf("  Gamma         : M3b posterior means (%d x %d)\n", K, Q))
  cat("  z_eps         : list of S zero vectors (length K)\n")
} else {
  cat("  [INFO] M3b results not found. Using default initialisation.\n")

  init_fun <- function() {
    list(
      alpha     = rep(0, P),
      beta      = rep(0, P),
      log_kappa = log(10),
      tau       = rep(0.2, K),
      L_Omega   = diag(K),
      Gamma     = matrix(0, K, Q),
      z_eps     = rep(list(rep(0, K)), S)
    )
  }
}

cat("  [PASS] Initial values set.\n\n")


###############################################################################
## SECTION 4 : FIT M3b-W
###############################################################################
cat("--- 4. Fitting M3b-W (survey-weighted pseudo-posterior) ---\n")
cat("  chains = 4, parallel_chains = 4\n")
cat("  iter_warmup = 1000, iter_sampling = 1000\n")
cat("  adapt_delta = 0.95, max_treedepth = 12\n")
cat("  seed = 20250220\n\n")

fit_start <- Sys.time()

fit_m3bw <- tryCatch(
  m3bw_model$sample(
    data            = stan_data_m3bw,
    seed            = 20250220,
    chains          = 4,
    parallel_chains = 4,
    iter_warmup     = 1000,
    iter_sampling   = 1000,
    adapt_delta     = 0.95,
    max_treedepth   = 12,
    init            = init_fun,
    refresh         = 100
  ),
  error = function(e) {
    stop(
      "Stan sampling FAILED.\n",
      "  Error: ", conditionMessage(e),
      call. = FALSE
    )
  }
)

fit_end <- Sys.time()
fit_time <- difftime(fit_end, fit_start, units = "mins")
cat(sprintf("\n  Fitting completed in %.1f minutes.\n\n", as.numeric(fit_time)))


###############################################################################
## SECTION 5 : MCMC DIAGNOSTICS
###############################################################################
cat("--- 5. MCMC Diagnostics ---\n")

diag_pass <- TRUE

## 5a. Divergent transitions
n_divergent <- fit_m3bw$diagnostic_summary()$num_divergent
total_divergent <- sum(n_divergent)
cat(sprintf("  Divergent transitions: %d (per chain: %s)\n",
            total_divergent, paste(n_divergent, collapse = ", ")))
if (total_divergent > 0) {
  warning("  [WARN] Divergent transitions detected! Consider increasing adapt_delta.")
  diag_pass <- FALSE
} else {
  cat("  [PASS] No divergent transitions.\n")
}

## 5b. Max treedepth exceedances
n_max_treedepth <- fit_m3bw$diagnostic_summary()$num_max_treedepth
total_max_td <- sum(n_max_treedepth)
cat(sprintf("  Max treedepth hits: %d (per chain: %s)\n",
            total_max_td, paste(n_max_treedepth, collapse = ", ")))
if (total_max_td > 0) {
  warning("  [WARN] Max treedepth reached. Consider increasing max_treedepth.")
}

## 5c. Parameter summary -- fixed effects + hierarchical + Gamma
## Fixed effects
param_names_fixed <- c(paste0("alpha[", 1:P, "]"),
                       paste0("beta[", 1:P, "]"),
                       "log_kappa")

## Hierarchical parameters: tau[1:K]
param_names_tau <- paste0("tau[", 1:K, "]")
param_names_hier <- param_names_tau

## Gamma parameters: Gamma[k,q] for k=1:K, q=1:Q
param_names_gamma <- c()
for (k in 1:K) {
  for (q in 1:Q) {
    param_names_gamma <- c(param_names_gamma, sprintf("Gamma[%d,%d]", k, q))
  }
}

## Full summary for diagnostics (fixed + hierarchical + Gamma)
param_names_all <- c(param_names_fixed, param_names_hier, param_names_gamma)
param_summary <- fit_m3bw$summary(variables = param_names_all)

cat("\n  Fixed effect parameter summary:\n")
param_summary_fixed <- fit_m3bw$summary(variables = param_names_fixed)
print(param_summary_fixed, n = nrow(param_summary_fixed))

cat("\n  Hierarchical parameter summary (tau[1:K]):\n")
param_summary_hier <- fit_m3bw$summary(variables = param_names_hier)
print(param_summary_hier, n = nrow(param_summary_hier))

cat("\n  Gamma parameter summary:\n")
param_summary_gamma <- fit_m3bw$summary(variables = param_names_gamma)
print(param_summary_gamma, n = nrow(param_summary_gamma))

## 5d. R-hat check (on fixed + hierarchical + Gamma parameters)
max_rhat <- max(param_summary$rhat, na.rm = TRUE)
cat(sprintf("\n  Max R-hat (fixed+hier+Gamma): %.4f (threshold: 1.01)\n", max_rhat))
if (max_rhat > 1.01) {
  warning("  [WARN] R-hat > 1.01 detected. Chains may not have converged.")
  diag_pass <- FALSE
} else {
  cat("  [PASS] All R-hat < 1.01.\n")
}

## Also check R-hat on all delta parameters
delta_all_names <- c()
for (s in 1:S) {
  for (k in 1:K) {
    delta_all_names <- c(delta_all_names, sprintf("delta[%d,%d]", s, k))
  }
}
delta_all_summary <- fit_m3bw$summary(variables = delta_all_names)
max_rhat_delta <- max(delta_all_summary$rhat, na.rm = TRUE)
min_ess_delta  <- min(delta_all_summary$ess_bulk, na.rm = TRUE)
cat(sprintf("  Max R-hat (deltas): %.4f\n", max_rhat_delta))
cat(sprintf("  Min ESS_bulk (deltas): %.0f\n", min_ess_delta))
if (max_rhat_delta > 1.01) {
  warning("  [WARN] R-hat > 1.01 for some delta parameters.")
  diag_pass <- FALSE
} else {
  cat("  [PASS] All delta R-hat < 1.01.\n")
}

## 5e. ESS check (bulk and tail) on fixed + hierarchical + Gamma
min_ess_bulk <- min(param_summary$ess_bulk, na.rm = TRUE)
min_ess_tail <- min(param_summary$ess_tail, na.rm = TRUE)
cat(sprintf("  Min ESS (bulk, fixed+hier+Gamma): %.0f (threshold: 400)\n", min_ess_bulk))
cat(sprintf("  Min ESS (tail, fixed+hier+Gamma): %.0f (threshold: 400)\n", min_ess_tail))

if (min_ess_bulk < 400) {
  warning("  [WARN] ESS_bulk < 400 for some parameters.")
  diag_pass <- FALSE
} else {
  cat("  [PASS] All ESS_bulk > 400.\n")
}

if (min_ess_tail < 400) {
  warning("  [WARN] ESS_tail < 400 for some parameters.")
  diag_pass <- FALSE
} else {
  cat("  [PASS] All ESS_tail > 400.\n")
}

if (diag_pass) {
  cat("\n  [PASS] ALL MCMC DIAGNOSTICS PASSED.\n\n")
} else {
  cat("\n  [WARN] SOME DIAGNOSTICS FAILED. Review above warnings.\n\n")
}


###############################################################################
## SECTION 6 : COMPARE WEIGHTED VS UNWEIGHTED FIXED EFFECTS
###############################################################################
cat("--- 6. Weighted vs unweighted fixed effect comparison ---\n")
cat("  Key expectation: slight shifts in intensive margin (informative margin).\n\n")

## Extract M3b-W posterior means and CIs
alpha_means_w <- param_summary_fixed$mean[grepl("^alpha", param_summary_fixed$variable)]
beta_means_w  <- param_summary_fixed$mean[grepl("^beta",  param_summary_fixed$variable)]
log_kappa_mean_w <- param_summary_fixed$mean[param_summary_fixed$variable == "log_kappa"]
kappa_mean_w  <- exp(log_kappa_mean_w)

## NOTE: Naive MCMC quantiles (q5/q95) are NOT used for final inference.
## Design-corrected Wald CIs from the sandwich estimator (62_cholesky_transform.R)
## are the primary reporting intervals. See 09b_step6-technical-issues.qmd.

## Covariate names for display
cov_names <- c("intercept", "poverty", "urban", "black", "hispanic")

## Labels for tau
tau_labels <- c(paste0("ext_", cov_names), paste0("int_", cov_names))

## Hierarchical parameter means
tau_means_w <- param_summary_hier$mean

if (file.exists(RESULTS_M3B_PATH)) {
  results_m3b <- readRDS(RESULTS_M3B_PATH)
  alpha_means_uw <- as.numeric(results_m3b$alpha_means)
  beta_means_uw  <- as.numeric(results_m3b$beta_means)
  log_kappa_mean_uw <- as.numeric(results_m3b$log_kappa_mean)

  cat(sprintf("  %-12s %10s %10s %10s %10s\n",
              "Parameter", "Unweighted", "Weighted", "Shift", "Pct_shift"))
  cat(sprintf("  %s\n", paste(rep("-", 60), collapse = "")))

  ## Extensive margin (alpha)
  for (k in 1:P) {
    shift <- alpha_means_w[k] - alpha_means_uw[k]
    pct_shift <- ifelse(abs(alpha_means_uw[k]) > 1e-6,
                        100 * shift / abs(alpha_means_uw[k]), NA)
    cat(sprintf("  alpha[%d] %-7s %+10.4f %+10.4f %+10.4f %+9.1f%%\n",
                k, cov_names[k], alpha_means_uw[k], alpha_means_w[k],
                shift, pct_shift))
  }

  ## Intensive margin (beta)
  for (k in 1:P) {
    shift <- beta_means_w[k] - beta_means_uw[k]
    pct_shift <- ifelse(abs(beta_means_uw[k]) > 1e-6,
                        100 * shift / abs(beta_means_uw[k]), NA)
    cat(sprintf("  beta[%d]  %-7s %+10.4f %+10.4f %+10.4f %+9.1f%%\n",
                k, cov_names[k], beta_means_uw[k], beta_means_w[k],
                shift, pct_shift))
  }

  ## Dispersion
  shift_lk <- log_kappa_mean_w - log_kappa_mean_uw
  pct_shift_lk <- 100 * shift_lk / abs(log_kappa_mean_uw)
  cat(sprintf("  log_kappa   %+10.4f %+10.4f %+10.4f %+9.1f%%\n",
              log_kappa_mean_uw, log_kappa_mean_w, shift_lk, pct_shift_lk))

  ## Summarize: which margin shows bigger shifts?
  ext_total_shift <- sum(abs(alpha_means_w - alpha_means_uw))
  int_total_shift <- sum(abs(beta_means_w  - beta_means_uw))
  cat(sprintf("\n  Total absolute shift (extensive): %.4f\n", ext_total_shift))
  cat(sprintf("  Total absolute shift (intensive): %.4f\n", int_total_shift))

  if (int_total_shift > ext_total_shift) {
    cat("  [EXPECTED] Intensive margin shows larger shifts (informative margin).\n")
  } else {
    cat("  [NOTE] Extensive margin shows larger shifts (unexpected; check informativeness).\n")
  }
} else {
  cat("  [INFO] M3b unweighted results not found. Printing weighted-only summaries.\n\n")

  cat("  Extensive margin (alpha):\n")
  for (k in 1:P) {
    cat(sprintf("    alpha[%d] %-10s = %+.4f\n", k, cov_names[k], alpha_means_w[k]))
  }

  cat("\n  Intensive margin (beta):\n")
  for (k in 1:P) {
    cat(sprintf("    beta[%d]  %-10s = %+.4f\n", k, cov_names[k], beta_means_w[k]))
  }

  cat(sprintf("\n  log_kappa = %.4f, kappa = %.2f\n", log_kappa_mean_w, kappa_mean_w))
}

## Poverty reversal check under weighting
alpha_poverty_w <- alpha_means_w[2]
beta_poverty_w  <- beta_means_w[2]

cat("\n  === POVERTY REVERSAL CHECK (WEIGHTED) ===\n")
cat(sprintf("    alpha_poverty = %+.4f (expected: negative)\n", alpha_poverty_w))
cat(sprintf("    beta_poverty  = %+.4f (expected: positive)\n", beta_poverty_w))

if (alpha_poverty_w < 0) {
  cat("    [PASS] alpha_poverty < 0 (higher poverty -> less likely to serve IT)\n")
} else {
  warning("    [WARN] alpha_poverty >= 0 -- unexpected sign!")
}

if (beta_poverty_w > 0) {
  cat("    [PASS] beta_poverty > 0 (higher poverty -> higher IT share)\n")
} else {
  warning("    [WARN] beta_poverty <= 0 -- unexpected sign!")
}

cat("\n")


###############################################################################
## SECTION 7 : EXTRACT SCORE MATRICES FOR SANDWICH VARIANCE
###############################################################################
cat("--- 7. Extracting score matrices for sandwich variance ---\n")
cat("  Strategy: compute posterior mean of per-observation scores.\n")
cat("  These will be used in 61_sandwich_variance.R with cluster-robust J.\n\n")

## Extract score_ext: N x P matrix at each draw
## draws format: draws_matrix is (n_draws) x (N*P) with column naming score_ext[i,p]
cat("  Extracting score_ext ...\n")
score_ext_draws <- fit_m3bw$draws("score_ext", format = "draws_matrix")
n_draws_total <- nrow(score_ext_draws)
cat(sprintf("    score_ext draws: %d draws x %d columns (N*P = %d*%d)\n",
            n_draws_total, ncol(score_ext_draws), N, P))

## Posterior mean: average across draws -> vector of length N*P
score_ext_post_mean <- colMeans(score_ext_draws)
## Reshape to N x P matrix
score_ext_mat <- matrix(score_ext_post_mean, nrow = N, ncol = P, byrow = FALSE)

cat("  Extracting score_int ...\n")
score_int_draws <- fit_m3bw$draws("score_int", format = "draws_matrix")
score_int_post_mean <- colMeans(score_int_draws)
score_int_mat <- matrix(score_int_post_mean, nrow = N, ncol = P, byrow = FALSE)

cat("  Extracting score_kappa ...\n")
score_kappa_draws <- fit_m3bw$draws("score_kappa", format = "draws_matrix")
score_kappa_post_mean <- as.numeric(colMeans(score_kappa_draws))

cat(sprintf("    score_ext  : N x P = %d x %d\n", nrow(score_ext_mat), ncol(score_ext_mat)))
cat(sprintf("    score_int  : N x P = %d x %d\n", nrow(score_int_mat), ncol(score_int_mat)))
cat(sprintf("    score_kappa: length %d\n", length(score_kappa_post_mean)))

## Summary statistics of scores
cat("\n  Score summary statistics (posterior means):\n")
cat(sprintf("    score_ext  -- mean: %.6f, sd: %.6f, range: [%.4f, %.4f]\n",
            mean(score_ext_mat), sd(score_ext_mat),
            min(score_ext_mat), max(score_ext_mat)))
cat(sprintf("    score_int  -- mean: %.6f, sd: %.6f, range: [%.4f, %.4f]\n",
            mean(score_int_mat), sd(score_int_mat),
            min(score_int_mat), max(score_int_mat)))
cat(sprintf("    score_kappa -- mean: %.6f, sd: %.6f, range: [%.4f, %.4f]\n",
            mean(score_kappa_post_mean), sd(score_kappa_post_mean),
            min(score_kappa_post_mean), max(score_kappa_post_mean)))

## Score means should be approximately zero at the pseudo-MLE
## (not exactly zero because we have priors and random effects)
cat(sprintf("\n  Mean score (should be near zero if priors weak):\n"))
cat(sprintf("    mean(score_ext)  per covariate: [%s]\n",
            paste(sprintf("%+.6f", colMeans(score_ext_mat)), collapse = ", ")))
cat(sprintf("    mean(score_int)  per covariate: [%s]\n",
            paste(sprintf("%+.6f", colMeans(score_int_mat)), collapse = ", ")))
cat(sprintf("    mean(score_kappa): %+.6f\n", mean(score_kappa_post_mean)))

## Save scores alongside survey design variables for cluster-robust J
cat("\n  Saving score matrices + survey design info ...\n")

scores_list <- list(
  ## Posterior mean score matrices
  score_ext   = score_ext_mat,       # N x P
  score_int   = score_int_mat,       # N x P
  score_kappa = score_kappa_post_mean,  # length N

  ## Normalized survey weights
  w_tilde = w_tilde,                 # length N

  ## Survey design variables for cluster aggregation
  stratum_idx = full_stan_data$stratum_idx,   # length N
  psu_idx     = full_stan_data$psu_idx,       # length N

  ## Dimensions
  N = N, P = P, S = S, Q = Q, K = K,

  ## Parameter names (for constructing the full score vector)
  ## Full fixed-effect score: s_i = [score_ext_i (P), score_int_i (P), score_kappa_i (1)]
  ## Total length: 2P + 1 = 11
  param_names = c(paste0("alpha[", 1:P, "]"),
                  paste0("beta[", 1:P, "]"),
                  "log_kappa"),
  n_fixed = 2 * P + 1,

  ## Metadata
  description = paste0(
    "Posterior mean scores from M3b-W fit. ",
    "Use in 61_sandwich_variance.R for cluster-robust meat matrix J. ",
    "Score vectors are UNWEIGHTED: s_i = d log f(y_i|theta) / d theta. ",
    "Weight multiplication happens during J construction."
  ),
  timestamp = Sys.time()
)

saveRDS(scores_list, SCORES_OUT)
cat(sprintf("  Saved: %s\n", SCORES_OUT))
cat(sprintf("    File size: %.1f KB\n", file.info(SCORES_OUT)$size / 1024))
cat("  [PASS] Score matrices extracted and saved.\n\n")


###############################################################################
## SECTION 8 : LOO-CV (USING UNWEIGHTED LOG_LIK)
###############################################################################
cat("--- 8. LOO-CV computation (using unweighted log_lik) ---\n")
cat("  Note: LOO uses unweighted pointwise log-lik for comparable model selection.\n\n")

## Extract log_lik array
log_lik <- fit_m3bw$draws("log_lik", format = "matrix")
cat(sprintf("  log_lik dimensions: %d draws x %d observations\n",
            nrow(log_lik), ncol(log_lik)))

## Compute LOO
loo_m3bw <- tryCatch(
  loo(log_lik, cores = 4),
  error = function(e) {
    warning("LOO computation failed: ", conditionMessage(e))
    NULL
  }
)

if (!is.null(loo_m3bw)) {
  cat("\n  LOO-CV summary (M3b-W):\n")
  print(loo_m3bw)

  ## Check Pareto k diagnostics
  k_values <- loo_m3bw$diagnostics$pareto_k
  n_bad_k <- sum(k_values > 0.7)
  cat(sprintf("\n  Pareto k > 0.7: %d / %d observations (%.1f%%)\n",
              n_bad_k, length(k_values), 100 * n_bad_k / length(k_values)))

  if (n_bad_k > 0) {
    cat(sprintf("  [NOTE] %d observations with problematic Pareto k values.\n", n_bad_k))
  } else {
    cat("  [PASS] All Pareto k values < 0.7.\n")
  }

  ## Compare with M3b unweighted LOO
  if (file.exists(RESULTS_M3B_PATH)) {
    results_m3b <- readRDS(RESULTS_M3B_PATH)
    loo_m3b <- results_m3b$loo

    if (!is.null(loo_m3b)) {
      cat("\n  === LOO-CV COMPARISON: M3b-W (weighted) vs M3b (unweighted) ===\n")
      cat("  (Caveat: these LOOs are not directly comparable in the usual sense;\n")
      cat("   the weighted model targets a different pseudo-posterior.\n")
      cat("   Comparison is informative, not a formal model selection test.)\n\n")

      elpd_m3b_uw <- loo_m3b$estimates["elpd_loo", "Estimate"]
      elpd_m3b_w  <- loo_m3bw$estimates["elpd_loo", "Estimate"]
      cat(sprintf("    M3b (unweighted) ELPD_loo = %.1f\n", elpd_m3b_uw))
      cat(sprintf("    M3b-W (weighted) ELPD_loo = %.1f\n", elpd_m3b_w))
      cat(sprintf("    Difference                = %.1f (positive = M3b-W better)\n",
                  elpd_m3b_w - elpd_m3b_uw))

      ## Formal loo_compare
      comp_weighted <- loo_compare(loo_m3b, loo_m3bw)
      cat("\n  loo_compare output (M3b-W vs M3b):\n")
      print(comp_weighted)
    }
  }
} else {
  cat("  [WARN] LOO-CV computation failed. Skipping.\n")
}

cat("\n")


###############################################################################
## SECTION 9 : GAMMA ANALYSIS UNDER WEIGHTING
###############################################################################
cat("--- 9. Policy moderator analysis (Gamma) under survey weighting ---\n")
cat("  Checking which policy moderators remain significant under the pseudo-posterior.\n\n")

v_col_names <- c("intercept", "MR_pctile_std", "TieredReim", "ITaddon")
gamma_row_labels <- c(paste0("ext_", cov_names), paste0("int_", cov_names))

## Build Gamma summary table
gamma_table_w <- data.frame(
  row_idx     = integer(K * Q),
  col_idx     = integer(K * Q),
  row_label   = character(K * Q),
  col_label   = character(K * Q),
  margin      = character(K * Q),
  covariate   = character(K * Q),
  policy      = character(K * Q),
  post_mean   = numeric(K * Q),
  post_sd     = numeric(K * Q),
  q025        = numeric(K * Q),
  q975        = numeric(K * Q),
  prob_pos    = numeric(K * Q),
  stringsAsFactors = FALSE
)

idx <- 0
for (k in 1:K) {
  for (q in 1:Q) {
    idx <- idx + 1
    var_name <- sprintf("Gamma[%d,%d]", k, q)
    draws_kq <- as.numeric(fit_m3bw$draws(var_name, format = "matrix"))

    margin <- ifelse(k <= P, "extensive", "intensive")
    cov_idx <- ifelse(k <= P, k, k - P)

    gamma_table_w$row_idx[idx]   <- k
    gamma_table_w$col_idx[idx]   <- q
    gamma_table_w$row_label[idx] <- gamma_row_labels[k]
    gamma_table_w$col_label[idx] <- v_col_names[q]
    gamma_table_w$margin[idx]    <- margin
    gamma_table_w$covariate[idx] <- cov_names[cov_idx]
    gamma_table_w$policy[idx]    <- v_col_names[q]
    gamma_table_w$post_mean[idx] <- mean(draws_kq)
    gamma_table_w$post_sd[idx]   <- sd(draws_kq)
    gamma_table_w$q025[idx]      <- quantile(draws_kq, 0.025)
    gamma_table_w$q975[idx]      <- quantile(draws_kq, 0.975)
    gamma_table_w$prob_pos[idx]  <- mean(draws_kq > 0)
  }
}

## Print formatted Gamma table
cat("  === GAMMA MATRIX (WEIGHTED): POSTERIOR SUMMARIES ===\n\n")
cat(sprintf("  %-22s %-15s %10s %10s %10s %10s %10s\n",
            "Parameter", "Description", "PostMean", "PostSD",
            "Q2.5", "Q97.5", "P(>0)"))
cat(sprintf("  %s\n", paste(rep("-", 95), collapse = "")))

for (i in 1:nrow(gamma_table_w)) {
  g <- gamma_table_w[i, ]
  param_name <- sprintf("Gamma[%d,%d]", g$row_idx, g$col_idx)
  desc <- sprintf("%s.%s", g$row_label, g$col_label)

  excludes_zero <- (g$q025 > 0) | (g$q975 < 0)
  sig_mark <- ifelse(excludes_zero, " *", "  ")

  cat(sprintf("  %-22s %-15s %+10.4f %10.4f %+10.4f %+10.4f %10.3f%s\n",
              param_name, desc,
              g$post_mean, g$post_sd, g$q025, g$q975, g$prob_pos, sig_mark))
}

n_sig_w <- sum((gamma_table_w$q025 > 0) | (gamma_table_w$q975 < 0))
cat(sprintf("\n  Significant Gamma elements (95%% CI excludes 0): %d / %d\n",
            n_sig_w, nrow(gamma_table_w)))

## Gamma as K x Q matrix (posterior means)
gamma_mean_mat_w <- matrix(NA, K, Q)
for (k in 1:K) {
  for (q in 1:Q) {
    row_filter <- gamma_table_w$row_idx == k & gamma_table_w$col_idx == q
    gamma_mean_mat_w[k, q] <- gamma_table_w$post_mean[row_filter]
  }
}

cat("\n  Gamma matrix (weighted posterior means, K x Q):\n")
cat(sprintf("    %15s", ""))
cat(sprintf("%12s", v_col_names), "\n")
for (k in 1:K) {
  cat(sprintf("    %-15s", gamma_row_labels[k]))
  cat(sprintf("%+12.4f", gamma_mean_mat_w[k, ]))
  cat("\n")
}

## Compare with unweighted Gamma if available
if (file.exists(RESULTS_M3B_PATH)) {
  results_m3b <- readRDS(RESULTS_M3B_PATH)
  if (!is.null(results_m3b$gamma_table)) {
    cat("\n  === GAMMA SIGNIFICANCE COMPARISON: UNWEIGHTED vs WEIGHTED ===\n")
    cat(sprintf("  %-22s %12s %12s %12s %12s\n",
                "Parameter", "UW_mean", "W_mean", "UW_sig?", "W_sig?"))
    cat(sprintf("  %s\n", paste(rep("-", 75), collapse = "")))

    gt_uw <- results_m3b$gamma_table
    for (i in 1:nrow(gamma_table_w)) {
      gw <- gamma_table_w[i, ]
      gu <- gt_uw[gt_uw$row_idx == gw$row_idx & gt_uw$col_idx == gw$col_idx, ]

      if (nrow(gu) == 1) {
        sig_uw <- ifelse((gu$q025 > 0) | (gu$q975 < 0), "YES", "no")
        sig_w  <- ifelse((gw$q025 > 0) | (gw$q975 < 0), "YES", "no")
        param_name <- sprintf("Gamma[%d,%d]", gw$row_idx, gw$col_idx)

        cat(sprintf("  %-22s %+12.4f %+12.4f %12s %12s\n",
                    param_name, gu$post_mean, gw$post_mean, sig_uw, sig_w))
      }
    }
  }
}

cat("\n")


###############################################################################
## SECTION 10 : SAVE RESULTS
###############################################################################
cat("--- 10. Saving results ---\n")

## 10a. Save the CmdStanR fit object
fit_m3bw$save_object(FIT_OUT)
cat(sprintf("  Saved fit object: %s\n", FIT_OUT))
cat(sprintf("    File size: %.1f MB\n",
            file.info(FIT_OUT)$size / 1024^2))

## 10b. Assemble and save results list
results_m3bw <- list(
  ## Model info
  model      = "M3b-W",
  model_desc = "Survey-Weighted Policy Moderator SVC Hurdle Beta-Binomial",

  ## Timing
  fit_time_mins = as.numeric(fit_time),

  ## Parameter summary (fixed effects)
  param_summary = param_summary_fixed,

  ## Hierarchical parameter summary
  hier_summary = param_summary_hier,

  ## Gamma parameter summary
  gamma_summary = param_summary_gamma,

  ## Key estimates
  alpha_means    = setNames(alpha_means_w, cov_names),
  beta_means     = setNames(beta_means_w, cov_names),
  log_kappa_mean = log_kappa_mean_w,
  kappa_mean     = kappa_mean_w,
  tau_means      = setNames(tau_means_w, tau_labels),

  ## Gamma matrix (posterior means)
  gamma_mean_mat = gamma_mean_mat_w,

  ## Gamma detailed table
  gamma_table = gamma_table_w,

  ## Weight info
  weight_info = list(
    w_mean     = w_mean,
    w_sd       = w_sd,
    w_min      = w_min,
    w_max      = w_max,
    w_cv       = w_cv,
    ess_kish   = ess_kish,
    deff_kish  = deff_kish,
    sum_w_tilde = sum(w_tilde)
  ),

  ## Score summary (for reference; full scores in scores_m3b_weighted.rds)
  score_summary = list(
    score_ext_col_means  = colMeans(score_ext_mat),
    score_int_col_means  = colMeans(score_int_mat),
    score_kappa_mean     = mean(score_kappa_post_mean),
    score_ext_col_sds    = apply(score_ext_mat, 2, sd),
    score_int_col_sds    = apply(score_int_mat, 2, sd),
    score_kappa_sd       = sd(score_kappa_post_mean)
  ),

  ## Diagnostics
  diagnostics = list(
    n_divergent      = total_divergent,
    n_max_treedepth  = total_max_td,
    max_rhat         = max_rhat,
    max_rhat_delta   = max_rhat_delta,
    min_ess_bulk     = min_ess_bulk,
    min_ess_tail     = min_ess_tail,
    min_ess_delta    = min_ess_delta,
    all_pass         = diag_pass
  ),

  ## LOO-CV
  loo = loo_m3bw,

  ## Timestamp
  timestamp = Sys.time()
)

saveRDS(results_m3bw, RESULTS_OUT)
cat(sprintf("  Saved results: %s\n", RESULTS_OUT))
cat(sprintf("    File size: %.1f KB\n",
            file.info(RESULTS_OUT)$size / 1024))


###############################################################################
## SECTION 11 : FINAL SUMMARY
###############################################################################
cat("\n==============================================================\n")
cat("  M3b-W FITTING SUMMARY\n")
cat("==============================================================\n")

n_fixed  <- P * 2 + 1                    # alpha + beta + log_kappa
n_hier   <- K + K * (K - 1) / 2          # tau[K] + L_Omega free elements
n_gamma  <- K * Q                         # Gamma[K, Q]
n_re     <- K * S                         # delta[S, K]
n_params_total <- n_fixed + n_hier + n_gamma + n_re

cat(sprintf("\n  Model: Survey-Weighted Policy Moderator SVC HBB (M3b-W)\n"))
cat(sprintf("  N = %d, P = %d, S = %d, Q = %d, K = 2P = %d\n", N, P, S, Q, K))
cat(sprintf("  Parameters: %d fixed + %d hier + %d Gamma + %d RE = %d total\n",
            n_fixed, n_hier, n_gamma, n_re, n_params_total))
cat(sprintf("  Fit time: %.1f minutes\n", as.numeric(fit_time)))

cat(sprintf("\n  Survey Weights:\n"))
cat(sprintf("    Kish ESS  = %.0f (of N=%d)\n", ess_kish, N))
cat(sprintf("    DEFF_kish = %.2f\n", deff_kish))
cat(sprintf("    Weight CV = %.4f, range = [%.4f, %.4f]\n", w_cv, w_min, w_max))

cat(sprintf("\n  MCMC Diagnostics:\n"))
cat(sprintf("    Divergent transitions  : %d %s\n",
            total_divergent, ifelse(total_divergent == 0, "[PASS]", "[WARN]")))
cat(sprintf("    Max R-hat (fixed+Gamma): %.4f %s\n",
            max_rhat, ifelse(max_rhat < 1.01, "[PASS]", "[WARN]")))
cat(sprintf("    Max R-hat (deltas)     : %.4f %s\n",
            max_rhat_delta, ifelse(max_rhat_delta < 1.01, "[PASS]", "[WARN]")))
cat(sprintf("    Min ESS (bulk)         : %.0f %s\n",
            min_ess_bulk, ifelse(min_ess_bulk > 400, "[PASS]", "[WARN]")))
cat(sprintf("    Min ESS (tail)         : %.0f %s\n",
            min_ess_tail, ifelse(min_ess_tail > 400, "[PASS]", "[WARN]")))

cat(sprintf("\n  Key Results (Weighted Pseudo-Posterior):\n"))
cat(sprintf("    alpha_poverty = %+.4f %s\n",
            alpha_poverty_w, ifelse(alpha_poverty_w < 0, "[PASS] (<0)", "[WARN]")))
cat(sprintf("    beta_poverty  = %+.4f %s\n",
            beta_poverty_w, ifelse(beta_poverty_w > 0, "[PASS] (>0)", "[WARN]")))
cat(sprintf("    kappa         = %.2f %s\n",
            kappa_mean_w, ifelse(kappa_mean_w > 3 & kappa_mean_w < 30, "[PASS]", "[WARN]")))

## Weighted vs unweighted comparison in summary
if (file.exists(RESULTS_M3B_PATH)) {
  results_m3b <- readRDS(RESULTS_M3B_PATH)
  alpha_means_uw <- as.numeric(results_m3b$alpha_means)
  beta_means_uw  <- as.numeric(results_m3b$beta_means)

  cat(sprintf("\n  Weighted vs Unweighted Shifts:\n"))
  cat(sprintf("    %-15s %10s %10s %10s\n", "Parameter", "Unweighted", "Weighted", "Shift"))
  cat(sprintf("    %s\n", paste(rep("-", 48), collapse = "")))
  for (k in 1:P) {
    cat(sprintf("    alpha[%d] %-7s %+10.4f %+10.4f %+10.4f\n",
                k, cov_names[k], alpha_means_uw[k], alpha_means_w[k],
                alpha_means_w[k] - alpha_means_uw[k]))
  }
  for (k in 1:P) {
    cat(sprintf("    beta[%d]  %-7s %+10.4f %+10.4f %+10.4f\n",
                k, cov_names[k], beta_means_uw[k], beta_means_w[k],
                beta_means_w[k] - beta_means_uw[k]))
  }
}

cat(sprintf("\n  Gamma (policy moderators, significant at 95%%):\n"))
for (i in 1:nrow(gamma_table_w)) {
  g <- gamma_table_w[i, ]
  excludes_zero <- (g$q025 > 0) | (g$q975 < 0)
  if (excludes_zero) {
    cat(sprintf("    Gamma[%d,%d] %-22s = %+.4f  95%% CI [%+.4f, %+.4f] *\n",
                g$row_idx, g$col_idx,
                sprintf("%s.%s", g$row_label, g$col_label),
                g$post_mean, g$q025, g$q975))
  }
}
if (n_sig_w == 0) {
  cat("    (none significant at 95%% level)\n")
}

if (!is.null(loo_m3bw)) {
  cat(sprintf("\n  LOO-CV:\n"))
  cat(sprintf("    ELPD_loo = %.1f (SE = %.1f)\n",
              loo_m3bw$estimates["elpd_loo", "Estimate"],
              loo_m3bw$estimates["elpd_loo", "SE"]))
  cat(sprintf("    p_loo    = %.1f\n",
              loo_m3bw$estimates["p_loo", "Estimate"]))
  cat(sprintf("    Pareto k > 0.7: %d / %d\n",
              sum(loo_m3bw$diagnostics$pareto_k > 0.7),
              length(loo_m3bw$diagnostics$pareto_k)))
}

cat(sprintf("\n  Score matrices saved for sandwich variance:\n"))
cat(sprintf("    %s\n", SCORES_OUT))
cat(sprintf("    score_ext : %d x %d\n", N, P))
cat(sprintf("    score_int : %d x %d\n", N, P))
cat(sprintf("    score_kappa : %d x 1\n", N))
cat(sprintf("    + w_tilde, stratum_idx, psu_idx for cluster-robust J\n"))

cat(sprintf("\n  Output files:\n"))
cat(sprintf("    %s\n", FIT_OUT))
cat(sprintf("    %s\n", RESULTS_OUT))
cat(sprintf("    %s\n", SCORES_OUT))

cat("\n==============================================================\n")
cat("  M3b-W FITTING COMPLETE.\n")
if (diag_pass) {
  cat("  ALL DIAGNOSTICS PASSED.\n")
} else {
  cat("  SOME DIAGNOSTICS FAILED. Review before proceeding.\n")
}
cat("  Next: 61_sandwich_variance.R (cluster-robust sandwich correction)\n")
cat("==============================================================\n")
