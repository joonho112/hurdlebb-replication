## =============================================================================
## 30_fit_m2.R -- Fit Block-Diagonal SVC Hurdle Beta-Binomial Model (M2)
## =============================================================================
## Purpose : Compile and fit the Block-Diagonal SVC HBB model (M2)
##           on NSECE 2019 data (unweighted), run diagnostics, LOO-CV
##           comparison with M1/M0, PPC, and state-level poverty reversal.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Track   : A (requires NSECE restricted-use data; see data/DATA_ACCESS.md)
## Inputs  : data/precomputed/stan_data.rds
##           stan/hbb_m2.stan
##           code/helpers/utils.R
##           data/precomputed/results_m1.rds
##           data/precomputed/results_m0.rds
## Outputs : data/precomputed/fit_m2.rds
##           data/precomputed/results_m2.rds
## =============================================================================

cat("==============================================================\n")
cat("  HBB Replication: M2 Fitting  (Phase 3)\n")
cat("  Block-Diagonal SVC Hurdle Beta-Binomial\n")
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
STAN_DATA_PATH  <- file.path(PROJECT_ROOT, "data/precomputed/stan_data.rds")
STAN_MODEL_PATH <- file.path(PROJECT_ROOT, "stan/hbb_m2.stan")
OUTPUT_DIR      <- file.path(PROJECT_ROOT, "data/precomputed")
FIT_OUT         <- file.path(OUTPUT_DIR, "fit_m2.rds")
RESULTS_OUT     <- file.path(OUTPUT_DIR, "results_m2.rds")
RESULTS_M1_PATH <- file.path(OUTPUT_DIR, "results_m1.rds")
RESULTS_M0_PATH <- file.path(OUTPUT_DIR, "results_m0.rds")

## Ensure output directory exists
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)


###############################################################################
## SECTION 1 : LOAD DATA
###############################################################################
cat("--- 1. Loading Stan data ---\n")

stopifnot("Stan data file not found" = file.exists(STAN_DATA_PATH))
full_stan_data <- readRDS(STAN_DATA_PATH)
cat(sprintf("  Loaded: %s\n", STAN_DATA_PATH))
cat(sprintf("  N = %d, P = %d, S = %d, N_pos = %d\n",
            full_stan_data$N, full_stan_data$P, full_stan_data$S,
            full_stan_data$N_pos))

## Build M2-specific data list: N, P, S, y, n_trial, z, X, state
## M2 uses the same data structure as M1 (random effects handled in Stan)
stan_data_m2 <- list(
  N       = full_stan_data$N,
  P       = full_stan_data$P,
  S       = full_stan_data$S,
  y       = full_stan_data$y,
  n_trial = full_stan_data$n_trial,
  z       = full_stan_data$z,
  X       = full_stan_data$X,
  state   = full_stan_data$state
)

cat(sprintf("  M2 data list: N=%d, P=%d, S=%d\n",
            stan_data_m2$N, stan_data_m2$P, stan_data_m2$S))
cat(sprintf("  Zero rate: %.1f%%\n", 100 * mean(stan_data_m2$z == 0)))

## State sample sizes
state_ns <- table(stan_data_m2$state)
cat(sprintf("  State sample sizes: min=%d, median=%d, max=%d\n",
            min(state_ns), median(state_ns), max(state_ns)))

## M2 random effects dimensionality
n_re_per_state <- 2 * stan_data_m2$P   # P for ext + P for int
n_re_total     <- n_re_per_state * stan_data_m2$S
cat(sprintf("  Random effects: %d per state (%d ext + %d int) x %d states = %d total\n",
            n_re_per_state, stan_data_m2$P, stan_data_m2$P,
            stan_data_m2$S, n_re_total))
cat("  [PASS] Data loaded and M2 data list constructed.\n\n")


###############################################################################
## SECTION 2 : COMPILE STAN MODEL
###############################################################################
cat("--- 2. Compiling Stan model ---\n")

stopifnot("Stan model file not found" = file.exists(STAN_MODEL_PATH))
cat(sprintf("  Model file: %s\n", STAN_MODEL_PATH))

## Compile (cmdstanr caches compiled models; recompiles only if .stan changed)
m2_model <- tryCatch(
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
## SECTION 3 : INITIALISATION FROM M1 POSTERIORS (WARM START)
###############################################################################
cat("--- 3. Setting initial values ---\n")

P <- stan_data_m2$P
S <- stan_data_m2$S

if (file.exists(RESULTS_M1_PATH)) {
  results_m1 <- readRDS(RESULTS_M1_PATH)
  cat("  [INFO] M1 results found. Using M1 posteriors for warm start.\n")

  ## Extract M1 posterior means
  m1_alpha     <- as.numeric(results_m1$alpha_means)
  m1_beta      <- as.numeric(results_m1$beta_means)
  m1_log_kappa <- as.numeric(results_m1$log_kappa_mean)
  m1_tau       <- as.numeric(results_m1$tau_means)   # length 2: (ext, int)

  ## For M2: tau_ext[1:P], tau_int[1:P]
  ##   - intercept scale from M1; other covariates start small (0.1)
  init_tau_ext <- c(m1_tau[1], rep(0.1, P - 1))
  init_tau_int <- c(m1_tau[2], rep(0.1, P - 1))

  init_fun <- function() {
    list(
      alpha     = m1_alpha,
      beta      = m1_beta,
      log_kappa = m1_log_kappa,
      tau_ext   = init_tau_ext,
      tau_int   = init_tau_int,
      L_ext     = diag(P),
      L_int     = diag(P),
      z_ext     = rep(list(rep(0, P)), S),
      z_int     = rep(list(rep(0, P)), S)
    )
  }

  cat(sprintf("  alpha init    : [%s]\n",
              paste(sprintf("%+.3f", m1_alpha), collapse = ", ")))
  cat(sprintf("  beta init     : [%s]\n",
              paste(sprintf("%+.3f", m1_beta), collapse = ", ")))
  cat(sprintf("  log_kappa init: %.3f\n", m1_log_kappa))
  cat(sprintf("  tau_ext init  : [%s]\n",
              paste(sprintf("%.3f", init_tau_ext), collapse = ", ")))
  cat(sprintf("  tau_int init  : [%s]\n",
              paste(sprintf("%.3f", init_tau_int), collapse = ", ")))
  cat("  L_ext, L_int  : identity matrices (P x P)\n")
  cat("  z_ext, z_int  : matrices of zeros (P x S)\n")
} else {
  cat("  [INFO] M1 results not found. Using default initialisation.\n")

  init_fun <- function() {
    list(
      alpha     = rep(0, P),
      beta      = rep(0, P),
      log_kappa = log(10),
      tau_ext   = rep(0.3, P),
      tau_int   = rep(0.3, P),
      L_ext     = diag(P),
      L_int     = diag(P),
      z_ext     = rep(list(rep(0, P)), S),
      z_int     = rep(list(rep(0, P)), S)
    )
  }
}

cat("  [PASS] Initial values set.\n\n")


###############################################################################
## SECTION 4 : FIT M2
###############################################################################
cat("--- 4. Fitting M2 ---\n")
cat("  chains = 4, parallel_chains = 4\n")
cat("  iter_warmup = 1000, iter_sampling = 1000\n")
cat("  adapt_delta = 0.95, max_treedepth = 12\n")
cat("  seed = 20250220\n\n")

fit_start <- Sys.time()

fit_m2 <- tryCatch(
  m2_model$sample(
    data            = stan_data_m2,
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
n_divergent <- fit_m2$diagnostic_summary()$num_divergent
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
n_max_treedepth <- fit_m2$diagnostic_summary()$num_max_treedepth
total_max_td <- sum(n_max_treedepth)
cat(sprintf("  Max treedepth hits: %d (per chain: %s)\n",
            total_max_td, paste(n_max_treedepth, collapse = ", ")))
if (total_max_td > 0) {
  warning("  [WARN] Max treedepth reached. Consider increasing max_treedepth.")
}

## 5c. Parameter summary — fixed effects + hierarchical parameters
## Fixed effects
param_names_fixed <- c(paste0("alpha[", 1:P, "]"),
                       paste0("beta[", 1:P, "]"),
                       "log_kappa")

## Hierarchical parameters: tau_ext[1:P], tau_int[1:P]
param_names_tau_ext <- paste0("tau_ext[", 1:P, "]")
param_names_tau_int <- paste0("tau_int[", 1:P, "]")
param_names_hier <- c(param_names_tau_ext, param_names_tau_int)

## Selected delta values (a few states for monitoring)
delta_sample_states <- c(1, 10, 25, 40, 51)
param_names_delta_sample <- c()
for (s in delta_sample_states) {
  for (p in 1:P) {
    param_names_delta_sample <- c(param_names_delta_sample,
                                   sprintf("delta_ext[%d,%d]", s, p),
                                   sprintf("delta_int[%d,%d]", s, p))
  }
}

## Full summary for diagnostics (fixed + hierarchical)
param_names_all <- c(param_names_fixed, param_names_hier)
param_summary <- fit_m2$summary(variables = param_names_all)

cat("\n  Fixed effect parameter summary:\n")
param_summary_fixed <- fit_m2$summary(variables = param_names_fixed)
print(param_summary_fixed, n = nrow(param_summary_fixed))

cat("\n  Hierarchical parameter summary (tau_ext, tau_int):\n")
param_summary_hier <- fit_m2$summary(variables = param_names_hier)
print(param_summary_hier, n = nrow(param_summary_hier))

cat("\n  Selected delta (state random coefficients) summary:\n")
param_summary_delta_sample <- fit_m2$summary(variables = param_names_delta_sample)
print(param_summary_delta_sample, n = min(nrow(param_summary_delta_sample), 20))

## 5d. R-hat check (on fixed + hierarchical parameters)
max_rhat <- max(param_summary$rhat, na.rm = TRUE)
cat(sprintf("\n  Max R-hat (fixed+hier): %.4f (threshold: 1.01)\n", max_rhat))
if (max_rhat > 1.01) {
  warning("  [WARN] R-hat > 1.01 detected. Chains may not have converged.")
  diag_pass <- FALSE
} else {
  cat("  [PASS] All R-hat < 1.01.\n")
}

## Also check R-hat on all delta parameters
delta_ext_all_names <- c()
delta_int_all_names <- c()
for (s in 1:S) {
  for (p in 1:P) {
    delta_ext_all_names <- c(delta_ext_all_names, sprintf("delta_ext[%d,%d]", s, p))
    delta_int_all_names <- c(delta_int_all_names, sprintf("delta_int[%d,%d]", s, p))
  }
}
delta_all_names <- c(delta_ext_all_names, delta_int_all_names)
delta_all_summary <- fit_m2$summary(variables = delta_all_names)
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

## 5e. ESS check (bulk and tail) on fixed + hierarchical
min_ess_bulk <- min(param_summary$ess_bulk, na.rm = TRUE)
min_ess_tail <- min(param_summary$ess_tail, na.rm = TRUE)
cat(sprintf("  Min ESS (bulk, fixed+hier): %.0f (threshold: 400)\n", min_ess_bulk))
cat(sprintf("  Min ESS (tail, fixed+hier): %.0f (threshold: 400)\n", min_ess_tail))

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
## SECTION 6 : KEY PARAMETER CHECKS
###############################################################################
cat("--- 6. Key parameter checks ---\n")

## Extract posterior means for fixed effects
alpha_means <- param_summary_fixed$mean[grepl("^alpha", param_summary_fixed$variable)]
beta_means  <- param_summary_fixed$mean[grepl("^beta",  param_summary_fixed$variable)]
log_kappa_mean <- param_summary_fixed$mean[param_summary_fixed$variable == "log_kappa"]
kappa_mean  <- exp(log_kappa_mean)

## Hierarchical parameter means
tau_ext_means <- param_summary_hier$mean[grepl("^tau_ext", param_summary_hier$variable)]
tau_int_means <- param_summary_hier$mean[grepl("^tau_int", param_summary_hier$variable)]

## Covariate names for display
cov_names <- c("intercept", "poverty", "urban", "black", "hispanic")

cat("\n  Extensive margin (alpha): P(serve IT)\n")
for (k in seq_along(alpha_means)) {
  cat(sprintf("    alpha[%d] %-10s = %+.4f\n", k, cov_names[k], alpha_means[k]))
}

cat("\n  Intensive margin (beta): IT share | serve IT\n")
for (k in seq_along(beta_means)) {
  cat(sprintf("    beta[%d]  %-10s = %+.4f\n", k, cov_names[k], beta_means[k]))
}

cat(sprintf("\n  Dispersion:\n"))
cat(sprintf("    log_kappa = %.4f\n", log_kappa_mean))
cat(sprintf("    kappa     = %.2f\n", kappa_mean))

cat(sprintf("\n  Hierarchical scale parameters (tau_ext):\n"))
for (k in seq_along(tau_ext_means)) {
  cat(sprintf("    tau_ext[%d] %-10s = %.4f\n", k, cov_names[k], tau_ext_means[k]))
}

cat(sprintf("\n  Hierarchical scale parameters (tau_int):\n"))
for (k in seq_along(tau_int_means)) {
  cat(sprintf("    tau_int[%d] %-10s = %.4f\n", k, cov_names[k], tau_int_means[k]))
}

## 6a. Extract correlation matrices
## Try Omega_ext first (generated quantities)
omega_ext_available <- tryCatch({
  fit_m2$draws("Omega_ext[1,2]", format = "matrix")
  TRUE
}, error = function(e) FALSE)

if (omega_ext_available) {
  cat("\n  Omega_ext (extensive margin correlation matrix) posterior means:\n")
  omega_ext_mat <- matrix(NA, P, P)
  for (i in 1:P) {
    for (j in 1:P) {
      draws_ij <- fit_m2$draws(sprintf("Omega_ext[%d,%d]", i, j), format = "matrix")
      omega_ext_mat[i, j] <- mean(draws_ij)
    }
  }
  cat("    ")
  cat(sprintf("%10s", cov_names), "\n")
  for (i in 1:P) {
    cat(sprintf("    %-10s", cov_names[i]))
    cat(sprintf("%10.3f", omega_ext_mat[i, ]))
    cat("\n")
  }

  cat("\n  Omega_int (intensive margin correlation matrix) posterior means:\n")
  omega_int_mat <- matrix(NA, P, P)
  for (i in 1:P) {
    for (j in 1:P) {
      draws_ij <- fit_m2$draws(sprintf("Omega_int[%d,%d]", i, j), format = "matrix")
      omega_int_mat[i, j] <- mean(draws_ij)
    }
  }
  cat("    ")
  cat(sprintf("%10s", cov_names), "\n")
  for (i in 1:P) {
    cat(sprintf("    %-10s", cov_names[i]))
    cat(sprintf("%10.3f", omega_int_mat[i, ]))
    cat("\n")
  }
} else {
  cat("\n  [NOTE] Omega matrices not available in generated quantities.\n")
  cat("         Correlation matrices will be computed from L_ext/L_int draws.\n")
  omega_ext_mat <- NULL
  omega_int_mat <- NULL
}

## 6b. Poverty reversal check (population-level)
alpha_poverty <- alpha_means[2]
beta_poverty  <- beta_means[2]

cat("\n  === POVERTY REVERSAL CHECK (POPULATION-LEVEL) ===\n")
cat(sprintf("    alpha_poverty = %+.4f (expected: negative)\n", alpha_poverty))
cat(sprintf("    beta_poverty  = %+.4f (expected: positive)\n", beta_poverty))

if (alpha_poverty < 0) {
  cat("    [PASS] alpha_poverty < 0 (higher poverty -> less likely to serve IT)\n")
} else {
  warning("    [WARN] alpha_poverty >= 0 -- unexpected sign!")
}

if (beta_poverty > 0) {
  cat("    [PASS] beta_poverty > 0 (higher poverty -> higher IT share)\n")
} else {
  warning("    [WARN] beta_poverty <= 0 -- unexpected sign!")
}

## 6c. State variation check: tau values
cat("\n  === STATE VARIATION CHECK ===\n")
cat("  Extensive margin tau:\n")
for (k in seq_along(tau_ext_means)) {
  status <- ifelse(tau_ext_means[k] > 0.01, "[PASS]", "[NOTE: near zero]")
  cat(sprintf("    tau_ext[%d] %-10s = %.4f %s\n",
              k, cov_names[k], tau_ext_means[k], status))
}
cat("  Intensive margin tau:\n")
for (k in seq_along(tau_int_means)) {
  status <- ifelse(tau_int_means[k] > 0.01, "[PASS]", "[NOTE: near zero]")
  cat(sprintf("    tau_int[%d] %-10s = %.4f %s\n",
              k, cov_names[k], tau_int_means[k], status))
}

## Compare M2 intercept tau with M1 tau (should be similar)
if (file.exists(RESULTS_M1_PATH)) {
  cat("\n  === COMPARISON WITH M1 tau ===\n")
  cat(sprintf("    M1 tau_ext (intercept) = %.4f,  M2 tau_ext[1] = %.4f\n",
              as.numeric(results_m1$tau_means[1]), tau_ext_means[1]))
  cat(sprintf("    M1 tau_int (intercept) = %.4f,  M2 tau_int[1] = %.4f\n",
              as.numeric(results_m1$tau_means[2]), tau_int_means[1]))
}

## 6d. Overdispersion check
cat(sprintf("\n  === OVERDISPERSION CHECK ===\n"))
cat(sprintf("    kappa = %.2f (expected: ~5-15 for 12x overdispersion)\n", kappa_mean))

if (kappa_mean > 3 && kappa_mean < 30) {
  cat("    [PASS] kappa in reasonable range.\n")
} else {
  warning(sprintf("    [WARN] kappa = %.2f outside expected range [3, 30].", kappa_mean))
}

cat("\n")


###############################################################################
## SECTION 7 : STATE-LEVEL POVERTY REVERSAL ANALYSIS
###############################################################################
cat("--- 7. State-level poverty reversal analysis ---\n")

## Extract posterior draws for delta_ext[s,2] and delta_int[s,2]
## (poverty coefficient random effects)
## State-specific total poverty effect:
##   alpha_poverty_s = alpha[2] + delta_ext[s, 2]
##   beta_poverty_s  = beta[2]  + delta_int[s, 2]

## Extract alpha[2] and beta[2] draws
alpha2_draws <- fit_m2$draws("alpha[2]", format = "matrix")  # n_draws x 1
beta2_draws  <- fit_m2$draws("beta[2]",  format = "matrix")  # n_draws x 1
n_draws <- nrow(alpha2_draws)

## Extract delta_ext[s,2] and delta_int[s,2] for all states
delta_ext_pov_names <- sprintf("delta_ext[%d,2]", 1:S)
delta_int_pov_names <- sprintf("delta_int[%d,2]", 1:S)
delta_ext_pov_draws <- fit_m2$draws(delta_ext_pov_names, format = "matrix")  # n_draws x S
delta_int_pov_draws <- fit_m2$draws(delta_int_pov_names, format = "matrix")  # n_draws x S

## Compute state-specific total poverty effects for each draw
## alpha_poverty_s[d, s] = alpha[2][d] + delta_ext[s,2][d]
## beta_poverty_s[d, s]  = beta[2][d]  + delta_int[s,2][d]
alpha_poverty_s <- sweep(delta_ext_pov_draws, 1, as.numeric(alpha2_draws), "+")
beta_poverty_s  <- sweep(delta_int_pov_draws, 1, as.numeric(beta2_draws), "+")

## For each state: posterior mean and P(reversal pattern)
## Reversal = alpha_poverty_s < 0 AND beta_poverty_s > 0
state_poverty_table <- data.frame(
  state        = 1:S,
  n_obs        = as.integer(state_ns),
  alpha_pov_mean = colMeans(alpha_poverty_s),
  alpha_pov_q05  = apply(alpha_poverty_s, 2, quantile, probs = 0.05),
  alpha_pov_q95  = apply(alpha_poverty_s, 2, quantile, probs = 0.95),
  beta_pov_mean  = colMeans(beta_poverty_s),
  beta_pov_q05   = apply(beta_poverty_s, 2, quantile, probs = 0.05),
  beta_pov_q95   = apply(beta_poverty_s, 2, quantile, probs = 0.95),
  prob_reversal  = colMeans(alpha_poverty_s < 0 & beta_poverty_s > 0),
  stringsAsFactors = FALSE
)

## Classify states by posterior mean
state_poverty_table$reversal_mean <- (state_poverty_table$alpha_pov_mean < 0 &
                                       state_poverty_table$beta_pov_mean > 0)
## Classify by high probability (> 0.5)
state_poverty_table$reversal_prob50 <- state_poverty_table$prob_reversal > 0.5

n_reversal_mean   <- sum(state_poverty_table$reversal_mean)
n_reversal_prob50 <- sum(state_poverty_table$reversal_prob50)

cat(sprintf("\n  States showing poverty reversal (posterior mean): %d / %d (%.0f%%)\n",
            n_reversal_mean, S, 100 * n_reversal_mean / S))
cat(sprintf("  States showing poverty reversal (P > 0.5):       %d / %d (%.0f%%)\n",
            n_reversal_prob50, S, 100 * n_reversal_prob50 / S))
cat(sprintf("  Expected from M1 analysis: ~22/51 (43%%)\n"))

## Print full state table
cat("\n  State-level poverty effects:\n\n")
cat(sprintf("  %5s %5s %10s %10s %10s %10s %10s %10s %10s %8s\n",
            "State", "N_obs",
            "a_pov_mn", "a_pov_05", "a_pov_95",
            "b_pov_mn", "b_pov_05", "b_pov_95",
            "P(rev)", "Pattern"))
cat(sprintf("  %s\n", paste(rep("-", 100), collapse = "")))

for (s in 1:S) {
  r <- state_poverty_table[s, ]
  pattern <- ifelse(r$reversal_mean, "REVERSAL", "")
  cat(sprintf("  %5d %5d %+10.4f %+10.4f %+10.4f %+10.4f %+10.4f %+10.4f %10.3f %8s\n",
              r$state, r$n_obs,
              r$alpha_pov_mean, r$alpha_pov_q05, r$alpha_pov_q95,
              r$beta_pov_mean, r$beta_pov_q05, r$beta_pov_q95,
              r$prob_reversal, pattern))
}

## Summary of patterns
cat("\n  === POVERTY REVERSAL PATTERN SUMMARY ===\n")
cat(sprintf("    Classic reversal (alpha<0, beta>0): %d states\n", n_reversal_mean))
n_both_neg <- sum(state_poverty_table$alpha_pov_mean < 0 &
                   state_poverty_table$beta_pov_mean <= 0)
n_both_pos <- sum(state_poverty_table$alpha_pov_mean >= 0 &
                   state_poverty_table$beta_pov_mean > 0)
n_neither  <- sum(state_poverty_table$alpha_pov_mean >= 0 &
                   state_poverty_table$beta_pov_mean <= 0)
cat(sprintf("    alpha<0, beta<=0 (poverty barrier both): %d states\n", n_both_neg))
cat(sprintf("    alpha>=0, beta>0 (poverty positive both): %d states\n", n_both_pos))
cat(sprintf("    alpha>=0, beta<=0 (no poverty effect):    %d states\n", n_neither))

cat("\n")


###############################################################################
## SECTION 8 : LOO-CV AND COMPARISON WITH M1/M0
###############################################################################
cat("--- 8. LOO-CV computation and M1/M0 comparison ---\n")

## Extract log_lik array
log_lik <- fit_m2$draws("log_lik", format = "matrix")
cat(sprintf("  log_lik dimensions: %d draws x %d observations\n",
            nrow(log_lik), ncol(log_lik)))

## Compute LOO
loo_m2 <- tryCatch(
  loo(log_lik, cores = 4),
  error = function(e) {
    warning("LOO computation failed: ", conditionMessage(e))
    NULL
  }
)

if (!is.null(loo_m2)) {
  cat("\n  LOO-CV summary (M2):\n")
  print(loo_m2)

  ## Check Pareto k diagnostics
  k_values <- loo_m2$diagnostics$pareto_k
  n_bad_k <- sum(k_values > 0.7)
  cat(sprintf("\n  Pareto k > 0.7: %d / %d observations (%.1f%%)\n",
              n_bad_k, length(k_values), 100 * n_bad_k / length(k_values)))

  if (n_bad_k > 0) {
    cat(sprintf("  [NOTE] %d observations with problematic Pareto k values.\n", n_bad_k))
    cat("         Consider moment matching or reloo for these observations.\n")
  } else {
    cat("  [PASS] All Pareto k values < 0.7.\n")
  }

  ## 8a. LOO comparison with M1
  if (file.exists(RESULTS_M1_PATH)) {
    results_m1 <- readRDS(RESULTS_M1_PATH)
    loo_m1 <- results_m1$loo

    if (!is.null(loo_m1)) {
      cat("\n  === LOO-CV COMPARISON: M2 vs M1 ===\n")

      elpd_m1 <- loo_m1$estimates["elpd_loo", "Estimate"]
      elpd_m2 <- loo_m2$estimates["elpd_loo", "Estimate"]
      cat(sprintf("    M1 ELPD_loo = %.1f\n", elpd_m1))
      cat(sprintf("    M2 ELPD_loo = %.1f\n", elpd_m2))
      cat(sprintf("    Difference  = %.1f (positive = M2 better)\n",
                  elpd_m2 - elpd_m1))

      ## Formal loo_compare
      comp_m2_m1 <- loo_compare(loo_m1, loo_m2)
      cat("\n  loo_compare output (M2 vs M1):\n")
      print(comp_m2_m1)

      ## Interpretation
      elpd_diff <- comp_m2_m1[2, "elpd_diff"]
      se_diff   <- comp_m2_m1[2, "se_diff"]
      cat(sprintf("\n    ELPD difference: %.1f (SE = %.1f)\n",
                  elpd_diff, se_diff))

      if (abs(elpd_diff) > 2 * se_diff) {
        better_model <- rownames(comp_m2_m1)[1]
        cat(sprintf("    [PASS] Difference > 2*SE: %s is clearly preferred.\n",
                    better_model))
      } else {
        cat("    [NOTE] Difference < 2*SE: models are comparable.\n")
      }
    }
  }

  ## 8b. LOO comparison with M0
  if (file.exists(RESULTS_M0_PATH)) {
    results_m0 <- readRDS(RESULTS_M0_PATH)
    loo_m0 <- results_m0$loo

    if (!is.null(loo_m0)) {
      cat("\n  === LOO-CV COMPARISON: M2 vs M0 ===\n")

      elpd_m0 <- loo_m0$estimates["elpd_loo", "Estimate"]
      cat(sprintf("    M0 ELPD_loo = %.1f\n", elpd_m0))
      cat(sprintf("    M2 ELPD_loo = %.1f\n", elpd_m2))
      cat(sprintf("    Difference  = %.1f (positive = M2 better)\n",
                  elpd_m2 - elpd_m0))
    }
  }

  ## 8c. Three-way comparison if all three LOO objects available
  if (file.exists(RESULTS_M0_PATH) && file.exists(RESULTS_M1_PATH)) {
    results_m0 <- readRDS(RESULTS_M0_PATH)
    results_m1 <- readRDS(RESULTS_M1_PATH)
    if (!is.null(results_m0$loo) && !is.null(results_m1$loo)) {
      cat("\n  === THREE-WAY LOO COMPARISON: M0 vs M1 vs M2 ===\n")
      comp_all <- loo_compare(results_m0$loo, results_m1$loo, loo_m2)
      print(comp_all)
      cat("\n")
    }
  }
} else {
  cat("  [WARN] LOO-CV computation failed. Skipping.\n")
}

cat("\n")


###############################################################################
## SECTION 9 : POSTERIOR PREDICTIVE CHECKS
###############################################################################
cat("--- 9. Posterior predictive checks ---\n")

## Use y_rep from Stan generated quantities for PPC (simpler and more reliable)
y_rep_available <- tryCatch({
  fit_m2$draws("y_rep[1]", format = "matrix")
  TRUE
}, error = function(e) FALSE)

N <- stan_data_m2$N
y_obs <- stan_data_m2$y
n_trial <- stan_data_m2$n_trial
z_obs <- stan_data_m2$z

observed_zero_rate <- mean(z_obs == 0)
observed_mean_share_pos <- mean(y_obs[z_obs == 1] / n_trial[z_obs == 1])

cat(sprintf("  Observed zero rate (structural): %.3f (%.1f%%)\n",
            observed_zero_rate, 100 * observed_zero_rate))
cat(sprintf("  Observed mean IT share (y/n | z=1): %.4f\n",
            observed_mean_share_pos))

if (y_rep_available) {
  cat("\n  Using y_rep from Stan generated quantities for PPC ...\n")

  y_rep_draws <- fit_m2$draws("y_rep", format = "matrix")  # n_draws x N
  n_draws_ppc <- nrow(y_rep_draws)
  cat(sprintf("  Number of posterior draws: %d\n", n_draws_ppc))

  ## Subsample for speed if needed
  n_ppc_use <- min(200, n_draws_ppc)
  draw_idx <- seq(1, n_draws_ppc, length.out = n_ppc_use) |> round() |> unique()
  n_ppc_use <- length(draw_idx)

  cat(sprintf("  Using %d draws for PPC summaries ...\n", n_ppc_use))

  ppc_zero_rates <- numeric(n_ppc_use)
  ppc_mean_share <- numeric(n_ppc_use)

  for (d in seq_along(draw_idx)) {
    dd <- draw_idx[d]
    y_rep_d <- as.integer(y_rep_draws[dd, ])

    ## Zero rate
    ppc_zero_rates[d] <- mean(y_rep_d == 0)

    ## Mean IT share among positive draws
    pos_idx <- which(y_rep_d > 0)
    if (length(pos_idx) > 0) {
      ppc_mean_share[d] <- mean(y_rep_d[pos_idx] / n_trial[pos_idx])
    } else {
      ppc_mean_share[d] <- NA_real_
    }
  }

} else {
  cat("\n  y_rep not available. Computing PPC from parameter draws ...\n")

  ## Extract posterior draws for alpha, beta, log_kappa, and all deltas
  alpha_draws <- fit_m2$draws("alpha", format = "matrix")  # n_draws x P
  beta_draws  <- fit_m2$draws("beta",  format = "matrix")  # n_draws x P
  lk_draws    <- fit_m2$draws("log_kappa", format = "matrix")  # n_draws x 1

  ## Extract all delta_ext and delta_int draws
  ## delta_ext[s,p] for s=1:S, p=1:P
  delta_ext_names_all <- c()
  delta_int_names_all <- c()
  for (s in 1:S) {
    for (p in 1:P) {
      delta_ext_names_all <- c(delta_ext_names_all, sprintf("delta_ext[%d,%d]", s, p))
      delta_int_names_all <- c(delta_int_names_all, sprintf("delta_int[%d,%d]", s, p))
    }
  }
  delta_ext_all_draws <- fit_m2$draws(delta_ext_names_all, format = "matrix")
  delta_int_all_draws <- fit_m2$draws(delta_int_names_all, format = "matrix")

  n_draws_ppc <- nrow(alpha_draws)
  n_ppc_use <- min(200, n_draws_ppc)
  draw_idx <- seq(1, n_draws_ppc, length.out = n_ppc_use) |> round() |> unique()
  n_ppc_use <- length(draw_idx)

  cat(sprintf("  Number of posterior draws: %d, using %d for PPC\n",
              n_draws_ppc, n_ppc_use))

  X <- stan_data_m2$X
  state_idx <- stan_data_m2$state

  ppc_zero_rates <- numeric(n_ppc_use)
  ppc_mean_share <- numeric(n_ppc_use)

  for (d in seq_along(draw_idx)) {
    dd <- draw_idx[d]

    ## Compute linear predictors with state-varying coefficients
    ## eta_ext[i] = X_i * alpha + X_i * delta_ext[state[i], .]
    ## eta_int[i] = X_i * beta  + X_i * delta_int[state[i], .]
    alpha_d <- as.numeric(alpha_draws[dd, ])
    beta_d  <- as.numeric(beta_draws[dd, ])
    kappa_d <- exp(lk_draws[dd, 1])

    eta_ext <- numeric(N)
    eta_int <- numeric(N)
    for (i in 1:N) {
      s <- state_idx[i]
      ## Extract delta_ext[s, 1:P] for this draw
      ## Column indices in the flattened matrix: (s-1)*P + 1:P
      col_start <- (s - 1) * P
      d_ext_s <- as.numeric(delta_ext_all_draws[dd, col_start + 1:P])
      d_int_s <- as.numeric(delta_int_all_draws[dd, col_start + 1:P])
      x_i <- as.numeric(X[i, ])
      eta_ext[i] <- sum(x_i * (alpha_d + d_ext_s))
      eta_int[i] <- sum(x_i * (beta_d  + d_int_s))
    }

    q_i  <- inv_logit(eta_ext)
    mu_i <- inv_logit(eta_int)
    mu_i <- pmin(pmax(mu_i, 1e-8), 1 - 1e-8)

    ## Simulate z (extensive margin)
    z_sim <- rbinom(N, size = 1, prob = q_i)

    ## Simulate y (intensive margin) for z_sim == 1
    y_sim <- integer(N)
    pos_idx <- which(z_sim == 1)
    for (i in pos_idx) {
      y_sim[i] <- rztbetabinom(1L, size = n_trial[i], mu = mu_i[i], kappa = kappa_d)
    }

    ## Compute summaries
    ppc_zero_rates[d] <- mean(z_sim == 0)
    if (length(pos_idx) > 0) {
      ppc_mean_share[d] <- mean(y_sim[pos_idx] / n_trial[pos_idx])
    } else {
      ppc_mean_share[d] <- NA_real_
    }
  }
}

## PPC results
cat("\n  PPC: Zero rate (structural)\n")
cat(sprintf("    Observed          : %.3f\n", observed_zero_rate))
cat(sprintf("    Posterior mean    : %.3f\n", mean(ppc_zero_rates)))
cat(sprintf("    Posterior 95%% CI  : [%.3f, %.3f]\n",
            quantile(ppc_zero_rates, 0.025),
            quantile(ppc_zero_rates, 0.975)))

zero_rate_covered <- (observed_zero_rate >= quantile(ppc_zero_rates, 0.025) &
                      observed_zero_rate <= quantile(ppc_zero_rates, 0.975))
if (zero_rate_covered) {
  cat("    [PASS] Observed zero rate within 95% posterior predictive interval.\n")
} else {
  cat("    [NOTE] Observed zero rate outside 95% PPI. Model may be miscalibrated.\n")
}

cat("\n  PPC: Mean IT share (y/n | z=1)\n")
ppc_share_valid <- ppc_mean_share[!is.na(ppc_mean_share)]
cat(sprintf("    Observed          : %.4f\n", observed_mean_share_pos))
cat(sprintf("    Posterior mean    : %.4f\n", mean(ppc_share_valid)))
cat(sprintf("    Posterior 95%% CI  : [%.4f, %.4f]\n",
            quantile(ppc_share_valid, 0.025),
            quantile(ppc_share_valid, 0.975)))

share_covered <- (observed_mean_share_pos >= quantile(ppc_share_valid, 0.025) &
                  observed_mean_share_pos <= quantile(ppc_share_valid, 0.975))
if (share_covered) {
  cat("    [PASS] Observed mean IT share within 95% posterior predictive interval.\n")
} else {
  cat("    [NOTE] Observed mean IT share outside 95% PPI.\n")
}

cat("\n")


###############################################################################
## SECTION 10 : SAVE RESULTS
###############################################################################
cat("--- 10. Saving results ---\n")

## 10a. Save the CmdStanR fit object
fit_m2$save_object(FIT_OUT)
cat(sprintf("  Saved fit object: %s\n", FIT_OUT))
cat(sprintf("    File size: %.1f MB\n",
            file.info(FIT_OUT)$size / 1024^2))

## 10b. Assemble and save results list
results_m2 <- list(
  ## Model info
  model      = "M2",
  model_desc = "Block-Diagonal SVC Hurdle Beta-Binomial (unweighted)",

  ## Timing
  fit_time_mins = as.numeric(fit_time),

  ## Parameter summary (fixed effects)
  param_summary = param_summary_fixed,

  ## Hierarchical parameter summary
  hier_summary = param_summary_hier,

  ## Key estimates
  alpha_means    = setNames(alpha_means, cov_names),
  beta_means     = setNames(beta_means, cov_names),
  log_kappa_mean = log_kappa_mean,
  kappa_mean     = kappa_mean,
  tau_ext_means  = setNames(tau_ext_means, cov_names),
  tau_int_means  = setNames(tau_int_means, cov_names),

  ## Correlation matrices (if available)
  omega_ext_mat = omega_ext_mat,
  omega_int_mat = omega_int_mat,

  ## State-level poverty analysis
  state_poverty_table   = state_poverty_table,
  n_reversal_mean       = n_reversal_mean,
  n_reversal_prob50     = n_reversal_prob50,

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
  loo = loo_m2,

  ## PPC
  ppc = list(
    observed_zero_rate     = observed_zero_rate,
    ppc_zero_rates         = ppc_zero_rates,
    ppc_zero_rate_mean     = mean(ppc_zero_rates),
    ppc_zero_rate_95ci     = quantile(ppc_zero_rates, c(0.025, 0.975)),
    observed_mean_share    = observed_mean_share_pos,
    ppc_mean_share         = ppc_share_valid,
    ppc_mean_share_mean    = mean(ppc_share_valid),
    ppc_mean_share_95ci    = quantile(ppc_share_valid, c(0.025, 0.975))
  ),

  ## Stan data used (for reproducibility)
  stan_data_m2 = stan_data_m2,

  ## Timestamp
  timestamp = Sys.time()
)

saveRDS(results_m2, RESULTS_OUT)
cat(sprintf("  Saved results: %s\n", RESULTS_OUT))
cat(sprintf("    File size: %.1f KB\n",
            file.info(RESULTS_OUT)$size / 1024))


###############################################################################
## SECTION 11 : FINAL SUMMARY
###############################################################################
cat("\n==============================================================\n")
cat("  M2 FITTING SUMMARY\n")
cat("==============================================================\n")

n_fixed  <- P * 2 + 1                          # alpha + beta + log_kappa
n_hier   <- 2 * P + P*(P-1)/2 + P*(P-1)/2     # tau_ext + tau_int + L_ext + L_int
n_re     <- 2 * P * S                          # delta_ext + delta_int
n_params_total <- n_fixed + n_hier + n_re

cat(sprintf("\n  Model: Block-Diagonal SVC Hurdle Beta-Binomial (M2)\n"))
cat(sprintf("  N = %d, P = %d, S = %d\n", N, P, S))
cat(sprintf("  Parameters: %d fixed + %d hierarchical + %d random effects = %d total\n",
            n_fixed, n_hier, n_re, n_params_total))
cat(sprintf("  Fit time: %.1f minutes\n", as.numeric(fit_time)))

cat(sprintf("\n  MCMC Diagnostics:\n"))
cat(sprintf("    Divergent transitions : %d %s\n",
            total_divergent, ifelse(total_divergent == 0, "[PASS]", "[WARN]")))
cat(sprintf("    Max R-hat (fixed)     : %.4f %s\n",
            max_rhat, ifelse(max_rhat < 1.01, "[PASS]", "[WARN]")))
cat(sprintf("    Max R-hat (deltas)    : %.4f %s\n",
            max_rhat_delta, ifelse(max_rhat_delta < 1.01, "[PASS]", "[WARN]")))
cat(sprintf("    Min ESS (bulk)        : %.0f %s\n",
            min_ess_bulk, ifelse(min_ess_bulk > 400, "[PASS]", "[WARN]")))
cat(sprintf("    Min ESS (tail)        : %.0f %s\n",
            min_ess_tail, ifelse(min_ess_tail > 400, "[PASS]", "[WARN]")))

cat(sprintf("\n  Key Results:\n"))
cat(sprintf("    alpha_poverty = %+.4f %s\n",
            alpha_poverty, ifelse(alpha_poverty < 0, "[PASS] (<0)", "[WARN]")))
cat(sprintf("    beta_poverty  = %+.4f %s\n",
            beta_poverty, ifelse(beta_poverty > 0, "[PASS] (>0)", "[WARN]")))
cat(sprintf("    kappa         = %.2f %s\n",
            kappa_mean, ifelse(kappa_mean > 3 & kappa_mean < 30, "[PASS]", "[WARN]")))

cat(sprintf("\n  Tau (state variation scale):\n"))
cat("    Extensive margin:\n")
for (k in 1:P) {
  cat(sprintf("      tau_ext[%d] %-10s = %.4f\n", k, cov_names[k], tau_ext_means[k]))
}
cat("    Intensive margin:\n")
for (k in 1:P) {
  cat(sprintf("      tau_int[%d] %-10s = %.4f\n", k, cov_names[k], tau_int_means[k]))
}

cat(sprintf("\n  Poverty Reversal (state-level):\n"))
cat(sprintf("    States with reversal (posterior mean): %d / %d (%.0f%%)\n",
            n_reversal_mean, S, 100 * n_reversal_mean / S))
cat(sprintf("    States with reversal (P > 0.5):       %d / %d (%.0f%%)\n",
            n_reversal_prob50, S, 100 * n_reversal_prob50 / S))
cat(sprintf("    Expected: ~22/51 (43%%)\n"))

cat(sprintf("\n  PPC:\n"))
cat(sprintf("    Zero rate: observed=%.3f, predicted=%.3f\n",
            observed_zero_rate, mean(ppc_zero_rates)))
cat(sprintf("    IT share:  observed=%.4f, predicted=%.4f\n",
            observed_mean_share_pos, mean(ppc_share_valid)))

if (!is.null(loo_m2)) {
  cat(sprintf("\n  LOO-CV:\n"))
  cat(sprintf("    ELPD_loo   = %.1f (SE = %.1f)\n",
              loo_m2$estimates["elpd_loo", "Estimate"],
              loo_m2$estimates["elpd_loo", "SE"]))
  cat(sprintf("    p_loo      = %.1f\n",
              loo_m2$estimates["p_loo", "Estimate"]))
  cat(sprintf("    Pareto k > 0.7: %d / %d\n",
              sum(loo_m2$diagnostics$pareto_k > 0.7),
              length(loo_m2$diagnostics$pareto_k)))

  if (file.exists(RESULTS_M1_PATH)) {
    results_m1 <- readRDS(RESULTS_M1_PATH)
    if (!is.null(results_m1$loo)) {
      cat(sprintf("\n  LOO Comparison:\n"))
      cat(sprintf("    M0 ELPD = %.1f\n",
                  readRDS(RESULTS_M0_PATH)$loo$estimates["elpd_loo", "Estimate"]))
      cat(sprintf("    M1 ELPD = %.1f  (M1-M0 = +%.1f)\n",
                  results_m1$loo$estimates["elpd_loo", "Estimate"],
                  results_m1$loo$estimates["elpd_loo", "Estimate"] -
                  readRDS(RESULTS_M0_PATH)$loo$estimates["elpd_loo", "Estimate"]))
      cat(sprintf("    M2 ELPD = %.1f  (M2-M1 = +%.1f, M2-M0 = +%.1f)\n",
                  loo_m2$estimates["elpd_loo", "Estimate"],
                  loo_m2$estimates["elpd_loo", "Estimate"] -
                  results_m1$loo$estimates["elpd_loo", "Estimate"],
                  loo_m2$estimates["elpd_loo", "Estimate"] -
                  readRDS(RESULTS_M0_PATH)$loo$estimates["elpd_loo", "Estimate"]))
    }
  }
}

cat(sprintf("\n  Output files:\n"))
cat(sprintf("    %s\n", FIT_OUT))
cat(sprintf("    %s\n", RESULTS_OUT))

cat("\n==============================================================\n")
cat("  M2 FITTING COMPLETE.\n")
if (diag_pass) {
  cat("  ALL DIAGNOSTICS PASSED. Ready for M3.\n")
} else {
  cat("  SOME DIAGNOSTICS FAILED. Review before proceeding to M3.\n")
}
cat("==============================================================\n")
