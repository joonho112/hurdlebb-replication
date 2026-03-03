## =============================================================================
## 40_fit_m3a.R -- Fit Cross-Margin Covariance SVC Model (M3a)
## =============================================================================
## Purpose : Compile and fit the Cross-Margin Covariance SVC HBB model (M3a)
##           on NSECE 2019 data (unweighted), run diagnostics, LOO-CV
##           comparison, PPC, and cross-margin correlation analysis.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Track   : A (requires NSECE restricted-use data; see data/DATA_ACCESS.md)
## Inputs  : data/precomputed/stan_data.rds
##           stan/hbb_m3a.stan
##           code/helpers/utils.R
##           data/precomputed/results_m2.rds
##           data/precomputed/results_m1.rds
##           data/precomputed/results_m0.rds
## Outputs : data/precomputed/fit_m3a.rds
##           data/precomputed/results_m3a.rds
## =============================================================================

cat("==============================================================\n")
cat("  HBB Replication: M3a Fitting  (Phase 4)\n")
cat("  Cross-Margin Covariance SVC Hurdle Beta-Binomial\n")
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
STAN_MODEL_PATH <- file.path(PROJECT_ROOT, "stan/hbb_m3a.stan")
OUTPUT_DIR      <- file.path(PROJECT_ROOT, "data/precomputed")
FIT_OUT         <- file.path(OUTPUT_DIR, "fit_m3a.rds")
RESULTS_OUT     <- file.path(OUTPUT_DIR, "results_m3a.rds")
RESULTS_M2_PATH <- file.path(OUTPUT_DIR, "results_m2.rds")
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

## Build M3a-specific data list: same as M2 (N, P, S, y, n_trial, z, X, state)
stan_data_m3a <- list(
  N       = full_stan_data$N,
  P       = full_stan_data$P,
  S       = full_stan_data$S,
  y       = full_stan_data$y,
  n_trial = full_stan_data$n_trial,
  z       = full_stan_data$z,
  X       = full_stan_data$X,
  state   = full_stan_data$state
)

P <- stan_data_m3a$P
S <- stan_data_m3a$S
K <- 2 * P  # Joint dimension: P extensive + P intensive = 10

cat(sprintf("  M3a data list: N=%d, P=%d, S=%d, K=2P=%d\n",
            stan_data_m3a$N, P, S, K))
cat(sprintf("  Zero rate: %.1f%%\n", 100 * mean(stan_data_m3a$z == 0)))

## State sample sizes
state_ns <- table(stan_data_m3a$state)
cat(sprintf("  State sample sizes: min=%d, median=%d, max=%d\n",
            min(state_ns), median(state_ns), max(state_ns)))

## M3a random effects dimensionality
n_re_per_state <- K
n_re_total     <- K * S
n_corr_params  <- K * (K - 1) / 2
cat(sprintf("  Random effects: %d per state x %d states = %d total\n",
            n_re_per_state, S, n_re_total))
cat(sprintf("  Correlation parameters: %d (from %dx%d Omega)\n",
            n_corr_params, K, K))
cat(sprintf("  Cross-margin correlations: %d (P x P off-diagonal block)\n",
            P * P))
cat("  [PASS] Data loaded and M3a data list constructed.\n\n")


###############################################################################
## SECTION 2 : COMPILE STAN MODEL
###############################################################################
cat("--- 2. Compiling Stan model ---\n")

stopifnot("Stan model file not found" = file.exists(STAN_MODEL_PATH))
cat(sprintf("  Model file: %s\n", STAN_MODEL_PATH))

## Compile (cmdstanr caches compiled models; recompiles only if .stan changed)
m3a_model <- tryCatch(
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
## SECTION 3 : INITIALISATION FROM M2 POSTERIORS (WARM START)
###############################################################################
cat("--- 3. Setting initial values ---\n")

if (file.exists(RESULTS_M2_PATH)) {
  results_m2 <- readRDS(RESULTS_M2_PATH)
  cat("  [INFO] M2 results found. Using M2 posteriors for warm start.\n")

  ## Extract M2 posterior means
  m2_alpha     <- as.numeric(results_m2$alpha_means)
  m2_beta      <- as.numeric(results_m2$beta_means)
  m2_log_kappa <- as.numeric(results_m2$log_kappa_mean)
  m2_tau_ext   <- as.numeric(results_m2$tau_ext_means)   # length P
  m2_tau_int   <- as.numeric(results_m2$tau_int_means)   # length P

  ## For M3a: tau is a single vector of length K=2P
  ## tau[1:P] = extensive, tau[(P+1):(2P)] = intensive
  init_tau <- c(m2_tau_ext, m2_tau_int)

  init_fun <- function() {
    list(
      alpha     = m2_alpha,
      beta      = m2_beta,
      log_kappa = m2_log_kappa,
      tau       = init_tau,
      L_Omega   = diag(K),
      z_eps     = rep(list(rep(0, K)), S)
    )
  }

  cat(sprintf("  alpha init    : [%s]\n",
              paste(sprintf("%+.3f", m2_alpha), collapse = ", ")))
  cat(sprintf("  beta init     : [%s]\n",
              paste(sprintf("%+.3f", m2_beta), collapse = ", ")))
  cat(sprintf("  log_kappa init: %.3f\n", m2_log_kappa))
  cat(sprintf("  tau init      : [%s]\n",
              paste(sprintf("%.3f", init_tau), collapse = ", ")))
  cat("  L_Omega       : identity matrix (K x K)\n")
  cat("  z_eps         : list of S zero vectors (length K)\n")
} else {
  cat("  [INFO] M2 results not found. Using default initialisation.\n")

  init_fun <- function() {
    list(
      alpha     = rep(0, P),
      beta      = rep(0, P),
      log_kappa = log(10),
      tau       = rep(0.2, K),
      L_Omega   = diag(K),
      z_eps     = rep(list(rep(0, K)), S)
    )
  }
}

cat("  [PASS] Initial values set.\n\n")


###############################################################################
## SECTION 4 : FIT M3a
###############################################################################
cat("--- 4. Fitting M3a ---\n")
cat("  chains = 4, parallel_chains = 4\n")
cat("  iter_warmup = 1000, iter_sampling = 1000\n")
cat("  adapt_delta = 0.95, max_treedepth = 12\n")
cat("  seed = 20250220\n\n")

fit_start <- Sys.time()

fit_m3a <- tryCatch(
  m3a_model$sample(
    data            = stan_data_m3a,
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
n_divergent <- fit_m3a$diagnostic_summary()$num_divergent
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
n_max_treedepth <- fit_m3a$diagnostic_summary()$num_max_treedepth
total_max_td <- sum(n_max_treedepth)
cat(sprintf("  Max treedepth hits: %d (per chain: %s)\n",
            total_max_td, paste(n_max_treedepth, collapse = ", ")))
if (total_max_td > 0) {
  warning("  [WARN] Max treedepth reached. Consider increasing max_treedepth.")
}

## 5c. Parameter summary -- fixed effects + hierarchical parameters
## Fixed effects
param_names_fixed <- c(paste0("alpha[", 1:P, "]"),
                       paste0("beta[", 1:P, "]"),
                       "log_kappa")

## Hierarchical parameters: tau[1:K]
param_names_tau <- paste0("tau[", 1:K, "]")
param_names_hier <- param_names_tau

## Full summary for diagnostics (fixed + hierarchical)
param_names_all <- c(param_names_fixed, param_names_hier)
param_summary <- fit_m3a$summary(variables = param_names_all)

cat("\n  Fixed effect parameter summary:\n")
param_summary_fixed <- fit_m3a$summary(variables = param_names_fixed)
print(param_summary_fixed, n = nrow(param_summary_fixed))

cat("\n  Hierarchical parameter summary (tau[1:K]):\n")
param_summary_hier <- fit_m3a$summary(variables = param_names_hier)
print(param_summary_hier, n = nrow(param_summary_hier))

## Selected delta values (a few states for monitoring)
## In M3a, delta[s] is a vector of length K; Stan names: delta[s,k]
delta_sample_states <- c(1, 10, 25, 40, 51)
param_names_delta_sample <- c()
for (s in delta_sample_states) {
  for (k in 1:K) {
    param_names_delta_sample <- c(param_names_delta_sample,
                                   sprintf("delta[%d,%d]", s, k))
  }
}

cat("\n  Selected delta (state random coefficients) summary:\n")
param_summary_delta_sample <- fit_m3a$summary(variables = param_names_delta_sample)
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
delta_all_names <- c()
for (s in 1:S) {
  for (k in 1:K) {
    delta_all_names <- c(delta_all_names, sprintf("delta[%d,%d]", s, k))
  }
}
delta_all_summary <- fit_m3a$summary(variables = delta_all_names)
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
tau_means <- param_summary_hier$mean

## Covariate names for display
cov_names <- c("intercept", "poverty", "urban", "black", "hispanic")

## Labels for tau: first P are extensive, last P are intensive
tau_labels <- c(paste0("ext_", cov_names), paste0("int_", cov_names))

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

cat(sprintf("\n  Hierarchical scale parameters (tau[1:K]):\n"))
for (k in 1:K) {
  margin_label <- ifelse(k <= P, "ext", "int")
  cov_idx <- ifelse(k <= P, k, k - P)
  cat(sprintf("    tau[%2d] %-15s = %.4f\n",
              k, tau_labels[k], tau_means[k]))
}

## 6a. Extract Omega correlation matrix (from generated quantities)
cat("\n  Omega (K x K correlation matrix) posterior means:\n")
omega_mat <- matrix(NA, K, K)
for (i in 1:K) {
  for (j in 1:K) {
    draws_ij <- fit_m3a$draws(sprintf("Omega[%d,%d]", i, j), format = "matrix")
    omega_mat[i, j] <- mean(draws_ij)
  }
}

## Print full Omega matrix
cat("    ")
cat(sprintf("%8s", tau_labels), "\n")
for (i in 1:K) {
  cat(sprintf("    %-15s", tau_labels[i]))
  cat(sprintf("%8.3f", omega_mat[i, ]))
  cat("\n")
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
for (k in 1:P) {
  status <- ifelse(tau_means[k] > 0.01, "[PASS]", "[NOTE: near zero]")
  cat(sprintf("    tau[%d] %-10s = %.4f %s\n",
              k, cov_names[k], tau_means[k], status))
}
cat("  Intensive margin tau:\n")
for (k in 1:P) {
  status <- ifelse(tau_means[P + k] > 0.01, "[PASS]", "[NOTE: near zero]")
  cat(sprintf("    tau[%d] %-10s = %.4f %s\n",
              P + k, cov_names[k], tau_means[P + k], status))
}

## Compare M3a tau with M2 tau (should be similar)
if (file.exists(RESULTS_M2_PATH)) {
  cat("\n  === COMPARISON WITH M2 tau ===\n")
  cat(sprintf("    %-15s %10s %10s\n", "Parameter", "M2", "M3a"))
  cat(sprintf("    %s\n", paste(rep("-", 38), collapse = "")))
  for (k in 1:P) {
    cat(sprintf("    %-15s %10.4f %10.4f\n",
                paste0("tau_ext_", cov_names[k]),
                as.numeric(results_m2$tau_ext_means[k]),
                tau_means[k]))
  }
  for (k in 1:P) {
    cat(sprintf("    %-15s %10.4f %10.4f\n",
                paste0("tau_int_", cov_names[k]),
                as.numeric(results_m2$tau_int_means[k]),
                tau_means[P + k]))
  }
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
## SECTION 7 : CROSS-MARGIN CORRELATION ANALYSIS (KEY NOVELTY)
###############################################################################
cat("--- 7. Cross-margin correlation analysis ---\n")
cat("  This is the core methodological contribution of M3a.\n")
cat("  The 10x10 Omega has a cross-margin block: Omega[k, P+j] for k,j=1,...,P\n\n")

## 7a. Extract the cross-margin correlation block
## Omega[k, P+k] for k = 1,...,P gives the "same covariate, cross margin" correlations
## CmdStanR variable names: Omega[1,6], Omega[2,7], ..., Omega[5,10]

cat("  === CROSS-MARGIN CORRELATIONS: SAME COVARIATE (DIAGONAL) ===\n")
cat("  rho_cross[k] = Omega[k, P+k] (extensive k <-> intensive k)\n\n")

cat(sprintf("  %-15s %10s %10s %10s %10s %10s\n",
            "Covariate", "PostMean", "PostSD", "Q2.5", "Q97.5", "P(rho<0)"))
cat(sprintf("  %s\n", paste(rep("-", 65), collapse = "")))

cross_diag_means   <- numeric(P)
cross_diag_sd      <- numeric(P)
cross_diag_q025    <- numeric(P)
cross_diag_q975    <- numeric(P)
cross_diag_prob_neg <- numeric(P)

for (k in 1:P) {
  var_name <- sprintf("Omega[%d,%d]", k, P + k)
  draws_k <- as.numeric(fit_m3a$draws(var_name, format = "matrix"))

  cross_diag_means[k]    <- mean(draws_k)
  cross_diag_sd[k]       <- sd(draws_k)
  cross_diag_q025[k]     <- quantile(draws_k, 0.025)
  cross_diag_q975[k]     <- quantile(draws_k, 0.975)
  cross_diag_prob_neg[k] <- mean(draws_k < 0)

  cat(sprintf("  %-15s %+10.4f %10.4f %+10.4f %+10.4f %10.3f\n",
              cov_names[k],
              cross_diag_means[k], cross_diag_sd[k],
              cross_diag_q025[k], cross_diag_q975[k],
              cross_diag_prob_neg[k]))
}

## Check significance (95% CI excludes zero)
cat("\n  Significance assessment (95% CI excludes zero):\n")
for (k in 1:P) {
  excludes_zero <- (cross_diag_q025[k] > 0) | (cross_diag_q975[k] < 0)
  direction <- ifelse(cross_diag_means[k] < 0, "negative", "positive")
  if (excludes_zero) {
    cat(sprintf("    rho_cross[%d] %-10s: [SIGNIFICANT] %s (95%% CI: [%.3f, %.3f])\n",
                k, cov_names[k], direction, cross_diag_q025[k], cross_diag_q975[k]))
  } else {
    cat(sprintf("    rho_cross[%d] %-10s: [not signif.] (95%% CI: [%.3f, %.3f])\n",
                k, cov_names[k], cross_diag_q025[k], cross_diag_q975[k]))
  }
}

## 7b. Full cross-margin block (P x P off-diagonal block of Omega)
cat("\n  === FULL CROSS-MARGIN CORRELATION BLOCK ===\n")
cat("  Omega[k, P+j] for k=1,...,P and j=1,...,P\n\n")

cross_block_mean <- matrix(NA, P, P)
cross_block_sd   <- matrix(NA, P, P)
cross_block_pneg <- matrix(NA, P, P)

for (k in 1:P) {
  for (j in 1:P) {
    var_name <- sprintf("Omega[%d,%d]", k, P + j)
    draws_kj <- as.numeric(fit_m3a$draws(var_name, format = "matrix"))
    cross_block_mean[k, j] <- mean(draws_kj)
    cross_block_sd[k, j]   <- sd(draws_kj)
    cross_block_pneg[k, j] <- mean(draws_kj < 0)
  }
}

cat("  Posterior means:\n")
cat(sprintf("    %15s", ""))
cat(sprintf("%12s", paste0("int_", cov_names)), "\n")
for (k in 1:P) {
  cat(sprintf("    %-15s", paste0("ext_", cov_names[k])))
  cat(sprintf("%+12.4f", cross_block_mean[k, ]))
  cat("\n")
}

cat("\n  Posterior SDs:\n")
cat(sprintf("    %15s", ""))
cat(sprintf("%12s", paste0("int_", cov_names)), "\n")
for (k in 1:P) {
  cat(sprintf("    %-15s", paste0("ext_", cov_names[k])))
  cat(sprintf("%12.4f", cross_block_sd[k, ]))
  cat("\n")
}

cat("\n  P(rho < 0):\n")
cat(sprintf("    %15s", ""))
cat(sprintf("%12s", paste0("int_", cov_names)), "\n")
for (k in 1:P) {
  cat(sprintf("    %-15s", paste0("ext_", cov_names[k])))
  cat(sprintf("%12.3f", cross_block_pneg[k, ]))
  cat("\n")
}

## 7c. Key substantive interpretation
cat("\n  === KEY SUBSTANTIVE FINDINGS ===\n")

## Poverty cross-margin correlation
cat(sprintf("  Poverty cross-margin (Omega[2,%d]):\n", P + 2))
cat(sprintf("    Posterior mean = %+.4f (SD = %.4f)\n",
            cross_diag_means[2], cross_diag_sd[2]))
cat(sprintf("    95%% CI = [%+.4f, %+.4f]\n",
            cross_diag_q025[2], cross_diag_q975[2]))
cat(sprintf("    P(rho < 0) = %.3f\n", cross_diag_prob_neg[2]))

if (cross_diag_prob_neg[2] > 0.95) {
  cat("    [PASS] Strong evidence for NEGATIVE cross-margin poverty correlation.\n")
  cat("    Interpretation: States where poverty is a stronger barrier to serving IT\n")
  cat("    (more negative alpha_pov) tend to have higher IT share among those who serve\n")
  cat("    (more positive beta_pov). This supports the 'compensating behavior' mechanism.\n")
} else if (cross_diag_prob_neg[2] > 0.80) {
  cat("    [NOTE] Moderate evidence for negative cross-margin poverty correlation.\n")
} else {
  cat("    [NOTE] Weak or no evidence for negative cross-margin poverty correlation.\n")
}

## Intercept cross-margin correlation
cat(sprintf("\n  Intercept cross-margin (Omega[1,%d]):\n", P + 1))
cat(sprintf("    Posterior mean = %+.4f (SD = %.4f)\n",
            cross_diag_means[1], cross_diag_sd[1]))
cat(sprintf("    95%% CI = [%+.4f, %+.4f]\n",
            cross_diag_q025[1], cross_diag_q975[1]))
cat(sprintf("    P(rho < 0) = %.3f\n", cross_diag_prob_neg[1]))

cat("\n")


###############################################################################
## SECTION 8 : STATE-LEVEL POVERTY REVERSAL ANALYSIS
###############################################################################
cat("--- 8. State-level poverty reversal analysis ---\n")

## In M3a, delta[s,k] where k=1,...,P is extensive, k=P+1,...,2P is intensive
## State-specific total poverty effect:
##   alpha_poverty_s = alpha[2] + delta[s, 2]
##   beta_poverty_s  = beta[2]  + delta[s, P+2]

## Extract alpha[2] and beta[2] draws
alpha2_draws <- fit_m3a$draws("alpha[2]", format = "matrix")  # n_draws x 1
beta2_draws  <- fit_m3a$draws("beta[2]",  format = "matrix")  # n_draws x 1
n_draws <- nrow(alpha2_draws)

## Extract delta[s,2] and delta[s,P+2] for all states
delta_ext_pov_names <- sprintf("delta[%d,2]", 1:S)
delta_int_pov_names <- sprintf("delta[%d,%d]", 1:S, P + 2)
delta_ext_pov_draws <- fit_m3a$draws(delta_ext_pov_names, format = "matrix")  # n_draws x S
delta_int_pov_draws <- fit_m3a$draws(delta_int_pov_names, format = "matrix")  # n_draws x S

## Compute state-specific total poverty effects for each draw
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
cat(sprintf("  Expected from M2 analysis: ~47/51 (92%%)\n"))

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
## SECTION 9 : LOO-CV AND COMPARISON WITH M2/M1/M0
###############################################################################
cat("--- 9. LOO-CV computation and M2/M1/M0 comparison ---\n")

## Extract log_lik array
log_lik <- fit_m3a$draws("log_lik", format = "matrix")
cat(sprintf("  log_lik dimensions: %d draws x %d observations\n",
            nrow(log_lik), ncol(log_lik)))

## Compute LOO
loo_m3a <- tryCatch(
  loo(log_lik, cores = 4),
  error = function(e) {
    warning("LOO computation failed: ", conditionMessage(e))
    NULL
  }
)

if (!is.null(loo_m3a)) {
  cat("\n  LOO-CV summary (M3a):\n")
  print(loo_m3a)

  ## Check Pareto k diagnostics
  k_values <- loo_m3a$diagnostics$pareto_k
  n_bad_k <- sum(k_values > 0.7)
  cat(sprintf("\n  Pareto k > 0.7: %d / %d observations (%.1f%%)\n",
              n_bad_k, length(k_values), 100 * n_bad_k / length(k_values)))

  if (n_bad_k > 0) {
    cat(sprintf("  [NOTE] %d observations with problematic Pareto k values.\n", n_bad_k))
    cat("         Consider moment matching or reloo for these observations.\n")
  } else {
    cat("  [PASS] All Pareto k values < 0.7.\n")
  }

  ## 9a. LOO comparison with M2
  if (file.exists(RESULTS_M2_PATH)) {
    results_m2 <- readRDS(RESULTS_M2_PATH)
    loo_m2 <- results_m2$loo

    if (!is.null(loo_m2)) {
      cat("\n  === LOO-CV COMPARISON: M3a vs M2 (CRITICAL TEST) ===\n")
      cat("  (Does cross-margin covariance help?)\n\n")

      elpd_m2  <- loo_m2$estimates["elpd_loo", "Estimate"]
      elpd_m3a <- loo_m3a$estimates["elpd_loo", "Estimate"]
      cat(sprintf("    M2  ELPD_loo = %.1f\n", elpd_m2))
      cat(sprintf("    M3a ELPD_loo = %.1f\n", elpd_m3a))
      cat(sprintf("    Difference   = %.1f (positive = M3a better)\n",
                  elpd_m3a - elpd_m2))

      ## Formal loo_compare
      comp_m3a_m2 <- loo_compare(loo_m2, loo_m3a)
      cat("\n  loo_compare output (M3a vs M2):\n")
      print(comp_m3a_m2)

      ## Interpretation
      elpd_diff <- comp_m3a_m2[2, "elpd_diff"]
      se_diff   <- comp_m3a_m2[2, "se_diff"]
      cat(sprintf("\n    ELPD difference: %.1f (SE = %.1f)\n",
                  elpd_diff, se_diff))

      if (abs(elpd_diff) > 2 * se_diff) {
        better_model <- rownames(comp_m3a_m2)[1]
        cat(sprintf("    [PASS] Difference > 2*SE: %s is clearly preferred.\n",
                    better_model))
      } else {
        cat("    [NOTE] Difference < 2*SE: models are comparable.\n")
        cat("    Cross-margin covariance may not improve predictive accuracy,\n")
        cat("    but the correlations themselves are still of substantive interest.\n")
      }
    }
  }

  ## 9b. LOO comparison with M1
  if (file.exists(RESULTS_M1_PATH)) {
    results_m1 <- readRDS(RESULTS_M1_PATH)
    loo_m1 <- results_m1$loo

    if (!is.null(loo_m1)) {
      cat("\n  === LOO-CV COMPARISON: M3a vs M1 ===\n")

      elpd_m1 <- loo_m1$estimates["elpd_loo", "Estimate"]
      cat(sprintf("    M1  ELPD_loo = %.1f\n", elpd_m1))
      cat(sprintf("    M3a ELPD_loo = %.1f\n", elpd_m3a))
      cat(sprintf("    Difference   = %.1f (positive = M3a better)\n",
                  elpd_m3a - elpd_m1))
    }
  }

  ## 9c. LOO comparison with M0
  if (file.exists(RESULTS_M0_PATH)) {
    results_m0 <- readRDS(RESULTS_M0_PATH)
    loo_m0 <- results_m0$loo

    if (!is.null(loo_m0)) {
      cat("\n  === LOO-CV COMPARISON: M3a vs M0 ===\n")

      elpd_m0 <- loo_m0$estimates["elpd_loo", "Estimate"]
      cat(sprintf("    M0  ELPD_loo = %.1f\n", elpd_m0))
      cat(sprintf("    M3a ELPD_loo = %.1f\n", elpd_m3a))
      cat(sprintf("    Difference   = %.1f (positive = M3a better)\n",
                  elpd_m3a - elpd_m0))
    }
  }

  ## 9d. Four-way comparison if all LOO objects available
  if (file.exists(RESULTS_M0_PATH) && file.exists(RESULTS_M1_PATH) &&
      file.exists(RESULTS_M2_PATH)) {
    results_m0 <- readRDS(RESULTS_M0_PATH)
    results_m1 <- readRDS(RESULTS_M1_PATH)
    results_m2 <- readRDS(RESULTS_M2_PATH)
    if (!is.null(results_m0$loo) && !is.null(results_m1$loo) &&
        !is.null(results_m2$loo)) {
      cat("\n  === FOUR-WAY LOO COMPARISON: M0 vs M1 vs M2 vs M3a ===\n")
      comp_all <- loo_compare(results_m0$loo, results_m1$loo,
                              results_m2$loo, loo_m3a)
      print(comp_all)

      cat("\n  LOO progression:\n")
      elpd_m0 <- results_m0$loo$estimates["elpd_loo", "Estimate"]
      elpd_m1 <- results_m1$loo$estimates["elpd_loo", "Estimate"]
      elpd_m2 <- results_m2$loo$estimates["elpd_loo", "Estimate"]
      cat(sprintf("    M0 ELPD = %.1f\n", elpd_m0))
      cat(sprintf("    M1 ELPD = %.1f  (M1-M0 = %+.1f)\n",
                  elpd_m1, elpd_m1 - elpd_m0))
      cat(sprintf("    M2 ELPD = %.1f  (M2-M1 = %+.1f, M2-M0 = %+.1f)\n",
                  elpd_m2, elpd_m2 - elpd_m1, elpd_m2 - elpd_m0))
      cat(sprintf("    M3a ELPD = %.1f  (M3a-M2 = %+.1f, M3a-M0 = %+.1f)\n",
                  elpd_m3a, elpd_m3a - elpd_m2, elpd_m3a - elpd_m0))
      cat("\n")
    }
  }
} else {
  cat("  [WARN] LOO-CV computation failed. Skipping.\n")
}

cat("\n")


###############################################################################
## SECTION 10 : POSTERIOR PREDICTIVE CHECKS
###############################################################################
cat("--- 10. Posterior predictive checks ---\n")

## Use y_rep from Stan generated quantities for PPC
y_rep_available <- tryCatch({
  fit_m3a$draws("y_rep[1]", format = "matrix")
  TRUE
}, error = function(e) FALSE)

N <- stan_data_m3a$N
y_obs <- stan_data_m3a$y
n_trial <- stan_data_m3a$n_trial
z_obs <- stan_data_m3a$z

observed_zero_rate <- mean(z_obs == 0)
observed_mean_share_pos <- mean(y_obs[z_obs == 1] / n_trial[z_obs == 1])

cat(sprintf("  Observed zero rate (structural): %.3f (%.1f%%)\n",
            observed_zero_rate, 100 * observed_zero_rate))
cat(sprintf("  Observed mean IT share (y/n | z=1): %.4f\n",
            observed_mean_share_pos))

if (y_rep_available) {
  cat("\n  Using y_rep from Stan generated quantities for PPC ...\n")

  y_rep_draws <- fit_m3a$draws("y_rep", format = "matrix")  # n_draws x N
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
  alpha_draws <- fit_m3a$draws("alpha", format = "matrix")  # n_draws x P
  beta_draws  <- fit_m3a$draws("beta",  format = "matrix")  # n_draws x P
  lk_draws    <- fit_m3a$draws("log_kappa", format = "matrix")  # n_draws x 1

  ## Extract all delta draws
  ## delta[s,k] for s=1:S, k=1:K
  delta_all_names_ppc <- c()
  for (s in 1:S) {
    for (k in 1:K) {
      delta_all_names_ppc <- c(delta_all_names_ppc, sprintf("delta[%d,%d]", s, k))
    }
  }
  delta_all_draws <- fit_m3a$draws(delta_all_names_ppc, format = "matrix")

  n_draws_ppc <- nrow(alpha_draws)
  n_ppc_use <- min(200, n_draws_ppc)
  draw_idx <- seq(1, n_draws_ppc, length.out = n_ppc_use) |> round() |> unique()
  n_ppc_use <- length(draw_idx)

  cat(sprintf("  Number of posterior draws: %d, using %d for PPC\n",
              n_draws_ppc, n_ppc_use))

  X <- stan_data_m3a$X
  state_idx <- stan_data_m3a$state

  ppc_zero_rates <- numeric(n_ppc_use)
  ppc_mean_share <- numeric(n_ppc_use)

  for (d in seq_along(draw_idx)) {
    dd <- draw_idx[d]

    ## Compute linear predictors with state-varying coefficients
    ## delta[s,1:P] = extensive, delta[s,(P+1):(2P)] = intensive
    alpha_d <- as.numeric(alpha_draws[dd, ])
    beta_d  <- as.numeric(beta_draws[dd, ])
    kappa_d <- exp(lk_draws[dd, 1])

    eta_ext <- numeric(N)
    eta_int <- numeric(N)
    for (i in 1:N) {
      s <- state_idx[i]
      ## Column indices in the flattened matrix: (s-1)*K + 1:K
      col_start <- (s - 1) * K
      d_s <- as.numeric(delta_all_draws[dd, col_start + 1:K])
      d_ext_s <- d_s[1:P]
      d_int_s <- d_s[(P + 1):K]
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
## SECTION 11 : SAVE RESULTS
###############################################################################
cat("--- 11. Saving results ---\n")

## 11a. Save the CmdStanR fit object
fit_m3a$save_object(FIT_OUT)
cat(sprintf("  Saved fit object: %s\n", FIT_OUT))
cat(sprintf("    File size: %.1f MB\n",
            file.info(FIT_OUT)$size / 1024^2))

## 11b. Assemble and save results list
results_m3a <- list(
  ## Model info
  model      = "M3a",
  model_desc = "Cross-Margin Covariance SVC Hurdle Beta-Binomial (unweighted)",

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
  tau_means      = setNames(tau_means, tau_labels),

  ## Full Omega correlation matrix (posterior mean)
  omega_mat = omega_mat,

  ## Cross-margin correlation block (posterior summaries)
  cross_block_mean = cross_block_mean,
  cross_block_sd   = cross_block_sd,
  cross_block_pneg = cross_block_pneg,

  ## Cross-margin diagonal (same-covariate, cross-margin)
  cross_diag_means    = setNames(cross_diag_means, cov_names),
  cross_diag_sd       = setNames(cross_diag_sd, cov_names),
  cross_diag_q025     = setNames(cross_diag_q025, cov_names),
  cross_diag_q975     = setNames(cross_diag_q975, cov_names),
  cross_diag_prob_neg = setNames(cross_diag_prob_neg, cov_names),

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
  loo = loo_m3a,

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
  stan_data_m3a = stan_data_m3a,

  ## Timestamp
  timestamp = Sys.time()
)

saveRDS(results_m3a, RESULTS_OUT)
cat(sprintf("  Saved results: %s\n", RESULTS_OUT))
cat(sprintf("    File size: %.1f KB\n",
            file.info(RESULTS_OUT)$size / 1024))


###############################################################################
## SECTION 12 : FINAL SUMMARY
###############################################################################
cat("\n==============================================================\n")
cat("  M3a FITTING SUMMARY\n")
cat("==============================================================\n")

n_fixed  <- P * 2 + 1                    # alpha + beta + log_kappa
n_hier   <- K + K * (K - 1) / 2          # tau[K] + L_Omega free elements
n_re     <- K * S                         # delta[S, K]
n_params_total <- n_fixed + n_hier + n_re

cat(sprintf("\n  Model: Cross-Margin Covariance SVC Hurdle Beta-Binomial (M3a)\n"))
cat(sprintf("  N = %d, P = %d, S = %d, K = 2P = %d\n", N, P, S, K))
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
  cat(sprintf("      tau[%d] %-10s = %.4f\n", k, cov_names[k], tau_means[k]))
}
cat("    Intensive margin:\n")
for (k in 1:P) {
  cat(sprintf("      tau[%d] %-10s = %.4f\n", P + k, cov_names[k], tau_means[P + k]))
}

cat(sprintf("\n  Cross-Margin Correlations (rho_cross = Omega[k, P+k]):\n"))
for (k in 1:P) {
  excludes_zero <- (cross_diag_q025[k] > 0) | (cross_diag_q975[k] < 0)
  sig_mark <- ifelse(excludes_zero, "*", " ")
  cat(sprintf("    rho_cross[%d] %-10s = %+.4f  95%% CI [%+.4f, %+.4f]  P(<0)=%.3f %s\n",
              k, cov_names[k],
              cross_diag_means[k],
              cross_diag_q025[k], cross_diag_q975[k],
              cross_diag_prob_neg[k],
              sig_mark))
}

cat(sprintf("\n  Poverty Reversal (state-level):\n"))
cat(sprintf("    States with reversal (posterior mean): %d / %d (%.0f%%)\n",
            n_reversal_mean, S, 100 * n_reversal_mean / S))
cat(sprintf("    States with reversal (P > 0.5):       %d / %d (%.0f%%)\n",
            n_reversal_prob50, S, 100 * n_reversal_prob50 / S))

cat(sprintf("\n  PPC:\n"))
cat(sprintf("    Zero rate: observed=%.3f, predicted=%.3f\n",
            observed_zero_rate, mean(ppc_zero_rates)))
cat(sprintf("    IT share:  observed=%.4f, predicted=%.4f\n",
            observed_mean_share_pos, mean(ppc_share_valid)))

if (!is.null(loo_m3a)) {
  cat(sprintf("\n  LOO-CV:\n"))
  cat(sprintf("    ELPD_loo   = %.1f (SE = %.1f)\n",
              loo_m3a$estimates["elpd_loo", "Estimate"],
              loo_m3a$estimates["elpd_loo", "SE"]))
  cat(sprintf("    p_loo      = %.1f\n",
              loo_m3a$estimates["p_loo", "Estimate"]))
  cat(sprintf("    Pareto k > 0.7: %d / %d\n",
              sum(loo_m3a$diagnostics$pareto_k > 0.7),
              length(loo_m3a$diagnostics$pareto_k)))

  if (file.exists(RESULTS_M2_PATH) && file.exists(RESULTS_M1_PATH) &&
      file.exists(RESULTS_M0_PATH)) {
    results_m0 <- readRDS(RESULTS_M0_PATH)
    results_m1 <- readRDS(RESULTS_M1_PATH)
    results_m2 <- readRDS(RESULTS_M2_PATH)
    if (!is.null(results_m0$loo) && !is.null(results_m1$loo) &&
        !is.null(results_m2$loo)) {
      elpd_m0 <- results_m0$loo$estimates["elpd_loo", "Estimate"]
      elpd_m1 <- results_m1$loo$estimates["elpd_loo", "Estimate"]
      elpd_m2 <- results_m2$loo$estimates["elpd_loo", "Estimate"]
      elpd_m3a_val <- loo_m3a$estimates["elpd_loo", "Estimate"]

      cat(sprintf("\n  LOO Comparison:\n"))
      cat(sprintf("    M0  ELPD = %.1f\n", elpd_m0))
      cat(sprintf("    M1  ELPD = %.1f  (M1-M0 = %+.1f)\n",
                  elpd_m1, elpd_m1 - elpd_m0))
      cat(sprintf("    M2  ELPD = %.1f  (M2-M1 = %+.1f)\n",
                  elpd_m2, elpd_m2 - elpd_m1))
      cat(sprintf("    M3a ELPD = %.1f  (M3a-M2 = %+.1f, M3a-M0 = %+.1f)\n",
                  elpd_m3a_val, elpd_m3a_val - elpd_m2, elpd_m3a_val - elpd_m0))
    }
  }
}

cat(sprintf("\n  Output files:\n"))
cat(sprintf("    %s\n", FIT_OUT))
cat(sprintf("    %s\n", RESULTS_OUT))

cat("\n==============================================================\n")
cat("  M3a FITTING COMPLETE.\n")
if (diag_pass) {
  cat("  ALL DIAGNOSTICS PASSED. Ready for M3b.\n")
} else {
  cat("  SOME DIAGNOSTICS FAILED. Review before proceeding to M3b.\n")
}
cat("==============================================================\n")
