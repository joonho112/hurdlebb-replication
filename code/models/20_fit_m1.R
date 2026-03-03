## =============================================================================
## 20_fit_m1.R -- Fit Random Intercepts Hurdle Beta-Binomial Model (M1)
## =============================================================================
## Purpose : Compile and fit the Random Intercepts HBB model (M1)
##           on NSECE 2019 data (unweighted), run diagnostics, LOO-CV
##           comparison with M0, and posterior predictive checks.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Track   : A (requires NSECE restricted-use data; see data/DATA_ACCESS.md)
## Inputs  : data/precomputed/stan_data.rds
##           stan/hbb_m1.stan
##           code/helpers/utils.R
##           data/precomputed/results_m0.rds
## Outputs : data/precomputed/fit_m1.rds
##           data/precomputed/results_m1.rds
## =============================================================================

cat("==============================================================\n")
cat("  HBB Replication: M1 Fitting  (Phase 2)\n")
cat("  Random Intercepts Hurdle Beta-Binomial\n")
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
STAN_MODEL_PATH <- file.path(PROJECT_ROOT, "stan/hbb_m1.stan")
OUTPUT_DIR      <- file.path(PROJECT_ROOT, "data/precomputed")
FIT_OUT         <- file.path(OUTPUT_DIR, "fit_m1.rds")
RESULTS_OUT     <- file.path(OUTPUT_DIR, "results_m1.rds")
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

## Build M1-specific data list: N, P, S, y, n_trial, z, X, state
## M1 adds state random intercepts — no survey weights yet
stan_data_m1 <- list(
  N       = full_stan_data$N,
  P       = full_stan_data$P,
  S       = full_stan_data$S,
  y       = full_stan_data$y,
  n_trial = full_stan_data$n_trial,
  z       = full_stan_data$z,
  X       = full_stan_data$X,
  state   = full_stan_data$state
)

cat(sprintf("  M1 data list: N=%d, P=%d, S=%d\n",
            stan_data_m1$N, stan_data_m1$P, stan_data_m1$S))
cat(sprintf("  Zero rate: %.1f%%\n", 100 * mean(stan_data_m1$z == 0)))

## State sample sizes
state_ns <- table(stan_data_m1$state)
cat(sprintf("  State sample sizes: min=%d, median=%d, max=%d\n",
            min(state_ns), median(state_ns), max(state_ns)))
cat("  [PASS] Data loaded and M1 data list constructed.\n\n")


###############################################################################
## SECTION 2 : COMPILE STAN MODEL
###############################################################################
cat("--- 2. Compiling Stan model ---\n")

stopifnot("Stan model file not found" = file.exists(STAN_MODEL_PATH))
cat(sprintf("  Model file: %s\n", STAN_MODEL_PATH))

## Compile (cmdstanr caches compiled models; recompiles only if .stan changed)
m1_model <- tryCatch(
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
## SECTION 3 : INITIALISATION FROM M0 POSTERIORS (WARM START)
###############################################################################
cat("--- 3. Setting initial values ---\n")

if (file.exists(RESULTS_M0_PATH)) {
  results_m0 <- readRDS(RESULTS_M0_PATH)
  cat("  [INFO] M0 results found. Using M0 posteriors for warm start.\n")

  ## Extract M0 posterior means
  m0_alpha     <- as.numeric(results_m0$alpha_means)
  m0_beta      <- as.numeric(results_m0$beta_means)
  m0_log_kappa <- as.numeric(results_m0$log_kappa_mean)

  init_fun <- function() {
    list(
      alpha     = m0_alpha,
      beta      = m0_beta,
      log_kappa = m0_log_kappa,
      tau       = c(0.3, 0.3),
      L_Omega   = diag(2),
      z_delta   = matrix(0, nrow = 2, ncol = stan_data_m1$S)
    )
  }

  cat(sprintf("  alpha init    : [%s]\n",
              paste(sprintf("%+.3f", m0_alpha), collapse = ", ")))
  cat(sprintf("  beta init     : [%s]\n",
              paste(sprintf("%+.3f", m0_beta), collapse = ", ")))
  cat(sprintf("  log_kappa init: %.3f\n", m0_log_kappa))
  cat("  tau init      : [0.300, 0.300]\n")
  cat("  z_delta init  : matrix of zeros (2 x S)\n")
} else {
  cat("  [INFO] M0 results not found. Using default initialisation.\n")

  init_fun <- function() {
    list(
      alpha     = rep(0, stan_data_m1$P),
      beta      = rep(0, stan_data_m1$P),
      log_kappa = log(10),
      tau       = c(0.3, 0.3),
      L_Omega   = diag(2),
      z_delta   = matrix(0, nrow = 2, ncol = stan_data_m1$S)
    )
  }
}

cat("  [PASS] Initial values set.\n\n")


###############################################################################
## SECTION 4 : FIT M1
###############################################################################
cat("--- 4. Fitting M1 ---\n")
cat("  chains = 4, parallel_chains = 4\n")
cat("  iter_warmup = 1000, iter_sampling = 1000\n")
cat("  adapt_delta = 0.95, max_treedepth = 12\n")
cat("  seed = 20250220\n\n")

fit_start <- Sys.time()

fit_m1 <- tryCatch(
  m1_model$sample(
    data            = stan_data_m1,
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
n_divergent <- fit_m1$diagnostic_summary()$num_divergent
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
n_max_treedepth <- fit_m1$diagnostic_summary()$num_max_treedepth
total_max_td <- sum(n_max_treedepth)
cat(sprintf("  Max treedepth hits: %d (per chain: %s)\n",
            total_max_td, paste(n_max_treedepth, collapse = ", ")))
if (total_max_td > 0) {
  warning("  [WARN] Max treedepth reached. Consider increasing max_treedepth.")
}

## 5c. Parameter summary — fixed effects + hierarchical parameters
## Fixed effects
param_names_fixed <- c(paste0("alpha[", 1:stan_data_m1$P, "]"),
                       paste0("beta[", 1:stan_data_m1$P, "]"),
                       "log_kappa")
## Hierarchical parameters
param_names_hier <- c("tau[1]", "tau[2]", "L_Omega[1,1]", "L_Omega[1,2]",
                      "L_Omega[2,1]", "L_Omega[2,2]")

## Selected delta values (a few states for monitoring)
delta_sample_states <- c(1, 10, 25, 40, 51)
param_names_delta <- c()
for (s in delta_sample_states) {
  param_names_delta <- c(param_names_delta,
                         sprintf("delta[1,%d]", s),
                         sprintf("delta[2,%d]", s))
}

## Full summary for diagnostics (fixed + hierarchical)
param_names_all <- c(param_names_fixed, param_names_hier)
param_summary <- fit_m1$summary(variables = param_names_all)

cat("\n  Fixed effect parameter summary:\n")
param_summary_fixed <- fit_m1$summary(variables = param_names_fixed)
print(param_summary_fixed, n = nrow(param_summary_fixed))

cat("\n  Hierarchical parameter summary:\n")
param_summary_hier <- fit_m1$summary(variables = param_names_hier)
print(param_summary_hier, n = nrow(param_summary_hier))

cat("\n  Selected delta (state random intercepts) summary:\n")
param_summary_delta <- fit_m1$summary(variables = param_names_delta)
print(param_summary_delta, n = nrow(param_summary_delta))

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
for (s in 1:stan_data_m1$S) {
  delta_all_names <- c(delta_all_names,
                       sprintf("delta[1,%d]", s),
                       sprintf("delta[2,%d]", s))
}
delta_all_summary <- fit_m1$summary(variables = delta_all_names)
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
tau_means <- param_summary_hier$mean[grepl("^tau", param_summary_hier$variable)]

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

cat(sprintf("\n  Hierarchical scale parameters:\n"))
cat(sprintf("    tau[1] (extensive) = %.4f\n", tau_means[1]))
cat(sprintf("    tau[2] (intensive) = %.4f\n", tau_means[2]))

## 6a. Extract correlation between random intercepts
## Omega = L_Omega * L_Omega', so Omega[1,2] = rho
## In Stan, with cholesky_factor_corr[2], Omega = multiply_lower_tri_self_transpose(L_Omega)
## We compute rho from the generated quantities if available, or from L_Omega draws
## Omega[1,2] = L_Omega[2,1] * L_Omega[1,1]
## For a 2x2 Cholesky of a correlation matrix:
##   L = [[1, 0], [L21, sqrt(1-L21^2)]]
##   Omega[1,2] = L21 * 1 = L21

## Try to extract Omega directly if available in generated quantities
omega_available <- tryCatch({
  fit_m1$draws("Omega[1,2]", format = "matrix")
  TRUE
}, error = function(e) FALSE)

if (omega_available) {
  rho_draws <- fit_m1$draws("Omega[1,2]", format = "matrix")
  rho_mean <- mean(rho_draws)
  rho_ci <- quantile(rho_draws, c(0.025, 0.975))
} else {
  ## Compute from L_Omega
  L21_draws <- fit_m1$draws("L_Omega[2,1]", format = "matrix")
  rho_draws <- L21_draws  # For 2x2 corr Cholesky, L[2,1] = rho
  rho_mean <- mean(rho_draws)
  rho_ci <- quantile(rho_draws, c(0.025, 0.975))
}

cat(sprintf("\n  Correlation (rho) between extensive and intensive intercepts:\n"))
cat(sprintf("    rho = %+.4f  95%% CI: [%+.4f, %+.4f]\n",
            rho_mean, rho_ci[1], rho_ci[2]))

## 6b. Poverty reversal check
alpha_poverty <- alpha_means[2]
beta_poverty  <- beta_means[2]

cat("\n  === POVERTY REVERSAL CHECK ===\n")
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

## 6c. State variation check: tau > 0
cat("\n  === STATE VARIATION CHECK ===\n")
cat(sprintf("    tau[1] = %.4f (expected: > 0)\n", tau_means[1]))
cat(sprintf("    tau[2] = %.4f (expected: > 0)\n", tau_means[2]))

if (tau_means[1] > 0.01) {
  cat("    [PASS] tau[1] > 0: meaningful state variation in extensive margin.\n")
} else {
  cat("    [NOTE] tau[1] near zero: limited state variation in extensive margin.\n")
}

if (tau_means[2] > 0.01) {
  cat("    [PASS] tau[2] > 0: meaningful state variation in intensive margin.\n")
} else {
  cat("    [NOTE] tau[2] near zero: limited state variation in intensive margin.\n")
}

## 6d. Correlation check: expected negative
cat("\n  === CROSS-MARGIN CORRELATION CHECK ===\n")
cat(sprintf("    rho = %+.4f (expected: negative per theory)\n", rho_mean))

if (rho_mean < 0) {
  cat("    [PASS] rho < 0: negative cross-margin correlation as expected.\n")
} else {
  cat("    [NOTE] rho >= 0: positive or zero cross-margin correlation.\n")
  cat("           Theory predicts negative, but data may differ.\n")
}

## 6e. Overdispersion check
cat(sprintf("\n  === OVERDISPERSION CHECK ===\n"))
cat(sprintf("    kappa = %.2f (expected: ~5-15 for 12x overdispersion)\n", kappa_mean))

if (kappa_mean > 3 && kappa_mean < 30) {
  cat("    [PASS] kappa in reasonable range.\n")
} else {
  warning(sprintf("    [WARN] kappa = %.2f outside expected range [3, 30].", kappa_mean))
}

cat("\n")


###############################################################################
## SECTION 7 : LOO-CV AND COMPARISON WITH M0
###############################################################################
cat("--- 7. LOO-CV computation and M0 comparison ---\n")

## Extract log_lik array
log_lik <- fit_m1$draws("log_lik", format = "matrix")
cat(sprintf("  log_lik dimensions: %d draws x %d observations\n",
            nrow(log_lik), ncol(log_lik)))

## Compute LOO
loo_m1 <- tryCatch(
  loo(log_lik, cores = 4),
  error = function(e) {
    warning("LOO computation failed: ", conditionMessage(e))
    NULL
  }
)

if (!is.null(loo_m1)) {
  cat("\n  LOO-CV summary (M1):\n")
  print(loo_m1)

  ## Check Pareto k diagnostics
  k_values <- loo_m1$diagnostics$pareto_k
  n_bad_k <- sum(k_values > 0.7)
  cat(sprintf("\n  Pareto k > 0.7: %d / %d observations (%.1f%%)\n",
              n_bad_k, length(k_values), 100 * n_bad_k / length(k_values)))

  if (n_bad_k > 0) {
    cat(sprintf("  [NOTE] %d observations with problematic Pareto k values.\n", n_bad_k))
    cat("         Consider moment matching or reloo for these observations.\n")
  } else {
    cat("  [PASS] All Pareto k values < 0.7.\n")
  }

  ## 7b. LOO comparison with M0
  if (file.exists(RESULTS_M0_PATH)) {
    results_m0 <- readRDS(RESULTS_M0_PATH)
    loo_m0 <- results_m0$loo

    if (!is.null(loo_m0)) {
      cat("\n  === LOO-CV COMPARISON: M1 vs M0 ===\n")

      elpd_m0 <- loo_m0$estimates["elpd_loo", "Estimate"]
      elpd_m1 <- loo_m1$estimates["elpd_loo", "Estimate"]
      cat(sprintf("    M0 ELPD_loo = %.1f\n", elpd_m0))
      cat(sprintf("    M1 ELPD_loo = %.1f\n", elpd_m1))
      cat(sprintf("    Difference  = %.1f (positive = M1 better)\n",
                  elpd_m1 - elpd_m0))

      ## Formal loo_compare
      comp <- loo_compare(loo_m0, loo_m1)
      cat("\n  loo_compare output:\n")
      print(comp)

      ## Interpretation
      elpd_diff <- comp[2, "elpd_diff"]
      se_diff   <- comp[2, "se_diff"]
      cat(sprintf("\n    ELPD difference: %.1f (SE = %.1f)\n",
                  elpd_diff, se_diff))

      if (abs(elpd_diff) > 2 * se_diff) {
        better_model <- rownames(comp)[1]
        cat(sprintf("    [PASS] Difference > 2*SE: %s is clearly preferred.\n",
                    better_model))
      } else {
        cat("    [NOTE] Difference < 2*SE: models are comparable.\n")
      }
    } else {
      cat("  [NOTE] M0 LOO object not available for comparison.\n")
    }
  } else {
    cat("  [NOTE] M0 results not found. Skipping LOO comparison.\n")
  }
} else {
  cat("  [WARN] LOO-CV computation failed. Skipping.\n")
}

cat("\n")


###############################################################################
## SECTION 8 : POSTERIOR PREDICTIVE CHECKS
###############################################################################
cat("--- 8. Posterior predictive checks ---\n")

## Extract posterior draws for alpha, beta, log_kappa, and delta
alpha_draws <- fit_m1$draws("alpha", format = "matrix")  # n_draws x P
beta_draws  <- fit_m1$draws("beta",  format = "matrix")  # n_draws x P
lk_draws    <- fit_m1$draws("log_kappa", format = "matrix")  # n_draws x 1

## Extract all delta draws: delta[margin, state]
## delta[1, s] = extensive intercept for state s
## delta[2, s] = intensive intercept for state s
delta1_names <- paste0("delta[1,", 1:stan_data_m1$S, "]")
delta2_names <- paste0("delta[2,", 1:stan_data_m1$S, "]")
delta1_draws <- fit_m1$draws(delta1_names, format = "matrix")  # n_draws x S
delta2_draws <- fit_m1$draws(delta2_names, format = "matrix")  # n_draws x S

n_draws <- nrow(alpha_draws)
cat(sprintf("  Number of posterior draws: %d\n", n_draws))

## Observed statistics
N <- stan_data_m1$N
X <- stan_data_m1$X
y_obs <- stan_data_m1$y
n_trial <- stan_data_m1$n_trial
z_obs <- stan_data_m1$z
state_idx <- stan_data_m1$state

observed_zero_rate <- mean(z_obs == 0)
observed_mean_share_pos <- mean(y_obs[z_obs == 1] / n_trial[z_obs == 1])

cat(sprintf("  Observed zero rate (structural): %.3f (%.1f%%)\n",
            observed_zero_rate, 100 * observed_zero_rate))
cat(sprintf("  Observed mean IT share (y/n | z=1): %.4f\n",
            observed_mean_share_pos))

## 8a. PPC: Simulate from posterior draws
n_ppc_draws <- min(200, n_draws)
draw_idx <- seq(1, n_draws, length.out = n_ppc_draws) |> round() |> unique()
n_ppc_draws <- length(draw_idx)

cat(sprintf("\n  Running PPC with %d posterior draws ...\n", n_ppc_draws))

ppc_zero_rates <- numeric(n_ppc_draws)
ppc_mean_share <- numeric(n_ppc_draws)

for (d in seq_along(draw_idx)) {
  dd <- draw_idx[d]

  ## Compute linear predictors with state random intercepts
  ## eta_ext[i] = X_i * alpha + delta[1, state[i]]
  ## eta_int[i] = X_i * beta  + delta[2, state[i]]
  eta_ext <- as.numeric(X %*% as.numeric(alpha_draws[dd, ])) +
             as.numeric(delta1_draws[dd, state_idx])
  eta_int <- as.numeric(X %*% as.numeric(beta_draws[dd, ])) +
             as.numeric(delta2_draws[dd, state_idx])

  q_i     <- inv_logit(eta_ext)
  mu_i    <- inv_logit(eta_int)
  mu_i    <- pmin(pmax(mu_i, 1e-8), 1 - 1e-8)
  kappa_d <- exp(lk_draws[dd, 1])

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
## SECTION 9 : SAVE RESULTS
###############################################################################
cat("--- 9. Saving results ---\n")

## 9a. Save the CmdStanR fit object
fit_m1$save_object(FIT_OUT)
cat(sprintf("  Saved fit object: %s\n", FIT_OUT))
cat(sprintf("    File size: %.1f MB\n",
            file.info(FIT_OUT)$size / 1024^2))

## 9b. Assemble and save results list
results_m1 <- list(
  ## Model info
  model      = "M1",
  model_desc = "Random Intercepts Hurdle Beta-Binomial (unweighted)",

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
  tau_means      = setNames(tau_means, c("extensive", "intensive")),
  rho_mean       = rho_mean,
  rho_95ci       = rho_ci,

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
  loo = loo_m1,

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
  stan_data_m1 = stan_data_m1,

  ## Timestamp
  timestamp = Sys.time()
)

saveRDS(results_m1, RESULTS_OUT)
cat(sprintf("  Saved results: %s\n", RESULTS_OUT))
cat(sprintf("    File size: %.1f KB\n",
            file.info(RESULTS_OUT)$size / 1024))


###############################################################################
## SECTION 10 : FINAL SUMMARY
###############################################################################
cat("\n==============================================================\n")
cat("  M1 FITTING SUMMARY\n")
cat("==============================================================\n")

n_params_total <- stan_data_m1$P * 2 + 1 + 2 + 1 + 2 * stan_data_m1$S
cat(sprintf("\n  Model: Random Intercepts Hurdle Beta-Binomial (M1)\n"))
cat(sprintf("  N = %d, P = %d, S = %d\n", N, stan_data_m1$P, stan_data_m1$S))
cat(sprintf("  Parameters: %d fixed + %d hierarchical + %d random effects = %d total\n",
            stan_data_m1$P * 2 + 1, 3, 2 * stan_data_m1$S, n_params_total))
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
cat(sprintf("    tau[1] (ext)  = %.4f %s\n",
            tau_means[1], ifelse(tau_means[1] > 0.01, "[PASS]", "[NOTE]")))
cat(sprintf("    tau[2] (int)  = %.4f %s\n",
            tau_means[2], ifelse(tau_means[2] > 0.01, "[PASS]", "[NOTE]")))
cat(sprintf("    rho           = %+.4f %s\n",
            rho_mean, ifelse(rho_mean < 0, "[PASS] (<0)", "[NOTE]")))

cat(sprintf("\n  PPC:\n"))
cat(sprintf("    Zero rate: observed=%.3f, predicted=%.3f\n",
            observed_zero_rate, mean(ppc_zero_rates)))
cat(sprintf("    IT share:  observed=%.4f, predicted=%.4f\n",
            observed_mean_share_pos, mean(ppc_share_valid)))

if (!is.null(loo_m1)) {
  cat(sprintf("\n  LOO-CV:\n"))
  cat(sprintf("    ELPD_loo   = %.1f (SE = %.1f)\n",
              loo_m1$estimates["elpd_loo", "Estimate"],
              loo_m1$estimates["elpd_loo", "SE"]))
  cat(sprintf("    p_loo      = %.1f\n",
              loo_m1$estimates["p_loo", "Estimate"]))
  cat(sprintf("    Pareto k > 0.7: %d / %d\n",
              sum(loo_m1$diagnostics$pareto_k > 0.7),
              length(loo_m1$diagnostics$pareto_k)))

  if (file.exists(RESULTS_M0_PATH)) {
    results_m0 <- readRDS(RESULTS_M0_PATH)
    if (!is.null(results_m0$loo)) {
      cat(sprintf("\n  LOO Comparison (M1 vs M0):\n"))
      cat(sprintf("    M0 ELPD = %.1f,  M1 ELPD = %.1f\n",
                  results_m0$loo$estimates["elpd_loo", "Estimate"],
                  loo_m1$estimates["elpd_loo", "Estimate"]))
      cat(sprintf("    Improvement = %.1f\n",
                  loo_m1$estimates["elpd_loo", "Estimate"] -
                  results_m0$loo$estimates["elpd_loo", "Estimate"]))
    }
  }
}

cat(sprintf("\n  Output files:\n"))
cat(sprintf("    %s\n", FIT_OUT))
cat(sprintf("    %s\n", RESULTS_OUT))

cat("\n==============================================================\n")
cat("  M1 FITTING COMPLETE.\n")
if (diag_pass) {
  cat("  ALL DIAGNOSTICS PASSED. Ready for M2.\n")
} else {
  cat("  SOME DIAGNOSTICS FAILED. Review before proceeding to M2.\n")
}
cat("==============================================================\n")
