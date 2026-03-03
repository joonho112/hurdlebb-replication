## =============================================================================
## 10_fit_m0.R -- Fit Pooled Hurdle Beta-Binomial Model (M0)
## =============================================================================
## Purpose : Compile and fit the Pooled HBB model (M0, no state effects)
##           on NSECE 2019 data (unweighted), run diagnostics, LOO-CV,
##           and posterior predictive checks.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Track   : A (requires NSECE restricted-use data; see data/DATA_ACCESS.md)
## Inputs  : data/precomputed/stan_data.rds
##           stan/hbb_m0.stan
##           code/helpers/utils.R
## Outputs : data/precomputed/fit_m0.rds
##           data/precomputed/results_m0.rds
## =============================================================================

cat("==============================================================\n")
cat("  HBB Replication: M0 Fitting  (Phase 1)\n")
cat("  Pooled Hurdle Beta-Binomial (no state effects)\n")
cat("==============================================================\n\n")

# ── 0. Setup ────────────────────────────────────────────────────────────────

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()

## Source helper functions
source(file.path(PROJECT_ROOT, "code/helpers/utils.R"))

## Load required packages
library(cmdstanr)
library(posterior)
library(loo)
library(dplyr, warn.conflicts = FALSE)

## Paths
STAN_DATA_PATH <- file.path(PROJECT_ROOT, "data/precomputed/stan_data.rds")
STAN_MODEL_PATH <- file.path(PROJECT_ROOT, "stan/hbb_m0.stan")
OUTPUT_DIR <- file.path(PROJECT_ROOT, "data/precomputed")
FIT_OUT <- file.path(OUTPUT_DIR, "fit_m0.rds")
RESULTS_OUT <- file.path(OUTPUT_DIR, "results_m0.rds")

## Ensure output directory exists
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)


###############################################################################
## SECTION 1 : LOAD DATA
###############################################################################
cat("--- 1. Loading Stan data ---\n")

stopifnot("Stan data file not found" = file.exists(STAN_DATA_PATH))
full_stan_data <- readRDS(STAN_DATA_PATH)
cat(sprintf("  Loaded: %s\n", STAN_DATA_PATH))
cat(sprintf("  N = %d, P = %d, N_pos = %d\n",
            full_stan_data$N, full_stan_data$P, full_stan_data$N_pos))

## Build M0-specific data list: only N, P, y, n_trial, z, X
## M0 is the pooled model — no state effects, no survey weights
stan_data_m0 <- list(
  N       = full_stan_data$N,
  P       = full_stan_data$P,
  y       = full_stan_data$y,
  n_trial = full_stan_data$n_trial,
  z       = full_stan_data$z,
  X       = full_stan_data$X
)

cat(sprintf("  M0 data list: N=%d, P=%d\n", stan_data_m0$N, stan_data_m0$P))
cat(sprintf("  Zero rate: %.1f%%\n", 100 * mean(stan_data_m0$z == 0)))
cat("  [PASS] Data loaded and M0 data list constructed.\n\n")


###############################################################################
## SECTION 2 : COMPILE STAN MODEL
###############################################################################
cat("--- 2. Compiling Stan model ---\n")

stopifnot("Stan model file not found" = file.exists(STAN_MODEL_PATH))
cat(sprintf("  Model file: %s\n", STAN_MODEL_PATH))

## Compile (cmdstanr caches compiled models; recompiles only if .stan changed)
m0_model <- tryCatch(
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
## SECTION 3 : FIT M0
###############################################################################
cat("--- 3. Fitting M0 ---\n")
cat("  chains = 4, parallel_chains = 4\n")
cat("  iter_warmup = 500, iter_sampling = 1000\n")
cat("  adapt_delta = 0.90, max_treedepth = 12\n")
cat("  seed = 20250220\n\n")

fit_start <- Sys.time()

fit_m0 <- tryCatch(
  m0_model$sample(
    data            = stan_data_m0,
    seed            = 20250220,
    chains          = 4,
    parallel_chains = 4,
    iter_warmup     = 500,
    iter_sampling   = 1000,
    adapt_delta     = 0.90,
    max_treedepth   = 12,
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
## SECTION 4 : MCMC DIAGNOSTICS
###############################################################################
cat("--- 4. MCMC Diagnostics ---\n")

diag_pass <- TRUE

## 4a. Divergent transitions
n_divergent <- fit_m0$diagnostic_summary()$num_divergent
total_divergent <- sum(n_divergent)
cat(sprintf("  Divergent transitions: %d (per chain: %s)\n",
            total_divergent, paste(n_divergent, collapse = ", ")))
if (total_divergent > 0) {
  warning("  [WARN] Divergent transitions detected! Consider increasing adapt_delta.")
  diag_pass <- FALSE
} else {
  cat("  [PASS] No divergent transitions.\n")
}

## 4b. Max treedepth exceedances
n_max_treedepth <- fit_m0$diagnostic_summary()$num_max_treedepth
total_max_td <- sum(n_max_treedepth)
cat(sprintf("  Max treedepth hits: %d (per chain: %s)\n",
            total_max_td, paste(n_max_treedepth, collapse = ", ")))
if (total_max_td > 0) {
  warning("  [WARN] Max treedepth reached. Consider increasing max_treedepth.")
}

## 4c. Parameter summary (alpha, beta, log_kappa)
param_names <- c(paste0("alpha[", 1:stan_data_m0$P, "]"),
                 paste0("beta[", 1:stan_data_m0$P, "]"),
                 "log_kappa")
param_summary <- fit_m0$summary(variables = param_names)

cat("\n  Parameter summary:\n")
print(param_summary, n = nrow(param_summary))

## 4d. R-hat check
max_rhat <- max(param_summary$rhat, na.rm = TRUE)
cat(sprintf("\n  Max R-hat: %.4f (threshold: 1.01)\n", max_rhat))
if (max_rhat > 1.01) {
  warning("  [WARN] R-hat > 1.01 detected. Chains may not have converged.")
  diag_pass <- FALSE
} else {
  cat("  [PASS] All R-hat < 1.01.\n")
}

## 4e. ESS check (bulk and tail)
min_ess_bulk <- min(param_summary$ess_bulk, na.rm = TRUE)
min_ess_tail <- min(param_summary$ess_tail, na.rm = TRUE)
cat(sprintf("  Min ESS (bulk): %.0f (threshold: 400)\n", min_ess_bulk))
cat(sprintf("  Min ESS (tail): %.0f (threshold: 400)\n", min_ess_tail))

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
## SECTION 5 : KEY PARAMETER CHECKS
###############################################################################
cat("--- 5. Key parameter checks ---\n")

## Extract posterior means
alpha_means <- param_summary$mean[grepl("^alpha", param_summary$variable)]
beta_means  <- param_summary$mean[grepl("^beta",  param_summary$variable)]
log_kappa_mean <- param_summary$mean[param_summary$variable == "log_kappa"]
kappa_mean  <- exp(log_kappa_mean)

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

## 5a. Poverty reversal check
## alpha_poverty (alpha[2]) should be NEGATIVE: more poverty -> less likely to serve IT
## beta_poverty  (beta[2])  should be POSITIVE: more poverty -> higher IT share
alpha_poverty <- alpha_means[2]
beta_poverty  <- beta_means[2]

cat("\n  === POVERTY REVERSAL CHECK ===\n")
cat(sprintf("    alpha_poverty = %+.4f (expected: negative)\n", alpha_poverty))
cat(sprintf("    beta_poverty  = %+.4f (expected: positive)\n", beta_poverty))

if (alpha_poverty < 0) {
  cat("    [PASS] alpha_poverty < 0 (higher poverty -> less likely to serve IT)\n")
} else {
  warning("    [WARN] alpha_poverty >= 0 — unexpected sign!")
}

if (beta_poverty > 0) {
  cat("    [PASS] beta_poverty > 0 (higher poverty -> higher IT share)\n")
} else {
  warning("    [WARN] beta_poverty <= 0 — unexpected sign!")
}

## 5b. Overdispersion check
## Expect kappa around 5-15 (12x overdispersion from binomial)
cat(sprintf("\n  === OVERDISPERSION CHECK ===\n"))
cat(sprintf("    kappa = %.2f (expected: ~5-15 for 12x overdispersion)\n", kappa_mean))

if (kappa_mean > 3 && kappa_mean < 30) {
  cat("    [PASS] kappa in reasonable range.\n")
} else {
  warning(sprintf("    [WARN] kappa = %.2f outside expected range [3, 30].", kappa_mean))
}

cat("\n")


###############################################################################
## SECTION 6 : LOO-CV
###############################################################################
cat("--- 6. LOO-CV computation ---\n")

## Extract log_lik array
log_lik <- fit_m0$draws("log_lik", format = "matrix")
cat(sprintf("  log_lik dimensions: %d draws x %d observations\n",
            nrow(log_lik), ncol(log_lik)))

## Compute LOO
loo_m0 <- tryCatch(
  loo(log_lik, cores = 4),
  error = function(e) {
    warning("LOO computation failed: ", conditionMessage(e))
    NULL
  }
)

if (!is.null(loo_m0)) {
  cat("\n  LOO-CV summary:\n")
  print(loo_m0)

  ## Check Pareto k diagnostics
  k_values <- loo_m0$diagnostics$pareto_k
  n_bad_k <- sum(k_values > 0.7)
  cat(sprintf("\n  Pareto k > 0.7: %d / %d observations (%.1f%%)\n",
              n_bad_k, length(k_values), 100 * n_bad_k / length(k_values)))

  if (n_bad_k > 0) {
    cat(sprintf("  [NOTE] %d observations with problematic Pareto k values.\n", n_bad_k))
    cat("         Consider moment matching or reloo for these observations.\n")
  } else {
    cat("  [PASS] All Pareto k values < 0.7.\n")
  }
} else {
  cat("  [WARN] LOO-CV computation failed. Skipping.\n")
}

cat("\n")


###############################################################################
## SECTION 7 : POSTERIOR PREDICTIVE CHECKS
###############################################################################
cat("--- 7. Posterior predictive checks ---\n")

## Extract posterior draws for alpha, beta, log_kappa
alpha_draws <- fit_m0$draws("alpha", format = "matrix")  # n_draws x P
beta_draws  <- fit_m0$draws("beta",  format = "matrix")  # n_draws x P
lk_draws    <- fit_m0$draws("log_kappa", format = "matrix")  # n_draws x 1

n_draws <- nrow(alpha_draws)
cat(sprintf("  Number of posterior draws: %d\n", n_draws))

## Observed statistics
N <- stan_data_m0$N
X <- stan_data_m0$X
y_obs <- stan_data_m0$y
n_trial <- stan_data_m0$n_trial
z_obs <- stan_data_m0$z

observed_zero_rate <- mean(z_obs == 0)
observed_mean_share_pos <- mean(y_obs[z_obs == 1] / n_trial[z_obs == 1])

cat(sprintf("  Observed zero rate (structural): %.3f (%.1f%%)\n",
            observed_zero_rate, 100 * observed_zero_rate))
cat(sprintf("  Observed mean IT share (y/n | z=1): %.4f\n",
            observed_mean_share_pos))

## 7a. PPC: Simulate zero rate from posterior draws
## For a subsample of draws (computational efficiency)
n_ppc_draws <- min(200, n_draws)
draw_idx <- seq(1, n_draws, length.out = n_ppc_draws) |> round() |> unique()
n_ppc_draws <- length(draw_idx)

cat(sprintf("\n  Running PPC with %d posterior draws ...\n", n_ppc_draws))

ppc_zero_rates <- numeric(n_ppc_draws)
ppc_mean_share <- numeric(n_ppc_draws)

for (d in seq_along(draw_idx)) {
  dd <- draw_idx[d]

  ## Compute linear predictors for all observations
  ## Use drop() to convert named vector to plain numeric for %*% compatibility
  eta_ext <- as.numeric(X %*% as.numeric(alpha_draws[dd, ]))
  eta_int <- as.numeric(X %*% as.numeric(beta_draws[dd, ]))
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
## SECTION 8 : SAVE RESULTS
###############################################################################
cat("--- 8. Saving results ---\n")

## 8a. Save the CmdStanR fit object
fit_m0$save_object(FIT_OUT)
cat(sprintf("  Saved fit object: %s\n", FIT_OUT))
cat(sprintf("    File size: %.1f MB\n",
            file.info(FIT_OUT)$size / 1024^2))

## 8b. Assemble and save results list
results_m0 <- list(
  ## Model info
  model     = "M0",
  model_desc = "Pooled Hurdle Beta-Binomial (unweighted)",

  ## Timing
  fit_time_mins = as.numeric(fit_time),

  ## Parameter summary
  param_summary = param_summary,

  ## Key estimates
  alpha_means    = setNames(alpha_means, cov_names),
  beta_means     = setNames(beta_means, cov_names),
  log_kappa_mean = log_kappa_mean,
  kappa_mean     = kappa_mean,

  ## Diagnostics
  diagnostics = list(
    n_divergent     = total_divergent,
    n_max_treedepth = total_max_td,
    max_rhat        = max_rhat,
    min_ess_bulk    = min_ess_bulk,
    min_ess_tail    = min_ess_tail,
    all_pass        = diag_pass
  ),

  ## LOO-CV
  loo = loo_m0,

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
  stan_data_m0 = stan_data_m0,

  ## Timestamp
  timestamp = Sys.time()
)

saveRDS(results_m0, RESULTS_OUT)
cat(sprintf("  Saved results: %s\n", RESULTS_OUT))
cat(sprintf("    File size: %.1f KB\n",
            file.info(RESULTS_OUT)$size / 1024))


###############################################################################
## SECTION 9 : FINAL SUMMARY
###############################################################################
cat("\n==============================================================\n")
cat("  M0 FITTING SUMMARY\n")
cat("==============================================================\n")

cat(sprintf("\n  Model: Pooled Hurdle Beta-Binomial (M0)\n"))
cat(sprintf("  N = %d, P = %d, 11 parameters\n", N, stan_data_m0$P))
cat(sprintf("  Fit time: %.1f minutes\n", as.numeric(fit_time)))

cat(sprintf("\n  MCMC Diagnostics:\n"))
cat(sprintf("    Divergent transitions : %d %s\n",
            total_divergent, ifelse(total_divergent == 0, "[PASS]", "[WARN]")))
cat(sprintf("    Max R-hat             : %.4f %s\n",
            max_rhat, ifelse(max_rhat < 1.01, "[PASS]", "[WARN]")))
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

cat(sprintf("\n  PPC:\n"))
cat(sprintf("    Zero rate: observed=%.3f, predicted=%.3f\n",
            observed_zero_rate, mean(ppc_zero_rates)))
cat(sprintf("    IT share:  observed=%.4f, predicted=%.4f\n",
            observed_mean_share_pos, mean(ppc_share_valid)))

if (!is.null(loo_m0)) {
  cat(sprintf("\n  LOO-CV:\n"))
  cat(sprintf("    ELPD_loo   = %.1f (SE = %.1f)\n",
              loo_m0$estimates["elpd_loo", "Estimate"],
              loo_m0$estimates["elpd_loo", "SE"]))
  cat(sprintf("    p_loo      = %.1f\n",
              loo_m0$estimates["p_loo", "Estimate"]))
  cat(sprintf("    Pareto k > 0.7: %d / %d\n",
              sum(loo_m0$diagnostics$pareto_k > 0.7),
              length(loo_m0$diagnostics$pareto_k)))
}

cat(sprintf("\n  Output files:\n"))
cat(sprintf("    %s\n", FIT_OUT))
cat(sprintf("    %s\n", RESULTS_OUT))

cat("\n==============================================================\n")
cat("  M0 FITTING COMPLETE.\n")
if (diag_pass) {
  cat("  ALL DIAGNOSTICS PASSED. Ready for M1.\n")
} else {
  cat("  SOME DIAGNOSTICS FAILED. Review before proceeding to M1.\n")
}
cat("==============================================================\n")
