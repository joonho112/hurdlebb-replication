## =============================================================================
## 50_fit_m3b.R -- Fit Policy Moderator SVC Model (M3b)
## =============================================================================
## Purpose : Compile and fit the Policy Moderator SVC HBB model (M3b)
##           on NSECE 2019 data (unweighted), run diagnostics, LOO-CV
##           comparison, PPC, and policy moderator (Gamma) analysis.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Track   : A (requires NSECE restricted-use data; see data/DATA_ACCESS.md)
## Inputs  : data/precomputed/stan_data.rds
##           stan/hbb_m3b.stan
##           code/helpers/utils.R
##           data/precomputed/results_m3a.rds
##           data/precomputed/results_m2.rds
##           data/precomputed/results_m1.rds
##           data/precomputed/results_m0.rds
## Outputs : data/precomputed/fit_m3b.rds
##           data/precomputed/results_m3b.rds
## =============================================================================

cat("==============================================================\n")
cat("  HBB Replication: M3b Fitting  (Phase 5)\n")
cat("  Policy Moderator SVC Hurdle Beta-Binomial\n")
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
STAN_MODEL_PATH  <- file.path(PROJECT_ROOT, "stan/hbb_m3b.stan")
OUTPUT_DIR       <- file.path(PROJECT_ROOT, "data/precomputed")
FIT_OUT          <- file.path(OUTPUT_DIR, "fit_m3b.rds")
RESULTS_OUT      <- file.path(OUTPUT_DIR, "results_m3b.rds")
RESULTS_M3A_PATH <- file.path(OUTPUT_DIR, "results_m3a.rds")
RESULTS_M2_PATH  <- file.path(OUTPUT_DIR, "results_m2.rds")
RESULTS_M1_PATH  <- file.path(OUTPUT_DIR, "results_m1.rds")
RESULTS_M0_PATH  <- file.path(OUTPUT_DIR, "results_m0.rds")

## Ensure output directory exists
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)


###############################################################################
## SECTION 1 : LOAD DATA
###############################################################################
cat("--- 1. Loading Stan data ---\n")

stopifnot("Stan data file not found" = file.exists(STAN_DATA_PATH))
full_stan_data <- readRDS(STAN_DATA_PATH)
cat(sprintf("  Loaded: %s\n", STAN_DATA_PATH))
cat(sprintf("  N = %d, P = %d, S = %d, Q = %d, N_pos = %d\n",
            full_stan_data$N, full_stan_data$P, full_stan_data$S,
            full_stan_data$Q, full_stan_data$N_pos))

## Build M3b-specific data list: M3a data + Q and v_state
stan_data_m3b <- list(
  N       = full_stan_data$N,
  P       = full_stan_data$P,
  S       = full_stan_data$S,
  Q       = full_stan_data$Q,
  y       = full_stan_data$y,
  n_trial = full_stan_data$n_trial,
  z       = full_stan_data$z,
  X       = full_stan_data$X,
  state   = full_stan_data$state,
  v_state = full_stan_data$V    # S x Q state-level policy design matrix
)

P <- stan_data_m3b$P
S <- stan_data_m3b$S
Q <- stan_data_m3b$Q
K <- 2 * P  # Joint dimension: P extensive + P intensive = 10

cat(sprintf("  M3b data list: N=%d, P=%d, S=%d, Q=%d, K=2P=%d\n",
            stan_data_m3b$N, P, S, Q, K))
cat(sprintf("  Zero rate: %.1f%%\n", 100 * mean(stan_data_m3b$z == 0)))
cat(sprintf("  Gamma dimensions: K x Q = %d x %d = %d parameters\n",
            K, Q, K * Q))

## State sample sizes
state_ns <- table(stan_data_m3b$state)
cat(sprintf("  State sample sizes: min=%d, median=%d, max=%d\n",
            min(state_ns), median(state_ns), max(state_ns)))

## Policy covariate summary
cat("\n  Policy covariates (v_state: S x Q):\n")
v_col_names <- c("intercept", "MR_pctile_std", "TieredReim", "ITaddon")
for (q in 1:Q) {
  vals <- stan_data_m3b$v_state[, q]
  cat(sprintf("    v_state[,%-2d] %-15s mean=%.3f, sd=%.3f, range=[%.3f, %.3f]\n",
              q, v_col_names[q], mean(vals), sd(vals), min(vals), max(vals)))
}

## M3b random effects + policy moderator dimensionality
n_re_per_state <- K
n_re_total     <- K * S
n_corr_params  <- K * (K - 1) / 2
n_gamma_params <- K * Q
cat(sprintf("\n  Random effects: %d per state x %d states = %d total\n",
            n_re_per_state, S, n_re_total))
cat(sprintf("  Correlation parameters: %d (from %dx%d Omega)\n",
            n_corr_params, K, K))
cat(sprintf("  Policy moderator parameters (Gamma): %d x %d = %d\n",
            K, Q, n_gamma_params))
cat("  [PASS] Data loaded and M3b data list constructed.\n\n")


###############################################################################
## SECTION 2 : COMPILE STAN MODEL
###############################################################################
cat("--- 2. Compiling Stan model ---\n")

stopifnot("Stan model file not found" = file.exists(STAN_MODEL_PATH))
cat(sprintf("  Model file: %s\n", STAN_MODEL_PATH))

## Compile (cmdstanr caches compiled models; recompiles only if .stan changed)
m3b_model <- tryCatch(
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
## SECTION 3 : INITIALISATION FROM M3a POSTERIORS (WARM START)
###############################################################################
cat("--- 3. Setting initial values ---\n")

if (file.exists(RESULTS_M3A_PATH)) {
  results_m3a <- readRDS(RESULTS_M3A_PATH)
  cat("  [INFO] M3a results found. Using M3a posteriors for warm start.\n")

  ## Extract M3a posterior means
  m3a_alpha     <- as.numeric(results_m3a$alpha_means)
  m3a_beta      <- as.numeric(results_m3a$beta_means)
  m3a_log_kappa <- as.numeric(results_m3a$log_kappa_mean)
  m3a_tau       <- as.numeric(results_m3a$tau_means)  # length K

  init_fun <- function() {
    list(
      alpha     = m3a_alpha,
      beta      = m3a_beta,
      log_kappa = m3a_log_kappa,
      tau       = m3a_tau,
      L_Omega   = diag(K),
      Gamma     = matrix(0, K, Q),    # Start Gamma at zero (no policy effects)
      z_eps     = rep(list(rep(0, K)), S)
    )
  }

  cat(sprintf("  alpha init    : [%s]\n",
              paste(sprintf("%+.3f", m3a_alpha), collapse = ", ")))
  cat(sprintf("  beta init     : [%s]\n",
              paste(sprintf("%+.3f", m3a_beta), collapse = ", ")))
  cat(sprintf("  log_kappa init: %.3f\n", m3a_log_kappa))
  cat(sprintf("  tau init      : [%s]\n",
              paste(sprintf("%.3f", m3a_tau), collapse = ", ")))
  cat("  L_Omega       : identity matrix (K x K)\n")
  cat("  Gamma         : zero matrix (K x Q)\n")
  cat("  z_eps         : list of S zero vectors (length K)\n")
} else {
  cat("  [INFO] M3a results not found. Using default initialisation.\n")

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
## SECTION 4 : FIT M3b
###############################################################################
cat("--- 4. Fitting M3b ---\n")
cat("  chains = 4, parallel_chains = 4\n")
cat("  iter_warmup = 1000, iter_sampling = 1000\n")
cat("  adapt_delta = 0.95, max_treedepth = 12\n")
cat("  seed = 20250220\n\n")

fit_start <- Sys.time()

fit_m3b <- tryCatch(
  m3b_model$sample(
    data            = stan_data_m3b,
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
n_divergent <- fit_m3b$diagnostic_summary()$num_divergent
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
n_max_treedepth <- fit_m3b$diagnostic_summary()$num_max_treedepth
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

## Gamma parameters: Gamma[k,q] for k=1:K, q=1:Q
param_names_gamma <- c()
for (k in 1:K) {
  for (q in 1:Q) {
    param_names_gamma <- c(param_names_gamma, sprintf("Gamma[%d,%d]", k, q))
  }
}

## Full summary for diagnostics (fixed + hierarchical + Gamma)
param_names_all <- c(param_names_fixed, param_names_hier, param_names_gamma)
param_summary <- fit_m3b$summary(variables = param_names_all)

cat("\n  Fixed effect parameter summary:\n")
param_summary_fixed <- fit_m3b$summary(variables = param_names_fixed)
print(param_summary_fixed, n = nrow(param_summary_fixed))

cat("\n  Hierarchical parameter summary (tau[1:K]):\n")
param_summary_hier <- fit_m3b$summary(variables = param_names_hier)
print(param_summary_hier, n = nrow(param_summary_hier))

cat("\n  Gamma parameter summary:\n")
param_summary_gamma <- fit_m3b$summary(variables = param_names_gamma)
print(param_summary_gamma, n = nrow(param_summary_gamma))

## Selected delta values (a few states for monitoring)
delta_sample_states <- c(1, 10, 25, 40, 51)
param_names_delta_sample <- c()
for (s in delta_sample_states) {
  for (k in 1:K) {
    param_names_delta_sample <- c(param_names_delta_sample,
                                   sprintf("delta[%d,%d]", s, k))
  }
}

cat("\n  Selected delta (state random coefficients) summary:\n")
param_summary_delta_sample <- fit_m3b$summary(variables = param_names_delta_sample)
print(param_summary_delta_sample, n = min(nrow(param_summary_delta_sample), 20))

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
delta_all_summary <- fit_m3b$summary(variables = delta_all_names)
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
  cat(sprintf("    tau[%2d] %-15s = %.4f\n",
              k, tau_labels[k], tau_means[k]))
}

## 6a. Extract Omega correlation matrix (from generated quantities)
cat("\n  Omega (K x K correlation matrix) posterior means:\n")
omega_mat <- matrix(NA, K, K)
for (i in 1:K) {
  for (j in 1:K) {
    draws_ij <- fit_m3b$draws(sprintf("Omega[%d,%d]", i, j), format = "matrix")
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

## 6c. Overdispersion check
cat(sprintf("\n  === OVERDISPERSION CHECK ===\n"))
cat(sprintf("    kappa = %.2f (expected: ~5-15 for 12x overdispersion)\n", kappa_mean))

if (kappa_mean > 3 && kappa_mean < 30) {
  cat("    [PASS] kappa in reasonable range.\n")
} else {
  warning(sprintf("    [WARN] kappa = %.2f outside expected range [3, 30].", kappa_mean))
}

cat("\n")


###############################################################################
## SECTION 7 : POLICY MODERATOR ANALYSIS (KEY NEW ANALYSIS)
###############################################################################
cat("--- 7. Policy moderator analysis (Gamma matrix) ---\n")
cat("  Gamma is K x Q = 10 x 4 = 40 parameters.\n")
cat("  Rows 1-5:  extensive margin (intercept, poverty, urban, black, hispanic)\n")
cat("  Rows 6-10: intensive margin (intercept, poverty, urban, black, hispanic)\n")
cat("  Cols: 1=intercept, 2=MR_pctile_std, 3=TieredReim, 4=ITaddon\n\n")

## Extract all Gamma draws
gamma_draws <- fit_m3b$draws("Gamma", format = "matrix")

## Build structured Gamma summary table
gamma_row_labels <- c(paste0("ext_", cov_names), paste0("int_", cov_names))
gamma_col_labels <- v_col_names  # intercept, MR_pctile_std, TieredReim, ITaddon

gamma_table <- data.frame(
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
    draws_kq <- as.numeric(fit_m3b$draws(var_name, format = "matrix"))

    margin <- ifelse(k <= P, "extensive", "intensive")
    cov_idx <- ifelse(k <= P, k, k - P)

    gamma_table$row_idx[idx]   <- k
    gamma_table$col_idx[idx]   <- q
    gamma_table$row_label[idx] <- gamma_row_labels[k]
    gamma_table$col_label[idx] <- gamma_col_labels[q]
    gamma_table$margin[idx]    <- margin
    gamma_table$covariate[idx] <- cov_names[cov_idx]
    gamma_table$policy[idx]    <- gamma_col_labels[q]
    gamma_table$post_mean[idx] <- mean(draws_kq)
    gamma_table$post_sd[idx]   <- sd(draws_kq)
    gamma_table$q025[idx]      <- quantile(draws_kq, 0.025)
    gamma_table$q975[idx]      <- quantile(draws_kq, 0.975)
    gamma_table$prob_pos[idx]  <- mean(draws_kq > 0)
  }
}

## Print formatted Gamma table
cat("  === FULL GAMMA MATRIX: POSTERIOR SUMMARIES ===\n\n")
cat(sprintf("  %-22s %-15s %10s %10s %10s %10s %10s\n",
            "Parameter", "Description", "PostMean", "PostSD",
            "Q2.5", "Q97.5", "P(>0)"))
cat(sprintf("  %s\n", paste(rep("-", 95), collapse = "")))

for (i in 1:nrow(gamma_table)) {
  g <- gamma_table[i, ]
  param_name <- sprintf("Gamma[%d,%d]", g$row_idx, g$col_idx)
  desc <- sprintf("%s.%s", g$row_label, g$col_label)

  ## Mark significance: 95% CI excludes zero
  excludes_zero <- (g$q025 > 0) | (g$q975 < 0)
  sig_mark <- ifelse(excludes_zero, " *", "  ")

  cat(sprintf("  %-22s %-15s %+10.4f %10.4f %+10.4f %+10.4f %10.3f%s\n",
              param_name, desc,
              g$post_mean, g$post_sd, g$q025, g$q975, g$prob_pos, sig_mark))
}

## Count significant Gamma elements
n_sig <- sum((gamma_table$q025 > 0) | (gamma_table$q975 < 0))
cat(sprintf("\n  Significant Gamma elements (95%% CI excludes 0): %d / %d\n",
            n_sig, nrow(gamma_table)))

## 7a. Gamma as K x Q matrix (posterior means)
cat("\n  Gamma matrix (posterior means, K x Q):\n")
gamma_mean_mat <- matrix(NA, K, Q)
for (k in 1:K) {
  for (q in 1:Q) {
    row_filter <- gamma_table$row_idx == k & gamma_table$col_idx == q
    gamma_mean_mat[k, q] <- gamma_table$post_mean[row_filter]
  }
}

cat(sprintf("    %15s", ""))
cat(sprintf("%12s", gamma_col_labels), "\n")
for (k in 1:K) {
  cat(sprintf("    %-15s", gamma_row_labels[k]))
  cat(sprintf("%+12.4f", gamma_mean_mat[k, ]))
  cat("\n")
}

## 7b. Key substantive policy questions
cat("\n  === KEY POLICY MODERATOR QUESTIONS ===\n")

## Q1: TieredReim effect on extensive margin intercept -> Gamma[1, 3]
g13 <- gamma_table[gamma_table$row_idx == 1 & gamma_table$col_idx == 3, ]
cat(sprintf("\n  Q1: Does TieredReim affect extensive margin intercept?\n"))
cat(sprintf("      Gamma[1,3] = %+.4f (SD=%.4f), 95%% CI [%+.4f, %+.4f], P(>0)=%.3f\n",
            g13$post_mean, g13$post_sd, g13$q025, g13$q975, g13$prob_pos))
sig13 <- (g13$q025 > 0) | (g13$q975 < 0)
if (sig13) {
  direction <- ifelse(g13$post_mean > 0, "INCREASES", "DECREASES")
  cat(sprintf("      [SIGNIFICANT] TieredReim %s the baseline probability of serving IT.\n",
              direction))
} else {
  cat("      [Not significant] No clear evidence TieredReim affects extensive margin baseline.\n")
}

## Q2: ITaddon effect on intensive margin -> Gamma[6, 4]
g64 <- gamma_table[gamma_table$row_idx == (P + 1) & gamma_table$col_idx == 4, ]
cat(sprintf("\n  Q2: Does ITaddon affect intensive margin intercept?\n"))
cat(sprintf("      Gamma[%d,4] = %+.4f (SD=%.4f), 95%% CI [%+.4f, %+.4f], P(>0)=%.3f\n",
            P + 1, g64$post_mean, g64$post_sd, g64$q025, g64$q975, g64$prob_pos))
sig64 <- (g64$q025 > 0) | (g64$q975 < 0)
if (sig64) {
  direction <- ifelse(g64$post_mean > 0, "INCREASES", "DECREASES")
  cat(sprintf("      [SIGNIFICANT] ITaddon %s the baseline IT share among servers.\n",
              direction))
} else {
  cat("      [Not significant] No clear evidence ITaddon affects intensive margin baseline.\n")
}

## Q3: MR_pctile moderates poverty effect on extensive margin -> Gamma[2, 2]
g22 <- gamma_table[gamma_table$row_idx == 2 & gamma_table$col_idx == 2, ]
cat(sprintf("\n  Q3: Does MR_pctile moderate the poverty effect on extensive margin?\n"))
cat(sprintf("      Gamma[2,2] = %+.4f (SD=%.4f), 95%% CI [%+.4f, %+.4f], P(>0)=%.3f\n",
            g22$post_mean, g22$post_sd, g22$q025, g22$q975, g22$prob_pos))
sig22 <- (g22$q025 > 0) | (g22$q975 < 0)
if (sig22) {
  direction <- ifelse(g22$post_mean > 0,
                      "ATTENUATES the negative poverty barrier",
                      "AMPLIFIES the negative poverty barrier")
  cat(sprintf("      [SIGNIFICANT] Higher market rates %s.\n", direction))
} else {
  cat("      [Not significant] No clear moderation of poverty barrier by market rates.\n")
}

## Q4: Do any policy variables significantly reduce the poverty reversal?
cat(sprintf("\n  Q4: Do policy variables affect the poverty reversal?\n"))
cat("      Poverty reversal = negative alpha_poverty + positive beta_poverty\n")
cat("      Reducing reversal means: making alpha_poverty less negative OR beta_poverty less positive\n\n")

## Extensive poverty row (k=2): all Q policy columns
for (q in 1:Q) {
  g2q <- gamma_table[gamma_table$row_idx == 2 & gamma_table$col_idx == q, ]
  sig <- (g2q$q025 > 0) | (g2q$q975 < 0)
  cat(sprintf("      Gamma[2,%d] ext_poverty x %-15s = %+.4f  95%%CI [%+.4f, %+.4f] %s\n",
              q, gamma_col_labels[q], g2q$post_mean, g2q$q025, g2q$q975,
              ifelse(sig, "[SIG]", "")))
}

## Intensive poverty row (k=P+2): all Q policy columns
for (q in 1:Q) {
  g_pov_int <- gamma_table[gamma_table$row_idx == (P + 2) & gamma_table$col_idx == q, ]
  sig <- (g_pov_int$q025 > 0) | (g_pov_int$q975 < 0)
  cat(sprintf("      Gamma[%d,%d] int_poverty x %-15s = %+.4f  95%%CI [%+.4f, %+.4f] %s\n",
              P + 2, q, gamma_col_labels[q],
              g_pov_int$post_mean, g_pov_int$q025, g_pov_int$q975,
              ifelse(sig, "[SIG]", "")))
}

## 7c. Summary by margin: any column-3 or column-4 significant?
cat("\n  === POLICY EFFECT SUMMARY BY MARGIN ===\n")

for (margin_label in c("extensive", "intensive")) {
  margin_rows <- gamma_table[gamma_table$margin == margin_label, ]
  ## Exclude intercept column (q=1); focus on policy variables (q=2,3,4)
  policy_rows <- margin_rows[margin_rows$col_idx > 1, ]
  n_sig_margin <- sum((policy_rows$q025 > 0) | (policy_rows$q975 < 0))
  cat(sprintf("  %s margin: %d / %d significant policy effects (excluding intercept col)\n",
              margin_label, n_sig_margin, nrow(policy_rows)))
}

cat("\n")


###############################################################################
## SECTION 8 : TAU COMPARISON WITH M3a
###############################################################################
cat("--- 8. Tau comparison with M3a (residual state variation) ---\n")
cat("  If policy covariates explain state variation, M3b tau should be SMALLER than M3a tau.\n\n")

if (file.exists(RESULTS_M3A_PATH)) {
  results_m3a <- readRDS(RESULTS_M3A_PATH)
  m3a_tau_means <- as.numeric(results_m3a$tau_means)

  cat(sprintf("  %-20s %10s %10s %10s %10s\n",
              "Parameter", "M3a_tau", "M3b_tau", "Reduction", "Pct_red"))
  cat(sprintf("  %s\n", paste(rep("-", 65), collapse = "")))

  tau_reduction <- numeric(K)
  tau_pct_reduction <- numeric(K)

  for (k in 1:K) {
    reduction <- m3a_tau_means[k] - tau_means[k]
    pct_red   <- 100 * reduction / m3a_tau_means[k]
    tau_reduction[k]     <- reduction
    tau_pct_reduction[k] <- pct_red

    cat(sprintf("  tau[%2d] %-12s %10.4f %10.4f %+10.4f %+10.1f%%\n",
                k, tau_labels[k], m3a_tau_means[k], tau_means[k],
                reduction, pct_red))
  }

  ## Overall reduction
  mean_pct_red <- mean(tau_pct_reduction)
  cat(sprintf("\n  Mean tau reduction: %.1f%%\n", mean_pct_red))

  ## Separate by margin
  ext_mean_red <- mean(tau_pct_reduction[1:P])
  int_mean_red <- mean(tau_pct_reduction[(P + 1):K])
  cat(sprintf("  Extensive margin mean reduction: %.1f%%\n", ext_mean_red))
  cat(sprintf("  Intensive margin mean reduction: %.1f%%\n", int_mean_red))

  if (mean_pct_red > 5) {
    cat("  [PASS] Policy covariates explain some state variation (>5%% mean reduction).\n")
  } else if (mean_pct_red > 0) {
    cat("  [NOTE] Modest tau reduction. Policy covariates explain limited variation.\n")
  } else {
    cat("  [NOTE] No tau reduction. Policy covariates may not explain state variation.\n")
  }
} else {
  cat("  [INFO] M3a results not found. Skipping tau comparison.\n")
}

cat("\n")


###############################################################################
## SECTION 9 : CROSS-MARGIN CORRELATION ANALYSIS
###############################################################################
cat("--- 9. Cross-margin correlation analysis ---\n")
cat("  Checking if cross-margin correlations change after accounting for policy moderators.\n\n")

cat("  === CROSS-MARGIN CORRELATIONS: SAME COVARIATE (DIAGONAL) ===\n")
cat(sprintf("  %-15s %10s %10s %10s %10s %10s\n",
            "Covariate", "PostMean", "PostSD", "Q2.5", "Q97.5", "P(rho<0)"))
cat(sprintf("  %s\n", paste(rep("-", 65), collapse = "")))

cross_diag_means    <- numeric(P)
cross_diag_sd       <- numeric(P)
cross_diag_q025     <- numeric(P)
cross_diag_q975     <- numeric(P)
cross_diag_prob_neg <- numeric(P)

for (k in 1:P) {
  var_name <- sprintf("Omega[%d,%d]", k, P + k)
  draws_k <- as.numeric(fit_m3b$draws(var_name, format = "matrix"))

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

## Compare with M3a cross-margin correlations
if (file.exists(RESULTS_M3A_PATH)) {
  results_m3a <- readRDS(RESULTS_M3A_PATH)
  m3a_cross_diag <- as.numeric(results_m3a$cross_diag_means)

  cat("\n  === CROSS-MARGIN CORRELATION COMPARISON: M3a vs M3b ===\n")
  cat(sprintf("  %-15s %10s %10s %10s\n",
              "Covariate", "M3a_rho", "M3b_rho", "Change"))
  cat(sprintf("  %s\n", paste(rep("-", 50), collapse = "")))

  for (k in 1:P) {
    change <- cross_diag_means[k] - m3a_cross_diag[k]
    cat(sprintf("  %-15s %+10.4f %+10.4f %+10.4f\n",
                cov_names[k], m3a_cross_diag[k], cross_diag_means[k], change))
  }
}

cat("\n")


###############################################################################
## SECTION 10 : STATE-LEVEL POVERTY REVERSAL ANALYSIS
###############################################################################
cat("--- 10. State-level poverty reversal analysis ---\n")

## In M3b, delta[s,k] is residual after policy adjustment.
## The total state-specific coefficient is:
##   alpha_poverty_s = alpha[2] + Gamma[2,]' * v_state[s] + delta[s, 2]
##   beta_poverty_s  = beta[2]  + Gamma[P+2,]' * v_state[s] + delta[s, P+2]

## Extract draws
alpha2_draws <- fit_m3b$draws("alpha[2]", format = "matrix")
beta2_draws  <- fit_m3b$draws("beta[2]",  format = "matrix")
n_draws <- nrow(alpha2_draws)

## Extract Gamma[2,q] and Gamma[P+2,q] draws for poverty row
gamma_ext_pov_draws <- matrix(NA, n_draws, Q)  # Gamma[2,1:Q]
gamma_int_pov_draws <- matrix(NA, n_draws, Q)  # Gamma[P+2,1:Q]
for (q in 1:Q) {
  gamma_ext_pov_draws[, q] <- as.numeric(
    fit_m3b$draws(sprintf("Gamma[2,%d]", q), format = "matrix")
  )
  gamma_int_pov_draws[, q] <- as.numeric(
    fit_m3b$draws(sprintf("Gamma[%d,%d]", P + 2, q), format = "matrix")
  )
}

## Extract delta[s,2] and delta[s,P+2] for all states
delta_ext_pov_names <- sprintf("delta[%d,2]", 1:S)
delta_int_pov_names <- sprintf("delta[%d,%d]", 1:S, P + 2)
delta_ext_pov_draws <- fit_m3b$draws(delta_ext_pov_names, format = "matrix")
delta_int_pov_draws <- fit_m3b$draws(delta_int_pov_names, format = "matrix")

## v_state matrix: S x Q
v_mat <- stan_data_m3b$v_state

## Compute state-specific total poverty effects for each draw:
##   alpha_poverty_s[d,s] = alpha[2][d] + v_state[s,] %*% Gamma[2,][d] + delta[s,2][d]
##   beta_poverty_s[d,s]  = beta[2][d]  + v_state[s,] %*% Gamma[P+2,][d] + delta[s,P+2][d]

alpha_poverty_s <- matrix(NA, n_draws, S)
beta_poverty_s  <- matrix(NA, n_draws, S)

for (s in 1:S) {
  v_s <- as.numeric(v_mat[s, ])  # Q-vector

  ## Policy mean contribution: Gamma[2,] %*% v_state[s] for each draw
  policy_ext_s <- gamma_ext_pov_draws %*% v_s  # n_draws x 1
  policy_int_s <- gamma_int_pov_draws %*% v_s  # n_draws x 1

  alpha_poverty_s[, s] <- as.numeric(alpha2_draws) + as.numeric(policy_ext_s) +
                           as.numeric(delta_ext_pov_draws[, s])
  beta_poverty_s[, s]  <- as.numeric(beta2_draws) + as.numeric(policy_int_s) +
                           as.numeric(delta_int_pov_draws[, s])
}

## For each state: posterior mean and P(reversal pattern)
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

## Classify states
state_poverty_table$reversal_mean <- (state_poverty_table$alpha_pov_mean < 0 &
                                       state_poverty_table$beta_pov_mean > 0)
state_poverty_table$reversal_prob50 <- state_poverty_table$prob_reversal > 0.5

n_reversal_mean   <- sum(state_poverty_table$reversal_mean)
n_reversal_prob50 <- sum(state_poverty_table$reversal_prob50)

cat(sprintf("\n  States showing poverty reversal (posterior mean): %d / %d (%.0f%%)\n",
            n_reversal_mean, S, 100 * n_reversal_mean / S))
cat(sprintf("  States showing poverty reversal (P > 0.5):       %d / %d (%.0f%%)\n",
            n_reversal_prob50, S, 100 * n_reversal_prob50 / S))

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
## SECTION 11 : LOO-CV AND FIVE-WAY COMPARISON
###############################################################################
cat("--- 11. LOO-CV computation and five-way comparison ---\n")

## Extract log_lik array
log_lik <- fit_m3b$draws("log_lik", format = "matrix")
cat(sprintf("  log_lik dimensions: %d draws x %d observations\n",
            nrow(log_lik), ncol(log_lik)))

## Compute LOO
loo_m3b <- tryCatch(
  loo(log_lik, cores = 4),
  error = function(e) {
    warning("LOO computation failed: ", conditionMessage(e))
    NULL
  }
)

if (!is.null(loo_m3b)) {
  cat("\n  LOO-CV summary (M3b):\n")
  print(loo_m3b)

  ## Check Pareto k diagnostics
  k_values <- loo_m3b$diagnostics$pareto_k
  n_bad_k <- sum(k_values > 0.7)
  cat(sprintf("\n  Pareto k > 0.7: %d / %d observations (%.1f%%)\n",
              n_bad_k, length(k_values), 100 * n_bad_k / length(k_values)))

  if (n_bad_k > 0) {
    cat(sprintf("  [NOTE] %d observations with problematic Pareto k values.\n", n_bad_k))
  } else {
    cat("  [PASS] All Pareto k values < 0.7.\n")
  }

  ## 11a. LOO comparison with M3a (CRITICAL TEST: Does policy moderation help?)
  if (file.exists(RESULTS_M3A_PATH)) {
    results_m3a <- readRDS(RESULTS_M3A_PATH)
    loo_m3a <- results_m3a$loo

    if (!is.null(loo_m3a)) {
      cat("\n  === LOO-CV COMPARISON: M3b vs M3a (CRITICAL TEST) ===\n")
      cat("  (Does adding policy moderators to the mean structure help?)\n\n")

      elpd_m3a <- loo_m3a$estimates["elpd_loo", "Estimate"]
      elpd_m3b <- loo_m3b$estimates["elpd_loo", "Estimate"]
      cat(sprintf("    M3a ELPD_loo = %.1f\n", elpd_m3a))
      cat(sprintf("    M3b ELPD_loo = %.1f\n", elpd_m3b))
      cat(sprintf("    Difference   = %.1f (positive = M3b better)\n",
                  elpd_m3b - elpd_m3a))

      ## Formal loo_compare
      comp_m3b_m3a <- loo_compare(loo_m3a, loo_m3b)
      cat("\n  loo_compare output (M3b vs M3a):\n")
      print(comp_m3b_m3a)

      ## Interpretation
      elpd_diff <- comp_m3b_m3a[2, "elpd_diff"]
      se_diff   <- comp_m3b_m3a[2, "se_diff"]
      cat(sprintf("\n    ELPD difference: %.1f (SE = %.1f)\n",
                  elpd_diff, se_diff))

      if (abs(elpd_diff) > 2 * se_diff) {
        better_model <- rownames(comp_m3b_m3a)[1]
        cat(sprintf("    [PASS] Difference > 2*SE: %s is clearly preferred.\n",
                    better_model))
      } else {
        cat("    [NOTE] Difference < 2*SE: models are comparable.\n")
        cat("    Policy moderators may not improve predictive accuracy,\n")
        cat("    but Gamma coefficients remain informative for understanding\n")
        cat("    how policy shapes state variation in the poverty reversal.\n")
      }
    }
  }

  ## 11b. Five-way comparison if all LOO objects available
  all_results_exist <- file.exists(RESULTS_M0_PATH) &&
                       file.exists(RESULTS_M1_PATH) &&
                       file.exists(RESULTS_M2_PATH) &&
                       file.exists(RESULTS_M3A_PATH)

  if (all_results_exist) {
    results_m0  <- readRDS(RESULTS_M0_PATH)
    results_m1  <- readRDS(RESULTS_M1_PATH)
    results_m2  <- readRDS(RESULTS_M2_PATH)
    results_m3a <- readRDS(RESULTS_M3A_PATH)

    if (!is.null(results_m0$loo) && !is.null(results_m1$loo) &&
        !is.null(results_m2$loo) && !is.null(results_m3a$loo)) {

      cat("\n  === FIVE-WAY LOO COMPARISON: M0 vs M1 vs M2 vs M3a vs M3b ===\n")
      comp_all <- loo_compare(results_m0$loo, results_m1$loo,
                              results_m2$loo, results_m3a$loo, loo_m3b)
      print(comp_all)

      elpd_m0  <- results_m0$loo$estimates["elpd_loo", "Estimate"]
      elpd_m1  <- results_m1$loo$estimates["elpd_loo", "Estimate"]
      elpd_m2  <- results_m2$loo$estimates["elpd_loo", "Estimate"]
      elpd_m3a <- results_m3a$loo$estimates["elpd_loo", "Estimate"]
      elpd_m3b <- loo_m3b$estimates["elpd_loo", "Estimate"]

      cat("\n  LOO progression:\n")
      cat(sprintf("    M0  ELPD = %.1f\n", elpd_m0))
      cat(sprintf("    M1  ELPD = %.1f  (M1-M0 = %+.1f)\n",
                  elpd_m1, elpd_m1 - elpd_m0))
      cat(sprintf("    M2  ELPD = %.1f  (M2-M1 = %+.1f, M2-M0 = %+.1f)\n",
                  elpd_m2, elpd_m2 - elpd_m1, elpd_m2 - elpd_m0))
      cat(sprintf("    M3a ELPD = %.1f  (M3a-M2 = %+.1f, M3a-M0 = %+.1f)\n",
                  elpd_m3a, elpd_m3a - elpd_m2, elpd_m3a - elpd_m0))
      cat(sprintf("    M3b ELPD = %.1f  (M3b-M3a = %+.1f, M3b-M0 = %+.1f)\n",
                  elpd_m3b, elpd_m3b - elpd_m3a, elpd_m3b - elpd_m0))
      cat("\n")
    }
  }
} else {
  cat("  [WARN] LOO-CV computation failed. Skipping.\n")
}

cat("\n")


###############################################################################
## SECTION 12 : POSTERIOR PREDICTIVE CHECKS
###############################################################################
cat("--- 12. Posterior predictive checks ---\n")

## Use y_rep from Stan generated quantities for PPC
y_rep_available <- tryCatch({
  fit_m3b$draws("y_rep[1]", format = "matrix")
  TRUE
}, error = function(e) FALSE)

N <- stan_data_m3b$N
y_obs <- stan_data_m3b$y
n_trial <- stan_data_m3b$n_trial
z_obs <- stan_data_m3b$z

observed_zero_rate <- mean(z_obs == 0)
observed_mean_share_pos <- mean(y_obs[z_obs == 1] / n_trial[z_obs == 1])

cat(sprintf("  Observed zero rate (structural): %.3f (%.1f%%)\n",
            observed_zero_rate, 100 * observed_zero_rate))
cat(sprintf("  Observed mean IT share (y/n | z=1): %.4f\n",
            observed_mean_share_pos))

if (y_rep_available) {
  cat("\n  Using y_rep from Stan generated quantities for PPC ...\n")

  y_rep_draws <- fit_m3b$draws("y_rep", format = "matrix")  # n_draws x N
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
  alpha_draws <- fit_m3b$draws("alpha", format = "matrix")  # n_draws x P
  beta_draws  <- fit_m3b$draws("beta",  format = "matrix")  # n_draws x P
  lk_draws    <- fit_m3b$draws("log_kappa", format = "matrix")  # n_draws x 1

  ## Extract all delta draws
  delta_all_names_ppc <- c()
  for (s in 1:S) {
    for (k in 1:K) {
      delta_all_names_ppc <- c(delta_all_names_ppc, sprintf("delta[%d,%d]", s, k))
    }
  }
  delta_all_draws <- fit_m3b$draws(delta_all_names_ppc, format = "matrix")

  ## Extract Gamma draws for mean structure
  gamma_all_draws <- fit_m3b$draws("Gamma", format = "matrix")  # n_draws x (K*Q)

  n_draws_ppc <- nrow(alpha_draws)
  n_ppc_use <- min(200, n_draws_ppc)
  draw_idx <- seq(1, n_draws_ppc, length.out = n_ppc_use) |> round() |> unique()
  n_ppc_use <- length(draw_idx)

  cat(sprintf("  Number of posterior draws: %d, using %d for PPC\n",
              n_draws_ppc, n_ppc_use))

  X <- stan_data_m3b$X
  state_idx <- stan_data_m3b$state
  v_mat <- stan_data_m3b$v_state

  ppc_zero_rates <- numeric(n_ppc_use)
  ppc_mean_share <- numeric(n_ppc_use)

  for (d in seq_along(draw_idx)) {
    dd <- draw_idx[d]

    alpha_d <- as.numeric(alpha_draws[dd, ])
    beta_d  <- as.numeric(beta_draws[dd, ])
    kappa_d <- exp(lk_draws[dd, 1])

    ## Reconstruct Gamma matrix for this draw
    gamma_d <- matrix(as.numeric(gamma_all_draws[dd, ]), nrow = K, ncol = Q,
                      byrow = FALSE)

    eta_ext <- numeric(N)
    eta_int <- numeric(N)
    for (i in 1:N) {
      s <- state_idx[i]
      col_start <- (s - 1) * K
      d_s <- as.numeric(delta_all_draws[dd, col_start + 1:K])

      ## Total state-specific coefficient = Gamma * v_state[s] + delta[s]
      mu_delta_s <- gamma_d %*% as.numeric(v_mat[s, ])  # K-vector
      total_s <- as.numeric(mu_delta_s) + d_s

      d_ext_s <- total_s[1:P]
      d_int_s <- total_s[(P + 1):K]
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
## SECTION 13 : SAVE RESULTS
###############################################################################
cat("--- 13. Saving results ---\n")

## 13a. Save the CmdStanR fit object
fit_m3b$save_object(FIT_OUT)
cat(sprintf("  Saved fit object: %s\n", FIT_OUT))
cat(sprintf("    File size: %.1f MB\n",
            file.info(FIT_OUT)$size / 1024^2))

## 13b. Assemble and save results list
results_m3b <- list(
  ## Model info
  model      = "M3b",
  model_desc = "Policy Moderator SVC Hurdle Beta-Binomial (unweighted)",

  ## Timing
  fit_time_mins = as.numeric(fit_time),

  ## Parameter summary (fixed effects)
  param_summary = param_summary_fixed,

  ## Hierarchical parameter summary
  hier_summary = param_summary_hier,

  ## Gamma parameter summary
  gamma_summary = param_summary_gamma,

  ## Key estimates
  alpha_means    = setNames(alpha_means, cov_names),
  beta_means     = setNames(beta_means, cov_names),
  log_kappa_mean = log_kappa_mean,
  kappa_mean     = kappa_mean,
  tau_means      = setNames(tau_means, tau_labels),

  ## Gamma matrix (posterior means)
  gamma_mean_mat = gamma_mean_mat,

  ## Gamma detailed table
  gamma_table = gamma_table,

  ## Full Omega correlation matrix (posterior mean)
  omega_mat = omega_mat,

  ## Cross-margin correlation diagonal (same-covariate, cross-margin)
  cross_diag_means    = setNames(cross_diag_means, cov_names),
  cross_diag_sd       = setNames(cross_diag_sd, cov_names),
  cross_diag_q025     = setNames(cross_diag_q025, cov_names),
  cross_diag_q975     = setNames(cross_diag_q975, cov_names),
  cross_diag_prob_neg = setNames(cross_diag_prob_neg, cov_names),

  ## Tau comparison with M3a
  tau_reduction     = if (exists("tau_reduction")) tau_reduction else NULL,
  tau_pct_reduction = if (exists("tau_pct_reduction")) tau_pct_reduction else NULL,

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
  loo = loo_m3b,

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
  stan_data_m3b = stan_data_m3b,

  ## Timestamp
  timestamp = Sys.time()
)

saveRDS(results_m3b, RESULTS_OUT)
cat(sprintf("  Saved results: %s\n", RESULTS_OUT))
cat(sprintf("    File size: %.1f KB\n",
            file.info(RESULTS_OUT)$size / 1024))


###############################################################################
## SECTION 14 : FINAL SUMMARY
###############################################################################
cat("\n==============================================================\n")
cat("  M3b FITTING SUMMARY\n")
cat("==============================================================\n")

n_fixed  <- P * 2 + 1                    # alpha + beta + log_kappa
n_hier   <- K + K * (K - 1) / 2          # tau[K] + L_Omega free elements
n_gamma  <- K * Q                         # Gamma[K, Q]
n_re     <- K * S                         # delta[S, K]
n_params_total <- n_fixed + n_hier + n_gamma + n_re

cat(sprintf("\n  Model: Policy Moderator SVC Hurdle Beta-Binomial (M3b)\n"))
cat(sprintf("  N = %d, P = %d, S = %d, Q = %d, K = 2P = %d\n", N, P, S, Q, K))
cat(sprintf("  Parameters: %d fixed + %d hier + %d Gamma + %d RE = %d total\n",
            n_fixed, n_hier, n_gamma, n_re, n_params_total))
cat(sprintf("  Fit time: %.1f minutes\n", as.numeric(fit_time)))

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

cat(sprintf("\n  Key Results:\n"))
cat(sprintf("    alpha_poverty = %+.4f %s\n",
            alpha_poverty, ifelse(alpha_poverty < 0, "[PASS] (<0)", "[WARN]")))
cat(sprintf("    beta_poverty  = %+.4f %s\n",
            beta_poverty, ifelse(beta_poverty > 0, "[PASS] (>0)", "[WARN]")))
cat(sprintf("    kappa         = %.2f %s\n",
            kappa_mean, ifelse(kappa_mean > 3 & kappa_mean < 30, "[PASS]", "[WARN]")))

cat(sprintf("\n  Tau (state variation scale, with M3a comparison):\n"))
if (file.exists(RESULTS_M3A_PATH)) {
  cat(sprintf("    %-15s %10s %10s %10s\n", "Parameter", "M3a", "M3b", "Pct_red"))
  cat(sprintf("    %s\n", paste(rep("-", 48), collapse = "")))
  results_m3a <- readRDS(RESULTS_M3A_PATH)
  m3a_tau_means <- as.numeric(results_m3a$tau_means)
  for (k in 1:K) {
    pct_red <- 100 * (m3a_tau_means[k] - tau_means[k]) / m3a_tau_means[k]
    cat(sprintf("    tau[%2d] %-12s %8.4f %8.4f %+8.1f%%\n",
                k, tau_labels[k], m3a_tau_means[k], tau_means[k], pct_red))
  }
} else {
  for (k in 1:K) {
    cat(sprintf("    tau[%2d] %-12s = %.4f\n", k, tau_labels[k], tau_means[k]))
  }
}

cat(sprintf("\n  Gamma (policy moderators, significant at 95%%):\n"))
for (i in 1:nrow(gamma_table)) {
  g <- gamma_table[i, ]
  excludes_zero <- (g$q025 > 0) | (g$q975 < 0)
  if (excludes_zero) {
    cat(sprintf("    Gamma[%d,%d] %-22s = %+.4f  95%% CI [%+.4f, %+.4f] *\n",
                g$row_idx, g$col_idx,
                sprintf("%s.%s", g$row_label, g$col_label),
                g$post_mean, g$q025, g$q975))
  }
}
if (n_sig == 0) {
  cat("    (none significant at 95%% level)\n")
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

if (!is.null(loo_m3b)) {
  cat(sprintf("\n  LOO-CV:\n"))
  cat(sprintf("    ELPD_loo   = %.1f (SE = %.1f)\n",
              loo_m3b$estimates["elpd_loo", "Estimate"],
              loo_m3b$estimates["elpd_loo", "SE"]))
  cat(sprintf("    p_loo      = %.1f\n",
              loo_m3b$estimates["p_loo", "Estimate"]))
  cat(sprintf("    Pareto k > 0.7: %d / %d\n",
              sum(loo_m3b$diagnostics$pareto_k > 0.7),
              length(loo_m3b$diagnostics$pareto_k)))

  if (all_results_exist) {
    results_m0  <- readRDS(RESULTS_M0_PATH)
    results_m1  <- readRDS(RESULTS_M1_PATH)
    results_m2  <- readRDS(RESULTS_M2_PATH)
    results_m3a <- readRDS(RESULTS_M3A_PATH)
    if (!is.null(results_m0$loo) && !is.null(results_m1$loo) &&
        !is.null(results_m2$loo) && !is.null(results_m3a$loo)) {
      elpd_m0   <- results_m0$loo$estimates["elpd_loo", "Estimate"]
      elpd_m1   <- results_m1$loo$estimates["elpd_loo", "Estimate"]
      elpd_m2   <- results_m2$loo$estimates["elpd_loo", "Estimate"]
      elpd_m3a  <- results_m3a$loo$estimates["elpd_loo", "Estimate"]
      elpd_m3b_val <- loo_m3b$estimates["elpd_loo", "Estimate"]

      cat(sprintf("\n  LOO Comparison (5-way):\n"))
      cat(sprintf("    M0  ELPD = %.1f\n", elpd_m0))
      cat(sprintf("    M1  ELPD = %.1f  (M1-M0 = %+.1f)\n",
                  elpd_m1, elpd_m1 - elpd_m0))
      cat(sprintf("    M2  ELPD = %.1f  (M2-M1 = %+.1f)\n",
                  elpd_m2, elpd_m2 - elpd_m1))
      cat(sprintf("    M3a ELPD = %.1f  (M3a-M2 = %+.1f)\n",
                  elpd_m3a, elpd_m3a - elpd_m2))
      cat(sprintf("    M3b ELPD = %.1f  (M3b-M3a = %+.1f, M3b-M0 = %+.1f)\n",
                  elpd_m3b_val, elpd_m3b_val - elpd_m3a, elpd_m3b_val - elpd_m0))
    }
  }
}

cat(sprintf("\n  Output files:\n"))
cat(sprintf("    %s\n", FIT_OUT))
cat(sprintf("    %s\n", RESULTS_OUT))

cat("\n==============================================================\n")
cat("  M3b FITTING COMPLETE.\n")
if (diag_pass) {
  cat("  ALL DIAGNOSTICS PASSED.\n")
} else {
  cat("  SOME DIAGNOSTICS FAILED. Review before proceeding.\n")
}
cat("==============================================================\n")
