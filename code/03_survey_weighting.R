## =============================================================================
## 03_survey_weighting.R -- Sandwich Variance and Cholesky Correction
## =============================================================================
## Purpose : Compute the cluster-robust sandwich variance estimator
##           V_sand = H_obs^{-1} * J_cluster * H_obs^{-1} and apply the
##           Williams-Savitsky (2021) Cholesky affine transformation to
##           correct pseudo-posterior MCMC draws for survey design effects.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Theory:
##   The weighted pseudo-posterior yields draws whose covariance (Sigma_MCMC)
##   does NOT equal H_obs^{-1} in hierarchical models because priors and
##   random effects dominate the marginal posterior of fixed effects.
##
##   V_sand = H_obs^{-1} * J_cluster * H_obs^{-1}
##
##   where H_obs is the block-diagonal observed Fisher information and
##   J_cluster is the cluster-robust "meat" matrix.
##
##   Cholesky correction (Theorem 4.1):
##     theta*^(m) = theta_hat + A * (theta^(m) - theta_hat)
##     where A = L_sand * L_MCMC^{-1}
##
## Inputs:
##   data/precomputed/results_m3b_weighted.rds -- Weighted fit results
##   data/precomputed/fit_m3b_weighted.rds     -- CmdStanR fit object
##   data/precomputed/stan_data.rds            -- Stan data
##   data/precomputed/scores_m3b_weighted.rds  -- Pre-saved posterior mean scores
##
## Outputs:
##   data/precomputed/sandwich_variance.rds    -- J_cluster, V_sand, DER, etc.
##   data/precomputed/cholesky_correction.rds  -- Corrected draws, CIs, etc.
##
## Usage:
##   source("code/03_survey_weighting.R")
## =============================================================================

cat("==============================================================\n")
cat("  Sandwich Variance Estimator + Cholesky Correction\n")
cat("==============================================================\n\n")


###############################################################################
## SECTION 0 : SETUP
###############################################################################

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()

## Source helper functions
source(file.path(PROJECT_ROOT, "code/helpers/utils.R"))

## Load required packages
library(cmdstanr)
library(posterior)
library(dplyr, warn.conflicts = FALSE)
library(Matrix)

## Paths
STAN_DATA_PATH        <- file.path(PROJECT_ROOT, "data/precomputed/stan_data.rds")
FIT_WEIGHTED_PATH     <- file.path(PROJECT_ROOT, "data/precomputed/fit_m3b_weighted.rds")
RESULTS_WEIGHTED_PATH <- file.path(PROJECT_ROOT, "data/precomputed/results_m3b_weighted.rds")
SCORES_PATH           <- file.path(PROJECT_ROOT, "data/precomputed/scores_m3b_weighted.rds")
OUTPUT_DIR            <- file.path(PROJECT_ROOT, "data/precomputed")
SANDWICH_OUT          <- file.path(OUTPUT_DIR, "sandwich_variance.rds")
CORRECTION_OUT        <- file.path(OUTPUT_DIR, "cholesky_correction.rds")

## Ensure output directory exists
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)


###############################################################################
## PART A : SANDWICH VARIANCE ESTIMATOR
###############################################################################
cat("\n")
cat("##############################################################\n")
cat("##  Part A: Sandwich Variance Estimator                     ##\n")
cat("##############################################################\n\n")

###############################################################################
## SECTION 1 : LOAD INPUTS
###############################################################################
cat("--- 1. Loading inputs ---\n")

## 1a. Load Stan data (contains survey design variables)
stopifnot("Stan data file not found" = file.exists(STAN_DATA_PATH))
stan_data <- readRDS(STAN_DATA_PATH)
cat(sprintf("  Loaded stan_data: %s\n", STAN_DATA_PATH))
cat(sprintf("    N = %d, P = %d, S = %d\n",
            stan_data$N, stan_data$P, stan_data$S))
cat(sprintf("    n_strata = %d, n_psu = %d\n",
            stan_data$n_strata, stan_data$n_psu))

## 1b. Load weighted fit results
stopifnot("Weighted results file not found" = file.exists(RESULTS_WEIGHTED_PATH))
results_weighted <- readRDS(RESULTS_WEIGHTED_PATH)
cat(sprintf("  Loaded weighted results: %s\n", RESULTS_WEIGHTED_PATH))

## 1c. Load the CmdStanR fit object (for MCMC draws)
stopifnot("Weighted fit file not found" = file.exists(FIT_WEIGHTED_PATH))
fit_weighted <- readRDS(FIT_WEIGHTED_PATH)
cat(sprintf("  Loaded weighted fit: %s\n", FIT_WEIGHTED_PATH))

## Extract key dimensions
N <- stan_data$N
P <- stan_data$P
S <- stan_data$S
D <- 2 * P + 1  # total fixed-effect dimension: alpha(P) + beta(P) + log_kappa(1) = 11

cat(sprintf("  Fixed-effect dimension D = 2P + 1 = %d\n", D))
cat("  [PASS] All inputs loaded.\n\n")


###############################################################################
## SECTION 2 : EXTRACT FIXED-EFFECT DRAWS AND LOAD POSTERIOR MEAN SCORES
###############################################################################
cat("--- 2. Extracting fixed-effect draws and loading posterior mean scores ---\n")

## 2a. Fixed-effect parameter names (for Sigma_MCMC)
param_names_fixed <- c(paste0("alpha[", 1:P, "]"),
                        paste0("beta[", 1:P, "]"),
                        "log_kappa")

## Human-readable labels for reporting
param_labels <- c(
  "alpha_intercept", "alpha_poverty", "alpha_urban", "alpha_black", "alpha_hispanic",
  "beta_intercept",  "beta_poverty",  "beta_urban",  "beta_black",  "beta_hispanic",
  "log_kappa"
)

cat(sprintf("  Fixed-effect parameters (%d):\n", D))
for (d in seq_len(D)) {
  cat(sprintf("    [%2d] %-25s  (%s)\n", d, param_labels[d], param_names_fixed[d]))
}

## 2b. Extract MCMC draws for fixed effects: M x D matrix
draws_fixed <- fit_weighted$draws(variables = param_names_fixed, format = "matrix")
M <- nrow(draws_fixed)
cat(sprintf("\n  MCMC draws: M = %d draws x D = %d parameters\n", M, D))

## 2c. Load pre-saved posterior mean scores
cat("\n  Loading pre-saved posterior mean scores ...\n")

if (file.exists(SCORES_PATH)) {
  scores_saved <- readRDS(SCORES_PATH)
  cat(sprintf("  Loaded: %s\n", SCORES_PATH))

  score_ext_mat     <- scores_saved$score_ext       # N x P
  score_int_mat     <- scores_saved$score_int       # N x P
  score_kappa_means <- scores_saved$score_kappa     # length N

  cat(sprintf("  score_ext_mat: %d x %d\n", nrow(score_ext_mat), ncol(score_ext_mat)))
  cat(sprintf("  score_int_mat: %d x %d\n", nrow(score_int_mat), ncol(score_int_mat)))
  cat(sprintf("  score_kappa:   length %d\n", length(score_kappa_means)))

  ## Validate dimensions
  stopifnot(nrow(score_ext_mat) == N, ncol(score_ext_mat) == P)
  stopifnot(nrow(score_int_mat) == N, ncol(score_int_mat) == P)
  stopifnot(length(score_kappa_means) == N)
} else {
  ## Fallback: extract directly from fit object (slow but functional)
  cat("  [WARN] Pre-saved scores not found. Extracting from fit object (slow) ...\n")

  score_ext_draws <- fit_weighted$draws("score_ext", format = "draws_matrix")
  score_ext_mat   <- matrix(colMeans(score_ext_draws), nrow = N, ncol = P, byrow = FALSE)

  score_int_draws <- fit_weighted$draws("score_int", format = "draws_matrix")
  score_int_mat   <- matrix(colMeans(score_int_draws), nrow = N, ncol = P, byrow = FALSE)

  score_kappa_draws <- fit_weighted$draws("score_kappa", format = "draws_matrix")
  score_kappa_means <- as.numeric(colMeans(score_kappa_draws))

  cat(sprintf("  score_ext_mat: %d x %d\n", nrow(score_ext_mat), ncol(score_ext_mat)))
  cat(sprintf("  score_int_mat: %d x %d\n", nrow(score_int_mat), ncol(score_int_mat)))
  cat(sprintf("  score_kappa:   length %d\n", length(score_kappa_means)))
}

## 2d. Stack into N x D score matrix: S_i = [score_ext_i, score_int_i, score_kappa_i]
S_mat <- cbind(score_ext_mat, score_int_mat, score_kappa_means)
colnames(S_mat) <- param_labels
cat(sprintf("\n  Full score matrix S_mat: %d x %d\n", nrow(S_mat), ncol(S_mat)))

## Quick sanity check: score means should be near zero at the MLE/posterior mode
score_col_means <- colMeans(S_mat)
cat("\n  Score column means (should be near zero at posterior mean):\n")
for (d in seq_len(D)) {
  cat(sprintf("    %-25s  mean = %+.6f\n", param_labels[d], score_col_means[d]))
}

cat("\n  [PASS] Scores loaded and stacked.\n\n")


###############################################################################
## SECTION 3 : COMPUTE Sigma_MCMC (EMPIRICAL POSTERIOR COVARIANCE)
###############################################################################
cat("--- 3. Computing Sigma_MCMC (empirical posterior covariance) ---\n")

## Sigma_MCMC = cov(draws_fixed)  # D x D
Sigma_MCMC <- cov(draws_fixed)
cat(sprintf("  Sigma_MCMC dimensions: %d x %d\n", nrow(Sigma_MCMC), ncol(Sigma_MCMC)))

## Check positive definiteness
eig_Sigma <- eigen(Sigma_MCMC, symmetric = TRUE, only.values = TRUE)$values
min_eig_Sigma <- min(eig_Sigma)
max_eig_Sigma <- max(eig_Sigma)
cond_number_Sigma <- max_eig_Sigma / min_eig_Sigma

cat(sprintf("  Eigenvalue range: [%.2e, %.2e]\n", min_eig_Sigma, max_eig_Sigma))
cat(sprintf("  Condition number: %.2e\n", cond_number_Sigma))

if (min_eig_Sigma > 0) {
  cat("  [PASS] Sigma_MCMC is positive definite.\n")
} else {
  warning("  [WARN] Sigma_MCMC is NOT positive definite! Min eigenvalue = ",
          sprintf("%.2e", min_eig_Sigma))
}

## Print diagonal (posterior variances)
cat("\n  Sigma_MCMC diagonal (posterior variances):\n")
for (d in seq_len(D)) {
  cat(sprintf("    %-25s  Var = %.6e  SD = %.6f\n",
              param_labels[d], Sigma_MCMC[d, d], sqrt(Sigma_MCMC[d, d])))
}

cat("\n")


###############################################################################
## SECTION 4 : COMPUTE CLUSTER-ROBUST MEAT MATRIX J_cluster
###############################################################################
cat("--- 4. Computing cluster-robust meat matrix J_cluster ---\n")
cat("  J_cluster = sum_h (C_h/(C_h-1)) * sum_c (s_bar_hc - s_bar_h)(s_bar_hc - s_bar_h)'\n")
cat("  where s_bar_hc = sum_{i in PSU(h,c)} w_tilde[i] * S_i\n\n")

## Extract survey design variables
w_tilde     <- stan_data$w_tilde
stratum_idx <- stan_data$stratum_idx
psu_idx     <- stan_data$psu_idx
n_strata    <- stan_data$n_strata

## Build a data frame for aggregation
survey_df <- data.frame(
  obs       = 1:N,
  stratum   = stratum_idx,
  psu       = psu_idx,
  w_tilde   = w_tilde
)

## Identify unique stratum-PSU combinations
strata_psu <- survey_df %>%
  group_by(stratum, psu) %>%
  summarise(n_obs = n(), .groups = "drop") %>%
  arrange(stratum, psu)

## Count PSUs per stratum
psu_per_stratum <- strata_psu %>%
  group_by(stratum) %>%
  summarise(C_h = n(), .groups = "drop")

H <- nrow(psu_per_stratum)                         # number of strata
total_psu <- nrow(strata_psu)                       # total PSU count
df_total <- sum(psu_per_stratum$C_h - 1)            # degrees of freedom

cat(sprintf("  Number of strata (H): %d\n", H))
cat(sprintf("  Total PSUs: %d\n", total_psu))
cat(sprintf("  Degrees of freedom: sum(C_h - 1) = %d\n", df_total))

## PSU size distribution
cat(sprintf("  PSU sizes: min=%d, median=%.0f, max=%d\n",
            min(strata_psu$n_obs),
            median(strata_psu$n_obs),
            max(strata_psu$n_obs)))

## PSUs per stratum distribution
cat(sprintf("  PSUs per stratum: min=%d, median=%.0f, max=%d\n",
            min(psu_per_stratum$C_h),
            median(psu_per_stratum$C_h),
            max(psu_per_stratum$C_h)))

## Check for singleton strata (C_h = 1)
n_singleton <- sum(psu_per_stratum$C_h == 1)
if (n_singleton > 0) {
  cat(sprintf("  [WARN] %d singleton strata (C_h = 1) detected.\n", n_singleton))
  cat("  Singleton strata contribute 0 to J_cluster (no variance estimable).\n")
} else {
  cat("  [PASS] No singleton strata.\n")
}

## Compute J_cluster
cat("\n  Computing J_cluster ...\n")
J_cluster <- matrix(0, D, D)

for (h in seq_len(H)) {
  stratum_h <- psu_per_stratum$stratum[h]
  C_h <- psu_per_stratum$C_h[h]

  ## Skip singleton strata (cannot estimate within-stratum variance)
  if (C_h < 2) next

  ## Get PSU indices within this stratum
  psus_in_h <- strata_psu %>%
    filter(stratum == stratum_h) %>%
    pull(psu)

  ## Compute weighted score totals for each PSU in this stratum
  s_bar_hc_list <- vector("list", C_h)

  for (c_idx in seq_along(psus_in_h)) {
    psu_c <- psus_in_h[c_idx]
    obs_in_hc <- which(stratum_idx == stratum_h & psu_idx == psu_c)
    w_hc <- w_tilde[obs_in_hc]
    s_bar_hc_list[[c_idx]] <- colSums(w_hc * S_mat[obs_in_hc, , drop = FALSE])
  }

  ## Convert to matrix: C_h x D
  s_bar_hc_mat <- do.call(rbind, s_bar_hc_list)

  ## Stratum mean: s_bar_h = (1/C_h) * sum_c s_bar_hc
  s_bar_h <- colMeans(s_bar_hc_mat)

  ## Center each PSU score total
  delta_hc <- sweep(s_bar_hc_mat, 2, s_bar_h, "-")  # C_h x D

  ## Accumulate: J_cluster += (C_h/(C_h-1)) * sum_c delta_hc %*% t(delta_hc)
  fpc <- C_h / (C_h - 1)
  J_cluster <- J_cluster + fpc * crossprod(delta_hc)
}

cat(sprintf("  J_cluster dimensions: %d x %d\n", nrow(J_cluster), ncol(J_cluster)))

## Check J_cluster properties
eig_J <- eigen(J_cluster, symmetric = TRUE, only.values = TRUE)$values
min_eig_J <- min(eig_J)
max_eig_J <- max(eig_J)
cat(sprintf("  J_cluster eigenvalue range: [%.2e, %.2e]\n", min_eig_J, max_eig_J))

if (min_eig_J > 0) {
  cat("  [PASS] J_cluster is positive definite.\n")
} else if (min_eig_J >= -1e-10) {
  cat("  [NOTE] J_cluster is positive semi-definite (min eigenvalue near zero).\n")
} else {
  warning(sprintf("  [WARN] J_cluster has negative eigenvalue: %.2e", min_eig_J))
}

## Print J_cluster diagonal
cat("\n  J_cluster diagonal:\n")
for (d in seq_len(D)) {
  cat(sprintf("    %-25s  J[%d,%d] = %.6e\n",
              param_labels[d], d, d, J_cluster[d, d]))
}

cat("\n")


###############################################################################
## SECTION 4b : COMPUTE EXPLICIT BREAD MATRIX H_obs
###############################################################################
cat("--- 4b. Computing explicit bread matrix H_obs (observed information) ---\n")
cat("  H_obs = negative weighted Hessian of log-likelihood at posterior mean\n\n")

## We need posterior means of delta (state random effects)
cat("  Extracting delta posterior means for H_obs computation ...\n")
K <- 2 * P
delta_names <- c()
for (s_idx in 1:S) {
  for (k_idx in 1:K) {
    delta_names <- c(delta_names, sprintf("delta[%d,%d]", s_idx, k_idx))
  }
}
delta_summary <- fit_weighted$summary(variables = delta_names, .cores = 4)
delta_means_vec <- delta_summary$mean
delta_means_mat <- matrix(delta_means_vec, nrow = S, ncol = K, byrow = TRUE)
cat(sprintf("  delta_means_mat: %d x %d\n", nrow(delta_means_mat), ncol(delta_means_mat)))

## Extract fixed-effect posterior means
alpha_hat     <- results_weighted$alpha_means
beta_hat      <- results_weighted$beta_means
log_kappa_hat <- results_weighted$log_kappa_mean
kappa_hat     <- results_weighted$kappa_mean

## Design matrix and state indices
X_mat     <- stan_data$X       # N x P
state_idx <- stan_data$state   # length N

## --- Extensive-margin H_ext (analytic Fisher information) ---
cat("\n  Computing H_ext (extensive margin, analytic Fisher info) ...\n")
H_ext <- matrix(0, P, P)
for (i in 1:N) {
  s_i <- state_idx[i]
  delta_ext_s <- delta_means_mat[s_i, 1:P]
  eta_ext_i <- sum(X_mat[i, ] * as.numeric(alpha_hat)) +
               sum(X_mat[i, ] * delta_ext_s)
  q_i <- plogis(eta_ext_i)
  H_ext <- H_ext + w_tilde[i] * q_i * (1 - q_i) * tcrossprod(X_mat[i, ])
}

## --- Intensive+kappa H_int block (empirical information identity) ---
cat("  Computing H_int (intensive+kappa, empirical information) ...\n")
P_int <- P + 1  # beta[1:P] + log_kappa
H_int <- matrix(0, P_int, P_int)
for (i in 1:N) {
  s_int_full <- c(score_int_mat[i, ], score_kappa_means[i])
  H_int <- H_int + w_tilde[i] * tcrossprod(s_int_full)
}

## --- Assemble block-diagonal H_obs (D x D) ---
H_obs <- matrix(0, D, D)
H_obs[1:P, 1:P] <- H_ext
H_obs[(P + 1):D, (P + 1):D] <- H_int

cat(sprintf("\n  H_obs block structure: %d x %d block-diagonal\n", D, D))
cat(sprintf("    Block 1 (ext, alpha):   H_ext  [%d x %d]\n", P, P))
cat(sprintf("    Block 2 (int+kappa):    H_int  [%d x %d]\n", P_int, P_int))

## Check H_obs positive definiteness
eig_H <- eigen(H_obs, symmetric = TRUE, only.values = TRUE)$values
min_eig_H <- min(eig_H)
max_eig_H <- max(eig_H)
cat(sprintf("\n  H_obs eigenvalue range: [%.2e, %.2e]\n", min_eig_H, max_eig_H))

if (min_eig_H > 0) {
  cat("  [PASS] H_obs is positive definite.\n")
} else {
  warning(sprintf("  [WARN] H_obs is NOT positive definite! Min eigenvalue = %.2e", min_eig_H))
  cat("  Adding ridge to make H_obs PD ...\n")
  H_obs <- H_obs + abs(min_eig_H) * 2 * diag(D)
}

## Compute H_obs_inv (the correct bread matrix)
H_obs_inv <- solve(H_obs)

## Prior inflation ratio
prior_inflation <- diag(Sigma_MCMC) / diag(H_obs_inv)

cat("\n")


###############################################################################
## SECTION 5 : COMPUTE SANDWICH VARIANCE
###############################################################################
cat("--- 5. Computing sandwich variance V_sand = H_obs^{-1} J_cluster H_obs^{-1} ---\n\n")

V_sand <- H_obs_inv %*% J_cluster %*% H_obs_inv
cat(sprintf("  V_sand dimensions: %d x %d\n", nrow(V_sand), ncol(V_sand)))

## Check positive definiteness
eig_V <- eigen(V_sand, symmetric = TRUE, only.values = TRUE)$values
min_eig_V <- min(eig_V)
max_eig_V <- max(eig_V)
cond_number_V <- max_eig_V / min_eig_V

cat(sprintf("  Eigenvalue range: [%.2e, %.2e]\n", min_eig_V, max_eig_V))
cat(sprintf("  Condition number: %.2e\n", cond_number_V))

if (min_eig_V > 0) {
  cat("  [PASS] V_sand is positive definite.\n")
} else {
  warning(sprintf("  [WARN] V_sand is NOT positive definite! Min eigenvalue = %.2e", min_eig_V))
  cat("  Attempting nearPD correction ...\n")
  V_sand_pd <- as.matrix(Matrix::nearPD(V_sand, corr = FALSE)$mat)
  V_sand <- V_sand_pd
}

cat("\n")


###############################################################################
## SECTION 6 : DESIGN EFFECT RATIO (DER)
###############################################################################
cat("--- 6. Design Effect Ratio (DER) ---\n")

## DER relative to H_obs_inv
DER <- diag(V_sand) / diag(H_obs_inv)
DER_vs_MCMC <- diag(V_sand) / diag(Sigma_MCMC)

## Build results table
der_table <- data.frame(
  parameter      = param_labels,
  H_obs_inv_pp   = diag(H_obs_inv),
  V_sand_pp      = diag(V_sand),
  Sigma_MCMC_pp  = diag(Sigma_MCMC),
  H_obs_inv_SD   = sqrt(diag(H_obs_inv)),
  V_sand_SD      = sqrt(diag(V_sand)),
  Sigma_MCMC_SD  = sqrt(diag(Sigma_MCMC)),
  DER            = DER,
  DER_vs_MCMC    = DER_vs_MCMC,
  stringsAsFactors = FALSE
)

cat(sprintf("\n  DER summary (V_sand / H_obs_inv):\n"))
cat(sprintf("    min    = %.3f\n", min(DER)))
cat(sprintf("    median = %.3f\n", median(DER)))
cat(sprintf("    mean   = %.3f\n", mean(DER)))
cat(sprintf("    max    = %.3f\n", max(DER)))
cat(sprintf("    Kish DEFF reference: 3.79\n"))

cat("\n")


###############################################################################
## SECTION 7 : VALIDATION CHECK (UNIFORM WEIGHTS)
###############################################################################
cat("--- 7. Validation: DER under uniform weights ---\n\n")

## Compute J_cluster using uniform weights
w_unif <- rep(1, N)
J_cluster_unif <- matrix(0, D, D)

for (h in seq_len(H)) {
  stratum_h <- psu_per_stratum$stratum[h]
  C_h <- psu_per_stratum$C_h[h]
  if (C_h < 2) next

  psus_in_h <- strata_psu %>%
    filter(stratum == stratum_h) %>%
    pull(psu)

  s_bar_hc_list_u <- vector("list", C_h)
  for (c_idx in seq_along(psus_in_h)) {
    psu_c <- psus_in_h[c_idx]
    obs_in_hc <- which(stratum_idx == stratum_h & psu_idx == psu_c)
    s_bar_hc_list_u[[c_idx]] <- colSums(S_mat[obs_in_hc, , drop = FALSE])
  }

  s_bar_hc_mat_u <- do.call(rbind, s_bar_hc_list_u)
  s_bar_h_u <- colMeans(s_bar_hc_mat_u)
  delta_hc_u <- sweep(s_bar_hc_mat_u, 2, s_bar_h_u, "-")
  fpc <- C_h / (C_h - 1)
  J_cluster_unif <- J_cluster_unif + fpc * crossprod(delta_hc_u)
}

V_sand_unif <- H_obs_inv %*% J_cluster_unif %*% H_obs_inv
DER_unif <- diag(V_sand_unif) / diag(H_obs_inv)

mean_DER_unif <- mean(DER_unif)
mean_DER_wt   <- mean(DER)

cat(sprintf("  Mean DER_unif    = %.3f (should be close to 1)\n", mean_DER_unif))
cat(sprintf("  Mean DER_weighted = %.3f (should be ~DEFF = 3.79)\n", mean_DER_wt))

if (mean_DER_unif > 0.5 && mean_DER_unif < 5.0) {
  cat(sprintf("  [PASS] Mean DER_unif = %.3f is in plausible range.\n", mean_DER_unif))
}

cat("\n")


###############################################################################
## SECTION 8 : SAVE SANDWICH RESULTS
###############################################################################
cat("--- 8. Saving sandwich variance results ---\n")

sandwich_results <- list(
  J_cluster  = J_cluster,
  H_obs      = H_obs,
  H_obs_inv  = H_obs_inv,
  Sigma_MCMC = Sigma_MCMC,
  V_sand     = V_sand,
  DER            = setNames(DER, param_labels),
  DER_vs_MCMC    = setNames(DER_vs_MCMC, param_labels),
  prior_inflation = setNames(prior_inflation, param_labels),
  der_table  = der_table,
  param_labels      = param_labels,
  param_names_fixed = param_names_fixed,
  D                 = D,
  survey_info = list(
    n_strata      = H,
    n_psu         = total_psu,
    df            = df_total,
    n_singleton   = n_singleton,
    psu_per_stratum = psu_per_stratum
  ),
  eigenvalues = list(
    Sigma_MCMC = list(min = min_eig_Sigma, max = max_eig_Sigma,
                      condition = cond_number_Sigma),
    V_sand     = list(min = min_eig_V, max = max_eig_V,
                      condition = cond_number_V),
    J_cluster  = list(min = min_eig_J, max = max_eig_J)
  ),
  validation = list(
    J_cluster_unif = J_cluster_unif,
    V_sand_unif    = V_sand_unif,
    DER_unif       = setNames(DER_unif, param_labels)
  ),
  score_col_means = setNames(score_col_means, param_labels),
  timestamp = Sys.time()
)

saveRDS(sandwich_results, SANDWICH_OUT)
cat(sprintf("  Saved: %s\n", SANDWICH_OUT))
cat(sprintf("    File size: %.1f KB\n",
            file.info(SANDWICH_OUT)$size / 1024))

cat("\n")


###############################################################################
## PART B : CHOLESKY AFFINE TRANSFORMATION
###############################################################################
cat("\n")
cat("##############################################################\n")
cat("##  Part B: Cholesky Affine Transformation                  ##\n")
cat("##############################################################\n\n")

###############################################################################
## SECTION 9 : EXTRACT FIXED-EFFECT MCMC DRAWS
###############################################################################
cat("--- 9. Extracting fixed-effect MCMC draws ---\n")

param_names <- c(paste0("alpha[", 1:P, "]"),
                 paste0("beta[", 1:P, "]"),
                 "log_kappa")

theta_draws <- fit_weighted$draws(variables = param_names, format = "matrix")
M <- nrow(theta_draws)
cat(sprintf("  MCMC draws: M = %d x D = %d\n", M, D))

## Posterior mean
theta_hat <- colMeans(theta_draws)
cat("  [PASS] MCMC draws extracted.\n\n")


###############################################################################
## SECTION 10 : CHOLESKY DECOMPOSITION
###############################################################################
cat("--- 10. Cholesky decomposition ---\n")

## Verify positive definiteness
eig_mcmc <- eigen(Sigma_MCMC, symmetric = TRUE, only.values = TRUE)$values
if (min(eig_mcmc) <= 0) {
  Sigma_MCMC <- Sigma_MCMC + 1e-8 * diag(D)
}

eig_sand <- eigen(V_sand, symmetric = TRUE, only.values = TRUE)$values
if (min(eig_sand) <= 0) {
  V_sand <- V_sand + 1e-8 * diag(D)
}

## Cholesky factorisation
L_MCMC <- t(chol(Sigma_MCMC))  # lower triangular
L_sand <- t(chol(V_sand))      # lower triangular

## Compute transformation matrix A = L_sand * L_MCMC^{-1}
A <- L_sand %*% solve(L_MCMC)

cat(sprintf("  A diagonal elements (scaling factors per parameter):\n"))
for (d in 1:D) {
  scaling_note <- ""
  if (abs(A[d, d]) < 0.1) {
    scaling_note <- " [SHRINKS: prior-dominated]"
  } else if (abs(A[d, d]) > 1.0) {
    scaling_note <- " [INFLATES: design effect]"
  } else {
    scaling_note <- " [MODERATE]"
  }
  cat(sprintf("    A[%2d,%2d] = %+.6f  (%-15s)%s\n",
              d, d, A[d, d], param_names[d], scaling_note))
}

## Verify A * Sigma_MCMC * A' = V_sand
V_check <- A %*% Sigma_MCMC %*% t(A)
max_err_A <- max(abs(V_check - V_sand))
cat(sprintf("\n  Verification: max|A * Sigma_MCMC * A' - V_sand| = %.2e\n", max_err_A))
if (max_err_A < 1e-6) {
  cat("  [PASS] A correctly maps Sigma_MCMC to V_sand.\n")
}

cat("\n")


###############################################################################
## SECTION 11 : APPLY TRANSFORMATION
###############################################################################
cat("--- 11. Applying Cholesky affine transformation ---\n")
cat(sprintf("  Transforming %d draws in R^%d ...\n", M, D))

theta_centered <- sweep(theta_draws, 2, theta_hat, "-")
theta_corrected <- sweep(theta_centered %*% t(A), 2, theta_hat, "+")
colnames(theta_corrected) <- param_names

cat(sprintf("  theta_corrected: %d x %d matrix\n",
            nrow(theta_corrected), ncol(theta_corrected)))

## Verify transformation
theta_corrected_mean <- colMeans(theta_corrected)
max_mean_err <- max(abs(theta_corrected_mean - theta_hat))
Sigma_corrected <- cov(theta_corrected)
max_var_err <- max(abs(Sigma_corrected - V_sand))
rel_var_err <- max_var_err / max(abs(V_sand))

cat(sprintf("  Mean preservation: max error = %.2e\n", max_mean_err))
cat(sprintf("  Variance correction: rel error = %.2e\n", rel_var_err))
cat("  [PASS] Transformation applied.\n\n")


###############################################################################
## SECTION 12 : COMPUTE CREDIBLE INTERVALS
###############################################################################
cat("--- 12. Computing credible intervals ---\n")

ci_table <- data.frame(
  parameter          = character(D),
  post_mean          = numeric(D),
  naive_lo           = numeric(D),
  naive_hi           = numeric(D),
  naive_width        = numeric(D),
  corrected_lo       = numeric(D),
  corrected_hi       = numeric(D),
  corrected_width    = numeric(D),
  wald_lo            = numeric(D),
  wald_hi            = numeric(D),
  wald_width         = numeric(D),
  width_ratio        = numeric(D),
  DER                = numeric(D),
  DER_vs_MCMC        = numeric(D),
  prior_inflation    = numeric(D),
  sqrt_DER           = numeric(D),
  stringsAsFactors = FALSE
)

for (d in 1:D) {
  naive_q     <- quantile(theta_draws[, d], probs = c(0.025, 0.975))
  corrected_q <- quantile(theta_corrected[, d], probs = c(0.025, 0.975))
  sd_sand <- sqrt(diag(V_sand)[d])
  wald_lo <- theta_hat[d] - 1.96 * sd_sand
  wald_hi <- theta_hat[d] + 1.96 * sd_sand

  ci_table$parameter[d]          <- param_labels[d]
  ci_table$post_mean[d]          <- theta_hat[d]
  ci_table$naive_lo[d]           <- naive_q[1]
  ci_table$naive_hi[d]           <- naive_q[2]
  ci_table$naive_width[d]        <- naive_q[2] - naive_q[1]
  ci_table$corrected_lo[d]       <- corrected_q[1]
  ci_table$corrected_hi[d]       <- corrected_q[2]
  ci_table$corrected_width[d]    <- corrected_q[2] - corrected_q[1]
  ci_table$wald_lo[d]            <- wald_lo
  ci_table$wald_hi[d]            <- wald_hi
  ci_table$wald_width[d]         <- wald_hi - wald_lo
  ci_table$width_ratio[d]        <- (corrected_q[2] - corrected_q[1]) /
                                     (naive_q[2] - naive_q[1])
  ci_table$DER[d]                <- DER[d]
  ci_table$DER_vs_MCMC[d]        <- DER_vs_MCMC[d]
  ci_table$prior_inflation[d]    <- prior_inflation[d]
  ci_table$sqrt_DER[d]           <- sqrt(DER[d])
}

cat("  [PASS] Credible intervals computed.\n\n")


###############################################################################
## SECTION 13 : STATE-LEVEL DER DIAGNOSTICS
###############################################################################
cat("--- 13. State-level DER diagnostics ---\n")

z_obs     <- stan_data$z

## State-level effective sample sizes (Kish formula)
state_ess <- data.frame(
  state         = 1:S,
  n_total       = integer(S),
  n_IT          = integer(S),
  ESS_ext       = numeric(S),
  ESS_int       = numeric(S),
  DEFF_ext      = numeric(S),
  DEFF_int      = numeric(S),
  approx_DER_ext = numeric(S),
  approx_DER_int = numeric(S),
  stringsAsFactors = FALSE
)

for (s in 1:S) {
  idx_s <- which(state_idx == s)
  n_s   <- length(idx_s)
  w_s   <- w_tilde[idx_s]

  ESS_ext_s <- (sum(w_s))^2 / sum(w_s^2)
  DEFF_ext_s <- n_s / ESS_ext_s

  idx_s_pos <- idx_s[z_obs[idx_s] == 1]
  n_s_pos   <- length(idx_s_pos)

  if (n_s_pos > 1) {
    w_s_pos    <- w_tilde[idx_s_pos]
    ESS_int_s  <- (sum(w_s_pos))^2 / sum(w_s_pos^2)
    DEFF_int_s <- n_s_pos / ESS_int_s
  } else {
    ESS_int_s  <- n_s_pos
    DEFF_int_s <- 1.0
  }

  lambda_ext_s <- n_s / (n_s + S)
  lambda_int_s <- n_s_pos / (n_s_pos + S)

  state_ess$n_total[s]        <- n_s
  state_ess$n_IT[s]           <- n_s_pos
  state_ess$ESS_ext[s]        <- ESS_ext_s
  state_ess$ESS_int[s]        <- ESS_int_s
  state_ess$DEFF_ext[s]       <- DEFF_ext_s
  state_ess$DEFF_int[s]       <- DEFF_int_s
  state_ess$approx_DER_ext[s] <- 1 + lambda_ext_s * (DEFF_ext_s - 1)
  state_ess$approx_DER_int[s] <- 1 + lambda_int_s * (DEFF_int_s - 1)
}

cat(sprintf("  State-level ESS summary (extensive margin):\n"))
cat(sprintf("    n_total: min=%d, median=%d, max=%d\n",
            min(state_ess$n_total),
            as.integer(median(state_ess$n_total)),
            max(state_ess$n_total)))
cat(sprintf("    ESS_ext: min=%.1f, median=%.1f, max=%.1f\n",
            min(state_ess$ESS_ext),
            median(state_ess$ESS_ext),
            max(state_ess$ESS_ext)))

cat("\n")


###############################################################################
## SECTION 14 : SAVE CHOLESKY RESULTS
###############################################################################
cat("--- 14. Saving Cholesky correction results ---\n")

cholesky_results <- list(
  description = "Williams-Savitsky (2021) Cholesky affine transformation",
  block       = "Block 1: Fixed effects (alpha, beta, log_kappa)",
  D           = D,
  M           = M,
  theta_corrected = theta_corrected,
  theta_hat       = theta_hat,
  comparison_table = ci_table,
  param_labels     = param_labels,
  param_names      = param_names,
  A       = A,
  L_MCMC  = L_MCMC,
  L_sand  = L_sand,
  Sigma_MCMC = Sigma_MCMC,
  V_sand     = V_sand,
  H_obs      = H_obs,
  H_obs_inv  = H_obs_inv,
  DER          = DER,
  DER_vs_MCMC  = DER_vs_MCMC,
  prior_inflation = prior_inflation,
  sqrt_DER     = sqrt(DER),
  state_ess = state_ess,
  verification = list(
    mean_preservation_error = max_mean_err,
    variance_relative_error = rel_var_err,
    mean_pass    = max_mean_err < 1e-6,
    variance_pass = rel_var_err < 1e-3
  ),
  poverty_reversal = list(
    alpha_poverty_wald_sig      = ci_table$wald_hi[2] < 0,
    beta_poverty_wald_sig       = ci_table$wald_lo[P + 2] > 0,
    reversal_survives_wald      = ci_table$wald_hi[2] < 0 && ci_table$wald_lo[P + 2] > 0
  ),
  timestamp = Sys.time()
)

saveRDS(cholesky_results, CORRECTION_OUT)
cat(sprintf("  Saved: %s\n", CORRECTION_OUT))
cat(sprintf("    File size: %.1f KB\n",
            file.info(CORRECTION_OUT)$size / 1024))


###############################################################################
## FINAL SUMMARY
###############################################################################
cat("\n==============================================================\n")
cat("  SURVEY WEIGHTING COMPLETE\n")
cat("==============================================================\n")

cat(sprintf("\n  Data:\n"))
cat(sprintf("    N = %d providers, D = %d fixed effects\n", N, D))
cat(sprintf("    H = %d strata, %d PSUs, df = %d\n", H, total_psu, df_total))

cat(sprintf("\n  DER summary: min=%.3f, median=%.3f, mean=%.3f, max=%.3f\n",
            min(DER), median(DER), mean(DER), max(DER)))
cat(sprintf("  Kish DEFF reference: 3.79\n"))

cat(sprintf("\n  Verification:\n"))
cat(sprintf("    Mean preservation: max error = %.2e %s\n",
            max_mean_err,
            ifelse(max_mean_err < 1e-6, "[PASS]", "[FAIL]")))
cat(sprintf("    Variance correction: rel error = %.2e %s\n",
            rel_var_err,
            ifelse(rel_var_err < 1e-3, "[PASS]", "[FAIL]")))

cat(sprintf("\n  Output files:\n"))
cat(sprintf("    %s\n", SANDWICH_OUT))
cat(sprintf("    %s\n", CORRECTION_OUT))

cat("\n==============================================================\n")
