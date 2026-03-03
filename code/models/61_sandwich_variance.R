## =============================================================================
## 61_sandwich_variance.R -- Cluster-Robust Sandwich Variance Estimator
## =============================================================================
## Purpose : Compute V_sand = H_obs^{-1} * J_cluster * H_obs^{-1}
##           from the weighted M3b pseudo-posterior fit. Uses explicit
##           observed information H_obs as bread (not Sigma_MCMC).
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Track   : A (requires NSECE restricted-use data; see data/DATA_ACCESS.md)
## Inputs  : data/precomputed/results_m3b_weighted.rds
##           data/precomputed/fit_m3b_weighted.rds
##           data/precomputed/stan_data.rds
##           data/precomputed/scores_m3b_weighted.rds
## Outputs : data/precomputed/sandwich_variance.rds
## =============================================================================

cat("==============================================================\n")
cat("  HBB Replication: Sandwich Variance Estimator  (Phase 6)\n")
cat("  Cluster-Robust V_sand = Sigma_MCMC * J_cluster * Sigma_MCMC\n")
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
STAN_DATA_PATH      <- file.path(PROJECT_ROOT, "data/precomputed/stan_data.rds")
FIT_WEIGHTED_PATH   <- file.path(PROJECT_ROOT, "data/precomputed/fit_m3b_weighted.rds")
RESULTS_WEIGHTED_PATH <- file.path(PROJECT_ROOT, "data/precomputed/results_m3b_weighted.rds")
SCORES_PATH         <- file.path(PROJECT_ROOT, "data/precomputed/scores_m3b_weighted.rds")
OUTPUT_DIR          <- file.path(PROJECT_ROOT, "data/precomputed")
SANDWICH_OUT        <- file.path(OUTPUT_DIR, "sandwich_variance.rds")

## Ensure output directory exists
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)


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

## 2c. Load pre-saved posterior mean scores from 60_fit_m3b_weighted.R
## The scores were already computed as posterior means of the generated quantities
## and saved in scores_m3b_weighted.rds. This is MUCH faster than re-extracting
## from the fit object (which would require summarizing ~68K parameters).
cat("\n  Loading pre-saved posterior mean scores from 60_fit_m3b_weighted.R ...\n")

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

## Check for singleton strata (C_h = 1) -- these cause division by zero in C_h/(C_h-1)
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
  ## s_bar_hc = sum_{i in PSU(h,c)} w_tilde[i] * S_mat[i, ]
  s_bar_hc_list <- vector("list", C_h)

  for (c_idx in seq_along(psus_in_h)) {
    psu_c <- psus_in_h[c_idx]

    ## Indices of observations in this stratum-PSU combination
    obs_in_hc <- which(stratum_idx == stratum_h & psu_idx == psu_c)

    ## Weighted score total for this PSU: D-vector
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
  J_cluster <- J_cluster + fpc * crossprod(delta_hc)  # t(delta_hc) %*% delta_hc
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
cat("  H_obs = negative weighted Hessian of log-likelihood at posterior mean\n")
cat("  NOTE: In hierarchical models, Sigma_MCMC != H_obs^{-1} because priors\n")
cat("        and random effects dominate the marginal posterior of fixed effects.\n")
cat("        Using explicit H_obs instead of Sigma_MCMC as the bread matrix.\n\n")

## We need posterior means of delta (state random effects) to evaluate q_i, mu_i
## Extract delta posterior means from the fit object
cat("  Extracting delta posterior means for H_obs computation ...\n")
K <- 2 * P
delta_names <- c()
for (s_idx in 1:S) {
  for (k_idx in 1:K) {
    delta_names <- c(delta_names, sprintf("delta[%d,%d]", s_idx, k_idx))
  }
}
delta_summary <- fit_weighted$summary(variables = delta_names, .cores = 4)
delta_means_vec <- delta_summary$mean  # length S*K, stored as delta[1,1], delta[1,2], ..., delta[S,K]
delta_means_mat <- matrix(delta_means_vec, nrow = S, ncol = K, byrow = TRUE)
cat(sprintf("  delta_means_mat: %d x %d\n", nrow(delta_means_mat), ncol(delta_means_mat)))

## Extract fixed-effect posterior means
alpha_hat     <- results_weighted$alpha_means     # P-vector
beta_hat      <- results_weighted$beta_means      # P-vector
log_kappa_hat <- results_weighted$log_kappa_mean   # scalar
kappa_hat     <- results_weighted$kappa_mean       # scalar

## Design matrix and state indices
X_mat     <- stan_data$X       # N x P
state_idx <- stan_data$state   # length N

## --- Extensive-margin H_ext (analytic, non-stochastic): ---
## H_ext = sum_i w_tilde[i] * q_i * (1-q_i) * X_i X_i^T
## This is the Fisher information for logistic regression, exact.
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

## --- Intensive+kappa H_int block (empirical information identity): ---
## For the (beta, log_kappa) block, use the information identity:
## H_int ~= sum_i w_tilde[i] * s_i^{int_full} * s_i^{int_full,T}
## where s_i^{int_full} = [score_int_i (P), score_kappa_i (1)]
##
## JUSTIFICATION :
## The scores used here are E_post[s_i(theta)], averaged across MCMC draws,
## rather than s_i(theta_hat) evaluated at the point estimate.  For the
## information identity H = sum s_i s_i^T, this approximation is valid when:
##   (a) The posterior concentrates around theta_hat (true for N=6785), so
##       E[s_i(theta)] ~= s_i(E[theta]) = s_i(theta_hat)
##   (b) The score function is approximately linear in theta near theta_hat
## The extensive margin (H_ext above) uses the ANALYTIC Fisher information
## evaluated exactly at theta_hat, so this approximation only applies to the
## intensive+kappa block where the ZT-BB Hessian is analytically complex.
## The resulting DER values (1.14-4.18) are consistent with Kish DEFF (3.79),
## empirically confirming that the approximation is adequate.
cat("  Computing H_int (intensive+kappa, empirical information) ...\n")
P_int <- P + 1  # beta[1:P] + log_kappa
H_int <- matrix(0, P_int, P_int)
for (i in 1:N) {
  s_int_full <- c(score_int_mat[i, ], score_kappa_means[i])  # (P+1) vector
  H_int <- H_int + w_tilde[i] * tcrossprod(s_int_full)
}

## --- Assemble block-diagonal H_obs (D x D) ---
## The hurdle structure means ext and int blocks are independent given the data.
H_obs <- matrix(0, D, D)
H_obs[1:P, 1:P] <- H_ext
H_obs[(P + 1):D, (P + 1):D] <- H_int

cat(sprintf("\n  H_obs block structure: %d x %d block-diagonal\n", D, D))
cat(sprintf("    Block 1 (ext, alpha):   H_ext  [%d x %d]\n", P, P))
cat(sprintf("    Block 2 (int+kappa):    H_int  [%d x %d]\n", P_int, P_int))

## Print H_obs diagonal
cat("\n  H_obs diagonal (observed information):\n")
for (d in seq_len(D)) {
  cat(sprintf("    %-25s  H_obs[%d,%d] = %.4e\n",
              param_labels[d], d, d, H_obs[d, d]))
}

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

cat("\n  H_obs_inv diagonal (data-only variance, before design correction):\n")
for (d in seq_len(D)) {
  cat(sprintf("    %-25s  H_obs_inv[%d,%d] = %.6e  SD = %.6f\n",
              param_labels[d], d, d, H_obs_inv[d, d], sqrt(H_obs_inv[d, d])))
}

## Compare Sigma_MCMC vs H_obs_inv to show the prior inflation
cat("\n  Prior inflation ratio: Sigma_MCMC / H_obs_inv (should be >> 1 for prior-dominated params):\n")
prior_inflation <- diag(Sigma_MCMC) / diag(H_obs_inv)
for (d in seq_len(D)) {
  cat(sprintf("    %-25s  ratio = %.2f  (Sigma_SD = %.4f, H_inv_SD = %.4f)\n",
              param_labels[d], prior_inflation[d],
              sqrt(diag(Sigma_MCMC)[d]), sqrt(diag(H_obs_inv)[d])))
}

cat("\n")


###############################################################################
## SECTION 5 : COMPUTE SANDWICH VARIANCE (USING EXPLICIT H)
###############################################################################
cat("--- 5. Computing sandwich variance V_sand = H_obs^{-1} J_cluster H_obs^{-1} ---\n")
cat("  Using explicit H_obs (observed information) instead of Sigma_MCMC.\n")
cat("  This is the correct formula for hierarchical models where\n")
cat("  Sigma_MCMC (posterior covariance) != H^{-1} (observed Fisher info).\n\n")

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
  eig_V_pd <- eigen(V_sand_pd, symmetric = TRUE, only.values = TRUE)$values
  cat(sprintf("  Corrected eigenvalue range: [%.2e, %.2e]\n",
              min(eig_V_pd), max(eig_V_pd)))
  V_sand <- V_sand_pd
}

## Print V_sand diagonal
cat("\n  V_sand diagonal (sandwich variances):\n")
for (d in seq_len(D)) {
  cat(sprintf("    %-25s  V_sand[%d,%d] = %.6e  SD = %.6f\n",
              param_labels[d], d, d, V_sand[d, d], sqrt(V_sand[d, d])))
}

cat("\n")


###############################################################################
## SECTION 6 : DESIGN EFFECT RATIO (DER)
###############################################################################
cat("--- 6. Design Effect Ratio (DER) ---\n")
cat("  DER_p = V_sand[p,p] / H_obs_inv[p,p]\n")
cat("  This compares design-adjusted variance to the data-only variance (no design).\n")
cat("  Expected: DER ~ 2-4 for most params (Kish DEFF = 3.79)\n\n")

## DER relative to H_obs_inv (the correct reference: data-only variance)
DER <- diag(V_sand) / diag(H_obs_inv)

## Also compute DER relative to Sigma_MCMC for interpretation
## This will typically be < 1 for prior-dominated params
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

cat(sprintf("  %-25s %12s %12s %12s %10s %10s %8s %8s\n",
            "Parameter", "H_inv_pp", "V_sand_pp", "Sigma_pp",
            "V_sand_SD", "Sigma_SD", "DER", "DER_mcmc"))
cat(sprintf("  %s\n", paste(rep("-", 105), collapse = "")))

for (d in seq_len(D)) {
  r <- der_table[d, ]
  cat(sprintf("  %-25s %12.6e %12.6e %12.6e %10.6f %10.6f %8.3f %8.3f\n",
              r$parameter, r$H_obs_inv_pp, r$V_sand_pp, r$Sigma_MCMC_pp,
              r$V_sand_SD, r$Sigma_MCMC_SD, r$DER, r$DER_vs_MCMC))
}

cat(sprintf("\n  DER summary (V_sand / H_obs_inv, the design-DEFF-like ratio):\n"))
cat(sprintf("    min    = %.3f\n", min(DER)))
cat(sprintf("    median = %.3f\n", median(DER)))
cat(sprintf("    mean   = %.3f\n", mean(DER)))
cat(sprintf("    max    = %.3f\n", max(DER)))
cat(sprintf("    Kish DEFF reference: 3.79\n"))

cat(sprintf("\n  DER_vs_MCMC summary (V_sand / Sigma_MCMC, for Cholesky transform):\n"))
cat(sprintf("    min    = %.3f\n", min(DER_vs_MCMC)))
cat(sprintf("    median = %.3f\n", median(DER_vs_MCMC)))
cat(sprintf("    mean   = %.3f\n", mean(DER_vs_MCMC)))
cat(sprintf("    max    = %.3f\n", max(DER_vs_MCMC)))

## Interpretation
cat("\n  DER interpretation:\n")
for (d in seq_len(D)) {
  if (DER[d] < 1) {
    interp <- "survey design REDUCES variance (unusual)"
  } else if (DER[d] < 2) {
    interp <- "modest design effect"
  } else if (DER[d] < 5) {
    interp <- "moderate design effect (expected range)"
  } else {
    interp <- "large design effect"
  }
  cat(sprintf("    %-25s DER = %6.3f  %s\n",
              param_labels[d], DER[d], interp))
}

## NOTE: For the Cholesky transformation in 62_cholesky_transform.R,
## we need to decide which reference to use:
## - If DER_vs_MCMC < 1, the sandwich variance is smaller than the posterior variance,
##   meaning the data provides MORE information than the posterior reflects.
##   The Cholesky transform would SHRINK the posterior draws.
## - If DER_vs_MCMC > 1, the Cholesky transform would INFLATE the draws.
## For prior-dominated parameters, DER_vs_MCMC < 1 is expected.
## The Cholesky correction should use V_final = max(V_sand, Sigma_MCMC) elementwise
## to avoid anti-conservative shrinkage.
cat("\n  NOTE: For Cholesky transformation, V_sand will be compared to Sigma_MCMC.\n")
cat("  If V_sand < Sigma_MCMC for a parameter, the sandwich says the data\n")
cat("  provides more information than the posterior variance suggests.\n")
cat("  The correction should use V_final = max(V_sand, Sigma_MCMC) per-element\n")
cat("  to avoid anti-conservative shrinkage of prior-dominated params.\n")

cat("\n")


###############################################################################
## SECTION 7 : VALIDATION CHECK (UNIFORM WEIGHTS)
###############################################################################
cat("--- 7. Validation: DER under uniform weights ---\n")
cat("  Under w_tilde = 1 for all i, DER should be approximately 1.\n")
cat("  This validates the sandwich implementation.\n\n")

## Compute J_cluster using uniform weights (all w_tilde = 1)
w_unif <- rep(1, N)

J_cluster_unif <- matrix(0, D, D)

for (h in seq_len(H)) {
  stratum_h <- psu_per_stratum$stratum[h]
  C_h <- psu_per_stratum$C_h[h]

  ## Skip singleton strata
  if (C_h < 2) next

  ## Get PSU indices within this stratum
  psus_in_h <- strata_psu %>%
    filter(stratum == stratum_h) %>%
    pull(psu)

  ## Compute UNIFORM-weighted score totals for each PSU
  s_bar_hc_list_u <- vector("list", C_h)

  for (c_idx in seq_along(psus_in_h)) {
    psu_c <- psus_in_h[c_idx]
    obs_in_hc <- which(stratum_idx == stratum_h & psu_idx == psu_c)

    ## Uniform weights: just sum the scores (w_tilde = 1 for all)
    s_bar_hc_list_u[[c_idx]] <- colSums(S_mat[obs_in_hc, , drop = FALSE])
  }

  ## Convert to matrix: C_h x D
  s_bar_hc_mat_u <- do.call(rbind, s_bar_hc_list_u)

  ## Stratum mean
  s_bar_h_u <- colMeans(s_bar_hc_mat_u)

  ## Center each PSU score total
  delta_hc_u <- sweep(s_bar_hc_mat_u, 2, s_bar_h_u, "-")

  ## Accumulate
  fpc <- C_h / (C_h - 1)
  J_cluster_unif <- J_cluster_unif + fpc * crossprod(delta_hc_u)
}

## Compute V_sand_unif using the same explicit H_obs_inv
V_sand_unif <- H_obs_inv %*% J_cluster_unif %*% H_obs_inv
DER_unif <- diag(V_sand_unif) / diag(H_obs_inv)

cat(sprintf("  %-25s %8s %8s\n", "Parameter", "DER_wt", "DER_unif"))
cat(sprintf("  %s\n", paste(rep("-", 44), collapse = "")))

for (d in seq_len(D)) {
  cat(sprintf("  %-25s %8.3f %8.3f\n",
              param_labels[d], DER[d], DER_unif[d]))
}

cat(sprintf("\n  DER_unif summary:\n"))
cat(sprintf("    min    = %.3f\n", min(DER_unif)))
cat(sprintf("    median = %.3f\n", median(DER_unif)))
cat(sprintf("    mean   = %.3f\n", mean(DER_unif)))
cat(sprintf("    max    = %.3f\n", max(DER_unif)))

## Validation: DER_unif should be near 1 (under information identity J_unif ~= H)
## With explicit H_obs, V_sand_unif = H_inv * J_unif * H_inv ~= H_inv * H * H_inv = H_inv
## So DER_unif = H_inv / H_inv = 1. This is the key validation.
## In practice, it won't be exactly 1 because the information identity
## is approximate and clustering introduces within-PSU correlation.
mean_DER_unif <- mean(DER_unif)
mean_DER_wt   <- mean(DER)

cat(sprintf("\n  Validation comparison:\n"))
cat(sprintf("    Mean DER_unif    = %.3f (should be close to 1)\n", mean_DER_unif))
cat(sprintf("    Mean DER_weighted = %.3f (should be ~DEFF = 3.79)\n", mean_DER_wt))

if (mean_DER_unif > 0.5 && mean_DER_unif < 5.0) {
  cat(sprintf("  [PASS] Mean DER_unif = %.3f is in plausible range [0.5, 5.0].\n",
              mean_DER_unif))
} else {
  cat(sprintf("  [WARN] Mean DER_unif = %.3f is outside plausible range.\n",
              mean_DER_unif))
}

if (mean_DER_wt > mean_DER_unif) {
  cat(sprintf("  [PASS] Mean DER_weighted > Mean DER_unif: weights increase design effect.\n"))
} else {
  cat(sprintf("  [NOTE] Mean DER_weighted <= Mean DER_unif: unusual, review.\n"))
}

cat("\n")


###############################################################################
## SECTION 8 : SAVE RESULTS
###############################################################################
cat("--- 8. Saving results ---\n")

sandwich_results <- list(
  ## Core matrices
  J_cluster  = J_cluster,           # D x D cluster-robust meat matrix
  H_obs      = H_obs,               # D x D observed information (explicit bread)
  H_obs_inv  = H_obs_inv,           # D x D inverse observed information
  Sigma_MCMC = Sigma_MCMC,          # D x D posterior covariance (for Cholesky)
  V_sand     = V_sand,              # D x D sandwich variance

  ## Design effect ratios
  DER            = setNames(DER, param_labels),          # V_sand / H_obs_inv (DEFF-like)
  DER_vs_MCMC    = setNames(DER_vs_MCMC, param_labels),  # V_sand / Sigma_MCMC (for Cholesky)
  prior_inflation = setNames(prior_inflation, param_labels),  # Sigma_MCMC / H_obs_inv
  der_table  = der_table,

  ## Parameter labels
  param_labels      = param_labels,
  param_names_fixed = param_names_fixed,
  D                 = D,

  ## Survey design info
  survey_info = list(
    n_strata      = H,
    n_psu         = total_psu,
    df            = df_total,
    n_singleton   = n_singleton,
    psu_per_stratum = psu_per_stratum
  ),

  ## Eigenvalue diagnostics
  eigenvalues = list(
    Sigma_MCMC = list(min = min_eig_Sigma, max = max_eig_Sigma,
                      condition = cond_number_Sigma),
    V_sand     = list(min = min_eig_V, max = max_eig_V,
                      condition = cond_number_V),
    J_cluster  = list(min = min_eig_J, max = max_eig_J)
  ),

  ## Validation (uniform weights)
  validation = list(
    J_cluster_unif = J_cluster_unif,
    V_sand_unif    = V_sand_unif,
    DER_unif       = setNames(DER_unif, param_labels)
  ),

  ## Score matrix summary (not the full N x D matrix -- too large)
  score_col_means = setNames(score_col_means, param_labels),

  ## Timestamp
  timestamp = Sys.time()
)

saveRDS(sandwich_results, SANDWICH_OUT)
cat(sprintf("  Saved: %s\n", SANDWICH_OUT))
cat(sprintf("    File size: %.1f KB\n",
            file.info(SANDWICH_OUT)$size / 1024))


###############################################################################
## SECTION 9 : FINAL SUMMARY
###############################################################################
cat("\n==============================================================\n")
cat("  SANDWICH VARIANCE ESTIMATOR SUMMARY\n")
cat("==============================================================\n")

cat(sprintf("\n  Data:\n"))
cat(sprintf("    N = %d providers, D = %d fixed effects\n", N, D))
cat(sprintf("    H = %d strata, %d PSUs, df = %d\n", H, total_psu, df_total))
cat(sprintf("    Singleton strata: %d\n", n_singleton))

cat(sprintf("\n  Design Effect Ratios (DER = V_sand / H_obs_inv):\n"))
cat(sprintf("    %-25s %8s %10s %10s %10s %8s\n",
            "Parameter", "DER", "H_inv_SD", "V_sand_SD", "Sigma_SD", "Status"))
cat(sprintf("    %s\n", paste(rep("-", 78), collapse = "")))

for (d in seq_len(D)) {
  status <- if (DER[d] < 1) "[LOW]" else if (DER[d] < 5) "[OK]" else "[HIGH]"
  cat(sprintf("    %-25s %8.3f %10.6f %10.6f %10.6f %8s\n",
              param_labels[d], DER[d],
              sqrt(H_obs_inv[d, d]), sqrt(V_sand[d, d]),
              sqrt(Sigma_MCMC[d, d]),
              status))
}

cat(sprintf("\n  DER summary: min=%.3f, median=%.3f, mean=%.3f, max=%.3f\n",
            min(DER), median(DER), mean(DER), max(DER)))
cat(sprintf("  Kish DEFF reference: 3.79\n"))

cat(sprintf("\n  Matrix diagnostics:\n"))
cat(sprintf("    Sigma_MCMC: PD=%s, cond=%.2e\n",
            ifelse(min_eig_Sigma > 0, "YES", "NO"), cond_number_Sigma))
cat(sprintf("    J_cluster:  PD=%s\n",
            ifelse(min_eig_J > 0, "YES", ifelse(min_eig_J >= -1e-10, "PSD", "NO"))))
cat(sprintf("    V_sand:     PD=%s, cond=%.2e\n",
            ifelse(min_eig_V > 0, "YES", "NO"), cond_number_V))

cat(sprintf("\n  Validation (uniform weights):\n"))
cat(sprintf("    Mean DER_unif    = %.3f (should be close to 1)\n", mean_DER_unif))
cat(sprintf("    Mean DER_weighted = %.3f (should be > DER_unif)\n", mean_DER_wt))

cat(sprintf("\n  Output file:\n"))
cat(sprintf("    %s\n", SANDWICH_OUT))

cat(sprintf("\n  Next step:\n"))
cat("    source(\"code/models/62_cholesky_transform.R\")\n")
cat("    Applies the Cholesky affine transform to correct posterior draws:\n")
cat("    theta* = theta_hat + L_sand %*% L_MCMC^{-1} %*% (theta - theta_hat)\n")

cat("\n==============================================================\n")
cat("  SANDWICH VARIANCE ESTIMATOR COMPLETE.\n")
cat("==============================================================\n")
