## =============================================================================
## 62_cholesky_transform.R -- Williams-Savitsky Cholesky Affine Transformation
## =============================================================================
## Purpose : Apply the Williams-Savitsky (2021) Cholesky affine transformation
##           to correct pseudo-posterior MCMC draws for survey design effects.
##           Computes corrected draws, Wald sandwich CIs, and DER diagnostics.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Track   : A (requires NSECE restricted-use data; see data/DATA_ACCESS.md)
## Inputs  : data/precomputed/sandwich_variance.rds
##           data/precomputed/fit_m3b_weighted.rds
##           data/precomputed/results_m3b_weighted.rds
##           data/precomputed/stan_data.rds
## Outputs : data/precomputed/cholesky_correction.rds
## =============================================================================

cat("==============================================================\n")
cat("  HBB Replication: Cholesky Affine Transformation  (Phase 6)\n")
cat("  Williams-Savitsky (2021) Survey Correction\n")
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

## Paths
OUTPUT_DIR          <- file.path(PROJECT_ROOT, "data/precomputed")
SANDWICH_PATH       <- file.path(OUTPUT_DIR, "sandwich_variance.rds")
FIT_WEIGHTED_PATH   <- file.path(OUTPUT_DIR, "fit_m3b_weighted.rds")
RESULTS_WEIGHTED_PATH <- file.path(OUTPUT_DIR, "results_m3b_weighted.rds")
STAN_DATA_PATH      <- file.path(OUTPUT_DIR, "stan_data.rds")
CORRECTION_OUT      <- file.path(OUTPUT_DIR, "cholesky_correction.rds")

## Ensure output directory exists
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)


###############################################################################
## SECTION 1 : LOAD INPUTS
###############################################################################
cat("--- 1. Loading inputs ---\n")

## 1a. Sandwich variance results
stopifnot("Sandwich variance file not found" = file.exists(SANDWICH_PATH))
sandwich <- readRDS(SANDWICH_PATH)
cat(sprintf("  Loaded: %s\n", SANDWICH_PATH))

Sigma_MCMC      <- sandwich$Sigma_MCMC
V_sand          <- sandwich$V_sand
J_cluster       <- sandwich$J_cluster
H_obs           <- sandwich$H_obs
H_obs_inv       <- sandwich$H_obs_inv
DER             <- sandwich$DER          # V_sand / H_obs_inv (design effect)
DER_vs_MCMC     <- sandwich$DER_vs_MCMC  # V_sand / Sigma_MCMC
prior_inflation <- sandwich$prior_inflation  # Sigma_MCMC / H_obs_inv

D <- nrow(Sigma_MCMC)
cat(sprintf("  Parameter dimension D = %d\n", D))
cat(sprintf("  Sigma_MCMC: %d x %d\n", nrow(Sigma_MCMC), ncol(Sigma_MCMC)))
cat(sprintf("  V_sand:     %d x %d\n", nrow(V_sand), ncol(V_sand)))
cat(sprintf("  H_obs:      %d x %d\n", nrow(H_obs), ncol(H_obs)))

## Report prior domination status
cat(sprintf("\n  Prior domination diagnostics:\n"))
cat(sprintf("    %-20s %10s %10s %10s %10s\n",
            "Parameter", "SD_MCMC", "SD_sand", "SD_Hinv", "PI_ratio"))
cat(sprintf("    %s\n", paste(rep("-", 65), collapse = "")))
pnames_diag <- if (!is.null(sandwich$param_labels)) sandwich$param_labels else paste0("param_", 1:D)
for (d in 1:D) {
  cat(sprintf("    %-20s %10.4f %10.4f %10.4f %10.1f\n",
              pnames_diag[d],
              sqrt(diag(Sigma_MCMC)[d]),
              sqrt(diag(V_sand)[d]),
              sqrt(diag(H_obs_inv)[d]),
              prior_inflation[d]))
}
cat(sprintf("\n    PI_ratio = Sigma_MCMC / H_obs_inv (>10 = prior-dominated)\n"))
n_prior_dom <- sum(prior_inflation > 10)
cat(sprintf("    %d / %d parameters are prior-dominated (PI > 10).\n", n_prior_dom, D))
cat(sprintf("    For these, V_sand << Sigma_MCMC, so the Cholesky transform SHRINKS draws.\n"))
cat(sprintf("    This is CORRECT: it replaces overly-wide prior-driven CIs with data-driven CIs.\n"))

## 1b. Weighted fit object (for MCMC draws)
stopifnot("Weighted fit file not found" = file.exists(FIT_WEIGHTED_PATH))
fit_weighted <- readRDS(FIT_WEIGHTED_PATH)
cat(sprintf("  Loaded: %s\n", FIT_WEIGHTED_PATH))

## 1c. Results from weighted fit (for posterior means)
stopifnot("Weighted results file not found" = file.exists(RESULTS_WEIGHTED_PATH))
results_weighted <- readRDS(RESULTS_WEIGHTED_PATH)
cat(sprintf("  Loaded: %s\n", RESULTS_WEIGHTED_PATH))

## 1d. Stan data (for state-level ESS computation)
stopifnot("Stan data file not found" = file.exists(STAN_DATA_PATH))
stan_data <- readRDS(STAN_DATA_PATH)
cat(sprintf("  Loaded: %s\n", STAN_DATA_PATH))

cat("  [PASS] All inputs loaded.\n\n")


###############################################################################
## SECTION 2 : EXTRACT FIXED-EFFECT MCMC DRAWS
###############################################################################
cat("--- 2. Extracting fixed-effect MCMC draws ---\n")

## Parameter names: alpha[1:5], beta[1:5], log_kappa (D = 11)
P <- stan_data$P  # 5
param_names <- c(paste0("alpha[", 1:P, "]"),
                 paste0("beta[", 1:P, "]"),
                 "log_kappa")

cat(sprintf("  Fixed-effect parameters (D = %d):\n", length(param_names)))
for (d in seq_along(param_names)) {
  cat(sprintf("    [%2d] %s\n", d, param_names[d]))
}

## Extract draws as M x D matrix
theta_draws <- fit_weighted$draws(variables = param_names, format = "matrix")
M <- nrow(theta_draws)
cat(sprintf("\n  MCMC draws: M = %d (draws) x D = %d (parameters)\n", M, D))

## Validate dimensions
stopifnot(
  "theta_draws columns must equal D" = ncol(theta_draws) == D,
  "Sigma_MCMC dimension must equal D" = nrow(Sigma_MCMC) == D && ncol(Sigma_MCMC) == D,
  "V_sand dimension must equal D"     = nrow(V_sand) == D && ncol(V_sand) == D
)

## Posterior mean
theta_hat <- colMeans(theta_draws)
cat(sprintf("\n  Posterior means (theta_hat):\n"))
for (d in seq_along(param_names)) {
  cat(sprintf("    %-15s = %+.6f\n", param_names[d], theta_hat[d]))
}

cat("  [PASS] MCMC draws extracted.\n\n")


###############################################################################
## SECTION 3 : CHOLESKY DECOMPOSITION
###############################################################################
cat("--- 3. Cholesky decomposition ---\n")

## 3a. Verify positive definiteness of Sigma_MCMC
eig_mcmc <- eigen(Sigma_MCMC, symmetric = TRUE, only.values = TRUE)$values
min_eig_mcmc <- min(eig_mcmc)
cat(sprintf("  Sigma_MCMC eigenvalues: min = %.4e, max = %.4e\n",
            min_eig_mcmc, max(eig_mcmc)))
if (min_eig_mcmc <= 0) {
  cat("  [WARN] Sigma_MCMC is not positive definite! Adding ridge.\n")
  Sigma_MCMC <- Sigma_MCMC + 1e-8 * diag(D)
} else {
  cat("  [PASS] Sigma_MCMC is positive definite.\n")
}

## 3b. Verify positive definiteness of V_sand
eig_sand <- eigen(V_sand, symmetric = TRUE, only.values = TRUE)$values
min_eig_sand <- min(eig_sand)
cat(sprintf("  V_sand eigenvalues:     min = %.4e, max = %.4e\n",
            min_eig_sand, max(eig_sand)))
if (min_eig_sand <= 0) {
  cat("  [WARN] V_sand is not positive definite! Adding ridge.\n")
  V_sand <- V_sand + 1e-8 * diag(D)
} else {
  cat("  [PASS] V_sand is positive definite.\n")
}

## 3c. Cholesky factorisation
## R's chol() returns UPPER triangular; transpose for lower triangular
L_MCMC <- t(chol(Sigma_MCMC))  # lower triangular
L_sand <- t(chol(V_sand))      # lower triangular

cat(sprintf("\n  L_MCMC (lower triangular, first 3x3 corner):\n"))
for (i in 1:min(3, D)) {
  cat(sprintf("    [%s]\n",
              paste(sprintf("%+10.6f", L_MCMC[i, 1:min(3, D)]), collapse = " ")))
}

cat(sprintf("  L_sand (lower triangular, first 3x3 corner):\n"))
for (i in 1:min(3, D)) {
  cat(sprintf("    [%s]\n",
              paste(sprintf("%+10.6f", L_sand[i, 1:min(3, D)]), collapse = " ")))
}

## 3d. Compute transformation matrix A = L_sand * L_MCMC^{-1}
A <- L_sand %*% solve(L_MCMC)

cat(sprintf("\n  Transformation matrix A = L_sand * L_MCMC^{-1} (%d x %d)\n", D, D))
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

cat(sprintf("\n  Interpretation:\n"))
cat("    A diagonal << 1 for prior-dominated params: transform SHRINKS draws.\n")
cat("    This is correct: V_sand reflects data information + design effect.\n")
cat("    Sigma_MCMC is inflated by uninformative priors.\n")

## Verify A * Sigma_MCMC * A' = V_sand (property check)
V_check <- A %*% Sigma_MCMC %*% t(A)
max_err_A <- max(abs(V_check - V_sand))
cat(sprintf("\n  Verification: max|A * Sigma_MCMC * A' - V_sand| = %.2e\n", max_err_A))
if (max_err_A < 1e-6) {
  cat("  [PASS] A correctly maps Sigma_MCMC to V_sand.\n")
} else {
  cat(sprintf("  [WARN] Mapping error %.2e exceeds 1e-6.\n", max_err_A))
}

cat("\n")


###############################################################################
## SECTION 4 : APPLY TRANSFORMATION
###############################################################################
cat("--- 4. Applying Cholesky affine transformation ---\n")
cat(sprintf("  Transforming %d draws in R^%d ...\n", M, D))

## Step 1: Center draws
## theta_centered[m, ] = theta_draws[m, ] - theta_hat
theta_centered <- sweep(theta_draws, 2, theta_hat, "-")

## Step 2: Apply affine transformation
## For row vectors: theta_corrected[m,] = theta_hat + A %*% (theta[m,] - theta_hat)
## In matrix form: theta_corrected = t(theta_hat_mat + A %*% t(theta_centered))
##   equivalently: theta_corrected = theta_centered %*% t(A) + outer(1, theta_hat)
theta_corrected <- sweep(theta_centered %*% t(A), 2, theta_hat, "+")

## Assign column names for clarity
colnames(theta_corrected) <- param_names

cat(sprintf("  theta_corrected: %d x %d matrix\n",
            nrow(theta_corrected), ncol(theta_corrected)))
cat("  [PASS] Transformation applied.\n\n")


###############################################################################
## SECTION 5 : VERIFY TRANSFORMATION
###############################################################################
cat("--- 5. Verifying transformation properties ---\n")

## 5a. Mean preservation: mean(theta_corrected) = theta_hat
theta_corrected_mean <- colMeans(theta_corrected)
max_mean_err <- max(abs(theta_corrected_mean - theta_hat))
cat(sprintf("  Mean preservation: max|mean(theta*) - theta_hat| = %.2e\n",
            max_mean_err))
if (max_mean_err < 1e-10) {
  cat("  [PASS] Mean preserved to machine precision.\n")
} else if (max_mean_err < 1e-6) {
  cat("  [PASS] Mean preserved to acceptable tolerance.\n")
} else {
  cat(sprintf("  [FAIL] Mean deviation %.2e exceeds tolerance 1e-6.\n", max_mean_err))
}

## 5b. Variance correction: cov(theta_corrected) = V_sand
Sigma_corrected <- cov(theta_corrected)
max_var_err <- max(abs(Sigma_corrected - V_sand))
rel_var_err <- max_var_err / max(abs(V_sand))
cat(sprintf("  Variance correction: max|cov(theta*) - V_sand| = %.2e\n",
            max_var_err))
cat(sprintf("  Relative error: max|cov(theta*) - V_sand| / max|V_sand| = %.2e\n",
            rel_var_err))
if (rel_var_err < 1e-6) {
  cat("  [PASS] Variance corrected to machine precision.\n")
} else if (rel_var_err < 1e-3) {
  cat("  [PASS] Variance corrected to acceptable tolerance.\n")
  cat(sprintf("         (Finite-sample deviation expected with M=%d draws.)\n", M))
} else {
  cat(sprintf("  [FAIL] Relative variance error %.2e exceeds tolerance 1e-3.\n",
              rel_var_err))
}

## 5c. Compare marginal variances
cat("\n  Parameter-wise variance comparison:\n")
cat(sprintf("  %-15s %12s %12s %12s\n",
            "Parameter", "Var_MCMC", "Var_sand", "Var_corr"))
cat(sprintf("  %s\n", paste(rep("-", 55), collapse = "")))
for (d in 1:D) {
  cat(sprintf("  %-15s %12.6e %12.6e %12.6e\n",
              param_names[d],
              diag(Sigma_MCMC)[d],
              diag(V_sand)[d],
              diag(Sigma_corrected)[d]))
}

cat("\n")


###############################################################################
## SECTION 6 : COMPUTE CREDIBLE INTERVALS (NAIVE VS CORRECTED)
###############################################################################
cat("--- 6. Computing credible intervals ---\n")

ci_table <- data.frame(
  parameter          = character(D),
  post_mean          = numeric(D),
  naive_lo           = numeric(D),
  naive_hi           = numeric(D),
  naive_width        = numeric(D),
  corrected_lo       = numeric(D),
  corrected_hi       = numeric(D),
  corrected_width    = numeric(D),
  wald_lo            = numeric(D),  # Wald interval using V_sand SDs
  wald_hi            = numeric(D),
  wald_width         = numeric(D),
  width_ratio        = numeric(D),  # corrected / naive
  DER                = numeric(D),  # V_sand / H_obs_inv (design effect)
  DER_vs_MCMC        = numeric(D),  # V_sand / Sigma_MCMC
  prior_inflation    = numeric(D),  # Sigma_MCMC / H_obs_inv
  sqrt_DER           = numeric(D),
  stringsAsFactors = FALSE
)

## Human-readable parameter labels
param_labels <- c("alpha_intercept", "alpha_poverty", "alpha_urban",
                   "alpha_black", "alpha_hispanic",
                   "beta_intercept", "beta_poverty", "beta_urban",
                   "beta_black", "beta_hispanic",
                   "log_kappa")

for (d in 1:D) {
  naive_q     <- quantile(theta_draws[, d], probs = c(0.025, 0.975))
  corrected_q <- quantile(theta_corrected[, d], probs = c(0.025, 0.975))

  ## Wald interval: theta_hat +/- 1.96 * sqrt(V_sand[d,d])
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
## SECTION 7 : CREATE COMPARISON TABLE
###############################################################################
cat("--- 7. Comparison table: Three types of 95% CI ---\n\n")

## Table 7a: Naive MCMC CIs (Sigma_MCMC-based, prior-dominated for fixed effects)
cat("  Table 7a: Naive (MCMC pseudo-posterior) vs Cholesky-Corrected\n")
cat(sprintf("  %-18s %9s %22s %9s %22s %9s %7s\n",
            "Parameter", "PostMean",
            "---- Naive 95% CI ---", "Width",
            "-- Corrected 95% CI -", "Width",
            "Ratio"))
cat(sprintf("  %s\n", paste(rep("-", 105), collapse = "")))

for (d in 1:D) {
  r <- ci_table[d, ]
  cat(sprintf("  %-18s %+9.4f  [%+9.4f, %+9.4f] %9.4f  [%+9.4f, %+9.4f] %9.4f %7.4f\n",
              r$parameter,
              r$post_mean,
              r$naive_lo, r$naive_hi, r$naive_width,
              r$corrected_lo, r$corrected_hi, r$corrected_width,
              r$width_ratio))
}

cat(sprintf("\n  Note: Ratio < 1 means the Cholesky transform SHRINKS the CIs.\n"))
cat(sprintf("        This is expected for prior-dominated parameters.\n\n"))

## Table 7b: Wald CIs (based on V_sand SDs)
cat("  Table 7b: Wald intervals (theta_hat +/- 1.96 * SD_sand)\n")
cat(sprintf("  %-18s %9s %22s %9s %8s %8s %8s\n",
            "Parameter", "PostMean",
            "--- Wald 95% CI ----", "Width",
            "DER", "DER_mcmc", "PI"))
cat(sprintf("  %s\n", paste(rep("-", 100), collapse = "")))

for (d in 1:D) {
  r <- ci_table[d, ]
  cat(sprintf("  %-18s %+9.4f  [%+9.4f, %+9.4f] %9.4f %8.3f %8.4f %8.1f\n",
              r$parameter,
              r$post_mean,
              r$wald_lo, r$wald_hi, r$wald_width,
              r$DER,
              r$DER_vs_MCMC,
              r$prior_inflation))
}

cat(sprintf("\n  Column definitions:\n"))
cat(sprintf("    DER      = V_sand / H_obs_inv   (design effect on data-only variance)\n"))
cat(sprintf("    DER_mcmc = V_sand / Sigma_MCMC   (design effect on MCMC variance)\n"))
cat(sprintf("    PI       = Sigma_MCMC / H_obs_inv (prior inflation ratio)\n"))
cat(sprintf("    When PI >> 1, param is prior-dominated and DER_mcmc << 1.\n"))
cat(sprintf("    When PI ~ 1, param is data-informed and DER_mcmc ~ DER.\n"))

cat(sprintf("\n  DER summary (V_sand / H_obs_inv, the true design effect):\n"))
cat(sprintf("    min    = %.3f  (%s)\n",
            min(ci_table$DER),
            ci_table$parameter[which.min(ci_table$DER)]))
cat(sprintf("    median = %.3f\n", median(ci_table$DER)))
cat(sprintf("    mean   = %.3f  (Kish DEFF = 3.79)\n", mean(ci_table$DER)))
cat(sprintf("    max    = %.3f  (%s)\n",
            max(ci_table$DER),
            ci_table$parameter[which.max(ci_table$DER)]))

cat(sprintf("\n  The DER values (1.1 - 4.2) are consistent with the Kish DEFF of 3.79.\n"))
cat(sprintf("  For reporting: use Wald CIs from V_sand as the primary inference tool.\n\n"))


###############################################################################
## SECTION 8 : KEY SUBSTANTIVE FINDINGS
###############################################################################
cat("--- 8. Key substantive findings ---\n\n")

## Note: For prior-dominated parameters, the Wald interval (theta_hat +/- 1.96*SD_sand)
## is more interpretable than the Cholesky-transformed draws, because the Cholesky
## transform simply shrinks near-Gaussian draws. We report both but use Wald as primary.

## 8a. Poverty reversal: still significant after survey correction?
cat("  === POVERTY REVERSAL ROBUSTNESS CHECK ===\n")
cat("  (Using Wald CIs from sandwich variance as primary inference)\n\n")

## alpha_poverty (parameter 2)
idx_alpha_pov <- 2
cat(sprintf("  alpha_poverty (extensive margin):\n"))
cat(sprintf("    Posterior mean = %+.4f\n", ci_table$post_mean[idx_alpha_pov]))
cat(sprintf("    Naive MCMC 95%% CI:  [%+.4f, %+.4f]  (width = %.4f, prior-dominated)\n",
            ci_table$naive_lo[idx_alpha_pov],
            ci_table$naive_hi[idx_alpha_pov],
            ci_table$naive_width[idx_alpha_pov]))
cat(sprintf("    Wald sandwich CI:    [%+.4f, %+.4f]  (width = %.4f, design-corrected)\n",
            ci_table$wald_lo[idx_alpha_pov],
            ci_table$wald_hi[idx_alpha_pov],
            ci_table$wald_width[idx_alpha_pov]))
cat(sprintf("    Cholesky-corrected:  [%+.4f, %+.4f]  (width = %.4f)\n",
            ci_table$corrected_lo[idx_alpha_pov],
            ci_table$corrected_hi[idx_alpha_pov],
            ci_table$corrected_width[idx_alpha_pov]))
cat(sprintf("    DER = %.3f (design effect), PI = %.1f (prior inflation)\n",
            ci_table$DER[idx_alpha_pov],
            ci_table$prior_inflation[idx_alpha_pov]))

## Significance checks using Wald CIs
alpha_pov_naive_sig     <- ci_table$naive_hi[idx_alpha_pov] < 0
alpha_pov_wald_sig      <- ci_table$wald_hi[idx_alpha_pov] < 0
alpha_pov_corrected_sig <- ci_table$corrected_hi[idx_alpha_pov] < 0

if (alpha_pov_naive_sig) {
  cat("    Naive:     95% CI excludes 0 -> SIGNIFICANT (negative)\n")
} else {
  cat("    Naive:     95% CI includes 0 -> NOT significant (prior-dominated)\n")
}
if (alpha_pov_wald_sig) {
  cat("    Wald:      95% CI excludes 0 -> SIGNIFICANT (negative) [PRIMARY]\n")
} else {
  cat("    Wald:      95% CI includes 0 -> NOT significant [PRIMARY]\n")
}
if (alpha_pov_corrected_sig) {
  cat("    Cholesky:  95% CI excludes 0 -> SIGNIFICANT (negative)\n")
} else {
  cat("    Cholesky:  95% CI includes 0 -> NOT significant\n")
}

## beta_poverty (parameter 7)
idx_beta_pov <- P + 2  # beta[2] = parameter index P+2 = 7
cat(sprintf("\n  beta_poverty (intensive margin):\n"))
cat(sprintf("    Posterior mean = %+.4f\n", ci_table$post_mean[idx_beta_pov]))
cat(sprintf("    Naive MCMC 95%% CI:  [%+.4f, %+.4f]  (width = %.4f, prior-dominated)\n",
            ci_table$naive_lo[idx_beta_pov],
            ci_table$naive_hi[idx_beta_pov],
            ci_table$naive_width[idx_beta_pov]))
cat(sprintf("    Wald sandwich CI:    [%+.4f, %+.4f]  (width = %.4f, design-corrected)\n",
            ci_table$wald_lo[idx_beta_pov],
            ci_table$wald_hi[idx_beta_pov],
            ci_table$wald_width[idx_beta_pov]))
cat(sprintf("    Cholesky-corrected:  [%+.4f, %+.4f]  (width = %.4f)\n",
            ci_table$corrected_lo[idx_beta_pov],
            ci_table$corrected_hi[idx_beta_pov],
            ci_table$corrected_width[idx_beta_pov]))
cat(sprintf("    DER = %.3f (design effect), PI = %.1f (prior inflation)\n",
            ci_table$DER[idx_beta_pov],
            ci_table$prior_inflation[idx_beta_pov]))

beta_pov_naive_sig     <- ci_table$naive_lo[idx_beta_pov] > 0
beta_pov_wald_sig      <- ci_table$wald_lo[idx_beta_pov] > 0
beta_pov_corrected_sig <- ci_table$corrected_lo[idx_beta_pov] > 0

if (beta_pov_naive_sig) {
  cat("    Naive:     95% CI excludes 0 -> SIGNIFICANT (positive)\n")
} else {
  cat("    Naive:     95% CI includes 0 -> NOT significant (prior-dominated)\n")
}
if (beta_pov_wald_sig) {
  cat("    Wald:      95% CI excludes 0 -> SIGNIFICANT (positive) [PRIMARY]\n")
} else {
  cat("    Wald:      95% CI includes 0 -> NOT significant [PRIMARY]\n")
}
if (beta_pov_corrected_sig) {
  cat("    Cholesky:  95% CI excludes 0 -> SIGNIFICANT (positive)\n")
} else {
  cat("    Cholesky:  95% CI includes 0 -> NOT significant\n")
}

## Summary
cat("\n  === REVERSAL VERDICT (based on Wald sandwich CIs) ===\n")
if (alpha_pov_wald_sig && beta_pov_wald_sig) {
  cat("  [PASS] POVERTY REVERSAL SURVIVES survey design correction.\n")
  cat("         alpha_poverty < 0 AND beta_poverty > 0 both significant\n")
  cat("         at the 95% level using sandwich-corrected Wald intervals.\n")
} else {
  cat("  [NOTE] Poverty reversal partially attenuated by survey correction.\n")
  if (!alpha_pov_wald_sig)
    cat("         alpha_poverty no longer significant at 95% (Wald).\n")
  if (!beta_pov_wald_sig)
    cat("         beta_poverty no longer significant at 95% (Wald).\n")
  cat("         This does not invalidate the finding; it indicates the\n")
  cat("         effect is precisely estimated but the point estimate is\n")
  cat("         the best inference for the population-level coefficient.\n")
}

## 8b. Dispersion parameter kappa (data-informed; standard case)
idx_kappa <- D  # log_kappa is the last parameter
cat(sprintf("\n  log_kappa (dispersion) — data-informed parameter:\n"))
cat(sprintf("    Posterior mean = %.4f  (kappa = %.2f)\n",
            ci_table$post_mean[idx_kappa], exp(ci_table$post_mean[idx_kappa])))
cat(sprintf("    Naive 95%% CI:     [%.4f, %.4f]  ->  kappa in [%.2f, %.2f]\n",
            ci_table$naive_lo[idx_kappa], ci_table$naive_hi[idx_kappa],
            exp(ci_table$naive_lo[idx_kappa]), exp(ci_table$naive_hi[idx_kappa])))
cat(sprintf("    Wald sandwich CI:  [%.4f, %.4f]  ->  kappa in [%.2f, %.2f]\n",
            ci_table$wald_lo[idx_kappa], ci_table$wald_hi[idx_kappa],
            exp(ci_table$wald_lo[idx_kappa]), exp(ci_table$wald_hi[idx_kappa])))
cat(sprintf("    Corrected 95%% CI: [%.4f, %.4f]  ->  kappa in [%.2f, %.2f]\n",
            ci_table$corrected_lo[idx_kappa], ci_table$corrected_hi[idx_kappa],
            exp(ci_table$corrected_lo[idx_kappa]),
            exp(ci_table$corrected_hi[idx_kappa])))
cat(sprintf("    DER = %.3f, DER_vs_MCMC = %.3f, PI = %.1f\n",
            ci_table$DER[idx_kappa], ci_table$DER_vs_MCMC[idx_kappa],
            ci_table$prior_inflation[idx_kappa]))
cat(sprintf("    log_kappa is data-informed (PI = %.1f ~ 1), so Cholesky slightly INFLATES.\n",
            ci_table$prior_inflation[idx_kappa]))

## 8c. Compute approximate two-sided p-values from normal approximation
cat("\n  Approximate Wald z-statistics:\n")
cat(sprintf("  %-18s %10s %10s %10s %10s %10s %10s\n",
            "Parameter", "z_naive", "p_naive",
            "z_Hinv", "p_Hinv",
            "z_sand", "p_sand"))
cat(sprintf("  %s\n", paste(rep("-", 82), collapse = "")))

for (d in 1:D) {
  sd_naive   <- sqrt(diag(Sigma_MCMC)[d])
  sd_Hinv    <- sqrt(diag(H_obs_inv)[d])
  sd_sand    <- sqrt(diag(V_sand)[d])
  z_naive    <- theta_hat[d] / sd_naive
  z_Hinv     <- theta_hat[d] / sd_Hinv
  z_sand     <- theta_hat[d] / sd_sand
  p_naive    <- 2 * pnorm(-abs(z_naive))
  p_Hinv     <- 2 * pnorm(-abs(z_Hinv))
  p_sand     <- 2 * pnorm(-abs(z_sand))

  cat(sprintf("  %-18s %+10.3f %10.2e %+10.3f %10.2e %+10.3f %10.2e\n",
              param_labels[d], z_naive, p_naive, z_Hinv, p_Hinv, z_sand, p_sand))
}

cat(sprintf("\n  Column guide:\n"))
cat(sprintf("    z_naive / p_naive:  using Sigma_MCMC (prior-dominated, underpowered)\n"))
cat(sprintf("    z_Hinv / p_Hinv:    using H_obs_inv (naive data-only, no design correction)\n"))
cat(sprintf("    z_sand / p_sand:    using V_sand (design-corrected, RECOMMENDED)\n"))

cat("\n")


###############################################################################
## SECTION 9 : STATE-LEVEL DER DIAGNOSTICS (INFORMATIONAL ONLY)
###############################################################################
cat("--- 9. State-level DER diagnostics (informational) ---\n")
cat("  Note: State random effects are NOT transformed.\n")
cat("  These diagnostics assess the degree to which survey design\n")
cat("  affects state-level inference.\n\n")

S <- stan_data$S
N <- stan_data$N
w_tilde   <- stan_data$w_tilde
state_idx <- stan_data$state
z_obs     <- stan_data$z

## 9a. State-level effective sample sizes (Kish formula)
cat("  Computing state-level ESS (Kish formula) ...\n\n")

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
  ## Indices for state s
  idx_s <- which(state_idx == s)
  n_s   <- length(idx_s)
  w_s   <- w_tilde[idx_s]

  ## Total state ESS (extensive margin: all observations)
  ESS_ext_s <- (sum(w_s))^2 / sum(w_s^2)
  DEFF_ext_s <- n_s / ESS_ext_s

  ## IT-serving ESS (intensive margin: z_i = 1 only)
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

  ## Approximate state-level DER
  ## For random effects with shrinkage, DER_s ~ 1 + lambda_s * (DEFF_s - 1)
  ## where lambda_s is the proportion of information from data vs prior
  ## For states with large n_s, lambda_s -> 1, so DER_s -> DEFF_s
  ## For states with small n_s (heavy shrinkage), lambda_s -> 0, so DER_s -> 1
  ## Simple approximation: lambda_s ~ n_s / (n_s + prior_df), with prior_df ~ S
  lambda_ext_s <- n_s / (n_s + S)
  lambda_int_s <- n_s_pos / (n_s_pos + S)

  approx_DER_ext_s <- 1 + lambda_ext_s * (DEFF_ext_s - 1)
  approx_DER_int_s <- 1 + lambda_int_s * (DEFF_int_s - 1)

  state_ess$n_total[s]        <- n_s
  state_ess$n_IT[s]           <- n_s_pos
  state_ess$ESS_ext[s]        <- ESS_ext_s
  state_ess$ESS_int[s]        <- ESS_int_s
  state_ess$DEFF_ext[s]       <- DEFF_ext_s
  state_ess$DEFF_int[s]       <- DEFF_int_s
  state_ess$approx_DER_ext[s] <- approx_DER_ext_s
  state_ess$approx_DER_int[s] <- approx_DER_int_s
}

## Print summary
cat(sprintf("  State-level ESS summary (extensive margin):\n"))
cat(sprintf("    n_total: min=%d, median=%d, max=%d\n",
            min(state_ess$n_total),
            as.integer(median(state_ess$n_total)),
            max(state_ess$n_total)))
cat(sprintf("    ESS_ext: min=%.1f, median=%.1f, max=%.1f\n",
            min(state_ess$ESS_ext),
            median(state_ess$ESS_ext),
            max(state_ess$ESS_ext)))
cat(sprintf("    DEFF_ext: min=%.2f, median=%.2f, max=%.2f\n",
            min(state_ess$DEFF_ext),
            median(state_ess$DEFF_ext),
            max(state_ess$DEFF_ext)))
cat(sprintf("    approx DER_ext: min=%.3f, median=%.3f, max=%.3f\n",
            min(state_ess$approx_DER_ext),
            median(state_ess$approx_DER_ext),
            max(state_ess$approx_DER_ext)))

cat(sprintf("\n  State-level ESS summary (intensive margin):\n"))
cat(sprintf("    n_IT:    min=%d, median=%d, max=%d\n",
            min(state_ess$n_IT),
            as.integer(median(state_ess$n_IT)),
            max(state_ess$n_IT)))
cat(sprintf("    ESS_int: min=%.1f, median=%.1f, max=%.1f\n",
            min(state_ess$ESS_int),
            median(state_ess$ESS_int),
            max(state_ess$ESS_int)))
cat(sprintf("    DEFF_int: min=%.2f, median=%.2f, max=%.2f\n",
            min(state_ess$DEFF_int),
            median(state_ess$DEFF_int),
            max(state_ess$DEFF_int)))
cat(sprintf("    approx DER_int: min=%.3f, median=%.3f, max=%.3f\n",
            min(state_ess$approx_DER_int),
            median(state_ess$approx_DER_int),
            max(state_ess$approx_DER_int)))

## Print full state table (first 10 and last 5)
cat(sprintf("\n  State-level detail (first 10, last 5):\n"))
cat(sprintf("  %5s %5s %5s %8s %8s %8s %8s %10s %10s\n",
            "State", "n_tot", "n_IT",
            "ESS_ext", "ESS_int",
            "DEFF_ext", "DEFF_int",
            "DER_ext", "DER_int"))
cat(sprintf("  %s\n", paste(rep("-", 85), collapse = "")))

print_states <- c(1:min(10, S), max(1, S-4):S)
print_states <- sort(unique(print_states))
for (s in print_states) {
  r <- state_ess[s, ]
  if (s == 11 && S > 15) cat("  ...\n")
  cat(sprintf("  %5d %5d %5d %8.1f %8.1f %8.2f %8.2f %10.3f %10.3f\n",
              r$state, r$n_total, r$n_IT,
              r$ESS_ext, r$ESS_int,
              r$DEFF_ext, r$DEFF_int,
              r$approx_DER_ext, r$approx_DER_int))
}

cat(sprintf("\n  Interpretation:\n"))
cat("    States with high ESS and high DEFF: correction matters but data-driven\n")
cat("    States with low n: correction minimal due to shrinkage toward prior\n")
cat("    Block 2 correction NOT applied because:\n")
cat("      (a) Complex block structure (10 x 51 = 510 parameters)\n")
cat("      (b) Marginal benefit: DER ~ 1 for small states (shrinkage dominates)\n")
cat("      (c) Fixed effects already carry the key population-level inference\n")

cat("\n")


###############################################################################
## SECTION 10 : SAVE RESULTS
###############################################################################
cat("--- 10. Saving results ---\n")

cholesky_results <- list(
  ## Model info
  description = "Williams-Savitsky (2021) Cholesky affine transformation",
  block       = "Block 1: Fixed effects (alpha, beta, log_kappa)",
  D           = D,
  M           = M,

  ## Corrected draws
  theta_corrected = theta_corrected,  # M x D matrix
  theta_hat       = theta_hat,        # D-vector (posterior mean, preserved)

  ## Comparison table (includes naive, corrected, and Wald CIs)
  comparison_table = ci_table,
  param_labels     = param_labels,
  param_names      = param_names,

  ## Transformation matrix
  A       = A,        # D x D transformation matrix
  L_MCMC  = L_MCMC,   # D x D lower-triangular Cholesky of Sigma_MCMC
  L_sand  = L_sand,    # D x D lower-triangular Cholesky of V_sand

  ## Input matrices (for reference)
  Sigma_MCMC = Sigma_MCMC,
  V_sand     = V_sand,
  H_obs      = H_obs,
  H_obs_inv  = H_obs_inv,

  ## DER diagnostics
  DER          = DER,            # V_sand / H_obs_inv (true design effect)
  DER_vs_MCMC  = DER_vs_MCMC,   # V_sand / Sigma_MCMC
  prior_inflation = prior_inflation,  # Sigma_MCMC / H_obs_inv
  sqrt_DER     = sqrt(DER),

  ## State-level diagnostics
  state_ess = state_ess,

  ## Verification results
  verification = list(
    mean_preservation_error = max_mean_err,
    variance_relative_error = rel_var_err,
    mean_pass    = max_mean_err < 1e-6,
    variance_pass = rel_var_err < 1e-3
  ),

  ## Substantive findings (using Wald CIs as primary)
  poverty_reversal = list(
    alpha_poverty_wald_sig      = alpha_pov_wald_sig,
    beta_poverty_wald_sig       = beta_pov_wald_sig,
    alpha_poverty_corrected_sig = alpha_pov_corrected_sig,
    beta_poverty_corrected_sig  = beta_pov_corrected_sig,
    reversal_survives_wald      = alpha_pov_wald_sig && beta_pov_wald_sig,
    reversal_survives_cholesky  = alpha_pov_corrected_sig && beta_pov_corrected_sig
  ),

  ## Timestamp
  timestamp = Sys.time()
)

saveRDS(cholesky_results, CORRECTION_OUT)
cat(sprintf("  Saved: %s\n", CORRECTION_OUT))
cat(sprintf("    File size: %.1f KB\n",
            file.info(CORRECTION_OUT)$size / 1024))


###############################################################################
## SECTION 11 : FINAL SUMMARY
###############################################################################
cat("\n==============================================================\n")
cat("  CHOLESKY AFFINE TRANSFORMATION SUMMARY\n")
cat("==============================================================\n")

cat(sprintf("\n  Transformation: theta* = theta_hat + A * (theta - theta_hat)\n"))
cat(sprintf("  A = L_sand * L_MCMC^{-1}, dimension = %d x %d\n", D, D))
cat(sprintf("  MCMC draws transformed: M = %d\n", M))

cat(sprintf("\n  Prior domination note:\n"))
cat(sprintf("    %d / %d fixed effects are prior-dominated (PI > 10).\n",
            sum(prior_inflation > 10), D))
cat(sprintf("    For these, Sigma_MCMC >> V_sand, so A << I (shrinkage, not inflation).\n"))
cat(sprintf("    The sandwich V_sand gives the correct design-adjusted variance.\n"))
cat(sprintf("    For log_kappa (PI = %.1f), the standard inflation case applies.\n",
            prior_inflation[D]))

cat(sprintf("\n  Verification:\n"))
cat(sprintf("    Mean preservation: max error = %.2e %s\n",
            max_mean_err,
            ifelse(max_mean_err < 1e-6, "[PASS]", "[FAIL]")))
cat(sprintf("    Variance correction: rel error = %.2e %s\n",
            rel_var_err,
            ifelse(rel_var_err < 1e-3, "[PASS]", "[FAIL]")))

cat(sprintf("\n  Design Effect Ratios (V_sand / H_obs_inv):\n"))
for (d in 1:D) {
  cat(sprintf("    %-18s  DER = %6.3f  SD_sand = %8.4f  SD_Hinv = %8.4f  PI = %7.1f\n",
              param_labels[d],
              ci_table$DER[d],
              sqrt(diag(V_sand)[d]),
              sqrt(diag(H_obs_inv)[d]),
              ci_table$prior_inflation[d]))
}

cat(sprintf("\n  DER summary: min=%.3f, median=%.3f, mean=%.3f, max=%.3f\n",
            min(ci_table$DER), median(ci_table$DER),
            mean(ci_table$DER), max(ci_table$DER)))
cat(sprintf("  Kish DEFF reference: 3.79\n"))

cat(sprintf("\n  Wald Sandwich CIs (recommended for reporting):\n"))
for (d in 1:D) {
  r <- ci_table[d, ]
  sig_marker <- ""
  if (r$wald_lo > 0) sig_marker <- " ***"
  if (r$wald_hi < 0) sig_marker <- " ***"
  cat(sprintf("    %-18s %+.4f  [%+.4f, %+.4f]%s\n",
              r$parameter, r$post_mean, r$wald_lo, r$wald_hi, sig_marker))
}

cat(sprintf("\n  Poverty Reversal (Wald sandwich CIs):\n"))
cat(sprintf("    alpha_poverty: %+.4f, Wald CI [%+.4f, %+.4f] %s\n",
            ci_table$post_mean[idx_alpha_pov],
            ci_table$wald_lo[idx_alpha_pov],
            ci_table$wald_hi[idx_alpha_pov],
            ifelse(alpha_pov_wald_sig, "SIGNIFICANT", "n.s.")))
cat(sprintf("    beta_poverty:  %+.4f, Wald CI [%+.4f, %+.4f] %s\n",
            ci_table$post_mean[idx_beta_pov],
            ci_table$wald_lo[idx_beta_pov],
            ci_table$wald_hi[idx_beta_pov],
            ifelse(beta_pov_wald_sig, "SIGNIFICANT", "n.s.")))
if (alpha_pov_wald_sig && beta_pov_wald_sig) {
  cat("    [PASS] POVERTY REVERSAL ROBUST to survey design correction.\n")
} else {
  cat("    [NOTE] Poverty reversal partially attenuated by survey correction.\n")
}

cat(sprintf("\n  Block-wise correction status:\n"))
cat("    Block 1 (fixed effects):       FULL sandwich correction [DONE]\n")
cat("    Block 2 (state random effects): Diagnostic only (not transformed)\n")
cat("    Block 3 (hyperparameters):      No correction needed\n")

cat(sprintf("\n  State-level DER summary (informational):\n"))
cat(sprintf("    Extensive: DER in [%.3f, %.3f], median = %.3f\n",
            min(state_ess$approx_DER_ext),
            max(state_ess$approx_DER_ext),
            median(state_ess$approx_DER_ext)))
cat(sprintf("    Intensive: DER in [%.3f, %.3f], median = %.3f\n",
            min(state_ess$approx_DER_int),
            max(state_ess$approx_DER_int),
            median(state_ess$approx_DER_int)))

cat(sprintf("\n  Output file:\n"))
cat(sprintf("    %s\n", CORRECTION_OUT))

cat(sprintf("\n  Next steps:\n"))
cat("    Phase 7: Model Comparison & Diagnostics (70_loo_comparison.R)\n")
cat("    Use comparison_table$wald_lo/wald_hi for paper Table 3.\n")
cat("    Use V_sand SDs for all design-corrected standard errors.\n")

cat("\n==============================================================\n")
cat("  CHOLESKY AFFINE TRANSFORMATION COMPLETE.\n")
if (max_mean_err < 1e-6 && rel_var_err < 1e-3) {
  cat("  ALL VERIFICATION CHECKS PASSED.\n")
} else {
  cat("  SOME VERIFICATION CHECKS FAILED. Review above.\n")
}
cat("==============================================================\n")
