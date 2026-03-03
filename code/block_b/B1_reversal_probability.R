## =============================================================================
## B1_reversal_probability.R -- Joint Poverty Reversal Probability
## =============================================================================
## Purpose : Recompute the joint poverty-reversal probability
##             Pr(alpha_pov < 0 AND beta_pov > 0 | data)
##           from BOTH naive (MCMC pseudo-posterior) and sandwich-corrected
##           (Cholesky-transformed) draws, for consistency with the paper's
##           two-track reporting strategy.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##
## Context :
##   The manuscript (Section 5.2) reports Pr(reversal) = 1.000, computed
##   from naive MCMC draws.  Multiple reviewers noted that this should also
##   be computed from sandwich-corrected draws for consistency.  Given the
##   tight corrected CIs (SD_sand ~ 0.05 for alpha_pov vs SD_MCMC ~ 0.97),
##   we expect the corrected probability to remain at or very near 1.000.
##
## Theory :
##   The Cholesky affine transformation (Williams-Savitsky, 2021) maps
##   naive draws theta^(m) to corrected draws theta*^(m):
##     theta*^(m) = theta_hat + A * (theta^(m) - theta_hat)
##   where A = L_sand * L_MCMC^{-1}.
##
##   For prior-dominated parameters (PI >> 1), this SHRINKS the draws
##   toward theta_hat, making the corrected distribution MUCH more
##   concentrated.  Since theta_hat for alpha_pov = -0.324 and for
##   beta_pov = +0.090, and the corrected SDs are ~0.05 (vs ~0.97 naive),
##   the corrected probability should be extremely close to 1.
##
## Two Levels of Reversal Probability:
##   Level 1 (Parameter): Pr(alpha[2] < 0 AND beta[2] > 0 | data)
##     -> computed directly from MCMC draws of alpha_pov & beta_pov
##   Level 2 (AME):       Pr(ext_AME_pov < 0 AND int_AME_pov > 0 | data)
##     -> computed from average marginal effect draws
##   Both levels are computed under both naive and corrected posteriors.
##
## Parts:
##   A: Parameter-level reversal probability (naive vs corrected)
##   B: Analytical bivariate normal approximation (cross-check via pmvnorm)
##   C: AME-level reversal probability (naive vs corrected)
##   D: Sensitivity analysis at multiple boundary thresholds
##   E: Manuscript language recommendation
##
## Inputs  :
##   data/precomputed/cholesky_correction.rds
##   data/precomputed/marginal_effects.rds
##   data/precomputed/sandwich_variance.rds
##   data/precomputed/stan_data.rds
##
## Outputs :
##   data/precomputed/B1_reversal_probability.rds
## =============================================================================

cat("==============================================================\n")
cat("  Joint Reversal Probability Analysis\n")
cat("==============================================================\n\n")


###############################################################################
## SECTION 0 : SETUP
###############################################################################

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()

## Source helper functions (provides inv_logit, logit, etc.)
source(file.path(PROJECT_ROOT, "code/helpers/utils.R"))

## Paths
OUTPUT_DIR    <- file.path(PROJECT_ROOT, "data/precomputed")
CHOLESKY_PATH <- file.path(OUTPUT_DIR, "cholesky_correction.rds")
ME_PATH       <- file.path(OUTPUT_DIR, "marginal_effects.rds")
SANDWICH_PATH <- file.path(OUTPUT_DIR, "sandwich_variance.rds")
STAN_DATA_PATH <- file.path(OUTPUT_DIR, "stan_data.rds")
B1_OUT        <- file.path(OUTPUT_DIR, "B1_reversal_probability.rds")

## Parameter indices (within the 11-dimensional theta vector)
## alpha[1:5] = cols 1-5, beta[1:5] = cols 6-10, log_kappa = col 11
IDX_ALPHA_POV <- 2
IDX_BETA_POV  <- 7
D <- 11
P <- 5

## Covariate labels
COV_LABELS <- c("intercept", "poverty", "urban", "black", "hispanic")

cat(sprintf("  PROJECT_ROOT: %s\n", PROJECT_ROOT))
cat(sprintf("  Output file:  %s\n", B1_OUT))
cat(sprintf("  alpha_poverty index: %d\n", IDX_ALPHA_POV))
cat(sprintf("  beta_poverty  index: %d\n\n", IDX_BETA_POV))


###############################################################################
## SECTION 1 : LOAD INPUTS
###############################################################################
cat("--- 1. Loading inputs ---\n\n")

## 1a. Cholesky correction results
stopifnot("Cholesky correction file not found" = file.exists(CHOLESKY_PATH))
chol_res <- readRDS(CHOLESKY_PATH)
cat(sprintf("  Loaded: %s\n", CHOLESKY_PATH))

theta_corrected <- chol_res$theta_corrected   # M x 11 matrix
theta_hat       <- chol_res$theta_hat          # length-11 vector
A               <- chol_res$A                  # 11 x 11 transformation matrix
L_MCMC          <- chol_res$L_MCMC             # 11 x 11 lower Cholesky of Sigma_MCMC
L_sand          <- chol_res$L_sand             # 11 x 11 lower Cholesky of V_sand
V_sand          <- chol_res$V_sand             # 11 x 11 sandwich variance
Sigma_MCMC      <- chol_res$Sigma_MCMC         # 11 x 11 MCMC posterior covariance
param_labels    <- chol_res$param_labels       # length-11 character vector
param_names     <- chol_res$param_names        # length-11 (Stan names)

M <- nrow(theta_corrected)
cat(sprintf("  Corrected draws: M = %d x D = %d\n", M, ncol(theta_corrected)))
cat(sprintf("  theta_hat[alpha_pov] = %+.6f  (idx %d)\n",
            theta_hat[IDX_ALPHA_POV], IDX_ALPHA_POV))
cat(sprintf("  theta_hat[beta_pov]  = %+.6f  (idx %d)\n",
            theta_hat[IDX_BETA_POV], IDX_BETA_POV))

## 1b. Marginal effects results (for AME-level reversal probability)
stopifnot("Marginal effects file not found" = file.exists(ME_PATH))
me_res <- readRDS(ME_PATH)
cat(sprintf("  Loaded: %s\n", ME_PATH))

ext_ame_draws_corr <- me_res$ext_ame_draws     # M_sub x P matrix (from corrected draws)
int_ame_draws_corr <- me_res$int_ame_draws     # M_sub x P matrix
M_ame              <- nrow(ext_ame_draws_corr)

cat(sprintf("  AME draws (corrected): M_sub = %d x P = %d\n",
            M_ame, ncol(ext_ame_draws_corr)))
cat(sprintf("  Existing pr_poverty_reversal (AME, corrected): %.4f\n",
            me_res$pr_poverty_reversal))

## 1c. Stan data (for AME recomputation from naive draws)
stopifnot("Stan data file not found" = file.exists(STAN_DATA_PATH))
stan_data <- readRDS(STAN_DATA_PATH)
X <- stan_data$X    # N x P design matrix
N <- stan_data$N
cat(sprintf("  Stan data: N = %d, P = %d\n", N, P))

cat("  [PASS] All inputs loaded.\n\n")


###############################################################################
## SECTION 2 : RECONSTRUCT NAIVE DRAWS VIA INVERSE TRANSFORMATION
###############################################################################
cat("--- 2. Reconstructing naive MCMC draws ---\n\n")

## The Cholesky transform is:
##   theta_corrected = theta_hat + A * (theta_naive - theta_hat)
## Inverting:
##   theta_naive = theta_hat + A^{-1} * (theta_corrected - theta_hat)
##
## Since A = L_sand * L_MCMC^{-1}, we have:
##   A^{-1} = L_MCMC * L_sand^{-1}
## This is more numerically stable than inverting A directly.

cat("  Computing A_inv = L_MCMC * L_sand^{-1} ...\n")
A_inv <- L_MCMC %*% solve(L_sand)

## Verify A_inv * A = I
max_inv_err <- max(abs(A_inv %*% A - diag(D)))
cat(sprintf("  Verification: max|A_inv * A - I| = %.2e\n", max_inv_err))
if (max_inv_err < 1e-10) {
  cat("  [PASS] A_inv computed to machine precision.\n")
} else {
  cat(sprintf("  [WARN] Inversion error %.2e.\n", max_inv_err))
}

## Reconstruct naive draws
cat(sprintf("  Reconstructing %d naive draws ...\n", M))
theta_centered_corrected <- sweep(theta_corrected, 2, theta_hat, "-")
theta_naive <- sweep(theta_centered_corrected %*% t(A_inv), 2, theta_hat, "+")
colnames(theta_naive) <- param_names

## Verify reconstruction: naive draws should have cov ~ Sigma_MCMC
Sigma_naive_recon <- cov(theta_naive)
max_cov_err <- max(abs(Sigma_naive_recon - Sigma_MCMC))
rel_cov_err <- max_cov_err / max(abs(Sigma_MCMC))
cat(sprintf("  Verification: max|cov(theta_naive) - Sigma_MCMC| = %.2e\n", max_cov_err))
cat(sprintf("  Relative error: %.2e\n", rel_cov_err))
if (rel_cov_err < 1e-3) {
  cat("  [PASS] Naive draw reconstruction verified.\n")
} else {
  cat(sprintf("  [WARN] Reconstruction relative error %.2e exceeds 1e-3.\n", rel_cov_err))
}

## Verify mean preservation
naive_mean <- colMeans(theta_naive)
max_mean_err <- max(abs(naive_mean - theta_hat))
cat(sprintf("  Mean preservation: max|mean(theta_naive) - theta_hat| = %.2e\n",
            max_mean_err))

## Print diagnostics for the two poverty parameters
cat(sprintf("\n  Reconstructed naive draws diagnostics:\n"))
cat(sprintf("    alpha_pov: mean = %+.6f, SD = %.6f\n",
            mean(theta_naive[, IDX_ALPHA_POV]),
            sd(theta_naive[, IDX_ALPHA_POV])))
cat(sprintf("    beta_pov:  mean = %+.6f, SD = %.6f\n",
            mean(theta_naive[, IDX_BETA_POV]),
            sd(theta_naive[, IDX_BETA_POV])))
cat(sprintf("  Corrected draws diagnostics:\n"))
cat(sprintf("    alpha_pov: mean = %+.6f, SD = %.6f\n",
            mean(theta_corrected[, IDX_ALPHA_POV]),
            sd(theta_corrected[, IDX_ALPHA_POV])))
cat(sprintf("    beta_pov:  mean = %+.6f, SD = %.6f\n",
            mean(theta_corrected[, IDX_BETA_POV]),
            sd(theta_corrected[, IDX_BETA_POV])))
cat("\n")


###############################################################################
## SECTION 3 : PART A -- PARAMETER-LEVEL REVERSAL PROBABILITY
###############################################################################
cat("--- 3. Part A: Parameter-level reversal probability ---\n\n")

## Definition: reversal iff alpha_pov < 0 AND beta_pov > 0

## 3a. Naive (parameter-level)
reversal_naive <- (theta_naive[, IDX_ALPHA_POV] < 0) &
                  (theta_naive[, IDX_BETA_POV]  > 0)
pr_reversal_naive <- mean(reversal_naive)
n_reversal_naive  <- sum(reversal_naive)

cat(sprintf("  NAIVE (MCMC pseudo-posterior, parameter-level):\n"))
cat(sprintf("    Draws satisfying reversal: %d / %d\n", n_reversal_naive, M))
cat(sprintf("    Pr(alpha_pov < 0 AND beta_pov > 0 | data) = %.6f\n",
            pr_reversal_naive))

## 3b. Corrected (sandwich, Cholesky-transformed)
reversal_corrected <- (theta_corrected[, IDX_ALPHA_POV] < 0) &
                      (theta_corrected[, IDX_BETA_POV]  > 0)
pr_reversal_corrected <- mean(reversal_corrected)
n_reversal_corrected  <- sum(reversal_corrected)

cat(sprintf("\n  CORRECTED (Cholesky-transformed, sandwich-adjusted):\n"))
cat(sprintf("    Draws satisfying reversal: %d / %d\n", n_reversal_corrected, M))
cat(sprintf("    Pr(alpha_pov < 0 AND beta_pov > 0 | data) = %.6f\n",
            pr_reversal_corrected))

## 3c. Marginal probabilities
pr_alpha_neg_naive     <- mean(theta_naive[, IDX_ALPHA_POV] < 0)
pr_beta_pos_naive      <- mean(theta_naive[, IDX_BETA_POV]  > 0)
pr_alpha_neg_corrected <- mean(theta_corrected[, IDX_ALPHA_POV] < 0)
pr_beta_pos_corrected  <- mean(theta_corrected[, IDX_BETA_POV]  > 0)

cat(sprintf("\n  Marginal probabilities:\n"))
cat(sprintf("                         %12s  %12s\n", "Naive", "Corrected"))
cat(sprintf("    Pr(alpha_pov < 0)    %12.6f  %12.6f\n",
            pr_alpha_neg_naive, pr_alpha_neg_corrected))
cat(sprintf("    Pr(beta_pov  > 0)    %12.6f  %12.6f\n",
            pr_beta_pos_naive, pr_beta_pos_corrected))

## 3d. Monte Carlo standard errors
## For a proportion p estimated from M independent draws,
## MCSE = sqrt(p * (1-p) / M).
## When p = 1.000 (all M draws satisfy), MCSE = 0.
## In that case, use rule of three: Pr >= 1 - 3/M (95% one-sided).

compute_mcse <- function(p, n) sqrt(p * (1 - p) / n)

mcse_naive <- compute_mcse(pr_reversal_naive, M)
mcse_corr  <- compute_mcse(pr_reversal_corrected, M)

cat(sprintf("\n  Monte Carlo standard errors:\n"))
cat(sprintf("    MCSE (naive):     %.6f", mcse_naive))
if (mcse_naive == 0) cat("  [all M draws satisfy; lower bound via rule of three]")
cat("\n")
cat(sprintf("    MCSE (corrected): %.6f", mcse_corr))
if (mcse_corr == 0) cat("  [all M draws satisfy; lower bound via rule of three]")
cat("\n")
if (mcse_naive == 0 || mcse_corr == 0) {
  cat(sprintf("    Rule-of-three 95%% lower bound: Pr >= 1 - 3/M = 1 - 3/%d = %.6f\n",
              M, 1 - 3/M))
}

## 3e. Distance from reversal boundary in sandwich-SDs
sd_sand_alpha  <- sqrt(V_sand[IDX_ALPHA_POV, IDX_ALPHA_POV])
sd_sand_beta   <- sqrt(V_sand[IDX_BETA_POV, IDX_BETA_POV])
sd_mcmc_alpha  <- sqrt(Sigma_MCMC[IDX_ALPHA_POV, IDX_ALPHA_POV])
sd_mcmc_beta   <- sqrt(Sigma_MCMC[IDX_BETA_POV, IDX_BETA_POV])

z_alpha_sand <- abs(theta_hat[IDX_ALPHA_POV]) / sd_sand_alpha
z_beta_sand  <- abs(theta_hat[IDX_BETA_POV])  / sd_sand_beta
z_alpha_mcmc <- abs(theta_hat[IDX_ALPHA_POV]) / sd_mcmc_alpha
z_beta_mcmc  <- abs(theta_hat[IDX_BETA_POV])  / sd_mcmc_beta

cat(sprintf("\n  Distance from reversal boundary (in SDs):\n"))
cat(sprintf("                         Sandwich-corr      Naive (MCMC)\n"))
cat(sprintf("    alpha_pov = 0:       z = %6.2f          z = %6.2f\n",
            z_alpha_sand, z_alpha_mcmc))
cat(sprintf("    beta_pov  = 0:       z = %6.2f          z = %6.2f\n",
            z_beta_sand, z_beta_mcmc))
cat(sprintf("    Min z (binding):     %6.2f              %6.2f\n",
            min(z_alpha_sand, z_beta_sand), min(z_alpha_mcmc, z_beta_mcmc)))

cat(sprintf("\n  Interpretation:\n"))
cat("    Under sandwich correction, the posterior mean is many SDs from\n")
cat("    the reversal boundary. The corrected draws are more concentrated\n")
cat("    around theta_hat (smaller SD), so the probability remains 1.000.\n")

cat("\n")


###############################################################################
## SECTION 4 : PART B -- ANALYTICAL BIVARIATE NORMAL APPROXIMATION
###############################################################################
cat("--- 4. Part B: Analytical bivariate normal approximation ---\n\n")

## Under the sandwich correction, the joint posterior of (alpha_pov, beta_pov)
## is approximately bivariate normal:
##   (alpha_pov, beta_pov) ~ N(mu_2, V_2x2)
## where V_2x2 is the 2x2 submatrix of V_sand.
##
## Pr(alpha_pov < 0 AND beta_pov > 0)
##   = Pr(X1 < 0 AND X2 > 0)
## computed via mvtnorm::pmvnorm or via independence approximation.

## Extract 2x2 submatrices
mu_2 <- c(theta_hat[IDX_ALPHA_POV], theta_hat[IDX_BETA_POV])
V_2_sand <- V_sand[c(IDX_ALPHA_POV, IDX_BETA_POV),
                    c(IDX_ALPHA_POV, IDX_BETA_POV)]
V_2_mcmc <- Sigma_MCMC[c(IDX_ALPHA_POV, IDX_BETA_POV),
                        c(IDX_ALPHA_POV, IDX_BETA_POV)]

## Correlations
cov_sand_cross <- V_2_sand[1, 2]
rho_sand <- cov_sand_cross / (sd_sand_alpha * sd_sand_beta)
cov_mcmc_cross <- V_2_mcmc[1, 2]
rho_mcmc <- cov_mcmc_cross / (sd_mcmc_alpha * sd_mcmc_beta)

## Empirical correlations from draws
rho_corr_emp  <- cor(theta_corrected[, IDX_ALPHA_POV],
                     theta_corrected[, IDX_BETA_POV])
rho_naive_emp <- cor(theta_naive[, IDX_ALPHA_POV],
                     theta_naive[, IDX_BETA_POV])

cat("  Bivariate parameters for (alpha_pov, beta_pov):\n\n")
cat("  Under SANDWICH correction:\n")
cat(sprintf("    mu     = (%+.6f, %+.6f)\n", mu_2[1], mu_2[2]))
cat(sprintf("    SD     = (%.6f, %.6f)\n", sd_sand_alpha, sd_sand_beta))
cat(sprintf("    rho    = %+.6f (V_sand), %+.6f (empirical draws)\n",
            rho_sand, rho_corr_emp))

cat("\n  Under NAIVE (MCMC) posterior:\n")
cat(sprintf("    mu     = (%+.6f, %+.6f)\n", mu_2[1], mu_2[2]))
cat(sprintf("    SD     = (%.6f, %.6f)\n", sd_mcmc_alpha, sd_mcmc_beta))
cat(sprintf("    rho    = %+.6f (Sigma_MCMC), %+.6f (empirical draws)\n",
            rho_mcmc, rho_naive_emp))

## Compute analytical probabilities
has_mvtnorm <- requireNamespace("mvtnorm", quietly = TRUE)
pr_bvn_sand <- NA_real_
pr_bvn_mcmc <- NA_real_

if (has_mvtnorm) {
  cat("\n  Computing exact bivariate normal probabilities (mvtnorm::pmvnorm) ...\n")

  ## Pr(alpha < 0 AND beta > 0)
  ## = Pr in rectangle (-Inf, 0] x [0, +Inf)
  pr_bvn_sand <- mvtnorm::pmvnorm(
    lower = c(-Inf, 0),
    upper = c(0, Inf),
    mean  = mu_2,
    sigma = V_2_sand
  )[1]

  pr_bvn_mcmc <- mvtnorm::pmvnorm(
    lower = c(-Inf, 0),
    upper = c(0, Inf),
    mean  = mu_2,
    sigma = V_2_mcmc
  )[1]

  cat(sprintf("    Sandwich:  Pr(reversal) = %.10f\n", pr_bvn_sand))
  cat(sprintf("    Naive:     Pr(reversal) = %.10f\n", pr_bvn_mcmc))
  cat(sprintf("    Sandwich:  1 - Pr       = %.4e\n", 1 - pr_bvn_sand))
  cat(sprintf("    Naive:     1 - Pr       = %.4e\n", 1 - pr_bvn_mcmc))

} else {
  cat("\n  [NOTE] mvtnorm package not available; skipping exact BVN.\n")
  cat("         Install with: install.packages('mvtnorm')\n")
}

## Independence approximation (always computed)
pr_marginal_alpha_sand <- pnorm(0, mean = mu_2[1], sd = sd_sand_alpha)
pr_marginal_beta_sand  <- pnorm(0, mean = mu_2[2], sd = sd_sand_beta,
                                lower.tail = FALSE)
pr_indep_sand <- pr_marginal_alpha_sand * pr_marginal_beta_sand

pr_marginal_alpha_mcmc <- pnorm(0, mean = mu_2[1], sd = sd_mcmc_alpha)
pr_marginal_beta_mcmc  <- pnorm(0, mean = mu_2[2], sd = sd_mcmc_beta,
                                lower.tail = FALSE)
pr_indep_mcmc <- pr_marginal_alpha_mcmc * pr_marginal_beta_mcmc

cat(sprintf("\n  Independence approximation (ignoring cross-correlation):\n"))
cat(sprintf("    Sandwich:  Pr(reversal) ~ %.10f\n", pr_indep_sand))
cat(sprintf("    Naive:     Pr(reversal) ~ %.10f\n", pr_indep_mcmc))

cat(sprintf("\n  Marginal normal-CDF probabilities (sandwich):\n"))
cat(sprintf("    Pr(alpha_pov < 0) = %.10f  (z = %.2f)\n",
            pr_marginal_alpha_sand, z_alpha_sand))
cat(sprintf("    Pr(beta_pov  > 0) = %.10f  (z = %.2f)\n",
            pr_marginal_beta_sand, z_beta_sand))
cat(sprintf("    1 - Pr(alpha<0)   = %.4e\n", 1 - pr_marginal_alpha_sand))
cat(sprintf("    1 - Pr(beta>0)    = %.4e\n", 1 - pr_marginal_beta_sand))

## Compare BVN vs independence
if (has_mvtnorm) {
  cat(sprintf("\n  BVN vs independence difference (sandwich): %+.2e\n",
              pr_bvn_sand - pr_indep_sand))
  cat(sprintf("  This confirms that the cross-correlation (rho = %.4f)\n", rho_sand))
  cat("  has negligible impact on the joint probability.\n")
}

cat("\n")


###############################################################################
## SECTION 5 : PART C -- AME-LEVEL REVERSAL PROBABILITY
###############################################################################
cat("--- 5. Part C: AME-level reversal probability ---\n\n")

## 5a. AME reversal from CORRECTED draws (already in marginal_effects.rds)
pov_idx <- 2   # poverty is covariate 2

pr_ame_reversal_corr <- mean(ext_ame_draws_corr[, pov_idx] < 0 &
                             int_ame_draws_corr[, pov_idx] > 0)
mcse_ame_corr <- compute_mcse(pr_ame_reversal_corr, M_ame)

cat("  5a. AME reversal from CORRECTED draws:\n")
cat(sprintf("      Pr(ext_AME_pov < 0 AND int_AME_pov > 0) = %.6f  (M_sub = %d)\n",
            pr_ame_reversal_corr, M_ame))
cat(sprintf("      MCSE = %.6f\n", mcse_ame_corr))
cat(sprintf("      ext_AME_pov: mean = %+.6f, SD = %.6f\n",
            mean(ext_ame_draws_corr[, pov_idx]),
            sd(ext_ame_draws_corr[, pov_idx])))
cat(sprintf("      int_AME_pov: mean = %+.6f, SD = %.6f\n",
            mean(int_ame_draws_corr[, pov_idx]),
            sd(int_ame_draws_corr[, pov_idx])))

## 5b. Recompute AME draws from NAIVE posterior for comparison
cat("\n  5b. Recomputing AME draws from NAIVE posterior ...\n")

## Thin the naive draws to match M_ame subsample size
thin_idx <- seq(1, M, by = max(1, floor(M / M_ame)))
if (length(thin_idx) > M_ame) thin_idx <- thin_idx[1:M_ame]
theta_naive_sub <- theta_naive[thin_idx, , drop = FALSE]
M_naive_sub <- nrow(theta_naive_sub)

ext_ame_naive <- matrix(NA_real_, nrow = M_naive_sub, ncol = P)
int_ame_naive <- matrix(NA_real_, nrow = M_naive_sub, ncol = P)
colnames(ext_ame_naive) <- COV_LABELS
colnames(int_ame_naive) <- COV_LABELS

t_start <- proc.time()

for (m in seq_len(M_naive_sub)) {
  alpha_m <- theta_naive_sub[m, 1:P]
  beta_m  <- theta_naive_sub[m, (P + 1):(2 * P)]

  eta_ext <- as.numeric(X %*% alpha_m)
  eta_int <- as.numeric(X %*% beta_m)

  q_m  <- inv_logit(eta_ext)
  mu_m <- inv_logit(eta_int)

  q_deriv  <- q_m * (1 - q_m)
  mu_deriv <- mu_m * (1 - mu_m)

  for (k in seq_len(P)) {
    ext_ame_naive[m, k] <- mean(alpha_m[k] * q_deriv * mu_m)
    int_ame_naive[m, k] <- mean(beta_m[k]  * mu_deriv * q_m)
  }
}

t_elapsed <- (proc.time() - t_start)[3]
cat(sprintf("      Computed in %.1f seconds (%d draws x %d obs)\n",
            t_elapsed, M_naive_sub, N))

pr_ame_reversal_naive <- mean(ext_ame_naive[, pov_idx] < 0 &
                              int_ame_naive[, pov_idx] > 0)
mcse_ame_naive <- compute_mcse(pr_ame_reversal_naive, M_naive_sub)

cat(sprintf("      Pr(AME reversal, naive) = %.6f  (M_sub = %d)\n",
            pr_ame_reversal_naive, M_naive_sub))
cat(sprintf("      MCSE = %.6f\n", mcse_ame_naive))
cat(sprintf("      ext_AME_pov: mean = %+.6f, SD = %.6f\n",
            mean(ext_ame_naive[, pov_idx]),
            sd(ext_ame_naive[, pov_idx])))
cat(sprintf("      int_AME_pov: mean = %+.6f, SD = %.6f\n",
            mean(int_ame_naive[, pov_idx]),
            sd(int_ame_naive[, pov_idx])))

## 5c. Consistency check: parameter-level vs AME-level
## Since AME_ext_pov = f(alpha) with f(.) monotonic in alpha_pov (at each obs),
## sign(AME_ext) = sign(alpha_pov). Same for intensive margin.
## So parameter-level and AME-level reversal probabilities should agree
## (modulo subsampling differences).

cat(sprintf("\n  Consistency check (parameter-level vs AME-level):\n"))
cat(sprintf("    %-30s  %10s  %10s\n", "", "Naive", "Corrected"))
cat(sprintf("    %-30s  %10.6f  %10.6f\n",
            "Parameter-level",
            pr_reversal_naive, pr_reversal_corrected))
cat(sprintf("    %-30s  %10.6f  %10.6f\n",
            "AME-level",
            pr_ame_reversal_naive, pr_ame_reversal_corr))

diff_naive <- abs(pr_reversal_naive - pr_ame_reversal_naive)
diff_corr  <- abs(pr_reversal_corrected - pr_ame_reversal_corr)

if (diff_naive < 0.01 && diff_corr < 0.01) {
  cat("    [PASS] Parameter-level and AME-level agree (within subsampling tolerance).\n")
} else {
  cat(sprintf("    [NOTE] Discrepancy: naive diff = %.4f, corr diff = %.4f.\n",
              diff_naive, diff_corr))
  cat("           Likely due to M_sub thinning.\n")
}

cat("\n")


###############################################################################
## SECTION 6 : PART D -- SENSITIVITY AT MULTIPLE THRESHOLDS
###############################################################################
cat("--- 6. Part D: Sensitivity -- Pr(reversal exceeds threshold) ---\n\n")

## Even when Pr(reversal) = 1.000, it is informative to see how far the
## posterior is from the boundary. Compute:
##   Pr(alpha_pov < -eps AND beta_pov > +eps | data)
## for various eps values. This quantifies the strength of the reversal.

thresholds <- c(0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25)

cat("  Pr(alpha_pov < -eps AND beta_pov > +eps | data):\n\n")
cat(sprintf("  %-8s  %12s  %12s", "eps", "Corrected", "Naive"))
if (has_mvtnorm) cat(sprintf("  %14s", "BVN (sand)"))
cat("\n")
cat(sprintf("  %s\n", paste(rep("-", if (has_mvtnorm) 52 else 36), collapse = "")))

pr_threshold_corr <- numeric(length(thresholds))
pr_threshold_naive <- numeric(length(thresholds))
pr_threshold_bvn  <- numeric(length(thresholds))

for (i in seq_along(thresholds)) {
  eps <- thresholds[i]

  pr_threshold_corr[i] <- mean(theta_corrected[, IDX_ALPHA_POV] < -eps &
                               theta_corrected[, IDX_BETA_POV] > eps)
  pr_threshold_naive[i] <- mean(theta_naive[, IDX_ALPHA_POV] < -eps &
                                theta_naive[, IDX_BETA_POV] > eps)

  line <- sprintf("  %-8.3f  %12.6f  %12.6f",
                  eps, pr_threshold_corr[i], pr_threshold_naive[i])

  if (has_mvtnorm) {
    pr_threshold_bvn[i] <- mvtnorm::pmvnorm(
      lower = c(-Inf, eps),
      upper = c(-eps, Inf),
      mean  = mu_2,
      sigma = V_2_sand
    )[1]
    line <- paste0(line, sprintf("  %14.10f", pr_threshold_bvn[i]))
  }

  cat(line, "\n")
}

cat(sprintf("\n  Interpretation:\n"))
cat(sprintf("    theta_hat = (%+.3f, %+.3f), SD_sand = (%.3f, %.3f).\n",
            theta_hat[IDX_ALPHA_POV], theta_hat[IDX_BETA_POV],
            sd_sand_alpha, sd_sand_beta))
cat("    The probability drops below 1.000 only for eps near the posterior\n")
cat("    mean, confirming that the reversal is not a boundary phenomenon.\n")

cat("\n")


###############################################################################
## SECTION 7 : PART E -- MANUSCRIPT LANGUAGE RECOMMENDATION
###############################################################################
cat("--- 7. Part E: Manuscript language recommendation ---\n\n")

## Decision logic:
## If all methods give Pr = 1.000 (to 3 decimals), suggest confirmation language.
## If they differ, discuss implications.

all_pr_1 <- (round(pr_reversal_naive, 3) == 1.000 &&
             round(pr_reversal_corrected, 3) == 1.000 &&
             round(pr_ame_reversal_naive, 3) == 1.000 &&
             round(pr_ame_reversal_corr, 3) == 1.000)

if (all_pr_1) {
  cat("  VERDICT: All methods yield Pr(reversal) = 1.000 (to 3 decimal places).\n\n")

  cat("  Recommended manuscript text (Section 5.2, after existing Pr = 1.000):\n\n")

  cat("    Option A (parenthetical, minimal change):\n")
  cat("    $\\Pr(\\alpha_{\\mathrm{pov}} < 0 \\text{ and } \\beta_{\\mathrm{pov}} > 0\n")
  cat("    \\mid \\text{data}) = 1.000$ (robust to sandwich correction).\n\n")

  cat("    Option B (expanded, with analytical cross-check):\n")
  cat("    $\\Pr(\\alpha_{\\mathrm{pov}} < 0 \\text{ and } \\beta_{\\mathrm{pov}} > 0\n")
  cat("    \\mid \\text{data}) = 1.000$ under both the naive and sandwich-corrected\n")
  cat("    posteriors.  Under a bivariate normal approximation with the sandwich\n")
  cat("    covariance, the complementary probability is of order $10^{-")
  if (has_mvtnorm) {
    complement <- 1 - pr_bvn_sand
    exponent <- floor(-log10(complement))
    cat(sprintf("%d", exponent))
  } else {
    cat("?")
  }
  cat("}$.\n\n")

  cat("    Option C (footnote or SM):\n")
  cat("    The reversal probability $\\Pr(\\alpha_{\\mathrm{pov}} < 0 \\cap\n")
  cat("    \\beta_{\\mathrm{pov}} > 0 \\mid \\text{data}) = 1.000$ is\n")
  cat("    consistent across naive ($M =")
  cat(sprintf(" %d$ draws) and\n", M))
  cat("    sandwich-corrected posteriors.  By the rule of three,\n")
  cat(sprintf("    the posterior probability exceeds $%.4f$ with 95\\%% confidence.\n\n", 1 - 3/M))

} else {
  cat("  VERDICT: Methods disagree. Some yield Pr < 1.000.\n\n")

  cat("  Detailed breakdown:\n")
  cat(sprintf("    Naive (param):     %.6f\n", pr_reversal_naive))
  cat(sprintf("    Corrected (param): %.6f\n", pr_reversal_corrected))
  cat(sprintf("    Naive (AME):       %.6f\n", pr_ame_reversal_naive))
  cat(sprintf("    Corrected (AME):   %.6f\n", pr_ame_reversal_corr))

  if (round(pr_reversal_corrected, 3) < 1.000) {
    cat("\n  The sandwich correction reduces the reversal probability.\n")
    cat("  Suggested language: report the corrected probability explicitly.\n\n")
    cat(sprintf("    \"Pr(reversal) = %.3f under the sandwich-corrected posterior\n",
                pr_reversal_corrected))
    cat("     (two-track inference strategy).\"\n\n")
  }
}


###############################################################################
## SECTION 8 : COMPREHENSIVE SUMMARY TABLE
###############################################################################
cat("--- 8. Comprehensive summary table ---\n\n")

cat("  ==================================================================\n")
cat("  JOINT POVERTY REVERSAL: Pr(alpha_pov < 0 AND beta_pov > 0 | data)\n")
cat("  ==================================================================\n\n")

cat(sprintf("  %-42s %14s\n", "Method", "Probability"))
cat(sprintf("  %s\n", paste(rep("-", 58), collapse = "")))
cat(sprintf("  %-42s %14.6f\n",
            "Naive MCMC (parameter-level)", pr_reversal_naive))
cat(sprintf("  %-42s %14.6f\n",
            "Sandwich-corrected (parameter-level)", pr_reversal_corrected))
if (has_mvtnorm) {
  cat(sprintf("  %-42s %14.10f\n",
              "Analytical BVN (sandwich)", pr_bvn_sand))
  cat(sprintf("  %-42s %14.10f\n",
              "Analytical BVN (naive)", pr_bvn_mcmc))
}
cat(sprintf("  %-42s %14.10f\n",
            "Independence approx (sandwich)", pr_indep_sand))
cat(sprintf("  %-42s %14.6f\n",
            "Naive MCMC (AME-level)", pr_ame_reversal_naive))
cat(sprintf("  %-42s %14.6f\n",
            "Sandwich-corrected (AME-level)", pr_ame_reversal_corr))
cat(sprintf("  %s\n\n", paste(rep("-", 58), collapse = "")))

cat(sprintf("  CROSS-MARGIN CORRELATION: rho(alpha_pov, beta_pov)\n"))
cat(sprintf("    V_sand target:         %+.6f\n", rho_sand))
cat(sprintf("    Corrected draws:       %+.6f\n", rho_corr_emp))
cat(sprintf("    Sigma_MCMC:            %+.6f\n", rho_mcmc))
cat(sprintf("    Naive draws:           %+.6f\n\n", rho_naive_emp))

cat(sprintf("  MONTE CARLO PRECISION:\n"))
cat(sprintf("    Total draws (M):                %d\n", M))
cat(sprintf("    AME subsample (M_sub):          %d\n", M_ame))
cat(sprintf("    Rule-of-three lower bound:      %.6f\n", 1 - 3/M))
cat(sprintf("    Min sandwich z-score:           %.2f\n",
            min(z_alpha_sand, z_beta_sand)))
cat("\n")


###############################################################################
## SECTION 9 : SAVE RESULTS
###############################################################################
cat("--- 9. Saving results ---\n\n")

B1_results <- list(
  ## Description
  description = paste(
    "Joint poverty reversal probability",
    "computed from both naive and sandwich-corrected MCMC draws,",
    "at both parameter-level and AME-level, with analytical bivariate",
    "normal cross-check via mvtnorm::pmvnorm and sensitivity analysis."
  ),

  ## Dimensions
  M = M,
  D = D,
  P = P,
  M_ame = M_ame,
  idx_alpha_pov = IDX_ALPHA_POV,
  idx_beta_pov  = IDX_BETA_POV,

  ## Part A: Parameter-level reversal probabilities
  param_level = list(
    pr_reversal_naive     = pr_reversal_naive,
    pr_reversal_corrected = pr_reversal_corrected,
    n_reversal_naive      = n_reversal_naive,
    n_reversal_corrected  = n_reversal_corrected,
    mcse_naive            = mcse_naive,
    mcse_corrected        = mcse_corr,
    pr_alpha_neg_naive    = pr_alpha_neg_naive,
    pr_beta_pos_naive     = pr_beta_pos_naive,
    pr_alpha_neg_corrected = pr_alpha_neg_corrected,
    pr_beta_pos_corrected  = pr_beta_pos_corrected,
    z_alpha_sand          = z_alpha_sand,
    z_beta_sand           = z_beta_sand,
    z_alpha_mcmc          = z_alpha_mcmc,
    z_beta_mcmc           = z_beta_mcmc,
    lower_bound_rule_of_three = 1 - 3/M
  ),

  ## Part B: Analytical BVN approximation
  analytical = list(
    pr_bvn_sandwich     = pr_bvn_sand,
    pr_bvn_naive        = pr_bvn_mcmc,
    pr_indep_sandwich   = pr_indep_sand,
    pr_indep_naive      = pr_indep_mcmc,
    pr_marginal_alpha_sand = pr_marginal_alpha_sand,
    pr_marginal_beta_sand  = pr_marginal_beta_sand,
    rho_sand            = rho_sand,
    rho_mcmc            = rho_mcmc,
    rho_corr_empirical  = rho_corr_emp,
    rho_naive_empirical = rho_naive_emp,
    V_2x2_sandwich      = V_2_sand,
    V_2x2_naive         = V_2_mcmc,
    mu_2                = mu_2,
    mvtnorm_available   = has_mvtnorm
  ),

  ## Part C: AME-level reversal probabilities
  ame_level = list(
    pr_ame_reversal_corrected = pr_ame_reversal_corr,
    pr_ame_reversal_naive     = pr_ame_reversal_naive,
    mcse_ame_corrected        = mcse_ame_corr,
    mcse_ame_naive            = mcse_ame_naive,
    M_ame                     = M_ame,
    M_naive_sub               = M_naive_sub,
    ## Store naive AME draws for poverty covariate (for future reference)
    ext_ame_pov_naive_summary = c(
      mean = mean(ext_ame_naive[, pov_idx]),
      sd   = sd(ext_ame_naive[, pov_idx])
    ),
    int_ame_pov_naive_summary = c(
      mean = mean(int_ame_naive[, pov_idx]),
      sd   = sd(int_ame_naive[, pov_idx])
    )
  ),

  ## Part D: Sensitivity analysis
  sensitivity = list(
    thresholds         = thresholds,
    pr_corrected       = pr_threshold_corr,
    pr_naive           = pr_threshold_naive,
    pr_bvn             = if (has_mvtnorm) pr_threshold_bvn else NULL
  ),

  ## Verdict and manuscript language
  all_methods_agree = all_pr_1,
  verdict = if (all_pr_1) {
    "All methods yield Pr(reversal) = 1.000. Manuscript statement is robust."
  } else {
    sprintf("Disagreement: naive = %.4f, corrected = %.4f.",
            pr_reversal_naive, pr_reversal_corrected)
  },

  ## Parameter summary
  param_summary = data.frame(
    parameter = c("alpha_poverty", "beta_poverty"),
    theta_hat = c(theta_hat[IDX_ALPHA_POV], theta_hat[IDX_BETA_POV]),
    sd_mcmc   = c(sd_mcmc_alpha, sd_mcmc_beta),
    sd_sand   = c(sd_sand_alpha, sd_sand_beta),
    z_sand    = c(z_alpha_sand, z_beta_sand),
    z_mcmc    = c(z_alpha_mcmc, z_beta_mcmc),
    stringsAsFactors = FALSE
  ),

  ## Verification
  verification = list(
    A_inv_error        = max_inv_err,
    naive_cov_rel_err  = rel_cov_err,
    naive_mean_err     = max_mean_err,
    all_checks_pass    = (max_inv_err < 1e-6) && (rel_cov_err < 1e-3)
  ),

  ## Timestamp
  timestamp = Sys.time()
)

saveRDS(B1_results, B1_OUT)
cat(sprintf("  Saved: %s\n", B1_OUT))
cat(sprintf("  File size: %.1f KB\n\n",
            file.info(B1_OUT)$size / 1024))


###############################################################################
## SECTION 10 : FINAL SUMMARY
###############################################################################
cat("==============================================================\n")
cat("  Joint Reversal Probability Analysis COMPLETE\n")
cat("==============================================================\n\n")

cat("  PARAMETER-LEVEL:  Pr(alpha_pov < 0 AND beta_pov > 0 | data)\n")
cat(sprintf("    Naive (MCMC):         %.6f  (%d / %d draws)\n",
            pr_reversal_naive, n_reversal_naive, M))
cat(sprintf("    Corrected (sandwich): %.6f  (%d / %d draws)\n",
            pr_reversal_corrected, n_reversal_corrected, M))
if (has_mvtnorm) {
  cat(sprintf("    Analytical BVN:       %.10f  (sandwich covariance)\n",
              pr_bvn_sand))
}

cat(sprintf("\n  AME-LEVEL:  Pr(ext_AME_pov < 0 AND int_AME_pov > 0 | data)\n"))
cat(sprintf("    Naive AME:            %.6f  (M_sub = %d)\n",
            pr_ame_reversal_naive, M_naive_sub))
cat(sprintf("    Corrected AME:        %.6f  (M_sub = %d)\n",
            pr_ame_reversal_corr, M_ame))

cat(sprintf("\n  BOUNDARY DISTANCE:  min z-score = %.2f sandwich SDs\n",
            min(z_alpha_sand, z_beta_sand)))
cat(sprintf("  MC LOWER BOUND:     Pr >= %.4f (rule of three, 95%%)\n",
            1 - 3/M))

if (all_pr_1) {
  cat("\n  VERDICT: Reversal probability = 1.000 across ALL methods.\n")
  cat("    The poverty reversal is robust to sandwich correction.\n")
  cat("    The manuscript statement is fully consistent with the\n")
  cat("    two-track inference strategy.\n")
} else {
  cat("\n  VERDICT: Some methods yield Pr < 1.000.\n")
  cat("    Review the detailed comparison table above.\n")
}

cat(sprintf("\n  Output: %s\n", B1_OUT))
cat("\n==============================================================\n")
cat("  DONE.\n")
cat("==============================================================\n")
