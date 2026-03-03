## =============================================================================
## 05_marginal_effects.R -- Average Marginal Effects Decomposition
## =============================================================================
## Purpose : Compute Average Marginal Effects (AME) decomposed into
##           extensive and intensive margin contributions.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Theory:
##   The Hurdle Beta-Binomial model implies:
##     E[IT_share_i] = P(z_i = 1 | X_i) * E[IT_share_i | z_i = 1, X_i]
##                   = q_i             *  mu_i
##
##   By the product rule, the marginal effect of covariate k is:
##     dE[y/n] / dx_k = alpha_k * q*(1-q) * mu  +  q * beta_k * mu*(1-mu)
##                       [extensive component]      [intensive component]
##
## Design Choice -- Population-Average AME (PA-AME):
##   Uses GLOBAL (population-level) coefficients alpha and beta only,
##   WITHOUT state random effects delta[s]. This ensures consistency
##   with the sandwich-corrected Wald intervals.
##
## Inputs:
##   data/precomputed/cholesky_correction.rds  -- theta_corrected (M x 11)
##   data/precomputed/stan_data.rds            -- X, z, n_trial, y, N, P
##   data/precomputed/sandwich_variance.rds    -- V_sand, theta_hat
##
## Outputs:
##   data/precomputed/marginal_effects.rds     -- AME draws, summary, decomp
##
## Usage:
##   source("code/05_marginal_effects.R")
## =============================================================================

cat("==============================================================\n")
cat("  Marginal Effects Decomposition\n")
cat("  AME = Extensive + Intensive Components\n")
cat("==============================================================\n\n")


###############################################################################
## SECTION 0 : SETUP
###############################################################################

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()

## Source helper functions (provides inv_logit, logit, etc.)
source(file.path(PROJECT_ROOT, "code/helpers/utils.R"))

## Load required packages
library(dplyr, warn.conflicts = FALSE)

## Paths
OUTPUT_DIR      <- file.path(PROJECT_ROOT, "data/precomputed")
CHOLESKY_PATH   <- file.path(OUTPUT_DIR, "cholesky_correction.rds")
STAN_DATA_PATH  <- file.path(OUTPUT_DIR, "stan_data.rds")
SANDWICH_PATH   <- file.path(OUTPUT_DIR, "sandwich_variance.rds")
ME_OUT          <- file.path(OUTPUT_DIR, "marginal_effects.rds")

## Subsample size for efficiency
M_SUB <- 1000    # use every 4th draw from M=4000

## Covariate labels (P=5)
COV_LABELS <- c("intercept", "poverty", "urban", "black", "hispanic")

## Parameter dimension
D <- 11  # alpha[1:5], beta[1:5], log_kappa
P <- 5


###############################################################################
## SECTION 1 : LOAD INPUTS
###############################################################################

cat("--- Section 1: Loading inputs ---\n\n")

## 1a. Cholesky-corrected MCMC draws
if (!file.exists(CHOLESKY_PATH)) {
  stop("Cholesky correction file not found: ", CHOLESKY_PATH,
       "\n  Run 03_survey_weighting.R first.")
}
chol_res <- readRDS(CHOLESKY_PATH)
theta_corrected <- chol_res$theta_corrected   # M x 11 matrix
theta_hat       <- chol_res$theta_hat          # length-11 vector
M_total         <- nrow(theta_corrected)

cat(sprintf("  Corrected MCMC draws: %d draws x %d parameters\n",
            M_total, ncol(theta_corrected)))
cat(sprintf("  theta_hat: %s\n", paste(sprintf("%+.4f", theta_hat), collapse = ", ")))

## 1b. Stan data (design matrix and outcomes)
if (!file.exists(STAN_DATA_PATH)) {
  stop("Stan data file not found: ", STAN_DATA_PATH,
       "\n  Run 01_data_preparation.R first.")
}
stan_data <- readRDS(STAN_DATA_PATH)
X       <- stan_data$X          # N x P matrix
N       <- stan_data$N          # number of observations
z       <- stan_data$z          # binary indicator (serve IT)
n_trial <- stan_data$n_trial    # total enrollment
y       <- stan_data$y          # IT enrollment (for those serving)

cat(sprintf("  Observations: N = %d, P = %d covariates\n", N, P))
cat(sprintf("  Serving IT: %d (%.1f%%),  Not serving: %d (%.1f%%)\n",
            sum(z), 100 * mean(z), sum(1 - z), 100 * mean(1 - z)))

## Verify dimensions
stopifnot(ncol(X) == P)
stopifnot(nrow(X) == N)
stopifnot(ncol(theta_corrected) == D)

## 1c. Sandwich variance (for Wald comparison)
if (!file.exists(SANDWICH_PATH)) {
  stop("Sandwich variance file not found: ", SANDWICH_PATH,
       "\n  Run 03_survey_weighting.R first.")
}
sandwich_res <- readRDS(SANDWICH_PATH)
V_sand <- sandwich_res$V_sand   # 11 x 11 matrix

cat(sprintf("  Sandwich variance: %d x %d\n", nrow(V_sand), ncol(V_sand)))

## 1d. Subsample MCMC draws (every 4th draw for efficiency)
thin_idx <- seq(1, M_total, by = max(1, floor(M_total / M_SUB)))
if (length(thin_idx) > M_SUB) thin_idx <- thin_idx[1:M_SUB]
theta_sub <- theta_corrected[thin_idx, , drop = FALSE]
M_use <- nrow(theta_sub)

cat(sprintf("  Subsampled: %d draws (from %d, every %d-th)\n\n",
            M_use, M_total, max(1, floor(M_total / M_SUB))))


###############################################################################
## SECTION 2 : COMPUTE AME FOR EACH MCMC DRAW
###############################################################################

cat("--- Section 2: Computing AME for each MCMC draw ---\n\n")

ext_ame   <- matrix(NA_real_, nrow = M_use, ncol = P)
int_ame   <- matrix(NA_real_, nrow = M_use, ncol = P)
total_ame <- matrix(NA_real_, nrow = M_use, ncol = P)
mean_q    <- numeric(M_use)
mean_mu   <- numeric(M_use)

colnames(ext_ame)   <- COV_LABELS
colnames(int_ame)   <- COV_LABELS
colnames(total_ame) <- COV_LABELS

t_start <- proc.time()

for (m in seq_len(M_use)) {

  ## Extract alpha and beta for this draw
  alpha_m <- theta_sub[m, 1:P]
  beta_m  <- theta_sub[m, (P + 1):(2 * P)]

  ## Compute linear predictors for ALL observations (vectorized)
  eta_ext <- as.numeric(X %*% alpha_m)
  eta_int <- as.numeric(X %*% beta_m)

  ## Apply inverse logit to get probabilities
  q_m  <- inv_logit(eta_ext)
  mu_m <- inv_logit(eta_int)

  ## Store diagnostics
  mean_q[m]  <- mean(q_m)
  mean_mu[m] <- mean(mu_m)

  ## Pre-compute the "kernel" terms once
  q_deriv  <- q_m * (1 - q_m)
  mu_deriv <- mu_m * (1 - mu_m)

  for (k in seq_len(P)) {
    ext_component <- alpha_m[k] * q_deriv * mu_m
    int_component <- beta_m[k] * mu_deriv * q_m

    ext_ame[m, k]   <- mean(ext_component)
    int_ame[m, k]   <- mean(int_component)
    total_ame[m, k] <- ext_ame[m, k] + int_ame[m, k]
  }
}

t_elapsed <- (proc.time() - t_start)[3]
cat(sprintf("  AME computation: %.1f seconds for %d draws x %d obs x %d covariates\n",
            t_elapsed, M_use, N, P))
cat(sprintf("  Mean P(serve IT): %.4f (posterior mean over draws)\n", mean(mean_q)))
cat(sprintf("  Mean E[IT share | serve]: %.4f (posterior mean over draws)\n\n",
            mean(mean_mu)))


###############################################################################
## SECTION 3 : SUMMARIZE AME (POSTERIOR MEAN, 95% CI)
###############################################################################

cat("--- Section 3: Posterior summary of AME ---\n\n")

## Helper function for summarizing one AME matrix
summarize_ame <- function(ame_mat, labels) {
  data.frame(
    covariate     = labels,
    post_mean     = colMeans(ame_mat),
    post_median   = apply(ame_mat, 2, median),
    ci_lo         = apply(ame_mat, 2, quantile, probs = 0.025),
    ci_hi         = apply(ame_mat, 2, quantile, probs = 0.975),
    post_sd       = apply(ame_mat, 2, sd),
    pr_positive   = colMeans(ame_mat > 0),
    stringsAsFactors = FALSE,
    row.names     = NULL
  )
}

ext_summary   <- summarize_ame(ext_ame,   COV_LABELS)
int_summary   <- summarize_ame(int_ame,   COV_LABELS)
total_summary <- summarize_ame(total_ame, COV_LABELS)

## Print formatted summary tables
cat("  EXTENSIVE MARGIN AME:  dP(z=1)/dx_k * E[mu]\n")
cat("  ----------------------------------------------------------------\n")
cat(sprintf("  %-12s  %9s  %9s   [%9s, %9s]  Pr(>0)\n",
            "Covariate", "Mean", "Median", "2.5%", "97.5%"))
cat("  ----------------------------------------------------------------\n")
for (i in seq_len(P)) {
  cat(sprintf("  %-12s  %+9.6f  %+9.6f   [%+9.6f, %+9.6f]  %.3f\n",
              ext_summary$covariate[i],
              ext_summary$post_mean[i],
              ext_summary$post_median[i],
              ext_summary$ci_lo[i],
              ext_summary$ci_hi[i],
              ext_summary$pr_positive[i]))
}

cat("\n  INTENSIVE MARGIN AME:  P(z=1) * dmu/dx_k\n")
cat("  ----------------------------------------------------------------\n")
cat(sprintf("  %-12s  %9s  %9s   [%9s, %9s]  Pr(>0)\n",
            "Covariate", "Mean", "Median", "2.5%", "97.5%"))
cat("  ----------------------------------------------------------------\n")
for (i in seq_len(P)) {
  cat(sprintf("  %-12s  %+9.6f  %+9.6f   [%+9.6f, %+9.6f]  %.3f\n",
              int_summary$covariate[i],
              int_summary$post_mean[i],
              int_summary$post_median[i],
              int_summary$ci_lo[i],
              int_summary$ci_hi[i],
              int_summary$pr_positive[i]))
}

cat("\n  TOTAL AME:  d E[y/n] / dx_k\n")
cat("  ----------------------------------------------------------------\n")
cat(sprintf("  %-12s  %9s  %9s   [%9s, %9s]  Pr(>0)\n",
            "Covariate", "Mean", "Median", "2.5%", "97.5%"))
cat("  ----------------------------------------------------------------\n")
for (i in seq_len(P)) {
  cat(sprintf("  %-12s  %+9.6f  %+9.6f   [%+9.6f, %+9.6f]  %.3f\n",
              total_summary$covariate[i],
              total_summary$post_mean[i],
              total_summary$post_median[i],
              total_summary$ci_lo[i],
              total_summary$ci_hi[i],
              total_summary$pr_positive[i]))
}
cat("\n")


###############################################################################
## SECTION 4 : DECOMPOSITION TABLE
###############################################################################

cat("--- Section 4: Decomposition table ---\n\n")

decomp_idx <- 2:P   # poverty, urban, black, hispanic

decomp_table <- data.frame(
  covariate    = COV_LABELS[decomp_idx],
  ext_ame      = ext_summary$post_mean[decomp_idx],
  ext_ci_lo    = ext_summary$ci_lo[decomp_idx],
  ext_ci_hi    = ext_summary$ci_hi[decomp_idx],
  int_ame      = int_summary$post_mean[decomp_idx],
  int_ci_lo    = int_summary$ci_lo[decomp_idx],
  int_ci_hi    = int_summary$ci_hi[decomp_idx],
  total_ame    = total_summary$post_mean[decomp_idx],
  total_ci_lo  = total_summary$ci_lo[decomp_idx],
  total_ci_hi  = total_summary$ci_hi[decomp_idx],
  stringsAsFactors = FALSE,
  row.names    = NULL
)

## Compute share of total from each margin
decomp_table$ext_share <- with(decomp_table,
  abs(ext_ame) / (abs(ext_ame) + abs(int_ame)) * 100
)
decomp_table$int_share <- with(decomp_table,
  abs(int_ame) / (abs(ext_ame) + abs(int_ame)) * 100
)

## Determine sign agreement
decomp_table$sign_pattern <- with(decomp_table,
  ifelse(sign(ext_ame) == sign(int_ame), "reinforcing", "opposing")
)

## Print formatted decomposition table
cat("  MARGINAL EFFECTS DECOMPOSITION\n")
cat("  ==========================================================================\n")
cat(sprintf("  %-10s  %10s  %10s  %10s  %6s  %6s  %-12s\n",
            "Covariate", "Ext_AME", "Int_AME", "Total_AME",
            "Ext%", "Int%", "Pattern"))
cat("  --------------------------------------------------------------------------\n")
for (i in seq_len(nrow(decomp_table))) {
  r <- decomp_table[i, ]
  cat(sprintf("  %-10s  %+10.6f  %+10.6f  %+10.6f  %5.1f%%  %5.1f%%  %-12s\n",
              r$covariate, r$ext_ame, r$int_ame, r$total_ame,
              r$ext_share, r$int_share, r$sign_pattern))
}
cat("  ==========================================================================\n")

## Highlight poverty reversal
pov_row <- decomp_table[decomp_table$covariate == "poverty", ]
cat(sprintf("\n  POVERTY REVERSAL:\n"))
cat(sprintf("    Extensive: %+.6f  (higher poverty => LESS likely to serve IT)\n",
            pov_row$ext_ame))
cat(sprintf("    Intensive: %+.6f  (higher poverty => HIGHER IT share | serve)\n",
            pov_row$int_ame))
cat(sprintf("    Total:     %+.6f  (net effect on E[IT share])\n",
            pov_row$total_ame))

## Posterior probability of reversal
pr_reversal <- mean(ext_ame[, "poverty"] < 0 & int_ame[, "poverty"] > 0)
cat(sprintf("    Pr(reversal): %.4f  (posterior prob of ext<0 AND int>0)\n\n",
            pr_reversal))


###############################################################################
## SECTION 5 : WALD-BASED AME (DELTA METHOD) vs POSTERIOR AME
###############################################################################

cat("--- Section 5: Wald-based AME comparison ---\n\n")

alpha_hat <- theta_hat[1:P]
beta_hat  <- theta_hat[(P + 1):(2 * P)]

eta_ext_hat <- as.numeric(X %*% alpha_hat)
eta_int_hat <- as.numeric(X %*% beta_hat)
q_hat       <- inv_logit(eta_ext_hat)
mu_hat      <- inv_logit(eta_int_hat)

q_deriv_hat  <- q_hat * (1 - q_hat)
mu_deriv_hat <- mu_hat * (1 - mu_hat)

## Point-estimate AME at theta_hat
ame_ext_hat   <- numeric(P)
ame_int_hat   <- numeric(P)
ame_total_hat <- numeric(P)

for (k in seq_len(P)) {
  ame_ext_hat[k]   <- mean(alpha_hat[k] * q_deriv_hat * mu_hat)
  ame_int_hat[k]   <- mean(beta_hat[k] * mu_deriv_hat * q_hat)
  ame_total_hat[k] <- ame_ext_hat[k] + ame_int_hat[k]
}

names(ame_ext_hat)   <- COV_LABELS
names(ame_int_hat)   <- COV_LABELS
names(ame_total_hat) <- COV_LABELS

## Numerical gradient of AME w.r.t. theta (Delta method SE)
eps <- 1e-5

grad_ame_total <- matrix(0, nrow = P, ncol = D)
grad_ame_ext   <- matrix(0, nrow = P, ncol = D)
grad_ame_int   <- matrix(0, nrow = P, ncol = D)

ame_at_theta <- function(theta_vec) {
  a <- theta_vec[1:P]
  b <- theta_vec[(P + 1):(2 * P)]

  eta_e <- as.numeric(X %*% a)
  eta_i <- as.numeric(X %*% b)
  q_v   <- inv_logit(eta_e)
  mu_v  <- inv_logit(eta_i)
  qd_v  <- q_v * (1 - q_v)
  md_v  <- mu_v * (1 - mu_v)

  ext_v   <- numeric(P)
  int_v   <- numeric(P)
  total_v <- numeric(P)
  for (kk in seq_len(P)) {
    ext_v[kk]   <- mean(a[kk] * qd_v * mu_v)
    int_v[kk]   <- mean(b[kk] * md_v * q_v)
    total_v[kk] <- ext_v[kk] + int_v[kk]
  }
  list(ext = ext_v, int = int_v, total = total_v)
}

for (d in seq_len(D)) {
  theta_plus  <- theta_hat;  theta_plus[d]  <- theta_plus[d]  + eps
  theta_minus <- theta_hat;  theta_minus[d] <- theta_minus[d] - eps

  ame_plus  <- ame_at_theta(theta_plus)
  ame_minus <- ame_at_theta(theta_minus)

  grad_ame_total[, d] <- (ame_plus$total - ame_minus$total) / (2 * eps)
  grad_ame_ext[, d]   <- (ame_plus$ext   - ame_minus$ext)   / (2 * eps)
  grad_ame_int[, d]   <- (ame_plus$int   - ame_minus$int)   / (2 * eps)
}

## Delta method variance: Var(AME_k) = grad_k' V_sand grad_k
ame_var_total <- numeric(P)
ame_var_ext   <- numeric(P)
ame_var_int   <- numeric(P)

for (k in seq_len(P)) {
  ame_var_total[k] <- as.numeric(grad_ame_total[k, ] %*% V_sand %*% grad_ame_total[k, ])
  ame_var_ext[k]   <- as.numeric(grad_ame_ext[k, ]   %*% V_sand %*% grad_ame_ext[k, ])
  ame_var_int[k]   <- as.numeric(grad_ame_int[k, ]   %*% V_sand %*% grad_ame_int[k, ])
}

ame_se_total <- sqrt(pmax(ame_var_total, 0))
ame_se_ext   <- sqrt(pmax(ame_var_ext, 0))
ame_se_int   <- sqrt(pmax(ame_var_int, 0))

## Wald 95% CI
wald_total_lo <- ame_total_hat - 1.96 * ame_se_total
wald_total_hi <- ame_total_hat + 1.96 * ame_se_total
wald_ext_lo   <- ame_ext_hat - 1.96 * ame_se_ext
wald_ext_hi   <- ame_ext_hat + 1.96 * ame_se_ext
wald_int_lo   <- ame_int_hat - 1.96 * ame_se_int
wald_int_hi   <- ame_int_hat + 1.96 * ame_se_int

## Build Wald-based summary table
wald_summary <- data.frame(
  covariate    = COV_LABELS,
  ext_point    = ame_ext_hat,
  ext_se       = ame_se_ext,
  ext_lo       = wald_ext_lo,
  ext_hi       = wald_ext_hi,
  int_point    = ame_int_hat,
  int_se       = ame_se_int,
  int_lo       = wald_int_lo,
  int_hi       = wald_int_hi,
  total_point  = ame_total_hat,
  total_se     = ame_se_total,
  total_lo     = wald_total_lo,
  total_hi     = wald_total_hi,
  stringsAsFactors = FALSE,
  row.names    = NULL
)

## Print comparison: posterior vs Wald
cat("  COMPARISON: Posterior AME vs Wald (Delta-method) AME\n")
cat("  ==========================================================================\n")
cat(sprintf("  %-10s  %-23s  %-23s  %7s\n",
            "Covariate", "Posterior [95% CI]", "Wald [95% CI]", "Agree?"))
cat("  --------------------------------------------------------------------------\n")
for (i in decomp_idx) {
  post_str <- sprintf("%+.6f [%+.5f, %+.5f]",
                      total_summary$post_mean[i],
                      total_summary$ci_lo[i],
                      total_summary$ci_hi[i])
  wald_str <- sprintf("%+.6f [%+.5f, %+.5f]",
                      ame_total_hat[i],
                      wald_total_lo[i],
                      wald_total_hi[i])

  post_sig <- (total_summary$ci_lo[i] > 0 & total_summary$ci_hi[i] > 0) |
              (total_summary$ci_lo[i] < 0 & total_summary$ci_hi[i] < 0)
  wald_sig <- (wald_total_lo[i] > 0 & wald_total_hi[i] > 0) |
              (wald_total_lo[i] < 0 & wald_total_hi[i] < 0)
  agree <- ifelse(post_sig == wald_sig && sign(ame_total_hat[i]) == sign(total_summary$post_mean[i]),
                  "Yes", "Differs")

  cat(sprintf("  %-10s  %-23s  %-23s  %7s\n",
              COV_LABELS[i], post_str, wald_str, agree))
}
cat("  ==========================================================================\n\n")


###############################################################################
## SECTION 6 : SAVE RESULTS
###############################################################################

cat("--- Section 6: Saving results ---\n\n")

results <- list(
  description = paste(
    "Average Marginal Effects (AME) decomposed into extensive and intensive",
    "margin contributions. Computed from Cholesky-corrected MCMC draws of",
    "the Hurdle Beta-Binomial model (M3b weighted, survey-corrected)."
  ),
  N        = N,
  P        = P,
  D        = D,
  M_total  = M_total,
  M_sub    = M_use,
  cov_labels  = COV_LABELS,
  decomp_idx  = decomp_idx,
  ext_ame_draws   = ext_ame,
  int_ame_draws   = int_ame,
  total_ame_draws = total_ame,
  mean_q  = mean_q,
  mean_mu = mean_mu,
  ext_summary   = ext_summary,
  int_summary   = int_summary,
  total_summary = total_summary,
  decomp_table = decomp_table,
  pr_poverty_reversal = pr_reversal,
  wald_summary    = wald_summary,
  grad_ame_total  = grad_ame_total,
  grad_ame_ext    = grad_ame_ext,
  grad_ame_int    = grad_ame_int,
  ame_ext_hat   = ame_ext_hat,
  ame_int_hat   = ame_int_hat,
  ame_total_hat = ame_total_hat,
  timestamp = Sys.time()
)

saveRDS(results, ME_OUT)
cat(sprintf("  Saved: %s\n", ME_OUT))
cat(sprintf("  File size: %.1f KB\n\n", file.size(ME_OUT) / 1024))


###############################################################################
## SECTION 7 : SUMMARY
###############################################################################

cat("==============================================================\n")
cat("  MARGINAL EFFECTS DECOMPOSITION -- SUMMARY\n")
cat("==============================================================\n\n")

cat(sprintf("  Model: Hierarchical Hurdle Beta-Binomial (M3b weighted)\n"))
cat(sprintf("  Observations: N = %d (%.1f%% serving IT)\n",
            N, 100 * mean(z)))
cat(sprintf("  MCMC draws used: %d (thinned from %d)\n\n", M_use, M_total))

cat(sprintf("  Population-level predictions (posterior mean):\n"))
cat(sprintf("    Mean P(serve IT):          %.4f\n", mean(mean_q)))
cat(sprintf("    Mean E[IT share | serve]:  %.4f\n", mean(mean_mu)))
cat(sprintf("    Mean E[IT share]:          %.4f\n\n",
            mean(mean_q) * mean(mean_mu)))

cat("  KEY DECOMPOSITION (non-intercept covariates):\n")
cat("  --------------------------------------------------------------------------\n")
cat(sprintf("  %-10s  %10s  %10s  %10s  %6s  %6s  %-12s\n",
            "Covariate", "Ext_AME", "Int_AME", "Total_AME",
            "Ext%", "Int%", "Pattern"))
cat("  --------------------------------------------------------------------------\n")
for (i in seq_len(nrow(decomp_table))) {
  r <- decomp_table[i, ]
  cat(sprintf("  %-10s  %+10.6f  %+10.6f  %+10.6f  %5.1f%%  %5.1f%%  %-12s\n",
              r$covariate, r$ext_ame, r$int_ame, r$total_ame,
              r$ext_share, r$int_share, r$sign_pattern))
}
cat("  --------------------------------------------------------------------------\n")

cat(sprintf("\n  POVERTY REVERSAL:\n"))
cat(sprintf("    Extensive margin:  %+.6f  (higher poverty => less likely to serve)\n",
            pov_row$ext_ame))
cat(sprintf("    Intensive margin:  %+.6f  (higher poverty => higher IT share)\n",
            pov_row$int_ame))
cat(sprintf("    Net total effect:  %+.6f\n", pov_row$total_ame))
cat(sprintf("    Pr(reversal):      %.4f\n", pr_reversal))

cat(sprintf("\n  Output file: %s\n", ME_OUT))

cat("\n==============================================================\n")
cat("  MARGINAL EFFECTS DECOMPOSITION COMPLETE.\n")
cat("==============================================================\n")
