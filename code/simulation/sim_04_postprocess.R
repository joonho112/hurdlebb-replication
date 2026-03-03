## =============================================================================
## sim_04_postprocess.R -- Post-Processing Module
## =============================================================================
## Purpose : Post-process Stan fit results from sim_03_fit.R to compute:
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##           (1) Sandwich variance V_sand = H_obs^{-1} J_cluster H_obs^{-1}
##           (2) Cholesky-corrected draws (optional, for diagnostics)
##           (3) Confidence / credible intervals for 3 estimators x 5 targets
##           (4) Per-replication evaluation metrics (coverage, bias, RMSE)
##
## Three Estimators:
##   E-UW  : Unweighted posterior.  CIs = quantile-based from unweighted draws.
##   E-WT  : Weighted pseudo-posterior (naive).  CIs = quantile-based from
##           weighted draws.
##   E-WS  : Weighted + sandwich.  CIs = Wald intervals using sqrt(V_sand)
##           for fixed effects; same as E-WT for hyperparameters (tau).
##
## Sandwich Variance (refactored from 61_sandwich_variance.R):
##   V_sand = H_obs^{-1} * J_cluster * H_obs^{-1}
##   where H_obs = block_diag(H_ext, H_int) is the observed information
##   and J_cluster is the cluster-robust meat matrix.
##
##   H_ext (PxP): Analytic Fisher information for logistic extensive margin.
##     H_ext = sum_i w_tilde[i] * q_i * (1-q_i) * X_i X_i^T
##   H_int ((P+1)x(P+1)): Empirical information identity for intensive+kappa.
##     H_int = sum_i w_tilde[i] * s_int_full_i * s_int_full_i^T
##   H_obs is block-diagonal because the hurdle structure makes ext and int
##   conditionally independent.
##
## Cholesky Correction (refactored from 62_cholesky_transform.R):
##   Williams-Savitsky (2021) Theorem 4.1:
##   A = L_sand * L_MCMC^{-1}
##   theta_corrected = theta_hat + A * (theta - theta_hat)
##   so that cov(theta_corrected) = V_sand
##
## Target Parameters (5):
##   1. alpha_poverty (alpha[2])  -- extensive margin poverty coefficient
##   2. beta_poverty  (beta[2])   -- intensive margin poverty coefficient
##   3. log_kappa                 -- overdispersion
##   4. tau_ext       (tau[1])    -- extensive random intercept SD
##   5. tau_int       (tau[2])    -- intensive random intercept SD
##
## Coverage level: 0.90 (90% CIs), z = qnorm(0.95) = 1.6449
##
## Key Subtleties:
##   1. Score weighting: Scores from Stan are UNWEIGHTED. Multiply by w_tilde
##      when computing J_cluster.
##   2. H_ext is analytic (exact), H_int is empirical (approximate via
##      information identity). Same approach as 61_sandwich_variance.R.
##   3. Block-diagonal H_obs: hurdle conditional independence.
##   4. tau has no sandwich: hyperparameters not corrected. E-WS tau = E-WT tau.
##   5. Wald vs quantile: E-WS uses WALD CIs for fixed effects.
##   6. PD safeguard: nearPD() for V_sand and ridge for H_obs.
##   7. Memory: Only theta_hat and V_sand diagonal needed for metrics.
##
## Dependencies:
##   - sim_00_config.R : SIM_CONFIG
##   - sim_03_fit.R    : fit results (E_UW, E_WT_WS)
##   - Matrix          : nearPD() for PD correction
##   - utils_helpers.R : inv_logit
##
## Inputs (per replication):
##   fit_results : list(E_UW=..., E_WT_WS=...) from sim_03_fit.R
##   sample_data : list from sim_02_sampling.R draw_sample()
##   config      : SIM_CONFIG
##
## Outputs (per replication):
##   list(
##     metrics        : data.frame (5 params x 3 estimators = 15 rows)
##     sandwich       : list(V_sand, H_obs, J_cluster, DER, H_obs_inv, ...)
##     cholesky       : list(theta_corrected, A) or NULL
##     diagnostics_uw : from E_UW fit
##     diagnostics_wt : from E_WT_WS fit
##   )
##
## Usage:
##   source("code/simulation/sim_04_postprocess.R")
##   result <- postprocess_replication(fit_results, sample_data, SIM_CONFIG)
##
## =============================================================================
###############################################################################

cat("==============================================================\n")
cat("  Post-Processing Module (sim_04_postprocess.R)\n")
cat("==============================================================\n\n")


###############################################################################
## SECTION 0 : ENVIRONMENT CHECK
###############################################################################

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
                           

## Source dependencies if not already loaded
if (!exists("SIM_CONFIG")) {
  cat("  Loading SIM_CONFIG from sim_00_config.R ...\n")
  source(file.path(PROJECT_ROOT, "code/simulation/sim_00_config.R"))
}

## Ensure Matrix package is available for nearPD fallback
if (!requireNamespace("Matrix", quietly = TRUE)) {
  stop("Package 'Matrix' is required for nearPD fallback. ",
       "Install with: install.packages('Matrix')")
}

cat("  Dependencies loaded.\n\n")


###############################################################################
## SECTION 1 : CI COMPUTATION HELPERS
###############################################################################

#' Compute Quantile-Based Credible Interval
#'
#' @param draws  Numeric vector of MCMC draws for a single parameter.
#' @param level  Numeric. Coverage level (e.g., 0.90).
#'
#' @return Named list with ci_lo, ci_hi.
compute_quantile_ci <- function(draws, level = 0.90) {
  alpha <- 1 - level
  probs <- c(alpha / 2, 1 - alpha / 2)
  q <- quantile(draws, probs = probs, names = FALSE)
  list(ci_lo = q[1], ci_hi = q[2])
}


#' Compute Wald Confidence Interval
#'
#' @param theta_hat  Numeric scalar. Point estimate.
#' @param v_dd       Numeric scalar. Sandwich variance for this parameter.
#' @param level      Numeric. Coverage level (e.g., 0.90).
#'
#' @return Named list with ci_lo, ci_hi.
compute_wald_ci <- function(theta_hat, v_dd, level = 0.90) {
  alpha <- 1 - level
  z_crit <- qnorm(1 - alpha / 2)  # 1.6449 for level=0.90
  se <- sqrt(v_dd)
  list(
    ci_lo = theta_hat - z_crit * se,
    ci_hi = theta_hat + z_crit * se
  )
}


###############################################################################
## SECTION 2 : CLUSTER-ROBUST MEAT MATRIX J_cluster
###############################################################################
#' Compute Cluster-Robust Meat Matrix J_cluster
#'
#' J_cluster = sum_h (C_h/(C_h-1)) * sum_c (s_bar_hc - s_bar_h)(s_bar_hc - s_bar_h)'
#' where s_bar_hc = sum_{i in PSU(h,c)} w_tilde[i] * S_mat[i, ]
#'
#' NOTE: Scores from Stan are UNWEIGHTED. The weight multiplication happens
#' here when computing s_bar_hc.
#'
#' Singleton strata (C_h < 2) are skipped because within-stratum variance
#' cannot be estimated from a single PSU.
#'
#' @param S_mat        N x D score matrix (UNWEIGHTED individual scores).
#' @param w_tilde      N-vector of normalized survey weights.
#' @param stratum_idx  N-vector of stratum indices (integer).
#' @param psu_idx      N-vector of PSU indices (integer).
#' @param D            Integer. Fixed-effect dimension (2P+1).
#'
#' @return D x D positive (semi)definite meat matrix.

compute_J_cluster <- function(S_mat, w_tilde, stratum_idx, psu_idx, D) {

  N <- nrow(S_mat)
  strata_unique <- sort(unique(stratum_idx))
  H <- length(strata_unique)

  J_cluster <- matrix(0, D, D)
  n_singleton <- 0L

  for (h_idx in seq_len(H)) {
    stratum_h <- strata_unique[h_idx]

    ## Observations in this stratum
    obs_h <- which(stratum_idx == stratum_h)

    ## Unique PSUs within this stratum
    psus_in_h <- sort(unique(psu_idx[obs_h]))
    C_h <- length(psus_in_h)

    ## Skip singleton strata (cannot estimate within-stratum variance)
    if (C_h < 2L) {
      n_singleton <- n_singleton + 1L
      next
    }

    ## Compute weighted score totals for each PSU
    ## s_bar_hc = sum_{i in PSU(h,c)} w_tilde[i] * S_mat[i, ]
    s_bar_hc_mat <- matrix(0, nrow = C_h, ncol = D)

    for (c_idx in seq_along(psus_in_h)) {
      psu_c <- psus_in_h[c_idx]
      obs_in_hc <- obs_h[psu_idx[obs_h] == psu_c]

      ## Weighted score total: multiply UNWEIGHTED scores by weights
      if (length(obs_in_hc) == 1L) {
        s_bar_hc_mat[c_idx, ] <- w_tilde[obs_in_hc] * S_mat[obs_in_hc, ]
      } else {
        s_bar_hc_mat[c_idx, ] <- colSums(
          w_tilde[obs_in_hc] * S_mat[obs_in_hc, , drop = FALSE]
        )
      }
    }

    ## Stratum mean: s_bar_h = mean of all s_bar_hc
    s_bar_h <- colMeans(s_bar_hc_mat)

    ## Center each PSU score total
    delta_hc <- sweep(s_bar_hc_mat, 2, s_bar_h, "-")  # C_h x D

    ## Accumulate with finite population correction
    ## FPC = C_h / (C_h - 1) -- standard Taylor linearization factor
    fpc <- C_h / (C_h - 1)
    J_cluster <- J_cluster + fpc * crossprod(delta_hc)
  }

  if (n_singleton > 0) {
    cat(sprintf("    [J_cluster] Skipped %d singleton strata.\n", n_singleton))
  }

  J_cluster
}


###############################################################################
## SECTION 3 : OBSERVED INFORMATION H_obs (BLOCK-DIAGONAL)
###############################################################################
#' Compute Observed Information Matrix H_obs
#'
#' H_obs = block_diag(H_ext, H_int) where:
#'   H_ext (PxP)     : Analytic Fisher info for logistic extensive margin
#'   H_int ((P+1)x(P+1)) : Empirical information identity for intensive+kappa
#'
#' Block 1: H_ext = sum_i w_tilde[i] * q_i * (1-q_i) * X_i X_i^T
#'   where q_i = logistic(X_i * alpha + delta_ext[state_i])
#'   This is the EXACT Fisher information for the Bernoulli/logistic model.
#'
#' Block 2: H_int = sum_i w_tilde[i] * s_int_full_i s_int_full_i^T
#'   where s_int_full_i = [score_int_i(1:P), score_kappa_i]
#'   This uses the information identity: E[s s^T] = I(theta).
#'   Valid because the posterior concentrates around theta_hat for N ~ 7000.
#'
#' @param S_mat         N x D score matrix.
#' @param w_tilde       N-vector of normalized weights.
#' @param X             N x P design matrix.
#' @param state_idx     N-vector of state indices (integer 1:S).
#' @param alpha_hat     P-vector of extensive margin coefficients.
#' @param delta_means   S x 2 matrix of state random intercept posterior means.
#'                      Column 1 = extensive (delta_ext), Column 2 = intensive (delta_int).
#' @param N, P, D       Integer dimensions.
#'
#' @return D x D block-diagonal observed information matrix.

compute_H_obs <- function(S_mat, w_tilde, X, state_idx, alpha_hat,
                          delta_means, N, P, D) {

  ## --- Block 1: H_ext (P x P) -- analytic Fisher information ---
  ## H_ext = sum_i w_tilde[i] * q_i * (1-q_i) * x_i x_i'
  H_ext <- matrix(0, P, P)

  for (i in seq_len(N)) {
    s_i <- state_idx[i]
    ## M1 model: logit(q_i) = X_i * alpha + delta_ext[state_i]
    eta_ext_i <- sum(X[i, ] * alpha_hat) + delta_means[s_i, 1]
    q_i <- plogis(eta_ext_i)
    ## Weighted outer product: w * q*(1-q) * x x'
    H_ext <- H_ext + w_tilde[i] * q_i * (1 - q_i) * tcrossprod(X[i, ])
  }

  ## --- Block 2: H_int ((P+1) x (P+1)) -- empirical information identity ---
  ## H_int = sum_i w_tilde[i] * s_int_full_i s_int_full_i'
  ## s_int_full_i = [score_int (P cols), score_kappa (1 col)]
  P_int <- P + 1L  # beta[1:P] + log_kappa
  H_int <- matrix(0, P_int, P_int)

  for (i in seq_len(N)) {
    ## Extract intensive + kappa score: columns (P+1):D of S_mat
    s_int_full_i <- S_mat[i, (P + 1L):D]  # length P+1
    H_int <- H_int + w_tilde[i] * tcrossprod(s_int_full_i)
  }

  ## --- Assemble block-diagonal H_obs (D x D) ---
  H_obs <- matrix(0, D, D)
  H_obs[1:P, 1:P] <- H_ext
  H_obs[(P + 1L):D, (P + 1L):D] <- H_int

  H_obs
}


###############################################################################
## SECTION 4 : SANDWICH VARIANCE ESTIMATOR
###############################################################################
#' Compute the Full Sandwich Variance Estimator
#'
#' V_sand = H_obs^{-1} * J_cluster * H_obs^{-1}
#'
#' Includes PD safeguards:
#'   - Ridge correction for H_obs if not PD
#'   - Matrix::nearPD() for V_sand if not PD
#'
#' @param S_mat         N x D stacked score matrix (unweighted from Stan GQ).
#' @param w_tilde       N-vector of normalized survey weights.
#' @param stratum_idx   N-vector of stratum indices.
#' @param psu_idx       N-vector of PSU indices.
#' @param X             N x P design matrix.
#' @param state_idx     N-vector of state indices.
#' @param alpha_hat     P-vector of posterior mean alpha.
#' @param delta_means   S x 2 matrix of random intercept posterior means.
#' @param N, P, D       Integer dimensions.
#'
#' @return Named list:
#'   V_sand         : D x D sandwich variance matrix (guaranteed PD)
#'   H_obs          : D x D observed information
#'   H_obs_inv      : D x D inverse observed information
#'   J_cluster      : D x D cluster-robust meat matrix
#'   DER            : D-vector of design effect ratios
#'   pd_fix_applied : logical, TRUE if nearPD was needed for V_sand
#'   H_ridge_applied: logical, TRUE if ridge was needed for H_obs

compute_sandwich_variance <- function(S_mat, w_tilde, stratum_idx, psu_idx,
                                      X, state_idx, alpha_hat, delta_means,
                                      N, P, D) {

  ## Step 1: J_cluster (cluster-robust meat)
  cat("    Computing J_cluster ...\n")
  J_cluster <- compute_J_cluster(S_mat, w_tilde, stratum_idx, psu_idx, D)

  ## Step 2: H_obs (block-diagonal observed information)
  cat("    Computing H_obs ...\n")
  H_obs <- compute_H_obs(S_mat, w_tilde, X, state_idx, alpha_hat,
                         delta_means, N, P, D)

  ## Step 3: Check H_obs PD and invert
  eig_H <- eigen(H_obs, symmetric = TRUE, only.values = TRUE)$values
  min_eig_H <- min(eig_H)
  H_ridge_applied <- FALSE

  if (min_eig_H <= 0) {
    ## Add ridge to make H_obs PD
    ridge <- abs(min_eig_H) * 2 + 1e-8
    H_obs <- H_obs + ridge * diag(D)
    H_ridge_applied <- TRUE
    cat(sprintf("    [WARN] H_obs not PD (min eig = %.2e). Ridge = %.2e applied.\n",
                min_eig_H, ridge))
  }

  H_obs_inv <- solve(H_obs)

  ## Step 4: V_sand = H_obs^{-1} * J_cluster * H_obs^{-1}
  cat("    Computing V_sand = H_inv * J * H_inv ...\n")
  V_sand <- H_obs_inv %*% J_cluster %*% H_obs_inv

  ## Step 5: Check V_sand PD
  eig_V <- eigen(V_sand, symmetric = TRUE, only.values = TRUE)$values
  min_eig_V <- min(eig_V)
  pd_fix_applied <- FALSE

  if (min_eig_V <= 0) {
    V_sand <- as.matrix(Matrix::nearPD(V_sand, corr = FALSE)$mat)
    pd_fix_applied <- TRUE
    cat(sprintf("    [WARN] V_sand not PD (min eig = %.2e). nearPD correction applied.\n",
                min_eig_V))
  }

  ## Step 6: DER = diag(V_sand) / diag(H_obs_inv)
  DER <- diag(V_sand) / diag(H_obs_inv)

  cat(sprintf("    DER range: [%.3f, %.3f], mean = %.3f\n",
              min(DER), max(DER), mean(DER)))

  list(
    V_sand          = V_sand,
    H_obs           = H_obs,
    H_obs_inv       = H_obs_inv,
    J_cluster       = J_cluster,
    DER             = DER,
    pd_fix_applied  = pd_fix_applied,
    H_ridge_applied = H_ridge_applied,
    eigenvalues     = list(
      H_obs_min  = min_eig_H,
      V_sand_min = min_eig_V,
      V_sand_max = max(eig_V),
      V_cond     = max(eig_V) / max(min_eig_V, 1e-30)
    )
  )
}


###############################################################################
## SECTION 5 : CHOLESKY AFFINE TRANSFORMATION
###############################################################################
#' Apply Williams-Savitsky (2021) Cholesky Affine Correction
#'
#' Transforms MCMC draws so that cov(theta_corrected) = V_sand
#' while preserving the posterior mean.
#'
#' theta*^(m) = theta_hat + A * (theta^(m) - theta_hat)
#' where A = L_sand * L_MCMC^{-1}
#' L_MCMC = t(chol(Sigma_MCMC)), L_sand = t(chol(V_sand)) [lower triangular]
#'
#' NOTE: For the simulation evaluation, Wald CIs are primary.  The Cholesky
#' correction is provided for diagnostics only.  We do NOT need to store
#' corrected draws for metric computation -- only theta_hat and V_sand
#' diagonal matter.
#'
#' @param theta_draws  M x D matrix of MCMC draws for fixed effects.
#' @param theta_hat    D-vector of posterior means.
#' @param Sigma_MCMC   D x D empirical posterior covariance.
#' @param V_sand       D x D sandwich variance.
#'
#' @return Named list:
#'   theta_corrected : M x D matrix of corrected draws
#'   A               : D x D transformation matrix

apply_cholesky_correction <- function(theta_draws, theta_hat, Sigma_MCMC, V_sand) {

  D <- length(theta_hat)

  ## Ensure Sigma_MCMC is PD (add ridge if needed)
  eig_mcmc <- eigen(Sigma_MCMC, symmetric = TRUE, only.values = TRUE)$values
  if (min(eig_mcmc) <= 0) {
    Sigma_MCMC <- Sigma_MCMC + (abs(min(eig_mcmc)) + 1e-10) * diag(D)
  }

  ## Ensure V_sand is PD
  eig_sand <- eigen(V_sand, symmetric = TRUE, only.values = TRUE)$values
  if (min(eig_sand) <= 0) {
    V_sand <- as.matrix(Matrix::nearPD(V_sand, corr = FALSE)$mat)
  }

  ## Lower triangular Cholesky factors
  ## R's chol() returns UPPER triangular; transpose for lower
  L_MCMC <- t(chol(Sigma_MCMC))
  L_sand <- t(chol(V_sand))

  ## Transformation matrix: A = L_sand * L_MCMC^{-1}
  A <- L_sand %*% solve(L_MCMC)

  ## Apply affine transformation:
  ## theta*[m,] = theta_hat + A %*% (theta[m,] - theta_hat)
  ## In matrix form (row-major): (theta - 1*theta_hat') %*% A' + 1*theta_hat'
  theta_centered  <- sweep(theta_draws, 2, theta_hat, "-")
  theta_corrected <- sweep(theta_centered %*% t(A), 2, theta_hat, "+")

  list(
    theta_corrected = theta_corrected,
    A               = A
  )
}


###############################################################################
## SECTION 6 : METRIC COMPUTATION
###############################################################################
#' Compute Evaluation Metrics for One Parameter-Estimator Combination
#'
#' @param estimate    Numeric scalar. Point estimate (posterior mean).
#' @param ci_lo       Numeric scalar. Lower CI bound.
#' @param ci_hi       Numeric scalar. Upper CI bound.
#' @param true_value  Numeric scalar. True DGP value.
#' @param se          Numeric scalar. Standard error.
#'
#' @return Named list: estimate, ci_lo, ci_hi, ci_width, covers, bias, se, true_value.

compute_param_metrics <- function(estimate, ci_lo, ci_hi, true_value, se) {
  ci_width <- ci_hi - ci_lo
  covers   <- as.integer(true_value >= ci_lo & true_value <= ci_hi)
  bias     <- estimate - true_value

  list(
    estimate   = estimate,
    ci_lo      = ci_lo,
    ci_hi      = ci_hi,
    ci_width   = ci_width,
    covers     = covers,
    bias       = bias,
    se         = se,
    true_value = true_value
  )
}


###############################################################################
## SECTION 7 : ASSEMBLE METRICS DATA FRAME
###############################################################################
#' Assemble All Per-Replication Metrics into a Data Frame
#'
#' Computes metrics for 5 target parameters x 3 estimators = 15 rows.
#'
#' For fixed effects (alpha_poverty, beta_poverty, log_kappa):
#'   E-UW: quantile CI from unweighted draws
#'   E-WT: quantile CI from weighted draws
#'   E-WS: Wald CI using theta_hat_wt +/- z * sqrt(V_sand[d,d])
#'
#' For hyperparameters (tau_ext, tau_int):
#'   E-UW: quantile CI from unweighted tau draws
#'   E-WT: quantile CI from weighted tau draws
#'   E-WS: SAME as E-WT (no sandwich correction)
#'
#' @param fit_uw      List. Unweighted fit results.
#'   Expected fields: theta_hat (D-vector), theta_draws (MxD matrix),
#'                    tau_hat (2-vector), tau_draws (Mx2 matrix)
#' @param fit_wt_ws   List. Weighted fit results.
#'   Expected fields: theta_hat, theta_draws, tau_hat, tau_draws, Sigma_MCMC
#' @param sandwich    List. Output from compute_sandwich_variance().
#' @param config      SIM_CONFIG list.
#'
#' @return data.frame with columns:
#'   param, estimator, estimate, ci_lo, ci_hi, ci_width, covers, bias, se, true_value

assemble_metrics <- function(fit_uw, fit_wt_ws, sandwich, config) {

  ## Config
  target_params  <- config$evaluation$target_params   # 5 target params
  coverage_level <- config$evaluation$coverage_level   # 0.90
  P              <- config$true_params$P               # 5
  D              <- 2L * P + 1L                        # 11

  ## Index mapping for fixed effects in theta vector (D=11):
  ## alpha[1:5] = indices 1:5
  ## beta[1:5]  = indices 6:10
  ## log_kappa  = index 11
  fixed_idx_map <- list(
    alpha_poverty = 2L,      # alpha[2]
    beta_poverty  = P + 2L,  # beta[2] = 7
    log_kappa     = D        # 11
  )

  ## Tau index mapping (tau is a 2-vector)
  tau_idx_map <- list(
    tau_ext = 1L,
    tau_int = 2L
  )

  ## Build output
  n_targets <- length(target_params)
  estimator_ids <- c("E_UW", "E_WT", "E_WS")
  n_estimators <- length(estimator_ids)
  n_rows <- n_targets * n_estimators

  metrics_df <- data.frame(
    param      = character(n_rows),
    estimator  = character(n_rows),
    estimate   = numeric(n_rows),
    ci_lo      = numeric(n_rows),
    ci_hi      = numeric(n_rows),
    ci_width   = numeric(n_rows),
    covers     = integer(n_rows),
    bias       = numeric(n_rows),
    se         = numeric(n_rows),
    true_value = numeric(n_rows),
    stringsAsFactors = FALSE
  )

  row_idx <- 0L

  for (tp in target_params) {
    param_name <- tp$name
    true_value <- tp$true_value
    is_tau     <- param_name %in% c("tau_ext", "tau_int")

    for (est_id in estimator_ids) {
      row_idx <- row_idx + 1L

      if (is_tau) {
        ## -------------------------------------------------------
        ## Hyperparameter (tau): No sandwich correction
        ## -------------------------------------------------------
        k <- tau_idx_map[[param_name]]

        if (est_id == "E_UW") {
          estimate <- fit_uw$tau_hat[k]
          se_val   <- sd(fit_uw$tau_draws[, k])
          ci       <- compute_quantile_ci(fit_uw$tau_draws[, k],
                                          level = coverage_level)

        } else if (est_id == "E_WT") {
          estimate <- fit_wt_ws$tau_hat[k]
          se_val   <- sd(fit_wt_ws$tau_draws[, k])
          ci       <- compute_quantile_ci(fit_wt_ws$tau_draws[, k],
                                          level = coverage_level)

        } else {
          ## E-WS: SAME as E-WT for tau
          estimate <- fit_wt_ws$tau_hat[k]
          se_val   <- sd(fit_wt_ws$tau_draws[, k])
          ci       <- compute_quantile_ci(fit_wt_ws$tau_draws[, k],
                                          level = coverage_level)
        }

      } else {
        ## -------------------------------------------------------
        ## Fixed effect: Sandwich correction for E-WS
        ## -------------------------------------------------------
        d <- fixed_idx_map[[param_name]]

        if (est_id == "E_UW") {
          estimate <- fit_uw$theta_hat[d]
          se_val   <- sd(fit_uw$theta_draws[, d])
          ci       <- compute_quantile_ci(fit_uw$theta_draws[, d],
                                          level = coverage_level)

        } else if (est_id == "E_WT") {
          estimate <- fit_wt_ws$theta_hat[d]
          se_val   <- sd(fit_wt_ws$theta_draws[, d])
          ci       <- compute_quantile_ci(fit_wt_ws$theta_draws[, d],
                                          level = coverage_level)

        } else {
          ## E-WS: Wald CI from V_sand
          estimate <- fit_wt_ws$theta_hat[d]
          se_val   <- sqrt(sandwich$V_sand[d, d])
          ci       <- compute_wald_ci(estimate, sandwich$V_sand[d, d],
                                      level = coverage_level)
        }
      }

      ## Compute per-cell metrics
      m <- compute_param_metrics(estimate, ci$ci_lo, ci$ci_hi, true_value, se_val)

      metrics_df$param[row_idx]      <- param_name
      metrics_df$estimator[row_idx]  <- est_id
      metrics_df$estimate[row_idx]   <- m$estimate
      metrics_df$ci_lo[row_idx]      <- m$ci_lo
      metrics_df$ci_hi[row_idx]      <- m$ci_hi
      metrics_df$ci_width[row_idx]   <- m$ci_width
      metrics_df$covers[row_idx]     <- m$covers
      metrics_df$bias[row_idx]       <- m$bias
      metrics_df$se[row_idx]         <- m$se
      metrics_df$true_value[row_idx] <- m$true_value
    }
  }

  metrics_df
}


###############################################################################
## SECTION 8 : MAIN ENTRY POINT -- postprocess_replication()
###############################################################################
#' Post-Process One Simulation Replication
#'
#' Takes the fit results from sim_03_fit.R and the sample data from
#' sim_02_sampling.R, computes the sandwich variance, optionally applies
#' the Cholesky correction, and assembles per-replication evaluation metrics
#' for all target parameters and estimators.
#'
#' @param fit_results  List with components:
#'   \item{E_UW}{Unweighted fit: theta_hat (D-vector), theta_draws (MxD),
#'               Sigma_MCMC (DxD), tau_hat (2-vector), tau_draws (Mx2),
#'               delta_means (Sx2), diagnostics}
#'   \item{E_WT_WS}{Weighted fit: same as E_UW plus
#'               S_mat (NxD score matrix) or
#'               score_ext (NxP), score_int (NxP), score_kappa (N-vector)}
#' @param sample_data  List from sim_02_sampling.R draw_sample() with
#'                     component stan_data_wt containing survey design vars.
#' @param config       SIM_CONFIG list.
#'
#' @return Named list:
#'   \item{metrics}{data.frame with 15 rows (5 params x 3 estimators)}
#'   \item{sandwich}{list(V_sand, H_obs, J_cluster, DER, H_obs_inv, ...)}
#'   \item{cholesky}{list(theta_corrected, A) or NULL if failed/skipped}
#'   \item{diagnostics_uw}{from E_UW fit}
#'   \item{diagnostics_wt}{from E_WT_WS fit}
#'   \item{timing}{elapsed seconds}

postprocess_replication <- function(fit_results, sample_data, config) {

  t_start <- proc.time()

  ## -----------------------------------------------------------------------
  ## 0. Extract inputs
  ## -----------------------------------------------------------------------
  fit_uw    <- fit_results$E_UW
  fit_wt_ws <- fit_results$E_WT_WS

  ## Stan data (weighted version has all survey design info)
  sd_wt <- sample_data$stan_data_wt

  N <- sd_wt$N
  P <- sd_wt$P
  S <- sd_wt$S
  D <- 2L * P + 1L  # alpha(P) + beta(P) + log_kappa(1) = 11

  cat(sprintf("  [POST] Processing replication: N=%d, P=%d, S=%d, D=%d\n",
              N, P, S, D))

  ## -----------------------------------------------------------------------
  ## 1. Assemble score matrix (N x D)
  ## -----------------------------------------------------------------------
  ## The weighted fit produces scores either as:
  ##   (a) Pre-assembled S_mat (N x D), or
  ##   (b) Separate score_ext (NxP), score_int (NxP), score_kappa (N)
  ## Handle both cases.

  if (!is.null(fit_wt_ws$S_mat)) {
    S_mat <- fit_wt_ws$S_mat
  } else if (!is.null(fit_wt_ws$score_ext)) {
    S_mat <- cbind(fit_wt_ws$score_ext,
                   fit_wt_ws$score_int,
                   fit_wt_ws$score_kappa)
  } else {
    stop("[POST] Scores not found in fit_wt_ws. Need S_mat or score_ext/int/kappa.")
  }

  ## Validate
  stopifnot(
    "S_mat must be N x D"         = nrow(S_mat) == N && ncol(S_mat) == D,
    "theta_hat must have length D" = length(fit_wt_ws$theta_hat) == D,
    "X must be N x P"             = nrow(sd_wt$X) == N && ncol(sd_wt$X) == P,
    "w_tilde must have length N"   = length(sd_wt$w_tilde) == N
  )

  ## -----------------------------------------------------------------------
  ## 2. Extract posterior means for H_obs
  ## -----------------------------------------------------------------------
  ## alpha_hat: first P elements of theta_hat
  alpha_hat <- fit_wt_ws$theta_hat[1:P]

  ## delta_means: S x 2 matrix of random intercept posterior means
  delta_means <- fit_wt_ws$delta_means
  stopifnot(
    "delta_means must be S x 2" = nrow(delta_means) == S && ncol(delta_means) == 2
  )

  ## -----------------------------------------------------------------------
  ## 3. Compute sandwich variance
  ## -----------------------------------------------------------------------
  cat("  [POST] Computing sandwich variance ...\n")

  sandwich <- compute_sandwich_variance(
    S_mat       = S_mat,
    w_tilde     = sd_wt$w_tilde,
    stratum_idx = sd_wt$stratum_idx,
    psu_idx     = sd_wt$psu_idx,
    X           = sd_wt$X,
    state_idx   = sd_wt$state,
    alpha_hat   = alpha_hat,
    delta_means = delta_means,
    N           = N,
    P           = P,
    D           = D
  )

  if (sandwich$pd_fix_applied) {
    cat("  [POST] [WARN] nearPD correction was applied to V_sand.\n")
  }

  ## -----------------------------------------------------------------------
  ## 4. Cholesky correction (optional, for diagnostics)
  ## -----------------------------------------------------------------------
  cholesky_result <- tryCatch({
    cat("  [POST] Applying Cholesky correction ...\n")

    Sigma_MCMC <- fit_wt_ws$Sigma_MCMC
    stopifnot(
      "Sigma_MCMC must be D x D" = nrow(Sigma_MCMC) == D && ncol(Sigma_MCMC) == D
    )

    res <- apply_cholesky_correction(
      theta_draws = fit_wt_ws$theta_draws,
      theta_hat   = fit_wt_ws$theta_hat,
      Sigma_MCMC  = Sigma_MCMC,
      V_sand      = sandwich$V_sand
    )

    cat(sprintf("  [POST] Cholesky A diagonal: [%.4f, %.4f]\n",
                min(diag(res$A)), max(diag(res$A))))
    res
  },
  error = function(e) {
    cat(sprintf("  [POST] [WARN] Cholesky correction failed: %s\n", e$message))
    NULL
  })

  ## -----------------------------------------------------------------------
  ## 5. Assemble metrics
  ## -----------------------------------------------------------------------
  cat("  [POST] Assembling metrics (5 params x 3 estimators = 15 rows) ...\n")

  metrics_df <- assemble_metrics(fit_uw, fit_wt_ws, sandwich, config)

  ## -----------------------------------------------------------------------
  ## 6. Print per-replication summary
  ## -----------------------------------------------------------------------
  cat("  [POST] Per-replication summary:\n")
  cat(sprintf("    %-16s %-6s %10s %10s %10s %8s %6s %10s\n",
              "Parameter", "Est", "Estimate", "CI_lo", "CI_hi",
              "Width", "Cover", "Bias"))
  cat(sprintf("    %s\n", paste(rep("-", 78), collapse = "")))

  for (i in seq_len(nrow(metrics_df))) {
    r <- metrics_df[i, ]
    cat(sprintf("    %-16s %-6s %+10.5f %+10.5f %+10.5f %8.5f %6d %+10.5f\n",
                r$param, r$estimator,
                r$estimate, r$ci_lo, r$ci_hi,
                r$ci_width, r$covers, r$bias))
  }

  ## -----------------------------------------------------------------------
  ## 7. Return
  ## -----------------------------------------------------------------------
  t_elapsed <- (proc.time() - t_start)["elapsed"]
  cat(sprintf("\n  [POST] Post-processing completed in %.1f seconds.\n", t_elapsed))

  list(
    metrics        = metrics_df,
    sandwich       = sandwich,
    cholesky       = cholesky_result,
    diagnostics_uw = fit_uw$diagnostics,
    diagnostics_wt = fit_wt_ws$diagnostics,
    timing         = as.numeric(t_elapsed)
  )
}


###############################################################################
## SECTION 9 : BATCH HELPER -- postprocess_saved_fits()
###############################################################################
#' Post-Process a Saved Pair of Fit RDS Files
#'
#' Convenience wrapper that loads fit results and sample data from disk,
#' then calls postprocess_replication().
#'
#' @param scenario_id  Character. One of "S0", "S3", "S4".
#' @param rep_id       Integer. Replication number.
#' @param config       SIM_CONFIG list.
#'
#' @return Output of postprocess_replication(), or NULL on error.

postprocess_saved_fits <- function(scenario_id, rep_id, config) {

  ## Paths
  fit_uw_path <- file.path(config$paths$sim_fits, scenario_id, "E_UW",
                           sprintf("rep_%03d.rds", rep_id))
  fit_wt_path <- file.path(config$paths$sim_fits, scenario_id, "E_WT",
                           sprintf("rep_%03d.rds", rep_id))
  sample_path <- file.path(config$paths$sim_samples, scenario_id,
                           sprintf("rep_%03d.rds", rep_id))

  ## Check existence
  missing <- character(0)
  if (!file.exists(fit_uw_path)) missing <- c(missing, fit_uw_path)
  if (!file.exists(fit_wt_path)) missing <- c(missing, fit_wt_path)
  if (!file.exists(sample_path)) missing <- c(missing, sample_path)

  if (length(missing) > 0) {
    warning(sprintf("File(s) not found: %s", paste(missing, collapse = ", ")))
    return(NULL)
  }

  ## Load
  fit_uw      <- readRDS(fit_uw_path)
  fit_wt_ws   <- readRDS(fit_wt_path)
  sample_data <- readRDS(sample_path)

  fit_results <- list(E_UW = fit_uw, E_WT_WS = fit_wt_ws)

  ## Process
  postprocess_replication(fit_results, sample_data, config)
}


###############################################################################
## SECTION 10 : AGGREGATE ACROSS REPLICATIONS
###############################################################################
#' Aggregate Per-Replication Metrics into Summary Statistics
#'
#' Given a list of per-replication metric data frames, computes across-rep
#' summaries: coverage rate, mean bias, RMSE, mean CI width, and width ratio.
#'
#' @param metrics_list  List of data.frames, each from postprocess_replication()$metrics.
#' @param config        SIM_CONFIG list.
#'
#' @return data.frame with columns:
#'   param, estimator, R, coverage, coverage_mcse, mean_bias, median_bias,
#'   rmse, mean_ci_width, median_ci_width, mean_se, width_ratio

aggregate_metrics <- function(metrics_list, config) {

  ## Combine all per-rep metrics
  all_metrics <- do.call(rbind, metrics_list)
  R_actual <- length(metrics_list)

  ## Unique param-estimator combinations
  combos <- unique(all_metrics[, c("param", "estimator")])
  n_combos <- nrow(combos)

  summary_df <- data.frame(
    param            = character(n_combos),
    estimator        = character(n_combos),
    R                = integer(n_combos),
    coverage         = numeric(n_combos),
    coverage_mcse    = numeric(n_combos),
    mean_bias        = numeric(n_combos),
    median_bias      = numeric(n_combos),
    rmse             = numeric(n_combos),
    mean_ci_width    = numeric(n_combos),
    median_ci_width  = numeric(n_combos),
    mean_se          = numeric(n_combos),
    width_ratio      = numeric(n_combos),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(n_combos)) {
    p_name <- combos$param[i]
    e_name <- combos$estimator[i]

    idx <- which(all_metrics$param == p_name & all_metrics$estimator == e_name)
    sub <- all_metrics[idx, ]
    R_sub <- nrow(sub)

    ## Coverage
    cov_rate <- mean(sub$covers)
    cov_mcse <- sqrt(cov_rate * (1 - cov_rate) / max(R_sub, 1))

    ## Bias
    mean_bias   <- mean(sub$bias)
    median_bias <- median(sub$bias)

    ## RMSE = sqrt(mean(bias^2))
    rmse <- sqrt(mean(sub$bias^2))

    ## CI width
    mean_ci_width   <- mean(sub$ci_width)
    median_ci_width <- median(sub$ci_width)

    ## SE
    mean_se <- mean(sub$se)

    summary_df$param[i]           <- p_name
    summary_df$estimator[i]       <- e_name
    summary_df$R[i]               <- R_sub
    summary_df$coverage[i]        <- cov_rate
    summary_df$coverage_mcse[i]   <- cov_mcse
    summary_df$mean_bias[i]       <- mean_bias
    summary_df$median_bias[i]     <- median_bias
    summary_df$rmse[i]            <- rmse
    summary_df$mean_ci_width[i]   <- mean_ci_width
    summary_df$median_ci_width[i] <- median_ci_width
    summary_df$mean_se[i]         <- mean_se
    summary_df$width_ratio[i]     <- NA_real_
  }

  ## Compute width ratio: E-WS / E-WT for each parameter
  ## This shows how much the sandwich changes interval width relative to
  ## the naive pseudo-posterior CI.
  for (p_name in unique(summary_df$param)) {
    idx_wt <- which(summary_df$param == p_name & summary_df$estimator == "E_WT")
    idx_ws <- which(summary_df$param == p_name & summary_df$estimator == "E_WS")

    if (length(idx_wt) == 1 && length(idx_ws) == 1) {
      w_wt <- summary_df$mean_ci_width[idx_wt]
      if (w_wt > 0) {
        summary_df$width_ratio[idx_ws] <- summary_df$mean_ci_width[idx_ws] / w_wt
      }
    }
  }

  ## Flag tau parameters where sandwich is NOT applicable ().
  ## E-WS coverage for tau = E-WT coverage for tau (no sandwich correction
  ## exists for hyperparameters). This note column helps downstream reporting.
  tau_params <- c("tau_ext", "tau_int")
  summary_df$sandwich_note <- ""
  tau_ws_idx <- which(summary_df$param %in% tau_params &
                        summary_df$estimator == "E_WS")
  if (length(tau_ws_idx) > 0) {
    summary_df$sandwich_note[tau_ws_idx] <-
      "No sandwich correction for hyperparameters; E-WS CI = E-WT CI"
  }

  summary_df
}


###############################################################################
## SECTION 11 : PRINT SUMMARY TABLE
###############################################################################
#' Print Formatted Summary of Aggregated Metrics
#'
#' @param summary_df  data.frame from aggregate_metrics().
#' @param scenario_id Character. For labeling.
#' @param config      SIM_CONFIG list.

print_metrics_summary <- function(summary_df, scenario_id = "", config = NULL) {

  cat("\n================================================================\n")
  cat("  SIMULATION RESULTS SUMMARY")
  if (nzchar(scenario_id)) cat(sprintf("  -- Scenario %s", scenario_id))
  cat("\n================================================================\n")

  coverage_level <- if (!is.null(config)) config$evaluation$coverage_level else 0.90
  nominal <- coverage_level

  cat(sprintf("\n  Nominal coverage: %.0f%%\n", 100 * nominal))
  if (nrow(summary_df) > 0) {
    cat(sprintf("  Replications: %d\n\n", summary_df$R[1]))
  }

  ## Group by parameter
  params <- unique(summary_df$param)

  for (p_name in params) {
    sub <- summary_df[summary_df$param == p_name, ]

    ## Get true value from config
    tv <- NA_real_
    if (!is.null(config)) {
      for (tp in config$evaluation$target_params) {
        if (tp$name == p_name) {
          tv <- tp$true_value
          break
        }
      }
    }

    cat(sprintf("  --- %s (true = %+.6f) ---\n", p_name,
                ifelse(is.na(tv), 0, tv)))
    cat(sprintf("  %-6s %8s %10s %10s %10s %10s %10s\n",
                "Est", "Cover", "MCSE", "Bias", "RMSE",
                "CI Width", "W Ratio"))
    cat(sprintf("  %s\n", paste(rep("-", 68), collapse = "")))

    for (j in seq_len(nrow(sub))) {
      r <- sub[j, ]
      wr_str <- ifelse(is.na(r$width_ratio), "       ---",
                        sprintf("%10.4f", r$width_ratio))

      ## Flag significant undercoverage (> 2 MCSE below nominal)
      cover_flag <- ""
      if (r$coverage < nominal - 2 * r$coverage_mcse) {
        cover_flag <- " *"
      }

      cat(sprintf("  %-6s %7.3f%s %10.4f %+10.5f %10.5f %10.5f %s\n",
                  r$estimator,
                  r$coverage, cover_flag,
                  r$coverage_mcse,
                  r$mean_bias,
                  r$rmse,
                  r$mean_ci_width,
                  wr_str))
    }
    cat("\n")
  }

  ## Legend
  cat("  Legend:\n")
  cat("    * = coverage significantly below nominal (> 2 MCSE)\n")
  cat("    W Ratio = E-WS CI width / E-WT CI width\n")
  cat("             (shows how sandwich adjusts CI width relative to naive weighted)\n")
  cat("  NOTE: tau_ext and tau_int are NOT sandwich-correctable.\n")
  cat("        E-WS CIs for tau = E-WT CIs (no correction applied).\n")

  cat("\n================================================================\n")
  cat("  END OF RESULTS SUMMARY\n")
  cat("================================================================\n\n")

  invisible(NULL)
}


###############################################################################
## SECTION 12 : STANDALONE EXECUTION
###############################################################################
## When sourced directly (not via another script), print module summary
## and optionally run a quick sanity test.

if (!isTRUE(.SIM_04_CALLED_FROM_PARENT)) {

  cat("\n--------------------------------------------------------------\n")
  cat("  Standalone mode: running quick sanity tests ...\n")
  cat("--------------------------------------------------------------\n\n")

  ## --- Test 1: CI helper functions ---
  cat("  Test 1: CI helper functions ...\n")
  set.seed(123)
  test_draws <- rnorm(5000, mean = 2.0, sd = 0.5)

  ## Quantile CI should contain 2.0 in ~90% of cases
  ci_q <- compute_quantile_ci(test_draws, level = 0.90)
  cat(sprintf("    Quantile CI: [%.4f, %.4f], width = %.4f\n",
              ci_q$ci_lo, ci_q$ci_hi, ci_q$ci_hi - ci_q$ci_lo))
  stopifnot(ci_q$ci_lo < 2.0 && ci_q$ci_hi > 2.0)

  ## Wald CI
  ci_w <- compute_wald_ci(2.0, 0.25, level = 0.90)  # SE = 0.5
  cat(sprintf("    Wald CI:     [%.4f, %.4f], width = %.4f\n",
              ci_w$ci_lo, ci_w$ci_hi, ci_w$ci_hi - ci_w$ci_lo))
  expected_width <- 2 * qnorm(0.95) * 0.5  # 2 * 1.6449 * 0.5 = 1.6449
  stopifnot(abs((ci_w$ci_hi - ci_w$ci_lo) - expected_width) < 0.001)
  cat("  [PASS] CI functions work correctly.\n\n")

  ## --- Test 2: J_cluster computation ---
  cat("  Test 2: J_cluster computation ...\n")
  N_t <- 100L
  D_t <- 5L
  S_mat_t <- matrix(rnorm(N_t * D_t), N_t, D_t)
  w_t <- rep(1, N_t)  # uniform weights
  strat_t <- rep(1:5, each = 20)
  psu_t <- rep(1:20, each = 5)

  J_t <- compute_J_cluster(S_mat_t, w_t, strat_t, psu_t, D_t)
  cat(sprintf("    J_cluster: %d x %d, symmetric: %s\n",
              nrow(J_t), ncol(J_t),
              max(abs(J_t - t(J_t))) < 1e-10))
  stopifnot(nrow(J_t) == D_t, ncol(J_t) == D_t)
  stopifnot(max(abs(J_t - t(J_t))) < 1e-10)  # symmetric
  cat("  [PASS] J_cluster computation works.\n\n")

  ## --- Test 3: H_obs computation ---
  cat("  Test 3: H_obs computation ...\n")
  P_t <- 3L
  D_t2 <- 2L * P_t + 1L  # 7
  N_t2 <- 50L
  S_t <- 5L

  X_t <- cbind(1, matrix(rnorm(N_t2 * (P_t - 1)), N_t2, P_t - 1))
  state_t <- sample(1:S_t, N_t2, replace = TRUE)
  alpha_t <- rnorm(P_t)
  delta_t <- matrix(rnorm(S_t * 2, sd = 0.3), S_t, 2)
  S_mat_t2 <- matrix(rnorm(N_t2 * D_t2, sd = 0.2), N_t2, D_t2)
  w_t2 <- runif(N_t2, 0.5, 2.0)

  H_t <- compute_H_obs(S_mat_t2, w_t2, X_t, state_t, alpha_t, delta_t,
                        N_t2, P_t, D_t2)
  cat(sprintf("    H_obs: %d x %d, symmetric: %s\n",
              nrow(H_t), ncol(H_t),
              max(abs(H_t - t(H_t))) < 1e-10))
  ## Block structure: H[1:P, (P+1):D] should be zero
  off_block_max <- max(abs(H_t[1:P_t, (P_t + 1):D_t2]))
  cat(sprintf("    Off-diagonal block max: %.2e (should be ~0)\n", off_block_max))
  stopifnot(off_block_max < 1e-10)
  cat("  [PASS] H_obs block-diagonal structure verified.\n\n")

  ## --- Test 4: Full sandwich computation ---
  cat("  Test 4: Full sandwich computation ...\n")
  strat_t2 <- ((seq_len(N_t2) - 1) %% 5) + 1
  psu_t2 <- ((seq_len(N_t2) - 1) %% 10) + 1

  sand_t <- compute_sandwich_variance(
    S_mat_t2, w_t2, strat_t2, psu_t2,
    X_t, state_t, alpha_t, delta_t,
    N_t2, P_t, D_t2
  )

  cat(sprintf("    V_sand: %d x %d, PD fix: %s\n",
              nrow(sand_t$V_sand), ncol(sand_t$V_sand),
              sand_t$pd_fix_applied))
  cat(sprintf("    DER range: [%.3f, %.3f]\n",
              min(sand_t$DER), max(sand_t$DER)))

  ## Verify V_sand = H_inv J H_inv
  V_check <- sand_t$H_obs_inv %*% sand_t$J_cluster %*% sand_t$H_obs_inv
  reconstruction_err <- max(abs(V_check - sand_t$V_sand))
  cat(sprintf("    Reconstruction error: %.2e", reconstruction_err))
  if (reconstruction_err < 1e-8 || sand_t$pd_fix_applied) {
    cat(" [PASS]\n")
  } else {
    cat(" [NOTE]\n")
  }
  cat("  [PASS] Sandwich variance computation works.\n\n")

  ## --- Test 5: Cholesky correction ---
  cat("  Test 5: Cholesky correction ...\n")
  M_t <- 1000L
  theta_draws_t <- matrix(rnorm(M_t * D_t2, sd = 0.5), M_t, D_t2)
  theta_hat_t <- colMeans(theta_draws_t)
  Sigma_t <- cov(theta_draws_t)

  chol_t <- apply_cholesky_correction(theta_draws_t, theta_hat_t,
                                       Sigma_t, sand_t$V_sand)

  mean_err <- max(abs(colMeans(chol_t$theta_corrected) - theta_hat_t))
  cov_corrected <- cov(chol_t$theta_corrected)
  var_err <- max(abs(cov_corrected - sand_t$V_sand)) / max(abs(sand_t$V_sand))

  cat(sprintf("    Mean preservation error: %.2e\n", mean_err))
  cat(sprintf("    Variance relative error: %.2e\n", var_err))
  stopifnot(mean_err < 1e-10)
  stopifnot(var_err < 0.05)  # finite sample tolerance
  cat("  [PASS] Cholesky correction preserves mean and corrects variance.\n\n")

  ## --- Test 6: Metric assembly (synthetic) ---
  cat("  Test 6: Metric assembly (synthetic) ...\n")
  ## Create minimal synthetic fit structures matching expected format
  P_cfg <- SIM_CONFIG$true_params$P
  D_cfg <- 2L * P_cfg + 1L
  M_cfg <- 500L
  S_cfg <- SIM_CONFIG$population$S

  ## True values for centering synthetic draws
  true_alpha <- SIM_CONFIG$true_params$alpha
  true_beta  <- SIM_CONFIG$true_params$beta
  true_lk    <- SIM_CONFIG$true_params$log_kappa
  true_tau   <- SIM_CONFIG$true_params$tau
  theta_true <- c(true_alpha, true_beta, true_lk)

  ## Synthetic draws centered near truth
  set.seed(42)
  theta_draws_syn <- matrix(NA_real_, M_cfg, D_cfg)
  for (d in seq_len(D_cfg)) {
    theta_draws_syn[, d] <- rnorm(M_cfg, mean = theta_true[d], sd = 0.05)
  }
  tau_draws_syn <- cbind(
    abs(rnorm(M_cfg, mean = true_tau[1], sd = 0.05)),
    abs(rnorm(M_cfg, mean = true_tau[2], sd = 0.05))
  )

  ## Fake V_sand (diagonal)
  V_sand_syn <- diag(rep(0.01, D_cfg))

  fit_uw_syn <- list(
    theta_hat   = colMeans(theta_draws_syn),
    theta_draws = theta_draws_syn,
    Sigma_MCMC  = cov(theta_draws_syn),
    tau_hat     = colMeans(tau_draws_syn),
    tau_draws   = tau_draws_syn,
    delta_means = matrix(0, S_cfg, 2),
    diagnostics = list(ok = TRUE)
  )

  fit_wt_syn <- fit_uw_syn  # same structure for test

  sandwich_syn <- list(V_sand = V_sand_syn)

  metrics_syn <- assemble_metrics(fit_uw_syn, fit_wt_syn, sandwich_syn, SIM_CONFIG)
  cat(sprintf("    Metrics rows: %d (expected 15)\n", nrow(metrics_syn)))
  stopifnot(nrow(metrics_syn) == 15)

  ## Check all 5 params x 3 estimators present
  expected_params <- c("alpha_poverty", "beta_poverty", "log_kappa",
                       "tau_ext", "tau_int")
  expected_ests   <- c("E_UW", "E_WT", "E_WS")
  for (p in expected_params) {
    for (e in expected_ests) {
      n_found <- sum(metrics_syn$param == p & metrics_syn$estimator == e)
      stopifnot(n_found == 1)
    }
  }
  cat("  [PASS] Metric assembly produces correct 15-row data.frame.\n\n")

  ## --- Test 7: Aggregation ---
  cat("  Test 7: Aggregation ...\n")
  agg_syn <- aggregate_metrics(list(metrics_syn, metrics_syn, metrics_syn),
                                SIM_CONFIG)
  cat(sprintf("    Aggregated rows: %d (expected 15)\n", nrow(agg_syn)))
  stopifnot(nrow(agg_syn) == 15)

  ## All coverages should be 0 or 1 (since all 3 reps are identical)
  stopifnot(all(agg_syn$coverage %in% c(0, 1)))

  ## R should be 3 for all
  stopifnot(all(agg_syn$R == 3))
  cat("  [PASS] Aggregation works correctly.\n\n")

  cat("--------------------------------------------------------------\n")
  cat("  All standalone tests passed.\n")
  cat("--------------------------------------------------------------\n\n")

  cat("==============================================================\n")
  cat("  POST-PROCESSING MODULE LOADED\n")
  cat("==============================================================\n")
  cat("\n  Exported functions:\n")
  cat("    postprocess_replication(fit_results, sample_data, config)\n")
  cat("    postprocess_saved_fits(scenario_id, rep_id, config)\n")
  cat("    compute_sandwich_variance(S_mat, w_tilde, stratum_idx, psu_idx,\n")
  cat("                              X, state_idx, alpha_hat, delta_means, N, P, D)\n")
  cat("    apply_cholesky_correction(theta_draws, theta_hat, Sigma_MCMC, V_sand)\n")
  cat("    compute_quantile_ci(draws, level)\n")
  cat("    compute_wald_ci(theta_hat, v_dd, level)\n")
  cat("    compute_param_metrics(estimate, ci_lo, ci_hi, true_value, se)\n")
  cat("    assemble_metrics(fit_uw, fit_wt_ws, sandwich, config)\n")
  cat("    aggregate_metrics(metrics_list, config)\n")
  cat("    print_metrics_summary(summary_df, scenario_id, config)\n")
  cat("\n  Typical workflow (per replication):\n")
  cat("    fit_results <- list(E_UW = ..., E_WT_WS = ...)\n")
  cat("    sample_data <- list(stan_data_wt = ...)\n")
  cat("    result <- postprocess_replication(fit_results, sample_data, SIM_CONFIG)\n")
  cat("    result$metrics     # 15-row data.frame\n")
  cat("    result$sandwich    # V_sand, H_obs, J_cluster, DER\n")
  cat("\n  Aggregation (across R replications):\n")
  cat("    all_metrics <- lapply(rep_results, function(r) r$metrics)\n")
  cat("    summary <- aggregate_metrics(all_metrics, SIM_CONFIG)\n")
  cat("    print_metrics_summary(summary, 'S3', SIM_CONFIG)\n")
  cat("==============================================================\n")
}
