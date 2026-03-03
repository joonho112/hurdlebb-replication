## =============================================================================
## sim_03_fit.R -- Stan Model Fitting Module
## =============================================================================
## Purpose : Stan model fitting wrapper for the simulation pipeline.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##           Compiles HBB M1 models (unweighted and weighted), fits them
##           to simulated survey samples, and extracts all quantities
##           needed for three estimators:
##
##           E-UW  (Unweighted):       hbb_m1.stan, model-based CIs
##           E-WT  (Weighted-naive):   hbb_m1_weighted.stan, pseudo-posterior CIs
##           E-WS  (Weighted-sandwich): same fit as E-WT, sandwich CIs (sim_04)
##
##           E-WT and E-WS share a single Stan fit. Only TWO fits per rep.
##
## Architecture:
##   sim_00_config.R -> sim_01_dgp.R -> sim_02_sampling.R
##     -> sim_03_fit.R (this file)
##     -> sim_04_postprocess.R
##
## Stan Models:
##   hbb_m1.stan           -- unweighted M1 (168 lines)
##   hbb_m1_weighted.stan  -- weighted M1 with score GQs (244 lines)
##
## Extracted Quantities (per fit):
##   theta_hat    : D-vector (D = 2P + 1 = 11) of posterior means
##   theta_draws  : M x D matrix (M = chains * iter_sampling = 8000)
##   Sigma_MCMC   : D x D empirical posterior covariance
##   tau_hat      : 2-vector [tau_ext, tau_int]
##   tau_draws    : M x 2
##   delta_means  : S x 2 posterior mean state random intercepts
##   diagnostics  : {rhat_max, ess_bulk_min, n_divergent, converged, ...}
##
##   Additionally for the weighted fit (E-WT / E-WS):
##   score_ext    : N x P posterior-mean extensive margin scores
##   score_int    : N x P posterior-mean intensive margin scores
##   score_kappa  : N-vector posterior-mean dispersion scores
##   S_mat        : N x D = cbind(score_ext, score_int, score_kappa)
##
## Memory Management:
##   After extracting quantities, the full CmdStanR fit object is NOT
##   retained. Each fit can consume ~400 MB for N ~ 7000, and retaining
##   them across R = 200 replications is infeasible.
##
## Dependencies:
##   - cmdstanr     : CmdStan interface
##   - posterior    : draws manipulation
##   - sim_00_config.R : SIM_CONFIG, get_rep_seed()
##
## Usage:
##   source("code/simulation/sim_03_fit.R")
##   models <- compile_stan_models(SIM_CONFIG)
##   result <- fit_all_estimators(stan_data_uw, stan_data_wt,
##                                 models, SIM_CONFIG, seed = 12345)
##
## Exports:
##   compile_stan_models()       -- compile both Stan models once
##   fit_unweighted()            -- fit E-UW
##   fit_weighted()              -- fit E-WT / E-WS (shared fit)
##   fit_all_estimators()        -- orchestrate both fits for one rep
##   extract_fixed_effects()     -- theta_hat, theta_draws, Sigma_MCMC
##   extract_random_effects()    -- tau_hat, tau_draws, delta_means
##   extract_scores()            -- score_ext, score_int, score_kappa, S_mat
##   extract_diagnostics()       -- MCMC convergence summary
##   check_convergence()         -- pass/fail against config thresholds
##   make_init_fun()             -- initial values for Stan sampler
##   save_fit_results()          -- save per-rep results to disk
##   load_fit_results()          -- reload from disk
##   print_fit_summary()         -- formatted diagnostic printout
##   run_fitting_pipeline()      -- full single-rep pipeline
##
## =============================================================================

cat("==============================================================\n")
cat("  Model Fitting Module  (sim_03_fit.R)\n")
cat("==============================================================\n\n")


###############################################################################
## SECTION 0 : ENVIRONMENT AND DEPENDENCIES
###############################################################################

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
                           

## Source config if not already loaded
if (!exists("SIM_CONFIG")) {
  cat("  Loading SIM_CONFIG from sim_00_config.R ...\n")
  source(file.path(PROJECT_ROOT, "code/simulation/sim_00_config.R"))
}

## Load required packages
if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  stop("Package 'cmdstanr' is required. Install from https://mc-stan.org/cmdstanr/")
}
if (!requireNamespace("posterior", quietly = TRUE)) {
  stop("Package 'posterior' is required. Install with: install.packages('posterior')")
}

library(cmdstanr)
library(posterior)

cat("  Dependencies loaded: cmdstanr, posterior\n\n")


###############################################################################
## SECTION 1 : COMPILE STAN MODELS
###############################################################################
#' Compile Both Stan Models (Unweighted and Weighted M1)
#'
#' Should be called ONCE before the replication loop. CmdStanR caches
#' compiled binaries; recompilation only occurs if the .stan file changed.
#'
#' @param config  SIM_CONFIG list (for $paths to .stan files)
#'
#' @return Named list:
#'   \item{m1_uw}{CmdStanModel for hbb_m1.stan}
#'   \item{m1_wt}{CmdStanModel for hbb_m1_weighted.stan}
#'   \item{m1_wt_noscores}{CmdStanModel for hbb_m1_weighted_noscores.stan (or NULL)}

compile_stan_models <- function(config) {

  cat("  [COMPILE] Compiling Stan models ...\n")
  t0 <- proc.time()

  path_uw <- config$paths$stan_model_m1
  path_wt <- config$paths$stan_model_m1_weighted

  if (!file.exists(path_uw)) {
    stop(sprintf("[COMPILE] Unweighted M1 not found: %s", path_uw))
  }
  if (!file.exists(path_wt)) {
    stop(sprintf("[COMPILE] Weighted M1 not found: %s", path_wt))
  }

  cat(sprintf("    Unweighted: %s\n", basename(path_uw)))
  m1_uw <- tryCatch(
    cmdstan_model(path_uw),
    error = function(e) {
      stop(sprintf("[COMPILE] Unweighted M1 failed: %s", conditionMessage(e)),
           call. = FALSE)
    }
  )
  cat("    Unweighted: OK\n")

  cat(sprintf("    Weighted:   %s\n", basename(path_wt)))
  m1_wt <- tryCatch(
    cmdstan_model(path_wt),
    error = function(e) {
      stop(sprintf("[COMPILE] Weighted M1 failed: %s", conditionMessage(e)),
           call. = FALSE)
    }
  )
  cat("    Weighted:   OK\n")

  ## --- Weighted M1 (no-scores variant: smaller CSV, no OOM on cloud) ---
  path_wt_ns <- sub("\\.stan$", "_noscores.stan", path_wt)
  if (file.exists(path_wt_ns)) {
    cat(sprintf("    Weighted (noscores): %s\n", basename(path_wt_ns)))
    m1_wt_noscores <- tryCatch(
      cmdstan_model(path_wt_ns),
      error = function(e) {
        cat(sprintf("    [COMPILE] Noscores variant failed: %s (will use standard weighted)\n",
                    conditionMessage(e)))
        NULL
      }
    )
    if (!is.null(m1_wt_noscores)) cat("    Weighted (noscores): OK\n")
  } else {
    m1_wt_noscores <- NULL
  }

  elapsed <- (proc.time() - t0)[["elapsed"]]
  n_models <- 2L + as.integer(!is.null(m1_wt_noscores))
  cat(sprintf("  [COMPILE] %d models compiled in %.1f sec.\n\n", n_models, elapsed))

  list(m1_uw = m1_uw, m1_wt = m1_wt, m1_wt_noscores = m1_wt_noscores)
}


###############################################################################
## SECTION 2 : INITIAL VALUE GENERATOR
###############################################################################
#' Create Initial Value Function for Stan Sampler
#'
#' Returns a function that generates a fresh set of initial values each
#' time it is called (one call per chain). Uses moderate defaults:
#' alpha and beta near zero, kappa near 10, tau small positive.
#' Small random perturbation ensures chains start from different points.
#'
#' @param P  Integer. Number of covariates (including intercept).
#' @param S  Integer. Number of states.
#'
#' @return Function suitable for the `init` argument of CmdStanR$sample().

make_init_fun <- function(P, S) {
  function() {
    list(
      alpha     = rnorm(P, 0, 0.05),
      beta      = rnorm(P, 0, 0.05),
      log_kappa = rnorm(1, log(10), 0.1),
      tau       = abs(rnorm(2, 0.3, 0.05)),
      L_Omega   = diag(2),
      z_delta   = matrix(rnorm(2 * S, 0, 0.1), nrow = 2, ncol = S)
    )
  }
}


###############################################################################
## SECTION 3 : EXTRACT FIXED EFFECTS
###############################################################################
#' Extract Fixed-Effect Parameter Posterior from a CmdStanR Fit
#'
#' The D = 2P + 1 fixed-effect parameters are ordered as:
#'   theta = [alpha[1], ..., alpha[P], beta[1], ..., beta[P], log_kappa]
#'
#' @param fit  CmdStanMCMC object.
#' @param P    Integer. Number of covariates.
#'
#' @return List with:
#'   \item{theta_hat}{D-vector of posterior means}
#'   \item{theta_draws}{M x D matrix of MCMC draws}
#'   \item{Sigma_MCMC}{D x D posterior covariance}
#'   \item{D}{Integer. Dimension of theta}

extract_fixed_effects <- function(fit, P) {

  D <- 2L * P + 1L

  ## Stan parameter names
  alpha_names <- paste0("alpha[", seq_len(P), "]")
  beta_names  <- paste0("beta[",  seq_len(P), "]")
  all_names   <- c(alpha_names, beta_names, "log_kappa")

  ## Extract as draws_matrix (posterior package class), then strip to plain matrix
  draws_raw <- fit$draws(variables = all_names, format = "draws_matrix")
  theta_draws <- as.matrix(draws_raw)
  colnames(theta_draws) <- all_names

  ## Posterior mean and covariance
  theta_hat  <- colMeans(theta_draws)
  Sigma_MCMC <- cov(theta_draws)

  list(
    theta_hat   = theta_hat,
    theta_draws = theta_draws,
    Sigma_MCMC  = Sigma_MCMC,
    D           = D
  )
}


###############################################################################
## SECTION 4 : EXTRACT RANDOM EFFECTS
###############################################################################
#' Extract Random-Effect Posterior from a CmdStanR Fit
#'
#' Extracts tau (scale), tau_draws, and delta posterior means.
#' delta[k, s] in Stan: k=1 (extensive), k=2 (intensive), s=1..S.
#' Returns delta as S x 2 matrix.
#'
#' @param fit  CmdStanMCMC object.
#' @param S    Integer. Number of states.
#'
#' @return List with:
#'   \item{tau_hat}{Named 2-vector [tau_ext, tau_int]}
#'   \item{tau_draws}{M x 2 matrix}
#'   \item{delta_means}{S x 2 matrix (col 1 = ext, col 2 = int)}

extract_random_effects <- function(fit, S) {

  ## --- tau ---
  tau_raw <- fit$draws(variables = c("tau[1]", "tau[2]"),
                       format = "draws_matrix")
  tau_draws <- as.matrix(tau_raw)
  colnames(tau_draws) <- c("tau_ext", "tau_int")
  tau_hat <- colMeans(tau_draws)

  ## --- delta: posterior means only (to save memory) ---
  ## delta[1, s] for s = 1..S => extensive margin
  ## delta[2, s] for s = 1..S => intensive margin
  delta_ext_names <- paste0("delta[1,", seq_len(S), "]")
  delta_int_names <- paste0("delta[2,", seq_len(S), "]")

  ## Extract, take colMeans, discard full draws
  delta_ext_raw <- fit$draws(variables = delta_ext_names,
                             format = "draws_matrix")
  delta_ext_means <- colMeans(as.matrix(delta_ext_raw))
  rm(delta_ext_raw)

  delta_int_raw <- fit$draws(variables = delta_int_names,
                             format = "draws_matrix")
  delta_int_means <- colMeans(as.matrix(delta_int_raw))
  rm(delta_int_raw)

  ## Assemble as S x 2
  delta_means <- cbind(ext = as.numeric(delta_ext_means),
                       int = as.numeric(delta_int_means))
  rownames(delta_means) <- paste0("s", seq_len(S))

  list(
    tau_hat     = tau_hat,
    tau_draws   = tau_draws,
    delta_means = delta_means
  )
}


###############################################################################
## SECTION 5 : EXTRACT SCORES (WEIGHTED FIT ONLY)
###############################################################################
#' Extract Posterior-Mean Score Vectors from a Weighted CmdStanR Fit
#'
#' The weighted Stan model (hbb_m1_weighted.stan) computes individual
#' UNWEIGHTED score vectors in generated quantities:
#'   score_ext[N, P]  -- d ell_i / d alpha (via chain rule)
#'   score_int[N, P]  -- d ell_i / d beta
#'   score_kappa[N]   -- d ell_i / d log_kappa
#'
#' CmdStan stores matrix[N, P] in column-major order:
#'   score_ext[1,1], score_ext[2,1], ..., score_ext[N,1],
#'   score_ext[1,2], ..., score_ext[N,P]
#' i.e., the first (row) index varies fastest.
#'
#' We take colMeans across MCMC draws to get the posterior-mean score
#' for each (i, p), then reshape to N x P using column-major fill
#' (byrow = FALSE).
#'
#' @param fit  CmdStanMCMC object (weighted model).
#' @param N    Integer. Number of observations.
#' @param P    Integer. Number of covariates.
#'
#' @return List with:
#'   \item{score_ext}{N x P matrix}
#'   \item{score_int}{N x P matrix}
#'   \item{score_kappa}{N-vector}
#'   \item{S_mat}{N x D matrix (D = 2P + 1)}
#'   Or NULL if extraction fails.

extract_scores <- function(fit, N, P) {

  D <- 2L * P + 1L

  ## --- score_ext[N, P] ---
  score_ext <- tryCatch({
    raw <- fit$draws("score_ext", format = "draws_matrix")
    pmean <- colMeans(as.matrix(raw))
    rm(raw)  # free memory immediately
    ## Column-major: first index (obs) varies fastest => byrow = FALSE
    matrix(pmean, nrow = N, ncol = P, byrow = FALSE)
  }, error = function(e) {
    cat(sprintf("    [SCORE] score_ext extraction FAILED: %s\n",
                conditionMessage(e)))
    NULL
  })

  ## --- score_int[N, P] ---
  score_int <- tryCatch({
    raw <- fit$draws("score_int", format = "draws_matrix")
    pmean <- colMeans(as.matrix(raw))
    rm(raw)
    matrix(pmean, nrow = N, ncol = P, byrow = FALSE)
  }, error = function(e) {
    cat(sprintf("    [SCORE] score_int extraction FAILED: %s\n",
                conditionMessage(e)))
    NULL
  })

  ## --- score_kappa[N] ---
  score_kappa <- tryCatch({
    raw <- fit$draws("score_kappa", format = "draws_matrix")
    pmean <- colMeans(as.matrix(raw))
    rm(raw)
    as.numeric(pmean)
  }, error = function(e) {
    cat(sprintf("    [SCORE] score_kappa extraction FAILED: %s\n",
                conditionMessage(e)))
    NULL
  })

  ## Check that all three components succeeded
  if (is.null(score_ext) || is.null(score_int) || is.null(score_kappa)) {
    cat("    [SCORE] At least one score component failed. Returning NULL.\n")
    return(NULL)
  }

  ## Dimension validation
  ok <- TRUE
  if (nrow(score_ext) != N || ncol(score_ext) != P) {
    cat(sprintf("    [SCORE] score_ext dim mismatch: %d x %d, expected %d x %d\n",
                nrow(score_ext), ncol(score_ext), N, P))
    ok <- FALSE
  }
  if (nrow(score_int) != N || ncol(score_int) != P) {
    cat(sprintf("    [SCORE] score_int dim mismatch: %d x %d, expected %d x %d\n",
                nrow(score_int), ncol(score_int), N, P))
    ok <- FALSE
  }
  if (length(score_kappa) != N) {
    cat(sprintf("    [SCORE] score_kappa length %d, expected %d\n",
                length(score_kappa), N))
    ok <- FALSE
  }

  if (!ok) {
    cat("    [SCORE] Dimension validation FAILED.\n")
    return(NULL)
  }

  ## Stack into N x D score matrix
  S_mat <- cbind(score_ext, score_int, score_kappa)

  ## Sanity: score column means should be near zero (at the posterior mean,
  ## the score is zero for the MLE; with priors this is approximate)
  col_means <- colMeans(S_mat)
  max_abs <- max(abs(col_means))
  if (max_abs > 2.0) {
    cat(sprintf("    [SCORE] WARNING: max |mean(score)| = %.4f (expected near 0)\n",
                max_abs))
  }

  list(
    score_ext   = score_ext,
    score_int   = score_int,
    score_kappa = score_kappa,
    S_mat       = S_mat
  )
}


###############################################################################
## SECTION 5b : SCORE COMPUTATION IN R (at posterior mean)
###############################################################################
#' Compute Score Vectors in R at Posterior Mean
#'
#' Replaces extract_scores() for cloud deployment: instead of extracting
#' 74,635 score parameters from 8,000 Stan draws (4.6 GB), computes scores
#' at the posterior mean theta_hat directly in R (< 0.6 MB).
#'
#' Mathematical formulas are line-by-line translations from
#' hbb_m1_weighted.stan generated quantities (lines 152-220).
#'
#' Extensive-margin score (all obs):
#'   s_ext[i,] = (z[i] - q_i) * X[i,]
#'   where q_i = plogis(X[i,] %*% alpha + delta_ext[state[i]])
#'
#' Intensive-margin score (z[i]=1 only):
#'   S_BB_mu = kappa * [psi(y+a) - psi(n-y+b) - psi(a) + psi(b)]
#'   Lambda = kappa * [psi(b+n) - psi(b)]
#'   trunc_corr = p0 * Lambda / (1-p0)
#'   s_int[i,] = (S_BB_mu - trunc_corr) * mu*(1-mu) * X[i,]
#'
#' Dispersion score (z[i]=1 only, w.r.t. log_kappa):
#'   S_BB_kappa = kappa * [mu*(psi(y+a)-psi(a)) + (1-mu)*(psi(n-y+b)-psi(b))
#'                + psi(kappa) - psi(n+kappa)]
#'   trunc_corr_kappa = p0*mu*kappa/(1-p0) * sum_{j=1}^{n-1} j/[(b+j)(kappa+j)]
#'   s_kappa[i] = S_BB_kappa - trunc_corr_kappa
#'
#' @param stan_data    List with N, P, S, y, n_trial, z, X, state.
#' @param theta_hat    D-vector: alpha[1:P], beta[1:P], log_kappa.
#' @param delta_means  S x 2 matrix (col1=ext, col2=int random intercepts).
#' @param P            Integer. Number of covariates.
#'
#' @return Same structure as extract_scores():
#'   list(score_ext (NxP), score_int (NxP), score_kappa (N-vector), S_mat (NxD))
#'   Or NULL if computation fails.

compute_scores_in_r <- function(stan_data, theta_hat, delta_means, P) {

  N <- stan_data$N
  D <- 2L * P + 1L

  ## Extract parameter components from theta_hat
  alpha     <- theta_hat[1:P]
  beta_coef <- theta_hat[(P + 1):(2 * P)]
  log_kappa <- theta_hat[2 * P + 1]
  kappa     <- exp(log_kappa)

  ## Data vectors
  y       <- stan_data$y
  n_trial <- stan_data$n_trial
  z       <- stan_data$z
  X       <- stan_data$X
  state   <- stan_data$state

  ## Pre-allocate output matrices
  score_ext   <- matrix(0, N, P)
  score_int   <- matrix(0, N, P)
  score_kappa <- numeric(N)

  ## Vectorized linear predictors (includes random intercepts)
  eta_ext <- as.numeric(X %*% alpha)     + delta_means[state, 1]
  eta_int <- as.numeric(X %*% beta_coef) + delta_means[state, 2]

  q_vec  <- plogis(eta_ext)
  mu_vec <- plogis(eta_int)
  a_vec  <- mu_vec * kappa
  b_vec  <- (1 - mu_vec) * kappa

  ## --- Extensive-margin score (all obs): vectorized ---
  ## s_ext[i,] = (z[i] - q_i) * X[i,]
  score_ext <- (z - q_vec) * X

  ## --- Intensive margin + kappa scores (z=1 only) ---
  pos_idx <- which(z == 1)

  for (i in pos_idx) {
    a_i  <- a_vec[i]
    b_i  <- b_vec[i]
    mu_i <- mu_vec[i]
    n_i  <- n_trial[i]
    y_i  <- y[i]

    ## p0 = P(Y=0 | BetaBin(n, a, b))
    log_p0 <- lgamma(b_i + n_i) + lgamma(kappa) -
              lgamma(b_i) - lgamma(kappa + n_i)
    p0_i <- exp(log_p0)

    ## --- Intensive score ---
    S_BB_mu <- kappa * (digamma(y_i + a_i) - digamma(n_i - y_i + b_i) -
                        digamma(a_i) + digamma(b_i))
    Lambda_i <- kappa * (digamma(b_i + n_i) - digamma(b_i))
    trunc_corr_mu <- p0_i * Lambda_i / (1 - p0_i)
    score_mu_i <- S_BB_mu - trunc_corr_mu  # MINUS sign (corrected)

    score_int[i, ] <- score_mu_i * mu_i * (1 - mu_i) * X[i, ]

    ## --- Kappa score (w.r.t. log_kappa) ---
    S_BB_kappa <- kappa * (
      mu_i * (digamma(y_i + a_i) - digamma(a_i)) +
      (1 - mu_i) * (digamma(n_i - y_i + b_i) - digamma(b_i)) +
      digamma(kappa) - digamma(n_i + kappa)
    )

    trunc_sum <- 0
    if (n_i > 1L) {
      j_seq <- seq_len(n_i - 1L)
      trunc_sum <- sum(j_seq / ((b_i + j_seq) * (kappa + j_seq)))
    }
    trunc_corr_kappa <- p0_i * mu_i * kappa * trunc_sum / (1 - p0_i)

    score_kappa[i] <- S_BB_kappa - trunc_corr_kappa  # MINUS sign (corrected)
  }

  ## --- Assemble N x D stacked score matrix ---
  S_mat <- cbind(score_ext, score_int, score_kappa)

  ## Dimension validation
  ok <- TRUE
  if (nrow(score_ext) != N || ncol(score_ext) != P) {
    cat(sprintf("    [SCORE-R] score_ext dim mismatch: %d x %d, expected %d x %d\n",
                nrow(score_ext), ncol(score_ext), N, P))
    ok <- FALSE
  }
  if (nrow(score_int) != N || ncol(score_int) != P) {
    cat(sprintf("    [SCORE-R] score_int dim mismatch: %d x %d, expected %d x %d\n",
                nrow(score_int), ncol(score_int), N, P))
    ok <- FALSE
  }
  if (length(score_kappa) != N) {
    cat(sprintf("    [SCORE-R] score_kappa length %d, expected %d\n",
                length(score_kappa), N))
    ok <- FALSE
  }
  if (!ok) {
    cat("    [SCORE-R] Dimension validation FAILED.\n")
    return(NULL)
  }

  ## Sanity: score column means should be near zero
  col_means <- colMeans(S_mat)
  max_abs <- max(abs(col_means))
  if (max_abs > 2.0) {
    cat(sprintf("    [SCORE-R] WARNING: max |mean(score)| = %.4f (expected near 0)\n",
                max_abs))
  }

  list(
    score_ext   = score_ext,
    score_int   = score_int,
    score_kappa = score_kappa,
    S_mat       = S_mat
  )
}


###############################################################################
## SECTION 6 : MCMC DIAGNOSTICS
###############################################################################
#' Extract MCMC Convergence Diagnostics from a CmdStanR Fit
#'
#' Collects R-hat, ESS, divergences, and treedepth for target parameters.
#'
#' @param fit            CmdStanMCMC object.
#' @param target_params  Character vector of Stan parameter names to check.
#'
#' @return List with:
#'   \item{rhat_max}{Maximum R-hat}
#'   \item{ess_bulk_min}{Minimum bulk ESS}
#'   \item{ess_tail_min}{Minimum tail ESS}
#'   \item{n_divergent}{Total divergent transitions (all chains)}
#'   \item{frac_divergent}{Fraction of all transitions that diverged}
#'   \item{n_max_treedepth}{Total max-treedepth exceedances}
#'   \item{param_summary}{Summary tibble from CmdStanR}

extract_diagnostics <- function(fit, target_params) {

  ## Parameter-level summaries
  param_summ <- tryCatch(
    fit$summary(variables = target_params),
    error = function(e) {
      warning(sprintf("[DIAG] summary() failed: %s", conditionMessage(e)))
      NULL
    }
  )

  if (is.null(param_summ)) {
    return(list(
      rhat_max        = NA_real_,
      ess_bulk_min    = NA_real_,
      ess_tail_min    = NA_real_,
      n_divergent     = NA_integer_,
      frac_divergent  = NA_real_,
      n_max_treedepth = NA_integer_,
      param_summary   = NULL
    ))
  }

  rhat_max     <- max(param_summ$rhat, na.rm = TRUE)
  ess_bulk_min <- min(param_summ$ess_bulk, na.rm = TRUE)
  ess_tail_min <- min(param_summ$ess_tail, na.rm = TRUE)

  ## Sampler diagnostics
  diag_summ <- tryCatch(
    fit$diagnostic_summary(quiet = TRUE),
    error = function(e) NULL
  )

  if (!is.null(diag_summ)) {
    n_divergent     <- sum(diag_summ$num_divergent)
    n_max_treedepth <- sum(diag_summ$num_max_treedepth)
  } else {
    n_divergent     <- NA_integer_
    n_max_treedepth <- NA_integer_
  }

  ## Total transitions for fraction calculation
  total_trans <- tryCatch({
    md <- fit$metadata()
    md$iter_sampling * length(diag_summ$num_divergent)
  }, error = function(e) NA_real_)

  frac_divergent <- if (!is.na(total_trans) && total_trans > 0 &&
                        !is.na(n_divergent)) {
    n_divergent / total_trans
  } else {
    NA_real_
  }

  list(
    rhat_max        = rhat_max,
    ess_bulk_min    = ess_bulk_min,
    ess_tail_min    = ess_tail_min,
    n_divergent     = as.integer(n_divergent),
    frac_divergent  = frac_divergent,
    n_max_treedepth = as.integer(n_max_treedepth),
    param_summary   = param_summ
  )
}


###############################################################################
## SECTION 7 : CONVERGENCE CHECK
###############################################################################
#' Check Whether MCMC Diagnostics Pass Thresholds
#'
#' Evaluates diagnostics against SIM_CONFIG$mcmc thresholds.
#'
#' @param diagnostics  List from extract_diagnostics().
#' @param config       SIM_CONFIG list.
#'
#' @return Logical. TRUE if all diagnostics pass.

check_convergence <- function(diagnostics, config) {

  mcmc <- config$mcmc
  pass <- TRUE
  msgs <- character(0)

  ## R-hat
  if (!is.na(diagnostics$rhat_max) &&
      diagnostics$rhat_max > mcmc$rhat_threshold) {
    msgs <- c(msgs, sprintf("Rhat=%.4f > %.2f",
                             diagnostics$rhat_max, mcmc$rhat_threshold))
    pass <- FALSE
  }

  ## ESS
  if (!is.na(diagnostics$ess_bulk_min) &&
      diagnostics$ess_bulk_min < mcmc$min_ess_bulk) {
    msgs <- c(msgs, sprintf("ESS_bulk=%.0f < %d",
                             diagnostics$ess_bulk_min, mcmc$min_ess_bulk))
    pass <- FALSE
  }

  ## Divergences
  if (!is.na(diagnostics$frac_divergent) &&
      diagnostics$frac_divergent > mcmc$max_divergent_frac) {
    msgs <- c(msgs, sprintf("div_frac=%.4f > %.4f (%d divergent)",
                             diagnostics$frac_divergent,
                             mcmc$max_divergent_frac,
                             diagnostics$n_divergent))
    pass <- FALSE
  }

  if (!pass && length(msgs) > 0) {
    for (m in msgs) cat(sprintf("      [WARN] %s\n", m))
  }

  pass
}


###############################################################################
## SECTION 8 : FIT UNWEIGHTED MODEL (E-UW)
###############################################################################
#' Fit the Unweighted M1 Hurdle Beta-Binomial
#'
#' Runs CmdStanR sampling on hbb_m1.stan (standard posterior, no weights).
#' Extracts theta, tau, delta, diagnostics.
#'
#' @param stan_data  Stan data list (N, P, S, y, n_trial, z, X, state).
#'                   May contain extra fields (e.g., stratum_idx) which
#'                   are stripped before passing to Stan.
#' @param models     List with $m1_uw (CmdStanModel).
#' @param config     SIM_CONFIG list.
#' @param seed       Integer seed for Stan sampling.
#'
#' @return List with: theta_hat, theta_draws, Sigma_MCMC, D, tau_hat,
#'         tau_draws, delta_means, diagnostics, converged, timing, fit_ok.
#'         Returns a list with fit_ok=FALSE and an error message on failure.

fit_unweighted <- function(stan_data, models, config, seed) {

  cat(sprintf("    [E-UW] Fitting unweighted M1 (seed=%d) ...\n", seed))
  t0 <- proc.time()

  P <- stan_data$P
  S <- stan_data$S
  N <- stan_data$N

  ## Strip to only fields expected by hbb_m1.stan data block
  stan_input <- list(
    N       = N,
    P       = P,
    S       = S,
    y       = stan_data$y,
    n_trial = stan_data$n_trial,
    z       = stan_data$z,
    X       = stan_data$X,
    state   = stan_data$state
  )

  init_fn <- make_init_fun(P, S)
  mcmc    <- config$mcmc

  ## --- Stan sampling ---
  fit <- tryCatch(
    models$m1_uw$sample(
      data            = stan_input,
      seed            = as.integer(seed),
      chains          = mcmc$chains,
      parallel_chains = mcmc$parallel_chains,
      iter_warmup     = mcmc$iter_warmup,
      iter_sampling   = mcmc$iter_sampling,
      adapt_delta     = mcmc$adapt_delta,
      max_treedepth   = mcmc$max_treedepth,
      init            = init_fn,
      refresh         = mcmc$refresh,
      show_messages   = FALSE,
      show_exceptions = FALSE
    ),
    error = function(e) {
      cat(sprintf("    [E-UW] SAMPLING FAILED: %s\n", conditionMessage(e)))
      NULL
    }
  )

  elapsed <- (proc.time() - t0)[["elapsed"]]

  if (is.null(fit)) {
    return(list(fit_ok = FALSE, error = "Stan sampling failed",
                timing = elapsed))
  }

  ## --- Extraction (wrapped in tryCatch for robustness) ---
  result <- tryCatch({

    fixed  <- extract_fixed_effects(fit, P)
    random <- extract_random_effects(fit, S)

    ## Diagnostic parameter names: fixed effects + tau + L_Omega elements
    diag_names <- c(
      paste0("alpha[", seq_len(P), "]"),
      paste0("beta[",  seq_len(P), "]"),
      "log_kappa",
      "tau[1]", "tau[2]"
    )
    diagnostics <- extract_diagnostics(fit, diag_names)
    converged   <- check_convergence(diagnostics, config)

    list(
      fit_ok      = TRUE,
      theta_hat   = fixed$theta_hat,
      theta_draws = fixed$theta_draws,
      Sigma_MCMC  = fixed$Sigma_MCMC,
      D           = fixed$D,
      tau_hat     = random$tau_hat,
      tau_draws   = random$tau_draws,
      delta_means = random$delta_means,
      diagnostics = diagnostics,
      converged   = converged,
      timing      = elapsed,
      seed        = seed
    )
  }, error = function(e) {
    cat(sprintf("    [E-UW] Extraction FAILED: %s\n", conditionMessage(e)))
    list(fit_ok = FALSE, error = conditionMessage(e), timing = elapsed)
  })

  ## Log summary
  if (isTRUE(result$fit_ok)) {
    d <- result$diagnostics
    cat(sprintf("    [E-UW] Done %.1f sec | Rhat=%.3f ESS=%d div=%d %s\n",
                elapsed,
                d$rhat_max, as.integer(d$ess_bulk_min), d$n_divergent,
                ifelse(result$converged, "[OK]", "[WARN]")))
  }

  result
}


###############################################################################
## SECTION 9 : FIT WEIGHTED MODEL (E-WT / E-WS)
###############################################################################
#' Fit the Weighted M1 Hurdle Beta-Binomial (Pseudo-Posterior)
#'
#' Single fit serves both E-WT (naive pseudo-posterior CIs from Sigma_MCMC)
#' and E-WS (sandwich-corrected CIs computed downstream in sim_04).
#'
#' Additionally extracts posterior-mean unweighted score vectors for the
#' cluster-robust sandwich variance estimator.
#'
#' @param stan_data  Stan data list (must include w_tilde).
#' @param models     List with $m1_wt (CmdStanModel).
#' @param config     SIM_CONFIG list.
#' @param seed       Integer seed for Stan sampling.
#'
#' @return List with same fields as fit_unweighted(), plus:
#'   score_ext (N x P), score_int (N x P), score_kappa (N), S_mat (N x D),
#'   scores_ok (logical), timing_fit, timing_scores.
#'   Returns list with fit_ok=FALSE on failure.

fit_weighted <- function(stan_data, models, config, seed) {

  cat(sprintf("    [E-WT] Fitting weighted M1 (seed=%d) ...\n", seed))
  t0 <- proc.time()

  P <- stan_data$P
  S <- stan_data$S
  N <- stan_data$N

  ## Validate w_tilde
  if (is.null(stan_data$w_tilde)) {
    cat("    [E-WT] ERROR: w_tilde is NULL.\n")
    return(list(fit_ok = FALSE, error = "w_tilde is NULL",
                timing = 0))
  }
  if (any(!is.finite(stan_data$w_tilde)) || any(stan_data$w_tilde <= 0)) {
    cat("    [E-WT] ERROR: w_tilde has non-positive or non-finite values.\n")
    return(list(fit_ok = FALSE, error = "w_tilde invalid",
                timing = 0))
  }
  w_sum_diff <- abs(sum(stan_data$w_tilde) - N)
  if (w_sum_diff > 1.0) {
    cat(sprintf("    [E-WT] WARNING: |sum(w_tilde) - N| = %.2f > 1.0\n",
                w_sum_diff))
  }

  ## Strip to fields expected by hbb_m1_weighted.stan data block
  stan_input <- list(
    N       = N,
    P       = P,
    S       = S,
    y       = stan_data$y,
    n_trial = stan_data$n_trial,
    z       = stan_data$z,
    X       = stan_data$X,
    state   = stan_data$state,
    w_tilde = stan_data$w_tilde
  )

  init_fn <- make_init_fun(P, S)
  mcmc    <- config$mcmc

  ## Choose Stan model: prefer noscores variant (no score GQ → smaller CSV)
  stan_model <- if (!is.null(models$m1_wt_noscores)) {
    cat("    [E-WT] Using noscores model variant.\n")
    models$m1_wt_noscores
  } else {
    models$m1_wt
  }

  ## --- Stan sampling ---
  fit <- tryCatch(
    stan_model$sample(
      data            = stan_input,
      seed            = as.integer(seed),
      chains          = mcmc$chains,
      parallel_chains = mcmc$parallel_chains,
      iter_warmup     = mcmc$iter_warmup,
      iter_sampling   = mcmc$iter_sampling,
      adapt_delta     = mcmc$adapt_delta,
      max_treedepth   = mcmc$max_treedepth,
      init            = init_fn,
      refresh         = mcmc$refresh,
      show_messages   = FALSE,
      show_exceptions = FALSE
    ),
    error = function(e) {
      cat(sprintf("    [E-WT] SAMPLING FAILED: %s\n", conditionMessage(e)))
      NULL
    }
  )

  t_fit <- (proc.time() - t0)[["elapsed"]]

  if (is.null(fit)) {
    return(list(fit_ok = FALSE, error = "Stan sampling failed",
                timing = t_fit))
  }

  ## --- Extract theta, tau, delta, diagnostics ---
  result <- tryCatch({

    fixed  <- extract_fixed_effects(fit, P)
    random <- extract_random_effects(fit, S)

    diag_names <- c(
      paste0("alpha[", seq_len(P), "]"),
      paste0("beta[",  seq_len(P), "]"),
      "log_kappa",
      "tau[1]", "tau[2]"
    )
    diagnostics <- extract_diagnostics(fit, diag_names)
    converged   <- check_convergence(diagnostics, config)

    list(
      fit_ok      = TRUE,
      theta_hat   = fixed$theta_hat,
      theta_draws = fixed$theta_draws,
      Sigma_MCMC  = fixed$Sigma_MCMC,
      D           = fixed$D,
      tau_hat     = random$tau_hat,
      tau_draws   = random$tau_draws,
      delta_means = random$delta_means,
      diagnostics = diagnostics,
      converged   = converged
    )
  }, error = function(e) {
    cat(sprintf("    [E-WT] Common extraction FAILED: %s\n", conditionMessage(e)))
    NULL
  })

  if (is.null(result)) {
    return(list(fit_ok = FALSE, error = "Common result extraction failed",
                timing = t_fit))
  }

  ## --- Compute score vectors in R at posterior mean (no Stan GQ needed) ---
  cat("    [E-WT] Computing scores in R at posterior mean ...\n")
  t_score0 <- proc.time()

  scores <- tryCatch(
    compute_scores_in_r(
      stan_data   = stan_input,
      theta_hat   = result$theta_hat,
      delta_means = result$delta_means,
      P           = P
    ),
    error = function(e) {
      cat(sprintf("    [E-WT] R score computation FAILED: %s\n",
                  conditionMessage(e)))
      NULL
    }
  )

  t_score <- (proc.time() - t_score0)[["elapsed"]]
  t_total <- (proc.time() - t0)[["elapsed"]]

  scores_ok <- !is.null(scores)
  if (scores_ok) {
    cat(sprintf("    [E-WT] Scores OK (%.1f sec): S_mat %d x %d\n",
                t_score, nrow(scores$S_mat), ncol(scores$S_mat)))
  } else {
    cat("    [E-WT] WARNING: Score computation failed. Sandwich not possible.\n")
  }

  ## --- Assemble final result ---
  result$score_ext     <- if (scores_ok) scores$score_ext   else NULL
  result$score_int     <- if (scores_ok) scores$score_int   else NULL
  result$score_kappa   <- if (scores_ok) scores$score_kappa else NULL
  result$S_mat         <- if (scores_ok) scores$S_mat       else NULL
  result$scores_ok     <- scores_ok
  result$timing        <- t_total
  result$timing_fit    <- t_fit
  result$timing_scores <- t_score
  result$seed          <- seed

  ## Log summary
  d <- result$diagnostics
  cat(sprintf("    [E-WT] Done %.1f sec (fit=%.1f, score=%.1f) | Rhat=%.3f ESS=%d div=%d %s\n",
              t_total, t_fit, t_score,
              d$rhat_max, as.integer(d$ess_bulk_min), d$n_divergent,
              ifelse(result$converged, "[OK]", "[WARN]")))

  result
}


###############################################################################
## SECTION 10 : FIT ALL ESTIMATORS (MAIN ENTRY POINT)
###############################################################################
#' Fit Both Stan Models for One Simulation Replication
#'
#' Orchestrates the unweighted and weighted fits.
#' E-WT and E-WS share the weighted fit (sandwich CIs in sim_04).
#'
#' @param stan_data_uw  Stan data list for unweighted model.
#' @param stan_data_wt  Stan data list for weighted model (includes w_tilde).
#' @param models        List from compile_stan_models().
#' @param config        SIM_CONFIG list.
#' @param seed          Integer. Base seed for this replication.
#'                      Unweighted uses seed; weighted uses seed + 50000.
#'
#' @return List with:
#'   \item{E_UW}{Result from fit_unweighted()}
#'   \item{E_WT_WS}{Result from fit_weighted()}
#'   \item{seed}{Base seed}
#'   \item{timing_total}{Total wall-clock time for both fits}
#'   \item{both_ok}{Logical. TRUE if both fits have fit_ok=TRUE}

fit_all_estimators <- function(stan_data_uw, stan_data_wt,
                                models, config, seed) {

  cat(sprintf("  [FIT-ALL] Fitting both estimators (seed=%d) ...\n", seed))
  t0 <- proc.time()

  ## Fit 1: Unweighted (E-UW)
  result_uw <- fit_unweighted(stan_data_uw, models, config, seed = seed)

  ## Fit 2: Weighted (E-WT / E-WS)
  ## Offset seed by 50000 to ensure independent chain initialization
  result_wt <- fit_weighted(stan_data_wt, models, config,
                             seed = seed + 50000L)

  t_total <- (proc.time() - t0)[["elapsed"]]

  uw_ok   <- isTRUE(result_uw$fit_ok)
  wt_ok   <- isTRUE(result_wt$fit_ok)
  both_ok <- uw_ok && wt_ok

  cat(sprintf("  [FIT-ALL] Done %.1f sec | E-UW: %s, E-WT/WS: %s\n",
              t_total,
              ifelse(uw_ok, "OK", "FAIL"),
              ifelse(wt_ok, "OK", "FAIL")))

  list(
    E_UW          = result_uw,
    E_WT_WS       = result_wt,
    seed          = seed,
    timing_total  = t_total,
    both_ok       = both_ok
  )
}


###############################################################################
## SECTION 11 : SAVE / LOAD FIT RESULTS
###############################################################################
#' Save Fit Results to Disk
#'
#' Saves the extracted quantities (NOT the full CmdStanR fit object) for
#' one replication to RDS files in the sim_fits directory hierarchy.
#'
#' File layout:
#'   fits/<scenario>/<estimator>/rep_NNN.rds
#'
#' E-WT and E-WS share a single saved file (in E_WT directory).
#'
#' @param fit_results  Output from fit_all_estimators().
#' @param config       SIM_CONFIG list (for paths).
#' @param scenario_id  Character ("S0", "S3", "S4").
#' @param rep_id       Integer (1..R).
#'
#' @return Character vector of saved file paths (invisibly).

save_fit_results <- function(fit_results, config, scenario_id, rep_id) {

  base_dir <- config$paths$sim_fits
  rep_tag  <- sprintf("rep_%03d.rds", rep_id)

  paths_saved <- character(0)

  ## --- E-UW ---
  uw_path <- file.path(base_dir, scenario_id, "E_UW", rep_tag)
  uw_data <- fit_results$E_UW
  saveRDS(uw_data, uw_path)
  paths_saved <- c(paths_saved, uw_path)

  ## --- E-WT/WS (shared) ---
  wt_path <- file.path(base_dir, scenario_id, "E_WT", rep_tag)
  wt_data <- fit_results$E_WT_WS
  saveRDS(wt_data, wt_path)
  paths_saved <- c(paths_saved, wt_path)

  cat(sprintf("    [SAVE] E-UW: %s (%.1f KB)\n",
              uw_path, file.info(uw_path)$size / 1024))
  cat(sprintf("    [SAVE] E-WT: %s (%.1f KB)\n",
              wt_path, file.info(wt_path)$size / 1024))

  invisible(paths_saved)
}


#' Load Fit Results from Disk
#'
#' @param config       SIM_CONFIG list.
#' @param scenario_id  Character ("S0", "S3", "S4").
#' @param rep_id       Integer (1..R).
#'
#' @return List with E_UW and E_WT_WS components (or NULL if file missing).

load_fit_results <- function(config, scenario_id, rep_id) {

  base_dir <- config$paths$sim_fits
  rep_tag  <- sprintf("rep_%03d.rds", rep_id)

  uw_path <- file.path(base_dir, scenario_id, "E_UW", rep_tag)
  wt_path <- file.path(base_dir, scenario_id, "E_WT", rep_tag)

  E_UW    <- if (file.exists(uw_path)) readRDS(uw_path) else NULL
  E_WT_WS <- if (file.exists(wt_path)) readRDS(wt_path) else NULL

  list(E_UW = E_UW, E_WT_WS = E_WT_WS)
}


###############################################################################
## SECTION 12 : PRINT FIT SUMMARY
###############################################################################
#' Print Formatted Fit Summary for One Replication
#'
#' @param fit_results  Output from fit_all_estimators().
#' @param scenario_id  Character.
#' @param rep_id       Integer.

print_fit_summary <- function(fit_results, scenario_id, rep_id) {

  cat(sprintf("\n  --- Fit Summary: %s / Rep %03d ---\n",
              scenario_id, rep_id))
  cat(sprintf("  Seed: %d, Total time: %.1f sec, Both OK: %s\n",
              fit_results$seed, fit_results$timing_total,
              ifelse(fit_results$both_ok, "YES", "NO")))

  ## E-UW
  uw <- fit_results$E_UW
  if (isTRUE(uw$fit_ok)) {
    d <- uw$diagnostics
    cat(sprintf("  E-UW: %.1fs, Rhat=%.3f, ESS=%d, div=%d, conv=%s\n",
                uw$timing, d$rhat_max, as.integer(d$ess_bulk_min),
                d$n_divergent, ifelse(uw$converged, "Y", "N")))
    cat(sprintf("        theta: [%s]\n",
                paste(sprintf("%+.4f", uw$theta_hat[1:min(5, length(uw$theta_hat))]),
                      collapse = ", ")))
  } else {
    cat(sprintf("  E-UW: FAILED (%s)\n",
                ifelse(is.null(uw$error), "unknown", uw$error)))
  }

  ## E-WT
  wt <- fit_results$E_WT_WS
  if (isTRUE(wt$fit_ok)) {
    d <- wt$diagnostics
    cat(sprintf("  E-WT: %.1fs (fit=%.1f, score=%.1f), Rhat=%.3f, ESS=%d, div=%d, conv=%s\n",
                wt$timing, wt$timing_fit, wt$timing_scores,
                d$rhat_max, as.integer(d$ess_bulk_min),
                d$n_divergent, ifelse(wt$converged, "Y", "N")))
    cat(sprintf("        theta: [%s]\n",
                paste(sprintf("%+.4f", wt$theta_hat[1:min(5, length(wt$theta_hat))]),
                      collapse = ", ")))
    if (isTRUE(wt$scores_ok)) {
      cat(sprintf("        S_mat: %d x %d [OK]\n",
                  nrow(wt$S_mat), ncol(wt$S_mat)))
    } else {
      cat("        S_mat: MISSING\n")
    }
  } else {
    cat(sprintf("  E-WT: FAILED (%s)\n",
                ifelse(is.null(wt$error), "unknown", wt$error)))
  }
  cat("\n")
}


###############################################################################
## SECTION 13 : FULL PIPELINE FOR ONE REPLICATION
###############################################################################
#' Run the Complete Fitting Pipeline for One Replication
#'
#' Entry point called from the main simulation loop. Takes a sample result
#' (from draw_sample()), computes the rep seed, fits both models, saves
#' results, and returns the output.
#'
#' @param sample_result  Output from draw_sample() (sim_02_sampling.R).
#'                       Must contain $stan_data_uw and $stan_data_wt.
#' @param models         Compiled Stan models from compile_stan_models().
#' @param config         SIM_CONFIG list.
#' @param scenario_id    Character ("S0", "S3", "S4").
#' @param rep_id         Integer (1..R).
#' @param save_to_disk   Logical. If TRUE, save results to sim_fits/.
#' @param verbose        Logical. If TRUE, print detailed summary.
#'
#' @return Output from fit_all_estimators(), with additional metadata.

run_fitting_pipeline <- function(sample_result, models, config,
                                  scenario_id, rep_id,
                                  save_to_disk = TRUE,
                                  verbose = FALSE) {

  seed <- get_rep_seed(config$seeds$base_seed, rep_id, scenario_id)

  N <- sample_result$stan_data_uw$N
  cat(sprintf("\n  ========== Fitting %s / Rep %03d (seed=%d, N=%d) ==========\n",
              scenario_id, rep_id, seed, N))

  ## Fit both estimators
  fit_results <- fit_all_estimators(
    stan_data_uw = sample_result$stan_data_uw,
    stan_data_wt = sample_result$stan_data_wt,
    models       = models,
    config       = config,
    seed         = seed
  )

  ## Attach metadata
  fit_results$scenario_id <- scenario_id
  fit_results$rep_id      <- rep_id
  fit_results$N           <- N

  ## Save to disk
  if (save_to_disk) {
    save_fit_results(fit_results, config, scenario_id, rep_id)
  }

  ## Verbose summary
  if (verbose) {
    print_fit_summary(fit_results, scenario_id, rep_id)
  }

  ## Garbage collection to free Stan fit memory
  invisible(gc(verbose = FALSE))

  fit_results
}


###############################################################################
## SECTION 14 : STANDALONE EXECUTION
###############################################################################
## When sourced directly (not via a parent script), compile models and
## optionally run a test fit if a sample exists.

if (!isTRUE(.SIM_03_CALLED_FROM_PARENT)) {

  cat("\n--------------------------------------------------------------\n")
  cat("  Standalone execution: testing fit pipeline ...\n")
  cat("--------------------------------------------------------------\n\n")

  pop_path    <- file.path(SIM_CONFIG$paths$sim_population, "pop_base.rds")
  sample_path <- file.path(SIM_CONFIG$paths$sim_samples, "S0", "rep_001.rds")

  if (!file.exists(pop_path)) {
    cat("  [SKIP] Population not found. Run sim_01_dgp.R first.\n")
    cat(sprintf("         Expected: %s\n", pop_path))
  } else if (!file.exists(sample_path)) {
    cat("  [SKIP] Sample not found. Run sim_02_sampling.R first.\n")
    cat(sprintf("         Expected: %s\n", sample_path))
  } else {

    cat("  [TEST] Loading sample ...\n")
    sample_result <- readRDS(sample_path)
    cat(sprintf("  [TEST] N=%d\n", sample_result$stan_data_uw$N))

    ## Compile models
    models <- compile_stan_models(SIM_CONFIG)

    ## Run the full pipeline for one replication
    fit_results <- run_fitting_pipeline(
      sample_result = sample_result,
      models        = models,
      config        = SIM_CONFIG,
      scenario_id   = "S0",
      rep_id        = 1L,
      save_to_disk  = TRUE,
      verbose       = TRUE
    )

    ## Quick validation: compare to DGP truth
    if (fit_results$both_ok) {
      cat("\n  --- Quick Validation ---\n")
      true_theta <- c(SIM_CONFIG$true_params$alpha,
                      SIM_CONFIG$true_params$beta,
                      SIM_CONFIG$true_params$log_kappa)

      uw_bias <- fit_results$E_UW$theta_hat - true_theta
      wt_bias <- fit_results$E_WT_WS$theta_hat - true_theta

      labels <- SIM_CONFIG$param_labels$fixed_labels
      cat(sprintf("  %-25s %10s %10s %10s\n", "Parameter", "Truth", "E-UW", "E-WT"))
      cat(sprintf("  %s\n", paste(rep("-", 57), collapse = "")))
      for (d in seq_along(true_theta)) {
        cat(sprintf("  %-25s %+10.4f %+10.4f %+10.4f\n",
                    labels[d], true_theta[d],
                    fit_results$E_UW$theta_hat[d],
                    fit_results$E_WT_WS$theta_hat[d]))
      }
    }
  }

  cat("\n==============================================================\n")
  cat("  FIT MODULE STANDALONE TEST COMPLETE\n")
  cat("==============================================================\n")
}


###############################################################################
## FINAL SUMMARY
###############################################################################

cat("==============================================================\n")
cat("  FITTING MODULE LOADED (sim_03_fit.R)\n")
cat("==============================================================\n")
cat("\n  Exported functions:\n")
cat("    compile_stan_models(config)          -- compile Stan models (2-3)\n")
cat("    fit_unweighted(stan_data, models, config, seed)\n")
cat("    fit_weighted(stan_data, models, config, seed)\n")
cat("    fit_all_estimators(sd_uw, sd_wt, models, config, seed)\n")
cat("    extract_fixed_effects(fit, P)        -- theta_hat, draws, Sigma\n")
cat("    extract_random_effects(fit, S)       -- tau_hat, tau_draws, delta\n")
cat("    extract_scores(fit, N, P)            -- S_mat for sandwich (Stan GQ)\n")
cat("    compute_scores_in_r(stan_data, theta_hat, delta_means, P) -- R scores\n")
cat("    extract_diagnostics(fit, params)     -- Rhat, ESS, divergences\n")
cat("    check_convergence(diagnostics, config)\n")
cat("    make_init_fun(P, S)\n")
cat("    save_fit_results(fit_results, config, scenario_id, rep_id)\n")
cat("    load_fit_results(config, scenario_id, rep_id)\n")
cat("    print_fit_summary(fit_results, scenario_id, rep_id)\n")
cat("    run_fitting_pipeline(sample_result, models, config, ...)\n")
cat("\n  Typical workflow:\n")
cat("    1. models <- compile_stan_models(SIM_CONFIG)\n")
cat("    2. sample <- readRDS('.../samples/S0/rep_001.rds')\n")
cat("    3. result <- fit_all_estimators(sample$stan_data_uw,\n")
cat("                   sample$stan_data_wt, models, SIM_CONFIG, seed=42)\n")
cat("    4. result$E_UW$theta_hat      # unweighted posterior means\n")
cat("    5. result$E_WT_WS$S_mat       # N x D score matrix for sandwich\n")
cat("    6. result$E_WT_WS$Sigma_MCMC  # naive pseudo-posterior covariance\n")
cat("    7. save_fit_results(result, SIM_CONFIG, 'S0', 1)\n")
cat("==============================================================\n")
