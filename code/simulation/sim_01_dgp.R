## =============================================================================
## sim_01_dgp.R -- Finite Population Generator
## =============================================================================
## Purpose : Generate a finite superpopulation of M=50,000 center-based
##           childcare providers from the Hurdle Beta-Binomial DGP (M1).
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##
## Model (M1: Random Intercepts Hurdle Beta-Binomial):
##   Part 1 (Extensive): z_i ~ Bernoulli(q_i)
##     logit(q_i) = X_i * alpha + delta_ext[state_i]
##   Part 2 (Intensive): y_i | z_i=1 ~ ZT-BetaBin(n_trial_i, mu_i, kappa)
##     logit(mu_i) = X_i * beta + delta_int[state_i]
##   Random effects:
##     (delta_ext[s], delta_int[s])' ~ N(0, diag(tau) * Omega * diag(tau))
##
## Dependency note:
##   This script loads data/precomputed/stan_data.rds for real NSECE covariate
##   distributions. Covariate profiles are bootstrapped from empirical data
##   to preserve non-normal marginals and state-specific clustering.
##
## Inputs:
##   - data/precomputed/simulation/sim_config.rds
##   - data/precomputed/stan_data.rds
##   - data/precomputed/simulation/empirical_calibration.rds
##   - code/helpers/utils.R
##
## Outputs:
##   - data/precomputed/simulation/population/pop_base.rds
##
## Dependencies: MASS (mvrnorm)
## =============================================================================

cat("==============================================================\n")
cat("  Finite Population Generator (sim_01_dgp.R)\n")
cat("==============================================================\n\n")


###############################################################################
## SECTION 0 : PATHS AND DEPENDENCIES
###############################################################################

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()


## Source utility functions (inv_logit, rztbetabinom, etc.)
source(file.path(PROJECT_ROOT, "code/helpers/utils.R"))

## Check MASS availability
if (!requireNamespace("MASS", quietly = TRUE)) {
  stop("Package 'MASS' is required for mvrnorm. Install with: install.packages('MASS')")
}


###############################################################################
## SECTION 1 : MAIN FUNCTION -- generate_finite_population()
###############################################################################
#' Generate a Finite Superpopulation from the Hurdle Beta-Binomial DGP
#'
#' Creates M providers across S=51 states with covariates bootstrapped from
#' the NSECE 2019 empirical data and outcomes generated from the M1 model
#' with known true parameter values.
#'
#' @param config  List. The SIM_CONFIG object from sim_00_config.R.
#' @param seed    Integer or NULL. Random seed for reproducibility.
#'                If NULL, uses config$seeds$population_seed.
#'
#' @return A list with components:
#'   \item{data}{Data frame with M rows: state, X (P cols), n_trial, z, y,
#'               eta_ext, eta_int, q, mu, delta_ext, delta_int}
#'   \item{delta}{Matrix S x 2 of state random intercepts}
#'   \item{config}{The config object used for generation}
#'   \item{seed}{The seed used}
#'   \item{diagnostics}{List of summary statistics for validation}
#'   \item{metadata}{List of generation metadata (timestamp, etc.)}

generate_finite_population <- function(config, seed = NULL) {

  t_start <- proc.time()

  ## -----------------------------------------------------------------------
  ## 0. Setup: seed, parameters, paths
  ## -----------------------------------------------------------------------
  if (is.null(seed)) {
    seed <- config$seeds$population_seed
  }
  set.seed(seed)
  cat(sprintf("  [DGP] Seed: %d\n", seed))

  ## Extract true parameters
  alpha     <- config$true_params$alpha       # length P
  beta      <- config$true_params$beta        # length P
  kappa     <- config$true_params$kappa       # scalar
  tau       <- config$true_params$tau         # length 2
  rho       <- config$true_params$rho         # scalar
  Sigma_delta <- config$true_params$Sigma_delta  # 2x2
  P         <- config$true_params$P           # 5 (incl intercept)
  S         <- config$population$S            # 51
  M         <- config$population$M            # 50000

  cat(sprintf("  [DGP] Population size M=%d, States S=%d, Covariates P=%d\n",
              M, S, P))
  cat(sprintf("  [DGP] kappa=%.4f, tau=(%.4f, %.4f), rho=%.4f\n",
              kappa, tau[1], tau[2], rho))

  ## -----------------------------------------------------------------------
  ## 1. Load empirical data
  ## -----------------------------------------------------------------------
  cat("  [DGP] Loading empirical data ...\n")

  stan_data_path <- config$paths$stan_data
  calib_path     <- file.path(config$paths$sim_output_root,
                              "empirical_calibration.rds")

  if (!file.exists(stan_data_path)) {
    stop(sprintf("stan_data.rds not found at: %s", stan_data_path))
  }
  if (!file.exists(calib_path)) {
    stop(sprintf("empirical_calibration.rds not found at: %s", calib_path))
  }

  stan_data <- readRDS(stan_data_path)
  emp_calib <- readRDS(calib_path)

  ## Validate loaded data
  stopifnot(
    "X"       %in% names(stan_data),
    "n_trial" %in% names(stan_data),
    "state"   %in% names(stan_data),
    ncol(stan_data$X) == P,
    nrow(stan_data$X) == stan_data$N
  )

  N_nsece <- stan_data$N
  X_nsece <- stan_data$X                    # N x P matrix
  n_trial_nsece <- stan_data$n_trial        # length N
  state_nsece   <- stan_data$state          # length N (1..51)

  cat(sprintf("  [DGP] NSECE data loaded: N=%d, P=%d, S=%d\n",
              N_nsece, P, S))

  ## -----------------------------------------------------------------------
  ## 2. State assignment: allocate M providers across S states
  ## -----------------------------------------------------------------------
  cat("  [DGP] Assigning providers to states ...\n")

  ## Use empirical state proportions from calibration
  state_props <- emp_calib$state_props       # length S, sums to 1.0
  stopifnot(length(state_props) == S,
            abs(sum(state_props) - 1.0) < 1e-6)

  ## Compute target state sizes (deterministic rounding to preserve M exactly)
  ## Use largest-remainder method (Hamilton method) for proportional allocation
  raw_sizes <- state_props * M
  floor_sizes <- floor(raw_sizes)
  remainders  <- raw_sizes - floor_sizes
  deficit     <- M - sum(floor_sizes)

  if (deficit > 0) {
    ## Award the extra seats to states with largest remainders
    top_idx <- order(remainders, decreasing = TRUE)[seq_len(deficit)]
    floor_sizes[top_idx] <- floor_sizes[top_idx] + 1L
  }

  M_s <- as.integer(floor_sizes)
  stopifnot(sum(M_s) == M, all(M_s >= 1))

  cat(sprintf("  [DGP] State sizes: min=%d, median=%d, max=%d, total=%d\n",
              min(M_s), as.integer(median(M_s)), max(M_s), sum(M_s)))

  ## -----------------------------------------------------------------------
  ## 3. Covariate generation: bootstrap from NSECE empirical data
  ## -----------------------------------------------------------------------
  cat("  [DGP] Generating covariates via empirical bootstrap ...\n")

  ## Pre-allocate output
  X_pop       <- matrix(NA_real_, nrow = M, ncol = P)
  state_pop   <- integer(M)
  n_trial_pop <- integer(M)

  ## Jitter SD for resampled-with-replacement covariates
  ## Small enough to preserve distribution shape, large enough to break ties
  JITTER_SD <- 0.05

  row_cursor <- 0L

  for (s in seq_len(S)) {

    target_n <- M_s[s]
    if (target_n == 0) next

    ## Find NSECE observations in state s
    idx_s <- which(state_nsece == s)
    nsece_n_s <- length(idx_s)

    if (nsece_n_s == 0) {
      ## Fallback: if a state has no NSECE observations (shouldn't happen),
      ## sample from the full NSECE dataset
      warning(sprintf("State %d has 0 NSECE observations; sampling from full dataset.", s))
      idx_s <- seq_len(N_nsece)
      nsece_n_s <- N_nsece
    }

    ## Extract state-level NSECE covariates and n_trial
    X_state <- X_nsece[idx_s, , drop = FALSE]           # nsece_n_s x P
    n_trial_state <- n_trial_nsece[idx_s]                # nsece_n_s

    ## Decide sampling strategy
    if (target_n <= nsece_n_s) {
      ## Subsample without replacement: preserves exact empirical distribution
      sample_idx <- sample.int(nsece_n_s, size = target_n, replace = FALSE)
      X_sampled <- X_state[sample_idx, , drop = FALSE]
      n_trial_sampled <- n_trial_state[sample_idx]
    } else {
      ## Oversample with replacement + jitter on non-intercept covariates
      sample_idx <- sample.int(nsece_n_s, size = target_n, replace = TRUE)
      X_sampled <- X_state[sample_idx, , drop = FALSE]
      n_trial_sampled <- n_trial_state[sample_idx]

      ## Add small jitter to continuous covariates (columns 2:P) to reduce
      ## exact ties. Column 1 is the intercept (always 1).
      jitter_matrix <- matrix(
        rnorm(target_n * (P - 1L), mean = 0, sd = JITTER_SD),
        nrow = target_n, ncol = P - 1L
      )
      X_sampled[, 2:P] <- X_sampled[, 2:P] + jitter_matrix
    }

    ## Also resample n_trial with replacement (independent of covariates)
    ## to provide additional variety in trial sizes
    n_trial_sampled <- sample(n_trial_state, size = target_n, replace = TRUE)

    ## Fill population arrays
    rows <- (row_cursor + 1L):(row_cursor + target_n)
    X_pop[rows, ]     <- X_sampled
    state_pop[rows]   <- s
    n_trial_pop[rows] <- n_trial_sampled
    row_cursor <- row_cursor + target_n
  }

  stopifnot(row_cursor == M)

  ## Ensure intercept column is exactly 1
  X_pop[, 1] <- 1.0

  ## Ensure n_trial >= 1 (should already be, but safeguard)
  n_trial_pop <- pmax(n_trial_pop, 1L)

  ## Report covariate summary
  cov_names <- c("intercept", "poverty", "urban", "black", "hispanic")
  cat("  [DGP] Population covariate summary (cols 2:5):\n")
  for (k in 2:P) {
    xk <- X_pop[, k]
    cat(sprintf("    %-10s: mean=%+.4f, sd=%.4f, skew=%.2f, range=[%.2f, %.2f]\n",
                cov_names[k], mean(xk), sd(xk),
                mean((xk - mean(xk))^3) / sd(xk)^3,
                min(xk), max(xk)))
  }
  cat(sprintf("  [DGP] n_trial: median=%d, mean=%.1f, range=[%d, %d]\n",
              median(n_trial_pop), mean(n_trial_pop),
              min(n_trial_pop), max(n_trial_pop)))

  ## -----------------------------------------------------------------------
  ## 4. Generate state random intercepts
  ## -----------------------------------------------------------------------
  cat("  [DGP] Generating state random intercepts ...\n")

  ## delta[s, ] = (delta_ext[s], delta_int[s]) ~ N(0, Sigma_delta)
  delta <- MASS::mvrnorm(n = S, mu = c(0, 0), Sigma = Sigma_delta)
  ## delta is S x 2 matrix: col 1 = extensive, col 2 = intensive

  cat(sprintf("  [DGP] delta_ext: mean=%.4f, sd=%.4f, range=[%.3f, %.3f]\n",
              mean(delta[, 1]), sd(delta[, 1]),
              min(delta[, 1]), max(delta[, 1])))
  cat(sprintf("  [DGP] delta_int: mean=%.4f, sd=%.4f, range=[%.3f, %.3f]\n",
              mean(delta[, 2]), sd(delta[, 2]),
              min(delta[, 2]), max(delta[, 2])))
  cat(sprintf("  [DGP] delta cor: %.4f (true rho=%.4f)\n",
              cor(delta[, 1], delta[, 2]), rho))

  ## -----------------------------------------------------------------------
  ## 5. Compute linear predictors and probabilities
  ## -----------------------------------------------------------------------
  cat("  [DGP] Computing linear predictors ...\n")

  ## Extensive margin: logit(q_i) = X_i * alpha + delta_ext[state_i]
  eta_ext <- as.numeric(X_pop %*% alpha) + delta[state_pop, 1]
  q_pop   <- inv_logit(eta_ext)

  ## Intensive margin: logit(mu_i) = X_i * beta + delta_int[state_i]
  eta_int <- as.numeric(X_pop %*% beta) + delta[state_pop, 2]
  mu_pop  <- inv_logit(eta_int)

  cat(sprintf("  [DGP] q (participation prob): mean=%.4f, range=[%.4f, %.4f]\n",
              mean(q_pop), min(q_pop), max(q_pop)))
  cat(sprintf("  [DGP] mu (intensity):         mean=%.4f, range=[%.4f, %.4f]\n",
              mean(mu_pop), min(mu_pop), max(mu_pop)))

  ## -----------------------------------------------------------------------
  ## 6. Generate outcomes
  ## -----------------------------------------------------------------------
  cat("  [DGP] Generating outcomes ...\n")

  ## Part 1: z_i ~ Bernoulli(q_i)
  z_pop <- rbinom(M, size = 1, prob = q_pop)

  zero_rate <- 1 - mean(z_pop)
  n_servers <- sum(z_pop == 1)
  cat(sprintf("  [DGP] Part 1: zero_rate=%.4f (target ~0.353), servers=%d/%d\n",
              zero_rate, n_servers, M))

  ## Part 2: y_i | z_i=1 ~ ZT-BetaBin(n_trial_i, mu_i, kappa)
  ## For z_i=0: y_i = 0 (structural zero)
  y_pop <- integer(M)

  ## Identify servers for ZT-BB sampling
  server_idx <- which(z_pop == 1)

  if (length(server_idx) > 0) {
    cat(sprintf("  [DGP] Drawing ZT-BetaBin for %d servers (rejection sampling) ...\n",
                length(server_idx)))

    ## Progress tracking for large draws
    n_servers_total <- length(server_idx)
    report_every <- max(1L, floor(n_servers_total / 10))

    for (ii in seq_along(server_idx)) {
      i <- server_idx[ii]
      y_pop[i] <- rztbetabinom(
        n_draws = 1L,
        size    = n_trial_pop[i],
        mu      = mu_pop[i],
        kappa   = kappa
      )

      ## Progress report every 10%
      if (ii %% report_every == 0) {
        pct <- round(100 * ii / n_servers_total)
        cat(sprintf("\r  [DGP]   Progress: %d%% (%d/%d)", pct, ii, n_servers_total))
      }
    }
    cat("\n")
  }

  ## Sanity: all z=0 should have y=0; all z=1 should have y>0
  stopifnot(all(y_pop[z_pop == 0] == 0))
  stopifnot(all(y_pop[z_pop == 1] > 0))

  ## Compute IT share among servers
  it_share_servers <- y_pop[server_idx] / n_trial_pop[server_idx]
  mean_it_share <- mean(it_share_servers)

  cat(sprintf("  [DGP] Part 2: mean IT share (servers)=%.4f (target ~0.478)\n",
              mean_it_share))
  cat(sprintf("  [DGP]         y range (servers): [%d, %d], median=%d\n",
              min(y_pop[server_idx]), max(y_pop[server_idx]),
              as.integer(median(y_pop[server_idx]))))

  ## -----------------------------------------------------------------------
  ## 7. Assemble population data frame
  ## -----------------------------------------------------------------------
  cat("  [DGP] Assembling population data frame ...\n")

  pop_df <- data.frame(
    provider_id = seq_len(M),
    state       = state_pop,
    n_trial     = n_trial_pop,
    z           = z_pop,
    y           = y_pop,
    eta_ext     = eta_ext,
    eta_int     = eta_int,
    q           = q_pop,
    mu          = mu_pop,
    delta_ext   = delta[state_pop, 1],
    delta_int   = delta[state_pop, 2]
  )

  ## Attach covariate columns (named)
  for (k in seq_len(P)) {
    pop_df[[cov_names[k]]] <- X_pop[, k]
  }

  ## -----------------------------------------------------------------------
  ## 8. Compute diagnostics
  ## -----------------------------------------------------------------------
  cat("  [DGP] Computing diagnostics ...\n")

  ## Per-state summaries
  state_summary <- data.frame(
    state       = seq_len(S),
    M_s         = M_s,
    zero_rate   = tapply(z_pop, state_pop, function(z) 1 - mean(z)),
    mean_q      = tapply(q_pop, state_pop, mean),
    mean_mu     = tapply(mu_pop, state_pop, mean),
    delta_ext   = delta[, 1],
    delta_int   = delta[, 2],
    stringsAsFactors = FALSE
  )

  ## Compute mean IT share per state (among servers only)
  it_share_by_state <- tapply(
    seq_len(M),
    state_pop,
    function(idx) {
      srv <- idx[z_pop[idx] == 1]
      if (length(srv) == 0) return(NA_real_)
      mean(y_pop[srv] / n_trial_pop[srv])
    }
  )
  state_summary$mean_it_share <- as.numeric(it_share_by_state)

  diagnostics <- list(
    M              = M,
    S              = S,
    P              = P,
    zero_rate      = zero_rate,
    mean_it_share  = mean_it_share,
    n_servers      = n_servers,
    mean_q         = mean(q_pop),
    mean_mu        = mean(mu_pop),
    n_trial_mean   = mean(n_trial_pop),
    n_trial_median = median(n_trial_pop),
    cov_means      = colMeans(X_pop[, 2:P]),
    cov_sds        = apply(X_pop[, 2:P], 2, sd),
    delta_empirical_cor = cor(delta[, 1], delta[, 2]),
    delta_empirical_sd  = apply(delta, 2, sd),
    state_summary  = state_summary
  )
  names(diagnostics$cov_means) <- cov_names[2:P]
  names(diagnostics$cov_sds)   <- cov_names[2:P]

  t_elapsed <- (proc.time() - t_start)["elapsed"]

  ## -----------------------------------------------------------------------
  ## 9. Package result
  ## -----------------------------------------------------------------------
  result <- list(
    data        = pop_df,
    X           = X_pop,
    delta       = delta,
    config      = config,
    seed        = seed,
    diagnostics = diagnostics,
    metadata    = list(
      timestamp    = Sys.time(),
      elapsed_sec  = as.numeric(t_elapsed),
      R_version    = paste(R.version$major, R.version$minor, sep = "."),
      generator    = "sim_01_dgp.R::generate_finite_population",
      description  = paste(
        "Finite superpopulation from Hurdle Beta-Binomial M1 DGP.",
        "Covariates bootstrapped from NSECE 2019 empirical data.",
        sprintf("M=%d, S=%d, P=%d, kappa=%.4f", M, S, P, kappa)
      )
    )
  )

  cat(sprintf("\n  [DGP] Population generated in %.1f seconds.\n", t_elapsed))

  result
}


###############################################################################
## SECTION 2 : VALIDATION FUNCTION
###############################################################################
#' Validate a Generated Finite Population
#'
#' Checks that the population conforms to expected dimensions, parameter
#' ranges, and approximate calibration targets from the NSECE 2019 data.
#'
#' @param pop    List. Output of generate_finite_population().
#' @param config List. SIM_CONFIG object.
#'
#' @return TRUE (invisibly) if all checks pass; stops with error otherwise.

validate_population <- function(pop, config) {

  cat("  [VALIDATE] Running population validation checks ...\n")

  checks_passed <- 0L
  checks_total  <- 0L
  warnings_list <- character(0)

  check_hard <- function(condition, msg) {
    checks_total <<- checks_total + 1L
    if (condition) {
      checks_passed <<- checks_passed + 1L
    } else {
      stop(sprintf("[VALIDATE] HARD FAIL: %s", msg), call. = FALSE)
    }
  }

  check_soft <- function(condition, msg) {
    checks_total <<- checks_total + 1L
    if (condition) {
      checks_passed <<- checks_passed + 1L
    } else {
      warnings_list <<- c(warnings_list, msg)
      cat(sprintf("    [WARN] %s\n", msg))
    }
  }

  df    <- pop$data
  M     <- config$population$M
  S     <- config$population$S
  P     <- config$true_params$P
  nsece <- config$nsece_reference

  ## --- Hard checks (must pass) ---

  ## Dimensions
  check_hard(nrow(df) == M,
             sprintf("Row count %d != M=%d", nrow(df), M))
  check_hard(ncol(pop$X) == P,
             sprintf("X has %d cols, expected P=%d", ncol(pop$X), P))
  check_hard(nrow(pop$X) == M,
             sprintf("X has %d rows, expected M=%d", nrow(pop$X), M))
  check_hard(nrow(pop$delta) == S,
             sprintf("delta has %d rows, expected S=%d", nrow(pop$delta), S))
  check_hard(ncol(pop$delta) == 2,
             sprintf("delta has %d cols, expected 2", ncol(pop$delta)))

  ## State coverage
  check_hard(length(unique(df$state)) == S,
             sprintf("Found %d unique states, expected S=%d",
                     length(unique(df$state)), S))
  check_hard(all(df$state >= 1 & df$state <= S),
             "State indices must be in [1, S]")

  ## Outcome validity
  check_hard(all(df$z %in% c(0L, 1L)),
             "z must be 0 or 1")
  check_hard(all(df$y[df$z == 0] == 0),
             "y must be 0 when z=0")
  check_hard(all(df$y[df$z == 1] > 0),
             "y must be > 0 when z=1")
  check_hard(all(df$y >= 0 & df$y <= df$n_trial),
             "y must be in [0, n_trial]")
  check_hard(all(df$n_trial >= 1),
             "n_trial must be >= 1")

  ## Probabilities in valid range
  check_hard(all(df$q >= 0 & df$q <= 1),
             "q must be in [0, 1]")
  check_hard(all(df$mu > 0 & df$mu < 1),
             "mu must be in (0, 1)")

  ## Intercept column
  check_hard(all(pop$X[, 1] == 1.0),
             "Intercept column must be all 1.0")

  ## --- Soft checks (calibration targets, allow tolerance) ---

  ## Zero rate: should be roughly 30-40% (NSECE is 35.3%)
  obs_zero_rate <- 1 - mean(df$z)
  check_soft(abs(obs_zero_rate - nsece$zero_rate) < 0.10,
             sprintf("zero_rate=%.4f differs from NSECE target %.4f by >0.10",
                     obs_zero_rate, nsece$zero_rate))

  ## Mean IT share among servers: should be roughly 40-60% (NSECE is 47.8%)
  server_idx <- which(df$z == 1)
  if (length(server_idx) > 0) {
    obs_it_share <- mean(df$y[server_idx] / df$n_trial[server_idx])
    check_soft(abs(obs_it_share - nsece$mean_it_share) < 0.10,
               sprintf("mean_it_share=%.4f differs from NSECE target %.4f by >0.10",
                       obs_it_share, nsece$mean_it_share))
  }

  ## n_trial median: should be within 20 of NSECE median (48)
  check_soft(abs(median(df$n_trial) - nsece$n_trial_median) < 20,
             sprintf("n_trial median=%d differs from NSECE %d by >20",
                     median(df$n_trial), nsece$n_trial_median))

  ## Covariate means near 0 (standardized)
  for (k in 2:P) {
    cov_mean <- mean(pop$X[, k])
    cov_name <- c("intercept", "poverty", "urban", "black", "hispanic")[k]
    check_soft(abs(cov_mean) < 0.15,
               sprintf("Covariate %s mean=%.4f (expected near 0)", cov_name, cov_mean))
  }

  ## State sizes: no extremely empty or dominant states
  state_sizes_obs <- table(df$state)
  check_soft(min(state_sizes_obs) >= 10,
             sprintf("Smallest state has %d providers (expected >=10)",
                     min(state_sizes_obs)))

  ## --- Summary ---
  if (length(warnings_list) > 0) {
    cat(sprintf("  [VALIDATE] %d/%d checks passed, %d soft warnings.\n",
                checks_passed, checks_total, length(warnings_list)))
  } else {
    cat(sprintf("  [VALIDATE] All %d/%d checks passed.\n",
                checks_passed, checks_total))
  }

  invisible(TRUE)
}


###############################################################################
## SECTION 3 : SUMMARY FUNCTION
###############################################################################
#' Summarize a Generated Finite Population
#'
#' Prints a detailed overview of the generated population including
#' covariate distributions, outcome statistics, and per-state summaries.
#'
#' @param pop List. Output of generate_finite_population().
#'
#' @return NULL (invisibly). Prints summary to console.

summarize_population <- function(pop) {

  df   <- pop$data
  diag <- pop$diagnostics
  meta <- pop$metadata
  X    <- pop$X
  P    <- diag$P

  cov_names <- c("intercept", "poverty", "urban", "black", "hispanic")

  cat("\n================================================================\n")
  cat("  FINITE POPULATION SUMMARY\n")
  cat("================================================================\n")

  ## --- Metadata ---
  cat(sprintf("\n  Generator:   %s\n", meta$generator))
  cat(sprintf("  Timestamp:   %s\n", format(meta$timestamp, "%Y-%m-%d %H:%M:%S")))
  cat(sprintf("  Seed:        %d\n", pop$seed))
  cat(sprintf("  Elapsed:     %.1f seconds\n", meta$elapsed_sec))
  cat(sprintf("  R version:   %s\n", meta$R_version))

  ## --- Dimensions ---
  cat(sprintf("\n  --- Dimensions ---\n"))
  cat(sprintf("  M = %d providers\n", diag$M))
  cat(sprintf("  S = %d states\n", diag$S))
  cat(sprintf("  P = %d covariates (incl. intercept)\n", diag$P))

  ## --- Outcome statistics ---
  cat(sprintf("\n  --- Outcomes ---\n"))
  cat(sprintf("  Zero rate (1 - mean(z)):    %.4f\n", diag$zero_rate))
  cat(sprintf("  Number of IT servers:       %d (%.1f%%)\n",
              diag$n_servers, 100 * diag$n_servers / diag$M))
  cat(sprintf("  Mean IT share (servers):    %.4f\n", diag$mean_it_share))
  cat(sprintf("  Mean q (participation):     %.4f\n", diag$mean_q))
  cat(sprintf("  Mean mu (intensity):        %.4f\n", diag$mean_mu))

  ## --- n_trial ---
  cat(sprintf("\n  --- n_trial (total 0-5 enrollment) ---\n"))
  cat(sprintf("  Mean:   %.1f\n", diag$n_trial_mean))
  cat(sprintf("  Median: %d\n", diag$n_trial_median))
  cat(sprintf("  Range:  [%d, %d]\n", min(df$n_trial), max(df$n_trial)))

  ## --- Covariates ---
  cat(sprintf("\n  --- Covariates (excluding intercept) ---\n"))
  cat(sprintf("  %-10s %8s %8s %8s %10s %10s\n",
              "Name", "Mean", "SD", "Skew", "Min", "Max"))
  cat(sprintf("  %s\n", paste(rep("-", 60), collapse = "")))
  for (k in 2:P) {
    xk <- X[, k]
    skew_k <- mean((xk - mean(xk))^3) / sd(xk)^3
    cat(sprintf("  %-10s %+8.4f %8.4f %8.2f %10.3f %10.3f\n",
                cov_names[k], mean(xk), sd(xk), skew_k, min(xk), max(xk)))
  }

  ## Covariate correlation matrix
  cat(sprintf("\n  Covariate correlation matrix (cols 2:%d):\n", P))
  cor_mat <- cor(X[, 2:P])
  rownames(cor_mat) <- cov_names[2:P]
  colnames(cor_mat) <- cov_names[2:P]
  print(round(cor_mat, 3))

  ## --- Random intercepts ---
  cat(sprintf("\n  --- Random Intercepts (delta) ---\n"))
  cat(sprintf("  delta_ext: mean=%+.4f, sd=%.4f, range=[%.3f, %.3f]\n",
              mean(pop$delta[, 1]), sd(pop$delta[, 1]),
              min(pop$delta[, 1]), max(pop$delta[, 1])))
  cat(sprintf("  delta_int: mean=%+.4f, sd=%.4f, range=[%.3f, %.3f]\n",
              mean(pop$delta[, 2]), sd(pop$delta[, 2]),
              min(pop$delta[, 2]), max(pop$delta[, 2])))
  cat(sprintf("  Empirical cor(delta_ext, delta_int): %.4f\n",
              diag$delta_empirical_cor))

  ## --- Per-state summary (top/bottom 5) ---
  ss <- diag$state_summary
  ss <- ss[order(ss$zero_rate), ]

  cat(sprintf("\n  --- Per-State Summary (sorted by zero rate) ---\n"))
  cat(sprintf("  %-6s %8s %10s %8s %12s\n",
              "State", "M_s", "Zero Rate", "Mean q", "IT Share"))
  cat(sprintf("  %s\n", paste(rep("-", 50), collapse = "")))

  ## Top 5 (lowest zero rate)
  for (i in 1:min(5, nrow(ss))) {
    cat(sprintf("  %-6d %8d %10.4f %8.4f %12.4f\n",
                ss$state[i], ss$M_s[i], ss$zero_rate[i],
                ss$mean_q[i],
                ifelse(is.na(ss$mean_it_share[i]), NA, ss$mean_it_share[i])))
  }
  cat("  ...\n")

  ## Bottom 5 (highest zero rate)
  n_ss <- nrow(ss)
  for (i in max(1, n_ss - 4):n_ss) {
    cat(sprintf("  %-6d %8d %10.4f %8.4f %12.4f\n",
                ss$state[i], ss$M_s[i], ss$zero_rate[i],
                ss$mean_q[i],
                ifelse(is.na(ss$mean_it_share[i]), NA, ss$mean_it_share[i])))
  }

  ## --- NSECE comparison ---
  nsece <- pop$config$nsece_reference
  cat(sprintf("\n  --- Calibration vs NSECE 2019 ---\n"))
  cat(sprintf("  %-25s %12s %12s\n", "Statistic", "Population", "NSECE"))
  cat(sprintf("  %s\n", paste(rep("-", 52), collapse = "")))
  cat(sprintf("  %-25s %12.4f %12.4f\n", "Zero rate",
              diag$zero_rate, nsece$zero_rate))
  cat(sprintf("  %-25s %12.4f %12.4f\n", "Mean IT share (servers)",
              diag$mean_it_share, nsece$mean_it_share))
  cat(sprintf("  %-25s %12.1f %12.1f\n", "n_trial mean",
              diag$n_trial_mean, nsece$n_trial_mean))
  cat(sprintf("  %-25s %12d %12d\n", "n_trial median",
              diag$n_trial_median, nsece$n_trial_median))

  cat("\n================================================================\n")
  cat("  END OF POPULATION SUMMARY\n")
  cat("================================================================\n\n")

  invisible(NULL)
}


###############################################################################
## SECTION 4 : STANDALONE EXECUTION
###############################################################################
## When sourced directly (not via another script), generate and save the
## population, then run validation and summary.

## Detect standalone execution: if SIM_CONFIG doesn't already exist in the
## calling environment, source the config file first.

if (!exists("SIM_CONFIG") || !is.list(SIM_CONFIG)) {
  cat("  [DGP] SIM_CONFIG not found; sourcing sim_00_config.R ...\n")
  source(file.path(PROJECT_ROOT, "code/simulation/sim_00_config.R"))
}

## Only run standalone block when this script is the top-level source
## (i.e., not being sourced by another simulation script that sets
## .SIM_01_CALLED_FROM_PARENT = TRUE)
if (!isTRUE(.SIM_01_CALLED_FROM_PARENT)) {

  cat("\n--------------------------------------------------------------\n")
  cat("  Standalone execution: generating population ...\n")
  cat("--------------------------------------------------------------\n\n")

  ## Generate population
  pop <- generate_finite_population(SIM_CONFIG)

  ## Validate
  validate_population(pop, SIM_CONFIG)

  ## Print summary
  summarize_population(pop)

  ## Save
  out_path <- file.path(SIM_CONFIG$paths$sim_population, "pop_base.rds")
  saveRDS(pop, out_path)
  cat(sprintf("  [DGP] Population saved: %s\n", out_path))
  cat(sprintf("         File size: %.1f MB\n",
              file.info(out_path)$size / 1024^2))

  cat("\n==============================================================\n")
  cat("  POPULATION GENERATION COMPLETE\n")
  cat(sprintf("  Output: %s\n", out_path))
  cat(sprintf("  Next step: source('code/simulation/sim_02_sampling.R')\n"))
  cat("==============================================================\n")
}
