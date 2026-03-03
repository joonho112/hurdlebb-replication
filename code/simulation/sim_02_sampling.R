## =============================================================================
## sim_02_sampling.R -- Poisson PPS Survey Sampling
## =============================================================================
## Purpose : Draw informative survey samples from the simulated population
##           using Poisson PPS sampling, compute survey weights, create
##           synthetic stratum/PSU structure for the sandwich estimator,
##           and prepare Stan data lists for model fitting.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##
## Sampling Approach: Direct Poisson PPS (probability proportional to size)
##   For each population unit i (i = 1, ..., M):
##     logit(pi_i) = gamma_0 + rho * y_star_i + x_design_i
##   where
##     y_star_i = standardized outcome (informativeness channel)
##     x_design_i = design-based variation (state size)
##     gamma_0 = calibrated intercept so that sum(pi_i) = N_target
##
##   rho controls informativeness:
##     rho = 0.00  (S0) : non-informative, CV(w) ~ 1.0,  DEFF ~ 2.0
##     rho = 0.15  (S3) : NSECE-calibrated, CV(w) ~ 1.67, DEFF ~ 3.79
##     rho = 0.50  (S4) : stress test, CV(w) ~ 2.0,  DEFF ~ 5.0
##
## Design Decisions:
##   - Poisson PPS rather than stratified cluster sampling for simplicity
##     and speed (called R=200 times per scenario).
##   - Inclusion probabilities capped at [0.02, 0.999] to ensure all
##     units have positive finite weights (max weight = 50).
##   - Normalized weights: w_tilde = w * N / sum(w) so sum(w_tilde) = N.
##   - Synthetic stratum/PSU structure assigned post-sampling for sandwich.
##   - x_design based on log(state_size) to create design-driven weight
##     variation even when rho = 0 (pure non-informative case).
##   - alpha_design (scale of x_design) is calibrated jointly with gamma_0
##     to match target CV(w) for each scenario.
##
## Population data frame (from sim_01_dgp.R) columns:
##   provider_id, state, n_trial, z, y, eta_ext, eta_int, q, mu,
##   delta_ext, delta_int, intercept, poverty, urban, black, hispanic
##
## Dependencies:
##   - sim_00_config.R : SIM_CONFIG, get_rep_seed()
##   - sim_01_dgp.R    : population data (pop_base.rds)
##   - utils_helpers.R : normalize_weights(), inv_logit(), logit()
##
## Usage : source("code/simulation/sim_02_sampling.R")
##         (from project root <PROJECT_ROOT>)
##
## Exports:
##   - draw_sample()              : main sampling function
##   - calibrate_inclusion()      : calibrate gamma_0 + alpha_design
##   - calibrate_all_scenarios()  : calibrate all 3 scenarios
##   - create_survey_structure()  : assign stratum/PSU to sample
##   - prepare_stan_data()        : build Stan data list from sample
##   - compute_weight_diagnostics() : weight distribution summary
##   - run_calibration_check()    : verify calibration matches targets
##   - run_sampling_pipeline()    : full pipeline for one rep
##   - print_sampling_summary()   : formatted summary
###############################################################################

cat("==============================================================\n")
cat("  Sampling Module  (sim_02_sampling.R)\n")
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

## Source utility helpers
source(file.path(PROJECT_ROOT, "code/helpers/utils.R"))

## Verify required functions exist
stopifnot(
  "normalize_weights not found" = exists("normalize_weights"),
  "inv_logit not found" = exists("inv_logit") || exists("plogis"),
  "SIM_CONFIG not found" = exists("SIM_CONFIG")
)

cat("  Dependencies loaded.\n\n")


###############################################################################
## SECTION 1 : COMPUTE STANDARDIZED OUTCOME (y_star)
###############################################################################
## y_star captures the outcome-informativeness channel. It is a continuous
## standardized measure of the outcome that determines how inclusion
## probability depends on the response variable.
##
## For z = 0 (non-servers): the "outcome" is 0 (no IT enrollment)
## For z = 1 (servers): the "outcome" is y_i / n_trial_i (IT share)
##
## We then standardize to mean 0, sd 1 across the full population.

compute_y_star <- function(pop) {
  ## Raw outcome: proportion for servers, 0 for non-servers
  raw_outcome <- ifelse(pop$z == 1, pop$y / pop$n_trial, 0)

  ## Standardize to mean 0, sd 1
  mu_raw <- mean(raw_outcome)
  sd_raw <- sd(raw_outcome)

  if (sd_raw < 1e-10) {
    warning("y_star has near-zero variance; returning zeros.")
    return(rep(0, length(raw_outcome)))
  }

  (raw_outcome - mu_raw) / sd_raw
}


###############################################################################
## SECTION 2 : COMPUTE DESIGN-BASED VARIATION (x_design)
###############################################################################
## x_design captures design-driven variation in inclusion probabilities
## that is NOT related to the outcome. In real surveys, larger states tend
## to have lower per-unit selection probabilities (because the same number
## of PSUs are sampled from a larger pool).
##
## x_design_i = alpha_design * (log(M_state[i]) - mean(log(M_state)))
##
## alpha_design is a scaling parameter calibrated to achieve the target
## CV(w) for each scenario. Larger alpha_design => more design-driven
## weight variation.
##
## Key insight: For S0 (rho=0), ALL weight variation comes from x_design.
## For S3/S4, weight variation is a combination of x_design and rho * y_star.

compute_x_design <- function(pop, alpha_design) {
  ## Tabulate state sizes in the population
  ## Population data frame uses column name "state" (integer 1:51)
  state_sizes <- table(pop$state)

  ## Map each unit to its state size
  unit_state_size <- as.numeric(state_sizes[as.character(pop$state)])

  ## Log state size, centered
  log_ss <- log(unit_state_size)
  log_ss_centered <- log_ss - mean(log_ss)

  ## Scale by alpha_design
  alpha_design * log_ss_centered
}


###############################################################################
## SECTION 3 : CALIBRATE INCLUSION PROBABILITIES
###############################################################################
## Two-stage calibration:
##   Stage 1: For a given alpha_design and rho, find gamma_0 so that
##            sum(pi_i) = N_target via uniroot.
##   Stage 2: Search over alpha_design to match the target CV(w).
##
## The calibration is done on the FULL population (M = 50,000) once per
## scenario. The calibrated parameters (gamma_0, alpha_design) are then
## re-used across all R = 200 replications.

calibrate_gamma0 <- function(rho, y_star, x_design, N_target, tol = 0.5) {
  ## Objective: sum(plogis(gamma0 + rho * y_star + x_design)) - N_target = 0
  f <- function(gamma0) {
    sum(plogis(gamma0 + rho * y_star + x_design)) - N_target
  }

  ## Check that the problem is solvable
  f_lo <- f(-15)
  f_hi <- f(15)

  if (f_lo > 0) {
    warning("Even gamma_0 = -15 gives sum(pi) > N_target; ",
            "the design variation is too large. Returning gamma_0 = -15.")
    return(-15)
  }
  if (f_hi < 0) {
    warning("Even gamma_0 = +15 gives sum(pi) < N_target; ",
            "N_target may be too large relative to M. Returning gamma_0 = +15.")
    return(15)
  }

  uniroot(f, interval = c(-15, 15), tol = tol * 0.01)$root
}


calibrate_inclusion <- function(pop, scenario, config, verbose = TRUE) {
  ## ----------------------------------------------------------------
  ## Inputs:
  ##   pop      : population data frame (with z, y, n_trial, state)
  ##   scenario : list with $rho, $target_cv_w, $id
  ##   config   : SIM_CONFIG
  ##
  ## Returns: list with
  ##   gamma_0      : calibrated intercept
  ##   alpha_design : calibrated design scale
  ##   y_star       : standardized outcome vector (length M)
  ##   x_design     : design variation vector (length M)
  ##   pi           : inclusion probability vector (length M)
  ##   cv_w         : achieved CV(w) = CV(1/pi)
  ##   achieved_N   : expected sample size = sum(pi)
  ## ----------------------------------------------------------------

  rho      <- scenario$rho
  target_N <- config$sampling$N_target
  target_cv <- scenario$target_cv_w
  M <- nrow(pop)

  if (verbose) {
    cat(sprintf("  Calibrating inclusion for %s (rho=%.2f, target CV=%.2f) ...\n",
                scenario$id, rho, target_cv))
  }

  ## Step 1: Compute y_star (fixed across calibration iterations)
  y_star <- compute_y_star(pop)

  ## Step 2: Search over alpha_design to match target CV(w)
  ##
  ## CV(w) = CV(1/pi). For fixed rho and y_star, increasing alpha_design
  ## increases the spread of x_design, which increases pi variation,
  ## which increases weight variation.
  ##
  ## We use optimize() on the squared difference from target CV.

  ## NOTE: We compute the EXPECTED sample-level CV(w), not the population-level.
  ## In Poisson PPS, the expected sample distribution of w = 1/pi is different
  ## from the population distribution of 1/pi, because units with higher pi
  ## are more likely to be included. The expected sample-level CV(w) is:
  ##   E_sample[w] = sum(pi * (1/pi)) / sum(pi) = M / sum(pi) = M/N
  ##   E_sample[w^2] = sum(pi * (1/pi)^2) / sum(pi) = sum(1/pi) / sum(pi)
  ##   Var_sample[w] = E[w^2] - E[w]^2
  ## This gives the correct expected weight distribution AMONG SAMPLED UNITS.

  ## Minimum inclusion probability (floor)
  ## pi_min = 0.02 gives max weight = 50 (reasonable upper bound)
  PI_MIN <- 0.02
  PI_MAX <- 0.999

  cv_from_alpha <- function(alpha_d) {
    x_des <- compute_x_design(pop, alpha_d)
    g0 <- calibrate_gamma0(rho, y_star, x_des, target_N)
    pi_vec <- plogis(g0 + rho * y_star + x_des)
    ## Clamp to [PI_MIN, PI_MAX]
    pi_vec <- pmax(PI_MIN, pmin(PI_MAX, pi_vec))
    ## Recompute g0 after clamping to maintain target N
    ## (clamping may change sum(pi), but typically the effect is small)

    ## Compute EXPECTED sample-level CV(w) = CV(1/pi) among sampled units
    ## Under Poisson sampling, unit i is included with prob pi_i.
    ## E_sample[w_i] = sum_i(pi_i * (1/pi_i)) / sum_i(pi_i) = M / N_expected
    ## E_sample[w_i^2] = sum_i(pi_i * (1/pi_i)^2) / sum_i(pi_i) = sum_i(1/pi_i) / N_expected
    w_pop <- 1 / pi_vec
    N_exp <- sum(pi_vec)
    Ew <- M / N_exp
    Ew2 <- sum(w_pop) / N_exp
    Vw <- Ew2 - Ew^2
    cv_w <- sqrt(max(Vw, 0)) / Ew
    cv_w
  }

  ## Objective: (CV(w) - target)^2
  obj <- function(alpha_d) {
    cv_achieved <- cv_from_alpha(alpha_d)
    (cv_achieved - target_cv)^2
  }

  ## Search over alpha_design in [0, 5]
  ## alpha_design = 0 means no design variation (all weight variation from rho)
  ## alpha_design = 5 is extreme
  opt <- optimize(obj, interval = c(0, 5), tol = 0.001)
  alpha_design_opt <- opt$minimum

  ## Compute final inclusion probabilities with optimized alpha_design
  x_design <- compute_x_design(pop, alpha_design_opt)
  gamma_0 <- calibrate_gamma0(rho, y_star, x_design, target_N)
  linear_pred <- gamma_0 + rho * y_star + x_design
  pi <- plogis(linear_pred)

  ## Clamp to [PI_MIN, PI_MAX]
  pi <- pmax(PI_MIN, pmin(PI_MAX, pi))

  ## Compute diagnostics on the inclusion probabilities
  ## Use EXPECTED sample-level CV (same formula as in cv_from_alpha)
  w_pop <- 1 / pi
  achieved_N <- sum(pi)
  Ew <- M / achieved_N
  Ew2 <- sum(w_pop) / achieved_N
  Vw <- Ew2 - Ew^2
  cv_w <- sqrt(max(Vw, 0)) / Ew

  ## Compute pre-clipping E[N] for diagnostic comparison
  pi_preclip <- plogis(linear_pred)
  preclip_N  <- sum(pi_preclip)
  n_clipped  <- sum(pi_preclip < PI_MIN) + sum(pi_preclip > PI_MAX)
  drift_pct  <- 100 * (achieved_N - target_N) / target_N

  if (verbose) {
    cat(sprintf("    alpha_design = %.4f\n", alpha_design_opt))
    cat(sprintf("    gamma_0      = %.4f\n", gamma_0))
    cat(sprintf("    E[N] pre-clip= %.1f (target: %d)\n", preclip_N, target_N))
    cat(sprintf("    E[N] post-clip= %.1f (drift: %+.1f%%)\n", achieved_N, drift_pct))
    cat(sprintf("    N clipped    = %d / %d units (%.1f%%)\n",
                n_clipped, M, 100 * n_clipped / M))
    cat(sprintf("    CV(w)        = %.4f (target: %.4f)\n", cv_w, target_cv))
    cat(sprintf("    pi range     = [%.4f, %.4f]\n", min(pi), max(pi)))
    cat(sprintf("    w range      = [%.2f, %.2f]\n", min(w_pop), max(w_pop)))
  }

  list(
    gamma_0      = gamma_0,
    alpha_design = alpha_design_opt,
    rho          = rho,
    y_star       = y_star,
    x_design     = x_design,
    pi           = pi,
    cv_w         = cv_w,
    expected_N   = achieved_N,
    expected_N_preclip = preclip_N,
    n_clipped    = n_clipped,
    drift_pct    = drift_pct,
    expected_cv_w = cv_w,
    target_cv_w  = target_cv,
    target_N     = target_N,
    scenario_id  = scenario$id
  )
}


###############################################################################
## SECTION 4 : DRAW POISSON PPS SAMPLE
###############################################################################
## Given calibrated inclusion probabilities, draw a Poisson sample and
## compute normalized weights.

draw_poisson_sample <- function(pop, calib, seed) {
  ## ----------------------------------------------------------------
  ## Inputs:
  ##   pop    : population data frame
  ##   calib  : calibration output from calibrate_inclusion()
  ##   seed   : integer seed for this replication
  ##
  ## Returns: data frame of sampled units with additional columns:
  ##   pi_i     : inclusion probability
  ##   w_raw    : raw weight = 1/pi_i
  ##   w_tilde  : normalized weight (sum = N_sample)
  ##   pop_idx  : original population index
  ## ----------------------------------------------------------------

  set.seed(seed)

  M <- nrow(pop)
  pi <- calib$pi

  ## Poisson sampling: include each unit independently with probability pi_i
  included <- rbinom(M, size = 1, prob = pi) == 1L
  N_sample <- sum(included)

  ## Extract sampled observations
  sample_df <- pop[included, , drop = FALSE]
  sample_df$pop_idx <- which(included)
  sample_df$pi_i    <- pi[included]
  sample_df$w_raw   <- 1 / sample_df$pi_i

  ## Normalize weights: sum(w_tilde) = N_sample
  sample_df$w_tilde <- sample_df$w_raw * N_sample / sum(sample_df$w_raw)

  ## Reset row names
  rownames(sample_df) <- NULL

  sample_df
}


###############################################################################
## SECTION 5 : CREATE SYNTHETIC SURVEY STRUCTURE
###############################################################################
## The sandwich variance estimator requires stratum and PSU indices.
## Since we used Poisson PPS (not stratified cluster sampling), we
## create a synthetic survey structure post-hoc.
##
## Strategy:
##   - Define strata based on state groups (states grouped to form ~30 strata)
##   - Within each stratum, assign units to PSUs (clusters of ~15-20 units)
##   - Ensure each stratum has >= 2 PSUs (required for variance estimation)
##
## This is a simplification of the real NSECE design but preserves the
## essential feature: within-PSU correlation induced by geographic clustering.

create_survey_structure <- function(sample_data, config) {
  ## ----------------------------------------------------------------
  ## Inputs:
  ##   sample_data : data frame from draw_poisson_sample()
  ##                 Must contain column "state" (integer 1:51)
  ##   config      : SIM_CONFIG
  ##
  ## Returns: sample_data with additional columns:
  ##   stratum_idx : stratum index (1, ..., H)
  ##   psu_idx     : PSU index (1, ..., total_psu)
  ## ----------------------------------------------------------------

  N <- nrow(sample_data)
  n_strata_target <- config$sampling$n_strata    # 30
  S <- config$population$S                       # 51

  ## --- Step 1: Assign states to strata ---
  ## Group the 51 states into ~30 strata. States with similar indices
  ## (roughly geographic grouping in alphabetical order) share a stratum.
  ## Small states may be merged; large states may form their own stratum.

  ## Count sample sizes per state
  state_counts <- table(factor(sample_data$state, levels = 1:S))
  state_order <- order(state_counts, decreasing = TRUE)

  ## Assign states to strata using round-robin on ordered states
  ## This balances stratum sizes
  state_to_stratum <- integer(S)
  for (i in seq_len(S)) {
    state_to_stratum[state_order[i]] <- ((i - 1) %% n_strata_target) + 1
  }

  ## Map each unit to its stratum
  sample_data$stratum_idx <- state_to_stratum[sample_data$state]

  ## --- Step 2: Assign PSUs within each stratum ---
  ## Target: ~15-20 observations per PSU (matching NSECE ~17 per PSU)
  ## Minimum: 2 PSUs per stratum (required for sandwich)

  target_psu_size <- 17L   # approximate observations per PSU
  psu_counter <- 0L

  sample_data$psu_idx <- NA_integer_

  for (h in seq_len(n_strata_target)) {
    idx_h <- which(sample_data$stratum_idx == h)
    n_h <- length(idx_h)

    if (n_h == 0) next

    ## Determine number of PSUs in this stratum
    n_psu_h <- max(2L, round(n_h / target_psu_size))

    ## Assign units within stratum to PSUs in a cyclic fashion
    ## (deterministic assignment based on row order for reproducibility)
    psu_assignment <- rep(seq_len(n_psu_h), length.out = n_h)

    sample_data$psu_idx[idx_h] <- psu_counter + psu_assignment
    psu_counter <- psu_counter + n_psu_h
  }

  ## --- Step 3: Verify structure ---
  ## Check: all units assigned, all strata have >= 2 PSUs, no NAs

  stopifnot(
    "NA in stratum_idx" = !any(is.na(sample_data$stratum_idx)),
    "NA in psu_idx"     = !any(is.na(sample_data$psu_idx))
  )

  ## Count PSUs per stratum and verify >= 2
  psu_per_stratum <- tapply(sample_data$psu_idx, sample_data$stratum_idx,
                            function(x) length(unique(x)))
  n_singleton <- sum(psu_per_stratum < 2)

  if (n_singleton > 0) {
    ## Fix singleton strata by merging with a neighboring stratum
    singleton_strata <- as.integer(names(psu_per_stratum[psu_per_stratum < 2]))
    for (sh in singleton_strata) {
      ## Merge with the next stratum (or previous if last)
      merge_target <- if (sh < n_strata_target) sh + 1 else sh - 1
      idx_sh <- which(sample_data$stratum_idx == sh)
      sample_data$stratum_idx[idx_sh] <- merge_target
    }
    ## Re-index strata to be contiguous
    unique_strata <- sort(unique(sample_data$stratum_idx))
    stratum_map <- setNames(seq_along(unique_strata), unique_strata)
    sample_data$stratum_idx <- as.integer(stratum_map[as.character(sample_data$stratum_idx)])

    ## Re-index PSUs to be contiguous
    unique_psu <- sort(unique(sample_data$psu_idx))
    psu_map <- setNames(seq_along(unique_psu), unique_psu)
    sample_data$psu_idx <- as.integer(psu_map[as.character(sample_data$psu_idx)])
  }

  ## Final summary
  n_strata_actual <- length(unique(sample_data$stratum_idx))
  n_psu_actual    <- length(unique(sample_data$psu_idx))

  attr(sample_data, "survey_structure") <- list(
    n_strata = n_strata_actual,
    n_psu    = n_psu_actual,
    psu_per_stratum = as.integer(tapply(sample_data$psu_idx,
                                        sample_data$stratum_idx,
                                        function(x) length(unique(x))))
  )

  sample_data
}


###############################################################################
## SECTION 6 : PREPARE STAN DATA LIST
###############################################################################
## Build the Stan data list in the exact format expected by
## hbb_m1.stan (unweighted) and hbb_m1_weighted.stan (weighted).
##
## Population data from sim_01_dgp.R has covariate columns named:
##   intercept, poverty, urban, black, hispanic
## Stan expects a design matrix X with columns:
##   [intercept, poverty, urban, black, hispanic]

prepare_stan_data <- function(sample_data, config, weighted = TRUE) {
  ## ----------------------------------------------------------------
  ## Inputs:
  ##   sample_data : data frame from create_survey_structure()
  ##                 Contains columns: state, intercept, poverty, urban,
  ##                 black, hispanic, z, y, n_trial, w_tilde,
  ##                 stratum_idx, psu_idx
  ##   config      : SIM_CONFIG
  ##   weighted    : logical; if TRUE, include w_tilde
  ##
  ## Returns: list matching Stan data block
  ## ----------------------------------------------------------------

  N <- nrow(sample_data)
  P <- config$true_params$P                # 5
  S <- config$population$S                 # 51

  ## Build design matrix X: intercept + 4 standardized covariates
  ## Covariate column names from sim_01_dgp.R
  cov_names <- c("intercept", "poverty", "urban", "black", "hispanic")

  ## Check that all columns exist
  missing_cols <- setdiff(cov_names, names(sample_data))
  if (length(missing_cols) > 0) {
    stop("Missing covariate columns in sample_data: ",
         paste(missing_cols, collapse = ", "),
         "\nAvailable columns: ", paste(names(sample_data), collapse = ", "))
  }

  X <- as.matrix(sample_data[, cov_names, drop = FALSE])
  colnames(X) <- cov_names

  ## Verify intercept column
  if (any(abs(X[, "intercept"] - 1) > 1e-10)) {
    warning("Intercept column is not exactly 1.0 for all observations. ",
            "Range: [", min(X[, "intercept"]), ", ", max(X[, "intercept"]), "]")
  }

  ## Core Stan data (matches hbb_m1.stan)
  stan_list <- list(
    N       = N,
    P       = P,
    S       = S,
    y       = as.integer(sample_data$y),
    n_trial = as.integer(sample_data$n_trial),
    z       = as.integer(sample_data$z),
    X       = X,
    state   = as.integer(sample_data$state)
  )

  ## Add weights for weighted models
  if (weighted) {
    stan_list$w_tilde <- sample_data$w_tilde
  }

  ## Add survey structure for sandwich (not passed to Stan, but stored
  ## in the same list for convenience in downstream scripts)
  stan_list$stratum_idx <- as.integer(sample_data$stratum_idx)
  stan_list$psu_idx     <- as.integer(sample_data$psu_idx)

  ss <- attr(sample_data, "survey_structure")
  if (!is.null(ss)) {
    stan_list$n_strata <- ss$n_strata
    stan_list$n_psu    <- ss$n_psu
  } else {
    stan_list$n_strata <- length(unique(sample_data$stratum_idx))
    stan_list$n_psu    <- length(unique(sample_data$psu_idx))
  }

  ## Validate
  stopifnot(
    "y length"       = length(stan_list$y) == N,
    "n_trial length" = length(stan_list$n_trial) == N,
    "z length"       = length(stan_list$z) == N,
    "X dimensions"   = nrow(stan_list$X) == N && ncol(stan_list$X) == P,
    "state length"   = length(stan_list$state) == N,
    "state range"    = all(stan_list$state >= 1 & stan_list$state <= S),
    "y <= n_trial"   = all(stan_list$y <= stan_list$n_trial),
    "z consistent"   = all((stan_list$y > 0) == (stan_list$z == 1)),
    "n_trial >= 1"   = all(stan_list$n_trial >= 1)
  )

  if (weighted) {
    stopifnot(
      "w_tilde length"   = length(stan_list$w_tilde) == N,
      "w_tilde positive" = all(stan_list$w_tilde > 0),
      "w_tilde sum"      = abs(sum(stan_list$w_tilde) - N) < 0.01
    )
  }

  stan_list
}


###############################################################################
## SECTION 7 : WEIGHT DIAGNOSTICS
###############################################################################

compute_weight_diagnostics <- function(sample_data) {
  ## ----------------------------------------------------------------
  ## Inputs:
  ##   sample_data : data frame with w_raw, w_tilde, pi_i
  ##
  ## Returns: list of weight diagnostic statistics
  ## ----------------------------------------------------------------

  w <- sample_data$w_raw
  w_t <- sample_data$w_tilde
  N <- length(w)

  ## CV of raw weights
  cv_w <- sd(w) / mean(w)

  ## Kish effective sample size and design effect
  kish_ess  <- (sum(w))^2 / sum(w^2)
  kish_deff <- N / kish_ess

  ## Equivalent: DEFF = 1 + CV^2
  deff_formula <- 1 + cv_w^2

  ## Weight quantiles
  w_quantiles <- quantile(w, probs = c(0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1))

  ## Normalized weight stats
  cv_wt <- sd(w_t) / mean(w_t)

  ## Inclusion probability stats
  pi <- sample_data$pi_i
  pi_quantiles <- quantile(pi, probs = c(0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1))

  ## Correlation of weights with outcome
  y_star_sample <- ifelse(sample_data$z == 1,
                          sample_data$y / sample_data$n_trial,
                          0)
  cor_w_outcome <- cor(w, y_star_sample)

  ## Weighted vs unweighted means
  zero_rate_uw <- mean(sample_data$z == 0)
  zero_rate_wt <- sum(w_t * (sample_data$z == 0)) / sum(w_t)

  it_share_uw <- NA_real_
  it_share_wt <- NA_real_
  if (sum(sample_data$z == 1) > 0) {
    srv_idx <- which(sample_data$z == 1)
    shares <- sample_data$y[srv_idx] / sample_data$n_trial[srv_idx]
    it_share_uw <- mean(shares)
    w_srv <- w_t[srv_idx]
    it_share_wt <- sum(w_srv * shares) / sum(w_srv)
  }

  list(
    N             = N,
    cv_w          = cv_w,
    cv_wt         = cv_wt,
    kish_ess      = kish_ess,
    kish_deff     = kish_deff,
    deff_formula  = deff_formula,
    w_mean        = mean(w),
    w_sd          = sd(w),
    w_min         = min(w),
    w_max         = max(w),
    w_quantiles   = w_quantiles,
    pi_min        = min(pi),
    pi_max        = max(pi),
    pi_quantiles  = pi_quantiles,
    cor_w_outcome = cor_w_outcome,
    sum_w_tilde   = sum(w_t),
    zero_rate_uw  = zero_rate_uw,
    zero_rate_wt  = zero_rate_wt,
    it_share_uw   = it_share_uw,
    it_share_wt   = it_share_wt
  )
}


###############################################################################
## SECTION 8 : MAIN SAMPLING FUNCTION (draw_sample)
###############################################################################
## This is the primary entry point called R=200 times per scenario.
## It orchestrates: sampling -> weighting -> survey structure -> Stan data.

draw_sample <- function(pop, scenario, config, calib, seed, verbose = FALSE) {
  ## ----------------------------------------------------------------
  ## Inputs:
  ##   pop      : population data frame (from sim_01_dgp.R)
  ##   scenario : list with $rho, $target_cv_w, $id
  ##   config   : SIM_CONFIG
  ##   calib    : pre-computed calibration from calibrate_inclusion()
  ##   seed     : integer seed for this replication
  ##   verbose  : logical; print progress messages
  ##
  ## Returns: list with
  ##   sample_data   : data frame of sampled units
  ##   stan_data_uw  : Stan data list for unweighted model (E-UW)
  ##   stan_data_wt  : Stan data list for weighted model (E-WT, E-WS)
  ##   diagnostics   : weight diagnostic summary
  ##   metadata      : seed, scenario_id, N, timing
  ## ----------------------------------------------------------------

  t0 <- proc.time()

  ## Step 1: Draw Poisson PPS sample
  sample_df <- draw_poisson_sample(pop, calib, seed)
  N_sample <- nrow(sample_df)

  if (verbose) {
    cat(sprintf("    Rep seed=%d: N=%d (target=%d)\n",
                seed, N_sample, config$sampling$N_target))
  }

  ## Step 2: Create synthetic survey structure
  sample_df <- create_survey_structure(sample_df, config)

  ## Step 3: Prepare Stan data (both weighted and unweighted)
  stan_data_uw <- prepare_stan_data(sample_df, config, weighted = FALSE)
  stan_data_wt <- prepare_stan_data(sample_df, config, weighted = TRUE)

  ## Step 4: Compute weight diagnostics
  diag <- compute_weight_diagnostics(sample_df)

  ## Timing
  elapsed <- (proc.time() - t0)[["elapsed"]]

  list(
    sample_data  = sample_df,
    stan_data_uw = stan_data_uw,
    stan_data_wt = stan_data_wt,
    diagnostics  = diag,
    metadata     = list(
      seed        = seed,
      scenario_id = scenario$id,
      N           = N_sample,
      elapsed_sec = elapsed
    )
  )
}


###############################################################################
## SECTION 9 : CALIBRATION VERIFICATION
###############################################################################
## Run a quick Monte Carlo check: draw B pilot samples and verify that
## the weight distribution matches targets on average.

run_calibration_check <- function(pop, scenario, config, calib,
                                  B = 20, verbose = TRUE) {
  ## ----------------------------------------------------------------
  ## Inputs:
  ##   pop, scenario, config, calib : as in draw_sample()
  ##   B       : number of pilot replications
  ##   verbose : print results
  ##
  ## Returns: data frame with columns: rep, N, cv_w, kish_deff, kish_ess,
  ##          cor_w_outcome, zero_rate
  ## ----------------------------------------------------------------

  if (verbose) {
    cat(sprintf("\n  Calibration check for %s (B=%d pilot samples) ...\n",
                scenario$id, B))
  }

  results <- data.frame(
    rep           = integer(B),
    N             = integer(B),
    cv_w          = numeric(B),
    kish_deff     = numeric(B),
    kish_ess      = numeric(B),
    cor_w_outcome = numeric(B),
    zero_rate     = numeric(B)
  )

  for (b in seq_len(B)) {
    pilot_seed <- config$seeds$base_seed + 99000L + as.integer(b)
    samp <- draw_poisson_sample(pop, calib, pilot_seed)
    diag <- compute_weight_diagnostics(samp)

    results$rep[b]           <- b
    results$N[b]             <- diag$N
    results$cv_w[b]          <- diag$cv_w
    results$kish_deff[b]     <- diag$kish_deff
    results$kish_ess[b]      <- diag$kish_ess
    results$cor_w_outcome[b] <- diag$cor_w_outcome
    results$zero_rate[b]     <- mean(samp$z == 0)
  }

  if (verbose) {
    cat(sprintf("    %-20s %8s %8s %8s\n", "Metric", "Mean", "SD", "Target"))
    cat(sprintf("    %s\n", paste(rep("-", 50), collapse = "")))
    cat(sprintf("    %-20s %8.1f %8.1f %8d\n",
                "N (sample size)",
                mean(results$N), sd(results$N),
                config$sampling$N_target))
    cat(sprintf("    %-20s %8.4f %8.4f %8.4f\n",
                "CV(w)",
                mean(results$cv_w), sd(results$cv_w),
                scenario$target_cv_w))
    cat(sprintf("    %-20s %8.4f %8.4f %8.4f\n",
                "Kish DEFF",
                mean(results$kish_deff), sd(results$kish_deff),
                scenario$target_kish_deff))
    cat(sprintf("    %-20s %8.1f %8.1f %8.1f\n",
                "Kish ESS",
                mean(results$kish_ess), sd(results$kish_ess),
                scenario$target_kish_ess))
    cat(sprintf("    %-20s %8.4f %8.4f %8s\n",
                "Cor(w, outcome)",
                mean(results$cor_w_outcome), sd(results$cor_w_outcome),
                ifelse(scenario$rho == 0, "~0", ">0")))
    cat(sprintf("    %-20s %8.4f %8.4f %8.4f\n",
                "Zero rate",
                mean(results$zero_rate), sd(results$zero_rate),
                config$nsece_reference$zero_rate))

    ## Pass/fail assessment
    cv_ok   <- abs(mean(results$cv_w) - scenario$target_cv_w) < 0.15
    n_ok    <- abs(mean(results$N) - config$sampling$N_target) < 200
    deff_ok <- abs(mean(results$kish_deff) - scenario$target_kish_deff) < 0.5

    cat(sprintf("\n    Assessment: CV(w) %s, N %s, DEFF %s\n",
                ifelse(cv_ok, "[PASS]", "[WARN]"),
                ifelse(n_ok, "[PASS]", "[WARN]"),
                ifelse(deff_ok, "[PASS]", "[WARN]")))
  }

  invisible(results)
}


###############################################################################
## SECTION 10 : PRINT SAMPLING MODULE SUMMARY
###############################################################################

print_sampling_summary <- function(calib, diag = NULL) {
  ## Print a formatted summary of the calibration and sample diagnostics
  cat(sprintf("\n  --- Sampling Summary for %s ---\n", calib$scenario_id))
  cat(sprintf("    rho           = %.4f\n", calib$rho))
  cat(sprintf("    gamma_0       = %.4f\n", calib$gamma_0))
  cat(sprintf("    alpha_design  = %.4f\n", calib$alpha_design))
  cat(sprintf("    E[N]          = %.1f\n", calib$achieved_N))
  cat(sprintf("    CV(w) target  = %.4f,  achieved = %.4f\n",
              SIM_CONFIG$scenarios[[calib$scenario_id]]$target_cv_w, calib$cv_w))

  if (!is.null(diag)) {
    cat(sprintf("\n    Sample diagnostics:\n"))
    cat(sprintf("      N           = %d\n", diag$N))
    cat(sprintf("      CV(w)       = %.4f\n", diag$cv_w))
    cat(sprintf("      Kish ESS    = %.1f\n", diag$kish_ess))
    cat(sprintf("      Kish DEFF   = %.4f\n", diag$kish_deff))
    cat(sprintf("      w range     = [%.2f, %.2f]\n", diag$w_min, diag$w_max))
    cat(sprintf("      Cor(w, y*)  = %.4f\n", diag$cor_w_outcome))
  }
  cat("\n")
}


###############################################################################
## SECTION 11 : BATCH CALIBRATION ACROSS ALL SCENARIOS
###############################################################################
## Convenience function to calibrate all three scenarios at once.
## Returns a named list of calibration objects.

calibrate_all_scenarios <- function(pop, config, verbose = TRUE) {
  ## ----------------------------------------------------------------
  ## Inputs:
  ##   pop    : population data frame
  ##   config : SIM_CONFIG
  ##   verbose: print progress
  ##
  ## Returns: named list (S0, S3, S4) of calibration objects
  ## ----------------------------------------------------------------

  calibrations <- list()

  for (sid in config$scenario_ids) {
    scenario <- config$scenarios[[sid]]
    calib <- calibrate_inclusion(pop, scenario, config, verbose = verbose)
    calibrations[[sid]] <- calib
  }

  if (verbose) {
    cat("\n  === Calibration Summary ===\n")
    cat(sprintf("  %-4s %10s %10s %10s %10s %10s\n",
                "ID", "rho", "gamma_0", "alpha_d", "E[N]", "CV(w)"))
    cat(sprintf("  %s\n", paste(rep("-", 56), collapse = "")))
    for (sid in config$scenario_ids) {
      c <- calibrations[[sid]]
      cat(sprintf("  %-4s %10.4f %10.4f %10.4f %10.1f %10.4f\n",
                  sid, c$rho, c$gamma_0, c$alpha_design,
                  c$achieved_N, c$cv_w))
    }
    cat("\n")
  }

  calibrations
}


###############################################################################
## SECTION 12 : FULL PIPELINE (CALIBRATE -> SAMPLE -> SAVE)
###############################################################################
## This function runs the complete sampling pipeline for one scenario and
## one replication. It is designed to be called from the main simulation
## loop script (sim_03_run.R).

run_sampling_pipeline <- function(pop, scenario_id, rep_id, config,
                                  calibrations, verbose = FALSE) {
  ## ----------------------------------------------------------------
  ## Inputs:
  ##   pop          : population data frame
  ##   scenario_id  : "S0", "S3", or "S4"
  ##   rep_id       : replication number (1, ..., R)
  ##   config       : SIM_CONFIG
  ##   calibrations : output from calibrate_all_scenarios()
  ##   verbose      : print progress
  ##
  ## Returns: result from draw_sample()
  ##
  ## Side effect: saves sample to disk at
  ##   config$paths$sim_samples/<scenario_id>/rep_<rep_id>.rds
  ## ----------------------------------------------------------------

  scenario <- config$scenarios[[scenario_id]]
  calib    <- calibrations[[scenario_id]]
  seed     <- get_rep_seed(config$seeds$base_seed, rep_id, scenario_id)

  ## Draw sample
  result <- draw_sample(pop, scenario, config, calib, seed, verbose = verbose)

  ## Save to disk
  out_path <- file.path(config$paths$sim_samples, scenario_id,
                        sprintf("rep_%03d.rds", rep_id))
  saveRDS(result, out_path)

  if (verbose) {
    cat(sprintf("    Saved: %s (%.1f KB)\n",
                out_path, file.info(out_path)$size / 1024))
  }

  result
}


###############################################################################
## SECTION 13 : INFORMATIVENESS DIAGNOSTICS
###############################################################################
## Comprehensive diagnostics comparing sample with population to verify
## that the informativeness mechanism is working as intended.

compute_informativeness_diagnostics <- function(sample_result, pop) {
  ## ----------------------------------------------------------------
  ## Inputs:
  ##   sample_result : output from draw_sample()
  ##   pop           : full population data frame
  ##
  ## Returns: list with detailed informativeness metrics
  ## ----------------------------------------------------------------

  samp <- sample_result$sample_data
  diag <- sample_result$diagnostics
  w <- samp$w_tilde

  ## Population values (truth)
  pop_zero_rate  <- mean(pop$z == 0)
  pop_it_share   <- if (sum(pop$z == 1) > 0) {
    mean(pop$y[pop$z == 1] / pop$n_trial[pop$z == 1])
  } else NA_real_

  ## Covariate means in population
  pop_poverty_mean <- mean(pop$poverty)

  ## Sample unweighted means
  samp_zero_rate_uw <- mean(samp$z == 0)
  samp_it_share_uw  <- if (sum(samp$z == 1) > 0) {
    mean(samp$y[samp$z == 1] / samp$n_trial[samp$z == 1])
  } else NA_real_
  samp_poverty_uw <- mean(samp$poverty)

  ## Sample weighted means
  samp_zero_rate_wt <- sum(w * (samp$z == 0)) / sum(w)
  samp_it_share_wt  <- if (sum(samp$z == 1) > 0) {
    w_srv <- w[samp$z == 1]
    shares <- samp$y[samp$z == 1] / samp$n_trial[samp$z == 1]
    sum(w_srv * shares) / sum(w_srv)
  } else NA_real_
  samp_poverty_wt <- sum(w * samp$poverty) / sum(w)

  ## Weight-outcome correlations
  cor_w_z <- cor(w, samp$z)
  cor_w_share <- if (sum(samp$z == 1) > 10) {
    cor(w[samp$z == 1], samp$y[samp$z == 1] / samp$n_trial[samp$z == 1])
  } else NA_real_

  list(
    ## Population values
    pop_zero_rate  = pop_zero_rate,
    pop_it_share   = pop_it_share,
    pop_poverty    = pop_poverty_mean,

    ## Sample unweighted
    samp_zero_rate_uw = samp_zero_rate_uw,
    samp_it_share_uw  = samp_it_share_uw,
    samp_poverty_uw   = samp_poverty_uw,

    ## Sample weighted
    samp_zero_rate_wt = samp_zero_rate_wt,
    samp_it_share_wt  = samp_it_share_wt,
    samp_poverty_wt   = samp_poverty_wt,

    ## Bias from informative sampling (UW)
    bias_zero_rate_uw = samp_zero_rate_uw - pop_zero_rate,
    bias_it_share_uw  = samp_it_share_uw  - pop_it_share,
    bias_poverty_uw   = samp_poverty_uw   - pop_poverty_mean,

    ## Bias correction from weighting (WT)
    bias_zero_rate_wt = samp_zero_rate_wt - pop_zero_rate,
    bias_it_share_wt  = samp_it_share_wt  - pop_it_share,
    bias_poverty_wt   = samp_poverty_wt   - pop_poverty_mean,

    ## Informativeness indicators
    cor_w_z     = cor_w_z,
    cor_w_share = cor_w_share,

    ## Weight distribution (from sample diagnostics)
    cv_w      = diag$cv_w,
    kish_deff = diag$kish_deff,
    kish_ess  = diag$kish_ess
  )
}


###############################################################################
## FINAL SUMMARY
###############################################################################

cat("==============================================================\n")
cat("  SAMPLING MODULE LOADED\n")
cat("==============================================================\n")
cat("\n  Exported functions:\n")
cat("    draw_sample(pop, scenario, config, calib, seed)\n")
cat("    calibrate_inclusion(pop, scenario, config)\n")
cat("    calibrate_all_scenarios(pop, config)\n")
cat("    create_survey_structure(sample_data, config)\n")
cat("    prepare_stan_data(sample_data, config, weighted)\n")
cat("    compute_weight_diagnostics(sample_data)\n")
cat("    compute_informativeness_diagnostics(sample_result, pop)\n")
cat("    run_calibration_check(pop, scenario, config, calib, B)\n")
cat("    run_sampling_pipeline(pop, scenario_id, rep_id, config, calibrations)\n")
cat("    print_sampling_summary(calib, diag)\n")
cat("\n  Typical workflow:\n")
cat("    1. pop <- readRDS('data/precomputed/simulation/population/pop_base.rds')\n")
cat("    2. calibs <- calibrate_all_scenarios(pop$data, SIM_CONFIG)\n")
cat("    3. run_calibration_check(pop$data, SIM_CONFIG$scenarios$S3, SIM_CONFIG, calibs$S3)\n")
cat("    4. result <- draw_sample(pop$data, SIM_CONFIG$scenarios$S3, SIM_CONFIG, calibs$S3, seed=123)\n")
cat("    5. result$stan_data_wt  # pass to Stan for weighted fit\n")
cat("    6. result$diagnostics   # weight distribution summary\n")
cat("==============================================================\n")
