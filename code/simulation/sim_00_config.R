## =============================================================================
## sim_00_config.R -- Simulation Study Configuration
## =============================================================================
## Purpose : Define all global configuration settings for the simulation
##           study comparing 3 estimators across 3 scenarios for the
##           Hurdle Beta-Binomial model.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##
## This file is the single source of truth for:
##   - True DGP parameter values (calibrated to NSECE M1 unweighted fit)
##   - Scenario definitions (S0, S3, S4)
##   - Population and sampling design specifications
##   - MCMC tuning parameters
##   - Evaluation metrics and target parameters
##   - File paths and directory structure
##   - Seed management for reproducibility
##
## Design Decisions:
##   - True parameters from M1 (random intercepts, unweighted)
##   - 3 scenarios (S0, S3, S4) in main text
##   - R=200 replications with 90% CIs
##
## Outputs:
##   - SIM_CONFIG list (loaded into R environment)
##   - data/precomputed/simulation/sim_config.rds
## =============================================================================

cat("==============================================================\n")
cat("  Simulation Configuration\n")
cat("==============================================================\n\n")


###############################################################################
## SECTION 0 : PROJECT ROOT
###############################################################################

## CANONICAL path definition — all other sim_*.R scripts should inherit
## from SIM_CONFIG$paths$project_root rather than redefining this.

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()



###############################################################################
## SECTION 1 : TRUE PARAMETER VALUES (DGP TRUTH)
###############################################################################
## Source: M1 unweighted fit on NSECE 2019 (data/precomputed/results_m1.rds).
##
## The M1 model is:
##   Part 1 (Extensive): logit(q_i) = X_i * alpha + delta[1, state[i]]
##   Part 2 (Intensive): logit(mu_i) = X_i * beta  + delta[2, state[i]]
##   delta[s] ~ N(0, diag(tau) * Omega * diag(tau))
##
## Covariates (P=5): intercept, poverty, urban, black, hispanic
## All covariates are standardized (mean 0, sd 1) in the NSECE data.
##
## NOTE: These values are from the NSECE M1 posterior means. They serve as
## the "ground truth" for the simulation DGP. The simulation generates data
## from these exact parameter values, then evaluates whether estimation
## procedures can recover them.

true_params <- list(
  ## -- Fixed effects: extensive margin (logit scale) --
  ## alpha[1] = intercept (baseline log-odds of serving IT)
  ## alpha[2] = poverty   (higher poverty -> LESS likely to serve IT)
  ## alpha[3] = urban     (more urban -> MORE likely to serve IT)
  ## alpha[4] = black     (higher % Black -> LESS likely to serve IT)
  ## alpha[5] = hispanic  (higher % Hispanic -> LESS likely to serve IT)
  alpha = c(0.696310, -0.119138, 0.253365, -0.070343, -0.139187),

  ## -- Fixed effects: intensive margin (logit scale) --
  ## beta[1] = intercept  (baseline logit IT share)
  ## beta[2] = poverty    (higher poverty -> HIGHER IT share -- REVERSAL)
  ## beta[3] = urban      (more urban -> LOWER IT share)
  ## beta[4] = black      (higher % Black -> HIGHER IT share)
  ## beta[5] = hispanic   (higher % Hispanic -> HIGHER IT share)
  beta = c(-0.032165, 0.057150, -0.017980, 0.079764, 0.039883),

  ## -- Dispersion parameter --
  ## kappa controls overdispersion: smaller kappa = more overdispersion
  ## kappa=5.23 yields roughly 12x overdispersion relative to binomial
  log_kappa = 1.654774,
  kappa     = 5.2319,      # exp(1.654774)


  ## -- Random intercept scale parameters --
  ## tau[1] = SD of state random intercepts for extensive margin
  ## tau[2] = SD of state random intercepts for intensive margin
  ## tau[1] >> tau[2]: more state variation in WHETHER to serve IT
  ## than in HOW MUCH IT to serve
  tau = c(0.577095, 0.208078),

  ## -- Cross-margin correlation --
  ## rho > 0: states that are more likely to serve IT also tend to have
  ## slightly higher IT shares (positive cross-margin correlation)
  ## This was positive in NSECE data (rho=0.285), contrary to theoretical
  ## prediction of negative correlation.
  rho = 0.285266,

  ## -- Derived: 2x2 correlation matrix Omega --
  ## Omega = [[1, rho], [rho, 1]]
  Omega = matrix(c(1, 0.285266, 0.285266, 1), nrow = 2),

  ## -- Derived: 2x2 covariance matrix Sigma_delta --
  ## Sigma_delta = diag(tau) %*% Omega %*% diag(tau)
  ## Computed below after list construction
  Sigma_delta = NULL,

  ## -- Number of covariates (including intercept) --
  P = 5L
)

## Compute the covariance matrix for state random intercepts
tau_diag <- diag(true_params$tau)
true_params$Sigma_delta <- tau_diag %*% true_params$Omega %*% tau_diag

cat("  True parameter values (from M1 unweighted fit):\n")
cov_names <- c("intercept", "poverty", "urban", "black", "hispanic")
cat("    Extensive margin (alpha):\n")
for (k in seq_along(true_params$alpha)) {
  cat(sprintf("      alpha[%d] %-10s = %+.6f\n", k, cov_names[k], true_params$alpha[k]))
}
cat("    Intensive margin (beta):\n")
for (k in seq_along(true_params$beta)) {
  cat(sprintf("      beta[%d]  %-10s = %+.6f\n", k, cov_names[k], true_params$beta[k]))
}
cat(sprintf("    log_kappa = %.6f  (kappa = %.4f)\n",
            true_params$log_kappa, true_params$kappa))
cat(sprintf("    tau = (%.6f, %.6f)\n", true_params$tau[1], true_params$tau[2]))
cat(sprintf("    rho = %.6f\n", true_params$rho))
cat("\n")


###############################################################################
## SECTION 2 : EMPIRICAL DATA CHARACTERISTICS (NSECE 2019 REFERENCE)
###############################################################################
## These values characterize the real NSECE data and serve as calibration
## targets for the simulation. They are NOT parameters but descriptive stats.

nsece_reference <- list(
  N            = 6785L,        # analysis sample (after removing n_trial=0)
  P            = 5L,           # covariates including intercept
  S            = 51L,          # states (50 + DC)
  n_strata     = 30L,          # survey strata
  n_psu        = 415L,         # primary sampling units

  ## Outcome characteristics
  zero_rate       = 0.3527,    # proportion with z=0 (don't serve IT)
  mean_it_share   = 0.4776,    # mean(y/n | y>0) among IT servers
  n_trial_median  = 48L,       # median total 0-5 enrollment
  n_trial_mean    = 60.53,     # mean total 0-5 enrollment
  n_trial_min     = 1L,
  n_trial_max     = 378L,

  ## Survey weight characteristics
  w_cv            = 1.6625,    # coefficient of variation of weights
  kish_ess        = 1802.8,    # Kish effective sample size
  kish_deff       = 3.7636,    # Kish design effect factor
  w_min           = 1.0,       # minimum weight (approximately)
  w_max           = 462.0      # maximum weight (approximately)
)

cat("  NSECE 2019 reference characteristics:\n")
cat(sprintf("    N=%d, S=%d, strata=%d, PSUs=%d\n",
            nsece_reference$N, nsece_reference$S,
            nsece_reference$n_strata, nsece_reference$n_psu))
cat(sprintf("    Zero rate=%.3f, mean IT share (servers)=%.4f\n",
            nsece_reference$zero_rate, nsece_reference$mean_it_share))
cat(sprintf("    n_trial: median=%d, mean=%.1f, range=[%d,%d]\n",
            nsece_reference$n_trial_median, nsece_reference$n_trial_mean,
            nsece_reference$n_trial_min, nsece_reference$n_trial_max))
cat(sprintf("    Weight CV=%.4f, Kish ESS=%.1f, Kish DEFF=%.4f\n",
            nsece_reference$w_cv, nsece_reference$kish_ess, nsece_reference$kish_deff))
cat("\n")


###############################################################################
## SECTION 3 : SCENARIO DEFINITIONS
###############################################################################
## Three scenarios for the main text (S0, S3, S4).
## S1 and S2 are deferred to Supplementary Material SM-F.
##
## Informativeness is controlled via rho in the inclusion model:
##   log(pi_i) = c_0 + rho * y_i^*
## where y_i^* is a standardized version of the latent outcome and c_0
## is calibrated to achieve the target sample size.
##
## rho=0: non-informative (inclusion independent of outcome given covariates)
## rho>0: informative (larger outcomes sampled with higher probability)
##
## The target_cv_w, target_kish_deff, and target_kish_ess are calibration
## targets for the weight distribution. The actual weight distribution
## depends on the interaction of rho with the outcome distribution and is
## tuned during population generation.

scenarios <- list(

  ## --- S0: Non-informative baseline ---
  ## Purpose: Establish that all three estimators produce valid inference
  ## when the sampling design is non-informative. Under S0, the unweighted
  ## estimator is fully valid, and weighting/sandwich should not help (but
  ## also should not harm much beyond efficiency loss from weight variation).
  S0 = list(
    id               = "S0",
    label            = "Non-informative",
    description      = "Baseline: sampling independent of outcome given covariates",
    rho              = 0.0,
    target_cv_w      = 1.0,       # modest weight variation from design only
    target_kish_deff = 2.0,       # 1 + CV_w^2 = 1 + 1.0 = 2.0
    target_kish_ess  = 3500,      # N_target / DEFF = 7000 / 2.0
    expected_result  = "All estimators equivalent; coverage near nominal"
  ),

  ## --- S3: NSECE-calibrated ---
  ## Purpose: Realistic assessment matching the actual NSECE 2019 survey
  ## design characteristics. The informativeness parameter rho=0.15 is
  ## calibrated to produce a weight-outcome correlation and weight CV
  ## similar to the observed NSECE values.
  ## This scenario answers: "How well does our estimation procedure work
  ## under conditions similar to the actual data?"
  S3 = list(
    id               = "S3",
    label            = "NSECE-calibrated",
    description      = "Realistic: calibrated to NSECE 2019 survey characteristics",
    rho              = 0.15,
    target_cv_w      = 1.67,      # matches NSECE CV_w = 1.6625
    target_kish_deff = 3.79,      # matches NSECE Kish DEFF
    target_kish_ess  = 1800,      # matches NSECE Kish ESS ~ 1803
    expected_result  = "E-UW shows mild bias; E-WS achieves nominal coverage"
  ),

  ## --- S4: Stress test ---
  ## Purpose: Extreme scenario where sampling informativeness is strong.
  ## This establishes the upper bound on how badly unweighted inference
  ## can fail and demonstrates that the sandwich correction remains valid
  ## even under severe departure from non-informativeness.
  S4 = list(
    id               = "S4",
    label            = "Highly informative (stress test)",
    description      = "Stress test: strong outcome-dependent sampling",
    rho              = 0.50,
    target_cv_w      = 2.0,       # higher weight variation
    target_kish_deff = 5.0,       # 1 + CV_w^2 = 1 + 4.0 = 5.0
    target_kish_ess  = 1400,      # N_target / DEFF = 7000 / 5.0
    expected_result  = "E-UW biased; E-WT undercoverage; E-WS sandwich essential"
  )
)

## Scenario ordering for iteration
SCENARIO_IDS <- c("S0", "S3", "S4")

cat("  Scenario definitions:\n")
for (sid in SCENARIO_IDS) {
  sc <- scenarios[[sid]]
  cat(sprintf("    %s (%s): rho=%.2f, target_cv_w=%.2f, target_kish_deff=%.2f\n",
              sc$id, sc$label, sc$rho, sc$target_cv_w, sc$target_kish_deff))
}
cat("\n")


###############################################################################
## SECTION 4 : POPULATION SPECIFICATION
###############################################################################
## The simulation generates a finite superpopulation of M providers in S=51
## states. State sizes are proportional to observed NSECE state sample sizes.
## This ensures the state-level structure mimics the actual data.

population <- list(
  M = 50000L,                    # total superpopulation size
  S = 51L,                       # number of states

  ## State proportions: proportional to NSECE state sample sizes.
  ## These are approximate proportions based on the NSECE 2019 data.
  ## States ordered alphabetically: AL, AK, AZ, AR, CA, CO, CT, DC, DE, FL,
  ## GA, HI, ID, IL, IN, IA, KS, KY, LA, ME, MD, MA, MI, MN, MS, MO, MT,
  ## NE, NV, NH, NJ, NM, NY, NC, ND, OH, OK, OR, PA, RI, SC, SD, TN, TX,
  ## UT, VT, VA, WA, WV, WI, WY
  ##
  ## NOTE: These will be loaded from the actual NSECE data in sim_01_dgp.R
  ## via tabulating state_idx. The values here are placeholder proportions
  ## that will be overridden. We store them for documentation and for
  ## standalone use without access to the NSECE data.
  state_props = NULL,            # set from data in sim_01_dgp.R

  ## Covariate distribution parameters for the superpopulation.
  ## Provider-level covariates are drawn from a multivariate normal
  ## (after standardization, the design matrix uses mean-0, sd-1 covariates).
  ## The intercept column is always 1.
  ##
  ## For the standardized covariates (poverty, urban, black, hispanic):
  ##   x_k ~ N(mu_state_k, sigma_k^2) within each state
  ## The between-state variation in covariate means is drawn from the
  ## observed NSECE state-level means.
  x_within_state_sd = c(1.0, 1.0, 1.0, 1.0),  # within-state SD (standardized)

  ## n_trial (total 0-5 enrollment) distribution
  ## Drawn from a log-normal to match NSECE's right-skewed distribution:
  ##   n_trial ~ max(1, round(exp(N(log_n_trial_mu, log_n_trial_sd))))
  ## Calibrated to match: median=48, mean~60, range roughly [1, 400]
  log_n_trial_mu = 3.87,         # log(48) = 3.87; gives median ~ 48
  log_n_trial_sd = 0.80          # gives mean ~ 60 and range ~ [1, 400]
)

cat(sprintf("  Population: M=%d providers in S=%d states\n",
            population$M, population$S))
cat(sprintf("  n_trial distribution: log-normal(%.2f, %.2f)\n",
            population$log_n_trial_mu, population$log_n_trial_sd))
cat("\n")


###############################################################################
## SECTION 5 : SAMPLING DESIGN
###############################################################################
## The sampling design mimics the NSECE stratified cluster design.
## Within each replication:
##   1. Assign population units to strata and PSUs
##   2. Compute inclusion probabilities pi_i (scenario-dependent)
##   3. Sample PSUs within strata (probability proportional to size)
##   4. Include all providers within sampled PSUs
##   5. Compute weights w_i = 1/pi_i and normalize

sampling <- list(
  N_target    = 7000L,           # target sample size (NSECE has 6785)
  n_strata    = 30L,             # number of strata (matches NSECE)
  n_psu_target = 415L,           # target number of PSUs (matches NSECE)

  ## PSU structure
  ## Average PSU size in population: M / n_psu_total_pop
  ## Average PSU size in sample: N_target / n_psu_target ~ 17
  n_psu_per_stratum_mean = 14L,  # ~415/30 = ~14 PSUs per stratum

  ## Sampling fraction (approximate)
  ## N_target / M = 7000/50000 = 0.14
  sampling_fraction = 0.14,

  ## Weight normalization: w_tilde = w * N / sum(w) so sum(w_tilde) = N
  normalize_weights = TRUE
)

cat(sprintf("  Sampling design: N_target=%d, strata=%d, PSU_target=%d\n",
            sampling$N_target, sampling$n_strata, sampling$n_psu_target))
cat(sprintf("  Sampling fraction: %.2f\n", sampling$sampling_fraction))
cat("\n")


###############################################################################
## SECTION 6 : MCMC SETTINGS
###############################################################################
## MCMC tuning parameters for Stan fitting within each replication.
## These are chosen to balance computational cost with reliable inference.
##
## The M1 model has 116 parameters:
##   5 (alpha) + 5 (beta) + 1 (log_kappa) + 2 (tau) + 1 (rho via L_Omega)
##   + 102 (z_delta: 2 x 51) = 116
##
## Warmup of 1500 and sampling of 2000 per chain provide adequate ESS
## for the target parameters. With 4 chains = 8000 post-warmup draws.
## Total draws needed per parameter for reliable CI estimation: ESS > 400.

mcmc <- list(
  chains           = 4L,
  parallel_chains  = 4L,
  iter_warmup      = 1500L,
  iter_sampling    = 2000L,
  adapt_delta      = 0.95,       # high adapt_delta to avoid divergences
  max_treedepth    = 14L,        # generous treedepth for hierarchical model
  seed_offset      = 0L,         # added to per-rep seed for chain variation
  refresh          = 0L,         # suppress per-iteration output in batch mode

  ## Diagnostic thresholds (relaxed for simulation throughput — keeps reps in)
  rhat_threshold   = 1.05,       # relaxed from 1.01 for simulation efficiency
  min_ess_bulk     = 200L,       # relaxed from 400 (we have 4 chains x 2000)
  max_divergent_frac = 0.01,     # allow up to 1% divergent transitions

  ## Strict thresholds for QC reporting table
  ## These are NOT used for pass/fail filtering, only for diagnostic reporting.
  rhat_strict      = 1.01,
  min_ess_strict   = 400L,
  max_div_strict   = 0.0,        # zero divergences

  ## Total posterior draws per replication: 4 chains * 2000 = 8000
  total_draws      = 4L * 2000L  # 8000
)

cat(sprintf("  MCMC: %d chains x (%d warmup + %d sampling) = %d draws\n",
            mcmc$chains, mcmc$iter_warmup, mcmc$iter_sampling,
            mcmc$total_draws))
cat(sprintf("  adapt_delta=%.2f, max_treedepth=%d\n",
            mcmc$adapt_delta, mcmc$max_treedepth))
cat("\n")


###############################################################################
## SECTION 7 : EVALUATION SETTINGS
###############################################################################
## Defines which parameters to track, coverage levels, and number of
## replications.
##
## Target parameters for evaluation:
##   - alpha_poverty (alpha[2]): extensive margin poverty effect
##   - beta_poverty  (beta[2]):  intensive margin poverty effect
##   - log_kappa:                overdispersion
##   - tau[1]:                   extensive random intercept SD
##   - tau[2]:                   intensive random intercept SD
##
## These 5 parameters capture the core aspects of the model:
##   - The poverty reversal (alpha_poverty < 0, beta_poverty > 0)
##   - Overdispersion magnitude
##   - State-level heterogeneity in both margins

evaluation <- list(
  ## Number of simulation replications
  R = 200L,

  ## Coverage level for credible intervals
  ## Using 90% (not 95%) to increase power to detect undercoverage.
  ## MCSE for 90% coverage with R=200: sqrt(0.90 * 0.10 / 200) = 0.0212
  ## This means we can detect ~4 percentage point deviations from nominal.
  coverage_level = 0.90,

  ## Credible interval quantiles (symmetric)
  ci_lower = 0.05,              # (1 - 0.90) / 2
  ci_upper = 0.95,              # 1 - (1 - 0.90) / 2

  ## Target parameters: names, Stan parameter names, and indices
  ## These define exactly which parameters are tracked across replications.
  ## Target parameters with sandwich_applicable flag:
  ## Sandwich correction applies to fixed effects and log_kappa (via H_obs/J_cluster).
  ## Hyperparameters (tau) are NOT sandwich-correctable — E-WS tau = E-WT tau.
  
  target_params = list(
    list(
      name       = "alpha_poverty",
      stan_name  = "alpha[2]",
      true_value = 0.057150,     # NOTE: will be set programmatically below
      description = "Extensive margin poverty coefficient",
      sandwich_applicable = TRUE
    ),
    list(
      name       = "beta_poverty",
      stan_name  = "beta[2]",
      true_value = NULL,
      description = "Intensive margin poverty coefficient",
      sandwich_applicable = TRUE
    ),
    list(
      name       = "log_kappa",
      stan_name  = "log_kappa",
      true_value = NULL,
      description = "Log overdispersion parameter",
      sandwich_applicable = TRUE
    ),
    list(
      name       = "tau_ext",
      stan_name  = "tau[1]",
      true_value = NULL,
      description = "Extensive margin random intercept SD",
      sandwich_applicable = FALSE  # no sandwich for hyperparameters
    ),
    list(
      name       = "tau_int",
      stan_name  = "tau[2]",
      true_value = NULL,
      description = "Intensive margin random intercept SD",
      sandwich_applicable = FALSE  # no sandwich for hyperparameters
    )
  ),

  ## Evaluation metrics computed for each parameter x estimator x scenario
  metrics = c("coverage", "bias", "relative_bias", "rmse", "ci_width",
              "width_ratio"),

  ## Three estimators
  estimators = list(
    list(
      id    = "E_UW",
      label = "Unweighted (E-UW)",
      description = "Unweighted likelihood, model-based SEs (naive posterior CIs)",
      weighted    = FALSE,
      sandwich    = FALSE
    ),
    list(
      id    = "E_WT",
      label = "Weighted-naive (E-WT)",
      description = "Weighted pseudo-likelihood, model-based SEs (pseudo-posterior CIs)",
      weighted    = TRUE,
      sandwich    = FALSE
    ),
    list(
      id    = "E_WS",
      label = "Weighted-sandwich (E-WS)",
      description = "Weighted pseudo-likelihood, sandwich-corrected SEs (Cholesky-transformed CIs)",
      weighted    = TRUE,
      sandwich    = TRUE
    )
  ),

  ## MCSE formula: sqrt(p*(1-p)/R) where p is the coverage probability
  mcse_coverage = sqrt(0.90 * 0.10 / 200)  # = 0.02121
)

## Set true values programmatically from true_params
evaluation$target_params[[1]]$true_value <- true_params$alpha[2]   # alpha_poverty
evaluation$target_params[[2]]$true_value <- true_params$beta[2]    # beta_poverty
evaluation$target_params[[3]]$true_value <- true_params$log_kappa  # log_kappa
evaluation$target_params[[4]]$true_value <- true_params$tau[1]     # tau_ext
evaluation$target_params[[5]]$true_value <- true_params$tau[2]     # tau_int

cat("  Evaluation settings:\n")
cat(sprintf("    R=%d replications, coverage level=%.2f\n",
            evaluation$R, evaluation$coverage_level))
cat(sprintf("    MCSE for coverage: %.4f\n", evaluation$mcse_coverage))
cat("    Target parameters:\n")
for (tp in evaluation$target_params) {
  cat(sprintf("      %-16s %-12s = %+.6f  (%s)\n",
              tp$name, tp$stan_name, tp$true_value, tp$description))
}
cat("    Estimators:\n")
for (est in evaluation$estimators) {
  cat(sprintf("      %-6s %-28s weighted=%s, sandwich=%s\n",
              est$id, est$label, est$weighted, est$sandwich))
}
cat("\n")


###############################################################################
## SECTION 8 : FILE PATHS AND DIRECTORY STRUCTURE
###############################################################################
## All paths are absolute, relative to PROJECT_ROOT.
## The simulation output directory is structured as:
##   data/precomputed/simulation/
##     sim_config.rds               -- this config (for reproducibility)
##     population/                  -- generated populations per scenario
##       pop_S0.rds, pop_S3.rds, pop_S4.rds
##     samples/                     -- drawn samples per rep x scenario
##       S0/rep_001.rds, ..., S0/rep_200.rds
##       S3/rep_001.rds, ...
##       S4/rep_001.rds, ...
##     fits/                        -- Stan fit summaries (not full fit objects)
##       S0/E_UW/rep_001.rds, ...
##       S0/E_WT/rep_001.rds, ...
##       S0/E_WS/rep_001.rds, ...
##       S3/...
##       S4/...
##     results/                     -- aggregated results
##       sim_results_raw.rds        -- per-rep parameter estimates
##       sim_results_summary.rds    -- coverage/bias/RMSE tables
##       sim_diagnostics.rds        -- MCMC diagnostic summaries
##     figures/                     -- simulation figures

paths <- list(
  project_root = PROJECT_ROOT,

  ## Stan model files
  stan_model_m1         = file.path(PROJECT_ROOT, "stan/hbb_m1.stan"),
  stan_model_m1_weighted = file.path(PROJECT_ROOT, "stan/hbb_m1_weighted.stan"),
  ## For the simulation, we use hbb_m1.stan for E_UW and hbb_m1_weighted.stan
  ## for E_WT and E_WS. The weighted M1 model was created by adapting the
  ## M3b-W score formulas to the M1 random-intercept-only structure.

  ## Helper files
  utils_helpers = file.path(PROJECT_ROOT, "code/helpers/utils.R"),

  ## Reference data (for calibrating state proportions and covariate distributions)
  stan_data     = file.path(PROJECT_ROOT, "data/precomputed/stan_data.rds"),
  analysis_data = file.path(PROJECT_ROOT, "data/precomputed/analysis_data.rds"),
  results_m1    = file.path(PROJECT_ROOT, "data/precomputed/results_m1.rds"),
  standardization_params = file.path(PROJECT_ROOT, "data/precomputed/standardization_params.rds"),

  ## Simulation script directory
  sim_scripts = file.path(PROJECT_ROOT, "code/simulation"),

  ## Simulation output directories
  sim_output_root = file.path(PROJECT_ROOT, "data/precomputed/simulation"),
  sim_config_out  = file.path(PROJECT_ROOT, "data/precomputed/simulation/sim_config.rds"),
  sim_population  = file.path(PROJECT_ROOT, "data/precomputed/simulation/population"),
  sim_samples     = file.path(PROJECT_ROOT, "data/precomputed/simulation/samples"),
  sim_fits        = file.path(PROJECT_ROOT, "data/precomputed/simulation/fits"),
  sim_results     = file.path(PROJECT_ROOT, "data/precomputed/simulation/results"),
  sim_figures     = file.path(PROJECT_ROOT, "data/precomputed/simulation/figures"),

  ## Log file for simulation progress
  sim_log = file.path(PROJECT_ROOT, "data/precomputed/simulation/sim_progress.log")
)

## Create output directories if they don't exist
dirs_to_create <- c(
  paths$sim_output_root,
  paths$sim_population,
  paths$sim_samples,
  paths$sim_fits,
  paths$sim_results,
  paths$sim_figures
)

## Also create per-scenario subdirectories
for (sid in SCENARIO_IDS) {
  dirs_to_create <- c(
    dirs_to_create,
    file.path(paths$sim_samples, sid),
    file.path(paths$sim_fits, sid)
  )
  ## Per-estimator subdirectories within fits
  for (est in evaluation$estimators) {
    dirs_to_create <- c(
      dirs_to_create,
      file.path(paths$sim_fits, sid, est$id)
    )
  }
}

for (d in dirs_to_create) {
  if (!dir.exists(d)) {
    dir.create(d, recursive = TRUE, showWarnings = FALSE)
    cat(sprintf("  [CREATED] %s\n", d))
  }
}

cat("  Paths configured. Output root:\n")
cat(sprintf("    %s\n", paths$sim_output_root))
cat("\n")


###############################################################################
## SECTION 9 : SEED MANAGEMENT
###############################################################################
## Reproducibility requires deterministic seed assignment. We use a base seed
## and a helper function that produces a unique seed for each
## (replication, scenario) combination.
##
## Seed structure:
##   per_rep_seed = base_seed + (scenario_numeric * 10000) + rep_id
## where scenario_numeric maps: S0=0, S3=3, S4=4.
##
## This ensures:
##   - Different replications within a scenario have different seeds
##   - Different scenarios have non-overlapping seed ranges
##   - The entire simulation is reproducible from base_seed alone

seeds <- list(
  base_seed        = 20260220L,   # date-based for easy identification
  scenario_offsets = c(S0 = 0L, S3 = 30000L, S4 = 40000L),
  population_seed  = 20260221L    # separate seed for population generation
)

## Helper function: compute per-replication seed
## Arguments:
##   base_seed   : integer, the global base seed
##   rep_id      : integer, replication number (1, ..., R)
##   scenario_id : character, one of "S0", "S3", "S4"
##
## Returns: integer seed, guaranteed unique across all (rep, scenario) combos

get_rep_seed <- function(base_seed, rep_id, scenario_id) {
  scenario_offset <- switch(scenario_id,
    "S0" = 0L,
    "S1" = 10000L,
    "S2" = 20000L,
    "S3" = 30000L,
    "S4" = 40000L,
    stop(sprintf("Unknown scenario: %s", scenario_id))
  )
  base_seed + scenario_offset + as.integer(rep_id)
}

cat("  Seed management:\n")
cat(sprintf("    Base seed: %d\n", seeds$base_seed))
cat(sprintf("    Population seed: %d\n", seeds$population_seed))
cat("    Example per-rep seeds:\n")
for (sid in SCENARIO_IDS) {
  cat(sprintf("      %s rep 1: %d,  rep 100: %d,  rep 200: %d\n",
              sid,
              get_rep_seed(seeds$base_seed, 1, sid),
              get_rep_seed(seeds$base_seed, 100, sid),
              get_rep_seed(seeds$base_seed, 200, sid)))
}
cat("\n")


###############################################################################
## SECTION 10 : PARAMETER LABELS AND DISPLAY NAMES
###############################################################################
## Human-readable labels for tables and figures.

param_labels <- list(
  ## Covariate names (P=5)
  covariate_names = c("intercept", "poverty", "urban", "black", "hispanic"),

  ## Fixed effect labels (D = 2P + 1 = 11)
  fixed_labels = c(
    "alpha_intercept", "alpha_poverty", "alpha_urban",
    "alpha_black", "alpha_hispanic",
    "beta_intercept", "beta_poverty", "beta_urban",
    "beta_black", "beta_hispanic",
    "log_kappa"
  ),

  ## Pretty labels for tables/figures
  fixed_pretty = c(
    "Intercept (ext)", "Poverty (ext)", "Urban (ext)",
    "Black (ext)", "Hispanic (ext)",
    "Intercept (int)", "Poverty (int)", "Urban (int)",
    "Black (int)", "Hispanic (int)",
    "log kappa"
  ),

  ## Target parameter pretty labels
  target_pretty = c(
    alpha_poverty = "Poverty (extensive)",
    beta_poverty  = "Poverty (intensive)",
    log_kappa     = "log kappa",
    tau_ext       = "tau (extensive)",
    tau_int       = "tau (intensive)"
  ),

  ## Scenario labels for plotting
  scenario_pretty = c(
    S0 = "S0: Non-informative",
    S3 = "S3: NSECE-calibrated",
    S4 = "S4: Stress test"
  ),

  ## Estimator labels for plotting
  estimator_pretty = c(
    E_UW = "Unweighted",
    E_WT = "Weighted (naive)",
    E_WS = "Weighted (sandwich)"
  ),

  ## Estimator colors for figures (colorblind-friendly)
  estimator_colors = c(
    E_UW = "#E69F00",  # orange
    E_WT = "#56B4E9",  # sky blue
    E_WS = "#009E73"   # bluish green
  )
)


###############################################################################
## SECTION 11 : COMPUTATIONAL BUDGET
###############################################################################
## Estimated computational cost for planning and monitoring.
##
## Per replication (M1 on N~7000):
##   - Unweighted fit:  ~15 min (4 chains, adapt_delta=0.95)
##   - Weighted fit:    ~20 min
##   - Sandwich computation: ~2 min
##   - Total per rep: ~40 min
##
## Total budget:
##   200 reps x 3 scenarios x 40 min = 24,000 min = 400 hours
##   With 4 cores: ~100 wall-clock hours (4.2 days)
##   With parallelism across replications: much less

compute_budget <- list(
  est_per_rep_minutes = list(
    E_UW_fit     = 15,
    E_WT_fit     = 20,
    sandwich     = 2,
    overhead     = 3,
    total_per_rep = 40
  ),
  total_rep_scenarios  = 200 * 3,     # 600 total replications
  est_total_cpu_hours  = 200 * 3 * 40 / 60,  # 400 CPU-hours
  est_wallclock_hours  = 200 * 3 * 40 / 60 / 4,  # 100 hours with 4-core parallelism
  recommended_parallel = 4L            # number of replications to run in parallel
)

cat("  Computational budget estimate:\n")
cat(sprintf("    Per replication: ~%d min\n", compute_budget$est_per_rep_minutes$total_per_rep))
cat(sprintf("    Total replications: %d (R=%d x %d scenarios)\n",
            compute_budget$total_rep_scenarios, evaluation$R, length(SCENARIO_IDS)))
cat(sprintf("    Estimated total: %.0f CPU-hours, %.0f wall-clock hours (4 cores)\n",
            compute_budget$est_total_cpu_hours, compute_budget$est_wallclock_hours))
cat("\n")


###############################################################################
## SECTION 12 : ASSEMBLE MASTER CONFIG LIST
###############################################################################

SIM_CONFIG <- list(
  ## Core settings
  true_params     = true_params,
  nsece_reference = nsece_reference,
  scenarios       = scenarios,
  scenario_ids    = SCENARIO_IDS,
  population      = population,
  sampling        = sampling,
  mcmc            = mcmc,
  evaluation      = evaluation,
  paths           = paths,
  seeds           = seeds,
  param_labels    = param_labels,
  compute_budget  = compute_budget,

  ## Metadata
  version         = "1.0.0",
  created         = Sys.time(),
  description     = paste(
    "Simulation study configuration for the HBB model.",
    "Compares 3 estimators (E-UW, E-WT, E-WS) across 3 scenarios (S0, S3, S4)",
    "using a Hurdle Beta-Binomial model with state random intercepts (M1).",
    "True parameters calibrated to NSECE 2019 M1 unweighted fit.",
    sep = " "
  )
)

cat("  Master config list (SIM_CONFIG) assembled.\n")
cat(sprintf("    Sections: %d\n", length(SIM_CONFIG)))
cat(sprintf("    Version: %s\n", SIM_CONFIG$version))
cat("\n")


###############################################################################
## SECTION 13 : HELPER FUNCTIONS
###############################################################################

## --- get_rep_seed(): already defined above in Section 9 ---
## Re-export as part of the config for convenience.


## --- print_config(): formatted summary of the configuration ---
## Prints a human-readable overview of all key settings.

print_config <- function(config) {
  cat("\n")
  cat("================================================================\n")
  cat("  SIMULATION STUDY CONFIGURATION SUMMARY\n")
  cat("================================================================\n")

  cat(sprintf("\n  Version: %s\n", config$version))
  cat(sprintf("  Created: %s\n", format(config$created, "%Y-%m-%d %H:%M:%S")))

  ## True parameters
  cat("\n  --- True Parameters (DGP) ---\n")
  tp <- config$true_params
  cnames <- c("intercept", "poverty", "urban", "black", "hispanic")
  cat("  alpha: ")
  cat(paste(sprintf("%+.4f", tp$alpha), collapse = "  "))
  cat("\n  beta:  ")
  cat(paste(sprintf("%+.4f", tp$beta), collapse = "  "))
  cat(sprintf("\n  log_kappa = %.4f (kappa = %.2f)\n", tp$log_kappa, tp$kappa))
  cat(sprintf("  tau = (%.4f, %.4f),  rho = %.4f\n", tp$tau[1], tp$tau[2], tp$rho))

  ## Scenarios
  cat("\n  --- Scenarios ---\n")
  cat(sprintf("  %-4s %-30s %8s %8s %8s\n",
              "ID", "Label", "rho", "CV_w", "DEFF"))
  cat(sprintf("  %s\n", paste(rep("-", 62), collapse = "")))
  for (sid in config$scenario_ids) {
    sc <- config$scenarios[[sid]]
    cat(sprintf("  %-4s %-30s %8.2f %8.2f %8.2f\n",
                sc$id, sc$label, sc$rho, sc$target_cv_w, sc$target_kish_deff))
  }

  ## Sampling and MCMC
  cat(sprintf("\n  --- Sampling Design ---\n"))
  cat(sprintf("  Population M=%d, Sample N_target=%d, Strata=%d, PSU=%d\n",
              config$population$M, config$sampling$N_target,
              config$sampling$n_strata, config$sampling$n_psu_target))

  cat(sprintf("\n  --- MCMC ---\n"))
  cat(sprintf("  Chains=%d, Warmup=%d, Sampling=%d, adapt_delta=%.2f\n",
              config$mcmc$chains, config$mcmc$iter_warmup,
              config$mcmc$iter_sampling, config$mcmc$adapt_delta))

  ## Evaluation
  cat(sprintf("\n  --- Evaluation ---\n"))
  cat(sprintf("  R=%d replications, Coverage level=%.2f, MCSE=%.4f\n",
              config$evaluation$R, config$evaluation$coverage_level,
              config$evaluation$mcse_coverage))

  cat("\n  Target parameters:\n")
  for (tp_item in config$evaluation$target_params) {
    cat(sprintf("    %-16s = %+.6f\n", tp_item$name, tp_item$true_value))
  }

  cat("\n  Estimators:\n")
  for (est in config$evaluation$estimators) {
    cat(sprintf("    %-6s %-28s\n", est$id, est$label))
  }

  ## Computational budget
  cat(sprintf("\n  --- Budget ---\n"))
  cat(sprintf("  Estimated: %.0f CPU-hours, %.0f wall-clock hours (%d-core parallel)\n",
              config$compute_budget$est_total_cpu_hours,
              config$compute_budget$est_wallclock_hours,
              config$compute_budget$recommended_parallel))

  ## Seeds
  cat(sprintf("\n  --- Seeds ---\n"))
  cat(sprintf("  Base seed: %d, Population seed: %d\n",
              config$seeds$base_seed, config$seeds$population_seed))

  cat("\n================================================================\n")
  cat("  END OF CONFIGURATION SUMMARY\n")
  cat("================================================================\n\n")
}


## --- validate_config(): check internal consistency ---
## Returns TRUE if all checks pass, otherwise stops with error.

validate_config <- function(config) {
  cat("  Validating configuration ...\n")

  checks_passed <- 0L
  checks_total  <- 0L

  check <- function(condition, msg) {
    checks_total <<- checks_total + 1L
    if (condition) {
      checks_passed <<- checks_passed + 1L
    } else {
      stop(sprintf("Config validation FAILED: %s", msg), call. = FALSE)
    }
  }

  ## True parameter dimensions
  check(length(config$true_params$alpha) == config$true_params$P,
        "alpha length must equal P")
  check(length(config$true_params$beta) == config$true_params$P,
        "beta length must equal P")
  check(length(config$true_params$tau) == 2,
        "tau must have length 2")
  check(config$true_params$rho > -1 && config$true_params$rho < 1,
        "rho must be in (-1, 1)")
  check(config$true_params$kappa > 0,
        "kappa must be positive")
  check(abs(config$true_params$kappa - exp(config$true_params$log_kappa)) < 0.01,
        "kappa must equal exp(log_kappa)")
  check(nrow(config$true_params$Omega) == 2 && ncol(config$true_params$Omega) == 2,
        "Omega must be 2x2")
  check(nrow(config$true_params$Sigma_delta) == 2 && ncol(config$true_params$Sigma_delta) == 2,
        "Sigma_delta must be 2x2")

  ## Sigma_delta positive definite
  eig_Sigma <- eigen(config$true_params$Sigma_delta, symmetric = TRUE, only.values = TRUE)$values
  check(all(eig_Sigma > 0), "Sigma_delta must be positive definite")

  ## Scenarios
  check(length(config$scenario_ids) >= 1, "Must have at least one scenario")
  for (sid in config$scenario_ids) {
    check(sid %in% names(config$scenarios),
          sprintf("Scenario '%s' not defined", sid))
    sc <- config$scenarios[[sid]]
    check(sc$rho >= 0, sprintf("rho must be non-negative for scenario %s", sid))
    check(sc$target_cv_w > 0, sprintf("target_cv_w must be positive for scenario %s", sid))
  }

  ## Population
  check(config$population$M > config$sampling$N_target,
        "Population M must exceed sample N_target")
  check(config$population$S == 51, "S must be 51")

  ## Sampling
  check(config$sampling$N_target > 0, "N_target must be positive")
  check(config$sampling$n_strata > 0, "n_strata must be positive")

  ## MCMC
  check(config$mcmc$chains >= 2, "Must have at least 2 chains")
  check(config$mcmc$adapt_delta > 0.8 && config$mcmc$adapt_delta < 1.0,
        "adapt_delta must be in (0.8, 1.0)")

  ## Evaluation
  check(config$evaluation$R >= 50, "R must be at least 50")
  check(config$evaluation$coverage_level > 0.5 && config$evaluation$coverage_level < 1.0,
        "coverage_level must be in (0.5, 1.0)")
  check(length(config$evaluation$target_params) >= 1,
        "Must have at least one target parameter")

  ## All target params have non-NULL true values
  for (tp_item in config$evaluation$target_params) {
    check(!is.null(tp_item$true_value),
          sprintf("true_value must be set for target '%s'", tp_item$name))
  }

  ## Seeds
  check(is.integer(config$seeds$base_seed) || is.numeric(config$seeds$base_seed),
        "base_seed must be numeric")

  cat(sprintf("  [PASS] All %d validation checks passed.\n\n", checks_total))
  invisible(TRUE)
}


###############################################################################
## SECTION 14 : VALIDATE AND SAVE
###############################################################################

## Run validation
validate_config(SIM_CONFIG)

## Print summary
print_config(SIM_CONFIG)

## Save config as RDS
saveRDS(SIM_CONFIG, paths$sim_config_out)
cat(sprintf("  Saved config: %s\n", paths$sim_config_out))
cat(sprintf("    File size: %.1f KB\n",
            file.info(paths$sim_config_out)$size / 1024))


###############################################################################
## FINAL SUMMARY
###############################################################################
cat("\n==============================================================\n")
cat("  SIMULATION CONFIGURATION COMPLETE\n")
cat("==============================================================\n")
cat(sprintf("\n  Config object: SIM_CONFIG (in R environment)\n"))
cat(sprintf("  Config file:   %s\n", paths$sim_config_out))
cat(sprintf("  Seed function: get_rep_seed(base_seed, rep_id, scenario_id)\n"))
cat(sprintf("  Print helper:  print_config(SIM_CONFIG)\n"))
cat(sprintf("  Validation:    validate_config(SIM_CONFIG)\n"))
cat(sprintf("\n  Next step: source('code/simulation/sim_01_dgp.R')\n"))
cat("==============================================================\n")
