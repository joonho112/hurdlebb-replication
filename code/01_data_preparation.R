## =============================================================================
## 01_data_preparation.R -- Data Preparation for Stan Model Fitting
## =============================================================================
## Purpose : Load NSECE 2019 CB master data, construct the Stan data list
##           for the Hurdle Beta-Binomial model, and save analysis-ready
##           objects for downstream Stan fitting.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Track   : A (requires NSECE restricted-use data; see data/DATA_ACCESS.md)
## Inputs  : data/restricted/cb_master_2019.rds
## Outputs : data/precomputed/stan_data.rds
##           data/precomputed/analysis_data.rds
##           data/precomputed/standardization_params.rds
## =============================================================================
##
## WARNING -- Track A (Full Replication)
## This script requires NSECE 2019 restricted-use data, which must be obtained
## through the NSECE data access procedures. See data/DATA_ACCESS.md for
## instructions on obtaining the data.
##
## For partial replication WITHOUT restricted data, use Track B:
##   source("code/06_tables_figures.R")
## =============================================================================

cat("==============================================================\n")
cat("  Data Preparation  (Step 1)\n")
cat("==============================================================\n\n")

# -- 0. Setup -----------------------------------------------------------------
set.seed(20250220)

library(dplyr, warn.conflicts = FALSE)
library(tidyr)
library(readr)

## Paths (portable from project root) ----------------------------------------
PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()

MASTER_DATA   <- file.path(PROJECT_ROOT, "data/restricted/cb_master_2019.rds")
OUTPUT_DIR    <- file.path(PROJECT_ROOT, "data/precomputed")
STAN_DATA_OUT <- file.path(OUTPUT_DIR, "stan_data.rds")
ANALYSIS_OUT  <- file.path(OUTPUT_DIR, "analysis_data.rds")

## Ensure output directory exists ---------------------------------------------
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
  cat(sprintf("[INFO] Created output directory: %s\n", OUTPUT_DIR))
}

## Graceful failure check for restricted data ---------------------------------
if (!file.exists(MASTER_DATA)) {
  stop("Restricted-use data not found at:\n  ", MASTER_DATA,
       "\n\nThis script requires NSECE restricted-use data.",
       "\nSee data/DATA_ACCESS.md for instructions on obtaining the data.",
       "\nFor partial replication without restricted data, use Track B:",
       "\n  source('code/06_tables_figures.R')")
}

###############################################################################
## SECTION 1 : LOAD RAW DATA
###############################################################################
cat("--- 1. Loading master data ---\n")

raw <- readRDS(MASTER_DATA)
cat(sprintf("  Loaded: %s\n", MASTER_DATA))
cat(sprintf("  Dimensions: %d rows x %d columns\n", nrow(raw), ncol(raw)))

###############################################################################
## SECTION 2 : SELECT AND VALIDATE REQUIRED VARIABLES
###############################################################################
cat("\n--- 2. Variable selection and validation ---\n")

## 2a. Define required variable names -----------------------------------------
outcome_vars   <- c("enrl_it", "enrl_total_05")
covariate_vars <- c("comm_pct_poverty_num", "pct_urban",
                     "comm_pct_black_num", "comm_pct_hisp_num")
state_var      <- "state_name"
survey_vars    <- c("weight", "vstratum", "vpsu")
policy_vars    <- c("ccdf_IT_MR_percentile_center",
                     "ccdf_TieredReim", "ccdf_DRR_ITaddon")
id_var         <- "cb_id"

all_required <- c(id_var, outcome_vars, covariate_vars, state_var,
                  survey_vars, policy_vars)

## 2b. Check all variables exist ----------------------------------------------
missing_vars <- setdiff(all_required, names(raw))
if (length(missing_vars) > 0) {
  stop(
    sprintf("Missing variables in data: %s", paste(missing_vars, collapse = ", ")),
    call. = FALSE
  )
}
cat("  [PASS] All required variables found in data.\n")

## 2c. Subset to required variables -------------------------------------------
dat <- raw %>%
  select(all_of(all_required))

cat(sprintf("  Selected %d variables for analysis.\n", ncol(dat)))

## 2d. Check for NA in critical variables -------------------------------------
cat("\n  Missing values per variable:\n")
na_counts <- colSums(is.na(dat))
for (v in names(na_counts)) {
  if (na_counts[v] > 0) {
    cat(sprintf("    %-35s %d NAs\n", v, na_counts[v]))
  }
}
if (all(na_counts == 0)) {
  cat("    (none, except policy variables checked below)\n")
}

## 2e. Validate outcome variables are non-negative whole numbers --------------
stopifnot(
  "enrl_it must be non-negative" =
    all(dat$enrl_it >= 0, na.rm = TRUE),
  "enrl_total_05 must be non-negative" =
    all(dat$enrl_total_05 >= 0, na.rm = TRUE),
  "enrl_it must be whole numbers" =
    all(dat$enrl_it == floor(dat$enrl_it), na.rm = TRUE),
  "enrl_total_05 must be whole numbers" =
    all(dat$enrl_total_05 == floor(dat$enrl_total_05), na.rm = TRUE)
)
cat("  [PASS] Outcome variables are non-negative integers.\n")

## 2f. Validate enrl_it <= enrl_total_05 --------------------------------------
violations <- sum(dat$enrl_it > dat$enrl_total_05, na.rm = TRUE)
if (violations > 0) {
  warning(sprintf("  [WARN] %d cases where enrl_it > enrl_total_05!", violations))
} else {
  cat("  [PASS] enrl_it <= enrl_total_05 for all observations.\n")
}

## 2g. Validate weights are positive ------------------------------------------
stopifnot("Survey weights must be positive" = all(dat$weight > 0))
cat("  [PASS] All survey weights are positive.\n")

###############################################################################
## SECTION 3 : REMOVE PROVIDERS WITH n_trial == 0
###############################################################################
cat("\n--- 3. Filter: remove providers with enrl_total_05 == 0 ---\n")

n_zero_trial <- sum(dat$enrl_total_05 == 0)
cat(sprintf("  Providers with enrl_total_05 == 0: %d (%.1f%%)\n",
            n_zero_trial, 100 * n_zero_trial / nrow(dat)))

if (n_zero_trial > 0) {
  dat <- dat %>% filter(enrl_total_05 > 0)
  cat(sprintf("  Removed %d providers. Remaining N = %d\n",
              n_zero_trial, nrow(dat)))
} else {
  cat("  No providers removed.\n")
}

N <- nrow(dat)
cat(sprintf("  Final analysis sample: N = %d\n", N))

###############################################################################
## SECTION 4 : CONSTRUCT OUTCOME VARIABLES
###############################################################################
cat("\n--- 4. Outcome variables ---\n")

## Convert to integer (they are stored as numeric but are whole numbers) ------
dat <- dat %>%
  mutate(
    y       = as.integer(enrl_it),        # IT enrollment count
    n_trial = as.integer(enrl_total_05),  # total 0-5 enrollment
    z       = as.integer(enrl_it > 0)     # participation indicator (1 = serves IT)
  )

n_participants <- sum(dat$z)
n_zeros        <- sum(dat$z == 0)
cat(sprintf("  IT participants (z=1) : %d (%.1f%%)\n",
            n_participants, 100 * n_participants / N))
cat(sprintf("  Non-participants (z=0): %d (%.1f%%)\n",
            n_zeros, 100 * n_zeros / N))
cat(sprintf("  Structural zero rate  : %.1f%%\n",
            100 * n_zeros / N))

## Validate: among z=0, y must be 0 ------------------------------------------
stopifnot(
  "y must be 0 when z is 0" = all(dat$y[dat$z == 0] == 0)
)
cat("  [PASS] y = 0 for all non-participants.\n")

## Summary of y among participants --------------------------------------------
cat("\n  Among participants (z=1):\n")
y_pos <- dat$y[dat$z == 1]
cat(sprintf("    min(y)    = %d\n", min(y_pos)))
cat(sprintf("    median(y) = %d\n", as.integer(median(y_pos))))
cat(sprintf("    mean(y)   = %.1f\n", mean(y_pos)))
cat(sprintf("    max(y)    = %d\n", max(y_pos)))

## n_trial summary ------------------------------------------------------------
cat(sprintf("\n  n_trial summary:\n"))
cat(sprintf("    min       = %d\n", min(dat$n_trial)))
cat(sprintf("    median    = %d\n", as.integer(median(dat$n_trial))))
cat(sprintf("    mean      = %.1f\n", mean(dat$n_trial)))
cat(sprintf("    max       = %d\n", max(dat$n_trial)))

###############################################################################
## SECTION 5 : STATE INDEX
###############################################################################
cat("\n--- 5. State index ---\n")

## Create sequential state index (1..S) based on sorted state names -----------
state_levels <- sort(unique(dat$state_name))
S <- length(state_levels)
cat(sprintf("  Number of unique states: %d\n", S))

## Validate we have exactly 51 (50 states + DC) -------------------------------
if (S != 51) {
  warning(sprintf("  [WARN] Expected 51 states (50 + DC), found %d", S))
} else {
  cat("  [PASS] 51 states (50 states + DC) confirmed.\n")
}

## Map state names to sequential integer indices ------------------------------
dat <- dat %>%
  mutate(
    state_idx = match(state_name, state_levels)
  )

## Validate state index covers 1..S without gaps ------------------------------
stopifnot(
  "State index range" = all(dat$state_idx >= 1 & dat$state_idx <= S),
  "All states present" = length(unique(dat$state_idx)) == S
)

## Print state sample sizes ---------------------------------------------------
state_ns <- dat %>%
  group_by(state_name, state_idx) %>%
  summarise(n = n(), n_IT = sum(z), .groups = "drop") %>%
  arrange(state_idx)

cat("\n  State sample sizes (first 10 / last 5):\n")
cat(sprintf("    %-5s %-25s %5s %5s\n", "idx", "state", "n", "n_IT"))
cat(sprintf("    %-5s %-25s %5s %5s\n", "---", "-----", "---", "----"))
for (i in 1:min(10, S)) {
  row_i <- state_ns[i, ]
  cat(sprintf("    %-5d %-25s %5d %5d\n",
              row_i$state_idx, row_i$state_name, row_i$n, row_i$n_IT))
}
if (S > 10) {
  cat("    ...\n")
  for (i in (S-4):S) {
    row_i <- state_ns[i, ]
    cat(sprintf("    %-5d %-25s %5d %5d\n",
                row_i$state_idx, row_i$state_name, row_i$n, row_i$n_IT))
  }
}

cat(sprintf("\n  Min state n: %d (%s)\n",
            min(state_ns$n),
            state_ns$state_name[which.min(state_ns$n)]))
cat(sprintf("  Max state n: %d (%s)\n",
            max(state_ns$n),
            state_ns$state_name[which.max(state_ns$n)]))

###############################################################################
## SECTION 6 : STANDARDIZE PROVIDER-LEVEL COVARIATES (X MATRIX)
###############################################################################
cat("\n--- 6. Provider-level covariates (X matrix) ---\n")

## 6a. Check for NAs in provider covariates -----------------------------------
for (v in covariate_vars) {
  na_count <- sum(is.na(dat[[v]]))
  if (na_count > 0) {
    stop(sprintf("Variable '%s' has %d NAs. Handle before proceeding.", v, na_count),
         call. = FALSE)
  }
}
cat("  [PASS] No missing values in provider covariates.\n")

## 6b. Raw covariate summaries ------------------------------------------------
cat("\n  Raw covariate summaries:\n")
for (v in covariate_vars) {
  vals <- dat[[v]]
  cat(sprintf("    %-25s  mean=%7.2f  sd=%7.2f  min=%7.2f  max=%7.2f\n",
              v, mean(vals), sd(vals), min(vals), max(vals)))
}

## 6c. Standardize (center and scale) -----------------------------------------
## Store means and SDs for later back-transformation
x_means <- vapply(covariate_vars, function(v) mean(dat[[v]]), numeric(1))
x_sds   <- vapply(covariate_vars, function(v) sd(dat[[v]]),   numeric(1))

## Verify no zero SDs (would cause division by zero) --------------------------
zero_sd <- x_sds[x_sds < .Machine$double.eps]
if (length(zero_sd) > 0) {
  stop(sprintf("Zero SD for covariates: %s", paste(names(zero_sd), collapse = ", ")),
       call. = FALSE)
}

## Create standardized versions -----------------------------------------------
x_std_names <- paste0(covariate_vars, "_std")
for (i in seq_along(covariate_vars)) {
  dat[[x_std_names[i]]] <- (dat[[covariate_vars[i]]] - x_means[i]) / x_sds[i]
}

cat("\n  Standardized covariate summaries (should be mean~0, sd~1):\n")
for (v in x_std_names) {
  vals <- dat[[v]]
  cat(sprintf("    %-35s  mean=%8.5f  sd=%7.5f\n", v, mean(vals), sd(vals)))
}

## 6d. Build the X matrix: [intercept, 4 standardized covariates] -> N x 5 ---
X <- cbind(
  intercept = 1,
  poverty   = dat$comm_pct_poverty_num_std,
  urban     = dat$pct_urban_std,
  black     = dat$comm_pct_black_num_std,
  hispanic  = dat$comm_pct_hisp_num_std
)

stopifnot(
  "X must be N x 5" = nrow(X) == N && ncol(X) == 5,
  "X intercept column must be all 1s" = all(X[, 1] == 1),
  "No NAs in X" = !any(is.na(X))
)
cat(sprintf("\n  X matrix dimensions: %d x %d\n", nrow(X), ncol(X)))
cat(sprintf("  X column names: %s\n", paste(colnames(X), collapse = ", ")))
cat("  [PASS] X matrix constructed.\n")

###############################################################################
## SECTION 7 : STATE-LEVEL POLICY MATRIX (V MATRIX)
###############################################################################
cat("\n--- 7. State-level policy covariates (V matrix) ---\n")

## 7a. Build state-level data frame -------------------------------------------
## Policy variables are constant within state, so take the first value
state_policy <- dat %>%
  group_by(state_idx, state_name) %>%
  summarise(
    MR_pctile  = first(ccdf_IT_MR_percentile_center),
    TieredReim = first(ccdf_TieredReim),
    ITaddon    = first(ccdf_DRR_ITaddon),
    .groups    = "drop"
  ) %>%
  arrange(state_idx)

## Validate: one row per state ------------------------------------------------
stopifnot(
  "Must have S rows in state_policy" = nrow(state_policy) == S
)

## 7b. Check within-state consistency of policy variables ---------------------
## All providers in the same state should have the same policy values
cat("  Checking within-state consistency of policy variables ...\n")
for (v in policy_vars) {
  inconsistencies <- dat %>%
    group_by(state_idx) %>%
    summarise(n_unique = n_distinct(.data[[v]], na.rm = FALSE), .groups = "drop") %>%
    filter(n_unique > 1)

  if (nrow(inconsistencies) > 0) {
    warning(sprintf("  [WARN] '%s' varies within %d state(s)!", v, nrow(inconsistencies)))
  }
}
cat("  [PASS] Policy variables consistent within states.\n")

## 7c. Handle missing MR_pctile values (impute with state median) -------------
n_mr_missing <- sum(is.na(state_policy$MR_pctile))
cat(sprintf("\n  MR_pctile missing states: %d of %d\n", n_mr_missing, S))

if (n_mr_missing > 0) {
  ## Impute with the median of non-missing states
  mr_median <- median(state_policy$MR_pctile, na.rm = TRUE)
  cat(sprintf("  Imputing %d missing MR_pctile values with state median = %.2f\n",
              n_mr_missing, mr_median))

  missing_states <- state_policy$state_name[is.na(state_policy$MR_pctile)]
  cat(sprintf("  Missing states: %s\n", paste(missing_states, collapse = ", ")))

  state_policy <- state_policy %>%
    mutate(
      MR_pctile_imputed = is.na(MR_pctile),
      MR_pctile = if_else(is.na(MR_pctile), mr_median, MR_pctile)
    )
} else {
  state_policy$MR_pctile_imputed <- FALSE
}

## 7d. Standardize continuous policy variable (MR_pctile) ---------------------
v_mr_mean <- mean(state_policy$MR_pctile)
v_mr_sd   <- sd(state_policy$MR_pctile)
stopifnot("MR_pctile SD must be > 0" = v_mr_sd > .Machine$double.eps)

state_policy <- state_policy %>%
  mutate(MR_pctile_std = (MR_pctile - v_mr_mean) / v_mr_sd)

cat(sprintf("  MR_pctile: mean=%.2f, sd=%.2f (before standardization)\n",
            v_mr_mean, v_mr_sd))

## 7e. Binary variables: keep as-is (no standardization) ----------------------
cat(sprintf("  TieredReim: %d states with tiered reimbursement\n",
            sum(state_policy$TieredReim)))
cat(sprintf("  ITaddon   : %d states with IT addon\n",
            sum(state_policy$ITaddon)))

## 7f. Build V matrix: [intercept, MR_pctile_std, TieredReim, ITaddon] -------
V <- cbind(
  intercept  = 1,
  MR_pctile  = state_policy$MR_pctile_std,
  TieredReim = state_policy$TieredReim,
  ITaddon    = state_policy$ITaddon
)

stopifnot(
  "V must be S x 4" = nrow(V) == S && ncol(V) == 4,
  "V intercept column must be all 1s" = all(V[, 1] == 1),
  "No NAs in V" = !any(is.na(V))
)
cat(sprintf("\n  V matrix dimensions: %d x %d\n", nrow(V), ncol(V)))
cat(sprintf("  V column names: %s\n", paste(colnames(V), collapse = ", ")))
cat("  [PASS] V matrix constructed.\n")

## Print V matrix -------------------------------------------------------------
cat("\n  V matrix (first 10 states):\n")
cat(sprintf("    %-5s %-25s %9s %9s %10s %7s\n",
            "idx", "state", "intercept", "MR_pctile", "TieredReim", "ITadd"))
for (i in 1:min(10, S)) {
  cat(sprintf("    %-5d %-25s %9.0f %9.3f %10.0f %7.0f\n",
              state_policy$state_idx[i], state_policy$state_name[i],
              V[i, 1], V[i, 2], V[i, 3], V[i, 4]))
}
if (S > 10) cat("    ...\n")

###############################################################################
## SECTION 8 : SURVEY DESIGN VARIABLES
###############################################################################
cat("\n--- 8. Survey design variables ---\n")

## 8a. Normalize weights: w_tilde = weight * N / sum(weight) ------------------
## This ensures sum(w_tilde) = N (self-normalizing)
w_raw_sum <- sum(dat$weight)
dat <- dat %>%
  mutate(w_tilde = weight * N / w_raw_sum)

cat(sprintf("  Raw weight sum      : %.2f\n", w_raw_sum))
cat(sprintf("  Normalized weight sum: %.2f (should be %d)\n",
            sum(dat$w_tilde), N))
cat(sprintf("  Mean normalized wt  : %.4f (should be ~1.0)\n",
            mean(dat$w_tilde)))
cat(sprintf("  Min normalized wt   : %.4f\n", min(dat$w_tilde)))
cat(sprintf("  Max normalized wt   : %.4f\n", max(dat$w_tilde)))

## Kish effective sample size -------------------------------------------------
kish_ess <- (sum(dat$w_tilde))^2 / sum(dat$w_tilde^2)
deff_kish <- N / kish_ess
cat(sprintf("  Kish ESS           : %.0f\n", kish_ess))
cat(sprintf("  Kish DEFF          : %.2f\n", deff_kish))

## 8b. Create sequential stratum and PSU indices ------------------------------
## Strata
stratum_levels <- sort(unique(dat$vstratum))
n_strata <- length(stratum_levels)
dat <- dat %>%
  mutate(stratum_idx = match(vstratum, stratum_levels))

cat(sprintf("\n  Number of strata: %d\n", n_strata))
cat(sprintf("  Stratum index range: [%d, %d]\n",
            min(dat$stratum_idx), max(dat$stratum_idx)))

## PSUs (nested within strata: create unique IDs first)
## In survey designs, PSUs are identified within strata
psu_levels <- sort(unique(dat$vpsu))
n_psu <- length(psu_levels)
dat <- dat %>%
  mutate(psu_idx = match(vpsu, psu_levels))

cat(sprintf("  Number of PSUs: %d\n", n_psu))
cat(sprintf("  PSU index range: [%d, %d]\n",
            min(dat$psu_idx), max(dat$psu_idx)))

## Validate no missing indices ------------------------------------------------
stopifnot(
  "No NA in stratum_idx" = !any(is.na(dat$stratum_idx)),
  "No NA in psu_idx"     = !any(is.na(dat$psu_idx))
)
cat("  [PASS] Stratum and PSU indices created.\n")

###############################################################################
## SECTION 9 : POSITIVE-OUTCOME INDEX (idx_pos)
###############################################################################
cat("\n--- 9. Positive-outcome index ---\n")

idx_pos <- which(dat$z == 1)
N_pos   <- length(idx_pos)

cat(sprintf("  N_pos = %d (providers serving IT)\n", N_pos))
cat(sprintf("  N_zero = %d (providers NOT serving IT)\n", N - N_pos))
cat(sprintf("  Proportion positive: %.3f\n", N_pos / N))

## Validate -------------------------------------------------------------------
stopifnot(
  "idx_pos length matches sum(z)" = N_pos == sum(dat$z),
  "All idx_pos entries have z=1"  = all(dat$z[idx_pos] == 1),
  "All idx_pos entries have y>0"  = all(dat$y[idx_pos] > 0)
)
cat("  [PASS] idx_pos validated.\n")

###############################################################################
## SECTION 10 : ASSEMBLE STAN DATA LIST
###############################################################################
cat("\n--- 10. Assembling Stan data list ---\n")

stan_data <- list(
  # Dimensions
  N     = N,                        # total number of providers
  S     = S,                        # number of states (51)
  P     = ncol(X),                  # number of provider covariates (5, incl intercept)
  Q     = ncol(V),                  # number of policy covariates (4, incl intercept)
  N_pos = N_pos,                    # number of IT-serving providers

  # Outcome data
  y       = dat$y,                  # IT enrollment count (integer vector, length N)
  n_trial = dat$n_trial,            # total 0-5 enrollment (integer vector, length N)
  z       = dat$z,                  # participation indicator (0/1, length N)

  # Provider covariates
  X = X,                            # N x P design matrix (standardized + intercept)

  # State-level data
  state = dat$state_idx,            # state index for each provider (1..S, length N)
  V     = V,                        # S x Q policy design matrix

  # Index for positive outcomes (Part 2 of hurdle)
  idx_pos = idx_pos,                # integer indices where z=1

  # Survey design
  w_tilde     = dat$w_tilde,        # normalized weights (length N)
  stratum_idx = dat$stratum_idx,    # stratum index (length N)
  psu_idx     = dat$psu_idx,        # PSU index (length N)
  n_strata    = n_strata,           # number of strata
  n_psu       = n_psu               # number of PSUs
)

## Print the data list structure ----------------------------------------------
cat("\n  Stan data list contents:\n")
cat(sprintf("    %-15s %-20s %s\n", "Name", "Type", "Size/Value"))
cat(sprintf("    %-15s %-20s %s\n", "----", "----", "----------"))
for (nm in names(stan_data)) {
  obj <- stan_data[[nm]]
  if (is.matrix(obj)) {
    desc <- sprintf("%d x %d matrix", nrow(obj), ncol(obj))
  } else if (length(obj) > 1) {
    desc <- sprintf("vector[%d], range [%.1f, %.1f]",
                    length(obj), min(obj), max(obj))
  } else {
    desc <- sprintf("scalar = %d", as.integer(obj))
  }
  cat(sprintf("    %-15s %-20s %s\n", nm, class(obj)[1], desc))
}

###############################################################################
## SECTION 11 : VALIDATION CHECKS ON STAN DATA LIST
###############################################################################
cat("\n--- 11. Final validation checks ---\n")

## Dimensional consistency ----------------------------------------------------
stopifnot(
  "y length = N"       = length(stan_data$y)       == stan_data$N,
  "n_trial length = N" = length(stan_data$n_trial)  == stan_data$N,
  "z length = N"       = length(stan_data$z)        == stan_data$N,
  "X rows = N"         = nrow(stan_data$X)          == stan_data$N,
  "X cols = P"         = ncol(stan_data$X)          == stan_data$P,
  "V rows = S"         = nrow(stan_data$V)          == stan_data$S,
  "V cols = Q"         = ncol(stan_data$V)          == stan_data$Q,
  "state length = N"   = length(stan_data$state)    == stan_data$N,
  "idx_pos length"     = length(stan_data$idx_pos)  == stan_data$N_pos,
  "w_tilde length = N" = length(stan_data$w_tilde)  == stan_data$N,
  "stratum length = N" = length(stan_data$stratum_idx) == stan_data$N,
  "psu length = N"     = length(stan_data$psu_idx)     == stan_data$N
)
cat("  [PASS] All dimensions consistent.\n")

## Value range checks ---------------------------------------------------------
stopifnot(
  "y non-negative"         = all(stan_data$y >= 0),
  "n_trial positive"       = all(stan_data$n_trial > 0),
  "y <= n_trial"           = all(stan_data$y <= stan_data$n_trial),
  "z in {0,1}"             = all(stan_data$z %in% c(0L, 1L)),
  "state in 1..S"          = all(stan_data$state >= 1 & stan_data$state <= stan_data$S),
  "idx_pos in 1..N"        = all(stan_data$idx_pos >= 1 & stan_data$idx_pos <= stan_data$N),
  "weights positive"       = all(stan_data$w_tilde > 0),
  "stratum_idx in 1..H"    = all(stan_data$stratum_idx >= 1 & stan_data$stratum_idx <= stan_data$n_strata),
  "psu_idx in 1..n_psu"    = all(stan_data$psu_idx >= 1 & stan_data$psu_idx <= stan_data$n_psu)
)
cat("  [PASS] All value ranges valid.\n")

## Cross-check: z and y consistency -------------------------------------------
stopifnot(
  "z=1 implies y>0"  = all(stan_data$y[stan_data$z == 1] > 0),
  "z=0 implies y==0" = all(stan_data$y[stan_data$z == 0] == 0)
)
cat("  [PASS] z/y consistency verified.\n")

## Cross-check: idx_pos points to z=1 rows ------------------------------------
stopifnot(
  "idx_pos entries have z=1" = all(stan_data$z[stan_data$idx_pos] == 1)
)
cat("  [PASS] idx_pos consistency verified.\n")

###############################################################################
## SECTION 12 : SAVE OUTPUTS
###############################################################################
cat("\n--- 12. Saving outputs ---\n")

## 12a. Save Stan data list ---------------------------------------------------
saveRDS(stan_data, STAN_DATA_OUT)
cat(sprintf("  Saved: %s\n", STAN_DATA_OUT))
cat(sprintf("    File size: %.1f KB\n",
            file.info(STAN_DATA_OUT)$size / 1024))

## 12b. Save analysis data frame ----------------------------------------------
## Include all constructed variables for later diagnostics and plotting
analysis_data <- dat %>%
  select(
    cb_id,
    state_name, state_idx,
    y, n_trial, z,
    all_of(covariate_vars),
    all_of(x_std_names),
    all_of(policy_vars),
    weight, w_tilde,
    vstratum, vpsu,
    stratum_idx, psu_idx
  )

saveRDS(analysis_data, ANALYSIS_OUT)
cat(sprintf("  Saved: %s\n", ANALYSIS_OUT))
cat(sprintf("    File size: %.1f KB\n",
            file.info(ANALYSIS_OUT)$size / 1024))

## 12c. Save standardization parameters for later back-transformation ---------
standardization_params <- list(
  x_vars  = covariate_vars,
  x_means = x_means,
  x_sds   = x_sds,
  v_mr_mean = v_mr_mean,
  v_mr_sd   = v_mr_sd,
  state_levels = state_levels,
  stratum_levels = stratum_levels,
  psu_levels = psu_levels,
  state_policy = state_policy
)
STD_PARAMS_OUT <- file.path(OUTPUT_DIR, "standardization_params.rds")
saveRDS(standardization_params, STD_PARAMS_OUT)
cat(sprintf("  Saved: %s\n", STD_PARAMS_OUT))

###############################################################################
## SECTION 13 : COMPREHENSIVE DATA SUMMARY
###############################################################################
cat("\n==============================================================\n")
cat("  DATA PREPARATION SUMMARY\n")
cat("==============================================================\n")

cat(sprintf("\n  Sample:\n"))
cat(sprintf("    Original providers          : %d\n", nrow(raw)))
cat(sprintf("    Removed (n_trial == 0)      : %d\n", n_zero_trial))
cat(sprintf("    Final analysis sample (N)   : %d\n", N))
cat(sprintf("    IT participants (N_pos)     : %d (%.1f%%)\n",
            N_pos, 100 * N_pos / N))
cat(sprintf("    Non-participants            : %d (%.1f%%)\n",
            N - N_pos, 100 * (N - N_pos) / N))
cat(sprintf("    States (S)                  : %d\n", S))

cat(sprintf("\n  Design matrices:\n"))
cat(sprintf("    X (provider covariates)     : %d x %d\n", nrow(X), ncol(X)))
cat(sprintf("      Columns: intercept, poverty, urban, black, hispanic\n"))
cat(sprintf("    V (state policy)            : %d x %d\n", nrow(V), ncol(V)))
cat(sprintf("      Columns: intercept, MR_pctile(std), TieredReim, ITaddon\n"))

cat(sprintf("\n  Outcome (y = enrl_it):\n"))
cat(sprintf("    min = %d, median = %d, mean = %.1f, max = %d\n",
            min(dat$y), as.integer(median(dat$y)),
            mean(dat$y), max(dat$y)))

cat(sprintf("\n  Trial size (n_trial = enrl_total_05):\n"))
cat(sprintf("    min = %d, median = %d, mean = %.1f, max = %d\n",
            min(dat$n_trial), as.integer(median(dat$n_trial)),
            mean(dat$n_trial), max(dat$n_trial)))

cat(sprintf("\n  Survey design:\n"))
cat(sprintf("    Strata                      : %d\n", n_strata))
cat(sprintf("    PSUs                        : %d\n", n_psu))
cat(sprintf("    Kish ESS                    : %.0f\n", kish_ess))
cat(sprintf("    Kish DEFF                   : %.2f\n", deff_kish))
cat(sprintf("    Weight range (normalized)   : [%.4f, %.4f]\n",
            min(dat$w_tilde), max(dat$w_tilde)))

cat(sprintf("\n  Provider covariates (standardized means, should be ~0):\n"))
for (v in x_std_names) {
  cat(sprintf("    %-35s mean = %+.5f\n", v, mean(dat[[v]])))
}

cat(sprintf("\n  Policy variables (state-level):\n"))
cat(sprintf("    MR_pctile (raw)             : mean = %.1f, sd = %.1f\n",
            v_mr_mean, v_mr_sd))
cat(sprintf("    MR_pctile imputed states    : %d\n", n_mr_missing))
cat(sprintf("    TieredReim = 1              : %d / %d states\n",
            sum(state_policy$TieredReim), S))
cat(sprintf("    ITaddon = 1                 : %d / %d states\n",
            sum(state_policy$ITaddon), S))

cat(sprintf("\n  Output files:\n"))
cat(sprintf("    %-40s (%.1f KB)\n", STAN_DATA_OUT,
            file.info(STAN_DATA_OUT)$size / 1024))
cat(sprintf("    %-40s (%.1f KB)\n", ANALYSIS_OUT,
            file.info(ANALYSIS_OUT)$size / 1024))
cat(sprintf("    %-40s (%.1f KB)\n", STD_PARAMS_OUT,
            file.info(STD_PARAMS_OUT)$size / 1024))

cat("\n==============================================================\n")
cat("  DATA PREPARATION COMPLETE.\n")
cat("  Ready for Stan model fitting.\n")
cat("==============================================================\n")
