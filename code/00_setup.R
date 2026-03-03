## =============================================================================
## 00_setup.R -- Environment Setup and Dependency Check
## =============================================================================
## Purpose : Verify that the computing environment is ready for replication.
##           Checks R version, installs missing packages, verifies CmdStan
##           toolchain, and compiles/fits a trivial Stan model to confirm
##           everything works end-to-end.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Usage   : source("code/00_setup.R")
##           (from the replication package root directory)
## =============================================================================

cat("==============================================================\n")
cat("  Replication Package: Environment Setup\n")
cat("==============================================================\n\n")


## =============================================================================
## Section 1 : Project Root Detection
## =============================================================================

## Detect project root portably using here::here() or fallback
if (requireNamespace("here", quietly = TRUE)) {
    PROJECT_ROOT <- here::here()
} else {
    ## Fallback: assume the script is run from the replication root
    PROJECT_ROOT <- getwd()
}
cat(sprintf("[Setup] Project root: %s\n\n", PROJECT_ROOT))


## =============================================================================
## Section 2 : R Version Check
## =============================================================================

cat("--- 1. R Version Check ---\n")
r_ver <- getRversion()
cat(sprintf("  R version detected : %s\n", as.character(r_ver)))

if (r_ver < "4.3") {
    stop(
        sprintf(
            "R version >= 4.3 is required (found %s). Please upgrade R.",
            as.character(r_ver)
        ),
        call. = FALSE
    )
} else {
    cat("  [PASS] R version >= 4.3\n\n")
}


## =============================================================================
## Section 3 : Package Installation
## =============================================================================

cat("--- 2. Required Packages ---\n")

## Define required CRAN packages -----------------------------------------------
cran_pkgs <- c(
    "posterior",    # posterior summaries & diagnostics
    "loo",         # LOO-CV / WAIC
    "survey",      # complex survey design
    "dplyr",       # data wrangling
    "tidyr",       # data reshaping
    "readr",       # fast file I/O
    "ggplot2",     # plotting
    "patchwork",   # multi-panel plots
    "xtable",      # LaTeX table export
    "scales",      # axis scale formatting
    "viridis",     # colorblind-friendly palettes
    "sf",          # spatial (for maps)
    "maps",        # US state map data
    "ggrepel",     # non-overlapping text labels
    "bayesplot",   # MCMC diagnostic plots
    "forcats",     # factor manipulation
    "MASS",        # statistical functions
    "here"         # portable project paths
)

## Helper: install a missing CRAN package --------------------------------------
install_if_missing <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat(sprintf("  [INSTALL] Installing '%s' from CRAN ...\n", pkg))
        install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
        if (!requireNamespace(pkg, quietly = TRUE)) {
            stop(sprintf("Failed to install package '%s'.", pkg), call. = FALSE)
        }
        cat(sprintf("  [OK]      '%s' installed successfully.\n", pkg))
    }
}

## Check / install CRAN packages -----------------------------------------------
for (pkg in cran_pkgs) {
    install_if_missing(pkg)
}

## Check / install cmdstanr (from stan-dev R-universe) -------------------------
cmdstanr_installed <- requireNamespace("cmdstanr", quietly = TRUE)

if (!cmdstanr_installed) {
    cat("  [INSTALL] Installing 'cmdstanr' from stan-dev R-universe ...\n")
    install.packages(
        "cmdstanr",
        repos = c("https://stan-dev.r-universe.dev",
                   "https://cloud.r-project.org"),
        quiet = TRUE
    )
    if (!requireNamespace("cmdstanr", quietly = TRUE)) {
        stop(
            "Failed to install 'cmdstanr'. ",
            "Try: install.packages('cmdstanr', repos = c(",
            "'https://stan-dev.r-universe.dev', 'https://cloud.r-project.org'))",
            call. = FALSE
        )
    }
    cat("  [OK]      'cmdstanr' installed successfully.\n")
}

## Print package versions ------------------------------------------------------
all_pkgs <- c("cmdstanr", cran_pkgs)
pkg_versions <- vapply(all_pkgs, function(p) {
    as.character(packageVersion(p))
}, character(1))

cat("\n  Package versions:\n")
for (i in seq_along(all_pkgs)) {
    cat(sprintf("    %-12s  %s\n", all_pkgs[i], pkg_versions[i]))
}
cat("  [PASS] All required R packages available.\n\n")


## =============================================================================
## Section 4 : CmdStan Installation Check
## =============================================================================

cat("--- 3. CmdStan Check ---\n")
library(cmdstanr)

cmdstan_path_found <- tryCatch(
    {
        cmdstan_path()
        TRUE
    },
    error = function(e) FALSE
)

if (!cmdstan_path_found) {
    cat("\n")
    cat("  [ERROR] CmdStan is not installed.\n")
    cat("\n")
    cat("  CmdStan is required to compile and run Stan models.\n")
    cat("  To install CmdStan, run the following in R:\n")
    cat("\n")
    cat("    library(cmdstanr)\n")
    cat("    check_cmdstan_toolchain(fix = TRUE)\n")
    cat("    install_cmdstan(cores = parallel::detectCores())\n")
    cat("\n")
    cat("  This may take 5-15 minutes on first installation.\n")
    cat("  For detailed instructions, see:\n")
    cat("    https://mc-stan.org/cmdstanr/articles/cmdstanr.html\n")
    cat("\n")
    stop("CmdStan not found. Please install it before proceeding.",
         call. = FALSE)
}

## Verify CmdStan version ------------------------------------------------------
cs_path <- cmdstan_path()
cs_ver  <- cmdstan_version()
cat(sprintf("  CmdStan path    : %s\n", cs_path))
cat(sprintf("  CmdStan version : %s\n", cs_ver))

if (cs_ver < "2.33") {
    stop(
        sprintf(
            "CmdStan >= 2.33 is required (found %s). Run install_cmdstan().",
            cs_ver
        ),
        call. = FALSE
    )
} else {
    cat("  [PASS] CmdStan version >= 2.33\n\n")
}


## =============================================================================
## Section 5 : Stan Toolchain Smoke Test
## =============================================================================

cat("--- 4. Stan Toolchain Smoke Test ---\n")
cat("  Compiling and fitting a trivial normal-normal model ...\n")

## Write the test model to a temporary file ------------------------------------
test_stan_code <- "
data {
  int<lower=1> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 10);
  sigma ~ cauchy(0, 5);
  y ~ normal(mu, sigma);
}
"

test_stan_file <- file.path(tempdir(), "test_normal_normal.stan")
writeLines(test_stan_code, test_stan_file)

## Compile ---------------------------------------------------------------------
test_model <- tryCatch(
    cmdstan_model(test_stan_file),
    error = function(e) {
        stop(
            "Stan model compilation FAILED. Toolchain issue.\n",
            "  Error: ", conditionMessage(e), "\n",
            "  Try: cmdstanr::check_cmdstan_toolchain(fix = TRUE)",
            call. = FALSE
        )
    }
)
cat("  [PASS] Compilation successful.\n")

## Generate fake data and fit --------------------------------------------------
set.seed(42)
N_test <- 100
y_test <- rnorm(N_test, mean = 3.0, sd = 1.5)
test_data <- list(N = N_test, y = y_test)

test_fit <- tryCatch(
    test_model$sample(
        data            = test_data,
        seed            = 42,
        chains          = 4,
        parallel_chains = min(4, parallel::detectCores()),
        iter_warmup     = 500,
        iter_sampling   = 500,
        refresh         = 0,
        show_messages   = FALSE,
        show_exceptions = FALSE
    ),
    error = function(e) {
        stop(
            "Stan sampling FAILED.\n",
            "  Error: ", conditionMessage(e),
            call. = FALSE
        )
    }
)
cat("  [PASS] Sampling successful.\n")

## Quick diagnostics -----------------------------------------------------------
test_summary <- test_fit$summary(variables = c("mu", "sigma"))
cat("\n  Posterior summary (true mu=3.0, sigma=1.5):\n")
print(test_summary)

mu_hat    <- test_summary$mean[test_summary$variable == "mu"]
sigma_hat <- test_summary$mean[test_summary$variable == "sigma"]

cat(sprintf("  mu estimate    : %.3f  (true: 3.000)\n", mu_hat))
cat(sprintf("  sigma estimate : %.3f  (true: 1.500)\n", sigma_hat))
cat("  [PASS] Smoke test complete.\n\n")


## =============================================================================
## Section 6 : Environment Summary
## =============================================================================

cat("==============================================================\n")
cat("  ENVIRONMENT SUMMARY\n")
cat("==============================================================\n")
cat(sprintf("  Date/Time       : %s\n", Sys.time()))
cat(sprintf("  Platform        : %s\n", R.version$platform))
cat(sprintf("  R version       : %s  [PASS]\n", as.character(r_ver)))
cat(sprintf("  CmdStan version : %s  [PASS]\n", cs_ver))
cat(sprintf("  CmdStan path    : %s\n", cs_path))
cat(sprintf("  CPU cores       : %d\n", parallel::detectCores()))
cat(sprintf("  Project root    : %s\n", PROJECT_ROOT))
cat("\n  R Packages:\n")
for (i in seq_along(all_pkgs)) {
    cat(sprintf("    %-12s  %s\n", all_pkgs[i], pkg_versions[i]))
}
cat("\n  Stan toolchain  : [PASS] (trivial model compiled & fitted)\n")
cat("==============================================================\n")
cat("  ALL CHECKS PASSED. Environment is ready for replication.\n")
cat("==============================================================\n")

## Clean up temporary files ----------------------------------------------------
if (file.exists(test_stan_file)) file.remove(test_stan_file)
