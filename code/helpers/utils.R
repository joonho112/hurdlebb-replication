## =============================================================================
## utils.R -- Shared Utility Functions
## =============================================================================
## Purpose : Helper functions for beta-binomial computations, data transformations,
##           and common statistical operations used throughout the replication code.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================


###############################################################################
##
##  Contents:
##    1. log_betabinom_pmf        -- Log PMF of Beta-Binomial
##    2. log_zt_betabinom_pmf     -- Log PMF of Zero-Truncated Beta-Binomial
##    3. log1mexp                 -- Numerically stable log(1 - exp(x))
##    4. rztbetabinom             -- Random draws from ZT-BetaBinomial
##    5. normalize_weights        -- Normalize survey weights
##    6. inv_logit                -- Inverse logit
##    7. logit                    -- Logit
##    8. compute_zt_betabinom_mean -- Analytical mean of ZT-BetaBin
##    9. compute_betabinom_p0     -- P(Y=0) for BetaBin
##
###############################################################################


## =============================================================================
## Section 1 : Beta-Binomial PMF Functions
## =============================================================================

# -- 1. Log PMF of Beta-Binomial ----------------------------------------------
#
# BetaBin(y | n, mu, kappa):
#   a = mu * kappa,  b = (1 - mu) * kappa
#   log f = lchoose(n, y) + lbeta(y + a, n - y + b) - lbeta(a, b)
#
# Vectorised over y, n, mu, kappa (recycled via standard R rules).
# Returns -Inf for invalid inputs (y < 0, y > n, etc.).

log_betabinom_pmf <- function(y, n, mu, kappa) {

    ## Input validation --------------------------------------------------
    if (any(kappa <= 0, na.rm = TRUE))
        stop("kappa must be > 0")
    if (any(mu < 0 | mu > 1, na.rm = TRUE))
        stop("mu must be in [0, 1]")

    a <- mu * kappa
    b <- (1 - mu) * kappa

    log_f <- lchoose(n, y) + lbeta(y + a, n - y + b) - lbeta(a, b)

    ## Enforce support: y in {0, 1, ..., n}
    invalid <- (y < 0) | (y > n) | (y != round(y))
    log_f[invalid] <- -Inf

    log_f
}


# -- 2. Log PMF of Zero-Truncated Beta-Binomial -------------------------------
#
# ZT-BetaBin(y | n, mu, kappa) for y in {1, ..., n}:
#   log p0 via lgamma (numerically stable),
#   log f_zt = log_betabinom_pmf(y, n, mu, kappa) - log(1 - exp(log_p0))
#
# Uses log1mexp() = log(1 - exp(x)) for x < 0, avoiding catastrophic
# cancellation when p0 is near 0 or near 1.

log_zt_betabinom_pmf <- function(y, n, mu, kappa) {

    if (any(y < 1, na.rm = TRUE))
        stop("y must be > 0 for zero-truncated distribution")

    b <- (1 - mu) * kappa

    ## log P(Y = 0) via lgamma identity
    log_p0 <- lgamma(b + n) + lgamma(kappa) - lgamma(b) - lgamma(kappa + n)

    ## log(1 - p0) with numerically stable log1mexp
    ## log_p0 is always <= 0 (it's a log-probability)
    log_1mp0 <- log1mexp(-log_p0)

    log_f <- log_betabinom_pmf(y, n, mu, kappa)
    log_f_zt <- log_f - log_1mp0

    ## Enforce support: y in {1, ..., n}
    invalid <- (y < 1) | (y > n) | (y != round(y))
    log_f_zt[invalid] <- -Inf

    log_f_zt
}


## =============================================================================
## Section 2 : Numerical Stability Helpers
## =============================================================================

# -- 3. log(1 - exp(x)) for x < 0, from Machler (2012) -----------------------

log1mexp <- function(x) {
    ## x should be non-negative (we negate log_p0 above)
    out <- rep(NA_real_, length(x))
    small <- x <= log(2)
    out[small]  <- log(-expm1(-x[small]))
    out[!small] <- log1p(-exp(-x[!small]))
    out
}


## =============================================================================
## Section 3 : Random Variate Generation
## =============================================================================

# -- 4. Random ZT-BetaBinomial via rejection sampling -------------------------
#
# Draws from BetaBin(size, mu, kappa); rejects zeros.
# Falls back to manual Beta-Binomial if VGAM is unavailable.
# All arguments are scalar (the outer n_draws controls replication).

rztbetabinom <- function(n_draws, size, mu, kappa) {

    if (length(size) != 1 || length(mu) != 1 || length(kappa) != 1)
        stop("size, mu, kappa must each be scalar")
    if (size < 1) stop("size must be >= 1")
    if (mu <= 0 || mu >= 1) stop("mu must be in (0, 1) for ZT-BB sampling")
    if (kappa <= 0) stop("kappa must be > 0")

    a <- mu * kappa
    b <- (1 - mu) * kappa

    ## Choose BetaBin sampler ------------------------------------------------
    has_vgam <- requireNamespace("VGAM", quietly = TRUE)

    rbetabinom_one <- if (has_vgam) {
        function(nn) VGAM::rbetabinom.ab(nn, size = size, shape1 = a, shape2 = b)
    } else {
        function(nn) {
            p <- rbeta(nn, shape1 = a, shape2 = b)
            rbinom(nn, size = size, prob = p)
        }
    }

    ## Rejection sampling ----------------------------------------------------
    out   <- integer(0)
    max_iter <- 1000L  # safety cap on total batches
    iter  <- 0L

    while (length(out) < n_draws && iter < max_iter) {
        iter <- iter + 1L
        need <- n_draws - length(out)
        ## Over-draw by factor 1/(1-p0) + margin to reduce iterations
        p0_approx <- exp(lgamma(b + size) + lgamma(a + b) -
                         lgamma(b) - lgamma(a + b + size))
        draw_n <- ceiling(need / max(1 - p0_approx, 0.01)) + 10L
        draws  <- rbetabinom_one(draw_n)
        draws  <- draws[draws > 0L]
        out    <- c(out, draws)
    }

    if (length(out) < n_draws) {
        warning("Rejection sampling did not produce enough non-zero draws; ",
                "returning ", length(out), " draws (requested ", n_draws, ")")
    }

    out[seq_len(min(n_draws, length(out)))]
}


## =============================================================================
## Section 4 : Survey Weight Utilities
## =============================================================================

# -- 5. Normalize survey weights -----------------------------------------------
#
# Rescale so that sum(w_tilde) = length(w).

normalize_weights <- function(w) {
    if (any(w <= 0, na.rm = TRUE))
        stop("All weights must be positive")
    w * length(w) / sum(w)
}


## =============================================================================
## Section 5 : Link Functions
## =============================================================================

# -- 6. Inverse logit ---------------------------------------------------------

inv_logit <- function(x) {
    plogis(x)
}


# -- 7. Logit ------------------------------------------------------------------

logit <- function(p) {
    qlogis(p)
}


## =============================================================================
## Section 6 : Moment Computations
## =============================================================================

# -- 8. Analytical mean of ZT-BetaBin -----------------------------------------
#
# E[Y | Y > 0] = n * mu / (1 - p0)
# where p0 = P(Y = 0) under BetaBin(n, mu, kappa).
# Vectorised over n, mu, kappa.

compute_zt_betabinom_mean <- function(n, mu, kappa) {
    p0 <- compute_betabinom_p0(n, mu, kappa)
    n * mu / (1 - p0)
}


# -- 9. P(Y = 0) for BetaBin --------------------------------------------------
#
# p0 = B(b + n, a) / B(b, a)   where a = mu*kappa, b = (1-mu)*kappa
#    = exp( lgamma(b+n) + lgamma(kappa) - lgamma(b) - lgamma(kappa+n) )
#
# Vectorised over n, mu, kappa.

compute_betabinom_p0 <- function(n, mu, kappa) {
    if (any(kappa <= 0, na.rm = TRUE)) stop("kappa must be > 0")
    if (any(mu < 0 | mu > 1, na.rm = TRUE)) stop("mu must be in [0, 1]")

    b <- (1 - mu) * kappa

    log_p0 <- lgamma(b + n) + lgamma(kappa) - lgamma(b) - lgamma(kappa + n)
    exp(log_p0)
}
