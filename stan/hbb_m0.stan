// =============================================================================
// M0: Pooled Hurdle Beta-Binomial
// =============================================================================
// Simplest specification: pooled fixed effects, no state variation, no weights.
//
// Two-part hurdle:
//   Part 1 (Extensive): Bernoulli — does provider serve IT at all?
//   Part 2 (Intensive):  Zero-truncated Beta-Binomial — IT share among servers
//
// Covariates (P = 5): intercept + 4 standardized predictors
// =============================================================================

data {
  int<lower=1> N;                    // number of providers
  int<lower=1> P;                    // number of covariates (including intercept)
  array[N] int<lower=0> y;           // IT enrollment count
  array[N] int<lower=1> n_trial;     // total 0-5 enrollment (trials)
  array[N] int<lower=0, upper=1> z;  // participation indicator: z[i] = 1 iff y[i] > 0
  matrix[N, P] X;                    // design matrix (col 1 = intercept)
}

parameters {
  vector[P] alpha;       // extensive-margin coefficients (logit scale)
  vector[P] beta;        // intensive-margin coefficients (logit scale)
  real log_kappa;        // log concentration parameter
}

transformed parameters {
  real<lower=0> kappa = exp(log_kappa);
}

model {
  // --- Priors ---
  alpha ~ normal(0, 2.5);
  beta ~ normal(0, 2.5);
  log_kappa ~ normal(log(10), 1.5);

  // --- Precompute linear predictors ---
  vector[N] eta_ext = X * alpha;     // extensive-margin linear predictor
  vector[N] eta_int = X * beta;      // intensive-margin linear predictor

  // --- Likelihood ---
  for (i in 1:N) {
    real q_i = inv_logit(eta_ext[i]);       // P(serve IT)
    real mu_i = inv_logit(eta_int[i]);      // conditional mean IT share
    real a_i = mu_i * kappa;                // BB shape parameter alpha
    real b_i = (1 - mu_i) * kappa;          // BB shape parameter beta

    if (z[i] == 0) {
      // --- Structural zero: provider does not serve IT ---
      target += log1m(q_i);

    } else {
      // --- Positive count: log(q_i) + log_f_BB(y|n,a,b) - log(1 - p0) ---

      // log P(Y=0 | BB) via lgamma for numerical stability
      //   p0 = B(b+n, kappa) / B(b, kappa)
      //      = Gamma(b+n)*Gamma(kappa) / [Gamma(b)*Gamma(kappa+n)]
      real log_p0_i = lgamma(b_i + n_trial[i]) + lgamma(kappa)
                    - lgamma(b_i) - lgamma(kappa + n_trial[i]);

      // log(1 - p0): use log1m_exp which computes log(1 - exp(x)) for x <= 0
      real log_1mp0_i = log1m_exp(log_p0_i);

      // Beta-Binomial log-PMF:
      //   log_f_BB = lchoose(n, y) + lbeta(y + a, n - y + b) - lbeta(a, b)
      real log_fBB_i = lchoose(n_trial[i], y[i])
                     + lbeta(y[i] + a_i, n_trial[i] - y[i] + b_i)
                     - lbeta(a_i, b_i);

      // Hurdle contribution: log(q) + log_f_BB - log(1 - p0)
      target += log(q_i) + log_fBB_i - log_1mp0_i;
    }
  }
}

generated quantities {
  // --- Pointwise log-likelihood (for LOO-CV) ---
  vector[N] log_lik;

  // --- Posterior predictive replications ---
  array[N] int<lower=0> y_rep;

  {
    vector[N] eta_ext = X * alpha;
    vector[N] eta_int = X * beta;

    for (i in 1:N) {
      real q_i = inv_logit(eta_ext[i]);
      real mu_i = inv_logit(eta_int[i]);
      real a_i = mu_i * kappa;
      real b_i = (1 - mu_i) * kappa;

      if (z[i] == 0) {
        // --- Structural zero ---
        log_lik[i] = log1m(q_i);

      } else {
        // --- Positive count ---
        real log_p0_i = lgamma(b_i + n_trial[i]) + lgamma(kappa)
                      - lgamma(b_i) - lgamma(kappa + n_trial[i]);
        real log_1mp0_i = log1m_exp(log_p0_i);
        real log_fBB_i = lchoose(n_trial[i], y[i])
                       + lbeta(y[i] + a_i, n_trial[i] - y[i] + b_i)
                       - lbeta(a_i, b_i);

        log_lik[i] = log(q_i) + log_fBB_i - log_1mp0_i;
      }

      // --- Posterior predictive replication ---
      // Step 1: Draw participation
      int z_rep_i = bernoulli_rng(q_i);

      if (z_rep_i == 0) {
        y_rep[i] = 0;
      } else {
        // Step 2: Rejection sampling for zero-truncated Beta-Binomial
        //   Draw from BetaBin(n, a, b) via beta_rng + binomial_rng; reject if 0
        int draw = 0;
        int max_iter = 1000;   // safety cap to prevent infinite loops
        int iter = 0;
        while (draw == 0 && iter < max_iter) {
          real p_draw = beta_rng(a_i, b_i);
          draw = binomial_rng(n_trial[i], p_draw);
          iter += 1;
        }
        // If rejection sampling exhausted (extremely unlikely), set to 1
        y_rep[i] = (draw == 0) ? 1 : draw;
      }
    }
  }
}
