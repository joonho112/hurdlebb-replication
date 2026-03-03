// =============================================================================
// M1: Random Intercepts Hurdle Beta-Binomial
// =============================================================================
// Extends M0 by adding state-specific random intercepts to both margins.
//
// Two-part hurdle:
//   Part 1 (Extensive): Bernoulli — does provider serve IT at all?
//   Part 2 (Intensive):  Zero-truncated Beta-Binomial — IT share among servers
//
// State random intercepts:
//   (delta_ext[s], delta_int[s])' ~ N(0, Sigma_delta)
//   Sigma_delta = diag(tau) * Omega * diag(tau)
//
// Non-centered parameterization (NCP):
//   z_delta[k,s] ~ N(0,1)  iid, k=1,2; s=1,...,S
//   delta = diag(tau) * L_Omega * z_delta
//
// Covariates (P = 5): intercept + 4 standardized predictors
// =============================================================================

data {
  int<lower=1> N;                    // number of providers
  int<lower=1> P;                    // number of covariates (including intercept)
  int<lower=1> S;                    // number of states
  array[N] int<lower=0> y;           // IT enrollment count
  array[N] int<lower=1> n_trial;     // total 0-5 enrollment (trials)
  array[N] int<lower=0, upper=1> z;  // participation indicator: z[i] = 1 iff y[i] > 0
  matrix[N, P] X;                    // design matrix (col 1 = intercept)
  array[N] int<lower=1, upper=S> state;  // state index for each observation
}

parameters {
  vector[P] alpha;                   // extensive-margin fixed effects (logit scale)
  vector[P] beta;                    // intensive-margin fixed effects (logit scale)
  real log_kappa;                    // log concentration parameter

  // --- Random intercepts (NCP) ---
  vector<lower=0>[2] tau;            // scale of random intercepts: (tau_ext, tau_int)
  cholesky_factor_corr[2] L_Omega;   // Cholesky factor of 2x2 correlation matrix
  matrix[2, S] z_delta;              // raw standard normal variates for NCP
}

transformed parameters {
  real<lower=0> kappa = exp(log_kappa);

  // --- Recover actual random effects via NCP ---
  // delta[2, S] = diag_pre_multiply(tau, L_Omega) * z_delta
  //   row 1 = delta_ext[s], row 2 = delta_int[s]
  matrix[2, S] delta = diag_pre_multiply(tau, L_Omega) * z_delta;
}

model {
  // --- Priors: fixed effects ---
  alpha ~ normal(0, 2.5);
  beta ~ normal(0, 2.5);
  log_kappa ~ normal(log(10), 1.5);

  // --- Priors: random effects ---
  tau ~ normal(0, 1);               // half-normal (lower=0 enforced by constraint)
  L_Omega ~ lkj_corr_cholesky(2);   // LKJ(2) prior on correlation
  to_vector(z_delta) ~ std_normal(); // iid N(0,1) for NCP

  // --- Precompute fixed-effect linear predictors ---
  vector[N] eta_ext_fixed = X * alpha;
  vector[N] eta_int_fixed = X * beta;

  // --- Likelihood ---
  for (i in 1:N) {
    // Add state random intercept to fixed-effect linear predictor
    real eta_ext_i = eta_ext_fixed[i] + delta[1, state[i]];
    real eta_int_i = eta_int_fixed[i] + delta[2, state[i]];

    real q_i = inv_logit(eta_ext_i);       // P(serve IT)
    real mu_i = inv_logit(eta_int_i);      // conditional mean IT share
    real a_i = mu_i * kappa;               // BB shape parameter alpha
    real b_i = (1 - mu_i) * kappa;         // BB shape parameter beta

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

  // --- Correlation matrix (for reporting) ---
  corr_matrix[2] Omega = multiply_lower_tri_self_transpose(L_Omega);

  {
    vector[N] eta_ext_fixed = X * alpha;
    vector[N] eta_int_fixed = X * beta;

    for (i in 1:N) {
      // Add state random intercept
      real eta_ext_i = eta_ext_fixed[i] + delta[1, state[i]];
      real eta_int_i = eta_int_fixed[i] + delta[2, state[i]];

      real q_i = inv_logit(eta_ext_i);
      real mu_i = inv_logit(eta_int_i);
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
