// =============================================================================
// M2: Block-Diagonal State-Varying Coefficients Hurdle Beta-Binomial
// =============================================================================
// Extends M1 by allowing ALL P covariates to vary by state for both margins.
//
// Two-part hurdle:
//   Part 1 (Extensive): Bernoulli — does provider serve IT at all?
//   Part 2 (Intensive):  Zero-truncated Beta-Binomial — IT share among servers
//
// State-varying coefficients (SVC):
//   delta_ext[s] ~ N(0, Sigma_ext)      s = 1,...,S
//   delta_int[s] ~ N(0, Sigma_int)      s = 1,...,S
//
//   Two INDEPENDENT P×P covariance matrices (block-diagonal structure):
//     Sigma_ext = diag(tau_ext) * Omega_ext * diag(tau_ext)
//     Sigma_int = diag(tau_int) * Omega_int * diag(tau_int)
//   Cross-margin covariance is ZERO here (deferred to M3a).
//
// Linear predictors:
//   eta_ext[i] = X[i] * (alpha + delta_ext[state[i]])
//   eta_int[i] = X[i] * (beta  + delta_int[state[i]])
//
// Non-centered parameterization (NCP):
//   z_ext[s] ~ N(0, I_P)   iid across s
//   z_int[s] ~ N(0, I_P)   iid across s
//   delta_ext[s] = L_Sigma_ext * z_ext[s]
//   delta_int[s] = L_Sigma_int * z_int[s]
//   where L_Sigma = diag_pre_multiply(tau, L_Omega)
//
// Covariates (P = 5): intercept + 4 standardized predictors
// Total parameters: 5+5+1+5+5+10+10 + 51*5 + 51*5 = 551
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

  // --- Extensive margin random effects (NCP) ---
  vector<lower=0>[P] tau_ext;        // scales for extensive SVC
  cholesky_factor_corr[P] L_ext;     // Cholesky factor of P×P correlation matrix
  array[S] vector[P] z_ext;          // raw NCP variates for extensive margin

  // --- Intensive margin random effects (NCP) ---
  vector<lower=0>[P] tau_int;        // scales for intensive SVC
  cholesky_factor_corr[P] L_int;     // Cholesky factor of P×P correlation matrix
  array[S] vector[P] z_int;          // raw NCP variates for intensive margin
}

transformed parameters {
  real<lower=0> kappa = exp(log_kappa);

  // --- Recover state-varying coefficient deviations via NCP ---
  // delta_ext[s] = diag_pre_multiply(tau_ext, L_ext) * z_ext[s]   (vector[P])
  // delta_int[s] = diag_pre_multiply(tau_int, L_int) * z_int[s]   (vector[P])
  array[S] vector[P] delta_ext;
  array[S] vector[P] delta_int;
  {
    matrix[P, P] L_Sigma_ext = diag_pre_multiply(tau_ext, L_ext);
    matrix[P, P] L_Sigma_int = diag_pre_multiply(tau_int, L_int);
    for (s in 1:S) {
      delta_ext[s] = L_Sigma_ext * z_ext[s];
      delta_int[s] = L_Sigma_int * z_int[s];
    }
  }
}

model {
  // --- Priors: fixed effects ---
  alpha ~ normal(0, 2.5);
  beta ~ normal(0, 2.5);
  log_kappa ~ normal(log(10), 1.5);

  // --- Priors: extensive margin random effects ---
  tau_ext ~ normal(0, 1);               // half-normal (lower=0 enforced by constraint)
  L_ext ~ lkj_corr_cholesky(2);         // LKJ(2) prior on P×P correlation
  for (s in 1:S)
    z_ext[s] ~ std_normal();             // iid N(0,1) for NCP

  // --- Priors: intensive margin random effects ---
  tau_int ~ normal(0, 1);               // half-normal (lower=0 enforced by constraint)
  L_int ~ lkj_corr_cholesky(2);         // LKJ(2) prior on P×P correlation
  for (s in 1:S)
    z_int[s] ~ std_normal();             // iid N(0,1) for NCP

  // --- Precompute fixed-effect linear predictors ---
  vector[N] eta_ext_fixed = X * alpha;
  vector[N] eta_int_fixed = X * beta;

  // --- Likelihood ---
  for (i in 1:N) {
    // State-varying linear predictor:
    //   eta = X[i] * alpha + X[i] * delta[state[i]]
    //   X[i] is row_vector[P], delta[s] is vector[P], product is scalar
    real eta_ext_i = eta_ext_fixed[i] + X[i] * delta_ext[state[i]];
    real eta_int_i = eta_int_fixed[i] + X[i] * delta_int[state[i]];

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

  // --- Correlation matrices (for reporting) ---
  corr_matrix[P] Omega_ext = multiply_lower_tri_self_transpose(L_ext);
  corr_matrix[P] Omega_int = multiply_lower_tri_self_transpose(L_int);

  {
    vector[N] eta_ext_fixed = X * alpha;
    vector[N] eta_int_fixed = X * beta;

    for (i in 1:N) {
      // State-varying linear predictor
      real eta_ext_i = eta_ext_fixed[i] + X[i] * delta_ext[state[i]];
      real eta_int_i = eta_int_fixed[i] + X[i] * delta_int[state[i]];

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
