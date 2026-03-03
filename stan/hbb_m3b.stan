// =============================================================================
// M3b: Policy Moderators — Cross-Margin SVC Hurdle Beta-Binomial
// =============================================================================
// EXTENDS M3a by adding state-level policy covariates that EXPLAIN variation
// in the state-varying coefficients (random effects).
//
// Two-part hurdle:
//   Part 1 (Extensive): Bernoulli — does provider serve IT at all?
//   Part 2 (Intensive):  Zero-truncated Beta-Binomial — IT share among servers
//
// Key change from M3a:
//   M3a: delta[s] = L_Sigma * z_eps[s]                 (mean-zero random effects)
//   M3b: delta[s] = Gamma * v[s] + epsilon[s]          (policy-moderated random effects)
//
//   where:
//     Gamma       = K x Q matrix of policy moderator coefficients
//     v[s]        = Q-vector of state-level policy covariates (col 1 = intercept)
//     Gamma * v[s]  = policy-explained component of state variation (vector[K])
//     epsilon[s]  ~ N(0, Sigma_epsilon) = residual state variation
//     Sigma_epsilon = diag(tau) * Omega * diag(tau)
//
// Policy covariates (Q = 4):
//   v_state[s, 1] = 1 (intercept)
//   v_state[s, 2] = MR_pctile_std (CCDF market rate percentile, standardized)
//   v_state[s, 3] = TieredReim (tiered reimbursement, binary 0/1)
//   v_state[s, 4] = ITaddon (IT differential rate, binary 0/1)
//
// Gamma structure (K x Q = 10 x 4 = 40 new parameters):
//   Gamma[1:P, :]       = how policy covariates moderate EXTENSIVE margin coefficients
//   Gamma[(P+1):K, :]   = how policy covariates moderate INTENSIVE margin coefficients
//
// Identifiability note:
//   v_state column 1 = all 1s (intercept). The Gamma[,1] column absorbs the
//   mean of delta across states. The residual epsilon[s] is truly mean-zero.
//   alpha and beta remain identifiable as individual-level fixed effects.
//
// NCP for residuals (same structure as M3a, but applied to residuals):
//   z_eps[s] ~ N(0, I_K)   for s = 1,...,S
//   epsilon[s] = diag_pre_multiply(tau, L_Omega) * z_eps[s]
//   delta[s]   = Gamma * v_state[s]' + epsilon[s]
//
// Linear predictors (identical to M3a):
//   eta_ext[i] = X[i] * (alpha + delta[state[i]][1:P])
//   eta_int[i] = X[i] * (beta  + delta[state[i]][(P+1):K])
//
// Covariates: P = 5 (intercept + 4 standardized predictors)
// K = 2*P = 10 (joint random effect dimension)
// Total parameters: 5+5+1+40+10+45 + 51*10 = 616
//   (40 = Gamma elements, 45 = free elements in 10x10 Cholesky factor)
//   M3a had 576 params; M3b adds 40 Gamma parameters
// =============================================================================

data {
  int<lower=1> N;                    // number of providers
  int<lower=1> P;                    // number of provider covariates (including intercept)
  int<lower=1> S;                    // number of states
  int<lower=1> Q;                    // number of state-level policy covariates (including intercept)
  array[N] int<lower=0> y;           // IT enrollment count
  array[N] int<lower=1> n_trial;     // total 0-5 enrollment (trials)
  array[N] int<lower=0, upper=1> z;  // participation indicator: z[i] = 1 iff y[i] > 0
  matrix[N, P] X;                    // provider-level design matrix (col 1 = intercept)
  array[N] int<lower=1, upper=S> state;  // state index for each observation
  matrix[S, Q] v_state;             // state-level policy design matrix (col 1 = intercept)
}

transformed data {
  int K = 2 * P;  // joint random effect dimension (extensive + intensive)
}

parameters {
  vector[P] alpha;                   // extensive-margin fixed effects (logit scale)
  vector[P] beta;                    // intensive-margin fixed effects (logit scale)
  real log_kappa;                    // log concentration parameter

  // --- Policy moderator coefficients ---
  // Gamma[k, q] = effect of policy covariate q on random effect dimension k
  //   Rows 1:P     → extensive margin
  //   Rows (P+1):K → intensive margin
  matrix[K, Q] Gamma;

  // --- Residual random effects (NCP) ---
  vector<lower=0>[K] tau;            // residual scales: tau[1:P] = ext, tau[(P+1):K] = int
  cholesky_factor_corr[K] L_Omega;   // Cholesky factor of K×K residual correlation matrix
  array[S] vector[K] z_eps;          // raw NCP variates (K-dimensional per state)
}

transformed parameters {
  real<lower=0> kappa = exp(log_kappa);

  // --- Recover state-varying coefficients ---
  // delta[s] = Gamma * v[s] + epsilon[s]
  //   Gamma * v[s] = policy-explained component
  //   epsilon[s]   = residual state variation (via NCP)
  array[S] vector[K] delta;
  {
    matrix[K, K] L_Sigma = diag_pre_multiply(tau, L_Omega);
    for (s in 1:S) {
      // epsilon[s] = L_Sigma * z_eps[s]  (residual, mean-zero)
      vector[K] epsilon_s = L_Sigma * z_eps[s];
      // delta[s] = policy-explained + residual
      // v_state[s] is row_vector[Q]; transpose to column vector for matrix multiply
      delta[s] = Gamma * v_state[s]' + epsilon_s;
    }
  }
}

model {
  // --- Priors: fixed effects ---
  alpha ~ normal(0, 2.5);
  beta ~ normal(0, 2.5);
  log_kappa ~ normal(log(10), 1.5);

  // --- Prior: policy moderator coefficients ---
  // Default weak prior; each element of Gamma ~ N(0, 1)
  to_vector(Gamma) ~ normal(0, 1);

  // --- Priors: residual random effects ---
  tau ~ normal(0, 1);                 // half-normal (lower=0 enforced by constraint)
  L_Omega ~ lkj_corr_cholesky(2);    // LKJ(2) prior on K×K residual correlation
  for (s in 1:S)
    z_eps[s] ~ std_normal();          // iid N(0, I_K) for NCP

  // --- Precompute fixed-effect linear predictors ---
  vector[N] eta_ext_fixed = X * alpha;
  vector[N] eta_int_fixed = X * beta;

  // --- Likelihood ---
  for (i in 1:N) {
    // Extract margin-specific deviations from joint delta vector
    //   head(delta[s], P)  = delta_ext (first P elements)
    //   tail(delta[s], P)  = delta_int (last P elements)
    real eta_ext_i = eta_ext_fixed[i] + X[i] * head(delta[state[i]], P);
    real eta_int_i = eta_int_fixed[i] + X[i] * tail(delta[state[i]], P);

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

  // --- FULL K×K residual correlation matrix ---
  // Omega has block structure:
  //   Omega[1:P, 1:P]           = within-extensive residual correlations
  //   Omega[(P+1):K, (P+1):K]   = within-intensive residual correlations
  //   Omega[1:P, (P+1):K]       = cross-margin residual correlations
  // NOTE: This is the RESIDUAL correlation (after removing policy effects).
  //       Compare with M3a's Omega to see how much policy covariates explain.
  corr_matrix[K] Omega = multiply_lower_tri_self_transpose(L_Omega);

  {
    vector[N] eta_ext_fixed = X * alpha;
    vector[N] eta_int_fixed = X * beta;

    for (i in 1:N) {
      // Extract margin-specific deviations from joint delta vector
      real eta_ext_i = eta_ext_fixed[i] + X[i] * head(delta[state[i]], P);
      real eta_int_i = eta_int_fixed[i] + X[i] * tail(delta[state[i]], P);

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
