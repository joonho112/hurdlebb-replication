// =============================================================================
// M3b-W-LKJ: Survey-Weighted HBB with parameterized LKJ eta
// =============================================================================
// IDENTICAL to hbb_m3b_weighted.stan EXCEPT:
//   1. lkj_eta is a data input (not hardcoded)
//   2. Score vectors REMOVED (not needed for sensitivity analysis)
//   3. y_rep REMOVED (not needed for sensitivity analysis)
// This lightweight version fits significantly faster.
// =============================================================================

data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> S;
  int<lower=1> Q;
  array[N] int<lower=0> y;
  array[N] int<lower=1> n_trial;
  array[N] int<lower=0, upper=1> z;
  matrix[N, P] X;
  array[N] int<lower=1, upper=S> state;
  matrix[S, Q] v_state;
  vector<lower=0>[N] w_tilde;

  // --- LKJ sensitivity parameter (NEW) ---
  real<lower=0> lkj_eta;
}

transformed data {
  int K = 2 * P;
}

parameters {
  vector[P] alpha;
  vector[P] beta;
  real log_kappa;
  matrix[K, Q] Gamma;
  vector<lower=0>[K] tau;
  cholesky_factor_corr[K] L_Omega;
  array[S] vector[K] z_eps;
}

transformed parameters {
  real<lower=0> kappa = exp(log_kappa);
  array[S] vector[K] delta;
  {
    matrix[K, K] L_Sigma = diag_pre_multiply(tau, L_Omega);
    for (s in 1:S) {
      vector[K] epsilon_s = L_Sigma * z_eps[s];
      delta[s] = Gamma * v_state[s]' + epsilon_s;
    }
  }
}

model {
  // --- Priors ---
  alpha ~ normal(0, 2.5);
  beta ~ normal(0, 2.5);
  log_kappa ~ normal(log(10), 1.5);
  to_vector(Gamma) ~ normal(0, 1);
  tau ~ normal(0, 1);
  L_Omega ~ lkj_corr_cholesky(lkj_eta);  // PARAMETERIZED
  for (s in 1:S)
    z_eps[s] ~ std_normal();

  // --- Weighted likelihood ---
  vector[N] eta_ext_fixed = X * alpha;
  vector[N] eta_int_fixed = X * beta;

  for (i in 1:N) {
    real eta_ext_i = eta_ext_fixed[i] + X[i] * head(delta[state[i]], P);
    real eta_int_i = eta_int_fixed[i] + X[i] * tail(delta[state[i]], P);
    real q_i = inv_logit(eta_ext_i);
    real mu_i = inv_logit(eta_int_i);
    real a_i = mu_i * kappa;
    real b_i = (1 - mu_i) * kappa;

    if (z[i] == 0) {
      target += w_tilde[i] * log1m(q_i);
    } else {
      real log_p0_i = lgamma(b_i + n_trial[i]) + lgamma(kappa)
                    - lgamma(b_i) - lgamma(kappa + n_trial[i]);
      real log_1mp0_i = log1m_exp(log_p0_i);
      real log_fBB_i = lchoose(n_trial[i], y[i])
                     + lbeta(y[i] + a_i, n_trial[i] - y[i] + b_i)
                     - lbeta(a_i, b_i);
      target += w_tilde[i] * (log(q_i) + log_fBB_i - log_1mp0_i);
    }
  }
}

generated quantities {
  // --- Correlation matrix (from Cholesky) ---
  corr_matrix[K] Omega = multiply_lower_tri_self_transpose(L_Omega);

  // --- Pointwise log-likelihood (unweighted, for LOO-CV) ---
  vector[N] log_lik;
  {
    vector[N] eta_ext_fixed = X * alpha;
    vector[N] eta_int_fixed = X * beta;
    for (i in 1:N) {
      real eta_ext_i = eta_ext_fixed[i] + X[i] * head(delta[state[i]], P);
      real eta_int_i = eta_int_fixed[i] + X[i] * tail(delta[state[i]], P);
      real q_i = inv_logit(eta_ext_i);
      real mu_i = inv_logit(eta_int_i);
      real a_i = mu_i * kappa;
      real b_i = (1 - mu_i) * kappa;

      if (z[i] == 0) {
        log_lik[i] = log1m(q_i);
      } else {
        real log_p0_i = lgamma(b_i + n_trial[i]) + lgamma(kappa)
                      - lgamma(b_i) - lgamma(kappa + n_trial[i]);
        real log_1mp0_i = log1m_exp(log_p0_i);
        real log_fBB_i = lchoose(n_trial[i], y[i])
                       + lbeta(y[i] + a_i, n_trial[i] - y[i] + b_i)
                       - lbeta(a_i, b_i);
        log_lik[i] = log(q_i) + log_fBB_i - log_1mp0_i;
      }
    }
  }
}
