// =============================================================================
// M3b-W: Survey-Weighted Policy Moderator SVC Hurdle Beta-Binomial
// =============================================================================
// EXTENDS M3b by incorporating NSECE survey weights into the likelihood
// to form a pseudo-posterior (Savitsky & Toth, 2016; Williams & Savitsky, 2021).
//
// Key change from M3b:
//   M3b:  target += log_lik[i]                          (unweighted)
//   M3b-W: target += w_tilde[i] * log_lik[i]            (weighted pseudo-posterior)
//
// Weights:
//   w_tilde[i] = w[i] * N / sum(w)   (normalized to sum to N)
//   This ensures the pseudo-posterior is on the same "effective sample size"
//   scale as the unweighted posterior.
//
// Additional generated quantities:
//   - score_ext[i, 1:P]  : extensive-margin score vector (∂ℓ_i/∂α chain rule)
//   - score_int[i, 1:P]  : intensive-margin score vector (∂ℓ_i/∂β chain rule)
//   - score_kappa[i]     : dispersion score (∂ℓ_i/∂log_kappa)
//   These are needed for the cluster-robust sandwich variance estimator
//   computed in R (61_sandwich_variance.R).
//
// The sandwich correction procedure (Williams & Savitsky, 2021):
//   V_sand = H^{-1} J_cluster H^{-1}
//   Then Cholesky affine transform: θ* = θ̂ + L_sand L_MCMC^{-1} (θ - θ̂)
//   Applied in R via 62_cholesky_transform.R
//
// All other model structure identical to M3b:
//   - Two-part hurdle (Bernoulli + ZT-BetaBinomial)
//   - Policy moderators: delta[s] = Gamma * v[s] + epsilon[s]
//   - NCP for residuals
//   - Full K×K cross-margin correlation
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

  // --- Survey weights (NEW in M3b-W) ---
  vector<lower=0>[N] w_tilde;       // normalized survey weights: sum(w_tilde) = N
}

transformed data {
  int K = 2 * P;  // joint random effect dimension (extensive + intensive)
}

parameters {
  vector[P] alpha;                   // extensive-margin fixed effects (logit scale)
  vector[P] beta;                    // intensive-margin fixed effects (logit scale)
  real log_kappa;                    // log concentration parameter

  // --- Policy moderator coefficients ---
  matrix[K, Q] Gamma;

  // --- Residual random effects (NCP) ---
  vector<lower=0>[K] tau;            // residual scales
  cholesky_factor_corr[K] L_Omega;   // Cholesky factor of K×K residual correlation
  array[S] vector[K] z_eps;          // raw NCP variates
}

transformed parameters {
  real<lower=0> kappa = exp(log_kappa);

  // --- Recover state-varying coefficients ---
  // delta[s] = Gamma * v[s] + epsilon[s]
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
  // --- Priors: fixed effects ---
  alpha ~ normal(0, 2.5);
  beta ~ normal(0, 2.5);
  log_kappa ~ normal(log(10), 1.5);

  // --- Prior: policy moderator coefficients ---
  to_vector(Gamma) ~ normal(0, 1);

  // --- Priors: residual random effects ---
  tau ~ normal(0, 1);
  L_Omega ~ lkj_corr_cholesky(2);
  for (s in 1:S)
    z_eps[s] ~ std_normal();

  // --- Precompute fixed-effect linear predictors ---
  vector[N] eta_ext_fixed = X * alpha;
  vector[N] eta_int_fixed = X * beta;

  // --- WEIGHTED Likelihood (pseudo-posterior) ---
  // target += w_tilde[i] * log f(y_i | theta)
  for (i in 1:N) {
    real eta_ext_i = eta_ext_fixed[i] + X[i] * head(delta[state[i]], P);
    real eta_int_i = eta_int_fixed[i] + X[i] * tail(delta[state[i]], P);

    real q_i = inv_logit(eta_ext_i);
    real mu_i = inv_logit(eta_int_i);
    real a_i = mu_i * kappa;
    real b_i = (1 - mu_i) * kappa;

    if (z[i] == 0) {
      // --- Structural zero: weighted ---
      target += w_tilde[i] * log1m(q_i);

    } else {
      // --- Positive count: weighted ---
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
  // --- Pointwise log-likelihood (UNWEIGHTED for LOO-CV) ---
  vector[N] log_lik;

  // --- Posterior predictive replications ---
  array[N] int<lower=0> y_rep;

  // --- Residual correlation matrix ---
  corr_matrix[K] Omega = multiply_lower_tri_self_transpose(L_Omega);

  // --- Score vectors for sandwich variance (NEW in M3b-W) ---
  // These are UNWEIGHTED individual scores: s_i = ∂log f(y_i|θ)/∂θ
  // The weight multiplication and cluster aggregation happen in R.
  matrix[N, P] score_ext;           // extensive-margin score w.r.t. alpha
  matrix[N, P] score_int;           // intensive-margin score w.r.t. beta
  vector[N] score_kappa;            // dispersion score w.r.t. log_kappa

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

      // =========================================================
      // Extensive-margin score: s_i^ext = (z_i - q_i) * X_i
      // =========================================================
      score_ext[i] = (z[i] - q_i) * to_row_vector(X[i]);

      if (z[i] == 0) {
        // --- Structural zero ---
        log_lik[i] = log1m(q_i);
        score_int[i] = rep_row_vector(0, P);
        score_kappa[i] = 0;

      } else {
        // --- Positive count ---
        real log_p0_i = lgamma(b_i + n_trial[i]) + lgamma(kappa)
                      - lgamma(b_i) - lgamma(kappa + n_trial[i]);
        real log_1mp0_i = log1m_exp(log_p0_i);
        real log_fBB_i = lchoose(n_trial[i], y[i])
                       + lbeta(y[i] + a_i, n_trial[i] - y[i] + b_i)
                       - lbeta(a_i, b_i);
        real p0_i = exp(log_p0_i);

        log_lik[i] = log(q_i) + log_fBB_i - log_1mp0_i;

        // =========================================================
        // Intensive-margin score: s_i^int (CORRECTED MINUS sign)
        //   S_BB = κ [ψ(y+a) - ψ(n-y+b) - ψ(a) + ψ(b)]
        //   Λ_i  = κ [ψ(b+n) - ψ(b)]
        //   trunc_corr = p0 * Λ / (1-p0)
        //   score_mu = S_BB - trunc_corr    ← MINUS sign
        //   score_int = score_mu * μ(1-μ) * X_i
        // =========================================================
        real S_BB_mu = kappa * (digamma(y[i] + a_i)
                               - digamma(n_trial[i] - y[i] + b_i)
                               - digamma(a_i) + digamma(b_i));
        real Lambda_i = kappa * (digamma(b_i + n_trial[i]) - digamma(b_i));
        real trunc_corr_mu = p0_i * Lambda_i / (1 - p0_i);
        real score_mu_i = S_BB_mu - trunc_corr_mu;  // MINUS sign

        score_int[i] = score_mu_i * mu_i * (1 - mu_i) * to_row_vector(X[i]);

        // =========================================================
        // Dispersion score: s_i^κ  (w.r.t. log_kappa, MINUS sign)
        //   S_BB_kappa = κ[μ(ψ(y+a)-ψ(a)) + (1-μ)(ψ(n-y+b)-ψ(b)) + ψ(κ)-ψ(n+κ)]
        //   trunc_corr_kappa = p0*μ*κ/(1-p0) * Σ_{j=1}^{n-1} j/[(b+j)(κ+j)]
        //   score_kappa = S_BB_kappa - trunc_corr_kappa
        // =========================================================
        real S_BB_kappa = kappa * (
          mu_i * (digamma(y[i] + a_i) - digamma(a_i))
          + (1 - mu_i) * (digamma(n_trial[i] - y[i] + b_i) - digamma(b_i))
          + digamma(kappa) - digamma(n_trial[i] + kappa)
        );

        real trunc_sum = 0;
        for (j in 1:(n_trial[i] - 1)) {
          trunc_sum += 1.0 * j / ((b_i + j) * (kappa + j));
        }
        real trunc_corr_kappa = p0_i * mu_i * kappa * trunc_sum / (1 - p0_i);

        score_kappa[i] = S_BB_kappa - trunc_corr_kappa;  // MINUS sign
      }

      // --- Posterior predictive replication ---
      int z_rep_i = bernoulli_rng(q_i);

      if (z_rep_i == 0) {
        y_rep[i] = 0;
      } else {
        int draw = 0;
        int max_iter = 1000;
        int iter = 0;
        while (draw == 0 && iter < max_iter) {
          real p_draw = beta_rng(a_i, b_i);
          draw = binomial_rng(n_trial[i], p_draw);
          iter += 1;
        }
        y_rep[i] = (draw == 0) ? 1 : draw;
      }
    }
  }
}
