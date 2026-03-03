// =============================================================================
// M1-W: Survey-Weighted Random Intercepts Hurdle Beta-Binomial
// =============================================================================
// EXTENDS M1 by incorporating NSECE survey weights into the likelihood
// to form a pseudo-posterior (Savitsky & Toth, 2016; Williams & Savitsky, 2021).
//
// Key change from M1:
//   M1:   target += log_lik[i]                          (unweighted)
//   M1-W: target += w_tilde[i] * log_lik[i]            (weighted pseudo-posterior)
//
// Weights:
//   w_tilde[i] = w[i] * N / sum(w)   (normalized to sum to N)
//   This ensures the pseudo-posterior is on the same "effective sample size"
//   scale as the unweighted posterior.
//
// Model structure (same as M1):
//   Two-part hurdle:
//     Part 1 (Extensive): Bernoulli — does provider serve IT at all?
//     Part 2 (Intensive):  Zero-truncated Beta-Binomial — IT share among servers
//
//   State random intercepts:
//     (delta_ext[s], delta_int[s])' ~ N(0, Sigma_delta)
//     Sigma_delta = diag(tau) * Omega * diag(tau)
//
//   Non-centered parameterization (NCP):
//     z_delta[k,s] ~ N(0,1)  iid, k=1,2; s=1,...,S
//     delta = diag(tau) * L_Omega * z_delta
//
// Additional generated quantities:
//   - score_ext[i, 1:P]  : extensive-margin score vector (d ell_i / d alpha)
//   - score_int[i, 1:P]  : intensive-margin score vector (d ell_i / d beta)
//   - score_kappa[i]     : dispersion score (d ell_i / d log_kappa)
//   These are needed for the cluster-robust sandwich variance estimator
//   computed in R (61_sandwich_variance.R).
//
// NOTE: Score vectors are UNWEIGHTED individual scores.
//   Weight multiplication and cluster aggregation happen in R.
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

  // --- Survey weights (NEW in M1-W) ---
  vector<lower=0>[N] w_tilde;       // normalized survey weights: sum(w_tilde) = N
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

  // --- WEIGHTED Likelihood (pseudo-posterior) ---
  // target += w_tilde[i] * log f(y_i | theta)
  for (i in 1:N) {
    // Add state random intercept to fixed-effect linear predictor
    real eta_ext_i = eta_ext_fixed[i] + delta[1, state[i]];
    real eta_int_i = eta_int_fixed[i] + delta[2, state[i]];

    real q_i = inv_logit(eta_ext_i);       // P(serve IT)
    real mu_i = inv_logit(eta_int_i);      // conditional mean IT share
    real a_i = mu_i * kappa;               // BB shape parameter alpha
    real b_i = (1 - mu_i) * kappa;         // BB shape parameter beta

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

  // --- Correlation matrix (for reporting) ---
  corr_matrix[2] Omega = multiply_lower_tri_self_transpose(L_Omega);

  // --- Score vectors for sandwich variance (NEW in M1-W) ---
  // These are UNWEIGHTED individual scores: s_i = d log f(y_i|theta) / d theta
  // The weight multiplication and cluster aggregation happen in R.
  matrix[N, P] score_ext;           // extensive-margin score w.r.t. alpha
  matrix[N, P] score_int;           // intensive-margin score w.r.t. beta
  vector[N] score_kappa;            // dispersion score w.r.t. log_kappa

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

      // =========================================================
      // Extensive-margin score: s_i^ext = (z_i - q_i) * X_i
      //   Chain rule: d ell_i / d alpha_p = (d ell_i / d eta_ext) * X_ip
      //   For Bernoulli hurdle: d ell_i / d eta_ext = z_i - q_i
      //   (applies to both z=0 and z=1 cases)
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
        //   S_BB = kappa [psi(y+a) - psi(n-y+b) - psi(a) + psi(b)]
        //   Lambda_i = kappa [psi(b+n) - psi(b)]
        //   trunc_corr = p0 * Lambda / (1 - p0)
        //   score_mu = S_BB - trunc_corr    <-- MINUS sign
        //   score_int = score_mu * mu(1-mu) * X_i
        //
        // Chain rule: d ell_i / d beta_p = (d ell_i / d mu) * mu(1-mu) * X_ip
        //   where mu(1-mu) is the logistic derivative d mu / d eta_int
        // =========================================================
        real S_BB_mu = kappa * (digamma(y[i] + a_i)
                               - digamma(n_trial[i] - y[i] + b_i)
                               - digamma(a_i) + digamma(b_i));
        real Lambda_i = kappa * (digamma(b_i + n_trial[i]) - digamma(b_i));
        real trunc_corr_mu = p0_i * Lambda_i / (1 - p0_i);
        real score_mu_i = S_BB_mu - trunc_corr_mu;  // MINUS sign

        score_int[i] = score_mu_i * mu_i * (1 - mu_i) * to_row_vector(X[i]);

        // =========================================================
        // Dispersion score: s_i^kappa  (w.r.t. log_kappa, MINUS sign)
        //   S_BB_kappa = kappa[mu(psi(y+a)-psi(a)) + (1-mu)(psi(n-y+b)-psi(b))
        //                + psi(kappa) - psi(n+kappa)]
        //   trunc_corr_kappa = p0*mu*kappa/(1-p0) * sum_{j=1}^{n-1} j/[(b+j)(kappa+j)]
        //   score_kappa = S_BB_kappa - trunc_corr_kappa
        //
        // Note: the kappa multiplier at the front accounts for d/d(log kappa)
        //   via chain rule: d/d(log kappa) = kappa * d/d(kappa)
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
        // Rejection sampling for zero-truncated Beta-Binomial
        // Draw from BetaBin(n, a, b) via beta_rng + binomial_rng; reject if 0
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
