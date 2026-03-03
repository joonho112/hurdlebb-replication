// =============================================================================
// M1-W-NS: Survey-Weighted Random Intercepts Hurdle Beta-Binomial (No Scores)
// =============================================================================
// VARIANT of M1-W that OMITS score vectors from generated quantities.
//
// Purpose:
//   Score computation in Stan GQ produces 74,635 parameters per draw, causing:
//   (1) CSV bloat: 1.8 GB/chain (vs ~200 MB without scores)
//   (2) OOM on extraction: fit$draws("score_ext") loads 8000 x 34925 matrix
//   Score vectors are instead computed in R at the posterior mean (sim_03_fit.R).
//
// Differences from hbb_m1_weighted.stan:
//   - REMOVED: score_ext[N,P], score_int[N,P], score_kappa[N] from GQ
//   - KEPT:    log_lik[N], y_rep[N], Omega[2,2]
//   - data/parameters/transformed parameters/model blocks: IDENTICAL
//
// Model structure (same as M1-W):
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

  // NOTE: Score vectors (score_ext, score_int, score_kappa) are computed
  // in R at the posterior mean via compute_scores_in_r() in sim_03_fit.R.
  // This avoids the 74,635 extra GQ parameters per draw that cause
  // CSV bloat (1.8 GB/chain) and OOM on extraction.

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
