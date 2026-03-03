## =============================================================================
## 04_model_comparison.R -- LOO-CV and Posterior Predictive Checks
## =============================================================================
## Purpose : Compare M0-M3b via LOO-CV (unweighted models only) and compile
##           posterior predictive checks across all models.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Inputs:
##   data/precomputed/results_m0.rds       -- M0 results (with LOO + PPC)
##   data/precomputed/results_m1.rds       -- M1 results
##   data/precomputed/results_m2.rds       -- M2 results
##   data/precomputed/results_m3a.rds      -- M3a results
##   data/precomputed/results_m3b.rds      -- M3b results
##   data/precomputed/stan_data.rds        -- Stan data
##
## Outputs:
##   data/precomputed/loo_comparison.rds   -- LOO comparison table
##   data/precomputed/ppc_comparison.rds   -- PPC comparison table
##
## Usage:
##   source("code/04_model_comparison.R")
## =============================================================================

cat("==============================================================\n")
cat("  LOO-CV Model Comparison + Posterior Predictive Checks\n")
cat("==============================================================\n\n")

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
source(file.path(PROJECT_ROOT, "code/helpers/utils.R"))
library(loo)
library(dplyr, warn.conflicts = FALSE)

OUTPUT_DIR <- file.path(PROJECT_ROOT, "data/precomputed")


###############################################################################
## PART A : LOO-CV MODEL COMPARISON
###############################################################################
cat("\n")
cat("##############################################################\n")
cat("##  Part A: LOO-CV Model Comparison                         ##\n")
cat("##############################################################\n\n")

## --- 1. Load LOO objects ---
cat("--- 1. Loading LOO objects from all unweighted models ---\n")

model_names <- c("m0", "m1", "m2", "m3a", "m3b")
model_descriptions <- c(
  "M0: Pooled HBB (no state effects)",
  "M1: Random intercepts (NCP)",
  "M2: Block-diagonal SVC",
  "M3a: Cross-margin covariance",
  "M3b: Policy moderators"
)

loo_list <- list()
for (i in seq_along(model_names)) {
  res_path <- file.path(OUTPUT_DIR, paste0("results_", model_names[i], ".rds"))
  stopifnot(file.exists(res_path))
  res <- readRDS(res_path)
  stopifnot(!is.null(res$loo))
  loo_list[[model_names[i]]] <- res$loo

  elpd <- res$loo$estimates["elpd_loo", "Estimate"]
  se   <- res$loo$estimates["elpd_loo", "SE"]
  p_loo <- res$loo$estimates["p_loo", "Estimate"]
  n_bad <- sum(res$loo$diagnostics$pareto_k > 0.7)

  cat(sprintf("  %-35s  ELPD = %9.1f (SE = %6.1f)  p_loo = %7.1f  bad_k = %d\n",
              model_descriptions[i], elpd, se, p_loo, n_bad))
}
cat("  [PASS] All 5 LOO objects loaded.\n\n")

## --- 2. LOO comparison table ---
cat("--- 2. LOO comparison (loo::loo_compare) ---\n\n")

comp <- loo::loo_compare(loo_list)
print(comp)
cat("\n")

## --- 3. Pairwise differences ---
cat("--- 3. Pairwise ELPD differences (consecutive models) ---\n\n")

cat(sprintf("  %-12s  %10s %8s %8s %12s\n",
            "Comparison", "ELPD_diff", "SE_diff", "z-ratio", "Significant"))
cat(sprintf("  %s\n", paste(rep("-", 60), collapse = "")))

pairs <- list(
  c("m0", "m1"),
  c("m1", "m2"),
  c("m2", "m3a"),
  c("m3a", "m3b")
)

pairwise_table <- data.frame(
  comparison = character(),
  elpd_diff = numeric(),
  se_diff = numeric(),
  z_ratio = numeric(),
  significant = logical(),
  stringsAsFactors = FALSE
)

for (pair in pairs) {
  diff_obj <- loo::loo_compare(loo_list[pair])
  elpd_diff <- diff_obj[2, "elpd_diff"]
  se_diff   <- diff_obj[2, "se_diff"]
  z_ratio   <- elpd_diff / se_diff
  sig       <- abs(z_ratio) > 2

  comp_name <- paste0(pair[1], " -> ", pair[2])
  cat(sprintf("  %-12s  %+10.1f %8.1f %8.2f %12s\n",
              comp_name, elpd_diff, se_diff, z_ratio,
              ifelse(sig, "YES", "no")))

  pairwise_table <- rbind(pairwise_table, data.frame(
    comparison = comp_name,
    elpd_diff = elpd_diff,
    se_diff = se_diff,
    z_ratio = z_ratio,
    significant = sig,
    stringsAsFactors = FALSE
  ))
}

cat("\n")

## --- 4. Summary for paper ---
cat("--- 4. Summary table for paper (Table 5) ---\n\n")

summary_table <- data.frame(
  model = model_names,
  description = model_descriptions,
  n_params = c(11, 116, 551, 576, 616),
  elpd_loo = numeric(5),
  se_elpd = numeric(5),
  p_loo = numeric(5),
  delta_elpd = numeric(5),
  se_delta = numeric(5),
  bad_k = integer(5),
  stringsAsFactors = FALSE
)

for (i in seq_along(model_names)) {
  loo_i <- loo_list[[model_names[i]]]
  summary_table$elpd_loo[i] <- loo_i$estimates["elpd_loo", "Estimate"]
  summary_table$se_elpd[i]  <- loo_i$estimates["elpd_loo", "SE"]
  summary_table$p_loo[i]    <- loo_i$estimates["p_loo", "Estimate"]
  summary_table$bad_k[i]    <- sum(loo_i$diagnostics$pareto_k > 0.7)
}

# Delta relative to M0
for (i in seq_along(model_names)) {
  if (i == 1) {
    summary_table$delta_elpd[i] <- 0
    summary_table$se_delta[i] <- 0
  } else {
    diff_vs_m0 <- loo::loo_compare(loo_list[c("m0", model_names[i])])
    summary_table$delta_elpd[i] <- diff_vs_m0[2, "elpd_diff"]
    summary_table$se_delta[i]   <- diff_vs_m0[2, "se_diff"]
  }
}

cat(sprintf("  %-6s %-35s %7s %10s %8s %8s %10s %8s %5s\n",
            "Model", "Description", "Params", "ELPD_loo", "SE", "p_loo",
            "dELPD_M0", "SE_d", "bad_k"))
cat(sprintf("  %s\n", paste(rep("-", 110), collapse = "")))

for (i in 1:nrow(summary_table)) {
  r <- summary_table[i, ]
  delta_str <- ifelse(i == 1, "---", sprintf("%+10.1f", r$delta_elpd))
  se_d_str  <- ifelse(i == 1, "---", sprintf("%8.1f", r$se_delta))
  cat(sprintf("  %-6s %-35s %7d %10.1f %8.1f %8.1f %10s %8s %5d\n",
              r$model, r$description, r$n_params,
              r$elpd_loo, r$se_elpd, r$p_loo,
              delta_str, se_d_str, r$bad_k))
}

## --- 5. Pareto k diagnostics ---
cat("\n--- 5. Pareto k diagnostics ---\n\n")

for (i in seq_along(model_names)) {
  pk <- loo_list[[model_names[i]]]$diagnostics$pareto_k
  cat(sprintf("  %s (N = %d):\n", model_names[i], length(pk)))
  cat(sprintf("    k < 0.5 (good):     %d (%.1f%%)\n",
              sum(pk < 0.5), 100*mean(pk < 0.5)))
  cat(sprintf("    0.5 < k < 0.7 (ok): %d (%.1f%%)\n",
              sum(pk >= 0.5 & pk < 0.7), 100*mean(pk >= 0.5 & pk < 0.7)))
  cat(sprintf("    k > 0.7 (bad):      %d (%.1f%%)\n",
              sum(pk > 0.7), 100*mean(pk > 0.7)))
  cat(sprintf("    max k = %.3f\n\n", max(pk)))
}

## --- 6. Save LOO results ---
cat("--- 6. Saving LOO results ---\n")

loo_results <- list(
  loo_list = loo_list,
  comparison = comp,
  pairwise = pairwise_table,
  summary_table = summary_table,
  model_names = model_names,
  model_descriptions = model_descriptions,
  timestamp = Sys.time()
)

out_path <- file.path(OUTPUT_DIR, "loo_comparison.rds")
saveRDS(loo_results, out_path)
cat(sprintf("  Saved: %s\n", out_path))
cat(sprintf("    File size: %.1f KB\n\n", file.info(out_path)$size / 1024))


###############################################################################
## PART B : POSTERIOR PREDICTIVE CHECKS
###############################################################################
cat("\n")
cat("##############################################################\n")
cat("##  Part B: Posterior Predictive Checks                     ##\n")
cat("##############################################################\n\n")

## --- 7. Load observed statistics ---
cat("--- 7. Loading observed data statistics ---\n")

stan_data <- readRDS(file.path(OUTPUT_DIR, "stan_data.rds"))
N <- stan_data$N
z_obs <- stan_data$z
y_obs <- stan_data$y
n_trial_obs <- stan_data$n_trial
idx_pos <- which(z_obs == 1)
N_pos <- length(idx_pos)

obs_zero_rate <- mean(z_obs == 0)
obs_it_share  <- mean(y_obs[idx_pos] / n_trial_obs[idx_pos])

cat(sprintf("  N = %d, N_pos = %d (%.1f%% serve IT)\n", N, N_pos, 100*(1-obs_zero_rate)))
cat(sprintf("  Observed zero rate:     %.4f\n", obs_zero_rate))
cat(sprintf("  Observed mean IT share: %.4f\n", obs_it_share))
cat("\n")

## --- 8. Load PPC from all unweighted models ---
cat("--- 8. Loading PPC results from unweighted models ---\n\n")

model_labels <- c("M0: Pooled", "M1: Rand Int", "M2: Block SVC",
                   "M3a: Cross-Margin", "M3b: Policy Mod")

ppc_table <- data.frame(
  model = character(),
  label = character(),
  obs_zero_rate = numeric(),
  pred_zero_rate = numeric(),
  zero_rate_lo = numeric(),
  zero_rate_hi = numeric(),
  zero_rate_in_ci = logical(),
  obs_it_share = numeric(),
  pred_it_share = numeric(),
  it_share_lo = numeric(),
  it_share_hi = numeric(),
  it_share_in_ci = logical(),
  stringsAsFactors = FALSE
)

for (i in seq_along(model_names)) {
  res_path <- file.path(OUTPUT_DIR, paste0("results_", model_names[i], ".rds"))
  res <- readRDS(res_path)
  ppc <- res$ppc

  if (is.null(ppc)) {
    cat(sprintf("  %s: No PPC results found, skipping.\n", model_names[i]))
    next
  }

  pred_zr <- ppc$ppc_zero_rate_mean
  zr_lo   <- ppc$ppc_zero_rate_95ci[1]
  zr_hi   <- ppc$ppc_zero_rate_95ci[2]
  pred_it <- ppc$ppc_mean_share_mean
  it_lo   <- ppc$ppc_mean_share_95ci[1]
  it_hi   <- ppc$ppc_mean_share_95ci[2]

  zr_in <- obs_zero_rate >= zr_lo && obs_zero_rate <= zr_hi
  it_in <- obs_it_share >= it_lo && obs_it_share <= it_hi

  ppc_table <- rbind(ppc_table, data.frame(
    model = model_names[i],
    label = model_labels[i],
    obs_zero_rate = obs_zero_rate,
    pred_zero_rate = pred_zr,
    zero_rate_lo = zr_lo,
    zero_rate_hi = zr_hi,
    zero_rate_in_ci = zr_in,
    obs_it_share = obs_it_share,
    pred_it_share = pred_it,
    it_share_lo = it_lo,
    it_share_hi = it_hi,
    it_share_in_ci = it_in,
    stringsAsFactors = FALSE
  ))

  cat(sprintf("  %s: zero_rate=%.4f [%.4f, %.4f] %s | it_share=%.4f [%.4f, %.4f] %s\n",
              model_names[i],
              pred_zr, zr_lo, zr_hi, ifelse(zr_in, "OK", "MISS"),
              pred_it, it_lo, it_hi, ifelse(it_in, "OK", "MISS")))
}

cat("\n")

## --- 9. PPC Summary ---
cat("--- 9. PPC Summary Table ---\n\n")

cat(sprintf("  Observed: zero_rate = %.4f, IT_share = %.4f\n\n", obs_zero_rate, obs_it_share))

cat(sprintf("  %-20s  %24s  %24s\n",
            "", "--- Zero Rate PPC ---", "--- IT Share PPC ---"))
cat(sprintf("  %-20s  %8s %16s %4s  %8s %16s %4s\n",
            "Model", "Pred", "95% CI", "OK?",
            "Pred", "95% CI", "OK?"))
cat(sprintf("  %s\n", paste(rep("-", 85), collapse = "")))

for (i in 1:nrow(ppc_table)) {
  r <- ppc_table[i, ]
  cat(sprintf("  %-20s  %8.4f [%7.4f, %7.4f] %4s  %8.4f [%7.4f, %7.4f] %4s\n",
              r$label,
              r$pred_zero_rate, r$zero_rate_lo, r$zero_rate_hi,
              ifelse(r$zero_rate_in_ci, "OK", "MISS"),
              r$pred_it_share, r$it_share_lo, r$it_share_hi,
              ifelse(r$it_share_in_ci, "OK", "MISS")))
}

all_zero_ok <- all(ppc_table$zero_rate_in_ci)
all_it_ok   <- all(ppc_table$it_share_in_ci)

cat(sprintf("\n  Overall: Zero rate PPC %s | IT share PPC %s\n",
            ifelse(all_zero_ok, "[ALL PASS]", "[SOME MISS]"),
            ifelse(all_it_ok, "[ALL PASS]", "[SOME MISS]")))

## --- 10. Additional PPC statistics ---
cat("\n--- 10. Additional observed data statistics ---\n\n")

it_shares <- y_obs[idx_pos] / n_trial_obs[idx_pos]
cat(sprintf("  Distribution of IT shares (y/n among z=1):\n"))
cat(sprintf("    min=%.4f, Q1=%.4f, median=%.4f, mean=%.4f, Q3=%.4f, max=%.4f\n",
            min(it_shares), quantile(it_shares, 0.25), median(it_shares),
            mean(it_shares), quantile(it_shares, 0.75), max(it_shares)))

## Overdispersion check
binom_var <- mean(it_shares * (1 - it_shares) / n_trial_obs[idx_pos])
obs_var   <- var(it_shares)
od_ratio  <- obs_var / binom_var
cat(sprintf("\n  Overdispersion ratio: %.1f\n", od_ratio))

## --- 11. Save PPC results ---
cat("\n--- 11. Saving PPC results ---\n")

ppc_results <- list(
  ppc_table = ppc_table,
  observed = list(
    N = N,
    N_pos = N_pos,
    zero_rate = obs_zero_rate,
    it_share_mean = obs_it_share,
    it_share_dist = summary(it_shares),
    n_trial_dist = summary(n_trial_obs),
    overdispersion_ratio = od_ratio
  ),
  all_zero_ok = all_zero_ok,
  all_it_ok = all_it_ok,
  timestamp = Sys.time()
)

out_path <- file.path(OUTPUT_DIR, "ppc_comparison.rds")
saveRDS(ppc_results, out_path)
cat(sprintf("  Saved: %s\n", out_path))
cat(sprintf("    File size: %.1f KB\n", file.info(out_path)$size / 1024))

cat("\n==============================================================\n")
cat("  MODEL COMPARISON COMPLETE\n")
if (all_zero_ok && all_it_ok) {
  cat("  [PASS] All models recover key data features.\n")
}
cat("==============================================================\n")
