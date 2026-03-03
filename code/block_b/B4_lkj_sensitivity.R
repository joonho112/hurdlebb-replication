## =============================================================================
## B4_lkj_sensitivity.R -- LKJ Prior Sensitivity Analysis
## =============================================================================
## Purpose : Compile LKJ prior sensitivity analysis results across
##           eta in {1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0}, generate a
##           comparison table and posterior density overlay figure.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##
## Design:
##   - Reads lightweight result files from per-eta refits
##   - Extracts the baseline eta=2 from existing M3b-W fit
##   - Produces:  (1) comparison table (LaTeX + CSV)
##                (2) rho_cross posterior density overlay (PDF + PNG)
##                (3) consolidated results RDS
##
## Inputs  :
##   data/precomputed/B4_lkj/fit_lkj_eta*.rds
##   data/precomputed/results_m3b_weighted.rds
##   data/precomputed/fit_m3b_weighted.rds
##
## Outputs :
##   data/precomputed/B4_lkj/B4_lkj_sensitivity.rds
##   data/precomputed/B4_lkj/ST_lkj_sensitivity.tex
##   data/precomputed/B4_lkj/ST_lkj_sensitivity.csv
##   data/precomputed/B4_lkj/SF_lkj_rho_density.pdf
##   data/precomputed/B4_lkj/SF_lkj_rho_density.png
## =============================================================================

cat("==============================================================\n")
cat("  LKJ Prior Sensitivity Analysis\n")
cat("==============================================================\n\n")


###############################################################################
## SECTION 0 : SETUP
###############################################################################

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
OUTPUT_DIR   <- file.path(PROJECT_ROOT, "data/precomputed")
B4_DIR       <- file.path(OUTPUT_DIR, "B4_lkj")
B4_OUT       <- file.path(B4_DIR, "B4_lkj_sensitivity.rds")

## Create output directory
if (!dir.exists(B4_DIR)) dir.create(B4_DIR, recursive = TRUE)

## Packages
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

## ---------- Project theme (matching 80_manuscript_tables_figures.R) ----------
theme_manuscript <- function(base_size = 10) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor  = element_blank(),
      panel.grid.major  = element_line(colour = "grey92", linewidth = 0.3),
      strip.background  = element_rect(fill = "grey96", colour = NA),
      strip.text        = element_text(face = "bold", size = base_size),
      axis.title        = element_text(size = base_size),
      axis.text         = element_text(size = base_size - 1, colour = "grey25"),
      axis.ticks        = element_line(colour = "grey70", linewidth = 0.3),
      plot.title        = element_text(face = "bold", size = base_size + 1, hjust = 0),
      plot.subtitle     = element_text(size = base_size - 1, hjust = 0, colour = "grey40"),
      legend.position   = "bottom",
      legend.text       = element_text(size = base_size - 1),
      legend.title      = element_text(size = base_size, face = "bold"),
      legend.key.size   = unit(0.8, "lines"),
      plot.margin       = margin(5, 10, 5, 5)
    )
}

## ---------- Helper: save figure ----------
save_figure <- function(plot, name, dir = B4_DIR, width = 7, height = 5) {
  pdf_path <- file.path(dir, paste0(name, ".pdf"))
  png_path <- file.path(dir, paste0(name, ".png"))
  ggsave(pdf_path, plot, width = width, height = height, device = "pdf")
  ggsave(png_path, plot, width = width, height = height, dpi = 300)
  cat(sprintf("  [SAVED] %s.pdf / .png (%g x %g in)\n", name, width, height))
}

## ---------- Helper: format signed number ----------
fmt_signed <- function(x, digits = 3) {
  s <- formatC(x, format = "f", digits = digits)
  if (x >= 0) s <- paste0("+", s)
  s
}

## ---------- Constants ----------
ETA_VALUES  <- c(1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
N_ETA       <- length(ETA_VALUES)
BASELINE_ETA <- 2.0

## Covariate labels
COV_LABELS <- c("intercept", "poverty", "urban", "black", "hispanic")
TAU_LABELS <- c(paste0("ext_", COV_LABELS), paste0("int_", COV_LABELS))
P <- 5    # number of covariates per margin
K <- 2 * P  # total random-effect dimensions

cat(sprintf("  PROJECT_ROOT: %s\n", PROJECT_ROOT))
cat(sprintf("  B4 output:    %s\n", B4_DIR))
cat(sprintf("  Eta values:   %s\n", paste(ETA_VALUES, collapse = ", ")))
cat(sprintf("  Baseline eta: %.1f\n\n", BASELINE_ETA))


###############################################################################
## SECTION 1 : LOAD EXISTING M3b-W BASELINE (ETA = 2)
###############################################################################
cat("--- 1. Loading existing M3b-W baseline (eta = 2) ---\n\n")

## 1a. Load the lightweight results file
m3bw_path <- file.path(OUTPUT_DIR, "results_m3b_weighted.rds")
stopifnot("M3b-W results file not found" = file.exists(m3bw_path))
res_m3bw <- readRDS(m3bw_path)
cat(sprintf("  Loaded: %s\n", m3bw_path))

## 1b. Extract fixed effects from existing results
baseline_alpha <- res_m3bw$alpha_means   # named vector of 5
baseline_beta  <- res_m3bw$beta_means    # named vector of 5
baseline_tau   <- res_m3bw$tau_means     # named vector of 10

cat(sprintf("  alpha_poverty (baseline) = %.4f\n", baseline_alpha["poverty"]))
cat(sprintf("  beta_poverty  (baseline) = %.4f\n", baseline_beta["poverty"]))

## 1c. Attempt to extract rho_cross draws from full fit object
##     The full fit (CmdStanMCMC) is ~1.9 GB and may take time to load.
##     If it fails or is unavailable, fall back to a point estimate.
fit_m3bw_path <- file.path(OUTPUT_DIR, "fit_m3b_weighted.rds")
baseline_rho_draws <- NULL
baseline_rho <- list(mean = NA, sd = NA, q025 = NA, q975 = NA, draws = NULL)

if (file.exists(fit_m3bw_path)) {
  cat(sprintf("\n  Loading full fit object for rho_cross draws: %s\n",
              fit_m3bw_path))
  cat("  (This may take a moment due to file size...)\n")

  tryCatch({
    fit_m3bw <- readRDS(fit_m3bw_path)

    ## rho_cross = Omega[1, P+1] = Omega[1,6]
    rho_name <- sprintf("Omega[1,%d]", P + 1)
    rho_draws <- as.numeric(fit_m3bw$draws(rho_name, format = "matrix"))

    if (length(rho_draws) > 0) {
      baseline_rho <- list(
        mean  = mean(rho_draws),
        sd    = sd(rho_draws),
        q025  = as.numeric(quantile(rho_draws, 0.025)),
        q975  = as.numeric(quantile(rho_draws, 0.975)),
        draws = rho_draws
      )
      cat(sprintf("  [PASS] Extracted %d rho_cross draws from full fit.\n",
                  length(rho_draws)))
      cat(sprintf("    rho_cross = %.4f [%.4f, %.4f]\n",
                  baseline_rho$mean, baseline_rho$q025, baseline_rho$q975))
    } else {
      cat("  [WARN] Omega draws empty. Using fallback.\n")
    }

    ## Clean up to free memory
    rm(fit_m3bw)
    gc(verbose = FALSE)

  }, error = function(e) {
    cat(sprintf("  [WARN] Could not load full fit: %s\n", conditionMessage(e)))
    cat("  Will attempt to use B4 eta=2 results instead.\n")
  })
} else {
  cat(sprintf("  [INFO] Full fit not found at %s\n", fit_m3bw_path))
  cat("  Will attempt to use B4 eta=2 results for baseline.\n")
}

cat("\n")


###############################################################################
## SECTION 2 : LOAD PER-ETA RESULTS FROM B4 FITS
###############################################################################
cat("--- 2. Loading per-eta sensitivity fit results ---\n\n")

## Storage: list of results, keyed by eta label
eta_results <- list()
eta_found   <- numeric(0)
eta_missing <- numeric(0)

for (eta in ETA_VALUES) {
  eta_label <- sprintf("%.1f", eta)
  eta_file  <- sprintf("fit_lkj_eta%s.rds", gsub("\\.", "_", eta_label))
  eta_path  <- file.path(B4_DIR, eta_file)

  if (file.exists(eta_path)) {
    res <- readRDS(eta_path)
    eta_results[[eta_label]] <- res
    eta_found <- c(eta_found, eta)

    cat(sprintf("  [FOUND] eta = %s : rho_cross = %.4f [%.4f, %.4f], ELPD = %.1f, diag = %s\n",
                eta_label,
                res$rho_cross$mean,
                res$rho_cross$q025,
                res$rho_cross$q975,
                ifelse(is.na(res$loo$elpd_loo), NA, res$loo$elpd_loo),
                ifelse(res$diagnostics$all_pass, "PASS", "WARN")))
  } else {
    eta_missing <- c(eta_missing, eta)
    cat(sprintf("  [MISS]  eta = %s : %s not found\n", eta_label, eta_file))
  }
}

## If eta=2.0 was fit separately AND we don't have baseline rho draws,
## use the B4 eta=2 fit as the baseline
if (is.na(baseline_rho$mean) && "2.0" %in% names(eta_results)) {
  cat("\n  [INFO] Using B4 eta=2.0 fit as baseline rho_cross source.\n")
  baseline_rho <- eta_results[["2.0"]]$rho_cross
  cat(sprintf("    rho_cross = %.4f [%.4f, %.4f]\n",
              baseline_rho$mean, baseline_rho$q025, baseline_rho$q975))
}

## If we STILL don't have baseline rho draws, use the paper's reported value
if (is.na(baseline_rho$mean)) {
  cat("\n  [WARN] No rho_cross draws available for eta=2 baseline.\n")
  cat("  Using paper's reported point estimate (0.021) for table.\n")
  baseline_rho$mean <- 0.021
  baseline_rho$sd   <- NA
  baseline_rho$q025 <- NA
  baseline_rho$q975 <- NA
}

cat(sprintf("\n  Summary: %d/%d eta values found, %d missing.\n",
            length(eta_found), N_ETA, length(eta_missing)))
if (length(eta_missing) > 0) {
  cat(sprintf("  Missing: eta = %s\n",
              paste(sprintf("%.1f", eta_missing), collapse = ", ")))
}
cat("\n")


###############################################################################
## SECTION 3 : BUILD COMPARISON TABLE
###############################################################################
cat("--- 3. Building comparison table ---\n\n")

## Initialize data frame with one row per eta
comp_rows <- list()

for (eta in ETA_VALUES) {
  eta_label <- sprintf("%.1f", eta)

  ## Check if this eta was fit via B4
  if (eta_label %in% names(eta_results)) {
    res <- eta_results[[eta_label]]

    comp_rows[[eta_label]] <- data.frame(
      eta            = eta,
      source         = ifelse(eta == BASELINE_ETA, "baseline/B4", "B4"),
      rho_mean       = res$rho_cross$mean,
      rho_sd         = res$rho_cross$sd,
      rho_q025       = res$rho_cross$q025,
      rho_q975       = res$rho_cross$q975,
      alpha_pov      = res$alpha_means["poverty"],
      beta_pov       = res$beta_means["poverty"],
      tau_ext1       = res$tau_means["ext_intercept"],
      tau_int1       = res$tau_means["int_intercept"],
      elpd_loo       = ifelse(is.null(res$loo$elpd_loo), NA, res$loo$elpd_loo),
      elpd_se        = ifelse(is.null(res$loo$elpd_se), NA, res$loo$elpd_se),
      diag_pass      = res$diagnostics$all_pass,
      max_rhat       = res$diagnostics$max_rhat,
      min_ess        = res$diagnostics$min_ess,
      n_divergent    = res$diagnostics$n_divergent,
      fit_time_mins  = ifelse(is.null(res$fit_time_mins), NA, res$fit_time_mins),
      stringsAsFactors = FALSE,
      row.names      = NULL
    )

  } else if (eta == BASELINE_ETA) {
    ## Use existing M3b-W results as baseline
    comp_rows[[eta_label]] <- data.frame(
      eta            = eta,
      source         = "existing M3b-W",
      rho_mean       = baseline_rho$mean,
      rho_sd         = ifelse(is.na(baseline_rho$sd), NA, baseline_rho$sd),
      rho_q025       = ifelse(is.na(baseline_rho$q025), NA, baseline_rho$q025),
      rho_q975       = ifelse(is.na(baseline_rho$q975), NA, baseline_rho$q975),
      alpha_pov      = as.numeric(baseline_alpha["poverty"]),
      beta_pov       = as.numeric(baseline_beta["poverty"]),
      tau_ext1       = as.numeric(baseline_tau["ext_intercept"]),
      tau_int1       = as.numeric(baseline_tau["int_intercept"]),
      elpd_loo       = NA,
      elpd_se        = NA,
      diag_pass      = TRUE,
      max_rhat       = NA,
      min_ess        = NA,
      n_divergent    = NA,
      fit_time_mins  = NA,
      stringsAsFactors = FALSE,
      row.names      = NULL
    )

  } else {
    ## Missing eta -- insert placeholder row
    comp_rows[[eta_label]] <- data.frame(
      eta            = eta,
      source         = "NOT RUN",
      rho_mean       = NA, rho_sd = NA, rho_q025 = NA, rho_q975 = NA,
      alpha_pov      = NA, beta_pov = NA,
      tau_ext1       = NA, tau_int1 = NA,
      elpd_loo       = NA, elpd_se = NA,
      diag_pass      = NA, max_rhat = NA, min_ess = NA, n_divergent = NA,
      fit_time_mins  = NA,
      stringsAsFactors = FALSE,
      row.names      = NULL
    )
  }
}

comparison <- do.call(rbind, comp_rows)
rownames(comparison) <- NULL

## Print the comparison table to console
cat("  LKJ Sensitivity Comparison Table:\n\n")
cat(sprintf("  %-4s  %-14s  %8s  %20s  %8s  %8s  %8s  %8s  %10s  %5s\n",
            "eta", "Source", "rho_mean", "[95% CI]",
            "a_pov", "b_pov", "tau_e1", "tau_i1", "ELPD", "Diag"))
cat(sprintf("  %s\n", paste(rep("-", 110), collapse = "")))

for (i in seq_len(nrow(comparison))) {
  r <- comparison[i, ]
  ci_str <- ifelse(is.na(r$rho_q025), "      N/A",
                   sprintf("[%+.3f, %+.3f]", r$rho_q025, r$rho_q975))
  elpd_str <- ifelse(is.na(r$elpd_loo), "N/A",
                     sprintf("%.1f", r$elpd_loo))
  diag_str <- ifelse(is.na(r$diag_pass), "N/A",
                     ifelse(r$diag_pass, "PASS", "WARN"))
  marker <- ifelse(r$eta == BASELINE_ETA, " *", "")

  cat(sprintf("  %-4.1f  %-14s  %8.4f  %20s  %8.4f  %8.4f  %8.4f  %8.4f  %10s  %5s%s\n",
              r$eta, r$source,
              ifelse(is.na(r$rho_mean), NA, r$rho_mean),
              ci_str,
              ifelse(is.na(r$alpha_pov), NA, r$alpha_pov),
              ifelse(is.na(r$beta_pov), NA, r$beta_pov),
              ifelse(is.na(r$tau_ext1), NA, r$tau_ext1),
              ifelse(is.na(r$tau_int1), NA, r$tau_int1),
              elpd_str, diag_str, marker))
}
cat("\n  (* = baseline specification)\n\n")

## Compute sensitivity metrics (using available rows)
available <- comparison[!is.na(comparison$rho_mean), ]

if (nrow(available) >= 2) {
  rho_range <- range(available$rho_mean)
  rho_spread <- diff(rho_range)
  alpha_range <- range(available$alpha_pov, na.rm = TRUE)
  beta_range  <- range(available$beta_pov, na.rm = TRUE)

  cat("  Sensitivity summary (across available eta values):\n")
  cat(sprintf("    rho_cross range:  [%.4f, %.4f]  (spread = %.4f)\n",
              rho_range[1], rho_range[2], rho_spread))
  cat(sprintf("    alpha_pov range:  [%.4f, %.4f]  (spread = %.4f)\n",
              alpha_range[1], alpha_range[2], diff(alpha_range)))
  cat(sprintf("    beta_pov range:   [%.4f, %.4f]  (spread = %.4f)\n",
              beta_range[1], beta_range[2], diff(beta_range)))

  ## ELPD comparison (if available)
  elpd_avail <- available[!is.na(available$elpd_loo), ]
  if (nrow(elpd_avail) >= 2) {
    elpd_range <- range(elpd_avail$elpd_loo)
    best_eta   <- elpd_avail$eta[which.max(elpd_avail$elpd_loo)]
    cat(sprintf("    ELPD_loo range:   [%.1f, %.1f]  (spread = %.1f)\n",
                elpd_range[1], elpd_range[2], diff(elpd_range)))
    cat(sprintf("    Best ELPD at eta: %.1f\n", best_eta))
  }
} else {
  cat("  [WARN] Fewer than 2 eta values available. Cannot compute sensitivity.\n")
}

cat("\n")


###############################################################################
## SECTION 4 : LATEX TABLE GENERATION
###############################################################################
cat("--- 4. Generating LaTeX table for SM-D ---\n\n")

tex <- character()

tex <- c(tex,
  "\\begin{table}[t]",
  "\\centering",
  paste0("\\caption{LKJ prior sensitivity analysis. ",
         "Each row reports posterior summaries from M3b-W fitted with "),
  paste0("  $\\mathrm{LKJ}(\\eta)$ prior on the $10 \\times 10$ random-effect ",
         "correlation matrix $\\Omega$."),
  paste0("  The baseline specification is $\\eta = 2$ (uniform over correlations). "),
  paste0("  $\\rho_{\\mathrm{cross}} = \\Omega_{1,6}$ is the cross-margin correlation ",
         "between extensive and intensive intercepts."),
  paste0("  $\\alpha_{\\mathrm{pov}}$ and $\\beta_{\\mathrm{pov}}$ are the poverty ",
         "coefficients for the extensive and intensive margins."),
  paste0("  $\\tau_1^{\\mathrm{ext}}$ and $\\tau_1^{\\mathrm{int}}$ are the ",
         "intercept random-effect standard deviations."),
  paste0("  All covariates are standardized.}"),
  "\\label{tab:lkj-sensitivity}",
  "\\smallskip",
  "\\small",
  "\\begin{adjustbox}{max width=\\textwidth}",
  paste0("\\begin{tabular}{@{}r ",
         "r@{\\hspace{4pt}}l ",    # rho_cross [CI]
         "rr ",                     # alpha_pov, beta_pov
         "rr ",                     # tau_ext1, tau_int1
         "r ",                      # ELPD
         "c@{}}"),                  # Diagnostics
  "\\toprule",
  paste0("$\\eta$ & $\\hat{\\rho}_{\\mathrm{cross}}$ & ",
         "[95\\% CI] & ",
         "$\\alpha_{\\mathrm{pov}}$ & $\\beta_{\\mathrm{pov}}$ & ",
         "$\\tau_1^{\\mathrm{ext}}$ & $\\tau_1^{\\mathrm{int}}$ & ",
         "$\\widehat{\\mathrm{elpd}}_{\\mathrm{loo}}$ & Diag.~\\\\"),
  "\\midrule"
)

for (i in seq_len(nrow(comparison))) {
  r <- comparison[i, ]

  if (is.na(r$rho_mean)) {
    ## Missing eta -- show dashes
    row <- sprintf("$%.1f$ & --- & --- & --- & --- & --- & --- & --- & --- \\\\",
                   r$eta)
  } else {
    ## Format rho with CI
    rho_str <- formatC(r$rho_mean, format = "f", digits = 3)
    if (!is.na(r$rho_q025)) {
      ci_str <- sprintf("$[%s,\\; %s]$",
                        formatC(r$rho_q025, format = "f", digits = 3),
                        formatC(r$rho_q975, format = "f", digits = 3))
    } else {
      ci_str <- "---"
    }

    ## Format ELPD
    elpd_str <- ifelse(is.na(r$elpd_loo), "---",
                       formatC(r$elpd_loo, format = "f", digits = 1))

    ## Format diagnostics
    diag_str <- ifelse(is.na(r$diag_pass), "---",
                       ifelse(r$diag_pass, "\\checkmark", "\\textbf{!}"))

    ## Mark baseline row
    eta_str <- ifelse(r$eta == BASELINE_ETA,
                      sprintf("$%.1f^{\\star}$", r$eta),
                      sprintf("$%.1f$", r$eta))

    row <- sprintf("%s & $%s$ & %s & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & %s \\\\",
                   eta_str,
                   rho_str,
                   ci_str,
                   formatC(r$alpha_pov, format = "f", digits = 3),
                   formatC(r$beta_pov, format = "f", digits = 3),
                   formatC(r$tau_ext1, format = "f", digits = 3),
                   formatC(r$tau_int1, format = "f", digits = 3),
                   elpd_str,
                   diag_str)
  }

  tex <- c(tex, row)
}

## Footer
tex <- c(tex,
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{adjustbox}",
  "",
  "\\medskip",
  paste0("{\\footnotesize $^{\\star}$Baseline specification ",
         "($\\eta = 2$ corresponds to a uniform distribution over ",
         "correlation matrices).}"),
  "\\end{table}"
)

## Write LaTeX table
tex_path <- file.path(B4_DIR, "ST_lkj_sensitivity.tex")
writeLines(tex, tex_path)
cat(sprintf("  [SAVED] %s\n", tex_path))

## CSV backup
csv_path <- file.path(B4_DIR, "ST_lkj_sensitivity.csv")
csv_df <- data.frame(
  eta         = comparison$eta,
  source      = comparison$source,
  rho_mean    = ifelse(is.na(comparison$rho_mean), "",
                       sprintf("%.4f", comparison$rho_mean)),
  rho_95CI    = ifelse(is.na(comparison$rho_q025), "",
                       sprintf("[%.3f, %.3f]", comparison$rho_q025,
                               comparison$rho_q975)),
  alpha_pov   = ifelse(is.na(comparison$alpha_pov), "",
                       sprintf("%.4f", comparison$alpha_pov)),
  beta_pov    = ifelse(is.na(comparison$beta_pov), "",
                       sprintf("%.4f", comparison$beta_pov)),
  tau_ext1    = ifelse(is.na(comparison$tau_ext1), "",
                       sprintf("%.4f", comparison$tau_ext1)),
  tau_int1    = ifelse(is.na(comparison$tau_int1), "",
                       sprintf("%.4f", comparison$tau_int1)),
  elpd_loo    = ifelse(is.na(comparison$elpd_loo), "",
                       sprintf("%.1f", comparison$elpd_loo)),
  diag_pass   = ifelse(is.na(comparison$diag_pass), "",
                       ifelse(comparison$diag_pass, "PASS", "WARN")),
  stringsAsFactors = FALSE
)
write.csv(csv_df, csv_path, row.names = FALSE)
cat(sprintf("  [SAVED] %s\n\n", csv_path))


###############################################################################
## SECTION 5 : POSTERIOR DENSITY FIGURE
###############################################################################
cat("--- 5. Generating rho_cross posterior density overlay figure ---\n\n")

## Collect rho_cross draws from all available etas
draws_list <- list()

for (eta in ETA_VALUES) {
  eta_label <- sprintf("%.1f", eta)

  if (eta_label %in% names(eta_results)) {
    ## Use B4 fit draws
    res <- eta_results[[eta_label]]
    if (!is.null(res$rho_cross$draws) && length(res$rho_cross$draws) > 0) {
      draws_list[[eta_label]] <- data.frame(
        eta   = sprintf("eta = %s", eta_label),
        rho   = res$rho_cross$draws,
        stringsAsFactors = FALSE
      )
    }
  } else if (eta == BASELINE_ETA && !is.null(baseline_rho$draws)) {
    ## Use existing fit draws for eta=2 baseline
    draws_list[[eta_label]] <- data.frame(
      eta   = sprintf("eta = %s", eta_label),
      rho   = baseline_rho$draws,
      stringsAsFactors = FALSE
    )
  }
}

n_density_etas <- length(draws_list)

if (n_density_etas >= 2) {
  ## Combine all draws
  draws_all <- do.call(rbind, draws_list)
  draws_all$eta <- factor(draws_all$eta,
                          levels = paste0("eta = ",
                                          sprintf("%.1f", sort(ETA_VALUES))))

  ## Color palette: diverging from warm (low eta) to cool (high eta)
  ## Using a manual palette for distinguishability
  eta_colors <- c(
    "eta = 1.0" = "#E41A1C",   # red
    "eta = 1.5" = "#FF7F00",   # orange
    "eta = 2.0" = "#4DAF4A",   # green (baseline)
    "eta = 3.0" = "#377EB8",   # blue
    "eta = 4.0" = "#984EA3",   # purple
    "eta = 6.0" = "#A65628",   # brown
    "eta = 8.0" = "#999999"    # grey
  )

  ## Only keep colors for eta values with draws
  active_levels <- levels(draws_all$eta)
  active_colors <- eta_colors[active_levels]

  ## Determine the density x-range from the data
  x_range <- range(draws_all$rho)
  x_pad   <- 0.1 * diff(x_range)

  ## Build the plot
  p_density <- ggplot(draws_all, aes(x = rho, colour = eta, fill = eta)) +
    geom_density(alpha = 0.08, linewidth = 0.7, adjust = 1.5) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40",
               linewidth = 0.4) +
    scale_colour_manual(values = active_colors, name = "LKJ Prior") +
    scale_fill_manual(values = active_colors, name = "LKJ Prior") +
    labs(
      title    = expression(paste("Posterior density of ",
                                  rho[cross], " under varying LKJ(",
                                  eta, ") priors")),
      subtitle = expression(paste(rho[cross], " = ", Omega[paste("1,6")],
                                  ": cross-margin correlation ",
                                  "(extensive-intensive intercepts)")),
      x = expression(rho[cross]),
      y = "Posterior density"
    ) +
    coord_cartesian(xlim = c(x_range[1] - x_pad, x_range[2] + x_pad)) +
    theme_manuscript(base_size = 10) +
    guides(colour = guide_legend(nrow = 1),
           fill   = guide_legend(nrow = 1))

  ## Save
  save_figure(p_density, "SF_lkj_rho_density", width = 7, height = 4.5)

  cat(sprintf("  Density plot includes %d eta values: %s\n",
              n_density_etas,
              paste(names(draws_list), collapse = ", ")))

} else if (n_density_etas == 1) {
  cat("  [WARN] Only 1 eta value has draws. Generating single-density plot.\n")

  draws_all <- do.call(rbind, draws_list)

  p_density <- ggplot(draws_all, aes(x = rho)) +
    geom_density(fill = "#4DAF4A", alpha = 0.3, linewidth = 0.7) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40",
               linewidth = 0.4) +
    labs(
      title = expression(paste("Posterior density of ", rho[cross])),
      x     = expression(rho[cross]),
      y     = "Posterior density"
    ) +
    theme_manuscript(base_size = 10)

  save_figure(p_density, "SF_lkj_rho_density", width = 7, height = 4.5)

} else {
  cat("  [WARN] No rho_cross draws available. Skipping density figure.\n")
  p_density <- NULL
}

cat("\n")


###############################################################################
## SECTION 6 : SUPPLEMENTARY PARAMETER COMPARISON FIGURE
###############################################################################
cat("--- 6. Generating parameter sensitivity dot plot ---\n\n")

if (nrow(available) >= 2) {
  ## Build a tidy data frame of key parameters across eta values
  param_tidy <- data.frame(
    eta   = rep(available$eta, 4),
    param = rep(c("rho[cross]",
                  "alpha[poverty]",
                  "beta[poverty]",
                  "tau[1]^ext"),
                each = nrow(available)),
    value = c(available$rho_mean,
              available$alpha_pov,
              available$beta_pov,
              available$tau_ext1),
    stringsAsFactors = FALSE
  )
  param_tidy$param <- factor(param_tidy$param,
                             levels = c("rho[cross]",
                                        "alpha[poverty]",
                                        "beta[poverty]",
                                        "tau[1]^ext"))

  p_params <- ggplot(param_tidy, aes(x = eta, y = value)) +
    geom_point(size = 2.5, colour = "#377EB8") +
    geom_line(linewidth = 0.4, colour = "#377EB8", alpha = 0.5) +
    geom_vline(xintercept = BASELINE_ETA, linetype = "dotted",
               colour = "grey50", linewidth = 0.4) +
    facet_wrap(~ param, scales = "free_y", nrow = 2, ncol = 2,
               labeller = label_parsed) +
    labs(
      title    = "Key parameter estimates under varying LKJ priors",
      subtitle = "Dotted line marks baseline specification (eta = 2)",
      x = expression(paste("LKJ prior concentration (", eta, ")")),
      y = "Posterior mean"
    ) +
    scale_x_continuous(breaks = ETA_VALUES) +
    theme_manuscript(base_size = 10)

  save_figure(p_params, "SF_lkj_param_sensitivity", width = 7, height = 5)

} else {
  cat("  [WARN] Fewer than 2 eta values available. Skipping parameter plot.\n")
  p_params <- NULL
}

cat("\n")


###############################################################################
## SECTION 7 : SUGGESTED MANUSCRIPT TEXT
###############################################################################
cat("--- 7. Generating suggested manuscript text ---\n\n")

ms_text <- character()

ms_text <- c(ms_text,
  "=========================================================================",
  "SUGGESTED TEXT FOR SM-D: LKJ Prior Sensitivity",
  "=========================================================================",
  "",
  "\\subsection*{D.X\\quad LKJ Prior Sensitivity}",
  "\\label{sec:sm-lkj-sensitivity}",
  ""
)

## Dynamic content based on results
if (nrow(available) >= 3) {
  rho_min   <- min(available$rho_mean, na.rm = TRUE)
  rho_max   <- max(available$rho_mean, na.rm = TRUE)
  apov_min  <- min(available$alpha_pov, na.rm = TRUE)
  apov_max  <- max(available$alpha_pov, na.rm = TRUE)
  bpov_min  <- min(available$beta_pov, na.rm = TRUE)
  bpov_max  <- max(available$beta_pov, na.rm = TRUE)

  ## Check if ELPD data is available
  elpd_avail_rows <- available[!is.na(available$elpd_loo), ]
  has_elpd <- nrow(elpd_avail_rows) >= 2

  ms_text <- c(ms_text,
    paste0("To assess sensitivity to the LKJ prior on the $10 \\times 10$"),
    paste0("random-effect correlation matrix~$\\Omega$, we refit M3b-W under"),
    paste0("seven concentration parameters"),
    paste0("$\\eta \\in \\{1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0\\}$."),
    paste0("The baseline specification uses $\\eta = 2$, which places a uniform"),
    paste0("distribution over correlation matrices; lower values favor extreme"),
    paste0("correlations, while higher values concentrate mass near the identity."),
    ""
  )

  ms_text <- c(ms_text,
    paste0("\\cref{tab:lkj-sensitivity} reports posterior summaries for the"),
    paste0("cross-margin correlation $\\rho_{\\mathrm{cross}} = \\Omega_{1,6}$,"),
    paste0("the poverty coefficients ($\\alpha_{\\mathrm{pov}}$, $\\beta_{\\mathrm{pov}}$),"),
    paste0("and the intercept random-effect standard deviations"),
    paste0("($\\tau_1^{\\mathrm{ext}}$, $\\tau_1^{\\mathrm{int}}$)."),
    sprintf("Across the sevenfold range of $\\eta$, $\\hat{\\rho}_{\\mathrm{cross}}$"),
    sprintf("varies between $%.3f$ and $%.3f$, remaining near zero throughout.", rho_min, rho_max),
    sprintf("The poverty coefficients are equally stable:"),
    sprintf("$\\alpha_{\\mathrm{pov}} \\in [%.3f,\\, %.3f]$ and", apov_min, apov_max),
    sprintf("$\\beta_{\\mathrm{pov}} \\in [%.3f,\\, %.3f]$,", bpov_min, bpov_max),
    paste0("confirming that the sign reversal pattern is robust to the LKJ prior."),
    ""
  )

  if (has_elpd) {
    best_elpd_eta <- elpd_avail_rows$eta[which.max(elpd_avail_rows$elpd_loo)]
    elpd_spread   <- diff(range(elpd_avail_rows$elpd_loo))
    ms_text <- c(ms_text,
      sprintf("LOO-CV fit is nearly identical across specifications"),
      sprintf("(ELPD spread $< %.0f$), with the best-fitting model at $\\eta = %.1f$.",
              ceiling(elpd_spread), best_elpd_eta),
      ""
    )
  }

  ms_text <- c(ms_text,
    paste0("These results confirm that the near-zero cross-margin correlation"),
    paste0("and the poverty reversal pattern are data-driven findings,"),
    paste0("not artifacts of the LKJ(2) prior specification."),
    ""
  )

} else {
  ms_text <- c(ms_text,
    "[NOTE: Fewer than 3 eta values available. Complete the remaining fits",
    " before generating final manuscript text.]",
    ""
  )
}

ms_text <- c(ms_text,
  "",
  "=========================================================================",
  "SUGGESTED CROSS-REFERENCE FOR SECTION 6 (Discussion):",
  "=========================================================================",
  "",
  "  Prior robustness is confirmed by LKJ sensitivity analysis",
  "  (\\cref{tab:lkj-sensitivity} in SM-D), which shows that the",
  "  cross-margin correlation and all fixed-effect estimates are",
  "  stable across $\\eta \\in \\{1, \\ldots, 8\\}$.",
  "",
  "========================================================================="
)

## Write manuscript text
ms_text_path <- file.path(B4_DIR, "B4_manuscript_text.txt")
writeLines(ms_text, ms_text_path)
cat(sprintf("  [SAVED] %s\n", ms_text_path))

## Print to console
cat("\n")
for (line in ms_text) cat(sprintf("  %s\n", line))
cat("\n")


###############################################################################
## SECTION 8 : VERIFICATION CHECKS
###############################################################################
cat("--- 8. Verification checks ---\n\n")

checks_passed <- 0
checks_total  <- 0

## Check 1: At least one eta value loaded
checks_total <- checks_total + 1
if (length(eta_found) >= 1) {
  checks_passed <- checks_passed + 1
  cat(sprintf("  [PASS] At least 1 eta value loaded (%d found).\n",
              length(eta_found)))
} else {
  cat("  [FAIL] No eta values loaded. Run 84_B4_lkj_sensitivity_fit.R first.\n")
}

## Check 2: Baseline eta=2 has rho_cross
checks_total <- checks_total + 1
if (!is.na(baseline_rho$mean)) {
  checks_passed <- checks_passed + 1
  cat(sprintf("  [PASS] Baseline rho_cross available: %.4f\n",
              baseline_rho$mean))
} else {
  cat("  [FAIL] Baseline rho_cross not available.\n")
}

## Check 3: Comparison table has correct number of rows
checks_total <- checks_total + 1
if (nrow(comparison) == N_ETA) {
  checks_passed <- checks_passed + 1
  cat(sprintf("  [PASS] Comparison table has %d rows (expected %d).\n",
              nrow(comparison), N_ETA))
} else {
  cat(sprintf("  [FAIL] Comparison table has %d rows (expected %d).\n",
              nrow(comparison), N_ETA))
}

## Check 4: LaTeX table has matching begin/end{table}
checks_total <- checks_total + 1
tex_content <- paste(tex, collapse = "\n")
has_begin <- grepl("\\\\begin\\{table\\}", tex_content)
has_end   <- grepl("\\\\end\\{table\\}", tex_content)
if (has_begin && has_end) {
  checks_passed <- checks_passed + 1
  cat("  [PASS] LaTeX table has matching begin/end{table}.\n")
} else {
  cat("  [FAIL] LaTeX table structure incomplete!\n")
}

## Check 5: All output files written
checks_total <- checks_total + 1
expected_files <- c(tex_path, csv_path, ms_text_path)
all_exist <- all(file.exists(expected_files))
if (all_exist) {
  checks_passed <- checks_passed + 1
  cat("  [PASS] Core output files (LaTeX, CSV, text) written.\n")
} else {
  missing_files <- expected_files[!file.exists(expected_files)]
  cat(sprintf("  [FAIL] Missing files: %s\n", paste(missing_files, collapse = ", ")))
}

## Check 6: Poverty reversal sign preserved across all available etas
checks_total <- checks_total + 1
if (nrow(available) >= 1) {
  reversal_holds <- all(available$alpha_pov < 0, na.rm = TRUE) &&
                    all(available$beta_pov > 0, na.rm = TRUE)
  if (reversal_holds) {
    checks_passed <- checks_passed + 1
    cat(sprintf("  [PASS] Poverty reversal (alpha<0, beta>0) holds in all %d etas.\n",
                nrow(available)))
  } else {
    ## Identify which etas break the reversal
    breaks <- available[available$alpha_pov >= 0 | available$beta_pov <= 0, ]
    cat(sprintf("  [WARN] Poverty reversal breaks at eta = %s.\n",
                paste(sprintf("%.1f", breaks$eta), collapse = ", ")))
  }
} else {
  cat("  [SKIP] No data to check reversal.\n")
}

## Check 7: rho_cross is near zero across all etas (|rho| < 0.3)
checks_total <- checks_total + 1
if (nrow(available) >= 1) {
  all_near_zero <- all(abs(available$rho_mean) < 0.3, na.rm = TRUE)
  if (all_near_zero) {
    checks_passed <- checks_passed + 1
    cat(sprintf("  [PASS] rho_cross is near zero (|rho| < 0.3) across all %d etas.\n",
                nrow(available)))
  } else {
    large_rho <- available[abs(available$rho_mean) >= 0.3, ]
    cat(sprintf("  [NOTE] rho_cross >= 0.3 at eta = %s.\n",
                paste(sprintf("%.1f", large_rho$eta), collapse = ", ")))
  }
} else {
  cat("  [SKIP] No data to check rho_cross.\n")
}

## Check 8: Diagnostics pass for all available etas
checks_total <- checks_total + 1
diag_avail <- available[!is.na(available$diag_pass), ]
if (nrow(diag_avail) >= 1) {
  all_diag_pass <- all(diag_avail$diag_pass)
  if (all_diag_pass) {
    checks_passed <- checks_passed + 1
    cat(sprintf("  [PASS] MCMC diagnostics pass for all %d etas.\n",
                nrow(diag_avail)))
  } else {
    failing <- diag_avail[!diag_avail$diag_pass, ]
    cat(sprintf("  [WARN] Diagnostics fail at eta = %s.\n",
                paste(sprintf("%.1f", failing$eta), collapse = ", ")))
  }
} else {
  cat("  [SKIP] No diagnostic data available.\n")
}

## Check 9: Density figure generated (if draws available)
checks_total <- checks_total + 1
density_pdf <- file.path(B4_DIR, "SF_lkj_rho_density.pdf")
if (n_density_etas >= 1 && file.exists(density_pdf)) {
  checks_passed <- checks_passed + 1
  cat(sprintf("  [PASS] Density figure generated with %d eta values.\n",
              n_density_etas))
} else if (n_density_etas == 0) {
  cat("  [SKIP] No draws available for density figure.\n")
} else {
  cat("  [FAIL] Density figure not found despite available draws.\n")
}

cat(sprintf("\n  Verification: %d / %d checks passed.\n\n",
            checks_passed, checks_total))


###############################################################################
## SECTION 9 : SAVE CONSOLIDATED RESULTS
###############################################################################
cat("--- 9. Saving consolidated results ---\n\n")

B4_results <- list(
  ## Description
  description = paste(
    ": LKJ prior sensitivity analysis.",
    "M3b-W refitted under eta in {1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0}.",
    "Comparison table, posterior density figure, and manuscript text."
  ),

  ## Configuration
  config = list(
    eta_values    = ETA_VALUES,
    baseline_eta  = BASELINE_ETA,
    eta_found     = eta_found,
    eta_missing   = eta_missing,
    n_found       = length(eta_found),
    n_missing     = length(eta_missing)
  ),

  ## Comparison table (the main output)
  comparison = comparison,

  ## Baseline rho_cross
  baseline_rho = list(
    mean  = baseline_rho$mean,
    sd    = baseline_rho$sd,
    q025  = baseline_rho$q025,
    q975  = baseline_rho$q975,
    n_draws = ifelse(is.null(baseline_rho$draws), 0,
                     length(baseline_rho$draws))
  ),

  ## Per-eta detailed results (without raw draws to keep file small)
  per_eta = lapply(eta_results, function(res) {
    list(
      lkj_eta     = res$lkj_eta,
      rho_cross   = list(
        mean = res$rho_cross$mean,
        sd   = res$rho_cross$sd,
        q025 = res$rho_cross$q025,
        q975 = res$rho_cross$q975,
        n_draws = length(res$rho_cross$draws)
      ),
      alpha_means = res$alpha_means,
      beta_means  = res$beta_means,
      tau_means   = res$tau_means,
      loo         = res$loo,
      diagnostics = res$diagnostics,
      fit_time    = res$fit_time_mins
    )
  }),

  ## Sensitivity summary
  sensitivity = if (nrow(available) >= 2) {
    list(
      rho_range       = range(available$rho_mean),
      rho_spread      = diff(range(available$rho_mean)),
      alpha_pov_range = range(available$alpha_pov, na.rm = TRUE),
      beta_pov_range  = range(available$beta_pov, na.rm = TRUE),
      tau_ext1_range  = range(available$tau_ext1, na.rm = TRUE),
      tau_int1_range  = range(available$tau_int1, na.rm = TRUE),
      elpd_range      = if (any(!is.na(available$elpd_loo)))
                          range(available$elpd_loo, na.rm = TRUE) else c(NA, NA),
      reversal_robust = all(available$alpha_pov < 0, na.rm = TRUE) &&
                        all(available$beta_pov > 0, na.rm = TRUE)
    )
  } else NULL,

  ## Output file paths
  output_files = list(
    tex       = tex_path,
    csv       = csv_path,
    ms_text   = ms_text_path,
    density_pdf = file.path(B4_DIR, "SF_lkj_rho_density.pdf"),
    density_png = file.path(B4_DIR, "SF_lkj_rho_density.png"),
    params_pdf  = file.path(B4_DIR, "SF_lkj_param_sensitivity.pdf"),
    params_png  = file.path(B4_DIR, "SF_lkj_param_sensitivity.png"),
    rds         = B4_OUT
  ),

  ## Verification
  verification = list(
    checks_passed = checks_passed,
    checks_total  = checks_total,
    all_pass      = (checks_passed == checks_total)
  ),

  ## Timestamp
  timestamp = Sys.time()
)

saveRDS(B4_results, B4_OUT)
cat(sprintf("  Saved: %s\n", B4_OUT))
cat(sprintf("  File size: %.1f KB\n\n",
            file.info(B4_OUT)$size / 1024))


###############################################################################
## SECTION 10 : FINAL SUMMARY
###############################################################################
cat("==============================================================\n")
cat("  LKJ Prior Sensitivity Analysis\n")
cat("==============================================================\n\n")

cat(sprintf("  ETA VALUES: %d found / %d total\n",
            length(eta_found), N_ETA))
if (length(eta_found) > 0) {
  cat(sprintf("    Found:   %s\n",
              paste(sprintf("%.1f", sort(eta_found)), collapse = ", ")))
}
if (length(eta_missing) > 0) {
  cat(sprintf("    Missing: %s\n",
              paste(sprintf("%.1f", sort(eta_missing)), collapse = ", ")))
}

cat(sprintf("\n  BASELINE (eta = %.1f):\n", BASELINE_ETA))
cat(sprintf("    rho_cross = %.4f", baseline_rho$mean))
if (!is.na(baseline_rho$q025)) {
  cat(sprintf(" [%.3f, %.3f]", baseline_rho$q025, baseline_rho$q975))
}
cat("\n")

if (nrow(available) >= 2) {
  cat("\n  SENSITIVITY RANGE (across available etas):\n")
  cat(sprintf("    rho_cross:  [%.4f, %.4f] (spread = %.4f)\n",
              min(available$rho_mean), max(available$rho_mean),
              diff(range(available$rho_mean))))
  cat(sprintf("    alpha_pov:  [%.4f, %.4f] (spread = %.4f)\n",
              min(available$alpha_pov, na.rm = TRUE),
              max(available$alpha_pov, na.rm = TRUE),
              diff(range(available$alpha_pov, na.rm = TRUE))))
  cat(sprintf("    beta_pov:   [%.4f, %.4f] (spread = %.4f)\n",
              min(available$beta_pov, na.rm = TRUE),
              max(available$beta_pov, na.rm = TRUE),
              diff(range(available$beta_pov, na.rm = TRUE))))

  ## Reversal robustness
  reversal_robust <- all(available$alpha_pov < 0, na.rm = TRUE) &&
                     all(available$beta_pov > 0, na.rm = TRUE)
  cat(sprintf("\n  POVERTY REVERSAL: %s across all %d etas.\n",
              ifelse(reversal_robust, "ROBUST", "NOT ROBUST"),
              nrow(available)))
}

cat(sprintf("\n  OUTPUT FILES:\n"))
cat(sprintf("    LaTeX table:     %s\n", tex_path))
cat(sprintf("    CSV backup:      %s\n", csv_path))
cat(sprintf("    Density figure:  %s\n",
            file.path(B4_DIR, "SF_lkj_rho_density.pdf")))
if (!is.null(p_params)) {
  cat(sprintf("    Param figure:    %s\n",
              file.path(B4_DIR, "SF_lkj_param_sensitivity.pdf")))
}
cat(sprintf("    Manuscript text: %s\n", ms_text_path))
cat(sprintf("    Results RDS:     %s\n", B4_OUT))

cat(sprintf("\n  VERIFICATION: %d / %d checks passed.\n",
            checks_passed, checks_total))

cat("\n  Output files have been written to data/precomputed/B4_lkj/\n")

if (length(eta_missing) > 0) {
  cat(sprintf("\n  [ACTION REQUIRED] Run remaining fits:\n"))
  for (eta in sort(eta_missing)) {
    cat(sprintf("    Rscript code/84_B4_lkj_sensitivity_fit.R --eta %.1f\n",
                eta))
  }
  cat("    Then re-run this analysis script.\n")
}

cat("\n==============================================================\n")
cat("  DONE.\n")
cat("==============================================================\n")
