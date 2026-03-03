## =============================================================================
## B5_misspec_analysis.R -- Misspecification Simulation Analysis
## =============================================================================
## Purpose : Aggregate misspecification simulation results (R=200),
##           compare with correctly-specified S3 results, produce manuscript
##           tables and figures.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##
## Requires: B5_rep_001.rds through B5_rep_200.rds in
##           data/precomputed/B5_misspec/results/
##
## Outputs:
##   Tables:  ST_B5_misspec.tex, ST_B5_misspec.csv
##   Figures: SF_B5_misspec_coverage.pdf/png, SF_B5_misspec_bias.pdf/png
##   Data:    B5_summary.rds, B5_raw_metrics.rds
## =============================================================================

cat("\n")
cat("##################################################################\n")
cat("##  B5: Misspecification Simulation — Analysis                  ##\n")
cat("##################################################################\n\n")

t_start <- proc.time()


###############################################################################
## SECTION 0 : SETUP
###############################################################################

cat("=== SECTION 0: Setup ===\n\n")

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
cat(sprintf("  Project root: %s\n", PROJECT_ROOT))

## Source simulation config (for true parameter values)
.SIM_01_CALLED_FROM_PARENT <- TRUE
.SIM_02_CALLED_FROM_PARENT <- TRUE
.SIM_03_CALLED_FROM_PARENT <- TRUE
.SIM_04_CALLED_FROM_PARENT <- TRUE
source(file.path(PROJECT_ROOT, "code/simulation/sim_00_config.R"))
source(file.path(PROJECT_ROOT, "code/simulation/sim_04_postprocess.R"))

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
  library(scales)
  library(xtable)
})
cat("  Packages loaded.\n")

## Paths
B5_ROOT     <- file.path(PROJECT_ROOT, "data/precomputed/B5_misspec")
B5_RESULTS  <- file.path(B5_ROOT, "results")
FIGURE_DIR  <- file.path(PROJECT_ROOT, "output/tables")
MANUSCRIPT_DIR <- file.path(PROJECT_ROOT, "output/figures")

## Original S3 simulation summary (correctly specified)
S3_SUMMARY_PATH <- file.path(PROJECT_ROOT,
  "data/precomputed/simulation/results/sim_summary_S3.rds")

## Constants
R_TARGET   <- 200L
NOMINAL    <- 0.90
MCSE_NOM   <- sqrt(NOMINAL * (1 - NOMINAL) / R_TARGET)  # 0.02121
PARAM_ORDER <- c("alpha_poverty", "beta_poverty", "log_kappa",
                  "tau_ext", "tau_int")
PARAM_LATEX <- c(
  "alpha_poverty" = "$\\alpha_{\\text{poverty}}$",
  "beta_poverty"  = "$\\beta_{\\text{poverty}}$",
  "log_kappa"     = "$\\log\\kappa$",
  "tau_ext"       = "$\\tau_{\\text{ext}}$",
  "tau_int"       = "$\\tau_{\\text{int}}$"
)
PARAM_DISPLAY <- c(
  "alpha_poverty" = "alpha[poverty]",
  "beta_poverty"  = "beta[poverty]",
  "log_kappa"     = "log(kappa)",
  "tau_ext"       = "tau[ext]",
  "tau_int"       = "tau[int]"
)
EST_LABELS <- c("E_UW" = "E-UW", "E_WT" = "E-WT", "E_WS" = "E-WS")
EST_COLORS_DISP <- c("E-UW" = "#4393C3", "E-WT" = "#D6604D", "E-WS" = "#1B7837")
EST_SHAPES_DISP <- c("E-UW" = 16, "E-WT" = 17, "E-WS" = 15)

TRUE_VAL_LOOKUP <- setNames(
  sapply(SIM_CONFIG$evaluation$target_params, function(tp) tp$true_value),
  sapply(SIM_CONFIG$evaluation$target_params, function(tp) tp$name)
)

## Theme
theme_manuscript <- function(base_size = 10) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor  = element_blank(),
      panel.grid.major  = element_line(color = "grey90", linewidth = 0.3),
      strip.background  = element_rect(fill = "grey95", color = "grey70"),
      strip.text        = element_text(face = "bold", size = base_size),
      axis.title        = element_text(size = base_size),
      axis.text         = element_text(size = base_size - 1),
      plot.title        = element_text(face = "bold", size = base_size + 1, hjust = 0),
      plot.subtitle     = element_text(size = base_size - 1, hjust = 0, color = "grey40"),
      legend.position   = "bottom",
      legend.text       = element_text(size = base_size - 1),
      legend.title      = element_text(size = base_size, face = "bold"),
      plot.margin       = margin(5, 10, 5, 5)
    )
}

save_figure <- function(plot, name, width = 7, height = 5) {
  pdf_path <- file.path(FIGURE_DIR, paste0(name, ".pdf"))
  png_path <- file.path(FIGURE_DIR, paste0(name, ".png"))
  ggsave(pdf_path, plot, width = width, height = height, device = "pdf")
  ggsave(png_path, plot, width = width, height = height, dpi = 300)
  ## Also copy to manuscript Figures/ directory
  manu_pdf <- file.path(MANUSCRIPT_DIR, paste0(name, ".pdf"))
  file.copy(pdf_path, manu_pdf, overwrite = TRUE)
  cat(sprintf("  [SAVED] %s.pdf / .png (%g x %g in)\n", name, width, height))
}

save_table <- function(df, name, caption = "", ...) {
  csv_path <- file.path(FIGURE_DIR, paste0(name, ".csv"))
  tex_path <- file.path(FIGURE_DIR, paste0(name, ".tex"))
  write.csv(df, csv_path, row.names = FALSE)
  xt <- xtable(df, caption = caption, ...)
  print(xt, file = tex_path, include.rownames = FALSE,
        booktabs = TRUE, sanitize.text.function = identity,
        floating = TRUE, table.placement = "htbp")
  ## Also copy to manuscript Figures/ directory
  manu_tex <- file.path(MANUSCRIPT_DIR, paste0(name, ".tex"))
  file.copy(tex_path, manu_tex, overwrite = TRUE)
  cat(sprintf("  [SAVED] %s.csv / .tex\n", name))
}

cat("  Setup complete.\n\n")


###############################################################################
## SECTION 1 : LOAD AND VALIDATE B5 RESULTS
###############################################################################

cat("=== SECTION 1: Load B5 Results ===\n\n")

## Find all B5_rep_NNN.rds files
rep_files <- list.files(B5_RESULTS, pattern = "^B5_rep_\\d{3}\\.rds$",
                        full.names = TRUE)
rep_ids   <- as.integer(gsub("^B5_rep_(\\d{3})\\.rds$", "\\1",
                              basename(rep_files)))
rep_ids   <- sort(rep_ids)

cat(sprintf("  Found %d / %d rep result files\n", length(rep_ids), R_TARGET))

if (length(rep_ids) == 0) {
  stop("[FATAL] No B5 result files found in ", B5_RESULTS)
}

## Load all results
all_results <- vector("list", length(rep_ids))
names(all_results) <- sprintf("rep_%03d", rep_ids)
n_ok   <- 0L
n_fail <- 0L
failed_reps <- integer(0)

for (i in seq_along(rep_ids)) {
  r <- readRDS(rep_files[i])
  all_results[[i]] <- r
  if (identical(r$status, "OK")) {
    n_ok <- n_ok + 1L
  } else {
    n_fail <- n_fail + 1L
    failed_reps <- c(failed_reps, rep_ids[i])
  }
}

cat(sprintf("  Successful: %d / %d (%.1f%%)\n",
            n_ok, length(rep_ids), 100 * n_ok / length(rep_ids)))
if (n_fail > 0) {
  cat(sprintf("  Failed reps: %s\n", paste(failed_reps, collapse = ", ")))
}

## Extract metrics from successful reps
metrics_list <- list()
for (i in seq_along(all_results)) {
  r <- all_results[[i]]
  if (identical(r$status, "OK") && !is.null(r$metrics)) {
    metrics_list[[length(metrics_list) + 1]] <- r$metrics
  }
}

R_valid <- length(metrics_list)
cat(sprintf("  Valid metrics: %d replications\n", R_valid))

## Extract timing info
timings <- sapply(all_results[sapply(all_results, function(r) r$status == "OK")],
                  function(r) r$timing$total)
if (length(timings) > 0) {
  cat(sprintf("  Per-rep timing: mean=%.1f, median=%.1f, range=[%.1f, %.1f] sec\n",
              mean(timings), median(timings), min(timings), max(timings)))
}

## Extract convergence diagnostics summary
n_diverg <- 0L
max_rhat <- 0
for (r in all_results) {
  if (identical(r$status, "OK") && !is.null(r$diagnostics)) {
    for (fit_diag in r$diagnostics) {
      if (!is.null(fit_diag$n_divergent)) {
        n_diverg <- n_diverg + fit_diag$n_divergent
      }
      if (!is.null(fit_diag$max_rhat)) {
        max_rhat <- max(max_rhat, fit_diag$max_rhat)
      }
    }
  }
}
cat(sprintf("  Total divergences: %d\n", n_diverg))
cat(sprintf("  Max R-hat across all fits: %.4f\n", max_rhat))

cat("\n")


###############################################################################
## SECTION 2 : AGGREGATE METRICS
###############################################################################

cat("=== SECTION 2: Aggregate Metrics ===\n\n")

## Use the same aggregate_metrics function from the simulation pipeline
b5_summary <- aggregate_metrics(metrics_list, SIM_CONFIG)
b5_summary$scenario_id <- "B5"

## Add relative bias
b5_summary$rel_bias_pct <- NA_real_
for (i in seq_len(nrow(b5_summary))) {
  tv <- TRUE_VAL_LOOKUP[b5_summary$param[i]]
  if (!is.na(tv) && abs(tv) > 1e-10) {
    b5_summary$rel_bias_pct[i] <- 100 * b5_summary$mean_bias[i] / tv
  }
}

## Coverage flag
b5_summary$cov_flag <- ifelse(
  b5_summary$coverage < NOMINAL - 2 * b5_summary$coverage_mcse, "*",
  ifelse(b5_summary$coverage > NOMINAL + 2 * b5_summary$coverage_mcse, "+", "")
)

## Print summary
print_metrics_summary(b5_summary, "B5 (Misspecified)", SIM_CONFIG)

## Also combine raw per-rep metrics
b5_raw <- do.call(rbind, metrics_list)
b5_raw$scenario_id <- "B5"
b5_raw$rep_id <- rep(seq_len(R_valid), each = length(PARAM_ORDER) * 3)

## Save
saveRDS(b5_summary, file.path(B5_RESULTS, "B5_summary.rds"))
saveRDS(b5_raw,     file.path(B5_RESULTS, "B5_raw_metrics.rds"))
cat(sprintf("  Saved: B5_summary.rds (%d rows)\n", nrow(b5_summary)))
cat(sprintf("  Saved: B5_raw_metrics.rds (%d rows)\n", nrow(b5_raw)))

cat("\n")


###############################################################################
## SECTION 3 : LOAD ORIGINAL S3 RESULTS FOR COMPARISON
###############################################################################

cat("=== SECTION 3: Load Original S3 Results ===\n\n")

if (!file.exists(S3_SUMMARY_PATH)) {
  stop("[FATAL] S3 summary not found: ", S3_SUMMARY_PATH)
}

s3_summary <- readRDS(S3_SUMMARY_PATH)
s3_summary$scenario_id <- "S3"

## Add relative bias to S3 if not present
if (!"rel_bias_pct" %in% names(s3_summary)) {
  s3_summary$rel_bias_pct <- NA_real_
  for (i in seq_len(nrow(s3_summary))) {
    tv <- TRUE_VAL_LOOKUP[s3_summary$param[i]]
    if (!is.na(tv) && abs(tv) > 1e-10) {
      s3_summary$rel_bias_pct[i] <- 100 * s3_summary$mean_bias[i] / tv
    }
  }
}

cat(sprintf("  S3 summary loaded: %d rows, R=%d\n",
            nrow(s3_summary), s3_summary$R[1]))

## Combine B5 and S3 for comparison
## Harmonize columns
common_cols <- intersect(names(b5_summary), names(s3_summary))
combined <- rbind(b5_summary[, common_cols], s3_summary[, common_cols])

cat(sprintf("  Combined comparison: %d rows\n", nrow(combined)))
cat("\n")


###############################################################################
## SECTION 4 : COMPARISON TABLE (ST_B5_misspec)
###############################################################################

cat("=== SECTION 4: Comparison Table ===\n\n")

## Build side-by-side comparison: S3 (correctly specified) vs B5 (misspecified)
## For each parameter x estimator: coverage, rel_bias, RMSE, width_ratio

build_comparison_table <- function(s3_df, b5_df) {
  rows <- list()

  for (p_name in PARAM_ORDER) {
    for (est_id in c("E_UW", "E_WT", "E_WS")) {
      s3_row <- s3_df[s3_df$param == p_name & s3_df$estimator == est_id, ]
      b5_row <- b5_df[b5_df$param == p_name & b5_df$estimator == est_id, ]

      if (nrow(s3_row) == 0 || nrow(b5_row) == 0) next

      ## Format coverage with dagger if outside 2*MCSE
      fmt_cov <- function(row) {
        val <- sprintf("%.1f", 100 * row$coverage)
        if (abs(row$coverage - NOMINAL) > 2 * row$coverage_mcse) {
          val <- paste0(val, "$^{\\dagger}$")
        }
        val
      }

      ## Format rel bias
      fmt_rb <- function(row) {
        if (is.na(row$rel_bias_pct)) return("---")
        sprintf("%+.1f", row$rel_bias_pct)
      }

      ## Format RMSE
      fmt_rmse <- function(row) sprintf("%.3f", row$rmse)

      ## Width ratio (only for E-WS)
      wr_s3 <- if (est_id == "E_WS" && !is.na(s3_row$width_ratio))
                  sprintf("%.2f", s3_row$width_ratio) else "---"
      wr_b5 <- if (est_id == "E_WS" && !is.na(b5_row$width_ratio))
                  sprintf("%.2f", b5_row$width_ratio) else "---"

      row <- data.frame(
        Parameter    = if (est_id == "E_UW") PARAM_LATEX[p_name] else "",
        Estimator    = EST_LABELS[est_id],
        `Cov (S3)`   = fmt_cov(s3_row),
        `Cov (B5)`   = fmt_cov(b5_row),
        `RB (S3)`    = fmt_rb(s3_row),
        `RB (B5)`    = fmt_rb(b5_row),
        `RMSE (S3)`  = fmt_rmse(s3_row),
        `RMSE (B5)`  = fmt_rmse(b5_row),
        `WR (S3)`    = wr_s3,
        `WR (B5)`    = wr_b5,
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
      rows[[length(rows) + 1]] <- row
    }
  }
  do.call(rbind, rows)
}

T_B5 <- build_comparison_table(s3_summary, b5_summary)
rownames(T_B5) <- NULL

save_table(T_B5, "ST_B5_misspec",
  caption = paste0(
    "Misspecification Robustness: Coverage (\\%), Relative Bias (\\%), RMSE, ",
    "and Width Ratio Under Correctly Specified (S3) vs.\\ Misspecified (B5) DGP. ",
    "B5 generates from M2 (state-varying poverty slopes), fits M1 (random intercepts). ",
    "$R = ", R_valid, "$. ",
    "Nominal coverage = 90\\%; $\\dagger$ indicates significant departure ",
    "(more than $2 \\times$ MCSE from nominal)."
  ))

cat("\n")


###############################################################################
## SECTION 5 : COVERAGE COMPARISON FIGURE (SF_B5_misspec_coverage)
###############################################################################

cat("=== SECTION 5: Coverage Comparison Figure ===\n\n")

## Prepare data for plotting
plot_data <- combined %>%
  filter(param %in% PARAM_ORDER) %>%
  mutate(
    param_f     = factor(param, levels = rev(PARAM_ORDER),
                         labels = rev(PARAM_DISPLAY[PARAM_ORDER])),
    est_label   = EST_LABELS[estimator],
    scenario_label = ifelse(scenario_id == "S3",
                            "S3: Correctly specified",
                            "B5: Misspecified (M2 truth, M1 fit)")
  )

## Coverage dot plot: side-by-side panels for S3 vs B5
p_cov <- ggplot(plot_data,
       aes(x = coverage, y = param_f, color = est_label,
           shape = est_label)) +
  geom_vline(xintercept = NOMINAL, linetype = "dashed", color = "grey50",
             linewidth = 0.4) +
  geom_vline(xintercept = NOMINAL + c(-2, 2) * MCSE_NOM,
             linetype = "dotted", color = "grey70", linewidth = 0.3) +
  geom_point(size = 3, position = position_dodge(width = 0.4)) +
  facet_wrap(~ scenario_label, ncol = 2) +
  scale_color_manual(values = EST_COLORS_DISP, name = "Estimator") +
  scale_shape_manual(values = EST_SHAPES_DISP, name = "Estimator") +
  scale_x_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0.70, 1.0),
    breaks = seq(0.70, 1.0, by = 0.05)
  ) +
  scale_y_discrete(labels = function(x) parse(text = x)) +
  labs(
    title = "Coverage Under Correct Specification vs. Misspecification",
    subtitle = sprintf("S3 scenario (DEFF %.1f), R = %d replications, 90%% nominal",
                       3.79, R_valid),
    x = "Empirical Coverage",
    y = NULL
  ) +
  theme_manuscript() +
  theme(legend.position = "bottom")

save_figure(p_cov, "SF_B5_misspec_coverage", width = 8, height = 4.5)


###############################################################################
## SECTION 6 : RELATIVE BIAS COMPARISON FIGURE (SF_B5_misspec_bias)
###############################################################################

cat("=== SECTION 6: Relative Bias Comparison Figure ===\n\n")

## Add relative bias for plotting
plot_data_rb <- plot_data %>%
  filter(!is.na(rel_bias_pct))

p_bias <- ggplot(plot_data_rb,
       aes(x = rel_bias_pct, y = param_f, color = est_label,
           shape = est_label)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50",
             linewidth = 0.4) +
  geom_point(size = 3, position = position_dodge(width = 0.4)) +
  facet_wrap(~ scenario_label, ncol = 2) +
  scale_color_manual(values = EST_COLORS_DISP, name = "Estimator") +
  scale_shape_manual(values = EST_SHAPES_DISP, name = "Estimator") +
  scale_y_discrete(labels = function(x) parse(text = x)) +
  labs(
    title = "Relative Bias Under Correct Specification vs. Misspecification",
    subtitle = sprintf("S3 scenario (DEFF %.1f), R = %d replications",
                       3.79, R_valid),
    x = "Relative Bias (%)",
    y = NULL
  ) +
  theme_manuscript() +
  theme(legend.position = "bottom")

save_figure(p_bias, "SF_B5_misspec_bias", width = 8, height = 4.5)


###############################################################################
## SECTION 7 : SUMMARY DIAGNOSTICS
###############################################################################

cat("=== SECTION 7: Summary Diagnostics ===\n\n")

## DER summary from B5 sandwich results
n_pd_fix <- sum(sapply(all_results, function(r) {
  if (!is.null(r$sandwich_summary) && !is.null(r$sandwich_summary$pd_fix)) {
    r$sandwich_summary$pd_fix
  } else {
    FALSE
  }
}))

der_values <- do.call(rbind, lapply(all_results[sapply(all_results, function(r)
    identical(r$status, "OK") && !is.null(r$sandwich_summary))], function(r) {
  der <- r$sandwich_summary$DER
  if (is.null(der) || length(der) == 0) return(NULL)
  data.frame(
    param = c("alpha_poverty", "beta_poverty", "log_kappa"),
    DER   = der[c(2, 7, 11)],  ## indices for poverty coeffs + log_kappa
    stringsAsFactors = FALSE
  )
}))

if (!is.null(der_values) && nrow(der_values) > 0) {
  cat("  DER summary (misspecification):\n")
  for (p in c("alpha_poverty", "beta_poverty", "log_kappa")) {
    d <- der_values$DER[der_values$param == p]
    cat(sprintf("    %-16s: mean=%.3f, median=%.3f, range=[%.3f, %.3f]\n",
                p, mean(d), median(d), min(d), max(d)))
  }
}

cat(sprintf("\n  nearPD corrections applied: %d / %d reps (%.1f%%)\n",
            n_pd_fix, R_valid, 100 * n_pd_fix / max(R_valid, 1)))


###############################################################################
## SECTION 8 : KEY FINDINGS SUMMARY
###############################################################################

cat("\n=== SECTION 8: Key Findings ===\n\n")

## Print side-by-side comparison for key parameters
cat("  Coverage comparison (S3 correct vs B5 misspecified):\n")
cat(sprintf("  %-16s %-6s %10s %10s %10s\n",
            "Parameter", "Est", "S3 Cov%", "B5 Cov%", "Delta"))
cat(sprintf("  %s\n", paste(rep("-", 58), collapse = "")))

for (p_name in PARAM_ORDER) {
  for (est_id in c("E_UW", "E_WT", "E_WS")) {
    s3_r <- s3_summary[s3_summary$param == p_name &
                         s3_summary$estimator == est_id, ]
    b5_r <- b5_summary[b5_summary$param == p_name &
                         b5_summary$estimator == est_id, ]
    if (nrow(s3_r) > 0 && nrow(b5_r) > 0) {
      delta <- 100 * (b5_r$coverage - s3_r$coverage)
      cat(sprintf("  %-16s %-6s %9.1f%% %9.1f%% %+9.1f pp\n",
                  p_name, EST_LABELS[est_id],
                  100 * s3_r$coverage, 100 * b5_r$coverage, delta))
    }
  }
}

## Key narrative findings
cat("\n  Key narrative findings:\n")

## 1. Does E-WS rescue coverage under misspecification?
ews_s3_cov <- b5_summary$coverage[b5_summary$estimator == "E_WS" &
                                     b5_summary$param == "alpha_poverty"]
ewt_b5_cov <- b5_summary$coverage[b5_summary$estimator == "E_WT" &
                                     b5_summary$param == "alpha_poverty"]
ews_b5_cov <- b5_summary$coverage[b5_summary$estimator == "E_WS" &
                                     b5_summary$param == "alpha_poverty"]

if (length(ews_b5_cov) > 0 && length(ewt_b5_cov) > 0) {
  cat(sprintf("    alpha_poverty E-WT coverage under misspec: %.1f%%\n",
              100 * ewt_b5_cov))
  cat(sprintf("    alpha_poverty E-WS coverage under misspec: %.1f%%\n",
              100 * ews_b5_cov))
  if (ews_b5_cov > ewt_b5_cov) {
    cat("    => Sandwich correction IMPROVES coverage under misspecification\n")
  } else {
    cat("    => Sandwich correction does NOT improve coverage under misspecification\n")
  }
}


###############################################################################
## SECTION 9 : FILE INVENTORY
###############################################################################

t_total <- (proc.time() - t_start)["elapsed"]

cat(sprintf("\n=== Analysis completed in %.1f sec (%.1f min) ===\n\n", t_total, t_total/60))

cat("  Output files:\n")
output_files <- c(
  file.path(FIGURE_DIR, "ST_B5_misspec.tex"),
  file.path(FIGURE_DIR, "ST_B5_misspec.csv"),
  file.path(FIGURE_DIR, "SF_B5_misspec_coverage.pdf"),
  file.path(FIGURE_DIR, "SF_B5_misspec_coverage.png"),
  file.path(FIGURE_DIR, "SF_B5_misspec_bias.pdf"),
  file.path(FIGURE_DIR, "SF_B5_misspec_bias.png"),
  file.path(B5_RESULTS, "B5_summary.rds"),
  file.path(B5_RESULTS, "B5_raw_metrics.rds")
)
for (f in output_files) {
  exists <- file.exists(f)
  size_str <- if (exists) sprintf("%.1f KB", file.info(f)$size / 1024) else "MISSING"
  cat(sprintf("    [%s] %s (%s)\n",
              if (exists) "OK" else "!!",
              basename(f), size_str))
}

cat("\n  B5 analysis done.\n")
