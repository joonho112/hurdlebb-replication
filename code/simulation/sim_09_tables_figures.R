## =============================================================================
## sim_09_tables_figures.R -- Simulation Tables and Figures
## =============================================================================
## Purpose : Aggregate 600 simulation replications (R=200 x 3 scenarios)
##           into manuscript-ready tables and figures (Table 7, Figure 7,
##           and supplementary materials ST7, ST8, SF7, SF8, SF9).
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Statistical details:
##   - Coverage: proportion of R reps where true value in CI
##   - MCSE for coverage: sqrt(p*(1-p)/R); at R=200, p=0.90: MCSE = 0.02121
##   - Relative bias = mean(estimate - true_value) / |true_value| x 100%
##   - RMSE = sqrt(mean((estimate - true_value)^2))
##   - Width ratio = mean_ci_width[E-WS] / mean_ci_width[E-WT]
##   - DER = diag(V_sand) / diag(H_obs_inv)
##   - tau no-sandwich: E-WS tau = E-WT tau (WR = 1.00 exactly)
##   - 90% coverage level: z = qnorm(0.95) = 1.6449
##
## Pipeline:
##   Section 0  : Setup (paths, packages, config, constants)
##   Section 1  : Sample regeneration (556 of 600 missing)
##   Section 2  : Batch post-processing with checkpointing
##   Section 3  : Aggregation across replications
##   Section 4  : Table 7 (main text -- publication quality)
##   Section 5  : Figure 7 (main text -- Cleveland coverage dot plot)
##   Section 6  : Supplementary Table ST7 (full metrics, all 45 rows)
##   Section 7  : Supplementary Table ST8 (DER summary)
##   Section 8  : Supplementary Figure SF7 (relative bias dot plot)
##   Section 9  : Supplementary Figure SF8 (CI width ratio boxplot)
##   Section 10 : Supplementary Figure SF9 (DER summary dot plot)
##   Section 11 : Validation & file inventory
##
## Style Conventions:
##   Colors: E-UW = #4393C3 (blue), E-WT = #D6604D (coral), E-WS = #1B7837 (green)
##   Shapes: E-UW = circle (16), E-WT = triangle (17), E-WS = square (15)
##
## Usage:
##   source("code/simulation/sim_09_tables_figures.R")
##   (from project root)
## =============================================================================

cat("\n")
cat("##################################################################\n")
cat("##  Simulation Tables & Figures                   ##\n")
cat("##  sim_09_tables_figures.R                                     ##\n")
cat("##################################################################\n\n")

SCRIPT_START <- proc.time()


###############################################################################
## SECTION 0 : SETUP
###############################################################################

cat("=== SECTION 0: Setup ===\n\n")

## --- 0a. Project root ---
PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
                           
cat(sprintf("  Project root: %s\n", PROJECT_ROOT))

## --- 0b. Source simulation modules with guard variables ---
.SIM_01_CALLED_FROM_PARENT <- TRUE
.SIM_02_CALLED_FROM_PARENT <- TRUE
.SIM_03_CALLED_FROM_PARENT <- TRUE
.SIM_04_CALLED_FROM_PARENT <- TRUE

cat("  Sourcing sim_00_config.R ...\n")
source(file.path(PROJECT_ROOT, "code/simulation/sim_00_config.R"))

cat("  Sourcing sim_01_dgp.R ...\n")
source(file.path(PROJECT_ROOT, "code/simulation/sim_01_dgp.R"))

cat("  Sourcing sim_02_sampling.R ...\n")
source(file.path(PROJECT_ROOT, "code/simulation/sim_02_sampling.R"))

cat("  Sourcing sim_04_postprocess.R ...\n")
source(file.path(PROJECT_ROOT, "code/simulation/sim_04_postprocess.R"))

## --- 0c. Load packages ---
## Note: tidyr and forcats NOT loaded as libraries; namespace-qualified when needed
cat("  Loading packages ...\n")
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
  library(scales)
  library(xtable)
})
cat("  Packages loaded: ggplot2, dplyr, patchwork, scales, xtable\n")

## --- 0d. Core constants ---
SCENARIOS  <- SIM_CONFIG$scenario_ids                      # c("S0", "S3", "S4")
R_TOTAL    <- SIM_CONFIG$evaluation$R                      # 200
N_TARGETS  <- length(SIM_CONFIG$evaluation$target_params)  # 5
NOMINAL    <- SIM_CONFIG$evaluation$coverage_level         # 0.90
MCSE_NOM   <- SIM_CONFIG$evaluation$mcse_coverage          # 0.02121
CKPT_EVERY <- 50L

## --- 0e. Output directories ---
## Simulation figures go alongside Step 6 outputs in output/tables/
RESULTS_DIR <- file.path(SIM_CONFIG$paths$sim_output_root, "results")
FIGURE_DIR  <- file.path(PROJECT_ROOT, "output/tables")

for (d in c(RESULTS_DIR, FIGURE_DIR)) {
  if (!dir.exists(d)) {
    dir.create(d, recursive = TRUE, showWarnings = FALSE)
    cat(sprintf("  [CREATED] %s\n", d))
  }
}
cat(sprintf("  Results directory: %s\n", RESULTS_DIR))
cat(sprintf("  Figure directory:  %s\n", FIGURE_DIR))

## --- 0f. Display constants ---

## Parameter ordering (top to bottom on y-axis)
PARAM_ORDER <- c("alpha_poverty", "beta_poverty", "log_kappa",
                  "tau_ext", "tau_int")

## Parameter display labels (plotmath expressions for ggplot axes)
PARAM_DISPLAY <- c(
  "alpha_poverty" = "alpha[poverty]",
  "beta_poverty"  = "beta[poverty]",
  "log_kappa"     = "log(kappa)",
  "tau_ext"       = "tau[ext]",
  "tau_int"       = "tau[int]"
)

## LaTeX labels for tables
PARAM_LATEX <- c(
  "alpha_poverty" = "$\\alpha_{\\text{poverty}}$",
  "beta_poverty"  = "$\\beta_{\\text{poverty}}$",
  "log_kappa"     = "$\\log\\kappa$",
  "tau_ext"       = "$\\tau_{\\text{ext}}$",
  "tau_int"       = "$\\tau_{\\text{int}}$"
)

## Scenario labels
SCENARIO_LABELS <- c(
  "S0" = "S0: Non-informative",
  "S3" = "S3: NSECE-calibrated",
  "S4" = "S4: Stress test"
)

## Scenario design effect factors (for table annotation)
SCENARIO_DEFF <- c("S0" = 2.0, "S3" = 3.79, "S4" = 5.0)

## Estimator labels, colors, and shapes (consistent with Step 6)
EST_LABELS <- c("E_UW" = "E-UW", "E_WT" = "E-WT", "E_WS" = "E-WS")
EST_COLORS <- c("E_UW" = "#4393C3", "E_WT" = "#D6604D", "E_WS" = "#1B7837")
EST_SHAPES <- c("E_UW" = 16, "E_WT" = 17, "E_WS" = 15)
## Display-label-keyed versions for scale_*_manual (avoid nested named vectors)
EST_COLORS_DISP <- c("E-UW" = "#4393C3", "E-WT" = "#D6604D", "E-WS" = "#1B7837")
EST_SHAPES_DISP <- c("E-UW" = 16, "E-WT" = 17, "E-WS" = 15)

## True-value lookup vector (avoids assuming aggregate_metrics output has true_value)
TRUE_VAL_LOOKUP <- setNames(
  sapply(SIM_CONFIG$evaluation$target_params, function(tp) tp$true_value),
  sapply(SIM_CONFIG$evaluation$target_params, function(tp) tp$name)
)

## Sandwich-applicable parameters (fixed effects only)
SANDWICH_PARAMS <- c("alpha_poverty", "beta_poverty", "log_kappa")

## Panel subsets for two-row figure layouts (F7, SF7)
FIXED_PARAMS  <- c("alpha_poverty", "beta_poverty", "log_kappa")
FIXED_DISPLAY <- PARAM_DISPLAY[FIXED_PARAMS]
HYPER_PARAMS  <- c("tau_ext", "tau_int")
HYPER_DISPLAY <- PARAM_DISPLAY[HYPER_PARAMS]

## Sandwich applicability flags per target parameter
SANDWICH_OK <- sapply(SIM_CONFIG$evaluation$target_params,
                       function(tp) tp$sandwich_applicable)
names(SANDWICH_OK) <- sapply(SIM_CONFIG$evaluation$target_params,
                              function(tp) tp$name)

## Target parameter indices in the theta vector (D=11)
P_DIM <- SIM_CONFIG$true_params$P  # 5
D_DIM <- 2L * P_DIM + 1L           # 11
TARGET_THETA_IDX <- c(
  alpha_poverty = 2L,
  beta_poverty  = P_DIM + 2L,  # 7
  log_kappa     = D_DIM         # 11
)


## --- 0g. Theme and save helpers (matching Step 6 exactly) ---

theme_manuscript <- function(base_size = 10) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor  = element_blank(),
      panel.grid.major  = element_line(color = "grey90", linewidth = 0.3),
      strip.background  = element_rect(fill = "grey95", color = "grey70"),
      strip.text        = element_text(face = "bold", size = base_size),
      axis.title        = element_text(size = base_size),
      axis.text         = element_text(size = base_size - 1),
      plot.title        = element_text(face = "bold", size = base_size + 1,
                                       hjust = 0),
      plot.subtitle     = element_text(size = base_size - 1, hjust = 0,
                                       color = "grey40"),
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
  cat(sprintf("  [SAVED] %s.csv / .tex\n", name))
}

## --- 0h. Utility: format elapsed time ---
fmt_elapsed <- function(seconds) {
  if (seconds < 60) return(sprintf("%.1fs", seconds))
  if (seconds < 3600) return(sprintf("%.1f min", seconds / 60))
  return(sprintf("%.1f hr", seconds / 3600))
}

cat("\n  Setup complete.\n\n")


###############################################################################
## SECTION 1 : SAMPLE REGENERATION
###############################################################################
## Many sample files are missing (generated on cloud VMs, not synced back).
## Regenerate deterministically from the population + calibrated inclusion
## probabilities. Seeds are identical to those used in Phase 4, so
## regenerated samples are bit-for-bit identical.

cat("=== SECTION 1: Sample Regeneration ===\n\n")

t1_start <- proc.time()

## --- 1a. Inventory existing samples ---
sample_inventory <- data.frame(
  scenario  = character(0),
  n_exist   = integer(0),
  n_missing = integer(0),
  stringsAsFactors = FALSE
)

for (sid in SCENARIOS) {
  sample_dir <- file.path(SIM_CONFIG$paths$sim_samples, sid)
  existing   <- list.files(sample_dir, pattern = "^rep_\\d{3}\\.rds$")
  n_exist    <- length(existing)
  n_missing  <- R_TOTAL - n_exist

  sample_inventory <- rbind(sample_inventory,
    data.frame(scenario = sid, n_exist = n_exist, n_missing = n_missing,
               stringsAsFactors = FALSE))
  cat(sprintf("  %s: %d existing, %d missing\n", sid, n_exist, n_missing))
}

total_missing <- sum(sample_inventory$n_missing)
cat(sprintf("\n  Total samples to regenerate: %d / %d\n\n",
            total_missing, length(SCENARIOS) * R_TOTAL))

if (total_missing > 0) {

  ## --- 1b. Load population ---
  pop_path <- file.path(SIM_CONFIG$paths$sim_population, "pop_base.rds")
  if (!file.exists(pop_path)) {
    stop("[FATAL] Population file not found: ", pop_path)
  }
  cat(sprintf("  Loading population: %s\n", pop_path))
  pop <- readRDS(pop_path)
  pop_data <- pop$data
  cat(sprintf("  Population loaded: M=%s providers\n",
              format(nrow(pop_data), big.mark = ",")))

  ## --- 1c. Calibrate all scenarios ---
  cat("  Calibrating inclusion probabilities ...\n")
  calibs <- calibrate_all_scenarios(pop_data, SIM_CONFIG, verbose = TRUE)

  ## --- 1d. Regenerate missing samples ---
  n_regenerated <- 0L
  regen_errors  <- 0L

  for (sid in SCENARIOS) {
    sample_dir <- file.path(SIM_CONFIG$paths$sim_samples, sid)
    cat(sprintf("\n  --- Regenerating samples for %s ---\n", sid))

    for (rep_id in seq_len(R_TOTAL)) {
      sample_path <- file.path(sample_dir, sprintf("rep_%03d.rds", rep_id))

      if (file.exists(sample_path)) next

      result <- tryCatch({
        seed <- get_rep_seed(SIM_CONFIG$seeds$base_seed, rep_id, sid)
        samp <- draw_sample(pop_data, SIM_CONFIG$scenarios[[sid]],
                            SIM_CONFIG, calibs[[sid]], seed, verbose = FALSE)
        saveRDS(samp, sample_path)
        n_regenerated <<- n_regenerated + 1L
        TRUE
      }, error = function(e) {
        cat(sprintf("  [ERROR] %s rep %03d: %s\n", sid, rep_id, e$message))
        regen_errors <<- regen_errors + 1L
        FALSE
      })

      if (rep_id %% 50 == 0) {
        cat(sprintf("    %s: %d/%d reps checked\n", sid, rep_id, R_TOTAL))
      }
    }
  }

  cat(sprintf("\n  Regeneration complete: %d created, %d errors\n",
              n_regenerated, regen_errors))

  ## --- 1e. Verification (spot-check stored vs regenerated) ---
  cat("\n  --- Verification: stored vs regenerated ---\n")
  n_verified <- 0L
  n_mismatch <- 0L

  for (sid in SCENARIOS) {
    for (rep_id in 1:3) {
      sample_path <- file.path(SIM_CONFIG$paths$sim_samples, sid,
                                sprintf("rep_%03d.rds", rep_id))
      if (!file.exists(sample_path)) next

      stored <- tryCatch(readRDS(sample_path), error = function(e) NULL)
      if (is.null(stored)) next

      seed <- get_rep_seed(SIM_CONFIG$seeds$base_seed, rep_id, sid)
      regen <- tryCatch(
        draw_sample(pop_data, SIM_CONFIG$scenarios[[sid]],
                    SIM_CONFIG, calibs[[sid]], seed, verbose = FALSE),
        error = function(e) NULL
      )
      if (is.null(regen)) next

      ## Compare w_tilde, stratum_idx, psu_idx, N
      match_w <- isTRUE(all.equal(stored$stan_data_wt$w_tilde,
                                   regen$stan_data_wt$w_tilde,
                                   tolerance = 1e-10))
      match_s <- identical(stored$stan_data_wt$stratum_idx,
                           regen$stan_data_wt$stratum_idx)
      match_p <- identical(stored$stan_data_wt$psu_idx,
                           regen$stan_data_wt$psu_idx)
      match_n <- identical(stored$stan_data_wt$N, regen$stan_data_wt$N)

      if (match_w && match_s && match_p && match_n) {
        n_verified <- n_verified + 1L
        cat(sprintf("    [PASS] %s rep %03d: all fields match\n", sid, rep_id))
      } else {
        n_mismatch <- n_mismatch + 1L
        cat(sprintf("    [FAIL] %s rep %03d: mismatch (w=%s s=%s p=%s n=%s)\n",
                    sid, rep_id, match_w, match_s, match_p, match_n))
      }
    }
  }

  cat(sprintf("  Verification: %d passed, %d mismatched\n",
              n_verified, n_mismatch))
  if (n_mismatch > 0) {
    warning("Sample verification mismatches detected!")
  }

  rm(pop, pop_data, calibs)
  invisible(gc(verbose = FALSE))

} else {
  cat("  All sample files already exist. Skipping regeneration.\n")
}

t1_elapsed <- (proc.time() - t1_start)["elapsed"]
cat(sprintf("\n  Section 1 completed in %s\n\n", fmt_elapsed(t1_elapsed)))


###############################################################################
## SECTION 2 : BATCH POST-PROCESSING WITH CHECKPOINTING
###############################################################################
## For each scenario x rep: postprocess_saved_fits() computes per-rep
## metrics (15 rows: 5 params x 3 estimators) and sandwich DER.
##
## Robustness features:
##   - tryCatch around every postprocess call
##   - Retry on warnings (some reps produce harmless warnings)
##   - Checkpoint every CKPT_EVERY=50 reps (resume from last checkpoint)
##   - Progress logging with timing per rep and ETA
##   - Memory cleanup after each rep
##
## Stores both V_diag AND H_diag from sandwich for DER cross-validation.

cat("=== SECTION 2: Batch Post-Processing ===\n\n")

t2_start <- proc.time()

all_scenario_results <- list()

for (sid in SCENARIOS) {

  cat(sprintf("  ========== Scenario %s ==========\n", sid))

  ## --- Check for checkpoint ---
  ckpt_path <- file.path(RESULTS_DIR, sprintf("checkpoint_%s.rds", sid))

  if (file.exists(ckpt_path)) {
    cat(sprintf("  [RESUME] Loading checkpoint: %s\n", ckpt_path))
    ckpt <- readRDS(ckpt_path)
    start_rep     <- ckpt$last_completed + 1L
    metrics_list  <- ckpt$metrics_list
    sandwich_list <- ckpt$sandwich_list
    diag_list     <- ckpt$diag_list
    timing_vec    <- ckpt$timing_vec
    n_failed      <- ckpt$n_failed
    cat(sprintf("  [RESUME] Resuming from rep %d (completed: %d, failed: %d)\n",
                start_rep, ckpt$last_completed, n_failed))
  } else {
    start_rep     <- 1L
    metrics_list  <- vector("list", R_TOTAL)
    sandwich_list <- vector("list", R_TOTAL)
    diag_list     <- vector("list", R_TOTAL)
    timing_vec    <- numeric(R_TOTAL)
    n_failed      <- 0L
  }

  ## Skip if already fully processed
  if (start_rep > R_TOTAL) {
    cat(sprintf("  [SKIP] %s already fully processed.\n\n", sid))
    all_scenario_results[[sid]] <- list(
      metrics_list  = metrics_list,
      sandwich_list = sandwich_list,
      diag_list     = diag_list,
      timing_vec    = timing_vec,
      n_failed      = n_failed
    )
    next
  }

  ## --- Main loop over reps ---
  batch_start <- proc.time()
  completed_in_batch <- 0L

  for (rep_id in start_rep:R_TOTAL) {

    t_rep <- proc.time()

    pp <- tryCatch(
      {
        ## Suppress verbose output from postprocess_saved_fits
        invisible(capture.output(
          result <- postprocess_saved_fits(sid, rep_id, SIM_CONFIG),
          type = "output"
        ))
        result
      },
      error = function(e) {
        cat(sprintf("  [ERROR] %s rep %03d: %s\n", sid, rep_id, e$message))
        NULL
      },
      warning = function(w) {
        cat(sprintf("  [WARN]  %s rep %03d: %s\n", sid, rep_id, w$message))
        ## Retry on warning (some reps produce harmless warnings)
        tryCatch(
          {
            invisible(capture.output(
              result <- postprocess_saved_fits(sid, rep_id, SIM_CONFIG),
              type = "output"
            ))
            result
          },
          error = function(e) {
            cat(sprintf("  [ERROR] %s rep %03d (retry): %s\n",
                        sid, rep_id, e$message))
            NULL
          }
        )
      }
    )

    rep_elapsed <- (proc.time() - t_rep)["elapsed"]
    timing_vec[rep_id] <- rep_elapsed

    if (!is.null(pp)) {
      metrics_list[[rep_id]] <- pp$metrics

      ## Store sandwich summary: DER vector (length D=11), V_sand diagonal,
      ## H_obs_inv diagonal (for DER cross-validation), PD fix flag
      sandwich_list[[rep_id]] <- list(
        DER     = pp$sandwich$DER,
        V_diag  = diag(pp$sandwich$V_sand),
        H_diag  = diag(pp$sandwich$H_obs_inv),
        pd_fix  = pp$sandwich$pd_fix_applied,
        H_ridge = pp$sandwich$H_ridge_applied
      )

      diag_list[[rep_id]] <- list(
        diagnostics_uw = pp$diagnostics_uw,
        diagnostics_wt = pp$diagnostics_wt
      )

      completed_in_batch <- completed_in_batch + 1L
    } else {
      n_failed <- n_failed + 1L
      cat(sprintf("  [SKIP] %s rep %03d: postprocess returned NULL\n",
                  sid, rep_id))
    }

    ## Memory cleanup
    rm(pp)
    invisible(gc(verbose = FALSE))

    ## --- Checkpoint ---
    if (rep_id %% CKPT_EVERY == 0) {
      saveRDS(
        list(last_completed = rep_id, metrics_list = metrics_list,
             sandwich_list = sandwich_list, diag_list = diag_list,
             timing_vec = timing_vec, n_failed = n_failed),
        ckpt_path
      )

      batch_elapsed <- (proc.time() - batch_start)["elapsed"]
      reps_done     <- rep_id - start_rep + 1L
      avg_per_rep   <- batch_elapsed / reps_done
      eta_seconds   <- avg_per_rep * (R_TOTAL - rep_id)

      cat(sprintf("  [CHECKPOINT] %s: %d/%d | %.1fs/rep | failed=%d | ETA: %s\n",
                  sid, rep_id, R_TOTAL, avg_per_rep, n_failed,
                  fmt_elapsed(eta_seconds)))
    }

    ## Progress every 10 reps (but not at checkpoints, which already log)
    if (rep_id %% 10 == 0 && rep_id %% CKPT_EVERY != 0) {
      cat(sprintf("    %s: %d/%d (%.1fs)\n", sid, rep_id, R_TOTAL,
                  rep_elapsed))
    }
  }

  ## --- Final checkpoint ---
  saveRDS(
    list(last_completed = R_TOTAL, metrics_list = metrics_list,
         sandwich_list = sandwich_list, diag_list = diag_list,
         timing_vec = timing_vec, n_failed = n_failed),
    ckpt_path
  )

  n_success <- sum(!sapply(metrics_list, is.null))
  batch_elapsed <- (proc.time() - batch_start)["elapsed"]

  cat(sprintf("\n  %s complete: %d/%d successful, %d failed, %s elapsed\n",
              sid, n_success, R_TOTAL, n_failed, fmt_elapsed(batch_elapsed)))
  if (any(timing_vec > 0)) {
    active <- timing_vec[timing_vec > 0]
    cat(sprintf("  Timing: mean=%.1fs, median=%.1fs, max=%.1fs per rep\n",
                mean(active), median(active), max(active)))
  }

  all_scenario_results[[sid]] <- list(
    metrics_list  = metrics_list,
    sandwich_list = sandwich_list,
    diag_list     = diag_list,
    timing_vec    = timing_vec,
    n_failed      = n_failed
  )

  cat("\n")
}

t2_elapsed <- (proc.time() - t2_start)["elapsed"]
cat(sprintf("  Section 2 completed in %s\n\n", fmt_elapsed(t2_elapsed)))


###############################################################################
## SECTION 3 : AGGREGATION
###############################################################################

cat("=== SECTION 3: Aggregation ===\n\n")

t3_start <- proc.time()

all_summaries   <- list()
all_raw_metrics <- list()

for (sid in SCENARIOS) {
  ml <- all_scenario_results[[sid]]$metrics_list
  valid_idx <- which(!sapply(ml, is.null))
  valid_metrics <- ml[valid_idx]

  n_valid <- length(valid_metrics)
  cat(sprintf("  %s: aggregating %d valid replications (of %d)\n",
              sid, n_valid, R_TOTAL))

  if (n_valid == 0) {
    cat(sprintf("  [WARN] %s: NO valid replications!\n", sid))
    next
  }

  ## Aggregate via aggregate_metrics()
  summary_df <- aggregate_metrics(valid_metrics, SIM_CONFIG)
  summary_df$scenario_id <- sid
  all_summaries[[sid]] <- summary_df

  ## Collect raw per-rep metrics
  raw_combined <- do.call(rbind, valid_metrics)
  raw_combined$scenario_id <- sid
  raw_combined$rep_id      <- rep(valid_idx, each = N_TARGETS * 3)
  all_raw_metrics[[sid]] <- raw_combined

  print_metrics_summary(summary_df, sid, SIM_CONFIG)
}

## Combine
combined_summary <- bind_rows(all_summaries)
rownames(combined_summary) <- NULL

combined_raw <- bind_rows(all_raw_metrics)
rownames(combined_raw) <- NULL

## Compute relative bias (%) using TRUE_VAL_LOOKUP
## (aggregate_metrics output does NOT include a true_value column)
combined_summary$rel_bias_pct <- NA_real_
for (i in seq_len(nrow(combined_summary))) {
  tv <- TRUE_VAL_LOOKUP[combined_summary$param[i]]
  if (!is.na(tv) && abs(tv) > 1e-10) {
    combined_summary$rel_bias_pct[i] <-
      100 * combined_summary$mean_bias[i] / tv
  }
}

## Coverage flag: significantly outside [nominal - 2*MCSE, nominal + 2*MCSE]
combined_summary$cov_flag <- ifelse(
  combined_summary$coverage < NOMINAL - 2 * combined_summary$coverage_mcse,
  "*",
  ifelse(
    combined_summary$coverage > NOMINAL + 2 * combined_summary$coverage_mcse,
    "+", ""
  )
)

## Save aggregated data
saveRDS(combined_summary, file.path(RESULTS_DIR, "sim_summary_all.rds"))
saveRDS(combined_raw,     file.path(RESULTS_DIR, "sim_raw_all.rds"))
cat(sprintf("  Saved: sim_summary_all.rds (%d rows)\n", nrow(combined_summary)))
cat(sprintf("  Saved: sim_raw_all.rds (%d rows)\n", nrow(combined_raw)))

## Save per-scenario summaries and sandwich data
for (sid in SCENARIOS) {
  if (!is.null(all_summaries[[sid]])) {
    saveRDS(all_summaries[[sid]],
            file.path(RESULTS_DIR, sprintf("sim_summary_%s.rds", sid)))
  }
  sl <- all_scenario_results[[sid]]$sandwich_list
  valid_sand <- sl[!sapply(sl, is.null)]
  if (length(valid_sand) > 0) {
    saveRDS(valid_sand,
            file.path(RESULTS_DIR, sprintf("sim_sandwich_%s.rds", sid)))
    cat(sprintf("  Saved: sim_sandwich_%s.rds (%d reps)\n",
                sid, length(valid_sand)))
  }
}

t3_elapsed <- (proc.time() - t3_start)["elapsed"]
cat(sprintf("\n  Section 3 completed in %s\n\n", fmt_elapsed(t3_elapsed)))


###############################################################################
## SECTION 4 : TABLE 7 (Main Text -- Publication Quality)
###############################################################################
## Layout: Three panels (A, B, C) stacked vertically for S0, S3, S4.
## Columns: Parameter | Coverage (%) by estimator | Rel. Bias (%) by estimator
##          | RMSE by estimator | Width Ratio (E-WS / E-WT)
##
## Coverage values outside [nominal - 2*MCSE, nominal + 2*MCSE] are flagged
## with a dagger ($\dagger$) in the LaTeX output.

cat("=== SECTION 4: Table 7 (Main Text) ===\n\n")

if (nrow(combined_summary) == 0) {
  cat("  [WARN] No data for Table 7. Skipping.\n\n")
} else {

  ## Helper: build one scenario panel
  build_panel_rows <- function(df_s, sid) {
    rows <- list()
    for (p_name in PARAM_ORDER) {
      sub <- df_s[df_s$param == p_name, ]
      if (nrow(sub) == 0) next

      ## Format coverage with dagger flag
      fmt_cov <- function(est_id) {
        r <- sub[sub$estimator == est_id, ]
        if (nrow(r) == 0) return("")
        val <- sprintf("%.1f", 100 * r$coverage)
        if (abs(r$coverage - NOMINAL) > 2 * r$coverage_mcse) {
          val <- paste0(val, "$^{\\dagger}$")
        }
        val
      }

      ## Format relative bias
      fmt_rb <- function(est_id) {
        r <- sub[sub$estimator == est_id, ]
        if (nrow(r) == 0 || is.na(r$rel_bias_pct)) return("---")
        sprintf("%+.1f", r$rel_bias_pct)
      }

      ## Format RMSE
      fmt_rmse <- function(est_id) {
        r <- sub[sub$estimator == est_id, ]
        if (nrow(r) == 0) return("")
        sprintf("%.3f", r$rmse)
      }

      ## Width ratio
      wr_row <- sub[sub$estimator == "E_WS", ]
      wr_str <- if (nrow(wr_row) > 0 && !is.na(wr_row$width_ratio))
                  sprintf("%.2f", wr_row$width_ratio) else "---"

      row <- data.frame(
        Parameter      = PARAM_LATEX[p_name],
        `Cov E-UW`     = fmt_cov("E_UW"),
        `Cov E-WT`     = fmt_cov("E_WT"),
        `Cov E-WS`     = fmt_cov("E_WS"),
        `RB E-UW`      = fmt_rb("E_UW"),
        `RB E-WT`      = fmt_rb("E_WT"),
        `RB E-WS`      = fmt_rb("E_WS"),
        `RMSE E-UW`    = fmt_rmse("E_UW"),
        `RMSE E-WT`    = fmt_rmse("E_WT"),
        `RMSE E-WS`    = fmt_rmse("E_WS"),
        WR             = wr_str,
        stringsAsFactors = FALSE,
        check.names      = FALSE
      )
      rows[[length(rows) + 1]] <- row
    }
    do.call(rbind, rows)
  }

  ## Build full table with panel headers
  t7_list <- list()
  panel_letters <- c("S0" = "A", "S3" = "B", "S4" = "C")

  for (sid in SCENARIOS) {
    df_s <- combined_summary[combined_summary$scenario_id == sid, ]
    if (nrow(df_s) == 0) next

    R_val <- df_s$R[1]
    panel_header <- sprintf(
      "\\textit{Panel %s: %s (DEFF $\\approx$ %.1f, R = %d)}",
      panel_letters[sid],
      SIM_CONFIG$scenarios[[sid]]$label,
      SCENARIO_DEFF[sid],
      R_val
    )

    ## Panel header row
    header_row <- data.frame(
      Parameter = panel_header,
      `Cov E-UW` = "", `Cov E-WT` = "", `Cov E-WS` = "",
      `RB E-UW` = "", `RB E-WT` = "", `RB E-WS` = "",
      `RMSE E-UW` = "", `RMSE E-WT` = "", `RMSE E-WS` = "",
      WR = "",
      stringsAsFactors = FALSE, check.names = FALSE
    )

    panel_body <- build_panel_rows(df_s, sid)
    t7_list[[sid]] <- rbind(header_row, panel_body)
  }

  T7 <- do.call(rbind, t7_list)
  rownames(T7) <- NULL

  save_table(T7, "T7_simulation_results",
    caption = paste0(
      "Simulation Study Results: Coverage (\\%), Relative Bias (\\%), RMSE, ",
      "and Width Ratio for Three Estimators Across Three Scenarios. ",
      "E-UW = unweighted; E-WT = weighted (naive pseudo-posterior); ",
      "E-WS = weighted (sandwich-corrected). ",
      "WR = E-WS/E-WT mean CI width ratio. ",
      "Nominal coverage = 90\\%. ",
      "$\\dagger$ = coverage deviates by $>$ 2 MCSE from nominal."
    )
  )

  ## Also save a plain-text CSV (no LaTeX markup)
  T7_plain <- T7
  T7_plain$Parameter <- gsub("\\$.*?\\$", "", T7_plain$Parameter)
  T7_plain$Parameter <- gsub("\\\\textit\\{|\\}", "", T7_plain$Parameter)
  for (col in names(T7_plain)) {
    T7_plain[[col]] <- gsub("\\$\\^\\{\\\\dagger\\}\\$", "*", T7_plain[[col]])
  }
  write.csv(T7_plain, file.path(FIGURE_DIR, "T7_simulation_results_plain.csv"),
            row.names = FALSE)
  cat("  [SAVED] T7_simulation_results_plain.csv\n")

  cat("  Table 7 generated.\n\n")
}


###############################################################################
## SECTION 5 : FIGURE 7 (Main Text -- Two-Panel Coverage Dot Plot)
###############################################################################
## Two-row patchwork layout:
##   (a) Fixed effects + log(kappa): 3 params, x-axis 20-100%
##   (b) Hyperparameters: 2 params, x-axis 0-100% (full range for tau 0%)
## This prevents tau's 0% coverage from compressing the informative region.
##   - Three horizontal facet panels by scenario (S0, S3, S4)
##   - Three estimators: color + shape (EST_COLORS_DISP), vertically dodged
##   - Error bars: 90% CI for coverage = +/- 1.645 * MCSE
##   - Reference: dashed line at 90% nominal; shaded band = nominal +/- 2 MCSE
##   - Output: 7.5 x 6 inches (PDF + PNG 300 DPI)

cat("=== SECTION 5: Figure 7 (Cleveland Coverage Dot Plot) ===\n\n")

if (nrow(combined_summary) == 0) {
  cat("  [WARN] No data for Figure 7. Skipping.\n\n")
} else {

  ## ---- (a) Fixed effects panel: separate factor with 3 levels ----
  f7a <- combined_summary %>%
    filter(param %in% FIXED_PARAMS) %>%
    mutate(
      cov_lo = pmax(0, coverage - qnorm(0.95) * coverage_mcse),
      cov_hi = pmin(1, coverage + qnorm(0.95) * coverage_mcse),
      param_label = factor(PARAM_DISPLAY[param],
                           levels = rev(FIXED_DISPLAY)),
      scenario_label = factor(SCENARIO_LABELS[scenario_id],
                              levels = SCENARIO_LABELS[SCENARIOS]),
      est_label = factor(EST_LABELS[estimator], levels = EST_LABELS)
    )

  f7a$y_num <- as.numeric(f7a$param_label)
  f7a$y_dodge <- f7a$y_num + case_when(
    f7a$estimator == "E_UW" ~  0.25,
    f7a$estimator == "E_WT" ~  0.00,
    f7a$estimator == "E_WS" ~ -0.25,
    TRUE ~ 0
  )

  p_f7a <- ggplot(f7a, aes(x = coverage, y = y_dodge,
                            color = est_label, shape = est_label)) +
    annotate("rect",
             xmin = NOMINAL - 2 * MCSE_NOM,
             xmax = NOMINAL + 2 * MCSE_NOM,
             ymin = -Inf, ymax = Inf,
             fill = "grey88", alpha = 0.5) +
    geom_vline(xintercept = NOMINAL,
               linetype = "dashed", color = "grey40", linewidth = 0.5) +
    geom_errorbarh(aes(xmin = cov_lo, xmax = cov_hi),
                   height = 0.15, linewidth = 0.45) +
    geom_point(size = 2.8) +
    facet_wrap(~ scenario_label, ncol = 3) +
    scale_color_manual(values = EST_COLORS_DISP, name = "Estimator") +
    scale_shape_manual(values = EST_SHAPES_DISP, name = "Estimator") +
    scale_x_continuous(
      breaks = seq(0.2, 1, 0.2),
      labels = percent_format(accuracy = 1),
      limits = c(0.10, 1.00)
    ) +
    scale_y_continuous(
      breaks = 1:3,
      labels = rev(parse(text = FIXED_DISPLAY)),
      expand = expansion(add = 0.5)
    ) +
    labs(x = NULL, y = NULL,
         subtitle = "(a) Fixed effects and overdispersion") +
    theme_manuscript(base_size = 10) +
    theme(
      panel.grid.major.y = element_blank(),
      legend.position    = "none",
      plot.subtitle      = element_text(face = "bold", size = 10,
                                        color = "black", hjust = 0)
    )

  ## ---- (b) Hyperparameters panel: separate factor with 2 levels ----
  f7b <- combined_summary %>%
    filter(param %in% HYPER_PARAMS) %>%
    mutate(
      cov_lo = pmax(0, coverage - qnorm(0.95) * coverage_mcse),
      cov_hi = pmin(1, coverage + qnorm(0.95) * coverage_mcse),
      param_label = factor(PARAM_DISPLAY[param],
                           levels = rev(HYPER_DISPLAY)),
      scenario_label = factor(SCENARIO_LABELS[scenario_id],
                              levels = SCENARIO_LABELS[SCENARIOS]),
      est_label = factor(EST_LABELS[estimator], levels = EST_LABELS)
    )

  f7b$y_num <- as.numeric(f7b$param_label)
  f7b$y_dodge <- f7b$y_num + case_when(
    f7b$estimator == "E_UW" ~  0.25,
    f7b$estimator == "E_WT" ~  0.00,
    f7b$estimator == "E_WS" ~ -0.25,
    TRUE ~ 0
  )

  p_f7b <- ggplot(f7b, aes(x = coverage, y = y_dodge,
                            color = est_label, shape = est_label)) +
    annotate("rect",
             xmin = NOMINAL - 2 * MCSE_NOM,
             xmax = NOMINAL + 2 * MCSE_NOM,
             ymin = -Inf, ymax = Inf,
             fill = "grey88", alpha = 0.5) +
    geom_vline(xintercept = NOMINAL,
               linetype = "dashed", color = "grey40", linewidth = 0.5) +
    geom_errorbarh(aes(xmin = cov_lo, xmax = cov_hi),
                   height = 0.15, linewidth = 0.45) +
    geom_point(size = 2.8) +
    facet_wrap(~ scenario_label, ncol = 3) +
    scale_color_manual(values = EST_COLORS_DISP, name = "Estimator") +
    scale_shape_manual(values = EST_SHAPES_DISP, name = "Estimator") +
    scale_x_continuous(
      breaks = seq(0, 1, 0.2),
      labels = percent_format(accuracy = 1),
      limits = c(-0.02, 1.02)
    ) +
    scale_y_continuous(
      breaks = 1:2,
      labels = rev(parse(text = HYPER_DISPLAY)),
      expand = expansion(add = 0.5)
    ) +
    labs(x = "Coverage Rate", y = NULL,
         subtitle = "(b) Hyperparameters (no sandwich correction; E-WS = E-WT)") +
    theme_manuscript(base_size = 10) +
    theme(
      panel.grid.major.y = element_blank(),
      legend.position    = "bottom",
      legend.key.width   = unit(1.2, "cm"),
      plot.subtitle      = element_text(face = "bold", size = 10,
                                        color = "black", hjust = 0),
      strip.text         = element_blank(),
      strip.background   = element_blank()
    ) +
    guides(color = guide_legend(override.aes = list(size = 3)))

  ## Compose with patchwork (two-row layout)
  p_f7 <- p_f7a / p_f7b +
    plot_layout(heights = c(3, 2)) +
    plot_annotation(
      title = "Simulation Study: 90% CI Coverage Rates",
      subtitle = paste0(
        "R = ", R_TOTAL,
        " replications per scenario; error bars = 90% CI for coverage; ",
        "grey band = nominal +/- 2 MCSE"
      ),
      theme = theme(
        plot.title = element_text(face = "bold", size = 12, hjust = 0),
        plot.subtitle = element_text(size = 9, hjust = 0, color = "grey40")
      )
    )

  save_figure(p_f7, "F7_simulation_coverage", width = 8.5, height = 6)
  cat("  Figure 7 generated.\n\n")
}


###############################################################################
## SECTION 6 : SUPPLEMENTARY TABLE ST7 (Full Simulation Metrics)
###############################################################################
## All 45 rows (3 scenarios x 5 params x 3 estimators) with every column
## from aggregate_metrics().

cat("=== SECTION 6: Supplementary Table ST7 (Full Metrics) ===\n\n")

if (nrow(combined_summary) == 0) {
  cat("  [WARN] No data for ST7. Skipping.\n\n")
} else {

  ST7 <- combined_summary %>%
    arrange(
      factor(scenario_id, levels = SCENARIOS),
      factor(param, levels = PARAM_ORDER),
      factor(estimator, levels = c("E_UW", "E_WT", "E_WS"))
    ) %>%
    transmute(
      Scenario       = SCENARIO_LABELS[scenario_id],
      Parameter      = PARAM_LATEX[param],
      Estimator      = EST_LABELS[estimator],
      R              = R,
      `True Value`   = sprintf("%+.6f", TRUE_VAL_LOOKUP[param]),
      `Coverage (\\%)` = sprintf("%.1f", 100 * coverage),
      MCSE           = sprintf("%.1f", 100 * coverage_mcse),
      `Mean Bias`    = sprintf("%+.5f", mean_bias),
      `Median Bias`  = sprintf("%+.5f", median_bias),
      RMSE           = sprintf("%.5f", rmse),
      `Mean CI Width` = sprintf("%.5f", mean_ci_width),
      `Median CI Width` = sprintf("%.5f", median_ci_width),
      `Mean SE`      = sprintf("%.5f", mean_se),
      `Width Ratio`  = ifelse(is.na(width_ratio), "---",
                               sprintf("%.3f", width_ratio)),
      Note           = ifelse(nchar(sandwich_note) > 0, "$\\star$", "")
    )

  save_table(ST7, "ST7_simulation_full",
    caption = paste0(
      "Full Simulation Results: All Scenarios, Parameters, and Estimators ",
      "(R = ", R_TOTAL, " replications, 90\\% nominal coverage). ",
      "$\\star$ = No sandwich correction for hyperparameters ",
      "(E-WS CI = E-WT CI for $\\tau$)."
    )
  )

  cat("  ST7 generated.\n\n")
}


###############################################################################
## SECTION 7 : SUPPLEMENTARY TABLE ST8 (DER Summary)
###############################################################################
## Design Effect Ratio summary for all D=11 fixed-effect parameters.
## Includes DER cross-validation: recompute DER from stored V_diag and H_diag
## to confirm consistency with stored DER values.

cat("=== SECTION 7: Supplementary Table ST8 (DER Summary) ===\n\n")

der_summary_list <- list()

for (sid in SCENARIOS) {
  sl <- all_scenario_results[[sid]]$sandwich_list
  valid_sl <- sl[!sapply(sl, is.null)]

  if (length(valid_sl) == 0) {
    cat(sprintf("  [WARN] %s: no valid sandwich results.\n", sid))
    next
  }

  der_mat <- do.call(rbind, lapply(valid_sl, function(x) x$DER))

  ## DER cross-validation: recompute from V_diag / H_diag
  v_mat <- do.call(rbind, lapply(valid_sl, function(x) x$V_diag))
  h_mat <- do.call(rbind, lapply(valid_sl, function(x) x$H_diag))
  der_check <- v_mat / h_mat

  if (ncol(der_mat) == ncol(der_check)) {
    max_diff <- max(abs(der_mat - der_check), na.rm = TRUE)
    cat(sprintf("  %s: DER consistency check max|diff| = %.2e [%s]\n",
                sid, max_diff, ifelse(max_diff < 1e-6, "PASS", "NOTE")))
  }

  ## Use fixed_pretty labels
  D_cols <- ncol(der_mat)
  param_labels_der <- SIM_CONFIG$param_labels$fixed_pretty
  if (length(param_labels_der) != D_cols) {
    param_labels_der <- paste0("$\\theta_{", 1:D_cols, "}$")
  }

  for (d in seq_len(D_cols)) {
    v <- der_mat[, d]
    der_summary_list[[length(der_summary_list) + 1]] <- data.frame(
      Scenario  = SCENARIO_LABELS[sid],
      Parameter = param_labels_der[d],
      R         = length(v),
      `Mean`    = sprintf("%.2f", mean(v, na.rm = TRUE)),
      `Median`  = sprintf("%.2f", median(v, na.rm = TRUE)),
      SD        = sprintf("%.2f", sd(v, na.rm = TRUE)),
      Min       = sprintf("%.2f", min(v, na.rm = TRUE)),
      Q25       = sprintf("%.2f", quantile(v, 0.25, na.rm = TRUE)),
      Q75       = sprintf("%.2f", quantile(v, 0.75, na.rm = TRUE)),
      Max       = sprintf("%.2f", max(v, na.rm = TRUE)),
      stringsAsFactors = FALSE, check.names = FALSE
    )
  }

  ## PD fix counts
  n_pd <- sum(sapply(valid_sl, function(x) isTRUE(x$pd_fix)))
  n_hr <- sum(sapply(valid_sl, function(x) isTRUE(x$H_ridge)))
  cat(sprintf("  %s: %d reps, PD-fix=%d, H-ridge=%d\n",
              sid, nrow(der_mat), n_pd, n_hr))
}

if (length(der_summary_list) > 0) {
  ST8 <- do.call(rbind, der_summary_list)
  rownames(ST8) <- NULL

  save_table(ST8, "ST8_DER_summary",
    caption = paste0(
      "Design Effect Ratios (DER) Across Simulation Replications. ",
      "DER = diag($V_{\\text{sand}}$) / diag($H_{\\text{obs}}^{-1}$). ",
      "All 11 fixed-effect parameters shown."
    )
  )

  ## Save raw DER matrices
  for (sid in SCENARIOS) {
    sl <- all_scenario_results[[sid]]$sandwich_list
    valid_sl <- sl[!sapply(sl, is.null)]
    if (length(valid_sl) > 0) {
      der_mat <- do.call(rbind, lapply(valid_sl, function(x) x$DER))
      saveRDS(der_mat, file.path(RESULTS_DIR, sprintf("sim_DER_%s.rds", sid)))
    }
  }

  cat("  ST8 generated.\n\n")
} else {
  cat("  [WARN] No DER data for ST8.\n\n")
}


###############################################################################
## SECTION 8 : SUPPLEMENTARY FIGURE SF7 (Two-Panel Relative Bias Lollipop)
###############################################################################
## Two-row patchwork layout matching F7 structure:
##   (a) Fixed effects + log(kappa): zoom to +/-25%
##   (b) Hyperparameters: full range to show massive tau bias
## Color-coded lollipop stems connecting to zero reference.
## Grey band = +/-5% acceptable zone in panel (a).
## Output: 7.5 x 6 inches (PDF + PNG 300 DPI)

cat("=== SECTION 8: Supplementary Figure SF7 (Relative Bias) ===\n\n")

if (nrow(combined_summary) == 0) {
  cat("  [WARN] No data for SF7. Skipping.\n\n")
} else {

  ## ---- (a) Fixed effects — zoom to +/-25% ----
  sf7a <- combined_summary %>%
    filter(!is.na(rel_bias_pct), param %in% FIXED_PARAMS) %>%
    mutate(
      param_label = factor(PARAM_DISPLAY[param],
                           levels = rev(FIXED_DISPLAY)),
      scenario_label = factor(SCENARIO_LABELS[scenario_id],
                              levels = SCENARIO_LABELS[SCENARIOS]),
      est_label = factor(EST_LABELS[estimator], levels = EST_LABELS)
    )

  sf7a$y_num <- as.numeric(sf7a$param_label)
  sf7a$y_dodge <- sf7a$y_num + case_when(
    sf7a$estimator == "E_UW" ~  0.25,
    sf7a$estimator == "E_WT" ~  0.00,
    sf7a$estimator == "E_WS" ~ -0.25,
    TRUE ~ 0
  )

  p_sf7a <- ggplot(sf7a, aes(x = rel_bias_pct, y = y_dodge,
                               color = est_label, shape = est_label)) +
    annotate("rect", xmin = -5, xmax = 5, ymin = -Inf, ymax = Inf,
             fill = "grey88", alpha = 0.4) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey40",
               linewidth = 0.5) +
    geom_segment(aes(x = 0, xend = rel_bias_pct, y = y_dodge, yend = y_dodge),
                 linewidth = 0.4, alpha = 0.5) +
    geom_point(size = 2.8) +
    facet_wrap(~ scenario_label, ncol = 3) +
    scale_color_manual(values = EST_COLORS_DISP, name = "Estimator") +
    scale_shape_manual(values = EST_SHAPES_DISP, name = "Estimator") +
    scale_x_continuous(breaks = seq(-25, 25, 5), limits = c(-25, 25)) +
    scale_y_continuous(
      breaks = 1:3,
      labels = rev(parse(text = FIXED_DISPLAY)),
      expand = expansion(add = 0.5)
    ) +
    labs(x = NULL, y = NULL,
         subtitle = "(a) Fixed effects and overdispersion") +
    theme_manuscript(base_size = 10) +
    theme(
      panel.grid.major.y = element_blank(),
      legend.position    = "none",
      plot.subtitle      = element_text(face = "bold", size = 10,
                                        color = "black", hjust = 0)
    )

  ## ---- (b) Hyperparameters — full range ----
  sf7b <- combined_summary %>%
    filter(!is.na(rel_bias_pct), param %in% HYPER_PARAMS) %>%
    mutate(
      param_label = factor(PARAM_DISPLAY[param],
                           levels = rev(HYPER_DISPLAY)),
      scenario_label = factor(SCENARIO_LABELS[scenario_id],
                              levels = SCENARIO_LABELS[SCENARIOS]),
      est_label = factor(EST_LABELS[estimator], levels = EST_LABELS)
    )

  sf7b$y_num <- as.numeric(sf7b$param_label)
  sf7b$y_dodge <- sf7b$y_num + case_when(
    sf7b$estimator == "E_UW" ~  0.25,
    sf7b$estimator == "E_WT" ~  0.00,
    sf7b$estimator == "E_WS" ~ -0.25,
    TRUE ~ 0
  )

  p_sf7b <- ggplot(sf7b, aes(x = rel_bias_pct, y = y_dodge,
                               color = est_label, shape = est_label)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey40",
               linewidth = 0.5) +
    geom_segment(aes(x = 0, xend = rel_bias_pct, y = y_dodge, yend = y_dodge),
                 linewidth = 0.4, alpha = 0.5) +
    geom_point(size = 2.8) +
    facet_wrap(~ scenario_label, ncol = 3) +
    scale_color_manual(values = EST_COLORS_DISP, name = "Estimator") +
    scale_shape_manual(values = EST_SHAPES_DISP, name = "Estimator") +
    scale_x_continuous(breaks = seq(-20, 120, 20)) +
    scale_y_continuous(
      breaks = 1:2,
      labels = rev(parse(text = HYPER_DISPLAY)),
      expand = expansion(add = 0.5)
    ) +
    labs(x = "Relative Bias (%)", y = NULL,
         subtitle = "(b) Hyperparameters (not sandwich-correctable)") +
    theme_manuscript(base_size = 10) +
    theme(
      panel.grid.major.y = element_blank(),
      legend.position    = "bottom",
      legend.key.width   = unit(1.2, "cm"),
      plot.subtitle      = element_text(face = "bold", size = 10,
                                        color = "black", hjust = 0),
      strip.text         = element_blank(),
      strip.background   = element_blank()
    ) +
    guides(color = guide_legend(override.aes = list(size = 3)))

  ## Compose with patchwork (two-row layout)
  p_sf7 <- p_sf7a / p_sf7b +
    plot_layout(heights = c(3, 2)) +
    plot_annotation(
      title = "Simulation Study: Relative Bias by Parameter and Estimator",
      subtitle = paste0(
        "R = ", R_TOTAL,
        " replications; relative bias = 100 x mean(bias) / true value; ",
        "grey band = +/-5%"
      ),
      theme = theme(
        plot.title = element_text(face = "bold", size = 12, hjust = 0),
        plot.subtitle = element_text(size = 9, hjust = 0, color = "grey40")
      )
    )

  save_figure(p_sf7, "SF7_simulation_bias", width = 7.5, height = 6)
  cat("  SF7 generated.\n\n")
}


###############################################################################
## SECTION 9 : SUPPLEMENTARY FIGURE SF8 (CI Width Ratio Violin + Boxplot)
###############################################################################
## Violin + boxplot overlay for richer distributional information.
## Median value labels above each distribution.
## Only for sandwich-applicable parameters (3 fixed effects).
## Faceted by scenario. Reference line at 1.0.
## Output: 7 x 4.5 inches (PDF + PNG 300 DPI)

cat("=== SECTION 9: Supplementary Figure SF8 (Width Ratio) ===\n\n")

if (nrow(combined_raw) == 0) {
  cat("  [WARN] No raw data for SF8. Skipping.\n\n")
} else {

  ## Compute per-rep width ratio: pivot E-WT and E-WS ci_width columns
  wr_data <- combined_raw %>%
    filter(param %in% SANDWICH_PARAMS,
           estimator %in% c("E_WT", "E_WS")) %>%
    select(scenario_id, param, rep_id, estimator, ci_width) %>%
    tidyr::pivot_wider(
      id_cols     = c(scenario_id, param, rep_id),
      names_from  = estimator,
      values_from = ci_width
    ) %>%
    mutate(width_ratio = E_WS / E_WT) %>%
    filter(!is.na(width_ratio), is.finite(width_ratio)) %>%
    mutate(
      param_label = factor(PARAM_DISPLAY[param],
                           levels = PARAM_DISPLAY[SANDWICH_PARAMS]),
      scenario_label = factor(SCENARIO_LABELS[scenario_id],
                              levels = SCENARIO_LABELS[SCENARIOS])
    )

  if (nrow(wr_data) > 0) {

    ## Median summaries for labels
    wr_medians <- wr_data %>%
      group_by(scenario_label, param_label) %>%
      summarise(median_wr = median(width_ratio), .groups = "drop")

    p_sf8 <- ggplot(wr_data, aes(x = param_label, y = width_ratio)) +
      ## Reference line at 1.0
      geom_hline(yintercept = 1.0, linetype = "dashed", color = "grey50",
                 linewidth = 0.5) +
      ## Violin (light fill)
      geom_violin(fill = "#4393C3", alpha = 0.15, color = "#4393C3",
                  linewidth = 0.4, width = 0.7, scale = "width") +
      ## Boxplot overlay (narrow, dark)
      geom_boxplot(
        fill    = "#4393C3",
        alpha   = 0.4,
        color   = "#2166AC",
        outlier.size  = 0.5,
        outlier.alpha = 0.3,
        outlier.color = "#4393C3",
        width   = 0.18,
        linewidth = 0.4
      ) +
      ## Median labels
      geom_text(data = wr_medians,
                aes(x = param_label, y = median_wr,
                    label = sprintf("%.2f", median_wr)),
                vjust = -1.2, size = 2.8, fontface = "bold",
                color = "#2166AC") +
      ## Facets
      facet_wrap(~ scenario_label, ncol = 3) +
      ## Axis labels using parsed math
      scale_x_discrete(
        labels = parse(text = PARAM_DISPLAY[SANDWICH_PARAMS])
      ) +
      scale_y_continuous(breaks = seq(0.5, 4, 0.5)) +
      ## Labels
      labs(
        x = NULL,
        y = "CI Width Ratio  (E-WS / E-WT)",
        title = "Distribution of Sandwich-to-Naive CI Width Ratios",
        subtitle = paste0(
          "R = ", R_TOTAL,
          " replications; ratio > 1 indicates sandwich widens CI ",
          "to account for design effect"
        )
      ) +
      theme_manuscript(base_size = 10) +
      theme(
        axis.text.x = element_text(size = 10),
        panel.spacing.x = unit(1.2, "lines")
      )

    save_figure(p_sf8, "SF8_width_ratio_distribution", width = 7, height = 4.5)
    cat("  SF8 generated.\n\n")

  } else {
    cat("  [SKIP] No width ratio data for SF8.\n\n")
  }
}


###############################################################################
## SECTION 10 : SUPPLEMENTARY FIGURE SF9 (DER Summary — All 11 Parameters)
###############################################################################
## Shows ALL 11 parameters (not just 3 target params) for comprehensive view.
## Color-coded by scenario (S0/S3/S4 gradient) to show DEFF scaling.
## Thick bars = IQR, thin bars = mean +/- 1 SD.
## Margin separators and labels (Extensive/Intensive/Overdispersion).
## Output: 7 x 5.5 inches (PDF + PNG 300 DPI)

cat("=== SECTION 10: Supplementary Figure SF9 (DER Summary) ===\n\n")

## Assemble DER for ALL 11 parameters (not just 3 target params)
der_all_list <- list()

for (sid in SCENARIOS) {
  sl <- all_scenario_results[[sid]]$sandwich_list
  valid_sl <- sl[!sapply(sl, is.null)]
  if (length(valid_sl) == 0) next

  der_mat <- do.call(rbind, lapply(valid_sl, function(x) x$DER))

  ## Parameter names for all 11 (5 alpha + 5 beta + log kappa)
  param_names <- c(
    paste0("alpha[", c("int", "pov", "urb", "blk", "his"), "]"),
    paste0("beta[", c("int", "pov", "urb", "blk", "his"), "]"),
    "log(kappa)"
  )
  colnames(der_mat) <- param_names

  for (j in seq_len(ncol(der_mat))) {
    der_all_list[[length(der_all_list) + 1]] <- data.frame(
      scenario_id = sid,
      param_idx   = j,
      param       = param_names[j],
      DER         = der_mat[, j],
      stringsAsFactors = FALSE
    )
  }
}

if (length(der_all_list) > 0) {

  der_all_df <- bind_rows(der_all_list)

  ## Classify into margin
  der_all_df$margin <- case_when(
    der_all_df$param_idx <= 5  ~ "Extensive",
    der_all_df$param_idx <= 10 ~ "Intensive",
    TRUE ~ "Overdispersion"
  )

  ## Summarize per parameter x scenario
  der_summ <- der_all_df %>%
    group_by(scenario_id, param, margin) %>%
    summarise(
      mean_DER   = mean(DER, na.rm = TRUE),
      sd_DER     = sd(DER, na.rm = TRUE),
      q25_DER    = quantile(DER, 0.25, na.rm = TRUE),
      median_DER = median(DER, na.rm = TRUE),
      q75_DER    = quantile(DER, 0.75, na.rm = TRUE),
      .groups    = "drop"
    ) %>%
    mutate(
      lo = pmax(0, mean_DER - sd_DER),
      hi = mean_DER + sd_DER,
      scenario_label = factor(SCENARIO_LABELS[scenario_id],
                              levels = SCENARIO_LABELS[SCENARIOS])
    )

  ## Set param factor order (grouped by margin, bottom-to-top)
  param_order_full <- c(
    "log(kappa)",
    "beta[his]", "beta[blk]", "beta[urb]", "beta[pov]", "beta[int]",
    "alpha[his]", "alpha[blk]", "alpha[urb]", "alpha[pov]", "alpha[int]"
  )
  der_summ$param_f <- factor(der_summ$param, levels = param_order_full)

  ## Scenario colors (gradient matching design effect intensity)
  SCEN_COLORS <- c("S0" = "#92C5DE", "S3" = "#4393C3", "S4" = "#2166AC")

  p_sf9 <- ggplot(der_summ, aes(x = mean_DER, y = param_f,
                                  color = scenario_id)) +
    ## DER = 1 reference
    geom_vline(xintercept = 1.0, linetype = "dashed", color = "grey50",
               linewidth = 0.4) +
    ## Thin whiskers: mean +/- 1 SD
    geom_errorbarh(aes(xmin = lo, xmax = hi),
                   height = 0.15, linewidth = 0.35,
                   position = position_dodge(width = 0.6)) +
    ## Thick bars: IQR
    geom_errorbarh(aes(xmin = q25_DER, xmax = q75_DER),
                   height = 0.3, linewidth = 0.65,
                   position = position_dodge(width = 0.6)) +
    ## Points (mean DER)
    geom_point(size = 2.2, position = position_dodge(width = 0.6)) +
    ## Scales
    scale_color_manual(
      values = SCEN_COLORS,
      labels = c("S0" = "S0 (DEFF~2)",
                 "S3" = "S3 (DEFF~3.8)",
                 "S4" = "S4 (DEFF~5)"),
      name = "Scenario"
    ) +
    scale_y_discrete(labels = parse(text = param_order_full)) +
    scale_x_continuous(breaks = seq(0, 20, 2)) +
    ## Margin separation: horizontal lines
    ## Factor order (bottom=1 to top=11):
    ##   1=log(kappa), 2=beta_his, ..., 6=beta_int, 7=alpha_his, ..., 11=alpha_int
    geom_hline(yintercept = 1.5, color = "grey70", linewidth = 0.3,
               linetype = "dotted") +
    geom_hline(yintercept = 6.5, color = "grey70", linewidth = 0.3,
               linetype = "dotted") +
    ## Margin labels (right-aligned annotation)
    annotate("text", x = Inf, y = 9, label = "Extensive", hjust = 1.05,
             size = 2.8, fontface = "italic", color = "grey50") +
    annotate("text", x = Inf, y = 4, label = "Intensive", hjust = 1.05,
             size = 2.8, fontface = "italic", color = "grey50") +
    annotate("text", x = Inf, y = 1, label = "Overdispersion", hjust = 1.05,
             size = 2.8, fontface = "italic", color = "grey50") +
    ## Labels
    labs(
      x = "Design Effect Ratio (DER)",
      y = NULL,
      title = "Design Effect Ratios Across All Parameters",
      subtitle = paste0(
        "R = ", R_TOTAL,
        " replications; thick bars = IQR; thin bars = mean \u00b1 1 SD; ",
        "DER > 1 indicates design effect"
      )
    ) +
    theme_manuscript(base_size = 10) +
    theme(
      panel.grid.major.y = element_blank(),
      legend.key.width   = unit(1, "cm"),
      panel.spacing      = unit(1, "lines")
    ) +
    guides(color = guide_legend(override.aes = list(size = 3)))

  save_figure(p_sf9, "SF9_DER_summary", width = 7, height = 5.5)
  cat("  SF9 generated.\n\n")

} else {
  cat("  [SKIP] No DER data for SF9.\n\n")
}


###############################################################################
## SECTION 11 : VALIDATION & FILE INVENTORY
###############################################################################
## Comprehensive validation with 8 sub-checks:
##   11a. Rep counts (>= 90% success required)
##   11b. Tau invariant: E-WS == E-WT for hyperparameters (strict 1e-10)
##   11c. Coverage anomalies (severe undercoverage > 3 MCSE)
##   11d. S0 coverage: all estimators near nominal (non-informative design)
##   11e. DER plausibility (no values > 100)
##   11f. DER cross-validation: V_diag / H_diag == DER
##   11g. Expected ordering (S4: E-WS >= E-UW for all theta params)
##   11h. Pilot R=5 comparison (if available)

cat("=== SECTION 11: Validation & Inventory ===\n\n")

validation_ok <- TRUE

## --- 11a. Rep counts ---
cat("  --- 11a. Rep counts ---\n")
for (sid in SCENARIOS) {
  ml <- all_scenario_results[[sid]]$metrics_list
  n_valid <- sum(!sapply(ml, is.null))
  n_failed <- all_scenario_results[[sid]]$n_failed

  cat(sprintf("  %s: %d valid, %d failed (of %d)\n",
              sid, n_valid, n_failed, R_TOTAL))

  if (n_valid < R_TOTAL * 0.90) {
    cat(sprintf("  [ALERT] %s: fewer than 90%% reps succeeded!\n", sid))
    validation_ok <- FALSE
  }
}

## --- 11b. Tau invariant: E-WS == E-WT (coverage, bias, WR=1.00) ---
## This is a hard invariant: E-WS tau CI = E-WT tau CI by construction.
cat("\n  --- 11b. Tau invariant: E-WS = E-WT (coverage, bias, WR=1.00) ---\n")
if (nrow(combined_summary) > 0) {
  tau_params <- c("tau_ext", "tau_int")

  for (sid in SCENARIOS) {
    for (tp in tau_params) {
      cov_wt <- combined_summary$coverage[combined_summary$scenario_id == sid &
                                            combined_summary$param == tp &
                                            combined_summary$estimator == "E_WT"]
      cov_ws <- combined_summary$coverage[combined_summary$scenario_id == sid &
                                            combined_summary$param == tp &
                                            combined_summary$estimator == "E_WS"]
      bias_wt <- combined_summary$mean_bias[combined_summary$scenario_id == sid &
                                              combined_summary$param == tp &
                                              combined_summary$estimator == "E_WT"]
      bias_ws <- combined_summary$mean_bias[combined_summary$scenario_id == sid &
                                              combined_summary$param == tp &
                                              combined_summary$estimator == "E_WS"]
      wr <- combined_summary$width_ratio[combined_summary$scenario_id == sid &
                                           combined_summary$param == tp &
                                           combined_summary$estimator == "E_WS"]

      if (length(cov_wt) == 1 && length(cov_ws) == 1) {
        cov_match  <- abs(cov_wt - cov_ws) < 1e-10
        bias_match <- abs(bias_wt - bias_ws) < 1e-10
        wr_one     <- !is.na(wr) && abs(wr - 1.0) < 1e-10

        status <- if (cov_match && bias_match && wr_one) "PASS" else "FAIL"
        cat(sprintf("    %s/%s: cov_diff=%.1e, bias_diff=%.1e, WR=%.6f [%s]\n",
                    sid, tp,
                    abs(cov_wt - cov_ws), abs(bias_wt - bias_ws),
                    ifelse(!is.na(wr), wr, -1), status))
        if (status == "FAIL") validation_ok <- FALSE
      }
    }
  }
}

## --- 11c. Coverage anomalies ---
cat("\n  --- 11c. Coverage anomalies ---\n")
if (nrow(combined_summary) > 0) {
  severe <- combined_summary %>%
    filter(coverage < NOMINAL - 3 * coverage_mcse)

  if (nrow(severe) > 0) {
    cat("  Severe undercoverage (> 3 MCSE below nominal):\n")
    for (i in seq_len(nrow(severe))) {
      cat(sprintf("    %s / %s / %s: %.1f%% (deficit = %.1f MCSE)\n",
                  severe$scenario_id[i], severe$param[i], severe$estimator[i],
                  severe$coverage[i] * 100,
                  (NOMINAL - severe$coverage[i]) / severe$coverage_mcse[i]))
    }
  } else {
    cat("  No severe undercoverage detected.\n")
  }
}

## --- 11d. S0 coverage: all estimators near nominal ---
cat("\n  --- 11d. S0 coverage (non-informative: all estimators near nominal) ---\n")
if (nrow(combined_summary) > 0) {
  s0 <- combined_summary[combined_summary$scenario_id == "S0", ]
  for (i in seq_len(nrow(s0))) {
    r <- s0[i, ]
    dev_mcse <- abs(r$coverage - NOMINAL) / MCSE_NOM
    flag <- if (dev_mcse > 2) " [>2 MCSE]" else ""
    cat(sprintf("    %-16s %-6s: %.3f (%.1f MCSE)%s\n",
                r$param, r$estimator, r$coverage, dev_mcse, flag))
  }
}

## --- 11e. DER plausibility ---
cat("\n  --- 11e. DER plausibility ---\n")
for (sid in SCENARIOS) {
  sl <- all_scenario_results[[sid]]$sandwich_list
  valid_sl <- sl[!sapply(sl, is.null)]
  if (length(valid_sl) == 0) next

  der_mat <- do.call(rbind, lapply(valid_sl, function(x) x$DER))
  cat(sprintf("  %s DER: mean=%.3f, range=[%.3f, %.3f]\n",
              sid, mean(der_mat, na.rm = TRUE),
              min(der_mat, na.rm = TRUE), max(der_mat, na.rm = TRUE)))

  if (any(der_mat > 100, na.rm = TRUE)) {
    n_large <- sum(der_mat > 100, na.rm = TRUE)
    cat(sprintf("  [ALERT] %s: %d DER values > 100\n", sid, n_large))
    validation_ok <- FALSE
  }
}

## --- 11f. DER cross-validation: V_diag / H_diag == DER ---
cat("\n  --- 11f. DER cross-validation (V_diag / H_diag == DER) ---\n")
for (sid in SCENARIOS) {
  sl <- all_scenario_results[[sid]]$sandwich_list
  valid_sl <- sl[!sapply(sl, is.null)]
  if (length(valid_sl) == 0) next

  ## Check if H_diag is stored
  has_h_diag <- !is.null(valid_sl[[1]]$H_diag)
  if (!has_h_diag) {
    cat(sprintf("  %s: H_diag not stored, skipping cross-validation.\n", sid))
    next
  }

  der_mat <- do.call(rbind, lapply(valid_sl, function(x) x$DER))
  v_mat   <- do.call(rbind, lapply(valid_sl, function(x) x$V_diag))
  h_mat   <- do.call(rbind, lapply(valid_sl, function(x) x$H_diag))
  der_recomputed <- v_mat / h_mat

  max_diff <- max(abs(der_mat - der_recomputed), na.rm = TRUE)
  cat(sprintf("  %s: max|DER_stored - V_diag/H_diag| = %.2e [%s]\n",
              sid, max_diff, ifelse(max_diff < 1e-6, "PASS", "NOTE")))
}

## --- 11g. Expected ordering (S4: E-WS >= E-UW for fixed effects) ---
cat("\n  --- 11g. Expected ordering (S4: E-WS >= E-UW for all theta) ---\n")
if (nrow(combined_summary) > 0) {
  for (pname in names(TARGET_THETA_IDX)) {
    s4_uw <- combined_summary %>%
      filter(scenario_id == "S4", param == pname, estimator == "E_UW")
    s4_ws <- combined_summary %>%
      filter(scenario_id == "S4", param == pname, estimator == "E_WS")

    if (nrow(s4_uw) == 1 && nrow(s4_ws) == 1) {
      cat(sprintf("  S4/%s: E-UW=%.3f, E-WS=%.3f, diff=%+.3f ",
                  pname, s4_uw$coverage, s4_ws$coverage,
                  s4_ws$coverage - s4_uw$coverage))
      if (s4_ws$coverage >= s4_uw$coverage - 2 * MCSE_NOM) {
        cat("[OK]\n")
      } else {
        cat("[NOTE: E-WS unexpectedly lower]\n")
      }
    }
  }
}

## --- 11h. Pilot R=5 comparison ---
cat("\n  --- 11h. Pilot R=5 comparison ---\n")
pilot_found <- FALSE
for (sid in SCENARIOS) {
  for (try_path in c(
    file.path(RESULTS_DIR, sprintf("sim_results_%s_R5.rds", sid)),
    file.path(SIM_CONFIG$paths$sim_results, sprintf("sim_results_%s_R5.rds", sid))
  )) {
    if (!file.exists(try_path)) next
    pp <- readRDS(try_path)
    if (is.null(pp$summary) || nrow(pp$summary) == 0) next

    pilot_found <- TRUE
    cat(sprintf("\n  %s pilot R=5 vs full R=%d:\n", sid, R_TOTAL))
    pilot_s <- pp$summary
    for (pname in c("alpha_poverty", "beta_poverty")) {
      for (est in c("E_UW", "E_WS")) {
        p_cov <- pilot_s$coverage[pilot_s$param == pname &
                                    pilot_s$estimator == est]
        f_cov <- combined_summary$coverage[combined_summary$param == pname &
                                             combined_summary$estimator == est &
                                             combined_summary$scenario_id == sid]
        if (length(p_cov) == 1 && length(f_cov) == 1) {
          cat(sprintf("    %-16s %-6s: pilot=%.2f, full=%.3f\n",
                      pname, est, p_cov, f_cov))
        }
      }
    }
    break
  }
}
if (!pilot_found) cat("  No pilot R=5 results found.\n")

## --- Coverage summary table ---
cat("\n  --- Coverage Summary (ranges across scenarios) ---\n")
cat(sprintf("  %-16s %-12s %-12s %-12s\n",
            "Parameter", "E-UW", "E-WT", "E-WS"))
cat(sprintf("  %s\n", paste(rep("-", 54), collapse = "")))

if (nrow(combined_summary) > 0) {
  for (p_name in PARAM_ORDER) {
    vals <- character(3)
    for (j in seq_along(c("E_UW", "E_WT", "E_WS"))) {
      est <- c("E_UW", "E_WT", "E_WS")[j]
      sub <- combined_summary[combined_summary$param == p_name &
                                combined_summary$estimator == est, ]
      if (nrow(sub) > 0) {
        vals[j] <- sprintf("%.0f--%.0f%%", 100 * min(sub$coverage),
                           100 * max(sub$coverage))
      } else {
        vals[j] <- "---"
      }
    }
    cat(sprintf("  %-16s %-12s %-12s %-12s\n", p_name, vals[1], vals[2], vals[3]))
  }
}

cat(sprintf("\n  Validation: %s\n",
            ifelse(validation_ok, "ALL CHECKS PASSED", "COMPLETED WITH ALERTS")))

## --- File inventory ---
cat("\n  --- Output File Inventory ---\n")

## Simulation-specific outputs
sim_patterns <- "^(T7|F7|ST7|ST8|SF7|SF8|SF9)_"
sim_files <- list.files(FIGURE_DIR, pattern = sim_patterns, full.names = TRUE)

if (length(sim_files) > 0) {
  cat("  Simulation outputs in output/tables/:\n")
  for (f in sort(sim_files)) {
    size_kb <- file.info(f)$size / 1024
    cat(sprintf("    %-55s %7.1f KB\n", basename(f), size_kb))
  }
  cat(sprintf("  Total: %d files\n", length(sim_files)))
}

## Results data files
rds_files <- list.files(RESULTS_DIR, pattern = "\\.rds$", full.names = TRUE)
if (length(rds_files) > 0) {
  cat("\n  Results data in simulation/results/:\n")
  for (f in sort(rds_files)) {
    size_kb <- file.info(f)$size / 1024
    cat(sprintf("    %-55s %7.1f KB\n", basename(f), size_kb))
  }
}


###############################################################################
## FINAL SUMMARY
###############################################################################

SCRIPT_ELAPSED <- (proc.time() - SCRIPT_START)["elapsed"]

cat("\n")
cat("##################################################################\n")
cat("##  SIMULATION TABLES & FIGURES COMPLETE                        ##\n")
cat("##################################################################\n\n")

cat(sprintf("  Total elapsed time: %s\n\n", fmt_elapsed(SCRIPT_ELAPSED)))

cat("  Per-section timing:\n")
cat(sprintf("    Section 1 (Sample regen):    %s\n", fmt_elapsed(t1_elapsed)))
cat(sprintf("    Section 2 (Post-processing): %s\n", fmt_elapsed(t2_elapsed)))
cat(sprintf("    Section 3 (Aggregation):     %s\n", fmt_elapsed(t3_elapsed)))

cat("\n  Outputs produced:\n")
cat("    MAIN TEXT:\n")
cat("      T7_simulation_results.csv/.tex    Coverage/Bias/RMSE table\n")
cat("      F7_simulation_coverage.pdf/.png   Cleveland coverage dot plot\n")
cat("    SUPPLEMENTARY:\n")
cat("      ST7_simulation_full.csv/.tex      Full metrics (all 45 rows)\n")
cat("      ST8_DER_summary.csv/.tex          DER across replications\n")
cat("      SF7_simulation_bias.pdf/.png      Relative bias dot plot\n")
cat("      SF8_width_ratio_distribution.pdf/.png  Width ratio boxplot\n")
cat("      SF9_DER_summary.pdf/.png          DER summary dot plot\n")
cat("    DATA:\n")
cat("      sim_summary_all.rds               Combined summary\n")
cat("      sim_raw_all.rds                   Raw per-rep metrics\n")
cat("      sim_summary_{S0,S3,S4}.rds        Per-scenario summaries\n")
cat("      sim_sandwich_{S0,S3,S4}.rds       Per-scenario sandwich\n")
cat("      sim_DER_{S0,S3,S4}.rds            Per-scenario DER matrices\n")

cat("\n  Scenario success rates:\n")
for (sid in SCENARIOS) {
  ml <- all_scenario_results[[sid]]$metrics_list
  n_valid <- sum(!sapply(ml, is.null))
  cat(sprintf("    %s: %d/%d (%.1f%%)\n",
              sid, n_valid, R_TOTAL, 100 * n_valid / R_TOTAL))
}

cat("\n##################################################################\n")
