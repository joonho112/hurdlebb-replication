## =============================================================================
## sim_05_run_parallel.R -- Parallel Simulation Execution
## =============================================================================
## Purpose : Parallel execution of the simulation pipeline across replications.
##           Uses parallel::mclapply (fork-based) to distribute replications
##           across cores while sharing population, calibrations, and compiled
##           Stan models across all workers.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Architecture:
##   1. Load config, source modules, compile models (ONCE in parent)
##   2. Generate or load population, calibrate all scenarios (ONCE)
##   3. For each scenario:
##      a. mclapply over rep_ids: sample -> fit -> postprocess
##      b. Aggregate per-rep metrics into summary table
##      c. Save all outputs
##
## Resource Management:
##   When running R_parallel replications simultaneously, each replication's
##   Stan fits run chains SEQUENTIALLY (parallel_chains=1) to avoid
##   oversubscription. Total CPU usage = n_cores * 1 chain = n_cores.
##
## Usage:
##   Rscript code/simulation/sim_05_run_parallel.R \
##     --scenario S0 --reps 5 --cores 10 [--pilot]
##
##   Or source and call run_parallel_simulation() interactively.
##
## Dependencies:
##   - sim_00_config.R     : SIM_CONFIG
##   - sim_01_dgp.R        : generate_finite_population()
##   - sim_02_sampling.R   : calibrate_inclusion(), draw_sample(), etc.
##   - sim_03_fit.R        : compile_stan_models(), run_fitting_pipeline(), etc.
##   - sim_04_postprocess.R: postprocess_replication(), aggregate_metrics()
##   - parallel            : mclapply (Unix fork)
## =============================================================================

cat("==============================================================\n")
cat("  Parallel Simulation Runner (sim_05)\n")
cat("==============================================================\n\n")


###############################################################################
## SECTION 0 : ENVIRONMENT AND DEPENDENCIES
###############################################################################

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
                           
SIM_DIR      <- file.path(PROJECT_ROOT, "code/simulation")

## Suppress standalone execution in submodules
.SIM_01_CALLED_FROM_PARENT <- TRUE
.SIM_02_CALLED_FROM_PARENT <- TRUE
.SIM_03_CALLED_FROM_PARENT <- TRUE
.SIM_04_CALLED_FROM_PARENT <- TRUE

## Source all modules
cat("  Loading simulation modules...\n")
source(file.path(SIM_DIR, "sim_00_config.R"))
source(file.path(SIM_DIR, "sim_01_dgp.R"))
source(file.path(SIM_DIR, "sim_02_sampling.R"))
source(file.path(SIM_DIR, "sim_03_fit.R"))
source(file.path(SIM_DIR, "sim_04_postprocess.R"))
source(file.path(SIM_DIR, "sim_06_checkpoint.R"))

## Load required packages
suppressPackageStartupMessages({
  library(parallel)
})

cat("  All modules loaded.\n\n")


###############################################################################
## SECTION 1 : SINGLE-REPLICATION WORKER FUNCTION
###############################################################################
#' Execute One Complete Replication (Sample -> Fit -> Postprocess)
#'
#' Designed to be called from mclapply(). All heavy objects (population,
#' calibrations, compiled models, config) are available via fork inheritance.
#'
#' @param rep_id        Integer. Replication number.
#' @param scenario_id   Character. One of "S0", "S3", "S4".
#' @param pop           Data frame. Population from generate_finite_population().
#' @param calibrations  Named list. Pre-computed calibrations from calibrate_all_scenarios().
#' @param models        List. Compiled Stan models from compile_stan_models().
#' @param config        SIM_CONFIG list (with possibly modified mcmc$parallel_chains).
#' @param save_intermediates Logical. Save sample and fit files to disk.
#'
#' @return Named list with:
#'   \item{rep_id}{replication number}
#'   \item{scenario_id}{scenario}
#'   \item{metrics}{15-row data.frame from postprocess_replication()}
#'   \item{sandwich_summary}{DER, V_sand diagonal, eigenvalues}
#'   \item{diagnostics}{convergence info for both fits}
#'   \item{timing}{list(sample, fit, postprocess, total)}
#'   \item{status}{"OK" or error message}

run_single_replication <- function(rep_id, scenario_id, pop, calibrations,
                                    models, config, save_intermediates = TRUE) {

  t_total_start <- proc.time()

  ## Result skeleton (for error handling)
  result <- list(
    rep_id       = rep_id,
    scenario_id  = scenario_id,
    status       = "ERROR",
    metrics      = NULL,
    sandwich_summary = NULL,
    diagnostics  = NULL,
    timing       = list(sample = NA, fit = NA, postprocess = NA, total = NA)
  )

  tryCatch({

    ## -----------------------------------------------------------------
    ## Step 1: Draw sample
    ## -----------------------------------------------------------------
    t1 <- proc.time()
    scenario <- config$scenarios[[scenario_id]]
    calib    <- calibrations[[scenario_id]]
    seed     <- get_rep_seed(config$seeds$base_seed, rep_id, scenario_id)

    sample_result <- draw_sample(pop, scenario, config, calib, seed,
                                  verbose = FALSE)

    ## Save sample to disk
    if (save_intermediates) {
      sample_path <- file.path(config$paths$sim_samples, scenario_id,
                                sprintf("rep_%03d.rds", rep_id))
      saveRDS(sample_result, sample_path)
    }

    result$timing$sample <- (proc.time() - t1)[3]

    ## -----------------------------------------------------------------
    ## Step 2: Fit models (E-UW and E-WT/WS)
    ## -----------------------------------------------------------------
    t2 <- proc.time()

    fit_results <- fit_all_estimators(
      stan_data_uw = sample_result$stan_data_uw,
      stan_data_wt = sample_result$stan_data_wt,
      models       = models,
      config       = config,
      seed         = seed
    )

    ## Attach metadata
    fit_results$scenario_id <- scenario_id
    fit_results$rep_id      <- rep_id
    fit_results$N           <- sample_result$stan_data_uw$N

    ## Save fits to disk
    if (save_intermediates) {
      save_fit_results(fit_results, config, scenario_id, rep_id)
    }

    result$timing$fit <- (proc.time() - t2)[3]

    ## Check convergence
    if (!fit_results$both_ok) {
      result$status <- "FIT_FAILED"
      result$diagnostics <- list(
        uw = fit_results$E_UW$diagnostics,
        wt = fit_results$E_WT_WS$diagnostics
      )
      cat(sprintf("  [WARN] %s rep %03d: fit failed (both_ok=FALSE)\n",
                  scenario_id, rep_id))
      return(result)
    }

    ## -----------------------------------------------------------------
    ## Step 3: Post-process (sandwich + metrics)
    ## -----------------------------------------------------------------
    t3 <- proc.time()

    pp_result <- postprocess_replication(fit_results, sample_result, config)

    result$timing$postprocess <- (proc.time() - t3)[3]

    ## -----------------------------------------------------------------
    ## Step 4: Package results
    ## -----------------------------------------------------------------
    result$metrics <- pp_result$metrics
    result$sandwich_summary <- list(
      DER         = pp_result$sandwich$DER,
      V_sand_diag = diag(pp_result$sandwich$V_sand),
      pd_fix      = pp_result$sandwich$pd_fix_applied,
      eigenvalues = pp_result$sandwich$eigenvalues
    )
    result$diagnostics <- list(
      uw = fit_results$E_UW$diagnostics,
      wt = fit_results$E_WT_WS$diagnostics
    )
    result$status <- "OK"

    ## Total timing
    result$timing$total <- (proc.time() - t_total_start)[3]

    cat(sprintf("  [OK] %s rep %03d: %.1f sec (sample=%.1f, fit=%.1f, post=%.1f)\n",
                scenario_id, rep_id,
                result$timing$total, result$timing$sample,
                result$timing$fit, result$timing$postprocess))

    ## Free memory
    rm(fit_results, pp_result, sample_result)
    invisible(gc(verbose = FALSE))

  }, error = function(e) {
    result$status <<- paste("ERROR:", conditionMessage(e))
    cat(sprintf("  [ERROR] %s rep %03d: %s\n",
                scenario_id, rep_id, conditionMessage(e)))
  })

  result
}


###############################################################################
## SECTION 2 : PARALLEL SIMULATION RUNNER
###############################################################################
#' Run Parallel Simulation for One Scenario
#'
#' @param scenario_id   Character. "S0", "S3", or "S4".
#' @param rep_ids       Integer vector. Which replications to run (e.g., 1:5).
#' @param n_cores       Integer. Number of parallel workers.
#' @param pop           Population data frame.
#' @param calibrations  Pre-computed calibrations.
#' @param models        Compiled Stan models.
#' @param config        SIM_CONFIG list.
#' @param save_intermediates  Logical. Save sample/fit files.
#'
#' @return list with:
#'   \item{scenario_id}{scenario}
#'   \item{rep_results}{list of per-rep results}
#'   \item{metrics_list}{list of per-rep metrics data.frames}
#'   \item{summary}{aggregated summary from aggregate_metrics()}
#'   \item{timing_total}{total wall-clock seconds}
#'   \item{n_ok}{number of successful replications}
#'   \item{n_failed}{number of failed replications}
#'   \item{failed_reps}{rep_ids that failed}

run_parallel_scenario <- function(scenario_id, rep_ids, n_cores,
                                   pop, calibrations, models, config,
                                   save_intermediates = TRUE) {

  cat(sprintf("\n\n########################################################\n"))
  cat(sprintf("  SCENARIO %s : %d replications on %d cores\n",
              scenario_id, length(rep_ids), n_cores))
  cat(sprintf("########################################################\n\n"))

  t_start <- proc.time()

  ## Modify config for sequential chains within each worker
  config_parallel <- config
  config_parallel$mcmc$parallel_chains <- 1L
  cat(sprintf("  Stan parallel_chains set to 1 (sequential within worker)\n"))
  cat(sprintf("  Each rep: 4 chains x sequential = ~4x slower per rep\n"))
  cat(sprintf("  But %d reps in parallel -> net speedup ~ %.1fx vs serial\n\n",
              n_cores, n_cores / 4))

  ## Run replications in parallel
  rep_results <- mclapply(
    X        = rep_ids,
    FUN      = run_single_replication,
    scenario_id = scenario_id,
    pop         = pop,
    calibrations = calibrations,
    models      = models,
    config      = config_parallel,
    save_intermediates = save_intermediates,
    mc.cores = n_cores,
    mc.preschedule = FALSE    # dynamic scheduling for load balancing
  )

  ## Name by rep_id
  names(rep_results) <- sprintf("rep_%03d", rep_ids)

  ## --- CHECKPOINT: Update status for each completed rep ---
  for (i in seq_along(rep_ids)) {
    r <- rep_results[[i]]
    rep_time <- if (!is.null(r$timing$total) && !is.na(r$timing$total)) {
      r$timing$total
    } else 0
    tryCatch(
      checkpoint_update_status(
        scenario_id  = scenario_id,
        rep_id       = rep_ids[i],
        rep_status   = r$status,
        rep_time_sec = rep_time,
        config       = config
      ),
      error = function(e) {
        cat(sprintf("  [CHECKPOINT WARN] Status update failed for rep %d: %s\n",
                    rep_ids[i], e$message))
      }
    )
  }

  ## --- CHECKPOINT: Periodic disk check ---
  tryCatch(checkpoint_check_disk(halt_on_low = TRUE),
           error = function(e) cat(sprintf("  [CHECKPOINT WARN] Disk check error: %s\n", e$message)))

  ## Separate successful and failed
  statuses  <- sapply(rep_results, function(r) r$status)
  ok_mask   <- statuses == "OK"
  n_ok      <- sum(ok_mask)
  n_failed  <- sum(!ok_mask)
  failed_reps <- rep_ids[!ok_mask]

  cat(sprintf("\n  === Scenario %s Summary ===\n", scenario_id))
  cat(sprintf("  Successful: %d / %d\n", n_ok, length(rep_ids)))

  if (n_failed > 0) {
    cat(sprintf("  FAILED reps: %s\n",
                paste(failed_reps, collapse = ", ")))
    for (fr in failed_reps) {
      r <- rep_results[[sprintf("rep_%03d", fr)]]
      cat(sprintf("    Rep %03d: %s\n", fr, r$status))
    }
  }

  ## Aggregate metrics (only successful reps)
  metrics_list <- lapply(rep_results[ok_mask], function(r) r$metrics)
  summary_df   <- NULL

  if (n_ok >= 2) {
    summary_df <- aggregate_metrics(metrics_list, config)
    cat(sprintf("\n  Aggregated metrics from %d replications:\n", n_ok))
    print_metrics_summary(summary_df, scenario_id, config)
  } else if (n_ok == 1) {
    cat("  Only 1 successful rep — skipping aggregation.\n")
  } else {
    cat("  No successful replications — nothing to aggregate.\n")
  }

  ## Timing
  timing_total <- (proc.time() - t_start)[3]
  cat(sprintf("\n  Total wall-clock time: %.1f sec (%.1f min)\n",
              timing_total, timing_total / 60))

  ## Per-rep timing summary
  timings <- sapply(rep_results[ok_mask], function(r) r$timing$total)
  if (length(timings) > 0) {
    cat(sprintf("  Per-rep timing: mean=%.1f, min=%.1f, max=%.1f sec\n",
                mean(timings), min(timings), max(timings)))
  }

  ## Return
  list(
    scenario_id  = scenario_id,
    rep_ids      = rep_ids,
    rep_results  = rep_results,
    metrics_list = metrics_list,
    summary      = summary_df,
    timing_total = timing_total,
    n_ok         = n_ok,
    n_failed     = n_failed,
    failed_reps  = failed_reps
  )
}


###############################################################################
## SECTION 3 : FULL SIMULATION DRIVER
###############################################################################
#' Run the Complete Parallel Simulation
#'
#' Top-level function: generates population, calibrates, compiles models,
#' then runs all scenarios in sequence (each scenario parallelized internally).
#'
#' @param scenario_ids   Character vector. Scenarios to run.
#' @param rep_range      Integer vector. Rep IDs to run (e.g., 1:200).
#' @param n_cores        Integer. Parallel workers per scenario.
#' @param pilot          Logical. If TRUE, use abbreviated labels in output.
#' @param save_intermediates  Logical. Save per-rep files.
#'
#' @return Named list of scenario results.

run_full_simulation <- function(scenario_ids = c("S0", "S3", "S4"),
                                 rep_range    = 1:200,
                                 n_cores      = 4L,
                                 pilot        = FALSE,
                                 save_intermediates = TRUE) {

  t_grand_start <- proc.time()

  mode_label <- if (pilot) "PILOT" else "FULL"
  cat(sprintf("\n\n============================================================\n"))
  cat(sprintf("  %s SIMULATION: %d scenarios x %d reps = %d total\n",
              mode_label, length(scenario_ids), length(rep_range),
              length(scenario_ids) * length(rep_range)))
  cat(sprintf("  Cores: %d\n", n_cores))
  cat(sprintf("============================================================\n\n"))

  ## -----------------------------------------------------------------
  ## Step 0: Checkpoint initialization
  ## -----------------------------------------------------------------
  tryCatch({
    checkpoint_init(SIM_CONFIG)
    cat("  Checkpoint system initialized.\n\n")
  }, error = function(e) {
    cat(sprintf("  [WARN] Checkpoint init failed (non-fatal): %s\n", e$message))
  })

  ## -----------------------------------------------------------------
  ## Step 1: Generate or load population
  ## -----------------------------------------------------------------
  pop_path <- file.path(SIM_CONFIG$paths$sim_population, "pop_base.rds")

  if (file.exists(pop_path)) {
    cat("  Loading existing population...\n")
    pop_obj <- readRDS(pop_path)
    pop <- pop_obj$data
    cat(sprintf("  Population loaded: M=%d providers\n", nrow(pop)))
  } else {
    cat("  Generating population (M=50,000)...\n")
    pop_obj <- generate_finite_population(SIM_CONFIG,
                                           seed = SIM_CONFIG$seeds$population_seed)
    pop <- pop_obj$data
    saveRDS(pop_obj, pop_path)
    cat(sprintf("  Population saved: %s\n", pop_path))
  }

  ## -----------------------------------------------------------------
  ## Step 2: Calibrate inclusion probabilities for all scenarios
  ## -----------------------------------------------------------------
  cat("\n  Calibrating inclusion probabilities...\n")
  calibrations <- calibrate_all_scenarios(pop, SIM_CONFIG, verbose = TRUE)

  ## -----------------------------------------------------------------
  ## Step 3: Compile Stan models (once)
  ## -----------------------------------------------------------------
  cat("  Compiling Stan models...\n")
  models <- compile_stan_models(SIM_CONFIG)
  cat("  Models compiled.\n\n")

  ## -----------------------------------------------------------------
  ## Step 4: Run each scenario
  ## -----------------------------------------------------------------
  all_results <- list()

  for (sid in scenario_ids) {
    result <- run_parallel_scenario(
      scenario_id  = sid,
      rep_ids      = rep_range,
      n_cores      = n_cores,
      pop          = pop,
      calibrations = calibrations,
      models       = models,
      config       = SIM_CONFIG,
      save_intermediates = save_intermediates
    )

    all_results[[sid]] <- result

    ## Save per-scenario results incrementally
    out_path <- file.path(SIM_CONFIG$paths$sim_results,
                          sprintf("sim_results_%s_R%d.rds",
                                  sid, length(rep_range)))
    saveRDS(result, out_path)
    cat(sprintf("  Results saved: %s\n", out_path))
  }

  ## -----------------------------------------------------------------
  ## Step 5: Grand summary
  ## -----------------------------------------------------------------
  t_grand_total <- (proc.time() - t_grand_start)[3]

  cat(sprintf("\n\n============================================================\n"))
  cat(sprintf("  %s SIMULATION COMPLETE\n", mode_label))
  cat(sprintf("============================================================\n"))
  cat(sprintf("  Total wall-clock: %.1f sec (%.1f min, %.2f hrs)\n",
              t_grand_total, t_grand_total / 60, t_grand_total / 3600))

  for (sid in scenario_ids) {
    r <- all_results[[sid]]
    cat(sprintf("  %s: %d/%d OK, %.1f min\n",
                sid, r$n_ok, length(r$rep_ids), r$timing_total / 60))
  }
  cat("\n")

  ## Save combined results
  combined_path <- file.path(SIM_CONFIG$paths$sim_results,
                              sprintf("sim_results_combined_%s_R%d.rds",
                                      paste(scenario_ids, collapse = "_"),
                                      length(rep_range)))
  saveRDS(all_results, combined_path)
  cat(sprintf("  Combined results: %s\n\n", combined_path))

  ## -----------------------------------------------------------------
  ## Step 6: Checkpoint finalization
  ## -----------------------------------------------------------------
  tryCatch({
    checkpoint_finalize(SIM_CONFIG)
  }, error = function(e) {
    cat(sprintf("  [WARN] Checkpoint finalization failed (non-fatal): %s\n", e$message))
  })

  invisible(all_results)
}


###############################################################################
## SECTION 4 : COMMAND-LINE INTERFACE
###############################################################################
## Parse command-line arguments when run via Rscript.

## Guard: Only run CLI parsing when this script is the main entry point.
## When source()'d from another script (like /tmp/sim_pilot_S0.R),
## the parent sets .SIM_05_CALLED_FROM_PARENT = TRUE to skip CLI.
if (!interactive() && !isTRUE(.SIM_05_CALLED_FROM_PARENT)) {

  args <- commandArgs(trailingOnly = TRUE)

  ## Defaults
  cli_scenario <- "S0"
  cli_reps     <- 5L
  cli_cores    <- 4L
  cli_pilot    <- TRUE

  ## Parse arguments
  i <- 1
  while (i <= length(args)) {
    switch(args[i],
      "--scenario" = {
        cli_scenario <- args[i + 1]
        i <- i + 2
      },
      "--reps" = {
        cli_reps <- as.integer(args[i + 1])
        i <- i + 2
      },
      "--cores" = {
        cli_cores <- as.integer(args[i + 1])
        i <- i + 2
      },
      "--pilot" = {
        cli_pilot <- TRUE
        i <- i + 1
      },
      "--full" = {
        cli_pilot <- FALSE
        i <- i + 1
      },
      "--all-scenarios" = {
        cli_scenario <- "all"
        i <- i + 1
      },
      {
        cat(sprintf("  Unknown argument: %s\n", args[i]))
        i <- i + 1
      }
    )
  }

  ## Determine scenario list
  if (cli_scenario == "all") {
    scenarios_to_run <- c("S0", "S3", "S4")
  } else {
    scenarios_to_run <- cli_scenario
  }

  cat(sprintf("  CLI mode: scenarios=%s, reps=1:%d, cores=%d, pilot=%s\n",
              paste(scenarios_to_run, collapse = ","),
              cli_reps, cli_cores, cli_pilot))

  ## Run
  run_full_simulation(
    scenario_ids = scenarios_to_run,
    rep_range    = 1:cli_reps,
    n_cores      = cli_cores,
    pilot        = cli_pilot,
    save_intermediates = TRUE
  )
}


###############################################################################
## SECTION 5 : CONVENIENCE FUNCTIONS FOR INTERACTIVE USE
###############################################################################

#' Quick Pilot Run (Interactive)
#'
#' Convenience wrapper for Phase 3 local pilot.
#' @param scenario_id  Character. Default "S0".
#' @param R            Integer. Number of reps. Default 5.
#' @param cores        Integer. Number of cores. Default 10.

run_pilot <- function(scenario_id = "S0", R = 5L, cores = 10L) {
  run_full_simulation(
    scenario_ids = scenario_id,
    rep_range    = 1:R,
    n_cores      = cores,
    pilot        = TRUE,
    save_intermediates = TRUE
  )
}


#' Resume Simulation from a Specific Rep
#'
#' For recovering from crashes or extending an existing run.
#' Checks which reps already have saved results and only runs missing ones.
#'
#' @param scenario_id  Character.
#' @param rep_range    Integer vector. Full range of desired reps.
#' @param cores        Integer.

run_resume <- function(scenario_id, rep_range = 1:200, cores = 10L) {

  ## Initialize checkpoint if needed
  tryCatch(checkpoint_init(SIM_CONFIG), error = function(e) NULL)

  ## Use checkpoint system to find reps to re-run (missing + corrupted)
  missing_reps <- checkpoint_get_rerun_reps(scenario_id, SIM_CONFIG, rep_range)

  ## Fallback: also check file existence directly (belt and suspenders)
  for (r in rep_range) {
    uw_path <- file.path(SIM_CONFIG$paths$sim_fits, scenario_id, "E_UW",
                          sprintf("rep_%03d.rds", r))
    wt_path <- file.path(SIM_CONFIG$paths$sim_fits, scenario_id, "E_WT",
                          sprintf("rep_%03d.rds", r))
    if (!file.exists(uw_path) || !file.exists(wt_path)) {
      missing_reps <- unique(c(missing_reps, r))
    }
  }
  missing_reps <- sort(missing_reps)

  existing_count <- length(rep_range) - length(missing_reps)

  if (length(missing_reps) == 0) {
    cat(sprintf("  All %d reps for %s already exist. Nothing to run.\n",
                length(rep_range), scenario_id))
    return(invisible(NULL))
  }

  cat(sprintf("  Resume %s: %d existing, %d to run. Running missing/corrupted reps.\n",
              scenario_id, existing_count, length(missing_reps)))

  ## Delete corrupted files before re-running
  tryCatch({
    status <- .read_status()
    if (!is.null(status)) {
      sc_idx <- which(sapply(status$scenarios, function(s) s$id) == scenario_id)
      if (length(sc_idx) > 0) {
        cr <- status$scenarios[[sc_idx]]$corrupted_reps
        corrupted <- if (!is.null(cr) && length(cr) > 0) as.integer(unlist(cr)) else integer(0)
        if (length(corrupted) > 0) {
          cat(sprintf("  Cleaning %d corrupted reps before re-run...\n",
                      length(corrupted)))
          checkpoint_delete_corrupted(scenario_id, corrupted, SIM_CONFIG)
        }
      }
    }
  }, error = function(e) NULL)

  run_full_simulation(
    scenario_ids = scenario_id,
    rep_range    = missing_reps,
    n_cores      = cores,
    pilot        = FALSE,
    save_intermediates = TRUE
  )
}


#' Post-Process All Saved Fits for a Scenario
#'
#' Useful when fits were saved but post-processing needs to be re-run
#' (e.g., after bug fix in sim_04_postprocess.R).
#'
#' @param scenario_id  Character.
#' @param rep_range    Integer vector.
#' @param cores        Integer.

reprocess_scenario <- function(scenario_id, rep_range = 1:200, cores = 10L) {

  cat(sprintf("\n  Reprocessing %s: reps %d-%d\n",
              scenario_id, min(rep_range), max(rep_range)))

  results <- mclapply(
    X = rep_range,
    FUN = function(rep_id) {
      tryCatch({
        pp <- postprocess_saved_fits(scenario_id, rep_id, SIM_CONFIG)
        if (is.null(pp)) return(NULL)
        list(rep_id = rep_id, metrics = pp$metrics, status = "OK")
      }, error = function(e) {
        list(rep_id = rep_id, metrics = NULL, status = paste("ERROR:", e$message))
      })
    },
    mc.cores = cores
  )

  ## Extract metrics
  ok_results <- Filter(function(r) !is.null(r) && r$status == "OK", results)
  metrics_list <- lapply(ok_results, function(r) r$metrics)

  cat(sprintf("  Reprocessed: %d OK / %d total\n",
              length(ok_results), length(rep_range)))

  if (length(metrics_list) >= 2) {
    summary_df <- aggregate_metrics(metrics_list, SIM_CONFIG)
    print_metrics_summary(summary_df, scenario_id, SIM_CONFIG)
    return(invisible(list(metrics_list = metrics_list, summary = summary_df)))
  }

  invisible(list(metrics_list = metrics_list, summary = NULL))
}

cat("\n  sim_05_run_parallel.R loaded.\n")
cat("  Functions: run_pilot(), run_full_simulation(), run_resume(), reprocess_scenario()\n")
cat("  Example: run_pilot('S0', R=5, cores=10)\n\n")
