#!/usr/bin/env Rscript
## =============================================================================
## run_production_all.R -- Production Simulation Run (All Scenarios)
## =============================================================================
## Purpose : Run all three simulation scenarios (S0, S3, S4) sequentially,
##           each with R=200 replications on 10 cores.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Usage:
##   nohup Rscript code/simulation/run_production_all.R \
##     > data/precomputed/simulation/log_production.txt 2>&1 &
##
## Uses run_resume() to skip already-completed replications.
## Estimated total: ~86 hours on 10 cores.
##
## Checkpoint system:
##   - JSON status file: checkpoint_status.json (real-time progress)
##   - Integrity scans at startup and after each scenario
##   - Automatic bundles (tar.gz) every 25 reps
##   - Disk space monitoring
##   - Final bundle on completion
##
## To monitor progress:
##   tail -f data/precomputed/simulation/log_production.txt
##   cat data/precomputed/simulation/checkpoint_status.json | python3 -m json.tool
##   ls data/precomputed/simulation/fits/S0/E_UW/ | wc -l   # count completed
## =============================================================================

cat("==============================================================\n")
cat("  PRODUCTION SIMULATION RUN\n")
cat("==============================================================\n")
cat(sprintf("Start: %s\n", Sys.time()))
cat(sprintf("PID: %d\n", Sys.getpid()))
cat(sprintf("Machine: %s\n", Sys.info()["nodename"]))
cat(sprintf("Cores available: %d\n", parallel::detectCores()))
cat(sprintf("Cores to use: 10\n"))
cat("Plan: S0 -> S3 -> S4, each R=200, 10 cores\n")
cat("==============================================================\n\n")

## Guard flag
.SIM_05_CALLED_FROM_PARENT <- TRUE

## Source parallel runner (includes sim_06_checkpoint.R)
.pr <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
                   
source(file.path(.pr, "code/simulation/sim_05_run_parallel.R"))

t_grand <- proc.time()

## ---- CHECKPOINT PREFLIGHT ----
cat("\n*** CHECKPOINT PREFLIGHT ***\n")
reps_to_run <- checkpoint_preflight(
  config       = SIM_CONFIG,
  scenario_ids = c("S0", "S3", "S4"),
  rep_range    = 1:200,
  notify_email = NULL,   # Set email here for notifications
  notify_slack = NULL    # Set Slack webhook URL here
)
cat("\n")

## ---- S0 ----
cat("\n\n*** STARTING S0 ***\n")
cat(sprintf("Time: %s\n", Sys.time()))
results_S0 <- run_resume(scenario_id = "S0", rep_range = 1:200, cores = 10L)
cat(sprintf("S0 done: %s\n\n", Sys.time()))

## ---- S3 ----
cat("\n\n*** STARTING S3 ***\n")
cat(sprintf("Time: %s\n", Sys.time()))
results_S3 <- run_resume(scenario_id = "S3", rep_range = 1:200, cores = 10L)
cat(sprintf("S3 done: %s\n\n", Sys.time()))

## ---- S4 ----
cat("\n\n*** STARTING S4 ***\n")
cat(sprintf("Time: %s\n", Sys.time()))
results_S4 <- run_resume(scenario_id = "S4", rep_range = 1:200, cores = 10L)
cat(sprintf("S4 done: %s\n\n", Sys.time()))

## ---- CHECKPOINT FINALIZE ----
cat("\n*** CHECKPOINT FINALIZE ***\n")
checkpoint_finalize(SIM_CONFIG)

## ---- Grand Summary ----
t_total <- (proc.time() - t_grand)[3]
cat("\n==============================================================\n")
cat("  PHASE 4 PRODUCTION RUN COMPLETE\n")
cat("==============================================================\n")
cat(sprintf("Total wall-clock: %.1f hours\n", t_total / 3600))
cat(sprintf("End: %s\n", Sys.time()))

## Print final checkpoint status
cat("\n")
cat(checkpoint_get_summary(SIM_CONFIG))
cat("\n")

cat("==============================================================\n")
