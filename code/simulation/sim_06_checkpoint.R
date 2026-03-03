## =============================================================================
## sim_06_checkpoint.R -- Checkpoint System for Simulation Runs
## =============================================================================
## Purpose : Comprehensive checkpoint system for server deployment.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Features:
##   1. STATUS LOG    -- Structured JSON status file updated after every rep
##   2. INTEGRITY     -- Validate saved .rds files (readability + structure)
##   3. BUNDLE        -- Periodic tar.gz snapshots for download/backup
##   4. DISK MONITOR  -- Warn and halt if disk space runs low
##   5. NOTIFICATION  -- Email/Slack alerts on milestones or errors
##   6. AUTO-REPAIR   -- Detect corrupted files and queue for re-run
##
## Architecture:
##   sim_06_checkpoint.R is a pure utility module -- no main() execution.
##   Functions are called by sim_05_run_parallel.R hooks.
##
## Integration Points (in sim_05):
##   - After each rep:       checkpoint_update_status()
##   - Every N reps:         checkpoint_create_bundle()
##   - Before/after run:     checkpoint_scan_integrity()
##   - At startup:           checkpoint_check_disk()
##   - On error:             checkpoint_notify("error", ...)
##   - At milestones:        checkpoint_notify("milestone", ...)
##
## Dependencies:
##   - sim_00_config.R : SIM_CONFIG (for paths)
##   - jsonlite        : JSON read/write (install if needed)
## =============================================================================

cat("  Loading checkpoint module (sim_06_checkpoint.R)...\n")

## ---- Soft dependency on jsonlite ----
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  cat("  [WARN] jsonlite not installed. Status logs will use basic format.\n")
  cat("  Install with: install.packages('jsonlite')\n")
  .CHECKPOINT_HAS_JSONLITE <- FALSE
} else {
  suppressPackageStartupMessages(library(jsonlite))
  .CHECKPOINT_HAS_JSONLITE <- TRUE
}


###############################################################################
## SECTION 0 : CHECKPOINT CONFIGURATION
###############################################################################

CHECKPOINT_CONFIG <- list(
  ## Status log file (JSON): single source of truth for progress

  status_file = NULL,  # set in checkpoint_init()

  ## Bundle settings
  bundle_dir       = NULL,       # set in checkpoint_init()
  bundle_every_n   = 25L,        # create bundle every N completed reps (per scenario)
  bundle_scenarios = TRUE,       # include per-scenario results in bundles
  bundle_compress  = "gzip",     # compression for tar

  ## Disk monitoring
  disk_warn_gb  = 5.0,   # warn when free space falls below this
  disk_halt_gb  = 2.0,   # halt execution when free space falls below this

  ## Integrity checks
  integrity_on_startup = TRUE,   # scan all files at startup
  integrity_on_bundle  = TRUE,   # verify before bundling
  max_repair_attempts  = 2L,     # max times to re-queue a corrupted rep

  ## Notification settings
  notify_email   = NULL,          # email address (NULL = disabled)
  notify_slack   = NULL,          # Slack webhook URL (NULL = disabled)
  notify_on      = c("milestone", "error", "completion"),
  milestone_reps = c(25, 50, 100, 150, 200)  # notify at these rep counts
)


###############################################################################
## SECTION 1 : INITIALIZATION
###############################################################################

#' Initialize Checkpoint System
#'
#' Creates checkpoint directory, status file, and validates environment.
#'
#' @param config  SIM_CONFIG list.
#' @param notify_email  Optional email for notifications.
#' @param notify_slack  Optional Slack webhook URL.
#' @return Invisible TRUE on success.

checkpoint_init <- function(config, notify_email = NULL, notify_slack = NULL) {

  ## Set paths
  CHECKPOINT_CONFIG$status_file <<- file.path(
    config$paths$sim_output_root, "checkpoint_status.json"
  )
  CHECKPOINT_CONFIG$bundle_dir <<- file.path(
    config$paths$sim_output_root, "bundles"
  )
  CHECKPOINT_CONFIG$notify_email <<- notify_email
  CHECKPOINT_CONFIG$notify_slack <<- notify_slack

  ## Create bundle directory
  if (!dir.exists(CHECKPOINT_CONFIG$bundle_dir)) {
    dir.create(CHECKPOINT_CONFIG$bundle_dir, recursive = TRUE)
    cat(sprintf("  [CHECKPOINT] Created bundle dir: %s\n",
                CHECKPOINT_CONFIG$bundle_dir))
  }

  ## Initialize status file if it doesn't exist
  if (!file.exists(CHECKPOINT_CONFIG$status_file)) {
    status <- .create_initial_status(config)
    .write_status(status)
    cat(sprintf("  [CHECKPOINT] Created status file: %s\n",
                CHECKPOINT_CONFIG$status_file))
  } else {
    cat(sprintf("  [CHECKPOINT] Existing status file found: %s\n",
                CHECKPOINT_CONFIG$status_file))
  }

  ## Initial disk check
  checkpoint_check_disk(halt_on_low = FALSE)

  cat("  [CHECKPOINT] Initialization complete.\n")
  invisible(TRUE)
}


#' Create Initial Status Structure
#' @keywords internal
.create_initial_status <- function(config) {
  list(
    project      = "HBB Simulation Study",
    version      = config$version,
    created      = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    last_updated = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    machine      = as.character(Sys.info()["nodename"]),
    pid          = Sys.getpid(),

    ## Per-scenario tracking
    scenarios = lapply(config$scenario_ids, function(sid) {
      list(
        id             = sid,
        status         = "pending",       # pending | running | completed | error
        total_reps     = config$evaluation$R,
        completed_reps = 0L,
        failed_reps    = integer(0),
        corrupted_reps = integer(0),
        last_rep       = NA_integer_,
        start_time     = NA_character_,
        last_rep_time  = NA_character_,
        elapsed_sec    = 0,
        avg_rep_sec    = NA_real_,
        eta_hours      = NA_real_,
        bundles_created = 0L
      )
    }),

    ## Global counters
    global = list(
      total_reps_all   = config$evaluation$R * length(config$scenario_ids),
      completed_all    = 0L,
      failed_all       = 0L,
      corrupted_all    = 0L,
      disk_free_gb     = NA_real_,
      last_bundle_time = NA_character_,
      errors           = list()
    )
  )
}


###############################################################################
## SECTION 2 : STATUS LOG (JSON)
###############################################################################

#' Update Status After a Completed Replication
#'
#' Called after every single replication. Updates the JSON status file with
#' current progress, timing, and ETA.
#'
#' @param scenario_id  Character. "S0", "S3", or "S4".
#' @param rep_id       Integer. Replication number.
#' @param rep_status   Character. "OK", "FIT_FAILED", or "ERROR: <msg>".
#' @param rep_time_sec Numeric. Wall-clock seconds for this rep.
#' @param config       SIM_CONFIG list (for path info).
#'
#' @return Invisible TRUE.

checkpoint_update_status <- function(scenario_id, rep_id, rep_status,
                                     rep_time_sec, config) {

  status <- .read_status()
  if (is.null(status)) return(invisible(FALSE))

  ## Find scenario index
  sc_idx <- which(sapply(status$scenarios, function(s) s$id) == scenario_id)
  if (length(sc_idx) == 0) {
    cat(sprintf("  [CHECKPOINT WARN] Unknown scenario: %s\n", scenario_id))
    return(invisible(FALSE))
  }

  sc <- status$scenarios[[sc_idx]]

  ## Update scenario info (NULL-safe: JSON roundtrip turns NA -> null -> NULL)
  st_missing <- is.null(sc$start_time) ||
                (length(sc$start_time) == 1 && (is.na(sc$start_time) || sc$start_time == "NA"))
  if (st_missing) {
    sc$start_time <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    sc$status     <- "running"
  }

  sc$last_rep      <- rep_id
  sc$last_rep_time <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")

  ## Ensure numeric fields are not NULL (JSON deserialization safety)
  if (is.null(sc$completed_reps))  sc$completed_reps  <- 0L
  if (is.null(sc$elapsed_sec))     sc$elapsed_sec     <- 0
  if (is.null(sc$bundles_created)) sc$bundles_created  <- 0L
  if (is.null(sc$total_reps))      sc$total_reps       <- config$evaluation$R
  if (is.null(sc$avg_rep_sec))     sc$avg_rep_sec      <- NA_real_
  if (is.null(sc$eta_hours))       sc$eta_hours        <- NA_real_
  if (is.null(status$global$completed_all))  status$global$completed_all  <- 0L
  if (is.null(status$global$failed_all))     status$global$failed_all     <- 0L
  if (is.null(status$global$corrupted_all))  status$global$corrupted_all  <- 0L
  if (is.null(status$global$total_reps_all)) status$global$total_reps_all <- config$evaluation$R * length(config$scenario_ids)
  if (is.null(status$global$errors))         status$global$errors         <- list()

  if (rep_status == "OK") {
    sc$completed_reps <- sc$completed_reps + 1L
    status$global$completed_all <- status$global$completed_all + 1L
  } else {
    existing_failed <- if (!is.null(sc$failed_reps) && length(sc$failed_reps) > 0) {
      as.integer(unlist(sc$failed_reps))
    } else integer(0)
    sc$failed_reps <- unique(c(existing_failed, rep_id))
    status$global$failed_all <- status$global$failed_all + 1L
    ## Log error details
    status$global$errors <- c(status$global$errors, list(list(
      scenario  = scenario_id,
      rep       = rep_id,
      status    = rep_status,
      time      = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    )))
  }

  ## Timing estimates
  sc$elapsed_sec <- sc$elapsed_sec + rep_time_sec
  if (sc$completed_reps > 0) {
    sc$avg_rep_sec <- sc$elapsed_sec / sc$completed_reps
    remaining <- sc$total_reps - sc$completed_reps - length(sc$failed_reps)
    sc$eta_hours <- (remaining * sc$avg_rep_sec) / 3600
  }

  ## Check if scenario complete
  if (sc$completed_reps + length(sc$failed_reps) >= sc$total_reps) {
    sc$status <- "completed"
  }

  status$scenarios[[sc_idx]] <- sc
  status$last_updated <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")

  .write_status(status)

  ## Print compact progress
  pct <- 100 * sc$completed_reps / sc$total_reps
  eta_str <- if (!is.na(sc$eta_hours)) sprintf("%.1fh", sc$eta_hours) else "?"
  cat(sprintf("  [CHECKPOINT] %s: %d/%d (%.0f%%) | last=rep_%03d | ETA=%s\n",
              scenario_id, sc$completed_reps, sc$total_reps,
              pct, rep_id, eta_str))

  ## Check if we should create a bundle
  if (sc$completed_reps > 0 &&
      sc$completed_reps %% CHECKPOINT_CONFIG$bundle_every_n == 0) {
    checkpoint_create_bundle(scenario_id, config,
                              label = sprintf("auto_%s_n%03d",
                                              scenario_id, sc$completed_reps))
  }

  ## Check if this is a milestone (for notifications)
  if (sc$completed_reps %in% CHECKPOINT_CONFIG$milestone_reps) {
    checkpoint_notify(
      type    = "milestone",
      message = sprintf("[%s] %d/%d reps completed (%.0f%%), ETA=%.1fh",
                        scenario_id, sc$completed_reps, sc$total_reps,
                        pct, ifelse(is.na(sc$eta_hours), 0, sc$eta_hours)),
      config  = config
    )
  }

  invisible(TRUE)
}


#' Read Current Status from JSON File
#' @keywords internal
.read_status <- function() {
  sf <- CHECKPOINT_CONFIG$status_file
  if (is.null(sf) || !file.exists(sf)) return(NULL)

  tryCatch({
    if (.CHECKPOINT_HAS_JSONLITE) {
      fromJSON(sf, simplifyVector = FALSE)
    } else {
      readRDS(sub("\\.json$", ".rds", sf))
    }
  }, error = function(e) {
    cat(sprintf("  [CHECKPOINT WARN] Failed to read status: %s\n", e$message))
    NULL
  })
}


#' Write Status to JSON File
#' @keywords internal
.write_status <- function(status) {
  sf <- CHECKPOINT_CONFIG$status_file
  if (is.null(sf)) return(invisible(FALSE))

  tryCatch({
    if (.CHECKPOINT_HAS_JSONLITE) {
      write(toJSON(status, pretty = TRUE, auto_unbox = TRUE), sf)
    } else {
      ## Fallback: save as RDS
      rds_path <- sub("\\.json$", ".rds", sf)
      saveRDS(status, rds_path)
    }
  }, error = function(e) {
    cat(sprintf("  [CHECKPOINT WARN] Failed to write status: %s\n", e$message))
  })

  invisible(TRUE)
}


#' Get Status Summary (for display)
#'
#' @param config  SIM_CONFIG list.
#' @return Character string with formatted summary.

checkpoint_get_summary <- function(config = NULL) {
  status <- .read_status()
  if (is.null(status)) return("  [CHECKPOINT] No status file found.\n")

  lines <- character(0)
  lines <- c(lines, sprintf("  === Checkpoint Status (%s) ===",
                             status$last_updated))

  for (sc in status$scenarios) {
    ## JSON null-safety
    completed <- if (is.null(sc$completed_reps)) 0L else as.integer(sc$completed_reps)
    total     <- if (is.null(sc$total_reps)) 200L else as.integer(sc$total_reps)
    avg_sec   <- tryCatch(as.numeric(sc$avg_rep_sec), warning = function(w) NA_real_, error = function(e) NA_real_)
    if (is.null(avg_sec) || length(avg_sec) == 0) avg_sec <- NA_real_
    eta_h     <- tryCatch(as.numeric(sc$eta_hours), warning = function(w) NA_real_, error = function(e) NA_real_)
    if (is.null(eta_h) || length(eta_h) == 0) eta_h <- NA_real_
    sc_status <- if (is.null(sc$status)) "?" else sc$status
    failed    <- if (!is.null(sc$failed_reps) && length(sc$failed_reps) > 0) {
      as.integer(unlist(sc$failed_reps))
    } else integer(0)

    pct <- 100 * completed / max(1, total)
    eta_str <- if (!is.na(eta_h)) sprintf("%.1fh", eta_h) else "?"
    fail_str <- if (length(failed) > 0) {
      sprintf(" | FAILED: %s", paste(failed, collapse = ","))
    } else ""

    lines <- c(lines, sprintf("  %s [%s]: %d/%d (%.0f%%) | avg=%.0fs | ETA=%s%s",
                               sc$id, sc_status,
                               completed, total, pct,
                               ifelse(is.na(avg_sec), 0, avg_sec),
                               eta_str, fail_str))
  }

  lines <- c(lines, sprintf("  Global: %d/%d completed, %d failed, %d corrupted",
                             status$global$completed_all,
                             status$global$total_reps_all,
                             status$global$failed_all,
                             status$global$corrupted_all))

  if (!is.na(status$global$disk_free_gb)) {
    lines <- c(lines, sprintf("  Disk free: %.1f GB", status$global$disk_free_gb))
  }

  paste(lines, collapse = "\n")
}


###############################################################################
## SECTION 3 : INTEGRITY VALIDATION
###############################################################################

#' Scan and Validate All Saved .rds Files
#'
#' Checks that each saved file is:
#'   (a) Readable by readRDS()
#'   (b) Contains expected structure (theta_hat, diagnostics, etc.)
#'   (c) Has non-zero file size
#'
#' @param scenario_id  Character. Which scenario to scan (or "all").
#' @param config       SIM_CONFIG list.
#' @param rep_range    Integer vector. Which reps to check.
#' @param verbose      Logical. Print per-file details.
#'
#' @return Data.frame with columns: scenario, rep, estimator, status, size_kb, error.

checkpoint_scan_integrity <- function(scenario_id = "all", config,
                                       rep_range = 1:200, verbose = FALSE) {

  scenarios <- if (scenario_id == "all") config$scenario_ids else scenario_id
  results <- data.frame(
    scenario = character(0), rep = integer(0), estimator = character(0),
    status = character(0), size_kb = numeric(0), error = character(0),
    stringsAsFactors = FALSE
  )

  for (sid in scenarios) {
    for (r in rep_range) {
      for (est in c("E_UW", "E_WT")) {
        fpath <- file.path(config$paths$sim_fits, sid, est,
                           sprintf("rep_%03d.rds", r))

        row <- .validate_single_file(fpath, sid, r, est, verbose)
        results <- rbind(results, row)
      }
    }
  }

  ## Summary
  n_total <- nrow(results)
  n_ok    <- sum(results$status == "OK")
  n_miss  <- sum(results$status == "MISSING")
  n_bad   <- sum(results$status %in% c("CORRUPT", "INVALID_STRUCTURE"))

  cat(sprintf("\n  [INTEGRITY] Scan complete: %d files checked\n", n_total))
  cat(sprintf("    OK:      %d\n", n_ok))
  cat(sprintf("    Missing: %d\n", n_miss))
  cat(sprintf("    Bad:     %d\n", n_bad))

  if (n_bad > 0) {
    cat("    Corrupted/Invalid files:\n")
    bad_rows <- results[results$status %in% c("CORRUPT", "INVALID_STRUCTURE"), ]
    for (i in seq_len(nrow(bad_rows))) {
      cat(sprintf("      %s %s rep_%03d: %s (%s)\n",
                  bad_rows$scenario[i], bad_rows$estimator[i],
                  bad_rows$rep[i], bad_rows$status[i], bad_rows$error[i]))
    }

    ## Update status file
    .update_corrupted_in_status(bad_rows, config)
  }

  invisible(results)
}


#' Validate a Single .rds File
#' @keywords internal
.validate_single_file <- function(fpath, scenario_id, rep_id, estimator, verbose) {

  row <- data.frame(
    scenario  = scenario_id,
    rep       = rep_id,
    estimator = estimator,
    status    = "OK",
    size_kb   = 0,
    error     = "",
    stringsAsFactors = FALSE
  )

  ## Check existence
  if (!file.exists(fpath)) {
    row$status <- "MISSING"
    return(row)
  }

  ## Check file size
  fsize <- file.info(fpath)$size
  row$size_kb <- fsize / 1024

  if (fsize == 0) {
    row$status <- "CORRUPT"
    row$error  <- "Zero-byte file"
    return(row)
  }

  ## Try reading
  obj <- tryCatch(readRDS(fpath), error = function(e) {
    row$status <<- "CORRUPT"
    row$error  <<- paste("readRDS failed:", e$message)
    NULL
  })

  if (is.null(obj)) return(row)

  ## Structure validation
  if (estimator == "E_UW") {
    required <- c("theta_hat", "diagnostics")
  } else {
    required <- c("theta_hat", "diagnostics")
  }

  missing_fields <- setdiff(required, names(obj))
  if (length(missing_fields) > 0) {
    row$status <- "INVALID_STRUCTURE"
    row$error  <- paste("Missing fields:", paste(missing_fields, collapse = ", "))
    return(row)
  }

  if (verbose) {
    cat(sprintf("    [OK] %s %s rep_%03d (%.1f KB)\n",
                scenario_id, estimator, rep_id, row$size_kb))
  }

  row
}


#' Update Corrupted Reps in Status File
#' @keywords internal
.update_corrupted_in_status <- function(bad_rows, config) {
  status <- .read_status()
  if (is.null(status)) return(invisible(NULL))

  for (i in seq_len(nrow(bad_rows))) {
    sid <- bad_rows$scenario[i]
    rid <- bad_rows$rep[i]

    sc_idx <- which(sapply(status$scenarios, function(s) s$id) == sid)
    if (length(sc_idx) > 0) {
      sc <- status$scenarios[[sc_idx]]
      sc$corrupted_reps <- unique(c(sc$corrupted_reps, rid))
      status$scenarios[[sc_idx]] <- sc
    }
  }

  status$global$corrupted_all <- sum(sapply(status$scenarios,
                                             function(s) length(s$corrupted_reps)))
  .write_status(status)
}


#' Get Reps That Need Re-Running (failed + corrupted)
#'
#' @param scenario_id  Character.
#' @param config       SIM_CONFIG list.
#' @param rep_range    Integer vector. Full target range.
#'
#' @return Integer vector of rep IDs that need re-running.

checkpoint_get_rerun_reps <- function(scenario_id, config, rep_range = 1:200) {

  ## 1. Missing reps (no file on disk)
  missing_reps <- integer(0)
  for (r in rep_range) {
    uw_path <- file.path(config$paths$sim_fits, scenario_id, "E_UW",
                         sprintf("rep_%03d.rds", r))
    wt_path <- file.path(config$paths$sim_fits, scenario_id, "E_WT",
                         sprintf("rep_%03d.rds", r))
    if (!file.exists(uw_path) || !file.exists(wt_path)) {
      missing_reps <- c(missing_reps, r)
    }
  }

  ## 2. Corrupted reps (from integrity scan)
  status <- .read_status()
  corrupted_reps <- integer(0)
  if (!is.null(status)) {
    sc_idx <- which(sapply(status$scenarios, function(s) s$id) == scenario_id)
    if (length(sc_idx) > 0) {
      cr <- status$scenarios[[sc_idx]]$corrupted_reps
      ## Handle JSON deserialization: may come back as list or NULL
      if (!is.null(cr) && length(cr) > 0) {
        corrupted_reps <- as.integer(unlist(cr))
      }
    }
  }

  rerun <- sort(unique(c(missing_reps, corrupted_reps)))

  cat(sprintf("  [CHECKPOINT] %s: %d missing + %d corrupted = %d to re-run\n",
              scenario_id, length(missing_reps), length(corrupted_reps),
              length(rerun)))

  rerun
}


#' Delete Corrupted Files to Force Clean Re-run
#'
#' @param scenario_id  Character.
#' @param rep_ids      Integer vector. Reps to delete.
#' @param config       SIM_CONFIG list.
#' @return Invisible integer count of files deleted.

checkpoint_delete_corrupted <- function(scenario_id, rep_ids, config) {
  deleted <- 0L
  for (r in rep_ids) {
    for (est in c("E_UW", "E_WT")) {
      fpath <- file.path(config$paths$sim_fits, scenario_id, est,
                         sprintf("rep_%03d.rds", r))
      if (file.exists(fpath)) {
        file.remove(fpath)
        deleted <- deleted + 1L
      }
    }
    ## Also delete sample file
    spath <- file.path(config$paths$sim_samples, scenario_id,
                       sprintf("rep_%03d.rds", r))
    if (file.exists(spath)) {
      file.remove(spath)
      deleted <- deleted + 1L
    }
  }
  cat(sprintf("  [CHECKPOINT] Deleted %d files for %d corrupted reps in %s\n",
              deleted, length(rep_ids), scenario_id))
  invisible(deleted)
}


###############################################################################
## SECTION 4 : BUNDLE SNAPSHOTS (tar.gz)
###############################################################################

#' Create a Checkpoint Bundle (tar.gz)
#'
#' Archives current simulation outputs for download/backup.
#' Includes: fits/, samples/, results/, status file, config.
#'
#' @param scenario_id  Character. "S0", "S3", "S4", or "all".
#' @param config       SIM_CONFIG list.
#' @param label        Character. Custom label for the bundle filename.
#' @param include_samples  Logical. Include sample files (larger bundles).
#'
#' @return Character. Path to the created bundle file.

checkpoint_create_bundle <- function(scenario_id = "all", config,
                                      label = NULL,
                                      include_samples = FALSE) {

  if (is.null(label)) {
    label <- format(Sys.time(), "%Y%m%d_%H%M%S")
  }

  bundle_name <- sprintf("checkpoint_%s_%s.tar.gz", label,
                          format(Sys.time(), "%Y%m%d_%H%M%S"))
  bundle_path <- file.path(CHECKPOINT_CONFIG$bundle_dir, bundle_name)

  ## Collect files to archive
  files_to_archive <- character(0)

  scenarios <- if (scenario_id == "all") config$scenario_ids else scenario_id

  for (sid in scenarios) {
    ## Fit files
    for (est in c("E_UW", "E_WT")) {
      fit_dir <- file.path(config$paths$sim_fits, sid, est)
      fit_files <- list.files(fit_dir, pattern = "\\.rds$", full.names = TRUE)
      files_to_archive <- c(files_to_archive, fit_files)
    }

    ## Sample files (optional — they can be large)
    if (include_samples) {
      sample_dir <- file.path(config$paths$sim_samples, sid)
      sample_files <- list.files(sample_dir, pattern = "\\.rds$",
                                  full.names = TRUE)
      files_to_archive <- c(files_to_archive, sample_files)
    }
  }

  ## Always include: status file, config, results
  meta_files <- c(
    CHECKPOINT_CONFIG$status_file,
    config$paths$sim_config_out
  )
  result_files <- list.files(config$paths$sim_results, pattern = "\\.rds$",
                              full.names = TRUE)
  files_to_archive <- c(files_to_archive, meta_files, result_files)
  files_to_archive <- files_to_archive[file.exists(files_to_archive)]

  if (length(files_to_archive) == 0) {
    cat("  [CHECKPOINT] No files to bundle.\n")
    return(invisible(NULL))
  }

  ## Write file list to temp file for tar
  filelist_tmp <- tempfile(fileext = ".txt")
  writeLines(files_to_archive, filelist_tmp)

  ## Create tar.gz using base R (cross-platform)
  t_start <- proc.time()

  cmd <- sprintf("tar -czf '%s' -T '%s' 2>/dev/null", bundle_path, filelist_tmp)
  ret <- system(cmd, intern = FALSE)

  if (ret != 0) {
    ## Fallback: use R's tar function
    tryCatch({
      tar(bundle_path, files = files_to_archive,
          compression = "gzip", tar = "internal")
    }, error = function(e) {
      cat(sprintf("  [CHECKPOINT ERROR] Bundle creation failed: %s\n", e$message))
      return(invisible(NULL))
    })
  }

  t_elapsed <- (proc.time() - t_start)[3]
  bundle_size_mb <- file.info(bundle_path)$size / (1024^2)

  cat(sprintf("  [CHECKPOINT] Bundle created: %s\n", basename(bundle_path)))
  cat(sprintf("    Files: %d, Size: %.1f MB, Time: %.1f sec\n",
              length(files_to_archive), bundle_size_mb, t_elapsed))

  ## Update status
  status <- .read_status()
  if (!is.null(status)) {
    status$global$last_bundle_time <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    for (sid in scenarios) {
      sc_idx <- which(sapply(status$scenarios, function(s) s$id) == sid)
      if (length(sc_idx) > 0) {
        status$scenarios[[sc_idx]]$bundles_created <-
          status$scenarios[[sc_idx]]$bundles_created + 1L
      }
    }
    .write_status(status)
  }

  ## Clean up old bundles (keep at most 5)
  .cleanup_old_bundles(config, keep_n = 5)

  invisible(bundle_path)
}


#' Clean Up Old Bundles (keep only N most recent)
#' @keywords internal
.cleanup_old_bundles <- function(config, keep_n = 5) {
  bundle_dir <- CHECKPOINT_CONFIG$bundle_dir
  if (!dir.exists(bundle_dir)) return(invisible(NULL))

  bundles <- list.files(bundle_dir, pattern = "\\.tar\\.gz$", full.names = TRUE)
  if (length(bundles) <= keep_n) return(invisible(NULL))

  ## Sort by modification time (oldest first)
  bundle_info <- file.info(bundles)
  bundle_info$path <- bundles
  bundle_info <- bundle_info[order(bundle_info$mtime), ]

  ## Remove oldest ones
  to_remove <- head(bundle_info$path, length(bundles) - keep_n)
  for (f in to_remove) {
    file.remove(f)
    cat(sprintf("  [CHECKPOINT] Removed old bundle: %s\n", basename(f)))
  }
}


###############################################################################
## SECTION 5 : DISK SPACE MONITORING
###############################################################################

#' Check Available Disk Space
#'
#' @param halt_on_low  Logical. If TRUE and disk < halt threshold, stop().
#' @return Numeric. Free disk space in GB (or NA if cannot determine).

checkpoint_check_disk <- function(halt_on_low = TRUE) {

  free_gb <- tryCatch({
    ## Use df command (works on macOS and Linux)
    output <- system("df -g / 2>/dev/null | tail -1 | awk '{print $4}'",
                     intern = TRUE)
    if (length(output) == 0 || output == "") {
      ## Try alternative (Linux with df -BG)
      output <- system("df -BG / 2>/dev/null | tail -1 | awk '{print $4}'",
                       intern = TRUE)
      as.numeric(gsub("G", "", output))
    } else {
      as.numeric(output)
    }
  }, error = function(e) NA_real_)

  if (is.na(free_gb)) {
    cat("  [CHECKPOINT] Could not determine disk space.\n")
    return(invisible(NA_real_))
  }

  ## Update status file
  status <- .read_status()
  if (!is.null(status)) {
    status$global$disk_free_gb <- free_gb
    .write_status(status)
  }

  ## Check thresholds
  if (free_gb < CHECKPOINT_CONFIG$disk_halt_gb && halt_on_low) {
    msg <- sprintf("DISK CRITICALLY LOW: %.1f GB free (threshold: %.1f GB). HALTING.",
                   free_gb, CHECKPOINT_CONFIG$disk_halt_gb)
    cat(sprintf("  [CHECKPOINT CRITICAL] %s\n", msg))
    checkpoint_notify("error", msg, config = NULL)
    stop(msg, call. = FALSE)
  } else if (free_gb < CHECKPOINT_CONFIG$disk_warn_gb) {
    cat(sprintf("  [CHECKPOINT WARN] Disk space low: %.1f GB free\n", free_gb))
  } else {
    cat(sprintf("  [CHECKPOINT] Disk space OK: %.1f GB free\n", free_gb))
  }

  invisible(free_gb)
}


###############################################################################
## SECTION 6 : NOTIFICATIONS
###############################################################################

#' Send Notification (Email or Slack)
#'
#' @param type     Character. "milestone", "error", or "completion".
#' @param message  Character. The notification message.
#' @param config   SIM_CONFIG list (or NULL).
#'
#' @return Invisible TRUE if sent, FALSE if skipped.

checkpoint_notify <- function(type, message, config = NULL) {

  if (!(type %in% CHECKPOINT_CONFIG$notify_on)) {
    return(invisible(FALSE))
  }

  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  full_msg <- sprintf("[%s] HBB Sim: [%s] %s", timestamp, type, message)

  ## Console output always
  cat(sprintf("  [NOTIFY:%s] %s\n", toupper(type), message))

  ## Email notification
  if (!is.null(CHECKPOINT_CONFIG$notify_email)) {
    .send_email_notification(full_msg, type)
  }

  ## Slack webhook notification
  if (!is.null(CHECKPOINT_CONFIG$notify_slack)) {
    .send_slack_notification(full_msg, type)
  }

  ## Also log to a notification log file
  log_path <- file.path(
    if (!is.null(config)) config$paths$sim_output_root
    else dirname(CHECKPOINT_CONFIG$status_file),
    "checkpoint_notifications.log"
  )
  tryCatch(
    cat(full_msg, "\n", file = log_path, append = TRUE),
    error = function(e) NULL
  )

  invisible(TRUE)
}


#' Send Email Notification via mailx/sendmail
#' @keywords internal
.send_email_notification <- function(message, type) {
  email <- CHECKPOINT_CONFIG$notify_email
  if (is.null(email)) return(invisible(FALSE))

  subject <- sprintf("HBB Sim [%s]", toupper(type))

  ## Try mailx (available on most Unix systems)
  cmd <- sprintf('echo "%s" | mailx -s "%s" "%s" 2>/dev/null',
                 gsub('"', '\\"', message), subject, email)

  ret <- tryCatch(system(cmd, intern = FALSE), error = function(e) 1)
  if (ret != 0) {
    cat(sprintf("  [CHECKPOINT] Email notification failed (mailx not available).\n"))
  }
  invisible(ret == 0)
}


#' Send Slack Webhook Notification
#' @keywords internal
.send_slack_notification <- function(message, type) {
  webhook_url <- CHECKPOINT_CONFIG$notify_slack
  if (is.null(webhook_url)) return(invisible(FALSE))

  ## Emoji by type
  emoji <- switch(type,
    "milestone" = ":white_check_mark:",
    "error"     = ":x:",
    "completion" = ":tada:",
    ":information_source:"
  )

  payload <- sprintf('{"text":"%s %s"}', emoji, gsub('"', '\\"', message))

  cmd <- sprintf("curl -s -X POST -H 'Content-type: application/json' --data '%s' '%s' 2>/dev/null",
                 payload, webhook_url)

  ret <- tryCatch(system(cmd, intern = FALSE), error = function(e) 1)
  if (ret != 0) {
    cat("  [CHECKPOINT] Slack notification failed.\n")
  }
  invisible(ret == 0)
}


###############################################################################
## SECTION 7 : CONVENIENCE WRAPPERS
###############################################################################

#' Full Pre-Run Check
#'
#' Performs all startup checks:
#'   1. Initialize status file
#'   2. Disk space check
#'   3. Integrity scan of existing files
#'   4. Report reps to run
#'
#' @param config         SIM_CONFIG list.
#' @param scenario_ids   Character vector. Scenarios to check.
#' @param rep_range      Integer vector.
#' @param notify_email   Optional email.
#' @param notify_slack   Optional Slack webhook.
#'
#' @return Named list of reps to run per scenario.

checkpoint_preflight <- function(config,
                                  scenario_ids = config$scenario_ids,
                                  rep_range = 1:200,
                                  notify_email = NULL,
                                  notify_slack = NULL) {

  cat("\n  ============================================\n")
  cat("  CHECKPOINT PREFLIGHT CHECK\n")
  cat("  ============================================\n\n")

  ## 1. Initialize
  checkpoint_init(config, notify_email = notify_email, notify_slack = notify_slack)

  ## 2. Disk check
  checkpoint_check_disk(halt_on_low = TRUE)

  ## 3. Integrity scan (only for existing files)
  if (CHECKPOINT_CONFIG$integrity_on_startup) {
    cat("\n  Scanning existing files for integrity...\n")
    integrity <- checkpoint_scan_integrity("all", config, rep_range, verbose = FALSE)
  }

  ## 4. Identify reps to run
  reps_to_run <- list()
  for (sid in scenario_ids) {
    rerun <- checkpoint_get_rerun_reps(sid, config, rep_range)
    reps_to_run[[sid]] <- rerun
  }

  total_to_run <- sum(sapply(reps_to_run, length))
  cat(sprintf("\n  Total reps to run: %d\n", total_to_run))
  for (sid in scenario_ids) {
    cat(sprintf("    %s: %d reps\n", sid, length(reps_to_run[[sid]])))
  }

  cat("\n  ============================================\n")
  cat("  PREFLIGHT COMPLETE\n")
  cat("  ============================================\n\n")

  invisible(reps_to_run)
}


#' Post-Run Finalization
#'
#' @param config  SIM_CONFIG list.
#' @return Invisible TRUE.

checkpoint_finalize <- function(config) {

  cat("\n  [CHECKPOINT] Running post-run finalization...\n")

  ## 1. Final integrity scan
  cat("  Final integrity scan...\n")
  integrity <- checkpoint_scan_integrity("all", config, verbose = FALSE)

  ## 2. Create final bundle
  cat("  Creating final bundle...\n")
  checkpoint_create_bundle("all", config, label = "FINAL",
                            include_samples = TRUE)

  ## 3. Print summary
  cat(checkpoint_get_summary(config))
  cat("\n")

  ## 4. Send completion notification
  status <- .read_status()
  if (!is.null(status)) {
    msg <- sprintf("SIMULATION COMPLETE: %d/%d reps (%.1f%% success), %d failed",
                   status$global$completed_all, status$global$total_reps_all,
                   100 * status$global$completed_all / max(1, status$global$total_reps_all),
                   status$global$failed_all)
    checkpoint_notify("completion", msg, config)
  }

  invisible(TRUE)
}


###############################################################################
## FINAL
###############################################################################

cat("  sim_06_checkpoint.R loaded.\n")
cat("  Functions: checkpoint_init(), checkpoint_update_status(),\n")
cat("             checkpoint_scan_integrity(), checkpoint_create_bundle(),\n")
cat("             checkpoint_check_disk(), checkpoint_notify(),\n")
cat("             checkpoint_preflight(), checkpoint_finalize(),\n")
cat("             checkpoint_get_summary(), checkpoint_get_rerun_reps()\n\n")
