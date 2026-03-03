## =============================================================================
## B2_m3b_comparison.R -- M3b vs M3b-W Comparison Table
## =============================================================================
## Purpose : Build a side-by-side comparison table of M3b (unweighted) vs
##           M3b-W (survey-weighted) fixed-effect and random-effect estimates,
##           highlighting the impact of survey weighting on point estimates,
##           sign preservation, and random-effect SDs.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##
## Design Philosophy:
##   - Produce a compact companion table for the main text
##   - Key questions for readers:
##     (1) Does weighting change the SIGN of any coefficient?
##     (2) How large are shifts relative to the sandwich SE?
##     (3) Do the random-effect SDs (tau) change substantially?
##
## Parts:
##   A: Fixed-effect comparison (11 rows x ~8 columns)
##   B: Random-effect SD comparison (10 tau elements)
##   C: Poverty reversal robustness check
##   D: LaTeX table generation (main-text + SM)
##   E: Summary statistics
##
## Inputs  :
##   data/precomputed/results_m3b.rds
##   data/precomputed/results_m3b_weighted.rds
##   data/precomputed/sandwich_variance.rds
##   data/precomputed/cholesky_correction.rds
##
## Outputs :
##   data/precomputed/B2_m3b_comparison.rds
##   output/tables/T_m3b_comparison.tex
##   output/tables/T_m3b_comparison.csv
##   output/tables/ST_tau_comparison.tex
##   output/tables/ST_tau_comparison.csv
## =============================================================================

cat("==============================================================\n")
cat("  M3b vs M3b-W Comparison\n")
cat("==============================================================\n\n")


###############################################################################
## SECTION 0 : SETUP
###############################################################################

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
OUTPUT_DIR   <- file.path(PROJECT_ROOT, "data/precomputed")
FIGURE_DIR   <- file.path(PROJECT_ROOT, "output/tables")
B2_OUT       <- file.path(OUTPUT_DIR, "B2_m3b_comparison.rds")

## Create output directory (should already exist)
if (!dir.exists(FIGURE_DIR)) dir.create(FIGURE_DIR, recursive = TRUE)

## Dimensions
D <- 11   # fixed effects
P <- 5    # covariates per margin

## Covariate labels
COV_LABELS <- c("intercept", "poverty", "urban", "black", "hispanic")

## Display labels for LaTeX (matching existing tab:fixed-effects)
PARAM_DISPLAY <- c(
  "Intercept", "Poverty", "Urban", "Black", "Hispanic",
  "Intercept", "Poverty", "Urban", "Black", "Hispanic",
  "$\\log\\kappa$"
)

## Internal parameter labels
PARAM_LABELS <- c(
  "alpha_intercept", "alpha_poverty", "alpha_urban",
  "alpha_black", "alpha_hispanic",
  "beta_intercept", "beta_poverty", "beta_urban",
  "beta_black", "beta_hispanic",
  "log_kappa"
)

## Tau labels (10 random-effect SDs)
TAU_LABELS <- c(
  "ext_intercept", "ext_poverty", "ext_urban", "ext_black", "ext_hispanic",
  "int_intercept", "int_poverty", "int_urban", "int_black", "int_hispanic"
)
TAU_DISPLAY <- c(
  "Intercept", "Poverty", "Urban", "Black", "Hispanic",
  "Intercept", "Poverty", "Urban", "Black", "Hispanic"
)

cat(sprintf("  PROJECT_ROOT: %s\n", PROJECT_ROOT))
cat(sprintf("  Output file:  %s\n\n", B2_OUT))


###############################################################################
## SECTION 1 : LOAD INPUTS
###############################################################################
cat("--- 1. Loading inputs ---\n\n")

## 1a. M3b (unweighted)
m3b_path <- file.path(OUTPUT_DIR, "results_m3b.rds")
stopifnot("M3b results file not found" = file.exists(m3b_path))
res_m3b <- readRDS(m3b_path)
cat(sprintf("  Loaded: %s\n", m3b_path))
cat(sprintf("    Model: %s\n", res_m3b$model_desc))

## 1b. M3b-W (survey-weighted)
m3bw_path <- file.path(OUTPUT_DIR, "results_m3b_weighted.rds")
stopifnot("M3b-W results file not found" = file.exists(m3bw_path))
res_m3bw <- readRDS(m3bw_path)
cat(sprintf("  Loaded: %s\n", m3bw_path))
cat(sprintf("    Model: %s\n", res_m3bw$model_desc))

## 1c. Sandwich variance (for Wald SEs of M3b-W)
sand_path <- file.path(OUTPUT_DIR, "sandwich_variance.rds")
stopifnot("Sandwich variance file not found" = file.exists(sand_path))
sand_var <- readRDS(sand_path)
cat(sprintf("  Loaded: %s\n", sand_path))

## 1d. Cholesky correction (for Wald CIs of M3b-W)
chol_path <- file.path(OUTPUT_DIR, "cholesky_correction.rds")
stopifnot("Cholesky correction file not found" = file.exists(chol_path))
chol_corr <- readRDS(chol_path)
cat(sprintf("  Loaded: %s\n\n", chol_path))

## Extract key quantities
V_sand   <- sand_var$V_sand         # 11 x 11 sandwich variance
se_wald  <- sqrt(diag(V_sand))      # sandwich SEs (length 11)
names(se_wald) <- PARAM_LABELS

ct <- chol_corr$comparison_table    # 11-row data.frame with wald_lo, wald_hi

cat("  [PASS] All 4 input files loaded (no heavy fit objects needed).\n\n")


###############################################################################
## SECTION 2 : PART A -- FIXED-EFFECT COMPARISON
###############################################################################
cat("--- 2. Part A: Fixed-effect comparison (11 parameters) ---\n\n")

## 2a. Extract posterior means from both models
mean_uw <- c(res_m3b$alpha_means, res_m3b$beta_means, res_m3b$log_kappa_mean)
mean_wt <- c(res_m3bw$alpha_means, res_m3bw$beta_means, res_m3bw$log_kappa_mean)
names(mean_uw) <- PARAM_LABELS
names(mean_wt) <- PARAM_LABELS

cat("  M3b (UW) posterior means:\n")
for (i in seq_len(D)) {
  cat(sprintf("    %-20s  %+.4f\n", PARAM_LABELS[i], mean_uw[i]))
}

cat("\n  M3b-W (WT) posterior means:\n")
for (i in seq_len(D)) {
  cat(sprintf("    %-20s  %+.4f\n", PARAM_LABELS[i], mean_wt[i]))
}

## 2b. Compute shift = WT - UW
shift <- mean_wt - mean_uw
names(shift) <- PARAM_LABELS

## 2c. Normalize shift by sandwich SE:
##     z_shift = shift / SE_wald
##     This gives a z-score-like measure of how many design-corrected SEs
##     the weighting moves the estimate.  It is NOT a formal test statistic
##     because SE_wald is the variance of M3b-W alone, not of the difference.
z_shift <- shift / se_wald
names(z_shift) <- PARAM_LABELS

cat("\n  Shift and normalized shift:\n")
cat(sprintf("  %-20s  %10s  %8s  %10s\n",
            "Parameter", "Shift", "SE_wald", "Shift/SE"))
cat(sprintf("  %s\n", paste(rep("-", 54), collapse = "")))
for (i in seq_len(D)) {
  cat(sprintf("  %-20s  %+10.4f  %8.4f  %+10.2f\n",
              PARAM_LABELS[i], shift[i], se_wald[i], z_shift[i]))
}

## 2d. Sign preservation check
sign_uw <- sign(mean_uw)
sign_wt <- sign(mean_wt)
sign_preserved <- (sign_uw == sign_wt)
names(sign_preserved) <- PARAM_LABELS

n_sign_change <- sum(!sign_preserved)

cat(sprintf("\n  Sign changes: %d / %d parameters\n", n_sign_change, D))
if (n_sign_change > 0) {
  for (i in which(!sign_preserved)) {
    cat(sprintf("    [SIGN CHANGE] %s: M3b = %+.4f -> M3b-W = %+.4f\n",
                PARAM_LABELS[i], mean_uw[i], mean_wt[i]))
  }
} else {
  cat("    All signs preserved under weighting.\n")
}

## 2e. Wald CI significance (from cholesky_correction comparison_table)
wald_lo <- ct$wald_lo
wald_hi <- ct$wald_hi
wald_sig <- (wald_lo > 0) | (wald_hi < 0)  # CI excludes zero
names(wald_sig) <- PARAM_LABELS

cat(sprintf("\n  M3b-W Wald 95%% CI significance: %d / %d significant\n",
            sum(wald_sig), D))
nonsig_params <- PARAM_LABELS[!wald_sig]
if (length(nonsig_params) > 0) {
  cat(sprintf("    Non-significant: %s\n", paste(nonsig_params, collapse = ", ")))
}

## 2f. Build the comparison data.frame
fixed_comparison <- data.frame(
  parameter      = PARAM_LABELS,
  display_label  = PARAM_DISPLAY,
  margin         = c(rep("Extensive", 5), rep("Intensive", 5), "Dispersion"),
  mean_uw        = as.numeric(mean_uw),
  mean_wt        = as.numeric(mean_wt),
  shift          = as.numeric(shift),
  se_wald        = as.numeric(se_wald),
  z_shift        = as.numeric(z_shift),
  wald_lo        = wald_lo,
  wald_hi        = wald_hi,
  wald_sig       = wald_sig,
  sign_preserved = sign_preserved,
  stringsAsFactors = FALSE,
  row.names      = NULL
)

cat("\n")


###############################################################################
## SECTION 3 : PART B -- RANDOM-EFFECT SD COMPARISON (TAU)
###############################################################################
cat("--- 3. Part B: Random-effect SD comparison (10 tau parameters) ---\n\n")

## 3a. Extract tau vectors
tau_uw <- res_m3b$tau_means     # named vector of length 10
tau_wt <- res_m3bw$tau_means    # named vector of length 10
stopifnot(length(tau_uw) == 10, length(tau_wt) == 10)
names(tau_uw) <- TAU_LABELS
names(tau_wt) <- TAU_LABELS

## 3b. Compute absolute and relative changes
tau_shift <- tau_wt - tau_uw
tau_rel   <- tau_shift / tau_uw   # (tau_wt - tau_uw) / tau_uw

cat("  Random-effect SD (tau) comparison:\n")
cat(sprintf("  %-20s  %8s  %8s  %10s  %10s\n",
            "Parameter", "M3b", "M3b-W", "Shift", "Rel.Change"))
cat(sprintf("  %s\n", paste(rep("-", 62), collapse = "")))

cat("  Extensive margin:\n")
for (i in 1:5) {
  cat(sprintf("    %-18s  %8.4f  %8.4f  %+10.4f  %+10.1f%%\n",
              COV_LABELS[i], tau_uw[i], tau_wt[i],
              tau_shift[i], 100 * tau_rel[i]))
}
cat("  Intensive margin:\n")
for (i in 6:10) {
  cat(sprintf("    %-18s  %8.4f  %8.4f  %+10.4f  %+10.1f%%\n",
              COV_LABELS[i - 5], tau_uw[i], tau_wt[i],
              tau_shift[i], 100 * tau_rel[i]))
}

## 3c. Build the tau comparison data.frame
tau_comparison <- data.frame(
  parameter     = TAU_LABELS,
  display_label = TAU_DISPLAY,
  margin        = rep(c("Extensive", "Intensive"), each = 5),
  tau_uw        = as.numeric(tau_uw),
  tau_wt        = as.numeric(tau_wt),
  tau_shift     = as.numeric(tau_shift),
  rel_change    = as.numeric(tau_rel),
  stringsAsFactors = FALSE,
  row.names     = NULL
)

## 3d. Tau summary
n_tau_increase <- sum(tau_shift > 0)
n_tau_decrease <- sum(tau_shift < 0)
median_rel_ext <- median(tau_rel[1:5])
median_rel_int <- median(tau_rel[6:10])
median_rel_all <- median(tau_rel)

cat(sprintf("\n  Tau summary:\n"))
cat(sprintf("    Median rel. change (extensive): %+.1f%%\n", 100 * median_rel_ext))
cat(sprintf("    Median rel. change (intensive): %+.1f%%\n", 100 * median_rel_int))
cat(sprintf("    Median rel. change (overall):   %+.1f%%\n", 100 * median_rel_all))
cat(sprintf("    Max |rel. change|:  %.1f%% (%s)\n",
            100 * max(abs(tau_rel)), TAU_LABELS[which.max(abs(tau_rel))]))
cat(sprintf("    Direction: %d increase, %d decrease\n",
            n_tau_increase, n_tau_decrease))

cat("\n")


###############################################################################
## SECTION 4 : PART C -- POVERTY REVERSAL ROBUSTNESS
###############################################################################
cat("--- 4. Part C: Poverty reversal robustness ---\n\n")

## The poverty reversal is the paper's central empirical finding:
##   alpha_poverty < 0  (poverty reduces extensive-margin participation)
##   beta_poverty  > 0  (poverty increases intensive-margin IT share)

alpha_pov_uw <- mean_uw["alpha_poverty"]
beta_pov_uw  <- mean_uw["beta_poverty"]
alpha_pov_wt <- mean_wt["alpha_poverty"]
beta_pov_wt  <- mean_wt["beta_poverty"]

reversal_uw <- (alpha_pov_uw < 0) && (beta_pov_uw > 0)
reversal_wt <- (alpha_pov_wt < 0) && (beta_pov_wt > 0)

cat("  Poverty reversal check:\n\n")
cat(sprintf("  %-25s  %12s  %12s\n", "", "M3b (UW)", "M3b-W (WT)"))
cat(sprintf("  %s\n", paste(rep("-", 52), collapse = "")))
cat(sprintf("  %-25s  %+12.4f  %+12.4f\n",
            "alpha (Poverty)", alpha_pov_uw, alpha_pov_wt))
cat(sprintf("  %-25s  %+12.4f  %+12.4f\n",
            "beta (Poverty)", beta_pov_uw, beta_pov_wt))
cat(sprintf("  %-25s  %12s  %12s\n",
            "alpha_pov < 0?",
            ifelse(alpha_pov_uw < 0, "YES", "NO"),
            ifelse(alpha_pov_wt < 0, "YES", "NO")))
cat(sprintf("  %-25s  %12s  %12s\n",
            "beta_pov > 0?",
            ifelse(beta_pov_uw > 0, "YES", "NO"),
            ifelse(beta_pov_wt > 0, "YES", "NO")))
cat(sprintf("  %-25s  %12s  %12s\n",
            "REVERSAL HOLDS?",
            ifelse(reversal_uw, "YES", "NO"),
            ifelse(reversal_wt, "YES", "NO")))

## M3b-W Wald significance for poverty coefficients
alpha_pov_wald_sig <- wald_sig["alpha_poverty"]  # CI excludes zero
beta_pov_wald_sig  <- wald_sig["beta_poverty"]

cat(sprintf("\n  M3b-W Wald significance:\n"))
cat(sprintf("    alpha_poverty: Wald 95%% CI [%.3f, %.3f] %s\n",
            wald_lo[2], wald_hi[2],
            ifelse(alpha_pov_wald_sig, "SIGNIFICANT", "not sig.")))
cat(sprintf("    beta_poverty:  Wald 95%% CI [%.3f, %.3f] %s\n",
            wald_lo[7], wald_hi[7],
            ifelse(beta_pov_wald_sig, "SIGNIFICANT", "not sig.")))

if (reversal_uw && reversal_wt) {
  cat("\n  [PASS] Poverty reversal holds in BOTH models.\n")
  cat("    The sign pattern (alpha_pov < 0, beta_pov > 0) is robust\n")
  cat("    to the inclusion of survey weights.\n")
} else {
  cat("\n  [NOTE] Poverty reversal does NOT hold in both models.\n")
}

## Magnitude comparison
cat(sprintf("\n  Shift in poverty parameters:\n"))
cat(sprintf("    alpha_pov: %+.4f -> %+.4f (shift = %+.4f, z = %+.2f SEs)\n",
            alpha_pov_uw, alpha_pov_wt,
            shift["alpha_poverty"], z_shift["alpha_poverty"]))
cat(sprintf("    beta_pov:  %+.4f -> %+.4f (shift = %+.4f, z = %+.2f SEs)\n",
            beta_pov_uw, beta_pov_wt,
            shift["beta_poverty"], z_shift["beta_poverty"]))

cat("\n")


###############################################################################
## SECTION 5 : PART D -- LATEX TABLE GENERATION
###############################################################################
cat("--- 5. Part D: LaTeX table generation ---\n\n")

## =========================================================================
## TABLE A: Main-text fixed-effect comparison (compact)
## =========================================================================
cat("  5a. Main-text table: Fixed-effect comparison ...\n")

## Format the Wald CI string with significance asterisk
format_wald_ci <- function(lo, hi, sig) {
  ci <- sprintf("$[%s,\\; %s]", formatC(lo, format = "f", digits = 3),
                formatC(hi, format = "f", digits = 3))
  if (sig) {
    ci <- paste0(ci, "^{*}$")
  } else {
    ci <- paste0(ci, "$")
  }
  ci
}

## Format a signed number with explicit + for positive values
fmt_signed <- function(x, digits = 3) {
  s <- formatC(x, format = "f", digits = digits)
  if (x >= 0) s <- paste0("+", s)
  s
}

## Build the LaTeX table line by line
tex <- character()

tex <- c(tex,
  "\\begin{table}[t]",
  "\\centering",
  paste0("\\caption{Comparison of unweighted (M3b) and survey-weighted (M3b-W) ",
         "fixed-effect"),
  paste0("  posterior means. ``Shift'' is M3b-W minus M3b. ",
         "``Shift/SE'' normalizes the shift"),
  paste0("  by the sandwich standard error, providing a scale-free measure of the ",
         "weighting"),
  paste0("  impact. ``Wald 95\\% CI'' is the sandwich-corrected interval for M3b-W ",
         "(\\cref{thm:cholesky})."),
  paste0("  Asterisks indicate significance at the 5\\% level. ",
         "All covariates are standardized.}"),
  "\\label{tab:m3b-comparison}",
  "\\smallskip",
  "\\small",
  "\\begin{adjustbox}{max width=\\textwidth}",
  "\\begin{tabular}{@{}l rr rr l@{}}",
  "\\toprule",
  "Parameter & M3b & M3b-W & Shift & Shift/SE & Wald 95\\% CI \\\\",
  "\\midrule"
)

## --- Extensive margin block ---
tex <- c(tex,
  "\\multicolumn{6}{@{}l}{\\textit{Extensive margin ($\\balpha$)}} \\\\[2pt]"
)

for (i in 1:5) {
  row <- sprintf("%-20s & $%s$ & $%s$ & $%s$ & $%s$ & %s \\\\",
                 PARAM_DISPLAY[i],
                 fmt_signed(mean_uw[i], 3),
                 fmt_signed(mean_wt[i], 3),
                 fmt_signed(shift[i], 3),
                 fmt_signed(z_shift[i], 2),
                 format_wald_ci(wald_lo[i], wald_hi[i], wald_sig[i]))
  ## Add vertical space after last extensive-margin row
  if (i == 5) row <- sub("\\\\\\\\$", "\\\\\\\\[4pt]", row)
  tex <- c(tex, row)
}

## --- Intensive margin block ---
tex <- c(tex,
  "\\multicolumn{6}{@{}l}{\\textit{Intensive margin ($\\bbeta$)}} \\\\[2pt]"
)

for (i in 6:10) {
  row <- sprintf("%-20s & $%s$ & $%s$ & $%s$ & $%s$ & %s \\\\",
                 PARAM_DISPLAY[i],
                 fmt_signed(mean_uw[i], 3),
                 fmt_signed(mean_wt[i], 3),
                 fmt_signed(shift[i], 3),
                 fmt_signed(z_shift[i], 2),
                 format_wald_ci(wald_lo[i], wald_hi[i], wald_sig[i]))
  if (i == 10) row <- sub("\\\\\\\\$", "\\\\\\\\[4pt]", row)
  tex <- c(tex, row)
}

## --- Overdispersion block ---
tex <- c(tex,
  "\\multicolumn{6}{@{}l}{\\textit{Overdispersion}} \\\\[2pt]"
)

row <- sprintf("%-20s & $%s$ & $%s$ & $%s$ & $%s$ & %s \\\\",
               PARAM_DISPLAY[11],
               fmt_signed(mean_uw[11], 3),
               fmt_signed(mean_wt[11], 3),
               fmt_signed(shift[11], 3),
               fmt_signed(z_shift[11], 2),
               format_wald_ci(wald_lo[11], wald_hi[11], wald_sig[11]))
tex <- c(tex, row)

## --- Footer ---
tex <- c(tex,
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{adjustbox}",
  "\\end{table}"
)

## Write main-text LaTeX table
tex_main_path <- file.path(FIGURE_DIR, "T_m3b_comparison.tex")
writeLines(tex, tex_main_path)
cat(sprintf("  [SAVED] %s\n", tex_main_path))

## CSV backup
csv_main_path <- file.path(FIGURE_DIR, "T_m3b_comparison.csv")
csv_df <- data.frame(
  Parameter   = PARAM_DISPLAY,
  Margin      = c(rep("Extensive", 5), rep("Intensive", 5), "Dispersion"),
  M3b         = sprintf("%.3f", mean_uw),
  M3b_W       = sprintf("%.3f", mean_wt),
  Shift       = sprintf("%+.3f", shift),
  SE_wald     = sprintf("%.4f", se_wald),
  Shift_SE    = sprintf("%+.2f", z_shift),
  Wald_95_CI  = sprintf("[%.3f, %.3f]", wald_lo, wald_hi),
  Significant = ifelse(wald_sig, "*", ""),
  stringsAsFactors = FALSE
)
write.csv(csv_df, csv_main_path, row.names = FALSE)
cat(sprintf("  [SAVED] %s\n\n", csv_main_path))


## =========================================================================
## TABLE B: SM random-effect SD comparison
## =========================================================================
cat("  5b. SM table: Random-effect SD comparison ...\n")

tex_sm <- character()

tex_sm <- c(tex_sm,
  "\\begin{table}[t]",
  "\\centering",
  paste0("\\caption{Random-effect standard deviations ($\\tau$): ",
         "unweighted (M3b) versus survey-weighted (M3b-W)."),
  paste0("  ``Rel.~$\\Delta$'' is the relative change ",
         "$(\\tau_{\\mathrm{WT}} - \\tau_{\\mathrm{UW}}) / ",
         "\\tau_{\\mathrm{UW}}$."),
  paste0("  Survey weighting substantially inflates extensive-margin ",
         "heterogeneity,"),
  paste0("  while intensive-margin effects are more ",
         "moderate.}\\label{tab:tau-comparison}"),
  "\\smallskip",
  "\\small",
  "\\begin{tabular}{@{}l rrr r@{}}",
  "\\toprule",
  "Parameter & M3b & M3b-W & Shift & Rel.~$\\Delta$ (\\%) \\\\",
  "\\midrule"
)

## Extensive margin
tex_sm <- c(tex_sm,
  paste0("\\multicolumn{5}{@{}l}{\\textit{Extensive margin ",
         "($\\btau^{\\mathrm{ext}}$)}} \\\\[2pt]")
)
for (i in 1:5) {
  row <- sprintf("%-15s & %.3f & %.3f & $%s$ & $%s$ \\\\",
                 TAU_DISPLAY[i],
                 tau_uw[i], tau_wt[i],
                 fmt_signed(tau_shift[i], 3),
                 fmt_signed(100 * tau_rel[i], 1))
  if (i == 5) row <- sub("\\\\\\\\$", "\\\\\\\\[4pt]", row)
  tex_sm <- c(tex_sm, row)
}

## Intensive margin
tex_sm <- c(tex_sm,
  paste0("\\multicolumn{5}{@{}l}{\\textit{Intensive margin ",
         "($\\btau^{\\mathrm{int}}$)}} \\\\[2pt]")
)
for (i in 6:10) {
  row <- sprintf("%-15s & %.3f & %.3f & $%s$ & $%s$ \\\\",
                 TAU_DISPLAY[i],
                 tau_uw[i], tau_wt[i],
                 fmt_signed(tau_shift[i], 3),
                 fmt_signed(100 * tau_rel[i], 1))
  tex_sm <- c(tex_sm, row)
}

## Footer
tex_sm <- c(tex_sm,
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}"
)

tex_sm_path <- file.path(FIGURE_DIR, "ST_tau_comparison.tex")
writeLines(tex_sm, tex_sm_path)
cat(sprintf("  [SAVED] %s\n", tex_sm_path))

## CSV backup
csv_sm_path <- file.path(FIGURE_DIR, "ST_tau_comparison.csv")
csv_tau <- data.frame(
  Margin      = rep(c("Extensive", "Intensive"), each = 5),
  Covariate   = TAU_DISPLAY,
  M3b         = sprintf("%.4f", tau_uw),
  M3b_W       = sprintf("%.4f", tau_wt),
  Shift       = sprintf("%+.4f", tau_shift),
  Rel_Change  = sprintf("%+.1f%%", 100 * tau_rel),
  stringsAsFactors = FALSE
)
write.csv(csv_tau, csv_sm_path, row.names = FALSE)
cat(sprintf("  [SAVED] %s\n\n", csv_sm_path))


###############################################################################
## SECTION 6 : PART E -- SUMMARY STATISTICS
###############################################################################
cat("--- 6. Part E: Summary statistics ---\n\n")

## 6a. Fixed-effect summary
max_abs_shift   <- max(abs(shift))
max_shift_param <- PARAM_LABELS[which.max(abs(shift))]
med_abs_shift   <- median(abs(shift))
max_z_shift     <- max(abs(z_shift))
max_z_param     <- PARAM_LABELS[which.max(abs(z_shift))]
med_z_shift     <- median(abs(z_shift))

cat("  FIXED-EFFECT SHIFTS:\n")
cat(sprintf("    Max |shift|:       %.4f  (%s)\n",
            max_abs_shift, max_shift_param))
cat(sprintf("    Median |shift|:    %.4f\n", med_abs_shift))
cat(sprintf("    Max |shift/SE|:    %.2f SEs  (%s)\n",
            max_z_shift, max_z_param))
cat(sprintf("    Median |shift/SE|: %.2f SEs\n", med_z_shift))
cat(sprintf("    Sign changes:      %d / %d\n", n_sign_change, D))
if (n_sign_change > 0) {
  cat(sprintf("      Changed: %s\n",
              paste(PARAM_LABELS[!sign_preserved], collapse = ", ")))
}

## 6b. Informal joint shift summary
sum_z2 <- sum(z_shift^2)

cat(sprintf("\n  INFORMAL JOINT SUMMARY:\n"))
cat(sprintf("    Sum of (shift/SE)^2: %.2f  (%d parameters)\n", sum_z2, D))
cat("    NOTE: Descriptive only. This is NOT a formal Hausman test because\n")
cat("    the sandwich SE is for M3b-W alone, not for the difference.\n")

## 6c. Kappa comparison
kappa_uw <- res_m3b$kappa_mean
kappa_wt <- res_m3bw$kappa_mean
kappa_shift <- kappa_wt - kappa_uw
kappa_rel   <- kappa_shift / kappa_uw

cat(sprintf("\n  OVERDISPERSION (kappa):\n"))
cat(sprintf("    M3b:   kappa = %.2f  (log_kappa = %.3f)\n",
            kappa_uw, mean_uw[11]))
cat(sprintf("    M3b-W: kappa = %.2f  (log_kappa = %.3f)\n",
            kappa_wt, mean_wt[11]))
cat(sprintf("    Shift: %+.2f (%+.1f%%)\n", kappa_shift, 100 * kappa_rel))

## 6d. Tau summary (reprise)
cat(sprintf("\n  RANDOM-EFFECT SD SHIFTS:\n"))
cat(sprintf("    Extensive median rel. change: %+.0f%%\n",
            100 * median_rel_ext))
cat(sprintf("    Intensive median rel. change: %+.0f%%\n",
            100 * median_rel_int))
cat(sprintf("    Overall median rel. change:   %+.0f%%\n",
            100 * median_rel_all))

## 6e. Interpretation
cat("\n  INTERPRETATION:\n")
cat("    Weighting shifts all fixed-effect posterior means but preserves\n")
cat("    all coefficient signs, including the poverty reversal pattern.\n")
cat(sprintf("    The shifts range from %.3f to %.3f on the logit scale\n",
            min(abs(shift)), max(abs(shift))))
cat(sprintf("    (%.1f--%.1f sandwich SEs). ",
            min(abs(z_shift)), max(abs(z_shift))))
cat("Random-effect SDs generally\n")
cat("    increase under weighting, particularly for the extensive margin\n")
cat(sprintf("    (median: %+.0f%%), ",
            100 * median_rel_ext))
cat("reflecting greater between-state heterogeneity\n")
cat("    once survey design effects are accounted for.\n")

cat("\n")


###############################################################################
## SECTION 7 : VERIFICATION
###############################################################################
cat("--- 7. Verification checks ---\n\n")

checks_passed <- 0
checks_total  <- 0

## Check 1: Parameter counts
checks_total <- checks_total + 1
if (length(mean_uw) == D && length(mean_wt) == D) {
  checks_passed <- checks_passed + 1
  cat("  [PASS] Parameter count: 11 fixed effects in both models.\n")
} else {
  cat("  [FAIL] Parameter count mismatch!\n")
}

## Check 2: M3b-W means match cholesky_correction table
checks_total <- checks_total + 1
max_diff <- max(abs(mean_wt - ct$post_mean))
if (max_diff < 1e-6) {
  checks_passed <- checks_passed + 1
  cat(sprintf("  [PASS] M3b-W means match cholesky_correction (max diff = %.1e).\n",
              max_diff))
} else {
  cat(sprintf("  [FAIL] M3b-W means mismatch (max diff = %.6f).\n", max_diff))
}

## Check 3: Poverty reversal preserved
checks_total <- checks_total + 1
if (reversal_uw && reversal_wt) {
  checks_passed <- checks_passed + 1
  cat("  [PASS] Poverty reversal preserved in both specifications.\n")
} else {
  cat("  [FAIL] Poverty reversal NOT preserved!\n")
}

## Check 4: Tau vectors have correct length
checks_total <- checks_total + 1
if (length(tau_uw) == 10 && length(tau_wt) == 10) {
  checks_passed <- checks_passed + 1
  cat("  [PASS] Tau vectors: 10 elements each.\n")
} else {
  cat("  [FAIL] Tau vector length mismatch!\n")
}

## Check 5: All output files written
checks_total <- checks_total + 1
all_exist <- all(file.exists(tex_main_path, csv_main_path,
                             tex_sm_path, csv_sm_path))
if (all_exist) {
  checks_passed <- checks_passed + 1
  cat("  [PASS] All 4 output files written successfully.\n")
} else {
  cat("  [FAIL] Some output files missing!\n")
}

## Check 6: LaTeX table compiles (basic syntax check)
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

cat(sprintf("\n  Verification: %d / %d checks passed.\n\n",
            checks_passed, checks_total))


###############################################################################
## SECTION 8 : SAVE RESULTS
###############################################################################
cat("--- 8. Saving results ---\n\n")

B2_results <- list(
  ## Description
  description = paste(
    "M3b vs M3b-W comparison table.",
    "Side-by-side comparison of unweighted and survey-weighted posterior",
    "means for fixed effects and random-effect SDs, with shift/SE",
    "normalization and poverty reversal robustness check."
  ),

  ## Fixed-effect comparison (main output)
  fixed_comparison = fixed_comparison,

  ## Tau comparison
  tau_comparison = tau_comparison,

  ## Poverty reversal
  poverty_reversal = list(
    alpha_pov_uw    = as.numeric(alpha_pov_uw),
    beta_pov_uw     = as.numeric(beta_pov_uw),
    alpha_pov_wt    = as.numeric(alpha_pov_wt),
    beta_pov_wt     = as.numeric(beta_pov_wt),
    reversal_uw     = reversal_uw,
    reversal_wt     = reversal_wt,
    both_hold       = reversal_uw && reversal_wt,
    alpha_pov_wald_sig = as.logical(alpha_pov_wald_sig),
    beta_pov_wald_sig  = as.logical(beta_pov_wald_sig)
  ),

  ## Summary statistics
  summary = list(
    max_abs_shift   = max_abs_shift,
    max_shift_param = max_shift_param,
    med_abs_shift   = med_abs_shift,
    max_z_shift     = max_z_shift,
    max_z_param     = max_z_param,
    med_z_shift     = med_z_shift,
    n_sign_changes  = n_sign_change,
    sign_changed_params = if (n_sign_change > 0) PARAM_LABELS[!sign_preserved] else character(0),
    sum_z_squared   = sum_z2,
    kappa_uw        = kappa_uw,
    kappa_wt        = kappa_wt,
    kappa_rel_change = kappa_rel,
    median_rel_change_ext = median_rel_ext,
    median_rel_change_int = median_rel_int,
    median_rel_change_all = median_rel_all,
    max_abs_rel_change_tau = max(abs(tau_rel)),
    max_abs_rel_change_tau_param = TAU_LABELS[which.max(abs(tau_rel))],
    n_tau_increase = n_tau_increase,
    n_tau_decrease = n_tau_decrease
  ),

  ## Output file paths
  output_files = list(
    main_tex = tex_main_path,
    main_csv = csv_main_path,
    sm_tex   = tex_sm_path,
    sm_csv   = csv_sm_path,
    rds      = B2_OUT
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

saveRDS(B2_results, B2_OUT)
cat(sprintf("  Saved: %s\n", B2_OUT))
cat(sprintf("  File size: %.1f KB\n\n",
            file.info(B2_OUT)$size / 1024))


###############################################################################
## SECTION 9 : FINAL SUMMARY
###############################################################################
cat("==============================================================\n")
cat("  M3b vs M3b-W Comparison COMPLETE\n")
cat("==============================================================\n\n")

cat("  FIXED EFFECTS (11 parameters):\n")
cat(sprintf("    Max |shift|:       %.4f (%s)\n", max_abs_shift, max_shift_param))
cat(sprintf("    Max |shift/SE|:    %.2f (%s)\n", max_z_shift, max_z_param))
cat(sprintf("    Median |shift/SE|: %.2f\n", med_z_shift))
cat(sprintf("    Sign changes:      %d / %d\n", n_sign_change, D))

cat(sprintf("\n  POVERTY REVERSAL:\n"))
cat(sprintf("    M3b:   alpha_pov = %+.3f, beta_pov = %+.3f -> %s\n",
            alpha_pov_uw, beta_pov_uw,
            ifelse(reversal_uw, "REVERSAL HOLDS", "NO REVERSAL")))
cat(sprintf("    M3b-W: alpha_pov = %+.3f, beta_pov = %+.3f -> %s\n",
            alpha_pov_wt, beta_pov_wt,
            ifelse(reversal_wt, "REVERSAL HOLDS", "NO REVERSAL")))

cat(sprintf("\n  RANDOM-EFFECT SDs (10 tau parameters):\n"))
cat(sprintf("    Extensive median rel. change: %+.0f%%\n",
            100 * median_rel_ext))
cat(sprintf("    Intensive median rel. change: %+.0f%%\n",
            100 * median_rel_int))

cat(sprintf("\n  OVERDISPERSION:\n"))
cat(sprintf("    kappa: %.2f (UW) -> %.2f (WT), %+.1f%%\n",
            kappa_uw, kappa_wt, 100 * kappa_rel))

cat(sprintf("\n  OUTPUT FILES:\n"))
cat(sprintf("    Main-text table: %s\n", tex_main_path))
cat(sprintf("    SM table:        %s\n", tex_sm_path))
cat(sprintf("    Results RDS:     %s\n", B2_OUT))

cat(sprintf("\n  VERIFICATION: %d / %d checks passed.\n",
            checks_passed, checks_total))

cat("\n==============================================================\n")
cat("  DONE.\n")
cat("==============================================================\n")
