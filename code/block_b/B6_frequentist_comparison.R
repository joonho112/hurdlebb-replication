## =============================================================================
## B6_frequentist_comparison.R -- Frequentist Contextualization
## =============================================================================
## Purpose : Compare survey-weighted frequentist (svyglm) and Bayesian HBB
##           estimates to validate design-consistency and contextualize the
##           Bayesian approach's value-added.
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
## =============================================================================
##
## Output:
##   output/tables/ST_B6_frequentist.tex
##   output/tables/ST_B6_frequentist.csv
##   data/precomputed/B6_frequentist_comparison.rds
## =============================================================================

library(survey)
library(dplyr)
library(xtable)

cat("=== Frequentist Contextualization ===\n\n")

## ---- Paths ----
PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
OUTPUT_DIR   <- file.path(PROJECT_ROOT, "data/precomputed")
FIG_DIR      <- file.path(PROJECT_ROOT, "output/tables")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

## ---- Load data ----
d  <- readRDS(file.path(OUTPUT_DIR, "analysis_data.rds"))
sv <- readRDS(file.path(OUTPUT_DIR, "sandwich_variance.rds"))
r_m3bw <- readRDS(file.path(OUTPUT_DIR, "results_m3b_weighted.rds"))
r_m3b  <- readRDS(file.path(OUTPUT_DIR, "results_m3b.rds"))

cat("Data loaded: N =", nrow(d), "\n")

## ---- Prepare variables ----
d <- d %>%
  mutate(
    serves_it  = z,
    it_share   = ifelse(n_trial > 0 & y > 0, y / n_trial, NA_real_),
    pct_poverty  = comm_pct_poverty_num_std,
    pct_urban    = pct_urban_std,
    pct_black    = comm_pct_black_num_std,
    pct_hispanic = comm_pct_hisp_num_std
  )

servers <- d %>% filter(serves_it == 1 & !is.na(it_share))
cat("Full sample: N =", nrow(d), "\n")
cat("Servers (IT > 0): N =", nrow(servers), "\n\n")

## ==========================================================================
## 1. Frequentist models
## ==========================================================================
cat("--- 1. Fitting frequentist models ---\n\n")

## 1a. Create survey design object
svy_full <- svydesign(
  ids     = ~vpsu,
  strata  = ~vstratum,
  weights = ~weight,
  data    = d,
  nest    = TRUE
)
svy_servers <- subset(svy_full, serves_it == 1 & !is.na(it_share))

## 1b. Unweighted models
m_uw_ext <- glm(serves_it ~ pct_poverty + pct_urban + pct_black + pct_hispanic,
                data = d, family = binomial)
m_uw_int <- lm(it_share ~ pct_poverty + pct_urban + pct_black + pct_hispanic,
               data = servers)

## 1c. Survey-weighted models (cluster-robust SEs)
m_wt_ext <- svyglm(serves_it ~ pct_poverty + pct_urban + pct_black + pct_hispanic,
                    design = svy_full, family = quasibinomial())
m_wt_int <- svyglm(it_share ~ pct_poverty + pct_urban + pct_black + pct_hispanic,
                    design = svy_servers)

cat("  Frequentist models fitted.\n\n")

## ==========================================================================
## 2. Extract and compare estimates
## ==========================================================================
cat("--- 2. Extracting estimates for comparison ---\n\n")

covars <- c("(Intercept)", "pct_poverty", "pct_urban", "pct_black", "pct_hispanic")
covar_labels <- c("Intercept", "Poverty", "Urban", "Black", "Hispanic")

## Helper: extract coefficient table
extract_coefs <- function(model, covars) {
  s <- summary(model)$coefficients
  data.frame(
    est = s[covars, "Estimate"],
    se  = s[covars, "Std. Error"],
    stringsAsFactors = FALSE
  )
}

## 2a. Frequentist: unweighted
freq_uw_ext <- extract_coefs(m_uw_ext, covars)
freq_uw_int <- extract_coefs(m_uw_int, covars)

## 2b. Frequentist: survey-weighted (cluster-robust)
freq_wt_ext <- extract_coefs(m_wt_ext, covars)
freq_wt_int <- extract_coefs(m_wt_int, covars)

## 2c. Bayesian HBB: pseudo-posterior means + sandwich SDs
## alpha = extensive, beta = intensive
hbb_alpha_means <- r_m3bw$alpha_means  # named vector
hbb_beta_means  <- r_m3bw$beta_means
hbb_sand_sd     <- sv$der_table$V_sand_SD  # 11 elements: 5 alpha + 5 beta + log_kappa
hbb_mcmc_sd     <- sv$der_table$Sigma_MCMC_SD
der_values      <- sv$DER

## ==========================================================================
## 3. Build comparison table
## ==========================================================================
cat("--- 3. Building comparison table ---\n\n")

## Extensive margin comparison
comp_ext <- data.frame(
  Margin    = "Extensive",
  Parameter = covar_labels,
  svyglm_est = freq_wt_ext$est,
  svyglm_se  = freq_wt_ext$se,
  HBB_est    = as.numeric(hbb_alpha_means),
  HBB_sand_se = hbb_sand_sd[1:5],
  DER        = der_values[1:5],
  stringsAsFactors = FALSE
)

## Intensive margin comparison
comp_int <- data.frame(
  Margin    = "Intensive",
  Parameter = covar_labels,
  svyglm_est = freq_wt_int$est,
  svyglm_se  = freq_wt_int$se,
  HBB_est    = as.numeric(hbb_beta_means),
  HBB_sand_se = hbb_sand_sd[6:10],
  DER        = der_values[6:10],
  stringsAsFactors = FALSE
)

comp <- rbind(comp_ext, comp_int)
comp$diff_est <- comp$HBB_est - comp$svyglm_est
comp$ratio_se <- comp$HBB_sand_se / comp$svyglm_se

## Print summary
cat("  Comparison table:\n\n")
print(comp %>%
        mutate(across(where(is.numeric), ~ round(., 4))) %>%
        select(Margin, Parameter, svyglm_est, HBB_est, diff_est,
               svyglm_se, HBB_sand_se, ratio_se, DER),
      row.names = FALSE)

## ==========================================================================
## 4. Structural limitations summary
## ==========================================================================
cat("\n--- 4. Structural limitations of svyglm ---\n\n")

limitations <- data.frame(
  Feature = c(
    "Bounded count distribution (beta-binomial)",
    "Zero-inflation / hurdle structure",
    "State-level random effects",
    "Cross-margin correlation",
    "Survey-weight correction for all above",
    "Overdispersion parameter (kappa)"
  ),
  svyglm = c("No", "No", "No", "No", "Partial", "No"),
  HBB    = c("Yes", "Yes", "Yes", "Yes", "Yes", "Yes"),
  stringsAsFactors = FALSE
)

cat("  Feature comparison:\n")
print(limitations, row.names = FALSE)

## ==========================================================================
## 5. Generate LaTeX table
## ==========================================================================
cat("\n--- 5. Generating output files ---\n\n")

## Format for LaTeX
fmt_comp <- comp %>%
  mutate(
    svyglm_cell = sprintf("%.3f (%.3f)", svyglm_est, svyglm_se),
    HBB_cell    = sprintf("%.3f (%.3f)", HBB_est, HBB_sand_se),
    diff_cell   = sprintf("%+.3f", diff_est),
    ratio_cell  = sprintf("%.2f", ratio_se),
    DER_cell    = sprintf("%.2f", DER)
  )

## LaTeX table (manual construction for better formatting)
tex_lines <- c(
  "% Generated by 86_B6_frequentist_comparison.R",
  "\\small",
  "\\begin{adjustbox}{max width=\\textwidth}",
  "\\begin{tabular}{@{}l l rr rr r r@{}}",
  "\\toprule",
  " & & \\multicolumn{2}{c}{svyglm (design-based)} &",
  "   \\multicolumn{2}{c}{HBB (sandwich-corrected)} & & \\\\",
  "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
  "Margin & Parameter & Est. & SE & Est. & SE & $\\Delta$Est. & SE Ratio \\\\",
  "\\midrule"
)

for (i in seq_len(nrow(fmt_comp))) {
  row <- fmt_comp[i, ]
  margin_str <- ifelse(i == 1 | (i == 6), row$Margin, "")
  tex_lines <- c(tex_lines, sprintf(
    "%s & %s & %.3f & %.3f & %.3f & %.3f & %+.3f & %.2f \\\\",
    margin_str, row$Parameter,
    row$svyglm_est, row$svyglm_se,
    row$HBB_est, row$HBB_sand_se,
    row$diff_est, row$ratio_se
  ))
  ## Add spacing between margins
  if (i == 5) tex_lines <- c(tex_lines, "[3pt]")
}

tex_lines <- c(tex_lines,
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{adjustbox}"
)

## Write LaTeX
tex_file <- file.path(FIG_DIR, "ST_B6_frequentist.tex")
writeLines(tex_lines, tex_file)
cat("  LaTeX table written:", tex_file, "\n")

## Write CSV
csv_file <- file.path(FIG_DIR, "ST_B6_frequentist.csv")
write.csv(comp, csv_file, row.names = FALSE)
cat("  CSV written:", csv_file, "\n")

## ==========================================================================
## 6. Save full results
## ==========================================================================
results <- list(
  comparison   = comp,
  limitations  = limitations,
  freq_uw_ext  = freq_uw_ext,
  freq_uw_int  = freq_uw_int,
  freq_wt_ext  = freq_wt_ext,
  freq_wt_int  = freq_wt_int,
  hbb_alpha    = hbb_alpha_means,
  hbb_beta     = hbb_beta_means,
  hbb_sand_sd  = hbb_sand_sd,
  der          = der_values,
  N_full       = nrow(d),
  N_servers    = nrow(servers),
  timestamp    = Sys.time()
)

rds_file <- file.path(OUTPUT_DIR, "B6_frequentist_comparison.rds")
saveRDS(results, rds_file)
cat("  Full results saved:", rds_file, "\n")

## ==========================================================================
## 7. Key findings
## ==========================================================================
cat("\n=== Key Findings ===\n\n")

cat("1. Point estimate concordance:\n")
cat("   Max |Delta| for extensive:", round(max(abs(comp$diff_est[1:5])), 4), "\n")
cat("   Max |Delta| for intensive:", round(max(abs(comp$diff_est[6:10])), 4), "\n")
cat("   Mean |Delta| overall:", round(mean(abs(comp$diff_est)), 4), "\n\n")

cat("2. SE ratio (HBB sandwich / svyglm):\n")
cat("   Extensive range:", round(min(comp$ratio_se[1:5]), 2),
    "--", round(max(comp$ratio_se[1:5]), 2), "\n")
cat("   Intensive range:", round(min(comp$ratio_se[6:10]), 2),
    "--", round(max(comp$ratio_se[6:10]), 2), "\n\n")

cat("3. DER range:", round(min(comp$DER), 2), "--", round(max(comp$DER), 2), "\n\n")

cat("DONE.\n")
