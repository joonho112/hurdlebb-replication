## =============================================================================
## 06_tables_figures.R -- Generate All Publication Tables and Figures
## =============================================================================
## Purpose : Reproduce every table and figure in the main text and supplementary
##           materials from pre-computed posterior summaries. This is the core
##           script for Track B (partial replication without restricted data).
## Author  : JoonHo Lee (jlee296@ua.edu)
## License : MIT
##
## Part of the replication package for:
##   "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded
##    Counts and Its Application to Childcare Enrollment"
##
## Track   : B (uses pre-computed results in data/precomputed/)
## Inputs  : data/precomputed/*.rds (model summaries, marginal effects, etc.)
## Outputs : output/tables/*.{csv,tex} and output/figures/*.{pdf,png}
##
## Usage:
##   source("code/00_setup.R")   # install packages, set PROJECT_ROOT
##   source("code/06_tables_figures.R")
## =============================================================================

cat("==============================================================\n")
cat("  Replication Package: Publication Tables & Figures\n")
cat("==============================================================\n\n")


###############################################################################
## SECTION 0 : SETUP
###############################################################################

PROJECT_ROOT <- if (requireNamespace("here", quietly = TRUE)) {
  here::here()
} else {
  ## Fallback: assume working directory is the repository root
  getwd()
}

DATA_DIR   <- file.path(PROJECT_ROOT, "data/precomputed")
TABLE_DIR  <- file.path(PROJECT_ROOT, "output/tables")
FIGURE_DIR <- file.path(PROJECT_ROOT, "output/figures")

## Create output directories if they don't exist
dir.create(file.path(PROJECT_ROOT, "output/tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(PROJECT_ROOT, "output/figures"), recursive = TRUE, showWarnings = FALSE)

## Packages
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(xtable)
  library(scales)
  library(viridis)
  library(maps)
  library(ggrepel)
  library(forcats)
})

## Source shared helpers
source(file.path(PROJECT_ROOT, "code/helpers/utils.R"))
source(file.path(PROJECT_ROOT, "code/helpers/theme_manuscript.R"))

## ---------- Helper: JRSS manuscript theme ----------
## Upgraded to align with hurdlebb vignette aesthetics while retaining
## theme_bw() box borders appropriate for print journals.
theme_manuscript <- function(base_size = 10) {
  theme_bw(base_size = base_size) +
    theme(
      ## Grid: lighter major lines (vignette style), no minor
      panel.grid.minor  = element_blank(),
      panel.grid.major  = element_line(colour = "grey92", linewidth = 0.3),
      ## Strip: borderless facet labels
      strip.background  = element_rect(fill = "grey96", colour = NA),
      strip.text        = element_text(face = "bold", size = base_size),
      ## Axes: softer text, thin ticks
      axis.title        = element_text(size = base_size),
      axis.text         = element_text(size = base_size - 1, colour = "grey25"),
      axis.ticks        = element_line(colour = "grey70", linewidth = 0.3),
      ## Titles
      plot.title        = element_text(face = "bold", size = base_size + 1, hjust = 0),
      plot.subtitle     = element_text(size = base_size - 1, hjust = 0, colour = "grey40"),
      ## Legend
      legend.position   = "bottom",
      legend.text       = element_text(size = base_size - 1),
      legend.title      = element_text(size = base_size, face = "bold"),
      legend.key.size   = unit(0.8, "lines"),
      ## Margins
      plot.margin       = margin(5, 10, 5, 5)
    )
}

## ---------- Shared aesthetic constants ----------

## Margin color palette (same across all 5 vignettes)
MARGIN_COLORS <- c(Extensive  = "#4393C3",   # blue
                   Intensive  = "#D6604D",   # red
                   Reference  = "#1B7837",   # green
                   Dispersion = "#762A83")   # purple

## Reversal pattern palette (F3, F4, F5)
REVERSAL_COLORS <- c("Classic Reversal"  = "#D6604D",
                     "Both Positive"     = "#4393C3",
                     "Both Negative"     = "#762A83",
                     "Opposite Reversal" = "#1B7837")

## Significance shape encoding
SHAPE_SIG    <- 16   # filled circle: CI excludes zero
SHAPE_NONSIG <-  1   # open circle: CI includes zero

## Small-state threshold
SMALL_N_THRESHOLD <- 40

## geom_pointrange() parameter presets
## ggplot2 4.0+: `size` controls point, `linewidth` controls line
POINTRANGE_DENSE <- list(linewidth = 0.35, size = 0.45)   # 51-state caterpillar
POINTRANGE_STD   <- list(linewidth = 0.7,  size = 1.25)   # standard coef plots

## ---------- Helper: save figure ----------
save_figure <- function(plot, name, width = 7, height = 5) {
  pdf_path <- file.path(FIGURE_DIR, paste0(name, ".pdf"))
  png_path <- file.path(FIGURE_DIR, paste0(name, ".png"))
  ## Use standard pdf device (cairo may not be available on all systems)
  ggsave(pdf_path, plot, width = width, height = height, device = "pdf")
  ggsave(png_path, plot, width = width, height = height, dpi = 300)
  cat(sprintf("  [SAVED] %s.pdf / .png (%g x %g in)\n", name, width, height))
}

## ---------- Helper: save table ----------
save_table <- function(df, name, caption = "", ...) {
  csv_path <- file.path(TABLE_DIR, paste0(name, ".csv"))
  tex_path <- file.path(TABLE_DIR, paste0(name, ".tex"))
  write.csv(df, csv_path, row.names = FALSE)
  xt <- xtable(df, caption = caption, ...)
  print(xt, file = tex_path, include.rownames = FALSE,
        booktabs = TRUE, sanitize.text.function = identity,
        floating = TRUE, table.placement = "htbp")
  cat(sprintf("  [SAVED] %s.csv / .tex\n", name))
}


###############################################################################
## SECTION 1 : LOAD ALL DATA
###############################################################################

cat("--- Section 1: Loading pre-computed analysis results ---\n")

analysis_data   <- readRDS(file.path(DATA_DIR, "analysis_data.rds"))
stan_data       <- readRDS(file.path(DATA_DIR, "stan_data.rds"))
std_params      <- readRDS(file.path(DATA_DIR, "standardization_params.rds"))
loo_comp        <- readRDS(file.path(DATA_DIR, "loo_comparison.rds"))
chol_corr       <- readRDS(file.path(DATA_DIR, "cholesky_correction.rds"))
sand_var        <- readRDS(file.path(DATA_DIR, "sandwich_variance.rds"))
res_m3b         <- readRDS(file.path(DATA_DIR, "results_m3b.rds"))
res_m3b_wt      <- readRDS(file.path(DATA_DIR, "results_m3b_weighted.rds"))
marg_eff        <- readRDS(file.path(DATA_DIR, "marginal_effects.rds"))
ppc_comp        <- readRDS(file.path(DATA_DIR, "ppc_comparison.rds"))

## Also load results for convergence diagnostics (ST5)
res_m0  <- readRDS(file.path(DATA_DIR, "results_m0.rds"))
res_m1  <- readRDS(file.path(DATA_DIR, "results_m1.rds"))
res_m2  <- readRDS(file.path(DATA_DIR, "results_m2.rds"))
res_m3a <- readRDS(file.path(DATA_DIR, "results_m3a.rds"))

cat("  [OK] All 14 .rds files loaded.\n\n")


###############################################################################
## SECTION 2 : T1 -- DATA SUMMARY TABLE
## === Table 1: Data Summary Statistics ===
###############################################################################

cat("--- Section 2: T1 -- Data Summary ---\n")

ad <- analysis_data  # shorthand
N  <- nrow(ad)
N_pos <- sum(ad$z == 1)
N_zero <- sum(ad$z == 0)

## Panel A: Outcome
it_share_pos <- ad$y[ad$z == 1] / ad$n_trial[ad$z == 1]

panel_a <- data.frame(
  Variable = c("Total providers (N)", "IT servers (z=1)", "Non-servers (z=0)",
               "Zero rate (%)", "Total enrollment (n_i): Mean",
               "Total enrollment (n_i): Median", "Total enrollment (n_i): Range",
               "IT enrollment (y_i | z=1): Mean", "IT enrollment (y_i | z=1): Median",
               "IT share (y/n | z=1): Mean", "IT share (y/n | z=1): SD"),
  Value = c(
    format(N, big.mark = ","),
    format(N_pos, big.mark = ","),
    format(N_zero, big.mark = ","),
    sprintf("%.1f", 100 * N_zero / N),
    sprintf("%.1f", mean(ad$n_trial)),
    sprintf("%.0f", median(ad$n_trial)),
    sprintf("%d--%d", min(ad$n_trial), max(ad$n_trial)),
    sprintf("%.1f", mean(ad$y[ad$z == 1])),
    sprintf("%.0f", median(ad$y[ad$z == 1])),
    sprintf("%.3f", mean(it_share_pos)),
    sprintf("%.3f", sd(it_share_pos))
  ),
  stringsAsFactors = FALSE
)

## Panel B: Covariates (unstandardized)
cov_names  <- c("Community poverty rate (%)", "Urban (%)", "Community % Black",
                 "Community % Hispanic")
cov_vars   <- c("comm_pct_poverty_num", "pct_urban", "comm_pct_black_num",
                 "comm_pct_hisp_num")

panel_b_rows <- lapply(seq_along(cov_vars), function(i) {
  x <- ad[[cov_vars[i]]]
  data.frame(
    Variable = cov_names[i],
    Value = sprintf("%.1f (%.1f) [%.1f, %.1f]",
                    mean(x, na.rm = TRUE), sd(x, na.rm = TRUE),
                    min(x, na.rm = TRUE), max(x, na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
})
panel_b <- do.call(rbind, panel_b_rows)

## Panel C: Survey design
w <- ad$weight
kish_ess <- sum(w)^2 / sum(w^2)
deff_kish <- N / kish_ess
cv_w <- sd(w) / mean(w)

panel_c <- data.frame(
  Variable = c("States (S)", "Strata", "Primary Sampling Units (PSUs)",
               "Weight range", "Weight CV", "Kish ESS",
               "Kish DEFF"),
  Value = c(
    "51 (50 + DC)",
    as.character(length(unique(ad$vstratum))),
    as.character(length(unique(ad$vpsu))),
    sprintf("%.0f--%.0f", min(w), max(w)),
    sprintf("%.2f", cv_w),
    sprintf("%.0f", kish_ess),
    sprintf("%.2f", deff_kish)
  ),
  stringsAsFactors = FALSE
)

T1 <- rbind(
  data.frame(Variable = "Panel A: Outcome", Value = "", stringsAsFactors = FALSE),
  panel_a,
  data.frame(Variable = "Panel B: Covariates (Mean (SD) [Min, Max])", Value = "",
             stringsAsFactors = FALSE),
  panel_b,
  data.frame(Variable = "Panel C: Survey Design", Value = "", stringsAsFactors = FALSE),
  panel_c
)

save_table(T1, "T1_data_summary", caption = "Data Summary: NSECE 2019 Center-Based Providers")


###############################################################################
## SECTION 3 : T2 -- STATE POLICY VARIABLES
## === Table 2: State Subsidy Policy Variables ===
###############################################################################

cat("--- Section 3: T2 -- State Policy Variables ---\n")

sp <- std_params$state_policy

## Get state sample sizes from res_m2 (loaded in Section 1)
m2_spt <- res_m2$state_poverty_table
m2_spt$state_name_full <- std_params$state_levels[m2_spt$state]
sp$n_obs <- m2_spt$n_obs[match(sp$state_name, m2_spt$state_name_full)]

## Add dagger for small states
sp$state_label <- ifelse(
  !is.na(sp$n_obs) & sp$n_obs < SMALL_N_THRESHOLD,
  paste0(sp$state_name, "$^{\\dagger}$"),
  sp$state_name
)

T2 <- sp %>%
  select(state_label, MR_pctile, TieredReim, ITaddon) %>%
  arrange(state_label) %>%
  rename(
    State = state_label,
    `MR Percentile` = MR_pctile,
    `Tiered Reimbursement` = TieredReim,
    `IT Addon` = ITaddon
  )

save_table(T2, "T2_state_policy",
           caption = "State Child Care Subsidy Policy Variables (51 States). $^{\\dagger}$\\,$N < 40$; estimates subject to strong hierarchical shrinkage.")


###############################################################################
## SECTION 4 : T3 -- LOO-CV MODEL COMPARISON
## === Table 3: Leave-One-Out Cross-Validation Model Comparison ===
###############################################################################

cat("--- Section 4: T3 -- LOO-CV Model Comparison ---\n")

st <- loo_comp$summary_table

## Add pairwise significance
pw <- loo_comp$pairwise
sig_vec <- c("---", ifelse(abs(pw$elpd_diff / pw$se_diff) > 2, "Yes", "No"))

T3 <- data.frame(
  Model = c("M0: Pooled", "M1: Random Intercepts", "M2: Block-Diagonal SVC",
            "M3a: Cross-Margin Cov.", "M3b: Policy Moderators"),
  Params = st$n_params,
  `ELPD(loo)` = sprintf("%.1f", st$elpd_loo),
  SE = sprintf("%.1f", st$se_elpd),
  `p(loo)` = sprintf("%.1f", st$p_loo),
  `dELPD vs M0` = sprintf("%.1f", st$delta_elpd),
  `SE(dELPD)` = sprintf("%.1f", st$se_delta),
  Significant = sig_vec,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

save_table(T3, "T3_loo_comparison",
           caption = "LOO-CV Model Comparison (M0--M3b)")


###############################################################################
## SECTION 5 : T4 -- FIXED EFFECTS WITH SANDWICH CIS + DER
## === Table 4: Fixed Effect Estimates with Sandwich-Corrected CIs ===
###############################################################################

cat("--- Section 5: T4 -- Fixed Effects + Sandwich CIs ---\n")

ct <- chol_corr$comparison_table

param_labels <- c(
  "alpha (Intercept)", "alpha (Poverty)", "alpha (Urban)",
  "alpha (Black)", "alpha (Hispanic)",
  "beta (Intercept)", "beta (Poverty)", "beta (Urban)",
  "beta (Black)", "beta (Hispanic)", "log(kappa)"
)

T4 <- data.frame(
  Parameter = param_labels,
  `Post Mean` = sprintf("%.3f", ct$post_mean),
  `Naive 95% CI` = sprintf("[%.3f, %.3f]", ct$naive_lo, ct$naive_hi),
  `Wald 95% CI` = sprintf("[%.3f, %.3f]", ct$wald_lo, ct$wald_hi),
  `Wald Width` = sprintf("%.3f", ct$wald_width),
  DER = sprintf("%.2f", ct$DER),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

save_table(T4, "T4_fixed_effects_sandwich",
           caption = "Fixed Effect Estimates with Sandwich-Corrected Confidence Intervals")


###############################################################################
## SECTION 6 : T5 -- GAMMA ESTIMATES (POLICY MODERATION)
## === Table 5: Policy Moderation Gamma Parameters ===
###############################################################################

cat("--- Section 6: T5 -- Gamma Estimates ---\n")

gt_wt <- res_m3b_wt$gamma_table

## Key subset: poverty row interactions + significant elements
## Show all poverty-interacting gammas (rows 5,6,7,8 and 25,26,27,28)
## plus other significant ones (prob_pos > 0.975 or < 0.025)
key_idx <- c(
  which(gt_wt$covariate == "poverty" & gt_wt$col_label != "intercept"),
  which(gt_wt$prob_pos > 0.975 | gt_wt$prob_pos < 0.025)
)
key_idx <- sort(unique(key_idx))

gt_key <- gt_wt[key_idx, ]

T5 <- data.frame(
  Margin = gt_key$margin,
  Covariate = gt_key$covariate,
  Policy = gt_key$policy,
  Mean = sprintf("%.3f", gt_key$post_mean),
  SD = sprintf("%.3f", gt_key$post_sd),
  `95% CI` = sprintf("[%.3f, %.3f]", gt_key$q025, gt_key$q975),
  `Pr(>0)` = sprintf("%.3f", gt_key$prob_pos),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

save_table(T5, "T5_gamma_estimates",
           caption = "Policy Moderation Estimates (Key Gamma Parameters, Weighted M3b)")


###############################################################################
## SECTION 7 : T6 -- MARGINAL EFFECTS DECOMPOSITION
## === Table 6: Average Marginal Effects Decomposition ===
###############################################################################

cat("--- Section 7: T6 -- Marginal Effects ---\n")

dt <- marg_eff$decomp_table

T6 <- data.frame(
  Covariate = c("Poverty", "Urban", "Black", "Hispanic"),
  `Ext AME` = sprintf("%.4f", dt$ext_ame),
  `Ext 95% CI` = sprintf("[%.4f, %.4f]", dt$ext_ci_lo, dt$ext_ci_hi),
  `Int AME` = sprintf("%.4f", dt$int_ame),
  `Int 95% CI` = sprintf("[%.4f, %.4f]", dt$int_ci_lo, dt$int_ci_hi),
  `Total AME` = sprintf("%.4f", dt$total_ame),
  `Total 95% CI` = sprintf("[%.4f, %.4f]", dt$total_ci_lo, dt$total_ci_hi),
  `Ext Share %` = sprintf("%.1f", dt$ext_share),
  Pattern = dt$sign_pattern,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

save_table(T6, "T6_marginal_effects",
           caption = "Average Marginal Effects Decomposition (Sandwich-Corrected Weighted M3b)")


###############################################################################
## SECTION 8 : F1 -- IT ENROLLMENT DISTRIBUTION
## === Figure 1: Distribution of IT Enrollment Share ===
###############################################################################

cat("--- Section 8: F1 -- IT Enrollment Distribution ---\n")

## Compute IT share including zeros
it_share_all <- ad$y / ad$n_trial  # 0 for z=0

## Separate zero and positive
df_f1 <- data.frame(
  it_share = it_share_all,
  z = factor(ad$z, levels = c(0, 1), labels = c("Non-server (z=0)", "IT server (z=1)"))
)

## Zero spike + histogram of positive
zero_pct <- 100 * mean(ad$z == 0)
mean_pos <- mean(it_share_pos)

p_f1 <- ggplot() +
  ## Histogram of positive IT shares
  geom_histogram(
    data = filter(df_f1, z == "IT server (z=1)"),
    aes(x = it_share, y = after_stat(count) / nrow(df_f1)),
    bins = 40, fill = MARGIN_COLORS["Extensive"],
    colour = "white", alpha = 0.8
  ) +
  ## Zero spike
  geom_col(
    data = data.frame(x = 0, y = N_zero / N),
    aes(x = x, y = y),
    width = 0.015, fill = MARGIN_COLORS["Intensive"], alpha = 0.9
  ) +
  ## Zero annotation -- arrow pointing to spike
  annotate("segment", x = 0.12, xend = 0.025, y = N_zero / N - 0.01, yend = N_zero / N - 0.005,
           arrow = arrow(length = unit(0.15, "cm"), type = "closed"),
           colour = MARGIN_COLORS["Intensive"], linewidth = 0.4) +
  annotate("text", x = 0.125, y = N_zero / N - 0.012,
           label = sprintf("%.1f%% structural zeros", zero_pct),
           size = 3, colour = MARGIN_COLORS["Intensive"],
           fontface = "bold", hjust = 0) +
  ## Mean line + annotation
  geom_vline(xintercept = mean_pos, linetype = "dashed", color = "#2166AC",
             linewidth = 0.5, alpha = 0.7) +
  annotate("text", x = mean_pos + 0.02, y = 0.06,
           label = sprintf("Mean = %.1f%%", mean_pos * 100),
           size = 2.8, color = "#2166AC", fontface = "italic", hjust = 0) +
  scale_x_continuous(labels = percent_format(), limits = c(-0.02, 1.02),
                     breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(labels = percent_format(), expand = expansion(mult = c(0, 0.05))) +
  labs(x = "IT Enrollment Share  (y / n)", y = "Proportion of Providers",
       title = "Distribution of Infant/Toddler Enrollment Share",
       subtitle = sprintf("N = %s center-based providers across 51 states (NSECE 2019)",
                          format(N, big.mark = ","))) +
  theme_manuscript()

save_figure(p_f1, "F1_it_distribution", width = 7, height = 4.5)


###############################################################################
## SECTION 9 : F2 -- POVERTY REVERSAL IN RAW DATA
## === Figure 2: Poverty Reversal Pattern in Raw Data ===
###############################################################################

cat("--- Section 9: F2 -- Poverty Reversal Raw Data ---\n")

## Bin poverty into deciles
ad_f2 <- ad %>%
  mutate(
    pov_bin = cut(comm_pct_poverty_num, breaks = quantile(comm_pct_poverty_num,
                  probs = seq(0, 1, 0.1), na.rm = TRUE), include.lowest = TRUE),
    pov_mid = as.numeric(pov_bin),  # bin index
    it_share_pos = ifelse(z == 1, y / n_trial, NA)
  )

## Compute mean poverty for each bin
bin_stats <- ad_f2 %>%
  group_by(pov_bin) %>%
  summarise(
    pov_mean = mean(comm_pct_poverty_num, na.rm = TRUE),
    participation = mean(z),
    it_share_srv = mean(it_share_pos, na.rm = TRUE),
    n_obs = n(),
    .groups = "drop"
  ) %>%
  filter(!is.na(pov_bin))

## Panel (a): Participation rate -- zoom to data range
p_f2a <- ggplot(bin_stats, aes(x = pov_mean, y = participation)) +
  geom_smooth(method = "loess", se = TRUE, colour = MARGIN_COLORS["Intensive"],
              fill = MARGIN_COLORS["Intensive"], alpha = 0.12, linewidth = 0.9) +
  geom_point(size = 2.8, colour = MARGIN_COLORS["Intensive"]) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0.50, 0.80),
                     breaks = seq(0.50, 0.80, 0.05)) +
  scale_x_continuous(breaks = seq(5, 35, 5)) +
  labs(x = "Community Poverty Rate (%)",
       y = "IT Participation Rate",
       subtitle = "(a) Extensive margin") +
  theme_manuscript(base_size = 9) +
  theme(plot.subtitle = element_text(face = "bold", size = 9.5,
                                     color = "black", hjust = 0,
                                     margin = margin(b = 5)))

## Panel (b): IT share among servers
p_f2b <- ggplot(bin_stats, aes(x = pov_mean, y = it_share_srv)) +
  geom_smooth(method = "loess", se = TRUE, colour = MARGIN_COLORS["Extensive"],
              fill = MARGIN_COLORS["Extensive"], alpha = 0.12, linewidth = 0.9) +
  geom_point(size = 2.8, colour = MARGIN_COLORS["Extensive"]) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     breaks = seq(0.42, 0.56, 0.02)) +
  scale_x_continuous(breaks = seq(5, 35, 5)) +
  labs(x = "Community Poverty Rate (%)",
       y = "Mean IT Share (among servers)",
       subtitle = "(b) Intensive margin") +
  theme_manuscript(base_size = 9) +
  theme(plot.subtitle = element_text(face = "bold", size = 9.5,
                                     color = "black", hjust = 0,
                                     margin = margin(b = 5)))

p_f2 <- p_f2a + p_f2b +
  plot_annotation(
    title = "Poverty Reversal Pattern in Raw Data",
    subtitle = "Poverty decile bins; loess smoother with 95% confidence band",
    theme = theme(
      plot.title = element_text(face = "bold", size = 11, hjust = 0,
                                margin = margin(b = 2)),
      plot.subtitle = element_text(size = 9, hjust = 0, color = "grey40",
                                   margin = margin(b = 8))
    )
  )

save_figure(p_f2, "F2_poverty_reversal_raw", width = 7.5, height = 4)


###############################################################################
## SECTION 10 : F3 -- STATE POVERTY COEFFICIENTS (CATERPILLAR)
## === Figure 3: State-Level Poverty Coefficient Caterpillar Plot ===
###############################################################################

cat("--- Section 10: F3 -- State Poverty Caterpillar ---\n")

spt <- res_m3b$state_poverty_table
state_names <- std_params$state_levels

## Add state names
spt$state_name <- state_names[spt$state]

## Classify reversal type
spt$type <- case_when(
  spt$alpha_pov_mean < 0 & spt$beta_pov_mean > 0 ~ "Classic Reversal",
  spt$alpha_pov_mean > 0 & spt$beta_pov_mean > 0 ~ "Both Positive",
  spt$alpha_pov_mean < 0 & spt$beta_pov_mean < 0 ~ "Both Negative",
  TRUE ~ "Opposite Reversal"
)

## Abbreviate state names for plotting
spt$state_abbr <- state.abb[match(spt$state_name, state.name)]
spt$state_abbr[is.na(spt$state_abbr)] <- "DC"  # District of Columbia

## Small-state label suppression (N < SMALL_N_THRESHOLD)
spt$small_state <- spt$n_obs < SMALL_N_THRESHOLD
f3_y_labels <- setNames(
  ifelse(spt$small_state, "", as.character(spt$state_abbr)),
  spt$state_abbr
)
cat(sprintf("  Small states (N < %d): %d states, labels suppressed\n",
            SMALL_N_THRESHOLD, sum(spt$small_state)))

## Sort by alpha_pov for caterpillar
spt <- spt %>% arrange(alpha_pov_mean)
spt$state_abbr <- factor(spt$state_abbr, levels = spt$state_abbr)

## Significance encoding: CI excludes zero -> filled, spans zero -> open
spt$alpha_sig <- ifelse(
  (spt$alpha_pov_q05 > 0) | (spt$alpha_pov_q95 < 0),
  "CI excludes zero", "CI spans zero"
)
spt$beta_sig <- ifelse(
  (spt$beta_pov_q05 > 0) | (spt$beta_pov_q95 < 0),
  "CI excludes zero", "CI spans zero"
)

## Panel (a): Extensive margin (alpha_pov)
p_f3a <- ggplot(spt, aes(x = alpha_pov_mean, y = state_abbr, colour = type)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50",
             linewidth = 0.4) +
  geom_pointrange(aes(xmin = alpha_pov_q05, xmax = alpha_pov_q95,
                      shape = alpha_sig, alpha = !small_state),
                  linewidth = POINTRANGE_DENSE$linewidth,
                  size = POINTRANGE_DENSE$size) +
  scale_colour_manual(values = REVERSAL_COLORS) +
  scale_shape_manual(values = c("CI excludes zero" = SHAPE_SIG,
                                "CI spans zero"    = SHAPE_NONSIG),
                     name = NULL) +
  scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = 0.35), guide = "none") +
  scale_y_discrete(labels = f3_y_labels) +
  labs(x = expression(tilde(alpha)[poverty*","*s]), y = NULL,
       subtitle = expression("(a) Extensive: " * tilde(alpha)[poverty*","*s]),
       colour = "Pattern") +
  theme_manuscript(base_size = 8) +
  theme(legend.position = "none",
        panel.grid.major.y = element_blank(),
        axis.text.y = element_text(size = 5.5),
        plot.subtitle = element_text(face = "bold", size = 9,
                                     colour = "black", hjust = 0))

## Panel (b): Intensive margin (beta_pov) -- independently sorted
spt_b <- spt %>% arrange(beta_pov_mean)
spt_b$state_abbr <- factor(spt_b$state_abbr, levels = spt_b$state_abbr)

p_f3b <- ggplot(spt_b, aes(x = beta_pov_mean, y = state_abbr, colour = type)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50",
             linewidth = 0.4) +
  geom_pointrange(aes(xmin = beta_pov_q05, xmax = beta_pov_q95,
                      shape = beta_sig, alpha = !small_state),
                  linewidth = POINTRANGE_DENSE$linewidth,
                  size = POINTRANGE_DENSE$size) +
  scale_colour_manual(values = REVERSAL_COLORS) +
  scale_shape_manual(values = c("CI excludes zero" = SHAPE_SIG,
                                "CI spans zero"    = SHAPE_NONSIG),
                     name = NULL) +
  scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = 0.35), guide = "none") +
  scale_y_discrete(labels = f3_y_labels) +
  labs(x = expression(tilde(beta)[poverty*","*s]), y = NULL,
       subtitle = expression("(b) Intensive: " * tilde(beta)[poverty*","*s]),
       colour = "Pattern") +
  theme_manuscript(base_size = 8) +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(size = 5.5),
        plot.subtitle = element_text(face = "bold", size = 9,
                                     colour = "black", hjust = 0))

p_f3 <- p_f3a + p_f3b +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "State-Level Poverty Coefficients (M3b, Unweighted)",
    subtitle = expression(
      "90% CIs; filled = CI excludes zero; states with " *
      italic(N) * " < 40 shown faded (labels suppressed)"
    ),
    theme = theme(
      plot.title = element_text(face = "bold", size = 11, hjust = 0),
      plot.subtitle = element_text(size = 9, hjust = 0, colour = "grey40")
    )
  ) &
  theme(legend.position = "bottom",
        legend.text = element_text(size = 8))

save_figure(p_f3, "F3_state_poverty_caterpillar", width = 7.5, height = 8.5)


###############################################################################
## SECTION 11 : F4 -- REVERSAL PROBABILITY MAP
## === Figure 4: Posterior Probability of Classic Poverty Reversal by State ===
###############################################################################

cat("--- Section 11: F4 -- Reversal Probability Map ---\n")

## Use M2 state_poverty_table which has meaningful prob_reversal values
## (M3b unweighted has near-zero prob_reversal due to prior domination)
spt_m2 <- readRDS(file.path(DATA_DIR, "results_m2.rds"))$state_poverty_table
spt_m2$state_name <- tolower(state_names[spt_m2$state])

## Get US map data
us_map <- map_data("state")

## Merge
map_df <- us_map %>%
  left_join(
    spt_m2 %>% select(state_name, prob_reversal),
    by = c("region" = "state_name")
  )

## DC is not in maps::map_data("state") -- skip it

p_f4 <- ggplot(map_df, aes(x = long, y = lat, group = group)) +
  geom_polygon(aes(fill = prob_reversal), color = "white", linewidth = 0.25) +
  scale_fill_viridis(
    option = "plasma",
    limits = c(0, 1),
    labels = percent_format(),
    name = "Pr(Classic Reversal)",
    guide = guide_colorbar(barwidth = 12, barheight = 0.5,
                           title.position = "top", title.hjust = 0.5)
  ) +
  coord_map("polyconic") +
  labs(title = "Probability of Classic Poverty Reversal by State",
       subtitle = "Posterior probability of classic reversal pattern from M2 (block-diagonal SVC)") +
  theme_manuscript() +
  theme(
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    legend.position = "bottom",
    legend.direction = "horizontal"
  )

save_figure(p_f4, "F4_reversal_probability_map", width = 7, height = 4.5)


###############################################################################
## SECTION 12 : F5 -- CROSS-MARGIN SCATTER
## === Figure 5: Cross-Margin Poverty Coefficient Scatter ===
###############################################################################

cat("--- Section 12: F5 -- Cross-Margin Scatter ---\n")

## Use M2 which has well-identified state effects
spt_f5 <- spt_m2
spt_f5$state_name_orig <- state_names[spt_f5$state]
spt_f5$state_abbr <- state.abb[match(spt_f5$state_name_orig, state.name)]
spt_f5$state_abbr[is.na(spt_f5$state_abbr)] <- "DC"

spt_f5$type <- case_when(
  spt_f5$alpha_pov_mean < 0 & spt_f5$beta_pov_mean > 0 ~ "Classic Reversal",
  spt_f5$alpha_pov_mean > 0 & spt_f5$beta_pov_mean > 0 ~ "Both Positive",
  spt_f5$alpha_pov_mean < 0 & spt_f5$beta_pov_mean < 0 ~ "Both Negative",
  TRUE ~ "Opposite Reversal"
)

## Small-state flag and display label
spt_f5$small_state <- spt_f5$n_obs < SMALL_N_THRESHOLD
spt_f5$display_label <- ifelse(spt_f5$small_state, "", spt_f5$state_abbr)
spt_f5$pt_size <- ifelse(spt_f5$small_state, 1.5, 2.5)
cat(sprintf("  F5 small states (N < %d): %d states, labels suppressed\n",
            SMALL_N_THRESHOLD, sum(spt_f5$small_state)))

## Compute axis range for quadrant label placement
f5_x_range <- range(spt_f5$alpha_pov_mean)
f5_y_range <- range(spt_f5$beta_pov_mean)

p_f5 <- ggplot(spt_f5, aes(x = alpha_pov_mean, y = beta_pov_mean)) +
  ## Quadrant shading
  annotate("rect", xmin = -Inf, xmax = 0, ymin = 0, ymax = Inf,
           fill = "#D6604D", alpha = 0.06) +
  annotate("rect", xmin = 0, xmax = Inf, ymin = 0, ymax = Inf,
           fill = "#4393C3", alpha = 0.06) +
  annotate("rect", xmin = -Inf, xmax = 0, ymin = -Inf, ymax = 0,
           fill = "#762A83", alpha = 0.06) +
  annotate("rect", xmin = 0, xmax = Inf, ymin = -Inf, ymax = 0,
           fill = "#1B7837", alpha = 0.06) +
  ## Reference lines
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.4) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.4) +
  ## Points (smaller for small states) and labels (suppressed for small states)
  geom_point(aes(colour = type, size = pt_size)) +
  scale_size_identity() +
  geom_text_repel(aes(label = display_label), size = 2.2, max.overlaps = 25,
                  segment.size = 0.2, segment.colour = "grey60",
                  min.segment.length = 0.3) +
  scale_colour_manual(values = REVERSAL_COLORS) +
  ## Quadrant labels -- all four corners
  annotate("text", x = f5_x_range[1] * 0.85, y = f5_y_range[2] * 0.95,
           label = "Classic\nReversal",
           size = 2.8, colour = "#D6604D", fontface = "italic", hjust = 0) +
  annotate("text", x = f5_x_range[2] * 0.85, y = f5_y_range[2] * 0.95,
           label = "Both\nPositive",
           size = 2.8, colour = "#4393C3", fontface = "italic", hjust = 1) +
  annotate("text", x = f5_x_range[1] * 0.85, y = f5_y_range[1] * 0.5,
           label = "Both\nNegative",
           size = 2.8, colour = "#762A83", fontface = "italic", hjust = 0) +
  annotate("text", x = f5_x_range[2] * 0.85, y = f5_y_range[1] * 0.5,
           label = "Opposite\nReversal",
           size = 2.8, colour = "#1B7837", fontface = "italic", hjust = 1) +
  labs(x = expression(tilde(alpha)[poverty*","*s] ~ "(Extensive margin)"),
       y = expression(tilde(beta)[poverty*","*s] ~ "(Intensive margin)"),
       title = "Cross-Margin Poverty Coefficients by State",
       subtitle = expression(
         "M2: Block-diagonal SVC; small points: " *
         italic(N) * " < 40 (labels suppressed)"
       ),
       colour = "Pattern") +
  theme_manuscript() +
  theme(legend.position = "bottom",
        legend.key.width = unit(0.8, "cm"))

save_figure(p_f5, "F5_cross_margin_scatter", width = 6.5, height = 6)


###############################################################################
## SECTION 13 : F6 -- SANDWICH CORRECTION IMPACT
## === Figure 6: Impact of Sandwich Variance Correction ===
###############################################################################

cat("--- Section 13: F6 -- Sandwich Correction Impact ---\n")

ct <- chol_corr$comparison_table

## Parsed math labels for y-axis
param_labels_math <- c(
  "alpha[intercept]", "alpha[poverty]", "alpha[urban]",
  "alpha[Black]", "alpha[Hispanic]",
  "beta[intercept]", "beta[poverty]", "beta[urban]",
  "beta[Black]", "beta[Hispanic]", "log(kappa)"
)

## Prepare data for forest plot
df_f6 <- data.frame(
  param = factor(param_labels, levels = rev(param_labels)),
  param_math = factor(param_labels_math, levels = rev(param_labels_math)),
  mean = ct$post_mean,
  naive_lo = ct$naive_lo,
  naive_hi = ct$naive_hi,
  wald_lo = ct$wald_lo,
  wald_hi = ct$wald_hi,
  DER = ct$DER
)

## Manual y positions for dodged CI bars (naive offset up, wald offset down)
df_f6$y_num <- as.numeric(df_f6$param)

## Panel (a): Forest plot -- overlaid CIs (layered-segment pattern)
p_f6a <- ggplot(df_f6) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50",
             linewidth = 0.4) +
  ## Naive MCMC CI (thick, translucent grey background)
  geom_segment(aes(x = naive_lo, xend = naive_hi,
                   y = y_num, yend = y_num),
               colour = "grey55", linewidth = 2.5, alpha = 0.25,
               lineend = "round") +
  ## Wald sandwich CI (thin, opaque red in front)
  geom_segment(aes(x = wald_lo, xend = wald_hi,
                   y = y_num, yend = y_num),
               colour = "#D6604D", linewidth = 0.8, alpha = 0.85,
               lineend = "round") +
  ## Point estimates
  geom_point(aes(x = mean, y = y_num), size = 1.8, colour = "black") +
  scale_y_continuous(
    breaks = seq_along(param_labels),
    labels = rev(parse(text = param_labels_math)),
    expand = expansion(add = 0.6)
  ) +
  labs(x = "Estimate", y = NULL,
       subtitle = "(a) Naive MCMC (grey band) vs Sandwich-Corrected (red) 95% CIs") +
  theme_manuscript(base_size = 9) +
  theme(panel.grid.major.y = element_blank(),
        plot.subtitle = element_text(face = "bold", size = 9,
                                     colour = "black", hjust = 0))

## Panel (b): DER bar chart with reference line and value labels
## Use param (factor) on y-axis for proper discrete bar spacing
p_f6b <- ggplot(df_f6, aes(x = DER, y = param)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50",
             linewidth = 0.4) +
  geom_col(fill = "#4393C3", alpha = 0.7, width = 0.6) +
  geom_text(aes(label = sprintf("%.2f", DER)), hjust = -0.15, size = 2.5,
            color = "#2166AC", fontface = "bold") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.25)),
                     breaks = seq(0, 5, 1)) +
  scale_y_discrete(labels = rev(parse(text = param_labels_math))) +
  labs(x = "Design Effect Ratio", y = NULL,
       subtitle = "(b) DER") +
  theme_manuscript(base_size = 9) +
  theme(panel.grid.major.y = element_blank(),
        plot.subtitle = element_text(face = "bold", size = 9,
                                     color = "black", hjust = 0))

p_f6 <- p_f6a + p_f6b +
  plot_layout(widths = c(2.2, 1)) +
  plot_annotation(
    title = "Impact of Sandwich Variance Correction",
    subtitle = "Naive MCMC CIs reflect prior width; Wald sandwich CIs reflect data + survey design",
    theme = theme(
      plot.title = element_text(face = "bold", size = 11, hjust = 0),
      plot.subtitle = element_text(size = 9, hjust = 0, color = "grey40")
    )
  )

save_figure(p_f6, "F6_sandwich_correction", width = 7.5, height = 5.5)


###############################################################################
## SECTION 14 : SUPPLEMENTARY TABLES (ST2-ST5)
## === Supplementary Table 2: All 51 State Poverty Effects ===
## === Supplementary Table 3: Full Gamma Parameter Matrix ===
## === Supplementary Table 4: Design Effect Ratios ===
## === Supplementary Table 5: Convergence Diagnostics ===
###############################################################################

cat("--- Section 14: Supplementary Tables ---\n")

## ---- ST2: All 51 State Poverty Effects ----
spt_st2 <- spt_m2  # M2 has well-identified state effects
spt_st2$state_name <- state_names[spt_st2$state]

## Add dagger for small states
spt_st2$state_label <- ifelse(
  spt_st2$n_obs < SMALL_N_THRESHOLD,
  paste0(spt_st2$state_name, "$^{\\dagger}$"),
  spt_st2$state_name
)

ST2 <- spt_st2 %>%
  select(state_label, n_obs, alpha_pov_mean, alpha_pov_q05, alpha_pov_q95,
         beta_pov_mean, beta_pov_q05, beta_pov_q95, prob_reversal) %>%
  arrange(state_label) %>%
  mutate(across(where(is.numeric) & !c(n_obs), ~ sprintf("%.3f", .))) %>%
  rename(
    State = state_label, N = n_obs,
    `alpha_pov Mean` = alpha_pov_mean,
    `alpha_pov Q05` = alpha_pov_q05, `alpha_pov Q95` = alpha_pov_q95,
    `beta_pov Mean` = beta_pov_mean,
    `beta_pov Q05` = beta_pov_q05, `beta_pov Q95` = beta_pov_q95,
    `Pr(Reversal)` = prob_reversal
  )

save_table(ST2, "ST2_state_poverty_all",
           caption = "State-Level Poverty Coefficients (All 51 States, M2). $^{\\dagger}$\\,$N < 40$; estimates subject to strong hierarchical shrinkage.")

## ---- ST3: Full Gamma Matrices ----
gt_full <- res_m3b_wt$gamma_table

ST3 <- gt_full %>%
  select(margin, covariate, policy, post_mean, post_sd, q025, q975, prob_pos) %>%
  mutate(across(where(is.numeric), ~ sprintf("%.3f", .))) %>%
  rename(
    Margin = margin, Covariate = covariate, Policy = policy,
    Mean = post_mean, SD = post_sd, Q025 = q025, Q975 = q975, `Pr(>0)` = prob_pos
  )

save_table(ST3, "ST3_gamma_full",
           caption = "Full Policy Moderation Parameters (Weighted M3b, 40 Elements)")

## ---- ST4: Design Effect Ratios ----
dt_full <- sand_var$der_table

ST4 <- data.frame(
  Parameter = param_labels,
  `H_obs_inv` = sprintf("%.6f", dt_full$H_obs_inv_pp),
  `V_sand` = sprintf("%.6f", dt_full$V_sand_pp),
  `Sigma_MCMC` = sprintf("%.4f", dt_full$Sigma_MCMC_pp),
  DER = sprintf("%.2f", dt_full$DER),
  DER_MCMC = sprintf("%.4f", dt_full$DER_vs_MCMC),
  `SD(H_obs)` = sprintf("%.4f", dt_full$H_obs_inv_SD),
  `SD(Sand)` = sprintf("%.4f", dt_full$V_sand_SD),
  `SD(MCMC)` = sprintf("%.4f", dt_full$Sigma_MCMC_SD),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

save_table(ST4, "ST4_design_effect_ratios",
           caption = "Design Effect Ratios: Full Diagnostic Table")

## ---- ST5: Convergence Diagnostics ----
all_results <- list(M0 = res_m0, M1 = res_m1, M2 = res_m2, M3a = res_m3a, M3b = res_m3b)

conv_rows <- lapply(names(all_results), function(mn) {
  diag <- all_results[[mn]]$diagnostics
  data.frame(
    Model = mn,
    Max_Rhat = if (!is.null(diag$max_rhat)) sprintf("%.4f", diag$max_rhat) else "NA",
    N_Rhat_gt_1.01 = if (!is.null(diag$n_rhat_gt101)) as.character(diag$n_rhat_gt101) else "NA",
    Min_ESS_bulk = if (!is.null(diag$min_ess_bulk)) sprintf("%.0f", diag$min_ess_bulk) else "NA",
    Min_ESS_tail = if (!is.null(diag$min_ess_tail)) sprintf("%.0f", diag$min_ess_tail) else "NA",
    N_divergences = if (!is.null(diag$n_divergent)) as.character(diag$n_divergent) else "NA",
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
})

ST5 <- do.call(rbind, conv_rows)

save_table(ST5, "ST5_convergence_diagnostics",
           caption = "Convergence Diagnostics Across All Models")


###############################################################################
## SECTION 15 : SUPPLEMENTARY FIGURES (SF1, SF3, SF6)
## === Supplementary Figure 1: Weight Distribution ===
## === Supplementary Figure 3: Posterior Predictive Checks ===
## === Supplementary Figure 6: Sandwich Correction Diagnostic Details ===
###############################################################################

cat("--- Section 15: Supplementary Figures ---\n")

## ---- SF1: Weight Distribution ----
df_sf1 <- data.frame(w = analysis_data$w_tilde)

w_stats <- sprintf(
  "N = %s\nMean = %.2f\nSD = %.2f\nCV = %.2f\nRange = [%.2f, %.2f]\nKish ESS = %.0f\nDEFF = %.2f",
  format(N, big.mark = ","), mean(df_sf1$w), sd(df_sf1$w), sd(df_sf1$w)/mean(df_sf1$w),
  min(df_sf1$w), max(df_sf1$w), kish_ess, deff_kish
)

p_sf1 <- ggplot(df_sf1, aes(x = w)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 60, fill = MARGIN_COLORS["Extensive"],
                 colour = "white", alpha = 0.7) +
  geom_density(colour = "#2166AC", linewidth = 0.6, adjust = 1.5) +
  geom_vline(xintercept = 1, linetype = "dashed",
             colour = MARGIN_COLORS["Intensive"],
             linewidth = 0.6, alpha = 0.8) +
  annotate("label", x = max(df_sf1$w) * 0.55, y = Inf, label = w_stats,
           size = 2.8, vjust = 1.3, hjust = 0, family = "mono",
           fill = "white", alpha = 0.85,
           label.r = unit(0.15, "lines")) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.02))) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(x = expression("Normalized Weight " * tilde(w)[i]),
       y = "Density",
       title = "Distribution of Normalized Survey Weights",
       subtitle = expression("Normalized: " * Sigma * tilde(w)[i] == N *
                             "; red dashed line = 1.0 (self-representing)")) +
  theme_manuscript()

save_figure(p_sf1, "SF1_weight_distribution", width = 7, height = 4.5)

## ---- SF3: Posterior Predictive Checks ----
ppc_t <- ppc_comp$ppc_table

df_sf3 <- ppc_t %>%
  select(label, obs_zero_rate, pred_zero_rate, zero_rate_lo, zero_rate_hi,
         obs_it_share, pred_it_share, it_share_lo, it_share_hi) %>%
  mutate(model_f = factor(label, levels = label))

## Shorten model labels for cleaner x-axis
model_short <- c("M0\nPooled", "M1\nRand.Int.", "M2\nBlock SVC",
                 "M3a\nCross-Cov.", "M3b\nPolicy Mod.")
df_sf3$model_short <- factor(model_short[as.numeric(df_sf3$model_f)],
                              levels = model_short)

## Panel (a): Zero rates
df_zr <- df_sf3 %>%
  select(model_short, obs_zero_rate, pred_zero_rate, zero_rate_lo, zero_rate_hi) %>%
  rename(obs = obs_zero_rate, pred = pred_zero_rate, lo = zero_rate_lo, hi = zero_rate_hi)

p_sf3a <- ggplot(df_zr, aes(x = model_short)) +
  geom_hline(aes(yintercept = obs[1]), linetype = "dashed",
             colour = MARGIN_COLORS["Intensive"], linewidth = 0.5) +
  geom_pointrange(aes(y = pred, ymin = lo, ymax = hi),
                  colour = MARGIN_COLORS["Extensive"],
                  linewidth = POINTRANGE_STD$linewidth,
                  size = POINTRANGE_STD$size) +
  scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
  labs(x = NULL, y = "Zero Rate",
       subtitle = "(a) Zero Rate PPC") +
  theme_manuscript(base_size = 9) +
  theme(axis.text.x = element_text(size = 7.5, lineheight = 0.9),
        plot.subtitle = element_text(face = "bold", size = 9.5,
                                     colour = "black", hjust = 0))

## Panel (b): IT share
df_is <- df_sf3 %>%
  select(model_short, obs_it_share, pred_it_share, it_share_lo, it_share_hi) %>%
  rename(obs = obs_it_share, pred = pred_it_share, lo = it_share_lo, hi = it_share_hi)

p_sf3b <- ggplot(df_is, aes(x = model_short)) +
  geom_hline(aes(yintercept = obs[1]), linetype = "dashed",
             colour = MARGIN_COLORS["Intensive"], linewidth = 0.5) +
  geom_pointrange(aes(y = pred, ymin = lo, ymax = hi),
                  colour = MARGIN_COLORS["Extensive"],
                  linewidth = POINTRANGE_STD$linewidth,
                  size = POINTRANGE_STD$size) +
  scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
  labs(x = NULL, y = "Mean IT Share",
       subtitle = "(b) IT Share PPC") +
  theme_manuscript(base_size = 9) +
  theme(axis.text.x = element_text(size = 7.5, lineheight = 0.9),
        plot.subtitle = element_text(face = "bold", size = 9.5,
                                     colour = "black", hjust = 0))

p_sf3 <- p_sf3a + p_sf3b +
  plot_annotation(
    title = "Posterior Predictive Checks Across All Models",
    subtitle = "Blue = predicted mean with 95% PPI; red dashed = observed",
    theme = theme(
      plot.title = element_text(face = "bold", size = 11, hjust = 0),
      plot.subtitle = element_text(size = 9, hjust = 0, colour = "grey40")
    )
  )

save_figure(p_sf3, "SF3_posterior_predictive", width = 7.5, height = 4.5)

## ---- SF6: Sandwich Correction Detail (3-panel) ----

## Prepare data
pi_vals <- chol_corr$prior_inflation
A_diag  <- diag(chol_corr$A)

df_sf6 <- data.frame(
  param = factor(param_labels, levels = param_labels),
  param_math = factor(param_labels_math, levels = param_labels_math),
  DER = ct$DER,
  DER_MCMC = ct$DER_vs_MCMC,
  PI = pi_vals,
  A_diag = A_diag
)

## Shared x-axis label function for parsed math
sf6_x_labels <- parse(text = param_labels_math)

## Panel (a): Prior Inflation -- horizontal bars for readability
p_sf6a <- ggplot(df_sf6, aes(x = PI, y = param)) +
  geom_vline(xintercept = 10, linetype = "dashed", color = "grey50",
             linewidth = 0.4) +
  geom_col(fill = "#D6604D", alpha = 0.7, width = 0.6) +
  geom_text(aes(label = sprintf("%.0f", PI)), hjust = -0.1, size = 2.3,
            color = "#B2182B") +
  ## "PI = 10" label: place at first factor level (log(kappa)) with nudge
  annotate("text", x = 10, y = 1, label = "PI = 10",
           size = 2.3, color = "grey50", hjust = -0.15, vjust = 2.2) +
  scale_x_log10(expand = expansion(mult = c(0, 0.15))) +
  scale_y_discrete(labels = sf6_x_labels,
                   expand = expansion(add = c(0.8, 0.5))) +
  labs(x = "Prior Inflation Ratio (log scale)", y = NULL,
       subtitle = "(a) Prior Inflation: Naive CI / Wald CI width ratio") +
  theme_manuscript(base_size = 8) +
  theme(panel.grid.major.y = element_blank(),
        plot.subtitle = element_text(face = "bold", size = 9,
                                     color = "black", hjust = 0))

## Panel (b): DER comparison -- horizontal grouped bars
df_sf6b <- df_sf6 %>%
  select(param, DER, DER_MCMC) %>%
  pivot_longer(cols = c(DER, DER_MCMC), names_to = "type", values_to = "value") %>%
  mutate(type_label = factor(
    ifelse(type == "DER", "DER (V_sand / H_obs_inv)", "DER_MCMC (V_sand / Sigma_MCMC)"),
    levels = c("DER (V_sand / H_obs_inv)", "DER_MCMC (V_sand / Sigma_MCMC)")
  ))

p_sf6b <- ggplot(df_sf6b, aes(x = value, y = param, fill = type_label)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50",
             linewidth = 0.4) +
  geom_col(position = position_dodge(width = 0.7), alpha = 0.7, width = 0.6) +
  scale_fill_manual(values = c("DER (V_sand / H_obs_inv)" = "#4393C3",
                               "DER_MCMC (V_sand / Sigma_MCMC)" = "#D6604D")) +
  scale_x_continuous(breaks = seq(0, 5, 1),
                     expand = expansion(mult = c(0, 0.05))) +
  scale_y_discrete(labels = sf6_x_labels) +
  labs(x = "Ratio", y = NULL,
       subtitle = "(b) DER vs DER-MCMC", fill = NULL) +
  theme_manuscript(base_size = 8) +
  theme(panel.grid.major.y = element_blank(),
        legend.position = "bottom",
        legend.text = element_text(size = 7),
        plot.subtitle = element_text(face = "bold", size = 9,
                                     color = "black", hjust = 0))

## Panel (c): Cholesky A diagonal -- horizontal bars
p_sf6c <- ggplot(df_sf6, aes(x = A_diag, y = param)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50",
             linewidth = 0.4) +
  geom_col(fill = "#1B7837", alpha = 0.7, width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", A_diag)), hjust = -0.1, size = 2.3,
            color = "#00441B") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.2)),
                     breaks = seq(0, 1, 0.2)) +
  scale_y_discrete(labels = sf6_x_labels) +
  labs(x = "A diagonal value", y = NULL,
       subtitle = "(c) Cholesky A Scaling Factor (A << 1 = prior-dominated)") +
  theme_manuscript(base_size = 8) +
  theme(panel.grid.major.y = element_blank(),
        plot.subtitle = element_text(face = "bold", size = 9,
                                     color = "black", hjust = 0))

p_sf6 <- p_sf6a / p_sf6b / p_sf6c +
  plot_annotation(
    title = "Sandwich Correction Diagnostic Details",
    subtitle = "Prior-dominated parameters (PI >> 10): Cholesky A << 1 (shrinks to data scale)",
    theme = theme(
      plot.title = element_text(face = "bold", size = 11, hjust = 0),
      plot.subtitle = element_text(size = 9, hjust = 0, color = "grey40")
    )
  )

save_figure(p_sf6, "SF6_sandwich_detail", width = 7, height = 9)


###############################################################################
## SECTION 16 : SUMMARY INVENTORY
###############################################################################

cat("\n")
cat("==============================================================\n")
cat("  OUTPUT INVENTORY\n")
cat("==============================================================\n\n")

all_tables  <- list.files(TABLE_DIR, full.names = FALSE)
all_figures <- list.files(FIGURE_DIR, full.names = FALSE)

cat(sprintf("  Total table files: %d\n", length(all_tables)))
cat(sprintf("  Total figure files: %d\n\n", length(all_figures)))

tables_csv <- grep("\\.csv$", all_tables, value = TRUE)
tables_tex <- grep("\\.tex$", all_tables, value = TRUE)
figures_all <- grep("\\.(pdf|png)$", all_figures, value = TRUE)

cat("  TABLES (.csv):\n")
for (f in sort(tables_csv)) cat(sprintf("    %s\n", f))
cat("\n  TABLES (.tex):\n")
for (f in sort(tables_tex)) cat(sprintf("    %s\n", f))
cat("\n  FIGURES (.pdf + .png):\n")
for (f in sort(figures_all)) cat(sprintf("    %s\n", f))

cat("\n  Done. All outputs in: output/tables/ and output/figures/\n")
cat("==============================================================\n")
