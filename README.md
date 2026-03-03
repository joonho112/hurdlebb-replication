# Replication Package: A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded Counts

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![R](https://img.shields.io/badge/R-%E2%89%A5%204.3-276DC3.svg)](https://www.r-project.org/)
[![Stan](https://img.shields.io/badge/CmdStan-%E2%89%A5%202.33-B2001E.svg)](https://mc-stan.org/users/interfaces/cmdstan)

**Paper:** "A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded Counts and Its Application to Childcare Enrollment"

**Author:** JoonHo Lee, The University of Alabama ([jlee296@ua.edu](mailto:jlee296@ua.edu))

**Preprint:** arXiv, March 2026

**Companion R package:** [`hurdlebb`](https://github.com/joonho112/hurdlebb) v0.1.0

---

## Paper Overview

This paper develops a Bayesian hierarchical hurdle beta-binomial (HBB) model for survey-weighted bounded counts. The model addresses three interrelated challenges that arise when analyzing enrollment proportions in large-scale surveys: (i) the bounded and discrete nature of the outcome (infant/toddler enrollment as a fraction of total capacity), (ii) structural zeros from providers that serve no infants or toddlers, and (iii) complex survey designs with unequal selection probabilities. The framework introduces a Cholesky-based sandwich variance correction that adjusts posterior credible intervals for survey design effects while preserving the hierarchical Bayesian structure. The methodology is applied to the 2019 National Survey of Early Care and Education (NSECE), modeling infant/toddler enrollment across 6,785 center-based childcare providers in 51 U.S. states and the District of Columbia.

---

## Quick Start (Track B -- Partial Replication)

Reproduce all publication tables and figures in approximately 10 minutes using pre-computed results, without restricted data or model refitting.

```r
# 1. Install required R packages
source("code/00_setup.R")

# 2. Generate all tables and figures
source("code/06_tables_figures.R")

# 3. Outputs appear in output/tables/ and output/figures/
```

---

## Requirements

### Software

| Software | Version | Purpose |
|:---------|:--------|:--------|
| R | >= 4.3 | Statistical computing |
| CmdStan | >= 2.33 | Stan model compilation and sampling (Track A only) |
| LaTeX | Any modern distribution | Manuscript compilation (optional) |

### R Packages

Install all dependencies at once with `source("code/00_setup.R")`, or install manually:

| Package | Purpose |
|:--------|:--------|
| `cmdstanr` | Interface to CmdStan for Bayesian model fitting |
| `posterior` | Posterior draws manipulation and summary |
| `loo` | LOO-CV and PSIS diagnostics |
| `survey` | Design-based survey inference (frequentist comparison) |
| `ggplot2` | Publication figures |
| `dplyr` | Data manipulation |
| `tidyr` | Data reshaping |
| `patchwork` | Multi-panel figure composition |
| `xtable` | LaTeX table generation |
| `sf` | Spatial data for choropleth maps |
| `maps` | U.S. state boundary data |
| `scales` | Axis formatting |
| `viridis` | Color scales |
| `ggrepel` | Non-overlapping text labels |
| `bayesplot` | MCMC diagnostic plots |
| `forcats` | Factor manipulation |
| `MASS` | Multivariate normal simulation |

---

## Repository Structure

```
hurdlebb-replication/
|
+-- README.md                          # This file
+-- LICENSE                            # MIT License
+-- .gitignore                         # Git ignore rules
+-- CITATION.cff                       # Machine-readable citation metadata
|
+-- code/
|   +-- 00_setup.R                     # Environment check + package installation
|   +-- 01_data_preparation.R          # Track A: restricted data --> stan_data.rds
|   +-- 02_model_fitting.R             # Master orchestrator (M0 --> M3b --> M3b-W)
|   +-- 03_survey_weighting.R          # Sandwich variance + Cholesky correction
|   +-- 04_model_comparison.R          # LOO-CV + posterior predictive checks
|   +-- 05_marginal_effects.R          # AME decomposition (LAE / LIE)
|   +-- 06_tables_figures.R            # Core Track B script (all publication outputs)
|   +-- 07_block_b_analyses.R          # Supplementary computational analyses
|   +-- helpers/
|   |   +-- utils.R                    # BetaBin PMF, logit/expit, shared utilities
|   |   +-- theme_manuscript.R         # ggplot2 publication theme
|   +-- models/                        # Individual model fitting scripts (Track A)
|   |   +-- 10_fit_m0.R               # M0: pooled HBB
|   |   +-- 20_fit_m1.R               # M1: random intercepts
|   |   +-- 30_fit_m2.R               # M2: block-diagonal SVC
|   |   +-- 40_fit_m3a.R              # M3a: full SVC
|   |   +-- 50_fit_m3b.R              # M3b: policy moderation
|   |   +-- 60_fit_m3b_weighted.R      # M3b-W: survey-weighted
|   |   +-- 61_sandwich_variance.R     # Sandwich variance computation
|   |   +-- 62_cholesky_transform.R    # Cholesky correction transform
|   +-- block_b/                       # Block B individual analysis scripts
|   |   +-- B1_reversal_probability.R  # Joint reversal probability
|   |   +-- B2_m3b_comparison.R        # M3b vs M3b-W comparison table
|   |   +-- B3_rho_cross.R            # Cross-component correlation tracking
|   |   +-- B4_lkj_sensitivity.R      # LKJ prior sensitivity analysis
|   |   +-- B5_misspec_simulation.R    # Misspecification simulation scenario
|   |   +-- B6_frequentist.R           # Frequentist contextualization
|   |   +-- B7_mcse.R                 # Monte Carlo standard errors
|   |   +-- B8_coverage_decomposition.R # Coverage gap decomposition
|   |   +-- B9_coverage_95.R          # 95% coverage results
|   +-- simulation/                    # Full simulation pipeline
|       +-- sim_00_config.R            # Simulation configuration and DGP parameters
|       +-- sim_01_generate.R          # Data generation from HBB DGP
|       +-- sim_02_fit.R              # Model fitting per replication
|       +-- sim_03_evaluate.R          # Coverage, bias, and RMSE evaluation
|       +-- sim_09_tables_figures.R    # Simulation tables and figures
|       +-- run_production_all.R       # Master launcher for full simulation
|
+-- stan/                              # Stan model files
|   +-- hbb_m0.stan                    # M0: pooled hurdle beta-binomial
|   +-- hbb_m1.stan                    # M1: state-level random intercepts
|   +-- hbb_m1_weighted.stan           # M1-W: weighted with score adjustments
|   +-- hbb_m1_weighted_noscores.stan  # M1-W: weighted without scores
|   +-- hbb_m2.stan                    # M2: block-diagonal SVC
|   +-- hbb_m3a.stan                   # M3a: full spatially varying coefficients
|   +-- hbb_m3b.stan                   # M3b: SVC + policy moderation
|   +-- hbb_m3b_weighted.stan          # M3b-W: survey-weighted (primary model)
|   +-- hbb_m3b_weighted_lkj.stan      # M3b-W-LKJ: LKJ prior sensitivity
|
+-- data/
|   +-- DATA_ACCESS.md                 # How to obtain NSECE restricted-use data
|   +-- precomputed/                   # Pre-computed results (~6 MB)
|   |   +-- (posterior summaries, LOO-CV results, AME tables, ...)
|   +-- precomputed/simulation/        # Aggregated simulation results
|       +-- (coverage, bias, RMSE summaries across 200 replications)
|
+-- output/
|   +-- tables/                        # Publication-ready tables (CSV + TEX)
|   +-- figures/                       # Publication-ready figures (PDF + PNG)
|
+-- manuscript/                        # arXiv preprint LaTeX source
    +-- main.tex                       # Master document
    +-- main.sty                       # Custom style file
    +-- references.bib                 # Bibliography
    +-- section1.tex ... section6.tex  # Main body sections
    +-- sm_a.tex ... sm_f8_coverage95.tex  # Supplementary material
    +-- Figures/                       # Compiled figure PDFs and table TEX files
```

---

## Two-Track Replication Design

This package supports two replication tracks to accommodate different levels of data access.

### Track B: Reproduce Tables and Figures (No Restricted Data)

**Time estimate:** ~10 minutes on a standard laptop.

Track B uses pre-computed posterior summaries and simulation results stored in `data/precomputed/`. It reproduces every publication table and figure without requiring the restricted-use NSECE data or refitting any Stan models.

**Step 1.** Verify your R environment:

```r
source("code/00_setup.R")
```

This script checks R version compatibility, installs any missing packages, and confirms that CmdStan is available (optional for Track B).

**Step 2.** Generate all tables and figures:

```r
source("code/06_tables_figures.R")
```

This single script reads from `data/precomputed/` and writes all publication outputs to `output/tables/` and `output/figures/`.

**Step 3.** (Optional) Run supplementary Block B analyses:

```r
source("code/07_block_b_analyses.R")
```

Produces supplementary tables and figures (LKJ sensitivity, misspecification coverage, frequentist comparison, coverage decomposition, etc.).

**Step 4.** Inspect outputs in `output/tables/` (CSV and TEX) and `output/figures/` (PDF and PNG).

---

### Track A: Full Replication (Restricted Data Required)

**Time estimate:** ~24 hours on a modern 8-core machine. The Stan model fitting stage (Step 3) is the primary bottleneck.

Track A reproduces the complete analysis pipeline from raw data through final outputs. It requires the NSECE 2019 restricted-use data. See `data/DATA_ACCESS.md` for detailed instructions on obtaining these data through OPRE/ICPSR.

**Step 1.** Place the NSECE restricted-use data:

```
data/restricted/cb_master_2019.rds
```

This directory is git-ignored and must never be committed to version control.

**Step 2.** Prepare the analysis data:

```r
source("code/01_data_preparation.R")
```

Produces `data/precomputed/stan_data.rds` (Stan input list) and `data/precomputed/analysis_data.rds` (tidy analysis data frame) with N = 6,785 providers across 51 jurisdictions.

**Step 3.** Fit all Stan models:

```r
source("code/02_model_fitting.R")
```

This master script sequentially fits six models (M0 through M3b-W) using CmdStan. Each model runs 4 chains with 2,000 warmup and 2,000 sampling iterations. Intermediate fit objects are saved to `data/precomputed/`.

**Step 4.** Compute sandwich variance corrections:

```r
source("code/03_survey_weighting.R")
```

Applies the Cholesky-based sandwich variance correction (Theorem 1 in the paper) to the M3b-W posterior, producing design-adjusted credible intervals.

**Step 5.** Model comparison:

```r
source("code/04_model_comparison.R")
```

Computes LOO-CV via PSIS for all model pairs and generates posterior predictive check summaries.

**Step 6.** Marginal effects:

```r
source("code/05_marginal_effects.R")
```

Decomposes average marginal effects (AME) into the logit-scale additive effect (LAE) and the logit-scale interaction effect (LIE) for each covariate.

**Step 7.** Generate all tables and figures:

```r
source("code/06_tables_figures.R")
```

**Step 8.** Run supplementary analyses:

```r
source("code/07_block_b_analyses.R")
```

---

## Simulation Study

The simulation study evaluates the frequentist operating characteristics (coverage, bias, RMSE) of the HBB model under three data-generating scenarios with R = 200 Monte Carlo replications each.

| Scenario | Label | Description |
|:---------|:------|:------------|
| S0 | Correct specification | Data generated from the true HBB DGP |
| S3 | Overdispersion misspecification | Inflated overdispersion relative to the fitted model |
| S4 | Correlation misspecification | Misspecified cross-component correlation structure |

### Running the Simulation

**Full production run** (~48 hours on an 8-core machine):

```r
source("code/simulation/run_production_all.R")
```

**Individual steps** (for debugging or partial runs):

```r
source("code/simulation/sim_00_config.R")       # Set DGP parameters
source("code/simulation/sim_01_generate.R")      # Generate synthetic datasets
source("code/simulation/sim_02_fit.R")           # Fit models (parallelized)
source("code/simulation/sim_03_evaluate.R")      # Compute coverage/bias/RMSE
source("code/simulation/sim_09_tables_figures.R") # Generate simulation outputs
```

Pre-computed simulation results are available in `data/precomputed/simulation/` for Track B users.

---

## Table and Figure Index

### Main Body Tables

| Paper | Internal ID | Description | Script |
|:------|:------------|:------------|:-------|
| Table 1 | T1 | Data summary statistics | `code/06_tables_figures.R` |
| Table 2 | T3 | LOO-CV model comparison | `code/06_tables_figures.R` |
| Table 3 | T4 | Fixed effects with sandwich standard errors | `code/06_tables_figures.R` |
| Table 4 | T5 | Policy moderator (Gamma) estimates | `code/06_tables_figures.R` |
| Table 5 | T6 | Average marginal effect decomposition | `code/06_tables_figures.R` |
| Table 6 | T7 | Simulation coverage results (90%) | `code/06_tables_figures.R` |

### Main Body Figures

| Paper | Internal ID | Description | Script |
|:------|:------------|:------------|:-------|
| Figure 1 | F1 | IT enrollment distribution | `code/06_tables_figures.R` |
| Figure 2 | F2 | Poverty reversal raw pattern | `code/06_tables_figures.R` |
| Figure 3 | F4 | Reversal probability choropleth map | `code/06_tables_figures.R` |
| Figure 4 | F5 | Cross-margin scatter plot | `code/06_tables_figures.R` |
| Figure 5 | F6 | Sandwich correction impact | `code/06_tables_figures.R` |
| Figure 6 | F7 | Simulation coverage comparison | `code/06_tables_figures.R` |

### Supplementary Tables

| ID | Description | Script |
|:---|:------------|:-------|
| ST-B2 | M3b vs M3b-W comparison | `code/07_block_b_analyses.R` |
| ST-B4 | LKJ prior sensitivity analysis | `code/07_block_b_analyses.R` |
| ST8 | Design effect ratio (DER) summary | `code/06_tables_figures.R` |
| ST-B5 | Misspecification simulation results | `code/07_block_b_analyses.R` |
| ST-B6 | Frequentist comparison | `code/07_block_b_analyses.R` |
| ST-B8 | Coverage gap decomposition | `code/07_block_b_analyses.R` |
| ST-B9 | 95% coverage results | `code/07_block_b_analyses.R` |
| ST-B3 | Cross-component correlation (rho_cross) | `code/07_block_b_analyses.R` |

### Supplementary Figures

| ID | Description | Script |
|:---|:------------|:-------|
| SF1 | Survey weight distribution | `code/06_tables_figures.R` |
| SF6 | Sandwich correction detail | `code/06_tables_figures.R` |
| SF7 | Simulation bias comparison | `code/06_tables_figures.R` |
| SF8 | Width ratio distribution | `code/06_tables_figures.R` |
| SF9 | DER summary plot | `code/06_tables_figures.R` |
| SF-B5 | Misspecification coverage | `code/07_block_b_analyses.R` |
| SF-LKJ | LKJ prior rho density | `code/07_block_b_analyses.R` |

---

## Model Progression

The paper develops the HBB model through a sequence of six nested specifications. Each model extends the previous one by relaxing a structural assumption.

```
M0  (Pooled HBB)
 |
 +-- M1  (State-level random intercepts)
      |
      +-- M2  (Block-diagonal spatially varying coefficients)
           |
           +-- M3a (Full SVC with cross-component correlation)
                |
                +-- M3b (SVC + state-level policy moderators)
                     |
                     +-- M3b-W (Survey-weighted with sandwich correction)
```

| Model | Description | Random Effects | Survey Weights |
|:------|:------------|:---------------|:---------------|
| M0 | Pooled hurdle beta-binomial | None | No |
| M1 | Random intercepts | State intercepts (hurdle + count) | No |
| M2 | Block-diagonal SVC | State slopes, no cross-component correlation | No |
| M3a | Full SVC | State slopes with cross-component correlation | No |
| M3b | Policy moderation | M3a + state-level policy covariates (Gamma) | No |
| M3b-W | Survey-weighted | M3b + pseudo-likelihood weighting + sandwich SE | Yes |

---

## Stan Model Files

| File | Model | Parameters | Description |
|:-----|:------|:-----------|:------------|
| `hbb_m0.stan` | M0 | Fixed effects only | Pooled hurdle beta-binomial |
| `hbb_m1.stan` | M1 | + state random intercepts | Hierarchical intercepts |
| `hbb_m1_weighted.stan` | M1-W | + survey weights, scores | Weighted with score adjustments |
| `hbb_m1_weighted_noscores.stan` | M1-W | + survey weights | Weighted without score terms |
| `hbb_m2.stan` | M2 | + block-diagonal Sigma | Uncorrelated SVC components |
| `hbb_m3a.stan` | M3a | + full Sigma | Cross-component correlation |
| `hbb_m3b.stan` | M3b | + Gamma (policy effects) | State-level policy moderation |
| `hbb_m3b_weighted.stan` | M3b-W | + pseudo-likelihood | Primary model (reported in paper) |
| `hbb_m3b_weighted_lkj.stan` | M3b-W-LKJ | + LKJ(2) prior | Prior sensitivity variant |

---

## Companion R Package

The [`hurdlebb`](https://github.com/joonho112/hurdlebb) R package (v0.1.0) provides a user-facing interface to the methodology developed in this paper. It includes:

- `hbb()`: Fit hurdle beta-binomial models with optional survey weighting
- `sandwich_correct()`: Apply the Cholesky-based sandwich variance correction
- `marginal_effects()`: Compute AME decomposition (LAE/LIE)
- `nsece_synth` and `nsece_synth_small`: Synthetic datasets for testing
- Vignettes with worked examples

**Installation:**

```r
# install.packages("remotes")
remotes::install_github("joonho112/hurdlebb")
```

---

## Citation

If you use this replication package or the associated methodology, please cite:

```bibtex
@article{Lee2026hbb,
  author  = {Lee, JoonHo},
  title   = {A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted
             Bounded Counts and Its Application to Childcare Enrollment},
  journal = {arXiv preprint},
  year    = {2026}
}
```

A machine-readable citation file is available in [`CITATION.cff`](CITATION.cff).

---

## License

This replication package is released under the [MIT License](LICENSE).

Copyright (c) 2026 JoonHo Lee.

---

## Contact

JoonHo Lee
Department of Educational Studies in Psychology, Research Methodology, and Counseling
The University of Alabama
[jlee296@ua.edu](mailto:jlee296@ua.edu)
