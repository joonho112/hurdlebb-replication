# Data Access Statement

**Replication Package for:**
*A Bayesian Hierarchical Hurdle Beta-Binomial Model for Survey-Weighted Bounded Counts and Its Application to Childcare Enrollment*

**Author:** JoonHo Lee, The University of Alabama (jlee296@ua.edu)

---

## 1. Overview

This replication package uses restricted-use data from the 2019 National Survey of Early Care and Education (NSECE). The restricted-use file is required because it contains provider-level enrollment counts, community-level covariates linked to Census tract identifiers, and survey design variables (strata, PSU indicators, and sampling weights) at a level of geographic detail that is suppressed in the public-use release.

The public-use NSECE data (ICPSR 37941) provides a subset of survey items but does not include:

- Provider-level infant/toddler enrollment counts disaggregated by age group
- Census tract-level community characteristics (poverty rate, racial/ethnic composition)
- Full survey design variables necessary for design-based inference (PSU identifiers, stratum indicators)
- State identifiers enabling hierarchical modeling across all 51 jurisdictions

Because these variables are central to the analysis, full replication of the empirical application requires the restricted-use file.


## 2. Data Source Information

| Field | Detail |
|:------|:-------|
| **Survey name** | National Survey of Early Care and Education (NSECE), 2019 |
| **Component** | Center-Based Provider Survey |
| **Sponsor** | Office of Planning, Research, and Evaluation (OPRE), Administration for Children and Families (ACF), U.S. Department of Health and Human Services |
| **Data collection** | NORC at the University of Chicago |
| **Data collection period** | November 2018 -- July 2019 |
| **Restricted-use archive** | ICPSR 38893 |
| **Public-use archive** | ICPSR 37941 |
| **Citation** | NSECE Project Team. (2022). *National Survey of Early Care and Education (NSECE), 2019: Restricted-Use Data* [Data set]. Inter-university Consortium for Political and Social Research. |


## 3. How to Apply for Restricted-Use Access

Access to the restricted-use NSECE data requires a data license agreement administered by OPRE. The application process involves the following steps:

1. **Contact OPRE.** Submit an initial request describing your research purpose and the specific restricted-use variables needed. The primary point of contact is:

   > Office of Planning, Research, and Evaluation (OPRE)\
   > Administration for Children and Families\
   > U.S. Department of Health and Human Services\
   > https://www.acf.hhs.gov/opre/project/national-survey-early-care-and-education-nsece

2. **Prepare the application.** The application package typically requires:
   - A research proposal describing the scientific objectives and the specific restricted variables needed
   - Institutional Review Board (IRB) approval or exemption documentation
   - A data security plan specifying physical and electronic safeguards (e.g., encrypted storage, restricted-access workstation, no remote access without VPN)
   - Signed data use agreement by the principal investigator and an authorized institutional representative
   - Evidence of human subjects research training (e.g., CITI certification)

3. **Review and approval.** OPRE and NORC review the application for completeness and compliance with federal data protection requirements. Reviewers may request revisions to the security plan or additional documentation.

4. **Data receipt.** Upon approval, the restricted-use data files are transmitted through a secure mechanism specified by NORC. The license stipulates conditions on storage, access, publication of results, and data destruction upon project completion.

**Typical processing time.** The review process generally takes 2--6 months from initial submission to data receipt, depending on the completeness of the application and institutional review timelines. Applicants should plan accordingly.


## 4. Required Variables

The analysis uses the following variables from the NSECE 2019 Center-Based Provider Survey. Variable names below correspond to the restricted-use codebook; consult the NSECE documentation for exact field labels.

### 4.1 Outcome Variables

| Variable | Description |
|:---------|:------------|
| IT enrollment count ($Y_i$) | Number of infants and toddlers (ages 0--2) currently enrolled at provider $i$ |
| Total enrollment capacity ($n_i$) | Total number of children (ages 0--5) currently enrolled at provider $i$ |

The outcome pair $(Y_i, n_i)$ defines the bounded count. Providers with $n_i = 0$ (24 cases) are excluded because the beta-binomial kernel requires a positive denominator, yielding a final analytic sample of $N = 6{,}785$.

### 4.2 Provider-Level Covariates

| Variable | Description |
|:---------|:------------|
| Community poverty rate | Poverty rate in the provider's Census tract (continuous, %) |
| Urban/rural indicator | Binary indicator for urban location based on Census tract classification |
| Community % Black | Percentage of Black residents in the provider's Census tract |
| Community % Hispanic | Percentage of Hispanic residents in the provider's Census tract |

All continuous covariates are standardized (centered and scaled) prior to model fitting.

### 4.3 Survey Design Variables

| Variable | Description |
|:---------|:------------|
| Sampling weight ($w_i$) | Final survey weight for provider $i$ (range: 1--462) |
| Stratum indicator | Stratum assignment (30 strata defined by state groupings) |
| PSU indicator | Primary sampling unit identifier (415 PSUs) |

### 4.4 Geographic and Policy Variables

| Variable | Description |
|:---------|:------------|
| State identifier | State FIPS code or equivalent (51 jurisdictions: 50 states + DC) |
| CCDF market rate percentile | State-level Child Care and Development Fund subsidy rate relative to market survey |
| Tiered reimbursement | Binary indicator for whether the state uses tiered subsidy reimbursement |
| IT rate add-on | Binary indicator for whether the state provides an infant/toddler rate differential |


## 5. Data Preparation Pipeline

Once you have obtained the restricted-use data, follow these steps to prepare the analysis files.

### Step 1: Place the master data file

Place the NSECE Center-Based Provider restricted-use data file (converted to R's `.rds` format) at:

```
data/restricted/cb_master_2019.rds
```

This directory is listed in `.gitignore` and must never be committed to version control.

### Step 2: Run the data preparation script

```r
source("code/01_data_preparation.R")
```

This script performs the following operations:
- Loads the raw restricted-use data
- Applies sample exclusion criteria ($n_i > 0$)
- Constructs the hurdle indicator ($z_i = \mathbf{1}[Y_i > 0]$)
- Standardizes continuous covariates
- Merges state-level policy variables
- Assembles the Stan data list (outcome vectors, design matrices, weight arrays, index mappings)

### Step 3: Verify outputs

The script produces two output files:

| File | Description |
|:-----|:------------|
| `data/precomputed/stan_data.rds` | Named list formatted for Stan input (used by model-fitting scripts) |
| `data/precomputed/analysis_data.rds` | Tidy data frame with all variables for post-estimation analysis |

Verify that the resulting data contain $N = 6{,}785$ providers across $S = 51$ states, with 4,392 IT-serving providers ($z_i = 1$) and 2,393 non-servers ($z_i = 0$).


## 6. Alternative: Partial Replication Without Restricted Data

Researchers who do not have access to the restricted-use NSECE data can still verify the computational pipeline and reproduce the analysis workflow using synthetic data.

### Synthetic Data via the `hurdlebb` R Package

The companion R package [`hurdlebb`](https://github.com/joonho112/hurdlebb) (v0.1.0) includes two synthetic datasets generated via a copula-based procedure that preserves the marginal distributions and correlation structure of the original data:

| Dataset | Rows | Purpose |
|:--------|-----:|:--------|
| `nsece_synth` | 6,785 | Full-scale synthetic data matching the dimensions of the analytic sample |
| `nsece_synth_small` | 500 | Lightweight subset for quick testing and development |

To use the synthetic data:

```r
# install.packages("remotes")
remotes::install_github("joonho112/hurdlebb")

library(hurdlebb)
data(nsece_synth)       # Full synthetic dataset
data(nsece_synth_small) # Small test dataset
```

These synthetic datasets can be used to:
- Verify that all code scripts execute without error
- Confirm the computational pipeline from data preparation through Stan model fitting
- Inspect the structure of model output objects

However, because the synthetic data are simulated rather than observed, **point estimates, credible intervals, and substantive conclusions will differ from the published results**. The synthetic data should not be used to draw policy-relevant inferences.

### Pre-Computed Results (Track B)

For researchers interested in reproducing figures and tables without refitting the models, all pre-computed posterior summaries, simulation results, and tabulated output are provided in:

```
data/precomputed/
```

The table- and figure-generating scripts (Track B in the replication guide) read directly from these pre-computed files and do not require access to the restricted-use data or re-estimation of the Stan models.

---

**Last updated:** March 2026

**Contact:** JoonHo Lee (jlee296@ua.edu)
