# Dataset documentation — BRFSS 2024

This directory documents the source data, variables, transformations, and derived files for the study:

> Barua, S., Jabran, S., Rodgers, C., and Fong, A. *Quantifying Behavioral Contributions to Diabetes Risk Using an Explainable Multiclass Machine Learning Framework with BRFSS 2024*. **Discover Public Health** (accepted October 2026; DOI and final citation pending).

**Availability:** Neither the original CDC public-use file nor the processed respondent-level analysis data are distributed in this repository. This README documents the intended dataset and outputs; it does **not** establish that the published results have been reproduced using the reorganized scripts.

## Source and citation

- **Provider:** U.S. Centers for Disease Control and Prevention (CDC).
- **Dataset:** 2024 Behavioral Risk Factor Surveillance System (BRFSS), public-use annual data.
- **Format used for the analysis:** SAS transport (`.XPT`).
- **Source and documentation:** [CDC BRFSS](https://www.cdc.gov/brfss/) and [BRFSS annual data](https://www.cdc.gov/brfss/annual_data/annual_data.htm).

Researchers using the data should cite the CDC BRFSS 2024 release and refer to its questionnaire, codebook, and calculated-variable documentation. When using the analysis implementation or reporting the findings, also cite the accompanying paper. Do not describe the processed data as a new independently collected dataset.

## Analytic sample and outcome

The final study reports **453,241** eligible respondents, with these **unweighted analytic-sample** class counts:

| Class code | Outcome | Respondents | Share |
| --- | --- | ---: | ---: |
| `0` | No diabetes | 376,125 | 82.99% |
| `1` | Prediabetes / borderline diabetes | 11,307 | 2.49% |
| `2` | Diabetes | 65,809 | 14.52% |
| **Total** | | **453,241** | **100%** |

The three-class outcome is derived from the self-reported `DIABETE4` variable:

| Original `DIABETE4` | Analysis label |
| --- | --- |
| `3` | `0`: No diabetes |
| `4` | `1`: Prediabetes / borderline diabetes |
| `1` | `2`: Diabetes |

Other `DIABETE4` codes are excluded from the analytic sample. The study's preprocessing script also drops duplicate rows after selecting the required analysis columns. These categories are based on *self-report*, not biomarkers or independent clinical adjudication. The 2.49% analytic-sample share of prediabetes is **not** a population prevalence estimate.

## Predictors

Two model specifications use **23 original variables** in total, organized into three prespecified domains:

| Domain | Number | BRFSS variable codes |
| --- | ---: | --- |
| Behavioral / health-status | 14 | `EXERANY2`, `_PHYS14D`, `SMOKE100`, `SMOKDAY2`, `_SMOKER3`, `_RFSMOK3`, `ALCDAY4`, `AVEDRNK4`, `DRNK3GE5`, `DRNKANY6`, `GENHLTH`, `PHYSHLTH`, `MENTHLTH`, `POORHLTH` |
| Anthropometric | 4 | `_BMI5`, `_BMI5CAT`, `_RFBMI5`, `WEIGHT2` |
| Demographic | 5 | `SEXVAR`, `_AGE80`, `_RACEGR3`, `EDUCA`, `INCOME3` |

**Model-1** uses the 14 behavioral/health-status variables. **Model-2** adds the 4 anthropometric and 5 demographic variables. The final study reports **122 encoded Model-2 features** after transformation.

`SEQNO` and `_STATE` are retained for record indexing and geographic analysis; `_LLCPWT` is retained for **evaluation-stage sampling-weight sensitivity only**. These are not predictors. The `_STATE` field includes jurisdictions/territories; geographic SHAP summaries are exploratory and should not be interpreted as comparisons of diabetes prevalence or risk.

## Preprocessing and split

The numbered script [`../src/01_preprocessing.py`](../src/01_preprocessing.py) represents the reorganized preprocessing implementation. Its intended order is:

1. Load the original SAS transport file; select the outcome, prespecified predictors, and three metadata fields.
2. Restrict to valid `DIABETE4` codes, remove duplicate analysis rows, and apply deterministic, variable-specific recoding. Examples: recode specified refusal/unknown values to missing; map `88` unhealthy-day values and documented no-consumption values to zero; convert valid metric-coded `WEIGHT2` observations into pounds.
3. Split once into **80% training (362,592)** and **20% held-out testing (90,649)**, stratified on the outcome, with `random_state=42`.
4. **Fit learned transformations on training data only** and apply the same fitted transformers to the held-out test data: numeric median imputation and scaling; categorical most-frequent imputation and one-hot encoding, ignoring unseen categories.

These are the script's intended rules, not an independent verification against the original `.XPT` data. For precise code-level mappings and edge cases, refer to the source script and the official 2024 CDC codebook.

## Local directory layout

```text
data/
├── README.md                # this file (tracked in Git)
├── raw/
│   └── CDC_BRFSS2024.XPT    # obtain from CDC; not tracked in Git
└── processed/
    ├── prepared_data.joblib      # generated; not tracked in Git
    ├── preprocessor_model1.joblib # generated; not tracked in Git
    ├── preprocessor_model2.joblib # generated; not tracked in Git
    └── preprocessing_summary.json # generated; not tracked in Git
```

The generated `prepared_data.joblib` is an **internal analysis artifact**, not a portable, independently curated CSV release. It contains transformed feature matrices, train/test labels, aligned subgroup and geographic metadata, train/test positions, and feature names. Saved preprocessors capture the learned training-only transformations. **Do not load untrusted `joblib`/pickle files**; loading them can execute code.

## Distribution and provenance

The `.gitignore` intentionally excludes raw and respondent-level processed files from Git by default. If distributing a processed dataset later, first verify its contents, output format, applicable CDC terms, and provenance. Document the original release/version, extraction date, precise selection and recoding rules, class counts, schema, missingness, software versions, checksum, and any differences from the accepted manuscript. Large public data should be placed in an appropriate versioned data archive and linked from here, rather than silently committed into Git history.

## Statistical interpretation

BRFSS employs a complex sampling design. The study's primary models and SHAP attribution were not fully survey-design adjusted. `_LLCPWT` sensitivity results do **not** substitute for design-aware variance estimation accounting for weights, strata, and clusters. Subgroup performance gaps are descriptive; neither subgroup results nor geographic SHAP summaries establish causal effects, fairness conclusions, or population-level diabetes risk estimates.
