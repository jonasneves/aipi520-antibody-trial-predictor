# Monoclonal Antibody Clinical Trial Success Predictor

[![ML Pipeline](https://github.com/jonasneves/aipi520-antibody-trial-predictor/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/jonasneves/aipi520-antibody-trial-predictor/actions/workflows/ml-pipeline.yml)

**Reports Portal:** [Model Dashboard](https://jonasneves.github.io/aipi520-antibody-trial-predictor) | [Raw Data EDA](https://jonasneves.github.io/aipi520-antibody-trial-predictor/eda_raw_data.html) | [Engineered Features EDA](https://jonasneves.github.io/aipi520-antibody-trial-predictor/eda_features.html)

## Overview

Predict clinical trial success for monoclonal antibody (mAb) therapeutics using machine learning. Built for Duke AIPI 520.

**Problem:** Monoclonal antibodies represent a $237B+ global market with $1-2B development costs per drug and high Phase 2/3 failure rates (~70%).

**Solution:** Machine learning classifiers trained on 7,094 Phase 2/3 antibody trials from ClinicalTrials.gov with 32 domain-specific features (antibody type, mechanism, biomarkers, trial design, temporal features) using stratified random validation.

## Dataset

| Source | Description | Coverage |
|--------|-------------|----------|
| [ClinicalTrials.gov Bulk XML](https://clinicaltrials.gov/data-api/about-api/bulk-data) | Complete trial registry | ~500K total trials available |
| Raw Antibody Trials | Phase 2/3 completed antibody trials | 7,116 trials collected |
| Processed Dataset | After labeling, temporal filtering & feature engineering | 7,094 trials (22 post-2024 trials filtered) |
| Features | Antibody-specific + traditional trial features (no text features) | 32 features engineered |

**Data Collection:** Bulk XML download from S3, streaming parse (no 11GB extraction needed), antibody classification by INN nomenclature (-umab/-zumab/-ximab/-omab).

## Model Performance

**Evaluation Method:** Random stratified train/test split (80/20)
- **Train:** 5,675 samples (80%)
- **Test:** 1,419 samples (20%)

**Best Model:** TBD (will be updated after re-training)

**Methodology:**
- Post-2024 trials filtered (22 trials)
- Text features excluded (50 TF-IDF features)
- `has_start_date` flag added for trials with missing dates (45% of dataset)
- Temporal features set to 0 for missing dates
- Stratified random split ensures balanced class distribution in train/test sets

**Note:** Previous time-based split (pre-2023 train, 2023+ test) showed temporal confounding - recent trials had 28.6% success vs 72.5% for older trials due to incomplete follow-up time. Random split provides more reliable performance estimates.

## Quick Start

```bash
make install    # Install dependencies
make pipeline   # Run complete pipeline (collect → label → features → train → dashboard)
```

View results:
```bash
open docs/index.html  # Model comparison dashboard
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML | scikit-learn, XGBoost, pandas, numpy |
| Data Processing | XML parsing, streaming (zipfile), antibody classification |
| Visualization | Plotly (interactive charts), HTML reports |
| Cloud Storage | AWS S3 (bulk XML hosting) |
| CI/CD | GitHub Actions (parallel model training) |
| Deployment | GitHub Pages (automated reports) |

## Project Structure

```
src/                    # ML pipeline
├── antibody_utils.py   # Antibody classification (INN nomenclature)
├── xml_parser.py       # Bulk XML parsing
├── data_labeling.py    # Success/failure classification
└── feature_engineering.py  # 32 feature extraction

scripts/
├── parse_bulk_xml.py   # S3 download & parse (~500K trials)
├── train_single_model.py  # Individual model training
├── generate_eda_*.py   # EDA report generation
└── charts/             # Plotly visualization modules

data/                   # Generated datasets (CSV)
models/                 # Trained models (.pkl)
docs/                   # HTML reports (GitHub Pages)
```

## Detailed Usage

### Installation
```bash
make install
# or: pip install -r requirements.txt
```

### Automated Pipeline (GitHub Actions)
Push to main branch triggers:
1. Bulk XML download from S3 (or use cache)
2. Data labeling (7,116 → 7,094 trials after temporal filtering) & feature engineering (32 features)
3. Models trained with time-based validation (Gradient Boosting, Random Forest, XGBoost, Logistic Regression, Decision Tree)
4. EDA reports & model dashboard generated with interactive Plotly visualizations
5. Results automatically deployed to GitHub Pages

### Manual Pipeline

Run complete pipeline:
```bash
make pipeline
```

Or individual steps:
```bash
make collect    # Download and parse bulk XML from S3 (7,116 trials)
make label      # Label trials as success/failure
make features   # Engineer 32 features
make train      # Train models with stratified random split
make dashboard  # Generate comparison dashboard with interactive charts
```

### Training Individual Models

```bash
python scripts/train_single_model.py <model_code> data/clinical_trials_features.csv models/
# Add --temporal-split flag to use time-based split (not recommended due to temporal confounding)
```

Model codes: `lr` (Logistic Regression), `dt` (Decision Tree), `rf` (Random Forest), `gb` (Gradient Boosting), `xgb` (XGBoost)

**Default:** Random stratified split (80% train, 20% test)

## Features (32 Total)

**Antibody-Specific (12 features):**
- Type classification: murine, chimeric, humanized, fully_human (4 binary features)
- Target mechanisms: checkpoint inhibitors (PD-1/PD-L1), growth factors (HER2/EGFR), cytokines, CD markers (4 binary features)
- Biomarker selection: HER2+, PD-L1+, EGFR+ (1 binary feature)
- Combination therapy indicators (1 binary feature)
- Favorable indication flags (1 binary feature)
- Antibody-specific metadata (1 feature)

**Traditional Trial Features (20 features):**
- Trial design: interventional/observational, phase indicators (6 features)
- Enrollment metrics: raw count, log-transformed count (2 features)
- Disease categories: cancer, cardiovascular, neurological, diabetes, infectious, respiratory, autoimmune, mental health (8 features)
- Sponsor: type (industry/NIH), trial count, major sponsor flag (3 features)
- Trial metadata: number of conditions (1 feature)

**Temporal Features (4 features):**
- `has_start_date`: Binary flag indicating valid start date (1 feature)
- Start year, years since start, recent trial flag (3 features)
- **Note:** 45% of trials have missing dates → set to 0

**Text Features:** Excluded (50 TF-IDF features not used)

## Methodology

1. **Data Collection:** Stream parse bulk XML from S3, filter Phase 2/3 interventional antibody trials → 7,116 trials
2. **Data Labeling:** Refined binary classification (Completed/Approved = success; Terminated/Withdrawn for efficacy/safety = failure; Administrative terminations excluded) → 7,094 trials (22 post-2024 trials filtered)
3. **Feature Engineering:** Extract 32 domain-specific features: 12 antibody-specific, 20 traditional trial, 4 temporal (no text features). `has_start_date` flag added for missing dates (45% of dataset)
4. **Model Training:** Train classifiers with stratified random split (Gradient Boosting, Random Forest, XGBoost, Logistic Regression, Decision Tree) using stratified 3-fold CV. 80/20 train/test split
5. **Evaluation:** ROC AUC (primary metric), plus F1, accuracy

## Background

**Market Context:** Monoclonal antibodies represent a $237+ billion global market (2023) growing at 11-12% CAGR, with blockbuster drugs like Keytruda ($25B/year) and Humira (peak $21B). Development costs $1-2 billion per drug over 10-15 years, with high Phase 2/3 failure risk (~70%).

**Clinical Trial Phases:**
- Phase 1: Safety/dosage testing
- Phase 2: Efficacy proof (highest failure rate)
- Phase 3: Large-scale validation vs. standard of care

## AI Usage Acknowledgment

**AI Assistants:** Claude Code (Anthropic) for code development, documentation, and domain knowledge research.

All code and analysis were reviewed, tested, and thoroughly understood by the team. The team takes full responsibility for the implementation and can explain all design decisions.

## References

1. Grand View Research. (2023). *Monoclonal Antibodies Market Size, Share & Trends Analysis Report*. https://www.grandviewresearch.com/industry-analysis/monoclonal-antibodies-market
2. Hay, M., et al. (2014). "Clinical development success rates for investigational drugs." *Nature Biotechnology*, 32, 40-51. DOI: 10.1038/nbt.2786
3. Merck & Co. (2024). *Keytruda Annual Revenue Report*. https://www.statista.com/statistics/1269401/revenues-of-keytruda/
4. AbbVie Inc. (2024). *Humira Revenue Data*. https://www.statista.com/statistics/318206/revenue-of-humira/
5. Congressional Budget Office. (2021). *Research and Development in the Pharmaceutical Industry*. https://www.cbo.gov/publication/57126
6. DiMasi, J.A., et al. (2016). "Innovation in the pharmaceutical industry: new estimates of R&D costs." *Journal of Health Economics*, 47, 20-33.
7. Fu, T., et al. (2022). "HINT: Hierarchical interaction network for clinical-trial-outcome predictions." *Patterns*, 3(4), Article 100445.
8. World Health Organization. *International Nonproprietary Names (INN) for Biological and Biotechnological Substances*.

## Authors

Jonas De Oliveira Neves, Sharmil Nanjappa Kallichanda, Dominic Tanzillo

Duke University - AIPI 520, 2025

## License

MIT
