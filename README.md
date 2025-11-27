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
- Missing start dates handled with binary flag; temporal features set to 0 (45% of dataset)
- Stratified random split (80/20 train/test)

**Note:** Time-based split showed temporal confounding (28.6% vs 72.5% success) due to incomplete follow-up. Random split provides more reliable estimates.

## Quick Start

```bash
make install    # Install dependencies
make pipeline   # Run complete pipeline (collect → label → features → train → reports)
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
├── base_report_generator.py  # Base class for report generation
├── generate_overview.py      # Model comparison dashboard
├── generate_eda_raw.py       # Raw data EDA report
├── generate_eda_features.py  # Features EDA report
└── charts/
    ├── utils.py        # Shared chart utilities
    ├── raw_data/       # Raw data visualizations
    └── features/       # Feature analysis charts

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
1. Bulk XML download from S3 (cached when available)
2. Data labeling & feature engineering → 7,094 trials, 32 features
3. Model training (5 classifiers with stratified random split)
4. Report generation (model dashboard + EDA reports)
5. Deployment to GitHub Pages

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
make reports    # Generate all reports (model dashboard + EDA reports)
```

### Training Individual Models

```bash
python scripts/train_single_model.py <model_code> data/clinical_trials_features.csv models/
# Use --temporal-split for time-based split (not recommended)
```

Model codes: `lr`, `dt`, `rf`, `gb`, `xgb`

Default: Random stratified split (80/20)

## Features (32 Total)

**Antibody-Specific (12):**
- Type classification: murine, chimeric, humanized, fully_human (4)
- Target mechanisms: checkpoint inhibitors (PD-1/PD-L1), growth factors (HER2/EGFR), cytokines, CD markers (4)
- Biomarker selection: HER2+, PD-L1+, EGFR+ (1)
- Combination therapy (1)
- Favorable indication (1)
- Antibody metadata (1)

**Traditional Trial Features (20):**
- Trial design: interventional/observational, phase indicators (6)
- Enrollment metrics: raw, log-transformed (2)
- Disease categories: cancer, cardiovascular, neurological, diabetes, infectious, respiratory, autoimmune, mental health (8)
- Sponsor: type, trial count, major sponsor flag (3)
- Number of conditions (1)

**Temporal Features (4):**
- `has_start_date` flag (1)
- Start year, years since start, recent trial flag (3)

## Methodology

1. **Data Collection:** Stream parse bulk XML from S3, filter Phase 2/3 interventional antibody trials → 7,116 trials
2. **Data Labeling:** Binary classification (Completed/Approved = success; Terminated/Withdrawn for efficacy/safety = failure) → 7,094 trials
3. **Feature Engineering:** 32 features (12 antibody-specific, 20 traditional, 4 temporal)
4. **Model Training:** Stratified random split (80/20), 3-fold CV
5. **Evaluation:** ROC AUC (primary), F1, accuracy

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
