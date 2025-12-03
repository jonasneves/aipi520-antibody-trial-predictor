# Monoclonal Antibody Clinical Trial Success Predictor

[![ML Pipeline](https://github.com/jonasneves/aipi520-antibody-trial-predictor/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/jonasneves/aipi520-antibody-trial-predictor/actions/workflows/ml-pipeline.yml)

**Reports Portal:** [Model Dashboard](https://jonasneves.github.io/aipi520-antibody-trial-predictor) | [Raw Data EDA](https://jonasneves.github.io/aipi520-antibody-trial-predictor/eda_raw_data.html) | [Engineered Features EDA](https://jonasneves.github.io/aipi520-antibody-trial-predictor/eda_features.html)

## Overview

Predict clinical trial success for monoclonal antibody (mAb) therapeutics using machine learning. Built for Duke AIPI 520.

## Dataset

| Source | Description | Coverage |
|--------|-------------|----------|
| [ClinicalTrials.gov Bulk XML](https://clinicaltrials.gov/data-api/about-api/bulk-data) | Complete trial registry | ~500K total trials available |

**Data Collection:** Bulk XML uploaded to S3, streaming parse (no 11GB extraction needed).

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

## AI Usage Acknowledgment

**AI Assistants:** Claude Code (Anthropic) for code development, documentation, and domain knowledge research.

All code and analysis were reviewed, tested, and thoroughly understood by the team. The team takes full responsibility for the implementation and can explain all design decisions.

## Authors

Jonas De Oliveira Neves, Sharmil Nanjappa Kallichanda, Dominic Tanzillo

Duke University - AIPI 520, 2025

## License

MIT
