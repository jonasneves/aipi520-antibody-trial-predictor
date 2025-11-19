# Predicting Monoclonal Antibody Clinical Trial Success

[![ML Pipeline](https://github.com/jonasneves/aipi520-project2/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/jonasneves/aipi520-project2/actions/workflows/ml-pipeline.yml)

## Project Overview

This project develops a machine learning model to predict the success of **monoclonal antibody (mAb) clinical trials**, focusing on one of the fastest-growing and most valuable therapeutic classes in pharmaceutical R&D.

## Background

**Market Context**: Monoclonal antibodies represent a $237+ billion global market (2023) [1] growing at 11-12% CAGR [1], with successful drugs like Keytruda ($25B/year) [3] and Humira (peak $21B) [4]. However, development costs $1-2 billion per drug [6] over 10-15 years [7], with high Phase 2/3 failure risk [2].

**Clinical Trial Phases**:
- **Phase 1**: Safety/dosage testing
- **Phase 2**: Efficacy proof (highest failure rate ~70%) [2]
- **Phase 3**: Large-scale validation vs. standard of care

## Project Structure

```
clinical-trial-prediction/
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── src/               # Source code for data processing and modeling
├── models/            # Trained model artifacts
├── reports/           # Written reports and documentation
└── requirements.txt   # Python dependencies
```

## Data Sources

- **ClinicalTrials.gov API**: Programmatic access to clinical trial data
- **Downloaded Dataset**: Complete dataset for offline analysis

## Methodology

1. **Data Collection**: ClinicalTrials.gov API (Phase 2/3 antibody trials)
2. **Data Labeling**: Binary classification (success/failure) based on trial status
3. **Feature Engineering**: 91 domain-specific features (antibody-specific + traditional)
4. **Model Training**: 5 models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
5. **Evaluation**: ROC AUC (primary), F1, accuracy, precision, recall with stratified cross-validation

## Features (91 Total)

**Antibody-Specific**: Type (-umab/-zumab/-ximab), target mechanism (checkpoint inhibitors, growth factors, cytokines), biomarker selection (HER2+, PD-L1+), combination therapy

**Traditional Trial**: Design (phase, study type), disease category, sponsor type/experience, enrollment size, temporal (duration, start date), historical success rates

## Models

5 models with class weighting: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (best: ROC AUC 0.824), XGBoost

## References

1. Grand View Research. (2023). *Monoclonal Antibodies Market Size, Share & Trends Analysis Report*. Market size: USD 237.61 billion in 2023, projected to reach USD 494.53 billion by 2030 at 11.04% CAGR. https://www.grandviewresearch.com/industry-analysis/monoclonal-antibodies-market

2. Hay, M., Thomas, D.W., Craighead, J.L., Economides, C., & Rosenthal, J. (2014). "Clinical development success rates for investigational drugs." *Nature Biotechnology*, 32, 40-51. DOI: 10.1038/nbt.2786

3. Merck & Co. (2024). *Keytruda Annual Revenue Report*. 2023 sales: $25.01 billion. https://www.statista.com/statistics/1269401/revenues-of-keytruda/

4. AbbVie Inc. (2024). *Humira Revenue Data*. Peak sales: $21.2 billion (2022); 2023 sales: $14.4 billion (post-biosimilar competition). https://www.statista.com/statistics/318206/revenue-of-humira/

5. Roche. *Herceptin (Trastuzumab) Sales Data*. Peak sales: ~$7 billion (2018); 2023 sales: $1.7 billion (biosimilar erosion).

6. Congressional Budget Office. (2021). *Research and Development in the Pharmaceutical Industry*. Estimated cost range: less than $1 billion to more than $2 billion per approved drug. https://www.cbo.gov/publication/57126

7. DiMasi, J.A., Grabowski, H.G., & Hansen, R.W. (2016). "Innovation in the pharmaceutical industry: new estimates of R&D costs." *Journal of Health Economics*, 47, 20-33. Average development timeline: 10-15 years.

8. Fu, T., Huang, K., Xiao, C., Glass, L. M., & Sun, J. (2022). "HINT: Hierarchical interaction network for clinical-trial-outcome predictions." *Patterns*, 3(4), Article 100445. DOI: 10.1016/j.patter.2022.100445

9. Elkin, M. E., & Zhu, X. (2021). "Predictive modeling of clinical trial terminations using feature engineering and embedding learning." *Scientific Reports*, 11, 3446. DOI: 10.1038/s41598-021-82840-x

10. World Health Organization. *International Nonproprietary Names (INN) for Biological and Biotechnological Substances*. Monoclonal antibody nomenclature guidelines: -umab (human), -zumab (humanized), -ximab (chimeric), -omab (murine).

## Team Members

- De Oliveira Neves, Jonas
- Kallichanda, Sharmil Nanjappa
- Tanzillo, Dominic

## Setup & Usage

### Installation
```bash
pip install -r requirements.txt
```

### Automated Pipeline (GitHub Actions)
Push to main branch triggers automated training of all 5 models in parallel.

### Manual Pipeline Execution

**Step 1: Data Preparation**
```bash
# Collect, label, and engineer features for antibody trials
python run_pipeline.py --steps all --max-studies 50000
# Outputs: data/clinical_trials_features.csv
```

**Step 2: Train Models** (train each model separately)
```bash
python scripts/train_single_model.py lr data/clinical_trials_features.csv results/
python scripts/train_single_model.py dt data/clinical_trials_features.csv results/
python scripts/train_single_model.py rf data/clinical_trials_features.csv results/
python scripts/train_single_model.py gb data/clinical_trials_features.csv results/
python scripts/train_single_model.py xgb data/clinical_trials_features.csv results/
# Model codes: lr=Logistic Regression, dt=Decision Tree, rf=Random Forest, gb=Gradient Boosting, xgb=XGBoost
```

**Step 3: Aggregate Results**
```bash
python scripts/aggregate_results.py results/ results/metrics_summary.json
```

**Step 4: Generate Dashboard**
```bash
python scripts/generate_report.py results/metrics_summary.json docs/index.html
# View: open docs/index.html
```

See `docs/ANTIBODY_FOCUS.md` for antibody-specific rationale.

## AI Usage Acknowledgment

This project was developed with assistance from AI tools (Claude/Anthropic) for code development, documentation, and domain knowledge research. All code and analysis were reviewed, tested, and thoroughly understood by the team. The team takes full responsibility for the implementation and can explain all design decisions.

## License

Academic project for educational purposes.
