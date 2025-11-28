#!/usr/bin/env python3
"""
Monoclonal Antibody Trial Success Prediction - Data Pipeline

This script runs the data preparation pipeline for antibody trials:
1. Data labeling (success/failure)
2. Feature engineering (including antibody-specific features)

Note:
- Data collection is handled by scripts/parse_bulk_xml.py (bulk XML download)
- Model training is handled separately by scripts/train_single_model.py
"""

import sys
from pathlib import Path
import argparse
import time
import pandas as pd

# Add src to path
sys.path.append('src')

from data_labeling import TrialOutcomeLabeler
from feature_engineering import TrialFeatureEngineer


def print_header(title):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")


def is_valid_csv(filepath):
    """Check if a CSV file exists and has valid data."""
    path = Path(filepath)
    if not path.exists():
        return False

    try:
        df = pd.read_csv(filepath)
        return len(df) > 0 and len(df.columns) > 0
    except Exception:
        return False


def run_data_labeling():
    """Step 1: Label clinical trials as success/failure"""
    print_header("STEP 1: DATA LABELING")

    data_file = "data/completed_phase2_3_trials.csv"

    if not is_valid_csv(data_file):
        print(f"Error: Valid data file not found at {data_file}")
        print("Please run data collection first: python scripts/parse_bulk_xml.py")
        sys.exit(1)

    print("Loading collected data...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} clinical trials")

    print("\nInitializing labeler...")
    labeler = TrialOutcomeLabeler()

    # Label the dataset
    print("\nLabeling trials...")
    df_labeled = labeler.label_dataset(df)

    # Create binary labels
    print("\nCreating binary labels (success=1, failure=0)...")
    df_binary = labeler.create_binary_labels(df_labeled, include_ambiguous=False)

    # Note: We do NOT balance the dataset here (undersampling).
    # Class imbalance is handled during model training using class_weight='balanced'.
    # This ensures we don't discard valuable data from the majority class.

    # Save all versions
    df_labeled.to_csv("data/clinical_trials_labeled.csv", index=False)
    df_binary.to_csv("data/clinical_trials_binary.csv", index=False)

    print(f"\n✓ Data labeling complete!")
    print(f"  - Labeled data: data/clinical_trials_labeled.csv")
    print(f"  - Binary labels: data/clinical_trials_binary.csv")


def run_feature_engineering():
    """Step 2: Engineer features from clinical trial data"""
    print_header("STEP 2: FEATURE ENGINEERING")

    data_file = "data/clinical_trials_binary.csv"
    features_file = "data/clinical_trials_features.csv"
    labels_file = "data/clinical_trials_labels.csv"

    # Check if features already exist
    if is_valid_csv(features_file):
        print(f"✓ Using cached features: {features_file}")
        df_modeling = pd.read_csv(features_file)
        print(f"  - Samples: {len(df_modeling)} trials")
        print(f"  - Features: {len(df_modeling.columns)} columns")
        return

    if not is_valid_csv(data_file):
        print(f"Error: Valid labeled data not found at {data_file}")
        print("Please run data labeling first: python run_pipeline.py --steps label")
        sys.exit(1)

    print("Loading labeled data...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} labeled trials")

    # Filter out post-2024 trials
    print("\nFiltering post-2024 trials...")
    df['start_date_parsed'] = pd.to_datetime(df['start_date'], errors='coerce')
    cutoff_date = pd.Timestamp('2024-12-31')
    future_trials = df[df['start_date_parsed'] > cutoff_date]
    if len(future_trials) > 0:
        print(f"  Removing {len(future_trials)} trials with start_date > 2024-12-31")
        df = df[(df['start_date_parsed'] <= cutoff_date) | (df['start_date_parsed'].isna())].copy()
    df = df.drop(columns=['start_date_parsed'])
    print(f"  Filtered dataset: {len(df)} trials")

    print("\nInitializing feature engineer...")
    engineer = TrialFeatureEngineer()

    # Extract all features (excluding text features)
    print("\nExtracting features (excluding TF-IDF text features)...")
    df_features = engineer.extract_all_features(df, fit=True, include_text_features=False)

    # Select features for modeling
    print("\nSelecting features for modeling...")
    df_modeling = engineer.select_features_for_modeling(df_features)

    # Save labels separately
    labels_df = df[['binary_outcome', 'outcome_label']].copy() if 'binary_outcome' in df.columns else None

    # Save engineered features without labels
    df_modeling.to_csv(features_file, index=False)
    print(f"\n✓ Features saved to {features_file}")
    print(f"  - Total features: {len(df_modeling.columns)}")
    print(f"  - Samples: {len(df_modeling)}")

    # Save labels separately
    if labels_df is not None:
        labels_df.to_csv(labels_file, index=False)
        print(f"\n✓ Labels saved separately to {labels_file}")

    print(f"\n✓ Feature engineering complete!")


def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(
        description='Clinical Trial Outcome Prediction Pipeline - Data Preparation'
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['label', 'features', 'all'],
        default=['all'],
        help='Steps to run (default: all). Note: Data collection is handled by scripts/parse_bulk_xml.py'
    )

    args = parser.parse_args()

    # Create necessary directories
    for dir_name in ["data", "models", "reports"]:
        Path(dir_name).mkdir(exist_ok=True)

    print_header("CLINICAL TRIAL OUTCOME PREDICTION - ML PIPELINE")
    print(f"Project: Predicting clinical trial success/failure")
    print(f"Data source: ClinicalTrials.gov Bulk XML (~500K trials)")
    print(f"Target: Phase 2 and Phase 3 completed trials")
    print(f"\nNote: Run scripts/parse_bulk_xml.py first to collect data from bulk XML")

    start_time = time.time()

    # Determine which steps to run
    steps = ['label', 'features'] if 'all' in args.steps else args.steps

    try:
        if 'label' in steps:
            run_data_labeling()

        if 'features' in steps:
            run_feature_engineering()

        elapsed = time.time() - start_time

        print_header("PIPELINE COMPLETE!")
        print(f"Total execution time: {elapsed/60:.1f} minutes")
        print(f"\nOutput files generated:")
        print(f"  - Data: data/clinical_trials_features.csv")
        print(f"  - Models: models/ directory")
        print(f"  - Reports: reports/model_evaluation_results.csv")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure to run previous steps first or use --steps all")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
