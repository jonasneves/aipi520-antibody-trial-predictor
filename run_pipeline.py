#!/usr/bin/env python3
"""
Monoclonal Antibody Trial Success Prediction - Data Pipeline

This script runs the data preparation pipeline for antibody trials:
1. Data collection from ClinicalTrials.gov (antibody trials only)
2. Data labeling (success/failure)
3. Feature engineering (including antibody-specific features)

Note: Model training is handled separately by scripts/train_single_model.py
"""

import sys
from pathlib import Path
import argparse
import time
import pandas as pd

# Add src to path
sys.path.append('src')

from data_collection import ClinicalTrialsAPI
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


def run_data_collection(max_studies=5000, force_download=False):
    """
    Step 1: Collect monoclonal antibody trial data

    Args:
        max_studies: Maximum number of studies to collect (0 = unlimited)
        force_download: Force re-download even if data exists
    """
    print_header("STEP 1: DATA COLLECTION")

    # Handle unlimited collection
    if max_studies == 0:
        max_studies = 999999
        print("Collecting unlimited studies (no max limit)")

    data_file = "data/completed_phase2_3_trials.csv"

    if is_valid_csv(data_file) and not force_download:
        df = pd.read_csv(data_file)
        print(f"✓ Valid data file found: {data_file}")
        print(f"  - {len(df)} trials loaded")
        print("Use --force-download to re-download")
        return
    elif Path(data_file).exists():
        print(f"Warning: Found invalid/empty data file at {data_file}")
        print("Re-downloading fresh data...")

    print("Initializing ClinicalTrials.gov API client...")
    api = ClinicalTrialsAPI(output_dir="data")

    query_params = {
        "query.term": (
            "(AREA[Phase]PHASE2 OR AREA[Phase]PHASE3) AND "
            "AREA[StudyType]INTERVENTIONAL AND "
            "("
            "AREA[OverallStatus]COMPLETED OR "
            "AREA[OverallStatus]TERMINATED OR "
            "AREA[OverallStatus]WITHDRAWN OR "
            "AREA[OverallStatus]SUSPENDED OR "
            "AREA[OverallStatus]APPROVED_FOR_MARKETING OR "
            "AREA[OverallStatus]AVAILABLE OR "
            "AREA[OverallStatus]NO_LONGER_AVAILABLE"
            ")"
        )
    }

    print(f"Fetching up to {max_studies} clinical trials...")
    print("This may take several minutes depending on network speed...")

    start_time = time.time()
    studies = api.search_studies(query_params=query_params, max_studies=max_studies)
    elapsed = time.time() - start_time

    print(f"\nCollected {len(studies)} studies in {elapsed:.1f} seconds")

    print("\nProcessing and filtering for antibody trials...")
    df = api.save_studies_to_csv(studies, "completed_phase2_3_trials.csv")
    api.save_raw_json(studies, "completed_phase2_3_trials_raw.json")

    antibody_count = df['is_antibody'].sum() if 'is_antibody' in df.columns else 0
    print(f"\n✓ Data collection complete!")
    print(f"  - CSV file: data/completed_phase2_3_trials.csv")
    print(f"  - JSON file: data/completed_phase2_3_trials_raw.json")
    print(f"  - Total studies: {len(df)}")
    print(f"  - Antibody trials: {antibody_count} ({antibody_count/len(df)*100:.1f}%)")


def run_data_labeling():
    """Step 2: Label clinical trials as success/failure"""
    print_header("STEP 2: DATA LABELING")

    data_file = "data/completed_phase2_3_trials.csv"

    if not is_valid_csv(data_file):
        print(f"Error: Valid data file not found at {data_file}")
        print("Please run data collection first: python run_pipeline.py --steps collect")
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

    # Create balanced dataset
    print("\nCreating balanced dataset...")
    df_balanced = labeler.balance_dataset(df_binary, method='undersample')

    # Save all versions
    df_labeled.to_csv("data/clinical_trials_labeled.csv", index=False)
    df_binary.to_csv("data/clinical_trials_binary.csv", index=False)
    df_balanced.to_csv("data/clinical_trials_balanced.csv", index=False)

    print(f"\n✓ Data labeling complete!")
    print(f"  - Labeled data: data/clinical_trials_labeled.csv")
    print(f"  - Binary labels: data/clinical_trials_binary.csv")
    print(f"  - Balanced data: data/clinical_trials_balanced.csv")


def run_feature_engineering():
    """Step 3: Engineer features from clinical trial data"""
    print_header("STEP 3: FEATURE ENGINEERING")

    data_file = "data/clinical_trials_binary.csv"
    features_file = "data/clinical_trials_features.csv"

    # Check if features already exist
    if is_valid_csv(features_file):
        print(f"✓ Using cached features: {features_file}")
        df_modeling = pd.read_csv(features_file)
        print(f"  - Samples: {len(df_modeling)} trials")
        print(f"  - Features: {len(df_modeling.columns)} columns")
        return

    if not is_valid_csv(data_file):
        print(f"Error: Valid labeled data not found at {data_file}")
        print("Please run data labeling first: python run_pipeline.py --steps collect label")
        sys.exit(1)

    print("Loading labeled data...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} labeled trials")

    print("\nInitializing feature engineer...")
    engineer = TrialFeatureEngineer()

    # Extract all features
    print("\nExtracting features...")
    df_features = engineer.extract_all_features(df, fit=True)

    # Select features for modeling
    print("\nSelecting features for modeling...")
    df_modeling = engineer.select_features_for_modeling(df_features)

    # Add labels back
    for label_col in ['binary_outcome', 'outcome_label']:
        if label_col in df.columns:
            df_modeling[label_col] = df[label_col]

    # Save engineered features
    df_modeling.to_csv(features_file, index=False)

    print(f"\n✓ Feature engineering complete!")
    print(f"  - Features file: {features_file}")
    print(f"  - Total features: {len(df_modeling.columns) - 2}")  # Excluding label columns
    print(f"  - Samples: {len(df_modeling)}")


def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(
        description='Clinical Trial Outcome Prediction Pipeline'
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['collect', 'label', 'features', 'model', 'all'],
        default=['all'],
        help='Steps to run (default: all)'
    )
    parser.add_argument(
        '--max-studies',
        type=int,
        default=5000,
        help='Maximum number of studies to collect (default: 5000)'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of data even if it exists'
    )

    args = parser.parse_args()

    # Create necessary directories
    for dir_name in ["data", "models", "reports"]:
        Path(dir_name).mkdir(exist_ok=True)

    print_header("CLINICAL TRIAL OUTCOME PREDICTION - ML PIPELINE")
    print(f"Project: Predicting clinical trial success/failure")
    print(f"Data source: ClinicalTrials.gov")
    print(f"Target: Phase 2 and Phase 3 completed trials")

    start_time = time.time()

    # Determine which steps to run
    steps = ['collect', 'label', 'features', 'model'] if 'all' in args.steps else args.steps

    try:
        if 'collect' in steps:
            run_data_collection(max_studies=args.max_studies, force_download=args.force_download)

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
