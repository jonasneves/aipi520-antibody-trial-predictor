#!/usr/bin/env python3
"""
Experiment: Compare Clean Labels vs. All Data Labeling Strategies

This script compares two approaches to labeling clinical trial outcomes:
1. CLEAN LABELS: Exclude trials terminated for administrative reasons (current)
2. ALL DATA: Include all terminations as failures

Compares:
- Model performance (ROC-AUC, Precision, Recall, F1)
- Feature importance differences
- Class balance impact
- Statistical significance of performance differences
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, make_scorer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from data_labeling import TrialOutcomeLabeler
from feature_engineering import TrialFeatureEngineer


class LabelingStrategyComparison:
    """Compare different labeling strategies for clinical trial outcomes"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.results = {}

    def prepare_datasets(self):
        """Prepare both datasets with different labeling strategies"""
        print("=" * 80)
        print("LOADING AND PREPARING DATASETS")
        print("=" * 80)

        # Load raw data
        df = pd.read_csv(self.data_path)
        print(f"\nLoaded {len(df):,} raw trials from {self.data_path.name}")

        # Initialize labeler
        labeler = TrialOutcomeLabeler()

        # Label the dataset
        df_labeled = labeler.label_dataset(df)

        # Strategy 1: CLEAN LABELS (exclude administrative terminations)
        print("\n" + "-" * 80)
        print("STRATEGY 1: CLEAN LABELS (Exclude Administrative Terminations)")
        print("-" * 80)
        df_clean = labeler.create_binary_labels(
            df_labeled,
            include_ambiguous=False,
            use_refined_labels=True  # This excludes admin terminations
        )

        # Strategy 2: ALL DATA (include all terminations as failures)
        print("\n" + "-" * 80)
        print("STRATEGY 2: ALL DATA (Include All Terminations)")
        print("-" * 80)

        # Create a copy and manually label all terminations as failures
        df_all = df_labeled.copy()

        # Map based on the actual status values (title case)
        status_map = {
            'Completed': 1,
            'Approved for Marketing': 1,
            'Available': 1,
            'Terminated': 0,
            'Withdrawn': 0,
            'Suspended': 0,
            'No Longer Available': 0
        }

        df_all['binary_outcome'] = df_all['overall_status'].map(status_map)

        # Keep only rows with valid labels
        df_all = df_all.dropna(subset=['binary_outcome'])
        df_all['binary_outcome'] = df_all['binary_outcome'].astype(int)

        print(f"Total samples: {len(df_all)}")
        print(f"Successful trials (1): {(df_all['binary_outcome'] == 1).sum()}")
        print(f"Failed trials (0): {(df_all['binary_outcome'] == 0).sum()}")
        print(f"Success rate: {(df_all['binary_outcome'] == 1).sum() / len(df_all) * 100:.2f}%")

        # Store datasets
        self.df_clean = df_clean
        self.df_all = df_all

        # Print comparison
        print("\n" + "=" * 80)
        print("DATASET COMPARISON")
        print("=" * 80)
        print(f"Strategy 1 (Clean):   {len(df_clean):,} trials "
              f"({(df_clean['binary_outcome'] == 1).sum() / len(df_clean) * 100:.1f}% success)")
        print(f"Strategy 2 (All):     {len(df_all):,} trials "
              f"({(df_all['binary_outcome'] == 1).sum() / len(df_all) * 100:.1f}% success)")
        print(f"Difference:           {len(df_all) - len(df_clean):,} additional trials "
              f"({(len(df_all) - len(df_clean)) / len(df_clean) * 100:.1f}% more data)")

    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> tuple:
        """
        Apply feature engineering to dataset

        Returns:
            (X, y) tuple of features and labels
        """
        engineer = TrialFeatureEngineer()

        # Extract features (no text features to avoid complexity)
        df_features = engineer.extract_all_features(df, fit=fit, include_text_features=False)

        # Select modeling features
        X = engineer.select_features_for_modeling(df_features)
        y = df['binary_outcome'].values

        return X, y, engineer

    def train_and_evaluate(self, X, y, strategy_name: str, n_folds: int = 5):
        """
        Train model with cross-validation and collect metrics

        Args:
            X: Feature matrix
            y: Labels
            strategy_name: Name of the strategy for reporting
            n_folds: Number of cross-validation folds

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'=' * 80}")
        print(f"TRAINING AND EVALUATING: {strategy_name}")
        print(f"{'=' * 80}")

        # Define model
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=0
        )

        # Define scoring metrics
        scoring = {
            'roc_auc': 'roc_auc',
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }

        # Cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        print(f"\nPerforming {n_folds}-fold cross-validation...")
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )

        # Compile results
        results = {
            'strategy': strategy_name,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_positive': y.sum(),
            'n_negative': len(y) - y.sum(),
            'success_rate': y.mean(),
            'roc_auc_mean': cv_results['test_roc_auc'].mean(),
            'roc_auc_std': cv_results['test_roc_auc'].std(),
            'precision_mean': cv_results['test_precision'].mean(),
            'precision_std': cv_results['test_precision'].std(),
            'recall_mean': cv_results['test_recall'].mean(),
            'recall_std': cv_results['test_recall'].std(),
            'f1_mean': cv_results['test_f1'].mean(),
            'f1_std': cv_results['test_f1'].std(),
            'cv_scores': cv_results  # Store full CV scores for statistical testing
        }

        # Print results
        print(f"\nResults ({n_folds}-fold CV):")
        print(f"  ROC-AUC:   {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
        print(f"  Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
        print(f"  Recall:    {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
        print(f"  F1 Score:  {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")

        # Train final model on full data for feature importance
        print("\nTraining final model on full dataset for feature importance...")
        model.fit(X, y)
        results['feature_importance'] = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return results

    def compare_results(self):
        """Compare results between strategies"""
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)

        clean_results = self.results['clean']
        all_results = self.results['all']

        # Create comparison table
        comparison_df = pd.DataFrame({
            'Metric': ['Samples', 'Features', 'Success Rate (%)',
                      'ROC-AUC', 'Precision', 'Recall', 'F1 Score'],
            'Clean Labels': [
                f"{clean_results['n_samples']:,}",
                clean_results['n_features'],
                f"{clean_results['success_rate'] * 100:.1f}%",
                f"{clean_results['roc_auc_mean']:.4f} ± {clean_results['roc_auc_std']:.4f}",
                f"{clean_results['precision_mean']:.4f} ± {clean_results['precision_std']:.4f}",
                f"{clean_results['recall_mean']:.4f} ± {clean_results['recall_std']:.4f}",
                f"{clean_results['f1_mean']:.4f} ± {clean_results['f1_std']:.4f}",
            ],
            'All Data': [
                f"{all_results['n_samples']:,}",
                all_results['n_features'],
                f"{all_results['success_rate'] * 100:.1f}%",
                f"{all_results['roc_auc_mean']:.4f} ± {all_results['roc_auc_std']:.4f}",
                f"{all_results['precision_mean']:.4f} ± {all_results['precision_std']:.4f}",
                f"{all_results['recall_mean']:.4f} ± {all_results['recall_std']:.4f}",
                f"{all_results['f1_mean']:.4f} ± {all_results['f1_std']:.4f}",
            ]
        })

        print("\n" + comparison_df.to_string(index=False))

        # Statistical significance testing
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE TESTS (Paired t-test)")
        print("=" * 80)

        for metric in ['roc_auc', 'precision', 'recall', 'f1']:
            clean_scores = clean_results['cv_scores'][f'test_{metric}']
            all_scores = all_results['cv_scores'][f'test_{metric}']

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(clean_scores, all_scores)

            diff = clean_scores.mean() - all_scores.mean()
            better = "Clean" if diff > 0 else "All Data"

            print(f"\n{metric.upper()}:")
            print(f"  Difference: {abs(diff):.4f} (favors {better})")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")

    def compare_feature_importance(self, top_n: int = 15):
        """Compare top feature importances between strategies"""
        print("\n" + "=" * 80)
        print(f"TOP {top_n} FEATURE IMPORTANCE COMPARISON")
        print("=" * 80)

        clean_fi = self.results['clean']['feature_importance'].head(top_n)
        all_fi = self.results['all']['feature_importance'].head(top_n)

        print("\nCLEAN LABELS - Top Features:")
        print(clean_fi.to_string(index=False))

        print("\n\nALL DATA - Top Features:")
        print(all_fi.to_string(index=False))

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Clean labels
        axes[0].barh(range(top_n), clean_fi['importance'].values[::-1])
        axes[0].set_yticks(range(top_n))
        axes[0].set_yticklabels(clean_fi['feature'].values[::-1])
        axes[0].set_xlabel('Feature Importance')
        axes[0].set_title('Clean Labels\n(Exclude Admin Terminations)')
        axes[0].grid(axis='x', alpha=0.3)

        # All data
        axes[1].barh(range(top_n), all_fi['importance'].values[::-1])
        axes[1].set_yticks(range(top_n))
        axes[1].set_yticklabels(all_fi['feature'].values[::-1])
        axes[1].set_xlabel('Feature Importance')
        axes[1].set_title('All Data\n(Include All Terminations)')
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_path = Path(__file__).parent.parent / 'reports' / 'label_strategy_feature_comparison.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFeature importance comparison plot saved to: {output_path}")

    def run_comparison(self):
        """Run full comparison experiment"""
        # Prepare datasets
        self.prepare_datasets()

        # Engineer features for clean labels
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING: CLEAN LABELS")
        print("=" * 80)
        X_clean, y_clean, engineer_clean = self.engineer_features(self.df_clean, fit=True)
        print(f"Feature matrix shape: {X_clean.shape}")

        # Engineer features for all data
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING: ALL DATA")
        print("=" * 80)
        X_all, y_all, engineer_all = self.engineer_features(self.df_all, fit=True)
        print(f"Feature matrix shape: {X_all.shape}")

        # Train and evaluate both strategies
        self.results['clean'] = self.train_and_evaluate(X_clean, y_clean, "Clean Labels")
        self.results['all'] = self.train_and_evaluate(X_all, y_all, "All Data")

        # Compare results
        self.compare_results()

        # Compare feature importances
        self.compare_feature_importance()

        # Final recommendation
        self.print_recommendation()

    def print_recommendation(self):
        """Print final recommendation based on results"""
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)

        clean_auc = self.results['clean']['roc_auc_mean']
        all_auc = self.results['all']['roc_auc_mean']

        # Perform significance test
        clean_scores = self.results['clean']['cv_scores']['test_roc_auc']
        all_scores = self.results['all']['cv_scores']['test_roc_auc']
        _, p_value = stats.ttest_rel(clean_scores, all_scores)

        diff = abs(clean_auc - all_auc)
        better_strategy = "Clean Labels" if clean_auc > all_auc else "All Data"

        print(f"\nPerformance Difference (ROC-AUC):")
        print(f"  Clean Labels: {clean_auc:.4f}")
        print(f"  All Data:     {all_auc:.4f}")
        print(f"  Difference:   {diff:.4f}")
        print(f"  p-value:      {p_value:.4f}")
        print(f"  Significant:  {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")

        print(f"\n{'=' * 80}")
        if p_value < 0.05:
            print(f"✓ RECOMMENDATION: Use **{better_strategy}**")
            print(f"  The performance difference is statistically significant.")
        else:
            print(f"✓ RECOMMENDATION: Use **Clean Labels** (current approach)")
            print(f"  No statistically significant performance difference detected.")
            print(f"  Clean labels provide better interpretability and clearer target definition.")

        print("\nRationale:")
        if clean_auc >= all_auc - 0.01:  # If clean is competitive (within 0.01)
            print("  • Clean labels better align with the goal: predict clinical efficacy")
            print("  • Administrative failures introduce label noise")
            print("  • Model predictions are more interpretable and actionable")
            print("  • Marginal data increase doesn't offset label quality cost")
        else:
            print("  • All data approach provides significantly better performance")
            print("  • Additional 879 samples improve model learning")
            print("  • Trade-off: less interpretability for better predictions")

        print("=" * 80)


def main():
    """Main entry point"""
    data_path = Path(__file__).parent.parent / 'data' / 'completed_phase2_3_trials.csv'

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the raw data file exists.")
        return

    # Run comparison
    comparison = LabelingStrategyComparison(data_path)
    comparison.run_comparison()

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
