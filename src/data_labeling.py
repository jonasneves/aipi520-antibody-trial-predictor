"""
Clinical Trial Outcome Labeling Module

This module provides functions to label clinical trials as successful or failed
based on their status and outcomes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class TrialOutcomeLabeler:
    """
    Labels clinical trials with success/failure outcomes based on multiple criteria
    """

    # Status categories that indicate success
    SUCCESS_STATUSES = {
        'COMPLETED',
        'APPROVED_FOR_MARKETING',
        'AVAILABLE'
    }

    # Status categories that indicate failure
    FAILURE_STATUSES = {
        'TERMINATED',
        'WITHDRAWN',
        'SUSPENDED',
        'NO_LONGER_AVAILABLE'
    }

    # Ambiguous statuses that need further analysis
    AMBIGUOUS_STATUSES = {
        'ACTIVE_NOT_RECRUITING',
        'RECRUITING',
        'NOT_YET_RECRUITING',
        'ENROLLING_BY_INVITATION',
        'UNKNOWN'
    }

    def __init__(self):
        self.label_counts = {
            'success': 0,
            'failure': 0,
            'ambiguous': 0,
            'unlabeled': 0
        }

    def label_by_status(self, status: str) -> str:
        """
        Label trial outcome based on overall status

        Args:
            status: Overall status of the trial

        Returns:
            Label: 'success', 'failure', 'ambiguous', or 'unlabeled'
        """
        status_upper = status.upper() if status else ''

        if status_upper in self.SUCCESS_STATUSES:
            return 'success'
        elif status_upper in self.FAILURE_STATUSES:
            return 'failure'
        elif status_upper in self.AMBIGUOUS_STATUSES:
            return 'ambiguous'
        else:
            return 'unlabeled'

    def label_by_results(self, has_results: bool, status: str) -> str:
        """
        Label trial outcome based on whether results are posted

        Args:
            has_results: Boolean indicating if results are available
            status: Overall status of the trial

        Returns:
            Label: 'success', 'failure', or 'ambiguous'
        """
        if has_results and status.upper() == 'COMPLETED':
            return 'success'
        elif status.upper() in self.FAILURE_STATUSES:
            return 'failure'
        else:
            return 'ambiguous'

    def label_by_termination_reason(self, status: str, why_stopped: str = '') -> Tuple[str, str]:
        """
        Provide more granular labeling based on termination reasons

        Args:
            status: Overall status of the trial
            why_stopped: Reason for stopping (if terminated/withdrawn)

        Returns:
            Tuple of (label, reason_category)
        """
        if status.upper() not in self.FAILURE_STATUSES:
            return 'not_terminated', 'N/A'

        if not why_stopped:
            return 'failure', 'unknown_reason'

        why_stopped_lower = why_stopped.lower()

        # Categorize termination reasons
        if any(term in why_stopped_lower for term in ['safety', 'adverse', 'toxicity', 'death']):
            return 'failure', 'safety_concerns'
        elif any(term in why_stopped_lower for term in ['efficacy', 'futility', 'lack of efficacy']):
            return 'failure', 'lack_of_efficacy'
        elif any(term in why_stopped_lower for term in ['enrollment', 'recruit', 'accrual']):
            return 'failure', 'enrollment_issues'
        elif any(term in why_stopped_lower for term in ['funding', 'financial', 'budget', 'sponsor']):
            return 'failure', 'funding_issues'
        elif any(term in why_stopped_lower for term in ['business', 'strategic', 'commercial']):
            return 'failure', 'business_decision'
        else:
            return 'failure', 'other_reason'

    def label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply labeling to entire dataset

        Args:
            df: DataFrame with clinical trials data

        Returns:
            DataFrame with added label columns
        """
        print("Labeling clinical trials dataset...")

        # Primary label based on status
        df['outcome_label'] = df['overall_status'].apply(self.label_by_status)

        # Secondary label based on results availability
        if 'has_results' in df.columns:
            df['outcome_label_results'] = df.apply(
                lambda row: self.label_by_results(row['has_results'], row['overall_status']),
                axis=1
            )

        # Termination reason analysis (if available)
        if 'why_stopped' in df.columns:
            termination_labels = df.apply(
                lambda row: self.label_by_termination_reason(
                    row['overall_status'],
                    row.get('why_stopped', '')
                ),
                axis=1
            )
            df['termination_label'] = [label for label, _ in termination_labels]
            df['termination_reason'] = [reason for _, reason in termination_labels]

        # Compute label statistics
        self._compute_label_stats(df)

        return df

    def _compute_label_stats(self, df: pd.DataFrame):
        """
        Compute and print statistics about label distribution

        Args:
            df: Labeled DataFrame
        """
        print("\n=== Label Distribution ===")
        print(df['outcome_label'].value_counts())

        self.label_counts['success'] = (df['outcome_label'] == 'success').sum()
        self.label_counts['failure'] = (df['outcome_label'] == 'failure').sum()
        self.label_counts['ambiguous'] = (df['outcome_label'] == 'ambiguous').sum()
        self.label_counts['unlabeled'] = (df['outcome_label'] == 'unlabeled').sum()

        total = len(df)
        print(f"\nSuccess rate: {self.label_counts['success'] / total * 100:.2f}%")
        print(f"Failure rate: {self.label_counts['failure'] / total * 100:.2f}%")
        print(f"Ambiguous: {self.label_counts['ambiguous'] / total * 100:.2f}%")

        if 'termination_reason' in df.columns:
            print("\n=== Termination Reasons ===")
            term_reasons = df[df['termination_reason'] != 'N/A']['termination_reason']
            if len(term_reasons) > 0:
                print(term_reasons.value_counts())

    def create_binary_labels(self, df: pd.DataFrame, include_ambiguous: bool = False) -> pd.DataFrame:
        """
        Create binary labels (0/1) for modeling

        Args:
            df: Labeled DataFrame
            include_ambiguous: Whether to keep ambiguous samples (will attempt to infer)

        Returns:
            DataFrame with binary labels
        """
        # Start with primary labels
        df_binary = df.copy()

        # Map to binary
        label_map = {
            'success': 1,
            'failure': 0
        }

        # Filter based on whether we include ambiguous
        if not include_ambiguous:
            # Only keep clear success/failure cases
            df_binary = df_binary[df_binary['outcome_label'].isin(['success', 'failure'])]
        else:
            # Try to infer labels for ambiguous cases
            def infer_ambiguous(row):
                if row['outcome_label'] in ['success', 'failure']:
                    return label_map.get(row['outcome_label'], -1)

                # Use secondary signals to infer
                if 'outcome_label_results' in row and row['outcome_label_results'] == 'success':
                    return 1
                elif 'termination_label' in row and row['termination_label'] == 'failure':
                    return 0
                else:
                    # Cannot infer, mark as -1 for removal or special handling
                    return -1

            df_binary['binary_outcome'] = df_binary.apply(infer_ambiguous, axis=1)
            df_binary = df_binary[df_binary['binary_outcome'] != -1]

        if 'binary_outcome' not in df_binary.columns:
            df_binary['binary_outcome'] = df_binary['outcome_label'].map(label_map)

        # Remove any rows that couldn't be mapped
        df_binary = df_binary.dropna(subset=['binary_outcome'])

        print(f"\n=== Binary Labels Created ===")
        print(f"Total samples: {len(df_binary)}")
        print(f"Successful trials (1): {(df_binary['binary_outcome'] == 1).sum()}")
        print(f"Failed trials (0): {(df_binary['binary_outcome'] == 0).sum()}")
        print(f"Success rate: {(df_binary['binary_outcome'] == 1).sum() / len(df_binary) * 100:.2f}%")

        return df_binary

    def balance_dataset(self, df: pd.DataFrame, method: str = 'undersample') -> pd.DataFrame:
        """
        Balance the dataset to handle class imbalance

        Args:
            df: DataFrame with binary_outcome column
            method: 'undersample', 'oversample', or 'none'

        Returns:
            Balanced DataFrame
        """
        if 'binary_outcome' not in df.columns:
            raise ValueError("DataFrame must have 'binary_outcome' column")

        if method == 'none':
            return df

        # Count classes
        success_count = (df['binary_outcome'] == 1).sum()
        failure_count = (df['binary_outcome'] == 0).sum()

        print(f"\n=== Balancing Dataset ({method}) ===")
        print(f"Original - Success: {success_count}, Failure: {failure_count}")

        if method == 'undersample':
            # Undersample majority class
            min_count = min(success_count, failure_count)

            df_success = df[df['binary_outcome'] == 1].sample(n=min_count, random_state=42)
            df_failure = df[df['binary_outcome'] == 0].sample(n=min_count, random_state=42)

            df_balanced = pd.concat([df_success, df_failure]).sample(frac=1, random_state=42)

        elif method == 'oversample':
            # Oversample minority class (simple duplication)
            max_count = max(success_count, failure_count)

            df_success = df[df['binary_outcome'] == 1]
            df_failure = df[df['binary_outcome'] == 0]

            # Oversample the minority class
            if success_count < failure_count:
                df_success = df_success.sample(n=max_count, replace=True, random_state=42)
            else:
                df_failure = df_failure.sample(n=max_count, replace=True, random_state=42)

            df_balanced = pd.concat([df_success, df_failure]).sample(frac=1, random_state=42)

        else:
            raise ValueError(f"Unknown balancing method: {method}")

        print(f"Balanced - Success: {(df_balanced['binary_outcome'] == 1).sum()}, "
              f"Failure: {(df_balanced['binary_outcome'] == 0).sum()}")

        return df_balanced


def main():
    """
    Example usage of the labeling module
    """
    # Load the collected data
    try:
        df = pd.read_csv("../data/completed_phase2_3_trials.csv")
    except FileNotFoundError:
        print("Data file not found. Please run data_collection.py first.")
        return

    # Initialize labeler
    labeler = TrialOutcomeLabeler()

    # Label the dataset
    df_labeled = labeler.label_dataset(df)

    # Create binary labels
    df_binary = labeler.create_binary_labels(df_labeled, include_ambiguous=False)

    # Save labeled data
    df_labeled.to_csv("../data/clinical_trials_labeled.csv", index=False)
    df_binary.to_csv("../data/clinical_trials_binary.csv", index=False)

    print(f"\nLabeled data saved to ../data/clinical_trials_labeled.csv")
    print(f"Binary labeled data saved to ../data/clinical_trials_binary.csv")

    # Optional: Create balanced dataset
    df_balanced = labeler.balance_dataset(df_binary, method='undersample')
    df_balanced.to_csv("../data/clinical_trials_balanced.csv", index=False)
    print(f"Balanced data saved to ../data/clinical_trials_balanced.csv")


if __name__ == "__main__":
    main()
