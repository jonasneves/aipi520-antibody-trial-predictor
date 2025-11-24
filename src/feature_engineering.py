"""
Clinical Trial Feature Engineering Module

This module extracts and engineers features from clinical trial data
for predictive modeling, with specialized features for antibody trials.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from data_collection import analyze_antibody


class TrialFeatureEngineer:
    """
    Engineers features from clinical trial data for machine learning models
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizers = {}
        self.sponsor_counts = {}  # Store sponsor counts from training data

    def extract_all_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Extract all features from the dataset

        Args:
            df: Input DataFrame with clinical trial data
            fit: Whether to fit transformers (True for training, False for test)

        Returns:
            DataFrame with engineered features
        """
        print("Extracting features from clinical trials...")

        df_features = df.copy()

        # Extract features in sequence
        df_features = self._extract_trial_characteristics(df_features)
        df_features = self._extract_temporal_features(df_features)
        df_features = self._extract_enrollment_features(df_features)
        df_features = self._extract_sponsor_features(df_features, fit)
        df_features = self._extract_condition_features(df_features, fit)
        df_features = self._extract_intervention_features(df_features, fit)
        df_features = self._extract_antibody_features(df_features)
        df_features = self._extract_phase_features(df_features)

        if 'brief_title' in df_features.columns:
            df_features = self._extract_text_features(df_features, fit)

        print(f"Feature extraction complete. Total features: {len(df_features.columns)}")

        return df_features

    def _extract_trial_characteristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic trial design characteristics"""
        # Study type encoding
        if 'study_type' in df.columns:
            df['is_interventional'] = (df['study_type'] == 'INTERVENTIONAL').astype(int)
            df['is_observational'] = (df['study_type'] == 'OBSERVATIONAL').astype(int)

        # Phase encoding
        if 'phases' in df.columns:
            for phase in ['EARLY_PHASE1', 'PHASE1', 'PHASE2', 'PHASE3', 'PHASE4']:
                df[f'is_{phase.lower()}'] = df['phases'].str.contains(phase, na=False).astype(int)

            # Count number of phases
            df['num_phases'] = sum(df[f'is_{p.lower()}'] for p in ['PHASE1', 'PHASE2', 'PHASE3', 'PHASE4', 'EARLY_PHASE1'])

        return df

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features related to timing"""
        if 'start_date' in df.columns:
            df['start_date_parsed'] = pd.to_datetime(df['start_date'], errors='coerce')

        if 'start_date_parsed' in df.columns:
            REFERENCE_YEAR = 2024

            df['start_year'] = df['start_date_parsed'].dt.year
            df['start_month'] = df['start_date_parsed'].dt.month
            df['start_quarter'] = df['start_date_parsed'].dt.quarter
            df['years_since_start'] = REFERENCE_YEAR - df['start_year']
            df['is_recent_trial'] = (df['years_since_start'] <= 5).astype(int)

        return df

    def _extract_enrollment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features related to patient enrollment"""
        if 'enrollment' not in df.columns:
            return df

        df['enrollment'] = pd.to_numeric(df['enrollment'], errors='coerce').fillna(0)
        df['enrollment_log'] = np.log1p(df['enrollment'])

        df['enrollment_category'] = pd.cut(
            df['enrollment'],
            bins=[0, 50, 100, 300, 1000, np.inf],
            labels=['very_small', 'small', 'medium', 'large', 'very_large']
        )

        return df

    def _extract_sponsor_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Extract features related to sponsors and organizations"""
        if 'sponsor_class' in df.columns:
            # One-hot encode sponsor class
            sponsor_dummies = pd.get_dummies(df['sponsor_class'], prefix='sponsor', drop_first=False)
            df = pd.concat([df, sponsor_dummies], axis=1)

            # Key sponsor types
            df['is_industry_sponsored'] = (df['sponsor_class'] == 'INDUSTRY').astype(int)
            df['is_nih_sponsored'] = (df['sponsor_class'] == 'NIH').astype(int)
            df['is_academic_sponsored'] = (
                df['sponsor_class'].isin(['FED', 'OTHER', 'OTHER_GOV'])
            ).astype(int)

        if 'sponsor_name' in df.columns:
            if fit:
                self.sponsor_counts = df['sponsor_name'].value_counts().to_dict()

            df['sponsor_trial_count'] = df['sponsor_name'].map(self.sponsor_counts).fillna(0)
            df['is_major_sponsor'] = (df['sponsor_trial_count'] > 20).astype(int)

        return df

    def _extract_categorical_features(self, df: pd.DataFrame, column: str, patterns: Dict[str, List[str]], prefix: str) -> pd.DataFrame:
        """
        Generic helper to extract categorical features from text columns

        Args:
            df: DataFrame
            column: Column name to search in
            patterns: Dict mapping category name to list of keywords
            prefix: Prefix for new feature columns

        Returns:
            DataFrame with new categorical features
        """
        if column not in df.columns:
            return df

        for category, keywords in patterns.items():
            pattern = '|'.join(keywords)
            df[f'{prefix}_{category}'] = df[column].str.lower().str.contains(
                pattern, na=False, regex=True
            ).astype(int)

        return df

    def _extract_condition_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Extract features from conditions/diseases"""
        if 'conditions' not in df.columns:
            return df

        # Number of conditions
        df['num_conditions'] = df['conditions'].str.split(',').apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        # Common condition categories
        condition_patterns = {
            'cancer': ['cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'melanoma'],
            'cardiovascular': ['heart', 'cardiac', 'cardiovascular', 'hypertension'],
            'neurological': ['alzheimer', 'parkinson', 'multiple sclerosis', 'epilepsy', 'stroke'],
            'diabetes': ['diabetes', 'diabetic'],
            'infectious': ['hiv', 'hepatitis', 'infection', 'covid', 'influenza'],
            'respiratory': ['asthma', 'copd', 'pulmonary', 'lung'],
            'autoimmune': ['arthritis', 'lupus', 'crohn', 'psoriasis'],
            'mental_health': ['depression', 'anxiety', 'schizophrenia', 'bipolar']
        }

        return self._extract_categorical_features(df, 'conditions', condition_patterns, 'condition')

    def _extract_intervention_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Extract features from interventions"""
        if 'intervention_types' not in df.columns:
            return df

        # Number of interventions
        df['num_interventions'] = df['intervention_types'].str.split(',').apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        # Common intervention types
        intervention_types = ['DRUG', 'BIOLOGICAL', 'DEVICE', 'PROCEDURE',
                             'BEHAVIORAL', 'DIETARY_SUPPLEMENT', 'RADIATION', 'OTHER']

        for int_type in intervention_types:
            df[f'intervention_{int_type.lower()}'] = df['intervention_types'].str.contains(
                int_type, na=False
            ).astype(int)

        # Combination therapy indicator
        df['is_combination_therapy'] = (df['num_interventions'] > 1).astype(int)

        return df

    def _extract_antibody_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract antibody-specific features for monoclonal antibody trials
        """
        # If antibody columns already exist from data collection, use them
        if 'is_antibody' in df.columns:
            if 'antibody_type' in df.columns:
                for ab_type in ['fully_human', 'humanized', 'chimeric', 'murine']:
                    df[f'is_{ab_type}_antibody'] = (df['antibody_type'] == ab_type).astype(int)

        # Otherwise, detect antibodies from intervention_names
        elif 'intervention_names' in df.columns:
            df['is_antibody'] = df['intervention_names'].apply(
                lambda x: any(analyze_antibody(name)[0] for name in str(x).split(',')) if pd.notna(x) else False
            )

            # Extract primary antibody name and type
            def get_primary_antibody(intervention_names):
                if pd.isna(intervention_names):
                    return '', 'not_antibody'
                names = str(intervention_names).split(',')
                for name in names:
                    is_ab, ab_type = analyze_antibody(name.strip())
                    if is_ab:
                        return name.strip(), ab_type
                return '', 'not_antibody'

            antibody_info = df['intervention_names'].apply(get_primary_antibody)
            df['antibody_name'] = antibody_info.apply(lambda x: x[0])
            df['antibody_type'] = antibody_info.apply(lambda x: x[1])

            for ab_type in ['fully_human', 'humanized', 'chimeric', 'murine']:
                df[f'is_{ab_type}_antibody'] = (df['antibody_type'] == ab_type).astype(int)

        # Extract target mechanism from title and conditions
        if 'brief_title' in df.columns:
            target_patterns = {
                'checkpoint_inhibitor': r'pd-1|pd-l1|ctla-4|lag-3|tim-3',
                'growth_factor': r'her2|egfr|vegf|igf|fgfr',
                'cytokine': r'\btnf\b|il-\d+|interleukin|interferon',
                'cd_marker': r'cd\d+|cd20|cd38|cd19|cd30|cd52',
            }

            title_text = df['brief_title'].fillna('').str.lower()
            condition_text = df['conditions'].fillna('').str.lower() if 'conditions' in df.columns else pd.Series([''] * len(df))
            combined_text = title_text + ' ' + condition_text

            for target_type, pattern in target_patterns.items():
                df[f'antibody_target_{target_type}'] = combined_text.str.contains(
                    pattern, case=False, regex=True, na=False
                ).astype(int)

            # Biomarker selection indicator
            biomarker_pattern = r'biomarker|positive|\+|expressing|mutation|her2\+|pd-l1\+|selected'
            df['has_biomarker_selection'] = df['brief_title'].str.contains(
                biomarker_pattern, case=False, regex=True, na=False
            ).astype(int)

            # Combination therapy detection
            combination_pattern = r'combination|plus|with|\+|and'
            df['antibody_combination'] = df['brief_title'].str.contains(
                combination_pattern, case=False, regex=True, na=False
            ).astype(int)

            # Favorable indications
            favorable_pattern = r'cancer|lymphoma|leukemia|myeloma|melanoma|arthritis|psoriasis|crohn'
            df['antibody_favorable_indication'] = combined_text.str.contains(
                favorable_pattern, case=False, regex=True, na=False
            ).astype(int)

        return df

    def _extract_phase_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract phase-specific features including historical success rates"""
        # Typical success rates by phase (from literature)
        phase_success_rates = {
            'PHASE1': 0.63,
            'PHASE2': 0.31,
            'PHASE3': 0.58,
            'PHASE4': 0.85
        }

        phase_risk_scores = {
            'EARLY_PHASE1': 0.7,
            'PHASE1': 0.6,
            'PHASE2': 0.9,  # Highest risk
            'PHASE3': 0.5,
            'PHASE4': 0.2   # Lowest risk
        }

        if 'phases' in df.columns:
            df['expected_success_rate'] = df['phases'].map(
                lambda x: phase_success_rates.get(x, 0.5) if pd.notna(x) else 0.5
            )

            df['phase_risk_score'] = df['phases'].map(phase_risk_scores).fillna(0.5)

        return df

    def _extract_text_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Extract features from text fields using TF-IDF"""
        if 'brief_title' not in df.columns:
            return df

        # Basic text features
        df['title_length'] = df['brief_title'].str.len()
        df['title_word_count'] = df['brief_title'].str.split().str.len()

        # TF-IDF features from title
        if fit:
            self.tfidf_vectorizers['title'] = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )
            title_tfidf = self.tfidf_vectorizers['title'].fit_transform(df['brief_title'].fillna(''))
        else:
            if 'title' in self.tfidf_vectorizers:
                title_tfidf = self.tfidf_vectorizers['title'].transform(df['brief_title'].fillna(''))
            else:
                return df

        # Add TF-IDF features to dataframe
        tfidf_df = pd.DataFrame(
            title_tfidf.toarray(),
            columns=[f'tfidf_title_{i}' for i in range(title_tfidf.shape[1])],
            index=df.index
        )

        df = pd.concat([df, tfidf_df], axis=1)

        return df

    def select_features_for_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare final features for modeling

        Args:
            df: DataFrame with all engineered features

        Returns:
            DataFrame with selected features ready for modeling
        """
        numeric_features = [
            'enrollment', 'enrollment_log', 'num_conditions', 'num_interventions',
            'num_phases', 'sponsor_trial_count',
            'start_year', 'years_since_start', 'phase_risk_score',
            'expected_success_rate', 'title_length', 'title_word_count'
        ]

        categorical_features = [
            'is_interventional', 'is_observational',
            'is_phase1', 'is_phase2', 'is_phase3', 'is_phase4',
            'is_industry_sponsored', 'is_nih_sponsored',
            'is_combination_therapy', 'is_recent_trial', 'is_major_sponsor'
        ]

        condition_features = [col for col in df.columns if col.startswith('condition_')]
        intervention_features = [col for col in df.columns if col.startswith('intervention_')]
        tfidf_features = [col for col in df.columns if col.startswith('tfidf_')]

        antibody_features = [col for col in df.columns if
                           col.startswith('antibody_') or
                           (col.startswith('is_') and 'antibody' in col) or
                           col == 'has_biomarker_selection']

        all_features = (
            numeric_features + categorical_features +
            condition_features + intervention_features +
            antibody_features + tfidf_features
        )

        # Select only features that exist
        available_features = [f for f in all_features if f in df.columns]

        print(f"Selected {len(available_features)} features for modeling")

        # Create modeling dataframe
        df_modeling = df[available_features].copy()

        # Clean data - consolidate fillna logic
        for col in df_modeling.columns:
            if df_modeling[col].dtype in ['float64', 'int64']:
                df_modeling[col] = df_modeling[col].fillna(df_modeling[col].median())
            else:
                df_modeling[col] = df_modeling[col].fillna(0)

        # Drop any remaining non-numeric columns
        non_numeric_cols = df_modeling.select_dtypes(include=['object']).columns.tolist()
        if non_numeric_cols:
            print(f"Warning: Dropping non-numeric columns: {non_numeric_cols}")
            df_modeling = df_modeling.drop(columns=non_numeric_cols)

        # Ensure all columns are numeric and handle any NaN from coercion
        df_modeling = df_modeling.apply(pd.to_numeric, errors='coerce').fillna(0)

        return df_modeling

    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for interpreting model importance"""
        return list(self.label_encoders.keys())


def main():
    """Example usage of feature engineering"""
    try:
        df = pd.read_csv("../data/clinical_trials_labeled.csv")
    except FileNotFoundError:
        print("Labeled data not found. Please run data_labeling.py first.")
        return

    engineer = TrialFeatureEngineer()

    # Extract all features
    df_features = engineer.extract_all_features(df, fit=True)

    # Select features for modeling
    df_modeling = engineer.select_features_for_modeling(df_features)

    # Add labels back
    if 'binary_outcome' in df.columns:
        df_modeling['binary_outcome'] = df['binary_outcome']
    if 'outcome_label' in df.columns:
        df_modeling['outcome_label'] = df['outcome_label']

    # Save engineered features
    output_file = "../data/clinical_trials_features.csv"
    df_modeling.to_csv(output_file, index=False)

    print(f"\nEngineered features saved to {output_file}")
    print(f"Feature matrix shape: {df_modeling.shape}")
    print("\n=== Feature Summary ===")
    print(f"Total features: {len(df_modeling.columns)}")
    print("\nFeature types:")
    print(df_modeling.dtypes.value_counts())


if __name__ == "__main__":
    main()
