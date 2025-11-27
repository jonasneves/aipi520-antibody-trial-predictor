#!/usr/bin/env python3
"""
Generate EDA report for engineered features

Analyzes clinical_trials_features.csv and generates an interactive HTML report with:
- Feature distributions
- Correlation heatmap
- Missing value analysis
- Feature importance (from quick Random Forest)
- Class balance
- Feature statistics
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Import base class and chart creation functions
sys.path.insert(0, str(Path(__file__).parent))
from base_report_generator import BaseReportGenerator
from charts.features import (
    create_missing_values_chart,
    create_class_balance_chart,
    create_feature_distribution_chart,
    create_feature_importance_chart,
    create_correlation_heatmap,
)


class FeaturesReportGenerator(BaseReportGenerator):
    """Report generator for engineered features"""

    def get_charts(self, df: pd.DataFrame) -> Dict:
        """Create all charts for features report"""
        return {
            'missing': create_missing_values_chart(df),
            'class_balance': create_class_balance_chart(df),
            'feature_dist': create_feature_distribution_chart(df),
            'importance': create_feature_importance_chart(df),
            'correlation': create_correlation_heatmap(df),
        }

    def build_summary_stats(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Build summary statistics for features"""
        # Compute feature statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['binary_outcome', 'outcome_label']]

        return [
            ('Samples', f'{len(df):,}'),
            ('Numeric Features', f'{len(feature_cols)}'),
            ('Missing Values', f'{df.isnull().sum().sum()}'),
            ('Total Columns', f'{len(df.columns)}'),
        ]

    def get_chart_sections(self) -> List[Tuple[str, List[str]]]:
        """Define chart sections for features report - no sections, flat list"""
        return None

    def get_report_title(self) -> str:
        """Get report title"""
        return 'Feature Engineering Analysis'

    def get_nav_active_states(self) -> Dict[str, str]:
        """Get navigation active states"""
        return {
            'overview': '',
            'raw': '',
            'features': 'active',
        }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate EDA report for engineered features')
    parser.add_argument('input_csv', help='Path to clinical_trials_features.csv')
    parser.add_argument('output_html', help='Path to output HTML report')
    parser.add_argument('--timestamp', default=None)
    parser.add_argument('--data-source', default='ClinicalTrials.gov API')
    parser.add_argument('--workflow-url', default='#')

    args = parser.parse_args()

    # Create generator and run
    generator = FeaturesReportGenerator()
    generator.run(
        input_csv=args.input_csv,
        output_html=args.output_html,
        timestamp=args.timestamp,
        data_source=args.data_source,
        workflow_url=args.workflow_url,
    )


if __name__ == '__main__':
    main()
