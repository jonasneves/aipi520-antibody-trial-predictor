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
            'class_balance': create_class_balance_chart(df),
            'feature_dist': create_feature_distribution_chart(df),
            'importance': create_feature_importance_chart(df),
            'correlation': create_correlation_heatmap(df),
        }

    def build_summary_stats(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Build summary statistics for features"""
        # Calculate stats matching raw data EDA for comparison
        total_trials = len(df)

        # Missing start dates (start_year == 0 means missing)
        missing_dates = (df['start_year'] == 0).sum() if 'start_year' in df.columns else 0
        missing_pct = (missing_dates / total_trials) * 100

        # Year range (excluding 0 which means missing)
        if 'start_year' in df.columns:
            valid_years = df[df['start_year'] > 0]['start_year']
            year_range = f"{valid_years.min():.0f} - {valid_years.max():.0f}" if not valid_years.empty else "N/A"
        else:
            year_range = "N/A"

        # Median enrollment
        median_enrollment = df['enrollment'].median() if 'enrollment' in df.columns else 0

        return [
            ('Total Trials', f'{total_trials:,}'),
            ('Missing Start Dates', f'{missing_dates:,} ({missing_pct:.1f}%)'),
            ('Year Range', year_range),
            ('Median Enrollment', f'{median_enrollment:.0f}'),
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
