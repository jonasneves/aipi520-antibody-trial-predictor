#!/usr/bin/env python3
"""
Generate EDA report for raw labeled data

Analyzes clinical_trials_binary.csv and generates an interactive HTML report with:
- Status distribution
- Phase distribution
- Sponsor breakdown
- Enrollment statistics
- Temporal trends
- Antibody Analysis:
  * Antibody type distribution
  * Success rate by antibody type
  * Temporal evolution of antibody types
  * Top tested antibody drugs
  * Antibody type usage by therapeutic area
- Class balance
- Top 20 most common conditions
- Intervention type distribution
- Phase vs Status cross-analysis heatmap
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Import base class and chart creation functions
sys.path.insert(0, str(Path(__file__).parent))
from base_report_generator import BaseReportGenerator
from charts.raw_data import (
    create_status_chart,
    create_phase_chart,
    create_sponsor_chart,
    create_enrollment_chart,
    create_temporal_chart,
    create_antibody_type_chart,
    create_antibody_success_chart,
    create_antibody_temporal_chart,
    create_top_antibodies_chart,
    create_antibody_by_area_chart,
    create_outcome_chart,
    create_conditions_chart,
    create_interventions_chart,
    create_phase_status_heatmap,
)


class RawDataReportGenerator(BaseReportGenerator):
    """Report generator for raw labeled data"""

    def get_charts(self, df: pd.DataFrame) -> Dict:
        """Create all charts for raw data report"""
        from charts.raw_data.combined_overview_chart import create_combined_overview_chart
        
        return {
            'combined_overview': create_combined_overview_chart(df),
            'sponsor': create_sponsor_chart(df),
            'enrollment': create_enrollment_chart(df),
            'temporal': create_temporal_chart(df),
            'antibody': create_antibody_type_chart(df),
            'antibody_success': create_antibody_success_chart(df),
            'antibody_temporal': create_antibody_temporal_chart(df),
            'top_antibodies': create_top_antibodies_chart(df),
            'antibody_by_area': create_antibody_by_area_chart(df),
            'conditions': create_conditions_chart(df),
            'interventions': create_interventions_chart(df),
            'phase_status': create_phase_status_heatmap(df),
        }

    def build_summary_stats(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Build summary statistics for raw data"""
        # Parse start year for year range calculation
        df_copy = df.copy()
        df_copy['start_date_parsed'] = pd.to_datetime(df_copy['start_date'], errors='coerce')
        df_copy['start_year'] = df_copy['start_date_parsed'].dt.year
        
        # Calculate missing dates
        total_trials = len(df)
        missing_dates = df_copy['start_year'].isna().sum()
        missing_pct = (missing_dates / total_trials) * 100
        
        valid_years = df_copy['start_year'].dropna()
        year_range = f"{valid_years.min():.0f} - {valid_years.max():.0f}" if not valid_years.empty else "N/A"

        return [
            ('Total Trials', f'{total_trials:,}'),
            ('Missing Start Dates', f'{missing_dates:,} ({missing_pct:.1f}%)'),
            ('Year Range', year_range),
            ('Median Enrollment', f'{df["enrollment"].median():.0f}'),
        ]

    def get_chart_sections(self) -> List[Tuple[str, List[str]]]:
        """Define chart sections for raw data report"""
        return [
            ('Trial Overview', ['combined_overview', 'phase_status']),
            ('Antibody Analysis', ['antibody', 'antibody_success', 'antibody_temporal', 'top_antibodies', 'antibody_by_area']),
            ('Study Characteristics', ['sponsor', 'enrollment', 'temporal']),
            ('Therapeutic Areas', ['conditions', 'interventions']),
        ]

    def get_report_title(self) -> str:
        """Get report title"""
        return 'Raw Data Exploratory Analysis'

    def get_nav_active_states(self) -> Dict[str, str]:
        """Get navigation active states"""
        return {
            'overview': '',
            'raw': 'active',
            'features': '',
        }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate EDA report for raw labeled data')
    parser.add_argument('input_csv', help='Path to clinical_trials_binary.csv')
    parser.add_argument('output_html', help='Path to output HTML report')
    parser.add_argument('--timestamp', default=None)
    parser.add_argument('--data-source', default='ClinicalTrials.gov API')
    parser.add_argument('--workflow-url', default='#')

    args = parser.parse_args()

    # Create generator and run
    generator = RawDataReportGenerator()
    generator.run(
        input_csv=args.input_csv,
        output_html=args.output_html,
        timestamp=args.timestamp,
        data_source=args.data_source,
        workflow_url=args.workflow_url,
    )


if __name__ == '__main__':
    main()
