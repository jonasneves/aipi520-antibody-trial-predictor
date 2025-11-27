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
from datetime import datetime

# Import chart creation functions
sys.path.insert(0, str(Path(__file__).parent))
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


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and prepare raw labeled data"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} trials with {len(df.columns)} columns")
    return df


def generate_html_report(df: pd.DataFrame, output_path: Path, metadata: dict):
    """Generate complete HTML report"""
    print("\nGenerating HTML report...")

    # Load template
    template_path = Path(__file__).parent / 'templates' / 'dashboard_template.html'
    with open(template_path, 'r') as f:
        template = f.read()

    # Create all charts
    charts = {
        'status': create_status_chart(df),
        'phase': create_phase_chart(df),
        'sponsor': create_sponsor_chart(df),
        'enrollment': create_enrollment_chart(df),
        'temporal': create_temporal_chart(df),
        'antibody': create_antibody_type_chart(df),
        'antibody_success': create_antibody_success_chart(df),
        'antibody_temporal': create_antibody_temporal_chart(df),
        'top_antibodies': create_top_antibodies_chart(df),
        'antibody_by_area': create_antibody_by_area_chart(df),
        'outcome': create_outcome_chart(df),
        'conditions': create_conditions_chart(df),
        'interventions': create_interventions_chart(df),
        'phase_status': create_phase_status_heatmap(df),
    }

    # Remove None charts
    charts = {k: v for k, v in charts.items() if v is not None}

    # Parse start year for summary stats
    df_copy = df.copy()
    df_copy['start_date_parsed'] = pd.to_datetime(df_copy['start_date'], errors='coerce')
    df_copy['start_year'] = df_copy['start_date_parsed'].dt.year

    # Build summary cards HTML
    summary_cards_html = '<div class="summary-stats">\n' + '\n'.join([
        f'<div class="stat-card">',
        f'    <div class="stat-value">{len(df):,}</div>',
        f'    <div class="stat-label">Total Trials</div>',
        f'</div>',
        f'<div class="stat-card">',
        f'    <div class="stat-value">{len(df.columns)}</div>',
        f'    <div class="stat-label">Features</div>',
        f'</div>',
        f'<div class="stat-card">',
        f'    <div class="stat-value">{df["enrollment"].median():.0f}</div>',
        f'    <div class="stat-label">Median Enrollment</div>',
        f'</div>',
        f'<div class="stat-card">',
        f'    <div class="stat-value">{df_copy["start_year"].min():.0f} - {df_copy["start_year"].max():.0f}</div>',
        f'    <div class="stat-label">Year Range</div>',
        f'</div>',
    ]) + '\n</div>'

    # Define chart sections for better organization
    chart_sections = [
        ('Trial Overview', ['status', 'phase', 'outcome', 'phase_status']),
        ('Antibody Analysis', ['antibody', 'antibody_success', 'antibody_temporal', 'top_antibodies', 'antibody_by_area']),
        ('Study Characteristics', ['sponsor', 'enrollment', 'temporal']),
        ('Therapeutic Areas', ['conditions', 'interventions']),
    ]

    # Build content HTML with charts organized by section
    content_parts = []
    for section_title, chart_names in chart_sections:
        # Check if any charts in this section exist
        section_charts = {name: charts.get(name) for name in chart_names if name in charts}

        if not section_charts:
            continue

        content_parts.append(f'<h2>{section_title}</h2>')

        for name, fig in section_charts.items():
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f'chart-{name}')
            content_parts.append(f'<div class="chart">{chart_html}</div>')

    content_html = '\n'.join(content_parts)

    # Replace placeholders in template
    html = template.replace('{{ title }}', 'Raw Data Exploratory Analysis')
    html = html.replace('{{ header_icon }}', '🔍 ')
    html = html.replace('{{ raw_active }}', 'active')
    html = html.replace('{{ features_active }}', '')
    html = html.replace('{{ metadata.timestamp }}', metadata['timestamp'])
    html = html.replace('{{ metadata.data_source }}', metadata.get('data_source', 'ClinicalTrials.gov API'))
    html = html.replace('{{ metadata.workflow_url }}', metadata.get('workflow_url', '#'))
    html = html.replace('{{ summary_cards }}', summary_cards_html)
    html = html.replace('{{ content }}', content_html)
    html = html.replace('{{ insights }}', '')  # No insights for raw data report

    # Write HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"✓ Report saved to {output_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate EDA report for raw labeled data')
    parser.add_argument('input_csv', help='Path to clinical_trials_binary.csv')
    parser.add_argument('output_html', help='Path to output HTML report')
    parser.add_argument('--timestamp', default=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
    parser.add_argument('--data-source', default='ClinicalTrials.gov API')
    parser.add_argument('--workflow-url', default='#')

    args = parser.parse_args()

    # Load data
    df = load_data(Path(args.input_csv))

    # Generate report
    metadata = {
        'timestamp': args.timestamp,
        'data_source': args.data_source,
        'workflow_url': args.workflow_url,
    }

    generate_html_report(df, Path(args.output_html), metadata)

    print("\n" + "="*60)
    print("✓ Raw Data EDA Report Generated Successfully")
    print("="*60)


if __name__ == '__main__':
    main()
