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
from datetime import datetime

# Import chart creation functions
sys.path.insert(0, str(Path(__file__).parent))
from charts.features import (
    create_missing_values_chart,
    create_class_balance_chart,
    create_feature_distribution_chart,
    create_feature_importance_chart,
    create_correlation_heatmap,
)


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and prepare engineered features"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} samples with {len(df.columns)} features")
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
        'missing': create_missing_values_chart(df),
        'class_balance': create_class_balance_chart(df),
        'feature_dist': create_feature_distribution_chart(df),
        'importance': create_feature_importance_chart(df),
        'correlation': create_correlation_heatmap(df),
    }

    # Remove None charts
    charts = {k: v for k, v in charts.items() if v is not None}

    # Compute feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['binary_outcome', 'outcome_label']]

    # Build summary cards HTML
    summary_cards_html = '<div class="summary-stats">\n' + '\n'.join([
        f'<div class="stat-card">',
        f'    <div class="stat-value">{len(df):,}</div>',
        f'    <div class="stat-label">Samples</div>',
        f'</div>',
        f'<div class="stat-card">',
        f'    <div class="stat-value">{len(feature_cols)}</div>',
        f'    <div class="stat-label">Numeric Features</div>',
        f'</div>',
        f'<div class="stat-card">',
        f'    <div class="stat-value">{df.isnull().sum().sum()}</div>',
        f'    <div class="stat-label">Missing Values</div>',
        f'</div>',
        f'<div class="stat-card">',
        f'    <div class="stat-value">{len(df.columns)}</div>',
        f'    <div class="stat-label">Total Columns</div>',
        f'</div>',
    ]) + '\n</div>'

    # Build content HTML with all charts
    content_parts = []
    for name, fig in charts.items():
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f'chart-{name}')
        content_parts.append(f'<div class="chart">{chart_html}</div>')

    content_html = '\n'.join(content_parts)

    # Replace placeholders in template
    html = template.replace('{{ title }}', 'Feature Engineering Analysis')
    html = html.replace('{{ header_icon }}', '🔬 ')
    html = html.replace('{{ raw_active }}', '')
    html = html.replace('{{ features_active }}', 'active')
    html = html.replace('{{ metadata.timestamp }}', metadata['timestamp'])
    html = html.replace('{{ metadata.data_source }}', metadata.get('data_source', 'ClinicalTrials.gov API'))
    html = html.replace('{{ metadata.workflow_url }}', metadata.get('workflow_url', '#'))
    html = html.replace('{{ summary_cards }}', summary_cards_html)
    html = html.replace('{{ content }}', content_html)
    html = html.replace('{{ insights }}', '')  # No insights for feature report

    # Write HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"✓ Report saved to {output_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate EDA report for engineered features')
    parser.add_argument('input_csv', help='Path to clinical_trials_features.csv')
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
    print("✓ Feature Engineering EDA Report Generated Successfully")
    print("="*60)


if __name__ == '__main__':
    main()
