#!/usr/bin/env python3
"""
Generate EDA report for raw labeled data

Analyzes clinical_trials_binary.csv and generates an interactive HTML report with:
- Status distribution
- Phase distribution
- Sponsor breakdown
- Enrollment statistics
- Temporal trends
- Antibody type distribution
- Class balance
- Top 20 most common conditions
- Intervention type distribution
- Phase vs Status cross-analysis heatmap
"""

import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and prepare raw labeled data"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} trials with {len(df.columns)} columns")
    return df

def create_status_chart(df: pd.DataFrame) -> go.Figure:
    """Create trial status distribution chart"""
    status_counts = df['overall_status'].value_counts()

    fig = go.Figure(data=[
        go.Bar(
            y=status_counts.index,
            x=status_counts.values,
            orientation='h',
            marker_color='steelblue',
            text=status_counts.values,
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Distribution of Trial Status',
        xaxis_title='Number of Trials',
        yaxis_title='Status',
        height=400,
        template='plotly_white'
    )

    return fig

def create_phase_chart(df: pd.DataFrame) -> go.Figure:
    """Create phase distribution chart"""
    phase_counts = df['phase'].value_counts() if 'phase' in df.columns else df['phases'].value_counts()

    fig = go.Figure(data=[
        go.Bar(
            x=phase_counts.index,
            y=phase_counts.values,
            marker_color='coral',
            text=phase_counts.values,
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Distribution of Trial Phases',
        xaxis_title='Phase',
        yaxis_title='Number of Trials',
        height=400,
        template='plotly_white'
    )

    return fig

def create_sponsor_chart(df: pd.DataFrame) -> go.Figure:
    """Create sponsor class distribution chart"""
    sponsor_counts = df['sponsor_class'].value_counts()

    fig = go.Figure(data=[
        go.Pie(
            labels=sponsor_counts.index,
            values=sponsor_counts.values,
            hole=0.3,
            textinfo='label+percent',
        )
    ])

    fig.update_layout(
        title='Distribution of Sponsor Types',
        height=400,
        template='plotly_white'
    )

    return fig

def create_enrollment_chart(df: pd.DataFrame) -> go.Figure:
    """Create enrollment distribution chart"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Enrollment Distribution', 'Log-Scale Enrollment')
    )

    # Raw enrollment
    fig.add_trace(
        go.Histogram(x=df['enrollment'].dropna(), nbinsx=50, marker_color='lightblue'),
        row=1, col=1
    )

    # Log-scale enrollment
    import numpy as np
    fig.add_trace(
        go.Histogram(x=np.log1p(df['enrollment'].dropna()), nbinsx=50, marker_color='lightcoral'),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Enrollment Count", row=1, col=1)
    fig.update_xaxes(title_text="Log(Enrollment + 1)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)

    fig.update_layout(
        title_text='Enrollment Analysis',
        height=400,
        showlegend=False,
        template='plotly_white'
    )

    return fig

def create_temporal_chart(df: pd.DataFrame) -> go.Figure:
    """Create temporal trends chart"""
    # Parse start dates
    df['start_date_parsed'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['start_year'] = df['start_date_parsed'].dt.year

    trials_by_year = df['start_year'].value_counts().sort_index()

    fig = go.Figure(data=[
        go.Scatter(
            x=trials_by_year.index,
            y=trials_by_year.values,
            mode='lines+markers',
            line=dict(width=2, color='steelblue'),
            marker=dict(size=8),
        )
    ])

    fig.update_layout(
        title='Trials Started by Year',
        xaxis_title='Year',
        yaxis_title='Number of Trials',
        height=400,
        template='plotly_white'
    )

    return fig

def create_antibody_type_chart(df: pd.DataFrame) -> go.Figure:
    """Create antibody type distribution chart"""
    if 'antibody_type' not in df.columns:
        return None

    ab_counts = df['antibody_type'].value_counts()

    fig = go.Figure(data=[
        go.Bar(
            x=ab_counts.index,
            y=ab_counts.values,
            marker_color='mediumpurple',
            text=ab_counts.values,
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Distribution of Antibody Types',
        xaxis_title='Antibody Type',
        yaxis_title='Number of Trials',
        height=400,
        template='plotly_white'
    )

    return fig

def create_outcome_chart(df: pd.DataFrame) -> go.Figure:
    """Create outcome distribution chart"""
    if 'binary_outcome' in df.columns:
        outcome_col = 'binary_outcome'
        labels = {0: 'Failure', 1: 'Success'}
    elif 'outcome_label' in df.columns:
        outcome_col = 'outcome_label'
        labels = None
    else:
        return None

    outcome_counts = df[outcome_col].value_counts()
    if labels:
        outcome_counts.index = outcome_counts.index.map(labels)

    fig = go.Figure(data=[
        go.Bar(
            x=outcome_counts.index,
            y=outcome_counts.values,
            marker_color=['#e74c3c', '#2ecc71'],
            text=outcome_counts.values,
            textposition='auto',
        )
    ])

    # Calculate class balance
    total = outcome_counts.sum()
    balance_text = "<br>".join([f"{idx}: {val:,} ({val/total*100:.1f}%)"
                                 for idx, val in outcome_counts.items()])

    fig.update_layout(
        title=f'Trial Outcome Distribution<br><sub>{balance_text}</sub>',
        xaxis_title='Outcome',
        yaxis_title='Number of Trials',
        height=400,
        template='plotly_white'
    )

    return fig

def create_condition_chart(df: pd.DataFrame) -> go.Figure:
    """Create top 20 conditions chart"""
    if 'conditions' not in df.columns:
        return None

    # Extract individual conditions
    all_conditions = []
    for conditions in df['conditions'].dropna():
        all_conditions.extend([c.strip() for c in str(conditions).split(',')])

    if not all_conditions:
        return None

    condition_counts = pd.Series(all_conditions).value_counts().head(20)

    fig = go.Figure(data=[
        go.Bar(
            y=condition_counts.index[::-1],  # Reverse for better readability
            x=condition_counts.values[::-1],
            orientation='h',
            marker_color='teal',
            text=condition_counts.values[::-1],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title=f'Top 20 Most Common Conditions<br><sub>Total unique conditions: {len(pd.Series(all_conditions).unique()):,}</sub>',
        xaxis_title='Number of Trials',
        yaxis_title='Condition',
        height=600,
        template='plotly_white'
    )

    return fig

def create_intervention_type_chart(df: pd.DataFrame) -> go.Figure:
    """Create intervention type distribution chart"""
    if 'intervention_types' not in df.columns:
        return None

    # Extract individual intervention types
    all_interventions = []
    for interventions in df['intervention_types'].dropna():
        all_interventions.extend([i.strip() for i in str(interventions).split(',')])

    if not all_interventions:
        return None

    intervention_counts = pd.Series(all_interventions).value_counts()

    fig = go.Figure(data=[
        go.Pie(
            labels=intervention_counts.index,
            values=intervention_counts.values,
            hole=0.3,
            textinfo='label+percent',
        )
    ])

    fig.update_layout(
        title='Distribution of Intervention Types',
        height=500,
        template='plotly_white'
    )

    return fig

def create_phase_status_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create phase vs status cross-analysis heatmap"""
    phase_col = 'phase' if 'phase' in df.columns else 'phases'

    # Create crosstab with percentages
    crosstab = pd.crosstab(df[phase_col], df['overall_status'], normalize='index') * 100

    # Round for display
    crosstab_display = crosstab.round(1)

    fig = go.Figure(data=go.Heatmap(
        z=crosstab.values,
        x=crosstab.columns,
        y=crosstab.index,
        colorscale='YlOrRd',
        text=crosstab_display.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title='Percentage')
    ))

    fig.update_layout(
        title='Clinical Trial Status by Phase (%)',
        xaxis_title='Status',
        yaxis_title='Phase',
        height=500,
        template='plotly_white'
    )

    return fig

def generate_html_report(df: pd.DataFrame, output_path: Path, metadata: dict):
    """Generate complete HTML report"""
    print("\nGenerating HTML report...")

    # Create all charts
    charts = {
        'status': create_status_chart(df),
        'phase': create_phase_chart(df),
        'sponsor': create_sponsor_chart(df),
        'enrollment': create_enrollment_chart(df),
        'temporal': create_temporal_chart(df),
        'antibody': create_antibody_type_chart(df),
        'outcome': create_outcome_chart(df),
        'conditions': create_condition_chart(df),
        'interventions': create_intervention_type_chart(df),
        'phase_status': create_phase_status_heatmap(df),
    }

    # Remove None charts
    charts = {k: v for k, v in charts.items() if v is not None}

    # Build HTML
    html_parts = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '    <meta charset="UTF-8">',
        '    <title>Raw Data EDA - Antibody Trial Predictor</title>',
        '    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>',
        '    <style>',
        '        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }',
        '        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }',
        '        h2 { color: #34495e; margin-top: 30px; }',
        '        .metadata { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }',
        '        .metadata p { margin: 5px 0; }',
        '        .chart { margin: 30px 0; }',
        '        .summary-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }',
        '        .stat-card { background: #3498db; color: white; padding: 20px; border-radius: 5px; text-align: center; }',
        '        .stat-value { font-size: 2em; font-weight: bold; }',
        '        .stat-label { font-size: 0.9em; opacity: 0.9; }',
        '    </style>',
        '</head>',
        '<body>',
        '    <div class="container">',
        '        <h1>📊 Raw Data Exploratory Analysis</h1>',
        '        <p><strong>Antibody Trial Success Predictor</strong></p>',
        '',
        '        <div class="metadata">',
        f'            <p><strong>Generated:</strong> {metadata["timestamp"]}</p>',
        f'            <p><strong>Data Source:</strong> {metadata.get("data_source", "ClinicalTrials.gov API")}</p>',
        f'            <p><strong>Workflow Run:</strong> <a href="{metadata.get("workflow_url", "#")}" target="_blank">View on GitHub</a></p>',
        '        </div>',
        '',
        '        <h2>📈 Summary Statistics</h2>',
        '        <div class="summary-stats">',
        f'            <div class="stat-card"><div class="stat-value">{len(df):,}</div><div class="stat-label">Total Trials</div></div>',
        f'            <div class="stat-card"><div class="stat-value">{len(df.columns)}</div><div class="stat-label">Features</div></div>',
        f'            <div class="stat-card"><div class="stat-value">{df["enrollment"].median():.0f}</div><div class="stat-label">Median Enrollment</div></div>',
        f'            <div class="stat-card"><div class="stat-value">{df["start_year"].min():.0f} - {df["start_year"].max():.0f}</div><div class="stat-label">Year Range</div></div>',
        '        </div>',
    ]

    # Add charts
    for name, fig in charts.items():
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f'chart-{name}')
        html_parts.extend([
            f'        <div class="chart">',
            f'            {chart_html}',
            f'        </div>',
        ])

    html_parts.extend([
        '    </div>',
        '</body>',
        '</html>',
    ])

    # Write HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(html_parts))

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
