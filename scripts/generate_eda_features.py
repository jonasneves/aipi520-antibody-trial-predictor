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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and prepare engineered features"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} samples with {len(df.columns)} features")
    return df

def create_missing_values_chart(df: pd.DataFrame) -> go.Figure:
    """Create missing values chart"""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values found!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="green")
        )
        fig.update_layout(
            title='Missing Values Analysis',
            height=300,
            template='plotly_white'
        )
        return fig

    missing_pct = (missing / len(df) * 100).round(2)

    fig = go.Figure(data=[
        go.Bar(
            y=missing.index,
            x=missing_pct.values,
            orientation='h',
            marker_color='coral',
            text=[f'{val}%' for val in missing_pct.values],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Missing Values by Feature',
        xaxis_title='Percentage Missing',
        yaxis_title='Feature',
        height=max(300, len(missing) * 25),
        template='plotly_white'
    )

    return fig

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap for numeric features"""
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude target variable
    feature_cols = [col for col in numeric_cols if col not in ['binary_outcome', 'outcome_label']]

    if len(feature_cols) == 0:
        return None

    # Compute correlation matrix (limit to top 30 features to keep readable)
    if len(feature_cols) > 30:
        # Use variance to select top features
        variances = df[feature_cols].var().sort_values(ascending=False)
        feature_cols = variances.head(30).index.tolist()

    corr_matrix = df[feature_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=f'Feature Correlation Heatmap (Top {len(feature_cols)} Features)',
        height=max(600, len(feature_cols) * 20),
        template='plotly_white'
    )

    return fig

def create_feature_importance_chart(df: pd.DataFrame) -> go.Figure:
    """Create quick feature importance using Random Forest"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    # Prepare data
    if 'binary_outcome' not in df.columns:
        return None

    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['binary_outcome', 'outcome_label']]

    if len(feature_cols) == 0:
        return None

    X = df[feature_cols].fillna(0)
    y = df['binary_outcome']

    # Quick Random Forest
    print("  Computing feature importance (this may take a minute)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Get top 20 features
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    fig = go.Figure(data=[
        go.Bar(
            y=importances['feature'],
            x=importances['importance'],
            orientation='h',
            marker_color='steelblue',
            text=importances['importance'].round(4),
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Top 20 Feature Importances (Random Forest)',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500,
        template='plotly_white'
    )

    return fig

def create_feature_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create distribution charts for key numeric features"""
    # Select a few important numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['binary_outcome', 'outcome_label']]

    # Pick top 6 by variance
    if len(feature_cols) > 6:
        variances = df[feature_cols].var().sort_values(ascending=False)
        feature_cols = variances.head(6).index.tolist()

    n_features = len(feature_cols)
    if n_features == 0:
        return None

    rows = (n_features + 2) // 3
    cols = min(3, n_features)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=feature_cols,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    for idx, col in enumerate(feature_cols):
        row = idx // 3 + 1
        col_pos = idx % 3 + 1

        fig.add_trace(
            go.Histogram(x=df[col].dropna(), nbinsx=30, marker_color='lightblue', name=col, showlegend=False),
            row=row, col=col_pos
        )

    fig.update_layout(
        title_text='Distribution of Key Features',
        height=rows * 300,
        template='plotly_white'
    )

    return fig

def create_class_balance_chart(df: pd.DataFrame) -> go.Figure:
    """Create class balance visualization"""
    if 'binary_outcome' not in df.columns:
        return None

    outcome_counts = df['binary_outcome'].value_counts()
    labels = {0: 'Failure', 1: 'Success'}
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

    total = outcome_counts.sum()
    balance_text = "<br>".join([f"{idx}: {val:,} ({val/total*100:.1f}%)"
                                 for idx, val in outcome_counts.items()])

    fig.update_layout(
        title=f'Class Balance<br><sub>{balance_text}</sub>',
        xaxis_title='Outcome',
        yaxis_title='Number of Samples',
        height=400,
        template='plotly_white'
    )

    return fig

def generate_html_report(df: pd.DataFrame, output_path: Path, metadata: dict):
    """Generate complete HTML report"""
    print("\nGenerating HTML report...")

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

    # Build HTML
    html_parts = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '    <meta charset="UTF-8">',
        '    <title>Feature Engineering EDA - Antibody Trial Predictor</title>',
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
        '        .stat-card { background: #9b59b6; color: white; padding: 20px; border-radius: 5px; text-align: center; }',
        '        .stat-value { font-size: 2em; font-weight: bold; }',
        '        .stat-label { font-size: 0.9em; opacity: 0.9; }',
        '    </style>',
        '</head>',
        '<body>',
        '    <div class="container">',
        '        <h1>🔧 Feature Engineering Analysis</h1>',
        '        <p><strong>Antibody Trial Success Predictor</strong></p>',
        '',
        '        <div class="metadata">',
        f'            <p><strong>Generated:</strong> {metadata["timestamp"]}</p>',
        f'            <p><strong>Data Source:</strong> {metadata.get("data_source", "ClinicalTrials.gov API")}</p>',
        f'            <p><strong>Workflow Run:</strong> <a href="{metadata.get("workflow_url", "#")}" target="_blank">View on GitHub</a></p>',
        '        </div>',
        '',
        '        <h2>📈 Feature Statistics</h2>',
        '        <div class="summary-stats">',
        f'            <div class="stat-card"><div class="stat-value">{len(df):,}</div><div class="stat-label">Samples</div></div>',
        f'            <div class="stat-card"><div class="stat-value">{len(feature_cols)}</div><div class="stat-label">Numeric Features</div></div>',
        f'            <div class="stat-card"><div class="stat-value">{df.isnull().sum().sum()}</div><div class="stat-label">Missing Values</div></div>',
        f'            <div class="stat-card"><div class="stat-value">{len(df.columns)}</div><div class="stat-label">Total Columns</div></div>',
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
